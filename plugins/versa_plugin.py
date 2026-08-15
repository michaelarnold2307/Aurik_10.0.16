"""versa_plugin — SingMOS Pro via VERSA Toolkit (2024).

Primär:  SingMOS Pro (South-Twilight/SingMOS v1.1.1, via torch.hub)
         Integration: VERSA / models/versa/versa/utterance_metrics/pseudo_mos.py
         Eingabe: float32 mono @ 16 kHz; Ausgabe: MOS ∈ [1.0, 5.0]
         Hub-Cache: models/versa/hub_cache/ (offline nach Installation)
         Referenz: https://arxiv.org/abs/2510.01812

Fallback: PQS-DSP-Gammatone (Bark-Filterbank + Sigmoid-MOS-Mapping)  # §V6 (copilot-instructions.md): logger.warning handled at call site

VERBOTEN laut Spec §4.4: PESQ, DNSMOS, NISQA, STOI, CDPAM.

Singleton-Pattern: get_versa_plugin() verwenden.
CPU-Only.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_VERSA_PATH = _ROOT / "models" / "versa"
_HUB_CACHE = str(_VERSA_PATH / "hub_cache")

# Offline-Betrieb: S3PRL-Cache auf gebündeltes models/s3prl/ setzen (vor s3prl-Import!).
# s3prl liest S3PRL_CACHE_HOME beim ersten Import — muss hier auf Modulebene gesetzt sein.
_S3PRL_BUNDLED = _ROOT / "models" / "s3prl"
if _S3PRL_BUNDLED.exists():
    os.environ.setdefault("S3PRL_CACHE_HOME", str(_S3PRL_BUNDLED))

# s3prl scannt beim Import alle Upstreams und loggt für jedes fehlende optionale
# Paket (ESPnet, Fairseq, …) eine WARNING. Diese sind harmlos — Aurik nutzt nur
# wav2vec2 (kein ESPnet/Fairseq erforderlich).  Logger-Filter einmalig setzen.
import logging as _logging

_logging.getLogger("s3prl.upstream.espnet_hubert.expert").setLevel(_logging.ERROR)
_logging.getLogger("s3prl.upstream.fairseq.expert").setLevel(_logging.ERROR)
del _logging
_MODEL_SR: int = 16_000
# SingMOS Pro / wav2vec2-large is designed for short utterances (≤30 s).
# Processing full-length tracks (3+ min) causes O(n) inference on CPU and
# leads to 20–40 min blocking calls per FC iteration. Cap at 30 s representative
# centre-slice — perceptually sufficient for MOS scoring (§Performance-Budget).
_SINGMOS_MAX_SAMPLES_16K: int = 30 * _MODEL_SR  # 30 s @ 16 kHz = 480 000 samples

_lock = threading.Lock()
_instance: VersaPlugin | None = None


@contextlib.contextmanager
def _suppress_s3prl_timm_deprecation():
    """Unterdrueckt nur die bekannte s3prl/timm-Import-Deprecation."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing from timm\.models\.layers is deprecated.*",
            category=FutureWarning,
            module=r"timm\.models\.layers",
        )
        yield


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VersaResult:
    """Ergebnis der VERSA MOS-Schätzung.

    Attributes:
        mos:          MOS-Wert ∈ [1.0, 5.0]
        model_used:   "singmos_pro" | "pqs_dsp_fallback"
        confidence:   Modell-Konfidenz ∈ [0, 1]
        sub_scores:   Optionale Teil-Scores (Signal, Hintergrund, Gesamt)
    """

    mos: float
    model_used: str
    confidence: float = 1.0
    sub_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # NaN/Inf-Guard (§3.1 Aurik Spec)
        if not math.isfinite(self.mos):
            self.mos = 3.0
        self.mos = float(np.clip(self.mos, 1.0, 5.0))


# ---------------------------------------------------------------------------
# VersaPlugin
# ---------------------------------------------------------------------------


class VersaPlugin:
    """SingMOS Pro via VERSA Toolkit — referenzfreier Musik-MOS.

    Primär: SingMOS Pro (South-Twilight/SingMOS v1.1.1) über VERSA pseudo_mos.
    Fallback: PQS-Gammatone-DSP (§4.4 Spec).  # §V6: logger.warning handled at call site

    Verwendung NUR für Qualitätsbewertung — keine Modifikation des Audios.
    """

    def __init__(self) -> None:
        self._predictor_dict: dict | None = None
        self._predictor_fs: dict | None = None
        self._pseudo_mos_metric = None
        self._model_loaded: bool = False
        self._load_attempted: bool = False
        self._load_lock = threading.Lock()
        # Lazy load: model is loaded on first score() call, NOT here.
        # Eager loading of SingMOS Pro (wav2vec2-large 606 MB) would cause OOM
        # when multiple plugins are imported simultaneously during test collection.

    @staticmethod
    def _vocal_confidence(audio: np.ndarray, sr: int) -> float:
        """Gibt vocal confidence [0, 1] from PANNs for SingMOS gating zurück."""
        try:
            from plugins.panns_plugin import get_panns_plugin

            tags = get_panns_plugin().get_tags(audio, sr)
        except Exception as exc:
            logger.debug("VERSA: PANNs unavailable for vocal gating: %s", exc)
            return 0.0

        singing_conf = float(tags.get("Singing voice", 0.0))
        vocals_conf = float(tags.get("Vocals", 0.0))
        max_vocal_conf = max(singing_conf, vocals_conf)
        logger.debug("VERSA: vocal_conf=%.3f", max_vocal_conf)
        return max_vocal_conf

    # Budget: wav2vec2-large (s3prl, ~600 MB) + SingMOS Pro checkpoint (~150 MB)
    _BUDGET_GB: float = 0.80

    def _try_load(self) -> None:
        """Loads SingMOS Pro via VERSA pseudo_mos; PQS-DSP fallback on error.

        Offline-Invariante: Lädt NUR wenn beide Checkpoint-Dateien lokal
        vorhanden sind. Kein torch.hub-Download im Produktionsbetrieb.
        Fehlende Weights → sofortiger PQS-DSP-Fallback (kein Netzwerkaufruf).
        """
        versa_pkg = _VERSA_PATH / "versa"
        _pm_path = _VERSA_PATH / "versa" / "utterance_metrics" / "pseudo_mos.py"
        if not versa_pkg.exists() or not _pm_path.exists():
            logger.info("VERSA toolkit nicht gefunden (%s) — PQS-DSP-Fallback.", versa_pkg)
            return

        # Offline check: SingMOS Pro checkpoint must be locally cached.
        # torch.hub caches the repo zip at hub_cache/<user>_SingMOS_<tag>/
        # and downloads the .pth checkpoint to hub_cache/checkpoints/.
        _hub_dir = _VERSA_PATH / "hub_cache"
        _singmos_checkpoint = _hub_dir / "checkpoints" / "ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
        # s3prl sucht wav2vec2-large in $S3PRL_CACHE_HOME/download/
        # Primär: gebündeltes models/s3prl/download/ (Offline-Betrieb)
        # Fallback: ~/.cache/s3prl/download/ (Legacy / Entwicklungsumgebung)
        _s3prl_bundled_dl = _S3PRL_BUNDLED / "download"
        _s3prl_sys_cache = Path(os.path.expanduser("~/.cache/s3prl/download"))

        def _has_wav2vec(_dir: Path) -> bool:
            return _dir.exists() and any("wav2vec_vox_new" in f.name for f in _dir.iterdir())

        _wav2vec2_cached = _has_wav2vec(_s3prl_bundled_dl) or _has_wav2vec(_s3prl_sys_cache)

        if not _singmos_checkpoint.exists():
            logger.info(
                "SingMOS Pro Checkpoint nicht lokal gefunden (%s) — PQS-DSP-Fallback. "
                "Für ML-Betrieb: models/versa/hub_cache/checkpoints/ befüllen "
                "(ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth).",
                _singmos_checkpoint,
            )
            return
        if not _wav2vec2_cached:
            logger.info(
                "wav2vec2-large (s3prl) nicht lokal gefunden ("
                "models/s3prl/download/ oder ~/.cache/s3prl/download/) "
                "— PQS-DSP-Fallback. Für ML-Betrieb: models/s3prl/ befüllen."
            )
            return

        try:
            from backend.core.ml_memory_budget import try_allocate as _try_alloc

            if not _try_alloc("VersaSingMOS", size_gb=self._BUDGET_GB):
                logger.warning("VERSA SingMOS Pro: ML-Budget erschöpft (%.2f GB) — PQS-Fallback.", self._BUDGET_GB)
                return
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)

        try:
            import importlib.util

            import torch

            # Load pseudo_mos.py directly to bypass versa/__init__.py which imports
            # pysptk (mcd_f0 dependency) that may not be installed on the target machine.
            _pm_path = _VERSA_PATH / "versa" / "utterance_metrics" / "pseudo_mos.py"
            _spec = importlib.util.spec_from_file_location("versa_pseudo_mos", str(_pm_path))
            _pm_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
            with _suppress_s3prl_timm_deprecation():
                _spec.loader.exec_module(_pm_mod)  # type: ignore[union-attr]
            pseudo_mos_setup = _pm_mod.pseudo_mos_setup
            pseudo_mos_metric = _pm_mod.pseudo_mos_metric

            torch.hub.set_dir(_HUB_CACHE)
            try:
                from backend.core.ml_device_manager import get_torch_device as _get_dev

                _versa_use_gpu = _get_dev("VersaSingMOS") != "cpu"
            except Exception:
                _versa_use_gpu = False
            # §v10.17: SingMOS always CPU — ROCm device mismatch fix
            _versa_use_gpu = False
            with _suppress_s3prl_timm_deprecation():
                predictor_dict, predictor_fs = pseudo_mos_setup(
                    predictor_types=["singmos_pro"],
                    predictor_args={"singmos_pro": {"fs": _MODEL_SR}},
                    cache_dir=_HUB_CACHE,
                    use_gpu=_versa_use_gpu,
                )
            self._predictor_dict = predictor_dict
            self._predictor_fs = predictor_fs
            self._pseudo_mos_metric = pseudo_mos_metric
            self._model_loaded = True

            # PLM lifecycle registration
            from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

            def _unload_singmos() -> None:
                if _instance is not None:
                    _instance._predictor_dict = None
                    _instance._predictor_fs = None
                    _instance._pseudo_mos_metric = None
                    _instance._model_loaded = False

            _reg_plm("VersaSingMOS", size_gb=self._BUDGET_GB, unload_fn=_unload_singmos)
            logger.info("✅ VERSA SingMOS Pro geladen (§4.4 — Musik-MOS-Primär)")
        except Exception as exc:
            logger.warning("VERSA SingMOS Pro nicht ladbar: %s — PQS-DSP-Fallback.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, audio: np.ndarray, sr: int) -> VersaResult:
        """Berechnet referenzfreien MOS-Wert für Musik- oder Sprachaufnahme.

        Args:
            audio: float32 mono/stereo, 48000 Hz
            sr:    Sample-Rate (muss 48000 sein)

        Returns:
            VersaResult mit MOS ∈ [1.0, 5.0] und Metadaten.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 1:
            mono = audio
        elif audio.ndim == 2:
            # Accept both layouts: [N, C] and [C, N].
            if audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
                mono = audio.mean(axis=0)
            elif audio.shape[1] <= 2 and audio.shape[0] > audio.shape[1]:
                mono = audio.mean(axis=1)
            else:
                mono = audio.mean(axis=-1)
        else:
            mono = np.ravel(audio)
        mono = np.clip(mono, -1.0, 1.0)

        # Tiered vocal gating: ≥0.7 pure SingMOS, 0.3–0.7 blend, <0.3 pure PQS
        _vocal_conf = self._vocal_confidence(mono, sr)

        if _vocal_conf < 0.3:
            return self._score_pqs_dsp(mono, sr)

        # Lazy load: attempt once, then use whatever is available.
        if not self._load_attempted:
            with self._load_lock:
                if not self._load_attempted:
                    self._try_load()
                    self._load_attempted = True

        if not (self._model_loaded and self._predictor_dict is not None):
            return self._score_pqs_dsp(mono, sr)

        if _vocal_conf >= 0.7:
            # High vocal confidence → pure SingMOS Pro
            return self._score_singmos_pro(mono, sr)

        # Blend zone (0.3–0.7): weighted average of SingMOS and PQS-DSP
        _singmos_result = self._score_singmos_pro(mono, sr)
        _pqs_result = self._score_pqs_dsp(mono, sr)
        # Linear blend weight: 0.3→0.0 SingMOS, 0.7→1.0 SingMOS
        _w = (_vocal_conf - 0.3) / 0.4
        _blended_mos = float(np.clip(_w * _singmos_result.mos + (1.0 - _w) * _pqs_result.mos, 1.0, 5.0))
        _blended_conf = float(_w * _singmos_result.confidence + (1.0 - _w) * _pqs_result.confidence)
        logger.debug(
            "VERSA: Blended MOS=%.3f (SingMOS=%.3f×%.2f + PQS=%.3f×%.2f)",
            _blended_mos,
            _singmos_result.mos,
            _w,
            _pqs_result.mos,
            1.0 - _w,
        )
        return VersaResult(mos=_blended_mos, model_used="singmos_pqs_blend", confidence=_blended_conf)

    # ------------------------------------------------------------------
    # SingMOS Pro Inference
    # ------------------------------------------------------------------

    def _score_singmos_pro(self, mono_48k: np.ndarray, sr: int) -> VersaResult:
        """SingMOS Pro MOS inference via VERSA pseudo_mos_metric.

        Resamples 48 kHz → 16 kHz (Lanczos), calls SingMOS Pro model,
        returns calibrated MOS ∈ [1.0, 5.0].
        """
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(sr, _MODEL_SR)
            mono_16k = resample_poly(mono_48k, _MODEL_SR // g, sr // g).astype(np.float32)
            mono_16k = np.nan_to_num(mono_16k, nan=0.0, posinf=0.0, neginf=0.0)
            mono_16k = np.clip(mono_16k, -1.0, 1.0)
            # Cap to 30 s centre-slice — wav2vec2-large inference is O(n); feeding
            # full-length tracks (> 30 s) causes multi-minute blocking calls on CPU.
            if mono_16k.size > _SINGMOS_MAX_SAMPLES_16K:
                _start = (mono_16k.size - _SINGMOS_MAX_SAMPLES_16K) // 2
                mono_16k = mono_16k[_start : _start + _SINGMOS_MAX_SAMPLES_16K]
                logger.debug(
                    "SingMOS: audio trimmed to %.0fs centre-slice for MOS scoring (full=%.0fs)",
                    _SINGMOS_MAX_SAMPLES_16K / _MODEL_SR,
                    mono_48k.size / sr,
                )
            if mono_16k.size < 320:
                logger.debug("SingMOS input too short (%d samples @16k) — using PQS fallback", mono_16k.size)
                return self._score_pqs_dsp(mono_48k, sr)

            # Robustly handle backend-specific shape expectations (1D vs 2D [B, T]).
            # Some SingMOS builds expect a batch dimension and raise "Dimension out of range"
            # for plain 1D vectors.
            _last_exc: Exception | None = None
            _scores = None
            assert self._pseudo_mos_metric is not None  # guarded by _model_loaded check
            # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during SingMOS inference
            _plm_versa = None
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_v

                _plm_versa = _get_plm_v()
                _plm_versa.set_active("VersaSingMOS", True)
            except Exception:
                logger.warning("versa_plugin.py::_score_singmos_pro fallback", exc_info=True)
            try:
                for _candidate in (mono_16k[np.newaxis, :], mono_16k):
                    try:
                        _scores = self._pseudo_mos_metric(
                            _candidate, _MODEL_SR, self._predictor_dict, self._predictor_fs
                        )
                        break
                    except Exception as _shape_exc:
                        _last_exc = _shape_exc
            finally:
                if _plm_versa is not None:
                    try:
                        _plm_versa.set_active("VersaSingMOS", False)
                    except Exception:
                        logger.warning("versa_plugin.py::_score_singmos_pro fallback", exc_info=True)

            if _scores is None:
                assert _last_exc is not None
                raise _last_exc

            if isinstance(_scores, dict):
                _mos_raw = _scores.get("singmos_pro", _scores.get("overall", 3.0))
            elif isinstance(_scores, (list, tuple)) and len(_scores) > 0:
                _first = _scores[0]
                _mos_raw = _first.get("singmos_pro", _first.get("overall", 3.0)) if isinstance(_first, dict) else _first
            else:
                _mos_raw = _scores

            if hasattr(_mos_raw, "item"):
                _mos_raw = _mos_raw.item()

            mos = float(np.clip(float(_mos_raw), 1.0, 5.0))
            if not math.isfinite(mos):
                mos = 3.0
            logger.debug("SingMOS Pro MOS: %.3f", mos)
            return VersaResult(mos=mos, model_used="singmos_pro", confidence=0.92)
        except Exception as exc:
            logger.warning("SingMOS Pro Inferenzfehler: %s — PQS-DSP-Fallback.", exc)
            return self._score_pqs_dsp(mono_48k, sr)

    # ------------------------------------------------------------------
    # PQS-DSP Fallback
    # ------------------------------------------------------------------

    def _score_pqs_dsp(self, mono: np.ndarray, sr: int) -> VersaResult:
        """PQS-Gammatone-DSP Fallback (§4.4 Spec).

        Formel:
            1. Gammatone-Filterbank (24 ERB-Kanäle, 50–8000 Hz)
            2. SNR per Kanal basierend auf Energieverhältnis harmonisch/nichharmonisch
            3. Frequenz-gewichteter SNR → MOS via Sigmoid-Mapping
               MOS = 1 + 4 · σ(0.2 · snr_dB − 1.5)
        """
        try:
            n = len(mono)
            if n < 512:
                return VersaResult(mos=3.0, model_used="pqs_dsp_fallback", confidence=0.40)

            # Energie-Schätzung via Gammatone-approximierter Bark-Filterbank
            spec = np.fft.rfft(mono[: min(n, 4 * sr)].astype(np.float64))
            mag = np.abs(spec)
            freqs = np.linspace(0, sr / 2, len(mag))

            # 24 Bark-Bänder
            bark_edges = [
                50,
                100,
                200,
                300,
                400,
                510,
                630,
                770,
                920,
                1080,
                1270,
                1480,
                1720,
                2000,
                2320,
                2700,
                3150,
                3700,
                4400,
                5300,
                6400,
                7700,
                9500,
                12000,
                15500,
            ]
            band_energies: list[float] = []
            for i in range(len(bark_edges) - 1):
                lo, hi = bark_edges[i], bark_edges[i + 1]
                mask = (freqs >= lo) & (freqs < hi)
                if mask.sum() > 0:
                    band_energies.append(float(np.mean(mag[mask] ** 2)))

            if not band_energies:
                return VersaResult(mos=3.0, model_used="pqs_dsp_fallback", confidence=0.40)

            total_e = float(np.mean(band_energies))
            rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
            # dB-normalized SNR: relative to local signal level (not fixed 0.01 floor)
            # This prevents penalizing quiet but high-quality recordings (pianissimo, classical).
            # Noise floor estimated from lowest-energy Bark bands (top 25% quietest).
            _sorted_be = sorted(band_energies)
            _noise_floor_e = float(np.mean(_sorted_be[: max(1, len(_sorted_be) // 4)])) + 1e-12
            _signal_e = float(np.mean(_sorted_be[len(_sorted_be) // 2 :])) + 1e-12
            snr_db = 10.0 * math.log10(_signal_e / _noise_floor_e)
            snr_db = float(np.clip(snr_db, -10.0, 50.0))

            # Frequency-weighted SNR aus Bark-Bändern (mittlere Bänder gewichtet)
            weights = np.array(
                [
                    0.5,
                    0.6,
                    0.8,
                    1.0,
                    1.2,
                    1.4,
                    1.5,
                    1.5,
                    1.4,
                    1.3,
                    1.2,
                    1.1,
                    1.0,
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                    0.4,
                    0.3,
                    0.2,
                    0.1,
                    0.1,
                ][: len(band_energies)]
            )
            w_e = np.array(band_energies[: len(weights)]) * weights
            freq_snr = 20.0 * math.log10(float(np.mean(w_e)) / (total_e + 1e-12) + 1e-5)
            combined_snr = 0.6 * snr_db + 0.4 * float(np.clip(freq_snr + 20, -10, 50))

            # MOS-Mapping: adaptive sigmoid slope based on signal level
            # Quiet signals (rms < -30 dBFS) get gentler slope to avoid floor-clamping
            _rms_db = 20.0 * math.log10(rms)
            _slope = 0.25 if _rms_db > -20.0 else max(0.12, 0.25 + 0.005 * _rms_db)
            z = _slope * combined_snr - 1.5
            sigma = 1.0 / (1.0 + math.exp(-z))
            mos = float(np.clip(1.0 + 4.0 * sigma, 1.0, 5.0))
            if not math.isfinite(mos):
                mos = 3.0

            return VersaResult(mos=mos, model_used="pqs_dsp_fallback", confidence=0.55)
        except Exception as exc:
            logger.error("PQS-DSP Fallback fehlgeschlagen: %s", exc)
            return VersaResult(mos=3.0, model_used="error", confidence=0.0)


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_versa_plugin() -> VersaPlugin:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VersaPlugin()
    return _instance


def get_loaded_versa_plugin() -> VersaPlugin | None:
    """Gibt nur eine bereits geladene VERSA-Instanz zurück, ohne Lazy-Load."""
    return _instance


def score_mos(audio: np.ndarray, sr: int) -> VersaResult:
    """Convenience-Wrapper für get_versa_plugin().score()."""
    return get_versa_plugin().score(audio, sr)
