"""sgmse_plugin — SGMSE+: Score-Based Generative Model for Speech Enhancement.

SGMSE+ (Richter et al. 2022) verwendet stochastische Differentialgleichungen (SDE)
zur Aufhebung von Rausch- und Hallprozessen. Überlegener Nachfolger von WPE für
kombinierte Enhancement + Dereverberation.

Verbesserung gegenüber WPE (2010):
    - SGMSE+ löst das inverse Problem via Score-Matching (p(clean|noisy))
    - Unterstützung breitbandiger 48 kHz Musikrestaurierung
    - Hallunterdrückung UND Rauschreduzierung in einem Schritt

Modell:
    Primär:   models/sgmse_plus/sgmse_plus.ts (~251 MB, TorchScript)
    Input:  [1, 2, n_fft//2+1, T] float32 (Real + Imag getrennt)
    Output: [1, 2, n_fft//2+1, T] float32 (denoised Real + Imag)
    Sigma:  Rauschpegel ∈ [0.01, 1.0] als skalarer Input

Fallback-Kaskade (§4.4):
    1. SGMSE+ TorchScript (dieser Plugin)
    2. WPE DSP (Nara-WPE, wpe_plugin.py)

Backward-Kompatibilität:
    Alle früheren Exporte (WpePlugin, SgmsePlugin, get_wpe_plugin, …)
    bleiben erhalten und zeigen auf wpe_plugin zur Rückwärtskompatibilität.

Referenz:
    Richter et al. "Speech Enhancement and Dereverberation with Diffusion-Based
    Generative Models" — IEEE/ACM TASLP 2022.
    https://github.com/sp-uhh/sgmse

Singleton: get_sgmse_plus_plugin() verwenden.
CPU-Only: CPUExecutionProvider.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_TS_PATH = _ROOT / "models" / "sgmse_plus" / "sgmse_plus.ts"
_CKPT_CANDIDATES = (
    _ROOT / "models" / "sgmse_plus" / "sgmse_plus_src_1.ckpt",
    _ROOT / "models" / "sgmse_plus" / "sgmse_wsj0_reverb.ckpt",
)


def _is_corrupt_torchscript_error(exc: BaseException) -> bool:
    """Erkennt deterministisch defekte/lokal falsche TorchScript-Dateien."""
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "invalid magic number for torchscript archive",
            "failed finding central directory",
            "not a zip archive",
            "constants.pkl",
            "version file not found",
        )
    )


def _quarantine_corrupt_torchscript(path: Path, exc: BaseException) -> Path | None:
    """Verschiebt ein defektes lokales TorchScript aus dem Produktionspfad."""
    if not path.exists():
        return None
    try:
        quarantine = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
        os.replace(path, quarantine)
        logger.warning(
            "SGMSE+: defektes TorchScript isoliert: %s -> %s (%s)",
            path.name,
            quarantine.name,
            str(exc).splitlines()[0],
        )
        return quarantine
    except Exception as quarantine_exc:
        logger.warning(
            "SGMSE+: defektes TorchScript erkannt, aber Quarantäne fehlgeschlagen (%s): %s",
            path,
            quarantine_exc,
        )
        return None


# Verarbeitungs-Konstanten (48 kHz)
_SR: int = 48_000
_DEFAULT_N_FFT: int = 510  # Match the bundled SGMSE checkpoints (256 frequency bins)
_DEFAULT_HOP: int = 128
_DEFAULT_WIN: int = 510

_lock_plus = threading.Lock()
_instance_plus: SGMSEPlusPlugin | None = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SgmseResult:
    """Ergebnis der SGMSE+ Enhancement-Inferenz.

    Attributes:
        audio:      Bereinigtes / Dereverb-Audio, float32 ∈ [-1, 1]
        sr:         Sample-Rate (48000)
        model_used: "sgmse_plus_torchscript" | "wpe_dsp_fallback"
        snr_improvement_db: Geschätzter SNR-Gewinn in dB
    """

    audio: np.ndarray
    sr: int
    model_used: str
    snr_improvement_db: float = 0.0

    def __post_init__(self) -> None:
        """Begrenzt audio to [-1, 1] and replace NaN/Inf."""
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = np.clip(self.audio, -1.0, 1.0)


# ---------------------------------------------------------------------------
# SGMSEPlusPlugin
# ---------------------------------------------------------------------------


class SGMSEPlusPlugin:
    """SGMSE+ Score-Based Speech/Music Enhancement (TorchScript-primary).

    Verarbeitet kombinierte Rausch- und Hallunterdrückung via score-basierter
    generativer Inferenz oder fällt auf WPE DSP zurück (§4.4 Spec).
    """

    FORWARD_TIMEOUT_S: float = 12.0

    def __init__(self) -> None:
        """Initialisiert SGMSE+ plugin and attempt model load."""
        self._session: Any = None
        self._ts_model: Any = None
        self._eager_model: Any = None
        self._eager_backbone: str = ""
        self._num_frames: int = 256
        self._model_loaded: bool = False
        self._n_fft: int = _DEFAULT_N_FFT
        self._hop: int = _DEFAULT_HOP
        self._win: int = _DEFAULT_WIN
        self._device: str = "cpu"  # set by _try_load or _try_load_from_checkpoint
        self._device: str = "cpu"  # type: ignore[no-redef]  # set by _try_load or _try_load_from_checkpoint
        self._load_model_geometry()
        self._try_load()

    def _load_model_geometry(self) -> None:
        """Lädt STFT geometry from bundled SGMSE checkpoint metadata.

        The exported TorchScript backbone is traced on tensors with 256 frequency
        bins. If the plugin uses a different frontend, the U-Net skip-connections
        inside NCSNPP fail deterministically with tensor-size mismatches.
        """
        try:
            import sys  # pylint: disable=import-outside-toplevel

            import torch  # pylint: disable=import-outside-toplevel

            sys.path.insert(0, str((_ROOT / "models" / "sgmse_plus").resolve()))
            for ckpt_path in _CKPT_CANDIDATES:
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(  # nosec B614 — trusted local model bundle
                    ckpt_path, map_location="cpu", weights_only=False
                )  # fairseq checkpoint contains custom objects
                hyper = ckpt.get("hyper_parameters")
                if not isinstance(hyper, dict):
                    continue
                n_fft = int(hyper.get("n_fft") or self._n_fft)
                hop = int(hyper.get("hop_length") or self._hop)
                if n_fft <= 0 or hop <= 0:
                    continue
                self._n_fft = n_fft
                self._hop = hop
                self._win = n_fft
                self._num_frames = int(hyper.get("num_frames") or self._num_frames)
                logger.info(
                    "SGMSE+ geometry from %s: n_fft=%d hop=%d win=%d num_frames=%d",
                    ckpt_path.name,
                    self._n_fft,
                    self._hop,
                    self._win,
                    self._num_frames,
                )
                return
        except Exception as exc:
            logger.debug("SGMSE+ checkpoint geometry unavailable (%s) — using defaults", exc)

    def _try_load(self) -> None:
        """Lädt SGMSE+ TorchScript; sonst WPE-Fallback."""
        try:
            from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                try_allocate as _try_alloc,
            )
        except Exception:
            _try_alloc = None  # type: ignore[assignment]

        if _TS_PATH.exists():
            try:
                import os as _os  # pylint: disable=import-outside-toplevel

                import torch  # pylint: disable=import-outside-toplevel

                torch.set_num_threads(_os.cpu_count() or 4)  # §2.37 CPU-Thread-Budget
                _budget_ok = _try_alloc is None or _try_alloc("SGMSE+", size_gb=0.12)
                if not _budget_ok:
                    try:
                        from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                            release as _rel2,
                        )

                        _rel2("SGMSE+")
                    except Exception:
                        logger.warning("sgmse_plugin.py::_try_load fallback", exc_info=True)
                    _budget_ok = _try_alloc is None or _try_alloc("SGMSE+", size_gb=0.12)
                if not _budget_ok:
                    logger.warning("SGMSE+: ML-Budget erschöpft — WPE-DSP-Fallback.")
                else:
                    try:
                        from backend.core.ml_device_manager import (  # pylint: disable=import-outside-toplevel
                            get_torch_device as _get_dev,
                        )

                        _dev = _get_dev("SGMSE")
                    except Exception:
                        _dev = "cpu"
                    try:
                        self._ts_model = torch.jit.load(str(_TS_PATH), map_location=_dev)  # nosec B614
                        self._ts_model.eval()
                        self._ts_model.to(_dev)
                        self._device = _dev
                    except Exception as _gpu_load_exc:
                        if _is_corrupt_torchscript_error(_gpu_load_exc):
                            _quarantine_corrupt_torchscript(_TS_PATH, _gpu_load_exc)
                            raise
                        if _dev != "cpu":
                            logger.warning("SGMSE+: GPU-Load fehlgeschlagen (%s) — CPU-Retry", _gpu_load_exc)
                            try:
                                self._ts_model = torch.jit.load(str(_TS_PATH), map_location="cpu")  # nosec B614
                            except Exception as _cpu_load_exc:
                                if _is_corrupt_torchscript_error(_cpu_load_exc):
                                    _quarantine_corrupt_torchscript(_TS_PATH, _cpu_load_exc)
                                raise
                            self._ts_model.eval()
                            self._device = "cpu"
                        else:
                            raise
                    self._model_loaded = True
                    logger.info("✅ SGMSE+ TorchScript geladen (%s, device=%s)", _TS_PATH.name, self._device)
                    try:
                        from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                            register_plugin as _reg_plm,
                        )

                        def _unload_sgmse(s: SGMSEPlusPlugin = self) -> None:
                            s._ts_model = None  # pylint: disable=protected-access
                            s._model_loaded = False  # pylint: disable=protected-access

                        _reg_plm("SGMSE+", size_gb=0.12, unload_fn=_unload_sgmse)
                    except Exception as _exc:
                        logger.debug("Plugin operation failed (non-critical): %s", _exc)
                    return
            except Exception as exc:
                logger.warning("SGMSE+ TorchScript nicht ladbar: %s — WPE-DSP-Fallback aktiv.", exc)
                try:
                    from backend.core.ml_memory_budget import release as _rel  # pylint: disable=import-outside-toplevel

                    _rel("SGMSE+")
                except Exception as _exc:
                    logger.debug("Plugin operation failed (non-critical): %s", _exc)

        # Recovery path: keep ML available via checkpoint-backed eager model.
        if self._try_load_from_checkpoint():
            self._model_loaded = True
            return

        logger.info(
            "SGMSE+ Modell nicht verfügbar (TorchScript: %s) — WPE-DSP-Fallback aktiv.",
            _TS_PATH,
        )

    def _try_load_from_checkpoint(self) -> bool:
        """Lädt SGMSE backbone directly from checkpoint as ML recovery path."""
        try:
            import sys  # pylint: disable=import-outside-toplevel

            import torch  # pylint: disable=import-outside-toplevel

            sys.path.insert(0, str((_ROOT / "models" / "sgmse_plus").resolve()))
            from sgmse.backbones import BackboneRegistry  # type: ignore  # pylint: disable=import-outside-toplevel

            for ckpt_path in _CKPT_CANDIDATES:
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(  # nosec B614 — trusted local model bundle
                    ckpt_path, map_location="cpu", weights_only=False
                )  # contains custom Lightning/fairseq objects
                hyper = ckpt.get("hyper_parameters")
                state = ckpt.get("state_dict")
                if not isinstance(hyper, dict) or not isinstance(state, dict):
                    continue
                backbone_name = str(hyper.get("backbone") or "").strip()
                if not backbone_name:
                    continue
                dnn_cls = BackboneRegistry.get_by_name(backbone_name)
                dnn = dnn_cls(**hyper)
                dnn_state = {k[4:]: v for k, v in state.items() if k.startswith("dnn.")}
                if not dnn_state:
                    continue
                dnn.load_state_dict(dnn_state, strict=False)
                dnn.eval()
                try:
                    from backend.core.ml_device_manager import (  # pylint: disable=import-outside-toplevel
                        get_torch_device as _get_dev,
                    )

                    _dev = _get_dev("SGMSE")
                except Exception:
                    _dev = "cpu"
                try:
                    dnn.to(_dev)
                    self._device = _dev
                except Exception as _gpu_to_exc:
                    logger.warning("SGMSE+ Checkpoint: GPU-Placement fehlgeschlagen (%s) — CPU", _gpu_to_exc)
                    dnn.to("cpu")
                    self._device = "cpu"
                self._eager_model = dnn
                self._eager_backbone = backbone_name
                logger.info(
                    "✅ SGMSE+ Checkpoint-ML geladen (%s, backbone=%s, device=%s)",
                    ckpt_path.name,
                    backbone_name,
                    self._device,
                )
                return True
        except Exception as exc:
            logger.warning("SGMSE+ Checkpoint-ML nicht ladbar: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Adaptive chunk size: 10 s when RAM is tight (< 6 GB free), else 30 s.
    # 10 s @ 48 kHz = 480k samples → STFT [1,2,256,3751] with bundled model geometry
    # peak ~300–400 MB.  30 s → ~1 GB peak.  Scales to RAM pressure.
    _MAX_CHUNK_SAMPLES_LARGE: int = 30 * _SR  # 30 s — plenty of RAM
    _MAX_CHUNK_SAMPLES_SMALL: int = 10 * _SR  # 10 s — RAM-tight mode
    # 10 ms Hanning crossfade at chunk boundaries
    _OVERLAP_SAMPLES: int = int(0.01 * _SR)

    def _get_available_ram_gb(self) -> float:
        """Gibt available RAM in GB, or inf if psutil unavailable zurück."""
        try:
            import psutil  # pylint: disable=import-outside-toplevel

            return float(psutil.virtual_memory().available / (1024**3))
        except Exception:
            logger.warning("sgmse_plugin.py::_get_available_ram_gb fallback", exc_info=True)
            return float("inf")

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        sigma: float = 0.5,
        max_runtime_s: float | None = None,
        panns_singing: float = 0.0,  # §0p v10.0.0: Vokal-Mode — konservativeres sigma bei Gesang
    ) -> SgmseResult:
        """Kombinierte Rausch-/Hallunterdrückung via SGMSE+ oder WPE-Fallback.

        Algorithm (TorchScript-Pfad):
            1. STFT → Real/Imag [1, 2, F, T]
            2. SGMSE+ forward: score-basiertes Denoising bei Sigma σ
               (Ornstein–Uhlenbeck SDE: dx = -½βx dt + √β dW, t ∈ [0,1])
            3. ISTFT aus Enhanced Complex Spektrum

        Chunked processing (10–30 s adaptive) to prevent OOM on long files.
        RAM guard: falls < 3 GB verfügbar → WPE-DSP-Fallback sofort.

        §0p Vokal-Mode: panns_singing ≥ 0.35 → sigma auf max. 0.35 begrenzt
        (konservativere Diffusion schützt Formanten und Vibrato-Modulation).

        Args:
            audio:         float32, 48000 Hz, mono oder stereo
            sr:            Sample-Rate (muss 48000 sein)
            sigma:         Rauschpegel-Schätzung ∈ [0.01, 1.0]. Standard 0.5 (adaptiv).
            panns_singing: PANNs Gesangs-Konfidenz. ≥ 0.35 → Vokal-Mode.

        Returns:
            SgmseResult mit bereinigtem Audio.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # §0p Vokal-Mode: konservativeres sigma schützt Formanten und Vibrato-Zonen.
        # panns_singing ≥ 0.35 → sigma_max=0.35; 0.25–0.35 → sigma_max=0.45 (weich).
        if float(panns_singing) >= 0.35:
            sigma = float(min(sigma, 0.35))
            logger.debug("§0p SGMSE+ Vokal-Mode: panns_singing=%.2f → sigma cap 0.35", panns_singing)
        elif float(panns_singing) >= 0.25:
            sigma = float(min(sigma, 0.45))

        # UV3 passes (2, N) channels-first; normalize to (N, 2) for uniform processing.
        _was_channels_first = audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2
        if _was_channels_first:
            audio = audio.T  # (2, N) → (N, 2)
        stereo = audio.ndim == 2 and audio.shape[1] == 2

        # ── RAM guard: < 3 GB available → WPE-DSP-Fallback ──────────
        # Threshold lowered from 4 GB to 3 GB to catch tighter RAM situations.
        # The SGMSE+ TorchScript SDE solver needs ≥1 GB headroom even for 10 s chunks.
        _use_ml = self._ts_model is not None
        if _use_ml:
            _avail_gb = self._get_available_ram_gb()
            if _avail_gb < 3.0:
                logger.warning(
                    "SGMSE+ RAM guard: nur %.1f GB frei (< 3 GB) — WPE-DSP-Fallback",
                    _avail_gb,
                )
                _use_ml = False

        def process_channel(ch: np.ndarray) -> np.ndarray:
            if _use_ml:
                try:
                    from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                        get_plugin_lifecycle_manager as _get_plm,
                    )

                    _plm = _get_plm()
                    _plm.set_active("SGMSE+", True)
                except Exception:
                    _plm = None
                try:
                    return self._enhance_chunked(ch, sigma, max_runtime_s=max_runtime_s)
                finally:
                    try:
                        if _plm is not None:
                            _plm.set_active("SGMSE+", False)
                    except Exception:
                        logger.warning("sgmse_plugin.py::process_channel fallback", exc_info=True)
            return self._wpe_fallback(ch, sr)

        if stereo:
            left = process_channel(audio[:, 0])
            right = process_channel(audio[:, 1])
            n = min(len(left), len(right), len(audio))
            out = np.stack([left[:n], right[:n]], axis=1)
        else:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            out = process_channel(mono)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        rms_in = float(np.sqrt(np.mean(audio**2))) + 1e-10
        rms_diff = float(np.sqrt(np.mean((out - audio) ** 2))) + 1e-10
        snr_imp = 20.0 * math.log10(rms_in / rms_diff) if rms_diff < rms_in else 0.0

        # Restore channels-first layout if input was (2, N)
        out_final = out.T if (_was_channels_first and out.ndim == 2) else out

        return SgmseResult(
            audio=out_final.astype(np.float32),
            sr=sr,
            model_used=("sgmse_plus_torchscript" if _use_ml else "wpe_dsp_fallback"),
            snr_improvement_db=float(np.clip(snr_imp, 0.0, 30.0)),
        )

    # ------------------------------------------------------------------
    # Chunked inference wrapper  (prevents OOM on long files)
    # ------------------------------------------------------------------

    def _enhance_chunked(self, mono: np.ndarray, sigma: float, max_runtime_s: float | None = None) -> np.ndarray:
        """Verarbeitet mono audio in adaptive chunks (10–30 s) with Hanning crossfade.

        Adapts chunk size to available RAM:
        - ≥ 6 GB free → 30 s chunks (~1 GB peak per chunk)
        - < 6 GB free → 10 s chunks (~300 MB peak per chunk)
        Falls back to WPE if RAM drops below 2 GB between chunks.
        """
        import gc  # pylint: disable=import-outside-toplevel

        n_total = len(mono)

        # Adaptive chunk sizing based on available RAM
        _avail = self._get_available_ram_gb()
        if _avail < 6.0:
            chunk_len = self._MAX_CHUNK_SAMPLES_SMALL  # 10 s
            logger.info("SGMSE+ adaptive chunk: 10 s (%.1f GB frei)", _avail)
        else:
            chunk_len = self._MAX_CHUNK_SAMPLES_LARGE  # 30 s

        if n_total <= chunk_len:
            return self._enhance_torchscript(mono, sigma)

        overlap = self._OVERLAP_SAMPLES
        step = chunk_len - overlap
        out = np.zeros(n_total, dtype=np.float32)
        fade_in = np.hanning(2 * overlap)[:overlap].astype(np.float32)
        fade_out = np.hanning(2 * overlap)[overlap:].astype(np.float32)

        # Minimum RAM headroom required before each chunk.
        # 30 s chunks peak ~1 GB in U-Net forward pass; add 2.5 GB safety margin
        # for glibc heap fragmentation and OS memory pressure.
        # 10 s chunks peak ~300 MB; 1.5 GB safety margin is sufficient.
        _HEADROOM_LARGE = 3.5  # GB needed before a 30 s chunk
        _HEADROOM_SMALL = 2.0  # GB needed before a 10 s chunk

        pos = 0
        chunk_idx = 0
        _t0 = time.perf_counter()
        _audio_s = max(1e-6, n_total / float(_SR))
        # Runtime budget: keep ML path for quality, but cap pathological tails.
        # CPU TorchScript on long material can be much slower than 2x real-time,
        # so the default budget must allow several chunks before fallback.
        # Budget default: 6x Echtzeit, min 180 s, max 1200 s (20 min) per channel.
        _base_budget_s = float(np.clip(6.0 * _audio_s, 180.0, 1200.0))
        if max_runtime_s is not None:
            # Caller-provided budget is honoured exactly (upper cap only to avoid unbounded runs).
            # Lower bound intentionally removed so callers (e.g. unit tests) can use tiny budgets.
            _runtime_budget_s = float(min(float(max_runtime_s), 1800.0))
        else:
            _runtime_budget_s = _base_budget_s
        while pos < n_total:
            end = min(pos + chunk_len, n_total)

            # ── Runtime guard: abort pathological slowdowns, keep partial ML result ──
            _elapsed = time.perf_counter() - _t0
            if chunk_idx > 0:
                _avg_chunk_s = _elapsed / float(chunk_idx)
                _remaining_chunks = max(0, int(np.ceil((n_total - pos) / max(1, step))))
                _projected_total_s = _elapsed + _avg_chunk_s * _remaining_chunks
                if _projected_total_s > _runtime_budget_s:
                    logger.warning(
                        "SGMSE+ runtime guard: projected %.1fs > budget %.1fs after %d chunks — rest via WPE-fallback",
                        _projected_total_s,
                        _runtime_budget_s,
                        chunk_idx,
                    )
                    rest = self._wpe_fallback(mono[pos:], _SR)
                    rest_len = min(len(rest), n_total - pos)
                    out[pos : pos + rest_len] = rest[:rest_len]
                    break

            if _elapsed > _runtime_budget_s:
                logger.warning(
                    "SGMSE+ runtime guard: elapsed %.1fs > budget %.1fs at chunk %d — rest via WPE-fallback",
                    _elapsed,
                    _runtime_budget_s,
                    chunk_idx + 1,
                )
                rest = self._wpe_fallback(mono[pos:], _SR)
                rest_len = min(len(rest), n_total - pos)
                out[pos : pos + rest_len] = rest[:rest_len]
                break

            # ── Pre-chunk RAM guard ───────────────────────────────────
            # Proactively free pages before allocating the next chunk to
            # give psutil an accurate picture of truly available RAM.
            gc.collect()
            try:
                import ctypes as _ct_pre  # pylint: disable=import-outside-toplevel

                _ct_pre.CDLL("libc.so.6").malloc_trim(0)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            _avail_pre = self._get_available_ram_gb()
            _headroom_needed = _HEADROOM_LARGE if chunk_len > self._MAX_CHUNK_SAMPLES_SMALL else _HEADROOM_SMALL
            if _avail_pre < _headroom_needed:
                logger.warning(
                    "SGMSE+ pre-chunk %d: %.1f GB frei < %.1f GB Headroom — WPE-Fallback für Rest (%.1f s)",
                    chunk_idx + 1,
                    _avail_pre,
                    _headroom_needed,
                    (n_total - pos) / _SR,
                )
                rest = self._wpe_fallback(mono[pos:], _SR)
                rest_len = min(len(rest), n_total - pos)
                out[pos : pos + rest_len] = rest[:rest_len]
                break

            chunk = mono[pos:end]
            enhanced = self._enhance_torchscript(chunk, sigma)

            if len(enhanced) < len(chunk):
                enhanced = np.pad(enhanced, (0, len(chunk) - len(enhanced)))
            elif len(enhanced) > len(chunk):
                enhanced = enhanced[: len(chunk)]

            if chunk_idx == 0:
                # First chunk: no fade-in, apply fade-out in overlap zone
                if end < n_total and len(enhanced) > overlap:
                    enhanced[-overlap:] *= fade_out
                out[pos:end] = enhanced
            else:
                # Apply fade-in at start of this chunk
                if len(enhanced) > overlap:
                    enhanced[:overlap] *= fade_in
                # Apply fade-out at end (unless last chunk)
                if end < n_total and len(enhanced) > overlap:
                    enhanced[-overlap:] *= fade_out
                # Overlap-add in crossfade zone
                out[pos : pos + overlap] += enhanced[:overlap]
                out[pos + overlap : end] = enhanced[overlap:]

            chunk_idx += 1
            pos += step

            # ── Aggressive memory cleanup between chunks ─────────────
            del chunk, enhanced
            gc.collect()
            # malloc_trim: return freed heap pages to OS immediately
            try:
                import ctypes as _ct  # pylint: disable=import-outside-toplevel

                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

            # RAM check between chunks — adaptive: shrink or bail out
            _avail_now = self._get_available_ram_gb()
            if _avail_now < 1.5:
                logger.warning(
                    "SGMSE+ chunk %d: nur %.1f GB frei (< 1.5 GB) — rest via WPE-Fallback",
                    chunk_idx,
                    _avail_now,
                )
                # Fill remaining with WPE fallback
                if pos < n_total:
                    rest = self._wpe_fallback(mono[pos:], _SR)
                    rest_len = min(len(rest), n_total - pos)
                    out[pos : pos + rest_len] = rest[:rest_len]
                break
            elif _avail_now < 4.0 and chunk_len > self._MAX_CHUNK_SAMPLES_SMALL:
                # Downgrade to smaller chunks for remaining audio
                chunk_len = self._MAX_CHUNK_SAMPLES_SMALL
                step = chunk_len - overlap
                logger.info("SGMSE+ RAM dropping (%.1f GB) — switching to 10 s chunks", _avail_now)

        logger.info(
            "SGMSE+ chunked: %d chunks (adaptive) für %.1f s Audio, %.1f GB frei (elapsed=%.1fs, budget=%.1fs)",
            chunk_idx,
            n_total / _SR,
            self._get_available_ram_gb(),
            time.perf_counter() - _t0,
            _runtime_budget_s,
        )
        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # ONNX Inference (SGMSE+ deterministic forward pass @ optimal sigma)
    # ------------------------------------------------------------------

    def _stft(self, mono: np.ndarray) -> tuple[np.ndarray, int]:
        """STFT → Complex Spectrogram."""
        from scipy.signal import stft as scipy_stft  # pylint: disable=import-outside-toplevel

        n_orig = len(mono)
        # §2.57: Clamp noverlap so it is always < min(nperseg, signal_length).
        # scipy auto-reduces nperseg to signal_length for short chunks, which
        # makes the fixed noverlap >= effective nperseg → ValueError.
        _noverlap = max(0, min(self._win - self._hop, n_orig - 1))
        _, _, Z = scipy_stft(
            mono.astype(np.float64),
            fs=_SR,
            nperseg=self._win,
            noverlap=_noverlap,
            window="hann",
            boundary="zeros",
        )
        return Z.astype(np.complex64), n_orig

    def _istft(self, Z: np.ndarray, n_orig: int) -> np.ndarray:
        """Inverse STFT."""
        from scipy.signal import istft as scipy_istft  # pylint: disable=import-outside-toplevel

        _noverlap = max(0, min(self._win - self._hop, n_orig - 1))
        _, x = scipy_istft(
            Z.astype(np.complex128),
            fs=_SR,
            nperseg=self._win,
            noverlap=_noverlap,
            window="hann",
            boundary="zeros",
        )
        x = x.astype(np.float32)
        if len(x) > n_orig:
            x = x[:n_orig]
        elif len(x) < n_orig:
            x = np.pad(x, (0, n_orig - len(x)))
        return x  # type: ignore[no-any-return]

    def _enhance_onnx(self, mono: np.ndarray, sigma: float) -> np.ndarray:
        """SGMSE+ ONNX-Inferenz: Score-Based Enhancement."""
        assert self._session is not None
        try:
            Z, n_orig = self._stft(mono)
            # SGMSE+ input: [1, 2, F, T] — Real und Imag als separate Kanäle
            real_c = Z.real[np.newaxis, np.newaxis].astype(np.float32)
            imag_c = Z.imag[np.newaxis, np.newaxis].astype(np.float32)
            inp = np.concatenate([real_c, imag_c], axis=1)  # [1, 2, F, T]

            # ── Pad time dimension to multiple of _UNET_ALIGN ──
            T_orig = inp.shape[3]
            T_pad = (self._UNET_ALIGN - T_orig % self._UNET_ALIGN) % self._UNET_ALIGN
            if T_pad > 0:
                inp = np.pad(inp, ((0, 0), (0, 0), (0, 0), (0, T_pad)), mode="constant")

            # Sigma als skalarer Input (falls Modell diesen Eingang erwartet)
            input_names = [i.name for i in self._session.get_inputs()]
            feed: dict[str, np.ndarray] = {input_names[0]: inp}
            if len(input_names) > 1:
                feed[input_names[1]] = np.array([[[[sigma]]]], dtype=np.float32)

            ort_out = self._session.run(None, feed)
            out_arr = np.asarray(ort_out[0], dtype=np.float32)  # [1, 2, F, T]

            # ── Remove time padding ──
            if T_pad > 0:
                out_arr = out_arr[:, :, :, :T_orig]

            out_real = out_arr[0, 0]
            out_imag = out_arr[0, 1] if out_arr.shape[1] >= 2 else np.zeros_like(out_real)

            Z_enhanced = (out_real + 1j * out_imag).astype(np.complex64)
            Z_enhanced = np.nan_to_num(Z_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

            result = self._istft(Z_enhanced, n_orig)
            return np.clip(np.nan_to_num(result, nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("SGMSE+ ONNX-Inferenzfehler: %s — WPE-Fallback.", exc)
            return self._wpe_fallback(mono, _SR)

    # NCSNPP U-Net has ~5 downsampling steps → time dim must be multiple of 2^5=32
    # to avoid encoder/decoder skip-connection shape mismatch (torch.cat RuntimeError).
    _UNET_ALIGN: int = 32

    def _enhance_torchscript(self, mono: np.ndarray, sigma: float) -> np.ndarray:
        """SGMSE+ TorchScript-Inferenz: score [B,2,F,T] aus STFT-Features.

        Memory-optimized: avoids redundant array copies, explicitly deletes
        intermediates, and calls gc.collect + malloc_trim after inference.
        Time-dimension is zero-padded to a multiple of _UNET_ALIGN to prevent
        encoder/decoder skip-connection shape mismatches in NCSNPP.
        """
        # Capture model refs atomically — PLM may set _ts_model/eager_model to None
        # between the None-check and the lambda invocation in _run_with_timeout (race cond.)
        _ts = self._ts_model
        _eager = self._eager_model
        if _ts is None and _eager is None:
            return self._wpe_fallback(mono, _SR)
        # Minimum-Input-Guard: NCSNPP U-Net 4×4 kernel requires STFT freq/time dims > 4.
        # Short segments produce too few STFT bins → RuntimeError:
        # "Kernel size can't be greater than actual input size (3×130)".
        _MIN_MONO_SAMPLES = _DEFAULT_N_FFT * 8  # 510*8 = 4080 ≈ 85 ms @ 48 kHz
        if len(mono) < _MIN_MONO_SAMPLES:
            logger.debug("SGMSE+: Segment zu kurz (%d < %d samples) → WPE-Fallback", len(mono), _MIN_MONO_SAMPLES)
            return self._wpe_fallback(mono, _SR)
        try:
            import gc  # pylint: disable=import-outside-toplevel

            import torch  # pylint: disable=import-outside-toplevel

            Z, n_orig = self._stft(mono)
            # Build [1, 2, F, T] tensor directly — no .copy() for y (same input)
            x_t = np.stack([Z.real, Z.imag], axis=0)[np.newaxis].astype(np.float32)
            del Z  # free complex spectrogram immediately (~44 MB per 30s)

            F_orig = x_t.shape[2]
            T_orig = x_t.shape[3]
            target_frames = max(32, int(self._num_frames))
            # Guard oversized frame windows from checkpoint metadata to prevent
            # single forward calls becoming unbounded on CPU.
            if target_frames > 256:
                logger.info("SGMSE+ frame cap: num_frames=%d -> 256", target_frames)
                target_frames = 256

            # CPU safety: reduce overlap for short clips to avoid test-timeout stalls
            # in TorchScript inference while keeping overlap-add for longer material.
            if T_orig <= target_frames * 4:
                step = target_frames
            else:
                step = max(16, target_frames // 2)

            # Process with model-native frame windows and overlap-add in STFT domain.
            out_acc = np.zeros((1, 2, F_orig, T_orig), dtype=np.float32)
            w_acc = np.zeros((1, 1, 1, T_orig), dtype=np.float32)
            starts = list(range(0, max(1, T_orig), step))
            if starts[-1] != max(0, T_orig - target_frames):
                starts.append(max(0, T_orig - target_frames))

            with torch.no_grad():
                t_t = torch.tensor([float(sigma)], dtype=torch.float32).to(self._device)

                def _identity_pin(x):
                    return x

                _pin_fn = _identity_pin
                try:
                    from backend.core.ml_device_manager import (  # pylint: disable=import-outside-toplevel
                        get_ml_device_manager as _get_mdm,
                    )

                    _pin_fn = _get_mdm().pin_tensor_rocm
                except Exception:
                    logger.warning("sgmse_plugin.py::_identity_pin fallback", exc_info=True)
                for s in starts:
                    e = min(s + target_frames, T_orig)
                    seg = x_t[:, :, :, s:e]
                    seg_len = seg.shape[3]
                    if seg_len < target_frames:
                        seg = np.pad(seg, ((0, 0), (0, 0), (0, 0), (0, target_frames - seg_len)), mode="constant")

                    _pinned = _pin_fn(seg)
                    xt_t = (_pinned if isinstance(_pinned, torch.Tensor) else torch.from_numpy(_pinned)).to(
                        self._device
                    )
                    y_t = xt_t
                    if _ts is not None:
                        out_t = self._run_with_timeout(
                            lambda xt_t=xt_t, y_t=y_t, t_t=t_t, _m=_ts: _m(xt_t, y_t, t_t),
                            timeout_s=self.FORWARD_TIMEOUT_S,
                        )
                    else:
                        if self._eager_backbone == "ncsnpp_v2":
                            out_t = self._run_with_timeout(
                                lambda xt_t=xt_t, y_t=y_t, t_t=t_t, _m=_eager: _m(xt_t, y_t, t_t),
                                timeout_s=self.FORWARD_TIMEOUT_S,
                            )
                        else:
                            x_complex = torch.complex(xt_t[:, 0], xt_t[:, 1])
                            y_complex = torch.complex(y_t[:, 0], y_t[:, 1])
                            dnn_input = torch.stack([x_complex, y_complex], dim=1)
                            out_complex = -self._run_with_timeout(
                                lambda dnn_input=dnn_input, t_t=t_t, _m=_eager: _m(dnn_input, t_t),
                                timeout_s=self.FORWARD_TIMEOUT_S,
                            )
                            out_t = torch.stack([out_complex.real, out_complex.imag], dim=1)

                    out_seg = out_t.detach().cpu().numpy().astype(np.float32)
                    # SGMSE+ TorchScript wrapper may return [B,2,1,F,T] (5D)
                    # instead of [B,2,F,T] (4D) — squeeze extra dim if present.
                    while out_seg.ndim > 4:
                        out_seg = out_seg.squeeze(2)
                    out_seg = out_seg[:, :, :F_orig, :seg_len]
                    if seg_len > 1:
                        w = np.hanning(seg_len).astype(np.float32)
                        if float(np.sum(w)) <= 1e-6:
                            w = np.ones(seg_len, dtype=np.float32)
                    else:
                        w = np.ones(1, dtype=np.float32)
                    out_acc[:, :, :, s:e] += out_seg * w[np.newaxis, np.newaxis, np.newaxis, :]
                    w_acc[:, :, :, s:e] += w[np.newaxis, np.newaxis, np.newaxis, :]

            out_arr = out_acc / np.maximum(w_acc, 1e-6)

            del x_t

            out_real = out_arr[0, 0]
            out_imag = out_arr[0, 1] if out_arr.shape[1] > 1 else np.zeros_like(out_real)
            del out_arr  # free full output array, keep only slices

            Z_enhanced = (out_real + 1j * out_imag).astype(np.complex64)
            del out_real, out_imag
            Z_enhanced = np.nan_to_num(Z_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
            result = self._istft(Z_enhanced, n_orig)
            del Z_enhanced

            # §v10.340 Polarity-Guard (do-no-harm): TorchScript/eager-Checkpoints
            # können global invertierte Ausgaben liefern (Score-Negation oder
            # abweichende STFT-Phasenkonvention). Deterministischer
            # Korrelationscheck gegen den Eingang — bei Antikorrelation wird
            # das Vorzeichen korrigiert. Korrekte Modelle sind unberührt.
            _res_ok = min(n_orig, len(result))
            _mono_ref = mono[:_res_ok]
            _mono_ref = _mono_ref - np.mean(_mono_ref)
            _res_ref = np.asarray(result)[:_res_ok] - np.mean(np.asarray(result)[:_res_ok])
            if float(np.dot(_mono_ref, _res_ref)) < 0.0:
                logger.warning("SGMSE+ Polarity-Guard: invertierte Ausgabe erkannt — Vorzeichen korrigiert")
                result = -np.asarray(result)

            # Aggressive cleanup: GC + malloc_trim to return memory to OS
            gc.collect()
            try:
                import ctypes as _ct  # pylint: disable=import-outside-toplevel

                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

            return np.clip(np.nan_to_num(result, nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
        except Exception as exc:
            # Detect Torch / system OOM and re-raise as Python MemoryError so
            # UV3's §2.39 OOM-Recovery-Checkpoint handler can fire instead of
            # silently swallowing the error and calling WPE (which may also OOM).
            _exc_str = str(exc).lower()
            _is_oom = any(kw in _exc_str for kw in ("not enough memory", "malloc", "out of memory", "allocat", "oom"))
            if _is_oom:
                logger.error(
                    "SGMSE+ TorchScript OOM erkannt: %s — re-raise als MemoryError für UV3-Checkpoint",
                    exc,
                )
                raise MemoryError(f"SGMSE+ Torch-OOM: {exc}") from exc
            if self._device != "cpu":
                logger.warning("SGMSE+: GPU-Inferenz fehlgeschlagen (%s) — CPU-Retry", exc)
                try:
                    for _m in (self._ts_model, self._eager_model):
                        if _m is not None:
                            _m.cpu()
                    self._device = "cpu"
                    try:
                        # pylint: disable-next=import-outside-toplevel
                        from backend.core.ml_device_manager import get_ml_device_manager as _mgr  # isort: skip

                        _mgr().report_gpu_error("SGMSE", exc)
                    except Exception:
                        logger.warning("sgmse_plugin.py::unknown fallback", exc_info=True)
                except Exception as _mv_exc:
                    logger.debug("SGMSE+ GPU→CPU move fehlgeschlagen: %s", _mv_exc)
                    self._device = "cpu"
                return self._enhance_torchscript(mono, sigma)
            logger.warning("SGMSE+ TorchScript-Inferenzfehler: %s — WPE-Fallback.", exc)
            return self._wpe_fallback(mono, _SR)

    def _run_with_timeout(self, fn: Any, timeout_s: float) -> Any:
        """Führt aus: callable with hard wall-clock timeout in a daemon thread."""
        box: dict[str, Any] = {}
        err: dict[str, Exception] = {}

        def _worker() -> None:
            try:
                box["value"] = fn()
            except Exception as exc:
                err["error"] = exc

        thread = threading.Thread(target=_worker, name="sgmse-forward", daemon=True)
        thread.start()
        thread.join(max(0.001, float(timeout_s)))
        if thread.is_alive():
            raise TimeoutError(f"SGMSE+ forward timeout after {timeout_s:.2f}s")
        if "error" in err:
            raise err["error"]
        return box.get("value")

    # ------------------------------------------------------------------
    # WPE DSP Fallback
    # ------------------------------------------------------------------

    def _wpe_fallback(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """WPE-Dereverberation als Fallback (wpe_plugin, §4.4)."""
        try:
            from plugins.wpe_plugin import (  # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported
                get_wpe_plugin,
            )

            plugin = get_wpe_plugin()
            result = plugin.enhance(mono, sr)
            if hasattr(result, "audio"):
                return np.clip(np.nan_to_num(result.audio.flatten(), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
            # Legacy: result ist ndarray
            arr = np.asarray(result, dtype=np.float32).flatten()
            return np.clip(np.nan_to_num(arr, nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.error("WPE-Fallback fehlgeschlagen: %s — Audio unverändert.", exc)
            return np.clip(np.nan_to_num(mono.copy(), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_sgmse_plus_plugin() -> SGMSEPlusPlugin:
    """Thread-sicherer Singleton-Accessor für SGMSE+."""
    global _instance_plus  # pylint: disable=global-statement
    if _instance_plus is None:
        with _lock_plus:
            if _instance_plus is None:
                _instance_plus = SGMSEPlusPlugin()
    return _instance_plus


def get_loaded_sgmse_plus_plugin() -> SGMSEPlusPlugin | None:
    """Gibt nur eine bereits geladene SGMSE+-Instanz zurück, ohne Lazy-Load."""
    return _instance_plus


def enhance_sgmse(audio: np.ndarray, sr: int, sigma: float = 0.5) -> SgmseResult:
    """Convenience-Wrapper für get_sgmse_plus_plugin().enhance()."""
    return get_sgmse_plus_plugin().enhance(audio, sr, sigma)


# ---------------------------------------------------------------------------
# Backward-Kompatibilität: WPE-Exporte aus wpe_plugin re-exportieren
# ---------------------------------------------------------------------------
from plugins.wpe_plugin import (  # pylint: disable=wrong-import-position
    SGMSEPlugin,
    SgmsePlugin,
    WpePlugin,
    enhance,
    get_sgmse_plugin,
    get_wpe_plugin,
)

__all__ = [
    "SGMSEPlugin",
    # Neue SGMSE+-Implementierung
    "SGMSEPlusPlugin",
    "SgmsePlugin",
    "SgmseResult",
    # Backward-Kompatibilität (WPE-Basis)
    "WpePlugin",
    "enhance",
    "enhance_sgmse",
    "get_sgmse_plugin",
    "get_sgmse_plus_plugin",
    "get_wpe_plugin",
]
