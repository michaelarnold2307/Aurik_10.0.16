"""Aurik 10.0.0 — Vocos Neural Vocoder Plugin

PRIMÄRER Vocoder: Vocos 0.1.0 (Siuzdak 2023, MIT)
Modell-Kaskade (§4.4 SOTA-Matrix, 3-Tier):
  1. Vocos ONNX 48 kHz nativ (models/vocos_48khz/vocos_48khz.onnx) — kein Resampling!
  2. Vocos ONNX 44.1 kHz     (vocos_mel_spec_44khz.onnx, volles Spektrum bis 22 kHz)
  3. Vocos ONNX 24 kHz       (vocos_mel_spec_24khz.onnx, 52 MB Release-Bundle)
  4. BigVGAN v2  (bigvgan_v2_plugin)                — neuronaler Fallback Stufe 1.5
  5. HiFi-GAN    (hifigan_plugin.HifiGanPlugin)     — neuronaler Fallback
  6. Griffin-Lim+ ≥ 32 Iterationen (DSP-Letzfall)  — nur wenn alle Modelle fehlen

Out-of-the-Box: Kein Download beim ersten Start.
"""

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy.signal import istft, resample_poly, stft

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Öffentliche SR-Konstanten (von Tests und externen Modulen verwendet)
# ---------------------------------------------------------------------------
MEL_SR_22K: int = 22_050  # Vocos 22 kHz Variante
MEL_SR_44K: int = 44_100  # Vocos 44 kHz Variante
AURIK_SR: int = 48_000  # Standard-Aurik-Arbeits-SR

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Priority: 48 kHz nativ (kein Resampling) → 44.1 kHz (volles Spektrum) → 24 kHz (Release-Bundle)
_MODEL_48K = os.path.join(_ROOT, "models", "vocos_48khz", "vocos_48khz.onnx")
_MODEL_44K = os.path.join(_ROOT, "models", "vocos", "vocos_mel_spec_44khz.onnx")
_MODEL_24K = os.path.join(_ROOT, "models", "vocos", "vocos_mel_spec_24khz.onnx")
_MODEL = _MODEL_44K  # für Rückwärtskompatibilität (bevorzugte Auflösung)
# Mel-Parameter je Modell-SR
_MEL_SR_48K = 48_000
_MEL_SR_44K = 44_100
_MEL_SR_24K = 24_000
_MEL_SR = _MEL_SR_24K  # Standardwert (wird zur Laufzeit angepasst)
_N_MELS_48K = 128
_N_MELS_44K = 128
_N_MELS_24K = 100
_N_MELS = _N_MELS_24K
_N_FFT_48K = 2048
_N_FFT_44K = 2048
_N_FFT_24K = 1024
_N_FFT = _N_FFT_24K
_HOP_48K = 256
_HOP_44K = 512
_HOP_24K = 256
_HOP = _HOP_24K
_WIN_48K = 2048
_WIN_44K = 2048
_WIN_24K = 1024
_WIN = _WIN_24K


# ---------------------------------------------------------------------------
# VocosResult — Ergebnis-Datenklasse (§3.6 Aurik Spec)
# ---------------------------------------------------------------------------
@dataclass
class VocosResult:
    """Ergebnis des Vocos-Vocoders."""

    audio: np.ndarray  # Synthetisiertes Audio, float32 ∈ [-1, 1]
    sr: int  # Ausgabe-Samplerate
    pqs_mos: float  # Perceptual Quality Score MOS ∈ [1.0, 5.0]
    model_used: str  # z.B. "vocos_onnx", "hifigan_fallback", "istft_phase_coherent_fallback"
    confidence: float  # Modell-Konfidenz ∈ [0, 1]
    mel_snr_db: float = 0.0  # Mel-Spektrogramm-SNR in dB
    model_sr: int = 0  # Interne Modell-SR (z.B. 24000)
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "sr": self.sr,
            "pqs_mos": self.pqs_mos,
            "model_used": self.model_used,
            "confidence": self.confidence,
            "mel_snr_db": self.mel_snr_db,
            "model_sr": self.model_sr,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# VocosPlugin
# ---------------------------------------------------------------------------
class VocosPlugin:
    """Vocos Neural Vocoder — Aurik 10.0.0 Implementierung.

    Modell-Kaskade (§4.4 SOTA-Matrix, 3-Tier):
        1. Vocos ONNX 48 kHz nativ — vocos_48khz/vocos_48khz.onnx (kein Resampling!)
        2. Vocos ONNX 44.1 kHz     — vocos_mel_spec_44khz.onnx
        3. Vocos ONNX 24 kHz       — vocos_mel_spec_24khz.onnx (Release-Bundle)
        4. BigVGAN v2  — bigvgan_v2_plugin.BigVGANv2Plugin (Stufe 1.5)
        5. HiFi-GAN    — hifigan_plugin.HifiGanPlugin.reconstruct()
        6. Griffin-Lim+ ≥ 32 It. — DSP-Letzfall (nur wenn alle Modelle fehlen)

    ONNX provider policy:
        Uses ml_device_manager.get_ort_providers("Vocos").
        GPU provider is preferred when supported; CPU remains mandatory fallback.

    Singleton-Pattern: get_vocos_plugin() verwenden.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._prefer_sr: int = _MEL_SR_48K  # 48 kHz nativ bevorzugt (kein Resampling)
        self._model_sr: int = _MEL_SR_24K
        self._mel_n_mels: int = _N_MELS_24K
        self._mel_n_fft: int = _N_FFT_24K
        self._mel_hop: int = _HOP_24K
        self._mel_win: int = _WIN_24K
        self._vocos_pypi = None
        self._onnx_session = None
        self._model_loaded: bool = False
        self._fallback_mode: str = "istft_phase_coherent_fallback"
        if model_path:
            self._try_load(model_path)
        else:
            # 48 kHz nativ zuerst (kein Resampling), dann 44.1 kHz, dann 24 kHz
            if os.path.exists(_MODEL_48K):
                self._try_load(_MODEL_48K)
            elif os.path.exists(_MODEL_44K):
                self._try_load(_MODEL_44K)
            elif os.path.exists(_MODEL_24K):
                self._try_load(_MODEL_24K)
            else:
                logger.warning("Vocos ONNX fehlt (48 kHz + 44 kHz + 24 kHz) — Griffin-Lim-Ersatzpfad.")

    def _try_load(self, path: str) -> None:
        """Versucht ONNX-Modell zu laden; setzt Fallback bei Fehler."""
        if not os.path.exists(path):
            logger.warning("Vocos ONNX fehlt: %s — Griffin-Lim-Ersatzpfad.", path)
            return
        # ── ML-Budget-Check VOR dem Laden (§5.1 OOM-Schutz) ──────────────────
        _allocated = False
        try:
            from backend.core.ml_memory_budget import release as _release
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate("Vocos", size_gb=0.12):
                try:
                    _release("Vocos")
                except Exception:
                    logger.warning("vocos_plugin.py::_try_laden Ersatzpfad", exc_info=True)
                if not try_allocate("Vocos", size_gb=0.12):
                    logger.warning("Vocos: ML-Grenze erschöpft — Griffin-Lim-Ersatzpfad")
                    return
            _allocated = True
        except ImportError as _imp_exc:
            logger.debug("Vocos: ml_memory_Grenze nicht verfügbar, Grenze-Pruefung übersprungen: %s", _imp_exc)
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                _providers = _get_prov("Vocos")
            except Exception:
                _providers = ["CPUExecutionProvider"]

            self._onnx_session = ort.InferenceSession(path, sess_options=opts, providers=_providers)
            self._model_loaded = True
            self._fallback_mode = "vocos_onnx"
            # Mel-Parameter je Modell-SR anpassen (48 kHz zuerst prüfen!)
            bname = os.path.basename(path)
            if "48" in bname:
                self._model_sr = _MEL_SR_48K
                self._mel_n_mels = _N_MELS_48K
                self._mel_n_fft = _N_FFT_48K
                self._mel_hop = _HOP_48K
                self._mel_win = _WIN_48K
                logger.info("Vocos 48 kHz ONNX geladen (nativ, kein Resampling): %s [providers=%s]", path, _providers)
            elif "44" in bname:
                self._model_sr = _MEL_SR_44K
                self._mel_n_mels = _N_MELS_44K
                self._mel_n_fft = _N_FFT_44K
                self._mel_hop = _HOP_44K
                self._mel_win = _WIN_44K
                logger.info(
                    "Vocos 44.1 kHz ONNX geladen: %s (volles Spektrum bis 22 kHz) [providers=%s]", path, _providers
                )
            else:
                self._model_sr = _MEL_SR_24K
                self._mel_n_mels = _N_MELS_24K
                self._mel_n_fft = _N_FFT_24K
                self._mel_hop = _HOP_24K
                self._mel_win = _WIN_24K
                logger.info("Vocos 24 kHz ONNX geladen: %s [providers=%s]", path, _providers)
            # ── PLM-Registrierung (LRU-Tracking, §5.1 OOM-Schutz) ─────────────────
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

                get_plugin_lifecycle_manager().register("Vocos", size_gb=0.12, unload_fn=self._do_unload)
            except ImportError as _plm_exc:
                logger.debug("Vocos: PluginLifecycleManager nicht verfügbar, LRU-Tracking deaktiviert: %s", _plm_exc)
        except Exception as exc:
            logger.warning("Vocos ONNX Fehler: %s — Griffin-Lim-Ersatzpfad.", exc)
            if _allocated:
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release("Vocos")
                except ImportError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_backend(self) -> str:
        """Name des aktiven Backends."""
        return self._fallback_mode

    @property
    def model_loaded(self) -> bool:
        """True wenn ein neuronales Modell geladen wurde."""
        return self._model_loaded

    def _do_unload(self) -> None:
        """Entlädt das ONNX-Modell (PLM-Callback, §5.1 OOM-Schutz)."""
        self._onnx_session = None
        self._model_loaded = False
        self._fallback_mode = "istft_phase_coherent_fallback"
        try:
            from backend.core.ml_memory_budget import release as _ml_release

            _ml_release("Vocos")
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # ------------------------------------------------------------------
    # Statische Hilfsmethoden (auch ohne Instanz nutzbar, §3.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Resampelt Audio von src_sr auf dst_sr (resample_poly, Lanczos-artig).

        Invariante: Ausgabe ist float32, NaN/Inf-frei.
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if src_sr == dst_sr:
            return audio
        g = gcd(src_sr, dst_sr)
        result = resample_poly(audio, dst_sr // g, src_sr // g)
        return np.nan_to_num(result.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _build_mel_filterbank(n_mels: int, n_freq_bins: int, sr: int, _n_fft: int) -> np.ndarray:
        """Baut Mel-Filterbank [n_mels, n_freq_bins], float32, non-negative.

        Formel: Hz ↔ Mel via 2595 * log10(1 + f/700)
        """
        hz_max = float(sr) / 2.0

        def hz_to_mel(f: float) -> float:
            return 2595.0 * math.log10(1.0 + max(f, 0.0) / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(0.0)
        mel_max = hz_to_mel(hz_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([mel_to_hz(float(m)) for m in mel_points])
        freqs = np.linspace(0.0, hz_max, n_freq_bins)

        fb = np.zeros((n_mels, n_freq_bins), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_low = hz_points[m - 1]
            f_center = hz_points[m]
            f_high = hz_points[m + 1]
            for k in range(n_freq_bins):
                f = freqs[k]
                if f_low <= f <= f_center:
                    denom = f_center - f_low
                    fb[m - 1, k] = (f - f_low) / denom if denom > 1e-10 else 0.0
                elif f_center < f <= f_high:
                    denom = f_high - f_center
                    fb[m - 1, k] = (f_high - f) / denom if denom > 1e-10 else 0.0

        return np.clip(fb, 0.0, None).astype(np.float32)

    @staticmethod
    def _compute_mel(
        audio: np.ndarray,
        sr: int,
        n_mels: int = 80,
        n_fft: int = _N_FFT_24K,
        hop: int = _HOP_24K,
    ) -> np.ndarray:
        """Berechnet Mel-Spektrogramm [n_mels, T], float32, finite.

        Formel: M = log(max(FB @ |STFT|^2, 1e-8))
        n_fft und hop müssen zum geladenen Modell passen (24k/44k/48k).
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n = len(audio)
        nperseg = min(n_fft, n) if n >= 2 else 2
        noverlap = max(0, nperseg - hop)
        try:
            _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
        except ValueError:
            nperseg = max(2, n)
            noverlap = nperseg - 1
            _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
        mag_sq = np.abs(Z) ** 2  # [n_freq_bins, T]
        n_freq_bins = mag_sq.shape[0]
        fb = VocosPlugin._build_mel_filterbank(n_mels, n_freq_bins, sr, nperseg)
        mel = fb @ mag_sq  # [n_mels, T]
        mel = np.log(np.maximum(mel, 1e-8)).astype(np.float32)
        return np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-18.4)

    @staticmethod
    def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
        """Schneidet auf target_len oder füllt mit Nullen auf.

        Invariante: len(result) == target_len, float32.
        """
        audio = audio.astype(np.float32)
        current = len(audio)
        if current == target_len:
            return audio
        elif current > target_len:
            return audio[:target_len]
        else:
            return np.pad(audio, (0, target_len - current)).astype(np.float32)

    # ------------------------------------------------------------------
    # Synthese-Methoden
    # ------------------------------------------------------------------

    def _synthesize_vocos_onnx(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, str, float]:
        """Vocos ONNX-Synthese via CPUExecutionProvider.

        Returns: (audio_out, model_name, confidence)

        Invarianten:
            - Ausgabe: float32 ∈ [-1, 1], kein NaN/Inf, len ≈ len(audio)
            - confidence ≥ 0.85 (neuronales Modell)
            - Stille-Eingabe → nahezu Stille im Ausgang
        """
        assert self._onnx_session is not None, "ONNX-Session nicht bereit"
        target_len = len(audio)
        model_sr = self._model_sr
        n_mels = self._mel_n_mels
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("Vocos", True)
        except Exception:
            logger.warning("vocos_plugin.py::_synthesize_vocos_onnx Ersatzpfad", exc_info=True)
        try:
            # 1. Resample auf Modell-SR (bei 48 kHz nativem Modell kein Resampling nötig)
            audio_model = self._resample(audio, sr, model_sr)

            # 2. Mel-Spektrogramm berechnen [n_mels, T] → [1, n_mels, T] (modellspezifisches n_fft/hop)
            mel = self._compute_mel(audio_model, model_sr, n_mels, n_fft=self._mel_n_fft, hop=self._mel_hop)

            # 3. ONNX Fixed-Shape-Input Guard (§ml-plugin-SKILL):
            # Check if model expects a fixed T dimension — if so, use chunked inference.
            _inp_meta = self._onnx_session.get_inputs()[0]
            input_name = _inp_meta.name
            _t_dim = _inp_meta.shape[2] if len(_inp_meta.shape) >= 3 else None
            _fixed_t = isinstance(_t_dim, int) and _t_dim > 0

            if _fixed_t:
                # Fixed-T model: chunk mel into _t_dim-frame segments, run each, concat waveforms.
                _chunk_t = int(_t_dim)
                _overlap_t = min(32, _chunk_t // 8)  # Overlap in mel frames (≈ crossfade region)
                _hop_samples = self._mel_hop  # samples per mel frame
                _n_mel_frames = mel.shape[1]
                waveform_segments: list[np.ndarray] = []
                _pos = 0
                while _pos < _n_mel_frames:
                    _end = min(_pos + _chunk_t, _n_mel_frames)
                    _seg = mel[:, _pos:_end]
                    if _seg.shape[1] < _chunk_t:
                        # Zero-pad last chunk to fixed size
                        _pad = np.zeros((_seg.shape[0], _chunk_t - _seg.shape[1]), dtype=np.float32)
                        _seg = np.concatenate([_seg, _pad], axis=1)
                    _seg_input = _seg[np.newaxis].astype(np.float32)
                    _seg_out = self._onnx_session.run(None, {input_name: _seg_input})
                    _seg_wav = np.asarray(_seg_out[0], dtype=np.float32).reshape(-1)
                    # Trim padding from last chunk
                    _valid_samples = (_end - _pos) * _hop_samples
                    waveform_segments.append(_seg_wav[:_valid_samples])
                    _pos += _chunk_t - _overlap_t
                # Simple concatenation with linear crossfade at chunk boundaries
                if len(waveform_segments) == 1:
                    waveform = waveform_segments[0]
                else:
                    _xfade_samples = _overlap_t * _hop_samples
                    waveform = waveform_segments[0]
                    for _next_seg in waveform_segments[1:]:
                        if len(waveform) < _xfade_samples or len(_next_seg) < _xfade_samples:
                            waveform = np.concatenate([waveform, _next_seg])
                        else:
                            _fade_out = np.linspace(1.0, 0.0, _xfade_samples, dtype=np.float32)
                            _fade_in = np.linspace(0.0, 1.0, _xfade_samples, dtype=np.float32)
                            waveform[-_xfade_samples:] = (
                                waveform[-_xfade_samples:] * _fade_out + _next_seg[:_xfade_samples] * _fade_in
                            )
                            waveform = np.concatenate([waveform, _next_seg[_xfade_samples:]])
                waveform = waveform.astype(np.float32)
            else:
                # Dynamic-T model: direct run; max-length guard to prevent OOM (60 s cap).
                _max_mel_frames = int(60 * model_sr / self._mel_hop)
                if mel.shape[1] > _max_mel_frames:
                    logger.debug(
                        "Vocos ONNX: mel T=%d > 60 s cap (%d) — truncating (dynamic-T model)",
                        mel.shape[1],
                        _max_mel_frames,
                    )
                    mel = mel[:, :_max_mel_frames]
                mel_input = mel[np.newaxis].astype(np.float32)  # [1, n_mels, T]
                ort_out = self._onnx_session.run(None, {input_name: mel_input})
                waveform = np.asarray(ort_out[0], dtype=np.float32).reshape(-1)

            # 4. Auf Aurik-SR resampeln und Länge angleichen
            waveform = self._resample(waveform, model_sr, sr)
            waveform = self._match_length(waveform, target_len)
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
            waveform = np.clip(waveform, -1.0, 1.0)
            return waveform, "vocos_onnx", 0.92
        except Exception as exc:
            logger.warning("Vocos ONNX-Inferenzfehler: %s — Audio-Passthrough.", exc)
            result = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0), "vocos_onnx_passthrough", 0.50
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("Vocos", False)
                except Exception:
                    logger.warning("vocos_plugin.py::unknown Ersatzpfad", exc_info=True)

    def _synthesize_bigvgan_v2(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, str, float]:
        """BigVGAN v2 Fallback-Synthese (Stufe 1.5, §4.4 SOTA-Matrix März 2026).

        Nutzt BigVGANv2Plugin — vollspektraler GAN-Vocoder bei 48 kHz nativ.
        Kein Resample-Verlust da Aurik nativ 48 kHz verarbeitet.

        Returns: (audio_out, model_name, confidence)
        """
        target_len = len(audio)
        try:
            from plugins.bigvgan_v2_plugin import BigVGANv2Plugin  # Lazy-Import

            _bvg = BigVGANv2Plugin()
            if not getattr(_bvg, "_model_loaded", False):
                return np.clip(audio.astype(np.float32), -1.0, 1.0), "bigvgan_v2_unavailable", 0.0
            result_obj = _bvg.synthesize(audio, sr, mode="studio2026")
            out = self._match_length(result_obj.audio, target_len)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = np.clip(out, -1.0, 1.0)
            logger.info("BigVGAN v2 Ersatzpfad erfolgreich (Stufe 1.5).")
            return out, "bigvgan_v2_fallback", 0.90
        except Exception as exc:
            logger.warning("BigVGAN v2 Ersatzpfad fehlgeschlagen: %s — HiFi-GAN wird versucht.", exc)
            return np.clip(audio.astype(np.float32), -1.0, 1.0), "bigvgan_v2_unavailable", 0.0

    def _synthesize_hifigan(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, str, float]:
        """HiFi-GAN Fallback-Synthese via hifigan_plugin (§4.4 SOTA-Matrix Stufe 2).

        Nutzt HifiGanPlugin.reconstruct() — Audio → 80-Band-Mel → ONNX → Waveform.
        Lazy-Import damit vocos_plugin ohne hifigan_plugin importierbar bleibt.

        Returns: (audio_out, model_name, confidence)

        Invarianten:
            - Ausgabe: float32 ∈ [-1, 1], kein NaN/Inf, len ≈ len(audio)
            - confidence = 0.78 (schlechter als Vocos, besser als Griffin-Lim)
            - Stille-Eingabe → nahezu Stille im Ausgang
        """
        target_len = len(audio)
        try:
            from plugins.hifigan_plugin import HifiGanPlugin  # Lazy-Import

            _hg = HifiGanPlugin()
            if not getattr(_hg, "_session", None):
                raise ImportError("HiFi-GAN Modell nicht verfügbar")
            out = _hg.reconstruct(audio, sr)
            out = self._match_length(out, target_len)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = np.clip(out, -1.0, 1.0)
            logger.info("HiFi-GAN Ersatzpfad erfolgreich.")
            return out, "hifigan_fallback", 0.78
        except Exception as exc:
            logger.warning("HiFi-GAN Ersatzpfad fehlgeschlagen: %s — Griffin-Lim wird verwendet.", exc)
            result = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0), "hifigan_unavailable", 0.40

    def _synthesize_istft_phase_coherent(
        self, audio: np.ndarray, sr: int, _n_iter: int = 32
    ) -> tuple[np.ndarray, str, float]:
        """Phase-coherent iSTFT fallback synthesis (Stufe 3 — last resort).

        Replaces former Griffin-Lim: uses original-phase iSTFT for transparent,
        deterministic reconstruction (§2.40, §4.5 — griffinlim() als Endschritt verboten).
        Only active when Vocos ONNX AND HiFi-GAN are both unavailable.

        Returns: (audio_out, model_name, confidence)

        Invarianten:
            - Ausgabe: float32 ∈ [-1, 1], kein NaN/Inf, len == len(audio)
            - confidence < 0.70 (schlechter als neuronale Modelle)
            - Stille-Eingabe → nahezu Stille im Ausgang
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        target_len = len(audio)
        if target_len < 2:
            return np.zeros(target_len, np.float32), "istft_phase_coherent_fallback", 0.30

        try:
            nperseg = min(_WIN, target_len)
            noverlap = max(0, nperseg - _HOP)
            _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
            mag = np.abs(Z)
            # Use original phase — no random (§2.40 determinism)
            orig_phase = np.angle(Z)
            # iSTFT with original phase as transparent phase-coherent fallback (§4.5: keine Griffin-Lim)
            _, result = istft(mag * np.exp(1j * orig_phase), fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
            result = np.nan_to_num(result.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            result = self._match_length(result, target_len)
            result = np.clip(result, -1.0, 1.0)
        except Exception as e:
            logger.warning("Verarbeitungsschritt-coherent iSTFT Ersatzpfad Fehler: %s", e)
            result = np.clip(audio.copy(), -1.0, 1.0)

        return result, "istft_phase_coherent_fallback", 0.60

    def _estimate_pqs_mos(self, original: np.ndarray, restored: np.ndarray, sr: int) -> float:
        """Schätzt PQS-MOS ∈ [1.0, 5.0].

        Formel: MOS = 1.0 + 4.0 * sigmoid(8 * (pearson_corr - 0.5))
        Invariante: Identisches Audio → MOS ≥ 4.0
        """
        try:
            orig_mel = self._compute_mel(original, sr)
            rest_mel = self._compute_mel(restored, sr)
            n = min(orig_mel.shape[1], rest_mel.shape[1])
            if n < 2:
                return 3.0
            a = orig_mel[:, :n].flatten().astype(np.float64)
            b = rest_mel[:, :n].flatten().astype(np.float64)
            std_a = float(np.std(a))
            std_b = float(np.std(b))
            if std_a < 1e-8 and std_b < 1e-8:
                return 4.5  # Beide Stille → identisch
            if std_a < 1e-8 or std_b < 1e-8:
                return 1.5
            _av = a - a.mean()
            _bv = b - b.mean()
            _nv = float(np.linalg.norm(_av))
            _nbv = float(np.linalg.norm(_bv))
            corr = float(np.dot(_av, _bv) / (_nv * _nbv + 1e-10))
            sim = float(np.clip(np.nan_to_num(corr), -1.0, 1.0))
            z = 8.0 * (sim - 0.5)
            sig = 1.0 / (1.0 + math.exp(-z))
            mos = 1.0 + 4.0 * sig
            return float(np.clip(mos, 1.0, 5.0))
        except Exception:
            logger.warning("vocos_plugin.py::_estimate_pqs_mos Ersatzpfad", exc_info=True)
            return 3.0

    def _mel_snr(self, original: np.ndarray, restored: np.ndarray, _sr: int) -> float:
        """Berechnet Signal-Rausch-Abstand in dB.

        Invariante: Identisches Audio → SNR ≥ 20 dB
        """
        try:
            n = min(len(original), len(restored))
            orig = original[:n].astype(np.float64)
            rest = restored[:n].astype(np.float64)
            signal_pow = float(np.sum(orig**2))
            noise_pow = float(np.sum((orig - rest) ** 2))
            if noise_pow < 1e-20:
                return 80.0  # Identisch
            if signal_pow < 1e-20:
                return 0.0
            return float(10.0 * math.log10(max(signal_pow / noise_pow, 1e-10)))
        except Exception:
            logger.warning("vocos_plugin.py::_mel_snr Ersatzpfad", exc_info=True)
            return 0.0

    def vocode(self, audio: np.ndarray, sr: int, mode: str = "studio2026") -> "VocosResult":
        """Synthetisiert Audio mit Vocos oder Griffin-Lim-Fallback.

        Args:
            audio: Input-Audio float32, shape [N]
            sr:    Sample-Rate (muss AURIK_SR = 48000 Hz sein)
            mode:  "studio2026" (erlaubt) oder "restoration" (verboten)

        Returns:
            VocosResult mit synthetisiertem Audio

        Raises:
            ValueError: wenn mode == "restoration"
            AssertionError: wenn sr != AURIK_SR
        """
        if mode == "restoration":
            raise ValueError(
                "Vocos im Restoration-Modus deaktiviert — nur Studio-2026-Modus erlaubt (§1.4 Aurik Spec)."
            )
        assert sr == AURIK_SR, f"SR muss {AURIK_SR} Hz sein, erhalten: {sr}"

        # NaN/Inf bereinigen
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # §4.4 Spec SOTA-Matrix — Fallback-Kaskade (aktualisiert März 2026):
        #   Stufe 1:   Vocos 44.1 kHz ONNX  (primär, volles Spektrum bis 22 kHz)
        #   Stufe 2:  Vocos 24 kHz ONNX    (falls 44.1 kHz fehlt)
        #   Stufe 1.5: BigVGAN v2 ONNX      (GAN-basierter Wideband-Vocoder, 48 kHz)
        #   Stufe 2:   HiFi-GAN ONNX        (neuronaler Fallback)
        #   Stufe 3:   Griffin-Lim+         (DSP-Letzfall, ≥ 32 Iter.)
        # Griffin-Lim als Endschritt VERBOTEN in Studio-2026 (§4.4).
        if self._model_loaded and self._onnx_session is not None:
            out_audio, model_name, confidence = self._synthesize_vocos_onnx(audio, sr)
        else:
            # Stufe 1.5: BigVGAN v2 als vollspektraler GAN-Fallback (48 kHz native)
            out_audio, model_name, confidence = self._synthesize_bigvgan_v2(audio, sr)
            if model_name == "bigvgan_v2_unavailable":
                # Stufe 2: HiFi-GAN als neuronaler Fallback
                out_audio, model_name, confidence = self._synthesize_hifigan(audio, sr)
                # Stufe 3: Phase-coherent iSTFT wenn alle neuronalen Modelle fehlen (§4.5)
                if model_name == "hifigan_unavailable":
                    out_audio, model_name, confidence = self._synthesize_istft_phase_coherent(audio, sr)
                    logger.warning(
                        "Vocos + BigVGAN v2 + HiFi-GAN nicht verfügbar — Verarbeitungsschritt-coherent iSTFT aktiv (Stufe 3, §4.5)."
                    )
        pqs = self._estimate_pqs_mos(audio, out_audio, sr)
        snr = self._mel_snr(audio, out_audio, sr)

        return VocosResult(
            audio=out_audio,
            sr=sr,
            pqs_mos=pqs,
            model_used=model_name,
            confidence=confidence,
            mel_snr_db=snr,
            model_sr=self._model_sr,
        )


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, §3.2 Aurik Spec)
# ---------------------------------------------------------------------------
_instance: VocosPlugin | None = None
_lock = threading.Lock()


def get_vocos_plugin() -> VocosPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance  # pylint: disable=global-statement  # §3.2 Singleton-Pattern (normativ vorgeschrieben)
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VocosPlugin()
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktionen
# ---------------------------------------------------------------------------


def vocode(audio: np.ndarray, sr: int, mode: str = "studio2026") -> VocosResult:
    """Convenience-Wrapper für get_vocos_plugin().vocode()."""
    return get_vocos_plugin().vocode(audio, sr, mode=mode)


def vocode_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convenience-Wrapper — gibt synthetisiertes Audio als np.ndarray zurück."""
    result = get_vocos_plugin().vocode(audio, sr, mode="studio2026")
    return result.audio
