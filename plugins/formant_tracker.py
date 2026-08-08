"""
FormantTracker – LPC Burg-basierter Formant-Tracker (§2.8, §4.4).

Algorithm (per Aurik 10.0.0 spec §4.4):
    LPC mit Burg-Algorithmus (Ordnung 16, 25 ms Frames), Root-Finding via
    np.roots(), Selektion stimmhafter Wurzeln (|r| < 1, angle > 0),
    Konvertierung zu Formant-Frequenzen F_k = angle_k · sr / (2π).

Math:
    LPC coefficients via Burg autocorrelation (bias-corrected, Ordnung 16)
    Roots r_k of polynomial [1, a_1, …, a_p]
    F_k [Hz]  = angle(r_k) · sr / (2π)          für angle(r_k) > 0, |r_k| < 1
    BW_k [Hz] = -log(|r_k|) · sr / π
    confidence_k = σ(−BW_k / sr)                 (enger BW → hohe Konfidenz)

Invariants:
    - NaN/Inf-safe
    - Mono conversion before analysis
    - Returns F1–F4 with per-formant confidence ∈ [0, 1]
    - Thread-safe singleton via get_formant_tracker()
    - Pure DSP — no external ML model required

References:
    §2.8 VocalAIEnhancement: FormantTracker High-Order LPC (Burg, Ordnung 16)
    §4.4 SOTA table: "High-Order LPC (Burg)" als Pflicht-Algorithmus
    §3.1 Numerische Robustheit
    §3.2 Singleton-Pattern
    Rabiner & Schafer (1978) — Digital Processing of Speech Signals
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
# DeepFormants CNN (Krug et al. 2022) — primärer Pfad, LPC als Fallback (§4.4)
_DEEPFORMANTS_ONNX = _ROOT / "models" / "deepformants" / "deepformants.onnx"
# DeepFormants Eingangs-SR: 16 kHz; intern resampled von 48 kHz
_DEEPFORMANTS_SR: int = 16_000
_DEEPFORMANTS_N_MELS: int = 128
_DEEPFORMANTS_N_FFT: int = 512  # 32 ms @ 16 kHz
_DEEPFORMANTS_HOP: int = 160  # 10 ms @ 16 kHz

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
LPC_ORDER: int = 32  # Spec §VERBOTEN: < 16; Richtig: 30–40 @ 48 kHz (war: 16)
FRAME_MS: float = 25.0  # Analysis frame [ms]
HOP_MS: float = 10.0  # Hop size [ms]
TARGET_SR: int = 48_000  # Internal processing SR
PREEMPHASIS: float = 0.97  # Pre-emphasis coefficient


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class FormantFrame:
    """Formanten eines einzelnen Analyse-Frames.

    Attributes:
        frequencies: F1–F4 in Hz (NaN wenn nicht bestimmbar)
        bandwidths: Zugehörige Bandbreiten in Hz
        confidences: Konfidenz pro Formant ∈ [0, 1]
    """

    frequencies: list[float] = field(default_factory=lambda: [float("nan")] * 4)
    bandwidths: list[float] = field(default_factory=lambda: [float("nan")] * 4)
    confidences: list[float] = field(default_factory=lambda: [0.0] * 4)


@dataclass
class FormantTrackingResult:
    """Ergebnis der Formant-Analyse.

    Attributes:
        formants: Median F1–F4 über alle validen Frames [Hz]
        formant_tracks: FormantFrame pro Analyse-Frame
        confidence: Gesamt-Konfidenz ∈ [0, 1]
        n_voiced_frames: Anzahl stimmhafter Frames (|LPC| > 0)
    """

    formants: list[float] = field(default_factory=lambda: [float("nan")] * 4)
    formant_tracks: list[FormantFrame] = field(default_factory=list)
    confidence: float = 0.0
    n_voiced_frames: int = 0


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------
class FormantTracker:
    """LPC Burg Ordnung-16 Formant-Tracker für Vokal-Analyse (§2.8, §4.4).

    Algorithm per frame:
        1. Pre-emphasis (α = 0.97)
        2. Hanning window
        3. Burg autocorrelation → LPC coefficients (order 16)
        4. np.roots([1, a_1, …, a_p]) → complex roots
        5. Select roots: |r| < 1.0 AND angle(r) > 0  (upper half unit circle)
        6. Sort by frequency; take top-4 as F1–F4
        7. Confidence: sigmoid(−BW / sr · 100)

    Args:
        lpc_order: LPC order (default 16 per spec)
        preemphasis: Pre-emphasis coefficient (default 0.97)
    """

    def __init__(
        self,
        lpc_order: int = LPC_ORDER,
        preemphasis: float = PREEMPHASIS,
    ) -> None:
        self._order = int(lpc_order)
        self._preemphasis = float(preemphasis)
        # DeepFormants ONNX-Session (optional; LPC-Fallback wenn nicht verfügbar)
        self._deepformants_session = None
        self._deepformants_loaded: bool = False
        self._try_load_deepformants()
        logger.debug(
            "FormantTracker initialisiert: lpc_order=%d, preemphasis=%.2f, deepformants=%s",
            self._order,
            self._preemphasis,
            "ONNX" if self._deepformants_loaded else "LPC-Fallback",
        )

    def _try_load_deepformants(self) -> None:
        """Lädt DeepFormants ONNX-Modell (optisch); LPC-Burg als Fallback."""
        if not _DEEPFORMANTS_ONNX.exists():
            logger.debug(
                "DeepFormants ONNX nicht gefunden (%s) — LPC-Burg-Ersatzpfad aktiv. "
                "Modell: https://github.com/vokandre/deepformants",
                _DEEPFORMANTS_ONNX,
            )
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("DeepFormants", size_gb=0.05):
                    logger.warning("DeepFormants: ML-Grenze erschöpft — LPC-Ersatzpfad.")
                    return
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            self._deepformants_session = ort.InferenceSession(
                str(_DEEPFORMANTS_ONNX),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._deepformants_loaded = True
            logger.info("✅ DeepFormants ONNX geladen — §4.4 primärer Formant-Tracker.")
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    "DeepFormants",
                    size_gb=0.05,
                    unload_fn=lambda s=self: (  # type: ignore[misc]
                        setattr(s, "_deepformants_session", None) or setattr(s, "_deepformants_loaded", False)  # type: ignore[func-returns-value]
                    ),
                )
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        except Exception as exc:
            logger.debug("DeepFormants ONNX nicht ladbar: %s — LPC-Burg-Ersatzpfad.", exc)
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("DeepFormants")
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def track(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> FormantTrackingResult:
        """Verfolgt formants F1–F4 in vocal audio.

        Primary path: DeepFormants CNN (ONNX, Krug et al. 2022) when model available.
        Fallback: LPC Burg order-16 (§4.4 DSP standard).  # §V6: logger.warning handled at call site

        Args:
            audio: float32/64 ndarray (mono or stereo)
            sample_rate: Sample rate in Hz (muss 48000 Hz sein)

        Returns:
            FormantTrackingResult with formants, tracks, confidence.

        Raises:
            ValueError: If audio is empty or sample_rate < 8000.
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        if audio.size == 0:
            raise ValueError("audio must not be empty")

        audio_f = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Primärer Pfad: DeepFormants CNN (wenn ONNX verfügbar)
        if self._deepformants_loaded and self._deepformants_session is not None:
            try:
                return self._analyze_deepformants(audio_f, sample_rate)
            except Exception as exc:
                logger.warning("DeepFormants ONNX fehlgeschlagen: %s — LPC-Burg-Ersatzpfad.", exc)

        # Fallback: LPC Burg (bisherige Implementierung)
        return self._analyze_lpc(audio_f, sample_rate)

    def _analyze_deepformants(
        self,
        audio_f: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> FormantTrackingResult:
        """DeepFormants CNN Formant-Tracking.

        Algorithm:
            1. Mono + Resample 48 kHz → 16 kHz
            2. Log-Mel Spectrogram [128 Bänder, n_fft=512, hop=160 @ 16 kHz]
            3. ONNX: [1, 128, T] → [1, 4, T] (F1–F4 per Frame in Hz)
            4. NaN-Guard + Post-Filter (physiologische Grenzen)
            5. Median-Aggregation pro Formant
        """
        assert self._deepformants_session is not None

        mono = audio_f if audio_f.ndim == 1 else audio_f.mean(axis=-1)
        mono_16k = self._resample_if_needed(mono, sample_rate, _DEEPFORMANTS_SR)

        # Log-Mel Spectrogram
        from scipy.signal import stft as scipy_stft

        _, _, Z = scipy_stft(
            mono_16k.astype(np.float64),
            fs=_DEEPFORMANTS_SR,
            nperseg=_DEEPFORMANTS_N_FFT,
            noverlap=_DEEPFORMANTS_N_FFT - _DEEPFORMANTS_HOP,
            window="hann",
        )
        mag = np.abs(Z).astype(np.float32)  # [freq_bins, T]

        # Mel-Filterbank (128 Bänder)
        try:
            import librosa

            mel_basis = librosa.filters.mel(
                sr=_DEEPFORMANTS_SR,
                n_fft=_DEEPFORMANTS_N_FFT,
                n_mels=_DEEPFORMANTS_N_MELS,
                fmin=50.0,
                fmax=8000.0,
            ).astype(np.float32)
        except ImportError:
            # Manuelle Dreiecks-Filterbank als Notfallpfad
            mel_basis = self._build_mel_basis(_DEEPFORMANTS_SR, _DEEPFORMANTS_N_FFT, _DEEPFORMANTS_N_MELS)

        mel_spec = mel_basis @ mag  # [128, T]
        log_mel = np.log1p(mel_spec * 1e4)  # Log-Skalierung
        log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0)

        # ONNX-Inferenz: [1, 128, T] → [1, 4, T]
        inp = log_mel[np.newaxis].astype(np.float32)  # [1, 128, T]
        inp_name = self._deepformants_session.get_inputs()[0].name
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("DeepFormants", True)
        except Exception:
            logger.warning("formant_tracker.py::_analyze_deepformants Ersatzpfad", exc_info=True)
        try:
            ort_out = self._deepformants_session.run(None, {inp_name: inp})
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("DeepFormants", False)
                except Exception:
                    logger.warning("formant_tracker.py::_analyze_deepformants Ersatzpfad", exc_info=True)
        formant_tracks_hz = np.asarray(ort_out[0], dtype=np.float32)  # [1, 4, T] oder [4, T]

        if formant_tracks_hz.ndim == 3:
            formant_tracks_hz = formant_tracks_hz[0]  # [4, T]

        formant_tracks_hz = np.nan_to_num(formant_tracks_hz, nan=0.0, posinf=0.0, neginf=0.0)

        # Physiologische Grenzen F1–F4 (Hz)
        _F_BOUNDS = [(200, 1200), (700, 3000), (1500, 4000), (2500, 5000)]
        for k, (fmin_k, fmax_k) in enumerate(_F_BOUNDS):
            formant_tracks_hz[k] = np.clip(formant_tracks_hz[k], fmin_k, fmax_k)

        # Frame-Objekte erstellen
        n_frames = formant_tracks_hz.shape[1]
        tracks: list[FormantFrame] = []
        for t in range(n_frames):
            freqs = [float(formant_tracks_hz[k, t]) for k in range(4)]
            confs = [0.8 if f > 0 else 0.0 for f in freqs]  # CNN: fester Basis-Konfidenzwert
            tracks.append(FormantFrame(frequencies=freqs, bandwidths=[0.0] * 4, confidences=confs))

        # Median-Aggregation
        median_formants: list[float] = []
        for k in range(4):
            vals = [formant_tracks_hz[k, t] for t in range(n_frames) if formant_tracks_hz[k, t] > 0]
            median_formants.append(float(np.median(vals)) if vals else float("nan"))

        confidence = 0.85  # DeepFormants CNN: hohe Konfidenz bei erfolgreichem ONNX-Lauf
        voiced = sum(1 for t in tracks if any(f > 0 for f in t.frequencies))

        logger.debug(
            "DeepFormants ONNX: F1=%.0f F2=%.0f F3=%.0f F4=%.0f Hz, conf=%.2f",
            *[f if math.isfinite(f) else 0 for f in median_formants],
            confidence,
        )
        return FormantTrackingResult(
            formants=median_formants,
            formant_tracks=tracks,
            confidence=confidence,
            n_voiced_frames=voiced,
        )

    @staticmethod
    def _build_mel_basis(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Einfache Dreiecks-Mel-Filterbank (Notfallpfad ohne librosa)."""
        freq_bins = n_fft // 2 + 1
        freqs = np.linspace(0, sr / 2, freq_bins)
        mel_min = 2595.0 * np.log10(1.0 + 50.0 / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        basis = np.zeros((n_mels, freq_bins), dtype=np.float32)
        for m in range(n_mels):
            l, c, r = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            for k, f in enumerate(freqs):
                if l <= f < c:
                    basis[m, k] = (f - l) / max(c - l, 1e-10)
                elif c <= f <= r:
                    basis[m, k] = (r - f) / max(r - c, 1e-10)
        return basis  # type: ignore[no-any-return]

    def _analyze_lpc(
        self,
        audio_f: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> FormantTrackingResult:
        """LPC Burg-basiertes Formant-Tracking (§4.4 DSP-Fallback).

        Spec §2.8: Downsampling auf 16 kHz vor LPC-Analyse → LPC Ord. 16 korrekt.
        (Alternative wäre Ord. 30–40 direkt bei 48 kHz; wir nutzen den Downsampling-Pfad,
        da _DEEPFORMANTS_SR=16000 bereits passend definiert ist.)
        """
        # §2.8: Downsample to 16 kHz for LPC analysis — LPC order 16 is correct at 16 kHz.
        # Using 48 kHz with order 16 would badly underfit the spectral envelope
        # (rule of thumb: order ≈ SR[kHz]×2+4 → at 48kHz: ~100; compromise 30–40;
        #  or: downsample to 16kHz → order 16 is correct per Rabiner & Schafer 1978).
        _LPC_SR: int = _DEEPFORMANTS_SR  # 16 000 Hz
        mono: npt.NDArray[np.float32] = audio_f if audio_f.ndim == 1 else audio_f.mean(axis=-1)
        mono = self._resample_if_needed(mono, sample_rate, _LPC_SR)
        sr = _LPC_SR

        # Pre-emphasis
        mono_pe = np.concatenate([[mono[0]], mono[1:] - self._preemphasis * mono[:-1]])
        mono_pe = mono_pe.astype(np.float32)

        frame_size = int(FRAME_MS * sr / 1000)
        hop_size = int(HOP_MS * sr / 1000)
        n = len(mono_pe)
        n_frames = max(1, 1 + (n - frame_size) // hop_size)
        window = np.hanning(frame_size).astype(np.float32)

        tracks: list[FormantFrame] = []
        voiced_count = 0

        for k in range(n_frames):
            start = k * hop_size
            frame = mono_pe[start : start + frame_size]
            if len(frame) < frame_size // 2:
                tracks.append(FormantFrame())
                continue
            # Zero-pad if shorter than frame_size
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[: len(frame)] = frame
            padded *= window

            lpc = self._burg_lpc(padded, self._order)
            if lpc is None:
                tracks.append(FormantFrame())
                continue

            frame_result = self._lpc_to_formants(lpc, sr)
            if frame_result is not None:
                voiced_count += 1
                tracks.append(frame_result)
            else:
                tracks.append(FormantFrame())

        # Aggregate: median F1–F4 over voiced frames
        all_freqs: list[list[float]] = [
            [t.frequencies[i] for t in tracks if math.isfinite(t.frequencies[i])] for i in range(4)
        ]
        median_formants = [float(np.median(vals)) if vals else float("nan") for vals in all_freqs]
        all_conf: list[float] = [c for t in tracks for c in t.confidences if math.isfinite(c) and c > 0]
        confidence = float(np.mean(all_conf)) if all_conf else 0.0

        result = FormantTrackingResult(
            formants=median_formants,
            formant_tracks=tracks,
            confidence=confidence,
            n_voiced_frames=voiced_count,
        )
        logger.debug(
            "FormantTracker: F1=%.0f F2=%.0f F3=%.0f F4=%.0f Hz, conf=%.2f",
            *[f if math.isfinite(f) else 0 for f in median_formants],
            confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _burg_lpc(
        frame: npt.NDArray[np.float32],
        order: int,
    ) -> npt.NDArray[np.float64] | None:
        """Berechnet LPC coefficients via Burg's method.

        Returns coefficient array [1, a_1, …, a_p] or None on failure.
        """
        try:
            # Try librosa's Burg if available (most accurate)
            from librosa.core.audio import _burg as _librosa_burg  # type: ignore[attr-defined]

            lpc = _librosa_burg(frame.astype(np.float64), order=order)
            return np.array([1.0, *list(-lpc)], dtype=np.float64)
        except (ImportError, AttributeError):
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        # Pure-numpy Burg autocorrelation (bias-corrected)
        x = frame.astype(np.float64)
        n = len(x)
        if n < order + 1:
            return None
        # Autocorrelation
        np.zeros(order + 1, dtype=np.float64)
        k_vec = np.zeros(order, dtype=np.float64)
        e = float(np.dot(x, x))
        if e < 1e-30:
            return None

        ef = x.copy()
        eb = x.copy()

        for m in range(order):
            num = -2.0 * float(np.dot(ef[m + 1 :], eb[m : n - 1]))
            den = float(np.dot(ef[m + 1 :], ef[m + 1 :]) + np.dot(eb[m : n - 1], eb[m : n - 1]))
            if abs(den) < 1e-30:
                return None
            km = num / den
            k_vec[m] = km

            # Update filter
            new_ef = ef[m + 1 :] + km * eb[m : n - 1]
            new_eb = eb[m : n - 1] + km * ef[m + 1 :]
            ef[m + 1 :] = new_ef
            eb[m : n - 1] = new_eb
            e *= 1.0 - km**2

        # Convert reflection coefficients to AR coefficients
        ar = np.zeros(order, dtype=np.float64)
        ar[0] = k_vec[0]
        for m in range(1, order):
            ar[m] = k_vec[m]
            ar[:m] += k_vec[m] * ar[m - 1 :: -1]

        return np.concatenate([[1.0], ar])

    def _lpc_to_formants(
        self,
        lpc: npt.NDArray[np.float64],
        sr: int,
    ) -> FormantFrame | None:
        """Konvertiert LPC polynomial to F1–F4 formant estimates."""
        try:
            roots = np.roots(lpc)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Select roots inside unit circle with positive imaginary part
        vocal_roots = [r for r in roots if abs(r) < 1.0 and np.angle(r) > 0.005]  # > ~0.3° — exclude DC
        if not vocal_roots:
            return None

        # Sort by angle (= frequency)
        vocal_roots.sort(key=lambda r: np.angle(r))

        freqs: list[float] = []
        bws: list[float] = []
        confs: list[float] = []

        for r in vocal_roots[:4]:
            angle = float(np.angle(r))
            mag = float(abs(r))
            f_hz = angle * sr / (2.0 * math.pi)
            bw_hz = -math.log(max(mag, 1e-9)) * sr / math.pi
            # Confidence: narrow bandwidth → high confidence (sigmoid)
            conf = 1.0 / (1.0 + math.exp(bw_hz / sr * 100 - 5))
            freqs.append(f_hz)
            bws.append(bw_hz)
            confs.append(float(np.clip(conf, 0.0, 1.0)))

        # Pad to 4 if fewer candidates
        while len(freqs) < 4:
            freqs.append(float("nan"))
            bws.append(float("nan"))
            confs.append(0.0)

        return FormantFrame(frequencies=freqs, bandwidths=bws, confidences=confs)

    @staticmethod
    def _resample_if_needed(
        audio: npt.NDArray[np.float32],
        src_sr: int,
        dst_sr: int,
    ) -> npt.NDArray[np.float32]:
        """Resample audio from src_sr to dst_sr."""
        if src_sr == dst_sr:
            return audio
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(dst_sr, src_sr)
            return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)  # type: ignore[no-any-return]
        except ImportError:
            orig_len = len(audio)
            new_len = int(orig_len * dst_sr / src_sr)
            return np.interp(
                np.linspace(0, orig_len - 1, new_len),
                np.arange(orig_len),
                audio,
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# Legacy-kompatibles Interface (drop-in für alten Code)
# ---------------------------------------------------------------------------
class FormantTrackerLegacy(FormantTracker):
    """Drop-in wrapper that exposes {'formants': [...], 'confidence': float}."""

    def track(  # type: ignore[override]
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> dict:  # type: ignore[override]
        result = super().track(audio, sample_rate)
        clean = [f if math.isfinite(f) else 0.0 for f in result.formants]
        return {"formants": clean[:3], "confidence": result.confidence}


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: FormantTracker | None = None
_lock = threading.Lock()


def get_formant_tracker() -> FormantTracker:
    """Thread-safe singleton (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FormantTracker()
    return _instance


def track_formants(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> FormantTrackingResult:
    """Convenience wrapper — Formant-Tracking ohne Klassen-Instantiierung.

    Args:
        audio: float32 audio (mono or stereo)
        sample_rate: Sample rate in Hz

    Returns:
        FormantTrackingResult with F1–F4 Hz, confidence.
    """
    return get_formant_tracker().track(audio, sample_rate)
