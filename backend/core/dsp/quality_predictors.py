"""
DSP-based quality predictors — DNSMOS P.835 and SingMOS proxies.

Lightweight, offline-first implementations that approximate the original ONNX-model
outputs (SIG/BAK/OVR ∈ [1, 5]) using acoustic feature extraction.

Exports (kanonisch per musical_goals.instructions.md):
    get_dnsmos_predictor()   → DNSMOSPredictor singleton
    get_singmos_predictor()  → SingMOSPredictor singleton

Usage:
    from backend.core.dsp.quality_predictors import get_dnsmos_predictor, get_singmos_predictor

    # Instrumental / general:
    result = get_dnsmos_predictor().predict(audio, sr)
    ovr_01  = (result["ovr"] - 1.0) / 4.0   # normalised → [0, 1]

    # Singing material (panns_singing >= 0.35):
    mos_15  = get_singmos_predictor().predict(audio, sr)  # → [1, 5]
    mos_01  = (mos_15 - 1.0) / 4.0           # normalised → [0, 1]

No ML models are required — all estimates are based on spectral, tonal and
harmonic features computed at 22 050 Hz analysis rate for speed.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ANALYSIS_SR = 22050  # lightweight analysis rate
_FRAME_LEN = 1024
_HOP_LEN = 256
_MAX_ANALYSIS_SAMPLES = _ANALYSIS_SR * 8  # cap at 8 s — stationary features


def _resample_to_analysis_sr(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Downmix to mono + decimate to _ANALYSIS_SR."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1)
    audio = np.asarray(audio, dtype=np.float64)
    if sr != _ANALYSIS_SR and sr > 0:
        try:
            from scipy.signal import decimate as _dec  # pylint: disable=import-outside-toplevel

            stride = max(1, round(sr / _ANALYSIS_SR))
            if stride > 1:
                audio = np.asarray(_dec(audio, stride, zero_phase=True), dtype=np.float64)
                sr = max(1, sr // stride)
        except Exception:
            stride = max(1, round(sr / _ANALYSIS_SR))
            if stride > 1:
                audio = audio[::stride]
                sr = max(1, sr // stride)
    # Cap length
    if len(audio) > _MAX_ANALYSIS_SAMPLES:
        start = (len(audio) - _MAX_ANALYSIS_SAMPLES) // 2
        audio = audio[start : start + _MAX_ANALYSIS_SAMPLES]
    return audio, sr


def _safe_stft(audio: np.ndarray) -> np.ndarray:
    """Gibt magnitude spectrogram (frames × bins) zurück."""
    n_fft = min(_FRAME_LEN, len(audio))
    hop = min(_HOP_LEN, max(1, n_fft // 4))
    num_frames = max(1, (len(audio) - n_fft) // hop + 1)
    frames = np.stack([audio[i * hop : i * hop + n_fft] * np.hanning(n_fft) for i in range(num_frames)], axis=0)
    return np.abs(np.fft.rfft(frames, n=n_fft, axis=1))  # type: ignore[no-any-return]  # (T, F)


def _estimate_snr_db(audio: np.ndarray) -> float:
    """Schätzt SNR in dB from signal + noise-floor percentile."""
    if len(audio) < 64:
        return 0.0
    rms_signal = float(np.sqrt(np.mean(audio**2) + 1e-12))
    # Noise floor ~ low-energy percentile of short frames
    frame_rms = [
        float(np.sqrt(np.mean(audio[i : i + 256] ** 2) + 1e-12)) for i in range(0, max(1, len(audio) - 256), 256)
    ]
    if not frame_rms:
        return 0.0
    noise_floor = float(np.percentile(frame_rms, 10)) + 1e-12
    return float(np.clip(20.0 * np.log10(rms_signal / noise_floor), 0.0, 60.0))


def _spectral_flatness(mag: np.ndarray) -> float:
    """Mean spectral flatness (0=pure tone, 1=white noise)."""
    eps = 1e-12
    geo_mean = float(np.exp(np.mean(np.log(mag + eps), axis=1)).mean())
    arith_mean = float(np.mean(mag))
    return float(np.clip(geo_mean / (arith_mean + eps), 0.0, 1.0))


def _spectral_contrast(mag: np.ndarray) -> float:
    """Mean spectral contrast across frames (dB)."""
    if mag.shape[0] < 2:
        return 0.0
    peak = np.percentile(mag, 99, axis=1)
    valley = np.percentile(mag, 10, axis=1) + 1e-12
    return float(np.mean(20.0 * np.log10(peak / valley + 1e-12)))


def _hnr_approx(audio: np.ndarray, sr: int) -> float:
    """
    Approximate HNR (harmonics-to-noise ratio) via ACF peak ratio.

    Returns HNR in dB, capped at 30 dB.
    """
    if len(audio) < 128:
        return 0.0
    # Use only a central segment of max 1 s
    seg_len = min(len(audio), sr)
    start = (len(audio) - seg_len) // 2
    seg = audio[start : start + seg_len]
    # V08-konform: FFT-basierte Auto-Korrelation statt O(n²) np.correlate
    from scipy.signal import fftconvolve

    acf = fftconvolve(seg, seg[::-1], mode="full")
    acf = acf[len(acf) // 2 :]
    acf /= acf[0] + 1e-12
    # Look for peak in F0 range 80–1000 Hz (lags)
    lag_min = max(1, sr // 1000)
    lag_max = max(lag_min + 1, sr // 80)
    lag_max = min(lag_max, len(acf) - 1)
    if lag_min >= lag_max:
        return 0.0
    peak_val = float(np.max(acf[lag_min:lag_max]))
    peak_val = float(np.clip(peak_val, -0.999, 0.999))
    hnr = 10.0 * np.log10((peak_val + 1e-12) / (1.0 - peak_val + 1e-12))
    return float(np.clip(hnr, 0.0, 30.0))


def _f0_stability(audio: np.ndarray, sr: int = 22050) -> float:  # sr reserved for future pyin fallback
    """
    Schätzt F0 stability (0–1): high = stable pitch / low = unstable / noisy.

    Uses zero-crossing-rate-based periodicity as a proxy.
    """
    _ = sr  # sr is intentionally unused — placeholder for future pyin-based fallback
    if len(audio) < 64:
        return 0.5
    frame_len = min(512, len(audio))
    hop = max(1, frame_len // 4)
    zcr_frames = []
    for i in range(0, max(1, len(audio) - frame_len), hop):
        frame = audio[i : i + frame_len]
        zcr_frames.append(float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0))
    if len(zcr_frames) < 2:
        return 0.5
    zcr_arr = np.array(zcr_frames)
    # Low-ZCR variance → pitch-stable
    norm_std = float(np.std(zcr_arr) / (np.mean(zcr_arr) + 1e-12))
    return float(np.clip(1.0 - min(1.0, norm_std), 0.0, 1.0))


def _formant_clarity(audio: np.ndarray, sr: int) -> float:
    """
    Schätzt formant clarity in the F1–F3 band (500–3000 Hz).

    High clarity → prominent energy bumps in LPC spectrum → natural vowels.
    Proxy: spectral contrast in 300–3500 Hz band.
    """
    if len(audio) < 32 or sr < 1:
        return 0.5
    mag = _safe_stft(audio)
    if mag.shape[1] < 4:
        return 0.5
    n_fft = min(_FRAME_LEN, len(audio))
    freq_res = (sr / 2.0) / (n_fft // 2 + 1)
    lo_bin = max(0, int(300 / freq_res))
    hi_bin = min(mag.shape[1], int(3500 / freq_res))
    if lo_bin >= hi_bin:
        return 0.5
    band_mag = mag[:, lo_bin:hi_bin]
    contrast = _spectral_contrast(band_mag)
    # Map 0–30 dB contrast to 0–1
    return float(np.clip(contrast / 30.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# DNSMOSPredictor
# ---------------------------------------------------------------------------


class DNSMOSPredictor:
    """
    Leichtgewichtiges DNSMOS P.835 proxy without ONNX model.

    Returns dict with keys "sig", "bak", "ovr" — each in [1, 5].
    """

    def predict(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Predict DNSMOS P.835 scores.

        Args:
            audio: Audio signal (mono or stereo), float32/float64.
            sr:    Sample rate in Hz.

        Returns:
            {"sig": float, "bak": float, "ovr": float} — each in [1, 5].
        """
        try:
            mono, _ = _resample_to_analysis_sr(audio, sr)
            if len(mono) < 8:
                return {"sig": 1.0, "bak": 1.0, "ovr": 1.0}

            mag = _safe_stft(mono)
            snr_db = _estimate_snr_db(mono)
            flatness = _spectral_flatness(mag)
            contrast = _spectral_contrast(mag)

            # SIG: signal clarity — driven by spectral contrast and SNR
            # SNR 0→0.0, SNR 40+→1.0; contrast 0→0.0, contrast 30+→1.0
            sig_raw = 0.55 * float(np.clip(snr_db / 40.0, 0.0, 1.0)) + 0.45 * float(np.clip(contrast / 30.0, 0.0, 1.0))

            # BAK: background noise — inverse of flatness (flat=noisy, tonal=clean)
            # Also penalise very low SNR
            bak_raw = 0.70 * float(np.clip(1.0 - flatness * 1.5, 0.0, 1.0)) + 0.30 * float(
                np.clip(snr_db / 50.0, 0.0, 1.0)
            )

            # OVR: geometric mean of SIG and BAK (perceptual blending)
            ovr_raw = float(np.sqrt(sig_raw * bak_raw + 1e-12))

            # Scale from [0, 1] → [1, 5]
            sig = float(np.clip(1.0 + sig_raw * 4.0, 1.0, 5.0))
            bak = float(np.clip(1.0 + bak_raw * 4.0, 1.0, 5.0))
            ovr = float(np.clip(1.0 + ovr_raw * 4.0, 1.0, 5.0))

            logger.debug(
                "DNSMOSPredictor: snr=%.1f dB flat=%.3f contrast=%.1f dB → sig=%.2f bak=%.2f ovr=%.2f",
                snr_db,
                flatness,
                contrast,
                sig,
                bak,
                ovr,
            )
            return {"sig": sig, "bak": bak, "ovr": ovr}

        except Exception as exc:
            logger.warning("§G23 DNSMOSPredictor.predict DSP-Ersatzpfad (returning defaults): %s", exc, exc_info=True)
            return {"sig": 3.0, "bak": 3.0, "ovr": 3.0}


# ---------------------------------------------------------------------------
# SingMOSPredictor
# ---------------------------------------------------------------------------


class SingMOSPredictor:
    """
    Leichtgewichtiges SingMOS proxy for singing quality estimation.

    Focuses on vocal-specific features: HNR, F0 stability, formant clarity,
    and breathiness (low HNR in low energy regions).

    Returns a scalar MOS in [1, 5].
    """

    def predict(self, audio: np.ndarray, sr: int) -> float:
        """
        Predict singing MOS score.

        Args:
            audio: Audio signal (mono or stereo), float32/float64.
            sr:    Sample rate in Hz.

        Returns:
            MOS score in [1, 5] (higher = better singing quality).
        """
        try:
            mono, eff_sr = _resample_to_analysis_sr(audio, sr)
            if len(mono) < 32:
                return 3.0

            # Feature extraction
            hnr_db = _hnr_approx(mono, eff_sr)
            f0_stab = _f0_stability(mono, eff_sr)
            formant_cl = _formant_clarity(mono, eff_sr)
            snr_db = _estimate_snr_db(mono)

            # HNR component: clean singing 15–25 dB; degraded < 5 dB
            hnr_score = float(np.clip(hnr_db / 20.0, 0.0, 1.0))

            # F0 stability: stable = natural singing
            f0_score = f0_stab

            # Formant clarity: F1–F3 prominence = intelligibility + naturalness
            fc_score = formant_cl

            # SNR component: background noise penalty
            snr_score = float(np.clip(snr_db / 40.0, 0.0, 1.0))

            # Weighted combination tuned for singing quality
            sing_raw = 0.35 * hnr_score + 0.25 * f0_score + 0.25 * fc_score + 0.15 * snr_score
            mos = float(np.clip(1.0 + sing_raw * 4.0, 1.0, 5.0))

            logger.debug(
                "SingMOSPredictor: hnr=%.1f dB f0_stab=%.3f formant_cl=%.3f snr=%.1f dB → mos=%.2f",
                hnr_db,
                f0_stab,
                formant_cl,
                snr_db,
                mos,
            )
            return mos

        except Exception as exc:
            logger.warning(
                "§G23 SingMOSPredictor.predict DSP-Ersatzpfad (returning default 3.0): %s", exc, exc_info=True
            )
            return 3.0


# ---------------------------------------------------------------------------
# Thread-safe singletons
# ---------------------------------------------------------------------------

_dnsmos_instance: DNSMOSPredictor | None = None
_dnsmos_lock = threading.Lock()

_singmos_instance: SingMOSPredictor | None = None
_singmos_lock = threading.Lock()


def get_dnsmos_predictor() -> DNSMOSPredictor:
    """Gibt the DNSMOSPredictor singleton (thread-safe) zurück."""
    global _dnsmos_instance  # pylint: disable=global-statement
    if _dnsmos_instance is None:
        with _dnsmos_lock:
            if _dnsmos_instance is None:
                _dnsmos_instance = DNSMOSPredictor()
    return _dnsmos_instance


def get_singmos_predictor() -> SingMOSPredictor:
    """Gibt the SingMOSPredictor singleton (thread-safe) zurück."""
    global _singmos_instance  # pylint: disable=global-statement
    if _singmos_instance is None:
        with _singmos_lock:
            if _singmos_instance is None:
                _singmos_instance = SingMOSPredictor()
    return _singmos_instance
