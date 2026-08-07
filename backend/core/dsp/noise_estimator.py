"""
backend/core/dsp/noise_estimator.py

IMCRA Noise Estimation — Canonical DSP module (DSP-Instructions, Aurik 10.0.0.x).

Implements Cohen's Improved Minima Controlled Recursive Averaging (IMCRA, 2003)
for non-stationary noise PSD estimation in music/speech.

Reference:
    Cohen (2003): "Noise Spectrum Estimation in Adverse Environments:
    Improved Minima Controlled Recursive Averaging."
    IEEE Trans. Speech Audio Process., 11(5), 466–475.

API (canonical per DSP-instructions §Noise-Estimation):
    noise_psd = compute_imcra_noise_estimate(audio, sr, alpha_d=0.85, alpha_s=0.9)
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from scipy.signal import stft as _scipy_stft

logger = logging.getLogger(__name__)

# ── Module-level defaults (DSP-instructions canonical) ───────────────────────
_DEFAULT_N_FFT = 2048
_DEFAULT_HOP = 512
_DEFAULT_ALPHA_D = 0.85  # recursive averaging weight (noise tracking speed)
_DEFAULT_ALPHA_S = 0.90  # smoothing weight for periodogram
_DEFAULT_Q = 0.50  # prior speech/signal absence probability
_B_MIN = 1.66  # IMCRA bias correction factor (Cohen 2003, Table I)
_INIT_FACTOR = 1.3  # conservative initialisation boost (DSP-instructions)
_INIT_DURATION_S = 2.0  # initialisation phase: first 2 seconds are conservative
_DELTA_DB = 5.0  # local indicator threshold in dB (IMCRA δ)
_M_SUBWINDOW = 4  # number of sub-windows for indicator (L in Cohen 2003)

# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: ImcraNoisEstimator | None = None
_lock = threading.Lock()


def get_noise_estimator() -> ImcraNoisEstimator:
    """Thread-safe singleton accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ImcraNoisEstimator()
    return _instance


class ImcraNoisEstimator:
    """IMCRA noise estimator — stateless per call, singleton for weight caching."""

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        alpha_d: float = _DEFAULT_ALPHA_D,
        alpha_s: float = _DEFAULT_ALPHA_S,
        n_fft: int = _DEFAULT_N_FFT,
        hop_length: int = _DEFAULT_HOP,
        q: float = _DEFAULT_Q,
    ) -> np.ndarray:
        """Führt aus: IMCRA noise estimation.

        Args:
            audio:       Mono float32 signal (1D).
            sr:          Sample rate in Hz.
            alpha_d:     Recursive averaging weight (0.85 per DSP-instructions).
            alpha_s:     Periodogram smoothing weight (0.90 per DSP-instructions).
            n_fft:       STFT window length in samples.
            hop_length:  STFT hop size in samples.
            q:           Prior signal absence probability (0.5 default).

        Returns:
            noise_psd: (n_freqs, n_frames) noise power spectral density estimate.
                       Same time-frequency grid as scipy.signal.stft with the
                       given n_fft / hop_length settings.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.mean(axis=0) if audio.ndim == 2 else audio.ravel()

        eps = 1e-12
        n_overlap = n_fft - hop_length
        _, _, Zxx = _scipy_stft(
            audio,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_overlap,
            window="hann",
            boundary="even",
            padded=True,
        )
        power = (np.abs(Zxx) ** 2).astype(np.float64)  # (n_freqs, n_frames)
        n_freqs, n_frames = power.shape

        # ── Initialisation phase — conservative (DSP-instructions: 1.3 × min) ──
        init_frames = max(1, int(_INIT_DURATION_S * sr / hop_length))
        init_end = min(init_frames, n_frames)
        S_min_init = np.min(power[:, :init_end], axis=1, keepdims=True)  # (F, 1)
        lambda_d = np.broadcast_to(np.maximum(_B_MIN * _INIT_FACTOR * S_min_init, eps), (n_freqs, 1)).copy()  # (F, 1)

        noise_psd = np.empty((n_freqs, n_frames), dtype=np.float64)

        # ── Sub-window length M for local indicator (Cohen 2003 §III-B) ──────────
        # M ≥ 3 frames; covers ~L × hop_length seconds of context
        M_local = max(3, int(0.15 * sr / hop_length))  # ~150 ms local window

        # Pre-compute smoothed periodogram (vectorised over time axis)
        # S_s(k,l) = α_s * S_s(k,l-1) + (1-α_s) * |Y(k,l)|²
        S_s = lambda_d.copy()  # initialise with conservative noise floor

        delta_lin = 10.0 ** (_DELTA_DB / 10.0)  # local indicator threshold

        for ll in range(n_frames):
            frame_power = power[:, ll : ll + 1]  # (F, 1)

            # Smoothed periodogram update
            S_s = alpha_s * S_s + (1.0 - alpha_s) * frame_power

            # Sliding minimum over past M_local frames (use available history)
            start = max(0, ll - M_local + 1)
            S_min = np.min(power[:, start : ll + 1], axis=1, keepdims=True)
            S_min = np.maximum(S_min, eps)

            # Bias-corrected noise floor candidate
            sigma_min = _B_MIN * S_min  # (F, 1)

            # Local activity indicator ζ_l: compares S_s to sigma_min
            # zeta > delta → signal/speech likely present
            zeta = np.clip(S_s / (sigma_min + eps), 0.0, 1000.0)
            indicator = (zeta > delta_lin).astype(np.float64)  # 1 = speech present

            # Speech presence probability via one-step likelihood ratio (IMCRA):
            # γ = a-posteriori SNR, ξ = a-priori SNR (instantaneous)
            gamma = np.clip(frame_power / (lambda_d + eps), 0.0, 1000.0)
            xi = np.maximum(gamma - 1.0, 0.0)
            nu = np.clip(xi * gamma / (xi + 1.0 + eps), 0.0, 500.0)
            # Likelihood ratio lambda_q (Cohen 2003, Eq. 13):
            lambda_q = (q / (1.0 - q + eps)) * (1.0 / (xi + 1.0 + eps)) * np.exp(np.clip(-xi + nu, -50.0, 50.0))
            # p(k,l) = Pr{speech present}
            p_speech = 1.0 / (1.0 + lambda_q)  # (F, 1)

            # Blend indicator with probabilistic estimate
            p_combined = np.clip(0.5 * p_speech + 0.5 * indicator, 0.0, 1.0)

            # Adaptive recursive averaging weight (Cohen 2003, Eq. 15):
            # α_k = α_d  when speech absent (p→0) → fast noise tracking
            # α_k → 1.0  when speech present   (p→1) → freeze noise estimate
            alpha_k = alpha_d + (1.0 - alpha_d) * p_combined

            # Conditional update: use sigma_min candidate when speech absent
            lambda_d = alpha_k * lambda_d + (1.0 - alpha_k) * sigma_min
            lambda_d = np.maximum(lambda_d, eps)

            noise_psd[:, ll] = lambda_d[:, 0]

        return noise_psd.astype(np.float32)  # type: ignore[no-any-return]


# ── Canonical public API (DSP-instructions) ──────────────────────────────────


def compute_imcra_noise_estimate(
    audio: np.ndarray,
    sr: int,
    alpha_d: float = _DEFAULT_ALPHA_D,
    alpha_s: float = _DEFAULT_ALPHA_S,
    n_fft: int = _DEFAULT_N_FFT,
    hop_length: int = _DEFAULT_HOP,
    q: float = _DEFAULT_Q,
) -> np.ndarray:
    """IMCRA noise PSD estimation (Cohen 2003).

    Canonical function referenced by DSP-instructions §Noise-Estimation.

    Args:
        audio:       Mono audio signal (1D float32 or float64).
        sr:          Sample rate in Hz (any value; no assert — analysis module).
        alpha_d:     Recursive averaging weight during noise tracking (0.85).
        alpha_s:     Periodogram smoothing weight (0.90).
        n_fft:       STFT window length (default 2048).
        hop_length:  STFT hop length (default 512).
        q:           Prior signal absence probability (0.5).

    Returns:
        noise_psd: float32 array of shape (n_freqs, n_frames) with estimated
                   noise power per STFT bin and frame.  Same time-frequency
                   grid as scipy.signal.stft(audio, nperseg=n_fft,
                   noverlap=n_fft-hop_length, boundary='reflect').

    Notes:
        - Initialphase (2s): conservative estimate = 1.3 × b_min × sliding-min
        - alpha_d=0.85, alpha_s=0.9 are the canonical values per DSP-instructions.
        - Does NOT assert sr == 48000 (analysis module, not processing phase).
    """
    try:
        return get_noise_estimator().estimate(
            audio=audio,
            sr=sr,
            alpha_d=alpha_d,
            alpha_s=alpha_s,
            n_fft=n_fft,
            hop_length=hop_length,
            q=q,
        )
    except Exception as _e:
        logger.warning("berechnen_imcra_noise_estimate Ersatzpfad (error: %s)", _e)
        # Fallback: simple sliding-minimum noise floor (1D averaged over time)
        n_overlap = n_fft - hop_length
        try:
            _, _, Zxx = _scipy_stft(audio, fs=sr, nperseg=n_fft, noverlap=n_overlap, window="hann", boundary="even")
            power = (np.abs(Zxx) ** 2).astype(np.float32)
            noise_floor = np.median(power, axis=1, keepdims=True)
            return np.broadcast_to(noise_floor, power.shape).copy()  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("noise_estimator.py::berechnen_imcra_noise_estimate Ersatzpfad: %s", e)
            return np.full((n_fft // 2 + 1, 1), 1e-8, dtype=np.float32)  # type: ignore[no-any-return]


__all__ = [
    "compute_imcra_noise_estimate",
    "get_noise_estimator",
    "ImcraNoisEstimator",
]
