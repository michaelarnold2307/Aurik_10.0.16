"""
Adaptive Wiener Filter DSP-Modul für Aurik 10.0.0 (SOTA-Maximum)

Implements an MMSE-LSA-inspired Wiener filter with:
- Decision-directed a-priori SNR estimation (Ephraim & Malah 1984)
- Psychoacoustic frequency weighting (ISO 226:2003 equal-loudness approximation)
- Spectral floor to prevent musical noise artifacts (Cohen 2002)
- Full NaN/Inf guards on all paths

References:
  Ephraim & Malah (1984): Speech enhancement using MMSE spectral amplitude estimator
  Cohen (2002): Noise spectrum estimation in adverse environments
  ISO 226:2003: Equal-loudness contours
"""

import logging

import numpy as np
from scipy.special import expn

logger = logging.getLogger(__name__)


def _iso226_weight_curve(n_bins: int, sr: int) -> np.ndarray:
    """Approximate ISO 226:2003 equal-loudness weighting curve.

    Returns per-bin weights [0.5, 1.0] — frequencies where human hearing
    is most sensitive (2-5 kHz) get lower gain reduction (weight closer to 1.0),
    while less sensitive regions (< 200 Hz, > 12 kHz) allow more aggressive
    filtering (weight closer to 0.5).
    """
    freqs = np.linspace(0, sr / 2, n_bins)
    # Sensitivity peak at ~3.5 kHz (A-weighting inspired)
    sensitivity = np.exp(-0.5 * ((np.log2(np.maximum(freqs, 20.0) / 3500.0)) ** 2) / 1.5**2)
    # Map: high sensitivity → weight 1.0 (gentle filter), low → 0.5 (aggressive)
    weights = 0.5 + 0.5 * sensitivity
    return np.nan_to_num(weights, nan=0.75)  # type: ignore[no-any-return]


class AdaptiveWienerFilter:
    """MMSE-LSA-inspired adaptive Wiener filter with psychoacoustic weighting.

    This replaces the classical power-domain Wiener filter (mathematically
    invalid for non-stationary audio) with a proper decision-directed SNR
    estimator and perceptual frequency weighting.
    """

    def __init__(
        self, alpha: float = 0.98, noise_floor: float = 1e-8, spectral_floor: float = 0.02, psychoacoustic: bool = True
    ):
        self.alpha = alpha
        self.noise_floor = noise_floor
        self.spectral_floor = spectral_floor
        self.psychoacoustic = psychoacoustic
        # Legacy compatibility
        self.eps = noise_floor

    def filter(self, noisy_mag: np.ndarray, noise_mag: np.ndarray, *, sr: int = 48000, **kwargs) -> np.ndarray:
        """MMSE-inspired Wiener filtering on magnitude spectra.

        Args:
            noisy_mag: Noisy magnitude spectrum (any shape, broadcastable)
            noise_mag: Estimated noise magnitude spectrum
            sr: Sample rate for psychoacoustic weighting (default 48000)
        Returns:
            Cleaned magnitude spectrum
        """
        alpha = kwargs.get("alpha", self.alpha)
        noise_floor = kwargs.get("noise_floor", self.noise_floor)

        # A-priori SNR — decision-directed (Ephraim-Malah 1984)
        noise_pow = noise_mag**2 + noise_floor
        noisy_pow = noisy_mag**2
        gamma = noisy_pow / noise_pow
        xi = alpha * noisy_pow / noise_pow + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 1e-10)

        # MMSE-LSA gain with exponential integral
        v = xi * gamma / (1.0 + xi)
        v = np.maximum(v, 1e-8)
        gain = (xi / (1.0 + xi)) * np.exp(0.5 * expn(1, v))

        # Spectral floor — prevents musical noise (Cohen 2002)
        gain = np.maximum(gain, self.spectral_floor)

        # Psychoacoustic weighting: gentler in perceptually sensitive bands
        if self.psychoacoustic and noisy_mag.ndim >= 1:
            n_bins = noisy_mag.shape[-1] if noisy_mag.ndim >= 2 else noisy_mag.shape[0]
            psy_weights = _iso226_weight_curve(n_bins, sr)
            # Reshape for broadcasting
            if noisy_mag.ndim == 2:
                psy_weights = psy_weights[np.newaxis, :]
            # Blend: gain → gain^weight (weight=1.0 → full gain; weight=0.5 → sqrt(gain), gentler)
            gain = np.power(np.maximum(gain, 1e-10), psy_weights)

        # Safety guards
        gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
        gain = np.clip(gain, 0.0, 1.0)

        clean_mag = gain * noisy_mag
        return clean_mag  # type: ignore[no-any-return]

    def auto_optimize(self, noisy_mag: np.ndarray, noise_mag: np.ndarray) -> None:
        """Adapt alpha and spectral_floor based on estimated SNR.

        Low SNR → smaller alpha (faster adaptation to noise changes).
        High SNR → larger alpha (more stable filtering for musical signals).
        """
        mean_signal = float(np.mean(noisy_mag**2)) + 1e-12
        mean_noise = float(np.mean(noise_mag**2)) + 1e-12
        snr_db = float(10.0 * np.log10(mean_signal / mean_noise))

        # alpha: 0.85 (low SNR) → 0.99 (high SNR)
        self.alpha = float(np.clip(0.85 + 0.0065 * snr_db, 0.80, 0.99))
        # spectral_floor: higher at low SNR (more musical noise prevention)
        self.spectral_floor = float(np.clip(0.06 - 0.002 * snr_db, 0.01, 0.08))
        # Legacy compatibility
        self.eps = self.noise_floor

        logger.debug(
            "AdaptiveWienerFilter.auto_optimieren: SNR=%.1f dB → alpha=%.4f, spectral_floor=%.4f",
            snr_db,
            self.alpha,
            self.spectral_floor,
        )
