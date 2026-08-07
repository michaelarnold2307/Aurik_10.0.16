"""
phase_rotation.py – Phasendrehung/Ausrichtung für AURIK 6.0.

Improves mono compatibility through phase alignment using all-pass filters.

Key features:
- Phase correlation analysis
- All-pass filter-based phase rotation
- Frequency-dependent phase adjustment
- Mono compatibility optimization

Critical for:
- Radio/TV broadcast (mono playback)
- Club systems (phase-sensitive)
- Streaming services (downmix to mono)

References:
- Brainworx bx_digital V3
- Voxengo PHA-979
- Sound on Sound Phase Alignment Guide
"""

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

logger = logging.getLogger(__name__)


class PhaseRotator:
    """
    Phase Rotation for Mono Compatibility.

    Uses all-pass filters to adjust phase relationships between L/R channels.
    """

    def __init__(self, target_correlation: float = 0.7, rotation_amount: float = 0.5):
        """
        Initialisiert Phase Rotator.

        Args:
            target_correlation: Target phase correlation (0.5-1.0)
            rotation_amount: Amount of phase rotation (0.0-1.0)
        """
        self.target_correlation = np.clip(target_correlation, 0.3, 1.0)
        self.rotation_amount = np.clip(rotation_amount, 0.0, 1.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Wendet an: phase rotation for mono compatibility.

        Args:
            audio: Input audio (must be stereo)
            sr: Sample rate

        Returns:
            Tuple of (phase-rotated audio, metrics dict)
        """
        if audio.ndim != 2:
            # Mono: Phase-Rotation als Identität, aber mit normkonformer Metrik und Hinweis
            logger.info("[PhaseRotator] Mono erkannt – keine Rotation nötig, Identität zurückgegeben.")
            return audio, {
                "skipped": False,
                "reason": "mono audio (Identität)",
                "correlation_before": 1.0,
                "correlation_after": 1.0,
                "mono_compatible": True,
            }
        left = audio[0]
        right = audio[1]

        # Measure initial phase correlation
        correlation_before = self._measure_phase_correlation(left, right)

        # Check if rotation is needed
        if abs(correlation_before - self.target_correlation) < 0.05:
            logger.info(
                f"[PhaseRotator] Verarbeitungsschritt correlation already optimal ({correlation_before:.3f}), skipping rotation."
            )
            return audio, {
                "skipped": True,
                "reason": "already optimal",
                "correlation_before": float(correlation_before),
            }

        # Apply phase rotation
        left_rotated, right_rotated = self._rotate_phase(left, right, sr)

        # Mix with original based on rotation_amount
        left_final = left * (1 - self.rotation_amount) + left_rotated * self.rotation_amount
        right_final = right * (1 - self.rotation_amount) + right_rotated * self.rotation_amount

        result = np.stack([left_final, right_final], axis=0)

        # Measure final phase correlation
        correlation_after = self._measure_phase_correlation(left_final, right_final)

        # Measure mono compatibility
        mono_before = (left + right) / 2
        mono_after = (left_final + right_final) / 2
        mono_energy_ratio = np.sqrt(np.mean(mono_after**2)) / (np.sqrt(np.mean(mono_before**2)) + 1e-10)

        metrics = {
            "correlation_before": float(correlation_before),
            "correlation_after": float(correlation_after),
            "correlation_improvement": float(correlation_after - correlation_before),
            "mono_energy_ratio": float(mono_energy_ratio),
            "mono_compatible": correlation_after > 0.5,
            "rotation_amount": self.rotation_amount,
        }

        return result, metrics

    def _rotate_phase(self, left: np.ndarray, right: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Wendet an: phase rotation using all-pass filters.

        Strategy:
        - Rotate right channel forward (all-pass cascade)
        - Rotate left channel slightly backward (inverse coefficient)
        - This preserves stereo width while improving correlation
        """
        # Right channel: full forward rotation
        right_rotated = self._apply_allpass_cascade(right, sr, forward=True)

        # Left channel: slight counter-rotation for stereo width preservation
        # Use 30% of the rotation strength in opposite direction
        left_rotated = self._apply_allpass_cascade(left, sr, forward=False)
        left_rotated = left * 0.7 + left_rotated * 0.3

        return left_rotated, right_rotated

    def _apply_allpass_cascade(self, signal_in: np.ndarray, sr: int, forward: bool = True) -> np.ndarray:
        """
        Wendet Kaskade von Allpass-Filtern zur Phasendrehung an.

        3-band approach with frequency-dependent rotation strength.
        """
        # Split into 3 bands
        low_band = self._extract_band(signal_in, sr, 20, 500)
        mid_band = self._extract_band(signal_in, sr, 500, 2000)
        high_band = self._extract_band(signal_in, sr, 2000, min(20000, sr // 2 - 1))

        # Direction factor
        direction = 1.0 if forward else -1.0

        # Apply phase shifts — mid frequencies get maximum rotation
        low_shifted = self._allpass_filter(low_band, sr, center_freq=250, strength=0.3 * direction)
        mid_shifted = self._allpass_filter(mid_band, sr, center_freq=1000, strength=1.0 * direction)
        high_shifted = self._allpass_filter(high_band, sr, center_freq=4000, strength=0.6 * direction)

        return low_shifted + mid_shifted + high_shifted  # type: ignore[no-any-return]

    def _extract_band(self, signal: np.ndarray, sr: int, low_freq: float, high_freq: float) -> np.ndarray:
        """Extrahiert frequency band using bandpass filter."""
        nyquist = sr / 2
        low = max(low_freq / nyquist, 0.001)
        high = min(high_freq / nyquist, 0.999)

        if low >= high:
            return np.zeros_like(signal)

        sos = butter(4, [low, high], btype="band", output="sos")
        return sosfilt(sos, signal)  # type: ignore[no-any-return]

    def _allpass_filter(self, signal: np.ndarray, sr: int, center_freq: float, strength: float) -> np.ndarray:
        """
        Wendet an: first-order all-pass filter for phase rotation.

        All-pass filter: H(z) = (a + z^-1) / (1 + a*z^-1)
        where a = (tan(π*fc/fs) - 1) / (tan(π*fc/fs) + 1)
        """
        # Calculate coefficient
        fc_normalized = center_freq / sr
        a = (np.tan(np.pi * fc_normalized) - 1) / (np.tan(np.pi * fc_normalized) + 1)

        # Scale by strength
        a = a * strength

        # Filter coefficients
        b = [a, 1.0]
        a_coeff = [1.0, a]

        # Apply filter
        filtered = lfilter(b, a_coeff, signal)

        return filtered  # type: ignore[no-any-return]

    def _measure_phase_correlation(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Misst phase correlation between L/R channels.

        Returns value between -1 (out of phase) and +1 (in phase).
        Target for mono compatibility: > 0.5
        """
        # Normalize signals
        left_norm = left / (np.sqrt(np.mean(left**2)) + 1e-10)
        right_norm = right / (np.sqrt(np.mean(right**2)) + 1e-10)

        # Guarded Pearson correlation — avoids NaN when signals are zero-energy (§VERBOTEN: np.corrcoef)
        _lc = left_norm - float(np.mean(left_norm))
        _rc = right_norm - float(np.mean(right_norm))
        _nl = float(np.linalg.norm(_lc))
        _nr = float(np.linalg.norm(_rc))
        if _nl < 1e-12 or _nr < 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.clip(np.dot(_lc, _rc) / (_nl * _nr), -1.0, 1.0))

        return correlation


def create_phase_rotator(target_correlation: float = 0.7, rotation_amount: float = 0.5) -> PhaseRotator:
    """Factory-Funktion zum Erstellen einer PhaseRotator-Instanz."""
    return PhaseRotator(target_correlation=target_correlation, rotation_amount=rotation_amount)


# Normkonformes CLI-Beispiel (siehe Doku):
# if __name__ == "__main__":
#     import soundfile as sf
#     # Load test audio (must be stereo)
#     audio, sr = sf.read("stereo_test.wav")
#     rotator = create_phase_rotator(target_correlation=0.7, rotation_amount=0.5)
#     rotated, metrics = rotator.process(audio.T, sr)  # Transpose to (channels, samples)
#     print(f"Phase Rotation applied:")
#     print(f"  Correlation Before: {metrics['correlation_before']:.3f}")
#     print(f"  Correlation After: {metrics['correlation_after']:.3f}")
#     print(f"  Improvement: {metrics['correlation_improvement']:.3f}")
#     print(f"  Mono Compatible: {metrics['mono_compatible']}")
#     sf.write("rotated_phase.wav", rotated.T, sr)
