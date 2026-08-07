"""
stereo_width_enhancer.py - Professional Mid-Side Stereo Width Enhancement

Production-ready stereo imaging tool for modern "wide" sound while maintaining
mono compatibility and phase coherence. Complements the AI-based stereo_enhancer.py.

Features:
- Mid-Side encoding/decoding
- Adjustable stereo width (0.0 = mono, 1.0 = normal, 2.0 = ultra-wide)
- Mono compatibility check
- Phase correlation monitoring
- Intelligent limiting to prevent extreme width

Author: AURIK Development Team
Version: 1.0.0
Date: 10. Februar 2026
"""

import numpy as np


class StereoWidthEnhancer:
    """
    Mid-Side based Professional Stereo Width Enhancement.

    Enhances stereo width by amplifying the Side channel (difference signal)
    while preserving the Mid channel (sum signal) for mono compatibility.

    Parameters
    ----------
    width_factor : float
        Stereo width multiplier (0.0-3.0):
        - 0.0 = Mono (sum both channels)
        - 1.0 = Original width (no change)
        - 1.5 = 50% wider (recommended for subtle enhancement)
        - 2.0 = 100% wider (dramatic effect)
        - 3.0 = Maximum safe width
    mono_check : bool
        If True, verify mono compatibility after processing
    phase_check : bool
        If True, monitor phase correlation
    safe_mode : bool
        If True, apply intelligent limiting to prevent extreme width
    """

    def __init__(
        self, width_factor: float = 1.5, mono_check: bool = True, phase_check: bool = True, safe_mode: bool = True
    ):
        self.width_factor = np.clip(width_factor, 0.0, 3.0)
        self.mono_check = mono_check
        self.phase_check = phase_check
        self.safe_mode = safe_mode

    def lr_to_ms(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Konvertiert Left/Right to Mid/Side.

        Mid = (L + R) / 2  (sum - mono compatible)
        Side = (L - R) / 2  (difference - stereo information)

        Parameters
        ----------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel

        Returns
        -------
        mid : np.ndarray
            Mid channel (sum)
        side : np.ndarray
            Side channel (difference)
        """
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        return mid, side

    def ms_to_lr(self, mid: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Konvertiert Mid/Side back to Left/Right.

        L = M + S
        R = M - S

        Parameters
        ----------
        mid : np.ndarray
            Mid channel
        side : np.ndarray
            Side channel

        Returns
        -------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel
        """
        left = mid + side
        right = mid - side
        return left, right

    def compute_phase_correlation(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Berechnet phase correlation between left and right channels.

        Returns
        -------
        correlation : float
            Phase correlation (-1.0 to +1.0):
            - +1.0 = Perfect correlation (mono)
            - 0.0 = No correlation (uncorrelated stereo)
            - -1.0 = Perfect anti-correlation (phase-reversed)
        """
        # Pearson correlation coefficient
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)

        numerator = np.sum(left_centered * right_centered)
        denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))

        correlation = numerator / denominator if denominator > 0 else 0.0

        return np.clip(correlation, -1.0, 1.0)  # type: ignore[no-any-return]

    def check_mono_compatibility(
        self, left: np.ndarray, right: np.ndarray, threshold_db: float = -40.0
    ) -> tuple[bool, float]:
        """
        Prüft if stereo signal is mono-compatible.

        When collapsed to mono (L+R), significant cancellation indicates
        phase problems or excessive width.

        Parameters
        ----------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel
        threshold_db : float
            Acceptable loss threshold in dB (default: -40 dB)

        Returns
        -------
        is_compatible : bool
            True if mono-compatible
        loss_db : float
            Energy loss when collapsed to mono (in dB)
        """
        # Compute stereo energy
        stereo_energy = np.mean(left**2 + right**2)

        # Compute mono energy (collapsed)
        mono = (left + right) / 2.0
        mono_energy = np.mean(mono**2) * 2  # *2 to account for both channels

        # Compute loss
        loss_db = 10 * np.log10(mono_energy / stereo_energy) if stereo_energy > 0 and mono_energy > 0 else -np.inf

        is_compatible = loss_db > threshold_db

        return is_compatible, loss_db

    def _apply_safe_limiting(self, mid: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Wendet an: intelligent limiting to prevent extreme width artifacts.

        Limits Side energy relative to Mid energy to prevent:
        - Phase issues
        - Excessive low-frequency spread
        - Mono incompatibility
        """
        # Compute RMS energies
        mid_rms = np.sqrt(np.mean(mid**2))
        side_rms = np.sqrt(np.mean(side**2))

        # Maximum safe Side/Mid ratio (empirically determined)
        max_ratio = 2.0

        if side_rms > 0 and mid_rms > 0:
            current_ratio = side_rms / mid_rms

            if current_ratio > max_ratio:
                # Scale down Side to maintain safe ratio
                scale_factor = max_ratio / current_ratio
                side = side * scale_factor

        return mid, side

    def process(self, audio: np.ndarray, return_metrics: bool = False) -> tuple[np.ndarray, dict | None]:
        """
        Wendet an: stereo width enhancement.

        Parameters
        ----------
        audio : np.ndarray
            Input stereo audio:
            - Shape (2, samples): channels-first format
            - Shape (samples, 2): samples-first format
        return_metrics : bool
            If True, return metrics dictionary

        Returns
        -------
        audio_enhanced : np.ndarray
            Enhanced audio with same shape as input
        metrics : dict, optional
            Processing metrics if return_metrics=True:
            - 'width_applied': Actual width factor applied
            - 'phase_correlation_input': Input phase correlation
            - 'phase_correlation_output': Output phase correlation
            - 'mono_compatible': Bool, mono compatibility check
            - 'mono_loss_db': Energy loss in mono (dB)
        """
        # Input validation
        if audio.ndim != 2:
            raise ValueError("Input must be stereo (2D array)")

        if audio.shape[0] == 2 and audio.shape[1] > 2:
            # Already (channels, samples)
            channels_first = True
            left = audio[0, :]
            right = audio[1, :]
        elif audio.shape[1] == 2 and audio.shape[0] > 2:
            # (samples, channels) format
            channels_first = False
            left = audio[:, 0]
            right = audio[:, 1]
        else:
            raise ValueError("Could not determine stereo format. Expected (2, N) or (N, 2)")

        # Measure input metrics
        if return_metrics or self.phase_check:
            phase_corr_input = self.compute_phase_correlation(left, right)

        # Convert to Mid-Side
        mid, side = self.lr_to_ms(left, right)

        # Apply width enhancement to Side channel
        side_enhanced = side * self.width_factor

        # Apply safe limiting if enabled
        if self.safe_mode:
            mid, side_enhanced = self._apply_safe_limiting(mid, side_enhanced)

        # Convert back to Left-Right
        left_enhanced, right_enhanced = self.ms_to_lr(mid, side_enhanced)

        # Check mono compatibility
        if return_metrics or self.mono_check:
            is_mono_compatible, mono_loss_db = self.check_mono_compatibility(left_enhanced, right_enhanced)

        # Measure output metrics
        if return_metrics or self.phase_check:
            phase_corr_output = self.compute_phase_correlation(left_enhanced, right_enhanced)

        # Reconstruct output in original format
        if channels_first:
            audio_enhanced = np.stack([left_enhanced, right_enhanced], axis=0)
        else:
            audio_enhanced = np.stack([left_enhanced, right_enhanced], axis=1)

        # Return with metrics if requested
        if return_metrics:
            metrics = {
                "width_applied": self.width_factor,
                "phase_correlation_input": phase_corr_input,
                "phase_correlation_output": phase_corr_output,
                "mono_compatible": is_mono_compatible,
                "mono_loss_db": mono_loss_db,
            }
            return audio_enhanced, metrics

        return audio_enhanced, None

    def analyze_stereo_field(self, audio: np.ndarray) -> dict:
        """
        Analysiert Stereofeld-Eigenschaften ohne Modifikation.

        Parameters
        ----------
        audio : np.ndarray
            Input stereo audio

        Returns
        -------
        analysis : dict
            Stereo field analysis:
            - 'width_estimate': Estimated current stereo width (0-2)
            - 'phase_correlation': Phase correlation (-1 to +1)
            - 'mid_energy_db': Mid channel energy (dB)
            - 'side_energy_db': Side channel energy (dB)
            - 'mono_compatible': Bool, mono compatibility
            - 'mono_loss_db': Mono collapse loss (dB)
        """
        # Parse stereo format
        if audio.shape[0] == 2:
            left, right = audio[0, :], audio[1, :]
        else:
            left, right = audio[:, 0], audio[:, 1]

        # Convert to Mid-Side
        mid, side = self.lr_to_ms(left, right)

        # Compute energies
        mid_energy = np.mean(mid**2)
        side_energy = np.mean(side**2)

        mid_energy_db = 10 * np.log10(mid_energy) if mid_energy > 0 else -np.inf
        side_energy_db = 10 * np.log10(side_energy) if side_energy > 0 else -np.inf

        # Estimate current width
        width_estimate = np.sqrt(side_energy / mid_energy) if mid_energy > 0 and side_energy > 0 else 0.0

        # Phase correlation
        phase_corr = self.compute_phase_correlation(left, right)

        # Mono compatibility
        is_mono_compatible, mono_loss_db = self.check_mono_compatibility(left, right)

        return {
            "width_estimate": width_estimate,
            "phase_correlation": phase_corr,
            "mid_energy_db": mid_energy_db,
            "side_energy_db": side_energy_db,
            "mono_compatible": is_mono_compatible,
            "mono_loss_db": mono_loss_db,
        }
