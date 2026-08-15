"""
Stereo Imaging Analyzer & Fixer Module

Implements GAP #21 from TIEFENANALYSE_MUSIKRESTAURATION_PROBLEME.md

Analysiert und korrigiert Stereo-Imaging-Probleme:
- Stereo-Breite (Width) - zu schmal oder zu breit
- Phase Correlation - Phase-Auslöschung Mono-Kompatibilität
- Stereo Balance - Links/Rechts-Ungleichgewicht
- Mid/Side Encoding - M/S Processing für präzise Corrections

Technik:
- Phase Correlation Analysis
- Mid/Side Encoding/Decoding
- Adaptive Stereo Width Enhancement/Reduction
- Balance Correction mit Headroom Preservation

Author: AURIK Development Team
Date: 9. Februar 2026
Version: 1.0.0
"""

import logging

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)
# pylint: disable=redefined-outer-name


class StereoImagingAnalyzer:
    """
    Analysiert Stereoabbildungs-Eigenschaften von Audiosignalen.

    Computes metrics like:
    - Phase correlation (mono compatibility)
    - Stereo width
    - Left/Right balance
    - Mid/Side content distribution
    """

    def __init__(self, frame_length_sec: float = 0.1):
        """
        Initialisiert Stereo Imaging Analyzer.

        Args:
            frame_length_sec: Duration of analysis frames (seconds)
        """
        self.frame_length_sec = frame_length_sec

    def analyze_phase_correlation(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sr: int,
    ) -> dict[str, float]:
        """
                Analysiert die Phasenkorrelation zwischen linkem und rechtem Kanal.

                Phase correlation ranges from -1 to +1:
                - +1: Perfect correlation (mono signal)
                - 0: No correlation (maximum stereo width)
                - -1: Perfect anti-correlation (phase reversed)

                Values below -0.5 indicate phase cancellation issues.

        envelope        Args:
                    left: Left channel audio
                    right: Right channel audio
                    sr: Sample rate

                Returns:
                    Dictionary with phase correlation metrics
        """
        # Frame-based analysis
        frame_length = int(self.frame_length_sec * sr)
        hop_length = frame_length // 2

        num_frames = (len(left) - frame_length) // hop_length + 1

        correlations = []

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length

            l_frame = left[start:end]
            r_frame = right[start:end]

            # Compute normalized cross-correlation at zero lag
            norm_l = np.linalg.norm(l_frame)
            norm_r = np.linalg.norm(r_frame)

            if norm_l > 1e-10 and norm_r > 1e-10:
                correlation = np.dot(l_frame, r_frame) / (norm_l * norm_r)
                correlations.append(correlation)

        if len(correlations) == 0:
            return {
                "phase_correlation_mean": 1.0,
                "phase_correlation_min": 1.0,
                "problematic_frames_ratio": 0.0,
            }

        correlations = np.array(correlations)  # type: ignore[assignment]

        # Count problematic frames (correlation < -0.5)
        problematic_frames = np.sum(correlations < -0.5)  # type: ignore[operator]
        problematic_ratio = problematic_frames / len(correlations)

        return {
            "phase_correlation_mean": float(np.mean(correlations)),
            "phase_correlation_min": float(np.min(correlations)),
            "problematic_frames_ratio": float(problematic_ratio),
        }

    def compute_stereo_width(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        """
        Berechnet effective stereo width.

        Width is computed based on the ratio of side (L-R) to mid (L+R) energy.

        Returns:
            Stereo width factor (0.0 = mono, 1.0 = normal, >1.0 = enhanced)
        """
        # Mid and Side signals
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        # Energy
        mid_energy = np.mean(mid**2)
        side_energy = np.mean(side**2)

        if mid_energy < 1e-10:
            return 0.0

        # Width factor: ratio of side to mid
        width = np.sqrt(side_energy / mid_energy)

        return float(width)

    def compute_balance(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> dict[str, float]:
        """
        Berechnet left/right balance.

        Args:
            left: Left channel
            right: Right channel

        Returns:
            Balance metrics
        """
        # RMS levels
        rms_left = np.sqrt(np.mean(left**2))
        rms_right = np.sqrt(np.mean(right**2))

        if rms_left < 1e-10 and rms_right < 1e-10:
            balance_db = 0.0
            balance_ratio = 1.0
        else:
            # Balance in dB
            balance_db = 20 * np.log10((rms_left + 1e-10) / (rms_right + 1e-10))
            balance_ratio = rms_left / (rms_right + 1e-10)

        return {
            "balance_db": float(balance_db),
            "balance_ratio": float(balance_ratio),
            "rms_left": float(rms_left),
            "rms_right": float(rms_right),
        }

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict:
        """
        Full stereo imaging analysis.

        Args:
            audio: Stereo audio (samples, 2) or (2, samples)
            sr: Sample rate

        Returns:
            Complete analysis metrics
        """
        # Auto-detect format
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
            # (channels, samples)
            left = audio[0]
            right = audio[1]
        else:
            # (samples, channels) - AURIK standard
            left = audio[:, 0]
            right = audio[:, 1]

        # Phase correlation analysis
        phase_metrics = self.analyze_phase_correlation(left, right, sr)

        # Stereo width
        width = self.compute_stereo_width(left, right)

        # Balance
        balance_metrics = self.compute_balance(left, right)

        # Combined metrics
        metrics = {
            **phase_metrics,
            "stereo_width": width,
            **balance_metrics,
        }

        return metrics


class StereoImagingFixer:
    """
    Corrects stereo imaging problems.

    Supports:
    - Width enhancement/reduction
    - Phase correlation improvement
    - Balance correction
    - M/S encoding/decoding
    """

    def __init__(
        self,
        target_width: float = 1.0,
        target_phase_correlation_min: float = -0.3,
        balance_tolerance_db: float = 1.0,
    ):
        """
        Initialisiert Stereo Imaging Fixer.

        Args:
            target_width: Target stereo width (1.0 = normal, <1.0 = narrower, >1.0 = wider)
            target_phase_correlation_min: Minimum acceptable phase correlation
            balance_tolerance_db: Maximum tolerable L/R imbalance (dB)
        """
        self.target_width = target_width
        self.target_phase_correlation_min = target_phase_correlation_min
        self.balance_tolerance_db = balance_tolerance_db

    def encode_mid_side(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Kodiert stereo signal to Mid/Side format.

        Args:
            left: Left channel
            right: Right channel

        Returns:
            Tuple of (mid, side)
        """
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        return mid, side

    def decode_mid_side(
        self,
        mid: np.ndarray,
        side: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dekodiert Mid/Side signal to stereo.

        Args:
            mid: Mid signal
            side: Side signal

        Returns:
            Tuple of (left, right)
        """
        left = mid + side
        right = mid - side
        return left, right

    def adjust_width(
        self,
        left: np.ndarray,
        right: np.ndarray,
        width_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Adjust stereo width.

        Args:
            left: Left channel
            right: Right channel
            width_factor: Width adjustment factor (1.0 = no change, <1.0 = narrower, >1.0 = wider)

        Returns:
            Adjusted (left, right)
        """
        # Encode to M/S
        mid, side = self.encode_mid_side(left, right)

        # Scale side signal
        side_adjusted = side * width_factor

        # Decode back to L/R
        left_out, right_out = self.decode_mid_side(mid, side_adjusted)

        # Safety only: no peak rescale, final clamp below.

        # NaN/Inf-Guard + Clipping
        left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
        right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)
        left_out = np.clip(left_out, -1.0, 1.0)
        right_out = np.clip(right_out, -1.0, 1.0)

        return left_out, right_out

    def correct_balance(
        self,
        left: np.ndarray,
        right: np.ndarray,
        target_balance_db: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Correct left/right balance.

        Args:
            left: Left channel
            right: Right channel
            target_balance_db: Target balance in dB (0 = equal, >0 = boost left, <0 = boost right)

        Returns:
            Balance-corrected (left, right)
        """
        # Current RMS levels
        rms_left = np.sqrt(np.mean(left**2))
        rms_right = np.sqrt(np.mean(right**2))

        if rms_left < 1e-10 or rms_right < 1e-10:
            return left.copy(), right.copy()

        # Current balance
        current_balance_db = 20 * np.log10(rms_left / rms_right)

        # Adjustment needed
        adjustment_db = target_balance_db - current_balance_db

        if abs(adjustment_db) < 0.1:  # Already balanced
            return left.copy(), right.copy()

        # Apply symmetric gain adjustment
        gain_left = 10 ** (adjustment_db / 40.0)  # Half the adjustment to left
        gain_right = 10 ** (-adjustment_db / 40.0)  # Half the adjustment to right

        left_out = left * gain_left
        right_out = right * gain_right

        # Safety only: no peak rescale, final clamp below.

        # NaN/Inf-Guard + Clipping
        left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
        right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)
        left_out = np.clip(left_out, -1.0, 1.0)
        right_out = np.clip(right_out, -1.0, 1.0)

        return left_out, right_out

    def fix_phase_cancellation(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sr: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fix phase cancellation issues.

        This detects frequency bands with severe phase cancellation
        and applies phase rotation to improve mono compatibility.

        Args:
            left: Left channel
            right: Right channel
            sr: Sample rate

        Returns:
            Phase-corrected (left, right)
        """
        # STFT for frequency-domain processing
        nperseg = 2048
        noverlap = nperseg // 2

        # Compute STFTs
        _f, _t, Zl = signal.stft(left, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, _, Zr = signal.stft(right, fs=sr, nperseg=nperseg, noverlap=noverlap)

        # Compute phase correlation per frequency band
        # correlation = Re(L * conj(R)) / (|L| * |R|)

        mag_l = np.abs(Zl) + 1e-10
        mag_r = np.abs(Zr) + 1e-10

        # Cross-correlation (complex)
        cross_corr = Zl * np.conj(Zr)

        # Normalized correlation (average over time)
        correlation = np.mean(np.real(cross_corr) / (mag_l * mag_r), axis=1)

        # Identify problematic frequency bands (correlation < threshold)
        problematic_bands = correlation < self.target_phase_correlation_min

        if np.any(problematic_bands):
            # Fix: Reduce side content in problematic bands
            # (gentle approach: reduce width in problematic bands only)

            # Convert to M/S in frequency domain
            Zm = (Zl + Zr) / 2.0
            Zs = (Zl - Zr) / 2.0

            # Attenuate side in problematic bands
            for i, is_problematic in enumerate(problematic_bands):
                if is_problematic:
                    # Reduce side by 50% in this band
                    Zs[i, :] *= 0.5

            # Convert back to L/R
            Zl_fixed = Zm + Zs
            Zr_fixed = Zm - Zs

            # Inverse STFT
            _, left_out = signal.istft(Zl_fixed, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, right_out = signal.istft(Zr_fixed, fs=sr, nperseg=nperseg, noverlap=noverlap)

            # Match length: istft kann kürzer als die Eingabe liefern → auf
            # Original-Länge auffüllen bzw. beschneiden (§Längen-Invariante).
            n_orig = len(left)
            left_out = np.asarray(left_out, dtype=np.float32)[:n_orig]
            right_out = np.asarray(right_out, dtype=np.float32)[:n_orig]
            if len(left_out) < n_orig:
                left_out = np.pad(left_out, (0, n_orig - len(left_out)))
            if len(right_out) < n_orig:
                right_out = np.pad(right_out, (0, n_orig - len(right_out)))

            return left_out, right_out

        return left.copy(), right.copy()

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        auto_correct: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Full stereo imaging correction pipeline.

        Args:
            audio: Stereo audio (samples, 2) or (2, samples)
            sr: Sample rate
            auto_correct: Automatically apply corrections based on analysis

        Returns:
            Tuple of (corrected_audio, metrics)
        """
        # Auto-detect format
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
            # (channels, samples) format
            left = audio[0]
            right = audio[1]
            input_format = "channels_first"
        else:
            # (samples, channels) - AURIK standard
            left = audio[:, 0]
            right = audio[:, 1]
            input_format = "channels_last"

        # Analyze current state
        analyzer = StereoImagingAnalyzer()
        metrics_before = analyzer.analyze(audio, sr)

        # Apply corrections if needed
        left_out = left.copy()
        right_out = right.copy()
        applied_corrections = []

        if auto_correct:
            # 1. Fix phase cancellation first
            if metrics_before["phase_correlation_min"] < self.target_phase_correlation_min:
                left_out, right_out = self.fix_phase_cancellation(left_out, right_out, sr)
                applied_corrections.append("phase_correction")

            # 2. Correct balance
            if abs(metrics_before["balance_db"]) > self.balance_tolerance_db:
                left_out, right_out = self.correct_balance(left_out, right_out, target_balance_db=0.0)
                applied_corrections.append("balance_correction")

            # 3. Adjust width if needed
            current_width = metrics_before["stereo_width"]
            if abs(current_width - self.target_width) > 0.1:
                width_factor = self.target_width / (current_width + 1e-10)
                # Limit adjustment to reasonable range
                width_factor = np.clip(width_factor, 0.5, 2.0)
                left_out, right_out = self.adjust_width(left_out, right_out, width_factor)
                applied_corrections.append("width_adjustment")

        # Reconstruct stereo audio in original format
        if input_format == "channels_first":
            audio_out = np.vstack([left_out, right_out])
        else:
            audio_out = np.column_stack([left_out, right_out])

        # Metrics after correction
        metrics_after = analyzer.analyze(audio_out, sr)

        # Combined metrics
        metrics = {
            "before": metrics_before,
            "after": metrics_after,
            "applied_corrections": applied_corrections,
            "num_corrections": len(applied_corrections),
        }

        return audio_out, metrics


if __name__ == "__main__":
    # Example usage

    # Create test signal with stereo imaging problems
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Mono source (poor stereo)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Create fake stereo with problems:
    # - Imbalanced (left louder)
    # - Narrow stereo field
    left = mono * 1.5 + 0.1 * np.random.randn(len(mono))
    right = mono * 0.8 + 0.05 * np.random.randn(len(mono))

    # Stereo audio (samples, channels)
    audio = np.column_stack([left, right])

    logger.info("Stereo Imaging Analysis & Correction Demo")
    logger.info(str("=" * 50))

    # Analyze
    analyzer = StereoImagingAnalyzer()
    metrics_before = analyzer.analyze(audio, sr)

    logger.info("\nAnalysis BEFORE correction:")
    logger.info("  Phase Correlation (mean): %.3f", metrics_before["phase_correlation_mean"])
    logger.info("  Phase Correlation (min): %.3f", metrics_before["phase_correlation_min"])
    logger.info("  Stereo Width: %.3f", metrics_before["stereo_width"])
    logger.info("  Balance: %.2f dB", metrics_before["balance_db"])
    logger.info("  Problematic Frames: %.1f%%", metrics_before["problematic_frames_ratio"] * 100)

    # Fix
    fixer = StereoImagingFixer(target_width=1.0, target_phase_correlation_min=-0.3)
    audio_fixed, metrics = fixer.process(audio, sr, auto_correct=True)

    logger.info("\nApplied Corrections:")
    for correction in metrics["applied_corrections"]:
        logger.info("  - %s", correction)

    logger.info("\nAnalysis AFTER correction:")
    logger.info("  Phase Correlation (mean): %.3f", metrics["after"]["phase_correlation_mean"])
    logger.info("  Phase Correlation (min): %.3f", metrics["after"]["phase_correlation_min"])
    logger.info("  Stereo Width: %.3f", metrics["after"]["stereo_width"])
    logger.info("  Balance: %.2f dB", metrics["after"]["balance_db"])
    logger.info("  Problematic Frames: %.1f%%", metrics["after"]["problematic_frames_ratio"] * 100)
