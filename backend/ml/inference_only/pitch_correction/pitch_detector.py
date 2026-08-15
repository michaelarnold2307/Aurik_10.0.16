"""
CREPE-based pitch detection with vibrato and glissando analysis.

This module provides high-quality pitch tracking using the CREPE model
(Convolutional Representation for Pitch Estimation) and additional
analysis to distinguish pitch errors from musical expression.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .logging_config import setup_logger

logger = logging.getLogger(__name__)

# FCPE pitch plugin (CREPE ONNX Fallback + pYIN DSP intern)
try:
    from plugins.fcpe_plugin import get_fcpe_plugin as _get_fcpe_plugin

    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False

logger = setup_logger("pitch_detector")


@dataclass
class PitchAnalysis:
    """
    Comprehensive pitch analysis result.

    Attributes:
        f0_hz: Fundamental frequency in Hz (time series)
        confidence: Confidence scores (0-1) for each frame
        times: Time stamps for each frame
        vibrato_detected: Whether vibrato is present
        glissando_detected: Whether glissando/slides are present
        pitch_errors: Detected pitch errors exceeding threshold
        epistemic_confidence: Overall confidence in analysis (0-1)
    """

    f0_hz: np.ndarray
    confidence: np.ndarray
    times: np.ndarray
    vibrato_detected: bool
    glissando_detected: bool
    pitch_errors: list[dict]
    epistemic_confidence: float


class CREPEPitchDetector:
    """
    SOTA neural pitch detector with musical expression analysis.

    Uses CREPE for accurate pitch tracking, plus custom analysis
    to detect vibrato, glissando, and distinguish errors from expression.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        model_capacity: str = "full",  # 'tiny', 'small', 'medium', 'large', 'full'
        step_size: int = 10,  # ms between predictions
        viterbi: bool = True,  # Use Viterbi smoothing
    ):
        """
        Initialisiert CREPE pitch detector.

        Args:
            sample_rate: Audio sample rate (Hz)
            model_capacity: CREPE model size (trade-off between speed and accuracy)
            step_size: Time between pitch predictions (ms)
            viterbi: Whether to use Viterbi smoothing for cleaner pitch tracks
        """
        self.sample_rate = sample_rate
        self.model_capacity = model_capacity
        self.step_size = step_size
        self.viterbi = viterbi

        if not CREPE_AVAILABLE:
            logger.warning(
                "CREPE not verfuegbar. Install with: pip install crepe-tf2. Falling back to basic pitch tracking."
            )

        logger.info(
            "CREPEPitchDetector initialisiert: sr=%s, model=%s, step=%sms", sample_rate, model_capacity, step_size
        )

    def detect(self, audio: np.ndarray, min_confidence: float = 0.85) -> PitchAnalysis:
        """
        Erkennt pitch and analyze musical expression.

        Args:
            audio: Audio signal (mono, float32)
            min_confidence: Minimum confidence threshold for pitch detection

        Returns:
            PitchAnalysis with comprehensive pitch information
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Normalize to [-1, 1]
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()

        # FCPE pitch tracking (CREPE ONNX Fallback + pYIN DSP intern)
        if CREPE_AVAILABLE:
            try:
                _r = _get_fcpe_plugin().analyze(audio, self.sample_rate)
                times, f0_hz, confidence = _r.times_s, _r.f0_hz, _r.voiced_prob
            except Exception:
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
                times, f0_hz, confidence = self._fallback_pitch_detection(audio)
        else:
            # Fallback: Simple autocorrelation-based pitch detection
            times, f0_hz, confidence = self._fallback_pitch_detection(audio)

        # Filter low-confidence regions
        f0_hz_filtered = f0_hz.copy()
        f0_hz_filtered[confidence < min_confidence] = 0

        # Musical expression analysis
        vibrato_detected = self._detect_vibrato(f0_hz_filtered, times)
        glissando_detected = self._detect_glissando(f0_hz_filtered, times)
        pitch_errors = self._detect_pitch_errors(f0_hz_filtered, confidence, times)

        # Epistemic confidence: Can we reliably distinguish errors from expression?
        epistemic_confidence = self._compute_epistemic_confidence(
            confidence, vibrato_detected, glissando_detected, pitch_errors
        )

        logger.info(
            f"Pitch detection vollstaendig: {len(f0_hz)} frames, "
            f"vibrato={vibrato_detected}, glissando={glissando_detected}, "
            f"errors={len(pitch_errors)}, epistemic_conf={epistemic_confidence:.2f}"
        )

        return PitchAnalysis(
            f0_hz=f0_hz_filtered,
            confidence=confidence,
            times=times,
            vibrato_detected=vibrato_detected,
            glissando_detected=glissando_detected,
            pitch_errors=pitch_errors,
            epistemic_confidence=epistemic_confidence,
        )

    def _fallback_pitch_detection(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Einfaches autocorrelation-based pitch detection as fallback.
        """
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(self.step_size / 1000 * self.sample_rate)

        n_frames = 1 + (len(audio) - frame_length) // hop_length
        times = np.arange(n_frames) * self.step_size / 1000
        f0_hz = np.zeros(n_frames)
        confidence = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break

            frame = audio[start:end]

            # Autocorrelation — FFT-based O(N log N)
            from backend.core.core_utils import fft_autocorr

            autocorr = fft_autocorr(frame)

            # Find first peak after zero lag
            min_lag = int(self.sample_rate / 500)  # Max 500 Hz
            max_lag = int(self.sample_rate / 80)  # Min 80 Hz

            if max_lag < len(autocorr):
                peak_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
                f0_hz[i] = self.sample_rate / peak_lag if peak_lag > 0 else 0
                confidence[i] = autocorr[peak_lag] / autocorr[0] if autocorr[0] > 0 else 0

        return times, f0_hz, confidence

    def _detect_vibrato(
        self,
        f0_hz: np.ndarray,
        times: np.ndarray,
        min_rate: float = 4.0,  # Hz, typical vibrato 4-8 Hz
        max_rate: float = 8.0,
        min_extent_cents: float = 20.0,  # Minimum vibrato depth
    ) -> bool:
        """
        Erkennt vibrato (periodic pitch variation).

        Vibrato characteristics:
        - Rate: 4-8 Hz (typical)
        - Extent: 20-100 cents
        - Regular periodic pattern
        """
        # Remove zeros (unvoiced regions)
        voiced_mask = f0_hz > 0
        if np.sum(voiced_mask) < 10:
            return False

        f0_voiced = f0_hz[voiced_mask]
        times_voiced = times[voiced_mask]

        # Convert to cents (semitone = 100 cents)
        f0_cents = 1200 * np.log2(f0_voiced / 440.0)

        # Detrend (remove slow pitch drift)
        from scipy.signal import detrend

        f0_detrended = detrend(f0_cents)

        # Check variation extent
        variation_extent = np.std(f0_detrended)
        if variation_extent < min_extent_cents:
            return False

        # FFT to detect periodicity
        from scipy.fft import rfftfreq

        f0_detrended_arr = np.asarray(f0_detrended, dtype=np.float64)
        fft = np.abs(np.fft.rfft(f0_detrended_arr))
        freqs = rfftfreq(len(f0_detrended_arr), d=np.mean(np.diff(times_voiced)))

        # Check for peak in vibrato frequency range
        mask = (freqs >= min_rate) & (freqs <= max_rate)
        if np.sum(mask) == 0:
            return False

        peak_power = np.max(fft[mask])
        mean_power = np.mean(fft)

        # Vibrato detected if clear peak in expected range
        return peak_power > 3 * mean_power  # type: ignore[no-any-return]

    def _detect_glissando(
        self,
        f0_hz: np.ndarray,
        times: np.ndarray,
        min_slope_cents_per_sec: float = 200.0,  # Minimum slope for glissando
    ) -> bool:
        """
        Erkennt glissando (continuous pitch slide).

        Glissando characteristics:
        - Continuous pitch change
        - Slope > 200 cents/sec
        - Duration > 0.2s
        """
        # Remove zeros
        voiced_mask = f0_hz > 0
        if np.sum(voiced_mask) < 10:
            return False

        f0_voiced = f0_hz[voiced_mask]
        times_voiced = times[voiced_mask]

        # Convert to cents
        f0_cents = 1200 * np.log2(f0_voiced / 440.0)

        # Calculate slopes (cents/sec)
        dt = np.diff(times_voiced)
        df = np.diff(f0_cents)
        slopes = np.abs(df / dt)

        # Detect sustained high slopes (glissando)
        high_slope_mask = slopes > min_slope_cents_per_sec

        if not np.any(high_slope_mask):
            return False

        # Check for sustained regions (> 0.2s)
        from scipy.ndimage import label

        label_result = label(high_slope_mask)
        if isinstance(label_result, tuple):
            labeled, n_regions = label_result
        else:
            labeled = label_result
            n_regions = int(np.max(labeled))

        for region_id in range(1, n_regions + 1):
            region_mask = labeled == region_id
            region_duration = np.sum(dt[region_mask])

            if region_duration > 0.2:  # 200ms sustained glissando
                return True

        return False

    def _detect_pitch_errors(
        self,
        f0_hz: np.ndarray,
        confidence: np.ndarray,
        times: np.ndarray,
        error_threshold_cents: float = 25.0,  # Minimum deviation to be considered error
    ) -> list[dict]:
        """
        Erkennt pitch errors (deviations from expected pitch).

        This is conservative: only detects obvious errors that are unlikely
        to be intentional musical expression.

        Strategy:
        1. Use a small median filter to track local pitch trends (robust to vibrato)
        2. Detect large deviations from the smoothed trend
        3. Combine with jump detection for sudden errors
        """
        errors: list[Any] = []

        # Remove zeros
        voiced_mask = (f0_hz > 0) & (confidence > 0.85)
        if np.sum(voiced_mask) < 10:
            return errors

        f0_voiced = f0_hz[voiced_mask]
        times_voiced = times[voiced_mask]
        confidence_voiced = confidence[voiced_mask]

        # Use small median filter to track local trends
        # Small window (5-7) tracks vibrato, larger jumps are preserved
        from scipy.signal import medfilt

        window_size = min(7, len(f0_voiced) if len(f0_voiced) % 2 == 1 else len(f0_voiced) - 1)
        if window_size < 3:
            return errors

        f0_expected = medfilt(f0_voiced, kernel_size=window_size)

        # Calculate deviations in cents from smoothed trend
        cents_deviation = 1200 * np.log2(f0_voiced / (f0_expected + 1e-10))

        # Find significant errors (persistent deviations from trend)
        error_mask = np.abs(cents_deviation) > error_threshold_cents

        # Also detect sudden LARGE jumps to catch abrupt pitch errors
        # Use higher threshold (50 cents) to avoid vibrato false positives
        jump_threshold_cents = max(50.0, error_threshold_cents * 2)
        jump_deviations = np.zeros(len(f0_voiced))

        if len(f0_voiced) > 1:
            pitch_diff_cents = 1200 * np.log2((f0_voiced[1:] + 1e-10) / (f0_voiced[:-1] + 1e-10))
            jump_mask = np.abs(pitch_diff_cents) > jump_threshold_cents
            # Store jump magnitudes
            jump_deviations[1:] = pitch_diff_cents
            # Extend jump_mask to match length of error_mask
            jump_mask_extended = np.zeros(len(f0_voiced), dtype=bool)
            jump_mask_extended[1:] = jump_mask
            error_mask = error_mask | jump_mask_extended

        if not np.any(error_mask):
            return errors

        # Group contiguous errors
        from scipy.ndimage import label

        label_result = label(error_mask)
        if isinstance(label_result, tuple):
            labeled, n_errors = label_result
        else:
            labeled = label_result
            n_errors = int(np.max(labeled))

        for error_id in range(1, n_errors + 1):
            error_region = labeled == error_id
            error_idx = np.where(error_region)[0]

            # Use jump deviation if it's larger than trend deviation
            region_cents_dev = cents_deviation[error_region]
            region_jump_dev = jump_deviations[error_region]

            # Prefer jump deviation if it's significantly larger
            if np.max(np.abs(region_jump_dev)) > np.max(np.abs(region_cents_dev)):
                max_dev = np.max(np.abs(region_jump_dev))
                # For discrete step errors, frame smoothing can spread a 100-cent jump
                # over multiple frames and bias the mean downward. Use the jump peak
                # as representative deviation when jump evidence dominates.
                mean_dev = float(np.sign(np.sum(region_jump_dev)) * max_dev)

                # Estimate stable-step deviation from context windows around the region.
                # This recovers full step size (e.g. ~100 cents) when transition frames
                # dilute the per-frame jump magnitude.
                if error_idx.size > 0:
                    i0 = int(error_idx[0])
                    i1 = int(error_idx[-1])
                    pre = f0_voiced[max(0, i0 - 4) : i0]
                    post = f0_voiced[i1 + 1 : i1 + 5]
                    if pre.size >= 2 and post.size >= 2:
                        pre_med = float(np.median(pre))
                        post_med = float(np.median(post))
                        if pre_med > 1e-6 and post_med > 1e-6:
                            step_dev = float(1200.0 * np.log2(post_med / pre_med))
                            if abs(step_dev) > abs(mean_dev) + 20.0:
                                mean_dev = step_dev
            else:
                max_dev = np.max(np.abs(region_cents_dev))
                mean_dev = np.mean(region_cents_dev)

                # Guard against transition-frame dilution on step-like pitch errors:
                # if the region has a clear dominant peak but the arithmetic mean is
                # much smaller, use the signed peak as representative error magnitude.
                if max_dev >= 2.0 * error_threshold_cents and np.abs(mean_dev) < 0.8 * max_dev:
                    mean_dev = float(np.sign(np.sum(region_cents_dev)) * max_dev)

            errors.append(
                {
                    "start_time": times_voiced[error_region][0],
                    "end_time": times_voiced[error_region][-1],
                    "mean_deviation_cents": mean_dev,
                    "max_deviation_cents": max_dev,
                    "mean_confidence": np.mean(confidence_voiced[error_region]),
                }
            )

        return errors

    def _compute_epistemic_confidence(
        self, confidence: np.ndarray, vibrato_detected: bool, glissando_detected: bool, pitch_errors: list[dict]
    ) -> float:
        """
        Berechnet epistemic confidence: How sure are we that we can distinguish.
        pitch errors from musical expression?

        Low confidence when:
        - Vibrato present AND errors detected (harder to distinguish)
        - Glissando present AND errors detected (intentional vs. error)
        - Low detection confidence

        High confidence when:
        - No musical expression (clear errors)
        - Musical expression BUT no errors (clearly only expression)
        """
        # Base confidence from pitch detection
        mean_confidence = np.mean(confidence[confidence > 0]) if np.any(confidence > 0) else 0

        # Reduce confidence only if expression AND errors coexist
        expression_penalty = 0.0

        if len(pitch_errors) > 0:
            # Errors detected - only penalize if expression might confuse detection
            if vibrato_detected:
                expression_penalty += 0.15
            if glissando_detected:
                expression_penalty += 0.20
        else:
            # No errors detected - no penalty even if expression present
            # (clear that expression is not being misidentified as errors)
            expression_penalty = 0.0

        epistemic_confidence = max(0.0, mean_confidence - expression_penalty)

        return epistemic_confidence
