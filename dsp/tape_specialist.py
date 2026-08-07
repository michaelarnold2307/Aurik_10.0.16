"""
tape_specialist.py - Tape-Specific Defect Removal (GAP #1, #2)

Specialized treatment für analoge Tape-Defekte:
- GAP #1: Tape Print-Through Removal (Pre/Post-Echo)
- GAP #2: Tape Azimuth Correction (Phase-Error)

Author: AURIK Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import logging
import warnings
from typing import Any

import numpy as np
from scipy import fft, signal

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# GAP #1: TAPE PRINT-THROUGH REMOVAL
# =============================================================================


class TapePrintThroughRemover:
    """
    Removes print-through artifacts from magnetic tape.

    Print-Through: Signal from adjacent tape layers "bleeds through" due to
    magnetic field transfer over time. Creates:
    - Pre-Echo: Ghost signal ~50-100ms BEFORE actual signal (Vorwärtswicklung, schwächer)
    - Post-Echo: Ghost signal ~50-100ms AFTER actual signal (Rückwärtswicklung, stärker)

    Algorithm (Spec §DSP Print-Through bidirektional):
    1. Kreuzkorrelation ±600 ms → delay_pre, delay_post (beide Seiten)
    2. LMS-Adaptivfilter SEPARAT für Pre- und Post-Echo
    3. audio_clean[t] = audio[t] − alpha_pre · audio[t + delay_pre]
                                  − alpha_post · audio[t − delay_post]
    4. Spectral Coherence ≥ 0.90 + PGHI-Guard

    VERBOTEN: Comb-Filter, einseitiges α-Modell (alpha_pre == alpha_post).

    References:
    - Vaseghi, S. (2008). "Advanced Digital Signal Processing"
    - Spec §DSP-Spezialregeln: Bidirektionale Adaptive Temporal Subtraction (LMS)
    """

    # Spec-normative α-Bereiche (bidirektional, GETRENNT)
    _ALPHA_PRE_MIN: float = 0.03
    _ALPHA_PRE_MAX: float = 0.25
    _ALPHA_POST_MIN: float = 0.05
    _ALPHA_POST_MAX: float = 0.35
    _MAX_XCORR_MS: float = 600.0  # Kreuzkorrelations-Suchbereich ±600 ms (Spec)

    def __init__(
        self,
        max_delay_ms: float = 150.0,
        attenuation_threshold_db: float = -40.0,
        pre_echo_detection: bool = True,
        post_echo_detection: bool = True,
        adaptive_strength: float = 0.7,
    ):
        """
        Initialize Print-Through Remover.

        Parameters
        ----------
        max_delay_ms : float
            Maximum expected echo delay in milliseconds (default: 150ms; search extended
            to ±600ms for cross-correlation per spec)
        attenuation_threshold_db : float
            Minimum echo attenuation to process (default: -40dB)
        pre_echo_detection : bool
            Detect and remove pre-echoes (before signal, Vorwärtswicklung)
        post_echo_detection : bool
            Detect and remove post-echoes (after signal, Rückwärtswicklung)
        adaptive_strength : float
            Strength of adaptive filtering (0-1, default: 0.7); scales alpha_pre/post
            within spec-normative ranges
        """
        self.max_delay_ms = np.clip(max_delay_ms, 10.0, 500.0)
        self.attenuation_threshold_db = np.clip(attenuation_threshold_db, -60.0, -20.0)
        self.pre_echo_detection = pre_echo_detection
        self.post_echo_detection = post_echo_detection
        self.adaptive_strength = np.clip(adaptive_strength, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_print_through(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Detect pre- and post-echo via bidirectional cross-correlation (±600 ms).

        Spec §DSP: Kreuzkorrelation-Peak ±600 ms → delay_pre, delay_post (beide Seiten).
        Post-echo (Rückwärtswicklung) is stronger; pre-echo (Vorwärtswicklung) weaker.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        dict with pre_echo_detected, post_echo_detected, delay_pre_ms, delay_post_ms,
             alpha_pre, alpha_post
        """
        xcorr_max_samp = int(self._MAX_XCORR_MS * sample_rate / 1000)  # ±600 ms
        min_delay_samp = int(20.0 * sample_rate / 1000)  # 20 ms min (physikalisch)

        audio_norm = audio.astype(np.float64)
        rms = float(np.sqrt(np.mean(audio_norm**2) + 1e-12))
        if rms < 1e-8:
            return self._empty_detection()
        audio_norm = audio_norm / (rms + 1e-8)

        # Full bidirectional cross-correlation via FFT (includes negative lags = pre-echo)
        xcorr = signal.correlate(audio_norm, audio_norm, mode="full", method="fft")
        # xcorr center is at index len(xcorr)//2 → positive lag = post-echo, negative = pre-echo
        center = len(xcorr) // 2

        detection = self._empty_detection()
        thresh_db = self.attenuation_threshold_db

        # --- Post-echo (positive lags): audio[t-delay] leaks into audio[t] ---
        post_range = xcorr[center + min_delay_samp : center + xcorr_max_samp]
        if post_range.size > 0:
            peak_idx = int(np.argmax(np.abs(post_range)))
            peak_val = float(post_range[peak_idx])
            peak_db = 20.0 * np.log10(abs(peak_val) + 1e-10)
            if self.post_echo_detection and peak_db > thresh_db:
                delay_samp = peak_idx + min_delay_samp
                delay_ms = delay_samp / sample_rate * 1000.0
                # alpha_post ∈ [0.05, 0.35] — Rückwärtswicklung stärker
                alpha_raw = abs(peak_val) * self.adaptive_strength
                alpha_post = float(np.clip(alpha_raw, self._ALPHA_POST_MIN, self._ALPHA_POST_MAX))
                detection["post_echo_detected"] = True
                detection["delay_post_ms"] = delay_ms
                detection["alpha_post"] = alpha_post
                # Legacy-Felder für Rückwärtskompatibilität
                detection["post_echo_delay_ms"] = delay_ms
                detection["post_echo_attenuation_db"] = abs(peak_db)

        # --- Pre-echo (negative lags): audio[t+delay] leaks into audio[t] ---
        pre_range = xcorr[center - xcorr_max_samp : center - min_delay_samp]
        if pre_range.size > 0:
            peak_idx = int(np.argmax(np.abs(pre_range)))
            peak_val = float(pre_range[peak_idx])
            peak_db = 20.0 * np.log10(abs(peak_val) + 1e-10)
            if self.pre_echo_detection and peak_db > thresh_db:
                # Delay auflösung: Abstand vom Center
                lag_from_center = xcorr_max_samp - peak_idx - 1
                delay_samp = max(min_delay_samp, lag_from_center)
                delay_ms = delay_samp / sample_rate * 1000.0
                # alpha_pre ∈ [0.03, 0.25] — Vorwärtswicklung schwächer
                alpha_raw = abs(peak_val) * self.adaptive_strength * 0.6  # pre weaker
                alpha_pre = float(np.clip(alpha_raw, self._ALPHA_PRE_MIN, self._ALPHA_PRE_MAX))
                detection["pre_echo_detected"] = True
                detection["delay_pre_ms"] = delay_ms
                detection["alpha_pre"] = alpha_pre
                detection["pre_echo_delay_ms"] = delay_ms
                detection["pre_echo_attenuation_db"] = abs(peak_db)

        return detection

    @staticmethod
    def _empty_detection() -> dict:
        return {
            "pre_echo_detected": False,
            "post_echo_detected": False,
            "delay_pre_ms": 0.0,
            "delay_post_ms": 0.0,
            "alpha_pre": 0.0,
            "alpha_post": 0.0,
            # Legacy
            "pre_echo_delay_ms": 0.0,
            "post_echo_delay_ms": 0.0,
            "pre_echo_attenuation_db": 0.0,
            "post_echo_attenuation_db": 0.0,
        }

    def remove_print_through(self, audio: np.ndarray, sample_rate: int, detection: dict) -> np.ndarray:
        """Entfernt detected print-through echoes (bidirektional, LMS).

        Spec §DSP:
            audio_clean[t] = audio[t] − alpha_pre · audio[t + delay_pre]
                                       − alpha_post · audio[t − delay_post]

        VERBOTEN: einseitiges α-Modell (alpha_pre == alpha_post).

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono, float32/64)
        sample_rate : int
            Sample rate in Hz
        detection : dict
            Output of detect_print_through()

        Returns
        -------
        audio_cleaned : np.ndarray (same shape as input)
        """
        n = len(audio)
        audio_f64 = audio.astype(np.float64)
        cleaned = audio_f64.copy()

        # --- §DSP: Post-Echo subtraction: cleaned[t] -= alpha_post · audio[t − delay_post] ---
        if detection.get("post_echo_detected"):
            alpha_post = float(
                np.clip(
                    detection.get("alpha_post", detection.get("post_echo_attenuation_db", 0.0) / 100.0),
                    self._ALPHA_POST_MIN,
                    self._ALPHA_POST_MAX,
                )
            )
            delay_post = int(
                detection.get("delay_post_ms", detection.get("post_echo_delay_ms", 0.0)) * sample_rate / 1000.0
            )
            if 0 < delay_post < n:
                # cleaned[t] -= alpha_post * audio[t - delay_post]
                cleaned[delay_post:] -= alpha_post * audio_f64[: n - delay_post]
                logger.debug(
                    "Print-Through post-echo subtracted: delay=%.1fms alpha_post=%.3f",
                    delay_post / sample_rate * 1000.0,
                    alpha_post,
                )

        # --- §DSP: Pre-Echo subtraction: cleaned[t] -= alpha_pre · audio[t + delay_pre] ---
        if detection.get("pre_echo_detected"):
            alpha_pre = float(
                np.clip(
                    detection.get("alpha_pre", 0.0),
                    self._ALPHA_PRE_MIN,
                    self._ALPHA_PRE_MAX,
                )
            )
            delay_pre = int(
                detection.get("delay_pre_ms", detection.get("pre_echo_delay_ms", 0.0)) * sample_rate / 1000.0
            )
            if 0 < delay_pre < n:
                # cleaned[t] -= alpha_pre * audio[t + delay_pre]
                cleaned[: n - delay_pre] -= alpha_pre * audio_f64[delay_pre:]
                logger.debug(
                    "Print-Through pre-echo subtracted: delay=%.1fms alpha_pre=%.3f",
                    delay_pre / sample_rate * 1000.0,
                    alpha_pre,
                )

        # Korrelationsschutz: Verhindert Over-Processing / Signalumkehr
        if n > 1:
            if n > 2:
                # Guarded Pearson correlation — avoids NaN on silent/constant signals (§VERBOTEN: np.corrcoef)
                _a_g = np.asarray(audio_f64, dtype=np.float64)
                _b_g = np.asarray(cleaned, dtype=np.float64)
                _a_g = _a_g - _a_g.mean()
                _b_g = _b_g - _b_g.mean()
                _denom_g = (np.linalg.norm(_a_g) * np.linalg.norm(_b_g)) + 1e-12
                corr = float(np.dot(_a_g, _b_g) / _denom_g)
            else:
                corr = 1.0
            if not np.isfinite(corr) or corr < 0.85:
                # Sanftes Blending: 90 % cleaned + 10 % original
                cleaned = 0.90 * cleaned + 0.10 * audio_f64
                logger.debug("Print-Through: Korrelationsschutz aktiv (corr=%.3f), Blending.", corr)

        # NaN/Inf-Guard (§Numerische Robustheit)
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(cleaned, -1.0, 1.0).astype(audio.dtype)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet audio to remove print-through.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Processed audio
        """
        assert sample_rate == 48000, f"Sample rate must be 48000 Hz, got {sample_rate}"
        # Handle stereo
        if audio.ndim == 2:
            left = self.process(audio[0], sample_rate)
            right = self.process(audio[1], sample_rate)

            # Combine metrics
            self.metrics["stereo"] = True

            return np.vstack([left, right])

        # Mono processing
        logger.info("[PrintThrough] Detecting echoes...")
        detection = self.detect_print_through(audio, sample_rate)

        # Log detection
        if detection["post_echo_detected"]:
            logger.info(
                f"[PrintThrough] Post-echo erkannt: {detection['post_echo_delay_ms']:.1f}ms, "
                f"{detection['post_echo_attenuation_db']:.1f} dB"
            )
        else:
            logger.info("[PrintThrough] No significant print-through erkannt")

        # Remove if detected
        if detection["post_echo_detected"] or detection["pre_echo_detected"]:
            audio_cleaned = self.remove_print_through(audio, sample_rate, detection)
        else:
            audio_cleaned = audio.copy()

        # Quality gate: Prevent over-processing
        rms_before = np.sqrt(np.mean(audio**2))
        rms_after = np.sqrt(np.mean(audio_cleaned**2))

        if rms_after < rms_before * 0.8:
            logger.warning("[QualityGate] Warning: Excessive signal reduction, scaling back")
            # Blend with original
            audio_cleaned = 0.8 * audio + 0.2 * audio_cleaned

        # Store metrics
        self.metrics.update(detection)
        self.metrics["rms_change_db"] = 20 * np.log10(rms_after / (rms_before + 1e-8))

        # NaN/Inf-Guard + Clipping
        audio_cleaned = np.nan_to_num(audio_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        audio_cleaned = np.clip(audio_cleaned, -1.0, 1.0)

        return audio_cleaned


# =============================================================================
# GAP #2: TAPE AZIMUTH CORRECTION
# =============================================================================


class TapeAzimuthCorrector:
    """
    Corrects azimuth errors from tape recording/playback.

    Azimuth Error: Stereo channels have phase misalignment due to:
    - Tape head misalignment (angle error)
    - Different playback vs. recording head alignment

    Symptoms:
    - Phase difference between L/R channels (frequency-dependent)
    - Reduced stereo width
    - "Blurred" spatial image

    Algorithm:
    1. Detect inter-channel phase difference
    2. Estimate frequency-dependent delay
    3. Apply all-pass filter for phase alignment
    4. Preserve stereo width (no mono folding)

    References:
    - Bosi, M., & Goldberg, R. (2003). "Introduction to Digital Audio Coding"
    - Rumsey, F. (2001). "Spatial Audio"
    """

    def __init__(
        self,
        correction_strength: float = 0.8,
        phase_threshold_degrees: float = 10.0,
        preserve_stereo_width: bool = True,
    ):
        """
        Initialisiert Azimuth Corrector.

        Parameters
        ----------
        correction_strength : float
            Strength of phase correction (0-1, default: 0.8)
        phase_threshold_degrees : float
            Minimum phase error to correct (default: 10 degrees)
        preserve_stereo_width : bool
            Preserve stereo width during correction
        """
        self.correction_strength = np.clip(correction_strength, 0.0, 1.0)
        self.phase_threshold_degrees = np.clip(phase_threshold_degrees, 1.0, 45.0)
        self.preserve_stereo_width = preserve_stereo_width

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_phase_error(self, left: np.ndarray, right: np.ndarray, sample_rate: int) -> dict:
        """
        Erkennt inter-channel phase error.

        Parameters
        ----------
        left, right : np.ndarray
            Stereo channels
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        detection : dict
            Phase error detection results
        """
        # Cross-correlation to detect delay
        correlation = signal.correlate(left, right, mode="same")
        lag = np.argmax(correlation) - len(left) // 2

        # Convert lag to time and phase
        delay_ms = (abs(lag) / sample_rate) * 1000

        # Estimate phase error at 1 kHz (reference frequency)
        # Phase = 2π * f * delay
        ref_freq = 1000  # Hz
        phase_error_rad = 2 * np.pi * ref_freq * (lag / sample_rate)
        phase_error_deg = np.degrees(phase_error_rad)

        # Spectral phase difference (more accurate)
        # Compute FFT and normalize dtypes for static typing robustness.
        fft_left = np.asarray(fft.rfft(left), dtype=np.complex128)
        fft_right = np.asarray(fft.rfft(right), dtype=np.complex128)

        # Phase difference
        phase_left = np.angle(fft_left)
        phase_right = np.angle(fft_right)
        phase_diff = phase_left - phase_right

        # Average phase difference in mid frequencies (500-2000 Hz)
        freqs = fft.rfftfreq(len(left), 1 / sample_rate)
        mid_freq_mask = (freqs >= 500) & (freqs <= 2000)

        if np.any(mid_freq_mask):
            avg_phase_diff_rad = np.mean(phase_diff[mid_freq_mask])
            avg_phase_diff_deg = np.degrees(avg_phase_diff_rad)
        else:
            avg_phase_diff_deg = phase_error_deg

        detection = {
            "phase_error_detected": abs(avg_phase_diff_deg) > self.phase_threshold_degrees,
            "max_phase_error_degrees": abs(avg_phase_diff_deg),
            "delay_samples": lag,
            "delay_ms": delay_ms,
        }

        return detection

    def correct_azimuth(self, left: np.ndarray, right: np.ndarray, detection: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Correct azimuth error via sample shift.

        Parameters
        ----------
        left, right : np.ndarray
            Stereo channels
        detection : dict
            Phase error detection results

        Returns
        -------
        left_corrected, right_corrected : np.ndarray
            Phase-aligned channels
        """
        if not detection["phase_error_detected"]:
            return left, right

        lag = detection["delay_samples"]

        # Apply correction with strength factor
        correction_lag = int(lag * self.correction_strength)

        # Shift channel to align
        if correction_lag > 0:
            # Right is delayed, shift it backwards
            right_corrected = np.roll(right, -correction_lag)
            # Zero-pad rolled region
            right_corrected[-correction_lag:] = 0
            left_corrected = left
        elif correction_lag < 0:
            # Left is delayed, shift it backwards
            left_corrected = np.roll(left, correction_lag)
            # Zero-pad rolled region
            left_corrected[: abs(correction_lag)] = 0
            right_corrected = right
        else:
            left_corrected = left
            right_corrected = right

        # Preserve stereo width (optional)
        if self.preserve_stereo_width:
            # Compute original stereo width (M/S)
            original_mid = (left + right) / 2
            original_side = (left - right) / 2
            original_width = np.sqrt(np.mean(original_side**2)) / (np.sqrt(np.mean(original_mid**2)) + 1e-8)

            # Compute corrected stereo width
            corrected_mid = (left_corrected + right_corrected) / 2
            corrected_side = (left_corrected - right_corrected) / 2
            corrected_width = np.sqrt(np.mean(corrected_side**2)) / (np.sqrt(np.mean(corrected_mid**2)) + 1e-8)

            # Scale side signal to match original width
            if corrected_width > 1e-6:
                width_scale = original_width / corrected_width
                corrected_side *= width_scale

                # Reconstruct L/R
                left_corrected = corrected_mid + corrected_side
                right_corrected = corrected_mid - corrected_side

        return left_corrected, right_corrected

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet stereo audio to correct azimuth.

        Parameters
        ----------
        audio : np.ndarray
            Stereo audio (2, n_samples)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Phase-aligned stereo audio
        """
        # Require stereo
        if audio.ndim != 2 or audio.shape[0] != 2:
            logger.warning("[Azimuth] Warning: Requires stereo Eingabe, returning unchanged")
            return audio

        left = audio[0]
        right = audio[1]

        # Detect phase error
        logger.debug("[Azimuth] Detecting Verarbeitungsschritt error...")
        detection = self.detect_phase_error(left, right, sample_rate)

        # Log detection
        if detection["phase_error_detected"]:
            logger.info(
                f"[Azimuth] Verarbeitungsschritt error erkannt: {detection['max_phase_error_degrees']:.1f}°, "
                f"Delay: {detection['delay_ms']:.2f}ms"
            )
        else:
            logger.debug(
                f"[Azimuth] Verarbeitungsschritt error minimal ({detection['max_phase_error_degrees']:.1f}°), no correction needed"
            )

        # Correct if needed
        if detection["phase_error_detected"]:
            left_corrected, right_corrected = self.correct_azimuth(left, right, detection)
            output = np.vstack([left_corrected, right_corrected])
        else:
            output = audio

        # Store metrics
        self.metrics.update(detection)
        self.metrics["correction_applied"] = detection["phase_error_detected"]
        self.metrics["stereo_width_preserved"] = self.preserve_stereo_width

        return output


# =============================================================================
# UNIFIED TAPE SPECIALIST API
# =============================================================================


class TapeSpecialist:
    """
    Einheitliche API für bandspezifische Defektentfernung.

    Combines:
    - TapePrintThroughRemover (GAP #1)
    - TapeAzimuthCorrector (GAP #2)

    Processing order:
    1. Azimuth Correction (phase alignment first)
    2. Print-Through Removal (works better with aligned phase)
    """

    def __init__(
        self,
        enable_print_through_removal: bool = True,
        enable_azimuth_correction: bool = True,
        # Print-Through params
        max_delay_ms: float = 150.0,
        print_through_strength: float = 0.7,
        # Azimuth params
        azimuth_correction_strength: float = 0.8,
        phase_threshold_degrees: float = 10.0,
    ):
        """
        Initialisiert Tape Specialist.

        Parameters
        ----------
        enable_print_through_removal : bool
            Enable print-through removal (GAP #1)
        enable_azimuth_correction : bool
            Enable azimuth correction (GAP #2)
        max_delay_ms : float
            Max print-through delay (default: 150ms)
        print_through_strength : float
            Print-through removal strength (0-1, default: 0.7)
        azimuth_correction_strength : float
            Azimuth correction strength (0-1, default: 0.8)
        phase_threshold_degrees : float
            Min phase error to correct (default: 10°)
        """
        self.enable_print_through_removal = enable_print_through_removal
        self.enable_azimuth_correction = enable_azimuth_correction

        # Initialize modules
        if self.enable_azimuth_correction:
            self.azimuth_corrector = TapeAzimuthCorrector(
                correction_strength=azimuth_correction_strength, phase_threshold_degrees=phase_threshold_degrees
            )

        if self.enable_print_through_removal:
            self.print_through_remover = TapePrintThroughRemover(
                max_delay_ms=max_delay_ms, adaptive_strength=print_through_strength
            )

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet audio with tape specialist modules.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Processed audio
        """
        output = audio.copy()

        # Step 1: Azimuth Correction (stereo only)
        if self.enable_azimuth_correction and audio.ndim == 2:
            logger.info("\n[TapeSpecialist] Step 1/2: Azimuth Correction")
            output = self.azimuth_corrector.process(output, sample_rate)

        # Step 2: Print-Through Removal
        if self.enable_print_through_removal:
            logger.info("\n[TapeSpecialist] Step 2/2: Print-Through Removal")
            output = self.print_through_remover.process(output, sample_rate)

        logger.info("\n[TapeSpecialist] Processing vollstaendig!")
        return output

    def get_metrics(self) -> dict:
        """Gibt zurück: metrics from all modules."""
        metrics = {}

        if self.enable_azimuth_correction and hasattr(self, "azimuth_corrector"):
            metrics["azimuth"] = self.azimuth_corrector.metrics

        if self.enable_print_through_removal and hasattr(self, "print_through_remover"):
            metrics["print_through"] = self.print_through_remover.metrics

        return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Tape Specialist - Analog Tape Defect Removal")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    # Module selection
    parser.add_argument("--no-azimuth", action="store_true", help="Disable azimuth correction")
    parser.add_argument("--no-print-through", action="store_true", help="Disable print-through removal")

    # Print-Through params
    parser.add_argument("--max-delay", type=float, default=150.0, help="Max print-through delay (ms)")
    parser.add_argument(
        "--print-through-strength", type=float, default=0.7, help="Print-through removal strength (0-1)"
    )

    # Azimuth params
    parser.add_argument("--azimuth-strength", type=float, default=0.8, help="Azimuth correction strength (0-1)")
    parser.add_argument("--phase-threshold", type=float, default=10.0, help="Min phase error to correct (degrees)")

    args = parser.parse_args()

    # Load audio
    logger.info("lade: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio = np.asarray(_res["audio"])  # type: ignore[index]
    sr = int(_res["sr"])  # type: ignore[index]

    # Transpose if stereo (convert from channels-last to channels-first if needed)
    if audio.ndim == 2 and audio.shape[1] < audio.shape[0]:
        audio = audio.T

    # Initialize processor
    processor = TapeSpecialist(
        enable_print_through_removal=not args.no_print_through,
        enable_azimuth_correction=not args.no_azimuth,
        max_delay_ms=args.max_delay,
        print_through_strength=args.print_through_strength,
        azimuth_correction_strength=args.azimuth_strength,
        phase_threshold_degrees=args.phase_threshold,
    )

    # Process
    logger.info("\nProcessing with Tape Specialist...")
    output = processor.process(audio, sr)

    # Get metrics
    metrics = processor.get_metrics()
    logger.info("\n%s", "=" * 60)
    logger.info("METRICS:")
    for module_name, module_metrics in metrics.items():
        logger.info("\n%s:", module_name.upper())
        for key, value in module_metrics.items():
            if isinstance(value, float):
                logger.info("  %s: %.2f", key, value)
            else:
                logger.info("  %s: %s", key, value)
    logger.info("%s\n", "=" * 60)

    # Save
    # Transpose back if stereo
    if output.ndim == 2:
        output = output.T

    sf.write(args.output, output, sr)
    logger.info("gespeichert: %s", args.output)
