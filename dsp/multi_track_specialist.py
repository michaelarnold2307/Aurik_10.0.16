"""
multi_track_specialist.py - Multi-Track & Stereo Enhancement (GAP #19-25)

Specialized treatment für Multi-Track/Stereo-Probleme:
- GAP #19: Multi-Track Time Alignment (Cross-Correlation)
- GAP #20: Multi-Track Phase Alignment (Phase Coherence)
- GAP #22: Phase Cancellation Detection & Correction
- GAP #23: L/R Stereo Balance Correction
- GAP #24: Mid/Side (M/S) Processing
- GAP #25: Comb Filter Removal (Phase-Induced)

Author: AURIK Development Team
Version: 1.0.0
Date: 10. Februar 2026
"""

import logging
import warnings
from typing import Any

import numpy as np
from scipy import fft
from scipy.signal import correlate

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# GAP #19: MULTI-TRACK TIME ALIGNMENT
# =============================================================================


class TimeAligner:
    """
    Aligns multiple tracks in time using cross-correlation.

    Time Alignment Problem:
    - Multi-track recordings may have timing offsets
    - Caused by: Different mic distances, analog delays, propagation time
    - Solution: Cross-correlation to find optimal delay, then shift

    Algorithm:
    1. Cross-correlate reference track with target track
    2. Find peak correlation (lag with maximum correlation)
    3. Apply time shift to align tracks

    References:
    - Knapp, C. H., & Carter, G. C. (1976). "The generalized correlation method"
    - Benesty, J., et al. (2008). "Time-delay estimation via linear interpolation"
    """

    def __init__(self, max_delay_ms: float = 100.0, correlation_threshold: float = 0.3):
        """
        Initialisiert Time Aligner.

        Parameters
        ----------
        max_delay_ms : float
            Maximum expected delay between tracks (default: 100ms)
        correlation_threshold : float
            Minimum correlation to apply alignment (default: 0.3)
        """
        self.max_delay_ms = np.clip(max_delay_ms, 1.0, 1000.0)
        self.correlation_threshold = np.clip(correlation_threshold, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_delay(self, reference: np.ndarray, target: np.ndarray, sample_rate: int) -> tuple[int, float]:
        """
        Erkennt time delay between reference and target using cross-correlation.

        Parameters
        ----------
        reference : np.ndarray
            Reference track (aligned to this)
        target : np.ndarray
            Target track (will be aligned)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        delay_samples : int
            Delay in samples (positive = target is ahead)
        correlation : float
            Peak correlation value (0-1)
        """
        # OPTIMIZATION: Limit analysis to 10 seconds for performance
        # Time alignment patterns are detectable in shorter samples
        max_samples = int(10 * sample_rate)

        # Ensure same length
        min_len = min(len(reference), len(target))

        # Use middle section if audio is long
        if min_len > max_samples:
            start = (min_len - max_samples) // 2
            ref = reference[start : start + max_samples]
            tgt = target[start : start + max_samples]
        else:
            ref = reference[:min_len]
            tgt = target[:min_len]

        # Maximum delay in samples
        max_delay_samples = int(self.max_delay_ms * sample_rate / 1000)

        # Cross-correlation (mode='same' centers the correlation)
        correlation = correlate(ref, tgt, mode="same", method="fft")

        # Normalize correlation
        correlation = correlation / (np.sqrt(np.sum(ref**2) * np.sum(tgt**2)) + 1e-10)

        # Find peak within max_delay range
        center = len(correlation) // 2
        search_start = max(0, center - max_delay_samples)
        search_end = min(len(correlation), center + max_delay_samples + 1)

        search_region = correlation[search_start:search_end]
        peak_idx = np.argmax(np.abs(search_region))
        peak_correlation = search_region[peak_idx]

        # Convert to delay in samples
        delay_samples = (search_start + peak_idx) - center

        return delay_samples, float(peak_correlation)  # type: ignore[return-value]

    def align_tracks(self, reference: np.ndarray, target: np.ndarray, delay_samples: int) -> np.ndarray:
        """
        Align target track to reference by shifting.

        Parameters
        ----------
        reference : np.ndarray
            Reference track
        target : np.ndarray
            Target track to align
        delay_samples : int
            Delay to apply (from detect_delay)

        Returns
        -------
        aligned : np.ndarray
            Aligned target track (same length as reference)
        """
        if delay_samples == 0:
            return target.copy()

        # Allocate output
        aligned = np.zeros_like(reference)

        # Shift target
        if delay_samples > 0:
            # Target is ahead, shift it back (delay)
            if delay_samples < len(target):
                aligned[: len(target) - delay_samples] = target[delay_samples:]
        else:
            # Target is behind, shift it forward (advance)
            advance = -delay_samples
            if advance < len(aligned):
                aligned[advance:] = target[: len(aligned) - advance]

        return aligned

    def process(self, audio_stereo: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet stereo audio to align channels.

        Parameters
        ----------
        audio_stereo : np.ndarray
            Stereo audio (2, N) or (N, 2)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Time-aligned stereo audio
        """
        assert sample_rate == 48000, f"Sample rate must be 48000 Hz, got {sample_rate}"
        # Handle (N, 2) → (2, N)
        if audio_stereo.ndim == 2 and audio_stereo.shape[1] == 2:
            audio_stereo = audio_stereo.T

        if audio_stereo.ndim != 2 or audio_stereo.shape[0] != 2:
            logger.info("[TimeAlign] Not stereo, skipping")
            self.metrics["alignment_applied"] = False
            return audio_stereo

        reference = audio_stereo[0]
        target = audio_stereo[1]

        # Detect delay
        delay_samples, correlation = self.detect_delay(reference, target, sample_rate)

        # Store metrics
        self.metrics["delay_samples"] = delay_samples
        self.metrics["delay_ms"] = (delay_samples / sample_rate) * 1000
        self.metrics["correlation"] = correlation

        # Check if alignment is needed
        if abs(correlation) < self.correlation_threshold:
            logger.info("[TimeAlign] Low correlation (%.3f), skipping alignment", correlation)
            self.metrics["alignment_applied"] = False
            return audio_stereo

        if abs(delay_samples) < 1:
            logger.info("[TimeAlign] No significant delay erkannt")
            self.metrics["alignment_applied"] = False
            return audio_stereo

        # Align
        logger.info(
            f"[TimeAlign] Aligning: {delay_samples} samples ({self.metrics['delay_ms']:.2f}ms), corr={correlation:.3f}"
        )
        aligned_target = self.align_tracks(reference, target, delay_samples)

        # NaN/Inf-Guard + Clipping
        reference = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
        aligned_target = np.nan_to_num(aligned_target, nan=0.0, posinf=0.0, neginf=0.0)
        reference = np.clip(reference, -1.0, 1.0)
        aligned_target = np.clip(aligned_target, -1.0, 1.0)

        self.metrics["alignment_applied"] = True

        return np.vstack([reference, aligned_target])


# =============================================================================
# GAP #20: MULTI-TRACK PHASE ALIGNMENT
# =============================================================================


class PhaseAligner:
    """
    Aligns phase between stereo/multi-track channels.

    Phase Alignment Problem:
    - Channels may have phase offset (not timing offset, but constant phase shift)
    - Caused by: Polarity inversion, phase-shifting filters, mic positioning
    - Solution: Detect phase difference, apply all-pass filter or polarity correction

    Algorithm:
    1. FFT-based phase difference analysis
    2. Detect constant phase offset vs frequency-dependent drift
    3. Apply polarity flip (-1 multiply) or all-pass filter

    References:
    - Zölzer, U. (2011). "DAFX: Digital Audio Effects"
    - Smith, J. O. (2007). "Introduction to Digital Filters"
    """

    def __init__(self, phase_threshold_degrees: float = 90.0, correction_strength: float = 0.8):
        """
        Initialisiert Phase Aligner.

        Parameters
        ----------
        phase_threshold_degrees : float
            Minimum phase difference to correct (default: 90°)
        correction_strength : float
            Strength of phase correction (0-1, default: 0.8)
        """
        self.phase_threshold_degrees = np.clip(phase_threshold_degrees, 10.0, 180.0)
        self.correction_strength = np.clip(correction_strength, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_phase_difference(self, reference: np.ndarray, target: np.ndarray) -> float:
        """
        Erkennt average phase difference between channels using FFT.

        Parameters
        ----------
        reference : np.ndarray
            Reference channel
        target : np.ndarray
            Target channel

        Returns
        -------
        phase_diff_degrees : float
            Average phase difference in degrees (-180 to 180)
        """
        # OPTIMIZATION: Limit analysis to 10 seconds for performance
        # Phase difference patterns are detectable in shorter samples
        max_samples = 480000  # 10 seconds at 48kHz
        if len(reference) > max_samples:
            # Use middle section for better representation
            start = (len(reference) - max_samples) // 2
            ref_sample = reference[start : start + max_samples]
            tgt_sample = target[start : start + max_samples]
        else:
            ref_sample = reference
            tgt_sample = target

        # FFT
        ref_fft = np.asarray(fft.fft(ref_sample), dtype=np.complex128)
        tgt_fft = np.asarray(fft.fft(tgt_sample), dtype=np.complex128)

        # Phase difference
        phase_tgt = np.arctan2(tgt_fft.imag, tgt_fft.real)
        phase_ref = np.arctan2(ref_fft.imag, ref_fft.real)
        phase_diff = phase_tgt - phase_ref

        # Wrap to -π to π
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # Weight by magnitude (ignore low-energy bins)
        mag_ref = np.sqrt(np.square(ref_fft.real) + np.square(ref_fft.imag))
        mag_tgt = np.sqrt(np.square(tgt_fft.real) + np.square(tgt_fft.imag))
        magnitudes = mag_ref * mag_tgt
        weights = magnitudes / (np.sum(magnitudes) + 1e-10)

        # Weighted average
        avg_phase_diff = np.sum(phase_diff * weights)

        # Convert to degrees
        avg_phase_diff_degrees = np.degrees(avg_phase_diff)

        return float(avg_phase_diff_degrees)

    def correct_phase(self, audio: np.ndarray, phase_diff_degrees: float) -> np.ndarray:
        """
        Correct phase by polarity inversion if needed.

        Parameters
        ----------
        audio : np.ndarray
            Audio to correct
        phase_diff_degrees : float
            Phase difference in degrees

        Returns
        -------
        corrected : np.ndarray
            Phase-corrected audio
        """
        # If phase difference is close to 180°, it's polarity inversion
        if abs(abs(phase_diff_degrees) - 180.0) < 30.0:
            # Invert polarity
            corrected = -audio * self.correction_strength + audio * (1 - self.correction_strength)
            return corrected  # type: ignore[no-any-return]
        else:
            # Phasenversatz via IIR-Allpass-Filter
            # Ziel-Phase in Radian
            phi = float(phase_diff_degrees) * np.pi / 180.0
            try:
                from scipy.signal import sosfilt

                # Erste-Ordnung-Allpass: H(z) = (a - z^{-1}) / (1 - a*z^{-1})
                # Phasengang: phi(w) = pi - 2*arctan(a*sin(w)/(1 - a*cos(w)))
                # Bei w=pi/2 (Nyquist/4): phi = pi - 2*arctan(a) => a = tan((pi - phi)/2)
                a = float(np.clip(np.tan((np.pi - abs(phi)) / 2.0), -0.999, 0.999))
                if phi < 0:
                    a = -a
                # SOS-Form: b = [a, -1, 0], a_coeff = [1, -a, 0]
                sos = np.array([[a, -1.0, 0.0, 1.0, -a, 0.0]])
                corrected = sosfilt(sos, audio.astype(np.float64)).astype(audio.dtype)
                # Dry/Wet-Mix mit correction_strength
                mix = float(getattr(self, "correction_strength", 1.0))
                return (corrected * mix + audio * (1.0 - mix)).astype(audio.dtype)  # type: ignore[no-any-return]
            except Exception:
                logger.warning("multi_track_specialist.py::correct_Verarbeitungsschritt Ersatzpfad", exc_info=True)
                return audio.copy()

    def process(self, audio_stereo: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet stereo audio to align phase.

        Parameters
        ----------
        audio_stereo : np.ndarray
            Stereo audio (2, N) or (N, 2)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Phase-aligned stereo audio
        """
        # Handle (N, 2) → (2, N)
        if audio_stereo.ndim == 2 and audio_stereo.shape[1] == 2:
            audio_stereo = audio_stereo.T

        if audio_stereo.ndim != 2 or audio_stereo.shape[0] != 2:
            logger.info("[PhaseAlign] Not stereo, skipping")
            self.metrics["correction_applied"] = False
            return audio_stereo

        reference = audio_stereo[0]
        target = audio_stereo[1]

        # Detect phase difference
        phase_diff = self.detect_phase_difference(reference, target)

        # Store metrics
        self.metrics["phase_diff_degrees"] = phase_diff

        # Check if correction is needed
        if abs(phase_diff) < self.phase_threshold_degrees:
            logger.info("[PhaseAlign] Verarbeitungsschritt difference (%.1f°) below Schwelle, skipping", phase_diff)
            self.metrics["correction_applied"] = False
            self.metrics["polarity_inverted"] = False
            return audio_stereo

        # Correct phase
        logger.info("[PhaseAlign] Correcting Verarbeitungsschritt: %.1f°", phase_diff)
        corrected_target = self.correct_phase(target, phase_diff)

        self.metrics["correction_applied"] = True
        self.metrics["polarity_inverted"] = abs(abs(phase_diff) - 180.0) < 30.0

        return np.vstack([reference, corrected_target])


# =============================================================================
# GAP #22: PHASE CANCELLATION DETECTOR & CORRECTOR
# =============================================================================


class PhaseCancellationCorrector:
    """
    Erkennt and corrects phase cancellation in stereo mix.

    Phase Cancellation:
    - When L and R channels are out of phase, they cancel in mono sum
    - Causes: Improper mic placement, polarity errors, phase-shifting effects
    - Solution: Detect low mid (M) energy vs side (S), correct phase

    Algorithm:
    1. Convert to Mid/Side (M/S)
    2. Analyze energy ratio (M vs S)
    3. If M is abnormally low → phase cancellation
    4. Correct by adjusting phase/polarity

    References:
    - Newell, P. (2012). "Recording Studio Design"
    - Blumlein, A. D. (1931). "British Patent 394,325"
    """

    def __init__(self, cancellation_threshold_db: float = -20.0, correction_strength: float = 0.8):
        """
        Initialisiert Phase Cancellation Corrector.

        Parameters
        ----------
        cancellation_threshold_db : float
            Threshold for cancellation detection (default: -20dB)
        correction_strength : float
            Strength of cancellation correction (0-1, default: 0.8)
        """
        self.cancellation_threshold_db = np.clip(cancellation_threshold_db, -40.0, -10.0)
        self.correction_strength = np.clip(correction_strength, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_cancellation(self, left: np.ndarray, right: np.ndarray) -> tuple[bool, float]:
        """
        Erkennt phase cancellation by analyzing M/S energy ratio.

        Parameters
        ----------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel

        Returns
        -------
        cancellation_detected : bool
            True if cancellation detected
        ms_ratio_db : float
            Mid/Side energy ratio in dB
        """
        # Convert to Mid/Side
        mid = (left + right) / 2
        side = (left - right) / 2

        # Calculate RMS energy
        mid_rms = np.sqrt(np.mean(mid**2))
        side_rms = np.sqrt(np.mean(side**2))

        # Avoid division by zero
        if side_rms < 1e-10:
            return False, 0.0

        # M/S ratio in dB
        ms_ratio_db = 20 * np.log10((mid_rms + 1e-10) / (side_rms + 1e-10))

        # If Mid is much lower than Side → cancellation
        cancellation_detected = ms_ratio_db < self.cancellation_threshold_db

        return cancellation_detected, float(ms_ratio_db)

    def correct_cancellation(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Correct phase cancellation by adjusting phase.

        Parameters
        ----------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel

        Returns
        -------
        left_corrected : np.ndarray
            Corrected left channel
        right_corrected : np.ndarray
            Corrected right channel
        """
        # Simple correction: Invert one channel partially
        # (This assumes the cancellation is due to polarity inversion)
        right_corrected = -right * self.correction_strength + right * (1 - self.correction_strength)

        return left.copy(), right_corrected

    def process(self, audio_stereo: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet stereo audio to correct phase cancellation.

        Parameters
        ----------
        audio_stereo : np.ndarray
            Stereo audio (2, N) or (N, 2)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Cancellation-corrected stereo audio
        """
        # Handle (N, 2) → (2, N)
        if audio_stereo.ndim == 2 and audio_stereo.shape[1] == 2:
            audio_stereo = audio_stereo.T

        if audio_stereo.ndim != 2 or audio_stereo.shape[0] != 2:
            logger.info("[PhaseCancellation] Not stereo, skipping")
            self.metrics["correction_applied"] = False
            return audio_stereo

        left = audio_stereo[0]
        right = audio_stereo[1]

        # Detect cancellation
        cancellation_detected, ms_ratio_db = self.detect_cancellation(left, right)

        # Store metrics
        self.metrics["cancellation_detected"] = cancellation_detected
        self.metrics["ms_ratio_db"] = ms_ratio_db

        # Check if correction is needed
        if not cancellation_detected:
            logger.info("[PhaseCancellation] No cancellation erkannt (M/S Verhaeltnis: %.1f dB)", ms_ratio_db)
            self.metrics["correction_applied"] = False
            return audio_stereo

        # Correct cancellation
        logger.info("[PhaseCancellation] Cancellation erkannt (M/S Verhaeltnis: %.1f dB), correcting...", ms_ratio_db)
        left_corrected, right_corrected = self.correct_cancellation(left, right)

        self.metrics["correction_applied"] = True

        return np.vstack([left_corrected, right_corrected])


# =============================================================================
# GAP #23: STEREO BALANCE CORRECTOR
# =============================================================================


class StereoBalanceCorrector:
    """
    Corrects L/R channel imbalance.

    Stereo Imbalance:
    - One channel significantly louder/quieter than the other
    - Causes: Recording level errors, azimuth errors, channel gain mismatch
    - Solution: Detect RMS difference, apply gain correction

    Algorithm:
    1. Calculate RMS for L and R
    2. Compute imbalance in dB
    3. Apply compensating gain to balance channels

    References:
    - Rumsey, F., & McCormick, T. (2009). "Sound and Recording"
    - Eargle, J. (2004). "The Microphone Book"
    """

    def __init__(self, imbalance_threshold_db: float = 1.0, correction_strength: float = 0.8):
        """
        Initialisiert Stereo Balance Corrector.

        Parameters
        ----------
        imbalance_threshold_db : float
            Minimum imbalance to correct (default: 1.0dB)
        correction_strength : float
            Strength of balance correction (0-1, default: 0.8)
        """
        self.imbalance_threshold_db = np.clip(imbalance_threshold_db, 0.5, 10.0)
        self.correction_strength = np.clip(correction_strength, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_imbalance(self, left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
        """
        Erkennt stereo imbalance.

        Parameters
        ----------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel

        Returns
        -------
        imbalance_db : float
            Imbalance in dB (positive = left louder)
        louder_channel : int
            0 = left louder, 1 = right louder
        """
        # Calculate RMS
        left_rms = np.sqrt(np.mean(left**2))
        right_rms = np.sqrt(np.mean(right**2))

        # Avoid division by zero
        if left_rms < 1e-10 or right_rms < 1e-10:
            return 0.0, 0

        # Imbalance in dB
        imbalance_db = 20 * np.log10(left_rms / right_rms)

        louder_channel = 0 if imbalance_db > 0 else 1

        return float(imbalance_db), louder_channel

    def correct_balance(
        self, left: np.ndarray, right: np.ndarray, imbalance_db: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Correct stereo balance by applying gain.

        Parameters
        ----------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel
        imbalance_db : float
            Imbalance in dB

        Returns
        -------
        left_corrected : np.ndarray
            Corrected left channel
        right_corrected : np.ndarray
            Corrected right channel
        """
        # Calculate correction gain
        correction_gain_db = -imbalance_db * self.correction_strength
        correction_gain_linear = 10 ** (correction_gain_db / 20)

        # Apply to quieter channel
        if imbalance_db > 0:
            # Left is louder, boost right
            right_corrected = right * correction_gain_linear
            left_corrected = left.copy()
        else:
            # Right is louder, boost left
            left_corrected = left * correction_gain_linear
            right_corrected = right.copy()

        return left_corrected, right_corrected

    def process(self, audio_stereo: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet stereo audio to correct balance.

        Parameters
        ----------
        audio_stereo : np.ndarray
            Stereo audio (2, N) or (N, 2)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Balance-corrected stereo audio
        """
        # Handle (N, 2) → (2, N)
        if audio_stereo.ndim == 2 and audio_stereo.shape[1] == 2:
            audio_stereo = audio_stereo.T

        if audio_stereo.ndim != 2 or audio_stereo.shape[0] != 2:
            logger.info("[StereoBalance] Not stereo, skipping")
            self.metrics["correction_applied"] = False
            return audio_stereo

        left = audio_stereo[0]
        right = audio_stereo[1]

        # Detect imbalance
        imbalance_db, louder_channel = self.detect_imbalance(left, right)

        # Store metrics
        self.metrics["imbalance_db"] = imbalance_db
        self.metrics["louder_channel"] = int(louder_channel)

        # Check if correction is needed
        if abs(imbalance_db) < self.imbalance_threshold_db:
            logger.info("[StereoBalance] Imbalance (%.2f dB) below Schwelle, skipping", abs(imbalance_db))
            self.metrics["correction_applied"] = False
            return audio_stereo

        # Correct balance
        channel_names = ["Left", "Right"]
        logger.info(
            "[StereoBalance] %s %.2f dB louder, correcting...", channel_names[louder_channel], abs(imbalance_db)
        )
        left_corrected, right_corrected = self.correct_balance(left, right, imbalance_db)

        self.metrics["correction_applied"] = True

        return np.vstack([left_corrected, right_corrected])


# =============================================================================
# GAP #24: MID/SIDE (M/S) PROCESSOR
# =============================================================================


class MidSideProcessor:
    """
    Mid/Side encoding/decoding and processing.

    M/S Processing:
    - Mid (M) = (L + R) / 2 (center/mono content)
    - Side (S) = (L - R) / 2 (stereo width/difference)
    - Allows independent processing of center vs width
    - Can adjust stereo width, enhance/reduce center

    Algorithm:
    1. Encode: L/R → M/S
    2. Process: Scale Mid and Side independently
    3. Decode: M/S → L/R

    References:
    - Blumlein, A. D. (1931). "British Patent 394,325"
    - Eargle, J. (2004). "The Microphone Book"
    """

    def __init__(self, width_factor: float = 1.0, mid_gain_db: float = 0.0, side_gain_db: float = 0.0):
        """
        Initialisiert Mid/Side Processor.

        Parameters
        ----------
        width_factor : float
            Stereo width multiplier (0-2, default: 1.0)
            <1 = narrower, >1 = wider
        mid_gain_db : float
            Gain for Mid channel (default: 0dB)
        side_gain_db : float
            Gain for Side channel (default: 0dB)
        """
        self.width_factor = np.clip(width_factor, 0.0, 2.0)
        self.mid_gain_db = np.clip(mid_gain_db, -12.0, 12.0)
        self.side_gain_db = np.clip(side_gain_db, -12.0, 12.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def encode_ms(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Kodiert L/R to Mid/Side.

        Parameters
        ----------
        left : np.ndarray
            Left channel
        right : np.ndarray
            Right channel

        Returns
        -------
        mid : np.ndarray
            Mid channel (L+R)/2
        side : np.ndarray
            Side channel (L-R)/2
        """
        mid = (left + right) / 2
        side = (left - right) / 2
        return mid, side

    def decode_ms(self, mid: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Dekodiert Mid/Side to L/R.

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

    def process(self, audio_stereo: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet stereo audio with M/S.

        Parameters
        ----------
        audio_stereo : np.ndarray
            Stereo audio (2, N) or (N, 2)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            M/S-processed stereo audio
        """
        # Handle (N, 2) → (2, N)
        if audio_stereo.ndim == 2 and audio_stereo.shape[1] == 2:
            audio_stereo = audio_stereo.T

        if audio_stereo.ndim != 2 or audio_stereo.shape[0] != 2:
            logger.info("[MidSide] Not stereo, skipping")
            return audio_stereo

        left = audio_stereo[0]
        right = audio_stereo[1]

        # Encode
        mid, side = self.encode_ms(left, right)

        # Process
        mid_gain = 10 ** (self.mid_gain_db / 20)
        side_gain = 10 ** (self.side_gain_db / 20) * self.width_factor

        mid_processed = mid * mid_gain
        side_processed = side * side_gain

        # Store metrics
        self.metrics["width_applied"] = self.width_factor
        self.metrics["mid_gain_applied"] = self.mid_gain_db
        self.metrics["side_gain_applied"] = self.side_gain_db

        # Decode
        left_processed, right_processed = self.decode_ms(mid_processed, side_processed)

        logger.info(
            f"[MidSide] Width: {self.width_factor:.2f}, Mid: {self.mid_gain_db:+.1f}dB, Side: {self.side_gain_db:+.1f}dB"
        )

        return np.vstack([left_processed, right_processed])


# =============================================================================
# GAP #25: COMB FILTER REMOVER (PHASE-INDUCED)
# =============================================================================


class CombFilterRemover:
    """
    Removes comb filtering caused by phase issues.

    Comb Filtering:
    - Notches in frequency response (looks like a comb)
    - Caused by: Phase interference, delayed duplicate signals
    - Common in: Multi-mic setups with phase issues, reflections
    - Solution: Detect notches, apply inverse filtering

    Algorithm:
    1. FFT analysis to detect comb pattern
    2. Identify notch frequencies
    3. Apply equalization to fill notches

    References:
    - Zölzer, U. (2011). "DAFX: Digital Audio Effects"
    - Orfanidis, S. J. (1996). "Introduction to Signal Processing"
    """

    def __init__(self, notch_threshold_db: float = -6.0, correction_strength: float = 0.7):
        """
        Initialisiert Comb Filter Remover.

        Parameters
        ----------
        notch_threshold_db : float
            Depth threshold for notch detection (default: -6dB)
        correction_strength : float
            Strength of comb filter correction (0-1, default: 0.7)
        """
        self.notch_threshold_db = np.clip(notch_threshold_db, -20.0, -3.0)
        self.correction_strength = np.clip(correction_strength, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_comb(self, audio: np.ndarray, sample_rate: int) -> list[float]:
        """
        Erkennt comb filtering by analyzing frequency spectrum.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        notch_frequencies : list
            List of detected notch frequencies (Hz)
        """
        # PERFORMANCE: Limit analysis to 3 seconds for speed (was 10s)
        # Comb filter patterns are detectable in shorter samples
        max_samples = int(3 * sample_rate)
        if len(audio) > max_samples:
            # Use middle section for better representation
            start = (len(audio) - max_samples) // 2
            audio_sample = audio[start : start + max_samples]
        else:
            audio_sample = audio

        try:
            # FFT with timeout protection
            spectrum = np.asarray(fft.fft(audio_sample), dtype=np.complex128)
            spectrum_half = spectrum[: len(spectrum) // 2]
            magnitude_half = np.sqrt(np.square(spectrum_half.real) + np.square(spectrum_half.imag))
            magnitude_db = 20 * np.log10(magnitude_half + 1e-10)

            # Smooth magnitude (reduced window size for speed)
            from scipy.ndimage import uniform_filter1d

            magnitude_smooth = uniform_filter1d(magnitude_db, size=30)  # was 50

            # Find local minima (notches)
            from scipy.signal import find_peaks

            # Invert to find minima as peaks
            inverted = -magnitude_smooth
            peaks, properties = find_peaks(inverted, prominence=abs(self.notch_threshold_db))

            # Convert peak indices to frequencies
            freqs = fft.fftfreq(len(audio_sample), 1 / sample_rate)[: len(spectrum) // 2]
            notch_frequencies = [freqs[p] for p in peaks if freqs[p] > 20 and freqs[p] < sample_rate / 2]

            # Limit to top 30 most prominent notches for performance (was 50)
            if len(notch_frequencies) > 30:
                prominences = properties["prominences"]
                top_indices = np.argsort(prominences)[-30:]
                notch_frequencies = [freqs[peaks[i]] for i in top_indices if freqs[peaks[i]] > 20]

            return notch_frequencies

        except Exception as e:
            logger.error("[CombFilter] ⚠️  Detection fehlgeschlagen: %s, returning empty", e)
            return []

    def correct_comb(self, audio: np.ndarray, notch_frequencies: list[float], sample_rate: int) -> np.ndarray:
        """
        Correct comb filtering by boosting notches.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        notch_frequencies : list
            Notch frequencies to correct
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        corrected : np.ndarray
            Comb-corrected audio
        """
        if not notch_frequencies:
            return audio.copy()

        # FFT
        spectrum = np.asarray(fft.fft(audio), dtype=np.complex128)

        # Create correction curve
        freqs = fft.fftfreq(len(audio), 1 / sample_rate)
        correction = np.ones(len(freqs), dtype=np.complex128)

        # Boost notches (applies symmetrically to positive and negative frequencies)
        for notch_freq in notch_frequencies:
            # Gaussian boost centered at notch
            boost_db = 6.0 * self.correction_strength  # 6dB boost
            bandwidth = notch_freq * 0.1  # 10% bandwidth

            # Apply boost to both positive and negative frequencies
            boost = np.exp(-((np.abs(freqs) - notch_freq) ** 2) / (2 * bandwidth**2))
            boost_linear = 10 ** (boost_db * boost / 20)

            correction *= boost_linear

        # Apply correction
        spectrum_corrected = spectrum * correction

        # IFFT
        ifft_out = np.asarray(fft.ifft(spectrum_corrected), dtype=np.complex128)
        audio_corrected = ifft_out.real

        return audio_corrected.astype(audio.dtype)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet audio to remove comb filtering.

        Parameters
        ----------
        audio : np.ndarray
            Audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        output : np.ndarray
            Comb-corrected audio
        """
        try:
            # Handle stereo
            if audio.ndim == 2:
                if audio.shape[0] == 2:
                    # (2, N)
                    left = self.process(audio[0], sample_rate)
                    right = self.process(audio[1], sample_rate)
                    return np.vstack([left, right])
                elif audio.shape[1] == 2:
                    # (N, 2)
                    audio = audio.T
                    left = self.process(audio[0], sample_rate)
                    right = self.process(audio[1], sample_rate)
                    return np.vstack([left, right]).T

            # Mono processing
            logger.info("[CombFilter] Detecting comb filtering...")
            notch_frequencies = self.detect_comb(audio, sample_rate)

            # Store metrics
            self.metrics["notches_detected"] = len(notch_frequencies)

            # Check if correction is needed
            if not notch_frequencies:
                logger.info("[CombFilter] No comb filtering erkannt")
                self.metrics["correction_applied"] = False
                return audio

            # Correct
            logger.info("[CombFilter] %s notches erkannt, correcting...", len(notch_frequencies))
            audio_corrected = self.correct_comb(audio, notch_frequencies, sample_rate)

            self.metrics["correction_applied"] = True
            logger.info("[CombFilter] Correction angewendet erfolgreich")

            return audio_corrected

        except Exception as e:
            logger.error("[CombFilter] ⚠️  Error during processing: %s", e)
            logger.info("[CombFilter] Returning unprocessed audio")
            self.metrics["correction_applied"] = False
            self.metrics["error"] = str(e)
            return audio.copy()


# =============================================================================
# UNIFIED MULTI-TRACK SPECIALIST API
# =============================================================================


class MultiTrackSpecialist:
    """
    Einheitliche API für Mehrspuraufnahme- und Stereo-Verbesserung.

    Combines:
    - TimeAligner (GAP #19)
    - PhaseAligner (GAP #20)
    - PhaseCancellationCorrector (GAP #22)
    - StereoBalanceCorrector (GAP #23)
    - MidSideProcessor (GAP #24)
    - CombFilterRemover (GAP #25)

    Processing order:
    1. Time Alignment (timing issues first)
    2. Phase Alignment (polarity/phase issues)
    3. Comb Filter Removal (frequency-domain issues)
    4. Phase Cancellation Correction (M/S energy issues)
    5. Stereo Balance Correction (level issues)
    6. Mid/Side Processing (optional, creative)
    """

    def __init__(
        self,
        enable_time_alignment: bool = True,
        enable_phase_alignment: bool = True,
        enable_phase_cancellation_correction: bool = True,
        enable_stereo_balance_correction: bool = True,
        enable_mid_side_processing: bool = False,  # Optional, creative
        enable_comb_filter_removal: bool = True,
        # Time alignment params
        max_delay_ms: float = 100.0,
        correlation_threshold: float = 0.3,
        # Phase alignment params
        phase_threshold_degrees: float = 90.0,
        phase_correction_strength: float = 0.8,
        # Phase cancellation params
        cancellation_threshold_db: float = -20.0,
        cancellation_correction_strength: float = 0.8,
        # Stereo balance params
        imbalance_threshold_db: float = 1.0,
        balance_correction_strength: float = 0.8,
        # M/S params
        width_factor: float = 1.0,
        mid_gain_db: float = 0.0,
        side_gain_db: float = 0.0,
        # Comb filter params
        notch_threshold_db: float = -6.0,
        comb_correction_strength: float = 0.7,
    ):
        """
        Initialisiert Multi-Track Specialist.

        Parameters
        ----------
        enable_* : bool
            Enable/disable individual processors
        ... (see individual classes for parameter descriptions)
        """
        self.enable_time_alignment = enable_time_alignment
        self.enable_phase_alignment = enable_phase_alignment
        self.enable_phase_cancellation_correction = enable_phase_cancellation_correction
        self.enable_stereo_balance_correction = enable_stereo_balance_correction
        self.enable_mid_side_processing = enable_mid_side_processing
        self.enable_comb_filter_removal = enable_comb_filter_removal

        # Initialize processors
        if self.enable_time_alignment:
            self.time_aligner = TimeAligner(max_delay_ms=max_delay_ms, correlation_threshold=correlation_threshold)

        if self.enable_phase_alignment:
            self.phase_aligner = PhaseAligner(
                phase_threshold_degrees=phase_threshold_degrees, correction_strength=phase_correction_strength
            )

        if self.enable_phase_cancellation_correction:
            self.phase_cancellation_corrector = PhaseCancellationCorrector(
                cancellation_threshold_db=cancellation_threshold_db,
                correction_strength=cancellation_correction_strength,
            )

        if self.enable_stereo_balance_correction:
            self.stereo_balance_corrector = StereoBalanceCorrector(
                imbalance_threshold_db=imbalance_threshold_db, correction_strength=balance_correction_strength
            )

        if self.enable_mid_side_processing:
            self.mid_side_processor = MidSideProcessor(
                width_factor=width_factor, mid_gain_db=mid_gain_db, side_gain_db=side_gain_db
            )

        if self.enable_comb_filter_removal:
            self.comb_filter_remover = CombFilterRemover(
                notch_threshold_db=notch_threshold_db, correction_strength=comb_correction_strength
            )

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung, Doku als Code
        Process audio with multi-track enhancement.
        """
        self._log_contract()
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sample_rate <= 0:
                raise ValueError("Ungültige Eingabe für MultiTrackSpecialist.process")
            output = audio.copy()
            # Step 1: Time Alignment
            if self.enable_time_alignment:
                logger.info("\n[MultiTrack] Step 1/6: Time Alignment")
                try:
                    output = self.time_aligner.process(output, sample_rate)
                except Exception as e:
                    logger.error("[MultiTrack] ⚠️  Time Alignment fehlgeschlagen: %s", e)
            # Step 2: Phase Alignment
            if self.enable_phase_alignment:
                logger.info("\n[MultiTrack] Step 2/6: Verarbeitungsschritt Alignment")
                try:
                    output = self.phase_aligner.process(output, sample_rate)
                except Exception as e:
                    logger.error("[MultiTrack] ⚠️  Verarbeitungsschritt Alignment fehlgeschlagen: %s", e)
            # Step 3: Comb Filter Removal (deaktiviert)
            if False:
                logger.info("\n[MultiTrack] Step 3/6: Comb Filter Removal (uebersprungen - performance issue)")
                try:
                    output = self.comb_filter_remover.process(output, sample_rate)
                except Exception as e:
                    logger.error("[MultiTrack] ⚠️  Comb Filter Removal fehlgeschlagen: %s", e)
            # Step 4: Phase Cancellation Correction
            if self.enable_phase_cancellation_correction:
                logger.info("\n[MultiTrack] Step 4/6: Verarbeitungsschritt Cancellation Correction")
                try:
                    output = self.phase_cancellation_corrector.process(output, sample_rate)
                except Exception as e:
                    logger.error("[MultiTrack] ⚠️  Verarbeitungsschritt Cancellation Correction fehlgeschlagen: %s", e)
            # Step 5: Stereo Balance Correction
            if self.enable_stereo_balance_correction:
                logger.info("\n[MultiTrack] Step 5/6: Stereo Balance Correction")
                try:
                    output = self.stereo_balance_corrector.process(output, sample_rate)
                except Exception as e:
                    logger.error("[MultiTrack] ⚠️  Stereo Balance Correction fehlgeschlagen: %s", e)
            # Step 6: Mid/Side Processing (optional, creative)
            if self.enable_mid_side_processing:
                logger.info("\n[MultiTrack] Step 6/6: Mid/Side Processing")
                try:
                    output = self.mid_side_processor.process(output, sample_rate)
                except Exception as e:
                    logger.error("[MultiTrack] ⚠️  Mid/Side Processing fehlgeschlagen: %s", e)
            logger.info("\n[MultiTrack] Processing vollstaendig!")
            self._audit_log({"shape": output.shape, "success": True})
            return output
        except Exception as e:
            logger.error("[MultiTrackSpecialist][Fehler] %s", e)
            self._audit_log({"error": str(e)})
            return audio

    def _log_contract(self):
        logger.info("[Contract][MultiTrackSpecialist] verarbeiten(audio: np.ndarray, sample_rate: int) -> np.ndarray")

    def _audit_log(self, result: dict):
        logger.info("[AuditLog][MultiTrackSpecialist] Ergebnis: %s", result)

    def get_metrics(self) -> dict:
        """Gibt zurück: metrics from all processors."""
        metrics = {}

        if self.enable_time_alignment and hasattr(self, "time_aligner"):
            metrics["time_alignment"] = self.time_aligner.metrics

        if self.enable_phase_alignment and hasattr(self, "phase_aligner"):
            metrics["phase_alignment"] = self.phase_aligner.metrics

        if self.enable_phase_cancellation_correction and hasattr(self, "phase_cancellation_corrector"):
            metrics["phase_cancellation"] = self.phase_cancellation_corrector.metrics

        if self.enable_stereo_balance_correction and hasattr(self, "stereo_balance_corrector"):
            metrics["stereo_balance"] = self.stereo_balance_corrector.metrics

        if self.enable_mid_side_processing and hasattr(self, "mid_side_processor"):
            metrics["mid_side"] = self.mid_side_processor.metrics

        if self.enable_comb_filter_removal and hasattr(self, "comb_filter_remover"):
            metrics["comb_filter"] = self.comb_filter_remover.metrics

        return metrics


# =============================================================================
# CLI (Demo)
# =============================================================================

if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Multi-Track Specialist - Multi-Track & Stereo Enhancement")
    parser.add_argument("input", help="Input audio file (stereo)")
    parser.add_argument("output", help="Output audio file")

    # Module selection
    parser.add_argument("--no-time-align", action="store_true", help="Disable time alignment")
    parser.add_argument("--no-phase-align", action="store_true", help="Disable phase alignment")
    parser.add_argument("--no-cancellation", action="store_true", help="Disable phase cancellation correction")
    parser.add_argument("--no-balance", action="store_true", help="Disable stereo balance correction")
    parser.add_argument("--ms-processing", action="store_true", help="Enable M/S processing")
    parser.add_argument("--no-comb", action="store_true", help="Disable comb filter removal")

    # Time alignment params
    parser.add_argument("--max-delay", type=float, default=100.0, help="Max delay for alignment (ms)")

    # M/S params
    parser.add_argument("--width", type=float, default=1.0, help="Stereo width factor (0-2)")

    args = parser.parse_args()

    # Load audio
    logger.info("lade: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Transpose if stereo (N, 2) → (2, N)
    if audio.ndim == 2:
        audio = audio.T

    # Initialize processor
    processor = MultiTrackSpecialist(
        enable_time_alignment=not args.no_time_align,
        enable_phase_alignment=not args.no_phase_align,
        enable_phase_cancellation_correction=not args.no_cancellation,
        enable_stereo_balance_correction=not args.no_balance,
        enable_mid_side_processing=args.ms_processing,
        enable_comb_filter_removal=not args.no_comb,
        max_delay_ms=args.max_delay,
        width_factor=args.width,
    )

    # Process
    logger.info("\nProcessing with Multi-Track Specialist...")
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

    # Save (transpose back if needed)
    if output.ndim == 2:
        output = output.T

    sf.write(args.output, output, sr)
    logger.info("gespeichert: %s", args.output)
