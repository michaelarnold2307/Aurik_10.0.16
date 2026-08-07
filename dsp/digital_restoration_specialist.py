"""
digital_restoration_specialist.py - Digital-Specific Defect Removal (GAP #3-5)

Specialized treatment für digitale Defekte:
- GAP #3: Codec Artifact Removal (MP3/AAC Pre-Echo, Spectral Holes)
- GAP #4: Packet Loss Concealment (Streaming Gaps)
- GAP #5: Jitter Correction (Clock Errors)

Author: AURIK Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import logging
import warnings
from typing import Any

import numpy as np
from scipy import interpolate, signal
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# GAP #3: CODEC ARTIFACT REMOVAL
# =============================================================================


class CodecArtifactRemover:
    """
    Entfernt codec-specific artifacts from compressed audio.

    Codec Artifacts:
    - MP3: Pre-echo (MDCT time-smearing before transients)
    - AAC: Spectral holes (quantization noise in quiet bands)
    - General: Time/frequency domain smearing, ringing

    Algorithm:
    1. Detect compression format (MP3 vs AAC characteristics)
    2. MP3: Transient detection + pre-echo removal
    3. AAC: Spectral analysis + hole filling
    4. Apply gentle smoothing to reduce artifacts

    References:
    - Brandenburg, K. (1999). "MP3 and AAC Explained"
    - Bosi, M., & Goldberg, R. (2003). "Introduction to Digital Audio Coding"
    """

    def __init__(
        self,
        pre_echo_threshold_db: float = -40.0,
        spectral_hole_threshold_db: float = -50.0,
        smoothing_strength: float = 0.6,
    ):
        """
        Initialisiert Codec Artifact Remover.

        Parameters
        ----------
        pre_echo_threshold_db : float
            Threshold for pre-echo detection (default: -40dB)
        spectral_hole_threshold_db : float
            Threshold for spectral hole detection (default: -50dB)
        smoothing_strength : float
            Strength of artifact smoothing (0-1, default: 0.6)
        """
        self.pre_echo_threshold_db = np.clip(pre_echo_threshold_db, -60.0, -20.0)
        self.spectral_hole_threshold_db = np.clip(spectral_hole_threshold_db, -80.0, -30.0)
        self.smoothing_strength = np.clip(smoothing_strength, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_pre_echo(self, audio: np.ndarray, sample_rate: int) -> list[int]:
        """
        Erkennt MP3 pre-echo artifacts (signal before transients).

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        transient_indices : list
            Sample indices where transients occur
        """
        # Compute envelope
        window_ms = 10
        window_samples = int(window_ms * sample_rate / 1000)

        # Simple RMS envelope
        envelope = np.sqrt(signal.convolve(audio**2, np.ones(window_samples) / window_samples, mode="same"))

        # Compute envelope derivative (rate of change)
        envelope_deriv = np.diff(envelope, prepend=envelope[0])

        # Find sudden increases (transients)
        threshold_linear = 10 ** (self.pre_echo_threshold_db / 20)
        transient_mask = envelope_deriv > (threshold_linear * np.max(envelope_deriv))

        # Find transient indices
        transient_indices = np.where(transient_mask)[0].tolist()

        return transient_indices  # type: ignore[no-any-return]

    def remove_pre_echo(self, audio: np.ndarray, transient_indices: list[int], sample_rate: int) -> np.ndarray:
        """
        Entfernt pre-echo artifacts before transients.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        transient_indices : list
            Sample indices of transients
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        audio_cleaned : np.ndarray
            Audio with pre-echo removed
        """
        audio_cleaned = audio.copy()

        # Pre-echo typically occurs 5-20ms before transient
        pre_echo_ms = 10
        pre_echo_samples = int(pre_echo_ms * sample_rate / 1000)

        for transient_idx in transient_indices:
            if transient_idx < pre_echo_samples:
                continue

            # Region before transient
            start_idx = max(0, transient_idx - pre_echo_samples)
            end_idx = transient_idx

            # Apply gentle fade-in from silence
            fade_length = end_idx - start_idx
            if fade_length > 0:
                fade_curve = np.linspace(0, 1, fade_length) ** 2  # Quadratic fade
                audio_cleaned[start_idx:end_idx] *= fade_curve * self.smoothing_strength

        return audio_cleaned

    def detect_spectral_holes(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Erkennt AAC spectral holes (quantization artifacts).

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        hole_mask : np.ndarray
            Boolean mask of spectral bins with holes
        """
        # Compute STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = signal.stft(audio, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Magnitude spectrum
        mag = np.abs(Zxx)

        # Convert to dB
        mag_db = 20 * np.log10(mag + 1e-8)

        # Detect holes: bins significantly quieter than neighbors
        threshold = self.spectral_hole_threshold_db

        # Median across time for each frequency
        mag_db_median = np.median(mag_db, axis=1, keepdims=True)

        # Holes are bins consistently below threshold
        hole_mask = mag_db_median < threshold

        return hole_mask.flatten()  # type: ignore[no-any-return]

    def fill_spectral_holes(self, audio: np.ndarray, hole_mask: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Fill spectral holes via interpolation.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        hole_mask : np.ndarray
            Boolean mask of hole frequencies
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        audio_filled : np.ndarray
            Audio with spectral holes filled
        """
        if not np.any(hole_mask):
            return audio

        # Compute STFT
        nperseg = 2048
        noverlap = nperseg // 2
        _f, _t, Zxx = signal.stft(audio, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Fill holes
        for i in range(Zxx.shape[1]):  # Time frames
            for j in range(Zxx.shape[0]):  # Frequency bins
                if hole_mask[j]:
                    # Interpolate from neighbors
                    neighbors = []
                    if j > 0:
                        neighbors.append(Zxx[j - 1, i])
                    if j < len(hole_mask) - 1:
                        neighbors.append(Zxx[j + 1, i])

                    if neighbors:
                        Zxx[j, i] = np.mean(neighbors) * self.smoothing_strength

        # Inverse STFT
        _, audio_filled = signal.istft(Zxx, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Match length
        if len(audio_filled) < len(audio):
            audio_filled = np.pad(audio_filled, (0, len(audio) - len(audio_filled)))
        elif len(audio_filled) > len(audio):
            audio_filled = audio_filled[: len(audio)]

        return audio_filled  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet audio to remove codec artifacts.

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
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo
        if audio.ndim == 2:
            left = self.process(audio[0], sample_rate)
            right = self.process(audio[1], sample_rate)

            self.metrics["stereo"] = True

            return np.vstack([left, right])

        # Mono processing
        logger.info("[CodecArtifact] Detecting pre-echo (MP3)...")
        transient_indices = self.detect_pre_echo(audio, sample_rate)

        # Remove pre-echo if detected
        if transient_indices:
            logger.info("[CodecArtifact] %s transients found, removing pre-echo", len(transient_indices))
            audio_cleaned = self.remove_pre_echo(audio, transient_indices, sample_rate)
        else:
            audio_cleaned = audio.copy()

        # Detect spectral holes
        logger.info("[CodecArtifact] Detecting spectral holes (AAC)...")
        hole_mask = self.detect_spectral_holes(audio_cleaned, sample_rate)

        n_holes = np.sum(hole_mask)
        if n_holes > 0:
            logger.info("[CodecArtifact] %s spectral holes found, filling...", n_holes)
            audio_cleaned = self.fill_spectral_holes(audio_cleaned, hole_mask, sample_rate)
        else:
            logger.info("[CodecArtifact] No significant spectral holes erkannt")

        # Store metrics
        self.metrics["pre_echo_detected"] = len(transient_indices) > 0
        self.metrics["num_transients"] = len(transient_indices)
        self.metrics["spectral_holes_found"] = n_holes
        self.metrics["smoothing_applied"] = self.smoothing_strength

        # Final NaN/Inf-Guard and clipping
        audio_cleaned = np.nan_to_num(audio_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        audio_cleaned = np.clip(audio_cleaned, -1.0, 1.0)

        return audio_cleaned


# =============================================================================
# GAP #4: PACKET LOSS CONCEALMENT
# =============================================================================


class PacketLossConcealer:
    """
    Conceals packet loss artifacts from streaming audio.

    Packet Loss: Missing audio data due to network transmission errors.
    Symptoms:
    - Gaps (complete silence or zero samples)
    - Discontinuities (sudden jumps)
    - Clicks/pops at loss boundaries

    Algorithm:
    1. Detect gaps (zero regions, discontinuities)
    2. Interpolate via surrounding context
    3. Apply crossfading for smooth transitions

    References:
    - Vaseghi, S. (2008). "Advanced Digital Signal Processing"
    - Godsill, S., et al. (2002). "Audio Packet Loss Concealment"
    """

    def __init__(self, gap_threshold_ms: float = 5.0, interpolation_method: str = "cubic"):
        """
        Initialisiert Packet Loss Concealer.

        Parameters
        ----------
        gap_threshold_ms : float
            Minimum gap length to conceal (default: 5ms)
        interpolation_method : str
            Interpolation method ('linear', 'cubic', default: 'cubic')
        """
        self.gap_threshold_ms = np.clip(gap_threshold_ms, 1.0, 100.0)
        self.interpolation_method = interpolation_method if interpolation_method in ["linear", "cubic"] else "cubic"

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_gaps(self, audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """
        Erkennt packet loss gaps (zero regions, discontinuities).

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        gaps : list of (start_idx, end_idx) tuples
            Detected gap regions
        """
        # Threshold for zero detection
        zero_threshold = 1e-6

        # Find zero samples
        is_zero = np.abs(audio) < zero_threshold

        # Find contiguous zero regions
        gaps = []
        in_gap = False
        gap_start = 0

        min_gap_samples = int(self.gap_threshold_ms * sample_rate / 1000)

        for i, zero_val in enumerate(is_zero):
            if zero_val and not in_gap:
                # Start of gap
                gap_start = i
                in_gap = True
            elif not zero_val and in_gap:
                # End of gap
                gap_length = i - gap_start
                if gap_length >= min_gap_samples:
                    gaps.append((gap_start, i))
                in_gap = False

        # Check if gap extends to end
        if in_gap:
            gap_length = len(audio) - gap_start
            if gap_length >= min_gap_samples:
                gaps.append((gap_start, len(audio)))

        return gaps

    def conceal_gap(self, audio: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
        """
        Conceal a single gap via interpolation.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        gap_start, gap_end : int
            Gap boundaries

        Returns
        -------
        audio_concealed : np.ndarray
            Audio with gap concealed
        """
        audio_concealed = audio.copy()

        # Context window: 20 samples before/after
        context_samples = 20

        # Get context before gap
        before_start = max(0, gap_start - context_samples)
        before_vals = audio[before_start:gap_start]

        # Get context after gap
        after_end = min(len(audio), gap_end + context_samples)
        after_vals = audio[gap_end:after_end]

        if len(before_vals) == 0 or len(after_vals) == 0:
            # Can't interpolate (gap at boundary)
            return audio_concealed

        # Interpolate
        # Use last/first values + interpolation method
        x_known = np.array([before_start, gap_start - 1, gap_end, after_end - 1])
        y_known = np.array([before_vals[0], before_vals[-1], after_vals[0], after_vals[-1]])

        x_interpolate = np.arange(gap_start, gap_end)

        if self.interpolation_method == "cubic" and len(x_known) >= 4:
            interp_func = interpolate.interp1d(x_known, y_known, kind="cubic", fill_value="extrapolate")
        else:
            interp_func = interpolate.interp1d(x_known, y_known, kind="linear", fill_value="extrapolate")

        audio_concealed[gap_start:gap_end] = interp_func(x_interpolate)

        return audio_concealed

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet audio to conceal packet loss.

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
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo
        if audio.ndim == 2:
            left = self.process(audio[0], sample_rate)
            right = self.process(audio[1], sample_rate)

            self.metrics["stereo"] = True

            return np.vstack([left, right])

        # Mono processing
        logger.info("[PacketLoss] Detecting gaps...")
        gaps = self.detect_gaps(audio, sample_rate)

        if not gaps:
            logger.info("[PacketLoss] No packet loss erkannt")
            self.metrics["gaps_detected"] = 0
            self.metrics["gaps_concealed"] = 0
            self.metrics["total_gap_duration_ms"] = 0.0
            return audio

        logger.info("[PacketLoss] %s gaps found, concealing...", len(gaps))

        # Conceal each gap
        audio_concealed = audio.copy()
        for gap_start, gap_end in gaps:
            audio_concealed = self.conceal_gap(audio_concealed, gap_start, gap_end)

        # Calculate total gap duration
        total_gap_samples = sum([end - start for start, end in gaps])
        total_gap_ms = (total_gap_samples / sample_rate) * 1000

        # Store metrics
        self.metrics["gaps_detected"] = len(gaps)
        self.metrics["gaps_concealed"] = len(gaps)
        self.metrics["total_gap_duration_ms"] = total_gap_ms

        # Final NaN/Inf-Guard and clipping
        audio_concealed = np.nan_to_num(audio_concealed, nan=0.0, posinf=0.0, neginf=0.0)
        audio_concealed = np.clip(audio_concealed, -1.0, 1.0)

        return audio_concealed


# =============================================================================
# GAP #5: JITTER CORRECTION
# =============================================================================


class JitterCorrector:
    """
    Corrects jitter artifacts from digital clock errors.

    Jitter: Time-base errors in digital audio clocking.
    Types:
    - Random jitter: White noise-like timing variations
    - Periodic jitter: Clock drift, phase-locked loop errors

    Symptoms:
    - Spectral smearing (phase noise)
    - Reduced clarity/definition
    - Audible as "harshness" in high frequencies

    Algorithm:
    1. Detect jitter via spectral analysis
    2. Estimate timing deviations
    3. Apply resampling correction (if significant)
    4. Gentle highpass filtering to reduce phase noise

    References:
    - Dunn, J. (2003). "The Etymology of Digital Audio"
    - Rumsey, F., & Watkinson, J. (1995). "The Digital Interface Handbook"
    """

    def __init__(self, jitter_threshold_ppm: float = 100.0, correction_strength: float = 0.7):
        """
        Initialisiert Jitter Corrector.

        Parameters
        ----------
        jitter_threshold_ppm : float
            Minimum jitter (parts per million) to correct (default: 100 ppm)
        correction_strength : float
            Strength of correction (0-1, default: 0.7)
        """
        self.jitter_threshold_ppm = np.clip(jitter_threshold_ppm, 10.0, 1000.0)
        self.correction_strength = np.clip(correction_strength, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

        # DSPContract
        self._log_contract()

    def _log_contract(self):
        """Protokolliert DSPContract for auditability."""

    def detect_jitter(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Erkennt jitter level via spectral analysis.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        jitter_ppm : float
            Estimated jitter in parts per million
        """
        # Simplified jitter detection:
        # High-frequency spectral noise is indicator of jitter

        # Highpass filter to isolate HF content
        sos = butter(4, 8000, "highpass", fs=sample_rate, output="sos")
        audio_hf = sosfilt(sos, audio)

        # Compute noise floor in HF
        hf_rms = np.sqrt(np.mean(audio_hf**2))
        full_rms = np.sqrt(np.mean(audio**2))

        # Jitter estimate (simplified heuristic)
        if full_rms > 1e-6:
            noise_ratio = hf_rms / full_rms
            # Convert to ppm (parts per million) - rough estimate
            jitter_ppm = noise_ratio * 1e6 * 0.1  # Scaling factor
        else:
            jitter_ppm = 0.0

        return jitter_ppm  # type: ignore[no-any-return]

    def correct_jitter(self, audio: np.ndarray, jitter_ppm: float, sample_rate: int) -> np.ndarray:
        """
        Correct jitter artifacts.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        jitter_ppm : float
            Detected jitter level
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        audio_corrected : np.ndarray
            Jitter-corrected audio
        """
        # Preserve input dtype
        input_dtype = audio.dtype

        # Apply gentle HF roll-off to reduce phase noise
        # (Simplified correction - full correction would require resampling)

        # Design gentle lowpass at ~18 kHz
        sos = butter(2, 18000, "lowpass", fs=sample_rate, output="sos")
        audio_filtered = sosfilt(sos, audio)

        # Blend with original based on correction strength
        alpha = jitter_ppm / 1000.0 * self.correction_strength  # Scale by jitter level
        alpha = np.clip(alpha, 0.0, 1.0)

        audio_corrected = (1 - alpha) * audio + alpha * audio_filtered

        # Cast back to original dtype
        return audio_corrected.astype(input_dtype)  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet audio to correct jitter.

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
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo
        if audio.ndim == 2:
            left = self.process(audio[0], sample_rate)
            right = self.process(audio[1], sample_rate)

            self.metrics["stereo"] = True

            return np.vstack([left, right])

        # Mono processing
        logger.info("[Jitter] Detecting jitter...")
        jitter_ppm = self.detect_jitter(audio, sample_rate)

        logger.info("[Jitter] Estimated jitter: %.1f ppm", jitter_ppm)

        # Correct if above threshold
        if jitter_ppm > self.jitter_threshold_ppm:
            logger.info("[Jitter] Jitter above Schwelle (%s ppm), correcting...", self.jitter_threshold_ppm)
            audio_corrected = self.correct_jitter(audio, jitter_ppm, sample_rate)
        else:
            logger.info("[Jitter] Jitter within acceptable range, no correction needed")
            audio_corrected = audio

        # Store metrics
        self.metrics["jitter_detected"] = jitter_ppm > self.jitter_threshold_ppm
        self.metrics["jitter_level_ppm"] = jitter_ppm
        self.metrics["correction_applied"] = jitter_ppm > self.jitter_threshold_ppm

        # Final NaN/Inf-Guard and clipping
        audio_corrected = np.nan_to_num(audio_corrected, nan=0.0, posinf=0.0, neginf=0.0)
        audio_corrected = np.clip(audio_corrected, -1.0, 1.0)

        return audio_corrected


# =============================================================================
# UNIFIED DIGITAL RESTORATION SPECIALIST API
# =============================================================================


class DigitalRestorationSpecialist:
    """
    Einheitliche API für digitalspezifische Defektentfernung.

    Combines:
    - CodecArtifactRemover (GAP #3)
    - PacketLossConcealer (GAP #4)
    - JitterCorrector (GAP #5)

    Processing order:
    1. Packet Loss Concealment (fill gaps first)
    2. Jitter Correction (timing/phase issues)
    3. Codec Artifact Removal (compression artifacts last)
    """

    def __init__(
        self,
        enable_codec_artifact_removal: bool = True,
        enable_packet_loss_concealment: bool = True,
        enable_jitter_correction: bool = True,
        # Codec params
        pre_echo_threshold_db: float = -40.0,
        spectral_hole_threshold_db: float = -50.0,
        codec_smoothing_strength: float = 0.6,
        # Packet loss params
        gap_threshold_ms: float = 5.0,
        interpolation_method: str = "cubic",
        # Jitter params
        jitter_threshold_ppm: float = 100.0,
        jitter_correction_strength: float = 0.7,
    ):
        """
        Initialisiert Digital Restoration Specialist.

        Parameters
        ----------
        enable_codec_artifact_removal : bool
            Enable codec artifact removal (GAP #3)
        enable_packet_loss_concealment : bool
            Enable packet loss concealment (GAP #4)
        enable_jitter_correction : bool
            Enable jitter correction (GAP #5)
        ... (see individual classes for parameter descriptions)
        """
        self.enable_codec_artifact_removal = enable_codec_artifact_removal
        self.enable_packet_loss_concealment = enable_packet_loss_concealment
        self.enable_jitter_correction = enable_jitter_correction

        # Initialize modules
        if self.enable_packet_loss_concealment:
            self.packet_loss_concealer = PacketLossConcealer(
                gap_threshold_ms=gap_threshold_ms, interpolation_method=interpolation_method
            )

        if self.enable_jitter_correction:
            self.jitter_corrector = JitterCorrector(
                jitter_threshold_ppm=jitter_threshold_ppm, correction_strength=jitter_correction_strength
            )

        if self.enable_codec_artifact_removal:
            self.codec_artifact_remover = CodecArtifactRemover(
                pre_echo_threshold_db=pre_echo_threshold_db,
                spectral_hole_threshold_db=spectral_hole_threshold_db,
                smoothing_strength=codec_smoothing_strength,
            )

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Verarbeitet audio with digital restoration modules.

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

        # Step 1: Packet Loss Concealment
        if self.enable_packet_loss_concealment:
            logger.info("\n[DigitalRestoration] Step 1/3: Packet Loss Concealment")
            output = self.packet_loss_concealer.process(output, sample_rate)

        # Step 2: Jitter Correction
        if self.enable_jitter_correction:
            logger.info("\n[DigitalRestoration] Step 2/3: Jitter Correction")
            output = self.jitter_corrector.process(output, sample_rate)

        # Step 3: Codec Artifact Removal
        if self.enable_codec_artifact_removal:
            logger.info("\n[DigitalRestoration] Step 3/3: Codec Artifact Removal")
            output = self.codec_artifact_remover.process(output, sample_rate)

        logger.info("\n[DigitalRestoration] Processing vollstaendig!")
        return output

    def get_metrics(self) -> dict:
        """Gibt zurück: metrics from all modules."""
        metrics = {}

        if self.enable_packet_loss_concealment and hasattr(self, "packet_loss_concealer"):
            metrics["packet_loss"] = self.packet_loss_concealer.metrics

        if self.enable_jitter_correction and hasattr(self, "jitter_corrector"):
            metrics["jitter"] = self.jitter_corrector.metrics

        if self.enable_codec_artifact_removal and hasattr(self, "codec_artifact_remover"):
            metrics["codec_artifacts"] = self.codec_artifact_remover.metrics

        return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Digital Restoration Specialist - Digital Defect Removal")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    # Module selection
    parser.add_argument("--no-codec", action="store_true", help="Disable codec artifact removal")
    parser.add_argument("--no-packet-loss", action="store_true", help="Disable packet loss concealment")
    parser.add_argument("--no-jitter", action="store_true", help="Disable jitter correction")

    # Codec params
    parser.add_argument("--pre-echo-threshold", type=float, default=-40.0, help="Pre-echo threshold (dB)")
    parser.add_argument("--codec-smoothing", type=float, default=0.6, help="Codec smoothing strength (0-1)")

    # Packet loss params
    parser.add_argument("--gap-threshold", type=float, default=5.0, help="Min gap threshold (ms)")

    # Jitter params
    parser.add_argument("--jitter-threshold", type=float, default=100.0, help="Jitter threshold (ppm)")
    parser.add_argument("--jitter-strength", type=float, default=0.7, help="Jitter correction strength (0-1)")

    args = parser.parse_args()

    # Load audio
    logger.info("lade: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Transpose if stereo
    if audio.ndim == 2:
        audio = audio.T

    # Initialize processor
    processor = DigitalRestorationSpecialist(
        enable_codec_artifact_removal=not args.no_codec,
        enable_packet_loss_concealment=not args.no_packet_loss,
        enable_jitter_correction=not args.no_jitter,
        pre_echo_threshold_db=args.pre_echo_threshold,
        codec_smoothing_strength=args.codec_smoothing,
        gap_threshold_ms=args.gap_threshold,
        jitter_threshold_ppm=args.jitter_threshold,
        jitter_correction_strength=args.jitter_strength,
    )

    # Process
    logger.info("\nProcessing with Digital Restoration Specialist...")
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
    if output.ndim == 2:
        output = output.T

    sf.write(args.output, output, sr)
    logger.info("gespeichert: %s", args.output)
