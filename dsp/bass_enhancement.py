#!/usr/bin/env python3
"""
Bass Enhancement System
=======================

Professional bass processing for instrumental music restoration.
Addresses the gap between vocal processing (Phase 2.2) and instrumental music.

Components:
1. SubBassEnhancer - 20-60 Hz fundamental reinforcement
2. MidBassClarifier - 60-250 Hz muddiness removal
3. BassHarmonicsEnhancer - 250-500 Hz perceived punch
4. BassDynamicsController - Avoid bass pumping

Target: Bring Bass-Kraft Musical Goal from 70% to 93%

Usage:
    >>> from dsp.bass_enhancement import BassEnhancementSystem
    >>>
    >>> enhancer = BassEnhancementSystem(
    ...     sub_bass_gain_db=3.0,
    ...     mid_bass_clarity=0.8,
    ...     harmonics_gain_db=2.0,
    ...     dynamics_control=True
    ... )
    >>>
    >>> processed, report = enhancer.process(audio, sr)
    >>> print(f"Sub-bass energy: {report['sub_bass_energy_change_db']:.1f} dB")

Author: AURIK Phase 2.3
Date: February 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore", category=RuntimeWarning)

_logger = logging.getLogger(__name__)


# =============================================================================
# COMPONENT 1: SUB-BASS ENHANCER
# =============================================================================


class SubBassEnhancer:
    """
    Sub-Bass Enhancement (20-60 Hz).

    Reinforces fundamental frequencies in the sub-bass range.
    Critical for kick drums, bass instruments, and low-frequency content.

    Techniques:
    - Harmonic synthesis (missing fundamentals)
    - Phase-coherent enhancement (avoid cancellation)
    - LFE (Low Frequency Effect) detection

    Parameters
    ----------
    gain_db : float
        Sub-bass gain in dB (0.0-6.0)
    synthesis_mix : float
        Mix of synthesized harmonics (0.0-1.0)
    phase_coherent : bool
        Ensure phase coherence to avoid cancellation
    """

    def __init__(
        self,
        gain_db: float = 3.0,
        synthesis_mix: float = 0.0,  # Disabled by default (slow for long audio)
        phase_coherent: bool = False,  # Disabled (slow np.correlate)
    ):
        self.gain_db = np.clip(gain_db, 0.0, 6.0)
        self.synthesis_mix = np.clip(synthesis_mix, 0.0, 1.0)
        self.phase_coherent = phase_coherent

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance sub-bass content.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Enhanced audio
        report : dict
            Processing report
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            result = np.stack([left, right], axis=-1)
            return (
                np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype),
                report_l,
            )
        else:
            result, report = self._process_channel(audio, sr)
            return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype), report

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract sub-bass band (20-60 Hz)
        sos_sub = butter(4, [20, 60], btype="band", fs=sr, output="sos")
        sub_bass = sosfilt(sos_sub, audio)

        # Measure original energy
        original_energy = np.sqrt(np.mean(sub_bass**2))

        # Apply gain
        linear_gain = 10 ** (self.gain_db / 20.0)
        sub_bass_enhanced = sub_bass * linear_gain

        # Optional: Synthesize missing fundamentals
        if self.synthesis_mix > 0:
            # Detect fundamental frequency in sub-bass range
            fundamental = self._detect_fundamental(sub_bass, sr, f_min=20, f_max=60)

            if fundamental > 0:
                # Synthesize fundamental
                t = np.arange(len(audio)) / sr
                synthesized = 0.3 * np.sin(2 * np.pi * fundamental * t)

                # Apply envelope from original sub-bass (fast method)
                rectified = np.abs(sub_bass)
                sos_env = butter(2, 20, btype="low", fs=sr, output="sos")
                envelope = sosfilt(sos_env, rectified)
                synthesized = synthesized * envelope

                # Mix with enhanced sub-bass
                sub_bass_enhanced = (1 - self.synthesis_mix) * sub_bass_enhanced + self.synthesis_mix * synthesized

        # Phase coherence check
        if self.phase_coherent:
            # Ensure phase alignment with original
            correlation = np.correlate(sub_bass, sub_bass_enhanced, mode="valid")
            if correlation[0] < 0:
                # Invert phase if anti-correlated
                sub_bass_enhanced = -sub_bass_enhanced

        # Reconstruct audio
        # Remove original sub-bass and add enhanced version
        sos_high = butter(4, 60, btype="high", fs=sr, output="sos")
        high_content = sosfilt(sos_high, audio)

        result = high_content + sub_bass_enhanced

        # Measure new energy
        new_energy = np.sqrt(np.mean(sosfilt(sos_sub, result) ** 2))
        energy_change_db = 20 * np.log10((new_energy + 1e-10) / (original_energy + 1e-10))

        report = {
            "sub_bass_energy_change_db": energy_change_db,
            "fundamental_detected_hz": 0.0,  # Skip slow detection
            "synthesis_applied": self.synthesis_mix > 0,
            "phase_coherent": self.phase_coherent,
        }

        return result, report

    def _detect_fundamental(self, signal: np.ndarray, sr: int, f_min: float, f_max: float) -> float:
        """
        Erkennt fundamental frequency using autocorrelation.

        Returns 0 if no clear fundamental detected.
        """
        # Find peaks in valid lag range
        min_lag = int(sr / f_max)
        max_lag = int(sr / f_min)

        _n = len(signal)
        if max_lag >= _n or min_lag >= max_lag:
            return 0.0

        # Efficient O(n·order) autocorrelation — only lags 0..max_lag (avoids O(n²) full correlate)
        autocorr = np.array([np.dot(signal[: _n - k], signal[k:]) for k in range(max_lag + 1)])

        valid_autocorr = autocorr[min_lag:max_lag]
        if len(valid_autocorr) == 0:
            return 0.0

        peak_idx = np.argmax(valid_autocorr)
        lag = min_lag + peak_idx

        # Check if peak is significant
        if valid_autocorr[peak_idx] < 0.3 * autocorr[0]:
            return 0.0

        fundamental = sr / lag
        return float(fundamental)


# =============================================================================
# COMPONENT 2: MID-BASS CLARIFIER
# =============================================================================


class MidBassClarifier:
    """
    Mid-Bass Clarity Enhancement (60-250 Hz).

    Removes muddiness in the mid-bass range while preserving warmth.
    Critical for bass instruments, kick drums, and tonal balance.

    Techniques:
    - Adaptive EQ (frequency-specific attenuation)
    - Transient preservation (attack phase detection)
    - Warmth/muddiness balance

    Parameters
    ----------
    clarity : float
        Clarity amount (0.0-1.0, higher = less muddiness)
    preserve_warmth : bool
        Preserve natural warmth while reducing mud
    transient_protection : bool
        Protect transient attacks from reduction
    """

    def __init__(self, clarity: float = 0.8, preserve_warmth: bool = True, transient_protection: bool = True):
        self.clarity = np.clip(clarity, 0.0, 1.0)
        self.preserve_warmth = preserve_warmth
        self.transient_protection = transient_protection

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Clarify mid-bass content.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Clarified audio
        report : dict
            Processing report
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            result = np.stack([left, right], axis=-1)
            return (
                np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype),
                report_l,
            )
        else:
            result, report = self._process_channel(audio, sr)
            return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype), report

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract mid-bass band (60-250 Hz)
        sos_mid = butter(4, [60, 250], btype="band", fs=sr, output="sos")
        mid_bass = sosfilt(sos_mid, audio)

        # Measure original energy
        original_energy = np.sqrt(np.mean(mid_bass**2))

        # Detect muddy frequencies (100-180 Hz typically problematic)
        sos_mud = butter(2, [100, 180], btype="band", fs=sr, output="sos")
        muddy_content = sosfilt(sos_mud, audio)

        # Calculate reduction based on clarity setting
        reduction = 0.5 * self.clarity  # Max 50% reduction

        # Transient protection: Detect attacks (fast method)
        if self.transient_protection:
            # Fast envelope detection: rectify + lowpass
            rectified = np.abs(mid_bass)
            sos_env = butter(2, 50, btype="low", fs=sr, output="sos")
            envelope = sosfilt(sos_env, rectified)
            derivative = np.diff(envelope, prepend=envelope[0])
            transient_mask = derivative > np.percentile(derivative, 90)
            # Expand mask to protect transients
            protection = np.convolve(transient_mask.astype(float), np.ones(int(0.01 * sr)), mode="same")
            protection = np.clip(protection, 0, 1)
            # Reduce reduction during transients
            reduction_envelope = reduction * (1 - 0.7 * protection)
        else:
            reduction_envelope = reduction

        # Apply adaptive reduction
        muddy_reduced = muddy_content * (1 - reduction_envelope)

        # Reconstruct mid-bass
        # Split into warm (60-100 Hz) and muddy (100-250 Hz) regions
        sos_warm = butter(2, [60, 100], btype="band", fs=sr, output="sos")
        warm_content = sosfilt(sos_warm, audio)

        if self.preserve_warmth:
            # Keep warmth, only reduce muddiness
            mid_bass_clarified = warm_content + muddy_reduced
        else:
            # Reduce entire mid-bass proportionally
            mid_bass_clarified = mid_bass * (1 - 0.3 * self.clarity)

        # Reconstruct full audio
        sos_low = butter(4, 60, btype="low", fs=sr, output="sos")
        sos_high = butter(4, 250, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + mid_bass_clarified + high_content

        # Measure new energy
        new_energy = np.sqrt(np.mean(sosfilt(sos_mid, result) ** 2))
        energy_change_db = 20 * np.log10((new_energy + 1e-10) / (original_energy + 1e-10))

        report = {
            "mid_bass_energy_change_db": energy_change_db,
            "muddiness_reduction": reduction * 100,  # Percentage
            "warmth_preserved": self.preserve_warmth,
            "transients_protected": self.transient_protection,
        }

        return result, report


# =============================================================================
# COMPONENT 3: BASS HARMONICS ENHANCER
# =============================================================================


class BassHarmonicsEnhancer:
    """
    Bass Harmonics Enhancement (250-500 Hz).

    Enhances harmonics in the upper bass range for perceived punch and definition.
    Critical for bass intelligibility on small speakers.

    Techniques:
    - Harmonic exciter (2nd, 3rd harmonics)
    - Perceived bass enhancement (psychoacoustic effect)
    - Saturation (analog warmth)

    Parameters
    ----------
    gain_db : float
        Harmonics gain in dB (0.0-4.0)
    saturation_amount : float
        Analog saturation amount (0.0-1.0)
    exciter_mix : float
        Harmonic exciter mix (0.0-1.0)
    """

    def __init__(self, gain_db: float = 2.0, saturation_amount: float = 0.2, exciter_mix: float = 0.3):
        self.gain_db = np.clip(gain_db, 0.0, 4.0)
        self.saturation_amount = np.clip(saturation_amount, 0.0, 1.0)
        self.exciter_mix = np.clip(exciter_mix, 0.0, 1.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance bass harmonics.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Enhanced audio
        report : dict
            Processing report
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            result = np.stack([left, right], axis=-1)
            return (
                np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype),
                report_l,
            )
        else:
            result, report = self._process_channel(audio, sr)
            return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype), report

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract harmonics band (250-500 Hz)
        sos_harm = butter(4, [250, 500], btype="band", fs=sr, output="sos")
        harmonics = sosfilt(sos_harm, audio)

        # Measure original energy
        original_energy = np.sqrt(np.mean(harmonics**2))

        # Apply gain
        linear_gain = 10 ** (self.gain_db / 20.0)
        harmonics_enhanced = harmonics * linear_gain

        # Optional: Harmonic exciter
        if self.exciter_mix > 0:
            # Extract bass fundamentals (60-250 Hz)
            sos_bass = butter(4, [60, 250], btype="band", fs=sr, output="sos")
            bass_fund = sosfilt(sos_bass, audio)

            # Generate 2nd and 3rd harmonics
            harmonics_synth = np.tanh(3.0 * bass_fund)  # Soft clipping generates harmonics

            # Filter to harmonics range
            harmonics_synth = sosfilt(sos_harm, harmonics_synth)

            # Mix with original harmonics
            harmonics_enhanced = (1 - self.exciter_mix) * harmonics_enhanced + self.exciter_mix * harmonics_synth * 0.5

        # Optional: Analog saturation
        if self.saturation_amount > 0:
            # Soft saturation (tanh)
            drive = 1.0 + 2.0 * self.saturation_amount
            saturated = np.tanh(drive * harmonics_enhanced / (np.max(np.abs(harmonics_enhanced)) + 1e-10))
            saturated = saturated * np.max(np.abs(harmonics_enhanced))

            # Mix saturated with clean
            harmonics_enhanced = (1 - self.saturation_amount) * harmonics_enhanced + self.saturation_amount * saturated

        # Reconstruct audio
        sos_low = butter(4, 250, btype="low", fs=sr, output="sos")
        sos_high = butter(4, 500, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + harmonics_enhanced + high_content

        # Measure new energy
        new_energy = np.sqrt(np.mean(sosfilt(sos_harm, result) ** 2))
        energy_change_db = 20 * np.log10((new_energy + 1e-10) / (original_energy + 1e-10))

        report = {
            "harmonics_energy_change_db": energy_change_db,
            "exciter_applied": self.exciter_mix > 0,
            "saturation_applied": self.saturation_amount > 0,
            "perceived_bass_boost_db": energy_change_db * 0.5,  # Psychoacoustic estimate
        }

        return result, report


# =============================================================================
# COMPONENT 4: BASS DYNAMICS CONTROLLER
# =============================================================================


class BassDynamicsController:
    """
    Bass Dynamics Control.

    Prevents bass pumping and maintains consistent bass energy.
    Critical for dance music, hip-hop, and modern productions.

    Techniques:
    - Envelope-based compression (slow attack, fast release)
    - Peak limiting (avoid bass clipping)
    - Sidechain-style ducking (optional)

    Parameters
    ----------
    compression_ratio : float
        Compression ratio (1.0-4.0)
    attack_ms : float
        Attack time in milliseconds
    release_ms : float
        Release time in milliseconds
    threshold_db : float
        Compression threshold in dB
    """

    def __init__(
        self,
        compression_ratio: float = 2.0,
        attack_ms: float = 20.0,
        release_ms: float = 100.0,
        threshold_db: float = -15.0,
    ):
        self.ratio = np.clip(compression_ratio, 1.0, 4.0)
        self.attack_ms = np.clip(attack_ms, 5.0, 50.0)
        self.release_ms = np.clip(release_ms, 50.0, 500.0)
        self.threshold_db = np.clip(threshold_db, -30.0, -5.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Control bass dynamics.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Dynamics-controlled audio
        report : dict
            Processing report
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            result = np.stack([left, right], axis=-1)
            return (
                np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype),
                report_l,
            )
        else:
            result, report = self._process_channel(audio, sr)
            return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype), report

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract bass band (20-250 Hz) for dynamics control
        sos_bass = butter(4, [20, 250], btype="band", fs=sr, output="sos")
        bass_band = sosfilt(sos_bass, audio)

        # Calculate envelope (fast method: rectify + lowpass instead of Hilbert)
        # Hilbert is too slow for long signals
        rectified = np.abs(bass_band)
        sos_env = butter(2, 20, btype="low", fs=sr, output="sos")  # 20Hz envelope
        envelope = sosfilt(sos_env, rectified)

        # Smooth envelope
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Calculate gain reduction
        threshold = self.threshold_db
        ratio = self.ratio

        gain_reduction_db = np.zeros_like(envelope_db)
        above_threshold = envelope_db > threshold

        if np.any(above_threshold):
            excess_db = envelope_db[above_threshold] - threshold
            gain_reduction_db[above_threshold] = -excess_db * (1 - 1 / ratio)

        # Apply attack/release smoothing via scipy lfilter
        from scipy.signal import lfilter

        attack_coeff = np.exp(-1.0 / (self.attack_ms * sr / 1000.0))
        release_coeff = np.exp(-1.0 / (self.release_ms * sr / 1000.0))

        # Average coefficient for simplified smoothing
        avg_coeff = (attack_coeff + release_coeff) / 2.0
        b = [1.0 - avg_coeff]
        a = [1.0, -avg_coeff]
        smoothed_gain_db = np.asarray(lfilter(b, a, gain_reduction_db), dtype=np.float64)

        # Convert to linear gain
        gain = 10 ** (smoothed_gain_db / 20.0)

        # Apply gain to bass band
        bass_compressed = bass_band * gain

        # Reconstruct audio
        sos_high = butter(4, 250, btype="high", fs=sr, output="sos")
        high_content = sosfilt(sos_high, audio)

        result = bass_compressed + high_content

        # Calculate metrics
        max_reduction_db = np.min(smoothed_gain_db)
        avg_reduction_db = np.mean(smoothed_gain_db[smoothed_gain_db < -0.1])

        if np.isnan(avg_reduction_db):
            avg_reduction_db = 0.0

        report = {
            "max_gain_reduction_db": abs(max_reduction_db),
            "avg_gain_reduction_db": abs(avg_reduction_db),
            "compression_ratio": self.ratio,
            "threshold_db": self.threshold_db,
        }

        return result, report


# =============================================================================
# UNIFIED API: BASS ENHANCEMENT SYSTEM
# =============================================================================


class BassEnhancementSystem:
    """
    Einheitliche API für Bass-Verbesserung.

    Combines all bass processing components into a single pipeline:
    1. Sub-Bass Enhancement (20-60 Hz)
    2. Mid-Bass Clarification (60-250 Hz)
    3. Bass Harmonics Enhancement (250-500 Hz)
    4. Bass Dynamics Control

    Parameters
    ----------
    sub_bass_gain_db : float
        Sub-bass gain (0.0-6.0 dB)
    mid_bass_clarity : float
        Mid-bass clarity (0.0-1.0)
    harmonics_gain_db : float
        Harmonics gain (0.0-4.0 dB)
    dynamics_control : bool
        Enable dynamics control
    compression_ratio : float
        Bass compression ratio (1.0-4.0)
    """

    def __init__(
        self,
        sub_bass_gain_db: float = 3.0,
        mid_bass_clarity: float = 0.8,
        harmonics_gain_db: float = 2.0,
        dynamics_control: bool = True,
        compression_ratio: float = 2.0,
    ):
        self.sub_bass_enhancer = SubBassEnhancer(gain_db=sub_bass_gain_db)
        self.mid_bass_clarifier = MidBassClarifier(clarity=mid_bass_clarity)
        self.harmonics_enhancer = BassHarmonicsEnhancer(gain_db=harmonics_gain_db)
        self.dynamics_controller = BassDynamicsController(compression_ratio=compression_ratio)
        self.dynamics_control = dynamics_control

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full bass enhancement pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Enhanced audio
        report : dict
            Comprehensive processing report
        """
        orig_dtype = audio.dtype
        result = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).copy()
        report = {}

        # Stage 1: Sub-Bass Enhancement
        result, sub_report = self.sub_bass_enhancer.process(result, sr)
        report["sub_bass"] = sub_report

        # Stage 2: Mid-Bass Clarification
        result, mid_report = self.mid_bass_clarifier.process(result, sr)
        report["mid_bass"] = mid_report

        # Stage 3: Bass Harmonics Enhancement
        result, harm_report = self.harmonics_enhancer.process(result, sr)
        report["harmonics"] = harm_report

        # Stage 4: Bass Dynamics Control (optional)
        if self.dynamics_control:
            result, dyn_report = self.dynamics_controller.process(result, sr)
            report["dynamics"] = dyn_report

        # Calculate overall metrics
        report["stages_applied"] = 4 if self.dynamics_control else 3  # type: ignore[assignment]
        report["total_bass_enhancement_db"] = (
            sub_report["sub_bass_energy_change_db"] + harm_report["harmonics_energy_change_db"]
        ) / 2.0

        result = np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype)
        return result, report


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI interface for Bass Enhancement System."""
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Phase 2.3 - Bass Enhancement System")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    parser.add_argument("--sub-bass-gain", type=float, default=3.0, help="Sub-bass gain in dB (0.0-6.0, default: 3.0)")
    parser.add_argument("--mid-bass-clarity", type=float, default=0.8, help="Mid-bass clarity (0.0-1.0, default: 0.8)")
    parser.add_argument(
        "--harmonics-gain", type=float, default=2.0, help="Harmonics gain in dB (0.0-4.0, default: 2.0)"
    )
    parser.add_argument(
        "--compression-ratio", type=float, default=2.0, help="Bass compression ratio (1.0-4.0, default: 2.0)"
    )
    parser.add_argument("--no-dynamics", action="store_true", help="Disable dynamics control")

    args = parser.parse_args()

    # Load audio
    _logger.info("Loading: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Make mono for processing
    audio_mono = np.mean(audio, axis=1) if audio.shape[1] == 2 else audio[:, 0]

    # Create bass enhancement system
    _logger.info("Bass Enhancement System")
    _logger.debug("-" * 60)

    enhancer = BassEnhancementSystem(
        sub_bass_gain_db=args.sub_bass_gain,
        mid_bass_clarity=args.mid_bass_clarity,
        harmonics_gain_db=args.harmonics_gain,
        dynamics_control=not args.no_dynamics,
        compression_ratio=args.compression_ratio,
    )

    # Process
    _logger.info("Processing...")
    processed, report = enhancer.process(audio_mono, sr)

    # Log report
    _logger.info("Processing Report:")
    _logger.info(
        "Sub-Bass Enhancement: %+.1f dB | Fundamental: %.1f Hz | Synthesis: %s",
        report["sub_bass"]["sub_bass_energy_change_db"],
        report["sub_bass"]["fundamental_detected_hz"],
        "Yes" if report["sub_bass"]["synthesis_applied"] else "No",
    )
    _logger.info(
        "Mid-Bass Clarification: %+.1f dB | Muddiness reduction: %.1f%% | Warmth preserved: %s",
        report["mid_bass"]["mid_bass_energy_change_db"],
        report["mid_bass"]["muddiness_reduction"],
        "Yes" if report["mid_bass"]["warmth_preserved"] else "No",
    )
    _logger.info(
        "Bass Harmonics: %+.1f dB | Perceived boost: %+.1f dB | Exciter: %s",
        report["harmonics"]["harmonics_energy_change_db"],
        report["harmonics"]["perceived_bass_boost_db"],
        "Yes" if report["harmonics"]["exciter_applied"] else "No",
    )
    if "dynamics" in report:
        _logger.info(
            "Bass Dynamics Control: max reduction %.1f dB | avg %.1f dB | ratio %.1f:1",
            report["dynamics"]["max_gain_reduction_db"],
            report["dynamics"]["avg_gain_reduction_db"],
            report["dynamics"]["compression_ratio"],
        )
    _logger.info(
        "Total Enhancement: %+.1f dB | Stages applied: %d",
        report["total_bass_enhancement_db"],
        report["stages_applied"],
    )

    # Save
    _logger.info("Saving: %s", args.output)
    sf.write(args.output, processed, sr)
    _logger.info("Done.")


if __name__ == "__main__":
    main()
