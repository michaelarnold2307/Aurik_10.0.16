#!/usr/bin/env python3
"""
Guitar/String Enhancement System
=================================

Professional guitar and string instrument processing for instrumental music restoration.
Addresses the gap between vocal processing (Phase 2.2) and instrumental music.

Components:
1. PickAttackEnhancer - String transient preservation and enhancement
2. StringResonanceEnhancer - Fundamental and overtone enhancement
3. FretNoiseReducer - Artistic balance of mechanical noise
4. AcousticBodyResonance - Natural acoustic character

Target: Bring Brillanz Musical Goal from 85% to 93% for guitar/string content

Usage:
    >>> from dsp.guitar_enhancement import GuitarEnhancementSystem
    >>>
    >>> enhancer = GuitarEnhancementSystem(
    ...     pick_attack_db=2.0,
    ...     string_resonance=0.8,
    ...     fret_noise_reduction=0.6,
    ...     body_resonance_db=1.5
    ... )
    >>>
    >>> processed, report = enhancer.process(audio, sr)
    >>> print(f"String clarity: {report['string_clarity_db']:.1f} dB")

Author: AURIK Phase 2.3
Date: February 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _match_lengths(*arrays):
    """Stellt sicher: all arrays have the same length (trim to minimum)."""
    min_len = min(len(arr) for arr in arrays)
    return tuple(arr[:min_len] for arr in arrays)


# =============================================================================
# COMPONENT 1: PICK ATTACK ENHANCER
# =============================================================================


class PickAttackEnhancer:
    """
    Pick Attack Enhancement.

    Preserves and enhances string transients from pick/pluck attacks.
    Critical for string articulation and playing dynamics.

    Techniques:
    - Transient detection and enhancement
    - Attack phase identification
    - Pick noise vs. musical transient separation

    Parameters
    ----------
    attack_db : float
        Attack enhancement gain in dB (0.0-4.0)
    transient_sharpness : float
        Transient sharpness (0.0-1.0)
    preserve_dynamics : bool
        Preserve playing dynamics
    """

    def __init__(self, attack_db: float = 2.0, transient_sharpness: float = 0.7, preserve_dynamics: bool = True):
        self.attack_db = np.clip(attack_db, 0.0, 4.0)
        self.transient_sharpness = np.clip(transient_sharpness, 0.0, 1.0)
        self.preserve_dynamics = preserve_dynamics

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance pick attacks.

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
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            # Ensure both channels have same length
            left, right = _match_lengths(left, right)
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # String transients are typically 2-6 kHz for guitars
        nyquist = sr / 2
        low_f = min(2000, nyquist * 0.25)
        high_f = min(6000, nyquist * 0.95)
        sos_transient = butter(4, [low_f, high_f], btype="band", fs=sr, output="sos")
        transient_band = sosfilt(sos_transient, audio)

        # Measure original energy
        original_energy = np.sqrt(np.mean(transient_band**2))

        # Detect transient events (pick attacks)
        envelope = np.abs(np.asarray(hilbert(transient_band), dtype=np.complex128))

        # Find attack phases (steep rises)
        derivative = np.diff(envelope, prepend=envelope[0])

        # Smooth derivative
        window_size = int(0.002 * sr)  # 2ms
        derivative_smooth = np.convolve(derivative, np.ones(window_size) / window_size, mode="same")

        # Identify transients (positive derivative above threshold)
        transient_threshold = np.percentile(np.abs(derivative_smooth), 90)
        attack_mask = (derivative_smooth > transient_threshold).astype(float)

        # Expand attack window
        attack_duration = int(0.010 * sr)  # 10ms attack window
        attack_mask_expanded = np.convolve(attack_mask, np.ones(attack_duration) / attack_duration, mode="same")
        attack_mask_expanded = np.clip(attack_mask_expanded, 0, 1)

        # Apply attack enhancement
        attack_gain = 10 ** (self.attack_db / 20.0)

        # If preserve dynamics, scale gain by envelope
        if self.preserve_dynamics:
            # Normalize envelope to [0, 1]
            envelope_norm = envelope / (np.max(envelope) + 1e-10)
            # Stronger enhancement for weaker attacks (compression-like)
            dynamic_gain = attack_gain * (1.0 + (1.0 - envelope_norm) * 0.3)
        else:
            dynamic_gain = attack_gain

        # Apply gain with attack mask
        result = audio.copy()
        enhancement = transient_band * attack_mask_expanded * (dynamic_gain - 1)
        # Ensure all arrays have same length
        min_len = min(len(result), len(enhancement), len(transient_band), len(attack_mask_expanded))
        result = result[:min_len]
        enhancement = enhancement[:min_len]
        transient_band = transient_band[:min_len]
        attack_mask_expanded = attack_mask_expanded[:min_len]
        result = result + enhancement

        # Optional: Transient sharpening
        if self.transient_sharpness > 0:
            # High-pass filter for super-sharp transients
            sos_sharp = butter(2, 4000, btype="high", fs=sr, output="sos")
            sharp_transients = sosfilt(sos_sharp, transient_band)

            # Apply only to attack mask regions
            sharp_enhancement = sharp_transients * attack_mask_expanded * self.transient_sharpness * 0.2
            # Ensure all arrays have same length
            min_len = min(len(result), len(sharp_enhancement))
            result = result[:min_len]
            sharp_enhancement = sharp_enhancement[:min_len]
            result = result + sharp_enhancement

        # Measure new energy
        new_energy = np.sqrt(np.mean(sosfilt(sos_transient, result) ** 2))
        energy_change_db = 20 * np.log10((new_energy + 1e-10) / (original_energy + 1e-10))

        # Count transient events
        num_transients = np.sum(attack_mask > 0.5)

        report = {
            "pick_attack_energy_change_db": energy_change_db,
            "transients_detected": int(num_transients),
            "dynamics_preserved": self.preserve_dynamics,
            "sharpening_applied": self.transient_sharpness > 0,
        }

        return result, report


# =============================================================================
# COMPONENT 2: STRING RESONANCE ENHANCER
# =============================================================================


class StringResonanceEnhancer:
    """
    String Resonance Enhancement.

    Enhances fundamental frequencies and overtones of string vibrations.
    Critical for tonal richness and harmonic content.

    Techniques:
    - Fundamental frequency enhancement (80-400 Hz)
    - Harmonic series enhancement (overtones)
    - Sympathetic resonance simulation

    Parameters
    ----------
    resonance : float
        String resonance amount (0.0-1.0)
    fundamental_db : float
        Fundamental frequency gain (0.0-3.0 dB)
    harmonic_enhancement : float
        Harmonic overtone enhancement (0.0-1.0)
    """

    def __init__(self, resonance: float = 0.8, fundamental_db: float = 2.0, harmonic_enhancement: float = 0.6):
        self.resonance = np.clip(resonance, 0.0, 1.0)
        self.fundamental_db = np.clip(fundamental_db, 0.0, 3.0)
        self.harmonic_enhancement = np.clip(harmonic_enhancement, 0.0, 1.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance string resonance.

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
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            # Ensure both channels have same length
            left, right = _match_lengths(left, right)
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        result = audio.copy()

        # Fundamental frequency range for guitars (82 Hz = E2 to ~400 Hz)
        sos_fundamental = butter(4, [80, 400], btype="band", fs=sr, output="sos")
        fundamental_band = sosfilt(sos_fundamental, audio)

        # Measure original energy
        fundamental_energy_orig = np.sqrt(np.mean(fundamental_band**2))

        # Enhance fundamental
        fundamental_gain = 10 ** (self.fundamental_db / 20.0)
        fundamental_enhanced = fundamental_band * fundamental_gain

        # Replace fundamental in result
        sos_low = butter(4, 80, btype="low", fs=sr, output="sos")
        sos_high = butter(4, 400, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        low_content, fundamental_enhanced, high_content = _match_lengths(
            low_content, fundamental_enhanced, high_content
        )
        result = low_content + fundamental_enhanced + high_content

        # Harmonic enhancement (overtone series)
        if self.harmonic_enhancement > 0:
            # Guitar harmonics are typically in 400 Hz - 3 kHz range
            sos_harmonics = butter(4, [400, 3000], btype="band", fs=sr, output="sos")
            harmonics_band = sosfilt(sos_harmonics, result)

            # Gentle harmonic exciter (soft saturation)
            harmonics_excited = np.tanh(harmonics_band * 2.0) * 0.5

            # Mix excited harmonics
            harmonics_enhanced = harmonics_band + harmonics_excited * self.harmonic_enhancement

            # Replace harmonics in result
            sos_low = butter(4, 400, btype="low", fs=sr, output="sos")
            sos_high = butter(4, 3000, btype="high", fs=sr, output="sos")

            low_content = sosfilt(sos_low, result)
            high_content = sosfilt(sos_high, result)

            low_content, harmonics_enhanced, high_content = _match_lengths(
                low_content, harmonics_enhanced, high_content
            )
            result = low_content + harmonics_enhanced + high_content

        # Measure new energy
        fundamental_energy_new = np.sqrt(np.mean(sosfilt(sos_fundamental, result) ** 2))
        fundamental_change_db = 20 * np.log10((fundamental_energy_new + 1e-10) / (fundamental_energy_orig + 1e-10))

        report = {
            "fundamental_energy_change_db": fundamental_change_db,
            "harmonic_enhancement_applied": self.harmonic_enhancement > 0,
            "resonance_amount": self.resonance,
        }

        return result, report


# =============================================================================
# COMPONENT 3: FRET NOISE REDUCER
# =============================================================================


class FretNoiseReducer:
    """
    Fret Noise Reduction.

    Reduces mechanical fret noise while maintaining realism.
    Critical for artistic balance between clean and natural sound.

    Techniques:
    - Fret squeak detection (2-8 kHz)
    - Artistic reduction (not complete removal)
    - Finger slide preservation

    Parameters
    ----------
    reduction : float
        Fret noise reduction amount (0.0-1.0)
    preserve_slides : bool
        Preserve intentional finger slides
    high_freq_cleanup : bool
        Clean up excessive high-frequency noise
    """

    def __init__(self, reduction: float = 0.6, preserve_slides: bool = True, high_freq_cleanup: bool = True):
        self.reduction = np.clip(reduction, 0.0, 1.0)
        self.preserve_slides = preserve_slides
        self.high_freq_cleanup = high_freq_cleanup

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Reduce fret noise.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Processed audio
        report : dict
            Processing report
        """
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            # Ensure both channels have same length
            left, right = _match_lengths(left, right)
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Fret noise is typically 2-8 kHz with harsh characteristics
        nyquist = sr / 2
        low_f = min(2000, nyquist * 0.25)
        high_f = min(8000, nyquist * 0.95)
        sos_fret = butter(4, [low_f, high_f], btype="band", fs=sr, output="sos")
        fret_band = sosfilt(sos_fret, audio)

        # Measure original noise energy
        noise_energy_orig = np.sqrt(np.mean(fret_band**2))

        # Detect noise-like content (high zero-crossing rate)
        zero_crossings = np.diff(np.sign(fret_band)) != 0
        zcr = np.convolve(zero_crossings.astype(float), np.ones(int(0.02 * sr)) / (0.02 * sr), mode="same")

        # High ZCR indicates noisy content
        noise_mask = zcr > np.percentile(zcr, 70)
        noise_mask = noise_mask.astype(float)

        # Smooth mask
        window_size = int(0.01 * sr)  # 10ms
        noise_mask_smooth = np.convolve(noise_mask, np.ones(window_size) / window_size, mode="same")

        # Preserve slides (longer continuous events)
        if self.preserve_slides:
            # Slides have lower ZCR but sustained energy
            envelope = np.abs(np.asarray(hilbert(fret_band), dtype=np.complex128))
            sustained_mask = envelope > np.percentile(envelope, 50)

            # Ensure masks have same length
            noise_mask_smooth, sustained_mask = _match_lengths(noise_mask_smooth, sustained_mask)

            # Reduce noise reduction where slides detected
            noise_mask_smooth = noise_mask_smooth * (1 - sustained_mask.astype(float) * 0.7)

        # Apply reduction
        reduction_factor = 1.0 - self.reduction
        # Ensure fret_band and noise_mask_smooth have same length
        fret_band, noise_mask_smooth = _match_lengths(fret_band, noise_mask_smooth)
        fret_band_reduced = fret_band * (1 - noise_mask_smooth * (1 - reduction_factor))

        # High-frequency cleanup (optional)
        if self.high_freq_cleanup:
            # Target harsh frequencies (4-8 kHz)
            nyquist = sr / 2
            low_f = min(4000, nyquist * 0.25)
            high_f = min(8000, nyquist * 0.95)
            sos_harsh = butter(4, [low_f, high_f], btype="band", fs=sr, output="sos")
            harsh_band = sosfilt(sos_harsh, fret_band_reduced)

            # Gentle reduction
            harsh_reduced = harsh_band * 0.7

            # Replace in fret band
            nyquist = sr / 2
            high_cutoff = min(8000, nyquist * 0.95)
            sos_low = butter(4, 4000, btype="low", fs=sr, output="sos")
            sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

            fret_low = sosfilt(sos_low, fret_band_reduced)
            fret_high = sosfilt(sos_high, fret_band_reduced)

            fret_band_reduced = fret_low + harsh_reduced + fret_high

        # Reconstruct audio
        nyquist = sr / 2
        high_cutoff = min(8000, nyquist * 0.95)
        sos_low = butter(4, 2000, btype="low", fs=sr, output="sos")
        sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        low_content, fret_band_reduced, high_content = _match_lengths(low_content, fret_band_reduced, high_content)
        result = low_content + fret_band_reduced + high_content

        # Measure reduction
        nyquist = sr / 2
        low_f = min(2000, nyquist * 0.25)
        high_f = min(8000, nyquist * 0.95)
        sos_fret = butter(4, [low_f, high_f], btype="band", fs=sr, output="sos")
        noise_energy_new = np.sqrt(np.mean(sosfilt(sos_fret, result) ** 2))

        reduction_db = 20 * np.log10((noise_energy_new + 1e-10) / (noise_energy_orig + 1e-10))
        reduction_percent = (1 - noise_energy_new / (noise_energy_orig + 1e-10)) * 100

        report = {
            "fret_noise_reduction_db": reduction_db,
            "fret_noise_reduction_percent": np.clip(reduction_percent, 0, 100),
            "slides_preserved": self.preserve_slides,
            "high_freq_cleanup_applied": self.high_freq_cleanup,
        }

        return result, report


# =============================================================================
# COMPONENT 4: ACOUSTIC BODY RESONANCE
# =============================================================================


class AcousticBodyResonance:
    """
    Acoustic Body Resonance Enhancement.

    Enhances natural resonance of acoustic guitar body.
    Critical for warmth and natural acoustic character.

    Techniques:
    - Body resonance peaks (100-300 Hz)
    - Soundhole/rosette resonance
    - Warmth enhancement

    Parameters
    ----------
    resonance_db : float
        Body resonance gain (0.0-3.0 dB)
    warmth : float
        Warmth enhancement (0.0-1.0)
    natural_character : bool
        Preserve natural acoustic character
    """

    def __init__(self, resonance_db: float = 1.5, warmth: float = 0.7, natural_character: bool = True):
        self.resonance_db = np.clip(resonance_db, 0.0, 3.0)
        self.warmth = np.clip(warmth, 0.0, 1.0)
        self.natural_character = natural_character

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance acoustic body resonance.

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
        # Handle stereo
        if audio.ndim == 2:
            left, report_l = self._process_channel(audio[:, 0], sr)
            right, _report_r = self._process_channel(audio[:, 1], sr)
            # Ensure both channels have same length
            left, right = _match_lengths(left, right)
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Body resonance range (100-300 Hz)
        sos_body = butter(4, [100, 300], btype="band", fs=sr, output="sos")
        body_band = sosfilt(sos_body, audio)

        # Measure original energy
        body_energy_orig = np.sqrt(np.mean(body_band**2))

        # Apply resonance gain
        resonance_gain = 10 ** (self.resonance_db / 20.0)
        body_enhanced = body_band * resonance_gain

        # Warmth enhancement (lower end of body resonance)
        if self.warmth > 0:
            sos_warm = butter(2, [100, 200], btype="band", fs=sr, output="sos")
            warmth_band = sosfilt(sos_warm, body_enhanced)

            # Extra warmth boost
            warmth_boosted = warmth_band * (1.0 + self.warmth * 0.3)

            # Mix back
            body_enhanced = body_enhanced + (warmth_boosted - warmth_band)

        # Natural character preservation (optional)
        if self.natural_character:
            # Avoid over-enhancement: use envelope-based limiting
            envelope = np.abs(np.asarray(hilbert(body_enhanced), dtype=np.complex128))

            # Gentle compression on loud passages
            threshold = np.percentile(envelope, 80)
            over_threshold = envelope > threshold

            # Soft limiting
            compression_ratio = 0.7
            body_enhanced[over_threshold] *= compression_ratio

        # Reconstruct audio
        sos_low = butter(4, 100, btype="low", fs=sr, output="sos")
        sos_high = butter(4, 300, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        low_content, body_enhanced, high_content = _match_lengths(low_content, body_enhanced, high_content)
        result = low_content + body_enhanced + high_content

        # Measure new energy
        body_energy_new = np.sqrt(np.mean(sosfilt(sos_body, result) ** 2))
        body_change_db = 20 * np.log10((body_energy_new + 1e-10) / (body_energy_orig + 1e-10))

        report = {
            "body_resonance_change_db": body_change_db,
            "warmth_applied": self.warmth > 0,
            "natural_character_preserved": self.natural_character,
        }

        return result, report


# =============================================================================
# UNIFIED API: GUITAR ENHANCEMENT SYSTEM
# =============================================================================


class GuitarEnhancementSystem:
    """
    Einheitliche API für Gitarren-/Saiten-Verbesserung.

    Combines all guitar processing components into a single pipeline:
    1. Pick Attack Enhancement
    2. String Resonance Enhancement
    3. Fret Noise Reduction
    4. Acoustic Body Resonance

    Parameters
    ----------
    pick_attack_db : float
        Pick attack gain (0.0-4.0 dB)
    string_resonance : float
        String resonance (0.0-1.0)
    fret_noise_reduction : float
        Fret noise reduction (0.0-1.0)
    body_resonance_db : float
        Body resonance gain (0.0-3.0 dB)
    """

    def __init__(
        self,
        pick_attack_db: float = 2.0,
        string_resonance: float = 0.8,
        fret_noise_reduction: float = 0.6,
        body_resonance_db: float = 1.5,
    ):
        self.pick_enhancer = PickAttackEnhancer(attack_db=pick_attack_db)
        self.resonance_enhancer = StringResonanceEnhancer(resonance=string_resonance)
        self.noise_reducer = FretNoiseReducer(reduction=fret_noise_reduction)
        self.body_enhancer = AcousticBodyResonance(resonance_db=body_resonance_db)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full guitar enhancement pipeline.

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
        result = audio.copy()
        report = {}

        # Stage 1: Pick Attack Enhancement
        result, pick_report = self.pick_enhancer.process(result, sr)
        report["pick_attack"] = pick_report

        # Stage 2: String Resonance Enhancement
        result, resonance_report = self.resonance_enhancer.process(result, sr)
        report["string_resonance"] = resonance_report

        # Stage 3: Fret Noise Reduction
        result, noise_report = self.noise_reducer.process(result, sr)
        report["fret_noise"] = noise_report

        # Stage 4: Acoustic Body Resonance
        result, body_report = self.body_enhancer.process(result, sr)
        report["body_resonance"] = body_report

        # Calculate overall metrics
        report["stages_applied"] = 4  # type: ignore[assignment]
        report["string_clarity_db"] = (
            pick_report["pick_attack_energy_change_db"] + resonance_report["fundamental_energy_change_db"]
        ) / 2.0

        return result, report


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI interface for Guitar Enhancement System."""
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Phase 2.3 - Guitar/String Enhancement System")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    parser.add_argument("--pick-attack", type=float, default=2.0, help="Pick attack gain in dB (0.0-4.0, default: 2.0)")
    parser.add_argument("--string-resonance", type=float, default=0.8, help="String resonance (0.0-1.0, default: 0.8)")
    parser.add_argument(
        "--fret-noise-reduction", type=float, default=0.6, help="Fret noise reduction (0.0-1.0, default: 0.6)"
    )
    parser.add_argument(
        "--body-resonance", type=float, default=1.5, help="Body resonance in dB (0.0-3.0, default: 1.5)"
    )

    args = parser.parse_args()

    # Load audio
    logger.info("lade: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Make mono for processing
    audio_mono = np.mean(audio, axis=1) if audio.shape[1] == 2 else audio[:, 0]

    # Create guitar enhancement system
    logger.info("\n🎸 Guitar/String Enhancement System")
    logger.info("=" * 60)

    enhancer = GuitarEnhancementSystem(
        pick_attack_db=args.pick_attack,
        string_resonance=args.string_resonance,
        fret_noise_reduction=args.fret_noise_reduction,
        body_resonance_db=args.body_resonance,
    )

    # Process
    logger.info("Processing...")
    processed, report = enhancer.process(audio_mono, sr)

    # Print report
    logger.info("\n📊 Processing Report:")
    logger.info("-" * 60)
    logger.info("Pick Attack: %.1f dB", report["pick_attack"]["pick_attack_energy_change_db"])
    logger.info("  Transients erkannt: %s", report["pick_attack"]["transients_detected"])

    logger.info("\nString Resonance: %.1f dB", report["string_resonance"]["fundamental_energy_change_db"])
    logger.info(
        f"  Harmonic enhancement: {'Yes' if report['string_resonance']['harmonic_enhancement_applied'] else 'No'}"
    )

    logger.info("\nFret Noise: %.1f dB", report["fret_noise"]["fret_noise_reduction_db"])
    logger.info("  Reduction: %.1f%%", report["fret_noise"]["fret_noise_reduction_percent"])

    logger.info("\nBody Resonance: %.1f dB", report["body_resonance"]["body_resonance_change_db"])
    logger.info("  Warmth angewendet: %s", "Yes" if report["body_resonance"]["warmth_applied"] else "No")

    logger.info("\nString Clarity: %.1f dB", report["string_clarity_db"])
    logger.info("Stages angewendet: %s", report["stages_applied"])

    # Save
    logger.info("\nspeichere: %s", args.output)
    sf.write(args.output, processed, sr)
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
