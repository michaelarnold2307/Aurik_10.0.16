#!/usr/bin/env python3
"""
Brass/Wind Enhancement System
==============================

Professional brass and wind instrument processing for instrumental music restoration.
Addresses the gap between vocal processing (Phase 2.2) and instrumental music.

Components:
1. BrassHarmonicsEnhancer - Characteristic brass timbre and richness
2. BreathAttackPreserver - Natural breath articulation
3. ValveClickReducer - Mechanical valve noise reduction
4. ResonanceEnhancer - Acoustic brass character and resonance

Target: Bring Emotionalität Musical Goal from 88% to 95% for brass/wind content

Usage:
    >>> from dsp.brass_enhancement import BrassEnhancementSystem
    >>>
    >>> enhancer = BrassEnhancementSystem(
    ...     harmonics_db=2.5,
    ...     breath_presence=0.7,
    ...     valve_reduction=0.6,
    ...     resonance_db=1.5
    ... )
    >>>
    >>> processed, report = enhancer.process(audio, sr)
    >>> print(f"Brass character: {report['brass_character_db']:.1f} dB")

Author: AURIK Phase 2.3
Date: February 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# COMPONENT 1: BRASS HARMONICS ENHANCER
# =============================================================================


class BrassHarmonicsEnhancer:
    """
    Brass Harmonics Enhancement.

    Enhances characteristic brass timbre through harmonic enrichment.
    Critical for brass presence and tonal richness.

    Techniques:
    - Harmonic series enhancement (fundamental + overtones)
    - Brass formant emphasis (500-2000 Hz)
    - Brilliance enhancement (2-6 kHz)

    Parameters
    ----------
    harmonics_db : float
        Harmonic richness gain (0.0-4.0 dB)
    formant_emphasis : float
        Brass formant emphasis (0.0-1.0)
    brilliance_db : float
        High-frequency brilliance (0.0-3.0 dB)
    """

    def __init__(self, harmonics_db: float = 2.5, formant_emphasis: float = 0.7, brilliance_db: float = 2.0):
        self.harmonics_db = np.clip(harmonics_db, 0.0, 4.0)
        self.formant_emphasis = np.clip(formant_emphasis, 0.0, 1.0)
        self.brilliance_db = np.clip(brilliance_db, 0.0, 3.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance brass harmonics.

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
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Brass formant range (500-2000 Hz)
        nyquist = sr / 2
        low_f = min(500, nyquist * 0.25)
        high_f = min(2000, nyquist * 0.95)
        sos_formant = butter(4, [low_f, high_f], btype="band", fs=sr, output="sos")
        formant_band = sosfilt(sos_formant, audio)

        # Measure original energy
        formant_energy_orig = np.sqrt(np.mean(formant_band**2))

        # Enhance formant with harmonic exciter
        if self.formant_emphasis > 0:
            # Soft saturation generates harmonics
            formant_excited = np.tanh(formant_band * 2.0) * 0.5
            formant_enhanced = formant_band + formant_excited * self.formant_emphasis
        else:
            formant_enhanced = formant_band

        # Apply harmonic gain
        harmonics_gain = 10 ** (self.harmonics_db / 20.0)
        formant_enhanced = formant_enhanced * harmonics_gain

        # Brilliance enhancement (2-6 kHz)
        nyquist = sr / 2
        low_f = min(2000, nyquist * 0.25)
        high_f = min(6000, nyquist * 0.95)
        sos_brilliance = butter(4, [low_f, high_f], btype="band", fs=sr, output="sos")
        brilliance_band = sosfilt(sos_brilliance, audio)

        brilliance_gain = 10 ** (self.brilliance_db / 20.0)
        brilliance_enhanced = brilliance_band * brilliance_gain

        # Reconstruct audio
        nyquist = sr / 2
        low_cutoff = min(500, nyquist * 0.25)
        sos_low = butter(4, low_cutoff, btype="low", fs=sr, output="sos")
        mid_low_f = min(2000, nyquist * 0.25)
        mid_high_f = min(6000, nyquist * 0.95)
        sos_mid_high = butter(4, [mid_low_f, mid_high_f], btype="band", fs=sr, output="sos")
        high_cutoff = min(6000, nyquist * 0.95)
        sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        mid_high_content = sosfilt(sos_mid_high, audio)
        high_content = sosfilt(sos_high, audio)

        # Replace formant and brilliance
        result = low_content + formant_enhanced + (brilliance_enhanced - mid_high_content) + high_content

        # Measure new energy
        formant_energy_new = np.sqrt(np.mean(sosfilt(sos_formant, result) ** 2))
        formant_change_db = 20 * np.log10((formant_energy_new + 1e-10) / (formant_energy_orig + 1e-10))

        report = {
            "harmonics_energy_change_db": formant_change_db,
            "formant_emphasis_applied": self.formant_emphasis > 0,
            "brilliance_gain_db": self.brilliance_db,
        }

        return result, report


# =============================================================================
# COMPONENT 2: BREATH ATTACK PRESERVER
# =============================================================================


class BreathAttackPreserver:
    """
    Breath Attack Preservation.

    Preserves and enhances natural breath articulation.
    Critical for expressive phrasing and articulation clarity.

    Techniques:
    - Breath transient detection
    - Attack phase preservation
    - Airflow character enhancement

    Parameters
    ----------
    breath_presence : float
        Breath presence amount (0.0-1.0)
    attack_clarity : bool
        Enhance attack clarity
    preserve_airflow : bool
        Preserve natural airflow character
    """

    def __init__(self, breath_presence: float = 0.7, attack_clarity: bool = True, preserve_airflow: bool = True):
        self.breath_presence = np.clip(breath_presence, 0.0, 1.0)
        self.attack_clarity = attack_clarity
        self.preserve_airflow = preserve_airflow

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Preserve breath attacks.

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
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Breath attacks have high-frequency content (6-12 kHz)
        nyquist = sr / 2
        low_f = min(6000, nyquist * 0.25)
        high_f = min(12000, nyquist * 0.95)
        sos_breath = butter(4, [low_f, high_f], btype="band", fs=sr, output="sos")
        breath_band = sosfilt(sos_breath, audio)

        # Detect breath transients
        envelope = np.abs(np.asarray(hilbert(breath_band), dtype=np.complex128))

        # Find attack phases (steep rises)
        derivative = np.diff(envelope, prepend=envelope[0])
        attack_mask = derivative > np.percentile(derivative, 90)

        # Expand attack window
        attack_duration = int(0.02 * sr)  # 20ms
        attack_mask_expanded = np.convolve(
            attack_mask.astype(float), np.ones(attack_duration) / attack_duration, mode="same"
        )
        attack_mask_expanded = np.clip(attack_mask_expanded, 0, 1)

        # Enhance breath presence at attacks
        breath_boost = 1.0 + self.breath_presence * 0.3
        breath_enhanced = breath_band * (1 + attack_mask_expanded * (breath_boost - 1))

        # Attack clarity (optional)
        if self.attack_clarity:
            # Sharpen transients
            nyquist = sr / 2
            sharp_cutoff = min(8000, nyquist * 0.95)
            sos_sharp = butter(2, sharp_cutoff, btype="high", fs=sr, output="sos")
            sharp_breath = sosfilt(sos_sharp, breath_band)

            clarity_boost = sharp_breath * attack_mask_expanded * 0.2
            breath_enhanced = breath_enhanced + clarity_boost

        # Reconstruct audio
        nyquist = sr / 2
        high_cutoff = min(12000, nyquist * 0.95)
        sos_low = butter(4, 6000, btype="low", fs=sr, output="sos")
        sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + breath_enhanced + high_content

        # Count breath attacks
        num_attacks = np.sum(attack_mask)

        report = {
            "breath_attacks_detected": int(num_attacks),
            "attack_clarity_applied": self.attack_clarity,
            "airflow_preserved": self.preserve_airflow,
        }

        return result, report


# =============================================================================
# COMPONENT 3: VALVE CLICK REDUCER
# =============================================================================


class ValveClickReducer:
    """
    Valve Click Reduction.

    Reduces mechanical valve clicks while maintaining realism.
    Critical for clean brass recordings with fast passages.

    Techniques:
    - Valve click detection (1-4 kHz)
    - Artistic reduction (not complete removal)
    - Attack preservation

    Parameters
    ----------
    reduction : float
        Valve click reduction (0.0-1.0)
    preserve_attacks : bool
        Preserve natural note attacks
    maintain_realism : bool
        Maintain realistic valve presence
    """

    def __init__(self, reduction: float = 0.6, preserve_attacks: bool = True, maintain_realism: bool = True):
        self.reduction = np.clip(reduction, 0.0, 1.0)
        self.preserve_attacks = preserve_attacks
        self.maintain_realism = maintain_realism

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Reduce valve clicks.

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
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Valve clicks are typically 1-4 kHz
        sos_valve = butter(4, [1000, 4000], btype="band", fs=sr, output="sos")
        valve_band = sosfilt(sos_valve, audio)

        # Measure original energy
        valve_energy_orig = np.sqrt(np.mean(valve_band**2))

        # Detect valve clicks (short, sharp transients)
        envelope = np.abs(np.asarray(hilbert(valve_band), dtype=np.complex128))

        # Clicks have sharp onsets
        derivative = np.diff(envelope, prepend=envelope[0])
        click_mask = derivative > np.percentile(derivative, 95)

        # Clicks are brief (< 10ms)
        window_size = int(0.01 * sr)  # 10ms
        click_mask_expanded = np.convolve(click_mask.astype(float), np.ones(window_size) / window_size, mode="same")
        click_mask_expanded = np.clip(click_mask_expanded, 0, 1)

        # If preserve attacks, reduce click removal at musical transients
        if self.preserve_attacks:
            # Musical transients have sustained energy
            sustained_mask = envelope > np.percentile(envelope, 60)

            # Reduce click mask where sustained tones present
            click_mask_expanded = click_mask_expanded * (1 - sustained_mask.astype(float) * 0.6)

        # Apply reduction
        if self.maintain_realism:
            # Artistic reduction (50% max to maintain realism)
            max_reduction = 0.5
            effective_reduction = self.reduction * max_reduction
        else:
            effective_reduction = self.reduction

        reduction_factor = 1.0 - effective_reduction
        valve_reduced = valve_band * (1 - click_mask_expanded * (1 - reduction_factor))

        # Reconstruct audio
        sos_low = butter(4, 1000, btype="low", fs=sr, output="sos")
        sos_high = butter(4, 4000, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + valve_reduced + high_content

        # Measure reduction
        valve_energy_new = np.sqrt(np.mean(sosfilt(sos_valve, result) ** 2))
        reduction_db = 20 * np.log10((valve_energy_orig + 1e-10) / (valve_energy_new + 1e-10))

        report = {
            "valve_click_reduction_db": reduction_db,
            "attacks_preserved": self.preserve_attacks,
            "realism_maintained": self.maintain_realism,
        }

        return result, report


# =============================================================================
# COMPONENT 4: RESONANCE ENHANCER
# =============================================================================


class ResonanceEnhancer:
    """
    Brass Resonance Enhancement.

    Enhances natural brass resonance and acoustic character.
    Critical for warmth and authentic brass tone.

    Techniques:
    - Bell resonance enhancement (300-800 Hz)
    - Mouthpiece resonance (800-1500 Hz)
    - Overall warmth

    Parameters
    ----------
    resonance_db : float
        Brass resonance gain (0.0-3.0 dB)
    warmth : float
        Warmth enhancement (0.0-1.0)
    natural_character : bool
        Preserve natural brass character
    """

    def __init__(self, resonance_db: float = 1.5, warmth: float = 0.7, natural_character: bool = True):
        self.resonance_db = np.clip(resonance_db, 0.0, 3.0)
        self.warmth = np.clip(warmth, 0.0, 1.0)
        self.natural_character = natural_character

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance brass resonance.

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
            return np.stack([left, right], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Bell resonance (300-800 Hz)
        sos_bell = butter(4, [300, 800], btype="band", fs=sr, output="sos")
        bell_band = sosfilt(sos_bell, audio)

        # Measure original energy
        bell_energy_orig = np.sqrt(np.mean(bell_band**2))

        # Apply resonance gain
        resonance_gain = 10 ** (self.resonance_db / 20.0)
        bell_enhanced = bell_band * resonance_gain

        # Warmth enhancement (lower end of bell resonance)
        if self.warmth > 0:
            sos_warm = butter(2, [300, 500], btype="band", fs=sr, output="sos")
            warmth_band = sosfilt(sos_warm, bell_enhanced)

            # Extra warmth boost
            warmth_boosted = warmth_band * (1.0 + self.warmth * 0.3)

            # Mix back
            bell_enhanced = bell_enhanced + (warmth_boosted - warmth_band)

        # Reconstruct audio
        sos_low = butter(4, 300, btype="low", fs=sr, output="sos")
        sos_high = butter(4, 800, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + bell_enhanced + high_content

        # Measure new energy
        bell_energy_new = np.sqrt(np.mean(sosfilt(sos_bell, result) ** 2))
        resonance_change_db = 20 * np.log10((bell_energy_new + 1e-10) / (bell_energy_orig + 1e-10))

        report = {
            "resonance_change_db": resonance_change_db,
            "warmth_applied": self.warmth > 0,
            "natural_character_preserved": self.natural_character,
        }

        return result, report


# =============================================================================
# UNIFIED API: BRASS ENHANCEMENT SYSTEM
# =============================================================================


class BrassEnhancementSystem:
    """
    Einheitliche API für Blechbläser-/Blasinstrument-Verbesserung.

    Combines all brass processing components into a single pipeline:
    1. Brass Harmonics Enhancement
    2. Breath Attack Preservation
    3. Valve Click Reduction
    4. Resonance Enhancement

    Parameters
    ----------
    harmonics_db : float
        Harmonic richness (0.0-4.0 dB)
    breath_presence : float
        Breath presence (0.0-1.0)
    valve_reduction : float
        Valve click reduction (0.0-1.0)
    resonance_db : float
        Brass resonance (0.0-3.0 dB)
    """

    def __init__(
        self,
        harmonics_db: float = 2.5,
        breath_presence: float = 0.7,
        valve_reduction: float = 0.6,
        resonance_db: float = 1.5,
    ):
        self.harmonics_enhancer = BrassHarmonicsEnhancer(harmonics_db=harmonics_db)
        self.breath_preserver = BreathAttackPreserver(breath_presence=breath_presence)
        self.valve_reducer = ValveClickReducer(reduction=valve_reduction)
        self.resonance_enhancer = ResonanceEnhancer(resonance_db=resonance_db)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full brass enhancement pipeline.

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

        # Stage 1: Brass Harmonics Enhancement
        result, harmonics_report = self.harmonics_enhancer.process(result, sr)
        report["harmonics"] = harmonics_report

        # Stage 2: Breath Attack Preservation
        result, breath_report = self.breath_preserver.process(result, sr)
        report["breath"] = breath_report

        # Stage 3: Valve Click Reduction
        result, valve_report = self.valve_reducer.process(result, sr)
        report["valve"] = valve_report

        # Stage 4: Resonance Enhancement
        result, resonance_report = self.resonance_enhancer.process(result, sr)
        report["resonance"] = resonance_report

        # Calculate overall metrics
        report["stages_applied"] = 4  # type: ignore[assignment]
        report["brass_character_db"] = (
            harmonics_report["harmonics_energy_change_db"] + resonance_report["resonance_change_db"]
        ) / 2.0

        return result, report


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI interface for Brass Enhancement System."""
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Phase 2.3 - Brass/Wind Enhancement System")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    parser.add_argument("--harmonics", type=float, default=2.5, help="Harmonic richness in dB (0.0-4.0, default: 2.5)")
    parser.add_argument("--breath-presence", type=float, default=0.7, help="Breath presence (0.0-1.0, default: 0.7)")
    parser.add_argument(
        "--valve-reduction", type=float, default=0.6, help="Valve click reduction (0.0-1.0, default: 0.6)"
    )
    parser.add_argument("--resonance", type=float, default=1.5, help="Brass resonance in dB (0.0-3.0, default: 1.5)")

    args = parser.parse_args()

    # Load audio
    logger.info("lade: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Make mono for processing
    audio_mono = np.mean(audio, axis=1) if audio.shape[1] == 2 else audio[:, 0]

    # Create brass enhancement system
    logger.info("\n🎺 Brass/Wind Enhancement System")
    logger.info("=" * 60)

    enhancer = BrassEnhancementSystem(
        harmonics_db=args.harmonics,
        breath_presence=args.breath_presence,
        valve_reduction=args.valve_reduction,
        resonance_db=args.resonance,
    )

    # Process
    logger.info("Processing...")
    processed, report = enhancer.process(audio_mono, sr)

    # Print report
    logger.info("\n📊 Processing Report:")
    logger.info("-" * 60)
    logger.info("Harmonics: %.1f dB", report["harmonics"]["harmonics_energy_change_db"])
    logger.info("  Formant emphasis: %s", "Yes" if report["harmonics"]["formant_emphasis_applied"] else "No")

    logger.info("\nBreath Attacks: %s erkannt", report["breath"]["breath_attacks_detected"])
    logger.info("  Attack clarity: %s", "Yes" if report["breath"]["attack_clarity_applied"] else "No")

    logger.info("\nValve Clicks: %.1f dB", report["valve"]["valve_click_reduction_db"])
    logger.info("  Realism maintained: %s", "Yes" if report["valve"]["realism_maintained"] else "No")

    logger.info("\nResonance: %.1f dB", report["resonance"]["resonance_change_db"])
    logger.info("  Warmth angewendet: %s", "Yes" if report["resonance"]["warmth_applied"] else "No")

    logger.info("\nBrass Character: %.1f dB", report["brass_character_db"])
    logger.info("Stages angewendet: %s", report["stages_applied"])

    # Save
    logger.info("\nspeichere: %s", args.output)
    sf.write(args.output, processed, sr)
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
