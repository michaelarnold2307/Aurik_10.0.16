#!/usr/bin/env python3
"""
Piano/Keys Restoration System
==============================

Professional piano and keyboard instrument restoration for instrumental music.
Addresses the gap between vocal processing (Phase 2.2) and instrumental music.

Components:
1. HammerNoiseReducer - Mechanical hammer artifact reduction
2. PedalNoiseReducer - Damper/sustain pedal noise reduction
3. KeyClickReducer - Key mechanism click reduction
4. TonalBalanceEnhancer - Bass/treble register balance

Target: Bring Natürlichkeit Musical Goal from 89% to 95% for piano/keys content

Usage:
    >>> from dsp.piano_restoration import PianoRestorationSystem
    >>>
    >>> restorer = PianoRestorationSystem(
    ...     hammer_noise_reduction=0.7,
    ...     pedal_noise_reduction=0.8,
    ...     key_click_reduction=0.6,
    ...     tonal_balance=0.5
    ... )
    >>>
    >>> processed, report = restorer.process(audio, sr)
    >>> print(f"Mechanical noise reduced: {report['mechanical_noise_db']:.1f} dB")

Author: AURIK Phase 2.3
Date: February 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

warnings.filterwarnings("ignore", category=RuntimeWarning)

_logger = logging.getLogger(__name__)


def _match_lengths(*arrays):
    """Stellt sicher: all arrays have the same length (trim to minimum)."""
    min_len = min(len(arr) for arr in arrays)
    return tuple(arr[:min_len] for arr in arrays)


# =============================================================================
# COMPONENT 1: HAMMER NOISE REDUCER
# =============================================================================


class HammerNoiseReducer:
    """
    Hammer Noise Reduction.

    Reduces mechanical noise from piano hammers striking strings.
    Critical for clean piano tone in historical recordings.

    Techniques:
    - Transient discrimination (musical vs. mechanical)
    - Attack phase preservation
    - Hammer thump reduction (20-100 Hz)

    Parameters
    ----------
    reduction : float
        Hammer noise reduction amount (0.0-1.0)
    preserve_attack : bool
        Preserve natural attack transients
    thump_reduction : bool
        Reduce low-frequency hammer thump
    """

    def __init__(self, reduction: float = 0.7, preserve_attack: bool = True, thump_reduction: bool = True):
        self.reduction = np.clip(reduction, 0.0, 1.0)
        self.preserve_attack = preserve_attack
        self.thump_reduction = thump_reduction

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für HammerNoiseReducer")
            if audio.ndim == 2:
                left, report_l = self._process_channel(audio[:, 0], sr)
                right, report_r = self._process_channel(audio[:, 1], sr)
                left, right = _match_lengths(left, right)
                result = np.stack([left, right], axis=-1)
                report = report_l
            else:
                result, report = self._process_channel(audio, sr)
            self._audit_log(report, sr)
            return result.astype(orig_dtype), report
        except Exception as e:
            _logger.error("HammerNoiseReducer process error: %s", e)
            report = {"hammer_noise_reduction_db": 0.0, "error": str(e)}
            self._audit_log(report, sr if "sr" in locals() else None)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return audio.astype(orig_dtype), report

    def _audit_log(self, report, sr=None):
        _logger.debug("HammerNoiseReducer audit: %s sr=%s", report, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        result = audio.copy()
        noise_reduced_db = 0.0

        # Hammer thump reduction (low-frequency mechanical noise)
        if self.thump_reduction:
            # Hammer thumps are often 20-100 Hz
            sos_thump = butter(4, [20, 100], btype="band", fs=sr, output="sos")
            thump_band = sosfilt(sos_thump, audio)

            # Measure original energy
            thump_energy_orig = np.sqrt(np.mean(thump_band**2))

            # Detect transients (musical notes have longer sustain)
            envelope = np.abs(np.asarray(hilbert(thump_band), dtype=np.complex128))

            # Sharp transients are likely hammer noise
            derivative = np.diff(envelope, prepend=envelope[0])
            transient_mask = derivative > np.percentile(derivative, 95)

            # Expand mask slightly
            window_size = int(0.005 * sr)  # 5ms
            transient_mask_expanded = np.convolve(
                transient_mask.astype(float), np.ones(window_size) / window_size, mode="same"
            )

            # Reduce thump at transients
            reduction_factor = 1.0 - self.reduction * 0.7  # Max 70% reduction
            thump_reduced = thump_band * (1 - transient_mask_expanded * (1 - reduction_factor))

            # Reconstruct
            sos_low = butter(4, 20, btype="low", fs=sr, output="sos")
            sos_high = butter(4, 100, btype="high", fs=sr, output="sos")

            low_content = sosfilt(sos_low, result)
            high_content = sosfilt(sos_high, result)

            result = low_content + thump_reduced + high_content

            # Measure reduction
            thump_energy_new = np.sqrt(np.mean(sosfilt(sos_thump, result) ** 2))
            noise_reduced_db = 20 * np.log10((thump_energy_orig + 1e-10) / (thump_energy_new + 1e-10))

        # High-frequency hammer click reduction (preserve attack)
        if self.preserve_attack:
            # Hammer clicks are 3-8 kHz range
            nyquist = sr / 2
            click_low = min(3000, nyquist * 0.6)
            click_high = min(8000, nyquist * 0.95)
            sos_click = butter(4, [click_low, click_high], btype="band", fs=sr, output="sos")
            click_band = sosfilt(sos_click, result)

            # Musical transients have more harmonic content
            # Mechanical clicks have noisier spectrum (higher ZCR)
            zero_crossings = np.diff(np.sign(click_band)) != 0
            zcr = np.convolve(zero_crossings.astype(float), np.ones(int(0.01 * sr)) / (0.01 * sr), mode="same")

            # High ZCR = mechanical click
            click_mask = (zcr > np.percentile(zcr, 75)).astype(float)

            # Smooth mask
            window_size = int(0.005 * sr)
            click_mask_smooth = np.convolve(click_mask, np.ones(window_size) / window_size, mode="same")

            # Ensure click_band and click_mask_smooth have same length
            click_band, click_mask_smooth = _match_lengths(click_band, click_mask_smooth)

            # Reduce clicks
            reduction_factor = 1.0 - self.reduction * 0.5  # Max 50% reduction
            click_reduced = click_band * (1 - click_mask_smooth * (1 - reduction_factor))

            # Reconstruct
            nyquist = sr / 2
            low_cutoff = min(3000, nyquist * 0.6)
            high_cutoff = min(8000, nyquist * 0.95)
            sos_low = butter(4, low_cutoff, btype="low", fs=sr, output="sos")
            sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

            low_content = sosfilt(sos_low, result)
            high_content = sosfilt(sos_high, result)

            # Ensure all components have same length
            low_content, click_reduced, high_content = _match_lengths(low_content, click_reduced, high_content)
            result = low_content + click_reduced + high_content

        report = {
            "hammer_noise_reduction_db": noise_reduced_db,
            "attack_preserved": self.preserve_attack,
            "thump_reduction_applied": self.thump_reduction,
        }

        return result, report


# =============================================================================
# COMPONENT 2: PEDAL NOISE REDUCER
# =============================================================================


class PedalNoiseReducer:
    """
    Pedal Noise Reduction.

    Reduces damper/sustain pedal mechanical noise.
    Critical for clean piano recordings with pedal usage.

    Techniques:
    - Low-frequency pedal thump detection
    - Damper resonance reduction
    - Pedal timing preservation

    Parameters
    ----------
    reduction : float
        Pedal noise reduction amount (0.0-1.0)
    damper_resonance_reduction : bool
        Reduce damper resonance artifacts
    preserve_sustain_character : bool
        Preserve natural sustain pedal character
    """

    def __init__(
        self, reduction: float = 0.8, damper_resonance_reduction: bool = True, preserve_sustain_character: bool = True
    ):
        self.reduction = np.clip(reduction, 0.0, 1.0)
        self.damper_resonance_reduction = damper_resonance_reduction
        self.preserve_sustain_character = preserve_sustain_character

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für PedalNoiseReducer")
            if audio.ndim == 2:
                left, report_l = self._process_channel(audio[:, 0], sr)
                right, report_r = self._process_channel(audio[:, 1], sr)
                left, right = _match_lengths(left, right)
                result = np.stack([left, right], axis=-1)
                report = report_l
            else:
                result, report = self._process_channel(audio, sr)
            self._audit_log(report, sr)
            return result.astype(orig_dtype), report
        except Exception as e:
            _logger.error("PedalNoiseReducer process error: %s", e)
            report = {"pedal_noise_reduction_db": 0.0, "error": str(e)}
            self._audit_log(report, sr if "sr" in locals() else None)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return audio.astype(orig_dtype), report

    def _audit_log(self, report, sr=None):
        _logger.debug("PedalNoiseReducer audit: %s sr=%s", report, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Pedal noise is typically low-frequency (10-80 Hz)
        sos_pedal = butter(4, [10, 80], btype="band", fs=sr, output="sos")
        pedal_band = sosfilt(sos_pedal, audio)

        # Measure original energy
        pedal_energy_orig = np.sqrt(np.mean(pedal_band**2))

        # Detect pedal events (sharp rises in low-frequency envelope)
        envelope = np.abs(np.asarray(hilbert(pedal_band), dtype=np.complex128))

        # Pedal events are sudden transients
        derivative = np.diff(envelope, prepend=envelope[0])
        pedal_events = derivative > np.percentile(derivative, 97)  # Top 3%

        # Expand event window
        event_duration = int(0.1 * sr)  # 100ms around pedal event
        pedal_mask = np.convolve(pedal_events.astype(float), np.ones(event_duration) / event_duration, mode="same")
        pedal_mask = np.clip(pedal_mask, 0, 1)

        # Reduce pedal band during events
        reduction_factor = 1.0 - self.reduction
        pedal_reduced = pedal_band * (1 - pedal_mask * (1 - reduction_factor))

        # Damper resonance reduction (optional)
        if self.damper_resonance_reduction:
            # Dampers can cause resonance in 150-300 Hz range
            sos_damper = butter(4, [150, 300], btype="band", fs=sr, output="sos")
            damper_band = sosfilt(sos_damper, audio)

            # Reduce resonance near pedal events
            damper_reduced = damper_band * (1 - pedal_mask * 0.3)  # 30% reduction

            # Reconstruct with damper reduction
            sos_low = butter(4, 150, btype="low", fs=sr, output="sos")
            sos_high = butter(4, 300, btype="high", fs=sr, output="sos")

            low_content = sosfilt(sos_low, audio)
            high_content = sosfilt(sos_high, audio)

            audio = low_content + damper_reduced + high_content

        # Reconstruct audio
        sos_low = butter(4, 10, btype="low", fs=sr, output="sos")
        sos_high = butter(4, 80, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + pedal_reduced + high_content

        # Measure reduction
        pedal_energy_new = np.sqrt(np.mean(sosfilt(sos_pedal, result) ** 2))
        reduction_db = 20 * np.log10((pedal_energy_orig + 1e-10) / (pedal_energy_new + 1e-10))

        # Count pedal events
        num_events = np.sum(pedal_events)

        report = {
            "pedal_noise_reduction_db": reduction_db,
            "pedal_events_detected": int(num_events),
            "damper_resonance_reduced": self.damper_resonance_reduction,
        }

        return result, report


# =============================================================================
# COMPONENT 3: KEY CLICK REDUCER
# =============================================================================


class KeyClickReducer:
    """
    Key Click Reduction.

    Reduces key mechanism clicks while maintaining attack.
    Critical for clean piano recordings, especially close-mic'd.

    Techniques:
    - Transient analysis (musical vs. mechanical)
    - Attack preservation
    - Click frequency attenuation (4-10 kHz)

    Parameters
    ----------
    reduction : float
        Key click reduction amount (0.0-1.0)
    preserve_attack : bool
        Preserve natural note attack
    aggressive_mode : bool
        More aggressive click removal
    """

    def __init__(self, reduction: float = 0.6, preserve_attack: bool = True, aggressive_mode: bool = False):
        self.reduction = np.clip(reduction, 0.0, 1.0)
        self.preserve_attack = preserve_attack
        self.aggressive_mode = aggressive_mode

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für KeyClickReducer")
            if audio.ndim == 2:
                left, report_l = self._process_channel(audio[:, 0], sr)
                right, report_r = self._process_channel(audio[:, 1], sr)
                left, right = _match_lengths(left, right)
                result = np.stack([left, right], axis=-1)
                report = report_l
            else:
                result, report = self._process_channel(audio, sr)
            self._audit_log(report, sr)
            return result.astype(orig_dtype), report
        except Exception as e:
            _logger.error("KeyClickReducer process error: %s", e)
            report = {"key_click_reduction_db": 0.0, "error": str(e)}
            self._audit_log(report, sr if "sr" in locals() else None)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return audio.astype(orig_dtype), report

    def _audit_log(self, report, sr=None):
        _logger.debug("KeyClickReducer audit: %s sr=%s", report, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Key clicks are high-frequency (4-10 kHz)
        nyquist = sr / 2
        click_low = min(4000, nyquist * 0.7)
        click_high = min(10000, nyquist * 0.95)
        sos_click = butter(4, [click_low, click_high], btype="band", fs=sr, output="sos")
        click_band = sosfilt(sos_click, audio)

        # Measure original energy
        click_energy_orig = np.sqrt(np.mean(click_band**2))

        # Detect clicks using zero-crossing rate
        zero_crossings = np.diff(np.sign(click_band)) != 0
        zcr = np.convolve(zero_crossings.astype(float), np.ones(int(0.005 * sr)) / (0.005 * sr), mode="same")

        # High ZCR indicates clicks (noisy transients)
        if self.aggressive_mode:
            click_threshold = np.percentile(zcr, 60)  # Top 40%
        else:
            click_threshold = np.percentile(zcr, 80)  # Top 20%

        click_mask = (zcr > click_threshold).astype(float)

        # Smooth mask
        window_size = int(0.003 * sr)  # 3ms
        click_mask_smooth = np.convolve(click_mask, np.ones(window_size) / window_size, mode="same")

        # If preserve attack, only reduce during non-musical transients
        if self.preserve_attack:
            # Musical transients have energy across wider frequency range
            sos_musical = butter(4, [1000, 4000], btype="band", fs=sr, output="sos")
            musical_band = sosfilt(sos_musical, audio)
            musical_envelope = np.abs(np.asarray(hilbert(musical_band), dtype=np.complex128))

            # Where musical content is strong, reduce click removal
            musical_threshold = np.percentile(musical_envelope, 70)
            musical_mask = (musical_envelope > musical_threshold).astype(float)

            # Smooth musical mask
            musical_mask_smooth = np.convolve(musical_mask, np.ones(int(0.01 * sr)) / (0.01 * sr), mode="same")

            # Ensure masks have same length
            click_mask_smooth, musical_mask_smooth = _match_lengths(click_mask_smooth, musical_mask_smooth)

            # Reduce click mask where musical content present
            click_mask_smooth = click_mask_smooth * (1 - musical_mask_smooth * 0.7)

        # Apply reduction
        reduction_factor = 1.0 - self.reduction
        # Ensure click_band and click_mask_smooth have same length
        click_band, click_mask_smooth = _match_lengths(click_band, click_mask_smooth)
        click_reduced = click_band * (1 - click_mask_smooth * (1 - reduction_factor))

        # Reconstruct audio
        nyquist = sr / 2
        low_cutoff = min(4000, nyquist * 0.7)
        high_cutoff = min(10000, nyquist * 0.95)
        sos_low = butter(4, low_cutoff, btype="low", fs=sr, output="sos")
        sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        # Ensure all components have same length
        low_content, click_reduced, high_content = _match_lengths(low_content, click_reduced, high_content)
        result = low_content + click_reduced + high_content

        # Measure reduction
        click_energy_new = np.sqrt(np.mean(sosfilt(sos_click, result) ** 2))
        reduction_db = 20 * np.log10((click_energy_orig + 1e-10) / (click_energy_new + 1e-10))
        reduction_percent = (1 - click_energy_new / (click_energy_orig + 1e-10)) * 100

        report = {
            "key_click_reduction_db": reduction_db,
            "key_click_reduction_percent": np.clip(reduction_percent, 0, 100),
            "attack_preserved": self.preserve_attack,
            "aggressive_mode": self.aggressive_mode,
        }

        return result, report


# =============================================================================
# COMPONENT 4: TONAL BALANCE ENHANCER
# =============================================================================


class TonalBalanceEnhancer:
    """
    Tonal Balance Enhancement.

    Balances bass and treble registers for natural piano sound.
    Critical for consistent tonal quality across keyboard.

    Techniques:
    - Bass register clarity (27-250 Hz, A0-B3)
    - Mid register warmth (250-1000 Hz, C4-B5)
    - Treble register brilliance (1-8 kHz, C6-C8)

    Parameters
    ----------
    balance : float
        Overall tonal balance (-1.0 to 1.0, negative = darker, positive = brighter)
    bass_clarity_db : float
        Bass register adjustment (-3.0 to 3.0 dB)
    treble_brilliance_db : float
        Treble register adjustment (-3.0 to 3.0 dB)
    """

    def __init__(self, balance: float = 0.0, bass_clarity_db: float = 0.0, treble_brilliance_db: float = 0.0):
        self.balance = np.clip(balance, -1.0, 1.0)
        self.bass_clarity_db = np.clip(bass_clarity_db, -3.0, 3.0)
        self.treble_brilliance_db = np.clip(treble_brilliance_db, -3.0, 3.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für TonalBalanceEnhancer")
            if audio.ndim == 2:
                left, report_l = self._process_channel(audio[:, 0], sr)
                right, report_r = self._process_channel(audio[:, 1], sr)
                left, right = _match_lengths(left, right)
                result = np.stack([left, right], axis=-1)
                report = report_l
            else:
                result, report = self._process_channel(audio, sr)
            self._audit_log(report, sr)
            return result.astype(orig_dtype), report
        except Exception as e:
            _logger.error("TonalBalanceEnhancer process error: %s", e)
            report = {"tonal_balance_db": 0.0, "error": str(e)}
            self._audit_log(report, sr if "sr" in locals() else None)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return audio.astype(orig_dtype), report

    def _audit_log(self, report, sr=None):
        _logger.debug("TonalBalanceEnhancer audit: %s sr=%s", report, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Bass register (27-250 Hz, A0-B3)
        sos_bass = butter(4, [27, 250], btype="band", fs=sr, output="sos")
        bass_band = sosfilt(sos_bass, audio)

        # Mid register (250-1000 Hz, C4-B5)
        sos_mid = butter(4, [250, 1000], btype="band", fs=sr, output="sos")
        mid_band = sosfilt(sos_mid, audio)

        # Treble register (1-8 kHz, C6-C8)
        nyquist = sr / 2
        treble_low = min(1000, nyquist * 0.3)
        treble_high = min(8000, nyquist * 0.95)
        sos_treble = butter(4, [treble_low, treble_high], btype="band", fs=sr, output="sos")
        treble_band = sosfilt(sos_treble, audio)

        # Apply balance adjustments
        # Overall balance affects bass vs. treble
        effective_bass_db = self.bass_clarity_db - self.balance
        effective_treble_db = self.treble_brilliance_db + self.balance

        bass_gain = 10 ** (effective_bass_db / 20.0)
        treble_gain = 10 ** (effective_treble_db / 20.0)

        bass_balanced = bass_band * bass_gain
        treble_balanced = treble_band * treble_gain

        # Reconstruct audio
        nyquist = sr / 2
        high_cutoff = min(8000, nyquist * 0.95)
        sos_low = butter(4, 27, btype="low", fs=sr, output="sos")
        sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + bass_balanced + mid_band + treble_balanced + high_content

        report = {
            "bass_adjustment_db": effective_bass_db,
            "treble_adjustment_db": effective_treble_db,
            "overall_balance": self.balance,
        }

        return result, report


# =============================================================================
# UNIFIED API: PIANO RESTORATION SYSTEM
# =============================================================================


class PianoRestorationSystem:
    """
    Einheitliche API für Klavier/Tasten-Restaurierung.

    Combines all piano processing components into a single pipeline:
    1. Hammer Noise Reduction
    2. Pedal Noise Reduction
    3. Key Click Reduction
    4. Tonal Balance Enhancement

    Parameters
    ----------
    hammer_noise_reduction : float
        Hammer noise reduction (0.0-1.0)
    pedal_noise_reduction : float
        Pedal noise reduction (0.0-1.0)
    key_click_reduction : float
        Key click reduction (0.0-1.0)
    tonal_balance : float
        Tonal balance (-1.0 to 1.0, negative = darker, positive = brighter)
    """

    def __init__(
        self,
        hammer_noise_reduction: float = 0.7,
        pedal_noise_reduction: float = 0.8,
        key_click_reduction: float = 0.6,
        tonal_balance: float = 0.0,
    ):
        self.hammer_reducer = HammerNoiseReducer(reduction=hammer_noise_reduction)
        self.pedal_reducer = PedalNoiseReducer(reduction=pedal_noise_reduction)
        self.click_reducer = KeyClickReducer(reduction=key_click_reduction)
        self.tonal_enhancer = TonalBalanceEnhancer(balance=tonal_balance)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full piano restoration pipeline with quality gate and audit logging.
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für PianoRestorationSystem")
            result, report = self._process(audio, sr)
            self._audit_log(report, sr)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result.astype(orig_dtype), report
        except Exception as e:
            _logger.error("PianoRestorationSystem process error: %s", e)
            report = {"piano_restoration_error": str(e)}
            self._audit_log(report, sr if "sr" in locals() else None)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            return audio.astype(orig_dtype), report

    def _process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full piano restoration pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Restored audio
        report : dict
            Comprehensive processing report
        """
        result = audio.copy()
        report = {}

        # Stage 1: Hammer Noise Reduction
        result, hammer_report = self.hammer_reducer.process(result, sr)
        report["hammer"] = hammer_report

        # Stage 2: Pedal Noise Reduction
        result, pedal_report = self.pedal_reducer.process(result, sr)
        report["pedal"] = pedal_report

        # Stage 3: Key Click Reduction
        result, click_report = self.click_reducer.process(result, sr)
        report["key_click"] = click_report

        # Stage 4: Tonal Balance Enhancement
        result, tonal_report = self.tonal_enhancer.process(result, sr)
        report["tonal_balance"] = tonal_report

        # Calculate overall metrics
        report["stages_applied"] = 4  # type: ignore[assignment]
        report["mechanical_noise_db"] = (
            hammer_report.get("hammer_noise_reduction_db", 0.0)
            + pedal_report.get("pedal_noise_reduction_db", 0.0)
            + click_report.get("key_click_reduction_db", 0.0)
        ) / 3.0

        return result, report

    def _audit_log(self, report, sr=None):
        _logger.debug("PianoRestorationSystem audit: %s sr=%s", report, sr)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI interface for Piano Restoration System."""
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Phase 2.3 - Piano/Keys Restoration System")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    parser.add_argument(
        "--hammer-reduction", type=float, default=0.7, help="Hammer noise reduction (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--pedal-reduction", type=float, default=0.8, help="Pedal noise reduction (0.0-1.0, default: 0.8)"
    )
    parser.add_argument(
        "--key-click-reduction", type=float, default=0.6, help="Key click reduction (0.0-1.0, default: 0.6)"
    )
    parser.add_argument("--tonal-balance", type=float, default=0.0, help="Tonal balance (-1.0 to 1.0, default: 0.0)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Load audio
    _logger.info("Loading: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Make mono for processing
    audio_mono = np.mean(audio, axis=1) if audio.shape[1] == 2 else audio[:, 0]

    # Create piano restoration system
    _logger.info("Piano/Keys Restoration System")

    restorer = PianoRestorationSystem(
        hammer_noise_reduction=args.hammer_reduction,
        pedal_noise_reduction=args.pedal_reduction,
        key_click_reduction=args.key_click_reduction,
        tonal_balance=args.tonal_balance,
    )

    # Process
    _logger.info("Processing...")
    processed, report = restorer.process(audio_mono, sr)

    # Log report
    _logger.info(
        "Hammer Noise: %+.1f dB | Attack preserved: %s",
        report["hammer"].get("hammer_noise_reduction_db", 0.0),
        report["hammer"].get("attack_preserved", False),
    )
    _logger.info(
        "Pedal Noise: %+.1f dB | Events detected: %s",
        report["pedal"].get("pedal_noise_reduction_db", 0.0),
        report["pedal"].get("pedal_events_detected", 0),
    )
    _logger.info(
        "Key Click: %+.1f dB | Reduction: %.1f%%",
        report["key_click"].get("key_click_reduction_db", 0.0),
        report["key_click"].get("key_click_reduction_percent", 0.0),
    )
    _logger.info(
        "Tonal Balance: %+.2f | Bass: %+.1f dB | Treble: %+.1f dB",
        report["tonal_balance"].get("overall_balance", 0.0),
        report["tonal_balance"].get("bass_adjustment_db", 0.0),
        report["tonal_balance"].get("treble_adjustment_db", 0.0),
    )
    _logger.info(
        "Mechanical Noise Reduction: %+.1f dB | Stages applied: %s",
        report.get("mechanical_noise_db", 0.0),
        report.get("stages_applied", 0),
    )

    # Save
    _logger.info("Saving: %s", args.output)
    sf.write(args.output, processed, sr)
    _logger.info("Done.")


if __name__ == "__main__":
    main()
