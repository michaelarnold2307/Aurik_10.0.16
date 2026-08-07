#!/usr/bin/env python3
"""
Drums/Percussion Enhancement System
====================================

Professional drums and percussion processing for instrumental music restoration.
Addresses the gap between vocal processing (Phase 2.2) and instrumental music.

Components:
1. KickDrumEnhancer - 20-80 Hz sub-bass punch
2. SnareCrackEnhancer - 200-400 Hz + 1-3 kHz articulation
3. HiHatClarifier - 8-12 kHz clarity and presence
4. CymbalShimmerEnhancer - 12-20 kHz shimmer and air

Target: Bring Transparenz Musical Goal from 87% to 93% for percussive content

Usage:
    >>> from dsp.drums_enhancement import DrumsEnhancementSystem
    >>>
    >>> enhancer = DrumsEnhancementSystem(
    ...     kick_gain_db=3.0,
    ...     snare_articulation=0.8,
    ...     hihat_clarity_db=2.0,
    ...     cymbal_shimmer_db=1.5
    ... )
    >>>
    >>> processed, report = enhancer.process(audio, sr)
    >>> print(f"Transient enhancement: {report['transient_energy_change_db']:.1f} dB")

Author: AURIK Phase 2.3
Date: February 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfilt

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# COMPONENT 1: KICK DRUM ENHANCER
# =============================================================================


class KickDrumEnhancer:
    """
    Kick Drum Enhancement (20-80 Hz).

    Enhances kick drum sub-bass punch and attack.
    Critical for electronic music, hip-hop, and modern productions.

    Techniques:
    - Sub-bass boost (fundamental reinforcement)
    - Attack transient shaping
    - Phase-coherent enhancement
    - Kick detection and isolation

    Parameters
    ----------
    gain_db : float
        Kick drum gain in dB (0.0-6.0)
    attack_enhancement : float
        Attack transient enhancement (0.0-1.0)
    detection_sensitivity : float
        Kick detection sensitivity (0.0-1.0)
    """

    def __init__(self, gain_db: float = 3.0, attack_enhancement: float = 0.5, detection_sensitivity: float = 0.7):
        self.gain_db = np.clip(gain_db, 0.0, 6.0)
        self.attack_enhancement = np.clip(attack_enhancement, 0.0, 1.0)
        self.detection_sensitivity = np.clip(detection_sensitivity, 0.3, 1.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance kick drums.

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
        # §2.51 Linked-Stereo: gain envelope from √(L²+R²)/√2 sidechain
        if audio.ndim == 2:
            mono_sc = np.sqrt((audio[:, 0] ** 2 + audio[:, 1] ** 2) / 2.0)
            mono_enh, report_l = self._process_channel(mono_sc, sr)
            _eps = 1e-10
            _g = np.clip(np.where(np.abs(mono_sc) > _eps, mono_enh / (mono_sc + _eps), 1.0), 0.0, 10.0)
            return np.stack([audio[:, 0] * _g, audio[:, 1] * _g], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract kick range (20-80 Hz)
        sos_kick = butter(4, [20, 80], btype="band", fs=sr, output="sos")
        kick_band = sosfilt(sos_kick, audio)

        # Measure original energy
        original_energy = np.sqrt(np.mean(kick_band**2))

        # Detect kick drum transients
        kick_events = self._detect_kick_events(kick_band, sr)

        # Apply gain to kick band
        linear_gain = 10 ** (self.gain_db / 20.0)
        kick_enhanced = kick_band * linear_gain

        # Enhance attack transients
        if self.attack_enhancement > 0 and len(kick_events) > 0:
            attack_mask = np.zeros_like(kick_enhanced)

            # Create attack enhancement mask around kick events
            attack_duration = int(0.02 * sr)  # 20ms attack window

            for event_idx in kick_events:
                start = max(0, event_idx)
                end = min(len(attack_mask), event_idx + attack_duration)

                # Exponential decay envelope
                window_len = end - start
                envelope = np.exp(-5 * np.arange(window_len) / window_len)
                attack_mask[start:end] += envelope

            # Clip mask to [0, 1]
            attack_mask = np.clip(attack_mask, 0, 1)

            # Apply attack enhancement
            attack_boost = 1.0 + self.attack_enhancement
            kick_enhanced = kick_enhanced * (1 + attack_mask * (attack_boost - 1))

        # Reconstruct audio
        sos_high = butter(4, 80, btype="high", fs=sr, output="sos")
        high_content = sosfilt(sos_high, audio)

        result = kick_enhanced + high_content

        # Measure new energy
        new_energy = np.sqrt(np.mean(sosfilt(sos_kick, result) ** 2))
        energy_change_db = 20 * np.log10((new_energy + 1e-10) / (original_energy + 1e-10))

        report = {
            "kick_energy_change_db": energy_change_db,
            "kick_events_detected": len(kick_events),
            "attack_enhancement_applied": self.attack_enhancement > 0,
        }

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result, report

    def _detect_kick_events(self, kick_band: np.ndarray, sr: int) -> list[int]:
        """
        Erkennt kick drum events using envelope.

        Returns list of sample indices where kicks occur.
        """
        # Calculate envelope
        envelope = np.abs(np.asarray(hilbert(kick_band), dtype=np.complex128))

        # Smooth envelope
        window_size = int(0.01 * sr)  # 10ms smoothing
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Find peaks
        # Dynamic threshold based on sensitivity
        threshold = np.percentile(envelope_smooth, 100 * (1 - self.detection_sensitivity))

        # Minimum distance between kicks (avoid double detection)
        min_distance = int(0.2 * sr)  # 200ms minimum

        peaks, _ = find_peaks(envelope_smooth, height=threshold, distance=min_distance)

        return list(peaks)


# =============================================================================
# COMPONENT 2: SNARE CRACK ENHANCER
# =============================================================================


class SnareCrackEnhancer:
    """
    Snare Crack Enhancement (200-400 Hz + 1-3 kHz).

    Enhances snare drum articulation and crack.
    Critical for groove, rhythm clarity, and snare presence.

    Techniques:
    - Body resonance enhancement (200-400 Hz)
    - Crack/wire enhancement (1-3 kHz)
    - Transient shaping

    Parameters
    ----------
    articulation : float
        Snare articulation enhancement (0.0-1.0)
    body_gain_db : float
        Snare body gain (0.0-4.0 dB)
    crack_gain_db : float
        Snare crack/wire gain (0.0-4.0 dB)
    """

    def __init__(self, articulation: float = 0.8, body_gain_db: float = 2.0, crack_gain_db: float = 2.5):
        self.articulation = np.clip(articulation, 0.0, 1.0)
        self.body_gain_db = np.clip(body_gain_db, 0.0, 4.0)
        self.crack_gain_db = np.clip(crack_gain_db, 0.0, 4.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance snare articulation.

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
        # §2.51 Linked-Stereo: gain envelope from √(L²+R²)/√2 sidechain
        if audio.ndim == 2:
            mono_sc = np.sqrt((audio[:, 0] ** 2 + audio[:, 1] ** 2) / 2.0)
            mono_enh, report_l = self._process_channel(mono_sc, sr)
            _eps = 1e-10
            _g = np.clip(np.where(np.abs(mono_sc) > _eps, mono_enh / (mono_sc + _eps), 1.0), 0.0, 10.0)
            return np.stack([audio[:, 0] * _g, audio[:, 1] * _g], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract snare body (200-400 Hz)
        sos_body = butter(4, [200, 400], btype="band", fs=sr, output="sos")
        snare_body = sosfilt(sos_body, audio)

        # Extract snare crack/wire (1-3 kHz)
        sos_crack = butter(4, [1000, 3000], btype="band", fs=sr, output="sos")
        snare_crack = sosfilt(sos_crack, audio)

        # Measure original energies
        body_energy_orig = np.sqrt(np.mean(snare_body**2))
        crack_energy_orig = np.sqrt(np.mean(snare_crack**2))

        # Detect snare transients (use mid-range for detection)
        sos_detect = butter(4, [300, 2000], btype="band", fs=sr, output="sos")
        detect_band = sosfilt(sos_detect, audio)
        snare_events = self._detect_snare_events(detect_band, sr)

        # Enhance body
        body_gain = 10 ** (self.body_gain_db / 20.0)
        snare_body_enhanced = snare_body * body_gain

        # Enhance crack
        crack_gain = 10 ** (self.crack_gain_db / 20.0)
        snare_crack_enhanced = snare_crack * crack_gain

        # Apply articulation enhancement (transient shaping)
        if self.articulation > 0 and len(snare_events) > 0:
            attack_mask = np.zeros_like(audio)

            # Create attack window around detected transients
            attack_duration = int(0.015 * sr)  # 15ms attack

            for event_idx in snare_events:
                start = max(0, event_idx)
                end = min(len(attack_mask), event_idx + attack_duration)

                # Sharp attack envelope
                window_len = end - start
                envelope = np.exp(-7 * np.arange(window_len) / window_len)
                attack_mask[start:end] += envelope

            attack_mask = np.clip(attack_mask, 0, 1)

            # Boost transients
            transient_boost = 1.0 + self.articulation * 0.5
            snare_crack_enhanced = snare_crack_enhanced * (1 + attack_mask * (transient_boost - 1))

        # Reconstruct audio
        sos_low = butter(4, 200, btype="low", fs=sr, output="sos")
        sos_mid_low = butter(4, [400, 1000], btype="band", fs=sr, output="sos")
        sos_high = butter(4, 3000, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        mid_low_content = sosfilt(sos_mid_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + snare_body_enhanced + mid_low_content + snare_crack_enhanced + high_content

        # Measure new energies
        body_energy_new = np.sqrt(np.mean(sosfilt(sos_body, result) ** 2))
        crack_energy_new = np.sqrt(np.mean(sosfilt(sos_crack, result) ** 2))

        body_change_db = 20 * np.log10((body_energy_new + 1e-10) / (body_energy_orig + 1e-10))
        crack_change_db = 20 * np.log10((crack_energy_new + 1e-10) / (crack_energy_orig + 1e-10))

        report = {
            "snare_body_energy_change_db": body_change_db,
            "snare_crack_energy_change_db": crack_change_db,
            "snare_events_detected": len(snare_events),
            "articulation_applied": self.articulation > 0,
        }

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result, report

    def _detect_snare_events(self, detect_band: np.ndarray, sr: int) -> list[int]:
        """Erkennt snare drum events."""
        envelope = np.abs(np.asarray(hilbert(detect_band), dtype=np.complex128))

        # Smooth
        window_size = int(0.005 * sr)  # 5ms
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Find peaks
        threshold = np.percentile(envelope_smooth, 95)  # Top 5%
        min_distance = int(0.15 * sr)  # 150ms minimum

        peaks, _ = find_peaks(envelope_smooth, height=threshold, distance=min_distance)

        return list(peaks)


# =============================================================================
# COMPONENT 3: HI-HAT CLARIFIER
# =============================================================================


class HiHatClarifier:
    """
    Hi-Hat Clarity Enhancement (8-12 kHz).

    Enhances hi-hat clarity and presence.
    Critical for groove articulation and time-keeping precision.

    Techniques:
    - High-frequency enhancement (8-12 kHz)
    - Transient sharpening
    - Bleed reduction (isolate hi-hat from other cymbals)

    Parameters
    ----------
    clarity_db : float
        Hi-hat clarity gain (0.0-3.0 dB)
    transient_sharpness : float
        Transient sharpening amount (0.0-1.0)
    reduce_bleed : bool
        Reduce cymbal bleed
    """

    def __init__(self, clarity_db: float = 2.0, transient_sharpness: float = 0.6, reduce_bleed: bool = True):
        self.clarity_db = np.clip(clarity_db, 0.0, 3.0)
        self.transient_sharpness = np.clip(transient_sharpness, 0.0, 1.0)
        self.reduce_bleed = reduce_bleed

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance hi-hat clarity.

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
        # §2.51 Linked-Stereo: gain envelope from √(L²+R²)/√2 sidechain
        if audio.ndim == 2:
            mono_sc = np.sqrt((audio[:, 0] ** 2 + audio[:, 1] ** 2) / 2.0)
            mono_enh, report_l = self._process_channel(mono_sc, sr)
            _eps = 1e-10
            _g = np.clip(np.where(np.abs(mono_sc) > _eps, mono_enh / (mono_sc + _eps), 1.0), 0.0, 10.0)
            return np.stack([audio[:, 0] * _g, audio[:, 1] * _g], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract hi-hat range (8-12 kHz)
        nyquist = sr / 2
        hihat_low = min(8000, nyquist * 0.85)
        hihat_high = min(12000, nyquist * 0.95)
        sos_hihat = butter(4, [hihat_low, hihat_high], btype="band", fs=sr, output="sos")
        hihat_band = sosfilt(sos_hihat, audio)

        # Measure original energy
        original_energy = np.sqrt(np.mean(hihat_band**2))

        # Apply clarity gain
        clarity_gain = 10 ** (self.clarity_db / 20.0)
        hihat_enhanced = hihat_band * clarity_gain

        # Transient sharpening
        if self.transient_sharpness > 0:
            # Detect hi-hat hits
            envelope = np.abs(np.asarray(hilbert(hihat_band), dtype=np.complex128))

            # Find transients (steep rises in envelope)
            derivative = np.diff(envelope, prepend=envelope[0])
            transient_mask = derivative > np.percentile(derivative, 95)

            # Expand mask slightly for attack phase
            kernel_size = int(0.005 * sr)  # 5ms
            transient_mask_expanded = np.convolve(transient_mask.astype(float), np.ones(kernel_size), mode="same")
            transient_mask_expanded = np.clip(transient_mask_expanded, 0, 1)

            # Apply transient boost
            transient_boost = 1.0 + self.transient_sharpness * 0.3
            hihat_enhanced = hihat_enhanced * (1 + transient_mask_expanded * (transient_boost - 1))

        # Reduce cymbal bleed (optional)
        if self.reduce_bleed:
            # Cymbals are typically in 12-20 kHz range
            nyquist = sr / 2
            cymbal_low = min(12000, nyquist * 0.75)
            cymbal_high = min(20000, nyquist * 0.95)
            sos_cymbal = butter(4, [cymbal_low, cymbal_high], btype="band", fs=sr, output="sos")
            cymbal_band = sosfilt(sos_cymbal, audio)

            # Gentle reduction of cymbal content in hi-hat band
            # (hi-hats are sharper, cymbals are more sustained)
            envelope_hihat = np.abs(np.asarray(hilbert(hihat_band), dtype=np.complex128))
            envelope_cymbal = np.abs(np.asarray(hilbert(cymbal_band), dtype=np.complex128))

            # Where cymbal is louder, reduce hi-hat enhancement
            cymbal_ratio = envelope_cymbal / (envelope_hihat + 1e-10)
            reduction_mask = np.clip(cymbal_ratio, 0, 1) * 0.3  # Max 30% reduction

            hihat_enhanced = hihat_enhanced * (1 - reduction_mask)

        # Reconstruct audio
        nyquist = sr / 2
        low_cutoff = min(8000, nyquist * 0.85)
        high_cutoff = min(12000, nyquist * 0.95)
        sos_low = butter(4, low_cutoff, btype="low", fs=sr, output="sos")
        sos_high = butter(4, high_cutoff, btype="high", fs=sr, output="sos")

        low_content = sosfilt(sos_low, audio)
        high_content = sosfilt(sos_high, audio)

        result = low_content + hihat_enhanced + high_content

        # Measure new energy
        new_energy = np.sqrt(np.mean(sosfilt(sos_hihat, result) ** 2))
        energy_change_db = 20 * np.log10((new_energy + 1e-10) / (original_energy + 1e-10))

        report = {
            "hihat_energy_change_db": energy_change_db,
            "transient_sharpening_applied": self.transient_sharpness > 0,
            "bleed_reduction_applied": self.reduce_bleed,
        }

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result, report


# =============================================================================
# COMPONENT 4: CYMBAL SHIMMER ENHANCER
# =============================================================================


class CymbalShimmerEnhancer:
    """
    Cymbal Shimmer Enhancement (12-20 kHz).

    Enhances cymbal shimmer and air.
    Critical for spatial impression and high-frequency brilliance.

    Techniques:
    - Air band enhancement (12-20 kHz)
    - Shimmer synthesis (harmonic content)
    - Decay enhancement (sustain preservation)

    Parameters
    ----------
    shimmer_db : float
        Cymbal shimmer gain (0.0-3.0 dB)
    air_enhancement : bool
        Enhance air band (15-20 kHz)
    decay_preservation : float
        Cymbal decay preservation (0.0-1.0)
    """

    def __init__(self, shimmer_db: float = 1.5, air_enhancement: bool = True, decay_preservation: float = 0.7):
        self.shimmer_db = np.clip(shimmer_db, 0.0, 3.0)
        self.air_enhancement = air_enhancement
        self.decay_preservation = np.clip(decay_preservation, 0.0, 1.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance cymbal shimmer.

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
        # §2.51 Linked-Stereo: gain envelope from √(L²+R²)/√2 sidechain
        if audio.ndim == 2:
            mono_sc = np.sqrt((audio[:, 0] ** 2 + audio[:, 1] ** 2) / 2.0)
            mono_enh, report_l = self._process_channel(mono_sc, sr)
            _eps = 1e-10
            _g = np.clip(np.where(np.abs(mono_sc) > _eps, mono_enh / (mono_sc + _eps), 1.0), 0.0, 10.0)
            return np.stack([audio[:, 0] * _g, audio[:, 1] * _g], axis=-1), report_l
        else:
            return self._process_channel(audio, sr)

    def _process_channel(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Verarbeitet einen einzelnen Kanal."""
        # Extract cymbal range (12-20 kHz)
        nyquist = sr / 2
        cymbal_low = min(12000, nyquist * 0.75)
        cymbal_high = min(20000, nyquist * 0.95)
        sos_cymbal = butter(4, [cymbal_low, cymbal_high], btype="band", fs=sr, output="sos")
        cymbal_band = sosfilt(sos_cymbal, audio)

        # Measure original energy
        original_energy = np.sqrt(np.mean(cymbal_band**2))

        # Apply shimmer gain
        shimmer_gain = 10 ** (self.shimmer_db / 20.0)
        cymbal_enhanced = cymbal_band * shimmer_gain

        # Optional: Air band enhancement (15-20 kHz)
        if self.air_enhancement:
            nyquist = sr / 2
            air_low = min(15000, nyquist * 0.85)
            air_high = min(20000, nyquist * 0.95)
            sos_air = butter(2, [air_low, air_high], btype="band", fs=sr, output="sos")
            air_band = sosfilt(sos_air, audio)

            # Extra boost for air
            air_boosted = air_band * 1.3

            # Mix air back
            cymbal_enhanced = cymbal_enhanced + (air_boosted - air_band)

        # Decay preservation
        if self.decay_preservation > 0:
            # Calculate envelope
            envelope = np.abs(np.asarray(hilbert(cymbal_band), dtype=np.complex128))

            # Smooth envelope
            window_size = int(0.02 * sr)  # 20ms
            envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

            # Identify decay phases (falling envelope)
            derivative = np.diff(envelope_smooth, prepend=envelope_smooth[0])
            decay_mask = (derivative < 0).astype(float)

            # Smooth decay mask
            decay_mask_smooth = np.convolve(decay_mask, np.ones(window_size) / window_size, mode="same")

            # Apply decay boost
            decay_boost = 1.0 + self.decay_preservation * 0.2  # Max 20% boost
            cymbal_enhanced = cymbal_enhanced * (1 + decay_mask_smooth * (decay_boost - 1))

        # Reconstruct audio
        nyquist = sr / 2
        low_cutoff = min(12000, nyquist * 0.95)
        sos_low = butter(4, low_cutoff, btype="low", fs=sr, output="sos")
        low_content = sosfilt(sos_low, audio)

        result = low_content + cymbal_enhanced

        # Measure new energy
        new_energy = np.sqrt(np.mean(sosfilt(sos_cymbal, result) ** 2))
        energy_change_db = 20 * np.log10((new_energy + 1e-10) / (original_energy + 1e-10))

        report = {
            "cymbal_energy_change_db": energy_change_db,
            "air_enhancement_applied": self.air_enhancement,
            "decay_preservation_applied": self.decay_preservation > 0,
        }

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result, report


# =============================================================================
# UNIFIED API: DRUMS ENHANCEMENT SYSTEM
# =============================================================================


class DrumsEnhancementSystem:
    """
    Einheitliche API für Schlagzeug-/Percussion-Verbesserung.

    Combines all drum processing components into a single pipeline:
    1. Kick Drum Enhancement (20-80 Hz)
    2. Snare Crack Enhancement (200-400 Hz + 1-3 kHz)
    3. Hi-Hat Clarification (8-12 kHz)
    4. Cymbal Shimmer Enhancement (12-20 kHz)

    Parameters
    ----------
    kick_gain_db : float
        Kick drum gain (0.0-6.0 dB)
    snare_articulation : float
        Snare articulation (0.0-1.0)
    hihat_clarity_db : float
        Hi-hat clarity gain (0.0-3.0 dB)
    cymbal_shimmer_db : float
        Cymbal shimmer gain (0.0-3.0 dB)
    """

    def __init__(
        self,
        kick_gain_db: float = 3.0,
        snare_articulation: float = 0.8,
        hihat_clarity_db: float = 2.0,
        cymbal_shimmer_db: float = 1.5,
    ):
        self.kick_enhancer = KickDrumEnhancer(gain_db=kick_gain_db)
        self.snare_enhancer = SnareCrackEnhancer(articulation=snare_articulation)
        self.hihat_clarifier = HiHatClarifier(clarity_db=hihat_clarity_db)
        self.cymbal_enhancer = CymbalShimmerEnhancer(shimmer_db=cymbal_shimmer_db)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full drums enhancement pipeline.

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

        # Stage 1: Kick Drum Enhancement
        result, kick_report = self.kick_enhancer.process(result, sr)
        report["kick"] = kick_report

        # Stage 2: Snare Crack Enhancement
        result, snare_report = self.snare_enhancer.process(result, sr)
        report["snare"] = snare_report

        # Stage 3: Hi-Hat Clarification
        result, hihat_report = self.hihat_clarifier.process(result, sr)
        report["hihat"] = hihat_report

        # Stage 4: Cymbal Shimmer Enhancement
        result, cymbal_report = self.cymbal_enhancer.process(result, sr)
        report["cymbal"] = cymbal_report

        # Calculate overall metrics
        report["stages_applied"] = 4  # type: ignore[assignment]
        report["transient_energy_change_db"] = (
            kick_report["kick_energy_change_db"] + snare_report["snare_crack_energy_change_db"]
        ) / 2.0

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result, report


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI interface for Drums Enhancement System."""
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Phase 2.3 - Drums/Percussion Enhancement System")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    parser.add_argument("--kick-gain", type=float, default=3.0, help="Kick drum gain in dB (0.0-6.0, default: 3.0)")
    parser.add_argument(
        "--snare-articulation", type=float, default=0.8, help="Snare articulation (0.0-1.0, default: 0.8)"
    )
    parser.add_argument("--hihat-clarity", type=float, default=2.0, help="Hi-hat clarity in dB (0.0-3.0, default: 2.0)")
    parser.add_argument(
        "--cymbal-shimmer", type=float, default=1.5, help="Cymbal shimmer in dB (0.0-3.0, default: 1.5)"
    )

    args = parser.parse_args()

    # Load audio
    logger.info("lade: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Make mono for processing
    audio_mono = np.mean(audio, axis=1) if audio.shape[1] == 2 else audio[:, 0]

    # Create drums enhancement system
    logger.info("\n🥁 Drums/Percussion Enhancement System")
    logger.info("=" * 60)

    enhancer = DrumsEnhancementSystem(
        kick_gain_db=args.kick_gain,
        snare_articulation=args.snare_articulation,
        hihat_clarity_db=args.hihat_clarity,
        cymbal_shimmer_db=args.cymbal_shimmer,
    )

    # Process
    logger.info("Processing...")
    processed, report = enhancer.process(audio_mono, sr)

    # Print report
    logger.info("\n📊 Processing Report:")
    logger.info("-" * 60)
    logger.info("Kick Drum: %.1f dB", report["kick"]["kick_energy_change_db"])
    logger.info("  Events erkannt: %s", report["kick"]["kick_events_detected"])

    logger.info("\nSnare Crack: %.1f dB", report["snare"]["snare_crack_energy_change_db"])
    logger.info("  Body: %.1f dB", report["snare"]["snare_body_energy_change_db"])
    logger.info("  Events erkannt: %s", report["snare"]["snare_events_detected"])

    logger.info("\nHi-Hat: %.1f dB", report["hihat"]["hihat_energy_change_db"])
    logger.info("  Transient sharpening: %s", "Yes" if report["hihat"]["transient_sharpening_applied"] else "No")

    logger.info("\nCymbal: %.1f dB", report["cymbal"]["cymbal_energy_change_db"])
    logger.info("  Air enhancement: %s", "Yes" if report["cymbal"]["air_enhancement_applied"] else "No")

    logger.info("\nTransient Enhancement: %.1f dB", report["transient_energy_change_db"])
    logger.info("Stages angewendet: %s", report["stages_applied"])

    # Save
    logger.info("\nspeichere: %s", args.output)
    sf.write(args.output, processed, sr)
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
