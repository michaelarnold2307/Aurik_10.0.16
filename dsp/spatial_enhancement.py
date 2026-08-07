#!/usr/bin/env python3
"""
Spatial Enhancement System
===========================

Professional spatial and ambience processing for instrumental music restoration.
Addresses the gap between vocal processing (Phase 2.2) and instrumental music.

Components:
1. DepthEnhancer - Reverb tail and early reflection preservation
2. WidthOptimizer - Stereo field optimization and mono compatibility
3. TexturePreserver - Atmospheric character and room ambience
4. SpatialLocalizer - 3D positioning and spatial imaging

Target: Bring Transparenz and Natürlichkeit Musical Goals +5% for spatial content

Usage:
    >>> from dsp.spatial_enhancement import SpatialEnhancementSystem
    >>>
    >>> enhancer = SpatialEnhancementSystem(
    ...     depth_db=2.0,
    ...     width_factor=1.2,
    ...     texture_preservation=0.7,
    ...     spatial_clarity=0.8
    ... )
    >>>
    >>> processed, report = enhancer.process(audio, sr)
    >>> print(f"Spatial enhancement: {report['spatial_quality']:.1f}")

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
# COMPONENT 1: DEPTH ENHANCER
# =============================================================================


class DepthEnhancer:
    """
    Depth Enhancement.

    Preserves and enhances reverb tails and early reflections.
    Critical for spatial depth and room character.

    Techniques:
    - Reverb tail preservation
    - Early reflection enhancement
    - Depth perception optimization

    Parameters
    ----------
    depth_db : float
        Reverb depth gain (0.0-3.0 dB)
    early_reflections : bool
        Enhance early reflections
    preserve_decay : bool
        Preserve natural reverb decay
    """

    def __init__(self, depth_db: float = 2.0, early_reflections: bool = True, preserve_decay: bool = True):
        self.depth_db = np.clip(depth_db, 0.0, 3.0)
        self.early_reflections = early_reflections
        self.preserve_decay = preserve_decay

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance spatial depth.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (stereo required)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Enhanced audio
        report : dict
            Processing report
        """
        # Require stereo for spatial processing
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        # Extract ambience (high frequencies contain more spatial info)
        sos_ambience = butter(4, 2000, btype="high", fs=sr, output="sos")
        ambience_l = sosfilt(sos_ambience, audio[:, 0])
        ambience_r = sosfilt(sos_ambience, audio[:, 1])

        # Measure original ambience energy
        ambience_energy_orig = np.sqrt(np.mean(ambience_l**2 + ambience_r**2))

        # Detect reverb tail (decaying envelope)
        envelope_l = np.abs(np.asarray(hilbert(ambience_l), dtype=np.complex128))
        envelope_r = np.abs(np.asarray(hilbert(ambience_r), dtype=np.complex128))
        envelope_avg = (envelope_l + envelope_r) / 2.0

        # Smooth envelope
        window_size = int(0.1 * sr)  # 100ms
        envelope_smooth = np.convolve(envelope_avg, np.ones(window_size) / window_size, mode="same")

        # Detect decay regions (falling envelope)
        derivative = np.diff(envelope_smooth, prepend=envelope_smooth[0])
        decay_mask = (derivative < 0).astype(float)

        # Smooth decay mask
        decay_mask_smooth = np.convolve(decay_mask, np.ones(window_size) / window_size, mode="same")

        # Apply depth gain to reverb tail
        if self.preserve_decay:
            depth_gain = 10 ** (self.depth_db / 20.0)

            # Boost during decay
            ambience_l_enhanced = ambience_l * (1 + decay_mask_smooth * (depth_gain - 1))
            ambience_r_enhanced = ambience_r * (1 + decay_mask_smooth * (depth_gain - 1))
        else:
            # Global gain
            depth_gain = 10 ** (self.depth_db / 20.0)
            ambience_l_enhanced = ambience_l * depth_gain
            ambience_r_enhanced = ambience_r * depth_gain

        # Early reflection enhancement (optional)
        if self.early_reflections:
            # Early reflections are 10-50ms after direct sound
            # Enhance transients in ambience
            transient_mask = derivative > np.percentile(derivative, 90)
            transient_mask_expanded = np.convolve(
                transient_mask.astype(float), np.ones(int(0.05 * sr)) / (0.05 * sr), mode="same"
            )

            # Boost early reflections
            early_boost = 1.2
            ambience_l_enhanced = ambience_l_enhanced * (1 + transient_mask_expanded * (early_boost - 1))
            ambience_r_enhanced = ambience_r_enhanced * (1 + transient_mask_expanded * (early_boost - 1))

        # Reconstruct audio
        sos_low = butter(4, 2000, btype="low", fs=sr, output="sos")
        low_l = sosfilt(sos_low, audio[:, 0])
        low_r = sosfilt(sos_low, audio[:, 1])

        result = np.stack([low_l + ambience_l_enhanced, low_r + ambience_r_enhanced], axis=-1)

        # Measure new ambience energy
        sos_ambience = butter(4, 2000, btype="high", fs=sr, output="sos")
        ambience_new_l = sosfilt(sos_ambience, result[:, 0])
        ambience_new_r = sosfilt(sos_ambience, result[:, 1])
        ambience_energy_new = np.sqrt(np.mean(ambience_new_l**2 + ambience_new_r**2))

        depth_change_db = 20 * np.log10((ambience_energy_new + 1e-10) / (ambience_energy_orig + 1e-10))

        report = {
            "depth_enhancement_db": depth_change_db,
            "early_reflections_enhanced": self.early_reflections,
            "decay_preserved": self.preserve_decay,
        }

        return result, report


# =============================================================================
# COMPONENT 2: WIDTH OPTIMIZER
# =============================================================================


class WidthOptimizer:
    """
    Stereo Width Optimization.

    Optimizes stereo width while maintaining mono compatibility.
    Critical for spatial impression and mix balance.

    Techniques:
    - Mid/Side processing
    - Width adjustment
    - Mono compatibility preservation

    Parameters
    ----------
    width_factor : float
        Stereo width factor (0.5-2.0, 1.0 = unchanged)
    mono_compatible : bool
        Ensure mono compatibility
    preserve_center : bool
        Preserve center content
    """

    def __init__(self, width_factor: float = 1.2, mono_compatible: bool = True, preserve_center: bool = True):
        self.width_factor = np.clip(width_factor, 0.5, 2.0)
        self.mono_compatible = mono_compatible
        self.preserve_center = preserve_center

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Optimiert stereo width.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (stereo required)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Width-optimized audio
        report : dict
            Processing report
        """
        # Require stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        left = audio[:, 0]
        right = audio[:, 1]

        # Convert to Mid/Side
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        # Measure original width
        mid_energy = np.sqrt(np.mean(mid**2))
        side_energy = np.sqrt(np.mean(side**2))
        original_width = side_energy / (mid_energy + 1e-10)

        # Apply width adjustment
        if self.preserve_center:
            # Only adjust side channel
            side_adjusted = side * self.width_factor
            mid_adjusted = mid
        else:
            # Adjust both with compensation
            side_adjusted = side * self.width_factor
            mid_adjusted = mid / np.sqrt(self.width_factor)

        # Mono compatibility check
        if self.mono_compatible:
            # Limit side gain to prevent phase cancellation in mono
            max_side_factor = 1.5
            if np.abs(self.width_factor) > max_side_factor:
                side_adjusted = side * max_side_factor

        # Convert back to Left/Right
        left_out = mid_adjusted + side_adjusted
        right_out = mid_adjusted - side_adjusted

        result = np.stack([left_out, right_out], axis=-1)

        # Measure new width
        mid_new = (left_out + right_out) / 2.0
        side_new = (left_out - right_out) / 2.0
        mid_energy_new = np.sqrt(np.mean(mid_new**2))
        side_energy_new = np.sqrt(np.mean(side_new**2))
        new_width = side_energy_new / (mid_energy_new + 1e-10)

        report = {
            "original_width": original_width,
            "new_width": new_width,
            "width_change": new_width / (original_width + 1e-10),
            "mono_compatible": self.mono_compatible,
        }

        return result, report


# =============================================================================
# COMPONENT 3: TEXTURE PRESERVER
# =============================================================================


class TexturePreserver:
    """
    Texture Preservation.

    Preserves atmospheric character and room ambience.
    Critical for natural spatial quality.

    Techniques:
    - Ambient texture enhancement
    - Room character preservation
    - Natural reverb balance

    Parameters
    ----------
    preservation : float
        Texture preservation amount (0.0-1.0)
    enhance_atmosphere : bool
        Enhance atmospheric content
    natural_balance : bool
        Maintain natural balance
    """

    def __init__(self, preservation: float = 0.7, enhance_atmosphere: bool = True, natural_balance: bool = True):
        self.preservation = np.clip(preservation, 0.0, 1.0)
        self.enhance_atmosphere = enhance_atmosphere
        self.natural_balance = natural_balance

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Preserve texture.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (stereo required)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Texture-enhanced audio
        report : dict
            Processing report
        """
        # Require stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        # Extract atmospheric content (uncorrelated stereo components)
        left = audio[:, 0]
        right = audio[:, 1]

        # Compute correlation
        correlation = np.correlate(left, right, mode="same")
        _ = correlation / (np.sqrt(np.sum(left**2) * np.sum(right**2)) + 1e-10)

        # Uncorrelated content = texture/ambience
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        # Measure texture energy
        texture_energy_orig = np.sqrt(np.mean(side**2))

        # Enhance atmospheric texture
        if self.enhance_atmosphere:
            # Boost side channel (contains room ambience)
            atmosphere_boost = 1.0 + self.preservation * 0.3
            side_enhanced = side * atmosphere_boost
        else:
            side_enhanced = side

        # Natural balance (optional)
        if self.natural_balance:
            # Prevent over-enhancement
            max_side_energy = np.sqrt(np.mean(mid**2)) * 0.8
            current_side_energy = np.sqrt(np.mean(side_enhanced**2))

            if current_side_energy > max_side_energy:
                side_enhanced = side_enhanced * (max_side_energy / (current_side_energy + 1e-10))

        # Reconstruct
        left_out = mid + side_enhanced
        right_out = mid - side_enhanced

        result = np.stack([left_out, right_out], axis=-1)

        # Measure texture energy
        side_new = (result[:, 0] - result[:, 1]) / 2.0
        texture_energy_new = np.sqrt(np.mean(side_new**2))

        texture_change_db = 20 * np.log10((texture_energy_new + 1e-10) / (texture_energy_orig + 1e-10))

        report = {
            "texture_enhancement_db": texture_change_db,
            "atmosphere_enhanced": self.enhance_atmosphere,
            "natural_balance_applied": self.natural_balance,
        }

        return result, report


# =============================================================================
# COMPONENT 4: SPATIAL LOCALIZER
# =============================================================================


class SpatialLocalizer:
    """
    Spatial Localization Enhancement.

    Enhances 3D positioning and spatial imaging.
    Critical for clarity and source localization.

    Techniques:
    - Haas effect enhancement
    - Precedence effect optimization
    - Spatial clarity

    Parameters
    ----------
    clarity : float
        Spatial clarity (0.0-1.0)
    haas_enhancement : bool
        Enhance Haas effect (precedence)
    maintain_naturalism : bool
        Maintain natural spatial characteristics
    """

    def __init__(self, clarity: float = 0.8, haas_enhancement: bool = True, maintain_naturalism: bool = True):
        self.clarity = np.clip(clarity, 0.0, 1.0)
        self.haas_enhancement = haas_enhancement
        self.maintain_naturalism = maintain_naturalism

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance spatial localization.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (stereo required)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Spatially enhanced audio
        report : dict
            Processing report
        """
        # Require stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        left = audio[:, 0]
        right = audio[:, 1]

        # Spatial clarity enhancement
        # Emphasize transients (improves localization)
        # Adapt filter frequencies to sample rate (max = 0.95 * Nyquist)
        nyquist = sr / 2
        low_freq = min(2000, nyquist * 0.25)
        high_freq = min(8000, nyquist * 0.95)
        sos_transient = butter(4, [low_freq, high_freq], btype="band", fs=sr, output="sos")
        transient_l = sosfilt(sos_transient, left)
        transient_r = sosfilt(sos_transient, right)

        # Detect transients
        envelope_l = np.abs(np.asarray(hilbert(transient_l), dtype=np.complex128))
        envelope_r = np.abs(np.asarray(hilbert(transient_r), dtype=np.complex128))

        derivative_l = np.diff(envelope_l, prepend=envelope_l[0])
        derivative_r = np.diff(envelope_r, prepend=envelope_r[0])

        transient_mask_l = derivative_l > np.percentile(derivative_l, 90)
        transient_mask_r = derivative_r > np.percentile(derivative_r, 90)

        # Expand masks
        window_size = int(0.01 * sr)  # 10ms
        transient_mask_l = np.convolve(transient_mask_l.astype(float), np.ones(window_size) / window_size, mode="same")
        transient_mask_r = np.convolve(transient_mask_r.astype(float), np.ones(window_size) / window_size, mode="same")

        # Enhance transients for clarity
        clarity_boost = 1.0 + self.clarity * 0.2

        left_enhanced = left + transient_l * transient_mask_l * (clarity_boost - 1)
        right_enhanced = right + transient_r * transient_mask_r * (clarity_boost - 1)

        # Haas effect enhancement (optional)
        if self.haas_enhancement and not self.maintain_naturalism:
            # Subtle time delay can enhance spatial perception
            # This is aggressive and may not be desired for natural recordings
            delay_samples = int(0.001 * sr)  # 1ms

            # Apply subtle cross-delay
            left_delayed = np.pad(right_enhanced, (delay_samples, 0), mode="constant")[: len(left_enhanced)]
            right_delayed = np.pad(left_enhanced, (delay_samples, 0), mode="constant")[: len(right_enhanced)]

            # Mix with original
            haas_mix = 0.1  # Very subtle
            left_enhanced = left_enhanced + left_delayed * haas_mix
            right_enhanced = right_enhanced + right_delayed * haas_mix

        result = np.stack([left_enhanced, right_enhanced], axis=-1)

        report = {
            "spatial_clarity_applied": self.clarity > 0,
            "haas_enhancement_applied": self.haas_enhancement and not self.maintain_naturalism,
            "naturalism_maintained": self.maintain_naturalism,
        }

        return result, report


# =============================================================================
# UNIFIED API: SPATIAL ENHANCEMENT SYSTEM
# =============================================================================


class SpatialEnhancementSystem:
    """
    Einheitliche API für räumliche Klangverbesserung.

    Combines all spatial processing components into a single pipeline:
    1. Depth Enhancement
    2. Width Optimization
    3. Texture Preservation
    4. Spatial Localization

    Parameters
    ----------
    depth_db : float
        Reverb depth gain (0.0-3.0 dB)
    width_factor : float
        Stereo width factor (0.5-2.0)
    texture_preservation : float
        Texture preservation (0.0-1.0)
    spatial_clarity : float
        Spatial clarity (0.0-1.0)
    """

    def __init__(
        self,
        depth_db: float = 2.0,
        width_factor: float = 1.2,
        texture_preservation: float = 0.7,
        spatial_clarity: float = 0.8,
    ):
        self.depth_enhancer = DepthEnhancer(depth_db=depth_db)
        self.width_optimizer = WidthOptimizer(width_factor=width_factor)
        self.texture_preserver = TexturePreserver(preservation=texture_preservation)
        self.spatial_localizer = SpatialLocalizer(clarity=spatial_clarity)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full spatial enhancement pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (stereo required)
        sr : int
            Sample rate in Hz

        Returns
        -------
        processed : np.ndarray
            Spatially enhanced audio
        report : dict
            Comprehensive processing report
        """
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        result = audio.copy()
        report = {}

        # Stage 1: Depth Enhancement
        result, depth_report = self.depth_enhancer.process(result, sr)
        report["depth"] = depth_report

        # Stage 2: Width Optimization
        result, width_report = self.width_optimizer.process(result, sr)
        report["width"] = width_report

        # Stage 3: Texture Preservation
        result, texture_report = self.texture_preserver.process(result, sr)
        report["texture"] = texture_report

        # Stage 4: Spatial Localization
        result, spatial_report = self.spatial_localizer.process(result, sr)
        report["spatial"] = spatial_report

        # Calculate overall metrics
        report["stages_applied"] = 4  # type: ignore[assignment]
        report["spatial_quality"] = report["width"]["width_change"] * 100  # Width improvement

        return result, report


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI interface for Spatial Enhancement System."""
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Phase 2.3 - Spatial Enhancement System")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")

    parser.add_argument("--depth", type=float, default=2.0, help="Reverb depth in dB (0.0-3.0, default: 2.0)")
    parser.add_argument("--width", type=float, default=1.2, help="Stereo width factor (0.5-2.0, default: 1.2)")
    parser.add_argument("--texture", type=float, default=0.7, help="Texture preservation (0.0-1.0, default: 0.7)")
    parser.add_argument("--clarity", type=float, default=0.8, help="Spatial clarity (0.0-1.0, default: 0.8)")

    args = parser.parse_args()

    # Load audio
    logger.info("lade: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Ensure stereo
    if audio.shape[1] == 1:
        audio = np.stack([audio[:, 0], audio[:, 0]], axis=-1)

    # Create spatial enhancement system
    logger.info("\n🌌 Spatial Enhancement System")
    logger.info("=" * 60)

    enhancer = SpatialEnhancementSystem(
        depth_db=args.depth, width_factor=args.width, texture_preservation=args.texture, spatial_clarity=args.clarity
    )

    # Process
    logger.info("Processing...")
    processed, report = enhancer.process(audio, sr)

    # Print report
    logger.info("\n📊 Processing Report:")
    logger.info("-" * 60)
    logger.info("Depth: %.1f dB", report["depth"]["depth_enhancement_db"])
    logger.info("  Early reflections: %s", "Yes" if report["depth"]["early_reflections_enhanced"] else "No")

    logger.info("\nWidth: %.2fx", report["width"]["width_change"])
    logger.info("  Originalsignal: %.2f", report["width"]["original_width"])
    logger.info("  New: %.2f", report["width"]["new_width"])

    logger.info("\nTexture: %.1f dB", report["texture"]["texture_enhancement_db"])
    logger.info("  Atmosphere: %s", "Yes" if report["texture"]["atmosphere_enhanced"] else "No")

    logger.info("\nSpatial Clarity: %s", "Applied" if report["spatial"]["spatial_clarity_applied"] else "Not applied")
    logger.info("  Haas enhancement: %s", "Yes" if report["spatial"]["haas_enhancement_applied"] else "No")

    logger.info("\nSpatial Quality: %.1f", report["spatial_quality"])
    logger.info("Stages angewendet: %s", report["stages_applied"])

    # Save
    logger.info("\nspeichere: %s", args.output)
    sf.write(args.output, processed, sr)
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
