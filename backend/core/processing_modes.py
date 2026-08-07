"""
Processing Modes für AURIK

Provides 2 Magic Button modes with predefined parameter sets:
- RESTORATION: Authentizität bewahren (Musik-Restauration)
- STUDIO_2026: Modern Highend Studio-Sound

HINWEIS: Forensic Analysis ist KEIN Mode, sondern eine feste Pipeline-Komponente,
die immer aktiv ist (Analyse, Detection, Metadata-Extraktion in Phase 1).

Author: AURIK Development Team
Version: 2.0 (Magic Button Edition)
Date: 2026-02-13
"""

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """
    Available Magic Button processing modes.

    Nur 2 user-wählbare Modi. Forensic ist KEIN Mode.
    """

    RESTORATION = "restoration"
    """Magic Button 1: Authentizität bewahren (Musik-Restauration).

    - Moderate denoise strength
    - Preserve all natural artifacts (breaths, room tone)
    - Gentle dynamics
    - Original LUFS/tonality maintained
    - Target: Transparent restoration with character preservation
    """

    STUDIO_2026 = "studio_2026"
    """Magic Button 2: Modern Highend Studio-Sound (Streaming-ready).

    - Aggressive noise reduction
    - Modern "air" (high frequency boost)
    - Competitive loudness
    - Tight dynamics
    - Target: Competitive streaming sound, maximum brilliance
    """

    @classmethod
    def from_string(cls, mode: str) -> "ProcessingMode":
        """Konvertiert string to ProcessingMode enum."""
        mode_lower = mode.lower()
        for m in cls:
            if m.value == mode_lower:
                return m
        raise ValueError(f"Invalid processing mode: {mode}. Valid modes: {[m.value for m in cls]}")


@dataclass
class ProcessingConfig:
    """Configuration for a specific processing mode."""

    # === Lyrics-Guided Enhancement ===
    enable_lyrics_guided: bool = False
    """Enable lyrics-guided vocal enhancement (World-First Innovation)."""

    # === Core Processing Parameters ===

    denoise_strength: float = 0.30
    """Denoise strength (0.0-1.0). Higher = more aggressive."""

    declip_strength: float = 0.50
    """Declipping strength (0.0-1.0). Higher = more aggressive reconstruction."""

    click_removal_sensitivity: float = 0.50
    """Click removal sensitivity (0.0-1.0). Higher = more clicks removed."""

    # === Authenticity Preservation ===

    preserve_breaths: bool = True
    """Preserve natural breaths (CRITICAL for music restoration)."""

    preserve_room_tone: bool = True
    """Preserve natural room ambience/reverb."""

    preserve_analog_character: bool = False
    """Preserve analog warmth (tape saturation, vinyl character)."""

    # === Dynamics & Loudness ===

    compression_ratio: float = 2.0
    """Dynamic range compression ratio (1.0=off, 10.0=brick-wall)."""

    compression_threshold_db: float = -20.0
    """Compression threshold in dB."""

    target_lufs: float | None = None
    """Target LUFS (None = keep original, -14.0 = streaming standard)."""

    # === Tonal Shaping ===

    high_freq_boost_db: float = 0.0
    """High frequency boost in dB (modern "air", 0=off)."""

    low_freq_rolloff_hz: int | None = None
    """Low frequency rolloff (None = off, 20-100 Hz)."""

    deesser_strength: float = 0.0
    """De-esser strength (0.0-1.0, for sibilance control)."""

    dereverb_strength: float = 0.0
    """De-reverb strength (0.0-1.0, for studio reverb removal)."""

    stereo_width_factor: float = 1.0
    """Stereo width enhancement factor (0.0=mono, 1.0=normal, 2.0=ultra-wide)."""

    true_peak_ceiling_dbtp: float = -1.0
    """True Peak ceiling in dBTP (EBU R128 recommends -1.0 dBTP for broadcast/streaming)."""

    # === Multiband Compression ===

    enable_multiband_compression: bool = False
    """Enable multiband compression for frequency-specific dynamics control."""

    multiband_bands: int = 3
    """Number of frequency bands for multiband compression (2-5)."""

    multiband_crossovers: tuple = (200, 2000)
    """Crossover frequencies in Hz (length = bands - 1)."""

    multiband_thresholds_db: tuple = (-24, -18, -12)
    """Compression thresholds per band in dB (length = bands)."""

    multiband_ratios: tuple = (2.0, 3.0, 4.0)
    """Compression ratios per band (length = bands)."""

    # === Spectral Repair ===

    enable_spectral_repair: bool = False
    """Enable spectral repair for digital artifacts (MP3, packet loss, codec artifacts)."""

    spectral_repair_strength: float = 0.7
    """Spectral repair strength (0.0-1.0)."""

    spectral_repair_hole_threshold_db: float = -60.0
    """Threshold for detecting spectral holes in dB below peak."""

    # === CD Noise Profile (§G8, §V14) ===

    enable_cd_noise_profile: bool = True
    """Enable CD-characteristic noise profile injection on export.

    Applies to BOTH Restoration and Studio 2026 modes (§V14).
    Noise is psychoacoustically masked (§G15) — only injected where
    the human ear would perceive silence/gaps.
    """

    # === Enhancement ===

    enable_enhancement: bool = True
    """Enable ResembleEnhance (vocal clarity, intelligibility)."""

    enhancement_strength: float = 0.50
    """Enhancement strength (0.0-1.0) if enabled."""

    # === Forensic / Integrity ===

    forensic_mode: bool = False
    """Enable forensic mode (minimal processing, full transparency)."""

    save_intermediate_steps: bool = False
    """Save intermediate audio after each phase (debugging/forensics)."""

    # === Mode Metadata ===

    mode_name: str = "custom"
    """Name of the processing mode."""

    description: str = "Custom processing configuration"
    """Human-readable description of this mode."""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary (for JSON serialization)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingConfig":
        """Erstellt from dictionary."""
        return cls(**data)

    def validate(self) -> None:
        """Validiert parameter ranges."""
        # Validate strength parameters (0.0-1.0)
        for param in [
            "denoise_strength",
            "declip_strength",
            "click_removal_sensitivity",
            "enhancement_strength",
            "deesser_strength",
        ]:
            value = getattr(self, param)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{param} must be in [0.0, 1.0], got {value}")

        # Validate compression ratio
        if not 1.0 <= self.compression_ratio <= 10.0:
            raise ValueError(f"compression_ratio must be in [1.0, 10.0], got {self.compression_ratio}")

        # Validate LUFS range
        if self.target_lufs is not None and not -30.0 <= self.target_lufs <= -5.0:
            raise ValueError(f"target_lufs must be in [-30.0, -5.0], got {self.target_lufs}")

        # Validate high freq boost
        if not -6.0 <= self.high_freq_boost_db <= 6.0:
            raise ValueError(f"high_freq_boost_db must be in [-6.0, 6.0], got {self.high_freq_boost_db}")

        # Validate dereverb strength
        if not 0.0 <= self.dereverb_strength <= 1.0:
            raise ValueError(f"dereverb_strength must be in [0.0, 1.0], got {self.dereverb_strength}")

        # Validate stereo width factor
        if not 0.0 <= self.stereo_width_factor <= 3.0:
            raise ValueError(f"stereo_width_factor must be in [0.0, 3.0], got {self.stereo_width_factor}")

        # Validate true peak ceiling
        if not -6.0 <= self.true_peak_ceiling_dbtp <= 0.0:
            raise ValueError(f"true_peak_ceiling_dbtp must be in [-6.0, 0.0], got {self.true_peak_ceiling_dbtp}")

        # Validate multiband compression parameters
        if self.enable_multiband_compression:
            if not 2 <= self.multiband_bands <= 5:
                raise ValueError(f"multiband_bands must be in [2, 5], got {self.multiband_bands}")

            if len(self.multiband_crossovers) != self.multiband_bands - 1:
                raise ValueError(
                    f"multiband_crossovers length must be {self.multiband_bands - 1}, "
                    f"got {len(self.multiband_crossovers)}"
                )

            if len(self.multiband_thresholds_db) != self.multiband_bands:
                raise ValueError(
                    f"multiband_thresholds_db length must be {self.multiband_bands}, "
                    f"got {len(self.multiband_thresholds_db)}"
                )

            if len(self.multiband_ratios) != self.multiband_bands:
                raise ValueError(
                    f"multiband_ratios length must be {self.multiband_bands}, got {len(self.multiband_ratios)}"
                )

        # Validate spectral repair parameters
        if self.enable_spectral_repair:
            if not 0.0 <= self.spectral_repair_strength <= 1.0:
                raise ValueError(f"spectral_repair_strength must be in [0.0, 1.0], got {self.spectral_repair_strength}")

            if not -80.0 <= self.spectral_repair_hole_threshold_db <= -40.0:
                raise ValueError(
                    "spectral_repair_hole_threshold_db must be in [-80.0, -40.0], "
                    f"got {self.spectral_repair_hole_threshold_db}"
                )


# === Predefined Mode Configurations ===

PROCESSING_CONFIGS: dict[ProcessingMode, ProcessingConfig] = {
    ProcessingMode.RESTORATION: ProcessingConfig(
        mode_name="restoration",
        description="Default - Authentizität bewahren (Musik-Restauration)",
        # Moderate processing
        denoise_strength=0.30,
        declip_strength=0.50,
        click_removal_sensitivity=0.50,
        # Preserve authenticity
        preserve_breaths=True,
        preserve_room_tone=True,
        preserve_analog_character=False,
        # Gentle dynamics
        compression_ratio=2.0,
        compression_threshold_db=-20.0,
        target_lufs=None,  # Keep original loudness
        # No tonal shaping
        high_freq_boost_db=0.0,
        low_freq_rolloff_hz=None,
        deesser_strength=0.0,
        dereverb_strength=0.0,  # No de-reverb (preserve natural ambience)
        # Stereo & Mastering
        stereo_width_factor=1.0,  # Original width (no enhancement)
        true_peak_ceiling_dbtp=-1.0,  # EBU R128 standard
        # Multiband Compression (disabled for restoration)
        enable_multiband_compression=False,
        multiband_bands=3,
        multiband_crossovers=(200, 2000),
        multiband_thresholds_db=(-24, -18, -12),
        multiband_ratios=(2.0, 3.0, 4.0),
        # Spectral Repair (disabled - preserve original artifacts)
        enable_spectral_repair=False,
        spectral_repair_strength=0.7,
        spectral_repair_hole_threshold_db=-60.0,
        # CD Noise Profile (§G8, §V14)
        enable_cd_noise_profile=True,
        # Enhancement
        enable_enhancement=True,
        enhancement_strength=0.50,
        # Not forensic
        forensic_mode=False,
        save_intermediate_steps=False,
    ),
    ProcessingMode.STUDIO_2026: ProcessingConfig(
        mode_name="studio_2026",
        description="Modern Highend Studio-Sound (Streaming-ready)",
        # Aggressive processing
        denoise_strength=0.50,
        declip_strength=0.60,
        click_removal_sensitivity=0.70,
        # Preserve breaths but allow modern shaping
        preserve_breaths=True,  # STILL critical for vocals!
        preserve_room_tone=False,  # Remove room ambience
        preserve_analog_character=False,
        # Competitive dynamics
        compression_ratio=4.0,
        compression_threshold_db=-18.0,
        target_lufs=-14.0,  # Streaming standard (Spotify, YouTube)
        # Modern "air"
        high_freq_boost_db=2.0,
        low_freq_rolloff_hz=30,  # Clean low end
        deesser_strength=0.30,  # Control sibilance
        dereverb_strength=0.50,  # Moderate de-reverb for studio clarity
        # Stereo & Mastering
        stereo_width_factor=1.5,  # Modern wide soundstage
        true_peak_ceiling_dbtp=-1.0,  # Streaming standard (EBU R128)
        # Multiband Compression (active for professional dynamics)
        enable_multiband_compression=True,
        multiband_bands=3,
        multiband_crossovers=(200, 2000),
        multiband_thresholds_db=(-24, -18, -12),
        multiband_ratios=(3.0, 4.0, 5.0),  # More aggressive for modern sound
        # Spectral Repair (active for cleaning digital artifacts)
        enable_spectral_repair=True,
        spectral_repair_strength=0.8,  # Strong repair for professional quality
        spectral_repair_hole_threshold_db=-60.0,
        # CD Noise Profile (§G8, §V14)
        enable_cd_noise_profile=True,
        # Strong enhancement
        enable_enhancement=True,
        enhancement_strength=0.70,
        # Not forensic
        forensic_mode=False,
        save_intermediate_steps=False,
    ),
}


def get_processing_config(mode: ProcessingMode) -> ProcessingConfig:
    """Gibt zurück: processing configuration for a given mode.

    Args:
        mode: Processing mode enum

    Returns:
        ProcessingConfig for the requested mode

    Raises:
        ValueError: If mode is invalid
    """
    if mode not in PROCESSING_CONFIGS:
        raise ValueError(f"No configuration for mode {mode}. Available modes: {[m.value for m in PROCESSING_CONFIGS]}")

    return PROCESSING_CONFIGS[mode]


def list_available_modes() -> dict[str, str]:
    """Listet auf: all available processing modes with descriptions.

    Returns:
        Dictionary mapping mode names to descriptions
    """
    return {mode.value: config.description for mode, config in PROCESSING_CONFIGS.items()}


def create_custom_config(**kwargs) -> ProcessingConfig:
    """Erstellt a custom processing configuration.

    Args:
        **kwargs: Parameter overrides (see ProcessingConfig fields)

    Returns:
        Custom ProcessingConfig

    Raises:
        ValueError: If any parameters are out of valid ranges

    Example:
        >>> config = create_custom_config(
        ...     denoise_strength=0.40,
        ...     preserve_breaths=True,
        ...     target_lufs=-16.0
        ... )
    """
    config = ProcessingConfig(**kwargs)
    config.validate()
    return config


# Convenience function
def get_config_by_name(mode_name: str) -> ProcessingConfig:
    """Gibt zurück: processing configuration by mode name (string).

    Args:
        mode_name: Mode name (e.g., "restoration", "studio_2026")

    Returns:
        ProcessingConfig for the requested mode

    Raises:
        ValueError: If mode name is invalid
    """
    mode = ProcessingMode.from_string(mode_name)
    return get_processing_config(mode)


if __name__ == "__main__":
    # Demo: List all available modes
    logger.debug("AURIK Processing Modes:")
    logger.debug("=" * 60)
    for mode_name, description in list_available_modes().items():
        logger.debug("\n%s:", mode_name.upper())
        logger.debug("  %s", description)

    logger.debug("\n" + "=" * 60)
    logger.debug("\nExample: RESTORATION Betriebsart parameters:")
    logger.debug("=" * 60)
    restoration = get_processing_config(ProcessingMode.RESTORATION)
    import json

    logger.debug(json.dumps(restoration.to_dict(), indent=2))
