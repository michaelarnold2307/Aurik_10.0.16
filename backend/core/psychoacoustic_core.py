#!/usr/bin/env python3
"""
Psychoacoustic Core - Unified Perceptual Processing Foundation
================================================================

Zentrales Modul für psychoacoustische Audio-Verarbeitung.
Integriert alle psychoacoustischen Komponenten in einer einheitlichen API.

Komponenten:
1. **Bark Scale Processor**: 24 Critical Bands Analysis
2. **Masking Analyzer**: Simultaneous + Temporal Masking
3. **Fletcher-Munson Curves**: Equal-Loudness Compensation

Psychoacoustic Prinzipien:
- Perception-First: Was Menschen hören ist wichtiger als was messbar ist
- Critical Bands: Frequenzauflösung des Gehörs (~24 Bark Bands)
- Masking: Laute Signale maskieren leisere in Zeit & Frequenz
- Equal-Loudness: Frequenzabhängige Lautheitswahrnehmung

Anwendungsbereiche:
- Audio Codecs (MP3, AAC, Opus)
- Noise Reduction & Enhancement
- Dynamic Range Control
- Tonal Balance Restoration
- Quality Assessment
- Mastering & Sound Design

Autor: Aurik v8.0 - Psychoacoustic Foundation
Lizenz: Proprietär
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

# Import psychoacoustic modules
from backend.core.bark_scale_processor import BarkScaleProcessor, BarkSpectrum, bark_to_hz, get_bark_bands, hz_to_bark
from backend.core.fletcher_munson_curves import FletcherMunsonConfig, FletcherMunsonProcessor
from backend.core.masking_analyzer import MaskingAnalyzer, MaskingConfig, MaskingProfile

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PSYCHOACOUSTIC ANALYSIS REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PsychoacousticAnalysis:
    """
    Comprehensive psychoacoustic analysis report.

    Attributes:
        bark_spectrum: Bark scale energy distribution
        masking_profile: Masking analysis results
        perceptual_centroid_bark: Spectral centroid in Bark scale
        perceptual_bandwidth_bark: Bandwidth in Bark scale
        masked_components_ratio: Ratio of masked (inaudible) components
        signal_to_mask_ratio_db: Average SMR in dB
        peak_bark_band: Most energetic Bark band
        peak_bark_energy: Energy of peak band
        frequency_balance: Dict of energy ratios (bass, mid, treble, air)
    """

    bark_spectrum: BarkSpectrum
    masking_profile: MaskingProfile
    perceptual_centroid_bark: float
    perceptual_bandwidth_bark: float
    masked_components_ratio: float
    signal_to_mask_ratio_db: float
    peak_bark_band: int
    peak_bark_energy: float
    frequency_balance: dict[str, float]

    def summary_dict(self) -> dict[str, Any]:
        """Gibt zurück: summary as dictionary."""
        return {
            "perceptual_centroid_bark": self.perceptual_centroid_bark,
            "perceptual_bandwidth_bark": self.perceptual_bandwidth_bark,
            "masked_components_ratio": self.masked_components_ratio,
            "signal_to_mask_ratio_db": self.signal_to_mask_ratio_db,
            "peak_bark_band": self.peak_bark_band,
            "peak_bark_energy": self.peak_bark_energy,
            "frequency_balance": self.frequency_balance,
        }

    def print_summary(self):
        """Gibt menschenlesbare Zusammenfassung der psychoakustischen Analyse aus."""
        logger.info("=" * 70)
        logger.info("PSYCHOACOUSTIC Analyse SUMMARY")
        logger.info("=" * 70)
        logger.info("  Perceptual Centroid: %.2f Bark", self.perceptual_centroid_bark)
        logger.info("  Perceptual Bandwidth: %.2f Bark", self.perceptual_bandwidth_bark)
        logger.info("  Masked Components: %.1f%%", self.masked_components_ratio * 100)
        logger.info("  Signal-to-Mask Verhaeltnis: %.1f dB", self.signal_to_mask_ratio_db)
        logger.info("  Peak Bark Band: #%d", self.peak_bark_band)
        logger.info("  Peak Energy: %.4f", self.peak_bark_energy)
        logger.info("  Frequency Balance:")
        for band, energy in self.frequency_balance.items():
            logger.info("    %s: %.1f%%", band.capitalize(), energy * 100)
        logger.info("=" * 70)


@dataclass
class PsychoacousticConfig:
    """
    Configuration for psychoacoustic core.

    Attributes:
        enable_bark_analysis: Enable Bark scale analysis
        enable_masking_analysis: Enable masking analysis
        enable_fletcher_munson: Enable equal-loudness compensation
        bark_normalize: Normalize Bark energies
        masking_temporal: Enable temporal masking
        masking_simultaneous: Enable simultaneous masking
        fletcher_target_phon: Target phon level for compensation
        fletcher_reference_phon: Reference phon level
    """

    enable_bark_analysis: bool = True
    enable_masking_analysis: bool = True
    enable_fletcher_munson: bool = True
    bark_normalize: bool = True
    masking_temporal: bool = True
    masking_simultaneous: bool = True
    fletcher_target_phon: int = 60
    fletcher_reference_phon: int = 80
    # §v10.0.5: LUFS-adaptives Phon-Level — wenn True, wird target_phon
    # aus dem Audio-LUFS-Pegel berechnet (leise = niedrigeres Phon).
    fletcher_adaptive_phon: bool = True  # §v10.0.5: LUFS-adaptiv ist Default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PSYCHOACOUSTIC CORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PsychoacousticCore:
    """
    Einheitliche psychoakustische Verarbeitungs-Engine.

    Combines Bark scale, masking, and equal-loudness processing
    into a single, easy-to-use interface.

    Features:
    - Comprehensive perceptual analysis
    - Frequency balance assessment
    - Masking-aware processing
    - Loudness compensation
    - Musical frequency band analysis

    Example:
        >>> core = PsychoacousticCore()
        >>> analysis = core.analyze(audio, sr=48000)
        >>> analysis.print_summary()
        >>>
        >>> # Apply loudness compensation
        >>> compensated = core.apply_loudness_compensation(audio, sr, listening_level='normal')
    """

    def __init__(self, config: PsychoacousticConfig | None = None):
        """
        Initialisiert psychoacoustic core.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or PsychoacousticConfig()

        # Initialize sub-processors
        self.bark_processor = BarkScaleProcessor() if self.config.enable_bark_analysis else None

        if self.config.enable_masking_analysis:
            masking_config = MaskingConfig(
                enable_temporal=self.config.masking_temporal, enable_simultaneous=self.config.masking_simultaneous
            )
            self.masking_analyzer = MaskingAnalyzer(masking_config)
        else:
            self.masking_analyzer = None  # type: ignore[assignment]

        if self.config.enable_fletcher_munson:
            fm_config = FletcherMunsonConfig(
                target_phon=self.config.fletcher_target_phon, reference_phon=self.config.fletcher_reference_phon
            )
            self.fm_processor = FletcherMunsonProcessor(fm_config)
        else:
            self.fm_processor = None  # type: ignore[assignment]

        logger.info("PsychoacousticCore initialisiert with all components aktiviert")

    def analyze(self, audio: np.ndarray, sr: int) -> PsychoacousticAnalysis:
        """
        Führt durch: comprehensive psychoacoustic analysis.

        Args:
            audio: Input audio (mono)
            sr: Sample rate

        Returns:
            PsychoacousticAnalysis object
        """
        # Bark scale analysis
        if self.bark_processor:
            bark_spectrum = self.bark_processor.analyze(audio, sr, normalize=self.config.bark_normalize)
            perceptual_centroid = bark_spectrum.get_spectral_centroid_bark()
            peak_band, peak_energy = bark_spectrum.get_peak_band()
            peak_bark_band = peak_band.index
        else:
            raise RuntimeError("Bark processor not enabled")

        # Masking analysis
        if self.masking_analyzer:
            masking_profile = self.masking_analyzer.analyze(audio, sr)
            masked_ratio = masking_profile.get_masked_components_ratio()
            smr_db = self.masking_analyzer.compute_smr(audio, sr)
        else:
            raise RuntimeError("Masking analyzer not enabled")

        # Compute perceptual bandwidth (Bark scale)
        # Bandwidth = spread around centroid containing 90% of energy
        cumsum = np.cumsum(bark_spectrum.energies)
        cumsum_norm = cumsum / cumsum[-1] if cumsum[-1] > 0 else np.zeros_like(cumsum)  # §3.1
        lower_idx = np.argmax(cumsum_norm >= 0.05)  # 5% threshold
        upper_idx = np.argmax(cumsum_norm >= 0.95)  # 95% threshold
        perceptual_bandwidth = float(upper_idx - lower_idx)

        # Frequency balance (musical bands)
        frequency_balance = self._compute_frequency_balance(bark_spectrum)

        return PsychoacousticAnalysis(
            bark_spectrum=bark_spectrum,
            masking_profile=masking_profile,
            perceptual_centroid_bark=perceptual_centroid,
            perceptual_bandwidth_bark=perceptual_bandwidth,
            masked_components_ratio=masked_ratio,
            signal_to_mask_ratio_db=smr_db,
            peak_bark_band=peak_bark_band,
            peak_bark_energy=float(peak_energy),
            frequency_balance=frequency_balance,
        )

    def _compute_frequency_balance(self, bark_spectrum: BarkSpectrum) -> dict[str, float]:
        """
        Berechnet energy distribution across musical frequency bands.

        Args:
            bark_spectrum: Bark spectrum

        Returns:
            Dict with energy ratios for bass, mid, treble, air
        """
        # Musical frequency bands (approximate Bark band mappings)
        # Bass: 20-200 Hz (Bark 0-2)
        # Mid: 200-2000 Hz (Bark 2-13)
        # Treble: 2-8 kHz (Bark 13-21)
        # Air: 8-16 kHz (Bark 21-23)

        bass_energy = bark_spectrum.get_energy_in_range(20, 200)
        mid_energy = bark_spectrum.get_energy_in_range(200, 2000)
        treble_energy = bark_spectrum.get_energy_in_range(2000, 8000)
        air_energy = bark_spectrum.get_energy_in_range(8000, 16000)

        return {
            "bass": float(bass_energy),
            "mid": float(mid_energy),
            "treble": float(treble_energy),
            "air": float(air_energy),
        }

    def apply_loudness_compensation(self, audio: np.ndarray, sr: int, listening_level: str = "normal") -> np.ndarray:
        """
        Wendet an: Fletcher-Munson loudness compensation.

        Args:
            audio: Input audio (mono)
            sr: Sample rate
            listening_level: 'quiet', 'normal', 'loud'

        Returns:
            Compensated audio
        """
        if not self.fm_processor:
            raise RuntimeError("Fletcher-Munson processor not enabled")

        level_map = {"quiet": (40, 80), "normal": (60, 80), "loud": (80, 100)}

        target_phon, reference_phon = level_map.get(listening_level, (60, 80))

        # §v10.0.5: LUFS-adaptives Phon-Level — überschreibt listening_level
        # wenn fletcher_adaptive_phon=True. Misst LUFS und berechnet
        # target_phon = clamp(60 + (LUFS+16)*1.5, 30, 80).
        if self.config.fletcher_adaptive_phon:
            try:
                from backend.core.fletcher_munson_curves import compute_adaptive_phon

                target_phon = compute_adaptive_phon(audio=audio, sr=sr, default_phon=target_phon)
                logger.debug(
                    "§v10.0.5 Adaptive Phon: target_phon=%d (from LUFS measurement)",
                    target_phon,
                )
            except Exception:
                logger.warning(
                    "⚠️ SOTA Fletcher-Munson adaptive phon nicht verfügbar (LUFS-Messung fehlgeschlagen) — Ersatzpfad zu listening_level-Map"
                )
                pass

        compensated, _ = self.fm_processor.apply_compensation(audio, sr, target_phon, reference_phon)
        compensated = np.nan_to_num(compensated, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(compensated, -1.0, 1.0)  # type: ignore[no-any-return]

    def remove_masked_components(self, audio: np.ndarray, sr: int, threshold_db: float = 0.0) -> np.ndarray:
        """
        Entfernt perceptually masked (inaudible) components.

        Useful for:
        - Audio coding (remove redundant information)
        - Noise reduction (preserve audible signal only)
        - Dynamic range optimization

        Args:
            audio: Input audio
            sr: Sample rate
            threshold_db: Additional threshold in dB

        Returns:
            Audio with masked components removed
        """
        if not self.masking_analyzer:
            raise RuntimeError("Masking analyzer not enabled")

        return self.masking_analyzer.apply_masking(audio, sr, threshold_db=threshold_db)

    def get_perceptual_eq_curve(
        self, frequencies: np.ndarray, target_balance: dict[str, float] | None = None
    ) -> np.ndarray:
        """
        Generiert perceptual EQ curve to achieve target balance.

        Args:
            frequencies: Frequency points for EQ curve
            target_balance: Target energy ratios (bass, mid, treble, air)

        Returns:
            EQ curve in dB (same shape as frequencies)
        """
        # Default: flat perceptual balance
        if target_balance is None:
            target_balance = {"bass": 0.20, "mid": 0.50, "treble": 0.25, "air": 0.05}

        # This is a simplified implementation
        # Production version would use iterative optimization

        eq_curve = np.zeros(len(frequencies))

        # Apply target balance as gains
        for freq_idx, freq in enumerate(frequencies):
            if freq < 200:
                eq_curve[freq_idx] = 10 * np.log10(target_balance["bass"] / 0.20 + 1e-10)
            elif freq < 2000:
                eq_curve[freq_idx] = 10 * np.log10(target_balance["mid"] / 0.50 + 1e-10)
            elif freq < 8000:
                eq_curve[freq_idx] = 10 * np.log10(target_balance["treble"] / 0.25 + 1e-10)
            else:
                eq_curve[freq_idx] = 10 * np.log10(target_balance["air"] / 0.05 + 1e-10)

        # Smooth curve
        from scipy.ndimage import gaussian_filter1d

        eq_curve = gaussian_filter1d(eq_curve, sigma=2.0)

        return eq_curve  # type: ignore[no-any-return]

    def get_bark_bands(self):
        """Gibt zurück: list of Bark bands."""
        return get_bark_bands()

    def hz_to_bark(self, freq_hz: float) -> float:
        """Konvertiert Hz to Bark scale."""
        return hz_to_bark(freq_hz)

    def bark_to_hz(self, bark: float) -> float:
        """Konvertiert Bark scale to Hz."""
        return bark_to_hz(bark)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_psychoacoustic(audio: np.ndarray, sr: int) -> PsychoacousticAnalysis:
    """
    Quick psychoacoustic analysis.

    Args:
        audio: Input audio (mono)
        sr: Sample rate

    Returns:
        PsychoacousticAnalysis object
    """
    core = PsychoacousticCore()
    return core.analyze(audio, sr)


def apply_perceptual_loudness_compensation(audio: np.ndarray, sr: int, listening_level: str = "normal") -> np.ndarray:
    """
    Quick loudness compensation.

    Args:
        audio: Input audio
        sr: Sample rate
        listening_level: 'quiet', 'normal', 'loud'

    Returns:
        Compensated audio
    """
    core = PsychoacousticCore()
    return core.apply_loudness_compensation(audio, sr, listening_level)


if __name__ == "__main__":
    """Demo psychoacoustic core"""
    logger.debug("\n" + "=" * 70)
    logger.debug("PSYCHOACOUSTIC CORE - Demo")
    logger.debug("=" * 70 + "\n")

    # Generate test signal
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Complex multi-frequency signal
    audio = (
        0.3 * np.sin(2 * np.pi * 100 * t)  # Bass
        + 0.5 * np.sin(2 * np.pi * 1000 * t)  # Mid (fundamental)
        + 0.2 * np.sin(2 * np.pi * 3000 * t)  # Treble
        + 0.1 * np.sin(2 * np.pi * 10000 * t)  # Air
        + 0.02 * np.random.randn(len(t))  # Noise
    )
    audio = audio / np.abs(audio).max()

    # Initialize core
    core = PsychoacousticCore()

    # Comprehensive analysis
    logger.debug("Performing comprehensive psychoacoustic Analyse...")
    analysis = core.analyze(audio, sr)

    # Print summary
    analysis.print_summary()

    # Test loudness compensation
    logger.debug("\nTesting Loudness Compensation...")
    compensated_quiet = core.apply_loudness_compensation(audio, sr, listening_level="quiet")
    compensated_normal = core.apply_loudness_compensation(audio, sr, listening_level="normal")

    logger.debug("  Originalsignal RMS: %.4f", np.sqrt(np.mean(audio**2)))
    logger.debug("  Quiet (40 phon) RMS: %.4f", np.sqrt(np.mean(compensated_quiet**2)))
    logger.debug("  Normal (60 phon) RMS: %.4f", np.sqrt(np.mean(compensated_normal**2)))

    # Test masked component removal
    logger.debug("\nRemoving masked (inaudible) components...")
    audio_unmasked = core.remove_masked_components(audio, sr)

    logger.debug("  Originalsignal RMS: %.4f", np.sqrt(np.mean(audio**2)))
    logger.debug("  After masking removal: %.4f", np.sqrt(np.mean(audio_unmasked**2)))
    logger.debug("  Energy reduction: %.1f%%", (1 - np.mean(audio_unmasked**2) / np.mean(audio**2)) * 100)

    # Test perceptual EQ
    logger.debug("\nGenerating Perceptual EQ Curve...")
    freqs = np.logspace(np.log10(20), np.log10(20000), 50)
    target_balance = {
        "bass": 0.15,  # Less bass
        "mid": 0.60,  # More mids
        "treble": 0.20,  # Less treble
        "air": 0.05,  # Normal air
    }

    eq_curve = core.get_perceptual_eq_curve(freqs, target_balance)

    logger.debug("\n  Target Balance:")
    for band, energy in target_balance.items():
        logger.debug("    %s: %.1f%%", band.capitalize(), energy * 100)

    logger.debug("\n  Sample EQ Points:")
    for i in [0, 10, 20, 30, 40]:
        if i < len(freqs):
            logger.debug("    %6.0f Hz: %+.2f dB", freqs[i], eq_curve[i])

    logger.debug("\n" + "=" * 70)
    logger.debug("Demo vollstaendig!")
    logger.debug("=" * 70 + "\n")
