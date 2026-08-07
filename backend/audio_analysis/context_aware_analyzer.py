"""
Context-Aware Audio Analysis
=============================

Genre-agnostic audio understanding through intrinsic audio features.
Replaces genre classification with objective audio characteristics.

This module analyzes audio content WITHOUT genre assumptions:
- Vocal Density: How much vocal vs instrumental content
- Dynamic Profile: Energy distribution and compression
- Spectral Signature: Frequency content and complexity
- Temporal Dynamics: Rhythmic and energetic characteristics
- Harmonic Complexity: Musical richness and structure

Author: Aurik Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class VocalDensity(Enum):
    """Vocal presence classification."""

    INSTRUMENTAL = "instrumental"  # <20% vocal content
    SPARSE_VOCAL = "sparse_vocal"  # 20-40% vocal
    BALANCED = "balanced"  # 40-60% vocal/instrumental
    VOCAL_DOMINANT = "vocal_dominant"  # 60-80% vocal
    VOCAL_ONLY = "vocal_only"  # >80% vocal (a cappella, speech)


class DynamicProfile(Enum):
    """Dynamic range characteristics."""

    HIGHLY_COMPRESSED = "highly_compressed"  # <6 dB crest factor
    COMPRESSED = "compressed"  # 6-10 dB crest factor
    MODERATE = "moderate"  # 10-14 dB crest factor
    DYNAMIC = "dynamic"  # 14-18 dB crest factor
    HIGHLY_DYNAMIC = "highly_dynamic"  # >18 dB crest factor (classical, jazz)


class SpectralProfile(Enum):
    """Spectral distribution characteristics."""

    BASS_HEAVY = "bass_heavy"  # Low-frequency dominant
    BALANCED = "balanced"  # Even distribution
    BRIGHT = "bright"  # High-frequency emphasis
    THIN = "thin"  # Missing low frequencies
    HARSH = "harsh"  # Excessive high frequencies


class TemporalCharacter(Enum):
    """Temporal/rhythmic characteristics."""

    AMBIENT = "ambient"  # Minimal transients, sustained
    GENTLE = "gentle"  # Soft transients, flowing
    MODERATE = "moderate"  # Balanced energy
    ENERGETIC = "energetic"  # Strong transients, active
    AGGRESSIVE = "aggressive"  # Very strong transients, intense


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class AudioContext:
    """
    Genre-agnostic audio context analysis.

    Based purely on measurable audio features, not genre labels.
    Provides actionable information for processing decisions.
    """

    # Core characteristics
    vocal_density: VocalDensity
    dynamic_profile: DynamicProfile
    spectral_profile: SpectralProfile
    temporal_character: TemporalCharacter

    # Detailed metrics
    vocal_percentage: float  # 0.0-1.0
    crest_factor_db: float  # Peak/RMS ratio in dB
    spectral_centroid_hz: float  # "Brightness" center frequency
    spectral_rolloff_hz: float  # 85% energy rolloff frequency
    zero_crossing_rate: float  # Transient/noise indicator
    harmonic_to_noise_ratio_db: float  # Musical vs noisy content

    # Processing recommendations (derived from features)
    recommended_deessing_strength: float  # 0.0-1.0
    recommended_denoise_strength: float  # 0.0-1.0
    preserve_transients: bool
    preserve_ambience: bool
    intelligibility_priority: float  # 0.0 (music) - 1.0 (speech)

    def __repr__(self) -> str:
        return (
            f"AudioContext(vocal={self.vocal_density.value}, "
            f"dynamics={self.dynamic_profile.value}, "
            f"spectral={self.spectral_profile.value})"
        )


# ============================================================================
# CONTEXT ANALYZER
# ============================================================================


class ContextAwareAnalyzer:
    """
    Genre-agnostic audio context analyzer.

    Analyzes intrinsic audio characteristics to guide processing decisions.
    No genre classification required - works for ANY audio content.

    Example:
        >>> analyzer = ContextAwareAnalyzer()
        >>> context = analyzer.analyze(audio, sr=48000)
        >>> logger.info(f"Vocal density: {context.vocal_density.value}")
        >>> logger.info(f"Recommended deessing: {context.recommended_deessing_strength}")
    """

    def __init__(self):
        """Initialisiert context analyzer."""
        logger.info("ContextAwareAnalyzer initialisiert (genre-agnostic)")

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> AudioContext:
        """
        Analysiert den Audio-Kontext aus intrinsischen Merkmalen.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate

        Returns:
            AudioContext with comprehensive analysis
        """
        # Convert to mono for analysis
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        logger.debug("analysiere %.2fs audio context", len(audio_mono) / sr)

        # 1. Vocal density analysis
        vocal_density, vocal_percentage = self._analyze_vocal_density(audio_mono, sr)

        # 2. Dynamic profile analysis
        dynamic_profile, crest_factor = self._analyze_dynamics(audio_mono)

        # 3. Spectral profile analysis
        spectral_profile, spectral_metrics = self._analyze_spectrum(audio_mono, sr)

        # 4. Temporal character analysis
        temporal_character, temporal_metrics = self._analyze_temporal_character(audio_mono, sr)

        # 5. Harmonic analysis
        hnr_db = self._analyze_harmonic_content(audio_mono, sr)

        # 6. Derive processing recommendations
        recommendations = self._derive_recommendations(
            vocal_density=vocal_density,
            dynamic_profile=dynamic_profile,
            spectral_profile=spectral_profile,
            temporal_character=temporal_character,
            vocal_percentage=vocal_percentage,
            crest_factor=crest_factor,
        )

        return AudioContext(
            vocal_density=vocal_density,
            dynamic_profile=dynamic_profile,
            spectral_profile=spectral_profile,
            temporal_character=temporal_character,
            vocal_percentage=vocal_percentage,
            crest_factor_db=crest_factor,
            spectral_centroid_hz=spectral_metrics["centroid"],
            spectral_rolloff_hz=spectral_metrics["rolloff"],
            zero_crossing_rate=temporal_metrics["zcr"],
            harmonic_to_noise_ratio_db=hnr_db,
            **recommendations,
        )

    def _analyze_vocal_density(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[VocalDensity, float]:
        """
        Analysiert Vokal- vs. Instrumental-Inhalt.

        Uses spectral characteristics typical of vocal vs instrumental:
        - Vocals: 80-1000 Hz fundamentals, strong 1-4kHz presence
        - Instruments: Broader spectral distribution
        """
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Vocal formant regions (strong indicator)
        vocal_region_1 = (freqs >= 300) & (freqs <= 1000)  # F1 region
        vocal_region_2 = (freqs >= 1000) & (freqs <= 3500)  # F2/F3 region

        # Instrumental regions
        bass_region = (freqs >= 40) & (freqs <= 300)
        high_region = (freqs >= 3500) & (freqs <= 8000)

        # Compute energies
        vocal_energy = np.sum(spectrum[vocal_region_1]) + np.sum(spectrum[vocal_region_2])
        instrumental_energy = np.sum(spectrum[bass_region]) + np.sum(spectrum[high_region])

        total_energy = vocal_energy + instrumental_energy
        vocal_percentage = 0.5 if total_energy == 0 else vocal_energy / total_energy

        # Classify density
        if vocal_percentage < 0.2:
            density = VocalDensity.INSTRUMENTAL
        elif vocal_percentage < 0.4:
            density = VocalDensity.SPARSE_VOCAL
        elif vocal_percentage < 0.6:
            density = VocalDensity.BALANCED
        elif vocal_percentage < 0.8:
            density = VocalDensity.VOCAL_DOMINANT
        else:
            density = VocalDensity.VOCAL_ONLY

        return density, float(vocal_percentage)

    def _analyze_dynamics(
        self,
        audio: np.ndarray,
    ) -> tuple[DynamicProfile, float]:
        """
        Analysiert Dynamikbereichseigenschaften.

        Crest factor = 20*log10(peak/RMS)
        """
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))

        crest_factor_db = 0.0 if rms == 0 else 20 * np.log10(peak / rms)

        # Classify profile
        if crest_factor_db < 6:
            profile = DynamicProfile.HIGHLY_COMPRESSED
        elif crest_factor_db < 10:
            profile = DynamicProfile.COMPRESSED
        elif crest_factor_db < 14:
            profile = DynamicProfile.MODERATE
        elif crest_factor_db < 18:
            profile = DynamicProfile.DYNAMIC
        else:
            profile = DynamicProfile.HIGHLY_DYNAMIC

        return profile, float(crest_factor_db)

    def _analyze_spectrum(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[SpectralProfile, dict[str, float]]:
        """
        Analysiert die spektrale Verteilung.
        """
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Spectral centroid (brightness)
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        # Spectral rolloff (85% energy point)
        cumsum = np.cumsum(spectrum)
        rolloff_threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

        # Energy distribution
        low_band = (freqs >= 20) & (freqs <= 300)
        mid_band = (freqs >= 300) & (freqs <= 4000)
        high_band = (freqs >= 4000) & (freqs <= sr / 2)

        low_energy = np.sum(spectrum[low_band])
        mid_energy = np.sum(spectrum[mid_band])
        high_energy = np.sum(spectrum[high_band])

        total_energy = low_energy + mid_energy + high_energy
        if total_energy == 0:
            low_ratio = mid_ratio = high_ratio = 0.33
        else:
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy

        # Classify profile
        if low_ratio > 0.5:
            profile = SpectralProfile.BASS_HEAVY
        elif high_ratio > 0.4:
            profile = SpectralProfile.BRIGHT
        elif high_ratio > 0.5:
            profile = SpectralProfile.HARSH
        elif low_ratio < 0.2:
            profile = SpectralProfile.THIN
        else:
            profile = SpectralProfile.BALANCED

        return profile, {
            "centroid": float(centroid),
            "rolloff": float(rolloff),
            "low_ratio": float(low_ratio),
            "mid_ratio": float(mid_ratio),
            "high_ratio": float(high_ratio),
        }

    def _analyze_temporal_character(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[TemporalCharacter, dict[str, float]]:
        """
        Analysiert temporale/rhythmische Eigenschaften.
        """
        # Zero crossing rate (transient indicator)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
        zcr = zero_crossings / len(audio)

        # RMS energy contour (smoothed)
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = frame_length // 2

        energy_contour = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            energy = np.sqrt(np.mean(frame**2))
            energy_contour.append(energy)

        energy_contour = np.array(energy_contour)  # type: ignore[assignment]

        # Energy variance (how much dynamics)
        energy_std = np.std(energy_contour) if len(energy_contour) > 0 else 0.0
        energy_mean = np.mean(energy_contour) if len(energy_contour) > 0 else 0.0

        if energy_mean == 0:
            energy_cv = 0.0
        else:
            energy_cv = energy_std / energy_mean  # Coefficient of variation

        # Classify character
        if energy_cv < 0.2 and zcr < 0.05:
            character = TemporalCharacter.AMBIENT
        elif energy_cv < 0.4 and zcr < 0.1:
            character = TemporalCharacter.GENTLE
        elif energy_cv < 0.7:
            character = TemporalCharacter.MODERATE
        elif energy_cv < 1.0:
            character = TemporalCharacter.ENERGETIC
        else:
            character = TemporalCharacter.AGGRESSIVE

        return character, {
            "zcr": float(zcr),
            "energy_cv": float(energy_cv),
        }

    def _analyze_harmonic_content(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Schätzt harmonic-to-noise ratio.

        Higher HNR = more musical/pitched content
        Lower HNR = more noisy/unpitched content
        """
        # Autocorrelation-based HNR estimation
        max_lag = int(sr / 50)  # Up to 50 Hz (low pitch limit)

        from backend.core.core_utils import fft_autocorr

        autocorr = fft_autocorr(audio, max_lag=max_lag)

        # Normalize
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Find first peak (fundamental period)
        if len(autocorr) > max_lag:
            peak_ac = np.max(autocorr[int(sr / 500) : max_lag])  # Ignore very low lags
        else:
            peak_ac = 0.5

        # HNR approximation
        if peak_ac < 0.1:
            hnr_db = -10.0  # Very noisy
        elif peak_ac > 0.9:
            hnr_db = 20.0  # Very harmonic
        else:
            # Linear mapping (rough approximation)
            hnr_db = (peak_ac - 0.5) * 40.0

        return float(np.clip(hnr_db, -10.0, 30.0))

    def _derive_recommendations(
        self,
        vocal_density: VocalDensity,
        dynamic_profile: DynamicProfile,
        spectral_profile: SpectralProfile,
        temporal_character: TemporalCharacter,
        vocal_percentage: float,
        crest_factor: float,
    ) -> dict:
        """
        Derive processing recommendations from audio characteristics.

        This replaces genre-based heuristics with data-driven decisions.
        """
        # De-essing strength (based on vocal content and dynamics)
        if vocal_density in [VocalDensity.VOCAL_ONLY, VocalDensity.VOCAL_DOMINANT]:
            # High vocal content → more potential for sibilance issues
            deessing_base = 0.7
        elif vocal_density == VocalDensity.BALANCED:
            deessing_base = 0.5
        else:
            # Mostly instrumental → minimal de-essing
            deessing_base = 0.3

        # Adjust for dynamics (compressed audio needs less de-essing)
        if dynamic_profile == DynamicProfile.HIGHLY_COMPRESSED:
            deessing_strength = deessing_base * 0.7
        elif dynamic_profile == DynamicProfile.COMPRESSED:
            deessing_strength = deessing_base * 0.85
        else:
            deessing_strength = deessing_base

        # Denoise strength (based on harmonic content and dynamics)
        if dynamic_profile in [DynamicProfile.HIGHLY_DYNAMIC, DynamicProfile.DYNAMIC]:
            # High dynamics → preserve carefully
            denoise_base = 0.3
        else:
            denoise_base = 0.5

        # Adjust for spectral profile
        if spectral_profile == SpectralProfile.HARSH:
            # Already harsh → careful with brightening
            denoise_strength = denoise_base * 0.8
        else:
            denoise_strength = denoise_base

        # Transient preservation (based on temporal character)
        preserve_transients = temporal_character in [
            TemporalCharacter.ENERGETIC,
            TemporalCharacter.AGGRESSIVE,
        ]

        # Ambience preservation (based on dynamics and temporal)
        preserve_ambience = dynamic_profile in [
            DynamicProfile.DYNAMIC,
            DynamicProfile.HIGHLY_DYNAMIC,
        ] and temporal_character in [TemporalCharacter.AMBIENT, TemporalCharacter.GENTLE]

        # Intelligibility priority (speech-like vs music-like)
        # Based on vocal density and dynamic characteristics
        if vocal_density == VocalDensity.VOCAL_ONLY:
            intelligibility_priority = 1.0  # Pure speech/singing
        elif vocal_density == VocalDensity.VOCAL_DOMINANT:
            intelligibility_priority = 0.8
        elif vocal_density == VocalDensity.BALANCED:
            intelligibility_priority = 0.5
        else:
            intelligibility_priority = 0.2  # Mostly instrumental

        return {
            "recommended_deessing_strength": float(np.clip(deessing_strength, 0.0, 1.0)),
            "recommended_denoise_strength": float(np.clip(denoise_strength, 0.0, 1.0)),
            "preserve_transients": preserve_transients,
            "preserve_ambience": preserve_ambience,
            "intelligibility_priority": float(intelligibility_priority),
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def analyze_audio_context(
    audio: np.ndarray,
    sr: int,
) -> AudioContext:
    """
    Convenience function for audio context analysis.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate

    Returns:
        AudioContext with comprehensive analysis

    Example:
        >>> context = analyze_audio_context(audio, sr=48000)
        >>> logger.info(f"Vocal density: {context.vocal_density.value}")
        >>> logger.info(f"Recommended deessing: {context.recommended_deessing_strength}")
    """
    analyzer = ContextAwareAnalyzer()
    return analyzer.analyze(audio, sr)
