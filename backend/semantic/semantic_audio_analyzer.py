"""
Semantic Audio Understanding
=============================

World's first genre-agnostic semantic audio analysis for restoration.

This module provides:
- Instrument detection (vocals, drums, bass, guitar, synth, percussion)
- Content characterization (transient vs. sustained)
- Semantic tagging without genre classification
- Processing recommendations based on audio content

Use Cases (Both Modes):
- RESTORATION: Process different instruments appropriately (preserve drum transients, smooth vocals)
- HIGHEND STUDIO: Optimize each instrument independently (punch drums, clarity vocals)

Author: Aurik Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


def _normalize_aurik_mode(aurik_mode: str | None) -> str:
    """Normalisiert external mode aliases to canonical values.

    Returns:
        "restoration" or "studio2026" (legacy aliases remain accepted).
    """
    _m = str(aurik_mode or "restoration").strip().lower().replace("_", "").replace(" ", "")
    if _m in {"studio2026", "studio", "highendstudio", "maximum"}:
        return "studio2026"
    return "restoration"


# ============================================================================
# ENUMS
# ============================================================================


class InstrumentType(Enum):
    """Erkannte Instrumententypen."""

    VOCALS = "vocals"  # Singing voice
    SPEECH = "speech"  # Spoken word
    DRUMS = "drums"  # Percussion, rhythmic
    BASS = "bass"  # Low-frequency melodic
    GUITAR = "guitar"  # String instrument (electric/acoustic)
    KEYS = "keys"  # Piano, organ, synth pads
    SYNTH = "synth"  # Electronic/synthetic sounds
    BRASS = "brass"  # Trumpet, sax, trombone
    STRINGS = "strings"  # Violin, cello, orchestral
    PERCUSSION = "percussion"  # Non-drum percussion
    AMBIENT = "ambient"  # Atmospheric textures
    UNKNOWN = "unknown"  # Unclassified content


class ContentCharacter(Enum):
    """Audio content characteristics."""

    HIGHLY_TRANSIENT = "highly_transient"  # Drums, percussion, attacks
    TRANSIENT = "transient"  # Plucked strings, staccato
    BALANCED = "balanced"  # Mixed transient/sustained
    SUSTAINED = "sustained"  # Pads, strings, legato
    HIGHLY_SUSTAINED = "highly_sustained"  # Drones, ambient, sustained notes


class ProcessingStrategy(Enum):
    """Recommended processing approaches."""

    PRESERVE_TRANSIENTS = "preserve_transients"  # Don't smooth attacks
    GENTLE_SMOOTHING = "gentle_smoothing"  # Light processing
    BALANCED_PROCESSING = "balanced_processing"  # Standard approach
    AGGRESSIVE_SMOOTHING = "aggressive_smoothing"  # Heavy processing
    PRESERVE_TEXTURE = "preserve_texture"  # Keep character intact


# ============================================================================
# DATA STRUCTURES
# ============================================================================


class InstrumentPresence:
    """Represents detected instrument with confidence."""

    def __init__(
        self,
        instrument: InstrumentType,
        confidence: float,
        time_percentage: float,
        frequency_range: tuple[float, float],
        energy_contribution: float,
    ):
        self.instrument = instrument
        self.confidence = confidence
        self.time_percentage = time_percentage
        self.frequency_range = frequency_range
        self.energy_contribution = energy_contribution

    @property
    def instrument_type(self):
        """Alias für instrument (API-Kompatibilität)."""
        return self.instrument


@dataclass
class SemanticProfile:
    @property
    def instruments(self):
        """Alias für detected_instruments (API-Kompatibilität)."""
        return self.detected_instruments

    # Provides instrument-aware processing guidance without genre labels.
    # Instrument detection
    detected_instruments: list[InstrumentPresence]
    dominant_instrument: InstrumentType

    # Content characteristics
    content_character: ContentCharacter
    transient_density: float  # 0.0-1.0 (how many transients per second)
    sustained_percentage: float  # 0.0-1.0 (percentage of sustained content)

    # Frequency content
    bass_energy: float  # 0.0-1.0 (20-250 Hz energy)
    mid_energy: float  # 0.0-1.0 (250-2000 Hz energy)
    high_energy: float  # 0.0-1.0 (2000-20000 Hz energy)

    # Processing recommendations
    recommended_strategy: ProcessingStrategy
    preserve_transients: bool
    enhance_clarity: bool
    reduce_harshness: bool

    # Mode-specific guidance
    restoration_notes: str
    studio_notes: str

    @property
    def processing_strategy(self):
        """Alias für recommended_strategy (API-Kompatibilität)."""
        return self.recommended_strategy

    def get_instrument_by_type(self, instrument: InstrumentType) -> InstrumentPresence | None:
        """Gibt zurück: instrument presence by type."""
        for inst in self.detected_instruments:
            if inst.instrument == instrument:
                return inst
        return None

    def has_instrument(self, instrument: InstrumentType, min_confidence: float = 0.3) -> bool:
        """Prüft if instrument is present with minimum confidence."""
        inst = self.get_instrument_by_type(instrument)
        return inst is not None and inst.confidence >= min_confidence

    def __repr__(self) -> str:
        instruments = ", ".join(
            f"{i.instrument.value}({i.confidence:.0%})"
            for i in sorted(self.detected_instruments, key=lambda x: x.confidence, reverse=True)[:3]
        )
        return (
            f"SemanticProfile(dominant={self.dominant_instrument.value}, "
            f"character={self.content_character.value}, instruments=[{instruments}])"
        )


# ============================================================================
# SEMANTIC AUDIO ANALYZER
# ============================================================================


class SemanticAudioAnalyzer:
    """
    Analysiert Audio semantisch ohne Genreklassifizierung.

    Detects instruments and content characteristics using
    intrinsic audio features only.
    """

    def __init__(self):
        """Initialisiert semantic analyzer."""
        logger.info("SemanticAudioAnalyzer initialisiert")

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        aurik_mode: str = "restoration",
    ) -> SemanticProfile:
        """
        Führt durch: semantic audio analysis.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            aurik_mode: "restoration" or "studio2026" (legacy aliases accepted)

        Returns:
            SemanticProfile with instrument detection and processing guidance
        """
        aurik_mode = _normalize_aurik_mode(aurik_mode)

        # Convert to mono
        if audio.ndim > 1:
            # Auto-detect stereo format: (samples, channels) vs (channels, samples)
            # If shape[0] < shape[1], it's likely (channels, samples) format (librosa-style)
            # If shape[1] <= 32 (reasonable max channels), it's likely (samples, channels) format
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples) - average across channels (axis=0)
                audio_mono = np.mean(audio, axis=0)
            else:
                # Format: (samples, channels) - average across channels (axis=1)
                audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        duration = len(audio_mono) / sr
        logger.debug("analysiere %.2fs audio semantically", duration)

        # 1. Detect instruments
        instruments = self._detect_instruments(audio_mono, sr)
        dominant = self._get_dominant_instrument(instruments)

        # 2. Analyze content characteristics
        content_char = self._analyze_content_character(audio_mono, sr)
        transient_density = self._compute_transient_density(audio_mono, sr)
        sustained_pct = self._compute_sustained_percentage(audio_mono, sr)

        # 3. Frequency analysis
        bass_energy, mid_energy, high_energy = self._analyze_frequency_bands(audio_mono, sr)

        # 4. Processing recommendations
        strategy = self._recommend_strategy(dominant, content_char, transient_density, aurik_mode)
        preserve_transients = self._should_preserve_transients(content_char, instruments)
        enhance_clarity = self._should_enhance_clarity(dominant, instruments)
        reduce_harshness = self._should_reduce_harshness(high_energy, dominant)

        # 5. Mode-specific notes
        restoration_notes = self._generate_restoration_notes(dominant, content_char, instruments)
        studio_notes = self._generate_studio_notes(dominant, content_char, instruments)

        return SemanticProfile(
            detected_instruments=instruments,
            dominant_instrument=dominant,
            content_character=content_char,
            transient_density=transient_density,
            sustained_percentage=sustained_pct,
            bass_energy=bass_energy,
            mid_energy=mid_energy,
            high_energy=high_energy,
            recommended_strategy=strategy,
            preserve_transients=preserve_transients,
            enhance_clarity=enhance_clarity,
            reduce_harshness=reduce_harshness,
            restoration_notes=restoration_notes,
            studio_notes=studio_notes,
        )

    # ========================================================================
    # INSTRUMENT DETECTION
    # ========================================================================

    def _detect_instruments(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[InstrumentPresence]:
        """
        Erkennt instruments using spectral and temporal features.

        This is a heuristic approach. Production version would use
        trained ML models (e.g., Demucs v4, MDX23C, or custom models).
        """
        instruments = []

        # Compute STFT
        f, _t, Zxx = signal.stft(audio, sr, nperseg=2048, boundary="even")
        power = np.abs(Zxx) ** 2

        # 1. Vocals detection (formant structure + harmonic energy)
        vocal_conf = self._detect_vocals(audio, sr, power, f)
        if vocal_conf > 0.1:
            instruments.append(
                InstrumentPresence(
                    instrument=InstrumentType.VOCALS,
                    confidence=vocal_conf,
                    time_percentage=0.6,  # Placeholder
                    frequency_range=(100.0, 5000.0),
                    energy_contribution=0.4,
                )
            )

        # 2. Drums detection (transient energy + rhythm)
        drum_conf = self._detect_drums(audio, sr)
        if drum_conf > 0.1:
            instruments.append(
                InstrumentPresence(
                    instrument=InstrumentType.DRUMS,
                    confidence=drum_conf,
                    time_percentage=0.7,
                    frequency_range=(50.0, 10000.0),
                    energy_contribution=0.3,
                )
            )

        # 3. Bass detection (low-frequency melodic content)
        bass_conf = self._detect_bass(power, f)
        if bass_conf > 0.1:
            instruments.append(
                InstrumentPresence(
                    instrument=InstrumentType.BASS,
                    confidence=bass_conf,
                    time_percentage=0.5,
                    frequency_range=(40.0, 250.0),
                    energy_contribution=0.2,
                )
            )

        # 4. Guitar detection (harmonic structure + mid-range)
        guitar_conf = self._detect_guitar(power, f)
        if guitar_conf > 0.1:
            instruments.append(
                InstrumentPresence(
                    instrument=InstrumentType.GUITAR,
                    confidence=guitar_conf,
                    time_percentage=0.5,
                    frequency_range=(80.0, 5000.0),
                    energy_contribution=0.25,
                )
            )

        # 5. Keys/synth detection
        keys_conf = self._detect_keys(power, f)
        if keys_conf > 0.1:
            instruments.append(
                InstrumentPresence(
                    instrument=InstrumentType.KEYS,
                    confidence=keys_conf,
                    time_percentage=0.6,
                    frequency_range=(50.0, 8000.0),
                    energy_contribution=0.3,
                )
            )

        # 6. Ambient/texture detection
        ambient_conf = self._detect_ambient(audio, sr)
        if ambient_conf > 0.1:
            instruments.append(
                InstrumentPresence(
                    instrument=InstrumentType.AMBIENT,
                    confidence=ambient_conf,
                    time_percentage=0.8,
                    frequency_range=(20.0, 20000.0),
                    energy_contribution=0.2,
                )
            )

        # If nothing detected, mark as unknown
        if len(instruments) == 0:
            instruments.append(
                InstrumentPresence(
                    instrument=InstrumentType.UNKNOWN,
                    confidence=1.0,
                    time_percentage=1.0,
                    frequency_range=(20.0, 20000.0),
                    energy_contribution=1.0,
                )
            )

        return instruments

    def _detect_vocals(
        self,
        audio: np.ndarray,
        sr: int,
        power: np.ndarray,
        f: np.ndarray,
    ) -> float:
        """Erkennt vocal presence (formants + harmonic structure)."""
        # Vocal formants typically in 300-3000 Hz range
        vocal_range_mask = (f >= 300) & (f <= 3000)
        vocal_energy = np.mean(power[vocal_range_mask, :])
        total_energy = np.mean(power)

        if total_energy == 0:
            return 0.0

        # Vocal energy ratio
        vocal_ratio = vocal_energy / total_energy

        # Check for harmonic structure (vocals are harmonic)
        harmonicity = self._compute_harmonicity(audio, sr)

        # Combine features
        confidence = min(1.0, (vocal_ratio * 2.0 + harmonicity) / 3.0)

        return float(confidence)

    def _detect_drums(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt drum presence (transient detection + rhythm)."""
        # Compute onset strength
        onset_env = self._compute_onset_envelope(audio, sr)

        # High onset strength = drum-like
        onset_strength = np.mean(onset_env)

        # Transient density
        transients = self._detect_transients(audio, sr)
        transient_density = len(transients) / (len(audio) / sr)

        # Normalize to 0-1
        drum_confidence = min(1.0, (onset_strength * 10.0 + transient_density / 5.0) / 2.0)

        return float(drum_confidence)

    def _detect_bass(self, power: np.ndarray, f: np.ndarray) -> float:
        """Erkennt bass presence (low-frequency melodic content)."""
        # Bass range: 40-250 Hz
        bass_mask = (f >= 40) & (f <= 250)
        bass_energy = np.mean(power[bass_mask, :])
        total_energy = np.mean(power)

        if total_energy == 0:
            return 0.0

        bass_ratio = bass_energy / total_energy

        # Bass confidence
        confidence = min(1.0, bass_ratio * 3.0)

        return float(confidence)

    def _detect_guitar(self, power: np.ndarray, f: np.ndarray) -> float:
        """Erkennt guitar presence (mid-range harmonic content)."""
        # Guitar fundamental range: ~80-1000 Hz
        guitar_mask = (f >= 80) & (f <= 1000)
        guitar_energy = np.mean(power[guitar_mask, :])
        total_energy = np.mean(power)

        if total_energy == 0:
            return 0.0

        # Check for harmonic richness in mid-range
        guitar_ratio = guitar_energy / total_energy

        confidence = min(1.0, guitar_ratio * 2.5)

        return float(confidence)

    def _detect_keys(self, power: np.ndarray, f: np.ndarray) -> float:
        """Erkennt keyboard/synth presence."""
        # Keys/synth have energy across wide frequency range
        wide_range_mask = (f >= 50) & (f <= 8000)
        keys_energy = np.mean(power[wide_range_mask, :])
        total_energy = np.mean(power)

        if total_energy == 0:
            return 0.0

        # Broadband energy indicates keys/synth
        keys_ratio = keys_energy / total_energy

        confidence = min(1.0, keys_ratio * 1.5)

        return float(confidence)

    def _detect_ambient(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt ambient/atmospheric content."""
        # Ambient = low transient density + sustained energy
        transients = self._detect_transients(audio, sr)
        transient_density = len(transients) / (len(audio) / sr)

        # Low transient density suggests ambient
        ambient_score = max(0.0, 1.0 - transient_density / 2.0)

        # Check for sustained energy
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0.01:
            ambient_score = min(1.0, ambient_score * 1.5)

        return float(ambient_score)

    def _get_dominant_instrument(
        self,
        instruments: list[InstrumentPresence],
    ) -> InstrumentType:
        """Bestimmt dominant instrument."""
        if len(instruments) == 0:
            return InstrumentType.UNKNOWN

        # Sort by confidence
        sorted_instruments = sorted(instruments, key=lambda x: x.confidence, reverse=True)
        return sorted_instruments[0].instrument

    # ========================================================================
    # CONTENT CHARACTER ANALYSIS
    # ========================================================================

    def _analyze_content_character(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> ContentCharacter:
        """Analysiert content character (transient vs. sustained)."""
        transient_density = self._compute_transient_density(audio, sr)

        # Classify based on transient density
        if transient_density > 8.0:
            return ContentCharacter.HIGHLY_TRANSIENT
        elif transient_density > 4.0:
            return ContentCharacter.TRANSIENT
        elif transient_density > 1.5:
            return ContentCharacter.BALANCED
        elif transient_density > 0.5:
            return ContentCharacter.SUSTAINED
        else:
            return ContentCharacter.HIGHLY_SUSTAINED

    def _compute_transient_density(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet transients per second."""
        transients = self._detect_transients(audio, sr)
        duration = len(audio) / sr

        if duration == 0:
            return 0.0

        return len(transients) / duration

    def _detect_transients(self, audio: np.ndarray, sr: int) -> list[int]:
        """Erkennt transient positions."""
        # Compute onset envelope
        onset_env = self._compute_onset_envelope(audio, sr)

        # Find peaks (transients)
        threshold = np.mean(onset_env) + 2.0 * np.std(onset_env)
        peaks, _ = signal.find_peaks(onset_env, height=threshold, distance=int(0.05 * sr))

        return peaks.tolist()  # type: ignore[no-any-return]

    def _compute_onset_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Berechnet onset strength envelope."""
        # Compute STFT
        _f, _t, Zxx = signal.stft(audio, sr, nperseg=2048, noverlap=1536, boundary="even")

        # Spectral flux (frame-to-frame change)
        spec_diff = np.diff(np.abs(Zxx), axis=1)
        spec_diff = np.maximum(spec_diff, 0)  # Half-wave rectify

        # Sum across frequencies
        onset_env = np.sum(spec_diff, axis=0)

        return onset_env  # type: ignore[no-any-return]

    def _compute_sustained_percentage(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet percentage of audio that is sustained."""
        # Sustained content has consistent energy

        # Segment audio into frames
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = frame_length // 2

        sustained_frames = 0
        total_frames = 0

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]

            # Check if frame is sustained (low variation)
            rms = np.sqrt(np.mean(frame**2))
            std = np.std(frame)

            if rms > 0.01:  # Not silence
                total_frames += 1
                if std / (rms + 1e-10) < 0.5:  # Low variation = sustained
                    sustained_frames += 1

        if total_frames == 0:
            return 0.0

        return sustained_frames / total_frames

    def _compute_harmonicity(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet harmonicity (0=noise, 1=harmonic)."""
        # Autocorrelation-based pitch detection

        # Segment and average
        frame_length = int(0.05 * sr)  # 50ms
        hop_length = frame_length // 2

        harmonicities = []

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]

            # Autocorrelation — FFT-based O(N log N)
            from backend.core.core_utils import fft_autocorr

            autocorr = fft_autocorr(frame)

            # Normalize
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]

                # Find peak (excluding zero lag)
                if len(autocorr) > 100:
                    peak_value = np.max(autocorr[100:])
                    harmonicities.append(peak_value)

        if len(harmonicities) == 0:
            return 0.0

        return float(np.mean(harmonicities))

    # ========================================================================
    # FREQUENCY ANALYSIS
    # ========================================================================

    def _analyze_frequency_bands(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[float, float, float]:
        """Analysiert energy in bass/mid/high bands."""
        # Compute power spectrum
        audio_arr = np.asarray(audio, dtype=np.float64)
        freqs = np.fft.fftfreq(len(audio_arr), 1 / sr)
        spectrum = np.abs(np.fft.fft(audio_arr))

        # Positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        spectrum = spectrum[pos_mask]

        # Energy in bands
        bass_mask = (freqs >= 20) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs < 2000)
        high_mask = (freqs >= 2000) & (freqs < 20000)

        bass_energy = np.sum(spectrum[bass_mask] ** 2)
        mid_energy = np.sum(spectrum[mid_mask] ** 2)
        high_energy = np.sum(spectrum[high_mask] ** 2)

        total_energy = bass_energy + mid_energy + high_energy

        if total_energy == 0:
            return 0.0, 0.0, 0.0

        return (
            float(bass_energy / total_energy),
            float(mid_energy / total_energy),
            float(high_energy / total_energy),
        )

    # ========================================================================
    # PROCESSING RECOMMENDATIONS
    # ========================================================================

    def _recommend_strategy(
        self,
        dominant: InstrumentType,
        content_char: ContentCharacter,
        transient_density: float,
        aurik_mode: str,
    ) -> ProcessingStrategy:
        """Recommend processing strategy based on content."""
        # Drums/percussion: preserve transients
        if dominant in [InstrumentType.DRUMS, InstrumentType.PERCUSSION]:
            return ProcessingStrategy.PRESERVE_TRANSIENTS

        # Vocals: balanced or gentle
        if dominant in [InstrumentType.VOCALS, InstrumentType.SPEECH]:
            if aurik_mode == "restoration":
                return ProcessingStrategy.GENTLE_SMOOTHING
            else:
                return ProcessingStrategy.BALANCED_PROCESSING

        # High transient density: preserve
        if content_char in [ContentCharacter.HIGHLY_TRANSIENT, ContentCharacter.TRANSIENT]:
            return ProcessingStrategy.PRESERVE_TRANSIENTS

        # Ambient/sustained: can smooth more
        if content_char in [ContentCharacter.HIGHLY_SUSTAINED]:
            return ProcessingStrategy.AGGRESSIVE_SMOOTHING

        # Default
        return ProcessingStrategy.BALANCED_PROCESSING

    def _should_preserve_transients(
        self,
        content_char: ContentCharacter,
        instruments: list[InstrumentPresence],
    ) -> bool:
        """Bestimmt if transients should be preserved."""
        # Preserve for drums/percussion
        for inst in instruments:
            if inst.instrument in [InstrumentType.DRUMS, InstrumentType.PERCUSSION]:
                if inst.confidence > 0.3:
                    return True

        # Preserve for highly transient content
        return content_char in [ContentCharacter.HIGHLY_TRANSIENT, ContentCharacter.TRANSIENT]

    def _should_enhance_clarity(
        self,
        dominant: InstrumentType,
        instruments: list[InstrumentPresence],
    ) -> bool:
        """Bestimmt if clarity enhancement is beneficial."""
        # Enhance clarity for vocals
        if dominant in [InstrumentType.VOCALS, InstrumentType.SPEECH]:
            return True

        # Enhance if vocals are present
        return any(
            inst.instrument in [InstrumentType.VOCALS, InstrumentType.SPEECH] and inst.confidence > 0.4
            for inst in instruments
        )

    def _should_reduce_harshness(
        self,
        high_energy: float,
        dominant: InstrumentType,
    ) -> bool:
        """Bestimmt if harshness reduction is beneficial."""
        # Reduce harshness if high-frequency energy is excessive
        if high_energy > 0.4:
            return True

        # Reduce for synths (can be harsh)
        return dominant == InstrumentType.SYNTH

    # ========================================================================
    # MODE-SPECIFIC NOTES
    # ========================================================================

    def _generate_restoration_notes(
        self,
        dominant: InstrumentType,
        content_char: ContentCharacter,
        instruments: list[InstrumentPresence],
    ) -> str:
        """Generiert restoration mode notes."""
        inst_str = ", ".join([i.instrument.value for i in instruments[:3]])

        if dominant == InstrumentType.VOCALS:
            return (
                f"VOCAL RESTORATION: Detected {inst_str}. "
                "Preserve natural vocal texture, gentle de-essing, "
                "maintain breath character."
            )
        elif dominant == InstrumentType.DRUMS:
            return (
                f"PERCUSSIVE RESTORATION: Detected {inst_str}. "
                "Preserve transient attacks, careful with compression, "
                "maintain original dynamics."
            )
        elif content_char in [ContentCharacter.HIGHLY_TRANSIENT, ContentCharacter.TRANSIENT]:
            return (
                f"TRANSIENT-RICH RESTORATION: Detected {inst_str}. Preserve attacks and dynamics, avoid over-smoothing."
            )
        elif content_char == ContentCharacter.HIGHLY_SUSTAINED:
            return (
                f"SUSTAINED RESTORATION: Detected {inst_str}. "
                "Focus on tonal balance, gentle noise reduction, "
                "preserve sustained character."
            )
        else:
            return f"BALANCED RESTORATION: Detected {inst_str}. Standard restoration approach with content awareness."

    def _generate_studio_notes(
        self,
        dominant: InstrumentType,
        content_char: ContentCharacter,
        instruments: list[InstrumentPresence],
    ) -> str:
        """Generiert studio mode notes."""
        inst_str = ", ".join([i.instrument.value for i in instruments[:3]])

        if dominant == InstrumentType.VOCALS:
            return (
                f"VOCAL PRODUCTION: Detected {inst_str}. "
                "Apply modern vocal chain: de-essing, compression, EQ for clarity. "
                "Optimize for streaming platforms."
            )
        elif dominant == InstrumentType.DRUMS:
            return (
                f"DRUM PRODUCTION: Detected {inst_str}. "
                "Enhance punch and clarity, parallel compression, "
                "optimize transients for impact."
            )
        elif dominant == InstrumentType.BASS:
            return (
                f"BASS PRODUCTION: Detected {inst_str}. Tighten low-end, enhance definition, ensure mix compatibility."
            )
        elif content_char in [ContentCharacter.HIGHLY_TRANSIENT, ContentCharacter.TRANSIENT]:
            return f"TRANSIENT PRODUCTION: Detected {inst_str}. Shape transients for modern punch, maintain clarity."
        else:
            return f"BALANCED PRODUCTION: Detected {inst_str}. Modern processing for streaming/broadcast standards."


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def analyze_semantic_content(
    audio: np.ndarray,
    sr: int,
    aurik_mode: str = "restoration",
) -> SemanticProfile:
    """
    Convenience function for semantic audio analysis.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
            aurik_mode: "restoration" or "studio2026" (legacy aliases accepted)

    Returns:
        SemanticProfile with instrument detection and processing guidance

    Example:
        >>> profile = analyze_semantic_content(audio, sr=48000, aurik_mode="restoration")
        >>> print(f"Dominant: {profile.dominant_instrument.value}")
        >>> print(f"Character: {profile.content_character.value}")
        >>> print(f"Preserve transients: {profile.preserve_transients}")
        >>>
        >>> # Check for specific instruments
        >>> if profile.has_instrument(InstrumentType.VOCALS):
        >>>     print("Vocals detected - apply vocal-specific processing")
    """
    analyzer = SemanticAudioAnalyzer()
    return analyzer.analyze(audio, sr, aurik_mode)
