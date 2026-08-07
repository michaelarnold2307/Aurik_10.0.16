"""
Integrated Lyrics-Guided Vocal Enhancement with Semantic Understanding
========================================================================

Combines lyrics-aware processing with semantic audio analysis for
comprehensive vocal enhancement that respects:
- Phoneme-level characteristics (sibilants, vowels, plosives)
- Word boundaries and breath timing
- Musical context and instrument presence
- Content character (transient vs sustained)

Author: AURIK Development Team
Version: 1.0.0
Date: 11. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.lyrics_guided.content_aware_processor import (
    ContentAwareProcessor,
    ContentSegment,
    ContentType,
    LyricsGuidedTimeline,
)
from backend.lyrics_guided.lyrics_aligner import (
    LyricsAligner,
    LyricsAlignment,
    PhonemeAlignment,
    WordAlignment,
)
from backend.semantic.semantic_audio_analyzer import (
    ContentCharacter,
    InstrumentType,
    SemanticAudioAnalyzer,
    SemanticProfile,  # type: ignore[misc]
)

logger = logging.getLogger(__name__)


def _normalize_aurik_mode(aurik_mode: str | None) -> str:
    """Normalisiert external mode aliases to canonical values."""
    _m = str(aurik_mode or "restoration").strip().lower().replace("_", "").replace(" ", "")
    if _m in {"studio2026", "studio", "highendstudio", "maximum"}:
        return "studio2026"
    return "restoration"


# ============================================================================
# ENUMS
# ============================================================================


class VocalProcessingMode(Enum):
    """Vocal processing modes based on lyrics and semantic analysis."""

    GENTLE_ENHANCEMENT = "gentle_enhancement"  # Minimal processing, preserve character
    STANDARD_VOCAL = "standard_vocal"  # Balanced vocal processing
    AGGRESSIVE_CLARITY = "aggressive_clarity"  # Maximum clarity and presence
    SIBILANCE_REDUCTION = "sibilance_reduction"  # Focus on de-essing
    BREATH_PRESERVATION = "breath_preservation"  # Keep natural breath
    TRANSIENT_AWARE = "transient_aware"  # Preserve vocal attacks


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class IntegratedVocalSegment:
    """Enhanced segment combining lyrics and semantic analysis."""

    # Time boundaries
    start_time: float
    end_time: float

    # Lyrics information
    lyrics_text: str | None
    phonemes: list[PhonemeAlignment]

    # Semantic information
    content_type: ContentType
    dominant_instrument: InstrumentType
    content_character: ContentCharacter

    # Processing decision
    processing_mode: VocalProcessingMode

    # Per-phoneme processing hints
    sibilant_reduction: float  # 0.0-1.0 (how much de-essing)
    transient_preservation: float  # 0.0-1.0 (preserve attacks)
    clarity_enhancement: float  # 0.0-1.0 (enhance presence)
    breath_handling: str  # "preserve", "reduce", "remove"

    notes: str = ""


@dataclass
class IntegratedVocalTimeline:
    """Complete timeline with integrated lyrics and semantic guidance."""

    segments: list[IntegratedVocalSegment]
    total_duration: float

    # High-level statistics
    vocal_percentage: float
    sibilant_percentage: float
    breath_percentage: float

    # Processing recommendations
    global_deessing_amount: float  # 0.0-1.0
    global_compression_ratio: float  # 1.0-20.0
    global_eq_boost_db: float  # -12.0 to +12.0 dB

    # Mode-specific strategies
    restoration_strategy: str
    studio_strategy: str

    def get_processing_at_time(self, timestamp: float) -> IntegratedVocalSegment | None:
        """Gibt zurück: processing information at specific timestamp."""
        for segment in self.segments:
            if segment.start_time <= timestamp < segment.end_time:
                return segment
        return None

    def __repr__(self) -> str:
        return (
            f"IntegratedVocalTimeline({len(self.segments)} segments, "
            f"{self.vocal_percentage:.1%} vocal, "
            f"{self.sibilant_percentage:.1%} sibilant)"
        )


# ============================================================================
# INTEGRATED VOCAL PROCESSOR
# ============================================================================


class IntegratedVocalProcessor:
    """
    Kombiniert lyrics-guided processing with semantic understanding.
    for comprehensive vocal enhancement.

    This processor:
    1. Aligns lyrics with phoneme-level precision
    2. Analyzes semantic audio content
    3. Generates integrated processing timeline
    4. Provides per-phoneme processing recommendations
    """

    def __init__(self):
        """Initialisiert integrated processor."""
        self.lyrics_aligner = LyricsAligner(use_whisper=True, use_mfa=True)
        self.content_processor = ContentAwareProcessor()
        self.semantic_analyzer = SemanticAudioAnalyzer()

        logger.info("IntegratedVocalProcessor initialisiert")

    def create_integrated_timeline(
        self,
        audio: np.ndarray,
        sr: int,
        aurik_mode: str = "restoration",
        language: str = "en",
        provided_lyrics: str | None = None,
    ) -> "IntegratedVocalTimeline":
        """
        Erstellt integrated vocal processing timeline.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            aurik_mode: "restoration" or "studio2026" (legacy aliases accepted)
            language: Language code for lyrics transcription
            provided_lyrics: Optional pre-known lyrics

        Returns:
            IntegratedVocalTimeline with comprehensive processing guidance
        """
        # Convert to mono for analysis
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
        # NaN/Inf-Guard
        audio_mono = np.nan_to_num(audio_mono, nan=0.0, posinf=0.0, neginf=0.0)
        audio_mono = np.clip(audio_mono, -1.0, 1.0)

        duration = len(audio_mono) / sr

        aurik_mode = _normalize_aurik_mode(aurik_mode)
        logger.info(
            f"Creating integrated vocal timeline (Betriebsart={aurik_mode}, duration={duration:.2f}s, language={language})"
        )

        # 1. Lyrics alignment (with phonemes)
        logger.info("📝 Step 1: Lyrics alignment...")
        lyrics_alignment = self.lyrics_aligner.align(audio_mono, sr, lyrics=provided_lyrics)

        # 2. Content analysis (vocal/instrumental segmentation)
        logger.info("🎵 Step 2: Content Analyse...")
        content_timeline = self.content_processor.create_processing_timeline(audio_mono, sr, aurik_mode, language)

        # 3. Semantic analysis (instrument detection, character)
        logger.info("🎸 Step 3: Semantic Analyse...")
        semantic_profile = self.semantic_analyzer.analyze(audio_mono, sr, aurik_mode)

        # 4. Integrate all information
        logger.info("🔗 Step 4: Integration...")
        integrated_segments = self._integrate_analyses(
            lyrics_alignment,
            content_timeline,
            semantic_profile,
            aurik_mode,
        )

        # 5. Compute global processing parameters
        logger.info("⚙️ Step 5: Global parameters...")
        global_params = self._compute_global_parameters(
            integrated_segments,
            semantic_profile,
            aurik_mode,
        )

        # 6. Generate strategies
        restoration_strategy = self._generate_restoration_strategy(integrated_segments, semantic_profile)
        studio_strategy = self._generate_studio_strategy(integrated_segments, semantic_profile)

        # Compute statistics
        vocal_duration = sum(
            seg.end_time - seg.start_time for seg in integrated_segments if seg.content_type == ContentType.VOCAL
        )
        vocal_percentage = vocal_duration / duration if duration > 0 else 0.0

        sibilant_duration = sum(
            p.end_time - p.start_time
            for seg in integrated_segments
            for p in seg.phonemes
            if p.phoneme_type == "sibilant"
        )
        sibilant_percentage = sibilant_duration / duration if duration > 0 else 0.0

        breath_duration = sum(
            seg.end_time - seg.start_time for seg in integrated_segments if seg.content_type == ContentType.BREATH
        )
        breath_percentage = breath_duration / duration if duration > 0 else 0.0

        timeline = IntegratedVocalTimeline(
            segments=integrated_segments,
            total_duration=duration,
            vocal_percentage=vocal_percentage,
            sibilant_percentage=sibilant_percentage,
            breath_percentage=breath_percentage,
            global_deessing_amount=global_params["deessing"],
            global_compression_ratio=global_params["compression"],
            global_eq_boost_db=global_params["eq_boost"],
            restoration_strategy=restoration_strategy,
            studio_strategy=studio_strategy,
        )

        logger.info("✅ Integrated timeline erstellt: %s", timeline)

        return timeline

    # ========================================================================
    # INTEGRATION LOGIC
    # ========================================================================

    def _integrate_analyses(
        self,
        lyrics_alignment: LyricsAlignment,
        content_timeline: LyricsGuidedTimeline,
        semantic_profile: SemanticProfile,
        aurik_mode: str,
    ) -> list[IntegratedVocalSegment]:
        """Integrate lyrics, content, and semantic analyses."""
        integrated_segments = []

        # Process each word from lyrics alignment
        for word in lyrics_alignment.words:
            # Find corresponding content segment
            content_seg = self._find_content_segment(word.start_time, content_timeline.segments)

            # Determine processing mode
            processing_mode = self._determine_processing_mode(
                word,
                content_seg,
                semantic_profile,
                aurik_mode,
            )

            # Compute per-phoneme parameters
            sibilant_reduction = self._compute_sibilant_reduction(word.phonemes, semantic_profile)
            transient_preservation = self._compute_transient_preservation(word.phonemes, semantic_profile)
            clarity_enhancement = self._compute_clarity_enhancement(word, semantic_profile, aurik_mode)
            breath_handling = self._determine_breath_handling(word, content_seg, aurik_mode)

            # Create integrated segment
            segment = IntegratedVocalSegment(
                start_time=word.start_time,
                end_time=word.end_time,
                lyrics_text=word.word,
                phonemes=word.phonemes,
                content_type=content_seg.content_type if content_seg else ContentType.VOCAL,
                dominant_instrument=semantic_profile.dominant_instrument,
                content_character=semantic_profile.content_character,
                processing_mode=processing_mode,
                sibilant_reduction=sibilant_reduction,
                transient_preservation=transient_preservation,
                clarity_enhancement=clarity_enhancement,
                breath_handling=breath_handling,
                notes=f"{word.word} ({len(word.phonemes)} phonemes)",
            )

            integrated_segments.append(segment)

        # Also add non-vocal segments (instrumental, breath, silence)
        for content_seg in content_timeline.segments:
            if content_seg.content_type != ContentType.VOCAL:
                # Check if this segment overlaps with any word
                overlaps = any(
                    seg.start_time <= content_seg.start_time < seg.end_time
                    or seg.start_time < content_seg.end_time <= seg.end_time
                    for seg in integrated_segments
                )

                if not overlaps:
                    # Add as non-vocal segment  # type: ignore[misc]
                    segment = IntegratedVocalSegment(
                        start_time=content_seg.start_time,
                        end_time=content_seg.end_time,
                        lyrics_text=None,
                        phonemes=[],
                        content_type=content_seg.content_type,
                        dominant_instrument=semantic_profile.dominant_instrument,
                        content_character=semantic_profile.content_character,
                        processing_mode=VocalProcessingMode.GENTLE_ENHANCEMENT,
                        sibilant_reduction=0.0,
                        transient_preservation=1.0,
                        clarity_enhancement=0.0,
                        breath_handling="preserve",
                        notes=content_seg.notes,
                    )
                    integrated_segments.append(segment)

        # Sort by start time
        integrated_segments.sort(key=lambda s: s.start_time)

        return integrated_segments

    def _find_content_segment(
        self,
        timestamp: float,
        segments: list[ContentSegment],
    ) -> ContentSegment | None:
        """Findet das Inhaltssegment beim gegebenen Zeitstempel."""
        for seg in segments:
            if seg.start_time <= timestamp < seg.end_time:
                return seg
        return None

    def _determine_processing_mode(
        self,
        word: WordAlignment,
        content_seg: ContentSegment | None,
        semantic_profile: SemanticProfile,
        aurik_mode: str,
    ) -> VocalProcessingMode:
        """Bestimmt processing mode for word."""
        # Check phoneme types
        has_sibilants = any(p.phoneme_type == "sibilant" for p in word.phonemes)
        has_plosives = any(p.phoneme_type == "plosive" for p in word.phonemes)

        # Restoration mode: more gentle
        if aurik_mode == "restoration":
            if has_sibilants:
                return VocalProcessingMode.SIBILANCE_REDUCTION
            elif has_plosives:
                return VocalProcessingMode.TRANSIENT_AWARE
            else:
                return VocalProcessingMode.GENTLE_ENHANCEMENT

        # Studio mode: more aggressive
        else:
            if has_sibilants or semantic_profile.enhance_clarity:
                return VocalProcessingMode.AGGRESSIVE_CLARITY
            else:
                return VocalProcessingMode.STANDARD_VOCAL

    def _compute_sibilant_reduction(
        self,
        phonemes: list[PhonemeAlignment],
        semantic_profile: SemanticProfile,
    ) -> float:
        """Berechnet de-essing amount based on sibilant content."""
        sibilant_count = sum(1 for p in phonemes if p.phoneme_type == "sibilant")

        if sibilant_count == 0:
            return 0.0

        # Base reduction: 0.3-0.7 depending on sibilant density
        sibilant_ratio = sibilant_count / len(phonemes) if len(phonemes) > 0 else 0.0
        base_reduction = min(0.7, 0.3 + sibilant_ratio * 0.4)

        # Adjust based on semantic profile
        if semantic_profile.reduce_harshness:
            base_reduction = min(1.0, base_reduction + 0.2)

        return base_reduction

    def _compute_transient_preservation(
        self,
        phonemes: list[PhonemeAlignment],
        semantic_profile: SemanticProfile,
    ) -> float:
        """Berechnet transient preservation amount."""
        # Count plosives (strong transients)
        plosive_count = sum(1 for p in phonemes if p.phoneme_type == "plosive")

        if plosive_count == 0:
            return 0.5  # Default

        # High plosive content = preserve more
        plosive_ratio = plosive_count / len(phonemes) if len(phonemes) > 0 else 0.0
        preservation = min(1.0, 0.5 + plosive_ratio * 0.5)

        # Adjust based on semantic profile
        if semantic_profile.preserve_transients:
            preservation = min(1.0, preservation + 0.2)

        return preservation

    def _compute_clarity_enhancement(
        self,
        word: WordAlignment,
        semantic_profile: SemanticProfile,
        aurik_mode: str,
    ) -> float:
        """Berechnet clarity enhancement amount."""
        # Base enhancement
        if aurik_mode == "restoration":
            base_enhancement = 0.3
        else:  # studio
            base_enhancement = 0.6

        # Adjust based on semantic profile
        if semantic_profile.enhance_clarity:
            base_enhancement = min(1.0, base_enhancement + 0.2)

        # Reduce for low-confidence words
        if word.confidence < 0.5:
            base_enhancement *= word.confidence * 2

        return base_enhancement

    def _determine_breath_handling(
        self,
        word: WordAlignment,
        content_seg: ContentSegment | None,
        aurik_mode: str,
    ) -> str:
        """Bestimmt how to handle breath sounds."""
        if content_seg and content_seg.content_type == ContentType.BREATH:
            if aurik_mode == "restoration":
                return "preserve"
            else:  # studio
                return "reduce"

        return "preserve"

    def _compute_global_parameters(
        self,
        segments: list[IntegratedVocalSegment],
        semantic_profile: SemanticProfile,
        aurik_mode: str,
    ) -> dict[str, float]:
        """Berechnet global processing parameters."""
        # Average sibilant reduction across all vocal segments
        vocal_segments = [s for s in segments if s.content_type == ContentType.VOCAL]

        if len(vocal_segments) > 0:
            avg_deessing = np.mean([s.sibilant_reduction for s in vocal_segments])
        else:
            avg_deessing = 0.3  # type: ignore[assignment]  # Default

        # Compression based on mode and content character
        if aurik_mode == "restoration":
            base_compression = 2.0  # Gentle
        else:  # studio
            base_compression = 4.0  # Moderate

        # Adjust for content character
        if semantic_profile.content_character in [ContentCharacter.HIGHLY_TRANSIENT]:
            base_compression *= 0.7  # Less compression for transient-rich
        elif semantic_profile.content_character in [ContentCharacter.HIGHLY_SUSTAINED]:
            base_compression *= 1.3  # More compression for sustained

        # EQ boost for clarity
        if aurik_mode == "studio2026" and semantic_profile.enhance_clarity:
            eq_boost = 3.0  # +3dB in presence range
        else:
            eq_boost = 1.0  # +1dB

        return {
            "deessing": float(avg_deessing),
            "compression": float(base_compression),
            "eq_boost": float(eq_boost),
        }

    # ========================================================================
    # STRATEGY GENERATION
    # ========================================================================

    def _generate_restoration_strategy(
        self,
        segments: list[IntegratedVocalSegment],
        semantic_profile: SemanticProfile,
    ) -> str:
        """Generiert restoration strategy description."""
        vocal_count = sum(1 for s in segments if s.content_type == ContentType.VOCAL)
        sibilant_count = sum(1 for s in segments for p in s.phonemes if p.phoneme_type == "sibilant")

        return (
            f"LYRICS-AWARE RESTORATION: {vocal_count} vocal segments with "
            f"{sibilant_count} sibilant phonemes detected. "
            "Applying phoneme-specific processing: gentle de-essing, "
            "transient preservation, natural breath handling. "
            f"Semantic profile: {semantic_profile.dominant_instrument.value}, "
            f"{semantic_profile.content_character.value}."
        )

    def _generate_studio_strategy(
        self,
        segments: list[IntegratedVocalSegment],
        semantic_profile: SemanticProfile,
    ) -> str:
        """Generiert studio strategy description."""
        vocal_count = sum(1 for s in segments if s.content_type == ContentType.VOCAL)

        return (
            f"LYRICS-AWARE PRODUCTION: {vocal_count} vocal segments with "
            "phoneme-level processing. Applying modern vocal chain: "
            "aggressive de-essing, clarity enhancement, breath control. "
            f"Optimized for {semantic_profile.dominant_instrument.value} "
            f"with {semantic_profile.content_character.value} character."
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def create_integrated_vocal_timeline(
    audio: np.ndarray,
    sr: int,
    aurik_mode: str = "restoration",
    language: str = "en",
    provided_lyrics: str | None = None,
) -> IntegratedVocalTimeline:
    """
    Convenience function for integrated vocal timeline creation.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
        aurik_mode: "restoration" or "studio2026" (legacy aliases accepted)
        language: Language code (en, de, es, etc.)
        provided_lyrics: Optional pre-known lyrics

    Returns:
        IntegratedVocalTimeline with comprehensive processing guidance

    Example:
        >>> timeline = create_integrated_vocal_timeline(
        ...     audio, sr=48000, aurik_mode="restoration", language="en"
        ... )
        >>> print(f"Vocal: {timeline.vocal_percentage:.1%}")
        >>> print(f"Sibilants: {timeline.sibilant_percentage:.1%}")
        >>> print(f"De-essing: {timeline.global_deessing_amount:.2f}")
        >>>
        >>> # Get processing at specific timestamp
        >>> seg = timeline.get_processing_at_time(2.5)
        >>> if seg:
        ...     print(f"Word: {seg.lyrics_text}")
        ...     print(f"Mode: {seg.processing_mode.value}")
        ...     print(f"Phonemes: {len(seg.phonemes)}")
    """
    processor = IntegratedVocalProcessor()
    return processor.create_integrated_timeline(audio, sr, aurik_mode, language, provided_lyrics)


if __name__ == "__main__":
    # Demo
    logger.info(str("=" * 80))
    logger.info("AURIK Integrated Vocal Processor Demo")
    logger.info(str("=" * 80))

    # Create test audio (5 seconds with simulated vocals)
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Simulated vocal pattern
    audio = np.zeros_like(t)

    # Vocal segments with varying characteristics
    # Segment 1: 0.5-1.5s (sustained vowel)
    mask1 = (t >= 0.5) & (t < 1.5)
    audio[mask1] = 0.3 * np.sin(2 * np.pi * 300 * t[mask1])

    # Segment 2: 2.0-3.0s (sibilant-rich)
    mask2 = (t >= 2.0) & (t < 3.0)
    audio[mask2] = 0.2 * np.random.randn(np.sum(mask2)).astype(np.float32)  # type: ignore[call-overload]  # Noise (sibilant-like)

    # Background
    audio += 0.01 * np.random.randn(len(audio)).astype(np.float32)

    # Process
    logger.info("\n🎤 Creating integrated vocal timeline...")
    try:
        timeline = create_integrated_vocal_timeline(audio, sr, aurik_mode="restoration", language="en")

        logger.info("\n✅ Timeline erstellt: %s", timeline)
        logger.info("\n📊 Statistics:")
        logger.info("   Vocal: %.1f", timeline.vocal_percentage)
        logger.info("   Sibilant: %.1f", timeline.sibilant_percentage)
        logger.info("   Breath: %.1f", timeline.breath_percentage)
        logger.info("\n⚙️ Global Parameters:")
        logger.info("   De-essing: %.2f", timeline.global_deessing_amount)
        logger.info("   Compression: %.1f:1", timeline.global_compression_ratio)
        logger.info("   EQ Boost: %+.1f dB", timeline.global_eq_boost_db)
        logger.info("\n📝 Strategy:")
        logger.info("   %s", timeline.restoration_strategy)

    except Exception as e:
        logger.error("⚠️ Demo fehlgeschlagen (expected if dependencies not installed): %s", e)

    logger.info(str("\n" + "=" * 80))
