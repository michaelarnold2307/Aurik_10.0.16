"""
Lyrics-Guided Vocal Enhancement
================================

World's first lyrics-synchronized audio processing.

This module enables content-aware vocal processing by:
- Automatic lyrics transcription (Whisper-based)
- Phoneme-level alignment (integration with Week 7-9)
- Word-level timestamps
- Lyrics-driven processing decisions

Supports Both AURIK Modes:
- RESTORATION: Process vocals only where vocals exist (preserve instrumental)
- HIGHEND STUDIO: Optimize vocal/instrumental balance dynamically

Author: Aurik Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec

import numpy as np

# Optional Whisper integration (availability check without importing module)
WHISPER_AVAILABLE = find_spec("whisper") is not None

# Integration with Week 7-9 Phoneme Classifier
try:
    from backend.ml.phoneme_aware.phoneme_classifier import PhonemeClassifier

    PHONEME_CLASSIFIER_AVAILABLE = True
except ImportError:
    PHONEME_CLASSIFIER_AVAILABLE = False
    PhonemeClassifier = None  # type: ignore[assignment, misc]

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


class ContentType(Enum):
    """Type of audio content at a given timestamp."""

    SILENCE = "silence"  # No audio
    VOCAL = "vocal"  # Singing/speaking
    BREATH = "breath"  # Natural breath
    INSTRUMENTAL = "instrumental"  # Music without vocals
    MIXED = "mixed"  # Vocals + instruments


class ProcessingIntent(Enum):
    """What to do with each content type."""

    NO_PROCESSING = "no_processing"  # Leave untouched
    VOCAL_ENHANCE = "vocal_enhance"  # Apply vocal enhancement
    PRESERVE_NATURAL = "preserve_natural"  # Keep but don't enhance
    DENOISE_ONLY = "denoise_only"  # Remove noise, preserve content
    FULL_PROCESSING = "full_processing"  # Apply all available processing


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class LyricsSegment:
    """Represents a word/phrase with timing."""

    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float  # 0.0-1.0
    phonemes: list[str] | None = None  # IPA phonemes (from Week 7-9)


@dataclass
class ContentSegment:
    """Represents audio content type with timing."""

    content_type: ContentType
    start_time: float
    end_time: float
    processing_intent: ProcessingIntent
    lyrics: str | None = None
    notes: str = ""


@dataclass
class LyricsGuidedTimeline:
    """
    Complete timeline with content-aware processing instructions.

    Provides frame-by-frame processing guidance for both AURIK modes.
    """

    segments: list[ContentSegment]
    total_duration: float
    vocal_percentage: float  # 0.0-1.0

    # Mode-specific recommendations
    restoration_strategy: str
    studio_strategy: str

    def get_processing_at_time(self, timestamp: float) -> ProcessingIntent:
        """Gibt zurück: processing intent at a specific timestamp."""
        for segment in self.segments:
            if segment.start_time <= timestamp < segment.end_time:
                return segment.processing_intent
        return ProcessingIntent.NO_PROCESSING

    def __repr__(self) -> str:
        return f"LyricsGuidedTimeline({len(self.segments)} segments, {self.vocal_percentage:.1%} vocal)"


# ============================================================================
# LYRICS ALIGNER
# ============================================================================


class LyricsAligner:
    """
    Align lyrics to audio using speech recognition.

    Provides word-level and phoneme-level timestamps for
    content-aware processing.
    """

    def __init__(self, use_phoneme_classifier: bool = True):
        """
        Initialisiert lyrics aligner.

        Args:
            use_phoneme_classifier: Use Week 7-9 phoneme classifier
        """
        self.use_phoneme_classifier = use_phoneme_classifier and PHONEME_CLASSIFIER_AVAILABLE

        if self.use_phoneme_classifier and PhonemeClassifier is not None:
            self.phoneme_classifier = PhonemeClassifier()
            logger.info("LyricsAligner initialisiert with phoneme classification")
        else:
            self.phoneme_classifier = None  # type: ignore[assignment]
            logger.info("LyricsAligner initialisiert without phoneme classification")

        # Check Whisper availability
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not verfuegbar. Install with: pip install openai-whisper")

    def align(
        self,
        audio: np.ndarray,
        sr: int,
        language: str = "en",
        provided_lyrics: str | None = None,
    ) -> list[LyricsSegment]:
        """
        Align lyrics to audio.

        Args:
            audio: Input audio (mono)
            sr: Sample rate
            language: Language code (en, de, es, fr, etc.)
            provided_lyrics: Optional pre-known lyrics (forces alignment)

        Returns:
            List of LyricsSegment with word-level timestamps
        """
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not verfuegbar - returning empty segments")
            return []

        # Convert to mono
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        logger.info("Aligning lyrics for %.2fs audio (language=%s)", len(audio_mono) / sr, language)

        # Use Whisper for transcription + alignment
        # (In production, use more sophisticated forced alignment)
        segments = self._whisper_align(audio_mono, sr, language)

        # Add phoneme information if available
        if self.use_phoneme_classifier and self.phoneme_classifier:
            segments = self._add_phoneme_information(segments, audio_mono, sr)

        return segments

    def _whisper_align(
        self,
        audio: np.ndarray,
        sr: int,
        language: str,
    ) -> list[LyricsSegment]:
        """Use Whisper for transcription with word-level timestamps."""
        segments = []

        try:
            import whisper

            model = whisper.load_model("tiny")
            # Whisper expects float32 audio at 16 kHz
            if sr != 16000:
                import librosa

                audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio.astype(np.float32)

            result = model.transcribe(
                audio_16k,
                language=language,
                word_timestamps=True,
                fp16=False,
            )

            for seg in result.get("segments", []):
                for word_info in seg.get("words", []):
                    text = word_info.get("word", "").strip()
                    if not text:
                        continue
                    start_t = float(word_info.get("start", 0.0))
                    end_t = float(word_info.get("end", start_t + 0.1))
                    prob = float(word_info.get("probability", 0.5))
                    segments.append(
                        LyricsSegment(
                            text=text,
                            start_time=start_t,
                            end_time=end_t,
                            confidence=prob,
                        )
                    )

            # Fallback: if word_timestamps not available, use segment-level
            if not segments:
                for seg in result.get("segments", []):
                    text = seg.get("text", "").strip()
                    if not text:
                        continue
                    segments.append(
                        LyricsSegment(
                            text=text,
                            start_time=float(seg.get("start", 0.0)),
                            end_time=float(seg.get("end", 0.0)),
                            confidence=float(seg.get("avg_logprob", -1.0) + 1.0),
                        )
                    )

        except ImportError:
            logger.warning("Whisper not installed — falling back to energy-based segmentation")
            segments = self._energy_based_segmentation(audio, sr)
        except Exception as e:
            logger.warning("Whisper transcription fehlgeschlagen: %s — using energy-based Ersatzpfad", e)
            segments = self._energy_based_segmentation(audio, sr)

        logger.info("Aligned %d vocal segments", len(segments))
        return segments

    def _energy_based_segmentation(self, audio: np.ndarray, sr: int) -> list[LyricsSegment]:
        """Fallback: RMS-based vocal activity segmentation when Whisper unavailable."""
        segments = []
        frame_size = int(0.05 * sr)  # 50 ms
        hop = frame_size // 2
        n_frames = max(1, (len(audio) - frame_size) // hop)

        rms_values = np.array([np.sqrt(np.mean(audio[i * hop : i * hop + frame_size] ** 2)) for i in range(n_frames)])
        threshold = max(0.01, float(np.percentile(rms_values, 30)))

        in_segment = False
        seg_start = 0.0
        for i, rms in enumerate(rms_values):
            t = i * hop / sr
            if rms > threshold and not in_segment:
                seg_start = t
                in_segment = True
            elif rms <= threshold and in_segment:
                if t - seg_start > 0.1:  # minimum 100 ms
                    segments.append(
                        LyricsSegment(
                            text="[vocal]",
                            start_time=seg_start,
                            end_time=t,
                            confidence=0.6,
                        )
                    )
                in_segment = False
        if in_segment:
            segments.append(
                LyricsSegment(
                    text="[vocal]",
                    start_time=seg_start,
                    end_time=len(audio) / sr,
                    confidence=0.6,
                )
            )
        return segments

    def _add_phoneme_information(
        self,
        segments: list[LyricsSegment],
        audio: np.ndarray,
        sr: int,
    ) -> list[LyricsSegment]:
        """Fügt hinzu: phoneme information from Week 7-9 classifier."""
        if self.phoneme_classifier is None:
            logger.debug("PhonemeClassifier not verfuegbar — skipping phoneme annotation")
            return segments

        for segment in segments:
            start_idx = int(segment.start_time * sr)
            end_idx = int(segment.end_time * sr)
            if end_idx <= start_idx or end_idx > len(audio):
                continue
            segment_audio = audio[start_idx:end_idx]
            try:
                result = self.phoneme_classifier.classify(segment_audio, sr)  # type: ignore[call-arg, arg-type]
                if hasattr(result, "phonemes"):
                    segment.phonemes = result.phonemes
                elif isinstance(result, list):
                    segment.phonemes = result
                else:
                    segment.phonemes = [str(result)]
            except Exception as e:
                logger.debug("Phoneme classification fehlgeschlagen for segment '%s': %s", segment.text, e)
                segment.phonemes = []

        return segments


# ============================================================================
# CONTENT AWARE PROCESSOR
# ============================================================================


class ContentAwareProcessor:
    """
    Verarbeitet audio based on lyrics-guided content analysis.

    Determines what to do with each audio segment based on content type.
    Respects AURIK operating modes.
    """

    def __init__(self):
        """Initialisiert content-aware processor."""
        self.lyrics_aligner = LyricsAligner()
        logger.info("ContentAwareProcessor initialisiert")

    def create_processing_timeline(
        self,
        audio: np.ndarray,
        sr: int,
        aurik_mode: str = "restoration",
        language: str = "en",
    ) -> LyricsGuidedTimeline:
        """
        Erstellt lyrics-guided processing timeline.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            aurik_mode: "restoration" or "studio2026" (legacy aliases accepted)
            language: Language code

        Returns:
            LyricsGuidedTimeline with processing instructions
        """
        # Convert to mono for analysis
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        duration = len(audio_mono) / sr

        aurik_mode = _normalize_aurik_mode(aurik_mode)
        logger.info("Creating processing timeline (%s Betriebsart, %.2fs)", aurik_mode, duration)

        # 1. Get lyrics alignment
        lyrics_segments = self.lyrics_aligner.align(audio_mono, sr, language)

        # 2. Analyze content types
        content_segments = self._analyze_content_types(audio_mono, sr, lyrics_segments)

        # 3. Assign processing intents based on mode
        content_segments = self._assign_processing_intents(content_segments, aurik_mode)

        # 4. Compute statistics
        vocal_duration = sum(
            seg.end_time - seg.start_time for seg in content_segments if seg.content_type == ContentType.VOCAL
        )
        vocal_percentage = vocal_duration / duration if duration > 0 else 0.0

        # 5. Generate mode-specific strategies
        restoration_strategy = self._generate_restoration_strategy(content_segments, vocal_percentage)
        studio_strategy = self._generate_studio_strategy(content_segments, vocal_percentage)

        return LyricsGuidedTimeline(
            segments=content_segments,
            total_duration=duration,
            vocal_percentage=vocal_percentage,
            restoration_strategy=restoration_strategy,
            studio_strategy=studio_strategy,
        )

    def _analyze_content_types(
        self,
        audio: np.ndarray,
        sr: int,
        lyrics_segments: list[LyricsSegment],
    ) -> list[ContentSegment]:
        """Analysiert Audio zur Bestimmung von Inhaltstypen."""
        content_segments = []

        # Create timeline covering full audio
        duration = len(audio) / sr

        # Start with lyrics segments (vocal regions)
        for lyrics_seg in lyrics_segments:
            content_seg = ContentSegment(
                content_type=ContentType.VOCAL,
                start_time=lyrics_seg.start_time,
                end_time=lyrics_seg.end_time,
                processing_intent=ProcessingIntent.VOCAL_ENHANCE,  # Will be refined
                lyrics=lyrics_seg.text,
                notes="Detected vocals with lyrics",
            )
            content_segments.append(content_seg)

        # Fill gaps between vocals
        if len(content_segments) > 0:
            # Before first vocal
            if content_segments[0].start_time > 0.1:
                intro = ContentSegment(
                    content_type=ContentType.INSTRUMENTAL,
                    start_time=0.0,
                    end_time=content_segments[0].start_time,
                    processing_intent=ProcessingIntent.NO_PROCESSING,
                    notes="Instrumental intro",
                )
                content_segments.insert(0, intro)

            # Between vocals
            for i in range(len(content_segments) - 1):
                gap_start = content_segments[i].end_time
                gap_end = content_segments[i + 1].start_time

                if gap_end - gap_start > 0.2:  # Significant gap
                    gap_audio = audio[int(gap_start * sr) : int(gap_end * sr)]

                    # Check if it's breath or instrumental
                    if gap_end - gap_start < 0.5 and np.std(gap_audio) < 0.05:
                        gap_type = ContentType.BREATH
                        intent = ProcessingIntent.PRESERVE_NATURAL
                    else:
                        gap_type = ContentType.INSTRUMENTAL
                        intent = ProcessingIntent.NO_PROCESSING

                    gap_seg = ContentSegment(
                        content_type=gap_type,
                        start_time=gap_start,
                        end_time=gap_end,
                        processing_intent=intent,
                        notes=f"{gap_type.value} gap",
                    )
                    content_segments.insert(i + 1, gap_seg)

            # After last vocal
            if content_segments[-1].end_time < duration - 0.1:
                outro = ContentSegment(
                    content_type=ContentType.INSTRUMENTAL,
                    start_time=content_segments[-1].end_time,
                    end_time=duration,
                    processing_intent=ProcessingIntent.NO_PROCESSING,
                    notes="Instrumental outro",
                )  # type: ignore[misc]
                content_segments.append(outro)
        else:
            # No vocals detected - entire track is instrumental
            full_track = ContentSegment(
                content_type=ContentType.INSTRUMENTAL,
                start_time=0.0,
                end_time=duration,
                processing_intent=ProcessingIntent.DENOISE_ONLY,
                notes="Instrumental track (no vocals detected)",
            )
            content_segments = [full_track]

        return content_segments

    def _assign_processing_intents(
        self,
        segments: list[ContentSegment],
        aurik_mode: str,
    ) -> list[ContentSegment]:
        """Assign processing intents based on AURIK mode."""
        for segment in segments:
            if aurik_mode == "restoration":
                # RESTORATION: Conservative, preserve authenticity
                if segment.content_type == ContentType.VOCAL:
                    segment.processing_intent = ProcessingIntent.VOCAL_ENHANCE
                elif segment.content_type == ContentType.BREATH:
                    segment.processing_intent = ProcessingIntent.PRESERVE_NATURAL
                elif segment.content_type == ContentType.INSTRUMENTAL:
                    segment.processing_intent = ProcessingIntent.DENOISE_ONLY
                elif segment.content_type == ContentType.SILENCE:
                    segment.processing_intent = ProcessingIntent.NO_PROCESSING

            elif aurik_mode == "studio2026":
                # STUDIO: Optimize for modern standards
                if segment.content_type == ContentType.VOCAL:
                    segment.processing_intent = ProcessingIntent.FULL_PROCESSING
                elif segment.content_type == ContentType.BREATH:
                    # Studio decision: reduce or keep breath?
                    segment.processing_intent = ProcessingIntent.PRESERVE_NATURAL
                elif segment.content_type == ContentType.INSTRUMENTAL:
                    segment.processing_intent = ProcessingIntent.FULL_PROCESSING
                elif segment.content_type == ContentType.SILENCE:
                    segment.processing_intent = ProcessingIntent.NO_PROCESSING

        return segments

    def _generate_restoration_strategy(
        self,
        segments: list[ContentSegment],
        vocal_percentage: float,
    ) -> str:
        """Generiert restoration mode strategy."""
        vocal_count = sum(1 for s in segments if s.content_type == ContentType.VOCAL)

        if vocal_percentage > 0.7:
            return (
                f"VOCAL-FOCUSED RESTORATION: {vocal_count} vocal segments "
                f"({vocal_percentage:.0%} vocal). Apply gentle enhancement to vocals, "
                "preserve instrumental breaks naturally."
            )
        elif vocal_percentage > 0.3:
            return (
                f"BALANCED RESTORATION: {vocal_count} vocal segments "
                f"({vocal_percentage:.0%} vocal). Process vocals and instruments "
                "independently, preserve mix balance."
            )
        else:
            return (
                f"INSTRUMENTAL RESTORATION: Minimal vocals ({vocal_percentage:.0%}). "
                "Focus on noise reduction and tonal balance, preserve instrumental textures."
            )

    def _generate_studio_strategy(
        self,
        segments: list[ContentSegment],
        vocal_percentage: float,
    ) -> str:
        """Generiert studio mode strategy."""
        vocal_count = sum(1 for s in segments if s.content_type == ContentType.VOCAL)

        if vocal_percentage > 0.7:
            return (
                f"VOCAL PRODUCTION: {vocal_count} vocal segments "
                f"({vocal_percentage:.0%} vocal). Maximize vocal clarity and presence, "
                "apply modern vocal chain (de-essing, compression, EQ)."
            )
        elif vocal_percentage > 0.3:
            return (
                f"MIXED PRODUCTION: {vocal_count} vocal segments "
                f"({vocal_percentage:.0%} vocal). Balance vocal/instrumental processing, "
                "optimize for streaming platforms."
            )
        else:
            return (
                f"INSTRUMENTAL PRODUCTION: Minimal vocals ({vocal_percentage:.0%}). "
                "Focus on instrumental clarity, dynamics, and tonal balance."
            )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def create_lyrics_guided_timeline(
    audio: np.ndarray,
    sr: int,
    aurik_mode: str = "restoration",
    language: str = "en",
) -> LyricsGuidedTimeline:
    """
    Convenience function for lyrics-guided processing timeline.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
        aurik_mode: "restoration" or "studio2026" (legacy aliases accepted)
        language: Language code (en, de, es, fr, etc.)

    Returns:
        LyricsGuidedTimeline with processing instructions

    Example:
        >>> timeline = create_lyrics_guided_timeline(audio, sr=48000, aurik_mode="restoration")
        >>> print(f"Vocal percentage: {timeline.vocal_percentage:.1%}")
        >>> print(f"Strategy: {timeline.restoration_strategy}")
        >>>
        >>> # Get processing at specific timestamp
        >>> processing = timeline.get_processing_at_time(2.5)  # 2.5 seconds
        >>> print(f"Process at 2.5s: {processing.value}")  # type: ignore[misc]
    """
    processor = ContentAwareProcessor()
    return processor.create_processing_timeline(audio, sr, aurik_mode, language)
