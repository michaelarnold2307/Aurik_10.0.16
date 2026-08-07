"""
Phoneme Detector using Wav2Vec2

This module provides automatic phoneme detection from audio using
Meta's Wav2Vec2 model trained on Common Voice dataset with eSpeak-ng
phoneme targets.

Key Features:
- IPA phoneme output
- Frame-level timestamps
- Confidence scores
- Multi-language support (60+ languages)
- Automatic resampling to 16kHz

Model: facebook/wav2vec2-lv-60-espeak-cv-ft
- Size: ~360MB
- Languages: 60+ supported via eSpeak-ng
- Output: IPA phonemes
- Frame resolution: 20ms

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Type checking imports (only for mypy/IDEs, not runtime)
if TYPE_CHECKING:
    from transformers import Wav2Vec2ForCTC as _Wav2Vec2ForCTC
    from transformers import Wav2Vec2Processor as _Wav2Vec2Processor

# Conditional imports (will be checked at runtime)
try:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    TRANSFORMERS_AVAILABLE = True
except Exception:
    # Broken optional torch installs must degrade to the DSP/non-ML path
    # instead of failing during module import or test collection.
    torch = None  # type: ignore[assignment]
    Wav2Vec2ForCTC = None  # type: ignore[misc]
    Wav2Vec2Processor = None  # type: ignore[misc]
    TRANSFORMERS_AVAILABLE = False
    # Warning will be shown when actually trying to use phoneme detection
    # (not at import time to avoid cluttering logs)

try:
    import librosa

    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

import logging

from backend.ml.phoneme_aware.logging_config import setup_logger

logger = logging.getLogger(__name__)

logger = setup_logger(__name__)

# Lokaler Wav2Vec2-Modell-Pfad — §13.3 bundled:true, kein HF-Download
_WAV2VEC2_LOCAL_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "wav2vec2"


class Language(Enum):
    """Supported languages for phoneme detection."""

    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"
    FRENCH = "fr"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    POLISH = "pl"
    # Add more as needed - Wav2Vec2 supports 60+ languages


@dataclass
class PhonemeSegment:
    """
    Erkanntes Phonem-Segment mit Timing und Konfidenz.

    Attributes:
        phoneme: IPA phoneme symbol (e.g., 'ɛ', 's', 'ʃ')
        start_time: Start time in seconds
        end_time: End time in seconds
        confidence: Detection confidence (0.0-1.0)
        frame_index: Original frame index in audio
    """

    phoneme: str
    start_time: float
    end_time: float
    confidence: float
    frame_index: int

    @property
    def duration(self) -> float:
        """Duration of phoneme segment in seconds."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (
            f"PhonemeSegment('{self.phoneme}', {self.start_time:.3f}-{self.end_time:.3f}s, conf={self.confidence:.2f})"
        )


@dataclass
class DetectionConfig:
    """
    Configuration for phoneme detection.

    Attributes:
        model_name: Lokaler Pfad zum Wav2Vec2-Modell (models/wav2vec2/)
        language: Target language (default: English)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        target_sample_rate: Target sample rate for model (16kHz for Wav2Vec2)
        use_gpu: Whether to use GPU if available
        cache_dir: Nicht genutzt — Modell wird immer aus lokalem Pfad geladen
    """

    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    language: Language = Language.ENGLISH
    min_confidence: float = 0.5
    target_sample_rate: int = 16000
    use_gpu: bool = True
    cache_dir: Path | None = None

    def __post_init__(self):
        """Validiert configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.target_sample_rate <= 0:
            raise ValueError(f"target_sample_rate must be positive, got {self.target_sample_rate}")


class PhonemeDetector:
    """
    Erkennt phonemes in audio using Wav2Vec2.

    This class provides automatic phoneme detection from audio signals using
    Meta's Wav2Vec2 model trained on Common Voice dataset with eSpeak-ng
    phoneme targets. The model outputs IPA phonemes with frame-level timing
    and confidence scores.

    Example:
        >>> detector = PhonemeDetector()
        >>> phonemes = detector.detect(audio, sr=44100)
        >>> for p in phonemes:
        ...     logger.info(f"{p.phoneme}: {p.start_time:.2f}-{p.end_time:.2f}s")
        h: 0.00-0.05s
        ɛ: 0.05-0.15s
        l: 0.15-0.20s
        oʊ: 0.20-0.35s

    Attributes:
        config: Detection configuration
        model: Wav2Vec2 model (loaded lazily)
        processor: Wav2Vec2 processor (loaded lazily)
        device: PyTorch device (cpu or cuda)
    """

    def __init__(self, config: DetectionConfig | None = None):
        """
        Initialisiert phoneme detector.

        Args:
            config: Detection configuration (uses defaults if None)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for PhonemeDetector. Install with: pip install transformers torch"
            )
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for PhonemeDetector. Install with: pip install librosa")

        self.config = config or DetectionConfig()
        self._model = None
        self._processor = None
        self._device = None

        logger.info("PhonemeDetector initialisiert with model: %s", self.config.model_name)

    def _resolved_model_name(self) -> str:
        """Löst auf: model source with local-first fallback for offline runtime."""
        if _WAV2VEC2_LOCAL_DIR.exists():
            return str(_WAV2VEC2_LOCAL_DIR)
        return self.config.model_name

    @property
    def device(self) -> Any:
        """Gibt zurück: PyTorch device (always cpu, §9.5 CPU-only policy)."""
        assert torch is not None
        if self._device is None:
            self._device = torch.device("cpu")  # type: ignore[assignment]
            logger.info("Using CPU for phoneme detection (§9.5 CPU-only)")
        return self._device

    @property
    def model(self) -> _Wav2Vec2ForCTC:
        """Lädt beim ersten Zugriff: Wav2Vec2 model."""
        if self._model is None:
            model_name = self._resolved_model_name()
            logger.info("lade model: %s", model_name)
            self._model = Wav2Vec2ForCTC.from_pretrained(  # type: ignore[assignment]  # nosec B615 — local_files_only wenn lokaler Pfad, sonst user-konfiguriertes Modell
                model_name,
                local_files_only=Path(model_name).exists(),
            )
            self._model = self._model.to(self.device)  # type: ignore[attr-defined]
            self._model.eval()  # type: ignore[attr-defined]  # Inference mode
            logger.info("Model geladen erfolgreich")
        return self._model  # type: ignore[return-value]

    @property
    def processor(self) -> _Wav2Vec2Processor:
        """Lädt beim ersten Zugriff: Wav2Vec2 processor."""
        if self._processor is None:
            model_name = self._resolved_model_name()
            logger.info("lade processor: %s", model_name)
            self._processor = Wav2Vec2Processor.from_pretrained(  # type: ignore[assignment]  # nosec B615 — local_files_only wenn lokaler Pfad, sonst user-konfiguriertes Modell
                model_name,
                local_files_only=Path(model_name).exists(),
            )
            logger.info("Processor geladen erfolgreich")
        return self._processor  # type: ignore[return-value]

    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio for Wav2Vec2.

        Steps:
        1. Convert stereo to mono if needed
        2. Resample to 16kHz
        3. Normalize to [-1, 1]

        Args:
            audio: Audio signal (shape: (samples,) or (2, samples))
            sr: Sample rate of input audio

        Returns:
            Preprocessed audio (mono, 16kHz, normalized)
        """
        # Convert stereo to mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
            logger.debug("Converted stereo to mono")

        # Resample to 16kHz if needed
        if sr != self.config.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.target_sample_rate)
            logger.debug("Resampled audio: %sHz → %sHz", sr, self.config.target_sample_rate)

        # Safety clip only (no peak normalization)
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def _decode_predictions(self, logits: Any, audio_length: int) -> list[tuple[str, int, float]]:
        """
        Dekodiert CTC predictions to phonemes.

        Uses CTC (Connectionist Temporal Classification) decoding to
        convert model logits to phoneme sequences with frame indices
        and confidence scores.

        Args:
            logits: Model output logits (shape: [time, vocab_size])
            audio_length: Original audio length in samples

        Returns:
            List of (phoneme, frame_index, confidence) tuples
        """
        assert torch is not None
        # Get token IDs (greedy decoding: argmax)
        predicted_ids = torch.argmax(logits, dim=-1)

        # Get confidence scores (softmax probabilities)
        probs = torch.softmax(logits, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]

        # Decode IDs to tokens
        # Note: CTC blank token is typically ID 0
        blank_token_id = 0

        phonemes_with_frames = []
        prev_token_id = None

        for frame_idx, (token_id, confidence) in enumerate(zip(predicted_ids.cpu().numpy(), confidences.cpu().numpy())):
            # Skip blank tokens
            if token_id == blank_token_id:
                prev_token_id = None
                continue

            # Skip repeated tokens (CTC collapses repeats)
            if token_id == prev_token_id:
                continue

            # Decode token ID to phoneme
            phoneme = self.processor.tokenizer.decode([token_id])

            # Skip special tokens
            if phoneme.startswith("[") or phoneme.startswith("<"):
                prev_token_id = token_id
                continue

            phonemes_with_frames.append((phoneme, frame_idx, float(confidence)))
            prev_token_id = token_id

        return phonemes_with_frames

    def _frames_to_time(self, frame_idx: int, total_frames: int, audio_duration: float) -> float:
        """
        Konvertiert frame index to time in seconds.

        Args:
            frame_idx: Frame index from model
            total_frames: Total number of frames
            audio_duration: Total audio duration in seconds

        Returns:
            Time in seconds
        """
        return (frame_idx / total_frames) * audio_duration

    def detect(
        self, audio: np.ndarray, sr: int, language: Language | None = None, min_confidence: float | None = None
    ) -> list[PhonemeSegment]:
        """
        Erkennt phonemes in audio.

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate of input audio
            language: Target language (overrides config)
            min_confidence: Minimum confidence (overrides config)

        Returns:
            List of detected phoneme segments

        Example:
            >>> detector = PhonemeDetector()
            >>> from backend.file_import import load_audio_file
            >>> _res = load_audio_file('speech.wav')
            >>> audio, sr = _res['audio'], int(_res['sr'])
            >>> phonemes = detector.detect(audio, sr)
            >>> logger.info("Detected %d phonemes", len(phonemes))
        """
        language = language or self.config.language
        min_confidence = min_confidence or self.config.min_confidence
        assert torch is not None

        logger.info(
            f"Detecting phonemes in {len(audio) / sr:.2f}s audio (lang={language.value}, min_conf={min_confidence})"
        )

        # Preprocess audio
        audio_processed = self._preprocess_audio(audio, sr)
        audio_duration = len(audio_processed) / self.config.target_sample_rate

        # Very short clips are often non-speech and can trigger tokenizer deps (phonemizer).
        if audio_duration < 0.2:
            logger.info(
                "Audio too short for reliable phoneme detection (%.3fs), returning empty Ergebnis", audio_duration
            )
            return []

        try:
            # Prepare input for model
            inputs = self.processor(audio_processed, sampling_rate=self.config.target_sample_rate, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)

            # Run model
            with torch.no_grad():
                logits = self.model(input_values).logits[0]  # [time, vocab_size]

            # Decode predictions
            phonemes_raw = self._decode_predictions(logits, len(audio_processed))
        except (ImportError, ModuleNotFoundError) as exc:
            # Optional tokenizer deps like phonemizer may be missing in lightweight envs.
            logger.warning("Phoneme detection dependency missing (%s); returning empty Ergebnis", exc)
            return []

        # Convert to PhonemeSegment objects
        total_frames = logits.shape[0]
        segments = []

        for phoneme, frame_idx, confidence in phonemes_raw:
            # Filter by confidence
            if confidence < min_confidence:
                continue

            # Compute timestamps
            # Each phoneme spans approximately one frame duration
            start_time = self._frames_to_time(frame_idx, total_frames, audio_duration)
            # Estimate end time (next frame or end of audio)
            end_time = self._frames_to_time(frame_idx + 1, total_frames, audio_duration)

            segment = PhonemeSegment(
                phoneme=phoneme.strip(),
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                frame_index=frame_idx,
            )
            segments.append(segment)

        logger.info("erkannt %s phonemes (filtered from %s by confidence)", len(segments), len(phonemes_raw))

        return segments

    def get_phoneme_timeline(
        self,
        segments: list[PhonemeSegment],
        audio_duration: float,
        frame_duration: float = 0.01,  # 10ms frames
    ) -> np.ndarray:
        """
        Konvertiert phoneme segments to frame-level timeline.

        Creates a dense timeline where each frame is labeled with the
        active phoneme at that time. Useful for synchronizing phoneme
        information with frame-by-frame audio processing.

        Args:
            segments: List of phoneme segments
            audio_duration: Total audio duration in seconds
            frame_duration: Duration of each frame in seconds

        Returns:
            Array of phoneme labels, one per frame (shape: [num_frames])
            Empty string '' indicates no phoneme (silence/breath)

        Example:
            >>> phonemes = detector.detect(audio, sr)
            >>> timeline = detector.get_phoneme_timeline(
            ...     phonemes, audio_duration=2.0
            ... )
            >>> # timeline[100] gives phoneme active at frame 100
        """
        num_frames = int(np.ceil(audio_duration / frame_duration))
        timeline = np.full(num_frames, "", dtype=object)

        for segment in segments:
            # Use small epsilon to handle floating-point boundary cases:
            # e.g. 0.3/0.1 = 2.9999... → int() = 2 (wrong), ceil-eps fixes this.
            start_frame = max(0, int(segment.start_time / frame_duration + 1e-9))
            end_frame = min(num_frames, int(np.ceil(segment.end_time / frame_duration - 1e-9)))

            # First-wins semantics: only fill frames that are still empty.
            # This ensures earlier (higher-priority) segments are not overwritten
            # by later segments that overlap the same frame region.
            for fi in range(start_frame, end_frame):
                if timeline[fi] == "":
                    timeline[fi] = segment.phoneme

        return timeline

    def get_statistics(self, segments: list[PhonemeSegment]) -> dict[str, Any]:
        """
        Berechnet statistics about detected phonemes.

        Args:
            segments: List of phoneme segments

        Returns:
            Dictionary with statistics:
            - total_phonemes: Total number of phonemes
            - unique_phonemes: Number of unique phonemes
            - avg_confidence: Average confidence score
            - min_confidence: Minimum confidence score
            - max_confidence: Maximum confidence score
            - avg_duration: Average phoneme duration
            - phoneme_counts: Dict of phoneme → count
        """
        if not segments:
            return {
                "total_phonemes": 0,
                "unique_phonemes": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "avg_duration": 0.0,
                "phoneme_counts": {},
            }

        confidences = [s.confidence for s in segments]
        durations = [s.duration for s in segments]
        phoneme_counts: dict[Any, Any] = {}

        for s in segments:
            phoneme_counts[s.phoneme] = phoneme_counts.get(s.phoneme, 0) + 1

        return {
            "total_phonemes": len(segments),
            "unique_phonemes": len(phoneme_counts),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "avg_duration": np.mean(durations),
            "phoneme_counts": phoneme_counts,
        }
