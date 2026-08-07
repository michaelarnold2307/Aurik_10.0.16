"""
Tests for PhonemeDetector - Wav2Vec2-based phoneme detection

Tests the phoneme detection logic including:
- Model initialization and configuration
- Audio preprocessing
- Phoneme detection (requires torch/transformers)
- Timeline generation
- Statistics computation

Note: Tests requiring torch/transformers will be skipped if not available.

Author: Aurik Development Team
Version: 1.0.0
"""

import numpy as np
import pytest

# Check if dependencies are available.
# Use broad except to catch UserWarning-as-error (-W error) from broken optional
# torch installs that trigger resampy → pkg_resources → DeprecationWarning.
torch = None
try:
    import torch  # type: ignore[assignment]

    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Try to import PhonemeDetector (may fail without dependencies)
from backend.ml.phoneme_aware import DetectionConfig, Language, PhonemeSegment

# Import PhonemeDetector separately to check availability
try:
    from backend.ml.phoneme_aware import PhonemeDetector

    DETECTOR_AVAILABLE = True
except Exception:
    DETECTOR_AVAILABLE = False


class TestPhonemeSegment:
    """Test PhonemeSegment dataclass"""

    def test_initialization(self):
        """Test segment initialization"""
        segment = PhonemeSegment(phoneme="s", start_time=0.0, end_time=0.1, confidence=0.95, frame_index=0)

        assert segment.phoneme == "s"
        assert segment.start_time == 0.0
        assert segment.end_time == 0.1
        assert segment.confidence == 0.95
        assert segment.frame_index == 0

    def test_duration_property(self):
        """Test duration property calculation"""
        segment = PhonemeSegment("a", 1.0, 1.5, 0.9, 100)
        assert segment.duration == 0.5

        segment2 = PhonemeSegment("e", 0.0, 0.02, 0.8, 0)
        assert abs(segment2.duration - 0.02) < 1e-6

    def test_repr(self):
        """Test string representation"""
        segment = PhonemeSegment("s", 0.0, 0.1, 0.95, 0)
        repr_str = repr(segment)

        assert "s" in repr_str
        assert "0.00" in repr_str or "0.0" in repr_str
        assert "0.95" in repr_str


class TestDetectionConfig:
    """Test DetectionConfig dataclass"""

    def test_default_initialization(self):
        """Test config with default values"""
        config = DetectionConfig()

        assert config.model_name == "facebook/wav2vec2-lv-60-espeak-cv-ft"
        assert config.language == Language.ENGLISH
        assert config.min_confidence == 0.5
        assert config.target_sample_rate == 16000
        assert config.use_gpu is True
        assert config.cache_dir is None

    def test_custom_initialization(self):
        """Test config with custom values"""
        config = DetectionConfig(
            model_name="custom/model",
            language=Language.GERMAN,
            min_confidence=0.7,
            target_sample_rate=22050,
            use_gpu=False,
            cache_dir="/tmp/cache",  # type: ignore[arg-type]
        )

        assert config.model_name == "custom/model"
        assert config.language == Language.GERMAN
        assert config.min_confidence == 0.7
        assert config.target_sample_rate == 22050
        assert config.use_gpu is False
        assert config.cache_dir == "/tmp/cache"

    def test_validation_min_confidence(self):
        """Test validation of min_confidence"""
        with pytest.raises(ValueError, match="min_confidence must be in"):
            DetectionConfig(min_confidence=-0.1)

        with pytest.raises(ValueError, match="min_confidence must be in"):
            DetectionConfig(min_confidence=1.5)

        # Valid values should work
        config1 = DetectionConfig(min_confidence=0.0)
        assert config1.min_confidence == 0.0

        config2 = DetectionConfig(min_confidence=1.0)
        assert config2.min_confidence == 1.0

    def test_validation_sample_rate(self):
        """Test validation of target_sample_rate"""
        with pytest.raises(ValueError, match="target_sample_rate must be positive"):
            DetectionConfig(target_sample_rate=0)

        with pytest.raises(ValueError, match="target_sample_rate must be positive"):
            DetectionConfig(target_sample_rate=-16000)

        # Valid values should work
        config = DetectionConfig(target_sample_rate=16000)
        assert config.target_sample_rate == 16000


class TestLanguageEnum:
    """Test Language enum"""

    def test_language_values(self):
        """Test language enum values"""
        assert Language.ENGLISH.value == "en"
        assert Language.GERMAN.value == "de"
        assert Language.SPANISH.value == "es"
        assert Language.FRENCH.value == "fr"
        assert Language.ITALIAN.value == "it"
        assert Language.PORTUGUESE.value == "pt"
        assert Language.DUTCH.value == "nl"
        assert Language.POLISH.value == "pl"

    def test_language_count(self):
        """Test expected number of languages"""
        languages = list(Language)
        assert len(languages) == 8


@pytest.mark.ml
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Requires torch/transformers")
class TestPhonemeDetectorInitialization:
    """Test PhonemeDetector initialization (without loading model)"""

    def test_initialization_default(self):
        """Test detector initialization with defaults"""
        detector = PhonemeDetector()

        assert detector.config.language == Language.ENGLISH
        assert detector.config.min_confidence == 0.5
        assert detector._model is None  # Lazy loading
        assert detector._processor is None
        assert detector._device is None

    def test_initialization_custom_config(self):
        """Test detector initialization with custom config"""
        config = DetectionConfig(language=Language.GERMAN, min_confidence=0.7, use_gpu=False)
        detector = PhonemeDetector(config)

        assert detector.config.language == Language.GERMAN
        assert detector.config.min_confidence == 0.7
        assert detector.config.use_gpu is False

    @pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Requires torch/transformers")
    def test_device_property_cpu(self):
        """Test device property (CPU mode)"""
        config = DetectionConfig(use_gpu=False)
        detector = PhonemeDetector(config)

        device = detector.device
        assert isinstance(device, torch.device)  # type: ignore[attr-defined]
        assert device.type == "cpu"

    def test_device_property_gpu(self):
        """Test device property in GPU mode request (CPU-only runtime policy)."""
        config = DetectionConfig(use_gpu=True)
        detector = PhonemeDetector(config)

        device = detector.device
        assert isinstance(device, torch.device)  # type: ignore[attr-defined]
        assert device.type == "cpu"


@pytest.mark.ml
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Requires torch/transformers")
class TestPhonemeDetectorPreprocessing:
    """Test audio preprocessing methods"""

    def test_preprocess_mono_audio(self):
        """Test preprocessing of mono audio"""
        detector = PhonemeDetector()

        # Generate mono audio at 16kHz (no resampling needed)
        sr = 16000
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))

        processed = detector._preprocess_audio(audio, sr)

        assert processed.ndim == 1  # Should remain mono
        assert len(processed) == len(audio)  # Same length (no resampling at 16kHz)

    def test_preprocess_stereo_to_mono(self):
        """Test stereo to mono conversion"""
        detector = PhonemeDetector()

        # Generate stereo audio
        sr = 16000
        duration = 1.0
        mono = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
        stereo = np.stack([mono, mono * 0.8])  # 2 channels

        processed = detector._preprocess_audio(stereo, sr)

        assert processed.ndim == 1  # Should be converted to mono
        assert len(processed) == mono.shape[0]

    def test_preprocess_resampling(self):
        """Test audio resampling"""
        detector = PhonemeDetector()

        # Generate audio at 44.1kHz (needs resampling to 16kHz)
        sr = 44100
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))

        processed = detector._preprocess_audio(audio, sr)

        assert processed.ndim == 1
        # Length should be approximately (16000/44100) * original
        expected_len = int(len(audio) * 16000 / sr)
        assert abs(len(processed) - expected_len) < 100  # Allow small difference

    def test_preprocess_normalization(self):
        """Test audio normalization"""
        detector = PhonemeDetector()

        # Generate audio with non-standard amplitude
        sr = 16000
        duration = 0.5
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 10.0

        processed = detector._preprocess_audio(audio, sr)

        # Should be normalized (max absolute value should be reasonable)
        assert np.max(np.abs(processed)) <= 1.0


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Requires torch/transformers")
class TestPhonemeDetectorDetection:
    """Test phoneme detection (requires model download)"""

    @pytest.fixture
    def detector(self):
        """Create detector for tests"""
        config = DetectionConfig(use_gpu=False, min_confidence=0.3)
        return PhonemeDetector(config)

    @pytest.fixture
    def test_audio(self):
        """Generate test audio"""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Simple sine wave as placeholder
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        return audio, sr

    @pytest.mark.slow  # Mark as slow because it downloads model
    def test_detect_basic(self, detector, test_audio):
        """Test basic phoneme detection"""
        audio, sr = test_audio

        phonemes = detector.detect(audio, sr)

        assert isinstance(phonemes, list)
        # With sine wave, may detect some phonemes or none
        # Just check structure is correct
        for segment in phonemes:
            assert isinstance(segment, PhonemeSegment)
            assert 0 <= segment.confidence <= 1.0
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time
            assert segment.duration > 0

    @pytest.mark.slow
    def test_detect_empty_audio(self, detector):
        """Test detection with empty audio"""
        audio = np.zeros(16000)  # 1 second of silence
        sr = 16000

        phonemes = detector.detect(audio, sr)

        # Should return empty list or very few phonemes
        assert isinstance(phonemes, list)
        assert len(phonemes) < 5  # Silence shouldn't produce many phonemes

    @pytest.mark.slow
    def test_detect_with_language_override(self, detector, test_audio):
        """Test detection with language override"""
        audio, sr = test_audio

        phonemes = detector.detect(audio, sr, language=Language.GERMAN)

        assert isinstance(phonemes, list)

    @pytest.mark.slow
    def test_detect_with_confidence_override(self, detector, test_audio):
        """Test detection with confidence threshold override"""
        audio, sr = test_audio

        # High confidence threshold should give fewer results
        phonemes_high = detector.detect(audio, sr, min_confidence=0.9)
        phonemes_low = detector.detect(audio, sr, min_confidence=0.1)

        assert isinstance(phonemes_high, list)
        assert isinstance(phonemes_low, list)
        assert len(phonemes_high) <= len(phonemes_low)


@pytest.mark.ml
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Requires torch/transformers")
class TestPhonemeDetectorTimeline:
    """Test phoneme timeline generation"""

    def test_get_phoneme_timeline_basic(self):
        """Test timeline generation from segments"""
        detector = PhonemeDetector()

        # Create test segments
        segments = [
            PhonemeSegment("h", 0.0, 0.1, 0.9, 0),
            PhonemeSegment("a", 0.1, 0.3, 0.9, 10),
            PhonemeSegment("s", 0.3, 0.4, 0.9, 30),
        ]

        timeline = detector.get_phoneme_timeline(segments, audio_duration=0.4, frame_duration=0.1)

        assert len(timeline) == 4  # 0.4s / 0.1s = 4 frames
        assert timeline[0] == "h"  # Frame 0: 0-0.1s
        assert timeline[1] == "a"  # Frame 1: 0.1-0.2s
        assert timeline[2] == "a"  # Frame 2: 0.2-0.3s
        assert timeline[3] == "s"  # Frame 3: 0.3-0.4s

    def test_get_phoneme_timeline_overlaps(self):
        """Test timeline with overlapping segments (takes first)"""
        detector = PhonemeDetector()

        segments = [
            PhonemeSegment("a", 0.0, 0.15, 0.9, 0),
            PhonemeSegment("b", 0.1, 0.2, 0.9, 10),
        ]

        timeline = detector.get_phoneme_timeline(segments, audio_duration=0.2, frame_duration=0.1)

        assert len(timeline) == 2
        assert timeline[0] == "a"  # Frame 0: 0-0.1s (covered by 'a')
        assert timeline[1] == "a"  # Frame 1: 0.1-0.2s ('a' comes first, takes precedence)

    def test_get_phoneme_timeline_gaps(self):
        """Test timeline with gaps (silence)"""
        detector = PhonemeDetector()

        segments = [
            PhonemeSegment("a", 0.0, 0.1, 0.9, 0),
            PhonemeSegment("s", 0.2, 0.3, 0.9, 20),
        ]

        timeline = detector.get_phoneme_timeline(segments, audio_duration=0.3, frame_duration=0.1)

        assert len(timeline) == 3
        assert timeline[0] == "a"
        assert timeline[1] == ""  # Gap
        assert timeline[2] == "s"


@pytest.mark.ml
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Requires torch/transformers")
class TestPhonemeDetectorStatistics:
    """Test statistics computation"""

    def test_get_statistics_basic(self):
        """Test statistics computation"""
        detector = PhonemeDetector()

        segments = [
            PhonemeSegment("s", 0.0, 0.1, 0.95, 0),
            PhonemeSegment("a", 0.1, 0.2, 0.90, 10),
            PhonemeSegment("s", 0.2, 0.3, 0.85, 20),
            PhonemeSegment("t", 0.3, 0.4, 0.80, 30),
        ]

        stats = detector.get_statistics(segments)

        assert stats["total_phonemes"] == 4
        assert stats["unique_phonemes"] == 3  # s, a, t
        assert abs(stats["avg_confidence"] - 0.875) < 0.01  # (0.95+0.90+0.85+0.80)/4
        assert abs(stats["avg_duration"] - 0.1) < 0.01  # All 0.1s
        assert stats["min_confidence"] == 0.80
        assert stats["max_confidence"] == 0.95

    def test_get_statistics_empty(self):
        """Test statistics with empty segments"""
        detector = PhonemeDetector()

        stats = detector.get_statistics([])

        assert stats["total_phonemes"] == 0
        assert stats["unique_phonemes"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["avg_duration"] == 0.0

    def test_get_statistics_phoneme_counts(self):
        """Test phoneme counts in statistics"""
        detector = PhonemeDetector()

        segments = [
            PhonemeSegment("s", 0.0, 0.1, 0.9, 0),
            PhonemeSegment("s", 0.1, 0.2, 0.9, 10),
            PhonemeSegment("a", 0.2, 0.3, 0.9, 20),
        ]

        stats = detector.get_statistics(segments)

        assert stats["phoneme_counts"]["s"] == 2
        assert stats["phoneme_counts"]["a"] == 1


@pytest.mark.ml
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Requires torch/transformers")
class TestPhonemeDetectorEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="PhonemeDetector requires torch/transformers")
    def test_detect_very_short_audio(self):
        """Test detection with very short audio"""
        detector = PhonemeDetector()

        # 0.1 second audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600))
        sr = 16000

        # Should not crash, may return empty list
        phonemes = detector.detect(audio, sr)
        assert isinstance(phonemes, list)

    @pytest.mark.skipif(not DETECTOR_AVAILABLE, reason="PhonemeDetector requires torch/transformers")
    def test_segment_duration_property(self):
        """Test segment duration calculation"""
        seg1 = PhonemeSegment("a", 0.0, 0.5, 0.9, 0)
        assert seg1.duration == 0.5

        seg2 = PhonemeSegment("b", 1.0, 1.001, 0.9, 100)
        assert abs(seg2.duration - 0.001) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
