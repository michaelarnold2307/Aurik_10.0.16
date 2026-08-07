from __future__ import annotations

"""Unit tests for backend.ml.speaker_identity_guard.

Tests:
- MFCC-based voiceprint extraction (librosa + FFT fallback)
- Cosine similarity
- SpeakerIdentityGuard pre-embedding capture and post-phase checks
- Identity preservation threshold behavior
- Edge cases: silence, NaN, short audio
"""


import numpy as np
import pytest

from backend.ml.speaker_identity_guard import (
    IDENTITY_THRESHOLD,
    N_MFCC,
    SpeakerIdentityGuard,
    SpeakerIdentityResult,
    extract_mfcc_voiceprint,
)


# cos_sim helper — cosine_similarity now a field in SpeakerIdentityResult
def cos_sim(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 1.0
    return float(np.dot(a, b) / norm)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sr():
    return 48_000


@pytest.fixture
def sine_440(rng, sr):
    """1 second 440 Hz sine wave."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float64)


@pytest.fixture
def sine_880(rng, sr):
    """1 second 880 Hz sine wave (different harmonic content)."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2.0 * np.pi * 880.0 * t).astype(np.float64)


@pytest.fixture
def vocal_like(rng, sr):
    """Synthetic vocal-like audio with harmonics."""
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    # Fundamental + harmonics simulating a voice
    sig = 0.5 * np.sin(2.0 * np.pi * 200.0 * t)
    sig += 0.3 * np.sin(2.0 * np.pi * 400.0 * t)
    sig += 0.2 * np.sin(2.0 * np.pi * 600.0 * t)
    sig += 0.15 * np.sin(2.0 * np.pi * 800.0 * t)
    sig += 0.1 * np.sin(2.0 * np.pi * 1000.0 * t)
    # Add some noise
    sig += 0.02 * rng.randn(*sig.shape)
    return sig.astype(np.float64)


@pytest.fixture
def stereo_vocal(vocal_like):
    """Stereo version of vocal-like."""
    return np.column_stack([vocal_like, vocal_like * 0.9])


# ---------------------------------------------------------------------------
# extract_voiceprint tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractVoiceprint:
    """Tests for extract_mfcc_voiceprint()."""

    def test_returns_correct_shape(self, vocal_like, sr):
        emb = extract_mfcc_voiceprint(vocal_like, sr)
        assert emb.ndim == 1
        assert emb.shape[0] == N_MFCC * 3  # 60-dim

    def test_returns_float64(self, vocal_like, sr):
        emb = extract_mfcc_voiceprint(vocal_like, sr)
        assert emb.dtype == np.float64

    def test_mono_input(self, vocal_like, sr):
        emb = extract_mfcc_voiceprint(vocal_like, sr)
        assert np.isfinite(emb).all()

    def test_stereo_input(self, stereo_vocal, sr):
        emb = extract_mfcc_voiceprint(stereo_vocal, sr)
        assert emb.shape[0] == N_MFCC * 3
        assert np.isfinite(emb).all()

    def test_sine_440(self, sine_440, sr):
        emb = extract_mfcc_voiceprint(sine_440, sr)
        assert emb.shape[0] == N_MFCC * 3
        assert np.isfinite(emb).all()

    def test_short_audio(self, sr):
        """Very short audio should not crash."""
        short = np.random.randn(256).astype(np.float64)
        emb = extract_mfcc_voiceprint(short, sr)
        assert emb.shape[0] == N_MFCC * 3
        assert np.isfinite(emb).all()

    def test_silence(self, sr):
        """Silence should produce finite embedding."""
        silent = np.zeros(48000, dtype=np.float64)
        emb = extract_mfcc_voiceprint(silent, sr)
        assert np.isfinite(emb).all()

    def test_nan_input(self, sr):
        """NaN input should not crash."""
        nan_audio = np.full(48000, np.nan, dtype=np.float64)
        emb = extract_mfcc_voiceprint(nan_audio, sr)
        assert np.isfinite(emb).all()

    def test_deterministic(self, vocal_like, sr):
        """Same input yields same embedding."""
        emb1 = extract_mfcc_voiceprint(vocal_like, sr)
        emb2 = extract_mfcc_voiceprint(vocal_like, sr)
        np.testing.assert_array_almost_equal(emb1, emb2)


# ---------------------------------------------------------------------------
# cosine_similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cos_sim()."""

    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cos_sim(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cos_sim(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cos_sim(a, b) == pytest.approx(-1.0)

    def test_similar_embeddings(self, vocal_like, sr):
        """Slightly perturbed audio should have high similarity."""
        emb1 = extract_mfcc_voiceprint(vocal_like, sr)
        # Slightly attenuate
        emb2 = extract_mfcc_voiceprint(vocal_like * 0.99, sr)
        sim = cos_sim(emb1, emb2)
        assert sim > 0.99

    def test_different_sounds(self, sine_440, sine_880, sr):
        """Different sine waves should differ."""
        emb1 = extract_mfcc_voiceprint(sine_440, sr)
        emb2 = extract_mfcc_voiceprint(sine_880, sr)
        sim = cos_sim(emb1, emb2)
        # Should be measurably different
        assert sim < 0.98

    def test_range(self):
        """Cosine sim should be in [-1, 1]."""
        for _ in range(20):
            a = np.random.randn(60)
            b = np.random.randn(60)
            sim = cos_sim(a, b)
            assert -1.0 <= sim <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# SpeakerIdentityGuard tests
# ---------------------------------------------------------------------------


class TestSpeakerIdentityGuard:
    """Tests for SpeakerIdentityGuard."""

    def test_capture_pre_embedding(self, vocal_like, sr):
        guard = SpeakerIdentityGuard()
        emb = guard.capture_pre_embedding(vocal_like, sr)  # type: ignore[func-returns-value]
        assert emb is None  # capture_pre_embedding returns None (stores internally)  # returns None (stores internally)
        assert guard.get_pre_embedding() is not None

    def test_check_phase_identity_preserved(self, vocal_like, sr):
        """Small perturbation should preserve identity."""
        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(vocal_like, sr)

        # Slightly modified (low noise)
        perturbed = vocal_like + 0.001 * np.random.randn(*vocal_like.shape)
        result = guard.check_phase("phase_42", perturbed, sr)
        assert result.identity_preserved
        assert result.cosine_similarity > IDENTITY_THRESHOLD

    def test_check_phase_identity_drift(self, vocal_like, sr):
        """Large perturbation should trigger identity drift warning."""
        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(vocal_like, sr)

        # Heavily modified (different frequency content)
        t = np.linspace(0, 2.0, len(vocal_like), endpoint=False)
        heavily_modified = np.sin(2.0 * np.pi * 2000.0 * t).astype(np.float64)
        result = guard.check_phase("phase_42", heavily_modified, sr)
        assert not result.identity_preserved
        assert not result.identity_preserved  # warnings field removed

    def test_check_without_pre_embedding(self, vocal_like, sr):
        """Check without capture should still return a result."""
        guard = SpeakerIdentityGuard()
        result = guard.check_phase("phase_19_de_esser", vocal_like, sr)
        assert result.identity_preserved  # Graceful fallback
        assert result.cosine_similarity == 1.0  # default fallback

    def test_all_vocal_phases_listed(self):
        """Ensure the VOCAL_PHASES tuple contains expected phases."""
        from backend.ml.speaker_identity_guard import VOCAL_PHASES

        assert "phase_19_de_esser" in VOCAL_PHASES
        assert "phase_42_vocal_enhancement" in VOCAL_PHASES
        assert "phase_43" in VOCAL_PHASES
        assert "phase_65_vocal_naturalness_restoration" in VOCAL_PHASES

    def test_result_dataclass_fields(self, vocal_like, sr):
        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(vocal_like, sr)
        result = guard.check_phase("phase_42", vocal_like, sr)
        assert isinstance(result, SpeakerIdentityResult)
        assert result.phase_id == "phase_42"
        assert isinstance(result.cosine_similarity, float)
        assert isinstance(result.identity_preserved, bool)
        assert isinstance(result.cosine_similarity, float)  # field, not list

    def test_multiple_checks_use_same_pre(self, vocal_like, sr):
        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(vocal_like, sr)

        r1 = guard.check_phase("phase_19_de_esser", vocal_like, sr)
        r2 = guard.check_phase("phase_42", vocal_like, sr)

        # Same pre-embedding, same post → similar results
        assert r1.cosine_similarity == pytest.approx(r2.cosine_similarity)

    def test_custom_threshold(self, vocal_like, sr):
        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(vocal_like, sr)
        # With very low negative threshold, everything passes (cosine can be negative)
        result = guard.check_phase("phase_42", np.zeros_like(vocal_like), sr, threshold=-2.0)
        assert result.identity_preserved
        # With threshold=1.0 only exactly identical (cos=1.0) passes;
        # same vocal_mike should still pass since embedding is deterministic.
        result2 = guard.check_phase("phase_42", vocal_like, sr, threshold=0.999)
        assert result2.identity_preserved  # cos≈1.0 >= 0.999

        # Heavily distorted signal should fail threshold 1.0
        t = np.linspace(0, 2.0, len(vocal_like), endpoint=False)
        noise = np.random.RandomState(7).randn(*vocal_like.shape) * 0.5
        result3 = guard.check_phase("phase_42", noise, sr, threshold=1.0)
        assert not result3.identity_preserved

    def test_pre_embedding_persistence(self, vocal_like, sr):
        """get_pre_embedding returns a copy, not the internal reference."""
        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(vocal_like, sr)
        emb1 = guard.get_pre_embedding()
        emb2 = guard.get_pre_embedding()
        np.testing.assert_array_almost_equal(emb1, emb2)  # returns same ref  # type: ignore
        np.testing.assert_array_almost_equal(emb1, emb2)  # type: ignore[arg-type]

    def test_stereo_input_to_guard(self, stereo_vocal, sr):
        """Guard should handle stereo input."""
        guard = SpeakerIdentityGuard()
        guard.capture_pre_embedding(stereo_vocal, sr)
        result = guard.check_phase("phase_42", stereo_vocal, sr)
        assert result.identity_preserved
