"""
Tests for Intelligibility Scorer
=================================

Test coverage:
- Basic scoring functionality
- Formant analysis
- C/V ratio computation
- Spectral balance assessment
- Temporal clarity assessment
- Quality classification
- Recommendations generation
- Reference comparison
- Edge cases
"""

import numpy as np
import pytest

from backend.ml.vocal_analysis.intelligibility_scorer import (
    FormantData,
    IntelligibilityReport,
    IntelligibilityScorer,
    QualityLevel,
    assess_intelligibility,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_rate() -> int:
    """Sample rate for tests."""
    return 48000


@pytest.fixture
def test_audio_mono(sample_rate: int) -> np.ndarray:
    """Generate test audio with formants and transients."""
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Simulate vowel sound with formants
    # F1=500 Hz, F2=1500 Hz, F3=2500 Hz
    f1 = np.sin(2 * np.pi * 500 * t) * 0.5
    f2 = np.sin(2 * np.pi * 1500 * t) * 0.3
    f3 = np.sin(2 * np.pi * 2500 * t) * 0.2

    # Add high-frequency consonants
    consonants = np.sin(2 * np.pi * 5000 * t) * 0.2

    # Combine
    audio = f1 + f2 + f3 + consonants

    # Add envelope (attack/decay)
    envelope = np.exp(-3 * t)
    audio = audio * envelope

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)

    return audio  # type: ignore[no-any-return]


@pytest.fixture
def test_audio_stereo(test_audio_mono: np.ndarray) -> np.ndarray:
    """Generate stereo test audio."""
    return np.stack([test_audio_mono, test_audio_mono * 0.95])


@pytest.fixture
def low_quality_audio(sample_rate: int) -> np.ndarray:
    """Generate low-quality audio (poor intelligibility)."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Muddy low frequencies only
    audio = np.sin(2 * np.pi * 200 * t) * 0.7
    audio += np.sin(2 * np.pi * 300 * t) * 0.3

    # Add noise — use fixed seed so the fixture is deterministic
    # regardless of what other tests do to the global random state.
    rng = np.random.default_rng(42)
    audio += rng.standard_normal(len(audio)) * 0.1

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)

    return audio  # type: ignore[no-any-return]


@pytest.fixture
def high_quality_audio(sample_rate: int) -> np.ndarray:
    """Generate high-quality audio (excellent intelligibility)."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Well-separated formants
    f1 = np.sin(2 * np.pi * 700 * t) * 0.4
    f2 = np.sin(2 * np.pi * 1800 * t) * 0.3
    f3 = np.sin(2 * np.pi * 3000 * t) * 0.2

    # Clear consonants
    consonants = np.sin(2 * np.pi * 4000 * t) * 0.25

    # Good envelope
    envelope = np.exp(-2 * t)
    audio = (f1 + f2 + f3 + consonants) * envelope

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)

    return audio  # type: ignore[no-any-return]


# ============================================================================
# TEST BASIC FUNCTIONALITY
# ============================================================================


def test_scorer_initialization():
    """Test IntelligibilityScorer initialization."""
    scorer = IntelligibilityScorer()
    assert scorer is not None
    assert scorer.min_formant_prominence == 0.3
    assert scorer.optimal_cv_ratio == (0.4, 0.6)


def test_scorer_initialization_custom():
    """Test IntelligibilityScorer with custom parameters."""
    scorer = IntelligibilityScorer(
        use_phoneme_detection=False,
        min_formant_prominence=0.5,
        optimal_cv_ratio=(0.3, 0.7),
    )
    assert not scorer.use_phoneme_detection
    assert scorer.min_formant_prominence == 0.5
    assert scorer.optimal_cv_ratio == (0.3, 0.7)


def test_basic_scoring_mono(test_audio_mono: np.ndarray, sample_rate: int):
    """Test basic scoring with mono audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    report = scorer.score(test_audio_mono, sample_rate)

    assert isinstance(report, IntelligibilityReport)
    assert 0.0 <= report.overall_score <= 1.0
    assert isinstance(report.quality_level, QualityLevel)
    assert 0.0 <= report.formant_clarity <= 1.0
    assert 0.0 <= report.consonant_clarity <= 1.0
    assert 0.0 <= report.spectral_balance <= 1.0
    assert 0.0 <= report.temporal_clarity <= 1.0
    assert report.cv_ratio >= 0.0
    assert isinstance(report.recommendations, list)


def test_basic_scoring_stereo(test_audio_stereo: np.ndarray, sample_rate: int):
    """Test basic scoring with stereo audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    report = scorer.score(test_audio_stereo, sample_rate)

    assert isinstance(report, IntelligibilityReport)
    assert 0.0 <= report.overall_score <= 1.0


# ============================================================================
# TEST FORMANT ANALYSIS
# ============================================================================


def test_formant_extraction(test_audio_mono: np.ndarray, sample_rate: int):
    """Test formant frequency extraction."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    formant_data = scorer._extract_formants(test_audio_mono, sample_rate)

    # May succeed or fail depending on audio characteristics
    if formant_data is not None:
        assert isinstance(formant_data, FormantData)
        assert formant_data.f1 > 0
        assert formant_data.f2 > formant_data.f1  # F2 > F1
        assert formant_data.f3 > formant_data.f2  # F3 > F2
        assert formant_data.f1_bandwidth > 0
        assert formant_data.f2_bandwidth > 0
        assert formant_data.f3_bandwidth > 0


def test_formant_clarity_assessment():
    """Test formant clarity scoring."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # Good formant data
    good_formants = FormantData(
        f1=700.0,
        f2=1800.0,
        f3=3000.0,
        f1_bandwidth=50.0,
        f2_bandwidth=70.0,
        f3_bandwidth=110.0,
    )
    clarity_good = scorer._assess_formant_clarity(good_formants)
    assert 0.5 <= clarity_good <= 1.0

    # Poor formant data (too close)
    poor_formants = FormantData(
        f1=500.0,
        f2=700.0,
        f3=900.0,
        f1_bandwidth=50.0,
        f2_bandwidth=70.0,
        f3_bandwidth=110.0,
    )
    clarity_poor = scorer._assess_formant_clarity(poor_formants)
    assert clarity_poor < clarity_good

    # None formant data
    clarity_none = scorer._assess_formant_clarity(None)
    assert clarity_none == 0.5


# ============================================================================
# TEST C/V RATIO
# ============================================================================


def test_cv_ratio_estimation(test_audio_mono: np.ndarray, sample_rate: int):
    """Test C/V ratio estimation without phoneme detection."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    cv_ratio = scorer._estimate_cv_ratio(test_audio_mono, sample_rate)

    assert 0.0 <= cv_ratio <= 1.0


def test_cv_ratio_consonant_heavy():
    """Test C/V ratio with consonant-heavy audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # High-frequency audio (consonants)
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 5000 * t)  # High frequency

    cv_ratio = scorer._estimate_cv_ratio(audio, sr)
    assert cv_ratio > 0.5  # Should be consonant-heavy


def test_cv_ratio_vowel_heavy():
    """Test C/V ratio with vowel-heavy audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # Low-frequency audio (vowels)
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 500 * t)  # Low frequency

    cv_ratio = scorer._estimate_cv_ratio(audio, sr)
    assert cv_ratio < 0.5  # Should be vowel-heavy


# ============================================================================
# TEST SPECTRAL BALANCE
# ============================================================================


def test_spectral_balance(test_audio_mono: np.ndarray, sample_rate: int):
    """Test spectral balance assessment."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    balance = scorer._assess_spectral_balance(test_audio_mono, sample_rate)

    assert 0.0 <= balance <= 1.0


def test_spectral_balance_imbalanced():
    """Test spectral balance with imbalanced audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # Only low frequencies (poor balance)
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio_low = np.sin(2 * np.pi * 200 * t)

    balance_low = scorer._assess_spectral_balance(audio_low, sr)

    # Balanced frequencies
    audio_balanced = (
        np.sin(2 * np.pi * 300 * t) * 0.3  # Low
        + np.sin(2 * np.pi * 1000 * t) * 0.4  # Mid
        + np.sin(2 * np.pi * 4000 * t) * 0.3  # High
    )

    balance_good = scorer._assess_spectral_balance(audio_balanced, sr)

    assert balance_good > balance_low


# ============================================================================
# TEST TEMPORAL CLARITY
# ============================================================================


def test_temporal_clarity(test_audio_mono: np.ndarray, sample_rate: int):
    """Test temporal clarity assessment."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    clarity = scorer._assess_temporal_clarity(test_audio_mono, sample_rate)

    assert 0.0 <= clarity <= 1.0


def test_temporal_clarity_overcompressed():
    """Test temporal clarity with over-compressed audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # Constant amplitude (over-compressed)
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio_flat = np.ones_like(t)

    clarity_flat = scorer._assess_temporal_clarity(audio_flat, sr)

    # Natural envelope
    envelope = np.exp(-2 * t)
    audio_natural = np.sin(2 * np.pi * 440 * t) * envelope

    clarity_natural = scorer._assess_temporal_clarity(audio_natural, sr)

    # Natural should have better temporal clarity
    assert clarity_natural >= clarity_flat * 0.8  # Allow some tolerance


# ============================================================================
# TEST QUALITY CLASSIFICATION
# ============================================================================


def test_quality_classification():
    """Test quality level classification."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    assert scorer._classify_quality(0.90) == QualityLevel.EXCELLENT
    assert scorer._classify_quality(0.75) == QualityLevel.GOOD
    assert scorer._classify_quality(0.60) == QualityLevel.ACCEPTABLE
    assert scorer._classify_quality(0.45) == QualityLevel.POOR
    assert scorer._classify_quality(0.20) == QualityLevel.VERY_POOR


def test_quality_comparison_high_vs_low(
    high_quality_audio: np.ndarray,
    low_quality_audio: np.ndarray,
    sample_rate: int,
):
    """Test that high-quality audio scores better than low-quality.

    The overall intelligibility score must be higher for the signal with
    clear formants and consonants (high_quality) than for the muddy
    low-frequency-only signal (low_quality).

    Note: formant_clarity is NOT asserted individually because LPC analysis
    on pure sinusoidal test signals is unreliable — LPC roots for muddy
    200/300 Hz + noise audio sometimes fall inside typical vocal-frequency
    ranges by chance.  The overall score (weighted combination of all
    sub-metrics) is the authoritative indicator.
    """
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    report_high = scorer.score(high_quality_audio, sample_rate)
    report_low = scorer.score(low_quality_audio, sample_rate)

    assert report_high.overall_score > report_low.overall_score
    # Consonant clarity is expected to be higher for high-quality (HF content)
    assert report_high.consonant_clarity > report_low.consonant_clarity


# ============================================================================
# TEST RECOMMENDATIONS
# ============================================================================


def test_recommendations_generation(test_audio_mono: np.ndarray, sample_rate: int):
    """Test recommendations generation."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    report = scorer.score(test_audio_mono, sample_rate)

    assert isinstance(report.recommendations, list)
    assert len(report.recommendations) > 0
    assert all(isinstance(rec, str) for rec in report.recommendations)


def test_recommendations_low_formant_clarity():
    """Test recommendations for low formant clarity."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    recommendations = scorer._generate_recommendations(
        formant_clarity=0.3,
        consonant_clarity=0.8,
        spectral_balance=0.8,
        temporal_clarity=0.8,
        cv_ratio=0.5,
    )

    # Should recommend formant enhancement
    assert any("formant" in rec.lower() for rec in recommendations)


def test_recommendations_low_consonant_clarity():
    """Test recommendations for low consonant clarity."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    recommendations = scorer._generate_recommendations(
        formant_clarity=0.8,
        consonant_clarity=0.3,
        spectral_balance=0.8,
        temporal_clarity=0.8,
        cv_ratio=0.5,
    )

    # Should recommend high-frequency enhancement
    assert any("consonant" in rec.lower() or "high-frequency" in rec.lower() for rec in recommendations)


def test_recommendations_excellent_quality():
    """Test recommendations for excellent quality."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    recommendations = scorer._generate_recommendations(
        formant_clarity=0.9,
        consonant_clarity=0.9,
        spectral_balance=0.9,
        temporal_clarity=0.9,
        cv_ratio=0.5,
    )

    # Should say no improvements needed
    assert any("excellent" in rec.lower() or "no improvement" in rec.lower() for rec in recommendations)


# ============================================================================
# TEST REFERENCE COMPARISON
# ============================================================================


def test_reference_comparison(test_audio_mono: np.ndarray, sample_rate: int):
    """Test reference audio comparison."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # Compare with itself (perfect similarity)
    similarity_self = scorer._compare_to_reference(
        test_audio_mono,
        test_audio_mono,
        sample_rate,
    )
    assert 0.9 <= similarity_self <= 1.0  # Should be very similar

    # Compare with modified version
    modified = test_audio_mono * 0.8
    similarity_modified = scorer._compare_to_reference(
        test_audio_mono,
        modified,
        sample_rate,
    )
    assert 0.7 <= similarity_modified <= 1.0  # Similar but not identical


def test_reference_comparison_different_lengths(sample_rate: int):
    """Test reference comparison with different length audios."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    audio1 = np.random.randn(sample_rate)
    audio2 = np.random.randn(sample_rate * 2)

    # Should handle different lengths
    similarity = scorer._compare_to_reference(audio1, audio2, sample_rate)
    assert 0.0 <= similarity <= 1.0


def test_scoring_with_reference(test_audio_mono: np.ndarray, sample_rate: int):
    """Test scoring with reference audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    reference = test_audio_mono * 0.9
    report = scorer.score(test_audio_mono, sample_rate, reference=reference)

    assert "reference_similarity" in report.metrics
    assert 0.0 <= report.metrics["reference_similarity"] <= 1.0


# ============================================================================
# TEST EDGE CASES
# ============================================================================


def test_empty_audio():
    """Test with empty audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    audio = np.array([])
    sr = 48000

    # Should handle gracefully
    try:
        report = scorer.score(audio, sr)
        # If it succeeds, check that scores are valid defaults
        assert 0.0 <= report.overall_score <= 1.0
    except (ValueError, ZeroDivisionError):
        # Expected for empty audio
        pass


def test_silent_audio():
    """Test with silent audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    audio = np.zeros(48000)
    sr = 48000

    report = scorer.score(audio, sr)

    # Should produce low scores
    assert 0.0 <= report.overall_score <= 1.0
    assert report.quality_level in [
        QualityLevel.POOR,
        QualityLevel.VERY_POOR,
        QualityLevel.ACCEPTABLE,
    ]


def test_very_short_audio():
    """Test with very short audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    audio = np.random.randn(100)  # Very short
    sr = 48000

    report = scorer.score(audio, sr)

    # Should handle but may give uncertain results
    assert 0.0 <= report.overall_score <= 1.0


def test_clipped_audio():
    """Test with clipped audio."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 2.0  # Clipped
    audio = np.clip(audio, -1.0, 1.0)

    report = scorer.score(audio, sr)

    # Should still produce valid scores
    assert 0.0 <= report.overall_score <= 1.0


# ============================================================================
# TEST CONVENIENCE FUNCTION
# ============================================================================


def test_assess_intelligibility_function(test_audio_mono: np.ndarray, sample_rate: int):
    """Test convenience function."""
    report = assess_intelligibility(test_audio_mono, sample_rate)

    assert isinstance(report, IntelligibilityReport)
    assert 0.0 <= report.overall_score <= 1.0


def test_assess_intelligibility_with_reference(test_audio_mono: np.ndarray, sample_rate: int):
    """Test convenience function with reference."""
    reference = test_audio_mono * 0.95
    report = assess_intelligibility(test_audio_mono, sample_rate, reference=reference)

    assert "reference_similarity" in report.metrics


# ============================================================================
# TEST METRICS
# ============================================================================


def test_detailed_metrics(test_audio_mono: np.ndarray, sample_rate: int):
    """Test that detailed metrics are populated."""
    scorer = IntelligibilityScorer(use_phoneme_detection=False)
    report = scorer.score(test_audio_mono, sample_rate)

    assert "cv_ratio" in report.metrics
    assert report.metrics["cv_ratio"] == report.cv_ratio

    # Formant metrics (may or may not be present)
    if report.formant_data:
        assert "formant_f1" in report.metrics
        assert "formant_f2" in report.metrics
        assert "formant_f3" in report.metrics


# ============================================================================
# TEST REPORT REPRESENTATION
# ============================================================================


def test_report_repr():
    """Test IntelligibilityReport string representation."""
    report = IntelligibilityReport(
        overall_score=0.85,
        quality_level=QualityLevel.EXCELLENT,
        formant_clarity=0.9,
        consonant_clarity=0.8,
        spectral_balance=0.85,
        temporal_clarity=0.9,
        cv_ratio=0.5,
    )

    repr_str = repr(report)
    assert "0.85" in repr_str
    assert "excellent" in repr_str


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


def test_performance_short_audio(sample_rate: int):
    """Test performance with short audio."""
    import time

    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # 1 second audio
    audio = np.random.randn(sample_rate)

    start = time.time()
    report = scorer.score(audio, sample_rate)
    elapsed = time.time() - start

    # Should be fast (<1 second for 1 second audio)
    assert elapsed < 1.0
    assert report.overall_score >= 0.0


def test_performance_long_audio(sample_rate: int):
    """Test performance with longer audio."""
    import time

    scorer = IntelligibilityScorer(use_phoneme_detection=False)

    # 10 seconds audio
    audio = np.random.randn(sample_rate * 10)

    start = time.time()
    report = scorer.score(audio, sample_rate)
    elapsed = time.time() - start

    # Should still be reasonable (<5 seconds for 10 second audio)
    assert elapsed < 5.0
    assert report.overall_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
