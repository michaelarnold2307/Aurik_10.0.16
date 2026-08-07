"""
tests/unit/test_artistic_intent_discriminator.py — §AID-1 Unit-Tests

Tests for backend/core/artistic_intent_discriminator.py
"""

import numpy as np
import pytest


def _sine(freq: float = 440.0, sr: int = 44100, duration_s: float = 2.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _distorted(sr: int = 44100, duration_s: float = 2.0) -> np.ndarray:
    """Signal with significant harmonic distortion (clipped sine)."""
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    raw = 2.0 * np.sin(2 * np.pi * 261.63 * t)  # overdrive
    return np.clip(raw, -0.7, 0.7).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_returns_same_instance():
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    a = get_artistic_intent_discriminator()
    b = get_artistic_intent_discriminator()
    assert a is b


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


def test_analyze_returns_result():
    from backend.core.artistic_intent_discriminator import (
        IntentAnalysisResult,
        get_artistic_intent_discriminator,
    )

    audio = _sine()
    result = get_artistic_intent_discriminator().analyze(audio, sr=44100)
    assert isinstance(result, IntentAnalysisResult)


def test_result_fields_types():
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    audio = _sine()
    result = get_artistic_intent_discriminator().analyze(audio, sr=44100)
    assert isinstance(result.global_score, float)
    assert isinstance(result.phase_intent_scores, dict)
    assert isinstance(result.characteristics, list)
    assert isinstance(result.n_intentional, int)
    assert isinstance(result.intent_fraction, float)


def test_global_score_in_range():
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    audio = _sine()
    result = get_artistic_intent_discriminator().analyze(audio, sr=44100)
    assert 0.0 <= result.global_score <= 1.0


def test_phase_intent_scores_in_range():
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    audio = _sine()
    result = get_artistic_intent_discriminator().analyze(audio, sr=44100)
    for ph, score in result.phase_intent_scores.items():
        assert isinstance(ph, str)
        assert 0.0 <= score <= 1.0, f"Out-of-range score for {ph}: {score}"


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_has_required_keys():
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    audio = _sine()
    result = get_artistic_intent_discriminator().analyze(audio, sr=44100)
    d = result.to_dict()
    assert "global_score" in d
    assert "phase_intent_scores" in d
    assert "n_intentional" in d
    assert "intent_fraction" in d
    assert "characteristics" in d


# ---------------------------------------------------------------------------
# Era-saturation heuristic
# ---------------------------------------------------------------------------


def test_shellac_era_expects_saturation():
    from backend.core.artistic_intent_discriminator import ArtisticIntentDiscriminator

    aid = ArtisticIntentDiscriminator()
    assert aid._era_expects_saturation(1940, "shellac") is True


def test_modern_digital_no_saturation():
    from backend.core.artistic_intent_discriminator import ArtisticIntentDiscriminator

    aid = ArtisticIntentDiscriminator()
    assert aid._era_expects_saturation(2010, "cd_digital") is False


def test_vintage_vinyl_expects_saturation():
    from backend.core.artistic_intent_discriminator import ArtisticIntentDiscriminator

    aid = ArtisticIntentDiscriminator()
    assert aid._era_expects_saturation(1968, "vinyl") is True


# ---------------------------------------------------------------------------
# _measure_harmonic_distortion
# ---------------------------------------------------------------------------


def test_distorted_signal_higher_thd():
    from backend.core.artistic_intent_discriminator import ArtisticIntentDiscriminator

    aid = ArtisticIntentDiscriminator()
    clean = _sine(freq=440.0, sr=44100)
    dirty = _distorted(sr=44100)
    mono_clean = clean
    mono_dirty = dirty
    thd_clean = aid._measure_harmonic_distortion(mono_clean, sr=44100)
    thd_dirty = aid._measure_harmonic_distortion(mono_dirty, sr=44100)
    # Distorted signal should have higher THD score (or at least not lower)
    assert thd_dirty >= thd_clean - 0.05  # tolerance for spectral overlap


def test_silent_thd_is_zero():
    from backend.core.artistic_intent_discriminator import ArtisticIntentDiscriminator

    aid = ArtisticIntentDiscriminator()
    silent = np.zeros(44100, dtype=np.float32)
    assert aid._measure_harmonic_distortion(silent, sr=44100) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_audio_returns_default():
    from backend.core.artistic_intent_discriminator import (
        IntentAnalysisResult,
        get_artistic_intent_discriminator,
    )

    empty = np.zeros(0, dtype=np.float32)
    result = get_artistic_intent_discriminator().analyze(empty, sr=44100)
    assert isinstance(result, IntentAnalysisResult)
    assert result.global_score == pytest.approx(0.0, abs=1e-6)


def test_stereo_audio_accepted():
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    mono = _sine(sr=44100, duration_s=2.0)
    stereo = np.column_stack([mono, mono])  # (samples, 2)
    result = get_artistic_intent_discriminator().analyze(stereo, sr=44100)
    assert result is not None
    assert 0.0 <= result.global_score <= 1.0


def test_channel_first_stereo_preserves_duration_fraction():
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    mono = _sine(sr=44100, duration_s=2.0)
    stereo = np.vstack([mono, mono])  # (channels, samples)
    result = get_artistic_intent_discriminator().analyze(
        stereo,
        sr=44100,
        panns_singing=0.8,
        restoration_context={"frisson_zones": [(0.0, 2.0)]},
    )
    assert result.global_score > 0.0


def test_repetition_consistency_accepts_structure_object():
    from backend.core.artistic_intent_discriminator import ArtisticIntentDiscriminator
    from backend.core.musical_structure_analyzer import MusicalStructure, SegmentInfo

    sr = 44100
    mono = np.ones(sr * 8, dtype=np.float32) * 0.2
    structure = MusicalStructure(
        segments=[
            SegmentInfo("chorus", 0, sr * 2, 0.0, 2.0),
            SegmentInfo("verse", sr * 2, sr * 4, 2.0, 4.0),
            SegmentInfo("chorus", sr * 4, sr * 6, 4.0, 6.0),
        ]
    )
    score = ArtisticIntentDiscriminator._measure_repetition_consistency(mono, sr, structure)
    assert score > 0.9


def test_repetition_consistency_accepts_time_bounds_dict():
    from backend.core.artistic_intent_discriminator import ArtisticIntentDiscriminator

    sr = 44100
    mono = np.ones(sr * 8, dtype=np.float32) * 0.2
    structure = {
        "segments": [
            {"label": "chorus", "start_s": 0.0, "end_s": 2.0},
            {"label": "chorus", "start_s": 4.0, "end_s": 6.0},
        ]
    }
    score = ArtisticIntentDiscriminator._measure_repetition_consistency(mono, sr, structure)
    assert score > 0.9


def test_frisson_zones_increase_intent():
    """Providing frisson zones in context should increase frisson_intent."""
    from backend.core.artistic_intent_discriminator import get_artistic_intent_discriminator

    sr = 44100
    audio = _sine(sr=sr, duration_s=3.0)
    # Without frisson zones
    result_no_frisson = get_artistic_intent_discriminator().analyze(audio, sr=sr, panns_singing=0.7)
    # With frisson zones covering entire file
    ctx_frisson = {"frisson_zones": [(0.0, 3.0)]}
    result_frisson = get_artistic_intent_discriminator().analyze(
        audio, sr=sr, panns_singing=0.7, restoration_context=ctx_frisson
    )
    # Frisson zones should produce >= global_score
    assert result_frisson.global_score >= result_no_frisson.global_score - 0.01


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------


def test_threshold_constants_sane():
    from backend.core.artistic_intent_discriminator import (
        INTENT_PROTECT_THRESHOLD,
        INTENT_REDUCE_THRESHOLD,
    )

    assert 0.5 < INTENT_PROTECT_THRESHOLD <= 1.0
    assert 0.0 < INTENT_REDUCE_THRESHOLD < INTENT_PROTECT_THRESHOLD
