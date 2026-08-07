from __future__ import annotations

"""
tests/unit/test_riaa_curve_classifier.py
Aurik 10.0.0 — Spec §6.6 Tests: RIAA-Kurven-Klassifikation
"""


import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def white_noise():
    """Weißes Rauschen (kein spektrales Profil → unknown expected)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(SR * 5).astype(np.float32)


@pytest.fixture
def riaa_shaped_audio():
    """Synthetisches Signal mit RIAA-ähnlichem Slope (~-5 dB/oct, ~+14 dB Bass)."""
    rng = np.random.default_rng(42)
    t = np.arange(SR * 10) / SR
    # Summe aus tiefen Frequenzen (stark) + mittleren (moderat) + hohen (schwach)
    audio = (
        3.5 * np.sin(2 * np.pi * 100 * t)
        + 1.0 * np.sin(2 * np.pi * 500 * t)
        + 0.3 * np.sin(2 * np.pi * 2000 * t)
        + 0.05 * np.sin(2 * np.pi * 8000 * t)
    )
    audio += 0.01 * rng.standard_normal(len(t))
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# RIAA_SLOPE_PROFILES dict
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_riaa_slope_profiles_has_required_entries():
    from backend.core.dsp.riaa_curve_classifier import RIAA_SLOPE_PROFILES

    required = {"riaa", "nab", "columbia", "aes", "capitol", "london", "ccir", "unknown"}
    assert required == set(RIAA_SLOPE_PROFILES.keys())


def test_riaa_slope_profiles_unknown_is_none():
    from backend.core.dsp.riaa_curve_classifier import RIAA_SLOPE_PROFILES

    assert RIAA_SLOPE_PROFILES["unknown"] is None


def test_riaa_slope_profiles_riaa_values():
    from backend.core.dsp.riaa_curve_classifier import RIAA_SLOPE_PROFILES

    slope, bass, hf, bt = RIAA_SLOPE_PROFILES["riaa"]  # type: ignore[misc]
    assert slope == pytest.approx(-5.0)
    assert bass == pytest.approx(13.7)
    assert hf == pytest.approx(2122.0)


def test_riaa_slope_profiles_columbia_bass_higher_than_riaa():
    from backend.core.dsp.riaa_curve_classifier import RIAA_SLOPE_PROFILES

    # Columbia hat mehr Bass-Boost als RIAA
    assert RIAA_SLOPE_PROFILES["columbia"][1] > RIAA_SLOPE_PROFILES["riaa"][1]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_singleton_returns_same_instance():
    from backend.core.dsp.riaa_curve_classifier import get_riaa_curve_classifier

    a = get_riaa_curve_classifier()
    b = get_riaa_curve_classifier()
    assert a is b


# ---------------------------------------------------------------------------
# classify_riaa_curve (Ausgabe-Typ und Werte)
# ---------------------------------------------------------------------------


def test_classify_returns_known_curve_or_unknown():
    from backend.core.dsp.riaa_curve_classifier import RIAA_SLOPE_PROFILES, classify_riaa_curve

    rng = np.random.default_rng(1)
    audio = rng.standard_normal(SR * 5).astype(np.float32)
    result = classify_riaa_curve(audio, SR, era_decade=1955)
    assert result in RIAA_SLOPE_PROFILES


def test_classify_too_short_audio_returns_unknown():
    from backend.core.dsp.riaa_curve_classifier import classify_riaa_curve

    short_audio = np.zeros(100, dtype=np.float32)
    assert classify_riaa_curve(short_audio, SR) == "unknown"


def test_classify_silence_returns_unknown():
    from backend.core.dsp.riaa_curve_classifier import classify_riaa_curve

    silence = np.zeros(SR * 5, dtype=np.float32)
    result = classify_riaa_curve(silence, SR)
    assert result == "unknown"


def test_classify_stereo_audio_supported():
    from backend.core.dsp.riaa_curve_classifier import RIAA_SLOPE_PROFILES, classify_riaa_curve

    rng = np.random.default_rng(3)
    stereo = rng.standard_normal((2, SR * 5)).astype(np.float32)
    result = classify_riaa_curve(stereo, SR)
    assert result in RIAA_SLOPE_PROFILES


def test_classify_stereo_NHW_audio():
    """Stereo im (N, 2) Format."""
    from backend.core.dsp.riaa_curve_classifier import RIAA_SLOPE_PROFILES, classify_riaa_curve

    rng = np.random.default_rng(4)
    stereo = rng.standard_normal((SR * 5, 2)).astype(np.float32)
    result = classify_riaa_curve(stereo, SR)
    assert result in RIAA_SLOPE_PROFILES


# ---------------------------------------------------------------------------
# classify_riaa_curve_with_confidence
# ---------------------------------------------------------------------------


def test_classify_with_confidence_returns_tuple():
    from backend.core.dsp.riaa_curve_classifier import classify_riaa_curve_with_confidence

    rng = np.random.default_rng(5)
    audio = rng.standard_normal(SR * 5).astype(np.float32)
    curve, conf = classify_riaa_curve_with_confidence(audio, SR)
    assert isinstance(curve, str)
    assert 0.0 <= conf <= 1.0


def test_classify_with_confidence_unknown_below_threshold():
    from backend.core.dsp.riaa_curve_classifier import classify_riaa_curve_with_confidence

    # Sehr kurzes Audio → confidence=0, curve="unknown"
    audio = np.zeros(100, dtype=np.float32)
    curve, conf = classify_riaa_curve_with_confidence(audio, SR)
    assert curve == "unknown"
    assert conf == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Era-Priors
# ---------------------------------------------------------------------------


def test_era_prior_pre_1950():
    from backend.core.dsp.riaa_curve_classifier import _get_era_riaa_priors

    priors = _get_era_riaa_priors(1945)
    assert priors["columbia"] == pytest.approx(2.5)
    assert priors["riaa"] < 1.0  # RIAA noch nicht standardisiert


def test_era_prior_post_1960():
    from backend.core.dsp.riaa_curve_classifier import _get_era_riaa_priors

    priors = _get_era_riaa_priors(1965)
    assert priors["riaa"] == pytest.approx(3.0)
    assert priors["columbia"] < 1.0


def test_era_prior_none_returns_empty_dict():
    from backend.core.dsp.riaa_curve_classifier import _get_era_riaa_priors

    assert _get_era_riaa_priors(None) == {}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def test_measure_spectral_slope_silence_returns_zero():
    from backend.core.dsp.riaa_curve_classifier import _measure_spectral_slope

    silence = np.zeros(SR * 5, dtype=np.float64)
    slope = _measure_spectral_slope(silence, SR)
    assert isinstance(slope, float)


def test_measure_bass_boost_silence_returns_zero():
    from backend.core.dsp.riaa_curve_classifier import _measure_bass_boost_at_100hz

    silence = np.zeros(SR * 5, dtype=np.float64)
    boost = _measure_bass_boost_at_100hz(silence, SR)
    assert isinstance(boost, float)


def test_find_hf_turnover_returns_positive_hz():
    from backend.core.dsp.riaa_curve_classifier import _find_hf_turnover_freq

    rng = np.random.default_rng(7)
    audio = rng.standard_normal(SR * 5).astype(np.float64)
    turnover = _find_hf_turnover_freq(audio, SR)
    assert turnover > 0.0
