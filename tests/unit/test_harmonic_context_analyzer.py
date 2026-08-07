import pytest

"""
tests/unit/test_harmonic_context_analyzer.py — §HCA-1 Unit-Tests

Tests for backend/core/harmonic_context_analyzer.py
"""

import warnings

import numpy as np


def _make_sine(freq_hz: float, sr: int = 44100, duration_s: float = 3.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]


def _make_chord(freqs: list, sr: int = 44100, duration_s: float = 3.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    sig = np.zeros_like(t)
    for f in freqs:
        sig += 0.25 * np.sin(2 * np.pi * f * t)
    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_returns_same_instance():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    a = get_harmonic_context_analyzer()
    b = get_harmonic_context_analyzer()
    assert a is b


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


def test_analyze_returns_result_fields():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    audio = _make_chord([261.63, 329.63, 392.0], sr=44100, duration_s=2.0)
    result = get_harmonic_context_analyzer().analyze(audio, sr=44100)

    assert hasattr(result, "chord_sequence")
    assert hasattr(result, "chord_confidence")
    assert hasattr(result, "key_root")
    assert hasattr(result, "key_mode")
    assert hasattr(result, "harmonic_density")
    assert hasattr(result, "harmonic_mask")
    assert hasattr(result, "analysis_confidence")
    assert hasattr(result, "sr")
    assert hasattr(result, "hop_length")


def test_analyze_result_types():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    audio = _make_chord([261.63, 329.63, 392.0], sr=44100, duration_s=2.0)
    result = get_harmonic_context_analyzer().analyze(audio, sr=44100)

    assert isinstance(result.chord_sequence, list)
    assert isinstance(result.key_root, str)
    assert isinstance(result.key_mode, str)
    assert isinstance(result.harmonic_density, np.ndarray)
    assert isinstance(result.harmonic_mask, np.ndarray)
    assert isinstance(result.analysis_confidence, float)
    assert 0.0 <= result.analysis_confidence <= 1.0


def test_harmonic_mask_shape():
    from backend.core.harmonic_context_analyzer import HarmonicContextAnalyzer, get_harmonic_context_analyzer

    sr = 44100
    duration_s = 2.0
    audio = _make_sine(440.0, sr=sr, duration_s=duration_s)
    result = get_harmonic_context_analyzer().analyze(audio, sr=sr)

    n_fft = HarmonicContextAnalyzer.N_FFT
    expected_bins = n_fft // 2 + 1
    assert result.harmonic_mask.shape[0] == expected_bins
    assert result.harmonic_mask.ndim == 2
    # Values should be in [0, 1]
    assert float(result.harmonic_mask.min()) >= 0.0
    assert float(result.harmonic_mask.max()) <= 1.0


def test_harmonic_density_range():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    audio = _make_chord([261.63, 329.63, 392.0], sr=44100, duration_s=2.0)
    result = get_harmonic_context_analyzer().analyze(audio, sr=44100)
    assert float(result.harmonic_density.min()) >= 0.0
    assert float(result.harmonic_density.max()) <= 1.0


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_serializable():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    audio = _make_sine(440.0, sr=44100, duration_s=1.5)
    result = get_harmonic_context_analyzer().analyze(audio, sr=44100)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "key_root" in d
    assert "key_mode" in d
    assert "analysis_confidence" in d
    # numpy arrays should be serialized as lists
    assert not isinstance(d.get("harmonic_density"), np.ndarray)
    assert not isinstance(d.get("harmonic_mask"), np.ndarray)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_silent_audio_does_not_crash():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    silent = np.zeros(44100, dtype=np.float32)
    result = get_harmonic_context_analyzer().analyze(silent, sr=44100)
    assert result is not None
    assert isinstance(result.key_root, str)


def test_very_short_audio_does_not_crash():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    short = np.random.randn(512).astype(np.float32)
    result = get_harmonic_context_analyzer().analyze(short, sr=44100)
    assert result is not None


def test_stereo_audio_accepted():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    mono = _make_chord([261.63, 329.63], sr=44100, duration_s=2.0)
    stereo = np.column_stack([mono, mono])  # (samples, 2)
    result = get_harmonic_context_analyzer().analyze(stereo, sr=44100)
    assert result is not None
    assert result.harmonic_mask.ndim == 2


def test_channels_first_short_stereo_does_not_warn_about_fft_length():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    mono = _make_chord([261.63, 329.63], sr=48000, duration_s=1.0)
    stereo = np.vstack([mono, mono])  # UV3 layout: (channels, samples)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
        warnings.filterwarnings("error", message=".*nperseg.*greater than input length.*", category=UserWarning)
        result = get_harmonic_context_analyzer().analyze(stereo, sr=48000)
    assert result is not None
    assert result.harmonic_mask.ndim == 2


# ---------------------------------------------------------------------------
# Modulation frames
# ---------------------------------------------------------------------------


def test_modulation_frames_is_list_of_ints():
    from backend.core.harmonic_context_analyzer import get_harmonic_context_analyzer

    audio = _make_chord([261.63, 329.63, 392.0], sr=44100, duration_s=3.0)
    result = get_harmonic_context_analyzer().analyze(audio, sr=44100)
    assert isinstance(result.modulation_frames, list)
    for f in result.modulation_frames:
        assert isinstance(f, int)
