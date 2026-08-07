import logging

import pytest

"""Unit tests for the productive DSP phoneme boundary detector."""

import numpy as np

logger = logging.getLogger(__name__)


def _fricative_noise(sr: int = 48000, duration: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 0.08, int(sr * duration)).astype(np.float32)
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, 3000.0, btype="highpass", fs=sr, output="sos")
        return sosfiltfilt(sos, noise).astype(np.float32)  # type: ignore[no-any-return]
    except Exception:
        logger.warning("test fallback", exc_info=True)
        return noise


@pytest.mark.unit
def test_detects_fricative_frames_from_high_band_noise():
    from backend.core.dsp.phoneme_boundary_detector import PhonemeClass, get_phoneme_features_dsp

    features = get_phoneme_features_dsp(_fricative_noise(), 48000, hop_length=512)
    classes = [feature.phoneme_class for feature in features]
    assert PhonemeClass.FRICATIVE in classes
    assert max(feature.high_band_ratio for feature in features) > 0.20


def test_protection_mask_marks_plosive_and_fricative_samples():
    from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_protection_mask_dsp

    audio = np.zeros(48000, dtype=np.float32)
    audio[12000:12200] = np.hanning(200).astype(np.float32) * 0.9
    audio[24000 : 24000 + 6000] = _fricative_noise(duration=6000 / 48000.0)

    mask = detect_phoneme_protection_mask_dsp(audio, 48000, hop_length=512)
    assert mask.dtype == bool
    assert mask.shape == audio.shape
    assert np.any(mask[11800:12400])
    assert np.any(mask[24000:30000])


def test_channels_last_protection_mask_preserves_sample_length():
    from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_protection_mask_dsp

    mono = _fricative_noise(duration=0.2)
    audio = np.stack([mono, mono * 0.8], axis=1)
    mask = detect_phoneme_protection_mask_dsp(audio, 48000, hop_length=512)
    assert mask.shape == (audio.shape[0],)
    assert mask.dtype == bool


def test_invalid_hop_length_is_sanitized():
    from backend.core.dsp.phoneme_boundary_detector import (
        detect_phoneme_boundaries_dsp,
        detect_phoneme_protection_mask_dsp,
        get_phoneme_features_dsp,
    )

    audio = _fricative_noise(duration=0.05)
    assert detect_phoneme_boundaries_dsp(audio, 48000, hop_length=0).dtype == bool
    assert detect_phoneme_protection_mask_dsp(audio, 48000, hop_length=0).shape == audio.shape
    assert get_phoneme_features_dsp(audio, 48000, hop_length=0)


def test_lge_classifier_uses_dsp_detector_for_fricative():
    from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

    phoneme_type = LyricsGuidedEnhancement._classify_phoneme_type(
        _fricative_noise(duration=0.18),
        48000,
        _mean_energy=0.8,
        is_stressed=True,
    )
    assert phoneme_type == "fricative_stressed"
