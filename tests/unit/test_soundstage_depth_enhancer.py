import numpy as np
import pytest

from dsp.soundstage_depth_enhancer import SoundstageDepthEnhancer


def _stereo_noise(sr: int = 48000, dur_s: float = 1.0) -> np.ndarray:
    n = int(sr * dur_s)
    rng = np.random.default_rng(1234)
    left = 0.1 * rng.standard_normal(n)
    right = 0.1 * rng.standard_normal(n)
    return np.stack([left, right], axis=1).astype(np.float64)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_soundstage_depth_enhancer_stereo_shape_and_finite() -> None:
    audio = _stereo_noise()
    enh = SoundstageDepthEnhancer(depth_amount=0.6, room_size=0.5)

    out, report = enh.process(audio, 48000)

    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
    assert 0.0 <= report.foreground_level <= 1.0
    assert 0.0 <= report.midground_level <= 1.0
    assert 0.0 <= report.background_level <= 1.0


def test_soundstage_depth_enhancer_handles_mono_input() -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    mono = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float64)
    enh = SoundstageDepthEnhancer(depth_amount=0.5, room_size=0.4)

    out, _ = enh.process(mono, 48000)

    assert out.ndim == 2
    assert out.shape[1] == 2
    assert np.isfinite(out).all()


def test_soundstage_depth_enhancer_rejects_wrong_sr() -> None:
    audio = _stereo_noise(sr=44100)
    enh = SoundstageDepthEnhancer()

    try:
        enh.process(audio, 44100)
        assert False, "Expected assertion for sample rate mismatch"
    except AssertionError as exc:
        assert "48000" in str(exc)
