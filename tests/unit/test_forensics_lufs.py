"""Unit tests for forensics integrated LUFS approximation."""

import numpy as np


def _sine(amplitude: float, sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


def test_integrated_lufs_increases_with_level():
    from backend.core.forensics.analysis_and_modules import _integrated_lufs_bs1770_approx

    quiet = _integrated_lufs_bs1770_approx(_sine(0.01), 48000)
    loud = _integrated_lufs_bs1770_approx(_sine(0.20), 48000)
    assert loud > quiet + 20.0
    assert -100.0 <= quiet <= 6.0
    assert -100.0 <= loud <= 6.0


def test_integrated_lufs_handles_stereo_and_silence():
    from backend.core.forensics.analysis_and_modules import _integrated_lufs_bs1770_approx

    stereo = np.stack([_sine(0.05), _sine(0.05)], axis=0)
    lufs = _integrated_lufs_bs1770_approx(stereo, 48000)
    silence = _integrated_lufs_bs1770_approx(np.zeros(48000, dtype=np.float32), 48000)
    assert -100.0 <= lufs <= 6.0
    assert silence <= -70.0
