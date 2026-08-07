"""tests/unit/test_phase_01_click_removal.py — §v10.700 I1: Unit-Tests für Phase 01."""

import numpy as np
import pytest

from backend.core.phases.phase_01_click_removal import ClickRemovalPhase


@pytest.fixture
def phase():
    return ClickRemovalPhase()


@pytest.fixture
def mono_audio():
    return (np.random.RandomState(42).randn(48000) * 0.1).astype(np.float32)


@pytest.fixture
def stereo_audio():
    return (np.random.RandomState(42).randn(48000, 2) * 0.1).astype(np.float32)


class TestClickRemovalPhase:
    def test_process_returns_ndarray(self, phase, mono_audio):
        result = phase.process(mono_audio, sample_rate=48000, material_type="vinyl")
        assert isinstance(result.audio, np.ndarray)

    def test_no_nan_inf(self, phase, mono_audio):
        result = phase.process(mono_audio, sample_rate=48000, material_type="vinyl")
        assert np.isfinite(result.audio).all()

    def test_not_silent(self, phase, mono_audio):
        result = phase.process(mono_audio, sample_rate=48000, material_type="vinyl")
        rms = float(np.sqrt(np.mean(result.audio**2)))
        assert rms > 1e-10, f"Output is silent (RMS={rms:.2e})"

    def test_stereo_preserved(self, phase, stereo_audio):
        result = phase.process(stereo_audio, sample_rate=48000, material_type="vinyl")
        assert result.audio.ndim == 2

    def test_clean_audio_unchanged(self, phase):
        """Sauberes Audio (keine Clicks) sollte nahezu unverändert bleiben."""
        t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
        clean = np.sin(2 * np.pi * 440 * t) * 0.5
        result = phase.process(clean, sample_rate=48000, material_type="vinyl")
        corr = float(np.corrcoef(clean, result.audio[: len(clean)])[0, 1])
        assert corr > 0.95, f"Clean audio degraded (corr={corr:.4f})"
