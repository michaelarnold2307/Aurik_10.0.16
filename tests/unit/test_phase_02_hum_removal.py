"""tests/unit/test_phase_02_hum_removal.py — §v10.700 I1: Unit-Tests für Phase 02 (Hum Removal)."""

import numpy as np
import pytest

from backend.core.phases.phase_02_hum_removal import HumRemovalPhase


@pytest.fixture
def phase():
    return HumRemovalPhase()


@pytest.fixture
def clean_audio():
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


@pytest.fixture
def hum_audio(clean_audio):
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    hum = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 150 * t)
    return (clean_audio + hum).astype(np.float32)


class TestHumRemovalPhase:
    def test_process_returns_ndarray(self, phase, hum_audio):
        result = phase.process(hum_audio, sample_rate=48000, material_type="vinyl")
        assert isinstance(result.audio, np.ndarray)

    def test_no_nan_inf(self, phase, hum_audio):
        result = phase.process(hum_audio, sample_rate=48000, material_type="vinyl")
        assert np.isfinite(result.audio).all()

    def test_not_silent(self, phase, clean_audio):
        result = phase.process(clean_audio, sample_rate=48000, material_type="vinyl")
        rms = float(np.sqrt(np.mean(result.audio**2)))
        assert rms > 1e-10, f"Output is silent (RMS={rms:.2e})"

    def test_reduces_hum(self, phase, clean_audio, hum_audio):
        """Hum Removal sollte 50-Hz-Anteil reduzieren."""
        result = phase.process(hum_audio, sample_rate=48000, material_type="vinyl")
        # Output sollte höhere Korrelation mit Clean haben als Input
        corr_in = float(np.corrcoef(clean_audio, hum_audio)[0, 1])
        corr_out = float(np.corrcoef(clean_audio, result.audio[: len(clean_audio)])[0, 1])
        assert corr_out > 0.80, f"Hum not reduced (corr_out={corr_out:.4f}, corr_in={corr_in:.4f})"
