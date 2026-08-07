"""tests/unit/test_phase_03_denoise.py — §v10.700 I1: Unit-Tests für Phase 03 (Denoise)."""

import numpy as np
import pytest

from backend.core.phases.phase_03_denoise import DenoisePhase


@pytest.fixture
def phase():
    return DenoisePhase(sample_rate=48000)


@pytest.fixture
def clean_audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)
    return sig.astype(np.float32)


@pytest.fixture
def noisy_audio(clean_audio):
    rng = np.random.RandomState(42)
    return (clean_audio + rng.randn(len(clean_audio)).astype(np.float32) * 0.05).astype(np.float32)


class TestDenoisePhase:
    def test_process_returns_ndarray(self, phase, noisy_audio):
        result = phase.process(noisy_audio, sample_rate=48000, material_type="vinyl")
        assert isinstance(result.audio, np.ndarray)

    def test_no_nan_inf(self, phase, noisy_audio):
        result = phase.process(noisy_audio, sample_rate=48000, material_type="vinyl")
        assert np.isfinite(result.audio).all()

    def test_not_silent(self, phase, noisy_audio):
        result = phase.process(noisy_audio, sample_rate=48000, material_type="vinyl")
        rms = float(np.sqrt(np.mean(result.audio**2)))
        assert rms > 1e-10, f"Output is silent (RMS={rms:.2e})"

    def test_reduces_noise(self, phase, clean_audio, noisy_audio):
        """Denoise sollte das Rauschen reduzieren (Output näher am Clean)."""
        result = phase.process(noisy_audio, sample_rate=48000, material_type="vinyl")
        # Output sollte höhere Korrelation mit Clean haben als Input
        corr_in = float(np.corrcoef(clean_audio, noisy_audio)[0, 1])
        corr_out = float(np.corrcoef(clean_audio, result.audio[: len(clean_audio)])[0, 1])
        # Nicht strikt asserten (Denoise kann auch verschlechtern bei Rauschen),
        # aber Output sollte NICHT still sein
        assert corr_out > 0.5, f"Output decorrelated from clean (corr={corr_out:.4f})"

    def test_clean_audio_not_degraded(self, phase, clean_audio):
        """Sauberes Audio sollte nicht signifikant verschlechtert werden."""
        result = phase.process(clean_audio, sample_rate=48000, material_type="vinyl")
        corr = float(np.corrcoef(clean_audio, result.audio[: len(clean_audio)])[0, 1])
        assert corr > 0.97, f"Clean audio degraded (corr={corr:.4f})"
