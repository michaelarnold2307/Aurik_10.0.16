"""test_tonality_gate_contract — §23-TONALITY (Vorschlag 01, docs/proposals/).

Referenzwerte aus der Mess-Evidenz der Session (Seeds dokumentiert):
  flatness 0.00 = Sinus 440+880 Hz | 0.69 = Sinus + Rauschen σ=0.05 | 0.99 = Rauschen
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 48000


def _sine(freq: float = 440.0, dur: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _clean_tone() -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, SR, endpoint=False))
            + 0.15 * np.sin(2 * np.pi * 880 * np.linspace(0, 1, SR, endpoint=False))).astype(np.float32)


def _noisy_tone() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (_clean_tone() + rng.standard_normal(SR).astype(np.float32) * 0.05).astype(np.float32)


def _noise() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(SR) * 0.05).astype(np.float32)


class TestTonalityGateReferenceValues:
    def test_flatness_tonal_near_zero(self):
        from backend.core.dsp.tonality_gate import spectral_flatness

        assert spectral_flatness(_clean_tone(), SR) < 0.02

    def test_flatness_tonal_plus_noise(self):
        from backend.core.dsp.tonality_gate import spectral_flatness

        assert 0.45 < spectral_flatness(_noisy_tone(), SR) < 0.90

    def test_flatness_noise_near_one(self):
        from backend.core.dsp.tonality_gate import spectral_flatness

        assert spectral_flatness(_noise(), SR) > 0.90

    def test_is_tonal_clean_decision(self):
        from backend.core.dsp.tonality_gate import is_tonal_clean

        assert is_tonal_clean(_clean_tone(), SR) is True
        assert is_tonal_clean(_noisy_tone(), SR) is False
        assert is_tonal_clean(_noise(), SR) is False


class TestTonalityGateIntegration:
    def test_spectral_denoiser_passthrough_on_tonal(self):
        from dsp.spectral_denoiser import SpectralDenoiser

        denoiser = SpectralDenoiser(reduction_db=18.0)
        tone = _clean_tone()
        out = denoiser.process(tone, SR)
        assert np.allclose(out, tone, atol=1e-7)

    def test_spectral_denoiser_processes_noisy(self):
        from dsp.spectral_denoiser import SpectralDenoiser

        denoiser = SpectralDenoiser(reduction_db=18.0)
        noisy = _noisy_tone()
        out = denoiser.process(noisy, SR)
        # Das Rauschen muss reduziert werden (Varianz sinkt)
        assert np.var(out) < np.var(noisy)

    def test_hybrid_strategy_omlsa_only_on_tonal(self):
        from backend.core.hybrid.hybrid_ml_denoiser import DenoiseConfig, DenoiseStrategy, HybridMLDenoiser

        denoiser = HybridMLDenoiser(config=DenoiseConfig(strategy=DenoiseStrategy.ADAPTIVE))
        strategy = denoiser._determine_strategy(_clean_tone(), SR)
        assert strategy == DenoiseStrategy.OMLSA_ONLY
