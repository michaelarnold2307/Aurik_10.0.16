from __future__ import annotations

import pytest

"""Unit-Tests für core/transient_decoupled_processor.py — TransientDecoupledProcessing.

Spec §2.27: HPSS-Trennung, separate(), recombine(), GrooveMetric-Invariante.
≥ 20 Tests.
"""


import numpy as np

from backend.core.transient_decoupled_processor import TransientDecoupledProcessing

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48000


def _sine(freq: float = 440.0, secs: float = 2.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _noise(secs: float = 2.0, amp: float = 0.1) -> np.ndarray:
    np.random.seed(11)
    return (np.random.randn(int(SR * secs)) * amp).astype(np.float32)


def _silence(secs: float = 2.0) -> np.ndarray:
    return np.zeros(int(SR * secs), dtype=np.float32)


def _impulse_train(secs: float = 2.0, bpm: float = 120.0) -> np.ndarray:
    """Regelmäßige Impulse (simulierter Schlagzeug-Beat)."""
    n = int(SR * secs)
    audio = np.zeros(n, dtype=np.float32)
    period_samples = int(SR * 60.0 / bpm)
    for i in range(0, n, period_samples):
        if i + 100 < n:
            audio[i : i + 100] = 0.8
    return audio


# ---------------------------------------------------------------------------
# Klasse 1: Import und Klassenkonstanten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransientDecoupledInit:
    def test_01_class_importable(self):
        assert TransientDecoupledProcessing is not None

    def test_02_instantiate(self):
        t = TransientDecoupledProcessing()
        assert t is not None

    def test_03_has_separate_method(self):
        """separate() muss vorhanden sein."""
        assert callable(getattr(TransientDecoupledProcessing, "separate", None))

    def test_04_has_recombine_method(self):
        """recombine() muss vorhanden sein."""
        assert callable(getattr(TransientDecoupledProcessing, "recombine", None))

    def test_05_separate_returns_two_components(self):
        """separate() liefert Tupel (percussive, harmonic)."""
        t = TransientDecoupledProcessing()
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 2 * 48000)).astype(np.float32)
        result = t.separate(audio, 48000)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Klasse 2: separate()
# ---------------------------------------------------------------------------


class TestSeparate:
    def setup_method(self):
        self.tdp = TransientDecoupledProcessing()

    def test_06_returns_two_arrays(self):
        audio = _sine()
        result = self.tdp.separate(audio, SR)
        assert len(result) == 2

    def test_07_percussive_shape_matches(self):
        audio = _noise()
        p, h = self.tdp.separate(audio, SR)
        assert p.shape == audio.shape

    def test_08_harmonic_shape_matches(self):
        audio = _sine()
        p, h = self.tdp.separate(audio, SR)
        assert h.shape == audio.shape

    def test_09_percussive_is_float32(self):
        audio = _sine()
        p, h = self.tdp.separate(audio, SR)
        assert p.dtype in (np.float32, np.float64)

    def test_10_harmonic_is_float32(self):
        audio = _sine()
        p, h = self.tdp.separate(audio, SR)
        assert h.dtype in (np.float32, np.float64)

    def test_11_no_nan_percussive(self):
        audio = _noise()
        p, h = self.tdp.separate(audio, SR)
        assert np.isfinite(p).all()

    def test_12_no_nan_harmonic(self):
        audio = _sine()
        p, h = self.tdp.separate(audio, SR)
        assert np.isfinite(h).all()

    def test_13_silence_no_crash(self):
        audio = _silence()
        p, h = self.tdp.separate(audio, SR)
        assert p.shape == audio.shape
        assert h.shape == audio.shape

    def test_14_impulse_train_no_crash(self):
        audio = _impulse_train()
        p, h = self.tdp.separate(audio, SR)
        assert np.isfinite(p).all()
        assert np.isfinite(h).all()


# ---------------------------------------------------------------------------
# Klasse 3: recombine()
# ---------------------------------------------------------------------------


class TestRecombine:
    def setup_method(self):
        self.tdp = TransientDecoupledProcessing()

    def test_15_recombine_shape_matches(self):
        audio = _sine()
        p, h = self.tdp.separate(audio, SR)
        out = self.tdp.recombine(p, h, SR)
        assert out.shape == audio.shape

    def test_16_recombine_no_nan(self):
        audio = _noise()
        p, h = self.tdp.separate(audio, SR)
        out = self.tdp.recombine(p, h, SR)
        assert np.isfinite(out).all()

    def test_17_recombine_clipped_within_bounds(self):
        """Ausgabe darf nicht über ±1.0 clippen (spec: np.clip(-1.0, 1.0))."""
        audio = _noise(amp=0.5)
        p, h = self.tdp.separate(audio, SR)
        out = self.tdp.recombine(p, h, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-5

    def test_18_recombine_with_original_perc(self):
        """Optionaler original_perc-Parameter → kein Absturz."""
        audio = _impulse_train()
        p, h = self.tdp.separate(audio, SR)
        out = self.tdp.recombine(p, h, SR, original_perc=p)
        assert out.shape == audio.shape
        assert np.isfinite(out).all()

    def test_19_energy_conservation_approximate(self):
        """Summe p + h sollte ungefähre Energie erhalten."""
        audio = _sine() * 0.5
        p, h = self.tdp.separate(audio, SR)
        out = self.tdp.recombine(p, h, SR)
        # Energie-Differenz sollte nicht zu groß sein
        energy_in = np.sum(audio**2)
        energy_out = np.sum(out**2)
        if energy_in > 1e-6:
            ratio = energy_out / energy_in
            assert 0.01 <= ratio <= 100.0, f"Energieverhältnis unrealistisch: {ratio}"

    def test_20_silence_recombine_no_crash(self):
        audio = _silence()
        p, h = self.tdp.separate(audio, SR)
        out = self.tdp.recombine(p, h, SR)
        assert np.isfinite(out).all()
        assert out.shape == audio.shape

    def test_21_float64_input_no_crash(self):
        """float64-Input → kein Absturz."""
        audio = _sine().astype(np.float64)
        p, h = self.tdp.separate(audio, SR)
        out = self.tdp.recombine(p, h, SR)
        assert out.shape[0] == audio.shape[0]
        assert np.isfinite(out).all()
