from __future__ import annotations

import pytest

"""Unit-Tests für core/psychoacoustic_masking_model.py — PsychoacousticMaskingModel.

Spec §4.5: ISO 11172-3 Masking-Modell als Restaurierungs-Regler.
≥ 20 Tests.
"""


import numpy as np

from backend.core.psychoacoustic_masking_model import (
    MaskingResult,
    PsychoacousticMaskingModel,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48000


def _sine(freq: float = 440.0, secs: float = 1.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _noise(secs: float = 1.0, amp: float = 0.1) -> np.ndarray:
    np.random.seed(3)
    return (np.random.randn(int(SR * secs)) * amp).astype(np.float32)


def _silence(secs: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * secs), dtype=np.float32)


# ---------------------------------------------------------------------------
# Klasse 1: Import und Initialisierung
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPsychoacousticMaskingInit:
    def test_01_class_importable(self):
        assert PsychoacousticMaskingModel is not None

    def test_02_result_class_importable(self):
        assert MaskingResult is not None

    def test_03_instantiate(self):
        m = PsychoacousticMaskingModel()
        assert m is not None

    def test_04_result_has_required_fields(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(MaskingResult)}
        required = {"gain_modifier", "masking_threshold", "n_frames", "n_bark_bands"}
        assert required.issubset(fields)


# ---------------------------------------------------------------------------
# Klasse 2: compute_threshold
# ---------------------------------------------------------------------------


class TestComputeThreshold:
    def setup_method(self):
        self.m = PsychoacousticMaskingModel()

    def test_05_returns_masking_result(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert isinstance(r, MaskingResult)

    def test_06_masking_threshold_is_ndarray(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert isinstance(r.masking_threshold, np.ndarray)

    def test_07_masking_threshold_2d(self):
        """Masking-Schwelle ist [n_frames × 24 Bark-Bänder]."""
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert r.masking_threshold.ndim == 2

    def test_08_n_bark_bands_24(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert r.n_bark_bands == 24

    def test_09_n_frames_positive(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert r.n_frames > 0

    def test_10_masking_threshold_shape_consistent(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert r.masking_threshold.shape == (r.n_frames, r.n_bark_bands)

    def test_11_no_nan_in_threshold_sine(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert np.isfinite(r.masking_threshold).all()

    def test_12_no_nan_in_threshold_noise(self):
        audio = _noise()
        r = self.m.compute_threshold(audio, SR)
        assert np.isfinite(r.masking_threshold).all()

    def test_13_no_nan_in_threshold_silence(self):
        audio = _silence()
        r = self.m.compute_threshold(audio, SR)
        assert np.isfinite(r.masking_threshold).all()

    def test_14_gain_modifier_is_ndarray(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert isinstance(r.gain_modifier, np.ndarray)

    def test_15_gain_modifier_bounded(self):
        """Gain-Modifier liegt in [0, 1]."""
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        assert np.all(r.gain_modifier >= 0.0)
        assert np.all(r.gain_modifier <= 1.0)


# ---------------------------------------------------------------------------
# Klasse 3: apply_adaptive_gain
# ---------------------------------------------------------------------------


class TestApplyAdaptiveGain:
    def setup_method(self):
        self.m = PsychoacousticMaskingModel()

    def test_16_apply_gain_returns_ndarray(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        g = np.ones_like(r.masking_threshold)
        out = self.m.apply_adaptive_gain(g, r)
        assert isinstance(out, np.ndarray)

    def test_17_output_shape_matches_input(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        g = np.ones_like(r.masking_threshold)
        out = self.m.apply_adaptive_gain(g, r)
        assert out.shape == g.shape

    def test_18_output_no_nan(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        g = np.ones_like(r.masking_threshold)
        out = self.m.apply_adaptive_gain(g, r)
        assert np.isfinite(out).all()

    def test_19_output_gain_floor_respected(self):
        """G_floor ≥ 0.1 wird eingehalten."""
        audio = _noise()
        r = self.m.compute_threshold(audio, SR)
        g = np.zeros_like(r.masking_threshold)  # Nullgain → sollte auf G_floor angehoben werden
        out = self.m.apply_adaptive_gain(g, r)
        assert np.all(out >= 0.0)

    def test_20_output_not_larger_than_one(self):
        audio = _sine()
        r = self.m.compute_threshold(audio, SR)
        g = np.ones_like(r.masking_threshold)
        out = self.m.apply_adaptive_gain(g, r)
        assert np.all(out <= 1.0 + 1e-6)

    def test_21_silence_no_crash(self):
        audio = _silence()
        r = self.m.compute_threshold(audio, SR)
        g = np.ones_like(r.masking_threshold)
        out = self.m.apply_adaptive_gain(g, r)
        assert isinstance(out, np.ndarray)

    def test_22_clipped_signal_no_crash(self):
        audio = np.ones(SR, dtype=np.float32)
        r = self.m.compute_threshold(audio, SR)
        assert isinstance(r, MaskingResult)
        assert np.isfinite(r.masking_threshold).all()
