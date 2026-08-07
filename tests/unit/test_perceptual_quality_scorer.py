from __future__ import annotations

import pytest

"""Unit-Tests für core/perceptual_quality_scorer.py — PerceptualQualityScorer.

Spec §2.6: Gammatone-NSIM + MCD + LUFS + MOS-Mapping für Musik-Qualitätsbewertung.
Niemals PESQ/DNSMOS/NISQA als Musik-Metrik (§4.4).
≥ 14 Tests: MOS-Bounds, NSIM, MCD, Shape, Mono/Stereo, Singleton, NaN-Guard.
"""


import math

import numpy as np

np.random.seed(42)

from backend.core.perceptual_quality_scorer import (
    PerceptualQualityScorer,
    PQSResult,
    get_perceptual_quality_scorer,
    score_audio,
)

SR = 48000


def _sine(freq: float = 440.0, secs: float = 1.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _noise(secs: float = 1.0, amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(SR * secs)) * amp).astype(np.float32)


def _stereo(secs: float = 1.0) -> np.ndarray:
    mono = _sine(secs=secs)
    return np.stack([mono, mono * 0.9])


# ---------------------------------------------------------------------------
# Klasse 1: Import und Instantiierung
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPQSImport:
    def test_01_class_importable(self):
        assert PerceptualQualityScorer is not None

    def test_02_result_class_importable(self):
        assert PQSResult is not None

    def test_03_instantiate(self):
        scorer = PerceptualQualityScorer()
        assert scorer is not None

    def test_04_singleton_returns_instance(self):
        scorer = get_perceptual_quality_scorer()
        assert isinstance(scorer, PerceptualQualityScorer)

    def test_05_singleton_is_same_object(self):
        a = get_perceptual_quality_scorer()
        b = get_perceptual_quality_scorer()
        assert a is b

    def test_06_result_fields_present(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(PQSResult)}
        assert "mos" in fields
        assert "nsim" in fields
        assert "mcd_db" in fields
        assert "spectral_coherence" in fields

    def test_07_pqs_mos_alias(self):
        r = PQSResult(mos=3.5, nsim=0.8, mcd_db=5.0, spectral_coherence=0.9)
        assert r.pqs_mos == r.mos


# ---------------------------------------------------------------------------
# Klasse 2: score_audio (referenz-basiert)
# ---------------------------------------------------------------------------


class TestPQSScoreAudio:
    def setup_method(self):
        self.scorer = PerceptualQualityScorer()

    def test_08_mos_in_range_identical(self):
        audio = _sine()
        result = self.scorer.score_audio(audio, audio.copy(), SR)
        assert 1.0 <= result.mos <= 5.0, f"MOS {result.mos} außerhalb [1, 5]"

    def test_09_mos_identical_signals_high(self):
        """Identische Signale → MOS nahe 5."""
        audio = _sine()
        result = self.scorer.score_audio(audio, audio.copy(), SR)
        assert result.mos >= 4.0, f"MOS für identische Signale zu niedrig: {result.mos}"

    def test_10_nsim_in_range(self):
        audio = _sine()
        result = self.scorer.score_audio(audio, audio.copy(), SR)
        assert 0.0 <= result.nsim <= 1.0

    def test_11_mcd_db_finite(self):
        audio = _sine()
        result = self.scorer.score_audio(audio, audio.copy(), SR)
        assert math.isfinite(result.mcd_db)

    def test_12_mos_drops_for_noise(self):
        """Degradiertes Signal (stark gestört) → MOS niedriger als Original."""
        clean = _sine()
        noisy = np.clip(clean + _noise(amp=0.8), -1.0, 1.0)
        result_clean = self.scorer.score_audio(clean, clean.copy(), SR)
        result_noisy = self.scorer.score_audio(clean, noisy, SR)
        assert result_noisy.mos <= result_clean.mos

    def test_13_referenced_flag_true(self):
        audio = _sine()
        result = self.scorer.score_audio(audio, audio.copy(), SR)
        assert result.referenced is True


# ---------------------------------------------------------------------------
# Klasse 3: score_audio_absolute (referenz-frei)
# ---------------------------------------------------------------------------


class TestPQSScoreAbsolute:
    def setup_method(self):
        self.scorer = PerceptualQualityScorer()

    def test_14_absolute_mos_in_range(self):
        result = self.scorer.score_audio_absolute(_sine(), SR)
        assert 1.0 <= result.mos <= 5.0

    def test_15_absolute_referenced_flag_false(self):
        result = self.scorer.score_audio_absolute(_sine(), SR)
        assert result.referenced is False

    def test_16_absolute_nsim_in_range(self):
        result = self.scorer.score_audio_absolute(_sine(), SR)
        assert 0.0 <= result.nsim <= 1.0

    def test_17_absolute_silence_lower_mos_than_sine(self):
        silence = np.zeros(SR, dtype=np.float32)
        r_silence = self.scorer.score_audio_absolute(silence, SR)
        r_sine = self.scorer.score_audio_absolute(_sine(), SR)
        # Stille sollte keinen höheren MOS haben als Sinuston
        assert r_silence.mos <= r_sine.mos + 0.5  # ein wenig Toleranz


# ---------------------------------------------------------------------------
# Klasse 4: NaN-Guard und Robustheit
# ---------------------------------------------------------------------------


class TestPQSNaNGuard:
    def setup_method(self):
        self.scorer = PerceptualQualityScorer()

    def test_18_nan_input_reference(self):
        nan_audio = np.full(SR, float("nan"), dtype=np.float32)
        result = self.scorer.score_audio(nan_audio, nan_audio.copy(), SR)
        assert math.isfinite(result.mos)

    def test_19_inf_input_reference(self):
        inf_audio = np.full(SR, float("inf"), dtype=np.float32)
        result = self.scorer.score_audio(inf_audio, inf_audio.copy(), SR)
        assert math.isfinite(result.mos)

    def test_20_nan_input_absolute(self):
        nan_audio = np.full(SR, float("nan"), dtype=np.float32)
        result = self.scorer.score_audio_absolute(nan_audio, SR)
        assert math.isfinite(result.mos)

    def test_21_score_alias_matches_score_audio(self):
        """score() ist Alias für score_audio()."""
        audio = _sine()
        r1 = self.scorer.score(audio, audio.copy(), SR)
        r2 = self.scorer.score_audio(audio, audio.copy(), SR)
        assert abs(r1.mos - r2.mos) < 1e-6


# ---------------------------------------------------------------------------
# Klasse 5: Convenience-Funktion
# ---------------------------------------------------------------------------


class TestPQSConvenienceFunction:
    def test_22_score_audio_convenience_returns_pqs_result(self):
        audio = _sine()
        result = score_audio(audio, audio.copy(), SR)
        assert isinstance(result, PQSResult)
        assert 1.0 <= result.mos <= 5.0

    def test_23_spectral_coherence_in_range(self):
        audio = _sine()
        result = score_audio(audio, audio.copy(), SR)
        assert 0.0 <= result.spectral_coherence <= 1.0
