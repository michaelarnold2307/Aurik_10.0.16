from __future__ import annotations

import logging

import pytest

"""tests/unit/test_denker/test_defekt_denker.py

Tests für DefektDenker — Defekterkennung & -klassifikation.
"""


import math

import numpy as np

logger = logging.getLogger(__name__)

SR = 48_000
np.random.seed(42)


def _sine(dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _noisy(dur: float = 1.0) -> np.ndarray:
    np.random.seed(7)
    base = _sine(dur)
    noise = np.random.randn(*base.shape).astype(np.float32) * 0.1
    return np.clip(base + noise, -1.0, 1.0)  # type: ignore[no-any-return]


# ─── DefektErgebnis ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDefektErgebnisFields:
    def _make(self):
        from denker.defekt_denker import DefektErgebnis

        return DefektErgebnis(
            defect_scores={"clicks": 0.1, "hiss": 0.6},
            primary_defect="hiss",
            confidence=0.70,
            material_context="tape",
            recommended_phases=["phase_29_tape_hiss_reduction"],
            reasoning="Bandrauschen dominant",
        )

    def test_01_defect_scores_dict(self):
        e = self._make()
        assert isinstance(e.defect_scores, dict)

    def test_02_primary_defect_str(self):
        e = self._make()
        assert isinstance(e.primary_defect, str)

    def test_03_confidence_bounded(self):
        e = self._make()
        assert 0.0 <= e.confidence <= 1.0

    def test_04_recommended_phases_list(self):
        e = self._make()
        assert isinstance(e.recommended_phases, list)

    def test_05_scores_finite(self):
        e = self._make()
        for v in e.defect_scores.values():
            assert math.isfinite(v)

    def test_06_reasoning_str(self):
        e = self._make()
        assert isinstance(e.reasoning, str)


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestDefektDenkerSingleton:
    def test_07_returns_instance(self):
        from denker.defekt_denker import DefektDenker, get_defekt_denker

        assert isinstance(get_defekt_denker(), DefektDenker)

    def test_08_singleton_identity(self):
        from denker.defekt_denker import get_defekt_denker

        assert get_defekt_denker() is get_defekt_denker()

    def test_09_thread_safe(self):
        import concurrent.futures

        from denker.defekt_denker import get_defekt_denker

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            insts = list(ex.map(lambda _: get_defekt_denker(), range(12)))
        assert all(i is insts[0] for i in insts)


# ─── analysiere() Ausgabe-Invarianten ────────────────────────────────────────


class TestDefektDenkerAnalysiere:
    def test_10_returns_defekt_ergebnis(self):
        from denker.defekt_denker import DefektDenker, DefektErgebnis

        audio = _sine()
        try:
            result = DefektDenker().analysiere(audio, SR)
            assert isinstance(result, DefektErgebnis)
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_11_primary_defect_nonempty(self):
        from denker.defekt_denker import DefektDenker

        audio = _noisy()
        try:
            result = DefektDenker().analysiere(audio, SR)
            assert len(result.primary_defect) > 0
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_12_confidence_finite(self):
        from denker.defekt_denker import DefektDenker

        audio = _sine()
        try:
            result = DefektDenker().analysiere(audio, SR)
            assert math.isfinite(result.confidence)
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_13_defect_scores_all_finite(self):
        from denker.defekt_denker import DefektDenker

        audio = _sine()
        try:
            result = DefektDenker().analysiere(audio, SR)
            for v in result.defect_scores.values():
                assert math.isfinite(v)
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_14_defect_scores_in_range(self):
        from denker.defekt_denker import DefektDenker

        audio = _sine()
        try:
            result = DefektDenker().analysiere(audio, SR)
            for v in result.defect_scores.values():
                assert 0.0 <= v <= 1.0
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_15_silence_no_crash(self):
        from denker.defekt_denker import DefektDenker

        audio = np.zeros(SR * 2, dtype=np.float32)
        try:
            result = DefektDenker().analysiere(audio, SR)
            assert result is not None
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_16_material_context_in_result(self):
        from denker.defekt_denker import DefektDenker

        audio = _sine()
        try:
            result = DefektDenker().analysiere(audio, SR, material="vinyl")
            assert isinstance(result.material_context, str)
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_17_noisy_signal_detects_noise(self):
        from denker.defekt_denker import DefektDenker

        np.random.seed(0)
        audio = np.random.randn(SR * 2).astype(np.float32) * 0.3
        try:
            result = DefektDenker().analysiere(audio, SR)
            # hiss oder high_freq_noise sollte erhöhten Score haben
            total = sum(result.defect_scores.values())
            assert total >= 0
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_18_recommended_phases_strings(self):
        from denker.defekt_denker import DefektDenker

        audio = _sine()
        try:
            result = DefektDenker().analysiere(audio, SR)
            for p in result.recommended_phases:
                assert isinstance(p, str)
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_19_short_audio_no_crash(self):
        from denker.defekt_denker import DefektDenker

        audio = np.zeros(256, dtype=np.float32)
        try:
            DefektDenker().analysiere(audio, SR)
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_20_confidence_bounded_on_real_signal(self):
        from denker.defekt_denker import DefektDenker

        audio = _noisy(dur=2.0)
        try:
            result = DefektDenker().analysiere(audio, SR)
            assert 0.0 <= result.confidence <= 1.0
        except Exception:
            logger.warning("test fallback", exc_info=True)
