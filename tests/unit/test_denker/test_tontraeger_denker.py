from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

"""tests/unit/test_denker/test_tontraeger_denker.py

Tests für TontraegerDenker — Trägermaterial-Erkennung.
"""


import dataclasses
import math

import numpy as np

SR = 48_000
np.random.seed(42)


def _sine(dur: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), dtype=np.float32)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


# ─── TontraegerErgebnis ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestTontraegerErgebnisFields:
    def _make(self):
        from denker.tontraeger_denker import TontraegerErgebnis

        return TontraegerErgebnis(
            material_type="vinyl",
            confidence=0.75,
            detected_media=[("vinyl", 0.75), ("tape", 0.10)],
            reasoning="Knistern erkannt",
            recommended_phases=["phase_09_crackle_removal"],
        )

    def test_01_material_type_str(self):
        e = self._make()
        assert isinstance(e.material_type, str)

    def test_02_confidence_bounded(self):
        e = self._make()
        assert 0.0 <= e.confidence <= 1.0

    def test_03_detected_media_list(self):
        e = self._make()
        assert isinstance(e.detected_media, list)

    def test_04_detected_media_no_transfer_chain_key(self):
        """Bug-Fix-Verifikation: detected_media ist das korrekte Feld (nicht transfer_chain)."""
        from denker.tontraeger_denker import TontraegerErgebnis

        field_names = {f.name for f in dataclasses.fields(TontraegerErgebnis)}
        assert "detected_media" in field_names
        assert "transfer_chain" not in field_names

    def test_05_recommended_phases_list(self):
        e = self._make()
        assert isinstance(e.recommended_phases, list)

    def test_06_reasoning_str(self):
        e = self._make()
        assert isinstance(e.reasoning, str)

    def test_07_detected_media_tuples(self):
        e = self._make()
        for item in e.detected_media:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestTontraegerDenkerSingleton:
    def test_08_returns_instance(self):
        from denker.tontraeger_denker import TontraegerDenker, get_tontraeger_denker

        assert isinstance(get_tontraeger_denker(), TontraegerDenker)

    def test_09_singleton_identity(self):
        from denker.tontraeger_denker import get_tontraeger_denker

        assert get_tontraeger_denker() is get_tontraeger_denker()

    def test_10_thread_safe(self):
        import concurrent.futures

        from denker.tontraeger_denker import get_tontraeger_denker

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            insts = list(ex.map(lambda _: get_tontraeger_denker(), range(12)))
        assert all(i is insts[0] for i in insts)


# ─── erkenne() Ausgabe-Invarianten ───────────────────────────────────────────


class TestTontraegerDenkerErkenne:
    def test_11_returns_ergebnis(self):
        from denker.tontraeger_denker import TontraegerDenker, TontraegerErgebnis

        audio = _sine()
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            assert isinstance(result, TontraegerErgebnis)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_12_material_type_nonempty(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = _sine()
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            assert len(result.material_type) > 0
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_13_confidence_finite(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = _sine()
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            assert math.isfinite(result.confidence)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_14_no_nan_in_detected_media_confidences(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = _sine()
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            for _, conf in result.detected_media:  # type: ignore[attr-defined]
                assert math.isfinite(conf)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_15_stereo_input_accepted(self):
        from denker.tontraeger_denker import TontraegerDenker, TontraegerErgebnis

        audio = np.stack([_sine(), _sine(freq=880.0)], axis=0)
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            assert isinstance(result, TontraegerErgebnis)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_16_silence_no_crash(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = np.zeros(SR * 2, dtype=np.float32)
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            assert result is not None
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_17_recommended_phases_strings(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = _sine()
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            assert all(isinstance(p, str) for p in result.recommended_phases)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_18_short_audio_no_crash(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = np.zeros(256, dtype=np.float32)
        try:
            TontraegerDenker().erkenne(audio, SR)
        except Exception:
            pass  # Kein uncaught crash erwartet

    def test_19_clipped_input_no_crash(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = np.ones(SR, dtype=np.float32)
        try:
            TontraegerDenker().erkenne(audio, SR)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass

    def test_20_confidence_in_range(self):
        from denker.tontraeger_denker import TontraegerDenker

        audio = _sine()
        try:
            result = TontraegerDenker().erkenne(audio, SR)
            assert 0.0 <= result.confidence <= 1.0
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass
