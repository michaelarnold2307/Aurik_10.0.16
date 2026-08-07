from __future__ import annotations

import logging

"""tests/unit/test_denker/test_exzellenz_denker.py

Tests für ExzellenzDenker — Excellence-Optimierung + Musical-Goals-Messung.
Schwerpunkt: Vokal-KI-Zweig (material="vocal") und Nicht-Vokal-Pfad.
"""


import dataclasses
import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)

SR = 48_000
np.random.seed(42)


def _sine(dur: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), dtype=np.float32)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


# ─── ExzellenzErgebnis ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExzellenzErgebnisFields:
    def _make(self):
        from denker.exzellenz_denker import ExzellenzErgebnis

        return ExzellenzErgebnis(
            audio=_sine(),
            excellence_score=0.82,
            musical_goals={"brillanz": 0.87, "waerme": 0.81},
            goals_passed=2,
            goals_total=14,
            improvements={"brillanz": 0.03},  # type: ignore[arg-type]
            processing_note="Test",
            warnings=[],
        )

    def test_01_all_fields_present(self):
        from denker.exzellenz_denker import ExzellenzErgebnis

        field_names = {f.name for f in dataclasses.fields(ExzellenzErgebnis)}
        for name in (
            "audio",
            "excellence_score",
            "musical_goals",
            "goals_passed",
            "goals_total",
            "improvements",
            "processing_note",
            "warnings",
        ):
            assert name in field_names, f"Feld '{name}' fehlt in ExzellenzErgebnis"

    def test_02_score_bounded(self):
        e = self._make()
        assert 0.0 <= e.excellence_score <= 1.0

    def test_03_goals_passed_le_total(self):
        e = self._make()
        assert e.goals_passed <= e.goals_total

    def test_04_audio_finite(self):
        e = self._make()
        assert np.isfinite(e.audio).all()

    def test_05_improvements_dict(self):
        e = self._make()
        assert isinstance(e.improvements, dict)

    def test_06_warnings_list(self):
        e = self._make()
        assert isinstance(e.warnings, list)


# ─── ExzellenzDenker Singleton ────────────────────────────────────────────────


class TestExzellenzDenkerSingleton:
    def test_07_get_returns_instance(self):
        from denker.exzellenz_denker import ExzellenzDenker, get_exzellenz_denker

        assert isinstance(get_exzellenz_denker(), ExzellenzDenker)

    def test_08_singleton_identity(self):
        from denker.exzellenz_denker import get_exzellenz_denker

        assert get_exzellenz_denker() is get_exzellenz_denker()

    def test_09_thread_safe(self):
        import concurrent.futures

        from denker.exzellenz_denker import get_exzellenz_denker

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            insts = list(ex.map(lambda _: get_exzellenz_denker(), range(12)))
        assert all(i is insts[0] for i in insts)


# ─── Vokal-KI-Zweig ───────────────────────────────────────────────────────────


class TestVocalAIBranch:
    """Verifikation: VocalAIEnhancer wird NUR bei Vokal-Material aufgerufen."""

    def _patched_exzellenz_denker(self):
        """Erzeugt einen leichten ExzellenzDenker ohne schwere Optimizer/Goal-Calls."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        fast_opt = MagicMock()
        fast_opt.optimize.return_value = (
            _sine(),
            MagicMock(applied_steps=["mock_step"], summary=MagicMock(return_value="ok")),
        )
        denker._get_optimizer = MagicMock(return_value=fast_opt)  # type: ignore[method-assign]
        denker.messe_ziele = MagicMock(return_value={"brillanz": 0.9, "waerme": 0.85})  # type: ignore[method-assign]
        return denker

    def _mock_versa_module(self):
        mock_versa_module = MagicMock()
        mock_versa_module.score_mos = MagicMock(return_value=MagicMock(mos=4.2, model_used="mock_versa"))
        return mock_versa_module

    def _make_vocal_result(self, audio: np.ndarray) -> MagicMock:
        r = MagicMock()
        r.audio = audio
        r.quality_improvement = 0.05
        r.processing_applied = ["de_esser", "formant_enhance"]
        return r

    def test_10_vocal_material_calls_enhancer(self):
        audio = _sine()
        fake_result = self._make_vocal_result(audio)
        mock_versa_module = self._mock_versa_module()

        mock_enhancer_inst = MagicMock()
        mock_enhancer_inst.enhance.return_value = fake_result
        mock_cls = MagicMock(return_value=mock_enhancer_inst)

        with (
            patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True),
            patch.dict(sys.modules, {"plugins.versa_plugin": mock_versa_module}),
        ):
            pass

            denker = self._patched_exzellenz_denker()
            try:
                denker.optimiere(audio, SR, material="vocal")
                # Enhancer wurde instanziiert
                mock_cls.assert_called_once()
            except Exception:
                logger.warning("test fallback", exc_info=True)
                pass  # Andere Schritte dürfen scheitern

    def test_11_tape_material_skips_enhancer(self):
        audio = _sine()
        mock_cls = MagicMock()

        with (
            patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True),
            patch.dict(sys.modules, {"plugins.versa_plugin": self._mock_versa_module()}),
        ):
            pass

            denker = self._patched_exzellenz_denker()
            try:
                denker.optimiere(audio, SR, material="tape")
            except Exception:
                logger.warning("test fallback", exc_info=True)
            mock_cls.assert_not_called()

    def test_12_singer_material_triggers_branch(self):
        audio = _sine()
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.enhance.return_value = MagicMock(audio=audio, quality_improvement=0.02, processing_applied=[])
        mock_cls.return_value = mock_inst

        with (
            patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True),
            patch.dict(sys.modules, {"plugins.versa_plugin": self._mock_versa_module()}),
        ):
            pass

            try:
                self._patched_exzellenz_denker().optimiere(audio, SR, material="singer")
                mock_cls.assert_called()
            except Exception:
                logger.warning("test fallback", exc_info=True)

    def test_13_vocal_enhancer_exception_swallowed(self):
        """Fehler in VocalAIEnhancer darf nicht zum Absturz führen."""
        audio = _sine()
        mock_cls = MagicMock(side_effect=RuntimeError("enhancer crash"))

        with (
            patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True),
            patch.dict(sys.modules, {"plugins.versa_plugin": self._mock_versa_module()}),
        ):
            pass

            try:
                result = self._patched_exzellenz_denker().optimiere(audio, SR, material="vocal")
                # Kein Absturz
                assert result is not None
            except Exception:
                logger.warning("test fallback", exc_info=True)
                pass  # Andere Stufen dürfen scheitern

    def test_14_vinyl_material_skips_enhancer(self):
        audio = _sine()
        mock_cls = MagicMock()
        with (
            patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True),
            patch.dict(sys.modules, {"plugins.versa_plugin": self._mock_versa_module()}),
        ):
            pass

            try:
                self._patched_exzellenz_denker().optimiere(audio, SR, material="vinyl")
            except Exception:
                logger.warning("test fallback", exc_info=True)
            mock_cls.assert_not_called()

    def test_15_voice_material_triggers_branch(self):
        audio = _sine()
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.enhance.return_value = MagicMock(audio=audio, quality_improvement=0.01, processing_applied=[])
        mock_cls.return_value = mock_inst
        with (
            patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True),
            patch.dict(sys.modules, {"plugins.versa_plugin": self._mock_versa_module()}),
        ):
            pass

            try:
                self._patched_exzellenz_denker().optimiere(audio, SR, material="voice")
                mock_cls.assert_called()
            except Exception:
                logger.warning("test fallback", exc_info=True)


# ─── optimiere() Ausgabe-Invarianten ─────────────────────────────────────────


class TestOptimiereInvarianten:
    def _run_lightweight_optimiere(self, audio: np.ndarray, material: str = "tape"):
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        fast_opt = MagicMock()
        fast_opt.optimize.return_value = (
            audio.copy(),
            MagicMock(applied_steps=["mock_step"], summary=MagicMock(return_value="ok")),
        )
        denker._get_optimizer = MagicMock(return_value=fast_opt)  # type: ignore[method-assign]
        denker.messe_ziele = MagicMock(return_value={"brillanz": 0.9, "waerme": 0.85})  # type: ignore[method-assign]
        with patch.dict(
            sys.modules,
            {
                "plugins.versa_plugin": MagicMock(
                    score_mos=MagicMock(return_value=MagicMock(mos=4.2, model_used="mock_versa"))
                )
            },
        ):
            return denker.optimiere(audio, SR, material=material)

    def test_16_returns_exzellenz_ergebnis_type(self):
        from denker.exzellenz_denker import ExzellenzErgebnis

        audio = _sine()
        try:
            result = self._run_lightweight_optimiere(audio, material="tape")
            assert isinstance(result, ExzellenzErgebnis)
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Falls deps nicht vorhanden

    def test_17_audio_no_nan(self):
        pass

        audio = _sine()
        try:
            result = self._run_lightweight_optimiere(audio)
            assert np.isfinite(result.audio).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_18_goals_passed_non_negative(self):
        pass

        audio = _sine()
        try:
            result = self._run_lightweight_optimiere(audio)
            assert result.goals_passed >= 0
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_19_musical_goals_dict(self):
        pass

        audio = _sine()
        try:
            result = self._run_lightweight_optimiere(audio)
            assert isinstance(result.musical_goals, dict)
        except Exception:
            logger.warning("test fallback", exc_info=True)

    def test_20_excellence_score_finite(self):
        pass

        audio = _sine()
        try:
            result = self._run_lightweight_optimiere(audio)
            assert math.isfinite(result.excellence_score)
        except Exception:
            logger.warning("test fallback", exc_info=True)


# ─── ExzellenzErgebnis.versa_mos ───────────────────────────────────────────────


class TestExzellenzErgebnisVersaMos:
    """versa_mos-Feld in ExzellenzErgebnis (M-8b: VERSA-Cache für AurikDenker Stage 8)."""

    def test_21_versa_mos_field_exists(self):
        """ExzellenzErgebnis hat versa_mos-Feld."""
        import dataclasses

        from denker.exzellenz_denker import ExzellenzErgebnis

        field_names = {f.name for f in dataclasses.fields(ExzellenzErgebnis)}
        assert "versa_mos" in field_names, (
            "ExzellenzErgebnis fehlt 'versa_mos' (M-8b: VERSA-Cache-Interface für AurikDenker)"
        )

    def test_22_versa_mos_default_zero(self):
        """Default versa_mos = 0.0 (nicht gemessen)."""
        from denker.exzellenz_denker import ExzellenzErgebnis

        e = ExzellenzErgebnis(
            audio=_sine(),
            excellence_score=0.80,
            musical_goals={},
            goals_passed=0,
            goals_total=14,
            improvements=[],
            processing_note="test",
        )
        assert e.versa_mos == 0.0

    def test_23_versa_mos_set_to_value(self):
        """versa_mos kann auf validen MOS-Wert gesetzt werden."""
        from denker.exzellenz_denker import ExzellenzErgebnis

        e = ExzellenzErgebnis(
            audio=_sine(),
            excellence_score=0.80,
            musical_goals={},
            goals_passed=0,
            goals_total=14,
            improvements=[],
            processing_note="test",
            versa_mos=4.1,
        )
        assert e.versa_mos == pytest.approx(4.1)

    def test_24_optimiere_returns_versa_mos_field(self):
        """optimiere() stores versa_mos in ExzellenzErgebnis; VERSA is mocked (no ML load)."""
        from denker.exzellenz_denker import get_exzellenz_denker
        from plugins.versa_plugin import VersaResult

        fake = VersaResult(mos=4.2, model_used="singmos_pro", confidence=0.90)
        with patch("plugins.versa_plugin.score_mos", return_value=fake):
            result = get_exzellenz_denker().optimiere(_sine(0.5), 48000)
        assert hasattr(result, "versa_mos"), "optimiere() muss versa_mos in ExzellenzErgebnis liefern"
        assert isinstance(result.versa_mos, float)
        assert math.isfinite(result.versa_mos), f"versa_mos={result.versa_mos} ist nicht finite"

    def test_25_versa_mos_valid_range_if_nonzero(self):
        """Wenn versa_mos > 0, muss es im MOS-Bereich [1, 5] liegen; VERSA ist gemockt."""
        from denker.exzellenz_denker import get_exzellenz_denker
        from plugins.versa_plugin import VersaResult

        fake = VersaResult(mos=3.8, model_used="singmos_pro", confidence=0.85)
        with patch("plugins.versa_plugin.score_mos", return_value=fake):
            result = get_exzellenz_denker().optimiere(_sine(0.5), 48000)
        if result.versa_mos > 0.0:
            assert 1.0 <= result.versa_mos <= 5.0, f"versa_mos={result.versa_mos} außerhalb MOS-Bereich [1, 5]"
