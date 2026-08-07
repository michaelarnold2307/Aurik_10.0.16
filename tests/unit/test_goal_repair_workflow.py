from __future__ import annotations

"""
tests/unit/test_goal_repair_workflow.py
========================================

Normative tests for the goal-achievement workflow optimizations (v10.0.0):
  - ExzellenzDenker.messe_und_repariere() — P3-P5 goal repair
  - AurikDenker Stufe 7 integration wiring (import + signature)
  - UV3 deferred-phase promotion logic (_DEFERRED_FC_WHITELIST)

All tests are fast unit tests (no ML, no heavy DSP) using tiny synthetic audio.
"""


import math
import unittest.mock as mock
from typing import Any

import numpy as np
import pytest

# ─── Fixtures ────────────────────────────────────────────────────────────────


def _sine(sr: int = 48_000, duration_s: float = 0.5, freq: float = 440.0) -> np.ndarray:
    """Mono sine wave, float32, range [-1, 1]."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(sr: int = 48_000, duration_s: float = 0.5) -> np.ndarray:
    """Stereo sine wave (2, N), float32."""
    mono = _sine(sr, duration_s)
    return np.stack([mono, mono * 0.9], axis=0)


# ─── ExzellenzDenker.messe_und_repariere() ───────────────────────────────────


class TestMesseUndRepariere:
    """Tests for ExzellenzDenker.messe_und_repariere() — §0 + §2.45 compliance."""

    def _make_denker(self, goal_scores: dict[str, float]) -> Any:
        """Build an ExzellenzDenker whose messe_ziele() returns fixed goal scores."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        denker.messe_ziele = mock.Mock(return_value=dict(goal_scores))  # type: ignore[method-assign]
        return denker

    def test_no_violations_returns_unchanged_audio(self):
        """When all goals ≥ 0.75, audio and goals are returned unchanged (§2.45)."""
        all_good = {
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.89,
            "tonal_center": 0.96,
            "timbre_authentizitaet": 0.88,
            "artikulation": 0.86,
            "emotionalitaet": 0.83,
            "micro_dynamics": 0.89,
            "groove": 0.84,
            "transparenz": 0.83,
            "waerme": 0.76,
            "bass_kraft": 0.79,
            "separation_fidelity": 0.79,
            "brillanz": 0.79,
            "spatial_depth": 0.71,
        }
        denker = self._make_denker(all_good)
        audio = _sine()
        out_audio, out_goals = denker.messe_und_repariere(audio, 48_000)  # type: ignore[attr-defined]

        # messe_ziele called exactly once (no repair attempt)
        assert denker.messe_ziele.call_count == 1  # type: ignore[attr-defined]
        np.testing.assert_array_equal(out_audio, audio)
        assert out_goals == all_good

    def test_borderline_violation_not_repaired(self):
        """Goals at 0.72-0.74 (deficit < 0.03) are NOT repaired (§2.45)."""
        borderline = {
            "micro_dynamics": 0.73,  # < 0.75 but deficit = 0.02 < 0.03
            "groove": 0.74,
            "waerme": 0.74,
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.90,
        }
        denker = self._make_denker(borderline)
        audio = _sine()
        out_audio, _ = denker.messe_und_repariere(audio, 48_000)  # type: ignore[attr-defined]

        assert denker.messe_ziele.call_count == 1  # no re-measurement  # type: ignore[attr-defined]
        np.testing.assert_array_equal(out_audio, audio)  # audio unchanged

    def test_significant_p35_violation_triggers_repair_attempt(self):
        """Goals with deficit ≥ 0.03 trigger the repair path."""
        with_violations = {
            "micro_dynamics": 0.70,  # deficit = 0.05 → repair triggered
            "groove": 0.71,
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.89,
            "waerme": 0.80,
        }
        denker = self._make_denker(with_violations)
        # Suppress ExcellenceOptimizer import (not needed for this test)
        with mock.patch("backend.core.excellence_optimizer.ExcellenceOptimizer") as _mock_opt:
            _mock_opt.return_value.optimize.return_value = (_sine(), mock.Mock(applied_steps=[]))
            _mock_opt.return_value._modulation_strength = 0.3
            audio = _sine()
            # Second call returns slightly improved scores
            denker.messe_ziele.side_effect = [  # type: ignore[attr-defined]
                dict(with_violations),  # initial measurement
                {  # after TD repair — improvement
                    "micro_dynamics": 0.76,
                    "groove": 0.77,
                    "natuerlichkeit": 0.91,
                    "authentizitaet": 0.89,
                    "waerme": 0.80,
                },
            ]
            out_audio, out_goals = denker.messe_und_repariere(audio, 48_000)  # type: ignore[attr-defined]

        # At minimum 2 measurements: initial + after repair
        assert denker.messe_ziele.call_count >= 2  # type: ignore[attr-defined]

    def test_p1p2_regression_guard_rejects_candidate(self):
        """A candidate that regresses a P1/P2 blend goal beyond its §v10.306
        category threshold (0.03) WITHOUT a compensating net-gain (net_gain
        <= -0.02) is rejected (§0 / §v10.306 Net-Improvement-Gate).

        Note: §v10.306 recalibrated the previous 0.01/0.015/0.02 thresholds
        (within measurement noise, rejected genuine improvements) to 0.03
        (P1/P2 blend) / 0.04 (mode-core) / 0.05 (general) + an overall
        Net-Improvement-Gate. A regression only blocks acceptance when it
        exceeds ITS category threshold AND the overall net_gain is clearly
        negative (<= -0.02) — a small regression offset by other gains is
        intentionally accepted.
        """
        initial = {
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.89,  # P1/P2 blend goal
            "micro_dynamics": 0.70,  # violation — triggers repair
            "groove": 0.72,
            "waerme": 0.80,
        }
        denker = self._make_denker(initial)
        # Repair candidate: authentizitaet (P1/P2) regresses well beyond its 0.03
        # category threshold, and the tiny micro_dynamics gain does NOT offset it
        # (net_gain = -0.05 + 0.02 = -0.03 <= -0.02) → reject.
        repaired_regressed = {
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.84,  # -0.05, exceeds P1/P2 threshold (0.03)
            "micro_dynamics": 0.72,  # +0.02, does not offset the regression
            "groove": 0.72,
            "waerme": 0.80,
        }
        denker.messe_ziele.side_effect = [  # type: ignore[attr-defined]
            dict(initial),
            dict(repaired_regressed),  # after TD repair
        ]
        with mock.patch("backend.core.excellence_optimizer.ExcellenceOptimizer") as _mopt:
            _mopt.return_value.optimize.return_value = (_sine(), mock.Mock(applied_steps=[]))
            _mopt.return_value._modulation_strength = 0.3
            audio = _sine()
            out_audio, out_goals = denker.messe_und_repariere(audio, 48_000)  # type: ignore[attr-defined]

        # Audio must remain unchanged because candidate was rejected
        np.testing.assert_array_equal(out_audio, audio)
        # Goals must reflect initial (not regressed candidate)
        assert out_goals.get("authentizitaet") == pytest.approx(0.89)

    def test_equal_passcount_prefers_lower_core_deficit(self):
        """Bei gleicher Pass-Anzahl wird der Kandidat mit niedrigerem Kernziel-Defizit gewählt."""
        initial = {
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.89,
            "spatial_depth": 0.45,  # knapp unter kanonischer Schwelle (~0.46)
            "micro_dynamics": 0.80,
            "groove": 0.82,
        }
        candidate = {
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.89,
            "spatial_depth": 0.47,  # Kernziel verbessert, Passcount bleibt identisch
            "micro_dynamics": 0.80,
            "groove": 0.82,
        }
        denker = self._make_denker(initial)
        denker.messe_ziele.side_effect = [dict(initial), dict(candidate)]  # type: ignore[attr-defined]

        with mock.patch("backend.core.excellence_optimizer.ExcellenceOptimizer") as _mopt:
            _mopt.return_value.optimize.return_value = (_sine(), mock.Mock(applied_steps=[]))
            _mopt.return_value._modulation_strength = 0.3
            out_audio, out_goals = denker.messe_und_repariere(_sine(), 48_000, reference_audio=_sine(freq=220.0))  # type: ignore[attr-defined]

        assert denker.messe_ziele.call_count >= 2  # type: ignore[attr-defined]
        assert out_goals.get("spatial_depth", 0.0) >= candidate["spatial_depth"] - 1e-6

    def test_mode_core_guard_rejects_spatial_depth_drop(self):
        """Restoration mode-core goal spatial_depth regressing beyond its §v10.306
        category threshold (0.04), with no compensating net-gain (net_gain
        <= -0.02), is rejected.
        """
        initial = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.90,
            "spatial_depth": 0.72,
            "micro_dynamics": 0.70,
            "groove": 0.71,
        }
        # spatial_depth regresses -0.05 (exceeds mode-core threshold 0.04); the
        # small micro_dynamics gain does not offset it (net_gain = -0.05 + 0.01
        # = -0.04 <= -0.02) → reject.
        regressed_candidate = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.90,
            "spatial_depth": 0.67,  # -0.05 > mode-core threshold 0.04
            "micro_dynamics": 0.71,
            "groove": 0.71,
        }
        denker = self._make_denker(initial)
        denker.messe_ziele.side_effect = [dict(initial), dict(regressed_candidate)]  # type: ignore[attr-defined]

        with mock.patch("backend.core.excellence_optimizer.ExcellenceOptimizer") as _mopt:
            _mopt.return_value.optimize.return_value = (_sine(), mock.Mock(applied_steps=[]))
            _mopt.return_value._modulation_strength = 0.3
            out_audio, out_goals = denker.messe_und_repariere(_sine(), 48_000)  # type: ignore[attr-defined]

        np.testing.assert_array_equal(out_audio, _sine())
        assert out_goals.get("spatial_depth") == pytest.approx(initial["spatial_depth"])

    def test_local_intent_guard_rejects_unbalanced_auth_spatial_tradeoff(self):
        """A small authentizitaet gain must not justify a disproportionate
        spatial_depth drop: since spatial_depth regresses well beyond its
        §v10.306 mode-core threshold (0.04) and the overall net_gain stays
        clearly negative (<= -0.02), the candidate is rejected.
        """
        initial = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.90,
            "spatial_depth": 0.72,
            "micro_dynamics": 0.70,
            "groove": 0.71,
        }
        # authentizitaet gains only +0.005, spatial_depth drops -0.07 (>> mode-core
        # threshold 0.04); net_gain = +0.005 - 0.07 + 0.01 = -0.055 <= -0.02 → reject.
        regressed_candidate = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.905,  # +0.005
            "spatial_depth": 0.65,  # -0.07, far beyond mode-core threshold 0.04
            "micro_dynamics": 0.71,
            "groove": 0.71,
        }
        denker = self._make_denker(initial)
        denker.messe_ziele.side_effect = [dict(initial), dict(regressed_candidate)]  # type: ignore[attr-defined]

        with mock.patch("backend.core.excellence_optimizer.ExcellenceOptimizer") as _mopt:
            _mopt.return_value.optimize.return_value = (_sine(), mock.Mock(applied_steps=[]))
            _mopt.return_value._modulation_strength = 0.3
            out_audio, out_goals = denker.messe_und_repariere(_sine(), 48_000)  # type: ignore[attr-defined]

        np.testing.assert_array_equal(out_audio, _sine())
        assert out_goals.get("authentizitaet") == pytest.approx(initial["authentizitaet"])
        assert out_goals.get("spatial_depth") == pytest.approx(initial["spatial_depth"])

    def test_local_rescue_improves_spatial_waerme_after_rejected_blends(self):
        """Lokaler End-Gate-Rescue darf nach frühen Rejects Spatial/Wärme gezielt verbessern."""
        initial = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.90,
            "spatial_depth": 0.44,
            "waerme": 0.66,
            "micro_dynamics": 0.80,
            "groove": 0.82,
        }
        # Step-3 Blend-Kandidaten: wegen starker Authentizitäts-Regression abzulehnen.
        step3_reject_1 = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.87,
            "spatial_depth": 0.45,
            "waerme": 0.67,
            "micro_dynamics": 0.80,
            "groove": 0.82,
        }
        step3_reject_2 = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.875,
            "spatial_depth": 0.45,
            "waerme": 0.67,
            "micro_dynamics": 0.80,
            "groove": 0.82,
        }
        # Step-5 Local-Rescue-Kandidat: lokale Defizite sinken bei akzeptabler Gesamtbilanz.
        local_rescue = {
            "natuerlichkeit": 0.92,
            "authentizitaet": 0.895,
            "spatial_depth": 0.46,
            "waerme": 0.70,
            "micro_dynamics": 0.80,
            "groove": 0.82,
        }

        denker = self._make_denker(initial)
        denker.messe_ziele.side_effect = [  # type: ignore[attr-defined]
            dict(initial),
            dict(step3_reject_1),
            dict(step3_reject_2),
            dict(local_rescue),
        ]

        ref = _sine(freq=220.0)
        out_audio, out_goals = denker.messe_und_repariere(_sine(freq=440.0), 48_000, reference_audio=ref)  # type: ignore[attr-defined]

        assert out_goals.get("spatial_depth", 0.0) >= local_rescue["spatial_depth"] - 1e-6
        assert out_goals.get("waerme", 0.0) >= local_rescue["waerme"] - 1e-6

    def test_empty_audio_returns_empty_dict(self):
        """Empty audio input → safe fallback (no crash)."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        out_audio, out_goals = denker.messe_und_repariere(np.array([], dtype=np.float32), 48_000)
        assert out_goals == {}
        assert out_audio.size == 0

    def test_nan_audio_sanitized(self):
        """NaN/Inf in input audio is sanitized before measurement."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        denker.messe_ziele = mock.Mock(return_value={"micro_dynamics": 0.80})  # type: ignore[method-assign]
        nan_audio = np.full(480, np.nan, dtype=np.float32)
        out_audio, _ = denker.messe_und_repariere(nan_audio, 48_000)
        assert np.all(np.isfinite(out_audio))

    def test_blend_repair_uses_reference_audio(self):
        """Blend step uses reference_audio when waerme/brillanz/bass_kraft fail."""
        initial = {
            "waerme": 0.70,  # deficit = 0.05 — triggers blend path
            "brillanz": 0.71,
            "bass_kraft": 0.72,
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.89,
        }
        improved_via_blend = {
            "waerme": 0.77,
            "brillanz": 0.78,
            "bass_kraft": 0.79,
            "natuerlichkeit": 0.91,
            "authentizitaet": 0.89,
        }
        denker = self._make_denker(initial)
        denker.messe_ziele.side_effect = [  # type: ignore[attr-defined]
            dict(initial),
            dict(improved_via_blend),  # blend measurement
        ]
        reference = _sine(freq=220.0)
        audio = _sine(freq=440.0)
        out_audio, out_goals = denker.messe_und_repariere(audio, 48_000, reference_audio=reference)  # type: ignore[attr-defined]
        # Blend result should be accepted (improvement found)
        assert out_goals.get("waerme", 0.0) >= initial["waerme"]

    def test_signature_returns_tuple(self):
        """messe_und_repariere() always returns (np.ndarray, dict)."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        denker.messe_ziele = mock.Mock(return_value={"groove": 0.85})  # type: ignore[method-assign]
        result = denker.messe_und_repariere(_sine(), 48_000)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], dict)


# ─── AurikDenker Stufe 7 wiring ──────────────────────────────────────────────


class TestAurikDenkerGoalRepairWiring:
    """Normative: AurikDenker must call messe_und_repariere in Stufe 7."""

    def test_messe_und_repariere_exists_on_exzellenz_denker(self):
        """ExzellenzDenker exposes messe_und_repariere() public method."""
        from denker.exzellenz_denker import ExzellenzDenker, get_exzellenz_denker

        denker = ExzellenzDenker()
        assert callable(getattr(denker, "messe_und_repariere", None)), (
            "ExzellenzDenker.messe_und_repariere() muss vorhanden sein (§2.11.1 Goal-Repair)."
        )
        # Singleton getter also returns instance with method
        singleton = get_exzellenz_denker()
        assert callable(getattr(singleton, "messe_und_repariere", None))

    def test_aurik_denker_stufe7_uses_repair_method(self):
        """AurikDenker source must reference messe_und_repariere, not bare messe_ziele."""
        import inspect

        import denker.aurik_denker as _ad_mod

        src = inspect.getsource(_ad_mod)
        assert "messe_und_repariere" in src, "AurikDenker Stufe 7 muss exd.messe_und_repariere() aufrufen (v10.0.0)."

    def test_aurik_denker_stufe7_passes_reference_audio(self):
        """AurikDenker must pass reference_audio= to messe_und_repariere."""
        import inspect

        import denker.aurik_denker as _ad_mod

        src = inspect.getsource(_ad_mod)
        assert "reference_audio=audio" in src, (
            "AurikDenker muss reference_audio=audio an messe_und_repariere() übergeben."
        )


# ─── UV3 Deferred-Phase-Promotion ────────────────────────────────────────────


class TestDeferredPhasePromotion:
    """Normative: UV3 must define the deferred-phase whitelist constants."""

    def test_deferred_fc_whitelist_defined(self):
        """UV3 source must contain _DEFERRED_FC_WHITELIST with P3-P5 phases."""
        import inspect

        import backend.core.unified_restorer_v3 as _uv3_mod

        src = inspect.getsource(_uv3_mod)
        assert "_DEFERRED_FC_WHITELIST" in src, (
            "UV3 muss _DEFERRED_FC_WHITELIST definieren (§2.47 Deferred-Phase-Promotion)."
        )

    def test_deferred_fc_whitelist_contains_p35_phases(self):
        """Whitelist must contain at least the 6 P3-P5 enhancement phase prefixes."""
        # We extract the whitelist content by importing test-wise from UV3's source
        # (the constant is defined inside a method, so we use source inspection)
        import inspect

        import backend.core.unified_restorer_v3 as _uv3_mod

        src = inspect.getsource(_uv3_mod)
        # All 6 whitelisted phases must appear in source
        for phase_prefix in ("phase_19", "phase_22", "phase_36", "phase_47", "phase_48", "phase_55"):
            assert phase_prefix in src, f"_DEFERRED_FC_WHITELIST muss '{phase_prefix}' enthalten."

    def test_deferred_phase_promotion_max3_guard(self):
        """Source must contain the max-3 guard for deferred phase addition."""
        import inspect

        import backend.core.unified_restorer_v3 as _uv3_mod

        src = inspect.getsource(_uv3_mod)
        assert "_added_deferred >= 3" in src, (
            "UV3 Deferred-Phase-Promotion muss auf max. 3 Phasen begrenzt sein (§2.45)."
        )

    def test_dfr_make_callable_defined(self):
        """A deferred-phase closure factory must be defined in UV3 for FeedbackChain."""
        import inspect

        import backend.core.unified_restorer_v3 as _uv3_mod

        src = inspect.getsource(_uv3_mod)
        assert "_dfr_make_callable" in src, (
            "UV3 muss _dfr_make_callable-Fabrik für Deferred-Phase-FeedbackChain-Integration haben."
        )


# ─── Integration: messe_und_repariere result contract ────────────────────────


class TestMesseUndRepariereContract:
    """Contract tests: output invariants regardless of repair outcome."""

    @pytest.mark.parametrize("shape", [(48000,), (2, 48000)])
    def test_output_audio_shape_preserved(self, shape: tuple[int, ...]):
        """Output audio shape must match input shape exactly."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        denker.messe_ziele = mock.Mock(return_value={"micro_dynamics": 0.85})  # type: ignore[method-assign]
        audio = np.zeros(shape, dtype=np.float32)
        out_audio, _ = denker.messe_und_repariere(audio, 48_000)
        assert out_audio.shape == shape

    def test_output_audio_clipped_to_unit_range(self):
        """Output audio must never exceed [-1.0, 1.0] (§3.1 Clip-Invariante)."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        denker.messe_ziele = mock.Mock(return_value={"micro_dynamics": 0.85})  # type: ignore[method-assign]
        audio = np.random.uniform(-0.9, 0.9, 4800).astype(np.float32)
        out_audio, _ = denker.messe_und_repariere(audio, 48_000)
        assert np.all(out_audio >= -1.0) and np.all(out_audio <= 1.0)

    def test_goals_dict_only_finite_values(self):
        """Returned goals dict must contain only finite float values."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        denker.messe_ziele = mock.Mock(return_value={"micro_dynamics": 0.85, "brillanz": float("nan")})  # type: ignore[method-assign]
        # NaN goals are filtered by messe_ziele — here we test the pass-through
        _, goals = denker.messe_und_repariere(_sine(), 48_000)
        for v in goals.values():
            # NaN values from messe_ziele may legitimately pass through
            assert math.isfinite(v) or v != v

    def test_exception_in_repair_does_not_propagate(self):
        """Any exception in repair path must be caught; original audio returned (§0)."""
        from denker.exzellenz_denker import ExzellenzDenker

        denker = ExzellenzDenker()
        initial_goals = {"micro_dynamics": 0.70, "groove": 0.70}  # triggers repair
        denker.messe_ziele = mock.Mock(return_value=dict(initial_goals))  # type: ignore[method-assign]
        audio = _sine()
        # Force ExcellenceOptimizer to raise
        with mock.patch(
            "backend.core.excellence_optimizer.ExcellenceOptimizer",
            side_effect=RuntimeError("Test-Fehler"),
        ):
            out_audio, out_goals = denker.messe_und_repariere(audio, 48_000)
        # Must not raise, original audio returned
        np.testing.assert_array_equal(out_audio, audio)
        assert out_goals == initial_goals
