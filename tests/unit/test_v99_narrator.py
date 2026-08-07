from __future__ import annotations

import pytest

"""
tests/unit/test_v99_narrator.py — RestorationNarrator Test-Suite (≥ 35 Tests)

Testet alle Zeige des RestorationNarrators:
  • NarratorResult-Struktur und -Serialisierung
  • _effective_quality (alle Gewichtungspfade)
  • _quality_stars / _difficulty_stars (alle Stufen)
  • _build_verdict (alle Qualitätsstufen)
  • _build_emotional_summary (diverse Materialien/Ären)
  • _build_trust_message (high/medium/low-Konfidenz)
  • _build_learning_message (alle GP-Stufen)
  • _build_era_context (alle Jahrzehnte)
  • NaN/Inf-Robustheit
  • Singleton-Pattern (Thread-Safety)
  • Convenience-Funktion narrate_restoration()
"""


import math
import threading

import numpy as np

from backend.core.restoration_narrator import (
    NarratorResult,
    RestorationNarrator,
    _defect_label,
    _material_label,
    get_narrator,
    narrate_restoration,
)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_narrator() -> RestorationNarrator:
    return RestorationNarrator()


def _base_narrate(**overrides):
    """Basisaufruf mit sicheren Standardwerten + optionalen Überschreibungen."""
    defaults = {
        "quality_estimate": 0.70,
        "material": "vinyl",
        "confidence": 0.65,
        "confidence_tier": "medium",
        "musical_goal_scores": {"brillanz": 0.88, "waerme": 0.82, "natuerlichkeit": 0.91},
        "musical_goals_passed": {"brillanz": True, "waerme": True, "natuerlichkeit": True},
        "top_defects": [("crackle", 0.75), ("hum", 0.40)],
        "executed_phases": 12,
        "era_decade": 1963,
        "era_label": "1960er Jahren",
        "pqs_mos": 4.2,
        "gp_observations": 5,
    }
    defaults.update(overrides)
    return narrate_restoration(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# 1. NarratorResult — Struktur & Serialisierung
# ===========================================================================


@pytest.mark.unit
class TestNarratorResult:
    def test_01_result_has_required_fields(self):
        """NarratorResult enthält alle Pflichtfelder."""
        r = _base_narrate()
        assert hasattr(r, "verdict")
        assert hasattr(r, "emotional_summary")
        assert hasattr(r, "comparison_hint")
        assert hasattr(r, "difficulty_stars")
        assert hasattr(r, "quality_stars")
        assert hasattr(r, "confidence_tier")
        assert hasattr(r, "defects_found")
        assert hasattr(r, "defects_fixed")
        assert hasattr(r, "gp_observations")

    def test_02_verdict_is_nonempty_string(self):
        r = _base_narrate()
        assert isinstance(r.verdict, str)
        assert len(r.verdict) > 5

    def test_03_emotional_summary_nonempty(self):
        r = _base_narrate()
        assert isinstance(r.emotional_summary, str)
        assert len(r.emotional_summary) > 10

    def test_04_comparison_hint_nonempty(self):
        r = _base_narrate()
        assert isinstance(r.comparison_hint, str)
        assert len(r.comparison_hint) > 5

    def test_05_quality_stars_in_range(self):
        r = _base_narrate(quality_estimate=0.75)
        assert 1 <= r.quality_stars <= 5

    def test_06_difficulty_stars_in_range(self):
        r = _base_narrate(confidence=0.30)
        assert 1 <= r.difficulty_stars <= 5

    def test_07_defects_found_is_list(self):
        r = _base_narrate()
        assert isinstance(r.defects_found, list)

    def test_08_defects_fixed_is_list(self):
        r = _base_narrate()
        assert isinstance(r.defects_fixed, list)

    def test_09_gp_observations_matches_input(self):
        r = _base_narrate(gp_observations=17)
        assert r.gp_observations == 17

    def test_10_as_dict_returns_dict(self):
        r = _base_narrate()
        d = r.as_dict()
        assert isinstance(d, dict)

    def test_11_as_dict_contains_all_keys(self):
        r = _base_narrate()
        d = r.as_dict()
        required_keys = {
            "verdict",
            "emotional_summary",
            "comparison_hint",
            "trust_message",
            "learning_message",
            "difficulty_stars",
            "quality_stars",
            "confidence_tier",
            "era_context",
            "defects_found",
            "defects_fixed",
            "gp_observations",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_12_as_dict_no_nan_inf(self):
        r = _base_narrate()
        d = r.as_dict()
        for k, v in d.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"NaN/Inf in key {k}"


# ===========================================================================
# 2. Qualitätsstufen — _quality_stars
# ===========================================================================


class TestQualityStars:
    N = _make_narrator()

    def test_13_excellent_gets_5_stars(self):
        assert self.N._quality_stars(0.85) == 5
        assert self.N._quality_stars(1.00) == 5

    def test_14_good_gets_4_stars(self):
        assert self.N._quality_stars(0.70) == 4

    def test_15_fair_gets_3_stars(self):
        assert self.N._quality_stars(0.55) == 3

    def test_16_poor_gets_2_stars(self):
        assert self.N._quality_stars(0.30) == 2

    def test_17_very_poor_gets_1_star(self):
        assert self.N._quality_stars(0.10) == 1
        assert self.N._quality_stars(0.00) == 1

    def test_18_boundary_exact(self):
        # Grenzwert genau auf Schwelle
        assert self.N._quality_stars(0.82) == 5
        assert self.N._quality_stars(0.819) == 4


# ===========================================================================
# 3. Schwierigkeitssterne — _difficulty_stars
# ===========================================================================


class TestDifficultyStars:
    N = _make_narrator()

    def test_19_low_confidence_raises_difficulty(self):
        stars_low = self.N._difficulty_stars(0.30, [("crackle", 0.8)])
        stars_high = self.N._difficulty_stars(0.90, [("crackle", 0.2)])
        assert stars_low >= stars_high

    def test_20_many_severe_defects_raise_difficulty(self):
        few_defects = [("crackle", 0.2)]
        many_defects = [("crackle", 0.8), ("hum", 0.9), ("dropouts", 0.85)]
        assert self.N._difficulty_stars(0.60, many_defects) >= self.N._difficulty_stars(0.60, few_defects)

    def test_21_max_is_5(self):
        assert self.N._difficulty_stars(0.10, [("x", 0.9)] * 10) <= 5

    def test_22_min_is_1(self):
        assert self.N._difficulty_stars(0.99, []) >= 1


# ===========================================================================
# 4. Effektive Qualität — _effective_quality
# ===========================================================================


class TestEffectiveQuality:
    N = _make_narrator()

    def test_23_without_mos_and_goals_uses_estimate(self):
        q = self.N._effective_quality(0.70, None, {})
        assert abs(q - 0.70) < 1e-6

    def test_24_with_mos_blended(self):
        # MOS=5.0 → q_mos=1.0; quality=0.0 → result > 0.0
        q = self.N._effective_quality(0.0, 5.0, {})
        assert q > 0.0

    def test_25_with_goals_blended(self):
        goals = {"a": 0.9, "b": 0.80}
        q = self.N._effective_quality(0.50, None, goals)
        assert 0.0 <= q <= 1.0

    def test_26_nan_quality_estimate_handled(self):
        # NaN in quality_estimate guard: narrate() wird aufgerufen
        r = _base_narrate(quality_estimate=float("nan"))
        assert 1 <= r.quality_stars <= 5

    def test_27_nan_pqs_mos_handled(self):
        r = _base_narrate(pqs_mos=float("nan"))
        assert 1 <= r.quality_stars <= 5

    def test_28_inf_quality_handled(self):
        r = _base_narrate(quality_estimate=float("inf"))
        assert 1 <= r.quality_stars <= 5

    def test_29_output_always_in_01(self):
        for q in np.linspace(0.0, 1.0, 20):
            eff = self.N._effective_quality(float(q), 4.0, {"x": 0.85})
            assert 0.0 <= eff <= 1.0, f"Out of range at q={q}: eff={eff}"


# ===========================================================================
# 5. Verdict — alle Zweige
# ===========================================================================


class TestVerdict:
    def test_30_excellent_verdict_positive(self):
        r = _base_narrate(quality_estimate=0.90, pqs_mos=4.8)
        assert "ja" in r.verdict.lower() or "ausgezeichnet" in r.verdict.lower() or "weltklasse" in r.verdict.lower()

    def test_31_good_verdict_mentions_improvement(self):
        r = _base_narrate(quality_estimate=0.70, pqs_mos=4.1)
        text = r.verdict.lower()
        assert any(w in text for w in ("ja", "besser", "klar", "erfolg", "behoben", "entfernt"))

    def test_32_poor_verdict_honest(self):
        r = _base_narrate(
            quality_estimate=0.15,
            pqs_mos=2.0,
            musical_goal_scores={},
            top_defects=[("dropouts", 0.95), ("hum", 0.90)],
        )
        # Ein ehrlicher Satz über schwieriges Material
        assert len(r.verdict) > 0

    def test_33_verdict_always_nonempty(self):
        """Verdict ist unter allen Bedingungen gefüllt."""
        for q in [0.0, 0.25, 0.50, 0.75, 1.0]:
            r = _base_narrate(quality_estimate=q, pqs_mos=None, musical_goal_scores={})
            assert len(r.verdict) > 0


# ===========================================================================
# 6. Trust Message — Konfidenz-Zweige
# ===========================================================================


class TestTrustMessage:
    N = _make_narrator()

    def test_34_high_confidence_no_trust_message(self):
        r = _base_narrate(confidence=0.90, confidence_tier="high")
        assert r.trust_message is None

    def test_35_low_confidence_gives_trust_message(self):
        r = _base_narrate(confidence=0.25, confidence_tier="low")
        assert r.trust_message is not None
        assert len(r.trust_message) > 10

    def test_36_medium_with_severe_defects_may_warn(self):
        r = _base_narrate(
            confidence=0.55,
            confidence_tier="medium",
            top_defects=[("dropouts", 0.95), ("hum", 0.85)],
        )
        # Kann eine Meldung enthalten (2 severe defects)
        # (kein Assert auf None — Material ist zulässig)
        assert isinstance(r.trust_message, (str, type(None)))

    def test_37_trust_message_in_german(self):
        r = _base_narrate(confidence=0.20, confidence_tier="low")
        if r.trust_message:
            # Mindestens eines der häufigen deutschen Wörter
            german_words = ("das", "die", "der", "ein", "und", "nicht", "wird")
            assert any(w in r.trust_message.lower() for w in german_words)


# ===========================================================================
# 7. Learning Message — GP-Zweige
# ===========================================================================


class TestLearningMessage:
    def test_38_zero_observations_gentle_intro(self):
        r = _base_narrate(gp_observations=0)
        assert r.learning_message is not None
        assert "erste" in r.learning_message.lower() or "ersten" in r.learning_message.lower()

    def test_39_few_observations(self):
        r = _base_narrate(gp_observations=2)
        assert r.learning_message is not None
        assert (
            "2" in r.learning_message
            or "zwei" in r.learning_message.lower()
            or "Erfahrungen" in r.learning_message
            or "ersten" in r.learning_message
        )

    def test_40_medium_observations(self):
        r = _base_narrate(gp_observations=8)
        assert r.learning_message is not None
        assert len(r.learning_message) > 5

    def test_41_many_observations_expert_message(self):
        r = _base_narrate(gp_observations=35)
        assert r.learning_message is not None
        assert "35" in r.learning_message or "Spezialist" in r.learning_message or "Aufnahmen" in r.learning_message

    def test_42_gp_message_never_empty(self):
        """Learning-Message ist unter allen Beobachtungs-Werten vorhanden."""
        for n_obs in [0, 1, 5, 11, 30, 100]:
            r = _base_narrate(gp_observations=n_obs)
            assert r.learning_message is not None


# ===========================================================================
# 8. Era-Kontext
# ===========================================================================


class TestEraContext:
    def test_43_no_era_is_none(self):
        r = _base_narrate(era_decade=None, era_label=None)
        assert r.era_context is None

    def test_44_era_1910_gives_context(self):
        r = _base_narrate(era_decade=1910, era_label="1910er Jahren")
        assert r.era_context is not None
        assert len(r.era_context) > 10

    def test_45_era_1940_gives_context(self):
        r = _base_narrate(era_decade=1940, era_label="1940er Jahren")
        assert r.era_context is not None

    def test_46_era_1970_gives_context(self):
        r = _base_narrate(era_decade=1970, era_label="1970er Jahren")
        assert r.era_context is not None

    def test_47_era_2020_gives_context(self):
        r = _base_narrate(era_decade=2020, era_label="2020er Jahren")
        assert r.era_context is not None

    def test_48_era_context_in_german(self):
        r = _base_narrate(era_decade=1955, era_label="1950er Jahren")
        if r.era_context:
            assert any(
                w in r.era_context.lower()
                for w in ("aufnahme", "klang", "studio", "röhren", "ära", "epoche", "goldene", "aufnahmen")
            )


# ===========================================================================
# 9. Defekt-Übersetzung
# ===========================================================================


class TestDefectTranslation:
    def test_49_known_defect_translated(self):
        assert "Knistern" in _defect_label("crackle")

    def test_50_unknown_defect_fallback(self):
        label = _defect_label("some_weird_defect_xyz")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_51_empty_defects_gives_empty_lists(self):
        r = _base_narrate(top_defects=[])
        assert r.defects_found == []
        assert r.defects_fixed == []

    def test_52_severe_defects_in_fixed(self):
        r = _base_narrate(top_defects=[("crackle", 0.80)])
        assert len(r.defects_fixed) >= 1

    def test_53_mild_defects_not_in_fixed(self):
        r = _base_narrate(top_defects=[("hum", 0.20)])
        # 0.20 < threshold 0.25 → not in found
        assert len(r.defects_found) == 0

    def test_54_material_label_known(self):
        assert "Schallplatte" in _material_label("vinyl")
        assert "Magnetband" in _material_label("tape")

    def test_55_material_label_unknown_fallback(self):
        label = _material_label("unknown_xyz")
        assert isinstance(label, str)
        assert len(label) > 0


# ===========================================================================
# 10. Diverse Materialien & Kantenfälle
# ===========================================================================


class TestEdgeCases:
    def test_56_all_goals_none(self):
        r = _base_narrate(musical_goal_scores=None, musical_goals_passed=None)
        assert isinstance(r.verdict, str)

    def test_57_tape_material(self):
        r = _base_narrate(material="tape", top_defects=[("wow_flutter", 0.70)])
        assert isinstance(r.emotional_summary, str)

    def test_58_shellac_material(self):
        r = _base_narrate(material="shellac", era_decade=1930, era_label="1930er Jahren")
        assert isinstance(r.emotional_summary, str)

    def test_59_no_defects_no_era(self):
        r = _base_narrate(top_defects=[], era_decade=None, era_label=None)
        assert isinstance(r.verdict, str)
        assert r.era_context is None

    def test_60_zero_phases(self):
        r = _base_narrate(executed_phases=0)
        assert isinstance(r.verdict, str)

    def test_61_quality_exactly_zero(self):
        r = _base_narrate(quality_estimate=0.0, pqs_mos=1.0, musical_goal_scores={})
        assert r.quality_stars == 1

    def test_62_quality_exactly_one(self):
        r = _base_narrate(quality_estimate=1.0, pqs_mos=5.0, musical_goal_scores={"brillanz": 1.0})
        assert r.quality_stars == 5

    def test_63_confidence_exactly_zero(self):
        r = _base_narrate(confidence=0.0, confidence_tier="low")
        assert r.trust_message is not None

    def test_64_confidence_exactly_one(self):
        r = _base_narrate(confidence=1.0, confidence_tier="high")
        assert r.trust_message is None


# ===========================================================================
# 11. Singleton & Thread-Safety
# ===========================================================================


class TestSingleton:
    def test_65_get_narrator_returns_instance(self):
        n = get_narrator()
        assert isinstance(n, RestorationNarrator)

    def test_66_singleton_same_object(self):
        n1 = get_narrator()
        n2 = get_narrator()
        assert n1 is n2

    def test_67_thread_safe_concurrent_access(self):
        """Mehrere Threads greifen gleichzeitig auf den Singleton zu — kein Fehler."""
        results = []
        errors = []

        def worker():
            try:
                n = get_narrator()
                r = n.narrate(
                    quality_estimate=0.70,
                    material="vinyl",
                    confidence=0.65,
                    gp_observations=5,
                )
                results.append(r.quality_stars)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == [], f"Thread-Fehler: {errors}"
        assert len(results) == 12
        assert all(1 <= s <= 5 for s in results)

    def test_68_convenience_function_returns_narrator_result(self):
        r = narrate_restoration(quality_estimate=0.80, material="tape")
        assert isinstance(r, NarratorResult)


# ===========================================================================
# 12. Konsistenz-Tests
# ===========================================================================


class TestConsistency:
    def test_69_deterministic_same_input(self):
        """Gleiche Eingabe → gleiche Ausgabe (deterministisch)."""
        kwargs = {
            "quality_estimate": 0.72,
            "material": "vinyl",
            "confidence": 0.65,
            "gp_observations": 7,
        }
        r1 = narrate_restoration(**kwargs)  # type: ignore[arg-type]
        r2 = narrate_restoration(**kwargs)  # type: ignore[arg-type]
        assert r1.verdict == r2.verdict
        assert r1.quality_stars == r2.quality_stars

    def test_70_higher_quality_higher_or_equal_stars(self):
        """Bessere Qualität → gleiche oder höhere Sternbewertung."""
        stars_low = _base_narrate(quality_estimate=0.20, pqs_mos=2.0, musical_goal_scores={}).quality_stars
        stars_high = _base_narrate(quality_estimate=0.90, pqs_mos=4.8, musical_goal_scores={"a": 0.95}).quality_stars
        assert stars_high >= stars_low

    def test_71_confidence_tier_stored_correctly(self):
        r = _base_narrate(confidence_tier="low")
        assert r.confidence_tier == "low"

    def test_72_gp_obs_zero_not_negative(self):
        r = _base_narrate(gp_observations=0)
        assert r.gp_observations == 0
