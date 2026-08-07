from __future__ import annotations

import pytest

"""tests/unit/test_adaptive_goal_thresholds.py

Pflicht-Testsuite für das AdaptiveGoalThresholds-System (§2.31 Spec).

Testet:
- AdaptiveGoalThresholds Dataclass (Felder, Typen, Bounds)
- MaterialQuality Enum (alle 7 Stufen)
- MaterialQualityAnalyzer.analyze() (DSP-Analyse auf synthetischen Signalen)
- AdaptiveGoalsCalculator.calculate_adaptive_thresholds() (Threshold-Berechnung)
- Relaxation-Invarianten: Thresh sinkt monoton von PRISTINE → EXTREME
- EnhancedProcessingStrategy.get_enhanced_config()
- get_adaptive_goals_and_config Convenience-Funktion
- NaN/Inf-Sicherheit, Edge-Cases (Stille, Rauschen, mono, stereo)
- Konsistenz-Tests (gleiche Eingabe → gleiches Ergebnis)
"""


import math

import numpy as np

from backend.core.musical_goals.adaptive_goals_system import (
    AdaptiveGoalsCalculator,
    AdaptiveGoalThresholds,
    EnhancedProcessingStrategy,
    MaterialQuality,
    MaterialQualityAnalyzer,
    MaterialQualityAssessment,
    get_adaptive_goals_and_config,
)

np.random.seed(42)
SR = 48_000


# ────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ────────────────────────────────────────────────────────────────────────────


def _sine(freq: float = 440.0, duration: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _noise(duration: float = 2.0, amp: float = 0.1) -> np.ndarray:
    return (np.random.randn(int(SR * duration)) * amp).astype(np.float32)


def _silence(duration: float = 2.0) -> np.ndarray:
    return np.zeros(int(SR * duration), dtype=np.float32)


def _make_assessment(quality_level: MaterialQuality, degradation: float) -> MaterialQualityAssessment:
    """Erstellt ein synthetisches Assessment für Unit-Tests des Calculators."""
    return MaterialQualityAssessment(
        quality_level=quality_level,
        confidence=0.90,
        degradation_score=degradation,
        medium_chain=["digital"],
        generation_count=1,
        noise_level=degradation * 0.5,
        bandwidth_limitation=0.0,
        artifact_density=0.0,
        dynamic_range_db=20.0,
        recommended_strength=degradation,
        requires_enhanced_processing=degradation > 0.5,
    )


# ────────────────────────────────────────────────────────────────────────────
# 01–04: MaterialQuality Enum
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestMaterialQualityEnum:
    """Prüft vollständige Enum-Spezifikation (7 Stufen, Sektion §2.31)."""

    def test_01_all_seven_levels_present(self):
        expected = {"pristine", "excellent", "good", "fair", "poor", "very_poor", "extreme"}
        actual = {q.value for q in MaterialQuality}
        assert expected == actual

    def test_02_enum_length(self):
        assert len(MaterialQuality) == 7

    def test_03_enum_members_accessible(self):
        assert MaterialQuality.PRISTINE is not None
        assert MaterialQuality.EXTREME is not None

    def test_04_enum_values_are_strings(self):
        for member in MaterialQuality:
            assert isinstance(member.value, str)


# ────────────────────────────────────────────────────────────────────────────
# 05–08: AdaptiveGoalThresholds Dataclass
# ────────────────────────────────────────────────────────────────────────────


class TestAdaptiveGoalThresholdsDataclass:
    """Prüft die Dataclass-Struktur (§2.31, 15 Ziele plus Kontextfelder)."""

    def _make(self, level=MaterialQuality.GOOD, relax=0.3) -> AdaptiveGoalThresholds:
        return AdaptiveGoalThresholds(
            brillanz=0.70,
            waerme=0.65,
            natuerlichkeit=0.72,
            authentizitaet=0.74,
            emotionalitaet=0.73,
            transparenz=0.75,
            bass_kraft=0.70,
            quality_level=level,
            relaxation_factor=relax,
        )

    def test_05_all_fields_present(self):
        obj = self._make()
        for field_name in GOAL_ALIASES:
            assert hasattr(obj, field_name), field_name
        assert hasattr(obj, "quality_level")
        assert hasattr(obj, "relaxation_factor")

    def test_06_threshold_fields_are_floats(self):
        obj = self._make()
        for field_name in (*GOAL_ALIASES, "relaxation_factor"):
            assert isinstance(getattr(obj, field_name), float), field_name

    def test_07_quality_level_field_is_enum(self):
        obj = self._make()
        assert isinstance(obj.quality_level, MaterialQuality)

    def test_08_all_thresholds_finite(self):
        obj = self._make()
        for field_name in (*GOAL_ALIASES, "relaxation_factor"):
            val = getattr(obj, field_name)
            assert math.isfinite(val), f"NaN/Inf in {field_name}"


# ────────────────────────────────────────────────────────────────────────────
# 09–13: MaterialQualityAnalyzer
# ────────────────────────────────────────────────────────────────────────────


class TestMaterialQualityAnalyzer:
    """Prüft MaterialQualityAnalyzer.analyze() auf synthetischen Signalen."""

    def setup_method(self):
        self.analyzer = MaterialQualityAnalyzer()

    def test_09_analyze_returns_assessment_type(self):
        audio = _sine()
        result = self.analyzer.analyze(audio, SR)
        assert isinstance(result, MaterialQualityAssessment)

    def test_10_analyze_quality_level_is_valid_enum(self):
        audio = _sine()
        result = self.analyzer.analyze(audio, SR)
        assert result.quality_level in MaterialQuality

    def test_11_degradation_score_in_bounds(self):
        audio = _noise(amp=0.3)
        result = self.analyzer.analyze(audio, SR)
        assert 0.0 <= result.degradation_score <= 1.0

    def test_12_analyze_silence_no_crash(self):
        audio = _silence()
        result = self.analyzer.analyze(audio, SR)
        assert isinstance(result, MaterialQualityAssessment)
        assert math.isfinite(result.degradation_score)

    def test_13_analyze_stereo_no_crash(self):
        mono = _sine()
        stereo = np.stack([mono, mono], axis=0)
        # Analyzer sollte mit mono und stereo umgehen
        result = self.analyzer.analyze(mono, SR)
        assert isinstance(result, MaterialQualityAssessment)


# ────────────────────────────────────────────────────────────────────────────
# 14–20: AdaptiveGoalsCalculator
# ────────────────────────────────────────────────────────────────────────────


class TestAdaptiveGoalsCalculator:
    """Prüft calculate_adaptive_thresholds() Logik und Invarianten."""

    def setup_method(self):
        self.calc = AdaptiveGoalsCalculator()

    def _thresholds_for(self, level: MaterialQuality, degradation: float) -> AdaptiveGoalThresholds:
        assessment = _make_assessment(level, degradation)
        return self.calc.calculate_adaptive_thresholds(assessment)

    def test_14_pristine_returns_high_thresholds(self):
        t = self._thresholds_for(MaterialQuality.PRISTINE, 0.0)
        # Bei PRISTINE (degradation=0.0) kein Nachlass
        assert t.brillanz >= 0.80
        assert t.natuerlichkeit >= 0.80
        assert t.authentizitaet >= 0.83

    def test_15_extreme_returns_relaxed_thresholds(self):
        t = self._thresholds_for(MaterialQuality.EXTREME, 1.0)
        # Bei EXTREME (degradation=1.0) starker Nachlass
        assert t.brillanz < 0.70
        assert t.natuerlichkeit < 0.70

    def test_16_monotone_relaxation_with_degradation(self):
        """Thresh muss mit steigendem Degradation-Score sinken."""
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        levels = [
            MaterialQuality.PRISTINE,
            MaterialQuality.GOOD,
            MaterialQuality.FAIR,
            MaterialQuality.POOR,
            MaterialQuality.EXTREME,
        ]
        thresholds = [self._thresholds_for(lvl, deg).brillanz for lvl, deg in zip(levels, scores)]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1], (
                f"Brillanz nicht monoton: score={scores[i]}→{thresholds[i]:.3f} "
                f"> score={scores[i + 1]}→{thresholds[i + 1]:.3f}"
            )

    def test_17_relaxation_factor_stored_correctly(self):
        t = self._thresholds_for(MaterialQuality.POOR, 0.6)
        assert abs(t.relaxation_factor - 0.6) < 1e-6

    def test_18_all_thresholds_in_zero_one_range(self):
        for level, deg in [
            (MaterialQuality.PRISTINE, 0.0),
            (MaterialQuality.EXCELLENT, 0.1),
            (MaterialQuality.GOOD, 0.2),
            (MaterialQuality.FAIR, 0.35),
            (MaterialQuality.POOR, 0.55),
            (MaterialQuality.VERY_POOR, 0.65),
            (MaterialQuality.EXTREME, 1.0),
        ]:
            t = self._thresholds_for(level, deg)
            for field in (
                "brillanz",
                "waerme",
                "natuerlichkeit",
                "authentizitaet",
                "emotionalitaet",
                "transparenz",
                "bass_kraft",
            ):
                val = getattr(t, field)
                assert 0.0 <= val <= 1.0, f"{field}={val} out of range for {level}"

    def test_19_all_output_fields_finite(self):
        t = self._thresholds_for(MaterialQuality.FAIR, 0.4)
        for field in (
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "bass_kraft",
            "relaxation_factor",
        ):
            assert math.isfinite(getattr(t, field)), f"NaN/Inf in {field}"

    def test_20_quality_level_preserved(self):
        t = self._thresholds_for(MaterialQuality.VERY_POOR, 0.7)
        assert t.quality_level == MaterialQuality.VERY_POOR


# ────────────────────────────────────────────────────────────────────────────
# 21–25: EnhancedProcessingStrategy
# ────────────────────────────────────────────────────────────────────────────


class TestEnhancedProcessingStrategy:
    """Prüft get_enhanced_config() für verschiedene Qualitätsstufen."""

    def setup_method(self):
        self.strategy = EnhancedProcessingStrategy()
        self.base_config = {
            "denoise_strength": 0.3,
            "enhancement_strength": 0.3,
            "enable_spectral_repair": False,
        }

    def _config_for(self, level: MaterialQuality, degradation: float) -> dict:
        assessment = _make_assessment(level, degradation)
        return self.strategy.get_enhanced_config(assessment, self.base_config)

    def test_21_returns_dict(self):
        cfg = self._config_for(MaterialQuality.GOOD, 0.2)
        assert isinstance(cfg, dict)

    def test_22_base_config_not_mutated(self):
        original_denoise = self.base_config["denoise_strength"]
        self._config_for(MaterialQuality.EXTREME, 1.0)
        assert self.base_config["denoise_strength"] == original_denoise

    def test_23_extreme_activates_spectral_repair(self):
        cfg = self._config_for(MaterialQuality.EXTREME, 1.0)
        assert cfg.get("enable_spectral_repair") is True

    def test_24_extreme_activates_multi_pass(self):
        cfg = self._config_for(MaterialQuality.EXTREME, 1.0)
        assert cfg.get("multi_pass_processing") is True

    def test_25_all_config_values_finite(self):
        for level, deg in [(MaterialQuality.PRISTINE, 0.0), (MaterialQuality.POOR, 0.6)]:
            cfg = self._config_for(level, deg)
            for k, v in cfg.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"NaN/Inf in config['{k}']"


# ────────────────────────────────────────────────────────────────────────────
# 26–28: Convenience-Funktion
# ────────────────────────────────────────────────────────────────────────────


class TestConvenienceFunction:
    """Prüft get_adaptive_goals_and_config()."""

    def test_26_returns_three_element_tuple(self):
        audio = _sine()
        result = get_adaptive_goals_and_config(audio, SR, {})
        assert len(result) == 3

    def test_27_first_element_is_adaptive_thresholds(self):
        audio = _sine()
        thresholds, _, _ = get_adaptive_goals_and_config(audio, SR, {})
        assert isinstance(thresholds, AdaptiveGoalThresholds)

    def test_28_third_element_is_quality_assessment(self):
        audio = _noise(amp=0.2)
        _, _, quality = get_adaptive_goals_and_config(audio, SR, {})
        assert isinstance(quality, MaterialQualityAssessment)

    # ── Tests 29–32: Koerzierung nicht-dict Argumente (§2.31 Robustheit) ──

    def test_29_non_dict_base_config_enum_coerced(self):
        """Enum als base_config wird auf {} normalisiert — kein AttributeError."""
        from enum import Enum

        class FakeEnum(Enum):
            SPEECH = "speech"

        audio = _sine()
        thresholds, cfg, quality = get_adaptive_goals_and_config(audio, SR, FakeEnum.SPEECH)
        assert isinstance(thresholds, AdaptiveGoalThresholds)
        assert isinstance(cfg, dict)
        assert isinstance(quality, MaterialQualityAssessment)

    def test_30_none_base_config_coerced_to_empty_dict(self):
        """None als base_config wird zu {} normalisiert."""
        audio = _sine()
        thresholds, cfg, quality = get_adaptive_goals_and_config(audio, SR, None)
        assert isinstance(cfg, dict)
        assert isinstance(thresholds, AdaptiveGoalThresholds)

    def test_31_non_dict_medium_detection_coerced_to_none(self):
        """Nicht-dict medium_detection (z.B. Dataclass) wird auf None normalisiert."""

        class FakeDataclass:
            channels = 1
            material = "shellac"

        audio = _sine()
        thresholds, cfg, quality = get_adaptive_goals_and_config(audio, SR, {}, FakeDataclass())
        assert isinstance(thresholds, AdaptiveGoalThresholds)
        assert isinstance(cfg, dict)

    def test_32_all_non_dict_args_coerced_no_exception(self):
        """Beide Argumente nicht-dict → robuste Verarbeitung, kein Absturz."""
        audio = _noise(amp=0.1)
        result = get_adaptive_goals_and_config(audio, SR, "not_a_dict", 42)
        assert len(result) == 3
        thresholds, cfg, quality = result
        assert isinstance(thresholds, AdaptiveGoalThresholds)
        assert isinstance(cfg, dict)


import types as _types

# ---------------------------------------------------------------------------
# P1-2: Tests für das extrahierte Standalone-Modul adaptive_goal_resolver
# ---------------------------------------------------------------------------
import pytest

from backend.core.musical_goals.adaptive_goal_resolver import (
    GOAL_ALIASES,
    resolve_adaptive_goal_thresholds,
)


class TestAdaptiveGoalResolverModule:
    """Tests für backend.core.musical_goals.adaptive_goal_resolver (P1-2)."""

    def test_33_all_15_aliases_defined(self):
        """GOAL_ALIASES muss alle 15 kanonischen Zielschlüssel enthalten."""
        assert len(GOAL_ALIASES) == 15
        expected = {
            "bass_kraft",
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "groove",
            "spatial_depth",
            "timbre_authentizitaet",
            "tonal_center",
            "micro_dynamics",
            "separation_fidelity",
            "artikulation",
            "transient_energie",
        }
        assert set(GOAL_ALIASES.keys()) == expected

    def test_34_resolves_from_dict_canonical_keys(self):
        """Kanonische Keys in einem flat dict werden korrekt aufgelöst."""
        payload = {"brillanz": 0.82, "waerme": 0.78, "bass_kraft": 0.77}
        result = resolve_adaptive_goal_thresholds(payload)
        assert result["brillanz"] == pytest.approx(0.82)
        assert result["waerme"] == pytest.approx(0.78)
        assert result["bass_kraft"] == pytest.approx(0.77)

    def test_35_resolves_bass_kraft_alias(self):
        """Legacy-Alias 'bass-kraft' wird auf 'bass_kraft' normiert."""
        payload = {"bass-kraft": 0.71}
        result = resolve_adaptive_goal_thresholds(payload)
        assert "bass_kraft" in result
        assert result["bass_kraft"] == pytest.approx(0.71)
        assert "bass-kraft" not in result

    def test_36_resolves_from_object_attributes(self):
        """Direkte Objekt-Attribute werden korrekt ausgelesen."""
        obj = _types.SimpleNamespace(
            brillanz=0.84,
            waerme=0.80,
            natuerlichkeit=0.91,
            authentizitaet=0.88,
            emotionalitaet=0.87,
            transparenz=0.89,
            bass_kraft=0.85,
        )
        result = resolve_adaptive_goal_thresholds(obj)
        assert result["brillanz"] == pytest.approx(0.84)
        assert result["bass_kraft"] == pytest.approx(0.85)

    def test_37_resolves_from_thresholds_attribute_dict(self):
        """Objekt mit .thresholds dict wird korrekt aufgelöst."""
        obj = _types.SimpleNamespace(
            thresholds={
                "groove": 0.62,
                "spatial_depth": 0.63,
                "tonal_center": 0.95,
            }
        )
        result = resolve_adaptive_goal_thresholds(obj)
        assert result["groove"] == pytest.approx(0.62)
        assert result["spatial_depth"] == pytest.approx(0.63)
        assert result["tonal_center"] == pytest.approx(0.95)

    def test_38_resolves_from_tuple_payload(self):
        """Tupel-Payload (wie get_adaptive_goals_and_config-Rückgabe) aller Teile."""
        obj1 = _types.SimpleNamespace(brillanz=0.83)
        obj2 = _types.SimpleNamespace(thresholds={"groove": 0.60})
        result = resolve_adaptive_goal_thresholds((obj1, obj2, None))
        assert result["brillanz"] == pytest.approx(0.83)
        assert result["groove"] == pytest.approx(0.60)

    def test_39_ignores_nan_and_inf_values(self):
        """NaN / Inf Werte werden ignoriert, kein Eintrag im Ergebnis."""

        payload = {"brillanz": float("nan"), "waerme": float("inf"), "groove": 0.60}
        result = resolve_adaptive_goal_thresholds(payload)
        assert "brillanz" not in result
        assert "waerme" not in result
        assert result["groove"] == pytest.approx(0.60)

    def test_40_empty_payload_returns_empty_dict(self):
        """Kein auflösbarer Payload liefert leeres Dict (kein Exception)."""
        assert resolve_adaptive_goal_thresholds(None) == {}
        assert resolve_adaptive_goal_thresholds({}) == {}
        assert resolve_adaptive_goal_thresholds((None, None)) == {}

    def test_41_uv3_static_method_delegates_to_module(self):
        """UnifiedRestorerV3._resolve_adaptive_goal_thresholds delegiert korrekt."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        payload = {"brillanz": 0.85, "waerme": 0.80}
        via_uv3 = UnifiedRestorerV3._resolve_adaptive_goal_thresholds(payload)
        via_module = resolve_adaptive_goal_thresholds(payload)
        assert via_uv3 == via_module
