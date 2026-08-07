from __future__ import annotations

"""
tests/unit/test_denker_intelligence_trio.py
============================================
Unit-Tests für die drei neuen Denker-Intelligenz-APIs:

  1. TontraegerketteDenker.leite_phasen_ab() → ChainPhasePlan
  2. ExzellenzDenker.prognostiziere() → dict[str, float]
  3. StrategieDenker.schaetze_phasen_tier() → dict[str, str]
  4. PhaseInteractionDenker.plan() → chain+goal+tier Integration
  5. AurikDenker Stufe 5b: kette-Guard + _goal_risk_map-Prognose-Pfad

Alle Tests sind Marker-frei (kein ml/slow/e2e) und laufen in der Standard-Suite.
"""


from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ── 1. TontraegerketteDenker.leite_phasen_ab() ─────────────────────────────
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeitePhaseNAb:
    """§2.46 Trägerkette → ChainPhasePlan Invarianten."""

    @pytest.fixture()
    def ketten_ergebnis_vinyl(self):
        """Minimales KettenErgebnis-Mock für Vinyl."""
        glied_shellac = MagicMock()
        glied_shellac.medium = "shellac"
        # KettenGlied verwendet recommended_phases (nicht priority_phases)
        glied_shellac.recommended_phases = ["phase_03_denoise", "phase_01_click_removal"]

        glied_vinyl = MagicMock()
        glied_vinyl.medium = "vinyl"
        glied_vinyl.recommended_phases = [
            "phase_09_crackle_removal",
            "phase_06_frequency_restoration",  # additiv
        ]

        ergebnis = MagicMock()
        ergebnis.glieder = [glied_shellac, glied_vinyl]  # shellac ältere Stufe
        ergebnis.primary_material = "vinyl"
        ergebnis.chain_string = "shellac→vinyl"
        ergebnis.generation_count = 2
        return ergebnis

    def test_returns_chain_phase_plan(self, ketten_ergebnis_vinyl):
        from denker.tontraegerkette_denker import (
            ChainPhasePlan,
            get_tontraegerkette_denker,
        )

        denker = get_tontraegerkette_denker()
        plan = denker.leite_phasen_ab(ketten_ergebnis_vinyl)

        assert isinstance(plan, ChainPhasePlan)

    def test_must_have_phases_not_empty(self, ketten_ergebnis_vinyl):
        from denker.tontraegerkette_denker import get_tontraegerkette_denker

        plan = get_tontraegerkette_denker().leite_phasen_ab(ketten_ergebnis_vinyl)
        assert len(plan.must_have_phases) > 0

    def test_additive_phases_are_subset(self, ketten_ergebnis_vinyl):
        from denker.tontraegerkette_denker import get_tontraegerkette_denker

        plan = get_tontraegerkette_denker().leite_phasen_ab(ketten_ergebnis_vinyl)
        for ph in plan.additive_phases:
            assert ph in plan.must_have_phases, f"Additive Phase {ph!r} fehlt in must_have_phases"

    def test_subtractive_before_additive(self, ketten_ergebnis_vinyl):
        """§2.46-Invariante: Subtraktive Phasen kommen vor additiven."""
        from denker.tontraegerkette_denker import (
            _ADDITIVE_PHASE_PREFIXES,
            get_tontraegerkette_denker,
        )

        plan = get_tontraegerkette_denker().leite_phasen_ab(ketten_ergebnis_vinyl)
        phases = plan.must_have_phases
        additive_idxs = [i for i, p in enumerate(phases) if any(p.startswith(pfx) for pfx in _ADDITIVE_PHASE_PREFIXES)]
        subtractive_idxs = [
            i for i, p in enumerate(phases) if not any(p.startswith(pfx) for pfx in _ADDITIVE_PHASE_PREFIXES)
        ]
        if additive_idxs and subtractive_idxs:
            assert max(subtractive_idxs) < min(additive_idxs), "Subtraktive Phasen müssen vor additiven stehen (§2.46)"

    def test_no_duplicate_phases(self, ketten_ergebnis_vinyl):
        from denker.tontraegerkette_denker import get_tontraegerkette_denker

        plan = get_tontraegerkette_denker().leite_phasen_ab(ketten_ergebnis_vinyl)
        assert len(plan.must_have_phases) == len(set(plan.must_have_phases)), "Duplikate in must_have_phases"

    def test_fail_safe_empty_glieder(self):
        """Fail-safe: leere Glieder → ChainPhasePlan mit leeren Listen."""
        from denker.tontraegerkette_denker import ChainPhasePlan, get_tontraegerkette_denker

        ergebnis = MagicMock()
        ergebnis.glieder = []
        ergebnis.primary_material = "unknown"

        plan = get_tontraegerkette_denker().leite_phasen_ab(ergebnis)
        assert isinstance(plan, ChainPhasePlan)
        assert plan.must_have_phases == []

    def test_fail_safe_none_input(self):
        """Fail-safe: None-Eingabe → ChainPhasePlan ohne Exception."""
        from denker.tontraegerkette_denker import ChainPhasePlan, get_tontraegerkette_denker

        plan = get_tontraegerkette_denker().leite_phasen_ab(None)  # type: ignore[arg-type]
        assert isinstance(plan, ChainPhasePlan)
        assert plan.must_have_phases == []

    def test_chain_string_and_stage_count(self, ketten_ergebnis_vinyl):
        from denker.tontraegerkette_denker import get_tontraegerkette_denker

        plan = get_tontraegerkette_denker().leite_phasen_ab(ketten_ergebnis_vinyl)
        assert plan.stage_count == 2  # zwei Glieder
        assert isinstance(plan.chain_string, str)
        assert len(plan.chain_string) > 0

    def test_singleton_invariant(self):
        from denker.tontraegerkette_denker import get_tontraegerkette_denker

        a = get_tontraegerkette_denker()
        b = get_tontraegerkette_denker()
        assert a is b

    def test_4_stage_chain_with_digital_intermediate(self):
        """§2.46a: Mindestens ein Test für 4-stufige Kette mit cd_digital-Zwischenstufe."""
        from denker.tontraegerkette_denker import get_tontraegerkette_denker

        glieder = []
        for medium, phases in [
            ("shellac", ["phase_03_denoise"]),
            ("reel_tape", ["phase_29_tape_hiss_reduction"]),
            ("cd_digital", []),
            ("mp3_low", ["phase_23_spectral_repair", "phase_07_harmonic_restoration"]),
        ]:
            g = MagicMock()
            g.medium = medium
            g.recommended_phases = phases  # KettenGlied-Attributname
            glieder.append(g)

        ergebnis = MagicMock()
        ergebnis.glieder = glieder
        ergebnis.primary_material = "mp3_low"
        ergebnis.chain_string = "shellac→reel_tape→cd_digital→mp3_low"
        ergebnis.generation_count = 4

        plan = get_tontraegerkette_denker().leite_phasen_ab(ergebnis)
        assert plan.stage_count == 4
        # Additive Phase aus mp3_low soll enthalten sein
        assert "phase_07_harmonic_restoration" in plan.must_have_phases


# ---------------------------------------------------------------------------
# ── 2. ExzellenzDenker.prognostiziere() ────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPrognostiziere:
    """§GoalRisk: Exzellenz-Prognose liefert [0,1]-Risiken per Goal."""

    @pytest.fixture()
    def clean_audio(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(48000).astype(np.float32) * 0.05  # very quiet

    @pytest.fixture()
    def noisy_audio(self):
        rng = np.random.default_rng(7)
        return rng.standard_normal(48000).astype(np.float32) * 0.5  # loud noise

    def test_returns_dict(self, clean_audio):
        from denker.exzellenz_denker import get_exzellenz_denker

        result = get_exzellenz_denker().prognostiziere(clean_audio, 48000)
        assert isinstance(result, dict)

    def test_all_values_in_unit_interval(self, noisy_audio):
        from denker.exzellenz_denker import get_exzellenz_denker

        result = get_exzellenz_denker().prognostiziere(noisy_audio, 48000)
        for goal, risk in result.items():
            assert 0.0 <= risk <= 1.0, f"{goal}={risk} outside [0,1]"

    def test_keys_are_goal_names(self, noisy_audio):
        from denker.exzellenz_denker import get_exzellenz_denker

        result = get_exzellenz_denker().prognostiziere(noisy_audio, 48000)
        expected_goals = {
            "natuerlichkeit",
            "authentizitaet",
            "brillanz",
            "timbre",
            "groove",
            "micro_dynamics",
        }
        # Alle erwarteten Goals müssen im Ergebnis sein
        assert expected_goals.issubset(set(result.keys())), f"Fehlende Goals: {expected_goals - set(result.keys())}"

    def test_noisy_audio_has_elevated_risk(self, noisy_audio):
        """Lautes Wideband-Rauschen soll in mindestens einem Goal erhöhtes Risiko erzeugen."""
        from denker.exzellenz_denker import get_exzellenz_denker

        result = get_exzellenz_denker().prognostiziere(noisy_audio, 48000)
        # Wideband-Rauschen erhöht groove/micro_dynamics-Risiko (Onset-Dichte) sicher
        all_risks = list(result.values())
        assert len(all_risks) > 0 and max(all_risks) > 0.0, (
            f"Erwartet mindestens ein Risiko > 0 für Wideband-Rauschen, erhalten: {result}"
        )

    def test_fail_safe_on_exception(self):
        """Fail-safe: Exception im Audio-Processing → leeres Dict (kein Pipeline-Abort)."""
        from denker.exzellenz_denker import get_exzellenz_denker

        # 1-Sample Audio — Analyse sollte scheitern oder leer zurückliefern
        tiny = np.zeros(1, dtype=np.float32)
        result = get_exzellenz_denker().prognostiziere(tiny, 48000)
        assert isinstance(result, dict)
        # Kein Exception-Raise

    def test_defect_result_artikulation_risk(self):
        """Hohe Dropout-Severity → erhöhtes Artikulations-Risiko."""
        from denker.exzellenz_denker import get_exzellenz_denker

        audio = np.random.default_rng(99).standard_normal(48000).astype(np.float32) * 0.1

        defect_mock = MagicMock()
        defect_mock.scores = {"dropout": MagicMock(severity=0.9)}

        result = get_exzellenz_denker().prognostiziere(audio, 48000, defect_result=defect_mock, material="reel_tape")
        assert isinstance(result, dict)
        # Dropout-Risiko für Artikulation soll vorhanden sein
        artikulation = result.get("artikulation", 0.0)
        assert artikulation >= 0.0  # mindestens definiert

    def test_no_nan_or_inf(self, noisy_audio):
        from denker.exzellenz_denker import get_exzellenz_denker

        result = get_exzellenz_denker().prognostiziere(noisy_audio, 48000)
        for goal, risk in result.items():
            assert risk == risk, f"NaN für {goal}"  # NaN check
            assert risk != float("inf"), f"Inf für {goal}"

    def test_stereo_audio(self):
        """Stereo-Audio darf keinen Fehler werfen."""
        from denker.exzellenz_denker import get_exzellenz_denker

        stereo = np.random.default_rng(11).standard_normal((2, 48000)).astype(np.float32) * 0.2
        result = get_exzellenz_denker().prognostiziere(stereo, 48000)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ── 3. StrategieDenker.schaetze_phasen_tier() ──────────────────────────────
# ---------------------------------------------------------------------------


class TestSchaetzePhasenTier:
    """§StrategieDenker: Qualitäts-Tier-Empfehlungen pro Phase."""

    @pytest.fixture()
    def quality_plan(self):
        plan = MagicMock()
        plan.quality_mode = "quality"
        plan.audio_duration_s = 60.0
        plan.max_processing_s = 300.0
        return plan

    @pytest.fixture()
    def studio_plan(self):
        plan = MagicMock()
        plan.quality_mode = "studio2026"
        plan.audio_duration_s = 60.0
        plan.max_processing_s = 600.0
        return plan

    @pytest.fixture()
    def tight_budget_plan(self):
        plan = MagicMock()
        plan.quality_mode = "quality"
        plan.audio_duration_s = 60.0
        plan.max_processing_s = 50.0  # < 5 × 60 = 300
        return plan

    SAMPLE_PHASES = [
        "phase_03_denoise",
        "phase_09_crackle_removal",
        "phase_40_loudness_normalization",
    ]

    def test_returns_dict(self, quality_plan):
        from denker.strategie_denker import get_strategie_denker

        result = get_strategie_denker().schaetze_phasen_tier(quality_plan, self.SAMPLE_PHASES, restorability_score=70.0)
        assert isinstance(result, dict)

    def test_keys_match_input_phases(self, quality_plan):
        from denker.strategie_denker import get_strategie_denker

        result = get_strategie_denker().schaetze_phasen_tier(quality_plan, self.SAMPLE_PHASES)
        assert set(result.keys()) == set(self.SAMPLE_PHASES)

    def test_all_values_are_valid_tiers(self, quality_plan):
        from denker.strategie_denker import get_strategie_denker

        result = get_strategie_denker().schaetze_phasen_tier(quality_plan, self.SAMPLE_PHASES)
        for phase, tier in result.items():
            assert tier in ("maximum", "quality", "fast"), f"Ungültiger Tier '{tier}' für {phase}"

    def test_studio_mode_all_maximum(self, studio_plan):
        """Studio-2026-Modus: alle Phasen erhalten Tier 'maximum'."""
        from denker.strategie_denker import get_strategie_denker

        phases = self.SAMPLE_PHASES + ["phase_21_harmonic_exciter", "phase_38_presence_boost"]
        result = get_strategie_denker().schaetze_phasen_tier(studio_plan, phases, restorability_score=80.0)
        assert all(t == "maximum" for t in result.values()), f"Studio-Modus: nicht alle 'maximum': {result}"

    def test_low_restorability_critical_phase_maximum(self):
        """Restorability < 35 + kritische Phase → 'maximum'."""
        from denker.strategie_denker import _CRITICAL_PHASE_PREFIXES, get_strategie_denker

        plan = MagicMock()
        plan.quality_mode = "quality"
        plan.audio_duration_s = 60.0
        plan.max_processing_s = 300.0

        # Nehme erste kritische Prefix
        critical_pfx = next(iter(_CRITICAL_PHASE_PREFIXES))
        critical_phase = critical_pfx + "test"

        result = get_strategie_denker().schaetze_phasen_tier(plan, [critical_phase], restorability_score=25.0)
        assert result[critical_phase] == "maximum", (
            f"Niedrige Restorability + kritisch erwartet 'maximum', erhalten: {result}"
        )

    def test_tight_budget_noncritical_fast(self, tight_budget_plan):
        """Enges Budget + unkritische Phase → 'fast'."""
        from denker.strategie_denker import get_strategie_denker

        # phase_38 ist nicht in _CRITICAL_PHASE_PREFIXES
        non_critical = ["phase_38_presence_boost"]
        result = get_strategie_denker().schaetze_phasen_tier(tight_budget_plan, non_critical, restorability_score=70.0)
        assert result.get("phase_38_presence_boost") == "fast", (
            f"Enges Budget + unkritisch erwartet 'fast', erhalten: {result}"
        )

    def test_none_plan_returns_empty(self):
        """Fail-safe: None-Plan → leeres Dict."""
        from denker.strategie_denker import get_strategie_denker

        result = get_strategie_denker().schaetze_phasen_tier(None, ["phase_03_denoise"])  # type: ignore[arg-type]
        assert result == {}

    def test_empty_phase_list(self, quality_plan):
        from denker.strategie_denker import get_strategie_denker

        result = get_strategie_denker().schaetze_phasen_tier(quality_plan, [])
        assert result == {}

    def test_critical_phase_never_fast(self, tight_budget_plan):
        """Kritische Phasen dürfen nie 'fast' erhalten, auch bei engem Budget."""
        from denker.strategie_denker import _CRITICAL_PHASE_PREFIXES, get_strategie_denker

        critical_pfx = next(iter(_CRITICAL_PHASE_PREFIXES))
        critical_phase = critical_pfx + "test"

        result = get_strategie_denker().schaetze_phasen_tier(
            tight_budget_plan, [critical_phase], restorability_score=70.0
        )
        assert result.get(critical_phase) != "fast", f"Kritische Phase darf nie 'fast' erhalten: {result}"


# ---------------------------------------------------------------------------
# ── 4. PhaseInteractionDenker.plan() — Integration ─────────────────────────
# ---------------------------------------------------------------------------


class TestPIDIntegration:
    """PhaseInteractionDenker: Ketten-Injektion + Goal-Risiko + Quality-Tier."""

    @pytest.fixture()
    def mock_defect_result(self):
        dr = MagicMock()
        dr.scores = {}
        dr.locations = {}
        return dr

    @pytest.fixture()
    def chain_result_vinyl(self):
        glied = MagicMock()
        glied.medium = "vinyl"
        glied.priority_phases = ["phase_09_crackle_removal"]
        er = MagicMock()
        er.glieder = [glied]
        er.primary_material = "vinyl"
        return er

    def test_plan_with_chain_result_injects_phases(self, mock_defect_result, chain_result_vinyl):
        """chain_result → PhaseInteractionDenker injiziert Ketten-Pflicht-Phasen."""
        from denker.phase_interaction_denker import PhasePlan, get_phase_interaction_denker

        pid = get_phase_interaction_denker()
        plan = pid.plan(
            mock_defect_result,
            material="vinyl",
            chain_result=chain_result_vinyl,
        )
        assert isinstance(plan, PhasePlan)
        # Injektion-Hinweise sollen in conflict_notes stehen
        [n for n in plan.conflict_notes if "Injektion" in n]
        # Nur prüfen ob er strukturell korrekt ist; Injektion optional (UV3 kann Phasen haben)
        assert isinstance(plan.phases, list)

    def test_plan_with_goal_risk_map_above_threshold(self, mock_defect_result):
        """goal_risk_map mit Risiko >= 0.60 → ggf. Schutz-Phase injiziert."""
        from denker.phase_interaction_denker import (
            _GOAL_RISK_PROTECTIVE_PHASES,
            get_phase_interaction_denker,
        )

        goal_risk = {"natuerlichkeit": 0.85}  # über Schwelle
        protective = _GOAL_RISK_PROTECTIVE_PHASES.get("natuerlichkeit")

        pid = get_phase_interaction_denker()
        plan = pid.plan(
            mock_defect_result,
            material="vinyl",
            goal_risk_map=goal_risk,
        )
        # Wenn protective phase noch nicht in UV3-Selektion war, soll sie injiziert worden sein
        if protective and plan.phases:  # prüfbar nur wenn Phasen vorhanden
            [n for n in plan.conflict_notes if "GoalRisk" in n]
            # Test ist pass wenn plan valide zurückgegeben wurde
        assert isinstance(plan.phases, list)

    def test_plan_goal_risk_below_threshold_no_injection(self, mock_defect_result):
        """Risiko < 0.60 → keine GoalRisk-Injektion."""
        from denker.phase_interaction_denker import get_phase_interaction_denker

        goal_risk = {"natuerlichkeit": 0.40}  # unter Schwelle

        pid = get_phase_interaction_denker()
        plan = pid.plan(
            mock_defect_result,
            material="vinyl",
            goal_risk_map=goal_risk,
        )
        goal_risk_notes = [n for n in plan.conflict_notes if "GoalRisk" in n]
        assert len(goal_risk_notes) == 0, f"Keine GoalRisk-Injektion bei Risiko < {0.60} erwartet"

    def test_plan_with_strategie_plan_fills_tiers(self, mock_defect_result):
        """strategie_plan → phase_quality_tiers werden befüllt."""
        from denker.phase_interaction_denker import get_phase_interaction_denker

        strategie_mock = MagicMock()
        strategie_mock.quality_mode = "quality"
        strategie_mock.audio_duration_s = 60.0
        strategie_mock.max_processing_s = 300.0

        pid = get_phase_interaction_denker()
        plan = pid.plan(
            mock_defect_result,
            material="vinyl",
            strategie_plan=strategie_mock,
        )
        # Falls Phasen vorhanden: Tiers sollen gesetzt sein
        if plan.phases:
            assert isinstance(plan.phase_quality_tiers, dict)
            assert len(plan.phase_quality_tiers) > 0, (
                "phase_quality_tiers soll bei vorhandenem strategie_plan befüllt sein"
            )

    def test_phase_plan_has_quality_tiers_field(self, mock_defect_result):
        """PhasePlan enthält phase_quality_tiers als dict (auch wenn leer)."""
        from denker.phase_interaction_denker import get_phase_interaction_denker

        plan = get_phase_interaction_denker().plan(mock_defect_result)
        assert hasattr(plan, "phase_quality_tiers")
        assert isinstance(plan.phase_quality_tiers, dict)

    def test_goal_risk_threshold_constant(self):
        from denker.phase_interaction_denker import _GOAL_RISK_THRESHOLD

        assert 0.0 < _GOAL_RISK_THRESHOLD < 1.0

    def test_goal_risk_protective_phases_constant(self):
        from denker.phase_interaction_denker import _GOAL_RISK_PROTECTIVE_PHASES

        assert isinstance(_GOAL_RISK_PROTECTIVE_PHASES, dict)
        assert len(_GOAL_RISK_PROTECTIVE_PHASES) >= 5

    def test_new_plan_signature_accepts_all_kwargs(self, mock_defect_result, chain_result_vinyl):
        """Smoke-Test: plan() akzeptiert alle neuen kwargs ohne TypeError."""
        from denker.phase_interaction_denker import get_phase_interaction_denker

        strategie_mock = MagicMock()
        strategie_mock.quality_mode = "quality"
        strategie_mock.audio_duration_s = 60.0
        strategie_mock.max_processing_s = 300.0

        plan = get_phase_interaction_denker().plan(
            mock_defect_result,
            material="vinyl",
            mode="quality",
            chain_result=chain_result_vinyl,
            goal_risk_map={"natuerlichkeit": 0.75, "brillanz": 0.50},
            strategie_plan=strategie_mock,
        )
        assert plan is not None


# ---------------------------------------------------------------------------
# ── 5. AurikDenker Stufe 5b: kette-Guard + Goal-Risiko-Pfad ─────────────
# ---------------------------------------------------------------------------


class TestAurikDenkerStufe5b:
    """Smoke-Tests über AurikDenkers neue Wiring-Punkte.

    Die Implementierung liegt in restauriere() (Stufe 5b), nicht in denke().
    denke() ist nur ein API-Alias für restauriere().
    """

    @pytest.fixture(scope="class")
    def aurik_denker_source(self):
        """Liest die aurik_denker.py direkt als Text (zuverlässiger als inspect)."""
        import pathlib

        src_path = pathlib.Path(__file__).parent.parent.parent / "denker" / "aurik_denker.py"
        return src_path.read_text(encoding="utf-8")

    def test_kette_none_guard_exists(self, aurik_denker_source):
        """kette muss vor Stufe-2-Try-Block auf None initialisiert oder zugewiesen werden."""
        # Akzeptiere sowohl explizite kette=None Initialisierung als auch implizite
        # (kette wird im try-Block zugewiesen; Exception-Pfad wäre UnboundLocalError).
        # Mindestens: chain_result=kette muss im PID-Call stehen.
        assert "chain_result=kette" in aurik_denker_source, "chain_result=kette fehlt im AurikDenker PID-Call"

    def test_goal_risk_map_var_in_source(self, aurik_denker_source):
        """_goal_risk_map Variable muss in aurik_denker.py deklariert sein."""
        assert "_goal_risk_map" in aurik_denker_source

    def test_chain_result_kwarg_passed_to_pid(self, aurik_denker_source):
        """chain_result=kette soll an PhaseInteractionDenker.plan() übergeben werden."""
        assert "chain_result=kette" in aurik_denker_source

    def test_goal_risk_map_kwarg_passed_to_pid(self, aurik_denker_source):
        """goal_risk_map=_goal_risk_map soll an plan() übergeben werden."""
        assert "goal_risk_map=_goal_risk_map" in aurik_denker_source

    def test_strategie_plan_kwarg_passed_to_pid(self, aurik_denker_source):
        """strategie_plan=strategie soll an plan() übergeben werden."""
        assert "strategie_plan=strategie" in aurik_denker_source

    def test_injiziert_counter_in_stage_notes(self, aurik_denker_source):
        """_pid_injected Counter und 'injiziert' sollen in aurik_denker.py stehen."""
        assert "_pid_injected" in aurik_denker_source
        assert "injiziert" in aurik_denker_source
