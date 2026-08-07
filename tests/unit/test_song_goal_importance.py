from __future__ import annotations

"""
Unit tests for §2.56 Song-Goal-Importance (backend/core/song_goal_importance.py).

Tests cover:
  - Default/neutral weights
  - Genre profiles produce differentiated weights
  - Era modifiers shift weights multiplicatively
  - Material modifiers shift weights
  - Vocal detection boosts artikulation/emotionalitaet
  - Restorability adjustment for degraded material
  - Studio 2026 mode boost
  - P1/P2 floor enforcement (min 0.70)
  - Global bounds [0.3, 2.0]
  - Unknown genre/era/material → neutral fallback
  - PMGG weighted regression semantics
  - CIG weighted drift semantics
  - GoalPriorityProtocol weighted conflict resolution
  - GoalPriorityProtocol weighted abort
"""


import pytest

from backend.core.song_goal_importance import (
    _P1P2_GOALS,
    _P1P2_WEIGHT_FLOOR,
    _WEIGHT_MAX,
    _WEIGHT_MIN,
    ALL_GOAL_NAMES,
    estimate_goal_importance,
    get_default_importance,
)

# ── Default / Neutral ──────────────────────────────────────────────────


@pytest.mark.unit
class TestDefaultImportance:
    def test_default_all_goals_present(self):
        d = get_default_importance()
        for g in ALL_GOAL_NAMES:
            assert g in d.weights, f"Missing goal: {g}"

    def test_default_all_weights_one(self):
        d = get_default_importance()
        for g, w in d.weights.items():
            assert w == 1.0, f"{g} weight should be 1.0, got {w}"

    def test_default_reason(self):
        d = get_default_importance()
        assert "default" in d.reason.lower()

    def test_weight_of_unknown_returns_one(self):
        d = get_default_importance()
        assert d.weight_of("nonexistent_goal") == 1.0


# ── estimate_goal_importance: Unknown inputs ────────────────────────


class TestUnknownInputs:
    def test_empty_genre(self):
        imp = estimate_goal_importance(genre_label="")
        for g in ALL_GOAL_NAMES:
            assert _WEIGHT_MIN <= imp.weights[g] <= _WEIGHT_MAX

    def test_unknown_material(self):
        imp = estimate_goal_importance(material_type="unknown_xyz")
        for g in ALL_GOAL_NAMES:
            assert _WEIGHT_MIN <= imp.weights[g] <= _WEIGHT_MAX

    def test_no_era(self):
        imp = estimate_goal_importance(era_decade="")
        for g in ALL_GOAL_NAMES:
            assert _WEIGHT_MIN <= imp.weights[g] <= _WEIGHT_MAX


# ── Genre profiles ──────────────────────────────────────────────────


class TestGenreProfiles:
    def test_jazz_groove_high(self):
        imp = estimate_goal_importance(genre_label="jazz")
        assert imp.weights["groove"] > 1.3

    def test_jazz_brillanz_low(self):
        imp = estimate_goal_importance(genre_label="jazz")
        assert imp.weights["brillanz"] < 1.0

    def test_metal_bass_kraft_high(self):
        imp = estimate_goal_importance(genre_label="metal")
        assert imp.weights["bass_kraft"] > 1.3

    def test_metal_micro_dynamics_low(self):
        imp = estimate_goal_importance(genre_label="metal")
        assert imp.weights["micro_dynamics"] < 1.0

    def test_schlager_waerme_high(self):
        imp = estimate_goal_importance(genre_label="schlager")
        assert imp.weights["waerme"] > 1.2

    def test_klassik_spatial_depth_high(self):
        imp = estimate_goal_importance(genre_label="klassik")
        assert imp.weights["spatial_depth"] > 1.4

    def test_electronic_bass_kraft_high(self):
        imp = estimate_goal_importance(genre_label="electronic")
        assert imp.weights["bass_kraft"] > 1.4

    def test_hip_hop_artikulation_high(self):
        imp = estimate_goal_importance(genre_label="hip-hop")
        assert imp.weights["artikulation"] > 1.2

    def test_reggae_groove_highest(self):
        imp = estimate_goal_importance(genre_label="reggae")
        assert imp.weights["groove"] > 1.5

    def test_soul_emotionalitaet_high(self):
        imp = estimate_goal_importance(genre_label="soul/r&b")
        assert imp.weights["emotionalitaet"] > 1.3

    def test_soul_alias_soul(self):
        imp = estimate_goal_importance(genre_label="soul")
        assert imp.weights["emotionalitaet"] > 1.3

    def test_folk_artikulation_high(self):
        imp = estimate_goal_importance(genre_label="folk")
        assert imp.weights["artikulation"] > 1.3

    def test_gospel_spatial_depth_high(self):
        imp = estimate_goal_importance(genre_label="gospel")
        assert imp.weights["spatial_depth"] > 1.4

    def test_different_genres_produce_different_weights(self):
        jazz = estimate_goal_importance(genre_label="jazz")
        metal = estimate_goal_importance(genre_label="metal")
        assert jazz.weights["waerme"] != metal.weights["waerme"]
        assert jazz.weights["groove"] != metal.weights["groove"]


# ── Era modifiers ───────────────────────────────────────────────────


class TestEraModifiers:
    def test_1920er_brillanz_reduced(self):
        imp = estimate_goal_importance(genre_label="pop", era_decade="1920er")
        neutral = estimate_goal_importance(genre_label="pop")
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_1920er_spatial_depth_reduced(self):
        imp = estimate_goal_importance(genre_label="pop", era_decade="1920er")
        neutral = estimate_goal_importance(genre_label="pop")
        assert imp.weights["spatial_depth"] < neutral.weights["spatial_depth"]

    def test_1970er_waerme_boosted(self):
        imp = estimate_goal_importance(genre_label="pop", era_decade="1970er")
        neutral = estimate_goal_importance(genre_label="pop")
        assert imp.weights["waerme"] > neutral.weights["waerme"]

    def test_modern_era_no_change(self):
        imp_90 = estimate_goal_importance(genre_label="pop", era_decade="1990er")
        imp_no = estimate_goal_importance(genre_label="pop")
        for g in ALL_GOAL_NAMES:
            assert abs(imp_90.weights[g] - imp_no.weights[g]) < 0.001


# ── Material modifiers ──────────────────────────────────────────────


class TestMaterialModifiers:
    def test_vinyl_waerme_boosted(self):
        imp = estimate_goal_importance(material_type="vinyl")
        assert imp.weights["waerme"] > 1.0

    def test_shellac_brillanz_reduced(self):
        imp = estimate_goal_importance(material_type="shellac")
        assert imp.weights["brillanz"] < 1.0

    def test_mp3_low_transparenz_boosted(self):
        imp = estimate_goal_importance(material_type="mp3_low")
        assert imp.weights["transparenz"] > 1.0

    def test_cd_digital_neutral(self):
        # §v10.15: estimate_goal_importance() always applies a non-neutral
        # "restoration_mode" baseline (unless is_studio_2026=True) — comparing
        # against an absolute 1.0 is wrong. cd_digital's _MATERIAL_WEIGHT_
        # MODIFIERS entry is genuinely empty {} ("no material modifier"), so the
        # correct comparison is against the SAME call without material_type
        # (both get the identical restoration_mode baseline, differing only in
        # the material dimension — which is absent for both).
        imp = estimate_goal_importance(material_type="cd_digital")
        baseline = estimate_goal_importance()
        for g in ALL_GOAL_NAMES:
            assert abs(imp.weights[g] - baseline.weights[g]) < 1e-9


# ── Vocal detection ─────────────────────────────────────────────────


class TestVocalDetection:
    def test_vocal_boosts_artikulation(self):
        imp_vocal = estimate_goal_importance(vocal_detected=True, vocal_confidence=0.8)
        imp_none = estimate_goal_importance(vocal_detected=False)
        assert imp_vocal.weights["artikulation"] > imp_none.weights["artikulation"]

    def test_vocal_boosts_emotionalitaet(self):
        imp_vocal = estimate_goal_importance(vocal_detected=True, vocal_confidence=0.8)
        imp_none = estimate_goal_importance(vocal_detected=False)
        assert imp_vocal.weights["emotionalitaet"] > imp_none.weights["emotionalitaet"]

    def test_low_confidence_vocal_small_effect(self):
        imp_low = estimate_goal_importance(vocal_detected=True, vocal_confidence=0.3)
        imp_high = estimate_goal_importance(vocal_detected=True, vocal_confidence=0.9)
        assert imp_high.weights["artikulation"] > imp_low.weights["artikulation"]

    def test_vocal_false_no_effect(self):
        imp = estimate_goal_importance(vocal_detected=False, vocal_confidence=0.9)
        neutral = estimate_goal_importance()
        assert abs(imp.weights["artikulation"] - neutral.weights["artikulation"]) < 0.001


# ── Restorability adjustment ────────────────────────────────────────


class TestRestorability:
    def test_low_restorability_reduces_brillanz(self):
        imp_low = estimate_goal_importance(restorability_score=10.0)
        imp_high = estimate_goal_importance(restorability_score=80.0)
        assert imp_low.weights["brillanz"] < imp_high.weights["brillanz"]

    def test_low_restorability_boosts_natuerlichkeit(self):
        imp_low = estimate_goal_importance(restorability_score=10.0)
        imp_high = estimate_goal_importance(restorability_score=80.0)
        assert imp_low.weights["natuerlichkeit"] > imp_high.weights["natuerlichkeit"]

    def test_high_restorability_no_adjustment(self):
        imp = estimate_goal_importance(restorability_score=80.0)
        neutral = estimate_goal_importance(restorability_score=50.0)
        # Both above 30 → no degradation adjustment
        for g in ALL_GOAL_NAMES:
            assert abs(imp.weights[g] - neutral.weights[g]) < 0.001


# ── Studio 2026 mode ────────────────────────────────────────────────


class TestStudio2026:
    def test_studio_boosts_brillanz(self):
        imp_st = estimate_goal_importance(is_studio_2026=True)
        imp_rest = estimate_goal_importance(is_studio_2026=False)
        assert imp_st.weights["brillanz"] > imp_rest.weights["brillanz"]

    def test_studio_boosts_transparenz(self):
        imp_st = estimate_goal_importance(is_studio_2026=True)
        imp_rest = estimate_goal_importance(is_studio_2026=False)
        assert imp_st.weights["transparenz"] > imp_rest.weights["transparenz"]


# ── Bounds enforcement ──────────────────────────────────────────────


class TestBoundsEnforcement:
    def test_p1p2_floor(self):
        """P1/P2 goals must never go below 0.70 even with aggressive modifiers."""
        imp = estimate_goal_importance(
            genre_label="metal",
            era_decade="1900er",
            material_type="wax_cylinder",
            restorability_score=5.0,
        )
        for g in _P1P2_GOALS:
            assert imp.weights[g] >= _P1P2_WEIGHT_FLOOR, (
                f"P1/P2 goal {g} weight {imp.weights[g]:.3f} < floor {_P1P2_WEIGHT_FLOOR}"
            )

    def test_global_min(self):
        """No weight below 0.3."""
        imp = estimate_goal_importance(
            genre_label="electronic",
            era_decade="1900er",
            material_type="wax_cylinder",
        )
        for g in ALL_GOAL_NAMES:
            assert imp.weights[g] >= _WEIGHT_MIN, f"Goal {g} weight {imp.weights[g]:.3f} < min {_WEIGHT_MIN}"

    def test_global_max(self):
        """No weight above 2.0."""
        imp = estimate_goal_importance(
            genre_label="reggae",
            era_decade="1970er",
            material_type="vinyl",
            vocal_detected=True,
            vocal_confidence=1.0,
            is_studio_2026=True,
        )
        for g in ALL_GOAL_NAMES:
            assert imp.weights[g] <= _WEIGHT_MAX, f"Goal {g} weight {imp.weights[g]:.3f} > max {_WEIGHT_MAX}"


# ── Combined scenario ──────────────────────────────────────────────


class TestCombinedScenario:
    def test_schlager_1970er_vinyl_scenario(self):
        """70er Vinyl Schlager: waerme HIGH, groove moderate, brillanz LOW."""
        imp = estimate_goal_importance(
            genre_label="schlager",
            era_decade="1970er",
            material_type="vinyl",
            vocal_detected=True,
            vocal_confidence=0.85,
            restorability_score=64.0,
        )
        assert imp.weights["waerme"] > 1.3  # Genre + Era + Material
        assert imp.weights["groove"] > 1.0  # Schlager is rhythmic
        assert imp.weights["brillanz"] < 1.0  # Era + genre
        assert imp.genre_profile == "schlager"
        assert imp.era_profile == "1970er"
        assert imp.material_profile == "vinyl"
        assert imp.vocal_detected is True

    def test_jazz_1950er_shellac(self):
        """50er Shellac Jazz: groove essential, spatial moderate, HF limited."""
        imp = estimate_goal_importance(
            genre_label="jazz",
            era_decade="1950er",
            material_type="shellac",
        )
        assert imp.weights["groove"] > 1.3
        assert imp.weights["brillanz"] < 0.8  # Era + material + genre
        assert imp.weights["waerme"] > 1.1


# ── Serialisation ───────────────────────────────────────────────────


class TestSerialisation:
    def test_as_dict_contains_all_fields(self):
        imp = estimate_goal_importance(genre_label="jazz")
        d = imp.as_dict()
        assert "weights" in d
        assert "reason" in d
        assert "genre_profile" in d
        assert len(d["weights"]) == 15

    def test_weights_method_returns_dict(self):
        imp = estimate_goal_importance()
        # SongGoalImportance.weights is a dict attribute
        assert isinstance(imp.weights, dict)
        assert len(imp.weights) == 15


# ── PMGG weighted regression integration ────────────────────────────


class TestPMGGWeightedRegression:
    def test_max_regression_weighted(self):
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        before = {"waerme": 0.80, "brillanz": 0.85}
        after = {"waerme": 0.76, "brillanz": 0.81}
        # Regression: waerme 0.04, brillanz 0.04

        # Uniform: max= 0.04
        reg_uniform = PerPhaseMusicalGoalsGate._max_regression(before, after, ["waerme", "brillanz"])
        assert abs(reg_uniform - 0.04) < 0.001

        # Weighted: waerme×2.0, brillanz×0.5
        weights = {"waerme": 2.0, "brillanz": 0.5}
        reg_weighted = PerPhaseMusicalGoalsGate._max_regression(
            before, after, ["waerme", "brillanz"], goal_weights=weights
        )
        # waerme: 0.04*2.0=0.08, brillanz: 0.04*0.5=0.02 → max=0.08
        assert abs(reg_weighted - 0.08) < 0.001

    def test_max_regression_no_weights_is_uniform(self):
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        before = {"waerme": 0.80}
        after = {"waerme": 0.76}
        reg1 = PerPhaseMusicalGoalsGate._max_regression(before, after, ["waerme"])
        reg2 = PerPhaseMusicalGoalsGate._max_regression(before, after, ["waerme"], goal_weights=None)
        assert abs(reg1 - reg2) < 0.001


# ── GoalPriorityProtocol weighted ───────────────────────────────────


class TestGPPWeighted:
    def test_resolve_conflict_equal_priority_weight_wins(self):
        from backend.core.goal_priority_protocol import GoalPriorityProtocol

        gpp = GoalPriorityProtocol()
        # transparenz and waerme are both P4
        weights = {"transparenz": 1.8, "waerme": 0.6}
        result = gpp.resolve_conflict(
            "transparenz",
            "waerme",
            0.05,
            0.05,
            goal_weights=weights,
        )
        assert result.winner == "transparenz"

    def test_resolve_conflict_priority_overrides_weight(self):
        from backend.core.goal_priority_protocol import GoalPriorityProtocol

        gpp = GoalPriorityProtocol()
        # natuerlichkeit (P1) vs. waerme (P4) — priority wins even if waerme has weight 2.0
        weights = {"natuerlichkeit": 0.7, "waerme": 2.0}
        result = gpp.resolve_conflict(
            "natuerlichkeit",
            "waerme",
            0.05,
            0.05,
            goal_weights=weights,
        )
        assert result.winner == "natuerlichkeit"

    def test_should_abort_weighted_stricter(self):
        from backend.core.goal_priority_protocol import GoalPriorityProtocol

        gpp = GoalPriorityProtocol()
        before = {"natuerlichkeit": 0.92}
        after = {"natuerlichkeit": 0.907}  # delta = 0.013

        # Without weights: 0.013 > epsilon(0.012) → abort
        res_no = gpp.should_abort_iteration(before, after)
        assert res_no.should_abort is True

        # With weight 2.0: effective_epsilon = 0.012/2.0 = 0.006 → 0.013 > 0.006 → abort
        res_w = gpp.should_abort_iteration(before, after, goal_weights={"natuerlichkeit": 2.0})
        assert res_w.should_abort is True

    def test_should_abort_weighted_more_lenient(self):
        from backend.core.goal_priority_protocol import GoalPriorityProtocol

        gpp = GoalPriorityProtocol()
        before = {"natuerlichkeit": 0.92}
        after = {"natuerlichkeit": 0.907}  # delta = 0.013

        # With weight 0.7: effective_epsilon = 0.012/0.7 ≈ 0.0171 → 0.013 < 0.0171 → no abort
        res = gpp.should_abort_iteration(before, after, goal_weights={"natuerlichkeit": 0.7})
        assert res.should_abort is False


# ── CIG weighted drift ──────────────────────────────────────────────


class TestCIGWeightedDrift:
    def test_interaction_guard_state_has_goal_weights(self):
        from backend.core.cumulative_interaction_guard import InteractionGuardState

        state = InteractionGuardState()
        assert hasattr(state, "goal_weights")
        assert state.goal_weights is None


# ── Audio-derived per-song fine-tuning ──────────────────────────────


class TestAudioSNR:
    def test_low_snr_reduces_brillanz(self):
        imp = estimate_goal_importance(snr_db=10.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_low_snr_boosts_transparenz(self):
        imp = estimate_goal_importance(snr_db=10.0)
        neutral = estimate_goal_importance()
        assert imp.weights["transparenz"] > neutral.weights["transparenz"]

    def test_high_snr_boosts_natuerlichkeit(self):
        imp = estimate_goal_importance(snr_db=45.0)
        neutral = estimate_goal_importance()
        assert imp.weights["natuerlichkeit"] > neutral.weights["natuerlichkeit"]

    def test_mid_snr_no_change(self):
        # §2.59: kontinuierliche SNR-Kurve statt hartem 15dB-Cutoff — die
        # Transparenz/Brillanz-Anpassung greift für JEDES snr_db < 30.0,
        # die Natuerlichkeit/Authentizitaet-Anpassung für JEDES snr_db > 35.0.
        # 25.0 dB ("mid" unter der alten 15dB-Cutoff-Annahme) liegt unter der
        # aktuellen 30dB-Schwelle und LÖST die Transparenz-Anpassung AUS.
        # Ein echter "no change"-Wert muss im neutralen Korridor [30, 35] liegen.
        imp = estimate_goal_importance(snr_db=32.0)
        neutral = estimate_goal_importance()
        for g in ALL_GOAL_NAMES:
            assert abs(imp.weights[g] - neutral.weights[g]) < 0.001


class TestAudioBandwidth:
    def test_narrow_bandwidth_reduces_brillanz(self):
        imp = estimate_goal_importance(effective_bandwidth_hz=4000.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_full_bandwidth_boosts_brillanz(self):
        imp = estimate_goal_importance(effective_bandwidth_hz=20000.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] > neutral.weights["brillanz"]


class TestAudioDynamics:
    def test_compressed_reduces_micro_dynamics(self):
        imp = estimate_goal_importance(dynamic_range_db=12.0)
        neutral = estimate_goal_importance()
        assert imp.weights["micro_dynamics"] < neutral.weights["micro_dynamics"]

    def test_wide_dynamics_boosts_micro_dynamics(self):
        imp = estimate_goal_importance(dynamic_range_db=60.0)
        neutral = estimate_goal_importance()
        assert imp.weights["micro_dynamics"] > neutral.weights["micro_dynamics"]


class TestAudioStereo:
    def test_poor_mono_compat_boosts_spatial(self):
        imp = estimate_goal_importance(stereo_mono_compat=0.2)
        neutral = estimate_goal_importance()
        assert imp.weights["spatial_depth"] > neutral.weights["spatial_depth"]

    def test_good_mono_compat_reduces_spatial(self):
        imp = estimate_goal_importance(stereo_mono_compat=0.95)
        neutral = estimate_goal_importance()
        assert imp.weights["spatial_depth"] < neutral.weights["spatial_depth"]


class TestAudioBPM:
    def test_fast_bpm_boosts_groove(self):
        imp = estimate_goal_importance(bpm=130.0)
        neutral = estimate_goal_importance()
        assert imp.weights["groove"] > neutral.weights["groove"]

    def test_slow_bpm_reduces_groove(self):
        imp = estimate_goal_importance(bpm=45.0)
        neutral = estimate_goal_importance()
        assert imp.weights["groove"] < neutral.weights["groove"]


class TestAudioDefects:
    def test_noise_defect_boosts_transparenz(self):
        # §2.59: defect_severities-Keys sind an DefectType.name.lower() gebunden.
        # "broadband_noise" existiert nicht als DefectType — kanonischer Name
        # fuer Rausch-Defekte in der _noise_sev-Pruefung ist "high_freq_noise".
        imp = estimate_goal_importance(defect_severities={"high_freq_noise": 0.8})
        neutral = estimate_goal_importance()
        assert imp.weights["transparenz"] > neutral.weights["transparenz"]

    def test_crackle_defect_boosts_groove(self):
        imp = estimate_goal_importance(defect_severities={"crackle": 0.7})
        neutral = estimate_goal_importance()
        assert imp.weights["groove"] > neutral.weights["groove"]

    def test_hf_loss_defect_reduces_brillanz(self):
        # §2.59: kanonischer Name ist "bandwidth_loss" (DefectType.BANDWIDTH_LOSS),
        # nicht "hf_loss" — letzteres ist kein DefectType-Wert.
        imp = estimate_goal_importance(defect_severities={"bandwidth_loss": 0.8})
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_wow_defect_boosts_tonal_center(self):
        imp = estimate_goal_importance(defect_severities={"wow": 0.5})
        neutral = estimate_goal_importance()
        assert imp.weights["tonal_center"] > neutral.weights["tonal_center"]


class TestAudioSpectralTilt:
    def test_dark_signal_reduces_brillanz(self):
        imp = estimate_goal_importance(spectral_tilt_db_per_oct=-6.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_bright_signal_boosts_brillanz(self):
        imp = estimate_goal_importance(spectral_tilt_db_per_oct=0.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] > neutral.weights["brillanz"]


class TestAudioCombinedWithLabels:
    def test_schlager_70er_vinyl_with_audio_features(self):
        """Full per-song analysis: labels + audio features."""
        imp = estimate_goal_importance(
            genre_label="schlager",
            era_decade="1970er",
            material_type="vinyl",
            vocal_detected=True,
            vocal_confidence=0.85,
            restorability_score=64.0,
            snr_db=14.3,
            effective_bandwidth_hz=12000.0,
            bpm=120.0,
            # §2.59: "broadband_noise" ist kein DefectType-Wert — kanonisch ist
            # "high_freq_noise" (siehe _noise_sev in song_goal_importance.py).
            defect_severities={"high_freq_noise": 0.4, "crackle": 0.6},
        )
        # Waerme still prioritised (genre+era+material) but soft-capped
        assert imp.weights["waerme"] > 1.2
        assert imp.weights["waerme"] < 1.8
        # All within bounds
        for g in ALL_GOAL_NAMES:
            assert _WEIGHT_MIN <= imp.weights[g] <= _WEIGHT_MAX
        # Reason should mention audio features — actual format is
        # "snr(14dB→transparenz×1.23)", not the literal substring "snr_low".
        assert "snr(" in imp.reason
        assert "schlager" in imp.reason.lower() or "genre" in imp.reason.lower()


# ── Carrier-chain degradation (§2.46/§2.46a) ───────────────────────


class TestCarrierChainDegradation:
    def test_single_gen_no_change(self):
        """1-generation chain (e.g. CD rip) has no carrier-chain adjustment."""
        imp = estimate_goal_importance(transfer_generation_count=1)
        neutral = estimate_goal_importance()
        for g in ALL_GOAL_NAMES:
            assert abs(imp.weights[g] - neutral.weights[g]) < 0.001

    def test_2gen_reduces_brillanz(self):
        imp = estimate_goal_importance(transfer_generation_count=2, source_fidelity_confidence=0.9)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_2gen_boosts_natuerlichkeit(self):
        imp = estimate_goal_importance(transfer_generation_count=2, source_fidelity_confidence=0.9)
        neutral = estimate_goal_importance()
        assert imp.weights["natuerlichkeit"] > neutral.weights["natuerlichkeit"]

    def test_4gen_stronger_than_2gen(self):
        """Deeper chain = larger adjustment."""
        imp2 = estimate_goal_importance(transfer_generation_count=2, source_fidelity_confidence=0.9)
        imp4 = estimate_goal_importance(transfer_generation_count=4, source_fidelity_confidence=0.9)
        assert imp4.weights["brillanz"] < imp2.weights["brillanz"]
        assert imp4.weights["natuerlichkeit"] > imp2.weights["natuerlichkeit"]

    def test_low_confidence_attenuates_effect(self):
        """With low confidence the carrier chain model is less trusted."""
        high = estimate_goal_importance(transfer_generation_count=4, source_fidelity_confidence=0.95)
        low = estimate_goal_importance(transfer_generation_count=4, source_fidelity_confidence=0.35)
        neutral = estimate_goal_importance()
        # Both below neutral, but low-confidence closer to neutral
        assert low.weights["brillanz"] > high.weights["brillanz"]
        assert low.weights["brillanz"] < neutral.weights["brillanz"]

    def test_deep_chain_boosts_tonal_center(self):
        imp = estimate_goal_importance(transfer_generation_count=4, source_fidelity_confidence=0.8)
        neutral = estimate_goal_importance()
        assert imp.weights["tonal_center"] > neutral.weights["tonal_center"]

    def test_hf_loss_reduces_brillanz(self):
        imp = estimate_goal_importance(cumulative_hf_loss_db=15.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_small_hf_loss_no_impact(self):
        """< 6 dB cumulative HF loss = below threshold, no adjustment."""
        imp = estimate_goal_importance(cumulative_hf_loss_db=4.0)
        neutral = estimate_goal_importance()
        for g in ALL_GOAL_NAMES:
            assert abs(imp.weights[g] - neutral.weights[g]) < 0.001

    def test_hf_loss_boosts_waerme(self):
        """HF loss means focus on warmth instead."""
        imp = estimate_goal_importance(cumulative_hf_loss_db=18.0)
        neutral = estimate_goal_importance()
        assert imp.weights["waerme"] > neutral.weights["waerme"]

    def test_reason_contains_chain_gen(self):
        imp = estimate_goal_importance(transfer_generation_count=3, source_fidelity_confidence=0.8)
        assert "chain_gen=3" in imp.reason

    def test_reason_contains_chain_hf_loss(self):
        imp = estimate_goal_importance(cumulative_hf_loss_db=15.0)
        assert "chain_hf_loss" in imp.reason

    def test_shellac_4gen_full_scenario(self):
        """Real scenario: shellac→tape→cassette→MP3 (4 generations).

        The original 1940s studio sound has been severely degraded across
        the carrier chain.  Goal weights must:
        - Strongly reduce brillanz (physically impossible to restore)
        - Boost timbre/authenticity (preserve what remains)
        - Keep spatial_depth expectations low
        """
        imp = estimate_goal_importance(
            genre_label="jazz",
            era_decade="1940er",
            material_type="shellac",
            restorability_score=22.0,
            snr_db=10.0,
            effective_bandwidth_hz=4500.0,
            transfer_generation_count=4,
            cumulative_hf_loss_db=20.0,
            source_fidelity_confidence=0.85,
            defect_severities={"broadband_noise": 0.7, "crackle": 0.8},
        )
        # brillanz should be very low — multiple factors push it down
        assert imp.weights["brillanz"] < 0.55
        # Core fidelity should be boosted
        assert imp.weights["natuerlichkeit"] > 1.1
        assert imp.weights["authentizitaet"] > 1.1
        assert imp.weights["timbre_authentizitaet"] > 1.0
        # All within bounds
        for g in ALL_GOAL_NAMES:
            assert _WEIGHT_MIN <= imp.weights[g] <= _WEIGHT_MAX
        # Reason tracks the carrier chain
        assert "chain_gen=4" in imp.reason


# ── Psychoacoustic features (Stage 3) ──────────────────────────────


class TestPsychoacousticRoughness:
    def test_high_roughness_boosts_artikulation(self):
        """High Zwicker roughness signals harsh audio → protect artikulation."""
        imp = estimate_goal_importance(roughness=0.6)
        neutral = estimate_goal_importance()
        assert imp.weights["artikulation"] > neutral.weights["artikulation"]

    def test_high_roughness_boosts_transparenz(self):
        imp = estimate_goal_importance(roughness=0.5)
        neutral = estimate_goal_importance()
        assert imp.weights["transparenz"] > neutral.weights["transparenz"]

    def test_high_roughness_boosts_natuerlichkeit(self):
        imp = estimate_goal_importance(roughness=0.6)
        neutral = estimate_goal_importance()
        assert imp.weights["natuerlichkeit"] > neutral.weights["natuerlichkeit"]

    def test_low_roughness_no_change(self):
        """Low roughness (smooth audio) should not significantly alter weights."""
        imp = estimate_goal_importance(roughness=0.05)
        neutral = estimate_goal_importance()
        assert abs(imp.weights["artikulation"] - neutral.weights["artikulation"]) < 0.08

    def test_reason_mentions_roughness(self):
        imp = estimate_goal_importance(roughness=0.6)
        assert "rough_high" in imp.reason


class TestPsychoacousticSharpness:
    def test_high_sharpness_boosts_brillanz(self):
        """High Aures sharpness means real HF emphasis → protect brillanz."""
        imp = estimate_goal_importance(sharpness=0.7)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] > neutral.weights["brillanz"]

    def test_low_sharpness_reduces_brillanz(self):
        """Perceptually dull signal → brillanz expectation lowered."""
        imp = estimate_goal_importance(sharpness=0.1)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_low_sharpness_boosts_waerme(self):
        imp = estimate_goal_importance(sharpness=0.1)
        neutral = estimate_goal_importance()
        assert imp.weights["waerme"] > neutral.weights["waerme"]

    def test_reason_mentions_sharpness(self):
        imp = estimate_goal_importance(sharpness=0.7)
        assert "sharp_high" in imp.reason


class TestPsychoacousticSpectralFlatness:
    def test_noisy_signal_boosts_transparenz(self):
        """High spectral flatness (noise-like) → transparenz critical."""
        imp = estimate_goal_importance(spectral_flatness=0.7)
        neutral = estimate_goal_importance()
        assert imp.weights["transparenz"] > neutral.weights["transparenz"]

    def test_noisy_signal_reduces_brillanz(self):
        imp = estimate_goal_importance(spectral_flatness=0.6)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_tonal_signal_boosts_tonal_center(self):
        """Low spectral flatness (tonal) → tonal_center goals defining."""
        imp = estimate_goal_importance(spectral_flatness=0.05)
        neutral = estimate_goal_importance()
        assert imp.weights["tonal_center"] > neutral.weights["tonal_center"]

    def test_reason_mentions_flatness(self):
        imp = estimate_goal_importance(spectral_flatness=0.7)
        assert "flat_noisy" in imp.reason


class TestPsychoacousticTonality:
    def test_strong_tonality_boosts_tonal_center(self):
        imp = estimate_goal_importance(tonality=0.8)
        neutral = estimate_goal_importance()
        assert imp.weights["tonal_center"] > neutral.weights["tonal_center"]

    def test_strong_tonality_boosts_timbre(self):
        imp = estimate_goal_importance(tonality=0.8)
        neutral = estimate_goal_importance()
        assert imp.weights["timbre_authentizitaet"] > neutral.weights["timbre_authentizitaet"]

    def test_weak_tonality_boosts_groove(self):
        """Weak tonality → percussive → groove is primary."""
        imp = estimate_goal_importance(tonality=0.1)
        neutral = estimate_goal_importance()
        assert imp.weights["groove"] > neutral.weights["groove"]

    def test_weak_tonality_boosts_micro_dynamics(self):
        imp = estimate_goal_importance(tonality=0.1)
        neutral = estimate_goal_importance()
        assert imp.weights["micro_dynamics"] > neutral.weights["micro_dynamics"]

    def test_reason_mentions_tonality(self):
        imp = estimate_goal_importance(tonality=0.8)
        assert "tonal_strong" in imp.reason


class TestPsychoacousticFrequencyBalance:
    def test_bass_heavy_boosts_bass_kraft(self):
        imp = estimate_goal_importance(frequency_balance={"bass": 0.55, "mid": 0.25, "treble": 0.15, "air": 0.05})
        neutral = estimate_goal_importance()
        assert imp.weights["bass_kraft"] > neutral.weights["bass_kraft"]

    def test_bass_heavy_boosts_waerme(self):
        imp = estimate_goal_importance(frequency_balance={"bass": 0.50, "mid": 0.25, "treble": 0.20, "air": 0.05})
        neutral = estimate_goal_importance()
        assert imp.weights["waerme"] > neutral.weights["waerme"]

    def test_bright_mix_boosts_brillanz(self):
        imp = estimate_goal_importance(frequency_balance={"bass": 0.10, "mid": 0.30, "treble": 0.35, "air": 0.25})
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] > neutral.weights["brillanz"]

    def test_dark_mix_reduces_brillanz(self):
        imp = estimate_goal_importance(frequency_balance={"bass": 0.40, "mid": 0.45, "treble": 0.05, "air": 0.03})
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_thin_bass_reduces_bass_kraft(self):
        imp = estimate_goal_importance(frequency_balance={"bass": 0.05, "mid": 0.50, "treble": 0.30, "air": 0.15})
        neutral = estimate_goal_importance()
        assert imp.weights["bass_kraft"] < neutral.weights["bass_kraft"]

    def test_reason_mentions_balance(self):
        imp = estimate_goal_importance(frequency_balance={"bass": 0.55, "mid": 0.25, "treble": 0.10, "air": 0.10})
        assert "fb_bass_heavy" in imp.reason


class TestPsychoacousticMaskedComponents:
    def test_high_masking_reduces_spatial(self):
        """Many masked components → spatial goals unrealistic."""
        imp = estimate_goal_importance(masked_components_ratio=0.6)
        neutral = estimate_goal_importance()
        assert imp.weights["spatial_depth"] < neutral.weights["spatial_depth"]

    def test_high_masking_reduces_separation(self):
        imp = estimate_goal_importance(masked_components_ratio=0.6)
        neutral = estimate_goal_importance()
        assert imp.weights["separation_fidelity"] < neutral.weights["separation_fidelity"]

    def test_high_masking_boosts_transparenz(self):
        """What's audible must be maximally clear."""
        imp = estimate_goal_importance(masked_components_ratio=0.6)
        neutral = estimate_goal_importance()
        assert imp.weights["transparenz"] > neutral.weights["transparenz"]

    def test_low_masking_boosts_spatial(self):
        """Almost everything audible → full fidelity achievable."""
        imp = estimate_goal_importance(masked_components_ratio=0.05)
        neutral = estimate_goal_importance()
        assert imp.weights["spatial_depth"] > neutral.weights["spatial_depth"]

    def test_reason_mentions_masking(self):
        imp = estimate_goal_importance(masked_components_ratio=0.7)
        assert "masked_high" in imp.reason


class TestPsychoacousticPerceptualCentroid:
    def test_dark_centroid_boosts_waerme(self):
        """Low perceptual centroid = warm/dark character."""
        imp = estimate_goal_importance(perceptual_centroid_bark=3.0)
        neutral = estimate_goal_importance()
        assert imp.weights["waerme"] > neutral.weights["waerme"]

    def test_dark_centroid_reduces_brillanz(self):
        imp = estimate_goal_importance(perceptual_centroid_bark=3.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] < neutral.weights["brillanz"]

    def test_bright_centroid_boosts_brillanz(self):
        imp = estimate_goal_importance(perceptual_centroid_bark=14.0)
        neutral = estimate_goal_importance()
        assert imp.weights["brillanz"] > neutral.weights["brillanz"]

    def test_bright_centroid_reduces_waerme(self):
        imp = estimate_goal_importance(perceptual_centroid_bark=14.0)
        neutral = estimate_goal_importance()
        assert imp.weights["waerme"] < neutral.weights["waerme"]

    def test_reason_mentions_centroid(self):
        imp = estimate_goal_importance(perceptual_centroid_bark=3.0)
        assert "centroid_dark" in imp.reason


class TestPsychoacousticCombinedScenario:
    def test_noisy_dark_vinyl_full_psychoacoustic(self):
        """Realistic scenario: noisy, dark, rough vinyl with heavy masking."""
        imp = estimate_goal_importance(
            genre_label="jazz",
            era_decade="1960er",
            material_type="vinyl",
            roughness=0.5,
            sharpness=0.15,
            spectral_flatness=0.6,
            tonality=0.7,
            frequency_balance={"bass": 0.40, "mid": 0.35, "treble": 0.15, "air": 0.10},
            masked_components_ratio=0.45,
            perceptual_centroid_bark=5.5,
        )
        # Result must be valid
        for g in ALL_GOAL_NAMES:
            assert _WEIGHT_MIN <= imp.weights[g] <= _WEIGHT_MAX
        # Should reflect noisy → transparenz high, rough → artikulation high
        assert imp.weights["transparenz"] > 1.1
        assert imp.weights["artikulation"] > 1.0
        # Dark + low sharpness → brillanz deprioritised
        assert imp.weights["brillanz"] < 1.0
        # Strong tonality → tonal_center boosted
        assert imp.weights["tonal_center"] > 1.0


# ── HNR / Harmonic Coherence / Crest Factor / Transient Density ─────


class TestHarmonicToNoiseRatio:
    """Tests calibrated for full-mix pitch-period HNR (range: -10 to +20 dB).
    Thresholds: >5 dB = clean, <0 dB = noisy.  Not isolated-vocal HNR!"""

    def test_clean_harmonics_boosts_timbre(self):
        """HNR > 5 dB = harmonics above noise in full mix → protect timbre."""
        imp = estimate_goal_importance(harmonic_to_noise_ratio_db=12.0)
        neutral = estimate_goal_importance()
        assert imp.weights["timbre_authentizitaet"] > neutral.weights["timbre_authentizitaet"]

    def test_clean_harmonics_boosts_tonal_center(self):
        imp = estimate_goal_importance(harmonic_to_noise_ratio_db=10.0)
        neutral = estimate_goal_importance()
        assert imp.weights["tonal_center"] > neutral.weights["tonal_center"]

    def test_clean_harmonics_boosts_artikulation(self):
        imp = estimate_goal_importance(harmonic_to_noise_ratio_db=8.0)
        neutral = estimate_goal_importance()
        assert imp.weights["artikulation"] > neutral.weights["artikulation"]

    def test_noisy_harmonics_boosts_transparenz(self):
        """HNR < 0 dB = harmonics buried in noise → transparenz critical."""
        imp = estimate_goal_importance(harmonic_to_noise_ratio_db=-5.0)
        neutral = estimate_goal_importance()
        assert imp.weights["transparenz"] > neutral.weights["transparenz"]

    def test_noisy_harmonics_reduces_timbre(self):
        imp = estimate_goal_importance(harmonic_to_noise_ratio_db=-3.0)
        neutral = estimate_goal_importance()
        assert imp.weights["timbre_authentizitaet"] < neutral.weights["timbre_authentizitaet"]

    def test_reason_mentions_hnr(self):
        imp = estimate_goal_importance(harmonic_to_noise_ratio_db=10.0)
        assert "hnr_clean" in imp.reason


class TestHarmonicCoherence:
    def test_strong_coherence_boosts_tonal(self):
        imp = estimate_goal_importance(harmonic_coherence=0.8)
        neutral = estimate_goal_importance()
        assert imp.weights["tonal_center"] > neutral.weights["tonal_center"]

    def test_strong_coherence_boosts_timbre(self):
        imp = estimate_goal_importance(harmonic_coherence=0.8)
        neutral = estimate_goal_importance()
        assert imp.weights["timbre_authentizitaet"] > neutral.weights["timbre_authentizitaet"]

    def test_weak_coherence_boosts_groove(self):
        """Weak coherence = percussive → groove critical."""
        imp = estimate_goal_importance(harmonic_coherence=0.2)
        neutral = estimate_goal_importance()
        assert imp.weights["groove"] > neutral.weights["groove"]

    def test_reason_mentions_coherence(self):
        imp = estimate_goal_importance(harmonic_coherence=0.8)
        assert "hcoh_strong" in imp.reason


class TestCrestFactor:
    """Tests calibrated for 99.9th-percentile crest factor.
    Thresholds: >12 dB = dynamic, <7 dB = compressed."""

    def test_dynamic_boosts_micro_dynamics(self):
        """High crest (99.9pctl) > 12 dB → protect micro_dynamics."""
        imp = estimate_goal_importance(crest_factor_db=15.0)
        neutral = estimate_goal_importance()
        assert imp.weights["micro_dynamics"] > neutral.weights["micro_dynamics"]

    def test_dynamic_boosts_groove(self):
        imp = estimate_goal_importance(crest_factor_db=14.0)
        neutral = estimate_goal_importance()
        assert imp.weights["groove"] > neutral.weights["groove"]

    def test_compressed_reduces_micro_dynamics(self):
        """Crest < 7 dB → already compressed, micro_dynamics less achievable."""
        imp = estimate_goal_importance(crest_factor_db=5.0)
        neutral = estimate_goal_importance()
        assert imp.weights["micro_dynamics"] < neutral.weights["micro_dynamics"]

    def test_reason_mentions_crest(self):
        imp = estimate_goal_importance(crest_factor_db=15.0)
        assert "crest_dynamic" in imp.reason


class TestTransientDensity:
    """Tests calibrated for STFT spectral-flux onsets (50ms min-gap).
    Thresholds: >8/s = percussive, <2/s = sustained."""

    def test_percussive_boosts_groove(self):
        """Transient density > 8/s → groove critical."""
        imp = estimate_goal_importance(transient_density=12.0)
        neutral = estimate_goal_importance()
        assert imp.weights["groove"] > neutral.weights["groove"]

    def test_percussive_boosts_artikulation(self):
        imp = estimate_goal_importance(transient_density=10.0)
        neutral = estimate_goal_importance()
        assert imp.weights["artikulation"] > neutral.weights["artikulation"]

    def test_sustained_boosts_timbre(self):
        """Transient density < 2/s → timbre defines texture."""
        imp = estimate_goal_importance(transient_density=0.5)
        neutral = estimate_goal_importance()
        assert imp.weights["timbre_authentizitaet"] > neutral.weights["timbre_authentizitaet"]

    def test_sustained_boosts_waerme(self):
        imp = estimate_goal_importance(transient_density=0.3)
        neutral = estimate_goal_importance()
        assert imp.weights["waerme"] > neutral.weights["waerme"]

    def test_reason_mentions_transients(self):
        imp = estimate_goal_importance(transient_density=12.0)
        assert "transient_dense" in imp.reason


# ── Masked Components Ratio sanity guard ────────────────────────────


class TestMaskedRatioSanityGuard:
    def test_extreme_high_masked_ignored(self):
        """masked_ratio ≥ 0.95 is measurement artifact → must be ignored."""
        imp_buggy = estimate_goal_importance(masked_components_ratio=1.0)
        neutral = estimate_goal_importance()
        # Should behave like neutral (no masking adjustment)
        assert abs(imp_buggy.weights["spatial_depth"] - neutral.weights["spatial_depth"]) < 0.01
        assert abs(imp_buggy.weights["separation_fidelity"] - neutral.weights["separation_fidelity"]) < 0.01

    def test_extreme_low_masked_ignored(self):
        """masked_ratio ≤ 0.01 is measurement artifact → must be ignored."""
        imp_buggy = estimate_goal_importance(masked_components_ratio=0.005)
        neutral = estimate_goal_importance()
        assert abs(imp_buggy.weights["spatial_depth"] - neutral.weights["spatial_depth"]) < 0.01

    def test_valid_masked_range_applies(self):
        """Values in valid range (0.01, 0.95) should still take effect."""
        imp = estimate_goal_importance(masked_components_ratio=0.6)
        neutral = estimate_goal_importance()
        assert imp.weights["spatial_depth"] < neutral.weights["spatial_depth"]


# ── Improved soft-cap (log-domain compression) ──────────────────────


class TestImprovedSoftCap:
    def test_extreme_stacking_tamed(self):
        """Multiple boosting stages should not exceed ~1.75 after soft-cap."""
        imp = estimate_goal_importance(
            genre_label="schlager",  # waerme boost
            era_decade="1970er",  # waerme boost
            material_type="vinyl",  # waerme boost
            sharpness=0.1,  # waerme boost
            frequency_balance={"bass": 0.55, "mid": 0.25, "treble": 0.10, "air": 0.10},  # waerme boost
            perceptual_centroid_bark=3.0,  # waerme boost
            transient_density=0.5,  # waerme boost
            crest_factor_db=6.0,  # waerme boost
        )
        # Even with 8 boosting sources, soft-cap must tame it
        # k=3.0 compression: asymptote at 1.5+0.33=1.83
        assert imp.weights["waerme"] <= 1.85
        assert imp.weights["waerme"] > 1.5  # Still reflects high importance

    def test_logarithmic_growth(self):
        """Soft-cap must grow sub-linearly above 1.5."""
        # Compare 2 different excess levels
        imp_moderate = estimate_goal_importance(
            genre_label="schlager",
            era_decade="1970er",
            material_type="vinyl",
        )
        imp_extreme = estimate_goal_importance(
            genre_label="schlager",
            era_decade="1970er",
            material_type="vinyl",
            sharpness=0.1,
            perceptual_centroid_bark=3.0,
            frequency_balance={"bass": 0.55, "mid": 0.25, "treble": 0.10, "air": 0.10},
        )
        # The extreme case adds 3 more boost sources but the gap should be small
        _delta = imp_extreme.weights["waerme"] - imp_moderate.weights["waerme"]
        assert _delta < 0.20  # Sub-linear growth: 3 extra boosts add < 0.20

    def test_below_0_5_compressed(self):
        """Values below 0.5 should also be log-compressed."""
        imp = estimate_goal_importance(
            genre_label="schlager",
            era_decade="1970er",
            material_type="vinyl",
            sharpness=0.1,
            spectral_flatness=0.7,
            perceptual_centroid_bark=3.0,
            frequency_balance={"bass": 0.55, "mid": 0.25, "treble": 0.01, "air": 0.01},
        )
        # brillanz should be reduced but not below 0.3
        assert imp.weights["brillanz"] >= _WEIGHT_MIN
        assert imp.weights["brillanz"] < 0.6


# ── Full-feature combined scenario (vocal + music preservation) ─────


class TestFullFeatureCombinedScenario:
    def test_vocal_schlager_full_analysis(self):
        """Elke Best scenario: Schlager/1970/vinyl + all features.

        Tests that vocal harmonics are protected while defect removal
        is still possible (weights must stay in useful range).
        """
        imp = estimate_goal_importance(
            genre_label="Deutscher Schlager",
            era_decade="1970",
            material_type="vinyl",
            restorability_score=63.5,
            vocal_detected=True,
            vocal_confidence=0.85,
            snr_db=9.2,
            effective_bandwidth_hz=22050.0,
            dynamic_range_db=12.8,
            bpm=136.0,
            defect_severities={"digital_artifacts": 1.0, "high_freq_noise": 1.0},
            spectral_tilt_db_per_oct=-3.2,
            transfer_generation_count=3,
            source_fidelity_confidence=0.40,
            roughness=0.76,
            sharpness=0.30,
            spectral_flatness=0.01,
            tonality=0.35,
            frequency_balance={"bass": 0.56, "mid": 0.80, "treble": 0.04, "air": 0.01},
            masked_components_ratio=1.0,  # Bug value → must be ignored
            perceptual_centroid_bark=3.66,
            harmonic_to_noise_ratio_db=12.0,
            harmonic_coherence=0.45,
            crest_factor_db=11.0,
            transient_density=3.2,
        )
        # All weights must be in valid range
        for g in ALL_GOAL_NAMES:
            assert _WEIGHT_MIN <= imp.weights[g] <= _WEIGHT_MAX, f"{g}={imp.weights[g]}"
        # Vocal character preserved: artikulation strong
        assert imp.weights["artikulation"] > 1.3
        # Warmth important (Schlager + vinyl + dark)
        assert imp.weights["waerme"] > 1.5
        # Transparenz important (roughness + vocal + noisy SNR)
        assert imp.weights["transparenz"] > 1.3
        # Brillanz low (dark centroid, low treble/air, carrier chain)
        assert imp.weights["brillanz"] < 0.7
        # Masked ratio=1.0 must be IGNORED → spatial_depth not unfairly punished
        estimate_goal_importance().weights["spatial_depth"]
        # With carrier chain + other factors it may be below neutral, but not
        # crushed by the bogus masked_ratio
        assert imp.weights["spatial_depth"] > 0.4


# ── Cross-Feature Interactions (Stage 5) ─────────────────────────────


class TestInteractionRoughTimesNoisy:
    def test_rough_noisy_synergy_transparenz(self):
        """Roughness + low SNR together → transparenz boost > either alone."""
        both = estimate_goal_importance(roughness=0.6, snr_db=10.0)
        rough_only = estimate_goal_importance(roughness=0.6, snr_db=45.0)
        noisy_only = estimate_goal_importance(roughness=0.05, snr_db=10.0)
        # Synergy: both > each individually
        assert both.weights["transparenz"] > rough_only.weights["transparenz"]
        assert both.weights["transparenz"] > noisy_only.weights["transparenz"]

    def test_interact_reason_tag(self):
        imp = estimate_goal_importance(roughness=0.7, snr_db=5.0)
        assert "interact_rough" in imp.reason


class TestInteractionVocalHNR:
    def test_vocal_clean_harmonics_boosts_timbre(self):
        """Clean harmonics (HNR > 5 dB, full-mix) + vocal → timbre synergy."""
        vocal_hnr = estimate_goal_importance(
            vocal_detected=True,
            vocal_confidence=0.9,
            harmonic_to_noise_ratio_db=12.0,
        )
        vocal_only = estimate_goal_importance(
            vocal_detected=True,
            vocal_confidence=0.9,
        )
        assert vocal_hnr.weights["timbre_authentizitaet"] > vocal_only.weights["timbre_authentizitaet"]


class TestInteractionBandwidthDark:
    def test_narrow_dark_reduces_brillanz(self):
        """Low BW + dark centroid → brillanz reduced synergistically."""
        both = estimate_goal_importance(
            effective_bandwidth_hz=7000.0,
            perceptual_centroid_bark=3.0,
        )
        bw_only = estimate_goal_importance(
            effective_bandwidth_hz=7000.0,
            perceptual_centroid_bark=8.0,
        )
        dark_only = estimate_goal_importance(
            effective_bandwidth_hz=20000.0,
            perceptual_centroid_bark=3.0,
        )
        assert both.weights["brillanz"] < bw_only.weights["brillanz"]
        assert both.weights["brillanz"] < dark_only.weights["brillanz"]


class TestInteractionCoherenceTonal:
    def test_coherent_tonal_boosts_tonal_center(self):
        """Strong pitch + tonality → tonal_center synergy."""
        both = estimate_goal_importance(harmonic_coherence=0.8, tonality=0.7)
        coh_only = estimate_goal_importance(harmonic_coherence=0.8, tonality=0.1)
        assert both.weights["tonal_center"] > coh_only.weights["tonal_center"]


class TestInteractionDynamicTransient:
    def test_dynamic_percussive_synergy(self):
        """Crest > 10 dB (99.9pctl) + transients > 5/s → groove synergy."""
        both = estimate_goal_importance(crest_factor_db=15.0, transient_density=8.0)
        crest_only = estimate_goal_importance(crest_factor_db=15.0, transient_density=0.5)
        assert both.weights["groove"] > crest_only.weights["groove"]


class TestInteractionChainNoisy:
    def test_degraded_chain_noisy_core_preservation(self):
        """Multi-gen chain + noise → natuerlichkeit/authentizitaet boosted."""
        both = estimate_goal_importance(
            transfer_generation_count=4,
            snr_db=8.0,
        )
        chain_only = estimate_goal_importance(
            transfer_generation_count=4,
            snr_db=30.0,  # Neutral SNR (no Step 6b trigger)
        )
        assert both.weights["natuerlichkeit"] > chain_only.weights["natuerlichkeit"]
        assert both.weights["authentizitaet"] > chain_only.weights["authentizitaet"]


# ── SongGoalImportance dataclass ────────────────────────────────────


class TestDataclass:
    def test_frozen(self):
        imp = estimate_goal_importance(genre_label="rock")
        with pytest.raises(AttributeError):
            imp.genre_profile = "pop"  # type: ignore[misc]

    def test_15_goals_in_weights(self):
        imp = estimate_goal_importance(genre_label="rock")
        assert len(imp.weights) == 15

    def test_all_goal_names_tuple(self):
        assert len(ALL_GOAL_NAMES) == 15
        assert "natuerlichkeit" in ALL_GOAL_NAMES
        assert "spatial_depth" in ALL_GOAL_NAMES
        assert "transient_energie" in ALL_GOAL_NAMES
