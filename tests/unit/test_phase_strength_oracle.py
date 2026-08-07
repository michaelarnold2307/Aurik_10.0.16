from __future__ import annotations

"""Tests fuer das phasenspezifische Strength-Orakel (§2.56b)."""


from types import SimpleNamespace

import numpy as np
import pytest

from backend.core.dsp.phase_strength_oracle import resolve_phase_strength_oracle
from backend.core.unified_restorer_v3 import UnifiedRestorerV3


def test_oracle_maps_sibilance_family_to_o7():
    prof = resolve_phase_strength_oracle(
        phase_id="phase_19_de_esser",
        phase_family="sibilance_control",
        current_strength=0.3,
        goal_gaps={"artikulation": 0.12, "vocal_quality": 0.15},
        goal_weights={"artikulation": 1.4, "vocal_quality": 1.8},
        defect_scores={"sibilance": 0.8},
        locality_factor=1.0,
        restorability_score=52.0,
        material_key="vinyl",
        song_calibration_profile={"restorability_score": 52.0},
        panns_singing=0.65,
    )
    assert prof.oracle_class == "O7_vocal_articulation"
    assert prof.ratio is not None
    assert prof.threshold_db is not None


def test_oracle_increases_control_with_higher_severity():
    mild = resolve_phase_strength_oracle(
        phase_id="phase_03_denoise",
        phase_family="subtractive_cleanup",
        current_strength=0.25,
        goal_gaps={"transparenz": 0.04},
        goal_weights={"transparenz": 1.2},
        defect_scores={"noise": 0.2},
        locality_factor=1.0,
        restorability_score=60.0,
        material_key="tape",
        song_calibration_profile=None,
        panns_singing=0.0,
    )
    hard = resolve_phase_strength_oracle(
        phase_id="phase_03_denoise",
        phase_family="subtractive_cleanup",
        current_strength=0.25,
        goal_gaps={"transparenz": 0.18, "natuerlichkeit": 0.12},
        goal_weights={"transparenz": 1.2, "natuerlichkeit": 1.4},
        defect_scores={"noise": 0.9},
        locality_factor=1.0,
        restorability_score=35.0,
        material_key="tape",
        song_calibration_profile=None,
        panns_singing=0.0,
    )
    assert hard.control_strength > mild.control_strength
    assert hard.wet_mix >= mild.wet_mix


def test_oracle_respects_locality_factor():
    wide = resolve_phase_strength_oracle(
        phase_id="phase_09_crackle_removal",
        phase_family="subtractive_cleanup",
        current_strength=0.5,
        goal_gaps={"natuerlichkeit": 0.10},
        goal_weights={"natuerlichkeit": 1.1},
        defect_scores={"crackle": 0.7},
        locality_factor=1.0,
        restorability_score=45.0,
        material_key="vinyl",
        song_calibration_profile=None,
        panns_singing=0.0,
    )
    local = resolve_phase_strength_oracle(
        phase_id="phase_09_crackle_removal",
        phase_family="subtractive_cleanup",
        current_strength=0.5,
        goal_gaps={"natuerlichkeit": 0.10},
        goal_weights={"natuerlichkeit": 1.1},
        defect_scores={"crackle": 0.7},
        locality_factor=0.4,
        restorability_score=45.0,
        material_key="vinyl",
        song_calibration_profile=None,
        panns_singing=0.0,
    )
    assert local.control_strength < wide.control_strength


def test_oracle_chain_factor_reduces_strength_for_multistage_chain():
    simple = resolve_phase_strength_oracle(
        phase_id="phase_03_denoise",
        phase_family="subtractive_cleanup",
        current_strength=0.6,
        goal_gaps={"transparenz": 0.20},
        goal_weights={"transparenz": 1.2},
        defect_scores={"noise": 0.95},
        locality_factor=1.0,
        restorability_score=35.0,
        material_key="vinyl",
        song_calibration_profile=None,
        panns_singing=0.0,
        transfer_chain=["vinyl"],
        chain_confidence=0.95,
    )
    chained = resolve_phase_strength_oracle(
        phase_id="phase_03_denoise",
        phase_family="subtractive_cleanup",
        current_strength=0.6,
        goal_gaps={"transparenz": 0.20},
        goal_weights={"transparenz": 1.2},
        defect_scores={"noise": 0.95},
        locality_factor=1.0,
        restorability_score=35.0,
        material_key="vinyl",
        song_calibration_profile=None,
        panns_singing=0.0,
        transfer_chain=["vinyl", "cassette", "mp3_low"],
        chain_confidence=0.95,
    )

    assert float(chained.hard_caps.get("chain_factor", 1.0)) < float(simple.hard_caps.get("chain_factor", 1.0))
    assert chained.control_strength < simple.control_strength


def test_oracle_voice_guard_dampens_strength_for_vocal_risk():
    baseline = resolve_phase_strength_oracle(
        phase_id="phase_19_de_esser",
        phase_family="sibilance_control",
        current_strength=0.7,
        goal_gaps={"artikulation": 0.20},
        goal_weights={"artikulation": 1.4},
        defect_scores={"sibilance": 0.9},
        locality_factor=1.0,
        restorability_score=35.0,
        material_key="vinyl",
        song_calibration_profile=None,
        panns_singing=0.62,
    )
    guarded = resolve_phase_strength_oracle(
        phase_id="phase_19_de_esser",
        phase_family="sibilance_control",
        current_strength=0.7,
        goal_gaps={"artikulation": 0.20},
        goal_weights={"artikulation": 1.4},
        defect_scores={"sibilance": 0.9},
        locality_factor=1.0,
        restorability_score=35.0,
        material_key="vinyl",
        song_calibration_profile=None,
        panns_singing=0.62,
        vocal_guard_metrics={
            "vqi": 0.58,
            "formant_integrity": 0.74,
            "vibrato_depth_preserved": 0.70,
            "micro_dynamic_correlation": 0.80,
        },
    )

    assert float(guarded.hard_caps.get("voice_guard_risk", 0.0)) > 0.0
    assert guarded.control_strength < baseline.control_strength
    assert guarded.wet_mix <= baseline.wet_mix


def test_oracle_voice_guard_not_applied_below_singing_threshold():
    with_guard = resolve_phase_strength_oracle(
        phase_id="phase_03_denoise",
        phase_family="subtractive_cleanup",
        current_strength=0.5,
        goal_gaps={"transparenz": 0.18},
        goal_weights={"transparenz": 1.2},
        defect_scores={"noise": 0.8},
        locality_factor=1.0,
        restorability_score=45.0,
        material_key="tape",
        song_calibration_profile=None,
        panns_singing=0.10,
        vocal_guard_metrics={
            "vqi": 0.4,
            "formant_integrity": 0.4,
            "vibrato_depth_preserved": 0.4,
            "micro_dynamic_correlation": 0.4,
        },
    )
    without_guard = resolve_phase_strength_oracle(
        phase_id="phase_03_denoise",
        phase_family="subtractive_cleanup",
        current_strength=0.5,
        goal_gaps={"transparenz": 0.18},
        goal_weights={"transparenz": 1.2},
        defect_scores={"noise": 0.8},
        locality_factor=1.0,
        restorability_score=45.0,
        material_key="tape",
        song_calibration_profile=None,
        panns_singing=0.10,
    )

    assert float(with_guard.hard_caps.get("voice_guard_risk", -1.0)) == pytest.approx(0.0)
    assert with_guard.control_strength == pytest.approx(without_guard.control_strength)
    assert with_guard.wet_mix == pytest.approx(without_guard.wet_mix)


def test_uv3_runtime_context_uses_vocal_guard_metrics_for_no_harm_damping():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_19_de_esser", name="De-Esser")
    audio = np.zeros(48_000, dtype=np.float32)

    kwargs_without = {
        "strength": 0.65,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "defect_scores": {"sibilance": 1.0},
    }
    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs_without,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"artikulation": 1.6, "vocal_quality": 1.4},
        rest_ctx=35.0,
    )
    strength_without = float(kwargs_without.get("strength", 0.0))  # type: ignore[arg-type]

    kwargs_with = {
        "strength": 0.65,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "defect_scores": {"sibilance": 1.0},
        "vocal_guard_metrics": {
            "vqi": 0.58,
            "formant_integrity": 0.72,
            "vibrato_depth_preserved": 0.70,
            "micro_dynamic_correlation": 0.80,
        },
    }
    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs_with,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"artikulation": 1.6, "vocal_quality": 1.4},
        rest_ctx=35.0,
    )
    profile_with = kwargs_with.get("phase_strength_oracle_profile", {})

    assert kwargs_with.get("phase_strength_oracle_class") == "O7_vocal_articulation"
    assert float(profile_with.get("hard_caps", {}).get("voice_guard_risk", 0.0)) > 0.0  # type: ignore[attr-defined]
    assert float(kwargs_with.get("strength", 0.0)) < strength_without  # type: ignore[arg-type]
    assert float(kwargs_with.get("phase_voice_guard_risk", 0.0)) > 0.0  # type: ignore[arg-type]
    assert kwargs_with.get("phase_voice_guard_damped") is True

    events = uv3._restoration_context.get("voice_guard_events", [])
    assert isinstance(events, list)
    assert events, "voice_guard_events sollte bei Dämpfung nicht leer sein"
    last = events[-1]
    assert last.get("phase_id") == "phase_19_de_esser"
    assert float(last.get("voice_guard_risk", 0.0)) > 0.0


def test_uv3_voice_guard_learning_scales_prior_strength_params():
    base = {
        "noise_reduction_strength": 0.72,
        "harmonic_boost_db": 2.4,
        "compression_ratio": 2.1,
        "ola_crossfade_ms": 22.0,
    }
    events = [
        {"phase_id": "phase_19_de_esser", "voice_guard_risk": 0.8},
        {"phase_id": "phase_03_denoise", "voice_guard_risk": 0.5},
    ]

    learned = UnifiedRestorerV3._derive_voice_guard_learning_from_events(events, base)

    assert isinstance(learned, dict)
    assert float(learned.get("noise_reduction_strength", 1.0)) < float(base["noise_reduction_strength"])
    assert float(learned.get("harmonic_boost_db", 99.0)) < float(base["harmonic_boost_db"])
    assert float(learned.get("compression_ratio", 99.0)) < float(base["compression_ratio"])
    assert float(learned.get("ola_crossfade_ms", 0.0)) == pytest.approx(float(base["ola_crossfade_ms"]))
    assert float(learned.get("_voice_guard_event_count", 0.0)) == pytest.approx(2.0)
    assert float(learned.get("_voice_guard_phase_count", 0.0)) == pytest.approx(2.0)
    assert 0.0 < float(learned.get("_voice_guard_learning_scale", 0.0)) < 1.0


def test_uv3_voice_guard_learning_is_blocked_on_negative_outcome():
    base = {
        "noise_reduction_strength": 0.72,
        "harmonic_boost_db": 2.4,
    }
    events = [{"phase_id": "phase_19_de_esser", "voice_guard_risk": 0.8}]

    learned = UnifiedRestorerV3._derive_voice_guard_learning_from_events(
        events,
        base,
        outcome_payload={"learn_enabled": False, "reason": "vqi_regression"},
    )

    assert float(learned.get("noise_reduction_strength", 0.0)) == pytest.approx(float(base["noise_reduction_strength"]))
    assert float(learned.get("harmonic_boost_db", 0.0)) == pytest.approx(float(base["harmonic_boost_db"]))
    assert float(learned.get("_voice_guard_learning_applied", 1.0)) == pytest.approx(0.0)
    assert learned.get("_voice_guard_learn_block_reason") == "vqi_regression"


def test_uv3_voice_guard_outcome_payload_detects_vqi_regression():
    events = [{"phase_id": "phase_19_de_esser", "voice_guard_risk": 0.7, "vqi": 0.80, "artifact_freedom": 0.97}]
    hpi_result = SimpleNamespace(hpi=0.42, artifact_freedom=0.97, vqi=0.75)

    outcome = UnifiedRestorerV3._build_voice_guard_outcome_payload(events, hpi_result)

    assert outcome.get("learn_enabled") is False
    assert outcome.get("reason") == "vqi_regression"
    assert float(outcome.get("vqi_delta", 0.0)) < 0.0


def test_uv3_voice_guard_coalition_payload_applies_outcome_weighted_factor():
    events = [
        {"phase_id": "phase_19_de_esser", "voice_guard_risk": 0.7},
        {"phase_id": "phase_20_reverb_reduction", "voice_guard_risk": 0.8},
        {"phase_id": "phase_49_advanced_dereverb", "voice_guard_risk": 0.6},
    ]
    coalitions = {
        "vocal_production": [
            "phase_19_de_esser",
            "phase_20_reverb_reduction",
            "phase_49_advanced_dereverb",
        ]
    }
    outcome = {"learn_enabled": True, "vqi_delta": 0.03, "artifact_delta": 0.01}

    payload = UnifiedRestorerV3._build_voice_guard_coalition_payload(
        events,
        coalitions,
        ["phase_19_de_esser", "phase_20_reverb_reduction", "phase_49_advanced_dereverb"],
        outcome,
    )

    assert 0.0 < float(payload.get("dominant_coalition_event_ratio", 0.0)) <= 1.0
    assert float(payload.get("dominant_coalition_event_count", 0.0)) >= 2.0
    assert 0.80 <= float(payload.get("coalition_learning_factor", 1.0)) < 1.0


def test_uv3_voice_guard_learning_includes_coalition_factor():
    base = {
        "noise_reduction_strength": 0.70,
        "compression_ratio": 2.0,
    }
    events = [
        {"phase_id": "phase_19_de_esser", "voice_guard_risk": 0.8},
        {"phase_id": "phase_20_reverb_reduction", "voice_guard_risk": 0.7},
    ]

    no_coal = UnifiedRestorerV3._derive_voice_guard_learning_from_events(
        events,
        base,
        outcome_payload={"learn_enabled": True, "vqi_delta": 0.02, "artifact_delta": 0.01},
    )
    with_coal = UnifiedRestorerV3._derive_voice_guard_learning_from_events(
        events,
        base,
        outcome_payload={"learn_enabled": True, "vqi_delta": 0.02, "artifact_delta": 0.01},
        coalition_payload={"coalition_learning_factor": 0.85, "dominant_coalition_event_ratio": 1.0},
    )

    assert float(with_coal.get("noise_reduction_strength", 1.0)) < float(no_coal.get("noise_reduction_strength", 1.0))
    assert float(with_coal.get("compression_ratio", 99.0)) < float(no_coal.get("compression_ratio", 99.0))
    assert float(with_coal.get("_voice_guard_coalition_factor", 0.0)) == pytest.approx(0.85)


def test_uv3_voice_guard_causal_credit_payload_from_coalition_deltas():
    phase_deltas = {
        "coalition:vocal_production": {
            "delta": {
                "artikulation": 0.06,
                "natuerlichkeit": 0.04,
                "transparenz": 0.03,
            }
        },
        "coalition:vinyl_disc": {
            "delta": {
                "artikulation": 0.01,
                "natuerlichkeit": 0.0,
                "transparenz": 0.01,
            }
        },
    }
    outcome = {"learn_enabled": True, "vqi_delta": 0.03, "artifact_delta": 0.01, "hpi_score": 0.42}

    payload = UnifiedRestorerV3._build_voice_guard_causal_credit_payload(phase_deltas, outcome)

    assert 0.0 < float(payload.get("causal_credit_confidence", 0.0)) <= 1.0
    assert payload.get("dominant_coalition") == "vocal_production"
    assert float(payload.get("dominant_coalition_credit", 0.0)) > 0.0
    c_map = payload.get("coalition_credit_map", {})
    assert isinstance(c_map, dict)
    assert float(c_map.get("vocal_production", 0.0)) > float(c_map.get("vinyl_disc", 0.0))


def test_uv3_voice_guard_learning_uses_causal_factor():
    base = {
        "noise_reduction_strength": 0.70,
        "compression_ratio": 2.0,
    }
    events = [
        {"phase_id": "phase_19_de_esser", "voice_guard_risk": 0.8},
        {"phase_id": "phase_20_reverb_reduction", "voice_guard_risk": 0.7},
    ]

    no_causal = UnifiedRestorerV3._derive_voice_guard_learning_from_events(
        events,
        base,
        outcome_payload={"learn_enabled": True, "vqi_delta": 0.02, "artifact_delta": 0.01},
    )
    with_causal = UnifiedRestorerV3._derive_voice_guard_learning_from_events(
        events,
        base,
        outcome_payload={"learn_enabled": True, "vqi_delta": 0.02, "artifact_delta": 0.01},
        causal_payload={"causal_credit_confidence": 0.9, "dominant_coalition": "vocal_production"},
    )

    assert float(with_causal.get("noise_reduction_strength", 1.0)) < float(
        no_causal.get("noise_reduction_strength", 1.0)
    )
    assert float(with_causal.get("compression_ratio", 99.0)) < float(no_causal.get("compression_ratio", 99.0))
    assert float(with_causal.get("_voice_guard_causal_factor", 0.0)) < 1.0
    assert float(with_causal.get("_causal_credit_confidence", 0.0)) == pytest.approx(0.9)


def test_uv3_voice_guard_counterfactual_payload_estimates_top1_drop():
    coalition_payload = {
        "dominant_coalition_event_ratio": 0.9,
        "dominant_coalition_event_count": 3.0,
    }
    causal_payload = {
        "dominant_coalition": "vocal_production",
        "dominant_coalition_credit": 0.42,
        "causal_credit_confidence": 0.66,
    }
    outcome_payload = {
        "learn_enabled": True,
        "vqi_delta": 0.03,
        "artifact_delta": 0.01,
        "hpi_score": 0.44,
    }

    payload = UnifiedRestorerV3._build_voice_guard_counterfactual_payload(
        coalition_payload,
        causal_payload,
        outcome_payload,
    )

    assert payload.get("dominant_coalition") == "vocal_production"
    assert 0.0 < float(payload.get("dominant_drop_estimate", 0.0)) <= 1.0
    assert 0.0 < float(payload.get("counterfactual_alignment", 0.0)) <= 1.0
    assert 0.0 < float(payload.get("counterfactual_confidence", 0.0)) <= 1.0


def test_uv3_voice_guard_learning_uses_counterfactual_factor():
    base = {
        "noise_reduction_strength": 0.70,
        "compression_ratio": 2.0,
    }
    events = [
        {"phase_id": "phase_19_de_esser", "voice_guard_risk": 0.35},
        {"phase_id": "phase_20_reverb_reduction", "voice_guard_risk": 0.25},
    ]

    no_cf = UnifiedRestorerV3._derive_voice_guard_learning_from_events(
        events,
        base,
        outcome_payload={"learn_enabled": True, "vqi_delta": 0.02, "artifact_delta": 0.01},
        causal_payload={"causal_credit_confidence": 0.5},
    )
    with_cf = UnifiedRestorerV3._derive_voice_guard_learning_from_events(
        events,
        base,
        outcome_payload={"learn_enabled": True, "vqi_delta": 0.02, "artifact_delta": 0.01},
        causal_payload={"causal_credit_confidence": 0.5},
        counterfactual_payload={
            "counterfactual_confidence": 0.8,
            "dominant_drop_estimate": 0.4,
            "counterfactual_alignment": 0.3,
            "dominant_coalition": "vocal_production",
        },
    )

    assert float(with_cf.get("_voice_guard_learning_scale", 1.0)) < float(no_cf.get("_voice_guard_learning_scale", 1.0))
    assert float(with_cf.get("noise_reduction_strength", 1.0)) < float(no_cf.get("noise_reduction_strength", 1.0))
    assert float(with_cf.get("compression_ratio", 99.0)) < float(no_cf.get("compression_ratio", 99.0))
    assert float(with_cf.get("_voice_guard_counterfactual_factor", 0.0)) < 1.0
    assert float(with_cf.get("_counterfactual_counterfactual_confidence", 0.0)) == pytest.approx(0.8)


def test_uv3_runtime_context_injects_oracle_profile_for_pilot_phase():
    class _DummyUV3:
        _PHASE_INTERVENTION_CLASS = UnifiedRestorerV3._PHASE_INTERVENTION_CLASS
        _conductor_strength_hints = {}
        _vintage_phase_strength_caps = {}
        _restoration_context = {
            "frisson_zones": [],
            "passaggio_zones": [],
            "vibrato_zones": [],
            "breath_segments": [],
            "soft_saturation_severity": 0.0,
            "soft_saturation_preserve": False,
            "primary_material": "vinyl",
        }
        _global_conservative_scalar = 1.0
        _panns_singing = 0.62
        _pmgg_ceiling_capped_targets = {}
        _song_goal_targets = {}

        def is_studio_mode(self):
            return False

        def _get_phase_calibration_scalar(self, _phase_id, _song_calibration):
            return 1.0

        def _resolve_phase_strength_oracle_rollout_mode(self, kwargs):
            return UnifiedRestorerV3._resolve_phase_strength_oracle_rollout_mode(self, kwargs)  # type: ignore[arg-type]

        def _canonical_phase_context_kwargs(self):
            return UnifiedRestorerV3._canonical_phase_context_kwargs(self)  # type: ignore[arg-type]

    uv3 = _DummyUV3()
    phase_meta = SimpleNamespace(phase_id="phase_19_de_esser", name="De-Esser")
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 0.55,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
    }

    wet_dry = UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,  # type: ignore[arg-type]
        phase_meta,
        audio,
        kwargs,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"artikulation": 1.4, "vocal_quality": 1.6},
        rest_ctx=45.0,
    )

    assert "phase_strength_oracle_profile" in kwargs
    assert kwargs.get("phase_strength_oracle_class") == "O7_vocal_articulation"
    assert 0.0 <= float(kwargs.get("strength", 0.0)) <= 1.0  # type: ignore[arg-type]
    assert 0.0 < float(wet_dry) <= 1.0


def test_uv3_oracle_rollout_pilot_is_non_empty_subset_of_canonical_phases():
    mapped = set(UnifiedRestorerV3._PHASE_INTERVENTION_CLASS.keys())
    rolled_out = set(UnifiedRestorerV3._PHASE_STRENGTH_ORACLE_PILOT_PHASES)
    assert rolled_out
    assert rolled_out.issubset(mapped)
    assert "phase_19_de_esser" in rolled_out


@pytest.mark.parametrize("phase_id", sorted(UnifiedRestorerV3._PHASE_STRENGTH_ORACLE_PILOT_PHASES))
def test_uv3_runtime_context_injects_oracle_telemetry_for_every_pilot_phase(phase_id):
    """Regression-Guard: alle Pilot-Phasen muessen Orakel-Telemetrie setzen.

    Dieser Test verhindert, dass bei kuenftigen Refactorings die per-Phase-Strength-
    Orakelintegration fuer einzelne Phasen stillschweigend entfernt oder umgangen wird.
    """

    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id=phase_id, name=phase_id)
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 0.55,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
    }

    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"natuerlichkeit": 1.0, "transparenz": 1.0, "vocal_quality": 1.0},
        rest_ctx=50.0,
    )

    profile = kwargs.get("phase_strength_oracle_profile")
    assert isinstance(profile, dict)
    assert kwargs.get("phase_strength_oracle_class") == profile.get("oracle_class")
    assert isinstance(profile.get("hard_caps"), dict)
    assert 0.0 <= float(kwargs.get("strength", 0.0)) <= 1.0  # type: ignore[arg-type]


def _make_dummy_uv3_for_runtime_hook():
    class _DummyUV3:
        _PHASE_INTERVENTION_CLASS = UnifiedRestorerV3._PHASE_INTERVENTION_CLASS
        _conductor_strength_hints = {}
        _vintage_phase_strength_caps = {}
        _restoration_context = {
            "frisson_zones": [],
            "passaggio_zones": [],
            "vibrato_zones": [],
            "breath_segments": [],
            "soft_saturation_severity": 0.0,
            "soft_saturation_preserve": False,
            "primary_material": "vinyl",
        }
        _global_conservative_scalar = 1.0
        _panns_singing = 0.62
        _pmgg_ceiling_capped_targets = {}
        _song_goal_targets = {}

        def is_studio_mode(self):
            return False

        def _get_phase_calibration_scalar(self, _phase_id, _song_calibration):
            return 1.0

        def _resolve_phase_strength_oracle_rollout_mode(self, kwargs):
            return UnifiedRestorerV3._resolve_phase_strength_oracle_rollout_mode(self, kwargs)  # type: ignore[arg-type]

        def _canonical_phase_context_kwargs(self):
            return UnifiedRestorerV3._canonical_phase_context_kwargs(self)  # type: ignore[arg-type]

    return _DummyUV3()


def test_uv3_runtime_context_applies_o8_cap_for_spectral_family():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_23_spectral_repair", name="Spectral Repair")
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 1.0,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "defect_scores": {"noise": 1.0},
        "transfer_chain": ["vinyl", "cassette", "mp3_low"],
        "material_confidence": 0.95,
    }

    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"transparenz": 1.4},
        rest_ctx=20.0,
    )

    profile = kwargs.get("phase_strength_oracle_profile", {})
    chain_factor = float(profile.get("hard_caps", {}).get("chain_factor", 1.0))  # type: ignore[attr-defined]
    assert kwargs.get("phase_strength_oracle_class") == "O8_generative_repair"
    assert float(kwargs.get("strength", 0.0)) <= 0.78 + 1e-9  # type: ignore[arg-type]
    assert profile.get("hard_caps", {}).get("max_strength") == pytest.approx(0.78 * (0.75 + 0.25 * chain_factor))  # type: ignore[attr-defined]
    assert float(profile.get("hard_caps", {}).get("chain_factor", 1.0)) < 0.9  # type: ignore[attr-defined]


def test_uv3_runtime_context_applies_o10_cap_for_output_family():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_40_loudness_normalization", name="Loudness")
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 1.0,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "defect_scores": {"loudness": 1.0},
        "transfer_chain": ["vinyl", "cassette", "mp3_low"],
        "material_confidence": 0.95,
    }

    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"natuerlichkeit": 1.2},
        rest_ctx=20.0,
    )

    profile = kwargs.get("phase_strength_oracle_profile", {})
    chain_factor = float(profile.get("hard_caps", {}).get("chain_factor", 1.0))  # type: ignore[attr-defined]
    assert kwargs.get("phase_strength_oracle_class") == "O10_output"
    assert float(kwargs.get("strength", 0.0)) <= 0.72 + 1e-9  # type: ignore[arg-type]
    assert profile.get("hard_caps", {}).get("max_strength") == pytest.approx(0.72 * (0.75 + 0.25 * chain_factor))  # type: ignore[attr-defined]
    assert float(profile.get("hard_caps", {}).get("chain_factor", 1.0)) < 0.9  # type: ignore[attr-defined]


def test_uv3_runtime_context_keeps_explicit_strength_but_sets_oracle_telemetry():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_19_de_esser", name="De-Esser")
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 0.33,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "defect_scores": {"sibilance": 1.0},
    }

    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=True,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"artikulation": 1.6, "vocal_quality": 1.4},
        rest_ctx=20.0,
    )

    assert float(kwargs.get("strength", 0.0)) == pytest.approx(0.33)  # type: ignore[arg-type]
    assert kwargs.get("phase_strength_oracle_class") == "O7_vocal_articulation"
    assert "phase_strength_oracle_profile" in kwargs


def test_uv3_runtime_context_keeps_explicit_strength_for_o10_output_phase():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_40_loudness_normalization", name="Loudness")
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 0.41,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "defect_scores": {"loudness": 1.0},
    }

    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=True,
        team_context_enabled=False,
        song_calibration=None,
        song_goal_weights={"natuerlichkeit": 1.2},
        rest_ctx=20.0,
    )

    assert float(kwargs.get("strength", 0.0)) == pytest.approx(0.41)  # type: ignore[arg-type]
    assert kwargs.get("phase_strength_oracle_class") == "O10_output"
    assert "phase_strength_oracle_profile" in kwargs


def test_uv3_runtime_context_uq_scalar_damps_non_explicit_strength_for_low_confidence():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_07_harmonic_restoration", name="Harmonic")
    audio = np.zeros(48_000, dtype=np.float32)

    kwargs_high = {
        "strength": 0.80,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "material_confidence": 0.95,
        "song_calibration_confidence": 0.95,
    }
    wet_high = UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs_high,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration={"pipeline_confidence": 0.95},
        song_goal_weights={"brillanz": 1.2},
        rest_ctx=35.0,
    )

    kwargs_low = {
        "strength": 0.80,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "material_confidence": 0.35,
        "song_calibration_confidence": 0.35,
    }
    wet_low = UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs_low,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration={"pipeline_confidence": 0.35},
        song_goal_weights={"brillanz": 1.2},
        rest_ctx=35.0,
    )

    assert float(kwargs_low.get("uq_confidence_scalar", 1.0)) < float(kwargs_high.get("uq_confidence_scalar", 1.0))  # type: ignore[arg-type]
    assert float(kwargs_low.get("strength", 0.0)) < float(kwargs_high.get("strength", 0.0))  # type: ignore[arg-type]
    assert float(wet_low) < float(wet_high)


def test_uv3_runtime_context_uq_scalar_keeps_explicit_strength_but_reduces_wetdry():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_48_stereo_width_enhancer", name="Stereo Width")
    audio = np.zeros(48_000, dtype=np.float32)

    kwargs = {
        "strength": 0.52,
        "sample_rate": 48_000,
        "material": "vinyl",
        "material_type": "vinyl",
        "material_confidence": 0.30,
        "song_calibration_confidence": 0.30,
    }
    wet_dry = UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=True,
        team_context_enabled=False,
        song_calibration={"pipeline_confidence": 0.30},
        song_goal_weights={"spatial_depth": 1.2},
        rest_ctx=25.0,
    )

    assert float(kwargs.get("strength", 0.0)) == pytest.approx(0.52)  # type: ignore[arg-type]
    assert float(kwargs.get("uq_confidence_scalar", 1.0)) < 1.0  # type: ignore[arg-type]
    assert float(wet_dry) < 1.0


def test_uv3_runtime_context_phase23_pre_hallucination_cap_for_tape_chain():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_23_spectral_repair", name="Spectral Repair")
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 0.92,
        "sample_rate": 48_000,
        "material": "cassette",
        "material_type": "cassette",
        "transfer_chain": ["vinyl", "cassette", "mp3_low"],
        "material_confidence": 0.85,
        "song_calibration_confidence": 0.85,
    }

    wet_dry = UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration={"pipeline_confidence": 0.85},
        song_goal_weights={"transparenz": 1.2},
        rest_ctx=40.0,
    )

    assert kwargs.get("phase23_pre_hallucination_cap") == pytest.approx(0.74)
    assert float(kwargs.get("strength", 1.0)) <= 0.74 + 1e-9  # type: ignore[arg-type]
    assert float(kwargs.get("uq_confidence_value", 0.0)) == pytest.approx(0.85)  # type: ignore[arg-type]
    assert 0.0 < float(wet_dry) <= 1.0


def test_uv3_runtime_context_phase23_pre_hallucination_cap_stricter_when_low_confidence():
    uv3 = _make_dummy_uv3_for_runtime_hook()
    phase_meta = SimpleNamespace(phase_id="phase_23_spectral_repair", name="Spectral Repair")
    audio = np.zeros(48_000, dtype=np.float32)
    kwargs = {
        "strength": 0.92,
        "sample_rate": 48_000,
        "material": "tape",
        "material_type": "tape",
        "transfer_chain": ["tape", "mp3_low"],
        "material_confidence": 0.55,
        "song_calibration_confidence": 0.55,
    }

    UnifiedRestorerV3._prepare_profiled_phase_runtime_context(
        uv3,
        phase_meta,
        audio,
        kwargs,
        strength_explicit=False,
        team_context_enabled=False,
        song_calibration={"pipeline_confidence": 0.55},
        song_goal_weights={"transparenz": 1.2},
        rest_ctx=40.0,
    )

    assert kwargs.get("phase23_pre_hallucination_cap") == pytest.approx(0.66)
    assert float(kwargs.get("strength", 1.0)) <= 0.66 + 1e-9  # type: ignore[arg-type]
    assert float(kwargs.get("uq_confidence_value", 0.0)) == pytest.approx(0.55)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "phase_family,expected_class,param_assertion,max_strength_cap",
    [
        ("general", "O1_general_repair", "ratio", 0.86),
        ("subtractive_cleanup", "O2_subtractive", "threshold", 0.92),
        ("tonal_restoration", "O3_spectral_balance", "spectral", 0.88),
        ("time_pitch_transport", "O4_time_pitch", "drive", 0.82),
        ("stereo_phase_geometry", "O5_stereo_field", "band", 0.78),
        ("dynamics_control", "O6_dynamics", "ratio", 0.84),
        ("sibilance_control", "O7_vocal_articulation", "threshold", 0.90),
        ("spectral_restoration", "O8_generative_repair", "spectral", 0.78),
        ("harmonic_noise_control", "O9_periodic_cancellation", "threshold_drive", 0.82),
        ("finalizer_output", "O10_output", "ratio", 0.72),
    ],
)
def test_oracle_o1_o10_class_guards(phase_family, expected_class, param_assertion, max_strength_cap):
    prof = resolve_phase_strength_oracle(
        phase_id="phase_test",
        phase_family=phase_family,
        current_strength=0.40,
        goal_gaps={"natuerlichkeit": 0.22, "transparenz": 0.18, "vocal_quality": 0.15},
        goal_weights={"natuerlichkeit": 1.2, "transparenz": 1.1, "vocal_quality": 1.3},
        defect_scores={"d1": 0.8},
        locality_factor=1.0,
        restorability_score=25.0,
        material_key="vinyl",
        song_calibration_profile=None,
        panns_singing=0.5,
    )

    assert prof.oracle_class == expected_class
    assert 0.0 <= prof.control_strength <= max_strength_cap + 1e-9
    assert 0.30 <= prof.wet_mix <= 0.98

    if param_assertion == "ratio":
        assert prof.ratio is not None
    elif param_assertion == "threshold":
        assert prof.threshold_db is not None
    elif param_assertion == "drive":
        assert prof.drive is not None
    elif param_assertion == "band":
        assert prof.band_profile is not None
    elif param_assertion == "spectral":
        assert prof.eq_gain_db is not None
        assert prof.drive is not None
        assert prof.band_profile is not None
    elif param_assertion == "threshold_drive":
        assert prof.threshold_db is not None
        assert prof.drive is not None
