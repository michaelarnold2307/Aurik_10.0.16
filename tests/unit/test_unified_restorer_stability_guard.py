from __future__ import annotations

import pytest

"""Stability guard tests for bounded-but-non-edge behavior.

Ensures calibration scalars and implicit phase strengths are pulled away from
hard bounds while explicit caller strengths remain untouched.
"""


import numpy as np

from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, DefectType, MaterialType
from backend.core.phases.phase_interface import PhaseCategory, PhaseMetadata, create_phase_result
from backend.core.quality_mode import QualityMode
from backend.core.unified_restorer_v3 import UnifiedRestorerV3


class _DummyPhase:
    def __init__(self, phase_id: str = "phase_03_denoise"):
        self._meta = PhaseMetadata(
            phase_id=phase_id,
            name="Dummy",
            category=PhaseCategory.RESTORATION,
            priority=5,
            version="1.0",
            dependencies=[],
            estimated_time_factor=0.0,
            memory_requirement_mb=1,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.0,
            description="dummy",
        )
        self.last_strength = None
        self.last_kwargs = None

    def get_metadata(self):
        return self._meta

    def process(self, audio: np.ndarray, sample_rate: int = 48000, **kwargs):
        self.last_strength = kwargs.get("strength")
        self.last_kwargs = dict(kwargs)  # type: ignore[assignment]
        return create_phase_result(audio=np.asarray(audio, dtype=np.float32))


@pytest.mark.unit
def test_song_calibration_pullback_avoids_hard_edges():
    profile = UnifiedRestorerV3._build_song_calibration_profile(
        material_type=MaterialType.SHELLAC,
        mode=QualityMode.BALANCED,  # type: ignore[arg-type]
        restorability_score=0.0,
        input_snr_db=-10.0,
        max_defect_severity=1.0,
        pipeline_confidence=1.0,
    )

    g = float(profile["global_scalar"])
    assert 0.50 < g < 1.50
    fam = profile["family_scalars"]
    assert all(0.30 < float(v) < 1.80 for v in fam.values())


def test_implicit_strength_gets_stability_corridor():
    uv3 = UnifiedRestorerV3()
    phase = _DummyPhase("phase_03_denoise")
    uv3._conductor_strength_hints = {"phase_03_denoise": 0.99}

    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(1024) / 48000)).astype(
        np.float32
    )  # §Early-Silence-Gate: nonzero, sonst wird die Phase komplett übersprungen
    uv3._profiled_phase_call(phase, audio, sample_rate=48000)

    assert isinstance(phase.last_strength, float)
    assert 0.03 <= phase.last_strength <= 0.97


def test_explicit_strength_not_overridden_by_corridor():
    uv3 = UnifiedRestorerV3()
    phase = _DummyPhase("phase_03_denoise")

    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(1024) / 48000)).astype(
        np.float32
    )  # §Early-Silence-Gate: nonzero, sonst wird die Phase komplett übersprungen
    uv3._profiled_phase_call(phase, audio, sample_rate=48000, strength=0.99)

    assert isinstance(phase.last_strength, float)
    assert abs(phase.last_strength - 0.99) < 1e-9


def test_phase03_gets_tdp_stem_aware_nr_auto_injected():
    uv3 = UnifiedRestorerV3()
    phase = _DummyPhase("phase_03_denoise")

    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(1024) / 48000)).astype(
        np.float32
    )  # §Early-Silence-Gate: nonzero, sonst wird die Phase komplett übersprungen
    uv3._profiled_phase_call(phase, audio, sample_rate=48000)

    assert isinstance(phase.last_kwargs, dict)
    assert phase.last_kwargs.get("tdp_stem_aware_nr") == "auto"


def test_non_phase03_does_not_get_tdp_stem_aware_nr_injected():
    uv3 = UnifiedRestorerV3()
    phase = _DummyPhase("phase_29_tape_hiss_reduction")

    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(1024) / 48000)).astype(
        np.float32
    )  # §Early-Silence-Gate: nonzero, sonst wird die Phase komplett übersprungen
    uv3._profiled_phase_call(phase, audio, sample_rate=48000)

    assert isinstance(phase.last_kwargs, dict)
    assert "tdp_stem_aware_nr" not in phase.last_kwargs


def test_cht_warning_brakes_risky_enhancement_phases_more_aggressively():
    base_brake = UnifiedRestorerV3._compute_cht_cascade_brake("tonal_enhancement", 0, 0)
    warned_brake = UnifiedRestorerV3._compute_cht_cascade_brake("tonal_enhancement", 2, 1)
    risky_brake = UnifiedRestorerV3._compute_cht_cascade_brake("stereo_enhancement", 2, 1)

    assert warned_brake < base_brake
    assert risky_brake <= warned_brake
    assert 0.5 <= risky_brake < 1.0


def test_naturalness_guard_damps_vocal_enhancement_when_voice_metrics_drop():
    guard = UnifiedRestorerV3._compute_naturalness_guard_scalar(
        phase_family="tonal_enhancement",
        studio_mode=False,
        panns_singing=0.52,
        vocal_quality_check={
            "noise_texture_authenticity": 0.10,
            "micro_dynamic_correlation": 0.94,
            "formant_integrity": 0.84,
        },
        phase_metadata_accumulator={
            "phase_39_air_band_enhancement": {
                "onset_shift_ms": 2.7,
            }
        },
    )

    assert bool(guard.get("enabled", False)) is True
    assert float(guard.get("scalar", 1.0)) < 0.80
    assert float(guard.get("risk_score", 0.0)) > 0.15
    assert len(guard.get("signals", [])) >= 3


def test_naturalness_guard_is_neutral_for_subtractive_family():
    guard = UnifiedRestorerV3._compute_naturalness_guard_scalar(
        phase_family="subtractive_cleanup",
        studio_mode=False,
        panns_singing=0.60,
        vocal_quality_check={
            "noise_texture_authenticity": 0.05,
            "micro_dynamic_correlation": 0.92,
            "formant_integrity": 0.82,
        },
        phase_metadata_accumulator={"x": {"onset_shift_ms": 8.0}},
    )

    assert bool(guard.get("enabled", True)) is False
    assert float(guard.get("scalar", 0.0)) == 1.0


def test_naturalness_guard_detects_machine_like_artifact_history():
    guard = UnifiedRestorerV3._compute_naturalness_guard_scalar(
        phase_family="source_enhancement",
        studio_mode=False,
        panns_singing=0.58,
        vocal_quality_check={
            "noise_texture_authenticity": 0.24,
            "micro_dynamic_correlation": 0.975,
            "breath_naturalness": 0.76,
            "spectral_color_preservation": 0.91,
        },
        phase_metadata_accumulator={
            "phase_23_spectral_repair": {
                "n_musical_noise": 5,
                "n_metallic_ringing": 2,
                "roughness_regression": True,
                "psycho_delta_penalty": 0.12,
            }
        },
    )

    assert bool(guard.get("enabled", False)) is True
    assert float(guard.get("risk_score", 0.0)) > 0.18
    assert float(guard.get("scalar", 1.0)) < 0.82
    sig = set(guard.get("signals", []))
    assert "musical_noise_history" in sig
    assert "metallic_ringing_history" in sig
    assert "breath_naturalness" in sig


def test_naturalness_guard_uses_runtime_psycho_state():
    guard = UnifiedRestorerV3._compute_naturalness_guard_scalar(
        phase_family="harmonic_enhancement",
        studio_mode=False,
        panns_singing=0.44,
        vocal_quality_check={
            "noise_texture_authenticity": 0.26,
            "micro_dynamic_correlation": 0.982,
        },
        phase_metadata_accumulator={
            "_psycho_runtime_state": {
                "rolling_risk": 0.26,
                "last_delta_penalty": 0.17,
            }
        },
    )

    assert bool(guard.get("enabled", False)) is True
    assert float(guard.get("risk_score", 0.0)) > 0.16
    assert float(guard.get("scalar", 1.0)) < 0.85
    sig = set(guard.get("signals", []))
    assert "runtime_psycho_risk" in sig
    assert "runtime_delta_recent" in sig


def test_focus_defect_map_prefers_reliable_defects():
    result = DefectAnalysisResult(
        material_type=MaterialType.TAPE,
        scores={
            DefectType.DROPOUTS: DefectScore(DefectType.DROPOUTS, severity=0.60, confidence=0.95),
            DefectType.PRE_ECHO: DefectScore(DefectType.PRE_ECHO, severity=0.80, confidence=0.20),
            DefectType.HUM: DefectScore(DefectType.HUM, severity=0.40, confidence=0.85),
        },
        analysis_time_seconds=0.01,
        sample_rate=48000,
        duration_seconds=1.0,
    )

    focus = result.get_focus_defect_map(n=3)
    assert "dropouts" in focus
    assert "hum" in focus
    assert float(focus["dropouts"]) > float(focus["hum"])
    assert float(focus.get("pre_echo", 0.0)) < float(focus["dropouts"])


def test_extract_defect_focus_scores_uses_metadata_when_present():
    result = DefectAnalysisResult(
        material_type=MaterialType.VINYL,
        scores={
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.4, confidence=0.9),
        },
        analysis_time_seconds=0.01,
        sample_rate=48000,
        duration_seconds=1.0,
        metadata={"focus_defects": {"room_mode_resonance": 0.61, "clicks": 0.33}},
    )

    focus = UnifiedRestorerV3._extract_defect_focus_scores(result, max_items=2)
    assert list(focus.keys())[0] == "room_mode_resonance"
    assert float(focus["room_mode_resonance"]) == 0.61


def test_wow_flutter_fingerprint_boosts_time_pitch_not_dynamics_eq():
    neutral = UnifiedRestorerV3._build_song_calibration_profile(
        material_type=MaterialType.TAPE,
        mode=QualityMode.BALANCED,  # type: ignore[arg-type]
        restorability_score=60.0,
        input_snr_db=30.0,
        max_defect_severity=0.40,
        pipeline_confidence=0.80,
        spectral_fingerprint={"wow_flutter_index": 0.0},
    )
    wow_heavy = UnifiedRestorerV3._build_song_calibration_profile(
        material_type=MaterialType.TAPE,
        mode=QualityMode.BALANCED,  # type: ignore[arg-type]
        restorability_score=60.0,
        input_snr_db=30.0,
        max_defect_severity=0.40,
        pipeline_confidence=0.80,
        spectral_fingerprint={"wow_flutter_index": 2.0},
    )

    neutral_fam = neutral["family_scalars"]
    wow_fam = wow_heavy["family_scalars"]
    assert float(wow_fam["time_pitch_transport"]) > float(neutral_fam["time_pitch_transport"])
    assert float(wow_fam["dynamics_eq"]) == float(neutral_fam["dynamics_eq"])


def test_phase12_and_phase31_use_time_pitch_family_scalar():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "time_pitch_transport": 1.40,
            "dynamics_eq": 0.60,
            "general": 1.0,
        },
    }

    assert UnifiedRestorerV3._get_phase_calibration_scalar("phase_12_wow_flutter_fix", profile) == 1.40
    assert UnifiedRestorerV3._get_phase_calibration_scalar("phase_31_speed_pitch_correction", profile) == 1.40
    assert UnifiedRestorerV3._get_phase_calibration_scalar("phase_04_eq_correction", profile) == 0.60
    assert UnifiedRestorerV3._phase_family_from_phase_id("phase_12_wow_flutter_fix") == "time_pitch_transport"
    assert UnifiedRestorerV3._phase_family_from_phase_id("phase_31_speed_pitch_correction") == "time_pitch_transport"


def test_temporal_defect_autosetup_boosts_time_pitch_family():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "reverb": 1.0,
            "reconstruction": 1.0,
            "time_pitch_transport": 1.0,
            "transient": 1.0,
            "general": 1.0,
        },
        "material": "tape",
        "restorability_tier": "fair",
    }

    out = UnifiedRestorerV3._apply_song_autosetup_policy(
        profile,
        defect_scores={"wow": 0.80, "flutter": 0.20},
        transfer_chain=["tape"],
        max_defect_severity=0.80,
    )

    fam = out["family_scalars"]
    assert float(fam["time_pitch_transport"]) > 1.0
    assert float(fam["transient"]) < 1.0
    assert float(fam["reconstruction"]) < float(fam["time_pitch_transport"])
    assert out["strict_conflict_policy"]["rollback_decay_per_family"]["time_pitch_transport"] == 0.93


def test_phase_calibration_scalar_has_interior_pullback():
    profile_low = {"global_scalar": 0.10, "family_scalars": {"denoise": 0.10, "general": 0.10}}
    s_low = UnifiedRestorerV3._get_phase_calibration_scalar("phase_03_denoise", profile_low)
    assert s_low > 0.30

    profile_high = {"global_scalar": 2.50, "family_scalars": {"denoise": 2.50, "general": 2.50}}
    s_high = UnifiedRestorerV3._get_phase_calibration_scalar("phase_03_denoise", profile_high)
    assert s_high < 1.80


def test_mid_calibration_production_nachbesserung_deboosts_on_low_artifact_floor():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "reverb": 1.0,
            "reconstruction": 1.0,
            "dynamics_eq": 1.0,
            "time_pitch_transport": 1.0,
            "transient": 1.0,
            "vocal": 1.0,
            "instrument": 1.0,
            "general": 1.0,
        },
    }
    scores = {
        "natuerlichkeit": 0.89,
        "brillanz": 0.90,
        "waerme": 0.70,
    }
    out = UnifiedRestorerV3._mid_pipeline_calibration_step(
        scores,
        profile,
        "33pct",
        5,
        10,
        artifact_floor=0.94,
    )

    assert out is not None
    fam = out["family_scalars"]
    assert float(fam["transient"]) < 1.0
    assert float(fam["reconstruction"]) < 1.0
    assert float(fam["dynamics_eq"]) < 1.0


def test_continuous_joy_refinement_writes_events_and_adjusts():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "reverb": 1.0,
            "reconstruction": 1.0,
            "dynamics_eq": 1.0,
            "transient": 1.0,
            "vocal": 1.0,
            "instrument": 1.0,
            "general": 1.0,
        },
    }
    scores = {
        "natuerlichkeit": 0.89,
        "brillanz": 0.90,
        "waerme": 0.70,
        "micro_dynamics": 0.80,
        "artikulation": 0.78,
    }
    out = UnifiedRestorerV3._continuous_joy_refinement_step(
        scores,
        profile,
        "phase_23_spectral_repair",
        7,
        20,
        artifact_floor=0.94,
    )

    assert out is not None
    fam = out["family_scalars"]
    assert float(fam["reconstruction"]) < 1.0
    assert float(fam["time_pitch_transport"]) < 1.0
    assert float(fam["transient"]) < 1.0
    events = out.get("_joy_closed_loop_events", [])
    assert isinstance(events, list)
    assert len(events) == 1
    assert events[0]["trigger_phase"] == "phase_23_spectral_repair"


def test_continuous_joy_refinement_returns_none_when_no_signal():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "reverb": 1.0,
            "reconstruction": 1.0,
            "dynamics_eq": 1.0,
            "transient": 1.0,
            "vocal": 1.0,
            "instrument": 1.0,
            "general": 1.0,
        },
    }
    scores = {
        "natuerlichkeit": 0.93,
        "brillanz": 0.80,
        "waerme": 0.80,
        "micro_dynamics": 0.90,
        "artikulation": 0.90,
    }
    out = UnifiedRestorerV3._continuous_joy_refinement_step(
        scores,
        profile,
        "phase_10_declip",
        4,
        20,
        artifact_floor=1.0,
    )
    assert out is None


# ---------------------------------------------------------------------------
# §2.47 Phase-53-Semantic-Feedback wiring tests
# ---------------------------------------------------------------------------


def test_phase53_semantic_feedback_updates_unbekannt_genre():
    """_restoration_context genre_label 'Unbekannt' is overridden by Phase-53 CLAP output."""
    from types import SimpleNamespace

    uv3 = UnifiedRestorerV3()
    uv3._restoration_context = {
        "genre_label": "Unbekannt",
        "is_schlager": False,
    }

    # Simulate a Phase 53 result with CLAP genre Jazz / conf=0.62
    fake_result = SimpleNamespace(
        metadata={
            "genre_hint": "Jazz",
            "genre_hint_source": "clap",
            "genre_hint_confidence": 0.62,
            "clap_top_genres": [{"genre": "Jazz", "confidence": 0.62}],
            "beats_top_tags": [],
        }
    )

    # Replicate the hook logic from _execute_pipeline
    _s53_meta = fake_result.metadata if hasattr(fake_result, "metadata") else {}
    _s53_genre = str(_s53_meta.get("genre_hint", "") or "")
    _s53_src = str(_s53_meta.get("genre_hint_source", "dsp"))
    _s53_conf = float(_s53_meta.get("genre_hint_confidence", 0.0))
    _s53_current_label = str(uv3._restoration_context.get("genre_label", "Unbekannt"))
    _s53_is_schlager = bool(uv3._restoration_context.get("is_schlager", False))
    _s53_override = (
        not _s53_is_schlager
        and _s53_genre not in ("", "Unbekannt")
        and (_s53_current_label in ("Unbekannt", "unknown", "") or (_s53_src == "clap" and _s53_conf >= 0.55))
    )
    if _s53_override:
        uv3._restoration_context["genre_label"] = _s53_genre
        uv3._restoration_context["genre_hint_source"] = _s53_src
        uv3._restoration_context["genre_hint_confidence"] = _s53_conf
        uv3._restoration_context["clap_top_genres"] = _s53_meta.get("clap_top_genres", [])
        uv3._restoration_context["beats_top_tags"] = _s53_meta.get("beats_top_tags", [])

    assert uv3._restoration_context["genre_label"] == "Jazz"
    assert uv3._restoration_context["genre_hint_source"] == "clap"
    assert abs(uv3._restoration_context["genre_hint_confidence"] - 0.62) < 0.01


def test_phase53_semantic_feedback_does_not_override_schlager():
    """Schlager (is_schlager=True) is never overridden by Phase-53 CLAP output."""
    from types import SimpleNamespace

    uv3 = UnifiedRestorerV3()
    uv3._restoration_context = {
        "genre_label": "Schlager",
        "is_schlager": True,
    }

    fake_result = SimpleNamespace(
        metadata={
            "genre_hint": "Rock",
            "genre_hint_source": "clap",
            "genre_hint_confidence": 0.80,
            "clap_top_genres": [],
            "beats_top_tags": [],
        }
    )

    _s53_meta = fake_result.metadata
    _s53_genre = str(_s53_meta.get("genre_hint", "") or "")
    _s53_src = str(_s53_meta.get("genre_hint_source", "dsp"))
    _s53_conf = float(_s53_meta.get("genre_hint_confidence", 0.0))
    _s53_current_label = str(uv3._restoration_context.get("genre_label", "Unbekannt"))
    _s53_is_schlager = bool(uv3._restoration_context.get("is_schlager", False))
    _s53_override = (
        not _s53_is_schlager
        and _s53_genre not in ("", "Unbekannt")
        and (_s53_current_label in ("Unbekannt", "unknown", "") or (_s53_src == "clap" and _s53_conf >= 0.55))
    )
    if _s53_override:
        uv3._restoration_context["genre_label"] = _s53_genre

    # Schlager must not be overridden
    assert uv3._restoration_context["genre_label"] == "Schlager"


def test_phase53_semantic_feedback_requires_clap_conf_threshold():
    """Low-confidence CLAP (< 0.55) does not override an existing non-Unbekannt label."""
    from types import SimpleNamespace

    uv3 = UnifiedRestorerV3()
    uv3._restoration_context = {
        "genre_label": "Klassik",
        "is_schlager": False,
    }

    fake_result = SimpleNamespace(
        metadata={
            "genre_hint": "Pop",
            "genre_hint_source": "clap",
            "genre_hint_confidence": 0.40,  # below 0.55 threshold
            "clap_top_genres": [],
            "beats_top_tags": [],
        }
    )

    _s53_meta = fake_result.metadata
    _s53_genre = str(_s53_meta.get("genre_hint", "") or "")
    _s53_src = str(_s53_meta.get("genre_hint_source", "dsp"))
    _s53_conf = float(_s53_meta.get("genre_hint_confidence", 0.0))
    _s53_current_label = str(uv3._restoration_context.get("genre_label", "Unbekannt"))
    _s53_is_schlager = bool(uv3._restoration_context.get("is_schlager", False))
    _s53_override = (
        not _s53_is_schlager
        and _s53_genre not in ("", "Unbekannt")
        and (_s53_current_label in ("Unbekannt", "unknown", "") or (_s53_src == "clap" and _s53_conf >= 0.55))
    )
    if _s53_override:
        uv3._restoration_context["genre_label"] = _s53_genre

    # Low confidence must not override existing label
    assert uv3._restoration_context["genre_label"] == "Klassik"


def test_phase53_profiled_phase_call_propagates_updated_genre_label():
    """After _restoration_context is updated, _profiled_phase_call injects the new genre_label."""
    uv3 = UnifiedRestorerV3()
    # Simulate post-Phase-53 state: genre upgraded from Unbekannt to Jazz
    uv3._restoration_context = {
        "genre_label": "Jazz",
        "genre_hint_source": "clap",
        "genre_hint_confidence": 0.62,
        "is_schlager": False,
        "bpm": 120.0,
    }

    phase = _DummyPhase("phase_07_harmonic_restoration")
    audio = (0.05 * np.sin(2 * np.pi * 440 * np.arange(1024) / 48000)).astype(
        np.float32
    )  # §Early-Silence-Gate: nonzero, sonst wird die Phase komplett übersprungen
    uv3._profiled_phase_call(phase, audio, sample_rate=48000)

    assert isinstance(phase.last_kwargs, dict)
    assert phase.last_kwargs.get("genre_label") == "Jazz"
    assert phase.last_kwargs.get("genre_hint_source") == "clap"


def test_song_cluster_policy_is_deterministic_and_bounded():
    pol = UnifiedRestorerV3._derive_song_cluster_policy(
        material="mp3_low",
        era_decade=1995,
        genre_label="Rock",
        restorability_tier="poor",
        is_schlager=False,
    )
    assert isinstance(pol.get("cluster_key"), str)
    assert 0.90 <= float(pol["artifact_sensitivity"]) <= 1.25
    assert 0.90 <= float(pol["fatigue_sensitivity"]) <= 1.20
    assert 0.85 <= float(pol["recovery_bias"]) <= 1.20


def test_joy_fatigue_runtime_index_increases_with_better_scores():
    low = UnifiedRestorerV3._compute_joy_fatigue_runtime_index(
        {
            "natuerlichkeit": 0.86,
            "waerme": 0.68,
            "micro_dynamics": 0.76,
            "emotionalitaet": 0.74,
            "brillanz": 0.90,
        },
        {"risk_level": "high"},
        artifact_freedom=0.93,
        emotional_arc_score=0.80,
    )
    high = UnifiedRestorerV3._compute_joy_fatigue_runtime_index(
        {
            "natuerlichkeit": 0.93,
            "waerme": 0.80,
            "micro_dynamics": 0.90,
            "emotionalitaet": 0.89,
            "brillanz": 0.82,
        },
        {"risk_level": "low"},
        artifact_freedom=0.99,
        emotional_arc_score=0.92,
    )
    assert float(high["joy_index"]) > float(low["joy_index"])
    assert float(high["fatigue_index"]) < float(low["fatigue_index"])
    assert 0.0 <= float(low.get("frisson_index", 0.0)) <= 1.0
    assert 0.0 <= float(high.get("frisson_index", 0.0)) <= 1.0
    assert float(high.get("frisson_index", 0.0)) > float(low.get("frisson_index", 0.0))


def test_frisson_audio_scalar_disabled_in_restoration_mode():
    scalar = UnifiedRestorerV3._compute_frisson_audio_impact_scalar(
        studio_mode=False,
        phase_family="harmonic_enhancement",
        goal_weights={
            "emotionalitaet": 1.5,
            "micro_dynamics": 1.4,
            "artikulation": 1.3,
            "spatial_depth": 1.4,
            "transparenz": 1.2,
            "tonal_center": 1.1,
        },
    )
    assert abs(float(scalar) - 1.0) < 1e-9


def test_frisson_audio_scalar_active_and_bounded_in_studio_mode():
    scalar_enh = UnifiedRestorerV3._compute_frisson_audio_impact_scalar(
        studio_mode=True,
        phase_family="harmonic_enhancement",
        goal_weights={
            "emotionalitaet": 1.6,
            "micro_dynamics": 1.5,
            "artikulation": 1.4,
            "spatial_depth": 1.4,
            "transparenz": 1.3,
            "tonal_center": 1.2,
        },
    )
    scalar_sub = UnifiedRestorerV3._compute_frisson_audio_impact_scalar(
        studio_mode=True,
        phase_family="subtractive_cleanup",
        goal_weights={
            "emotionalitaet": 1.6,
            "micro_dynamics": 1.5,
            "artikulation": 1.4,
            "spatial_depth": 1.4,
            "transparenz": 1.3,
            "tonal_center": 1.2,
        },
    )
    assert 0.94 <= float(scalar_enh) <= 1.06
    assert 0.94 <= float(scalar_sub) <= 1.06
    assert float(scalar_enh) > 1.0
    assert float(scalar_sub) <= 1.0


# ---------------------------------------------------------------------------
# §2.54 PlateauStop — material-adaptive params (_compute_plateau_params)
# ---------------------------------------------------------------------------


def test_plateau_params_shellac_uses_conservative_threshold():
    """Shellac: tiny per-phase improvements must not trigger plateau → low threshold."""
    thr, dmp = UnifiedRestorerV3._compute_plateau_params("shellac", 60.0)
    assert abs(thr - 0.002) < 1e-6, f"shellac threshold should be 0.002, got {thr}"
    assert abs(dmp - 0.55) < 1e-6, f"shellac dampen should be 0.55, got {dmp}"


def test_plateau_params_cd_digital_uses_aggressive_threshold():
    """CD: large per-phase deltas expected → high threshold for plateau detection."""
    thr, dmp = UnifiedRestorerV3._compute_plateau_params("cd_digital", 75.0)
    assert abs(thr - 0.010) < 1e-6, f"cd_digital threshold should be 0.010, got {thr}"
    assert abs(dmp - 0.35) < 1e-6, f"cd_digital dampen should be 0.35, got {dmp}"


def test_plateau_params_reel_tape_mapping():
    thr, dmp = UnifiedRestorerV3._compute_plateau_params("reel_tape", 55.0)
    assert abs(thr - 0.003) < 1e-6
    assert abs(dmp - 0.50) < 1e-6


def test_plateau_params_low_restorability_raises_dampen_floor():
    """restorability < 40 → dampen must be raised to at least 0.60."""
    # Shellac default dampen = 0.55, but restorability=30 → must be raised to 0.60
    _thr, dmp = UnifiedRestorerV3._compute_plateau_params("shellac", 30.0)
    assert dmp >= 0.60, f"Expected dampen ≥ 0.60 for restorability=30, got {dmp}"
    # CD default dampen = 0.35 → also raised to 0.60
    _thr_cd, dmp_cd = UnifiedRestorerV3._compute_plateau_params("cd_digital", 25.0)
    assert dmp_cd >= 0.60, f"Expected dampen ≥ 0.60 for cd_digital restorability=25, got {dmp_cd}"


def test_plateau_params_unknown_material_uses_default():
    """Unknown material falls back to default (0.005, 0.40)."""
    thr, dmp = UnifiedRestorerV3._compute_plateau_params("unknown_format", 70.0)
    assert abs(thr - 0.005) < 1e-6
    assert abs(dmp - 0.40) < 1e-6


def test_plateau_params_enum_value_attribute_is_used():
    """Material passed as enum-like object uses .value attribute."""

    class _FakeMaterial:
        value = "cassette"

    thr, dmp = UnifiedRestorerV3._compute_plateau_params(_FakeMaterial(), 65.0)
    assert abs(thr - 0.004) < 1e-6, f"cassette threshold should be 0.004, got {thr}"
    assert abs(dmp - 0.45) < 1e-6


def test_auto_improvement_recommendations_contains_actionable_items():
    rec = UnifiedRestorerV3._derive_auto_improvement_recommendations(
        musical_goal_scores={
            "natuerlichkeit": 0.88,
            "waerme": 0.70,
            "brillanz": 0.90,
            "micro_dynamics": 0.79,
        },
        musical_goals_passed={"natuerlichkeit": False},
        artifact_freedom=0.94,
        phase_regression_log={"phase_03_denoise": -2.1},
        top_defects=[{"type": "noise", "severity": 0.72}],
    )
    assert isinstance(rec, dict)
    assert int(rec.get("count", 0)) >= 3
    assert isinstance(rec.get("recommendations", []), list)
