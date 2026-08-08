from __future__ import annotations

import json
from pathlib import Path

import pytest

import audit.build_voice_first_runtime_report as runtime_report_module
from audit.build_voice_first_runtime_report import build_guard_coverage_manifest, build_runtime_report


@pytest.mark.unit
def test_build_runtime_report_passes_when_all_voice_first_blockers_true(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.42,
                        "prior_drift_ratio": 0.12,
                        "decision_stability_score": 0.91,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    assert report["required_passed"] == 3
    assert report["required_total"] == 3


def test_build_runtime_report_fails_when_blocker_missing(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.35,
                        "prior_drift_ratio": 0.08,
                        "decision_stability_score": 0.92,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is False
    assert report["required_passed"] < report["required_total"]


def test_build_runtime_report_fails_when_decision_quality_guard_breaks(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.0,
                        "prior_drift_ratio": 0.31,
                        "decision_stability_score": 0.71,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is False
    checks = report.get("checks", [])
    dq = next((c for c in checks if c.get("id") == "decision_quality_learning_guard"), None)
    assert isinstance(dq, dict)
    assert dq.get("passed") is False


def test_build_runtime_report_uses_guard_class_policy_matrix(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.20,
                        "prior_drift_ratio": 0.26,
                        "decision_stability_score": 0.79,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    original_guard_class = dict(runtime_report_module._GUARD_TO_CLASS)
    try:
        runtime_report_module._GUARD_TO_CLASS["decision_quality_learning_guard"] = "high"
        report = build_runtime_report(str(snapshot), str(out))
    finally:
        runtime_report_module._GUARD_TO_CLASS.clear()
        runtime_report_module._GUARD_TO_CLASS.update(original_guard_class)

    assert report["compliance_ok"] is True


def test_build_runtime_report_bridge_export_fidelity_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "fidelity_guards": {
                            "hf_hallucination_guard": {
                                "guard_fired_count": 99,
                                "max_delta_ratio": 0.99,
                            },
                            "spectral_tilt_guard": {
                                "guard_fired_count": 99,
                                "max_deviation_db_per_oct": 9.9,
                            },
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    fidelity = next((c for c in checks if c.get("id") == "bridge_export_fidelity_guard"), None)
    assert isinstance(fidelity, dict)
    assert fidelity.get("required") is False
    assert fidelity.get("passed") is False


def test_build_runtime_report_interaction_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "interaction_guard": {
                            "interaction_rollbacks": [{"phase_id": "phase_03"}] * 20,
                            "pipeline_stopped_early": True,
                            "stft_phases_count": 12,
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    interaction = next((c for c in checks if c.get("id") == "interaction_guard_runtime"), None)
    assert isinstance(interaction, dict)
    assert interaction.get("required") is False
    assert interaction.get("passed") is False


def test_build_runtime_report_bridge_import_status_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "startup_check_status": {
                            "available": False,
                            "failures": 2,
                            "last_error": "ImportError: startup_model_check missing",
                        },
                        "audio_exporter_status": {
                            "available": True,
                            "failures": 1,
                            "last_error": "ImportError: audio_exporter missing",
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    bridge_import = next((c for c in checks if c.get("id") == "bridge_import_status_runtime"), None)
    assert isinstance(bridge_import, dict)
    assert bridge_import.get("required") is False
    assert bridge_import.get("passed") is False


def test_build_runtime_report_dsp_ml_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "ml_guard_events": [{"phase_id": "phase_23"}] * 100,
                        "ml_fallbacks_used": [{"phase": "phase_24", "fallback": "dsp"}] * 40,  # §V6: logger.warning handled at call site
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    dsp_ml = next((c for c in checks if c.get("id") == "dsp_ml_guard_runtime"), None)
    assert isinstance(dsp_ml, dict)
    assert dsp_ml.get("required") is False
    assert dsp_ml.get("passed") is False


def test_build_runtime_report_vocal_perceptual_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "vocal_no_harm_rollback": True,
                        "vocal_no_harm_reason": "identity_loss",
                        "vocal_no_harm_gate": {
                            "requires_rollback": True,
                        },
                        "holistic_perceptual_gate": {
                            "passed": False,
                            "artifact_freedom": 0.80,
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    vp = next((c for c in checks if c.get("id") == "vocal_perceptual_runtime"), None)
    assert isinstance(vp, dict)
    assert vp.get("required") is False
    assert vp.get("passed") is False


def test_build_runtime_report_temporal_stereo_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "temporal_continuity": {
                            "phase_03": {
                                "critical": True,
                                "gain_step_db": 2.3,
                            }
                        },
                        "mono_compatibility_warning": True,
                        "onset_shift_ok": False,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    ts = next((c for c in checks if c.get("id") == "temporal_stereo_runtime"), None)
    assert isinstance(ts, dict)
    assert ts.get("required") is False
    assert ts.get("passed") is False


def test_build_runtime_report_recovery_execution_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "fail_reasons": ["artifact_freedom_low"],
                        "graceful_stop": True,
                        "recovery_certainty": {
                            "uncertainty_index": 0.95,
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    re_guard = next((c for c in checks if c.get("id") == "recovery_execution_runtime"), None)
    assert isinstance(re_guard, dict)
    assert re_guard.get("required") is False
    assert re_guard.get("passed") is False


def test_build_runtime_report_team_goal_guard_is_non_blocking(tmp_path: Path) -> None:
    snapshot = tmp_path / "current_voice_first_audit.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "vocal_quality_check": {
                        "vqi": True,
                        "formant_integrity": True,
                        "vibrato_depth_preserved": True,
                        "micro_dynamic_correlation": True,
                    },
                    "decision_quality": {
                        "learning_applied": True,
                        "causal_credit_confidence": 0.30,
                        "prior_drift_ratio": 0.10,
                        "decision_stability_score": 0.90,
                    },
                    "metadata": {
                        "team_coordination": {
                            "event_count": 99,
                        },
                        "goal_recovery": {
                            "attempted": True,
                            "resolved": False,
                            "final_violations": ["transparenz", "waerme"],
                        },
                        "length_corrections": ["phase_03", "phase_09", "phase_12", "phase_29", "phase_55"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runtime_report.json"

    report = build_runtime_report(str(snapshot), str(out))

    assert report["compliance_ok"] is True
    checks = report.get("checks", [])
    team_goal = next((c for c in checks if c.get("id") == "team_goal_runtime"), None)
    assert isinstance(team_goal, dict)
    assert team_goal.get("required") is False
    assert team_goal.get("passed") is False


def test_build_guard_coverage_manifest_contains_expected_checks(tmp_path: Path) -> None:
    out = tmp_path / "guard_coverage_manifest.json"

    manifest = build_guard_coverage_manifest(str(out))

    assert out.exists()
    assert manifest.get("schema") == "voice-first-guard-coverage-manifest.v1"
    checks = manifest.get("checks", [])
    assert isinstance(checks, list)
    ids = {item.get("id") for item in checks if isinstance(item, dict)}
    assert "decision_quality_learning_guard" in ids
    assert "team_goal_runtime" in ids
    assert "recovery_execution_runtime" in ids
    assert "bridge_import_status_runtime" in ids
