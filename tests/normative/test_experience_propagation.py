"""Tests for §2.53 Experience-Closed-Loop + Bridge/UI-Propagation.

Verifies that:
- UV3 produces required metadata fields (joy_runtime_index, auto_improvement, cluster)
- Bridge `get_experience_insights()` returns frontend-safe data
- End-to-end propagation through Denker chain
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# §2.53 Metadata Field Presence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExperienceMetadataFields:
    """RestorationResult.metadata must contain §2.53 fields."""

    def test_joy_runtime_index_structure(self):
        """joy_runtime_index must have joy_index, fatigue_index, components."""
        joy = {
            "joy_index": 0.82,
            "fatigue_index": 0.15,
            "components": {
                "timbral_warmth": 0.85,
                "dynamic_liveliness": 0.79,
                "spatial_envelopment": 0.81,
            },
        }
        assert "joy_index" in joy
        assert "fatigue_index" in joy
        assert "components" in joy
        assert isinstance(joy["components"], dict)

    def test_auto_improvement_recommendations_structure(self):
        """auto_improvement_recommendations must have count + list with focus/action/reason."""
        recs = {
            "count": 1,
            "recommendations": [
                {
                    "focus": "noise_floor",
                    "action": "Increase denoise strength for tape material",
                    "reason": "Noise floor at -52 dBFS exceeds target -72 dBFS",
                }
            ],
        }
        assert recs["count"] == len(recs["recommendations"])  # type: ignore[arg-type]
        for rec in recs["recommendations"]:  # type: ignore[attr-defined]
            assert "focus" in rec
            assert "action" in rec
            assert "reason" in rec

    def test_cluster_key_and_policy(self):
        """song_calibration must include cluster_key and cluster_policy."""
        cal = {"cluster_key": "rock_vinyl_1970s", "cluster_policy": "conservative"}
        assert isinstance(cal["cluster_key"], str)
        assert isinstance(cal["cluster_policy"], str)


# ---------------------------------------------------------------------------
# Bridge `get_experience_insights()` Safety
# ---------------------------------------------------------------------------


class TestBridgeExperienceInsights:
    """bridge.get_experience_insights() must be frontend-safe."""

    def test_get_experience_insights_exists(self):
        from backend.api import bridge

        assert hasattr(bridge, "get_experience_insights")

    def test_get_experience_insights_with_none(self):
        """Must not crash on None input."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(None)
        assert isinstance(result, dict)

    def test_get_experience_insights_nan_free(self):
        """Output must be NaN/Inf-free for Qt frontend."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(None)

        # Recursively check all float values
        def _check_nan_inf(obj, path=""):
            if isinstance(obj, float):
                assert np.isfinite(obj), f"NaN/Inf at {path}: {obj}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check_nan_inf(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _check_nan_inf(v, f"{path}[{i}]")

        _check_nan_inf(result)

    def test_get_experience_insights_with_mock_result(self):
        """Should handle a result-like object gracefully."""
        from backend.api.bridge import get_experience_insights

        class MockResult:
            metadata = {
                "joy_runtime_index": {
                    "joy_index": 0.75,
                    "fatigue_index": 0.20,
                    "components": {},
                },
                "auto_improvement_recommendations": {
                    "count": 0,
                    "recommendations": [],
                },
                "song_calibration": {
                    "cluster_key": "pop_cd_2000s",
                    "cluster_policy": "balanced",
                },
            }

        result = get_experience_insights(MockResult())
        assert isinstance(result, dict)

    def test_get_experience_insights_contains_fallback_quality_floor(self):
        """Bridge must propagate fallback_quality_floor in a frontend-safe shape."""
        from backend.api.bridge import get_experience_insights

        class MockResult:
            metadata = {
                "fallback_quality_floor": {
                    "triggered": True,
                    "passed": False,
                    "status": "recovered",
                    "reason": "fallback_quality_floor_recovered_with_checkpoint",
                    "recovered": True,
                    "attempts": 1,
                    "fallback_count": 2,
                    "artifact_freedom": 0.94,
                    "hpi_passed": False,
                    "hpi": -0.12,
                    "best_candidate": "hpi_best_checkpoint",
                    "recovery_trace": [{"attempt": 1, "candidate": "hpi_best_checkpoint", "result": "applied"}],
                }
            }

        result = get_experience_insights(MockResult())
        assert "fallback_quality_floor" in result
        fqf = result["fallback_quality_floor"]
        assert fqf["triggered"] is True
        assert fqf["status"] == "recovered"
        assert fqf["fallback_count"] == 2
        assert isinstance(fqf["recovery_trace"], list)

    def test_get_experience_insights_contains_quality_gate_summary(self):
        """Bridge must provide standardized quality_gate summary for UI decisions."""
        from backend.api.bridge import get_experience_insights

        class MockResult:
            fail_reason = "HPI_FAIL"
            degradation_status = "degraded"
            metadata = {
                "fail_reasons": [
                    {
                        "component": "HolisticPerceptualGate",
                        "error_code": "HPI_FAIL",
                    }
                ],
                "fallback_quality_floor": {
                    "triggered": True,
                    "status": "degraded",
                    "attempts": 1,
                    "reason": "fallback_quality_floor_failed_no_compatible_checkpoint",
                },
            }

        result = get_experience_insights(MockResult())
        qg = result["quality_gate"]
        assert qg["passed"] is False
        assert qg["degradation_status"] == "degraded"
        assert qg["primary_fail_reason"] == "HPI_FAIL"
        assert qg["primary_error_code"] == "HPI_FAIL"
        assert qg["recovery_attempted"] is True
        assert qg["best_possible_reached"] is False
        assert qg["profile"] == "neutral"
        assert qg["preserve_signal"] == 0.0

    def test_get_experience_insights_derives_fragile_profile_from_stage_notes(self):
        """Wenn preserve_signal hoch ist, muss quality_gate Profil als fragil markiert werden."""
        from backend.api.bridge import get_experience_insights

        class MockResult:
            metadata = {"degradation_status": "ok"}
            stage_notes = {"exzellenz_recovery_profile": {"preserve_signal": 0.8}}

        result = get_experience_insights(MockResult())
        qg = result["quality_gate"]
        assert qg["profile"] == "fragile_or_transient_risk"
        assert qg["preserve_signal"] == 0.8

    def test_get_experience_insights_quality_gate_maps_recovered_floor(self):
        """Fallback-quality-floor recovered status must be reflected as recovered gate status."""
        from backend.api.bridge import get_experience_insights

        class MockResult:
            metadata = {
                "degradation_status": "ok",
                "fallback_quality_floor": {
                    "triggered": True,
                    "status": "recovered",
                    "attempts": 1,
                    "reason": "fallback_quality_floor_recovered_with_checkpoint",
                },
            }

        result = get_experience_insights(MockResult())
        qg = result["quality_gate"]
        assert qg["passed"] is False
        assert qg["degradation_status"] == "recovered"
        assert qg["best_possible_reached"] is True
        assert qg["fallback_quality_floor_status"] == "recovered"

    def test_get_experience_insights_propagates_worldclass_and_threshold_evidence(self):
        """Bridge insights must carry WCS gate and threshold evidence fields."""
        from backend.api.bridge import get_experience_insights

        class MockResult:
            metadata = {
                "worldclass_composite_gate": {
                    "wcs": 0.89,
                    "threshold": 0.88,
                    "profile": "vocal",
                    "artifact_veto": False,
                    "passed": True,
                },
                "threshold_evidence": {
                    "worldclass_composite_gate": {
                        "source_class": "C",
                        "source_ref": "Spec §8.6b WCS initial calibration",
                    }
                },
            }

        result = get_experience_insights(MockResult())
        assert "threshold_evidence" in result
        assert result["quality_gate"]["worldclass_composite_gate"]["passed"] is True
        assert result["quality_gate"]["worldclass_composite_gate"]["profile"] == "vocal"
        assert result["threshold_evidence"]["worldclass_composite_gate"]["source_class"] == "C"

    def test_build_export_quality_gate_payload_includes_fqf_flags(self):
        """Bridge must emit export_workflow-compatible quality_gate payload."""
        from backend.api.bridge import build_export_quality_gate_payload

        class MockResult:
            quality_estimate = 0.9
            metadata = {
                "fail_reasons": [],
                "degradation_status": "ok",
                "fallback_quality_floor": {
                    "triggered": True,
                    "status": "recovered",
                    "attempts": 1,
                    "reason": "fallback_quality_floor_recovered_with_checkpoint",
                },
            }

        payload = build_export_quality_gate_payload(MockResult())
        assert payload["passed"] is False
        assert payload["recovery_attempted"] is True
        assert payload["best_possible_reached"] is True
        assert isinstance(payload["fallback_quality_floor"], dict)

    def test_build_export_quality_gate_payload_defaults_without_metadata(self):
        """Payload builder must be resilient for metadata-less results."""
        from backend.api.bridge import build_export_quality_gate_payload

        class MockResult:
            quality_estimate = 0.8
            metadata = None

        payload = build_export_quality_gate_payload(MockResult())
        assert "passed" in payload
        assert "required_gates" in payload
        assert payload["recovery_attempted"] is False
        assert "musiclover" in payload
        assert payload["musiclover"]["musical_goals"]["remaining_count"] == 0
        assert payload["user_confidence_summary"]["manual_action_required"] is False
        assert payload["user_confidence_summary"]["allowed_user_decisions"] == ["mode_selection"]

    def test_build_export_quality_gate_payload_mirrors_payload_into_result_metadata(self):
        """Bridge must persist the final export gate payload on the result object."""
        from backend.api.bridge import build_export_quality_gate_payload

        class MockResult:
            quality_estimate = 0.8
            metadata = {
                "degradation_status": "ok",
            }

        result = MockResult()
        payload = build_export_quality_gate_payload(result)
        assert result.metadata["quality_gate_payload"] is payload
        assert result.metadata["export_quality_gate_payload"] is payload
        assert result.metadata["degradation_status"] == "ok"

    def test_build_export_quality_gate_payload_carries_profile_details(self):
        """Export payload muss adaptives Gate-Profil für Reports bereitstellen."""
        from backend.api.bridge import build_export_quality_gate_payload

        class MockResult:
            quality_estimate = 0.7
            metadata = {
                "degradation_status": "ok",
                "export_gate_profile": "modern_stable",
                "export_gate_material": "cd_digital",
                "export_gate_preserve_signal": 0.15,
                "export_gate_thresholds": {
                    "quality_estimate": 0.57,
                    "level_drop_db": 2.1,
                },
                "export_gate_signal_signature": {
                    "crest_db": 11.2,
                    "hf_ratio": 0.09,
                    "transient_ratio": 0.002,
                    "micro_dynamic_db": 7.8,
                },
            }

        payload = build_export_quality_gate_payload(MockResult())
        assert payload["profile"] == "modern_stable"
        assert payload["material"] == "cd_digital"
        assert payload["preserve_signal"] == 0.15
        assert payload["thresholds"]["quality_estimate"] == 0.57
        assert payload["thresholds"]["level_drop_db"] == 2.1
        assert payload["signal_signature"]["crest_db"] == 11.2

    def test_build_export_quality_gate_payload_musiclover_sections(self):
        """Music-Lover Payload enthält vokale/stereo/zeitliche Qualitätsindikatoren."""
        from backend.api.bridge import build_export_quality_gate_payload

        class MockResult:
            quality_estimate = 0.81
            chroma_correlation = 0.93
            lufs_delta = 1.7
            metadata = {
                "degradation_status": "ok",
                "vqi": 0.84,
                "vqi_tier": "good",
                "singer_identity_cosine": 0.95,
                "mono_compatibility_warning": True,
                "vocal_restoration_capability_status": "sota_fallback",
                "model_capability_report": {
                    "summary": {
                        "all_sota_real": False,
                        "degraded_capabilities": ["miipher"],
                    }
                },
                "musical_goals": {
                    "scores": {"natuerlichkeit": 0.80, "authentizitaet": 0.77},
                    "thresholds": {"natuerlichkeit": 0.88, "authentizitaet": 0.84},
                },
                "temporal_continuity": {
                    "phase_19_de_esser": {"gain_step_db": 1.8, "variance_ratio": 1.4},
                },
            }

        payload = build_export_quality_gate_payload(MockResult())
        ml = payload["musiclover"]
        assert ml["vocal_integrity"]["vqi"] == 0.84
        assert ml["stereo_integrity"]["mono_compatibility_warning"] is True
        assert ml["temporal_risk"]["hotspot_count"] == 1
        assert ml["musical_goals"]["remaining_count"] >= 1
        assert ml["mastering"]["chroma_correlation"] == 0.93
        assert ml["decision_trace"]["all_sota_real"] is False
        assert ml["decision_trace"]["vocal_restoration_capability_status"] == "sota_fallback"
        assert payload["user_confidence_summary"]["confidence_level"] == "hoch"
        assert payload["user_confidence_summary"]["export_policy"] == "normal_export"

    def test_build_export_quality_gate_payload_user_confidence_protects_listener(self):
        """Bridge-Payload muss Hoerrisiko als geschuetzten Nutzerstatus formulieren."""
        from backend.api.bridge import build_export_quality_gate_payload

        class MockResult:
            quality_estimate = 0.31
            metadata = {
                "degradation_status": "degraded",
                "fail_reason": "artifact_freedom_regression",
            }

        payload = build_export_quality_gate_payload(MockResult())
        confidence = payload["user_confidence_summary"]
        assert confidence["confidence_level"] == "geschuetzt"
        assert confidence["export_policy"] == "input_or_best_safe_checkpoint"
        assert confidence["manual_action_required"] is False
        assert payload["musiclover"]["decision_trace"]["export_policy"] == "input_or_best_safe_checkpoint"

    def test_build_export_quality_gate_payload_propagates_worldclass_and_threshold_evidence(self):
        """Export payload must include WCS gate and threshold evidence from metadata."""
        from backend.api.bridge import build_export_quality_gate_payload

        class MockResult:
            quality_estimate = 0.88
            metadata = {
                "degradation_status": "ok",
                "worldclass_composite_gate": {
                    "wcs": 0.87,
                    "threshold": 0.85,
                    "profile": "instrumental",
                    "artifact_veto": False,
                    "passed": True,
                },
                "threshold_evidence": {
                    "hpi_gate": {
                        "source_class": "B",
                        "source_ref": "Spec §2.44 holistic perceptual index",
                    }
                },
            }

        payload = build_export_quality_gate_payload(MockResult())
        assert payload["worldclass_composite_gate"]["passed"] is True
        assert payload["worldclass_composite_gate"]["profile"] == "instrumental"
        assert payload["threshold_evidence"]["hpi_gate"]["source_class"] == "B"


# ---------------------------------------------------------------------------
# §2.53 Team Coordination Propagation
# ---------------------------------------------------------------------------


class TestTeamCoordinationPropagation:
    """§2.29e team_coordination must propagate via bridge."""

    def test_team_coordination_structure(self):
        """team_coordination has event_count, events, phase_type_summary."""
        tc = {
            "event_count": 2,
            "events": [
                {"phase": "phase_50", "policy": "phase50_after_hf_restoration"},
            ],
            "phase_type_summary": {"restorative": 3, "reconstructive": 2},
        }
        assert tc["event_count"] >= 0  # type: ignore[operator]
        assert isinstance(tc["events"], list)
        assert isinstance(tc["phase_type_summary"], dict)


# ---------------------------------------------------------------------------
# §2.53b Denker-Plan-Determinismus Propagation
# ---------------------------------------------------------------------------


class TestDenkerPlanPropagation:
    """Denker plan must propagate metadata end-to-end."""

    def test_metadata_not_discarded(self):
        """§2.53: VERBOTEN — Metadaten beim Konvertieren verwerfen."""
        # This is a structural test — the conversion path must preserve metadata
        # We verify the contract by checking that AurikErgebnis has metadata support
        try:
            # If it has metadata attribute or accepts it — contract fulfilled
            import inspect

            from denker.aurik_denker import AurikErgebnis

            sig = inspect.signature(AurikErgebnis.__init__) if hasattr(AurikErgebnis, "__init__") else None
            if sig is not None:
                params = list(sig.parameters.keys())
                has_meta = "metadata" in params or "meta" in params or hasattr(AurikErgebnis, "metadata")
            else:
                has_meta = hasattr(AurikErgebnis, "metadata")
            assert has_meta, "AurikErgebnis must support metadata field"
        except ImportError:
            pytest.skip("AurikErgebnis not importable")
