"""Build a deterministic runtime-spec style report from Voice-First audit snapshots.

This report mirrors the minimal contract consumed by audit/release_runtime_consistency.py
and is intended for CI failfast usage where live frontend/backend logs are not available.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
logger = logging.getLogger(__name__)

_REQUIRED_KEYS: tuple[str, ...] = (
    "vqi",
    "formant_integrity",
    "vibrato_depth_preserved",
    "micro_dynamic_correlation",
)

_GUARD_CLASS_POLICIES: dict[str, dict[str, float]] = {
    # Default-Matrix: profilbasiert, aktuell verhaltensneutral für bestehende Critical-Guards.
    "critical": {
        "max_allowed_prior_drift_ratio": 0.25,
        "min_decision_stability_score": 0.80,
    },
    "high": {
        "max_allowed_prior_drift_ratio": 0.28,
        "min_decision_stability_score": 0.78,
        "max_hf_guard_fired_count": 8.0,
        "max_spectral_tilt_guard_fired_count": 8.0,
        "max_hf_delta_ratio": 0.35,
        "max_spectral_tilt_deviation_db_per_oct": 4.5,
        "max_interaction_rollbacks": 6.0,
        "max_ml_fallbacks_used": 6.0,
        "max_ml_guard_events": 24.0,
        "min_holistic_artifact_freedom": 0.95,
        "max_temporal_critical_events": 0.0,
        "max_temporal_gain_step_db": 1.5,
        "max_fail_reasons_count": 0.0,
        "max_recovery_uncertainty_index": 0.85,
        "max_team_coordination_events": 32.0,
        "max_length_corrections": 4.0,
    },
    "medium": {
        "max_allowed_prior_drift_ratio": 0.32,
        "min_decision_stability_score": 0.74,
        "max_interaction_rollbacks": 8.0,
        "max_ml_fallbacks_used": 8.0,
        "max_ml_guard_events": 32.0,
    },
}

_GUARD_TO_CLASS: dict[str, str] = {
    "decision_quality_learning_guard": "critical",
    "bridge_export_fidelity_guard": "high",
    "bridge_import_status_runtime": "high",
    "interaction_guard_runtime": "high",
    "dsp_ml_guard_runtime": "high",
    "vocal_perceptual_runtime": "high",
    "temporal_stereo_runtime": "high",
    "recovery_execution_runtime": "high",
    "team_goal_runtime": "high",
}

_GUARD_POLICIES: dict[str, dict[str, float]] = {
    # Guard-spezifische Overrides bleiben möglich und haben Vorrang vor Klassen-Defaults.
    "decision_quality_learning_guard": {},
    "bridge_export_fidelity_guard": {},
    "bridge_import_status_runtime": {},
    "interaction_guard_runtime": {},
    "dsp_ml_guard_runtime": {},
    "vocal_perceptual_runtime": {},
    "temporal_stereo_runtime": {},
    "recovery_execution_runtime": {},
    "team_goal_runtime": {},
}

_CORE_REQUIRED_CHECKS: tuple[str, ...] = (
    "voice_first_snapshot_non_empty",
    "voice_first_blockers_runtime",
    "decision_quality_learning_guard",
)

_GUARD_FIELD_COVERAGE: dict[str, list[str]] = {
    "decision_quality_learning_guard": [
        "decision_quality.learning_applied",
        "decision_quality.causal_credit_confidence",
        "decision_quality.prior_drift_ratio",
        "decision_quality.decision_stability_score",
    ],
    "bridge_export_fidelity_guard": [
        "fidelity_guards.hf_hallucination_guard.guard_fired_count",
        "fidelity_guards.hf_hallucination_guard.max_delta_ratio",
        "fidelity_guards.spectral_tilt_guard.guard_fired_count",
        "fidelity_guards.spectral_tilt_guard.max_deviation_db_per_oct",
    ],
    "bridge_import_status_runtime": [
        "startup_check_status.available",
        "startup_check_status.failures",
        "pre_analysis_result_status.available",
        "pre_analysis_result_status.failures",
        "audio_exporter_status.available",
        "audio_exporter_status.failures",
        "ml_memory_budget_import_status.available",
        "ml_memory_budget_import_status.failures",
    ],
    "interaction_guard_runtime": [
        "interaction_guard.interaction_rollbacks",
        "interaction_guard.pipeline_stopped_early",
    ],
    "dsp_ml_guard_runtime": [
        "ml_guard_events",
        "ml_fallbacks_used",
    ],
    "vocal_perceptual_runtime": [
        "vocal_no_harm_gate.requires_rollback",
        "vocal_no_harm_rollback",
        "holistic_perceptual_gate.passed",
        "holistic_perceptual_gate.artifact_freedom",
    ],
    "temporal_stereo_runtime": [
        "temporal_continuity.*.critical",
        "temporal_continuity.*.gain_step_db",
        "mono_compatibility_warning",
        "onset_shift_ok",
    ],
    "recovery_execution_runtime": [
        "fail_reasons",
        "graceful_stop",
        "recovery_certainty.uncertainty_index",
    ],
    "team_goal_runtime": [
        "team_coordination.event_count",
        "goal_recovery.attempted",
        "goal_recovery.resolved",
        "goal_recovery.final_violations",
        "length_corrections",
    ],
}


def _get_guard_policy(guard_id: str) -> dict[str, float]:
    """Liefert effektive Policy für einen Guard aus Klasse + optionalem Override."""
    guard_class = _GUARD_TO_CLASS.get(guard_id, "medium")
    class_policy = _GUARD_CLASS_POLICIES.get(guard_class, _GUARD_CLASS_POLICIES["medium"])
    override_policy = _GUARD_POLICIES.get(guard_id, {})
    merged = dict(class_policy)
    merged.update(override_policy)
    return merged


def build_guard_coverage_manifest(
    output_path: str = "audit/voice_first_guard_coverage_manifest.json",
) -> dict[str, Any]:
    """Erzeugt ein kompaktes Manifest aller Runtime-Guard-Familien und Feldabdeckung."""
    checks: list[dict[str, Any]] = []

    for check_id in _CORE_REQUIRED_CHECKS:
        guard_class = _GUARD_TO_CLASS.get(check_id, "critical")
        checks.append(
            {
                "id": check_id,
                "required": True,
                "guard_class": guard_class,
                "effective_policy": _get_guard_policy(check_id),
                "field_coverage": list(_GUARD_FIELD_COVERAGE.get(check_id, [])),
            }
        )

    for guard_id in sorted(_GUARD_TO_CLASS.keys()):
        if guard_id in _CORE_REQUIRED_CHECKS:
            continue
        checks.append(
            {
                "id": guard_id,
                "required": False,
                "guard_class": _GUARD_TO_CLASS.get(guard_id, "medium"),
                "effective_policy": _get_guard_policy(guard_id),
                "field_coverage": list(_GUARD_FIELD_COVERAGE.get(guard_id, [])),
            }
        )

    manifest: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "schema": "voice-first-guard-coverage-manifest.v1",
        "check_count": len(checks),
        "required_count": sum(1 for item in checks if bool(item.get("required"))),
        "checks": checks,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _extract_decision_quality(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrahiere normalisierte Decision-Quality-Daten aus bekannten Snapshot-Strukturen."""
    candidates: list[Any] = [
        entry.get("decision_quality"),
        entry.get("learning_decision"),
    ]
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("decision_quality"))
        candidates.append(metadata.get("learning_decision"))

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def _extract_fidelity_guards(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extrahiert Bridge/Export Fidelity-Guard-Telemetrie aus bekannten Strukturen."""
    candidates: list[Any] = []

    # Export- und Sidecar-typische Ablage
    candidates.append(entry.get("fidelity_guards"))

    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("fidelity_guards"))
        candidates.append(metadata)

    # Manche Pfade legen die Felder unter features/scores ab
    features = entry.get("features")
    if isinstance(features, dict):
        candidates.append(features.get("fidelity_guards"))
        candidates.append(features)

    scores = entry.get("scores")
    if isinstance(scores, dict):
        candidates.append(scores.get("fidelity_guards"))
        candidates.append(scores)

    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("hf_hallucination_guard", "spectral_tilt_guard"):
            value = candidate.get(key)
            if isinstance(value, dict):
                result[key] = value
    return result


def _extract_interaction_guard(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrahiert CIG-Interaktionsguard-Telemetrie aus bekannten Snapshot-Strukturen."""
    candidates: list[Any] = [entry.get("interaction_guard")]

    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("interaction_guard"))

    features = entry.get("features")
    if isinstance(features, dict):
        candidates.append(features.get("interaction_guard"))

    scores = entry.get("scores")
    if isinstance(scores, dict):
        candidates.append(scores.get("interaction_guard"))

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def _extract_bridge_import_status_telemetry(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extrahiert Bridge-Importstatus-Telemetrie aus bekannten Snapshot-Strukturen."""
    out: dict[str, dict[str, Any]] = {}
    status_keys = (
        "startup_check_status",
        "pre_analysis_result_status",
        "audio_exporter_status",
        "ml_memory_budget_import_status",
    )

    candidates: list[Any] = [entry]
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
    features = entry.get("features")
    if isinstance(features, dict):
        candidates.append(features)
    scores = entry.get("scores")
    if isinstance(scores, dict):
        candidates.append(scores)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in status_keys:
            value = candidate.get(key)
            if isinstance(value, dict):
                out[key] = value

        nested = candidate.get("bridge_import_status")
        if isinstance(nested, dict):
            for key in status_keys:
                value = nested.get(key)
                if isinstance(value, dict):
                    out[key] = value

    return out


def _extract_ml_guard_telemetry(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extrahiert ML-Guard-Events und ML-Fallbacks aus bekannten Snapshot-Strukturen."""
    candidates: list[Any] = []

    for key in ("ml_guard_events", "ml_fallbacks_used"):
        candidates.append(entry.get(key))

    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in ("ml_guard_events", "ml_fallbacks_used"):
            candidates.append(metadata.get(key))

    features = entry.get("features")
    if isinstance(features, dict):
        for key in ("ml_guard_events", "ml_fallbacks_used"):
            candidates.append(features.get(key))

    scores = entry.get("scores")
    if isinstance(scores, dict):
        for key in ("ml_guard_events", "ml_fallbacks_used"):
            candidates.append(scores.get(key))

    ml_events: list[dict[str, Any]] = []
    ml_fallbacks: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            if not isinstance(item, dict):
                continue
            if "fallback" in item:
                ml_fallbacks.append(item)
            else:
                ml_events.append(item)

    return ml_events, ml_fallbacks


def _extract_vocal_perceptual_telemetry(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrahiert Vocal-Safety- und Holistic-Perceptual-Telemetrie aus Snapshot-Strukturen."""
    out: dict[str, Any] = {}

    candidates: list[Any] = [entry]
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
    features = entry.get("features")
    if isinstance(features, dict):
        candidates.append(features)
    scores = entry.get("scores")
    if isinstance(scores, dict):
        candidates.append(scores)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if "vocal_no_harm_gate" in candidate and isinstance(candidate.get("vocal_no_harm_gate"), dict):
            out["vocal_no_harm_gate"] = candidate.get("vocal_no_harm_gate")
        if "vocal_no_harm_rollback" in candidate and isinstance(candidate.get("vocal_no_harm_rollback"), bool):
            out["vocal_no_harm_rollback"] = candidate.get("vocal_no_harm_rollback")
        if "vocal_no_harm_reason" in candidate and isinstance(candidate.get("vocal_no_harm_reason"), str):
            out["vocal_no_harm_reason"] = candidate.get("vocal_no_harm_reason")
        if "holistic_perceptual_gate" in candidate and isinstance(candidate.get("holistic_perceptual_gate"), dict):
            out["holistic_perceptual_gate"] = candidate.get("holistic_perceptual_gate")

    return out


def _extract_temporal_stereo_telemetry(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrahiert Temporal-Continuity- und Stereo-Warntelemetrie aus Snapshot-Strukturen."""
    out: dict[str, Any] = {}

    candidates: list[Any] = [entry]
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
    features = entry.get("features")
    if isinstance(features, dict):
        candidates.append(features)
    scores = entry.get("scores")
    if isinstance(scores, dict):
        candidates.append(scores)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        tc = candidate.get("temporal_continuity")
        if isinstance(tc, dict):
            out["temporal_continuity"] = tc

        mono_warn = candidate.get("mono_compatibility_warning")
        if isinstance(mono_warn, bool):
            out["mono_compatibility_warning"] = mono_warn

        onset_ok = candidate.get("onset_shift_ok")
        if isinstance(onset_ok, bool):
            out["onset_shift_ok"] = onset_ok

    return out


def _extract_recovery_execution_telemetry(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrahiert Recovery-/Execution-Telemetrie aus Snapshot-Strukturen."""
    out: dict[str, Any] = {}

    candidates: list[Any] = [entry]
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
    features = entry.get("features")
    if isinstance(features, dict):
        candidates.append(features)
    scores = entry.get("scores")
    if isinstance(scores, dict):
        candidates.append(scores)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        fail_reasons = candidate.get("fail_reasons")
        if isinstance(fail_reasons, list):
            out["fail_reasons"] = [str(x) for x in fail_reasons]

        graceful_stop = candidate.get("graceful_stop")
        if isinstance(graceful_stop, bool):
            out["graceful_stop"] = graceful_stop

        recovery_certainty = candidate.get("recovery_certainty")
        if isinstance(recovery_certainty, dict):
            out["recovery_certainty"] = recovery_certainty

    return out


def _extract_team_goal_telemetry(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrahiert Team-/Goal-Recovery-/Length-Korrektur-Telemetrie aus Snapshot-Strukturen."""
    out: dict[str, Any] = {}

    candidates: list[Any] = [entry]
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
    features = entry.get("features")
    if isinstance(features, dict):
        candidates.append(features)
    scores = entry.get("scores")
    if isinstance(scores, dict):
        candidates.append(scores)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        team_coord = candidate.get("team_coordination")
        if isinstance(team_coord, dict):
            out["team_coordination"] = team_coord

        goal_recovery = candidate.get("goal_recovery")
        if isinstance(goal_recovery, dict):
            out["goal_recovery"] = goal_recovery

        length_corr = candidate.get("length_corrections")
        if isinstance(length_corr, list):
            out["length_corrections"] = [str(x) for x in length_corr]

    return out


def build_runtime_report(
    snapshot_path: str = "audit/current_voice_first_audit.json",
    output_path: str = "audit/runtime_spec_report_voice_first_runtime.json",
) -> dict[str, Any]:
    """Erzeuge einen deterministischen Runtime-Report aus einem Voice-First-Snapshot."""
    entries = _load_entries(Path(snapshot_path))
    checks: list[dict[str, Any]] = []
    decision_quality_policy = _get_guard_policy("decision_quality_learning_guard")
    max_allowed_prior_drift_ratio = float(decision_quality_policy.get("max_allowed_prior_drift_ratio", 0.25))
    min_decision_stability_score = float(decision_quality_policy.get("min_decision_stability_score", 0.80))

    non_empty = len(entries) > 0
    checks.append(
        {
            "id": "voice_first_snapshot_non_empty",
            "title": "Voice-First Snapshot enthält mindestens einen vokalen Lauf",
            "required": True,
            "passed": non_empty,
            "evidence": f"entries={len(entries)}",
        }
    )

    blockers_ok = True
    blocker_details: list[str] = []
    if non_empty:
        for idx, entry in enumerate(entries):
            vqc = entry.get("vocal_quality_check", {})
            if not isinstance(vqc, dict):
                blockers_ok = False
                blocker_details.append(f"entry[{idx}] missing vocal_quality_check")
                continue
            for key in _REQUIRED_KEYS:
                value = vqc.get(key)
                if not isinstance(value, bool) or value is not True:
                    blockers_ok = False
                    blocker_details.append(f"entry[{idx}] {key}={value!r}")
    else:
        blockers_ok = False
        blocker_details.append("snapshot empty")

    checks.append(
        {
            "id": "voice_first_blockers_runtime",
            "title": "Voice-First Runtime-Blocker vollständig erfüllt",
            "required": True,
            "passed": blockers_ok,
            "evidence": "; ".join(blocker_details) if blocker_details else "all required blockers true",
        }
    )

    decision_quality_ok = True
    decision_quality_details: list[str] = []
    if non_empty:
        for idx, entry in enumerate(entries):
            dq = _extract_decision_quality(entry)
            if not isinstance(dq, dict) or not dq:
                decision_quality_ok = False
                decision_quality_details.append(f"entry[{idx}] missing decision_quality")
                continue

            learning_applied = dq.get("learning_applied")
            causal_credit_conf = dq.get("causal_credit_confidence")
            prior_drift_ratio = dq.get("prior_drift_ratio")
            decision_stability_score = dq.get("decision_stability_score")

            if not isinstance(learning_applied, bool):
                decision_quality_ok = False
                decision_quality_details.append(f"entry[{idx}] learning_applied missing/bad")
                continue
            if not isinstance(causal_credit_conf, (int, float)):
                decision_quality_ok = False
                decision_quality_details.append(f"entry[{idx}] causal_credit_confidence missing/bad")
                continue
            if not isinstance(prior_drift_ratio, (int, float)):
                decision_quality_ok = False
                decision_quality_details.append(f"entry[{idx}] prior_drift_ratio missing/bad")
                continue
            if not isinstance(decision_stability_score, (int, float)):
                decision_quality_ok = False
                decision_quality_details.append(f"entry[{idx}] decision_stability_score missing/bad")
                continue

            if learning_applied and float(causal_credit_conf) <= 0.0:
                decision_quality_ok = False
                decision_quality_details.append(
                    f"entry[{idx}] learning_without_positive_causal_credit={causal_credit_conf!r}"
                )
            if float(prior_drift_ratio) > max_allowed_prior_drift_ratio:
                decision_quality_ok = False
                decision_quality_details.append(
                    f"entry[{idx}] prior_drift_ratio={prior_drift_ratio:.3f}>{max_allowed_prior_drift_ratio:.2f}"
                )
            if float(decision_stability_score) < min_decision_stability_score:
                decision_quality_ok = False
                decision_quality_details.append(
                    f"entry[{idx}] decision_stability_score={float(decision_stability_score):.3f}<{min_decision_stability_score:.2f}"
                )
    else:
        decision_quality_ok = False
        decision_quality_details.append("snapshot empty")

    checks.append(
        {
            "id": "decision_quality_learning_guard",
            "title": "Decision-Quality-Guard: Kausalität, Drift, Stabilität",
            "required": True,
            "passed": decision_quality_ok,
            "evidence": "; ".join(decision_quality_details)
            if decision_quality_details
            else "causal_safe=True drift_bound=True decision_stability=True",
        }
    )

    fidelity_policy = _get_guard_policy("bridge_export_fidelity_guard")
    max_hf_guard_fired_count = int(fidelity_policy.get("max_hf_guard_fired_count", 8.0))
    max_spectral_guard_fired_count = int(fidelity_policy.get("max_spectral_tilt_guard_fired_count", 8.0))
    max_hf_delta_ratio = float(fidelity_policy.get("max_hf_delta_ratio", 0.35))
    max_spectral_tilt_deviation = float(fidelity_policy.get("max_spectral_tilt_deviation_db_per_oct", 4.5))

    fidelity_guard_ok = True
    fidelity_details: list[str] = []
    telemetry_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            guards = _extract_fidelity_guards(entry)
            if not guards:
                continue
            telemetry_present = True

            hf_guard = guards.get("hf_hallucination_guard", {})
            if isinstance(hf_guard, dict):
                hf_fired = hf_guard.get("guard_fired_count")
                hf_ratio = hf_guard.get("max_delta_ratio")
                if isinstance(hf_fired, (int, float)) and int(hf_fired) > max_hf_guard_fired_count:
                    fidelity_guard_ok = False
                    fidelity_details.append(
                        f"entry[{idx}] hf_guard_fired_count={int(hf_fired)}>{max_hf_guard_fired_count}"
                    )
                if isinstance(hf_ratio, (int, float)) and float(hf_ratio) > max_hf_delta_ratio:
                    fidelity_guard_ok = False
                    fidelity_details.append(
                        f"entry[{idx}] hf_max_delta_ratio={float(hf_ratio):.3f}>{max_hf_delta_ratio:.3f}"
                    )

            st_guard = guards.get("spectral_tilt_guard", {})
            if isinstance(st_guard, dict):
                st_fired = st_guard.get("guard_fired_count")
                st_dev = st_guard.get("max_deviation_db_per_oct")
                if isinstance(st_fired, (int, float)) and int(st_fired) > max_spectral_guard_fired_count:
                    fidelity_guard_ok = False
                    fidelity_details.append(
                        f"entry[{idx}] spectral_guard_fired_count={int(st_fired)}>{max_spectral_guard_fired_count}"
                    )
                if isinstance(st_dev, (int, float)) and float(st_dev) > max_spectral_tilt_deviation:
                    fidelity_guard_ok = False
                    fidelity_details.append(
                        f"entry[{idx}] spectral_max_dev={float(st_dev):.3f}>{max_spectral_tilt_deviation:.3f}"
                    )

    checks.append(
        {
            "id": "bridge_export_fidelity_guard",
            "title": "Bridge/Export Fidelity-Guards im sicheren Bereich",
            "required": False,
            "passed": fidelity_guard_ok,
            "evidence": (
                "; ".join(fidelity_details)
                if fidelity_details
                else (
                    "fidelity telemetry absent (non-blocking)"
                    if not telemetry_present
                    else "hf/spectral fidelity guards within configured limits"
                )
            ),
        }
    )

    bridge_import_policy = _get_guard_policy("bridge_import_status_runtime")
    max_bridge_import_failures = int(bridge_import_policy.get("max_bridge_import_failures", 0.0))

    bridge_import_ok = True
    bridge_import_details: list[str] = []
    bridge_import_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            statuses = _extract_bridge_import_status_telemetry(entry)
            if not statuses:
                continue
            bridge_import_present = True

            for status_key, status_payload in statuses.items():
                available = status_payload.get("available")
                failures = status_payload.get("failures")
                last_error = status_payload.get("last_error")

                if isinstance(available, bool) and available is False:
                    bridge_import_ok = False
                    bridge_import_details.append(f"entry[{idx}] {status_key}.available=False")

                if isinstance(failures, (int, float)) and int(failures) > max_bridge_import_failures:
                    bridge_import_ok = False
                    bridge_import_details.append(
                        f"entry[{idx}] {status_key}.failures={int(failures)}>{max_bridge_import_failures}"
                    )

                if isinstance(last_error, str) and last_error.strip():
                    bridge_import_ok = False
                    bridge_import_details.append(f"entry[{idx}] {status_key}.last_error_set=True")

    checks.append(
        {
            "id": "bridge_import_status_runtime",
            "title": "Bridge-Importstatus-Telemetrie im sicheren Bereich",
            "required": False,
            "passed": bridge_import_ok,
            "evidence": (
                "; ".join(bridge_import_details)
                if bridge_import_details
                else (
                    "bridge import status telemetry absent (non-blocking)"
                    if not bridge_import_present
                    else "bridge import status signals within configured limits"
                )
            ),
        }
    )

    interaction_policy = _get_guard_policy("interaction_guard_runtime")
    max_interaction_rollbacks = int(interaction_policy.get("max_interaction_rollbacks", 6.0))

    interaction_guard_ok = True
    interaction_details: list[str] = []
    interaction_telemetry_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            interaction_guard = _extract_interaction_guard(entry)
            if not interaction_guard:
                continue
            interaction_telemetry_present = True

            rollbacks = interaction_guard.get("interaction_rollbacks")
            rollback_count = len(rollbacks) if isinstance(rollbacks, list) else 0
            if rollback_count > max_interaction_rollbacks:
                interaction_guard_ok = False
                interaction_details.append(
                    f"entry[{idx}] interaction_rollbacks={rollback_count}>{max_interaction_rollbacks}"
                )

            stopped_early = interaction_guard.get("pipeline_stopped_early")
            if isinstance(stopped_early, bool) and stopped_early:
                interaction_guard_ok = False
                interaction_details.append(f"entry[{idx}] interaction_guard_stopped_pipeline=True")

    checks.append(
        {
            "id": "interaction_guard_runtime",
            "title": "Interaction-Guard Runtime-Metadaten im sicheren Bereich",
            "required": False,
            "passed": interaction_guard_ok,
            "evidence": (
                "; ".join(interaction_details)
                if interaction_details
                else (
                    "interaction telemetry absent (non-blocking)"
                    if not interaction_telemetry_present
                    else "interaction guard within configured limits"
                )
            ),
        }
    )

    dsp_ml_policy = _get_guard_policy("dsp_ml_guard_runtime")
    max_ml_fallbacks_used = int(dsp_ml_policy.get("max_ml_fallbacks_used", 6.0))  # §V6: logger.warning handled at call site
    max_ml_guard_events = int(dsp_ml_policy.get("max_ml_guard_events", 24.0))

    dsp_ml_guard_ok = True
    dsp_ml_details: list[str] = []
    dsp_ml_telemetry_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            ml_events, ml_fallbacks = _extract_ml_guard_telemetry(entry)
            if not ml_events and not ml_fallbacks:
                continue
            dsp_ml_telemetry_present = True

            if len(ml_events) > max_ml_guard_events:
                dsp_ml_guard_ok = False
                dsp_ml_details.append(f"entry[{idx}] ml_guard_events={len(ml_events)}>{max_ml_guard_events}")

            if len(ml_fallbacks) > max_ml_fallbacks_used:
                dsp_ml_guard_ok = False
                dsp_ml_details.append(f"entry[{idx}] ml_fallbacks_used={len(ml_fallbacks)}>{max_ml_fallbacks_used}")

    checks.append(
        {
            "id": "dsp_ml_guard_runtime",
            "title": "DSP/ML-Guard-Telemetrie im sicheren Bereich",
            "required": False,
            "passed": dsp_ml_guard_ok,
            "evidence": (
                "; ".join(dsp_ml_details)
                if dsp_ml_details
                else (
                    "ml guard telemetry absent (non-blocking)"
                    if not dsp_ml_telemetry_present
                    else "ml guard events/fallbacks within configured limits"
                )
            ),
        }
    )

    vocal_perceptual_policy = _get_guard_policy("vocal_perceptual_runtime")
    min_holistic_artifact_freedom = float(vocal_perceptual_policy.get("min_holistic_artifact_freedom", 0.95))

    vocal_perceptual_ok = True
    vocal_perceptual_details: list[str] = []
    vocal_perceptual_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            telemetry = _extract_vocal_perceptual_telemetry(entry)
            if not telemetry:
                continue
            vocal_perceptual_present = True

            if bool(telemetry.get("vocal_no_harm_rollback", False)):
                vocal_perceptual_ok = False
                _reason = telemetry.get("vocal_no_harm_reason", "rollback")
                vocal_perceptual_details.append(f"entry[{idx}] vocal_no_harm_rollback=True reason={_reason}")

            vnh_gate = telemetry.get("vocal_no_harm_gate")
            if isinstance(vnh_gate, dict) and bool(vnh_gate.get("requires_rollback", False)):
                vocal_perceptual_ok = False
                vocal_perceptual_details.append(f"entry[{idx}] vocal_no_harm_gate.requires_rollback=True")

            hpg = telemetry.get("holistic_perceptual_gate")
            if isinstance(hpg, dict):
                if isinstance(hpg.get("passed"), bool) and not bool(hpg.get("passed")):
                    vocal_perceptual_ok = False
                    vocal_perceptual_details.append(f"entry[{idx}] holistic_perceptual_gate.passed=False")
                _artifact = hpg.get("artifact_freedom")
                if isinstance(_artifact, (int, float)) and float(_artifact) < min_holistic_artifact_freedom:
                    vocal_perceptual_ok = False
                    vocal_perceptual_details.append(
                        f"entry[{idx}] holistic_artifact_freedom={float(_artifact):.3f}<{min_holistic_artifact_freedom:.2f}"
                    )

    checks.append(
        {
            "id": "vocal_perceptual_runtime",
            "title": "Vocal-Safety und Holistic-Perceptual-Gate im sicheren Bereich",
            "required": False,
            "passed": vocal_perceptual_ok,
            "evidence": (
                "; ".join(vocal_perceptual_details)
                if vocal_perceptual_details
                else (
                    "vocal/perceptual telemetry absent (non-blocking)"
                    if not vocal_perceptual_present
                    else "vocal safety + holistic perceptual signals within configured limits"
                )
            ),
        }
    )

    temporal_stereo_policy = _get_guard_policy("temporal_stereo_runtime")
    max_temporal_critical_events = int(temporal_stereo_policy.get("max_temporal_critical_events", 0.0))
    max_temporal_gain_step_db = float(temporal_stereo_policy.get("max_temporal_gain_step_db", 1.5))

    temporal_stereo_ok = True
    temporal_stereo_details: list[str] = []
    temporal_stereo_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            telemetry = _extract_temporal_stereo_telemetry(entry)
            if not telemetry:
                continue
            temporal_stereo_present = True

            temporal = telemetry.get("temporal_continuity")
            if isinstance(temporal, dict):
                critical_count = 0
                max_gain_step = 0.0
                for phase_payload in temporal.values():
                    if not isinstance(phase_payload, dict):
                        continue
                    if bool(phase_payload.get("critical", False)):
                        critical_count += 1
                    _step = phase_payload.get("gain_step_db")
                    if isinstance(_step, (int, float)):
                        max_gain_step = max(max_gain_step, float(_step))

                if critical_count > max_temporal_critical_events:
                    temporal_stereo_ok = False
                    temporal_stereo_details.append(
                        f"entry[{idx}] temporal_critical_events={critical_count}>{max_temporal_critical_events}"
                    )
                if max_gain_step > max_temporal_gain_step_db:
                    temporal_stereo_ok = False
                    temporal_stereo_details.append(
                        f"entry[{idx}] temporal_max_gain_step_db={max_gain_step:.3f}>{max_temporal_gain_step_db:.2f}"
                    )

            if bool(telemetry.get("mono_compatibility_warning", False)):
                temporal_stereo_ok = False
                temporal_stereo_details.append(f"entry[{idx}] mono_compatibility_warning=True")

            if "onset_shift_ok" in telemetry and telemetry.get("onset_shift_ok") is False:
                temporal_stereo_ok = False
                temporal_stereo_details.append(f"entry[{idx}] onset_shift_ok=False")

    checks.append(
        {
            "id": "temporal_stereo_runtime",
            "title": "Temporal-Continuity und Stereo-Warnsignale im sicheren Bereich",
            "required": False,
            "passed": temporal_stereo_ok,
            "evidence": (
                "; ".join(temporal_stereo_details)
                if temporal_stereo_details
                else (
                    "temporal/stereo telemetry absent (non-blocking)"
                    if not temporal_stereo_present
                    else "temporal continuity + stereo warning signals within configured limits"
                )
            ),
        }
    )

    recovery_execution_policy = _get_guard_policy("recovery_execution_runtime")
    max_fail_reasons_count = int(recovery_execution_policy.get("max_fail_reasons_count", 0.0))
    max_recovery_uncertainty_index = float(recovery_execution_policy.get("max_recovery_uncertainty_index", 0.85))

    recovery_execution_ok = True
    recovery_execution_details: list[str] = []
    recovery_execution_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            telemetry = _extract_recovery_execution_telemetry(entry)
            if not telemetry:
                continue
            recovery_execution_present = True

            fail_reasons = telemetry.get("fail_reasons")
            if isinstance(fail_reasons, list) and len(fail_reasons) > max_fail_reasons_count:
                recovery_execution_ok = False
                recovery_execution_details.append(
                    f"entry[{idx}] fail_reasons_count={len(fail_reasons)}>{max_fail_reasons_count}"
                )

            if bool(telemetry.get("graceful_stop", False)):
                recovery_execution_ok = False
                recovery_execution_details.append(f"entry[{idx}] graceful_stop=True")

            rc = telemetry.get("recovery_certainty")
            if isinstance(rc, dict):
                uncertainty_index = rc.get("uncertainty_index")
                if (
                    isinstance(uncertainty_index, (int, float))
                    and float(uncertainty_index) > max_recovery_uncertainty_index
                ):
                    recovery_execution_ok = False
                    recovery_execution_details.append(
                        f"entry[{idx}] recovery_uncertainty_index={float(uncertainty_index):.3f}>{max_recovery_uncertainty_index:.2f}"
                    )

    checks.append(
        {
            "id": "recovery_execution_runtime",
            "title": "Recovery-/Execution-Signale im sicheren Bereich",
            "required": False,
            "passed": recovery_execution_ok,
            "evidence": (
                "; ".join(recovery_execution_details)
                if recovery_execution_details
                else (
                    "recovery/execution telemetry absent (non-blocking)"
                    if not recovery_execution_present
                    else "recovery certainty + execution telemetry within configured limits"
                )
            ),
        }
    )

    team_goal_policy = _get_guard_policy("team_goal_runtime")
    max_team_coord_events = int(team_goal_policy.get("max_team_coordination_events", 32.0))
    max_length_corrections = int(team_goal_policy.get("max_length_corrections", 4.0))

    team_goal_ok = True
    team_goal_details: list[str] = []
    team_goal_present = False
    if non_empty:
        for idx, entry in enumerate(entries):
            telemetry = _extract_team_goal_telemetry(entry)
            if not telemetry:
                continue
            team_goal_present = True

            team_coord = telemetry.get("team_coordination")
            if isinstance(team_coord, dict):
                event_count = team_coord.get("event_count")
                if isinstance(event_count, (int, float)) and int(event_count) > max_team_coord_events:
                    team_goal_ok = False
                    team_goal_details.append(
                        f"entry[{idx}] team_coordination_event_count={int(event_count)}>{max_team_coord_events}"
                    )

            goal_recovery = telemetry.get("goal_recovery")
            if isinstance(goal_recovery, dict):
                attempted = goal_recovery.get("attempted")
                resolved = goal_recovery.get("resolved")
                final_violations = goal_recovery.get("final_violations")
                if bool(attempted) and resolved is False:
                    team_goal_ok = False
                    team_goal_details.append(f"entry[{idx}] goal_recovery_attempted_but_unresolved=True")
                if isinstance(final_violations, list) and len(final_violations) > 0:
                    team_goal_ok = False
                    team_goal_details.append(f"entry[{idx}] goal_recovery_final_violations={len(final_violations)}")

            length_corr = telemetry.get("length_corrections")
            if isinstance(length_corr, list) and len(length_corr) > max_length_corrections:
                team_goal_ok = False
                team_goal_details.append(f"entry[{idx}] length_corrections={len(length_corr)}>{max_length_corrections}")

    checks.append(
        {
            "id": "team_goal_runtime",
            "title": "Team-Koordination und Goal-Recovery im sicheren Bereich",
            "required": False,
            "passed": team_goal_ok,
            "evidence": (
                "; ".join(team_goal_details)
                if team_goal_details
                else (
                    "team/goal telemetry absent (non-blocking)"
                    if not team_goal_present
                    else "team coordination + goal recovery signals within configured limits"
                )
            ),
        }
    )

    required_checks = [chk for chk in checks if chk.get("required")]
    required_passed = sum(1 for chk in required_checks if chk.get("passed") is True)
    required_total = len(required_checks)
    compliance_ok = required_total > 0 and required_passed == required_total

    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "snapshot_path": snapshot_path,
        "compliance_ok": compliance_ok,
        "required_passed": required_passed,
        "required_total": required_total,
        "checks": checks,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI-Einstieg: baut Runtime-Report und liefert Exit-Code nach Compliance-Status."""
    parser = argparse.ArgumentParser(description="Build runtime-spec report from Voice-First snapshot")
    parser.add_argument("--snapshot", default="audit/current_voice_first_audit.json")
    parser.add_argument("--output", default="audit/runtime_spec_report_voice_first_runtime.json")
    parser.add_argument("--coverage-output", default=None)
    args = parser.parse_args(argv)

    if isinstance(args.coverage_output, str) and args.coverage_output.strip():
        manifest = build_guard_coverage_manifest(output_path=args.coverage_output)
        print(
            "Voice-First guard coverage manifest: "
            f"checks={manifest.get('check_count')} required={manifest.get('required_count')} "
            f"output={args.coverage_output}"
        )

    report = build_runtime_report(snapshot_path=args.snapshot, output_path=args.output)
    print(
        "Voice-First runtime report: "
        f"required={report.get('required_passed')}/{report.get('required_total')} "
        f"compliance_ok={report.get('compliance_ok')}"
    )
    return 0 if bool(report.get("compliance_ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
