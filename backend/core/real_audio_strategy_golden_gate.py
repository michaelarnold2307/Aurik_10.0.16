"""Real-audio Golden-Set gate for autonomous restoration strategy planning.

This gate validates the layer after defect recognition: DefectScanner output is
fed into CausalDefectReasoner and DefectPhaseMapper, then checked against a
manifest of expected causes, required phases, forbidden phases, order rules, and
runtime budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from backend.core.causal_defect_reasoner import reason_about_defects
from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
from backend.core.defect_scanner import DefectScanner, MaterialType
from backend.core.real_audio_defect_golden_gate import _audio_from_import, _load_manifest
from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@dataclass(frozen=True)
class StrategyGateThresholds:
    """Thresholds for the real-audio strategy Golden-Set gate."""

    min_cause_topk_accuracy: float = 0.875
    min_phase_recall: float = 0.95
    min_phase_precision: float = 1.0
    max_forbidden_phase_violations: int = 0
    max_order_violations: int = 0
    max_runtime_factor: float = 1.50


@dataclass(frozen=True)
class StrategyCaseResult:
    """Serialisierbares Strategie-Bewertungsergebnis pro Fall."""

    case_id: str
    accepted_causes: tuple[str, ...]
    cause_top_k: int
    primary_cause: str
    top_causes: tuple[str, ...]
    cause_hit: bool
    required_phases: tuple[str, ...]
    missing_required_phases: tuple[str, ...]
    forbidden_phases: tuple[str, ...]
    forbidden_present: tuple[str, ...]
    ordered_before: tuple[tuple[str, str], ...]
    order_violations: tuple[tuple[str, str], ...]
    reasoner_phases: tuple[str, ...]
    mapper_phases: tuple[str, ...]
    combined_phases: tuple[str, ...]
    runtime_seconds: float
    duration_seconds: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StrategyGateResult:
    """Aggregate strategy gate result."""

    passed: bool
    cause_topk_accuracy: float
    phase_recall: float
    phase_precision: float
    forbidden_phase_violations: int
    order_violations: int
    runtime_factor: float
    fail_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RealAudioStrategyGoldenGateReport:
    """Serialisierbares Ergebnis eines Real-Audio-Strategie-Golden-Set-Gate-Laufs."""

    gate: StrategyGateResult
    cases: list[StrategyCaseResult]
    skipped_cases: list[dict[str, str]]
    manifest_path: str
    scanned_cases: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Gibt a JSON-serializable representation zurück."""
        return {
            "gate": asdict(self.gate),
            "cases": [asdict(case) for case in self.cases],
            "skipped_cases": self.skipped_cases,
            "manifest_path": self.manifest_path,
            "scanned_cases": self.scanned_cases,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _thresholds_from_manifest(payload: dict[str, Any]) -> StrategyGateThresholds:
    raw_thresholds = payload.get("thresholds")
    raw = raw_thresholds if isinstance(raw_thresholds, dict) else {}
    return StrategyGateThresholds(
        min_cause_topk_accuracy=float(raw.get("min_cause_topk_accuracy", 0.875)),
        min_phase_recall=float(raw.get("min_phase_recall", 0.95)),
        min_phase_precision=float(raw.get("min_phase_precision", 1.0)),
        max_forbidden_phase_violations=int(raw.get("max_forbidden_phase_violations", 0)),
        max_order_violations=int(raw.get("max_order_violations", 0)),
        max_runtime_factor=float(raw.get("max_runtime_factor", 1.50)),
    )


def _string_tuple(raw_value: object, field_name: str) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list):
        raise ValueError(f"'{field_name}' must be a list when present")
    values: list[str] = []
    for value in raw_value:
        text = str(value or "").strip()
        if text:
            values.append(text)
    return tuple(values)


def _order_rules(raw_value: object) -> tuple[tuple[str, str], ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list):
        raise ValueError("'ordered_before' must be a list when present")
    rules: list[tuple[str, str]] = []
    for item in raw_value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each 'ordered_before' entry must be a two-item list")
        before = str(item[0] or "").strip()
        after = str(item[1] or "").strip()
        if before and after:
            rules.append((before, after))
    return tuple(rules)


def _dedupe_phases(*phase_lists: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    phases: list[str] = []
    for phase_list in phase_lists:
        for phase_id in phase_list:
            if phase_id not in seen:
                phases.append(phase_id)
                seen.add(phase_id)
    return tuple(phases)


def _resolve_strategy_phase_coalitions(
    seed_phases: list[str],
    *,
    is_studio_2026: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Liefert aktive §2.67-Phasenkoalitionen für den Strategy-Gate-Pfad."""
    if not seed_phases:
        return {}
    try:
        resolved = UnifiedRestorerV3.get_active_phase_coalitions(
            seed_phases,
            is_studio_2026=is_studio_2026,
        )
        if not isinstance(resolved, dict):
            return {}
        normalized: dict[str, tuple[str, ...]] = {}
        for coalition_name, members in resolved.items():
            if not isinstance(coalition_name, str):
                continue
            if not isinstance(members, tuple):
                continue
            normalized_members = tuple(phase_id for phase_id in members if isinstance(phase_id, str))
            if len(normalized_members) >= 2:
                normalized[coalition_name] = normalized_members
        return normalized
    except Exception as e:
        logger.warning(
            "real_audio_strategy_golden_gate.py::_resolve_strategy_Verarbeitungsschritt_coalitions Ersatzpfad: %s", e
        )
        return {}


def _scan_strategy_case(raw_case: dict[str, Any], repo_root: Path, target_sr: int) -> StrategyCaseResult:
    case_id = str(raw_case.get("case_id", "") or "").strip()
    if not case_id:
        raise ValueError("Strategy case is missing 'case_id'")
    rel_path = str(raw_case.get("path", "") or "").strip()
    if not rel_path:
        raise ValueError(f"Strategy case {case_id} is missing 'path'")

    material = MaterialType(str(raw_case.get("material_type", "unknown") or "unknown"))
    path = repo_root / rel_path
    if not path.exists():
        raise FileNotFoundError(str(path))

    audio, sr, imported_duration = _audio_from_import(path, target_sr=target_sr)
    max_seconds = float(raw_case.get("max_seconds", 30.0))
    if max_seconds > 0.0:
        audio = audio[: min(len(audio), int(sr * max_seconds))]

    start_time = time.time()
    scanner = DefectScanner(sample_rate=sr)
    defect_result = scanner.scan(audio, sr, material_type=material, file_ext=path.suffix)
    defect_scores = {score.defect_type.value: score.severity for score in defect_result.get_top_defects(8)}
    plan = reason_about_defects(defect_scores, material=material.value, audio=audio, sample_rate=sr)
    reasoner_phases = [str(phase) for phase in plan.recommended_phases]
    active_phase_coalitions = _resolve_strategy_phase_coalitions(reasoner_phases, is_studio_2026=False)
    mapper_phases = DefectPhaseMapper().phases_for_defect_profile(
        list(defect_result.scores.values()),
        max_phases=12,
        mode="restoration",
        material=material.value,
        phase_coalitions=active_phase_coalitions if active_phase_coalitions else None,
    )
    runtime_seconds = float(time.time() - start_time)

    combined_phases = _dedupe_phases(reasoner_phases, mapper_phases)
    combined_set = set(combined_phases)
    top_causes = tuple(cause for cause, _prob in plan.ranked_causes[:10])
    accepted_causes = _string_tuple(raw_case.get("accepted_causes"), "accepted_causes")
    cause_top_k = int(raw_case.get("cause_top_k", 3))
    cause_hit = bool(set(accepted_causes) & set(top_causes[:cause_top_k])) if accepted_causes else True

    required_phases = _string_tuple(raw_case.get("required_phases"), "required_phases")
    missing_required = tuple(phase for phase in required_phases if phase not in combined_set)
    _case_forbidden = set(_string_tuple(raw_case.get("forbidden_phases"), "forbidden_phases"))
    forbidden_phases = tuple(sorted(set(_RESTORATION_FORBIDDEN_PHASES) | _case_forbidden))
    forbidden_present = tuple(phase for phase in forbidden_phases if phase in combined_set)

    ordered_before = _order_rules(raw_case.get("ordered_before"))
    phase_index = {phase: index for index, phase in enumerate(combined_phases)}
    order_violations = tuple(
        (before, after)
        for before, after in ordered_before
        if before in phase_index and after in phase_index and phase_index[before] > phase_index[after]
    )

    phase_to_coalition: dict[str, str] = {}
    for coalition_name, members in active_phase_coalitions.items():
        coalition_name_text = str(coalition_name)
        for phase_id in members:
            phase_to_coalition[str(phase_id)] = coalition_name_text
    coalition_counts: dict[str, int] = {}
    for phase_id in mapper_phases:
        coalition_for_phase = phase_to_coalition.get(phase_id)
        if coalition_for_phase:
            coalition_counts[coalition_for_phase] = coalition_counts.get(coalition_for_phase, 0) + 1
    dominant_coalition: str = ""
    dominant_coalition_ratio = 0.0
    if coalition_counts:
        dominant_item = max(coalition_counts.items(), key=lambda item: item[1])
        dominant_coalition = str(dominant_item[0])
        dominant_count = int(dominant_item[1])
        dominant_coalition_ratio = float(dominant_count) / max(len(mapper_phases), 1)

    return StrategyCaseResult(
        case_id=case_id,
        accepted_causes=accepted_causes,
        cause_top_k=cause_top_k,
        primary_cause=str(plan.primary_cause),
        top_causes=top_causes,
        cause_hit=cause_hit,
        required_phases=required_phases,
        missing_required_phases=missing_required,
        forbidden_phases=forbidden_phases,
        forbidden_present=forbidden_present,
        ordered_before=ordered_before,
        order_violations=order_violations,
        reasoner_phases=tuple(reasoner_phases),
        mapper_phases=tuple(mapper_phases),
        combined_phases=combined_phases,
        runtime_seconds=runtime_seconds,
        duration_seconds=float(defect_result.duration_seconds),
        metadata={
            "path": rel_path,
            "material": material.value,
            "imported_duration_seconds": imported_duration,
            "scan_duration_seconds": float(defect_result.duration_seconds),
            "sample_rate": int(defect_result.sample_rate),
            "description": str(raw_case.get("description", "") or ""),
            "defect_scores": {key: float(value) for key, value in defect_scores.items()},
            "active_phase_coalitions": {name: list(members) for name, members in active_phase_coalitions.items()},
            "dominant_coalition": dominant_coalition,
            "dominant_coalition_ratio": round(float(dominant_coalition_ratio), 3),
        },
    )


def _evaluate_strategy_gate(cases: list[StrategyCaseResult], thresholds: StrategyGateThresholds) -> StrategyGateResult:
    total_cases = max(len(cases), 1)
    cause_topk_accuracy = sum(1 for case in cases if case.cause_hit) / total_cases

    required_total = sum(len(case.required_phases) for case in cases)
    missing_total = sum(len(case.missing_required_phases) for case in cases)
    phase_recall = 1.0 if required_total == 0 else (required_total - missing_total) / required_total

    forbidden_total = sum(len(case.forbidden_present) for case in cases)
    phase_precision = 1.0 if forbidden_total == 0 else 0.0
    order_violations = sum(len(case.order_violations) for case in cases)
    total_runtime = sum(case.runtime_seconds for case in cases)
    total_duration = sum(max(case.duration_seconds, 1e-9) for case in cases)
    runtime_factor = total_runtime / total_duration

    fail_reasons: list[str] = []
    if cause_topk_accuracy < thresholds.min_cause_topk_accuracy:
        fail_reasons.append(f"cause_topk_accuracy {cause_topk_accuracy:.3f} < {thresholds.min_cause_topk_accuracy:.3f}")
    if phase_recall < thresholds.min_phase_recall:
        fail_reasons.append(f"phase_recall {phase_recall:.3f} < {thresholds.min_phase_recall:.3f}")
    if phase_precision < thresholds.min_phase_precision:
        fail_reasons.append(f"phase_precision {phase_precision:.3f} < {thresholds.min_phase_precision:.3f}")
    if forbidden_total > thresholds.max_forbidden_phase_violations:
        fail_reasons.append(
            f"forbidden_phase_violations {forbidden_total} > {thresholds.max_forbidden_phase_violations}"
        )
    if order_violations > thresholds.max_order_violations:
        fail_reasons.append(f"order_violations {order_violations} > {thresholds.max_order_violations}")
    if runtime_factor > thresholds.max_runtime_factor:
        fail_reasons.append(f"runtime_factor {runtime_factor:.3f} > {thresholds.max_runtime_factor:.3f}")

    return StrategyGateResult(
        passed=not fail_reasons,
        cause_topk_accuracy=float(cause_topk_accuracy),
        phase_recall=float(phase_recall),
        phase_precision=float(phase_precision),
        forbidden_phase_violations=int(forbidden_total),
        order_violations=int(order_violations),
        runtime_factor=float(runtime_factor),
        fail_reasons=tuple(fail_reasons),
    )


def run_real_audio_strategy_golden_gate(
    *,
    manifest_path: Path,
    repo_root: Path | None = None,
    allow_missing: bool = False,
    allow_empty: bool = False,
) -> RealAudioStrategyGoldenGateReport:
    """Führt aus: the manifest-driven real-audio strategy Golden-Set gate."""
    start_time = time.time()
    manifest_path = manifest_path.resolve()
    repo_root = (repo_root or manifest_path.parents[1]).resolve()
    payload = _load_manifest(manifest_path)
    thresholds = _thresholds_from_manifest(payload)
    target_sr = int(payload.get("target_sample_rate", 48_000))

    scanned_cases: list[StrategyCaseResult] = []
    skipped_cases: list[dict[str, str]] = []
    for raw_case in payload["cases"]:
        if not isinstance(raw_case, dict):
            raise ValueError("Every real-audio strategy manifest case must be an object")
        case_id = str(raw_case.get("case_id", "") or "unknown")
        if raw_case.get("active", True) is False:
            skipped_cases.append({"case_id": case_id, "reason": "inactive"})
            continue
        try:
            scanned_cases.append(_scan_strategy_case(raw_case, repo_root, target_sr))
        except FileNotFoundError as exc:
            if not allow_missing:
                raise
            skipped_cases.append({"case_id": case_id, "reason": f"missing_file:{exc}"})

    if not scanned_cases and not allow_empty:
        raise RuntimeError("Real-audio Strategy Golden-Set gate has no scanned active cases")

    gate = _evaluate_strategy_gate(scanned_cases, thresholds)
    elapsed = float(time.time() - start_time)
    return RealAudioStrategyGoldenGateReport(
        gate=gate,
        cases=scanned_cases,
        skipped_cases=skipped_cases,
        manifest_path=str(manifest_path),
        scanned_cases=len(scanned_cases),
        elapsed_seconds=elapsed,
    )


__all__ = [
    "RealAudioStrategyGoldenGateReport",
    "StrategyCaseResult",
    "StrategyGateResult",
    "StrategyGateThresholds",
    "run_real_audio_strategy_golden_gate",
]
