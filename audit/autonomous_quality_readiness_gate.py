#!/usr/bin/env python3
"""Autonomous quality-readiness gate for Aurik's non-interactive restoration flow.

from typing import Any
The gate does not ask the user for choices. It inspects three inputs that decide
whether Aurik can credibly claim fully automatic world-class restoration:

- real-audio corpus coverage and file integrity
- latest automatic quality benchmark report
- persisted MUSHRA calibration artifact for the MERT/OQS proxy

Default mode is report-only. Use ``--strict`` and/or the ``--require-*`` flags in
CI when these readiness dimensions should block a release.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
_DEFAULT_MATERIALS = {"shellac", "vinyl", "tape", "reel_tape", "cassette", "digital"}


@dataclass(frozen=True)
class ReadinessThresholds:
    """Minimum evidence thresholds for autonomous real-audio readiness."""

    min_entries: int = 20
    min_materials: int = 4
    min_vocal_entries: int = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_load_yaml(path: Path) -> dict[str, Any] | None:  # type: ignore[name-defined]
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("readiness: manifest unreadable %s: %s", path, exc)
        return None
    return raw if isinstance(raw, dict) else None


def _safe_load_json(path: Path) -> dict[str, Any] | None:  # type: ignore[name-defined]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("readiness: json unreadable %s: %s", path, exc)
        return None
    return raw if isinstance(raw, dict) else None


def collect_corpus_inventory(corpus_root: Path, thresholds: ReadinessThresholds) -> dict[str, Any]:  # type: ignore[name-defined]
    """Collect manifest coverage and file-integrity evidence from ``corpus/``."""
    corpus_root = Path(corpus_root)
    manifests = sorted(corpus_root.glob("*/manifest.yaml"))

    entries_total = 0
    files_present = 0
    audio_entries = 0
    vocal_entries = 0
    materials_with_entries: set[str] = set()
    conditions: dict[str, int] = {}
    missing_files: list[str] = []
    missing_license: list[str] = []
    missing_attribution: list[str] = []
    malformed_manifests: list[str] = []
    synthetic_entries = 0

    for manifest_path in manifests:
        manifest = _safe_load_yaml(manifest_path)
        if manifest is None:
            malformed_manifests.append(str(manifest_path))
            continue

        material = str(manifest.get("material", manifest_path.parent.name) or manifest_path.parent.name)
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            malformed_manifests.append(str(manifest_path))
            continue
        if entries:
            materials_with_entries.add(material)

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                malformed_manifests.append(f"{manifest_path}#{index}")
                continue

            entries_total += 1
            if bool(entry.get("vocal", False)):
                vocal_entries += 1
            condition = str(entry.get("condition", "unknown") or "unknown")
            conditions[condition] = conditions.get(condition, 0) + 1

            source_text = " ".join(
                str(entry.get(key, "") or "") for key in ("source", "source_attribution", "license")
            ).lower()
            if "synthetic" in source_text or "synthetisch" in source_text:
                synthetic_entries += 1

            if not entry.get("license"):
                missing_license.append(f"{manifest_path}:{index}")
            if not entry.get("source_attribution"):
                missing_attribution.append(f"{manifest_path}:{index}")

            file_value = str(entry.get("file", "") or "")
            if not file_value:
                missing_files.append(f"{manifest_path}:{index}:<missing file field>")
                continue
            if Path(file_value).suffix.lower() in _AUDIO_EXTENSIONS:
                audio_entries += 1
            audio_path = manifest_path.parent / file_value
            if audio_path.exists():
                files_present += 1
            else:
                missing_files.append(str(audio_path))

    corpus_ready = (
        corpus_root.is_dir()
        and entries_total >= thresholds.min_entries
        and len(materials_with_entries) >= thresholds.min_materials
        and vocal_entries >= thresholds.min_vocal_entries
        and not missing_files
        and not missing_license
        and not missing_attribution
        and not malformed_manifests
    )

    return {
        "root": str(corpus_root),
        "exists": corpus_root.is_dir(),
        "manifest_count": len(manifests),
        "entries_total": entries_total,
        "audio_entries": audio_entries,
        "files_present": files_present,
        "files_missing_count": len(missing_files),
        "missing_files": missing_files[:25],
        "materials_with_entries": sorted(materials_with_entries),
        "materials_with_entries_count": len(materials_with_entries),
        "conditions": dict(sorted(conditions.items())),
        "vocal_entries": vocal_entries,
        "synthetic_entries": synthetic_entries,
        "missing_license_count": len(missing_license),
        "missing_attribution_count": len(missing_attribution),
        "malformed_manifest_count": len(malformed_manifests),
        "malformed_manifests": malformed_manifests[:25],
        "thresholds": {
            "min_entries": thresholds.min_entries,
            "min_materials": thresholds.min_materials,
            "min_vocal_entries": thresholds.min_vocal_entries,
        },
        "ready": bool(corpus_ready),
    }


def load_latest_quality_benchmark(results_dir: Path) -> dict[str, Any]:  # type: ignore[name-defined]
    """Load the newest benchmark_*.json produced by auto_quality_benchmark."""
    results_dir = Path(results_dir)
    candidates = sorted(results_dir.glob("benchmark_*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    if not candidates:
        return {
            "results_dir": str(results_dir),
            "status": "no_report",
            "ready": False,
            "latest_report": "",
        }

    latest = candidates[-1]
    payload = _safe_load_json(latest) or {}
    aggregate_raw = payload.get("aggregate")
    aggregate: dict[str, Any] = dict(aggregate_raw) if isinstance(aggregate_raw, dict) else {}  # type: ignore[name-defined]
    files_total = int(payload.get("files_total", payload.get("files", 0)) or 0)
    files_ok = int(payload.get("files_ok", 0) or 0)
    files_failed = int(payload.get("files_failed", 0) or 0)
    has_pipeline_telemetry = "aurik_pipeline_fallbacks" in payload and "aurik_pipeline_ok" in payload
    aurik_pipeline_fallbacks = int(payload.get("aurik_pipeline_fallbacks", 0) or 0)
    aurik_pipeline_ok = int(payload.get("aurik_pipeline_ok", files_ok - aurik_pipeline_fallbacks) or 0)
    if not has_pipeline_telemetry:
        aurik_pipeline_ok = 0
        aurik_pipeline_fallbacks = 0
    coverage_ratio = float(payload.get("coverage_ratio", 1.0 if files_total > 0 else 0.0) or 0.0)
    ready = bool(
        files_total > 0
        and files_ok > 0
        and files_failed == 0
        and has_pipeline_telemetry
        and aurik_pipeline_fallbacks == 0
        and coverage_ratio >= 0.999
    )

    return {
        "results_dir": str(results_dir),
        "status": "ready" if ready else "attention",
        "ready": ready,
        "latest_report": str(latest),
        "files_total": files_total,
        "files_ok": files_ok,
        "files_failed": files_failed,
        "has_pipeline_telemetry": has_pipeline_telemetry,
        "aurik_pipeline_ok": aurik_pipeline_ok,
        "aurik_pipeline_fallbacks": aurik_pipeline_fallbacks,
        "coverage_ratio": coverage_ratio,
        "aggregate": aggregate,
    }


def _default_calibration_candidates() -> list[Path]:
    data_dir = Path(os.environ.get("AURIK_DATA_DIR", str(Path.home() / ".aurik")))
    return [
        data_dir / "mushra_calibration.json",
        Path("reports") / "mushra_calibration.json",
        Path("reports") / "mushra_calibration_v2.json",
    ]


def load_mushra_calibration_status(calibration_artifact: Path | None = None) -> dict[str, Any]:  # type: ignore[name-defined]
    """Inspect persisted MUSHRA calibration without requiring a listener at runtime."""
    candidates = [Path(calibration_artifact)] if calibration_artifact is not None else _default_calibration_candidates()
    for candidate in candidates:
        payload = _safe_load_json(candidate)
        if not payload:
            continue
        weights = payload.get("calibrated_weights")
        stage = int(payload.get("calibration_stage", 0) or 0)
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        ready = bool(isinstance(weights, dict) and weights and stage >= 2 and confidence > 0.0)
        return {
            "status": "ready" if ready else "attention",
            "ready": ready,
            "artifact": str(candidate),
            "calibration_stage": stage,
            "confidence": confidence,
            "weights_count": len(weights) if isinstance(weights, dict) else 0,
            "candidate_paths": [str(p) for p in candidates],
        }

    return {
        "status": "missing",
        "ready": False,
        "artifact": "",
        "calibration_stage": 1,
        "confidence": 0.0,
        "weights_count": 0,
        "candidate_paths": [str(p) for p in candidates],
    }


def build_autonomous_quality_readiness(
    *,
    corpus_root: Path,
    benchmark_results_dir: Path,
    calibration_artifact: Path | None = None,
    thresholds: ReadinessThresholds | None = None,
    require_benchmark: bool = False,
    require_panel_calibration: bool = False,
) -> dict[str, Any]:  # type: ignore[name-defined]
    """Build the non-interactive readiness verdict for Aurik quality evidence."""
    thresholds = thresholds or ReadinessThresholds()
    corpus = collect_corpus_inventory(corpus_root, thresholds)
    benchmark = load_latest_quality_benchmark(benchmark_results_dir)
    calibration = load_mushra_calibration_status(calibration_artifact)

    blocking_reasons: list[str] = []
    advisory_reasons: list[str] = []

    if not corpus.get("ready", False):
        blocking_reasons.append("corpus_not_ready")
    _benchmark_missing_schema = not bool(benchmark.get("has_pipeline_telemetry", False))
    _benchmark_partial = float(benchmark.get("coverage_ratio", 0.0) or 0.0) < 0.999
    if require_benchmark and not benchmark.get("ready", False):
        blocking_reasons.append("quality_benchmark_missing_or_failed")
    elif not benchmark.get("ready", False):
        if _benchmark_missing_schema:
            advisory_reasons.append("quality_benchmark_missing_pipeline_telemetry")
        elif _benchmark_partial:
            advisory_reasons.append("quality_benchmark_partial_corpus_coverage")
        else:
            advisory_reasons.append("quality_benchmark_missing_or_attention")
    if require_panel_calibration and not calibration.get("ready", False):
        blocking_reasons.append("mushra_panel_calibration_missing")
    elif not calibration.get("ready", False):
        advisory_reasons.append("mushra_panel_calibration_missing")

    if blocking_reasons:
        status = "blocked"
    elif advisory_reasons:
        status = "attention"
    else:
        status = "ready"

    return {
        "generated_at": _utc_now(),
        "status": status,
        "autonomy": {
            "manual_action_required": False,
            "allowed_user_decisions": ["mode_selection"],
            "runtime_user_prompts": False,
            "strict_mode_available": True,
        },
        "blocking_reasons": blocking_reasons,
        "advisory_reasons": advisory_reasons,
        "corpus": corpus,
        "quality_benchmark": benchmark,
        "mushra_calibration": calibration,
        "next_automated_actions": _next_automated_actions(blocking_reasons, advisory_reasons),
    }


def _next_automated_actions(blocking_reasons: list[str], advisory_reasons: list[str]) -> list[str]:
    reasons = set(blocking_reasons).union(advisory_reasons)
    actions: list[str] = []
    if "corpus_not_ready" in reasons:
        actions.append(
            "Populate corpus manifests with >=20 licensed audio files across >=4 materials, then rerun corpus gates."
        )
    if "quality_benchmark_missing_or_failed" in reasons or "quality_benchmark_missing_or_attention" in reasons:
        actions.append("Run benchmarks/quality/auto_quality_benchmark.py on corpus/ and store benchmark_*.json.")
    if "quality_benchmark_missing_pipeline_telemetry" in reasons:
        actions.append(
            "Re-run benchmarks/quality/auto_quality_benchmark.py with the current schema so aurik_pipeline_ok/fallbacks are recorded."
        )
    if "quality_benchmark_partial_corpus_coverage" in reasons:
        actions.append(
            "Run the full corpus benchmark without --max-files before treating the evidence as release-ready."
        )
    if "mushra_panel_calibration_missing" in reasons:
        actions.append("Import or generate a MUSHRA calibration artifact via MertMushraProxy.calibrate_from_panel().")
    return actions


def generate_autonomous_quality_readiness_report(
    *,
    corpus_root: Path,
    benchmark_results_dir: Path,
    output_json: Path,
    calibration_artifact: Path | None = None,
    thresholds: ReadinessThresholds | None = None,
    require_benchmark: bool = False,
    require_panel_calibration: bool = False,
) -> dict[str, Any]:  # type: ignore[name-defined]
    report = build_autonomous_quality_readiness(
        corpus_root=corpus_root,
        benchmark_results_dir=benchmark_results_dir,
        calibration_artifact=calibration_artifact,
        thresholds=thresholds,
        require_benchmark=require_benchmark,
        require_panel_calibration=require_panel_calibration,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate autonomous quality-readiness status for Aurik.")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument(
        "--benchmark-results-dir",
        default="benchmarks/quality/results",
        help="Directory containing benchmark_*.json reports",
    )
    parser.add_argument("--calibration-artifact", default="", help="Optional MUSHRA calibration artifact JSON")
    parser.add_argument("--output-json", default="audit/autonomous_quality_readiness.json")
    parser.add_argument("--min-entries", type=int, default=20)
    parser.add_argument("--min-materials", type=int, default=4)
    parser.add_argument("--min-vocal-entries", type=int, default=1)
    parser.add_argument("--require-benchmark", action="store_true")
    parser.add_argument("--require-panel-calibration", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit 1 unless status is ready")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    thresholds = ReadinessThresholds(
        min_entries=max(0, int(args.min_entries)),
        min_materials=max(0, int(args.min_materials)),
        min_vocal_entries=max(0, int(args.min_vocal_entries)),
    )
    calibration_artifact = Path(args.calibration_artifact) if str(args.calibration_artifact).strip() else None
    report = generate_autonomous_quality_readiness_report(
        corpus_root=Path(args.corpus_root),
        benchmark_results_dir=Path(args.benchmark_results_dir),
        calibration_artifact=calibration_artifact,
        output_json=Path(args.output_json),
        thresholds=thresholds,
        require_benchmark=bool(args.require_benchmark),
        require_panel_calibration=bool(args.require_panel_calibration),
    )
    print(json.dumps({"status": report["status"], "blocking_reasons": report["blocking_reasons"]}, ensure_ascii=False))
    return 1 if args.strict and report.get("status") != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
