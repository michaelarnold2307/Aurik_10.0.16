#!/usr/bin/env python3
"""Daily Real-Audio-Gate Trend Reporter (R5-R12).

Builds a compact daily status from existing UAT result files in audit/.
The report is fully offline and deterministic.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_R_CRITERIA = {"R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"}


@dataclass(frozen=True)
class DailyGatePoint:
    date: str
    source_file: str
    recommendation: str
    gates_passed: int
    gates_total: int
    r5_r12_passed: int
    r5_r12_total: int
    r5_r12_pass_rate: float


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        logger.warning("daily_real_audio_gate.py::_safe_int Ersatzpfad", exc_info=True)
        return default


def _safe_ts(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        return datetime.min
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        logger.warning("daily_real_audio_gate.py::_safe_ts Ersatzpfad", exc_info=True)
        return datetime.min


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug("ueberspringen unreadable UAT Ergebnis file %s: %s", path, exc)
        return None


def load_uat_runs(audit_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    runs: list[tuple[Path, dict[str, Any], datetime]] = []
    for path in sorted(audit_dir.glob("uat_results_*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        ts = _safe_ts(payload.get("generated_at"))
        runs.append((path, payload, ts))
    runs.sort(key=lambda item: (item[2], item[0].name))
    return [(p, data) for p, data, _ in runs]


def _extract_r5_r12_pass_counts(payload: dict[str, Any]) -> tuple[int, int]:
    rows = payload.get("restoration_criteria")
    if not isinstance(rows, list):
        return 0, len(_R_CRITERIA)

    by_id: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("criterion_id", "") or "")
        if cid in _R_CRITERIA:
            by_id[cid] = str(row.get("status", "") or "").upper()

    passed = sum(1 for cid in _R_CRITERIA if by_id.get(cid, "") == "PASSED")
    return passed, len(_R_CRITERIA)


def build_daily_points(runs: list[tuple[Path, dict[str, Any]]]) -> list[DailyGatePoint]:
    points: list[DailyGatePoint] = []
    for source_path, payload in runs:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        gates_passed = _safe_int(summary.get("gates_passed", 0), 0)  # type: ignore[union-attr]
        gates_total = _safe_int(summary.get("gates_total", 0), 0)  # type: ignore[union-attr]
        recommendation = str(summary.get("recommendation", "UNKNOWN") or "UNKNOWN")  # type: ignore[union-attr]

        r_passed, r_total = _extract_r5_r12_pass_counts(payload)
        rate = (float(r_passed) / float(r_total)) if r_total > 0 else 0.0

        ts = _safe_ts(payload.get("generated_at"))
        date_label = ts.date().isoformat() if ts != datetime.min else "unknown"

        points.append(
            DailyGatePoint(
                date=date_label,
                source_file=source_path.name,
                recommendation=recommendation,
                gates_passed=gates_passed,
                gates_total=gates_total,
                r5_r12_passed=r_passed,
                r5_r12_total=r_total,
                r5_r12_pass_rate=rate,
            )
        )
    return points


def build_status(points: list[DailyGatePoint]) -> dict[str, Any]:
    if not points:
        return {
            "generated_at": datetime.now().isoformat(),
            "status": "no_data",
            "latest": None,
            "trend": [],
        }

    latest = points[-1]
    latest_ready = (
        latest.gates_total > 0
        and latest.gates_passed >= latest.gates_total
        and latest.r5_r12_total > 0
        and latest.r5_r12_passed >= latest.r5_r12_total
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "status": "ready" if latest_ready else "attention",
        "latest": {
            "date": latest.date,
            "source_file": latest.source_file,
            "recommendation": latest.recommendation,
            "gates": {"passed": latest.gates_passed, "total": latest.gates_total},
            "r5_r12": {
                "passed": latest.r5_r12_passed,
                "total": latest.r5_r12_total,
                "pass_rate": latest.r5_r12_pass_rate,
            },
        },
        "trend": [
            {
                "date": p.date,
                "source_file": p.source_file,
                "recommendation": p.recommendation,
                "gates_passed": p.gates_passed,
                "gates_total": p.gates_total,
                "r5_r12_passed": p.r5_r12_passed,
                "r5_r12_total": p.r5_r12_total,
                "r5_r12_pass_rate": p.r5_r12_pass_rate,
            }
            for p in points
        ],
    }


def _markdown_from_status(status: dict[str, Any]) -> str:
    lines = [
        "# Daily Real-Audio-Gate Status",
        "",
        f"Generated: {status.get('generated_at', '')}",
        f"Status: {status.get('status', 'unknown')}",
        "",
    ]

    latest = status.get("latest")
    if isinstance(latest, dict):
        gates = latest.get("gates") if isinstance(latest.get("gates"), dict) else {}
        rset = latest.get("r5_r12") if isinstance(latest.get("r5_r12"), dict) else {}
        lines.extend(
            [
                "## Latest",
                "",
                f"- Date: {latest.get('date', '')}",
                f"- Source: {latest.get('source_file', '')}",
                f"- Recommendation: {latest.get('recommendation', 'UNKNOWN')}",
                f"- Gates: {_safe_int(gates.get('passed', 0))}/{_safe_int(gates.get('total', 0))}",  # type: ignore[union-attr]
                f"- R5-R12: {_safe_int(rset.get('passed', 0))}/{_safe_int(rset.get('total', 0))}",  # type: ignore[union-attr]
                "",
            ]
        )

    lines.extend(
        [
            "## Trend",
            "",
            "| Date | Source | Recommendation | Gates | R5-R12 |",
            "|---|---|---|---:|---:|",
        ]
    )

    trend = status.get("trend") if isinstance(status.get("trend"), list) else []
    for entry in trend:  # type: ignore[union-attr]
        if not isinstance(entry, dict):
            continue
        lines.append(
            "| {date} | {src} | {rec} | {gp}/{gt} | {rp}/{rt} |".format(
                date=str(entry.get("date", "")),
                src=str(entry.get("source_file", "")),
                rec=str(entry.get("recommendation", "UNKNOWN")),
                gp=_safe_int(entry.get("gates_passed", 0)),
                gt=_safe_int(entry.get("gates_total", 0)),
                rp=_safe_int(entry.get("r5_r12_passed", 0)),
                rt=_safe_int(entry.get("r5_r12_total", 0)),
            )
        )

    lines.append("")
    return "\n".join(lines)


def generate_daily_real_audio_gate_report(
    *,
    audit_dir: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    runs = load_uat_runs(audit_dir)
    points = build_daily_points(runs)
    status = build_status(points)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_markdown_from_status(status), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily real-audio gate trend status (R5-R12 + gates).")
    parser.add_argument("--audit-dir", default="audit", help="Directory with uat_results_*.json files")
    parser.add_argument("--json-output", default="audit/daily_real_audio_gate_status.json", help="Output JSON file")
    parser.add_argument("--md-output", default="audit/daily_real_audio_gate_status.md", help="Output Markdown file")
    args = parser.parse_args()

    status = generate_daily_real_audio_gate_report(
        audit_dir=Path(args.audit_dir),
        output_json=Path(args.json_output),
        output_md=Path(args.md_output),
    )

    return 0 if status.get("status") in {"ready", "attention", "no_data"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
