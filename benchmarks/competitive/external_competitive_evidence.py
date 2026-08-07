#!/usr/bin/env python3
"""External competitive evidence runner (Aurik vs RX11 vs CEDAR).

Builds a reproducible, matrix-aware comparison report from externally processed
outputs. This script is intended for release evidence generation where each
scenario has the same reference and input, but different processed outputs
(Aurik, iZotope RX 11, CEDAR).

Usage:
    python benchmarks/competitive/external_competitive_evidence.py \
        --manifest benchmarks/competitive/manifest_template.json \
        --output reports/competitive_external_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Project root for local imports when executed as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.mushra_evaluator import get_mushra_evaluator
from backend.file_import import load_audio_file

logger = logging.getLogger(__name__)

REQUIRED_MATERIALS: tuple[str, ...] = ("tape", "vinyl", "shellac", "digital", "vocal")
REQUIRED_DEFECT_CLASSES: tuple[str, ...] = (
    "hiss",
    "crackle",
    "dropout",
    "reverb",
    "hum",
    "codec",
)


@dataclass(frozen=True)
class ManifestItem:
    scenario_id: str
    material: str
    defect_class: str
    reference_path: str
    input_path: str
    aurik_path: str
    rx11_path: str
    cedar_path: str


class ManifestValidationError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ManifestValidationError("Manifest root must be an object")
    return data


def _require_str(value: Any, field: str, idx: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"items[{idx}].{field} must be a non-empty string")
    return value.strip()


def _parse_manifest(manifest_data: dict[str, Any]) -> list[ManifestItem]:
    items_raw = manifest_data.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise ManifestValidationError("Manifest must contain non-empty list 'items'")

    items: list[ManifestItem] = []
    for idx, entry in enumerate(items_raw):
        if not isinstance(entry, dict):
            raise ManifestValidationError(f"items[{idx}] must be an object")
        item = ManifestItem(
            scenario_id=_require_str(entry.get("scenario_id"), "scenario_id", idx),
            material=_require_str(entry.get("material"), "material", idx).lower(),
            defect_class=_require_str(entry.get("defect_class"), "defect_class", idx).lower(),
            reference_path=_require_str(entry.get("reference_path"), "reference_path", idx),
            input_path=_require_str(entry.get("input_path"), "input_path", idx),
            aurik_path=_require_str(entry.get("aurik_path"), "aurik_path", idx),
            rx11_path=_require_str(entry.get("rx11_path"), "rx11_path", idx),
            cedar_path=_require_str(entry.get("cedar_path"), "cedar_path", idx),
        )
        if item.material not in REQUIRED_MATERIALS:
            raise ManifestValidationError(
                f"items[{idx}].material='{item.material}' not in required set {REQUIRED_MATERIALS}"
            )
        if item.defect_class not in REQUIRED_DEFECT_CLASSES:
            raise ManifestValidationError(
                f"items[{idx}].defect_class='{item.defect_class}' not in required set {REQUIRED_DEFECT_CLASSES}"
            )
        items.append(item)
    return items


def _resolve_path(root: Path, path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else (root / p)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    result = load_audio_file(str(path), target_sr=48_000, mono=False, do_carrier_analysis=False)
    if not isinstance(result, dict):
        raise RuntimeError(f"Failed to load audio: {path} (no result dict)")
    if result.get("error"):
        raise RuntimeError(f"Failed to load audio: {path} ({result['error']})")
    audio = result.get("audio")
    sr = result.get("sr")
    if audio is None or sr is None:
        raise RuntimeError(f"Failed to load audio: {path} (missing audio/sr)")
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.size == 0:
        raise RuntimeError(f"Failed to load audio: {path} (empty audio)")
    return audio_np, int(sr)


def _cell_key(material: str, defect_class: str) -> str:
    return f"{material}::{defect_class}"


def _evaluate_manifest(
    manifest_path: Path,
    items: list[ManifestItem],
    require_full_matrix: bool,
    require_cedar: bool,
) -> dict[str, Any]:
    evaluator = get_mushra_evaluator()
    manifest_root = manifest_path.parent

    item_results: list[dict[str, Any]] = []
    cell_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for it in items:
        ref_path = _resolve_path(manifest_root, it.reference_path)
        in_path = _resolve_path(manifest_root, it.input_path)
        aurik_path = _resolve_path(manifest_root, it.aurik_path)
        rx11_path = _resolve_path(manifest_root, it.rx11_path)
        cedar_path = _resolve_path(manifest_root, it.cedar_path)

        # Input is loaded for completeness and reproducibility trace.
        _, _ = _load_audio(in_path)
        ref_audio, sr = _load_audio(ref_path)
        aurik_audio, _ = _load_audio(aurik_path)
        rx11_audio, _ = _load_audio(rx11_path)

        cedar_available = cedar_path.exists()
        cedar_audio: np.ndarray | None = None
        if cedar_available:
            cedar_audio, _ = _load_audio(cedar_path)
        elif require_cedar:
            raise RuntimeError(f"CEDAR output missing for scenario '{it.scenario_id}': {cedar_path}")

        aurik_res = evaluator.evaluate(ref_audio, aurik_audio, sr, compute_anchor=False)
        rx11_res = evaluator.evaluate(ref_audio, rx11_audio, sr, compute_anchor=False)

        cedar_res = None
        if cedar_audio is not None:
            cedar_res = evaluator.evaluate(ref_audio, cedar_audio, sr, compute_anchor=False)

        row = {
            "scenario_id": it.scenario_id,
            "material": it.material,
            "defect_class": it.defect_class,
            "aurik_oqs": float(aurik_res.mushra_score),
            "rx11_oqs": float(rx11_res.mushra_score),
            "cedar_oqs": float(cedar_res.mushra_score) if cedar_res else None,
            "aurik_grade": aurik_res.grade,
            "rx11_grade": rx11_res.grade,
            "cedar_grade": cedar_res.grade if cedar_res else None,
            "aurik_minus_rx11": float(aurik_res.mushra_score - rx11_res.mushra_score),
            "aurik_minus_cedar": (
                float(aurik_res.mushra_score - cedar_res.mushra_score) if cedar_res is not None else None
            ),
            "paths": {
                "reference": str(ref_path),
                "input": str(in_path),
                "aurik": str(aurik_path),
                "rx11": str(rx11_path),
                "cedar": str(cedar_path),
            },
        }
        item_results.append(row)

        k = _cell_key(it.material, it.defect_class)
        cell_acc[k]["aurik"].append(row["aurik_oqs"])  # type: ignore[arg-type]
        cell_acc[k]["rx11"].append(row["rx11_oqs"])  # type: ignore[arg-type]
        if row["cedar_oqs"] is not None:
            cell_acc[k]["cedar"].append(float(row["cedar_oqs"]))  # type: ignore[arg-type]

    required_cells = {_cell_key(m, d) for m in REQUIRED_MATERIALS for d in REQUIRED_DEFECT_CLASSES}
    present_cells = set(cell_acc.keys())
    missing_cells = sorted(required_cells - present_cells)

    if require_full_matrix and missing_cells:
        raise RuntimeError("Manifest does not cover full 5x6 matrix. Missing cells: " + ", ".join(missing_cells))

    cell_summary: dict[str, Any] = {}
    rx11_regressions: list[str] = []
    cedar_regressions: list[str] = []

    for key in sorted(present_cells):
        c = cell_acc[key]
        aurik_mean = float(np.mean(c["aurik"])) if c["aurik"] else float("nan")
        rx11_mean = float(np.mean(c["rx11"])) if c["rx11"] else float("nan")
        cedar_mean = float(np.mean(c["cedar"])) if c.get("cedar") else None

        aurik_minus_rx11 = aurik_mean - rx11_mean
        aurik_minus_cedar = (aurik_mean - cedar_mean) if cedar_mean is not None else None

        if math.isfinite(aurik_minus_rx11) and aurik_minus_rx11 < 0:
            rx11_regressions.append(key)
        if aurik_minus_cedar is not None and aurik_minus_cedar < 0:
            cedar_regressions.append(key)

        cell_summary[key] = {
            "aurik_mean_oqs": round(aurik_mean, 3),
            "rx11_mean_oqs": round(rx11_mean, 3),
            "cedar_mean_oqs": (round(cedar_mean, 3) if cedar_mean is not None else None),
            "aurik_minus_rx11": round(aurik_minus_rx11, 3),
            "aurik_minus_cedar": (round(aurik_minus_cedar, 3) if aurik_minus_cedar is not None else None),
            "n_items": len(c["aurik"]),
        }

    aurik_vals = np.array([r["aurik_oqs"] for r in item_results], dtype=np.float64)
    rx11_vals = np.array([r["rx11_oqs"] for r in item_results], dtype=np.float64)
    cedar_vals = np.array(
        [r["cedar_oqs"] for r in item_results if r["cedar_oqs"] is not None],
        dtype=np.float64,
    )

    wins_rx11 = int(np.sum(aurik_vals > rx11_vals))
    wins_cedar = int(np.sum(aurik_vals > cedar_vals)) if cedar_vals.size > 0 else None

    result = {
        "manifest": str(manifest_path),
        "items_evaluated": len(item_results),
        "matrix": {
            "required_materials": list(REQUIRED_MATERIALS),
            "required_defect_classes": list(REQUIRED_DEFECT_CLASSES),
            "required_cells": len(required_cells),
            "present_cells": len(present_cells),
            "missing_cells": missing_cells,
            "full_matrix_covered": len(missing_cells) == 0,
        },
        "summary": {
            "aurik_mean_oqs": round(float(np.mean(aurik_vals)), 3),
            "rx11_mean_oqs": round(float(np.mean(rx11_vals)), 3),
            "cedar_mean_oqs": (round(float(np.mean(cedar_vals)), 3) if cedar_vals.size > 0 else None),
            "aurik_vs_rx11_wins": wins_rx11,
            "aurik_vs_rx11_total": len(item_results),
            "aurik_vs_cedar_wins": wins_cedar,
            "aurik_vs_cedar_total": (int(cedar_vals.size) if cedar_vals.size > 0 else None),
        },
        "gates": {
            "aurik_vs_rx11_majority_win": wins_rx11 > (len(item_results) / 2.0),
            "aurik_vs_rx11_no_stratified_regressions": len(rx11_regressions) == 0,
            "aurik_vs_cedar_majority_win": (
                (wins_cedar > (int(cedar_vals.size) / 2.0)) if wins_cedar is not None else None
            ),
            "aurik_vs_cedar_no_stratified_regressions": (
                (len(cedar_regressions) == 0) if cedar_vals.size > 0 else None
            ),
            "release_competitive_ready": False,
        },
        "regressions": {
            "vs_rx11_cells": rx11_regressions,
            "vs_cedar_cells": cedar_regressions,
        },
        "cell_summary": cell_summary,
        "items": item_results,
    }

    rx11_ok = (
        result["gates"]["aurik_vs_rx11_majority_win"] and result["gates"]["aurik_vs_rx11_no_stratified_regressions"]  # type: ignore[index]
    )
    cedar_ok = True
    if require_cedar:
        cedar_ok = bool(
            result["gates"]["aurik_vs_cedar_majority_win"]  # type: ignore[index]
            and result["gates"]["aurik_vs_cedar_no_stratified_regressions"]  # type: ignore[index]
        )

    matrix_ok = result["matrix"]["full_matrix_covered"] if require_full_matrix else True  # type: ignore[index]
    result["gates"]["release_competitive_ready"] = bool(rx11_ok and cedar_ok and matrix_ok)  # type: ignore[index]

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run external competitive evidence (Aurik vs RX11 vs CEDAR) with OQS and 5x6 stratified gates.")
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/competitive_external_report.json",
        help="Output report path.",
    )
    parser.add_argument(
        "--allow-partial-matrix",
        action="store_true",
        help="Allow manifests that do not cover all 5x6 matrix cells.",
    )
    parser.add_argument(
        "--allow-missing-cedar",
        action="store_true",
        help="Allow missing CEDAR outputs (CEDAR gates become optional).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 2

    try:
        manifest_data = _read_json(manifest_path)
        items = _parse_manifest(manifest_data)
        report = _evaluate_manifest(
            manifest_path=manifest_path,
            items=items,
            require_full_matrix=not args.allow_partial_matrix,
            require_cedar=not args.allow_missing_cedar,
        )
    except (ManifestValidationError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        logger.error("Competitive evidence run failed: %s", exc)
        return 3

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path
    os.makedirs(output_path.parent, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Report written: %s", output_path)
    logger.info(
        "release_competitive_ready=%s | aurik_mean=%.2f vs rx11_mean=%.2f",
        report["gates"]["release_competitive_ready"],
        report["summary"]["aurik_mean_oqs"],
        report["summary"]["rx11_mean_oqs"],
    )
    if report["summary"]["cedar_mean_oqs"] is not None:
        logger.info("cedar_mean=%.2f", report["summary"]["cedar_mean_oqs"])

    return 0 if report["gates"]["release_competitive_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
