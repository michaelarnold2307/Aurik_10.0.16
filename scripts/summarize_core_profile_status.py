#!/usr/bin/env python3
"""Summarize core artifact state in a compact table.

Reports for core artifacts:
- file presence
- manifest entry presence
- manifest path/size/hash consistency (if file exists)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "manifest.json"


@dataclass(frozen=True)
class Artifact:
    name: str
    rel_path: str
    fallback: str


ARTIFACTS = [
    Artifact("fcpe", "models/fcpe/fcpe.onnx", "models/crepe/crepe.onnx|models/rmvpe/rmvpe.onnx"),
    Artifact("sgmse_plus", "models/sgmse_plus/sgmse_plus.ts", "wpe_dsp"),
    Artifact(
        "versa",
        "models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        "pqs_dsp",
    ),
    Artifact(
        "flow_matching",
        "models/flow_matching/flow_matching.onnx",
        "models/cqtdiff/score_network.pt|models/diffwave/diffwave_model.onnx",
    ),
    Artifact("gacela", "models/gacela/model/01_400000.pt", "dsp_exciter"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _yn(v: bool | None) -> str:
    if v is None:
        return "n/a"
    return "yes" if v else "no"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize core profile status")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary to stdout (in addition to table).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write JSON summary file.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_models: list[Any] = []
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest_models = data.get("models", []) if isinstance(data, dict) else []

    by_name: dict[str, dict] = {}
    if isinstance(manifest_models, list):
        for m in manifest_models:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                by_name[m["name"]] = m

    headers = [
        "artifact",
        "file",
        "manifest",
        "source_ckpt",
        "path_ok",
        "size_ok",
        "sha_ok",
        "release_mode",
        "runtime_ready",
        "resolved_by",
    ]
    widths = [14, 8, 10, 12, 8, 8, 8, 13, 14, 28]

    def row(cols: list[str]) -> str:
        return " ".join(c.ljust(w) for c, w in zip(cols, widths))

    print(row(headers))
    print("-" * (sum(widths) + len(widths) - 1))

    missing_files = 0
    missing_manifest = 0
    runtime_not_ready = 0
    rows: list[dict[str, str | bool | None]] = []

    for a in ARTIFACTS:
        path = ROOT / a.rel_path
        file_exists = path.exists()
        if not file_exists:
            missing_files += 1

        entry = by_name.get(a.name)
        manifest_exists = entry is not None
        if not manifest_exists:
            missing_manifest += 1

        path_ok: bool | None = None
        size_ok: bool | None = None
        sha_ok: bool | None = None
        source_ckpt: bool | None = None
        manifest_target_exists: bool = False

        if manifest_exists:
            bundled_path = str(entry.get("bundled_path", ""))  # type: ignore[union-attr]
            valid_paths = {a.rel_path}
            if a.name == "sgmse_plus":
                valid_paths.add("models/sgmse_plus/sgmse_plus.ts")
            path_ok = bundled_path in valid_paths

            target_path = ROOT / bundled_path if bundled_path else Path()
            manifest_target_exists = target_path.exists()
            if manifest_target_exists:
                size_ok = int(entry.get("size_bytes", -1)) == target_path.stat().st_size  # type: ignore[union-attr]
                sha_ok = str(entry.get("sha256", "")) == _sha256(target_path)  # type: ignore[union-attr]

        if a.name == "fcpe":
            crepe_fallback = (ROOT / "models/crepe/crepe.onnx").exists()
            rmvpe_backup = (ROOT / "models/rmvpe/rmvpe.onnx").exists()
            fallback_ok = crepe_fallback or rmvpe_backup
            runtime_ready = file_exists or fallback_ok
            resolved_by = (
                "primary"
                if file_exists
                else ("crepe_fallback" if crepe_fallback else ("rmvpe_backup" if rmvpe_backup else "missing"))
            )
            release_mode = "primary" if file_exists else ("fallback" if fallback_ok else "blocked")
        elif a.name == "sgmse_plus":
            source_ckpt = (ROOT / "models/sgmse_plus/sgmse_wsj0_reverb.ckpt").exists()
            runtime_ready = True
            ts_fallback = (ROOT / "models/sgmse_plus/sgmse_plus.ts").exists()
            resolved_by = (
                "primary" if file_exists else ("torchscript_fallback" if ts_fallback else "wpe_dsp_fallback")
            )  # §V6: logger.warning handled at call site
            release_mode = "primary" if file_exists else "fallback"
        elif a.name == "versa":
            runtime_ready = True
            resolved_by = "primary" if file_exists else "pqs_dsp_fallback"
            release_mode = "primary" if file_exists else "fallback"
        elif a.name == "flow_matching":
            fallback_ok = (ROOT / "models/cqtdiff/score_network.pt").exists() or (
                ROOT / "models/diffwave/diffwave_model.onnx"
            ).exists()
            runtime_ready = file_exists or fallback_ok
            resolved_by = "primary" if file_exists else ("cqtdiff_or_diffwave_fallback" if fallback_ok else "missing")
            release_mode = "primary" if file_exists else ("fallback" if fallback_ok else "blocked")
        else:
            runtime_ready = True
            resolved_by = "primary" if file_exists else "dsp_exciter_fallback"
            release_mode = "primary" if file_exists else "fallback"

        if not runtime_ready:
            runtime_not_ready += 1

        print(
            row(
                [
                    a.name,
                    _yn(file_exists),
                    _yn(manifest_exists),
                    _yn(source_ckpt),
                    _yn(path_ok),
                    _yn(size_ok),
                    _yn(sha_ok),
                    release_mode,
                    _yn(runtime_ready),
                    resolved_by,
                ]
            )
        )

        rows.append(
            {
                "artifact": a.name,
                "path": a.rel_path,
                "file": file_exists,
                "manifest": manifest_exists,
                "source_checkpoint": source_ckpt,
                "path_ok": path_ok,
                "size_ok": size_ok,
                "sha_ok": sha_ok,
                "release_mode": release_mode,
                "runtime_ready": runtime_ready,
                "resolved_by": resolved_by,
            }
        )

    print("\nSummary:")
    print(f"  missing_files={missing_files}")
    print(f"  missing_manifest_entries={missing_manifest}")
    print(f"  runtime_not_ready={runtime_not_ready}")

    payload = {
        "summary": {
            "missing_files": missing_files,
            "missing_manifest_entries": missing_manifest,
            "runtime_not_ready": runtime_not_ready,
        },
        "artifacts": rows,
    }

    if args.json:
        print("\nJSON:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"JSON written: {out}")

    # Non-failing summary script
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
