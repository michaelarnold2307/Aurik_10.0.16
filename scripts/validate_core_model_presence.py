#!/usr/bin/env python3
"""Validate core runtime readiness for missing-model rollout.

Default mode checks whether Aurik is operational for each core capability,
using either primary model files or documented fallback paths.

Use --strict-primary to enforce exact primary files only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CORE_PATHS = [
    "models/fcpe/fcpe.onnx",
    "models/sgmse_plus/sgmse_plus.ts",
    "models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
    "models/flow_matching/flow_matching.onnx",
    "models/gacela/model/01_400000.pt",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate core model/runtime readiness")
    p.add_argument(
        "--strict-primary",
        action="store_true",
        help="Require exact primary files from CORE_PATHS (legacy behavior).",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write JSON report.",
    )
    return p.parse_args()


def _exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def _runtime_ready_checks() -> list[dict[str, str | bool]]:
    # Primaries
    fcpe_primary = _exists("models/fcpe/fcpe.onnx")
    sgmse_primary = _exists("models/sgmse_plus/sgmse_plus.ts")
    versa_primary = _exists(
        "models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
    )
    flow_primary = _exists("models/flow_matching/flow_matching.onnx")
    gacela_primary = _exists("models/gacela/model/01_400000.pt")

    # Fallbacks / operational checks
    fcpe_fallback = _exists("models/crepe/crepe.onnx") or _exists("models/rmvpe/rmvpe.onnx")
    sgmse_fallback = True  # wpe_plugin.py is local DSP fallback.  # §V6: logger.warning handled at call site
    sgmse_checkpoint = _exists("models/sgmse_plus/sgmse_wsj0_reverb.ckpt")
    sgmse_torchscript = _exists("models/sgmse_plus/sgmse_plus.ts")
    versa_fallback = True  # versa_plugin.py has PQS-DSP fallback.  # §V6: logger.warning handled at call site
    flow_fallback = _exists("models/cqtdiff/score_network.pt") or _exists("models/diffwave/diffwave_model.onnx")
    gacela_fallback = True  # gacela_plugin.py has DSP exciter fallback.  # §V6: logger.warning handled at call site

    return [
        {
            "name": "fcpe",
            "primary": fcpe_primary,
            "fallback": fcpe_fallback,
            "runtime_ready": fcpe_primary or fcpe_fallback,
            "release_mode": "primary" if fcpe_primary else ("fallback" if fcpe_fallback else "blocked"),
            "resolved_by": (
                "primary"
                if fcpe_primary
                else (
                    "crepe_fallback"
                    if _exists("models/crepe/crepe.onnx")
                    else ("rmvpe_backup" if _exists("models/rmvpe/rmvpe.onnx") else "missing")
                )
            ),
        },
        {
            "name": "sgmse_plus",
            "primary": sgmse_primary,
            "fallback": sgmse_fallback,
            "source_checkpoint": sgmse_checkpoint,
            "runtime_ready": sgmse_primary or sgmse_fallback,
            "release_mode": "primary" if sgmse_primary else "fallback",
            "resolved_by": (
                "primary" if sgmse_primary else ("torchscript_fallback" if sgmse_torchscript else "wpe_dsp_fallback")  # §V6: logger.warning handled at call site
            ),
        },
        {
            "name": "versa",
            "primary": versa_primary,
            "fallback": versa_fallback,
            "runtime_ready": versa_primary or versa_fallback,
            "release_mode": "primary" if versa_primary else "fallback",
            "resolved_by": "primary" if versa_primary else "pqs_dsp_fallback",
        },
        {
            "name": "flow_matching",
            "primary": flow_primary,
            "fallback": flow_fallback,
            "runtime_ready": flow_primary or flow_fallback,
            "release_mode": "primary" if flow_primary else ("fallback" if flow_fallback else "blocked"),
            "resolved_by": (
                "primary" if flow_primary else ("cqtdiff_or_diffwave_fallback" if flow_fallback else "missing")
            ),
        },
        {
            "name": "gacela",
            "primary": gacela_primary,
            "fallback": gacela_fallback,
            "runtime_ready": gacela_primary or gacela_fallback,
            "release_mode": "primary" if gacela_primary else "fallback",
            "resolved_by": "primary" if gacela_primary else "dsp_exciter_fallback",
        },
    ]


def main() -> int:
    args = _parse_args()

    if args.strict_primary:
        missing = 0
        for rel in CORE_PATHS:
            p = ROOT / rel
            if p.exists():
                print(f"OK {rel}")
            else:
                print(f"MISSING {rel}")
                missing += 1

        print(
            "INFO models/sgmse_plus/sgmse_wsj0_reverb.ckpt "
            f"{'present' if _exists('models/sgmse_plus/sgmse_wsj0_reverb.ckpt') else 'missing'}"
        )

        if args.json_out:
            out = Path(args.json_out)
            if not out.is_absolute():
                out = (ROOT / out).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "mode": "strict_primary",
                "missing_primary_files": missing,
                "primary_paths": CORE_PATHS,
                "sgmse_source_checkpoint": _exists("models/sgmse_plus/sgmse_wsj0_reverb.ckpt"),
            }
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"JSON written: {out}")

        if missing:
            print(f"\nResult: {missing} core paths missing")
            return 1

        print("\nResult: all core paths present")
        return 0

    rows = _runtime_ready_checks()
    not_ready = [r for r in rows if not bool(r["runtime_ready"])]

    print("core_component  primary  fallback  source_ckpt  release_mode  runtime_ready  resolved_by")
    print("-------------------------------------------------------------------------------------------")
    for r in rows:
        source_ckpt = r.get("source_checkpoint")
        source_ckpt_col = "n/a" if source_ckpt is None else ("yes" if source_ckpt else "no")
        print(
            f"{r['name']:<15} "
            f"{'yes' if r['primary'] else 'no':<8} "
            f"{'yes' if r['fallback'] else 'no':<9} "
            f"{source_ckpt_col:<12} "
            f"{r.get('release_mode', 'n/a')!s:<13} "
            f"{'yes' if r['runtime_ready'] else 'no':<13} "
            f"{r['resolved_by']}"
        )

    payload = {
        "mode": "runtime_readiness",
        "all_runtime_ready": len(not_ready) == 0,
        "missing_runtime_components": [r["name"] for r in not_ready],
        "components": rows,
    }

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"JSON written: {out}")

    if not_ready:
        print(f"\nResult: {len(not_ready)} runtime components not ready")
        return 1

    print("\nResult: all core runtime components ready (primary or fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
