#!/usr/bin/env python3
"""Stepwise ONNX export readiness checker for Aurik model candidates.

This script does not modify production plugin code. It provides a practical,
step-by-step readiness report and performs non-destructive dry-run export tests
for candidates that already exist as TorchScript files.

Usage:
    .venv_aurik/bin/python scripts/onnx_export_readiness.py
"""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    src: Path
    expected_onnx: Path


PRIO1 = [
    Candidate("beats", "missing_dir", ROOT / "models" / "beats", ROOT / "models" / "beats" / "beats_iter3.onnx"),
    Candidate("rmvpe", "missing_dir", ROOT / "models" / "rmvpe", ROOT / "models" / "rmvpe" / "rmvpe.onnx"),
    Candidate(
        "flow_matching",
        "missing_dir",
        ROOT / "models" / "flow_matching",
        ROOT / "models" / "flow_matching" / "flow_matching.onnx",
    ),
    Candidate("mp_senet", "missing_dir", ROOT / "models" / "mp_senet", ROOT / "models" / "mp_senet" / "mp_senet.onnx"),
    Candidate(
        "gacela", "missing_dir", ROOT / "models" / "gacela", ROOT / "models" / "gacela" / "gacela_generator.onnx"
    ),
]

PRIO2_4 = [
    Candidate(
        "apollo",
        "torchscript_audio",
        ROOT / "models" / "apollo" / "apollo_model.pt",
        ROOT / "models" / "apollo" / "apollo_model.onnx",
    ),
    Candidate(
        "cqtdiff",
        "torchscript_score",
        ROOT / "models" / "cqtdiff" / "score_network.pt",
        ROOT / "models" / "cqtdiff" / "score_network.onnx",
    ),
    Candidate(
        "utmosv2",
        "checkpoint_only",
        ROOT / "models" / "utmosv2" / "fold0_s42_best_model.pth",
        ROOT / "models" / "utmosv2" / "utmosv2_ssl_encoder.onnx",
    ),
    Candidate(
        "laion_clap",
        "checkpoint_only",
        ROOT / "models" / "clap" / "music_audioset_epoch_15_esc_90.14.pt",
        ROOT / "models" / "clap" / "audio_encoder.onnx",
    ),
]


def _test_torchscript_export_audio(src: Path, out_path: Path) -> tuple[bool, str]:
    import torch

    model = torch.jit.load(str(src), map_location="cpu")
    model.eval()
    x = torch.zeros(1, 1, 44_100, dtype=torch.float32)
    torch.onnx.export(
        model,
        (x,),
        str(out_path),
        input_names=["audio"],
        output_names=["out"],
        opset_version=17,
    )
    return True, "exportable"


def _test_torchscript_export_score(src: Path, out_path: Path) -> tuple[bool, str]:
    import torch

    model = torch.jit.load(str(src), map_location="cpu")
    model.eval()
    x_noisy = torch.zeros(1, 65_536, dtype=torch.float32)
    sigma = torch.ones(1, 1, dtype=torch.float32)
    torch.onnx.export(
        model,
        (x_noisy, sigma),
        str(out_path),
        input_names=["x_noisy", "sigma"],
        output_names=["denoised"],
        opset_version=17,
        dynamic_axes={
            "x_noisy": {0: "batch", 1: "samples"},
            "sigma": {0: "batch"},
            "denoised": {0: "batch", 1: "samples"},
        },
    )
    return True, "exportable"


def _check_candidate(c: Candidate) -> tuple[str, str]:
    if c.expected_onnx.exists():
        return "ready", f"already ONNX: {c.expected_onnx}"

    if c.kind == "missing_dir":
        if not c.src.exists():
            return "blocked", f"missing model directory: {c.src}"
        return "blocked", f"directory exists but ONNX missing: {c.expected_onnx}"

    if c.kind == "checkpoint_only":
        if not c.src.exists():
            return "blocked", f"checkpoint missing: {c.src}"
        return "manual", "checkpoint exists; requires architecture loader before ONNX export"

    if c.kind in {"torchscript_audio", "torchscript_score"}:
        if not c.src.exists():
            return "blocked", f"torchscript missing: {c.src}"
        tmp_out = c.expected_onnx.with_name(c.expected_onnx.stem + "_tmp.onnx")
        try:
            if c.kind == "torchscript_audio":
                _test_torchscript_export_audio(c.src, tmp_out)
            else:
                _test_torchscript_export_score(c.src, tmp_out)
            smoke_ok, smoke_msg = _ort_smoke_test(c, tmp_out)
            if tmp_out.exists():
                tmp_out.unlink()
            if smoke_ok:
                return "exportable", f"dry-run ONNX export + ORT smoke test succeeded ({smoke_msg})"
            return "not-exportable", f"ORT smoke test failed: {smoke_msg}"
        except Exception as exc:
            if tmp_out.exists():
                tmp_out.unlink()
            return "not-exportable", f"{type(exc).__name__}: {exc}"

    return "unknown", "unsupported candidate type"


def _check_candidate_quick(c: Candidate) -> tuple[str, str]:
    """Fast checks only. Avoid heavy ONNX export attempts by default.

    This mode is safe for routine runs and CI-style health checks.
    """
    if c.expected_onnx.exists():
        return "ready", f"already ONNX: {c.expected_onnx}"

    if c.kind == "missing_dir":
        if not c.src.exists():
            return "blocked", f"missing model directory: {c.src}"
        return "blocked", f"directory exists but ONNX missing: {c.expected_onnx}"

    if c.kind == "checkpoint_only":
        if not c.src.exists():
            return "blocked", f"checkpoint missing: {c.src}"
        return "manual", "checkpoint exists; requires architecture loader before ONNX export"

    if c.kind in {"torchscript_audio", "torchscript_score"}:
        if not c.src.exists():
            return "blocked", f"torchscript missing: {c.src}"
        return "manual", "torchscript exists; run with --deep for actual ONNX export probe"

    return "unknown", "unsupported candidate type"


def _ort_smoke_test(c: Candidate, onnx_path: Path) -> tuple[bool, str]:
    """Run a minimal ONNX Runtime inference to verify generated model usability."""
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_names = [i.name for i in sess.get_inputs()]

        if c.kind == "torchscript_audio":
            feed = {input_names[0]: np.zeros((1, 1, 44_100), dtype=np.float32)}
        elif c.kind == "torchscript_score":
            feed = {
                input_names[0]: np.zeros((1, 65_536), dtype=np.float32),
                input_names[1]: np.ones((1, 1), dtype=np.float32),
            }
        else:
            return False, "no smoke-test shape rule for candidate"

        out = sess.run(None, feed)
        if not out:
            return False, "no outputs returned"
        return True, f"output_count={len(out)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stepwise ONNX export readiness checker")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Run real ONNX dry-run exports for torchscript candidates (can be slow).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional model names to check (e.g. --only apollo cqtdiff).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidates = PRIO1 + PRIO2_4
    only = {name.strip().lower() for name in args.only if name.strip()}
    if only:
        candidates = [c for c in candidates if c.name.lower() in only]

    print("=" * 108)
    print(f"{'Model':<15} {'Priority':<8} {'Status':<16} Details")
    print("=" * 108)

    for c in candidates:
        priority = "P1" if c in PRIO1 else "P2-P4"
        t0 = time.perf_counter()
        if args.deep:
            status, details = _check_candidate(c)
        else:
            status, details = _check_candidate_quick(c)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        details = f"{details} (t={dt_ms:.0f}ms)"
        print(f"{c.name:<15} {priority:<8} {status:<16} {details}")

    print("\nLegend:")
    print("- ready: ONNX already present")
    print("- exportable: ONNX dry-run works")
    print("- manual: architecture-specific exporter required")
    print("- blocked: source assets missing")
    print("- not-exportable: exporter failed with concrete operator/runtime error")
    print("\nMode:")
    print("- default: quick checks only (recommended for stepwise workflow)")
    print("- --deep : includes real ONNX dry-run for torchscript candidates")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
