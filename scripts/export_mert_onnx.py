"""
Export MERT-95M (and optionally MERT-330M) to ONNX INT8.

Plugin interface (mert_plugin._analyze_onnx):
  Input:  "input_values"  float32 [1, T]   (T audio samples @ 16 kHz)
  Output: [0]             float32 [1, F, D] (F frames, D hidden dims)
  Score:  np.mean(np.abs(output)) / 10.0   (clipped to [0, 1])

Usage:
    python scripts/export_mert_onnx.py [--model 330m]  # default: 95m
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", choices=["95m", "330m"], default="95m", help="Which MERT variant to export (default: 95m)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(model_id: str) -> Path:
    """Export MERT variant to FP32 ONNX, then quantize to INT8."""
    if model_id == "95m":
        src_dir = ROOT / "models" / "mert-95m"
        hidden_size = 768
    else:
        src_dir = ROOT / "models" / "mert-v1-330m"
        hidden_size = 1024  # MERT-330M dim

    out_dir = ROOT / "models" / "mert"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = out_dir / f"mert_{model_id}_fp32.onnx"
    int8_path = out_dir / "mert.onnx"  # plugin always reads this name

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"[1/4] Loading MERT-{model_id.upper()} from {src_dir} ...")
    sys.path.insert(0, str(src_dir))

    from transformers import AutoModel  # type: ignore

    model = AutoModel.from_pretrained(str(src_dir), trust_remote_code=True)
    model.eval()
    print(f"      Model type: {type(model).__name__}")

    # ------------------------------------------------------------------
    # 2. ONNX export (FP32)
    # ------------------------------------------------------------------
    SR = 16_000
    dummy_seconds = 1.0
    dummy = torch.zeros(1, int(SR * dummy_seconds))

    print(f"[2/4] Exporting FP32 ONNX → {fp32_path} ...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(fp32_path),
            opset_version=14,
            input_names=["input_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "samples"},
                "last_hidden_state": {0: "batch", 1: "frames"},
            },
            do_constant_folding=True,
        )

    size_fp32 = fp32_path.stat().st_size
    print(f"      FP32 ONNX size: {size_fp32 / 1e6:.1f} MB")

    # ------------------------------------------------------------------
    # 3. INT8 quantization (MatMul + Gemm only — Conv excluded;
    #    CPUExecutionProvider does not support ConvInteger)
    # ------------------------------------------------------------------
    print(f"[3/4] Quantizing to INT8 → {int8_path} ...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    size_int8 = int8_path.stat().st_size
    print(f"      INT8 ONNX size: {size_int8 / 1e6:.1f} MB  ({size_fp32 / size_int8:.1f}× smaller)")

    # Clean up FP32 intermediate
    fp32_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 4. Verify with OnnxRuntime
    # ------------------------------------------------------------------
    print("[4/4] Verifying with OnnxRuntime ...")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 3-second test clip
    test_audio = np.zeros((1, SR * 3), dtype=np.float32)
    outputs = sess.run(None, {input_name: test_audio})
    out_arr = outputs[0]
    print(f"      Output shape: {out_arr.shape}   (expect [1, *, {hidden_size}])")
    assert out_arr.ndim == 3, f"Expected 3D output, got shape {out_arr.shape}"
    assert out_arr.shape[0] == 1
    assert out_arr.shape[2] == hidden_size, f"Hidden size mismatch: {out_arr.shape[2]} vs {hidden_size}"
    assert np.isfinite(out_arr).all(), "NaN/Inf in output!"

    # Plugin-style score
    score = float(np.clip(np.mean(np.abs(out_arr)) / 10.0, 0.0, 1.0))
    print(f"      Plugin score (zero-input): {score:.4f}")

    sha = sha256_of_file(int8_path)
    print(f"      SHA-256: {sha}")
    print(f"\n✅  MERT-{model_id.upper()} ONNX saved → {int8_path}")
    return int8_path, sha  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    export(args.model)
