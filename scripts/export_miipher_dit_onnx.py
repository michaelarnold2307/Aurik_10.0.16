#!/usr/bin/env python3
"""
Export FlowMatchingDiT checkpoint → ONNX (§v10.14).

The MIIPHER-DiT plugin expects a FlowMatchingDiT ONNX model with:
  Input  x: [B, T, 1] float32 — degraded vocal waveform
  Input  t: [B]     float32 — flow-matching timestep (0.0–1.0)
  Output  : [B, T, 1] float32 — enhanced vocal waveform

Two modes:
  1. From checkpoint:  python scripts/export_miipher_dit_onnx.py --checkpoint PATH
  2. Fresh (untrained): python scripts/export_miipher_dit_onnx.py --fresh
     → exports a randomly-initialised model for structural verification.

Usage:
    python scripts/export_miipher_dit_onnx.py [--checkpoint PATH] [--output PATH] [--fresh]
"""

import argparse
import sys
from pathlib import Path

import onnx
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.miipher_dit.dit_model import FlowMatchingDiTExportWrapper, create_miipher_dit


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    dynamic_time: bool = True,
    dynamic_batch: bool = True,
    opset_version: int = 17,
):
    """Export model to ONNX with dynamic axes.

    The plugin passes inputs {"x": [1, T, 1], "t": [1]}.
    Both batch and time axes must be dynamic because T varies per chunk.
    """
    model.eval()
    model.to("cpu")

    # Dummy inputs matching the plugin's expected shapes
    # Use a moderate length (1s @ 48kHz) as the concrete shape
    dummy_T = 48000
    dummy_x = torch.randn(1, dummy_T, 1, dtype=torch.float32)
    dummy_t = torch.tensor([0.5], dtype=torch.float32)

    # Dynamic axes: batch dim 0 and time dim 1 for both input and output
    dynamic_axes = {}
    if dynamic_batch or dynamic_time:
        dyn_x = {}
        dyn_out = {}
        if dynamic_batch:
            dyn_x[0] = "batch"
            dyn_out[0] = "batch"
        if dynamic_time:
            dyn_x[1] = "time"
            dyn_out[1] = "time"
        dynamic_axes = {
            "x": dyn_x,
            "t": {0: "batch"} if dynamic_batch else {},
            "output": dyn_out,
        }

    print(f"Exporting ONNX model to: {output_path}")
    print(f"  Dummy input shape:  x={list(dummy_x.shape)}, t={list(dummy_t.shape)}")
    print(f"  Dynamic axes:       {dynamic_axes}")
    print(f"  Opset version:      {opset_version}")

    torch.onnx.export(
        model,
        (dummy_x, dummy_t),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["x", "t"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print("✅ ONNX model exported")

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX validation passed")

    # Report size and structure
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    size_gb = size_mb / 1024
    print(f"📦 Model size: {size_mb:.1f} MB ({size_gb:.2f} GB)")

    # Verify inputs/outputs
    print("📋 Graph inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else f"dynamic({d.dim_param})" for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape} ({inp.type.tensor_type.elem_type})")
    print("📋 Graph outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else f"dynamic({d.dim_param})" for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape} ({out.type.tensor_type.elem_type})")

    return output_path


def export_from_checkpoint(checkpoint_path: str, output_path: str, **kwargs):
    """Load a trained checkpoint and export to ONNX."""
    device = torch.device("cpu")
    model = create_miipher_dit()
    wrapper = FlowMatchingDiTExportWrapper(model)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    # Remove 'module.' prefix (DataParallel/DDP)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # The checkpoint may contain the raw model or the wrapper
    # Try loading into wrapper first, then into raw model
    try:
        wrapper.load_state_dict(state, strict=True)
        print("✅ Loaded into export wrapper")
    except RuntimeError:
        # Keys might be for the raw model (without "model." prefix)
        model_state = {k.replace("model.", ""): v for k, v in state.items()}
        try:
            model.load_state_dict(model_state, strict=True)
            print("✅ Loaded into raw model")
        except RuntimeError as e:
            # Report missing/unexpected keys and continue with partial load
            print(f"⚠️  Strict load failed: {e}")
            # Try non-strict
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                print(f"   Missing keys: {len(missing)}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)}")
            print("⚠️  Continuing with partially loaded weights")

    return export_to_onnx(wrapper, output_path, **kwargs)


def export_fresh(output_path: str, **kwargs):
    """Export a randomly-initialised (untrained) model to ONNX.

    This is useful for:
      - Verifying the model structure can be loaded by onnxruntime
      - Testing the plugin integration end-to-end
      - The model WILL NOT produce useful output — use a trained checkpoint
        for actual restoration work.
    """
    print("⚠️  Exporting UNTRAINED model (random weights)")
    print("   This model will NOT enhance audio — it outputs noise.")
    print("   Use --checkpoint to export a trained model.")

    model = create_miipher_dit()
    wrapper = FlowMatchingDiTExportWrapper(model)

    # Reset all weights with a standard initialisation
    # (already done in __init__, but be explicit)
    model._init_weights()

    return export_to_onnx(wrapper, output_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Export FlowMatchingDiT → ONNX (§v10.14)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/miipher_dit/flow_matching_dit.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Export a freshly-initialised (untrained) model",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Disable dynamic batch dimension",
    )
    parser.add_argument(
        "--static-time",
        action="store_true",
        help="Disable dynamic time dimension",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        dynamic_batch=not args.static_batch,
        dynamic_time=not args.static_time,
        opset_version=args.opset,
    )

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            print(f"❌ Checkpoint not found: {checkpoint}")
            print("   Train the model first: scripts/train_miipher_dit.py")
            sys.exit(1)
        export_from_checkpoint(str(checkpoint), str(output), **kwargs)
    elif args.fresh:
        export_fresh(str(output), **kwargs)
    else:
        print("No --checkpoint or --fresh specified.")
        print("Exporting untrained model by default (use --checkpoint for trained weights).")
        export_fresh(str(output), **kwargs)

    print(f"\n✅ Done: {output}")


if __name__ == "__main__":
    main()
