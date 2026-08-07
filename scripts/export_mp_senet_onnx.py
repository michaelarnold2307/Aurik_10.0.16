"""
Export MP-SENet generator checkpoint to ONNX.

Loads best_ckpt/g_best_vb (VoiceBank+DEMAND) and exports to
models/mp_senet/mp_senet.onnx with dynamic T-axis.

Usage:
    python scripts/export_mp_senet_onnx.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
MP_SENET_DIR = REPO_ROOT / "models" / "mp_senet"
CHECKPOINT = MP_SENET_DIR / "best_ckpt" / "g_best_vb"
CONFIG_FILE = MP_SENET_DIR / "best_ckpt" / "config.json"
OUTPUT_ONNX = MP_SENET_DIR / "mp_senet.onnx"

# Add mp_senet source to path
sys.path.insert(0, str(MP_SENET_DIR))


# ---------------------------------------------------------------------------
# Minimal AttrDict + model import
# ---------------------------------------------------------------------------
class AttrDict(dict):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, item: str) -> Any:
        return self[item]


def load_config(path: Path) -> AttrDict:
    with open(path) as f:
        return AttrDict(json.loads(f.read()))


# ---------------------------------------------------------------------------
# Monkey-patch: inject LearnableSigmoid2d without loading matplotlib/pesq
# utils.py imports matplotlib (broken under NumPy 2.x) only for plot helpers —
# the class itself is pure torch, so we inject it directly into sys.modules.
# ---------------------------------------------------------------------------
import types as _types


def _make_utils_stub() -> _types.ModuleType:
    import torch as _torch
    import torch.nn as _nn

    mod = _types.ModuleType("utils")

    class LearnableSigmoid2d(_nn.Module):
        def __init__(self, in_features: int, beta: float = 1.0) -> None:
            super().__init__()
            self.beta = beta
            self.slope = _nn.Parameter(_torch.ones(in_features, 1))
            self.slope.requiresGrad = True  # type: ignore[attr-defined]

        def forward(self, x: _torch.Tensor) -> _torch.Tensor:
            return self.beta * _torch.sigmoid(self.slope * x)

    mod.LearnableSigmoid2d = LearnableSigmoid2d  # type: ignore[attr-defined]
    return mod


sys.modules.setdefault("utils", _make_utils_stub())

# pesq is only used in training helpers (pesq_score / eval_pesq), not in forward()
try:
    pass
except ImportError:
    sys.modules.setdefault("pesq", _types.ModuleType("pesq"))


def build_model(h: AttrDict) -> nn.Module:
    from models.model import MPNet

    return MPNet(h)  # type: ignore[no-any-return]


def export():
    print(f"Config:     {CONFIG_FILE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output:     {OUTPUT_ONNX}")

    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    h = load_config(CONFIG_FILE)

    model = build_model(h)
    state_dict = torch.load(str(CHECKPOINT), map_location="cpu")
    model.load_state_dict(state_dict["generator"])
    model.eval()
    print("✅ Checkpoint loaded")

    # Dummy inputs: [B=1, F=n_fft//2+1, T=32]
    F = h.n_fft // 2 + 1  # 201
    T = 32
    noisy_amp = torch.randn(1, F, T)
    noisy_pha = torch.randn(1, F, T)

    with torch.no_grad():
        amp_out, pha_out, com_out = model(noisy_amp, noisy_pha)
    print(f"Forward OK — amp_out shape: {amp_out.shape}")

    print("Exporting to ONNX …")
    torch.onnx.export(
        model,
        (noisy_amp, noisy_pha),
        str(OUTPUT_ONNX),
        input_names=["noisy_amp", "noisy_pha"],
        output_names=["denoised_amp", "denoised_pha", "denoised_com"],
        dynamic_axes={
            "noisy_amp": {0: "batch", 2: "time"},
            "noisy_pha": {0: "batch", 2: "time"},
            "denoised_amp": {0: "batch", 2: "time"},
            "denoised_pha": {0: "batch", 2: "time"},
            "denoised_com": {0: "batch", 2: "time"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = OUTPUT_ONNX.stat().st_size / 1024 / 1024
    print(f"✅ Exported: {OUTPUT_ONNX}  ({size_mb:.1f} MB)")

    # Quick validation
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
        out = sess.run(
            None,
            {
                "noisy_amp": noisy_amp.numpy(),
                "noisy_pha": noisy_pha.numpy(),
            },
        )
        print(f"✅ ONNX validation OK — output shapes: {[o.shape for o in out]}")
    except ImportError:
        print("onnxruntime nicht installiert — Validierung übersprungen")
    except Exception as e:
        print(f"⚠️  ONNX validation error: {e}")


if __name__ == "__main__":
    export()
