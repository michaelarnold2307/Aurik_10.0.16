#!/usr/bin/env python3
"""Export LAION-CLAP audio encoder to ONNX (local checkpoint).

Exports audio embedding path using local CLAP checkpoint:
  models/clap/music_audioset_epoch_15_esc_90.14.pt

Input:
  waveform: float32 [B, T] @ 48 kHz
Output:
  audio_embedding: float32 [B, D] (L2-normalized)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CLAP_SRC = ROOT / "models" / "clap" / "src"
CKPT = ROOT / "models" / "clap" / "music_audioset_epoch_15_esc_90.14.pt"
OUT_PATH = ROOT / "models" / "clap" / "audio_encoder.onnx"


class _ClapAudioWrapper(torch.nn.Module):
    def __init__(self, clap_model: torch.nn.Module) -> None:
        super().__init__()
        self.clap = clap_model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        audio_dict = {"waveform": waveform}
        emb = self.clap(audio=audio_dict, text=None, device=waveform.device)
        emb = F.normalize(emb, dim=-1)
        return cast(torch.Tensor, emb)


def main() -> int:
    if not CKPT.exists():
        print(f"Fehler: Checkpoint fehlt: {CKPT}")
        return 1

    if str(CLAP_SRC) not in sys.path:
        sys.path.insert(0, str(CLAP_SRC))

    from laion_clap.clap_module.factory import create_model, load_state_dict

    torch.set_num_threads(max(1, os.cpu_count() or 4))

    model, _ = create_model(
        amodel_name="HTSAT-base",
        tmodel_name="roberta",
        pretrained="",
        precision="fp32",
        device=torch.device("cpu"),
        enable_fusion=False,
    )
    state = load_state_dict(str(CKPT), map_location="cpu", skip_params=True)
    # Audio-only export: text branch is not required and may differ by transformers version.
    state_audio = {k: v for k, v in state.items() if k.startswith("audio_branch.") or k.startswith("audio_projection")}
    model.load_state_dict(state_audio, strict=False)
    model.eval()

    wrapper = _ClapAudioWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 480_000, dtype=torch.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(OUT_PATH),
        input_names=["waveform"],
        output_names=["audio_embedding"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "audio_embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    m = onnx.load(str(OUT_PATH))
    onnx.checker.check_model(m)

    sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"waveform": np.zeros((1, 480_000), dtype=np.float32)})
    if not out or out[0].ndim != 2:
        print("Fehler: ORT-Smoke-Test fehlgeschlagen")
        return 2

    print(f"OK: {OUT_PATH}")
    print(f"Output shape: {out[0].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
