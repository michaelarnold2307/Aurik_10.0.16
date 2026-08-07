#!/usr/bin/env python3
"""Export UTMOSv2 audio encoder (wav2vec2-base backbone) to ONNX.

This exports the SSL backbone used by UTMOSv2 to a standalone ONNX model.
Output model is intended as a building block for UTMOS inference pipelines.

Input:
  input_values: float32 [B, T] @ 16 kHz
Output:
  last_hidden_state: float32 [B, T_frames, H]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "wav2vec2-base"
OUT_PATH = ROOT / "models" / "utmosv2" / "utmosv2_ssl_encoder.onnx"


class _W2VWrapper(torch.nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.model(input_values=input_values)
        return out.last_hidden_state  # type: ignore[no-any-return]


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"Fehler: Modellverzeichnis fehlt: {MODEL_DIR}")
        return 1

    torch.set_num_threads(max(1, os.cpu_count() or 4))

    model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.eval()
    wrapper = _W2VWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 16_000, dtype=torch.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(OUT_PATH),
        input_names=["input_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    m = onnx.load(str(OUT_PATH))
    onnx.checker.check_model(m)

    sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_values": np.zeros((1, 16_000), dtype=np.float32)})
    if not out or out[0].ndim != 3:
        print("Fehler: ORT-Smoke-Test fehlgeschlagen")
        return 2

    print(f"OK: {OUT_PATH}")
    print(f"Output shape: {out[0].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
