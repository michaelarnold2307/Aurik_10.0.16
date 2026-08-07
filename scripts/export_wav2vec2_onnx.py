#!/usr/bin/env python3
"""Export wav2vec2_forced_alignment ONNX (§2.36 Pflicht).

Converts models/wav2vec2/pytorch_model.bin → models/wav2vec2/wav2vec2_forced_alignment.onnx
then applies dynamic INT8 quantisation to reduce disk footprint.

Interface contract (must match backend/core/lyrics_guided_enhancement.py):
  Input : {"input_values": np.float32 (1, T)}   @ 16 kHz
  Output: [logits: np.float32 (1, T_frames, vocab_size)]

Usage:
    .venv_aurik/bin/python scripts/export_wav2vec2_onnx.py
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "wav2vec2"
ONNX_FP32 = MODEL_DIR / "wav2vec2_forced_alignment_fp32.onnx"
ONNX_INT8 = MODEL_DIR / "wav2vec2_forced_alignment.onnx"

# --------------------------------------------------------------------------- #
# 0. Sanity-checks                                                              #
# --------------------------------------------------------------------------- #
if not (MODEL_DIR / "pytorch_model.bin").exists():
    log.error("pytorch_model.bin not found in %s", MODEL_DIR)
    sys.exit(1)

if ONNX_INT8.exists():
    log.info("✅ %s already exists — delete it to re-export", ONNX_INT8.name)
    sys.exit(0)

# --------------------------------------------------------------------------- #
# 1. Load model (eager attention for ONNX compatibility)                       #
# --------------------------------------------------------------------------- #
log.info("Loading Wav2Vec2ForCTC from %s …", MODEL_DIR)
import torch
from transformers import Wav2Vec2ForCTC

torch.set_num_threads(max(1, os.cpu_count() or 4))

model = Wav2Vec2ForCTC.from_pretrained(
    str(MODEL_DIR),
    attn_implementation="eager",  # disable SDPA — required for ONNX export
    torch_dtype=torch.float32,
    local_files_only=True,
)
model.eval()
# Disable SpecAugment (active only during training)
model.config.apply_spec_augment = False

log.info(
    "Model loaded — vocab_size=%d, hidden_size=%d, num_layers=%d",
    model.config.vocab_size,
    model.config.hidden_size,
    model.config.num_hidden_layers,
)

# --------------------------------------------------------------------------- #
# 2. Dummy input: 1 s mono @ 16 kHz                                           #
# --------------------------------------------------------------------------- #
SEQ_LEN = 16_000
dummy = torch.zeros(1, SEQ_LEN, dtype=torch.float32)

# Warm-up pass to confirm output shape
with torch.no_grad():
    out = model(dummy)
logits = out.logits  # (1, T_frames, vocab_size)
log.info("Warm-up logits shape: %s  (vocab_size should be %d)", logits.shape, model.config.vocab_size)
assert logits.ndim == 3, f"Expected 3-D logits, got shape {logits.shape}"
assert logits.shape[2] == model.config.vocab_size, "vocab_size mismatch"

# --------------------------------------------------------------------------- #
# 3. torch.onnx.export → FP32 ONNX                                            #
# --------------------------------------------------------------------------- #
log.info("Exporting FP32 ONNX to %s …", ONNX_FP32)


# Wrap to return only logits tensor (onnx cannot handle return dataclass)
class _Wrapper(torch.nn.Module):
    def __init__(self, m: Wav2Vec2ForCTC) -> None:
        super().__init__()
        self._m = m

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self._m(input_values=input_values).logits  # type: ignore[no-any-return]


wrapper = _Wrapper(model)
wrapper.eval()

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(ONNX_FP32),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_frames"},
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )

fp32_mb = ONNX_FP32.stat().st_size / 1024 / 1024
log.info("FP32 ONNX written: %.0f MB", fp32_mb)

# --------------------------------------------------------------------------- #
# 4. Optional: verify FP32 model is loadable                                  #
# --------------------------------------------------------------------------- #
import onnx

model_check = onnx.load(str(ONNX_FP32))
onnx.checker.check_model(model_check)
log.info("ONNX model check passed.")

# --------------------------------------------------------------------------- #
# 5. Dynamic INT8 quantisation (reduces size ~4×, CPU runtime)                #
# --------------------------------------------------------------------------- #
log.info("Applying dynamic INT8 quantisation → %s …", ONNX_INT8)
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    model_input=str(ONNX_FP32),
    model_output=str(ONNX_INT8),
    weight_type=QuantType.QInt8,
    per_channel=False,  # per-tensor is more compatible with CPUExecutionProvider
    reduce_range=False,  # keep full INT8 range
    # Restrict to MatMul+Gemm only — Conv → ConvInteger is NOT supported by
    # CPUExecutionProvider and would crash at runtime.
    op_types_to_quantize=["MatMul", "Gemm"],
)

int8_mb = ONNX_INT8.stat().st_size / 1024 / 1024
log.info("INT8 ONNX written: %.0f MB  (compression: %.1f×)", int8_mb, fp32_mb / int8_mb if int8_mb else 0)

# --------------------------------------------------------------------------- #
# 6. End-to-end verification with ORT                                         #
# --------------------------------------------------------------------------- #
log.info("Verifying INT8 ONNX with ORT inference session …")
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    str(ONNX_INT8),
    providers=["CPUExecutionProvider"],
)
inp = np.zeros((1, SEQ_LEN), dtype=np.float32)
out_ort = sess.run(None, {"input_values": inp})
logits_ort = out_ort[0]

assert logits_ort.ndim == 3, f"ORT output ndim={logits_ort.ndim} ≠ 3"
assert logits_ort.shape[0] == 1, "batch dim mismatch"
assert logits_ort.shape[2] == model.config.vocab_size, (
    f"vocab_size: got {logits_ort.shape[2]}, expected {model.config.vocab_size}"
)
log.info("ORT verification passed — output shape: %s", logits_ort.shape)

# Remove intermediate fp32 file to save disk space
ONNX_FP32.unlink()
log.info("Removed intermediate FP32 file.")

log.info("=" * 60)
log.info("✅  wav2vec2_forced_alignment.onnx ready at:")
log.info("    %s", ONNX_INT8)
log.info("    Size: %.0f MB", int8_mb)
log.info("    Session input  : input_values  float32 (1, T)")
log.info("    Session output : logits         float32 (1, T_frames, %d)", model.config.vocab_size)
log.info("=" * 60)
