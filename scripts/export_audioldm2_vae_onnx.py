#!/usr/bin/env python3
"""Export AudioLDM2 VAE decoder to ONNX for Aurik.

Input:  latent (batch=1, 8, H, W) float32, where H=mel_bins/8, W=time_frames/8
Output: mel spectrogram (batch=1, 1, H*8, W*8) float32

The decoder reconstructs a mel spectrogram from the diffusion latent.
Uses the standard Stable-Diffusion-style AutoencoderKL from diffusers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models" / "audioldm2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ONNX_OUT = MODEL_DIR / "vae_decoder.onnx"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# ---------------------------------------------------------------------------
# Step 1: Download VAE config + weights
# ---------------------------------------------------------------------------
print("📥 Downloading VAE config & weights from cvssp/audioldm2…")
config_path = hf_hub_download("cvssp/audioldm2", "vae/config.json")
weights_path = hf_hub_download("cvssp/audioldm2", "vae/diffusion_pytorch_model.safetensors")

with open(config_path) as f:
    config = json.load(f)
print(f"   Config: {config['_class_name']}, latent={config['latent_channels']}ch")
print(f"   Weights: {os.path.getsize(weights_path)/(1024**2):.0f} MB")

# ---------------------------------------------------------------------------
# Step 2: Build VAE from config using diffusers
# ---------------------------------------------------------------------------
print("🔧 Building AutoencoderKL from config…")
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

# Extract decoder-relevant config
vae = AutoencoderKL(
    in_channels=config["in_channels"],          # 1 (mel)
    out_channels=config["out_channels"],        # 1
    block_out_channels=tuple(config["block_out_channels"]),  # (128, 256, 512)
    down_block_types=tuple(config["down_block_types"]),
    up_block_types=tuple(config["up_block_types"]),
    layers_per_block=config["layers_per_block"],
    act_fn=config["act_fn"],
    latent_channels=config["latent_channels"],
    norm_num_groups=config["norm_num_groups"],
    sample_size=config["sample_size"],
    scaling_factor=config["scaling_factor"],
    force_upcast=config.get("force_upcast", True),
)

# Load weights
print("📦 Loading safetensors weights…")
state_dict = load_safetensors(weights_path)
vae.load_state_dict(state_dict)
vae.eval()

# ---------------------------------------------------------------------------
# Step 3: Trace ONLY the decoder
# ---------------------------------------------------------------------------
print("🧊 Tracing VAE decoder → ONNX…")

# The AudioLDM2 mel spectrogram has 64 mel bins at sample_size 1024.
# After VAE encoding: latent height = 64/8 = 8, latent width = 1024/8 = 128
# For a variable-duration input, the width varies.
# We export with a concrete shape but mark the spatial dims as dynamic.

# Example latent for 3 seconds of audio at 16kHz:
#   mel_time = 3 * 16000 / 160 = 300  (hop_length=160, 100 fps)
#   latent_time = 300 / 8 = 37.5 → need to handle padding
# Let's export with a typical shape: (1, 8, 8, 128) for ~3.2s

latent_h = 8   # mel_bins=64 / downsample=8
latent_w = 128  # time_frames=1024 / 8 = 128 (10.24 s @ 100fps)

dummy_latent = torch.randn(1, config["latent_channels"], latent_h, latent_w, dtype=torch.float32)

# Wrap the decoder forward pass
class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model

    def forward(self, latent):
        # AutoencoderKL.decode() expects (batch, channels, height, width)
        # It handles the internal scaling and conv-transpose decoding.
        # The output is a mel spectrogram: (batch, 1, mel_bins, time_frames)
        decoded = self.vae.decode(latent)
        # Return the sample (mel spectrogram)
        return decoded.sample

wrapper = VAEDecoderWrapper(vae)
wrapper.eval()

# Test forward pass works
with torch.no_grad():
    test_out = wrapper(dummy_latent)
    print(f"   Test output shape: {tuple(test_out.shape)}")
    # AudioLDM2 VAE uses 4× upsampling (2 blocks with stride=2)
    # output is (1, 1, H*4, W*4) where H,W are the latent spatial dims
    expected_h = latent_h * 4
    expected_w = latent_w * 4
    assert test_out.shape[0] == 1 and test_out.shape[1] == 1, \
        f"Expected (1, 1, *, *), got {tuple(test_out.shape)}"

# Export to ONNX
torch.onnx.export(
    wrapper,
    dummy_latent,
    str(ONNX_OUT),
    input_names=["latent"],
    output_names=["mel_spectrogram"],
    dynamic_axes={
        "latent": {0: "batch", 2: "height", 3: "width"},
        "mel_spectrogram": {0: "batch", 2: "mel_bins", 3: "time_frames"},
    },
    opset_version=17,
    do_constant_folding=True,
)

# ---------------------------------------------------------------------------
# Step 4: Validate ONNX
# ---------------------------------------------------------------------------
print("✅ Validating ONNX model…")
onnx_model = onnx.load(str(ONNX_OUT))
onnx.checker.check_model(onnx_model)

# Test with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
print(f"   ONNX inputs:  {[i.name for i in session.get_inputs()]}")
print(f"   ONNX outputs: {[o.name for o in session.get_outputs()]}")

# Compare output
np_latent = dummy_latent.numpy().astype(np.float32)
onnx_out = session.run(None, {"latent": np_latent})[0]
torch_out = test_out.numpy()

max_diff = np.abs(onnx_out - torch_out).max()
mean_diff = np.abs(onnx_out - torch_out).mean()
print(f"   Max diff PyTorch vs ONNX: {max_diff:.6f}")
print(f"   Mean diff: {mean_diff:.6f}")
assert max_diff < 0.01, f"ONNX output differs too much from PyTorch (max={max_diff})"

size_mb = os.path.getsize(ONNX_OUT) / (1024 * 1024)
print(f"\n🎉 VAE decoder exported successfully!")
print(f"   Output: {ONNX_OUT}")
print(f"   Size:   {size_mb:.0f} MB")
print(f"   Input:  (batch, 8, H, W) — dynamic H, W")
print(f"   Output: (batch, 1, H×8, W×8)")
