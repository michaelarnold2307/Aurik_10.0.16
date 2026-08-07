#!/usr/bin/env python3
"""
BW Reconstructor V2 — Besseres Training mit Multi-Resolution-STFT-Loss
und Cutoff-Frequenz-Conditioning (FiLM).

Key improvements over V1:
  - Multi-Resolution STFT Loss (spezifisch für Frequenz-Rekonstruktion)
  - Cutoff-Frequenz als FiLM-Conditioning (Modell weiß, WO es rekonstruieren soll)
  - Größeres Modell (base_ch=24 statt 16)
  - Cosine-Warmup für besseres Fine-Tuning
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.signal import butter, sosfiltfilt

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128
N_FRAMES = 128
CUTOFFS = [3000, 5000, 8000, 11000]
_NOTES = np.array([130.81, 164.81, 196.00, 220.00, 261.63, 329.63, 392.00, 523.25, 659.25, 783.99])

# ═══════════════════════════════════════════════════════════════════════════
# Audio generation
# ═══════════════════════════════════════════════════════════════════════════


def make_synthetic(n_samples, sr):
    t = np.arange(n_samples, dtype=np.float64) / sr
    y = np.zeros(n_samples, dtype=np.float64)
    for _ in range(np.random.randint(1, 4)):
        f0 = np.random.choice(_NOTES) * np.random.uniform(0.5, 2.0)
        for h in range(1, np.random.randint(2, 7)):
            amp = 0.4 / (h ** np.random.uniform(0.6, 1.4))
            y += amp * np.sin(2 * np.pi * f0 * h * t + np.random.uniform(0, 2 * np.pi))
    noise = np.random.randn(n_samples).astype(np.float64)
    color = np.random.choice(["pink", "brown", "white"])
    if color == "pink":
        noise = np.cumsum(noise)
    elif color == "brown":
        noise = np.cumsum(np.cumsum(noise))
    noise /= np.abs(noise).max() + 1e-8
    y += 0.08 * noise
    y /= np.abs(y).max() + 1e-8
    return y.astype(np.float32)


def butter_lp(y, cutoff, sr, order=6):
    nyq = sr / 2
    if cutoff >= nyq * 0.98:
        return y.copy()
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Mel-Spectrogram + STFT utilities
# ═══════════════════════════════════════════════════════════════════════════

mel_t = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
    f_min=60,
    f_max=SR // 2,
    center=False,
)


def audio_to_log_mel(waveform):
    mel = mel_t(waveform)
    mel = torch.log(mel + 1e-6)
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std = mel.std(dim=(1, 2), keepdim=True) + 1e-8
    return (mel - mean) / std


# ═══════════════════════════════════════════════════════════════════════════
# FiLM Conditioning: inject cutoff frequency into the U-Net bottleneck
# ═══════════════════════════════════════════════════════════════════════════


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: generates scale & bias from cutoff Hz."""

    def __init__(self, condition_dim, feature_channels):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_channels)
        self.bias = nn.Linear(condition_dim, feature_channels)

    def forward(self, features, condition):
        # features: (B, C, H, W), condition: (B, cond_dim)
        s = self.scale(condition).unsqueeze(-1).unsqueeze(-1) + 1.0  # center around 1
        b = self.bias(condition).unsqueeze(-1).unsqueeze(-1)
        return features * s + b


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class CompactUNetV2(nn.Module):
    """U-Net with FiLM conditioning on cutoff frequency."""

    def __init__(self, base_ch=24):
        super().__init__()
        C = base_ch
        self.enc1 = ConvBlock(1, C)
        self.enc2 = ConvBlock(C, C * 2)
        self.enc3 = ConvBlock(C * 2, C * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(C * 4, C * 8)
        # FiLM at bottleneck: condition on cutoff frequency
        self.film_bottleneck = FiLMBlock(1, C * 8)
        self.up3 = nn.ConvTranspose2d(C * 8, C * 4, 2, 2)
        self.dec3 = ConvBlock(C * 8, C * 4)
        self.film_dec3 = FiLMBlock(1, C * 4)
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, 2)
        self.dec2 = ConvBlock(C * 4, C * 2)
        self.up1 = nn.ConvTranspose2d(C * 2, C, 2, 2)
        self.dec1 = ConvBlock(C * 2, C)
        self.final = nn.Conv2d(C, 1, 1)

    def forward(self, x, cutoff_normalized):
        # cutoff_normalized: (B, 1) in range [0, 1]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self.film_bottleneck(b, cutoff_normalized)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d3 = self.film_dec3(d3, cutoff_normalized)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Resolution STFT Loss
# ═══════════════════════════════════════════════════════════════════════════


def multi_resolution_stft_loss(pred, target):
    """Multi-scale spectral loss on Mel spectrograms."""
    loss = 0.0
    # Treat Mel spectrogram as 1D signals along frequency axis for each time frame
    # Use different FFT sizes for multi-resolution
    for n_fft in [32, 64, 128]:
        # STFT along mel bins (dim=2) for each batch
        B = pred.shape[0]
        total = 0
        for b in range(B):
            pred_spec = torch.stft(
                pred[b, 0, :, :],
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft, device=pred.device),
                return_complex=True,
                pad_mode="reflect",
            ).abs()
            target_spec = torch.stft(
                target[b, 0, :, :],
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft, device=target.device),
                return_complex=True,
                pad_mode="reflect",
            ).abs()
            total += F.l1_loss(pred_spec, target_spec)  # type: ignore[assignment]
        loss += total / B
    return loss / 3.0


def combined_loss(pred, target, cutoff_norm):
    """L1 + weighted MR-STFT loss, with cutoff-aware weighting."""
    loss_l1 = F.l1_loss(pred, target)
    loss_spec = multi_resolution_stft_loss(pred, target)

    # Cutoff-aware weight: more spectral loss for higher cutoffs (more to reconstruct)
    alpha = 0.3 + 0.5 * cutoff_norm.mean()  # 0.3 .. 0.8

    return loss_l1 + alpha * loss_spec


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════


def train(epochs=300, batch_size=6, lr=1e-4, base_ch=24, steps_per_epoch=200):
    device = torch.device("cpu")
    segment_samples = (N_FRAMES - 1) * HOP + N_FFT
    extra_samples = SR

    model = CompactUNetV2(base_ch=base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model: CompactUNetV2(base_ch={base_ch}) = {n_params:,} parameters")

    # Try loading previous checkpoint for fine-tuning
    checkpoint_path = Path("models/bw_reconstructor/best_model.pt")
    best_loss = float("inf")

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Only load compatible weights (ignore film blocks if missing from V1)
        model_state = model.state_dict()
        pretrained = {
            k: v for k, v in ckpt["model_state_dict"].items() if k in model_state and v.shape == model_state[k].shape
        }
        model_state.update(pretrained)
        model.load_state_dict(model_state, strict=False)
        loaded = len(pretrained)
        total = len(model_state)
        print(f"Loaded {loaded}/{total} weights from V1 checkpoint (transfer learning)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    out_dir = Path("models/bw_reconstructor")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Epochs: {epochs}, Batch: {batch_size}, Steps/epoch: {steps_per_epoch}")
    print(f"LR: {lr}, Multi-Resolution STFT Loss + FiLM conditioning")
    print()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            batch_x, batch_y, batch_c = [], [], []
            for _ in range(batch_size):
                y_full = make_synthetic(segment_samples + extra_samples, SR)
                cutoff = np.random.choice(CUTOFFS)
                y_lim = butter_lp(y_full, cutoff, SR)
                start = np.random.randint(0, len(y_full) - segment_samples)

                batch_x.append(torch.from_numpy(y_lim[start : start + segment_samples]))
                batch_y.append(torch.from_numpy(y_full[start : start + segment_samples]))
                # Normalize cutoff to [0, 1] (max possible is ~11kHz)
                batch_c.append(cutoff / 11025.0)

            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)
            c = torch.tensor(batch_c, dtype=torch.float32, device=device).unsqueeze(1)

            mel_x = audio_to_log_mel(x).unsqueeze(1)
            mel_y = audio_to_log_mel(y).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(mel_x, c)
            loss = combined_loss(pred, mel_y, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | Loss {avg_loss:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.1e} | {elapsed:.0f}s",
            flush=True,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "loss": avg_loss}, out_dir / "best_model.pt"
            )

    # Export ONNX
    model.eval()
    dummy_input = torch.randn(1, 1, N_MELS, N_FRAMES).to(device)
    dummy_cutoff = torch.tensor([[0.5]], dtype=torch.float32, device=device)

    class ExportWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, cutoff):
            return self.model(x, cutoff)

    export_model = ExportWrapper(model)
    onnx_path = out_dir / "bw_reconstructor_v2.onnx"

    torch.onnx.export(
        export_model,
        (dummy_input, dummy_cutoff),
        str(onnx_path),
        input_names=["input", "cutoff"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "freq", 3: "time"},
            "cutoff": {0: "batch"},
            "output": {0: "batch", 2: "freq", 3: "time"},
        },
        opset_version=17,
    )

    import onnx

    onnx.checker.check_model(str(onnx_path))
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\nONNX exported: {onnx_path} ({size_mb:.1f} MB)")
    print(f"Best loss: {best_loss:.4f}")
    print("Done!")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-ch", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.base_ch, args.steps)
