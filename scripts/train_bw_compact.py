#!/usr/bin/env python3
"""
Self-contained BW Reconstructor training — NO external data needed.
Generates synthetic audio on the fly, trains a compact U-Net on CPU.

Run:  python scripts/train_bw_compact.py
"""

import argparse
import math
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128
N_FRAMES = 128  # ~1.5 sec per segment
CUTOFFS = [3000, 5000, 8000, 11000]

# ═══════════════════════════════════════════════════════════════════════════
# Synthetic audio generator
# ═══════════════════════════════════════════════════════════════════════════

_NOTES = np.array([130.81, 164.81, 196.00, 220.00, 261.63, 329.63, 392.00, 523.25, 659.25, 783.99])


def make_synthetic(segment_samples: int, sr: int) -> np.ndarray:
    """Generate harmonic + noise audio."""
    t = np.arange(segment_samples, dtype=np.float64) / sr
    y = np.zeros(segment_samples, dtype=np.float64)

    n_notes = np.random.randint(1, 4)
    for _ in range(n_notes):
        f0 = np.random.choice(_NOTES) * np.random.uniform(0.5, 2.0)
        n_harm = np.random.randint(2, 7)
        for h in range(1, n_harm + 1):
            amp = 0.4 / (h ** np.random.uniform(0.6, 1.4))
            phase = np.random.uniform(0, 2 * np.pi)
            y += amp * np.sin(2 * np.pi * f0 * h * t + phase)

    # Add colored noise
    noise = np.random.randn(segment_samples).astype(np.float64)
    color = np.random.choice(["pink", "brown", "white"])
    if color == "pink":
        noise = np.cumsum(noise)
    elif color == "brown":
        noise = np.cumsum(np.cumsum(noise))
    noise /= np.abs(noise).max() + 1e-8
    y += 0.08 * noise

    y /= np.abs(y).max() + 1e-8
    return y.astype(np.float32)


def butter_lowpass(y: np.ndarray, cutoff_hz: float, sr: int, order: int = 6) -> np.ndarray:
    """Apply Butterworth lowpass filter."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2
    if cutoff_hz >= nyq * 0.98:
        return y.copy()
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════
# Mel spectrogram (torchaudio-based)
# ═══════════════════════════════════════════════════════════════════════════

_mel_t = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
    f_min=60,
    f_max=SR // 2,
    center=False,
)

_istft_t = torchaudio.transforms.InverseSpectrogram(
    n_fft=N_FFT,
    hop_length=HOP,
    normalized=True,
)


def audio_to_log_mel(waveform: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, n_mels, n_frames) log-mel, normalized."""
    mel = _mel_t(waveform)  # (B, n_mels, n_frames)
    mel = torch.log(mel + 1e-6)
    # per-sample norm
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std = mel.std(dim=(1, 2), keepdim=True) + 1e-8
    return cast(torch.Tensor, (mel - mean) / std)


def log_mel_to_audio(mel: torch.Tensor) -> torch.Tensor:
    """(B, n_mels, n_frames) log-mel -> (B, T) audio via Griffin-Lim."""
    mel = torch.exp(mel) - 1e-6
    mel = torch.clamp(mel, min=1e-8)
    # Convert mel to linear spec via pseudo-inverse (approximate)
    # We use a simpler approach: just do Griffin-Lim on the mel-to-linear approx
    N_FFT // 2 + 1
    mel_basis = _make_mel_basis(SR, N_FFT, N_MELS)
    mel_basis_pinv = torch.linalg.pinv(mel_basis)  # (n_fft_out, n_mels)
    # (B, n_mels, n_frames) -> (B, n_fft_out, n_frames)
    mag = torch.einsum("om,bmf->bof", mel_basis_pinv, mel)
    mag = torch.clamp(mag, min=0.0)
    return cast(torch.Tensor, _istft_t(mag.sqrt()))  # Griffin-Lim built into InverseSpectrogram


_mel_basis_cache = None


def _make_mel_basis(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
    global _mel_basis_cache
    if _mel_basis_cache is not None and _mel_basis_cache.shape == (n_mels, n_fft // 2 + 1):
        return _mel_basis_cache
    n_out = n_fft // 2 + 1
    f_min, f_max = 60.0, sr / 2.0
    mel_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = torch.floor((n_fft + 1) * hz_pts / sr).long().clamp(0, n_out - 1)
    basis = torch.zeros(n_mels, n_out)
    for m in range(n_mels):
        l, c, r = int(bins[m].item()), int(bins[m + 1].item()), int(bins[m + 2].item())
        if c > l:
            basis[m, l:c] = torch.linspace(0, 1, c - l)
        if r > c:
            basis[m, c:r] = torch.linspace(1, 0, r - c)
    _mel_basis_cache = basis
    return basis


# ═══════════════════════════════════════════════════════════════════════════
# U-Net (compact)
# ═══════════════════════════════════════════════════════════════════════════


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


class CompactUNet(nn.Module):
    """3-level U-Net: enc doubles channels, dec halves them, skip-cats match."""

    def __init__(self, base_ch=16):
        super().__init__()
        C = base_ch  # 16
        # Encoder: 1 -> C -> 2C -> 4C
        self.enc1 = ConvBlock(1, C)  # out: C
        self.enc2 = ConvBlock(C, C * 2)  # out: 2C
        self.enc3 = ConvBlock(C * 2, C * 4)  # out: 4C
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBlock(C * 4, C * 8)  # 4C -> 8C
        # Decoder: each up halves channels, cat doubles them back
        self.up3 = nn.ConvTranspose2d(C * 8, C * 4, 2, 2)  # 8C -> 4C
        self.dec3 = ConvBlock(C * 8, C * 4)  # cat(up3:4C, e3:4C)=8C -> 4C
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, 2)  # 4C -> 2C
        self.dec2 = ConvBlock(C * 4, C * 2)  # cat(up2:2C, e2:2C)=4C -> 2C
        self.up1 = nn.ConvTranspose2d(C * 2, C, 2, 2)  # 2C -> C
        self.dec1 = ConvBlock(C * 2, C)  # cat(up1:C, e1:C)=2C -> C
        self.final = nn.Conv2d(C, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)  # (B, C,   64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, 2C,  32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, 4C,  16, 16)
        b = self.bottleneck(self.pool(e3))  # (B, 8C,  8,   8)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # cat(4C, 4C)=8C -> 4C @ 16x16
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # cat(2C, 2C)=4C -> 2C @ 32x32
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # cat(C, C)=2C -> C @ 64x64
        return self.final(d1)  # (B, 1, 64, 64)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════


def train(
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 1e-3,
    base_ch: int = 16,
    steps_per_epoch: int = 200,
    output_dir: str = "models/bw_reconstructor",
):
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(
        f"Model: CompactUNet(base_ch={base_ch}) = {sum(p.numel() for p in CompactUNet(base_ch).parameters()):,} params"
    )

    model = CompactUNet(base_ch=base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    segment_samples = (N_FRAMES - 1) * HOP + N_FFT
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            # Generate batch of synthetic audio
            batch_input, batch_target = [], []
            for _ in range(batch_size):
                y_full = make_synthetic(segment_samples + SR, SR)
                cutoff = np.random.choice(CUTOFFS)
                y_limited = butter_lowpass(y_full, cutoff, SR)

                # Random crop to exact length
                start = np.random.randint(0, len(y_full) - segment_samples)
                y_full = y_full[start : start + segment_samples]
                y_limited = y_limited[start : start + segment_samples]

                batch_input.append(torch.from_numpy(y_limited))
                batch_target.append(torch.from_numpy(y_full))

            x = torch.stack(batch_input).to(device)  # (B, T)
            y = torch.stack(batch_target).to(device)

            mel_x = audio_to_log_mel(x).unsqueeze(1)  # (B, 1, M, F)
            mel_y = audio_to_log_mel(y).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(mel_x)

            loss_l1 = F.l1_loss(pred, mel_y)
            # Simple spectral loss
            loss_spec = 0.1 * F.mse_loss(pred[:, :, :, ::2], mel_y[:, :, :, ::2])
            loss = loss_l1 + loss_spec

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
                {"model_state_dict": model.state_dict(), "epoch": epoch, "loss": avg_loss},
                output_path / "best_model.pt",
            )

    # ── Export ONNX ───────────────────────────────────────────────────────
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, N_FRAMES).to(device)
    onnx_path = output_path / "bw_reconstructor.onnx"

    print(f"\nExporting ONNX to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "freq", 3: "time"}, "output": {0: "batch", 2: "freq", 3: "time"}},
        opset_version=17,
    )

    import onnx

    onnx.checker.check_model(str(onnx_path))
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"✅ ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
    print("✅ Training complete. Plug Aurik in: BWReconstructorPlugin(model_path=...)")

    return onnx_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output", type=str, default="models/bw_reconstructor")
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.base_ch, args.steps, args.output)
