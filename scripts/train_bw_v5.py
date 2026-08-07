#!/usr/bin/env python3
"""
BW Reconstructor V5 — Waveform-Domain U-Net + Multi-Resolution STFT Loss.

Statt Mel→Mel: Direkte Waveform-Rekonstruktion mit 1D-Convolutions.
Kein Griffin-Lim nötig — das Modell lernt Phase implizit über STFT-Loss.
Optional: BigVGAN als Post-Processor für "Air"-Frequenzen (> 16 kHz).

Architektur inspiriert von Demucs/Wave-U-Net:
  - 1D Encoder-Decoder mit GLU-Aktivierung
  - Skip-Connections auf jeder Ebene
  - LSTM-Bottleneck für Langzeit-Kontext
  - FiLM-Conditioning auf Cutoff-Frequenz

Loss: Multi-Resolution STFT Loss (3 FFT-Größen) + L1-Waveform
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

SR = 22050
CHUNK_SEC = 3.0  # seconds per training chunk
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
CUTOFFS = [3000, 5000, 8000, 11000]

# ═══════════════════════════════════════════════════════════════════════════
# Synthetic audio generator
# ═══════════════════════════════════════════════════════════════════════════

_NOTES = np.array([130.81, 164.81, 196.00, 220.00, 261.63, 329.63, 392.00, 523.25, 659.25, 783.99])


def make_synthetic(n_samples, sr):
    """Generate rich audio: harmonics + transients + noise."""
    t = np.arange(n_samples, dtype=np.float64) / sr
    y = np.zeros(n_samples, dtype=np.float64)

    # Harmonic content
    for _ in range(np.random.randint(1, 4)):
        f0 = np.random.choice(_NOTES) * np.random.uniform(0.5, 2.0)
        for h in range(1, np.random.randint(2, 8)):
            amp = 0.4 / (h ** np.random.uniform(0.5, 1.5))
            y += amp * np.sin(2 * np.pi * f0 * h * t + np.random.uniform(0, 2 * np.pi))

    # Transients (drum-like)
    for _ in range(np.random.randint(0, 3)):
        pos = np.random.randint(n_samples // 4, 3 * n_samples // 4)
        decay = np.exp(-np.arange(min(2000, n_samples - pos)) / np.random.uniform(50, 300))
        freq = np.random.uniform(200, 2000)
        pulse = decay * np.sin(2 * np.pi * freq * np.arange(len(decay)) / sr)
        y[pos : pos + len(pulse)] += 0.3 * pulse

    # Noise
    noise = np.random.randn(n_samples).astype(np.float64)
    color = np.random.choice(["pink", "brown", "white"])
    if color == "pink":
        noise = np.cumsum(noise)
    elif color == "brown":
        noise = np.cumsum(np.cumsum(noise))
    noise /= np.abs(noise).max() + 1e-8
    y += 0.05 * noise

    # Normalize
    peak = np.abs(y).max() + 1e-8
    y /= peak
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)


def butter_lp(y, cutoff, sr, order=6):
    nyq = sr / 2
    if cutoff >= nyq * 0.98:
        return y.copy()
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Waveform U-Net (1D)
# ═══════════════════════════════════════════════════════════════════════════


class FiLM1D(nn.Module):
    """1D FiLM: condition on cutoff frequency."""

    def __init__(self, cond_dim, feat_ch):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feat_ch)
        self.bias = nn.Linear(cond_dim, feat_ch)

    def forward(self, features, condition):
        # features: (B, C, T), condition: (B, cond_dim)
        s = self.scale(condition).unsqueeze(-1) + 1.0
        b = self.bias(condition).unsqueeze(-1)
        return features * s + b


class EncoderBlock(nn.Module):
    """Downsampling encoder block: Conv1d + GLU + optional downsample."""

    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Conv1d(in_ch, out_ch * 2, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm1d(out_ch * 2)
        if downsample:
            self.down = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.glu(x, dim=1)  # splits channels in half, applies gate
        if self.downsample:
            x = self.down(x)
        return x


class DecoderBlock(nn.Module):
    """Upsampling decoder block with skip connection."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1, bias=False)
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch * 2, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm1d(out_ch * 2)

    def forward(self, x, skip):
        x = self.up(x)
        # Align lengths
        if x.shape[-1] > skip.shape[-1]:
            x = x[..., : skip.shape[-1]]
        elif x.shape[-1] < skip.shape[-1]:
            skip = skip[..., : x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        return F.glu(x, dim=1)


class WaveformUNet(nn.Module):
    """1D U-Net for waveform reconstruction with FiLM conditioning."""

    def __init__(self, base_ch=32):
        super().__init__()
        C = base_ch

        # Encoder
        self.enc1 = EncoderBlock(1, C, downsample=True)  # T -> T/2
        self.enc2 = EncoderBlock(C, C * 2, downsample=True)  # T/2 -> T/4
        self.enc3 = EncoderBlock(C * 2, C * 4, downsample=True)  # T/4 -> T/8
        self.enc4 = EncoderBlock(C * 4, C * 8, downsample=True)  # T/8 -> T/16

        # LSTM Bottleneck
        self.lstm = nn.LSTM(C * 8, C * 8, num_layers=2, bidirectional=True, batch_first=True)
        self.lstm_proj = nn.Conv1d(C * 16, C * 8, 1)  # bidirectional -> single

        # FiLM at bottleneck
        self.film = FiLM1D(1, C * 8)

        # Decoder with skip connections
        self.dec4 = DecoderBlock(C * 8, C * 4, C * 4)  # T/16 -> T/8
        self.dec3 = DecoderBlock(C * 4, C * 2, C * 2)  # T/8 -> T/4
        self.dec2 = DecoderBlock(C * 2, C, C)  # T/4 -> T/2
        self.dec1 = DecoderBlock(C, 0, C)  # T/2 -> T  (no skip for input)

        self.final = nn.Sequential(
            nn.Conv1d(C, C, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(C, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, cutoff_norm):
        # x: (B, 1, T), cutoff_norm: (B, 1)
        e1 = self.enc1(x)  # (B, C,   T/2)
        e2 = self.enc2(e1)  # (B, 2C,  T/4)
        e3 = self.enc3(e2)  # (B, 4C,  T/8)
        e4 = self.enc4(e3)  # (B, 8C,  T/16)

        # LSTM: (B, 8C, T/16) -> (B, T/16, 8C) -> LSTM -> (B, T/16, 16C) -> (B, 8C, T/16)
        b_lstm = e4.permute(0, 2, 1)
        b_lstm, _ = self.lstm(b_lstm)
        b_lstm = b_lstm.permute(0, 2, 1)
        b = self.lstm_proj(b_lstm)

        # FiLM
        b = self.film(b, cutoff_norm)

        d4 = self.dec4(b, e3)  # T/8
        d3 = self.dec3(d4, e2)  # T/4
        d2 = self.dec2(d3, e1)  # T/2

        # Final block without skip
        d2_up = F.interpolate(d2, scale_factor=2, mode="linear", align_corners=False)
        if d2_up.shape[-1] > x.shape[-1]:
            d2_up = d2_up[..., : x.shape[-1]]
        d1 = self.final(d2_up)

        return d1


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Resolution STFT Loss (waveform domain)
# ═══════════════════════════════════════════════════════════════════════════


def mr_stft_loss(pred_wave, target_wave):
    """Multi-resolution STFT loss on waveforms."""
    loss = 0.0
    for n_fft in [512, 1024, 2048]:
        for hop in [n_fft // 4, n_fft // 2]:
            pred_spec = torch.stft(
                pred_wave.squeeze(1),
                n_fft=n_fft,
                hop_length=hop,
                window=torch.hann_window(n_fft, device=pred_wave.device),
                return_complex=True,
            )
            target_spec = torch.stft(
                target_wave.squeeze(1),
                n_fft=n_fft,
                hop_length=hop,
                window=torch.hann_window(n_fft, device=target_wave.device),
                return_complex=True,
            )
            # Spectral convergence + magnitude loss
            sc_loss = (pred_spec - target_spec).abs().pow(2).sum() / (target_spec.abs().pow(2).sum() + 1e-8)
            mag_loss = F.l1_loss(pred_spec.abs(), target_spec.abs())
            loss += sc_loss + mag_loss  # type: ignore[assignment]
    return loss / 6.0  # Average over 6 combinations


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════


def train(epochs=200, batch_size=4, lr=1e-4, base_ch=32, steps_per_epoch=200):
    device = torch.device("cpu")
    print(f"Device: {device}")

    model = WaveformUNet(base_ch=base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"WaveformUNet: {n_params:,} params")
    print(f"Chunk: {CHUNK_SAMPLES} samples ({CHUNK_SEC}s) @ {SR} Hz")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    out_dir = Path("models/bw_reconstructor")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try loading V3 generator weights for the convolutional layers (partial transfer)
    v3_path = Path("models/bw_reconstructor/best_model_v3.pt")
    if v3_path.exists():
        ckpt = torch.load(v3_path, map_location=device, weights_only=True)
        ms = model.state_dict()
        loaded = 0
        for k, v in ckpt["model_state_dict"].items():
            # Map 2D conv weights to 1D where possible
            k.replace("enc1", "enc1").replace("enc2", "enc2")
            if k in ms and v.shape == ms[k].shape:
                ms[k].copy_(v)
                loaded += 1
        model.load_state_dict(ms, strict=False)
        if loaded > 0:
            print(f"Loaded {loaded} compatible weights from V3")

    best_loss = float("inf")
    needs_pad = CHUNK_SAMPLES % (2**4)  # 4 downsamples = factor 16
    if needs_pad:
        target_len = ((CHUNK_SAMPLES // 16) + 1) * 16
        print(f"Padding target length: {CHUNK_SAMPLES} -> {target_len}")
    else:
        target_len = CHUNK_SAMPLES

    print(f"Epochs: {epochs}, Batch: {batch_size}, Steps/epoch: {steps_per_epoch}")
    print(f"LR: {lr}, MR-STFT Loss (waveform)")
    print()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            batch_x, batch_y, batch_c = [], [], []

            for _ in range(batch_size):
                y_full = make_synthetic(target_len + SR, SR)
                cutoff = np.random.choice(CUTOFFS)
                y_lim = butter_lp(y_full, cutoff, SR)

                # Random crop
                start = np.random.randint(0, len(y_full) - target_len)
                batch_x.append(torch.from_numpy(y_lim[start : start + target_len]))
                batch_y.append(torch.from_numpy(y_full[start : start + target_len]))
                batch_c.append(cutoff / 11025.0)

            x = torch.stack(batch_x).unsqueeze(1).to(device)  # (B, 1, T)
            y = torch.stack(batch_y).unsqueeze(1).to(device)
            c = torch.tensor(batch_c, dtype=torch.float32, device=device).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(x, c)

            loss_wave = F.l1_loss(pred, y)
            loss_stft = mr_stft_loss(pred, y)
            loss = loss_wave + 0.5 * loss_stft

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
                {"model_state_dict": model.state_dict(), "epoch": epoch, "loss": avg_loss}, out_dir / "best_model_v5.pt"
            )

    # Export ONNX
    model.eval()
    dummy_input = torch.randn(1, 1, target_len).to(device)
    dummy_cutoff = torch.tensor([[0.5]], dtype=torch.float32, device=device)

    class ExportWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, cutoff):
            return self.m(x, cutoff)

    onnx_path = out_dir / "bw_reconstructor_v5.onnx"
    torch.onnx.export(
        ExportWrapper(model),
        (dummy_input, dummy_cutoff),
        str(onnx_path),
        input_names=["waveform", "cutoff"],
        output_names=["waveform_out"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "time"},
            "cutoff": {0: "batch"},
            "waveform_out": {0: "batch", 2: "time"},
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-ch", type=int, default=32)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.base_ch)
