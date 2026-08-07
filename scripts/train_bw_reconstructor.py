#!/usr/bin/env python3
"""
Bandwidth Reconstructor — U-Net Training Pipeline für Aurik.

Trainiert ein schlankes U-Net (≈50 MB ONNX), das fehlende hohe Frequenzen
aus bandbegrenztem Audio rekonstruiert.

Voraussetzungen:
  pip install torch torchaudio numpy soundfile

Training:
  python scripts/train_bw_reconstructor.py --data MUSDB18-HQ/train --epochs 50

Das trainierte Modell wird als ONNX exportiert und in ~/.aurik/models/bw_reconstructor/
abgelegt, wo Auriks Plugin es findet.

Daten-Pipeline:
  1. Lade MUSDB18-HQ Track (7 Instrument-Spuren → Mixdown)
  2. Butterworth-Lowpass bei cutoff_freq (5/8/10 kHz)
  3. Input: bandbegrenztes Spektrogramm → U-Net → Output: volles Spektrogramm
  4. Loss: L1 + Multi-Resolution STFT Loss

Architektur: U-Net mit 4 Down/Up-Blöcken, ~2M Parameter, ~50 MB ONNX.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ── PyTorch-Imports (lazy, damit das Skript ohne GPU zumindest parsed) ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("❌ torch/torchaudio nicht installiert.")
    print("   pip install torch torchaudio numpy soundfile")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# U-Net Architektur
# ═══════════════════════════════════════════════════════════════════════════


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net für Spektrogramm-Inpainting (Bandbreiten-Rekonstruktion).

    Input:  [B, 1, 256, 256] — bandbegrenztes Mel-Spektrogramm
    Output: [B, 1, 256, 256] — volles Mel-Spektrogramm
    """

    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        f = features
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)
        self.final = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        b = self.bottleneck(p4)
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], 1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        return self.final(d1)


# ═══════════════════════════════════════════════════════════════════════════
# Daten-Pipeline
# ═══════════════════════════════════════════════════════════════════════════

MEL_PARAMS = {"n_fft": 2048, "hop_length": 512, "n_mels": 256, "sample_rate": 48000}
CUTOFF_FREQUENCIES = [5000, 8000, 10000]  # Shellac, Wax, Draht, Lackfolie


class BWDataset(Dataset):
    """Erzeugt bandbegrenzte Trainingspaare aus MUSDB18-HQ."""

    def __init__(self, data_dir: str, segment_seconds: float = 2.0, samples_per_track: int = 20):
        self.data_dir = Path(data_dir)
        self.segment_samples = int(segment_seconds * 48000)
        self.samples_per_track = samples_per_track
        self.tracks = sorted(self.data_dir.glob("**/mixture.wav"))
        if not self.tracks:
            raise FileNotFoundError(f"Keine mixture.wav in {data_dir} gefunden. MUSDB18-HQ Struktur erwartet.")
        print(
            f"Found {len(self.tracks)} tracks with {samples_per_track} segments each = {len(self.tracks) * samples_per_track} samples"
        )

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def __getitem__(self, idx):
        track_idx = idx // self.samples_per_track
        track_path = self.tracks[track_idx]

        # Lade Track
        audio, sr = torchaudio.load(str(track_path))
        audio = audio.mean(0)  # Mixdown zu Mono

        # Resample auf 48 kHz falls nötig
        if sr != 48000:
            audio = torchaudio.functional.resample(audio, sr, 48000)

        # Zufälliges Segment
        if len(audio) > self.segment_samples:
            start = np.random.randint(0, len(audio) - self.segment_samples)
            audio = audio[start : start + self.segment_samples]
        elif len(audio) < self.segment_samples:
            audio = F.pad(audio, (0, self.segment_samples - len(audio)))

        # Zufällige Cutoff-Frequenz
        cutoff = np.random.choice(CUTOFF_FREQUENCIES)

        # Bandbegrenzung via Butterworth
        audio_np = audio.numpy()
        limited = _butterworth_lowpass(audio_np, cutoff, 48000)

        # Mel-Spektrogramme
        mel_full = _to_mel(audio)
        mel_limited = _to_mel(torch.from_numpy(limited).float())

        # Normierung
        mel_full = (mel_full - mel_full.mean()) / (mel_full.std() + 1e-8)
        mel_limited = (mel_limited - mel_limited.mean()) / (mel_limited.std() + 1e-8)

        return mel_limited.unsqueeze(0), mel_full.unsqueeze(0)


def _butterworth_lowpass(audio: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(6, cutoff_hz / (sr / 2), btype="low", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)  # type: ignore[no-any-return]


def _to_mel(audio: torch.Tensor, sr: int = 48000, n_mels: int = 256) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels,
    )
    mel = mel_transform(audio)
    return torch.log(mel + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════


def multi_resolution_stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-Resolution STFT Loss für bessere spektrale Rekonstruktion."""
    loss = 0.0
    for n_fft in [512, 1024, 2048]:
        for hop in [n_fft // 4, n_fft // 2]:
            pred_stft = torch.stft(
                pred.squeeze(1),
                n_fft=n_fft,
                hop_length=hop,
                window=torch.hann_window(n_fft, device=pred.device),
                return_complex=True,
            )
            target_stft = torch.stft(
                target.squeeze(1),
                n_fft=n_fft,
                hop_length=hop,
                window=torch.hann_window(n_fft, device=target.device),
                return_complex=True,
            )
            loss += F.l1_loss(pred_stft.abs(), target_stft.abs())  # type: ignore[assignment]
    return loss / 6.0  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════


def train(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    output_dir: str = "~/.aurik/models/bw_reconstructor",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = BWDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = UNet(in_channels=1, out_channels=1, features=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (limited, full) in enumerate(dataloader):
            limited, full = limited.to(device), full.to(device)

            optimizer.zero_grad()
            pred = model(limited)

            loss_l1 = F.l1_loss(pred, full)
            loss_stft = multi_resolution_stft_loss(pred, full)
            loss = loss_l1 + 0.3 * loss_stft

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"\r  Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | Loss {loss.item():.4f}", end="")

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        print(f"\rEpoch {epoch + 1}/{epochs} | Avg Loss {avg_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path / "best_model.pt")
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")

    # Export ONNX
    print("\nExporting ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    onnx_path = output_path / "bw_reconstructor.onnx"

    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        input_names=["limited_mel"],
        output_names=["full_mel"],
        dynamic_axes={"limited_mel": {0: "batch"}, "full_mel": {0: "batch"}},
        opset_version=14,
    )

    # Größe prüfen
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"✓ ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
    print("✓ Training complete. Model ready for Aurik plugin.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Bandwidth Reconstructor")
    parser.add_argument("--data", required=True, help="MUSDB18-HQ train directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="~/.aurik/models/bw_reconstructor")
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size, args.lr, args.output)
