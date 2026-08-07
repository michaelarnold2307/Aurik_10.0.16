#!/usr/bin/env python3
"""
BW Reconstructor V3 — Adversarial Training (PatchGAN).

Neu gegenüber V2:
  - PatchGAN-Diskriminator (unterscheidet echte vs. rekonstruierte Mel-Patches)
  - Hinge-Loss für stabileres GAN-Training
  - Spectral Normalization im Diskriminator
  - Adaptive Loss-Balancierung (L1 + MR-STFT + GAN)
  - Diskriminator wird alle 2 Schritte aktualisiert (stabiler)

Generator = CompactUNetV2 (FiLM + U-Net), trainiert durch kombinierten Loss:
  L_total = lambda_L1 * L1 + lambda_STFT * MR-STFT + lambda_GAN * Hinge_G
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
# Audio generation (unverändert von V2)
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
# Mel spectrogram
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
# Generator (CompactUNetV2 — unverändert von V2)
# ═══════════════════════════════════════════════════════════════════════════


class FiLMBlock(nn.Module):
    def __init__(self, cond_dim, feat_ch):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feat_ch)
        self.bias = nn.Linear(cond_dim, feat_ch)

    def forward(self, features, condition):
        s = self.scale(condition).unsqueeze(-1).unsqueeze(-1) + 1.0
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


class Generator(nn.Module):
    def __init__(self, base_ch=24):
        super().__init__()
        C = base_ch
        self.enc1 = ConvBlock(1, C)
        self.enc2 = ConvBlock(C, C * 2)
        self.enc3 = ConvBlock(C * 2, C * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(C * 4, C * 8)
        self.film_bn = FiLMBlock(1, C * 8)
        self.up3 = nn.ConvTranspose2d(C * 8, C * 4, 2, 2)
        self.dec3 = ConvBlock(C * 8, C * 4)
        self.film_d3 = FiLMBlock(1, C * 4)
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, 2)
        self.dec2 = ConvBlock(C * 4, C * 2)
        self.up1 = nn.ConvTranspose2d(C * 2, C, 2, 2)
        self.dec1 = ConvBlock(C * 2, C)
        self.final = nn.Conv2d(C, 1, 1)

    def forward(self, x, cutoff_norm):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self.film_bn(b, cutoff_norm)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d3 = self.film_d3(d3, cutoff_norm)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# ═══════════════════════════════════════════════════════════════════════════
# PatchGAN Discriminator (leichtgewichtig, spectral-normed)
# ═══════════════════════════════════════════════════════════════════════════


class PatchGANDiscriminator(nn.Module):
    """PatchGAN: klassifiziert 16x16 Patches als real/fake."""

    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        C = base_ch

        def sn_conv(in_ch, out_ch, k, s, p):
            return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False))

        self.layers = nn.Sequential(
            sn_conv(in_channels, C, 4, 2, 1),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C, C * 2, 4, 2, 1),  # 64x64 -> 32x32
            nn.BatchNorm2d(C * 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C * 2, C * 4, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(C * 4),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C * 4, C * 8, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(C * 8),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C * 8, 1, 4, 1, 0),  # 8x8 -> 5x5
        )

    def forward(self, x):
        return self.layers(x)  # (B, 1, 5, 5) — Patch-Klassifikationen


# ═══════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════


def multi_resolution_stft_loss(pred, target):
    """Multi-scale spectral loss on Mel spectrogram rows (freq axis)."""
    B = pred.shape[0]
    loss = 0.0
    for n_fft in [32, 64, 128]:
        for b in range(B):
            ps = torch.stft(
                pred[b, 0, :, :],
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft, device=pred.device),
                return_complex=True,
                pad_mode="reflect",
            ).abs()
            ts = torch.stft(
                target[b, 0, :, :],
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft, device=target.device),
                return_complex=True,
                pad_mode="reflect",
            ).abs()
            loss += F.l1_loss(ps, ts)  # type: ignore[assignment]
    return loss / (B * 3)


def generator_loss(disc_fake):
    """Hinge-Generator-Loss: G will, dass D(fake) > 0."""
    return -disc_fake.mean()


def discriminator_loss(disc_real, disc_fake):
    """Hinge-Discriminator-Loss."""
    loss_real = F.relu(1.0 - disc_real).mean()
    loss_fake = F.relu(1.0 + disc_fake).mean()
    return loss_real + loss_fake


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════


def train(epochs=300, batch_size=6, lr=1e-4, base_ch=24, steps_per_epoch=200):
    device = torch.device("cpu")
    segment_samples = (N_FRAMES - 1) * HOP + N_FFT
    extra_samples = SR

    generator = Generator(base_ch=base_ch).to(device)
    discriminator = PatchGANDiscriminator(in_channels=1, base_ch=32).to(device)

    n_g = sum(p.numel() for p in generator.parameters())
    n_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Device: {device}")
    print(f"Generator: {n_g:,} params | Discriminator: {n_d:,} params")
    print(f"Total: {n_g + n_d:,} params")

    # Load V2 checkpoint for generator (transfer learning)
    ckpt_path = Path("models/bw_reconstructor/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        g_state = generator.state_dict()
        pretrained = {k: v for k, v in ckpt["model_state_dict"].items() if k in g_state and v.shape == g_state[k].shape}
        g_state.update(pretrained)
        generator.load_state_dict(g_state, strict=False)
        print(f"Loaded {len(pretrained)}/{len(g_state)} generator weights from V2 checkpoint")

    opt_g = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr * 0.5, weight_decay=1e-5, betas=(0.5, 0.999))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, epochs)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, epochs)

    out_dir = Path("models/bw_reconstructor")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss weights (progressive: start with L1+STFT, gradually add GAN)
    lambda_stft = 0.3
    lambda_gan_start = 0.0
    lambda_gan_end = 0.1

    print(f"Epochs: {epochs}, Batch: {batch_size}, Steps/epoch: {steps_per_epoch}")
    print(f"LR: {lr}, Hinge-GAN + PatchGAN Discriminator")
    print()

    best_loss = float("inf")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        # GAN weight ramps up over first 50 epochs
        gan_weight = lambda_gan_start + (lambda_gan_end - lambda_gan_start) * min(1.0, epoch / 50)

        total_g_loss = 0.0
        total_d_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            # --- Generate batch ---
            batch_x, batch_y, batch_c = [], [], []
            for _ in range(batch_size):
                y_full = make_synthetic(segment_samples + extra_samples, SR)
                cutoff = np.random.choice(CUTOFFS)
                y_lim = butter_lp(y_full, cutoff, SR)
                start = np.random.randint(0, len(y_full) - segment_samples)
                batch_x.append(torch.from_numpy(y_lim[start : start + segment_samples]))
                batch_y.append(torch.from_numpy(y_full[start : start + segment_samples]))
                batch_c.append(cutoff / 11025.0)

            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)
            c = torch.tensor(batch_c, dtype=torch.float32, device=device).unsqueeze(1)

            mel_x = audio_to_log_mel(x).unsqueeze(1)
            mel_y = audio_to_log_mel(y).unsqueeze(1)

            # --- Generator step ---
            opt_g.zero_grad()
            pred = generator(mel_x, c)

            loss_l1 = F.l1_loss(pred, mel_y)
            loss_stft = multi_resolution_stft_loss(pred, mel_y)

            if gan_weight > 0:
                disc_fake = discriminator(pred)
                loss_gan = generator_loss(disc_fake)
                g_loss = loss_l1 + lambda_stft * loss_stft + gan_weight * loss_gan
            else:
                g_loss = loss_l1 + lambda_stft * loss_stft

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 2.0)
            opt_g.step()

            # --- Discriminator step (every 2nd step for stability) ---
            d_loss = torch.tensor(0.0)
            if gan_weight > 0 and step % 2 == 0:
                opt_d.zero_grad()
                with torch.no_grad():
                    pred_detached = generator(mel_x, c).detach()

                disc_real = discriminator(mel_y)
                disc_fake = discriminator(pred_detached)
                d_loss = discriminator_loss(disc_real, disc_fake)

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 2.0)
                opt_d.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else 0.0

        avg_g = total_g_loss / steps_per_epoch
        avg_d = total_d_loss / steps_per_epoch if total_d_loss > 0 else 0

        sch_g.step()
        sch_d.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | G:{avg_g:.4f} D:{avg_d:.4f} "
            f"GAN:{gan_weight:.3f} | LR {sch_g.get_last_lr()[0]:.1e} | {elapsed:.0f}s",
            flush=True,
        )

        if avg_g < best_loss:
            best_loss = avg_g
            torch.save(
                {"model_state_dict": generator.state_dict(), "epoch": epoch, "loss": avg_g},
                out_dir / "best_model_v3.pt",
            )

    # Export ONNX (generator only)
    generator.eval()
    dummy_input = torch.randn(1, 1, N_MELS, N_FRAMES).to(device)
    dummy_cutoff = torch.tensor([[0.5]], dtype=torch.float32, device=device)

    class ExportWrapper(nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.gen = gen

        def forward(self, x, cutoff):
            return self.gen(x, cutoff)

    export_model = ExportWrapper(generator)
    onnx_path = out_dir / "bw_reconstructor_v3.onnx"

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
    print(f"Best G loss: {best_loss:.4f}")
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
