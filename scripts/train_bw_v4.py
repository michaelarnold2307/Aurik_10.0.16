#!/usr/bin/env python3
"""BW Reconstructor V4 — GAN + MUSDB18 (echte Musik)."""

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.signal import butter, sosfiltfilt

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128
N_FRAMES = 128
SEGMENT_SAMPLES = (N_FRAMES - 1) * HOP + N_FFT
CUTOFFS = [3000, 5000, 8000, 11000]


class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train"):
        self.data_dir = Path(data_dir) / split
        self.segment_samples = SEGMENT_SAMPLES
        self.tracks = sorted(self.data_dir.glob("*/mixture.wav"))
        if not self.tracks:
            raise FileNotFoundError(f"No mixture.wav in {self.data_dir}")
        print(f"MUSDB18 {split}: {len(self.tracks)} tracks, {self.segment_samples} samples/seg")

    def __len__(self):
        return len(self.tracks) * 10

    def __getitem__(self, idx):
        track_idx = idx % len(self.tracks)
        audio, file_sr = torchaudio.load(str(self.tracks[track_idx]))
        audio = audio.mean(0)
        if file_sr != SR:
            audio = torchaudio.functional.resample(audio, file_sr, SR)
        needed = self.segment_samples + SR
        if len(audio) < needed:
            audio = F.pad(audio, (0, needed - len(audio)))
        max_start = len(audio) - self.segment_samples
        start = random.randint(0, max_start)
        audio_seg = audio[start : start + self.segment_samples].numpy().astype(np.float64)
        peak = np.abs(audio_seg).max() + 1e-8
        audio_seg = (audio_seg / peak).astype(np.float64)
        cutoff = random.choice(CUTOFFS)
        sos = butter(6, cutoff / (SR / 2), btype="low", output="sos")
        audio_lim = sosfiltfilt(sos, audio_seg).astype(np.float32)
        return (
            torch.from_numpy(audio_lim),
            torch.from_numpy(audio_seg.astype(np.float32)),
            torch.tensor(cutoff / 11025.0, dtype=torch.float32),
        )


mel_t = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=60, f_max=SR // 2, center=False
)


def audio_to_log_mel(waveform):
    mel = mel_t(waveform)
    mel = torch.log(mel + 1e-6)
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std = mel.std(dim=(1, 2), keepdim=True) + 1e-8
    return (mel - mean) / std


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


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        C = base_ch

        def sn_conv(in_ch, out_ch, k, s, p):
            return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False))

        self.layers = nn.Sequential(
            sn_conv(in_channels, C, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C, C * 2, 4, 2, 1),
            nn.BatchNorm2d(C * 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C * 2, C * 4, 4, 2, 1),
            nn.BatchNorm2d(C * 4),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C * 4, C * 8, 4, 2, 1),
            nn.BatchNorm2d(C * 8),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(C * 8, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.layers(x)


def multi_resolution_stft_loss(pred, target):
    B, loss = pred.shape[0], 0.0
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
    return -disc_fake.mean()


def discriminator_loss(disc_real, disc_fake):
    return F.relu(1.0 - disc_real).mean() + F.relu(1.0 + disc_fake).mean()


def train(data_dir="models/musdb18hq", epochs=100, batch_size=6, lr=1e-4, base_ch=24):
    device = torch.device("cpu")
    print(f"Device: {device}\nData:   {data_dir}")
    dataset = MUSDB18Dataset(data_dir, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    generator = Generator(base_ch=base_ch).to(device)
    discriminator = PatchGANDiscriminator(in_channels=1, base_ch=32).to(device)
    print(
        f"Generator: {sum(p.numel() for p in generator.parameters()):,} params | Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params"
    )

    ckpt_path = Path("models/bw_reconstructor/best_model_v3.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        gs = generator.state_dict()
        pretrained = {k: v for k, v in ckpt["model_state_dict"].items() if k in gs and v.shape == gs[k].shape}
        gs.update(pretrained)
        generator.load_state_dict(gs, strict=False)
        print(f"Loaded {len(pretrained)}/{len(gs)} generator weights from V3 checkpoint")

    opt_g = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-5, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr * 0.5, weight_decay=1e-5, betas=(0.5, 0.999))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, epochs)
    out_dir = Path("models/bw_reconstructor")
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_stft = 0.3
    lambda_gan_end = 0.1
    steps_per_epoch = min(100, len(dataloader))
    print(f"Epochs: {epochs}, Batch: {batch_size}, Steps/epoch: {steps_per_epoch}\nLR: {lr}, Hinge-GAN + MUSDB18\n")
    best_loss = float("inf")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        gan_weight = lambda_gan_end * min(1.0, epoch / 50)
        total_g_loss = 0.0
        total_d_loss = 0.0
        t0 = time.time()
        for step, (x_audio, y_audio, cutoff_norm) in enumerate(dataloader):
            if step >= steps_per_epoch:
                break
            x_audio = x_audio.to(device)
            y_audio = y_audio.to(device)
            cutoff_norm = cutoff_norm.to(device).unsqueeze(1)
            mel_x = audio_to_log_mel(x_audio).unsqueeze(1)
            mel_y = audio_to_log_mel(y_audio).unsqueeze(1)
            # Generator
            opt_g.zero_grad()
            pred = generator(mel_x, cutoff_norm)
            loss_l1 = F.l1_loss(pred, mel_y)
            loss_stft = multi_resolution_stft_loss(pred, mel_y)
            if gan_weight > 0:
                loss_gan = generator_loss(discriminator(pred))
                g_loss = loss_l1 + lambda_stft * loss_stft + gan_weight * loss_gan
            else:
                g_loss = loss_l1 + lambda_stft * loss_stft
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 2.0)
            opt_g.step()
            # Discriminator
            d_loss = torch.tensor(0.0)
            if gan_weight > 0 and step % 2 == 0:
                opt_d.zero_grad()
                with torch.no_grad():
                    pred_detached = generator(mel_x, cutoff_norm).detach()
                d_loss = discriminator_loss(discriminator(mel_y), discriminator(pred_detached))
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 2.0)
                opt_d.step()
            total_g_loss += g_loss.item()
            if isinstance(d_loss, torch.Tensor):
                total_d_loss += d_loss.item()
        avg_g = total_g_loss / min(steps_per_epoch, step + 1)
        avg_d = total_d_loss / max(1, (step + 1) // 2)
        sch_g.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | G:{avg_g:.4f} D:{avg_d:.4f} GAN:{gan_weight:.3f} | LR {sch_g.get_last_lr()[0]:.1e} | {elapsed:.0f}s",
            flush=True,
        )
        if avg_g < best_loss:
            best_loss = avg_g
            torch.save(
                {"model_state_dict": generator.state_dict(), "epoch": epoch, "loss": avg_g},
                out_dir / "best_model_v4.pt",
            )

    # Export ONNX
    generator.eval()
    dummy_input = torch.randn(1, 1, N_MELS, N_FRAMES).to(device)
    dummy_cutoff = torch.tensor([[0.5]], dtype=torch.float32, device=device)

    class Wrapper(nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.gen = gen

        def forward(self, x, cutoff):
            return self.gen(x, cutoff)

    export_model = Wrapper(generator)
    onnx_path = out_dir / "bw_reconstructor_v4.onnx"
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
    print(
        f"\nONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)\nBest G loss: {best_loss:.4f}\nDone!"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
