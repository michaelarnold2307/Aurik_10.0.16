#!/usr/bin/env python3
"""
Fine-tune MP-SENet on MUSDB18 music data (§v10.17).

MP-SENet (Multi-Path SE Network) is Aurik's PRIMARY enhancement model,
running on ALL material. Originally trained on DNS5 Challenge (speech at 16kHz).
Fine-tuning on music replaces speech-specific PESQ/GAN training with
music-adapted spectral reconstruction.

Architecture: 4 TSC-Conformers, 64 channels, magnitude+phase input/output.
Native: 16kHz, n_fft=400, hop=100 → 201 freq bins.
Matches models/mp_senet/mp_senet.onnx (bundled ONNX).

Training: ~1-2 days on GPU (100 epochs, batch=8, 16kHz).

Usage:
    python scripts/train_mp_senet_musik.py --epochs 100 --batch-size 8 --lr 3e-5
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "models" / "mp_senet"))


# ── Constants (match config.json + ONNX model) ─────────────────────────────

SR = 16_000
N_FFT = 400
HOP = 100
WIN = 400
COMPRESS = 0.3  # magnitude compression exponent
FREQ_BINS = 201  # n_fft//2 + 1 = 201
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)  # 32000
CHUNK_FRAMES = CHUNK_SAMPLES // HOP  # 320
SNR_RANGE = (5.0, 20.0)

CHECKPOINT_DIR = _PROJECT / "models" / "mp_senet" / "finetuned"
BEST_PT = CHECKPOINT_DIR / "mp_senet_musik_best.pt"
LATEST_PT = CHECKPOINT_DIR / "checkpoint_latest.pt"


# ── STFT (matching mag_pha_stft from dataset.py) ──────────────────────────


def mag_pha_stft(y, n_fft=N_FFT, hop=HOP, win=WIN, compress=COMPRESS):
    """Magnitude + phase extraction from waveform. Returns (mag, pha, com)."""
    window = torch.hann_window(win, device=y.device)
    spec = torch.stft(
        y, n_fft, hop_length=hop, win_length=win, window=window, center=True, pad_mode="reflect", return_complex=True
    )
    spec_real = torch.view_as_real(spec)  # [B, F, T, 2]
    mag = torch.sqrt(spec_real.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(spec_real[..., 1] + 1e-10, spec_real[..., 0] + 1e-5)
    mag = torch.pow(mag, compress)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft=N_FFT, hop=HOP, win=WIN, compress=COMPRESS, length=None):
    """Inverse STFT from magnitude + phase. Returns waveform."""
    mag = torch.pow(mag, 1.0 / compress)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    window = torch.hann_window(win, device=com.device)
    return torch.istft(com, n_fft, hop_length=hop, win_length=win, window=window, center=True, length=length)


# ── Model Loading ─────────────────────────────────────────────────────────


def load_mpnet(device):
    """Load MPNet model, optionally with pre-trained weights."""
    from models.model import MPNet

    # Config matching ONNX model
    class H:
        dense_channel = 64
        compress_factor = COMPRESS
        num_tsconformers = 4
        beta = 2.0
        sampling_rate = SR
        n_fft = N_FFT
        hop_size = HOP
        win_size = WIN

    model = MPNet(H()).to(device)

    # Try loading pre-trained weights from ONNX companion checkpoint.
    # §v10.19-Fix: Offizielle MP-SENet-Releases heißen g_best_dns (DNS-Pretrained)
    # bzw. g_best_vb (VoiceBank) — „g_01000000" existierte hier nie, wodurch das
    # Finetune stillschweigend von Scratch lief.
    _pretrained_candidates = [
        _PROJECT / "models" / "mp_senet" / "best_ckpt" / "g_best_dns",
        _PROJECT / "models" / "mp_senet" / "best_ckpt" / "g_best_vb",
        _PROJECT / "models" / "mp_senet" / "best_ckpt" / "g_01000000",
    ]
    for pretrained in _pretrained_candidates:
        if pretrained.exists():
            print(f"Loading pre-trained weights from {pretrained}")
            state = torch.load(pretrained, map_location=device, weights_only=True)
            if "generator" in state:
                state = state["generator"]
            model.load_state_dict(state, strict=False)
            break
    else:
        print("WARNING: No pre-trained checkpoint found — training from scratch")

    return model


# ── Dataset ───────────────────────────────────────────────────────────────


class MusicMPSENetDataset(Dataset):
    """Streaming dataset from MUSDB18 stems."""

    def __init__(self, audio_files):
        self.files = audio_files

    def __len__(self):
        return len(self.files) * 20

    def _load(self, path):
        y, orig_sr = librosa.load(str(path), sr=None, mono=True)
        if orig_sr != SR:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=SR)
        y = y.astype(np.float32)
        if len(y) < CHUNK_SAMPLES:
            y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
        start = random.randint(0, max(0, len(y) - CHUNK_SAMPLES))
        return y[start : start + CHUNK_SAMPLES]

    def __getitem__(self, idx):
        clean = self._load(self.files[idx % len(self.files)])
        peak = np.abs(clean).max() + 1e-8
        clean = clean / peak

        noise = np.random.randn(CHUNK_SAMPLES).astype(np.float32)
        c = random.choice(["white", "pink", "brown"])
        if c == "pink":
            noise = np.cumsum(noise)
        elif c == "brown":
            noise = np.cumsum(np.cumsum(noise))
        noise = noise / (np.abs(noise).max() + 1e-8)

        snr_db = random.uniform(*SNR_RANGE)
        cr = np.sqrt(np.mean(clean**2) + 1e-8)
        nr = np.sqrt(np.mean(noise**2) + 1e-8)
        noise = noise * (cr / (10 ** (snr_db / 20))) / (nr + 1e-8)
        degraded = clean + noise

        dp = np.abs(degraded).max() + 1e-8
        return {"clean": torch.from_numpy(clean / dp), "degraded": torch.from_numpy(degraded / dp)}


# ── Loss ──────────────────────────────────────────────────────────────────


def spectral_loss(enhanced_mag, clean_mag, enhanced_com, clean_com):
    """Magnitude L1 + complex STFT loss."""
    loss_mag = F.l1_loss(enhanced_mag, clean_mag)

    # Complex domain loss
    loss_com = F.mse_loss(enhanced_com, clean_com)

    return loss_mag + 0.1 * loss_com


# ── Training ──────────────────────────────────────────────────────────────


def train(epochs=100, batch_size=8, lr=3e-5, steps_per_epoch=200, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Data
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    all_files = sorted(
        f
        for f in musdb.rglob("*.wav")
        if f.is_file() and any(s in f.stem for s in ["vocals", "drums", "bass", "other"])
    )
    if not all_files:
        print(f"ERROR: No stem files in {musdb}")
        return

    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * 0.2))
    train_files, val_files = all_files[n_val:], all_files[:n_val]

    train_ds = MusicMPSENetDataset(train_files)
    val_ds = MusicMPSENetDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Model
    model = load_mpnet(device)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: MP-SENet ({n_p:.2f}M) | Files: {len(all_files)} ({len(train_files)} train)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"Epochs: {epochs} | Batch: {batch_size} | Steps/ep: {steps_per_epoch} | LR: {lr}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break

            clean = batch["clean"].to(device)
            noisy = batch["degraded"].to(device)

            # STFT
            mag_c, pha_c, com_c = mag_pha_stft(clean)
            mag_n, pha_n, com_n = mag_pha_stft(noisy)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    enh_mag, enh_pha, enh_com = model(mag_n, pha_n)
                    loss = spectral_loss(enh_mag, mag_c, enh_com, com_c)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            else:
                enh_mag, enh_pha, enh_com = model(mag_n, pha_n)
                loss = spectral_loss(enh_mag, mag_c, enh_com, com_c)  # §v10.19-Fix: loss zuweisen (CPU-Pfad)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            train_loss += loss.item()

            if (step + 1) % 40 == 0:
                e = time.time() - t0
                eta = e / (step + 1) * (steps_per_epoch - step - 1) if step > 0 else 0
                print(
                    f"  Ep {epoch + 1:3d}/{epochs} | St {step + 1:3d}/{steps_per_epoch} | "
                    f"L {train_loss / (step + 1):.4f} | {e:.0f}s/{eta:.0f}s",
                    flush=True,
                )

        scheduler.step()
        avg_train = train_loss / min(steps_per_epoch, len(train_loader))

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                if vn >= 20:
                    break
                cv, nv = vb["clean"].to(device), vb["degraded"].to(device)
                mc, _, cc = mag_pha_stft(cv)
                mn, pn, _ = mag_pha_stft(nv)
                em, ep, ec = model(mn, pn)
                val_loss += spectral_loss(em, mc, ec, cc).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        print(
            f"Ep {epoch + 1:3d}/{epochs} | Tr {avg_train:.4f} | Val {avg_val:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s",
            flush=True,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
            },
            LATEST_PT,
        )
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_loss": avg_val}, BEST_PT)
            print(f"  >> Best: {best_val:.4f}")

    print(f"\nDone. Best val: {best_val:.4f} | {BEST_PT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune MP-SENet on music")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.steps_per_epoch, args.resume)
