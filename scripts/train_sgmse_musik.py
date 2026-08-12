#!/usr/bin/env python3
"""
Fine-tune SGMSE+ (Score-based Generative Model) on MUSDB18 music data (§v10.16).

Replaces speech-only training (VoiceBank-DEMAND, WSJ0, EARS-WHAM at 16kHz)
with music-specific fine-tuning at 48kHz.

Prerequisites:
  - Data prepared via scripts/prepare_sgmse_musik_data.py
  - Pre-trained checkpoint: models/sgmse_plus/sgmse_plus_src_1.ckpt
  - Ninja-free backbone patch applied (ncsnpp_utils/op/__init__.py)

Architecture:
  - Backbone: NCSNpp_48k (64.7M params), complex spectrogram input
  - SDE: OUVE (Ornstein-Uhlenbeck Variance Exploding)
  - Input: [B, 2, F, T] complex STFT, 48kHz
  - Output: score [B, 1, F, T] complex

Training: ~3-7 days on GPU (200 epochs, batch=4).

Usage:
    python scripts/train_sgmse_musik.py --epochs 200 --batch-size 4 --lr 3e-5
"""

import argparse, random, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "models" / "sgmse_plus"))


# ── SDE (Ornstein-Uhlenbeck Variance Exploding) ────────────────────────────

class OUVESDE:
    """Minimal OUV SDE for score matching. Matches sgmse.sdes.OUVESDE."""

    def __init__(self, sigma_min=0.05, sigma_max=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def perturb(self, x, t):
        """Add noise to x according to OUV process at time t."""
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        sigma = sigma.view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return x + sigma * noise, noise

    def loss_fn(self, model, x_clean, x_noisy, t):
        """Score-matching loss: ||score + noise/sigma||^2"""
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t.view(-1, 1, 1, 1)
        x_t = x_clean + sigma * torch.randn_like(x_clean)
        score_pred = model(x_t, t)
        target = -(x_t - x_clean) / (sigma ** 2 + 1e-8)
        return F.mse_loss(score_pred.real, target.real) + F.mse_loss(score_pred.imag, target.imag)


# ── Feature Extraction ──────────────────────────────────────────────────────

class STFTExtractor:
    """Compute complex STFT for SGMSE+ input."""

    def __init__(self, n_fft=1022, hop=256, device="cpu"):
        self.n_fft = n_fft
        self.hop = hop
        self.window = torch.hann_window(n_fft, device=device)

    def to(self, device):
        self.window = self.window.to(device)
        return self

    def __call__(self, audio):
        """audio: [B, T] → spec: [B, 2, F, T] complex"""
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True)
        return spec.unsqueeze(1)  # [B, 1, F, T] → stack as [B, 2, F, T]?


# ── Dataset (streaming from MUSDB18, no pre-generation needed) ──────────────

class SGMSE_Dataset(Dataset):
    """Streaming dataset: loads MUSDB18 stems directly, mixes noise on-the-fly."""

    def __init__(self, audio_files: list[Path]):
        self.files = audio_files

    def __len__(self):
        return len(self.files) * 20  # 20 chunks per file per epoch

    def _load(self, path: Path) -> np.ndarray:
        y, orig_sr = librosa.load(str(path), sr=None, mono=True)
        if orig_sr != 48000:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=48000)
        y = y.astype(np.float32)
        if len(y) < 192000:  # 4s @ 48kHz
            y = np.pad(y, (0, 192000 - len(y)), mode="reflect")
        start = random.randint(0, max(0, len(y) - 192000))
        return y[start:start + 192000]

    def __getitem__(self, idx):
        clean = self._load(self.files[idx % len(self.files)])
        peak = np.abs(clean).max() + 1e-8
        clean = clean / peak

        # Noise at random SNR
        noise = np.random.randn(192000).astype(np.float32)
        c = random.choice(["white", "pink", "brown"])
        if c == "pink": noise = np.cumsum(noise)
        elif c == "brown": noise = np.cumsum(np.cumsum(noise))
        noise = noise / (np.abs(noise).max() + 1e-8)

        snr_db = random.uniform(5.0, 20.0)
        cr = np.sqrt(np.mean(clean**2) + 1e-8)
        nr = np.sqrt(np.mean(noise**2) + 1e-8)
        noise = noise * (cr / (10**(snr_db/20))) / (nr + 1e-8)
        degraded = clean + noise

        dp = np.abs(degraded).max() + 1e-8
        return {"clean": torch.from_numpy(clean / dp),
                "noisy": torch.from_numpy(degraded / dp)}


# ── Training ────────────────────────────────────────────────────────────────

def train(
    epochs=200,
    batch_size=4,
    lr=3e-5,
    steps_per_epoch=200,
    ckpt_path="models/sgmse_plus/sgmse_plus_src_1.ckpt",
    resume=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Data — streaming directly from MUSDB18, no pre-generation
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    all_files = sorted(
        f for f in musdb.rglob("*.wav") if f.is_file()
        and any(s in f.stem for s in ["vocals", "drums", "bass", "other"])
    )
    if not all_files:
        print(f"ERROR: No stem files found in {musdb}")
        return

    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * 0.2))
    train_files, val_files = all_files[n_val:], all_files[:n_val]

    train_ds = SGMSE_Dataset(train_files)
    val_ds = SGMSE_Dataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=True)

    # Model
    from sgmse.backbones.ncsnpp_48k import NCSNpp_48k
    model = NCSNpp_48k().to(device)
    sde = OUVESDE()

    # Load pre-trained weights (partial — backbone only, not full Lightning ckpt)
    ckpt = Path(ckpt_path)
    if ckpt.exists():
        print(f"Loading pre-trained weights from {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        # Lightning checkpoint structure: state_dict has "model." prefix
        if "state_dict" in state:
            sd = {k.replace("model.", "").replace("_orig_mod.", ""): v
                  for k, v in state["state_dict"].items()}
        elif "model_state_dict" in state:
            sd = state["model_state_dict"]
        else:
            sd = state
        # Filter to backbone parameters only
        model_sd = {k: v for k, v in sd.items() if any(
            k.startswith(p) for p in ["enc", "dec", "output", "act", "norm"])}
        if model_sd:
            model.load_state_dict(model_sd, strict=False)
            print(f"  Loaded {len(model_sd)} backbone parameters")
        else:
            print("  WARNING: No backbone parameters found — starting from scratch")
    else:
        print(f"WARNING: No checkpoint at {ckpt} — training from scratch")

    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: NCSNpp_48k ({n_p:.1f}M) | Data: {len(train_ds)} train / {len(val_ds)} val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    out_dir = _PROJECT / "models" / "sgmse_plus" / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_epoch = 0

    if resume:
        rc = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(rc["model_state_dict"])
        start_epoch = rc.get("epoch", 0)
        best_val = rc.get("val_loss", float("inf"))

    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} → 1e-6")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)

            # STFT (complex)
            window = torch.hann_window(1022, device=device)
            spec_c = torch.stft(clean, n_fft=1022, hop_length=256, window=window, return_complex=True)
            spec_n = torch.stft(noisy, n_fft=1022, hop_length=256, window=window, return_complex=True)
            # Stack real+imag as complex: [B, F, T] → [B, 1, F, T] complex → [B, F, T]
            spec_c = spec_c.unsqueeze(1).contiguous()
            spec_n = spec_n.unsqueeze(1).contiguous()

            optimizer.zero_grad()
            t = torch.rand(batch_size, device=device)
            loss = sde.loss_fn(model, spec_c, spec_n, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item()

            if (step + 1) % 40 == 0:
                e = time.time() - t0
                eta = e / (step + 1) * (steps_per_epoch - step - 1) if step > 0 else 0
                print(f"  Ep {epoch+1:3d}/{epochs} | St {step+1:3d}/{steps_per_epoch} | "
                      f"L {train_loss/(step+1):.4f} | {e:.0f}s/{eta:.0f}s", flush=True)

        scheduler.step()
        avg_train = train_loss / min(steps_per_epoch, len(train_loader))

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                if vn >= 20: break
                cv, nv = vb["clean"].to(device), vb["noisy"].to(device)
                sc = torch.stft(cv, n_fft=1022, hop_length=256, window=window, return_complex=True).unsqueeze(1)
                sn = torch.stft(nv, n_fft=1022, hop_length=256, window=window, return_complex=True).unsqueeze(1)
                val_loss += sde.loss_fn(model, sc, sn, torch.rand(batch_size, device=device)).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        print(f"Ep {epoch+1:3d}/{epochs} | Tr {avg_train:.4f} | Val {avg_val:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.1e} | {time.time()-t0:.0f}s", flush=True)

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                    "optimizer_state_dict": optimizer.state_dict(), "val_loss": avg_val},
                   out_dir / "checkpoint_latest.pt")
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1, "val_loss": avg_val},
                       out_dir / "sgmse_musik_best.pt")
            print(f"  >> Best: {best_val:.4f}")

    print(f"\nDone. Best val: {best_val:.4f} | {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune SGMSE+ on music")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--ckpt", type=str, default="models/sgmse_plus/sgmse_plus_src_1.ckpt")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.steps_per_epoch,
          args.ckpt, args.resume)
