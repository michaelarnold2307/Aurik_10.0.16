#!/usr/bin/env python3
"""
Fine-tune DeepFilterNet v3.II on MUSDB18 music data (§v10.15).

Eliminates the Speech-Domain-Mismatch: instead of DNS-Challenge (speech+noise),
the model is trained on MUSDB18 stems with additive/musical degradation.

Dependencies (all in venv_rocm): torch, torchaudio, numpy, scipy, librosa, soundfile, onnx
NO libdf/Rust needed — ERB filterbank is pure NumPy (identical to plugin).

Architecture: DeepFilterNet3 (2.4M params), 48kHz, FFT=960, Hop=480.
Input:  feat_erb [B,1,32,T] + feat_spec [B,2,96,T]
Output: enh [B,2,96,T] (enhanced complex spectrogram)

Training: ~2-4 hours on GPU (50 epochs x 200 steps, batch 32).

Usage:
    /home/michael/.local/share/aurik/venv_rocm/bin/python scripts/train_df_musik.py
"""

from __future__ import annotations

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
sys.path.insert(0, str(_PROJECT / "models" / "deepfilternet_v3_ii" / "DeepFilterNet"))
sys.path.insert(0, str(_PROJECT / "models" / "deepfilternet_v3_ii" / "pyDF-data"))

# ── Constants (match plugins/deepfilternet_v3_ii_plugin.py) ─────────────────

SR = 48_000
N_FFT = 960
HOP = 480
N_ERB = 32
DF_BINS = 96
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
SNR_RANGE = (5.0, 20.0)

CHECKPOINT_DIR = _PROJECT / "models" / "deepfilternet_v3_ii" / "finetuned"
BEST_PT = CHECKPOINT_DIR / "dfn_musik_best.pt"
LATEST_PT = CHECKPOINT_DIR / "dfn_musik_latest.pt"


# ── ERB filterbank — pure NumPy (identical to plugin) ──────────────────────


def _build_erb_fb(n_fft=N_FFT, n_erb=N_ERB, sr=float(SR)):
    """ERB filterbank [n_erb, n_fft//2+1]. Zwicker ERB scale."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)

    def hz2erb(f):
        return 21.4 * np.log10(1.0 + f / 229.0 + 1e-9)

    erb_max = hz2erb(np.array([sr / 2]))[0]
    edges = np.linspace(hz2erb(np.array([0.0]))[0], erb_max, n_erb + 1)
    fb = np.zeros((n_erb, n_bins), dtype=np.float32)
    for b in range(n_erb):
        lo, hi = edges[b], edges[b + 1]
        mask = (hz2erb(freqs) >= lo) & (hz2erb(freqs) < hi)
        if mask.sum() > 0:
            fb[b, mask] = 1.0 / mask.sum()
    return fb


_ERB_FB_NP = _build_erb_fb()  # [32, 481]


# ── Feature Extractor (GPU-accelerated) ────────────────────────────────────


class DFNFeatureExtractor:
    """Compute ERB + complex STFT features for DeepFilterNet3.

    Output format (matches DfNet.forward expectation):
      feat_erb:  [B, 1, T, nb_erb=32]      — ERB log-energy per frame
      feat_spec: [B, 1, T, nb_df=96, 2]    — complex spec (real+imag in last dim)
      spec:      [B, 1, T, 481, 2]         — full complex spectrogram
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.window = torch.hann_window(N_FFT, device=device)
        self.erb_fb = torch.from_numpy(_ERB_FB_NP).to(device)  # [32, 481]

    def __call__(self, audio):
        """audio: [B, T] mono waveforms at 48kHz.
        Returns (feat_erb, feat_spec, spec).
        """
        # STFT: [B, freq=481, frames]
        spec = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, window=self.window, return_complex=True)
        T = spec.shape[2]

        # ERB: [B, 32, T] → [B, 1, T, 32]
        mag = spec[:, :481, :].abs()
        erb_e = torch.matmul(self.erb_fb, mag)  # [B, 32, T]
        feat_erb = torch.log1p(erb_e).unsqueeze(1).transpose(2, 3)  # [B, 1, T, 32]

        # feat_spec: [B, 1, T, 96, 2] — first 96 bins only
        spec96 = spec[:, :96, :]  # [B, 96, T]
        feat_spec = torch.stack([spec96.real, spec96.imag], dim=-1)  # [B, 96, T, 2]
        feat_spec = feat_spec.permute(0, 2, 1, 3).unsqueeze(1)  # [B, 1, T, 96, 2]

        # spec: [B, 1, T, 481, 2] — full freq bins
        full_spec = torch.stack([spec.real, spec.imag], dim=-1)  # [B, 481, T, 2]
        full_spec = full_spec.permute(0, 2, 1, 3).unsqueeze(1)  # [B, 1, T, 481, 2]

        return feat_erb, feat_spec, full_spec


# ── Dataset ────────────────────────────────────────────────────────────────


class MusicDenoiseDataset(Dataset):
    """Loads MUSDB18 stems, adds noise at random SNR, returns clean+noisy."""

    def __init__(self, audio_files, noise_files=None):
        self.files = audio_files
        self.noise_files = noise_files or []

    def __len__(self):
        return len(self.files) * 30

    def _load(self, path):
        import soundfile as sf

        # Read only the chunk we need (4.5s at native rate → ~200k samples)
        with sf.SoundFile(str(path)) as snd:
            sr = snd.samplerate
            # 4.5 seconds at native rate gives enough for resampling to 4.0s at 48kHz
            chunk_native = min(int(4.5 * sr), snd.frames)
            # Random start position in the file
            max_start = max(0, snd.frames - chunk_native)
            start_frame = random.randint(0, max_start)
            snd.seek(start_frame)
            y = snd.read(chunk_native, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)  # stereo → mono
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        y = y.astype(np.float32)
        if len(y) < CHUNK_SAMPLES:
            y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
        else:
            y = y[:CHUNK_SAMPLES]
        return y

    def _noise(self, length):
        if self.noise_files and random.random() < 0.5:
            try:
                n = self._load(random.choice(self.noise_files))
                return n / (np.abs(n).max() + np.float32(1e-8))
            except Exception:
                pass
        n = np.random.randn(length).astype(np.float32)
        c = random.choice(["white", "pink", "brown"])
        if c == "pink":
            n = np.cumsum(n)
        elif c == "brown":
            n = np.cumsum(np.cumsum(n))
        return n / (np.abs(n).max() + np.float32(1e-8))

    def __getitem__(self, idx):
        clean = self._load(self.files[idx % len(self.files)])
        peak = np.abs(clean).max() + np.float32(1e-8)
        clean = clean / peak

        # §v10.15-fix: Random gain augmentation (±3 dB) to combat overfitting
        gain = 10 ** (random.uniform(-3.0, 3.0) / 20.0)
        clean = clean * gain

        noise = self._noise(CHUNK_SAMPLES)
        snr_db = random.uniform(*SNR_RANGE)
        cr = np.sqrt(np.mean(clean**2) + np.float32(1e-8))
        nr = np.sqrt(np.mean(noise**2) + np.float32(1e-8))
        noise = noise * (cr / (np.float32(10 ** (snr_db / 20)))) / (nr + np.float32(1e-8))
        degraded = clean + noise

        dp = np.abs(degraded).max() + np.float32(1e-8)
        return {
            "clean": torch.from_numpy((clean / dp).astype(np.float32)),
            "degraded": torch.from_numpy((degraded / dp).astype(np.float32)),
        }


# ── Training ───────────────────────────────────────────────────────────────


def train(epochs=50, batch_size=32, lr=1e-4, steps_per_epoch=200, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DFNFeatureExtractor(device)

    # Data
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    corpus = _PROJECT / "corpus"
    vocals = sorted(f for f in musdb.rglob("vocals.wav") if f.is_file())
    instruments = sorted(f for f in musdb.rglob("*.wav") if f.is_file() and "vocals" not in f.stem)
    noise_files = sorted(corpus.rglob("*.wav")) if corpus.is_dir() else []
    noise_files = [f for f in noise_files if "clean" not in f.stem.lower()]
    # §v10.19: Corpus clean files as additional training sources
    corpus_clean = sorted(corpus.rglob("*clean*.wav")) if corpus.is_dir() else []
    all_files = vocals + instruments[: len(vocals)] + corpus_clean  # MUSDB + Corpus clean

    if not all_files:
        print("ERROR: No MUSDB18 files found")
        return

    n_train = int(0.8 * len(all_files))
    # §v10.15-fix: shuffle before split so val has balanced source types
    rng = random.Random(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    train_files, val_files = shuffled[:n_train], shuffled[n_train:]
    train_ds = MusicDenoiseDataset(train_files, noise_files)
    val_ds = MusicDenoiseDataset(val_files, noise_files)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, prefetch_factor=2
    )

    # Model
    from df.config import config

    config.use_defaults()
    from df.deepfilternet3 import init_model

    model = init_model().to(device)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: DeepFilterNet3 ({n_p:.2f}M) | Files: {len(all_files)} | Noise: {len(noise_files)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
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
        print(f"Epoch {epoch + 1}/{epochs} — LR {scheduler.get_last_lr()[0]:.1e} — starting", flush=True)

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            clean, noisy = batch["clean"].to(device), batch["degraded"].to(device)
            feb_n, fsp_n, spec_n = extractor(noisy)
            feb_c, fsp_c, spec_c = extractor(clean)
            optimizer.zero_grad()

            # No autocast — DeepFilterNet uses complex tensors (ComplexHalf unsupported on ROCm)
            enh, _, _, _ = model.forward(spec=spec_n, feat_erb=feb_n, feat_spec=fsp_n)
            loss = F.mse_loss(enh, spec_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_loss += loss.item()

            if (step + 1) % 10 == 0:
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
                feb_n2, fsp_n2, spec_n2 = extractor(nv)
                feb_c2, fsp_c2, spec_c2 = extractor(cv)
                enh, _, _, _ = model.forward(spec=spec_n2, feat_erb=feb_n2, feat_spec=fsp_n2)
                val_loss += F.mse_loss(enh, spec_c2).item()
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


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune DeepFilterNet on music")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.steps_per_epoch, args.resume)
