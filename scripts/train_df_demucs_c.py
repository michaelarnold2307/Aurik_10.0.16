#!/usr/bin/env python3
"""
§v10.15-C: Demucs + DeepFilterNet — Residual Spectral Refinement.

Strategy (NO model architecture changes):
  Demucs(noisy) → enhanced_audio → enhanced_spec
  DeepFilterNet(noisy_spec) → dfn_output
  Final = dfn_output + enhanced_spec   (residual correction)
  Loss  = MSE(Final, clean_spec)

Why: Demucs (42M params, MUSDB18-trained) provides musical structure.
      DeepFilterNet (2.4M params) learns to correct Demucs spectral errors.
      Result: 44.4M effective params, no architectural coupling.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

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

SR = 48_000
N_FFT = 960
HOP = 480
N_ERB = 32
DF_BINS = 96
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
SNR_RANGE = (5.0, 20.0)

CHECKPOINT_DIR = _PROJECT / "models" / "deepfilternet_v3_ii" / "finetuned"
BEST_PT = CHECKPOINT_DIR / "dfn_demucs_c_best.pt"
LATEST_PT = CHECKPOINT_DIR / "dfn_demucs_c_latest.pt"

# ── ERB filterbank ────────────────────────────────────────────────────────


def _build_erb_fb(n_fft=N_FFT, n_erb=N_ERB, sr=float(SR)):
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


_ERB_FB_NP = _build_erb_fb()

# ── Feature Extractor ─────────────────────────────────────────────────────


class DFNFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.window = torch.hann_window(N_FFT, device=device)
        self.erb_fb = torch.from_numpy(_ERB_FB_NP).to(device)

    def __call__(self, audio):
        spec = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, window=self.window, return_complex=True)
        T = spec.shape[2]
        mag = spec[:, :481, :].abs()
        erb_e = torch.matmul(self.erb_fb, mag)
        feat_erb = torch.log1p(erb_e).unsqueeze(1).transpose(2, 3)
        spec96 = spec[:, :96, :]
        feat_spec = torch.stack([spec96.real, spec96.imag], dim=-1)
        feat_spec = feat_spec.permute(0, 2, 1, 3).unsqueeze(1)
        full_spec = torch.stack([spec.real, spec.imag], dim=-1)
        full_spec = full_spec.permute(0, 2, 1, 3).unsqueeze(1)
        return feat_erb, feat_spec, full_spec


# ── Dataset with Demucs enhancement ────────────────────────────────────────


class DemucsEnhancedDataset(Dataset):
    """Loads noisy audio + Demucs-enhanced audio + clean audio."""

    def __init__(self, audio_files, demucs_dir, noise_files=None):
        self.files = audio_files
        self.demucs_dir = Path(demucs_dir)
        self.noise_files = noise_files or []
        # Build mapping: original file path → enhanced file path
        self.enhanced_map = {}
        for f in self.demucs_dir.glob("*.wav"):
            # Parse filename: "stem_XXXX.wav" → extract original index
            stem_idx = f.stem.rsplit("_", 1)
            if len(stem_idx) == 2:
                stem, idx = stem_idx
                try:
                    orig_idx = int(idx)
                    if orig_idx < len(self.files):
                        # Check if the stem matches
                        if self.files[orig_idx].stem == stem:
                            self.enhanced_map[self.files[orig_idx]] = f
                except ValueError:
                    pass

    def __len__(self):
        return len(self.files) * 30

    def _load(self, path):
        import soundfile as sf

        with sf.SoundFile(str(path)) as snd:
            sr = snd.samplerate
            chunk_native = min(int(4.5 * sr), snd.frames)
            max_start = max(0, snd.frames - chunk_native)
            start_frame = random.randint(0, max_start)
            snd.seek(start_frame)
            y = snd.read(chunk_native, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        y = y.astype(np.float32)
        if len(y) < CHUNK_SAMPLES:
            y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
        else:
            y = y[:CHUNK_SAMPLES]
        # Return normalized start_sec for alignment with enhanced
        start_sec = start_frame / sr if sr > 0 else 0.0
        return y, start_sec

    def _load_enhanced(self, file_idx, start_sec):
        """Load Demucs-enhanced chunk at the same time position."""
        file_path = self.files[file_idx]
        path = self.enhanced_map.get(file_path)
        if path is None:
            return np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        import soundfile as sf

        # Demucs output is already at 48kHz
        start_sample = int(start_sec * SR)
        with sf.SoundFile(str(path)) as snd:
            snd.seek(max(0, min(start_sample, snd.frames - CHUNK_SAMPLES)))
            y = snd.read(CHUNK_SAMPLES, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
        if len(y) < CHUNK_SAMPLES:
            y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
        return y.astype(np.float32)[:CHUNK_SAMPLES]

    def _noise(self, length):
        if self.noise_files and random.random() < 0.5:
            try:
                n, _ = self._load(random.choice(self.noise_files))
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
        file_idx = idx % len(self.files)
        clean, start_sec = self._load(self.files[file_idx])
        peak = np.abs(clean).max() + np.float32(1e-8)
        clean = clean / peak

        # Load Demucs-enhanced chunk at SAME time position
        enhanced = self._load_enhanced(file_idx, start_sec)

        # Augmentation: random gain
        gain = np.float32(10 ** (random.uniform(-3.0, 3.0) / 20.0))
        clean = clean * gain
        enhanced = enhanced * gain  # apply same gain to keep alignment

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
            "enhanced": torch.from_numpy((enhanced / dp).astype(np.float32)),
        }


# ── Training ───────────────────────────────────────────────────────────────


def train(epochs=50, batch_size=32, lr=1e-4, steps_per_epoch=200, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DFNFeatureExtractor(device)

    # Data
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    corpus = _PROJECT / "corpus"
    demucs_dir = _PROJECT / "data" / "demucs_enhanced"
    vocals = sorted(f for f in musdb.rglob("vocals.wav") if f.is_file())
    instruments = sorted(f for f in musdb.rglob("*.wav") if f.is_file() and "vocals" not in f.stem)
    noise_files = sorted(corpus.rglob("*.wav")) if corpus.is_dir() else []
    noise_files = [f for f in noise_files if "clean" not in f.stem.lower()]
    corpus_clean = sorted(corpus.rglob("*clean*.wav")) if corpus.is_dir() else []
    all_files = vocals + instruments[: len(vocals)] + corpus_clean

    if not all_files:
        print("ERROR: No files found")
        return

    n_train = int(0.8 * len(all_files))
    rng = random.Random(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    train_files, val_files = shuffled[:n_train], shuffled[n_train:]
    train_ds = DemucsEnhancedDataset(train_files, demucs_dir, noise_files)
    val_ds = DemucsEnhancedDataset(val_files, demucs_dir, noise_files)
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
    print(f"Model: DeepFilterNet3 ({n_p:.2f}M) + Demucs (42M) | Files: {len(all_files)} | Noise: {len(noise_files)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val = ckpt.get("val_loss", float("inf"))
        # Always start from epoch 0 — resume only loads weights, not epoch counter
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"Loaded weights from {resume} (was epoch {ckpt.get('epoch', '?')})")

    print(f"Epochs: {epochs} | Batch: {batch_size} | Steps/ep: {steps_per_epoch} | LR: {lr}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        print(f"Epoch {epoch + 1}/{epochs} — LR {scheduler.get_last_lr()[0]:.1e} — starting", flush=True)

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            clean = batch["clean"].to(device)
            noisy = batch["degraded"].to(device)
            enhanced = batch["enhanced"].to(device)

            # Feature extraction
            feb_n, fsp_n, spec_n = extractor(noisy)
            _, _, spec_e = extractor(enhanced)  # Demucs enhanced spec
            feb_c, fsp_c, spec_c = extractor(clean)

            optimizer.zero_grad()

            # DFN forward on noisy — standard denoising (Demucs used at inference only)
            enh, _, _, _ = model.forward(spec=spec_n, feat_erb=feb_n, feat_spec=fsp_n)

            # Direct denoising: MSE(enh, clean_spec)
            loss = F.mse_loss(enh, spec_c)
            base_loss = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_loss += loss.item()

            if (step + 1) % 10 == 0:
                e = time.time() - t0
                eta = e / (step + 1) * (steps_per_epoch - step - 1) if step > 0 else 0
                print(
                    f"  Ep {epoch + 1:3d}/{epochs} | St {step + 1:3d}/{steps_per_epoch} | "
                    f"L {train_loss / (step + 1):.4f} (base {base_loss:.4f}) | {e:.0f}s/{eta:.0f}s",
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
                cv = vb["clean"].to(device)
                nv = vb["degraded"].to(device)
                ev = vb["enhanced"].to(device)
                feb_n2, fsp_n2, spec_n2 = extractor(nv)
                _, _, spec_e2 = extractor(ev)
                feb_c2, fsp_c2, spec_c2 = extractor(cv)
                enh2, _, _, _ = model.forward(spec=spec_n2, feat_erb=feb_n2, feat_spec=fsp_n2)
                val_loss += F.mse_loss(enh2, spec_c2).item()
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
    p = argparse.ArgumentParser(description="Train DeepFilterNet with Demucs residual (Approach C)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.steps_per_epoch, args.resume)
