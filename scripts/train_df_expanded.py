#!/usr/bin/env python3
"""
§v10.21: Erweitertes DFN-Training mit FMA-small (8.000 Tracks) + MUSDB18.

Nutzt den PyTorch-GPU-Pfad direkt (ONNX-ROCm ist auf gfx1100 blockiert).
"""

from __future__ import annotations

import argparse, random, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

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
BEST_PT = CHECKPOINT_DIR / "dfn_expanded_best.pt"
LATEST_PT = CHECKPOINT_DIR / "dfn_expanded_latest.pt"

# ERB filterbank
def _build_erb_fb(n_fft=N_FFT, n_erb=N_ERB, sr=float(SR)):
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    def hz2erb(f): return 21.4 * np.log10(1.0 + f / 229.0 + 1e-9)
    erb_max = hz2erb(np.array([sr / 2]))[0]
    edges = np.linspace(hz2erb(np.array([0.0]))[0], erb_max, n_erb + 1)
    fb = np.zeros((n_erb, n_bins), dtype=np.float32)
    for b in range(n_erb):
        lo, hi = edges[b], edges[b + 1]
        mask = (hz2erb(freqs) >= lo) & (hz2erb(freqs) < hi)
        if mask.sum() > 0: fb[b, mask] = 1.0 / mask.sum()
    return fb

_ERB_FB_NP = _build_erb_fb()

class DFNFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.window = torch.hann_window(N_FFT, device=device)
        self.erb_fb = torch.from_numpy(_ERB_FB_NP).to(device)

    def __call__(self, audio):
        spec = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, window=self.window, return_complex=True)
        mag = spec.abs()
        erb_e = torch.matmul(self.erb_fb, mag)
        feat_erb = torch.log1p(erb_e).unsqueeze(1).transpose(2, 3)
        spec96 = spec[:, :DF_BINS, :]
        feat_spec = torch.stack([spec96.real, spec96.imag], dim=-1)
        feat_spec = feat_spec.permute(0, 2, 1, 3).unsqueeze(1)
        full_spec = torch.stack([spec.real, spec.imag], dim=-1)
        full_spec = full_spec.permute(0, 2, 1, 3).unsqueeze(1)
        return feat_erb, feat_spec, full_spec


class AudioDenoiseDataset(Dataset):
    """Loads clean audio from any source, adds noise."""
    def __init__(self, files, noise_files=None):
        self.files = files
        self.noise_files = noise_files or []

    def __len__(self):
        return max(len(self.files), 200) * 20  # More variety per file

    def _load(self, path):
        with sf.SoundFile(str(path)) as snd:
            sr = snd.samplerate
            chunk_native = min(int(4.5 * sr), snd.frames)
            max_start = max(0, snd.frames - chunk_native)
            start_frame = random.randint(0, max_start) if max_start > 0 else 0
            snd.seek(start_frame)
            y = snd.read(chunk_native, dtype='float32')
            if y.ndim > 1: y = y.mean(axis=1)
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
            except Exception: pass
        n = np.random.randn(length).astype(np.float32)
        c = random.choice(["white", "pink", "brown"])
        if c == "pink": n = np.cumsum(n)
        elif c == "brown": n = np.cumsum(np.cumsum(n))
        return n / (np.abs(n).max() + np.float32(1e-8))

    def __getitem__(self, idx):
        f = self.files[idx % len(self.files)]
        try:
            clean = self._load(f)
        except Exception:
            clean = np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.01
        peak = np.abs(clean).max() + np.float32(1e-8)
        clean = clean / peak
        gain = np.float32(10 ** (random.uniform(-3.0, 3.0) / 20.0))
        clean = clean * gain
        noise = self._noise(CHUNK_SAMPLES)
        snr_db = random.uniform(*SNR_RANGE)
        cr = np.sqrt(np.mean(clean**2) + np.float32(1e-8))
        nr = np.sqrt(np.mean(noise**2) + np.float32(1e-8))
        noise = noise * (cr / np.float32(10**(snr_db/20))) / (nr + np.float32(1e-8))
        degraded = clean + noise
        dp = np.abs(degraded).max() + np.float32(1e-8)
        return {"clean": torch.from_numpy((clean / dp).astype(np.float32)),
                "degraded": torch.from_numpy((degraded / dp).astype(np.float32))}


def train(epochs=50, batch_size=64, lr=1e-4, steps_per_epoch=300, resume=None):
    device = torch.device("cuda")
    extractor = DFNFeatureExtractor(device)

    # ── Data: FMA-small + MUSDB18 ──
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    fma = _PROJECT / "data" / "fma_small" / "fma_small"
    corpus = _PROJECT / "corpus"

    # Collect all clean audio files
    all_files = []
    # FMA: 8000 MP3s
    if fma.is_dir():
        fma_files = sorted(fma.rglob("*.mp3"))
        print(f"FMA-small: {len(fma_files)} MP3s")
        all_files.extend(fma_files)
    # MUSDB18: 500 WAVs
    musdb_files = sorted(musdb.rglob("*.wav"))
    print(f"MUSDB18: {len(musdb_files)} WAVs")
    all_files.extend(musdb_files)
    # Corpus
    corpus_files = sorted(corpus.rglob("*.wav")) if corpus.is_dir() else []
    print(f"Corpus: {len(corpus_files)} WAVs")
    all_files.extend(corpus_files)

    # Noise files (for realistic noise mixing)
    noise_files = [f for f in corpus_files if "clean" not in f.stem.lower()]

    n_train = int(0.9 * len(all_files))
    rng = random.Random(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:]

    print(f"Total: {len(all_files)} → Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = AudioDenoiseDataset(train_files, noise_files)
    val_ds = AudioDenoiseDataset(val_files, noise_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, drop_last=True, prefetch_factor=2)

    # ── Model (PyTorch GPU directly) ──
    from df.config import config; config.use_defaults()
    from df.deepfilternet3 import init_model
    model = init_model().to(device)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: DeepFilterNet3 ({n_p:.2f}M) | Device: {device}")

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded from {resume} (epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?'):.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    print(f"Epochs: {epochs} | Batch: {batch_size} | Steps/ep: {steps_per_epoch} | LR: {lr}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch: break
            clean = batch["clean"].to(device)
            noisy = batch["degraded"].to(device)

            feb_n, fsp_n, spec_n = extractor(noisy)
            feb_c, fsp_c, spec_c = extractor(clean)

            optimizer.zero_grad()
            enh, _, _, _ = model.forward(spec=spec_n, feat_erb=feb_n, feat_spec=fsp_n)
            loss = F.mse_loss(enh, spec_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_loss += loss.item()
            if (step + 1) % 50 == 0:
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
                cv = vb["clean"].to(device)
                nv = vb["degraded"].to(device)
                feb_n2, fsp_n2, spec_n2 = extractor(nv)
                feb_c2, fsp_c2, spec_c2 = extractor(cv)
                enh2, _, _, _ = model.forward(spec=spec_n2, feat_erb=feb_n2, feat_spec=fsp_n2)
                val_loss += F.mse_loss(enh2, spec_c2).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        print(f"Ep {epoch+1:3d}/{epochs} | Tr {avg_train:.4f} | Val {avg_val:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.1e} | {time.time()-t0:.0f}s", flush=True)

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_train, "val_loss": avg_val}, LATEST_PT)
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1, "val_loss": avg_val}, BEST_PT)
            print(f"  >> Best: {best_val:.4f}")

    print(f"\nDone. Best val: {best_val:.4f} | {BEST_PT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Expanded DFN training with FMA+Musdb")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=300)
    p.add_argument("--resume", type=str, default="models/deepfilternet_v3_ii/finetuned/dfn_musik_best.pt")
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.steps_per_epoch, args.resume)
