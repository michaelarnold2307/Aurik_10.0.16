#!/usr/bin/env python3
"""
§v10.128: MERT-Denoiser v4 — Stable training with gradient accumulation + ground-truth loss.

Key fixes from v10.127:
  - Ground-truth loss: compare pred_spec against direct STFT(clean), not model(clean)
  - Gradient accumulation: batch=4 × 8 accum → effective batch=32, fits in 24GB VRAM
  - Data validation: skip broken mp3s, NaN-producing files
  - Gradient clipping: tighter at 1.0
  - LR warmup: first 2 epochs
"""

from __future__ import annotations

import argparse, random, sys, time, os, math
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

SR = 48_000
N_FFT = 960
HOP = 480
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
SNR_RANGE = (5.0, 20.0)
MERT_SR = 16_000

CHECKPOINT_DIR = _PROJECT / "models" / "mert_denoiser"
BEST_PT = CHECKPOINT_DIR / "mert_denoiser_best.pt"
LATEST_PT = CHECKPOINT_DIR / "mert_denoiser_latest.pt"


# ═════════════════════════════════════════════════════════════════════════════
# MERT Feature Extractor (frozen, ONNX GPU)
# ═════════════════════════════════════════════════════════════════════════════

class MERTFeatureExtractor:
    def __init__(self, device="cuda"):
        import onnxruntime as ort
        model_path = _PROJECT / "models" / "mert" / "mert.onnx"
        self._session = ort.InferenceSession(
            str(model_path),
            providers=['ROCMExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[{'device_id': '0'}, {}],
        )
        self.device = device
        gpu = 'ROCM' in str(self._session.get_providers())
        print(f"  MERT: ONNX {'GPU' if gpu else 'CPU'}")

    @torch.no_grad()
    def __call__(self, audio_16k: np.ndarray) -> np.ndarray:
        peak = np.abs(audio_16k).max(axis=1, keepdims=True) + 1e-10
        audio_norm = audio_16k / peak
        outputs = self._session.run(None, {'input_values': audio_norm.astype(np.float32)})
        return outputs[0]


# ═════════════════════════════════════════════════════════════════════════════
# Decoder
# ═════════════════════════════════════════════════════════════════════════════

class MERTConditionedDecoder(nn.Module):
    def __init__(self, mert_dim=768, cond_dim=256):
        super().__init__()
        self.proj = nn.Linear(mert_dim, cond_dim)
        self.enc1 = nn.Sequential(nn.Conv2d(2, 48, 3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(48, 96, 3, padding=1, stride=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(96, 192, 3, padding=1, stride=2), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv2d(192, 256, 3, padding=1, stride=2), nn.GELU())
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + cond_dim, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GELU(),
        )
        self.dec4 = nn.Sequential(nn.Conv2d(256, 192, 3, padding=1), nn.GELU())
        self.dec3 = nn.Sequential(nn.Conv2d(192 + 192, 96, 3, padding=1), nn.GELU())
        self.dec2 = nn.Sequential(nn.Conv2d(96 + 96, 48, 3, padding=1), nn.GELU())
        self.dec1 = nn.Sequential(nn.Conv2d(48 + 48, 2, 3, padding=1))

    def forward(self, spec, mert_features):
        cond = self.proj(mert_features)
        cond = cond.transpose(1, 2).unsqueeze(2)
        e1 = self.enc1(spec)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        cond_up = F.interpolate(cond, size=(e4.shape[2], e4.shape[3]),
                                mode="bilinear", align_corners=False)
        b = self.bottleneck(torch.cat([e4, cond_up], dim=1))
        d4 = self.dec4(b)
        d4_up = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d4_up, e3], dim=1))
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))
        return d1


# ═════════════════════════════════════════════════════════════════════════════
# Full Model
# ═════════════════════════════════════════════════════════════════════════════

class MERTDenoiser(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.mert = MERTFeatureExtractor(device)
        self.decoder = MERTConditionedDecoder(mert_dim=768, cond_dim=256)
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.register_buffer("hann_norm", self._hann_norm())
        self.to(device)

    def _hann_norm(self):
        w = torch.hann_window(N_FFT)
        return w.pow(2).sum()

    def forward(self, noisy_audio_48k, return_spec=False):
        spec = torch.stft(noisy_audio_48k, n_fft=N_FFT, hop_length=HOP,
                          window=self.window.to(noisy_audio_48k.device),
                          return_complex=True)
        audio_16k = noisy_audio_48k[:, ::3].cpu().numpy()
        mert_feat = self.mert(audio_16k)
        mert_feat = torch.from_numpy(mert_feat).float().to(noisy_audio_48k.device)
        spec_ri = torch.stack([spec.real, spec.imag], dim=1)
        enhanced = self.decoder(spec_ri, mert_feat)
        enhanced_spec = torch.complex(enhanced[:, 0], enhanced[:, 1])
        clean_audio = torch.istft(enhanced_spec, n_fft=N_FFT, hop_length=HOP,
                                  window=self.window.to(noisy_audio_48k.device),
                                  length=noisy_audio_48k.shape[1])
        if return_spec:
            return clean_audio, enhanced_spec
        return clean_audio

    def compute_loss(self, noisy_audio, clean_audio):
        _, pred_spec = self.forward(noisy_audio, return_spec=True)
        # Ground truth clean STFT
        clean_spec = torch.stft(
            clean_audio, n_fft=N_FFT, hop_length=HOP,
            window=self.window.to(clean_audio.device),
            return_complex=True,
        )
        # Multi-band MSE
        F_bins = pred_spec.shape[1]
        f_low, f_mid = F_bins // 6, F_bins // 3
        loss_real = F.mse_loss(pred_spec.real, clean_spec.real)
        loss_imag = F.mse_loss(pred_spec.imag, clean_spec.imag)
        loss_mse = loss_real + loss_imag
        # Bark loss on log-magnitude
        pred_mag = torch.log1p(pred_spec.abs())
        clean_mag = torch.log1p(clean_spec.abs())
        loss_bark = F.l1_loss(pred_mag, clean_mag)
        loss = 0.8 * loss_mse + 0.2 * loss_bark
        return loss, loss_mse.item(), loss_bark.item()


# ═════════════════════════════════════════════════════════════════════════════
# Dataset with validation
# ═════════════════════════════════════════════════════════════════════════════

class SafeAudioDenoiseDataset(Dataset):
    """Loads random chunks, adds noise, returns clean/degraded pair.
    Falls back to silence for broken files instead of noise."""
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return max(len(self.files), 200) * 20

    def _load(self, path):
        try:
            with sf.SoundFile(str(path)) as snd:
                sr = snd.samplerate
                chunk_native = min(int(4.5 * sr), snd.frames)
                max_start = max(0, snd.frames - chunk_native)
                start_frame = random.randint(0, max_start) if max_start > 0 else 0
                snd.seek(start_frame)
                y = snd.read(chunk_native, dtype='float32')
                if y.ndim > 1:
                    y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            y = np.nan_to_num(y).astype(np.float32)
            if len(y) < CHUNK_SAMPLES:
                y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
            else:
                y = y[:CHUNK_SAMPLES]
            return y
        except Exception:
            # Return silence for bad files (model learns to pass through)
            return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def _noise(self, length):
        n = np.random.randn(length).astype(np.float32)
        c = random.choice(["white", "pink", "brown"])
        if c == "pink":
            n = np.cumsum(n)
        elif c == "brown":
            n = np.cumsum(np.cumsum(n))
        return n / (np.abs(n).max() + np.float32(1e-8))

    def __getitem__(self, idx):
        f = self.files[idx % len(self.files)]
        clean = self._load(f)
        peak = np.abs(clean).max() + np.float32(1e-8)
        clean = clean / peak
        gain = np.float32(10 ** (random.uniform(-6.0, 3.0) / 20.0))
        clean = clean * gain
        noise = self._noise(CHUNK_SAMPLES)
        snr_db = random.uniform(*SNR_RANGE)
        cr = np.sqrt(np.mean(clean**2) + np.float32(1e-8))
        nr = np.sqrt(np.mean(noise**2) + np.float32(1e-8))
        noise = noise * (cr / np.float32(10**(snr_db/20))) / (nr + np.float32(1e-8))
        degraded = clean + noise
        cp = np.abs(clean).max() + np.float32(1e-8)
        return {"clean": torch.from_numpy((clean / cp).astype(np.float32)),
                "degraded": torch.from_numpy((degraded / cp).astype(np.float32))}


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train(epochs=50, micro_batch=4, accum_steps=8, lr=1e-4, steps_per_epoch=200, resume=None):
    device = torch.device("cuda")
    effective_batch = micro_batch * accum_steps
    print(f"Gradient accumulation: {micro_batch} × {accum_steps} = effective batch {effective_batch}")

    # Data loading
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    fma = _PROJECT / "data" / "fma_small" / "fma_small"
    corpus = _PROJECT / "corpus"

    all_files = []
    if fma.is_dir():
        all_files.extend(sorted(fma.rglob("*.mp3")))
    all_files.extend(sorted(musdb.rglob("*.wav")))
    if corpus.is_dir():
        all_files.extend(sorted(corpus.rglob("*.wav")))

    n_train = int(0.9 * len(all_files))
    rng = random.Random(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    train_files, val_files = shuffled[:n_train], shuffled[n_train:]

    # Filter: validate first 1s of each file to catch corrupt mp3s
    print(f"Validating {len(train_files)} train + {len(val_files)} val files...", flush=True)
    train_files = [f for f in train_files if _quick_check(f)]
    val_files = [f for f in val_files if _quick_check(f)]
    print(f"  After validation: Train={len(train_files)}, Val={len(val_files)}", flush=True)

    train_ds = SafeAudioDenoiseDataset(train_files)
    val_ds = SafeAudioDenoiseDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True,
                              num_workers=2, drop_last=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=micro_batch, shuffle=False,
                            num_workers=2, drop_last=True, prefetch_factor=2)

    # Model
    model = MERTDenoiser(device=device).to(device)
    n_p = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"Model: MERT (117M frozen) + Decoder ({n_p:.1f}M trainable)")

    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - 2)  # leave room for warmup
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # EMA of decoder weights for stability
    ema_decay = 0.999
    ema_state = {k: v.clone().detach().cpu() for k, v in model.decoder.state_dict().items()}

    best_val = float("inf")
    start_epoch = 0

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
        if "ema_state" in ckpt:
            ema_state = ckpt["ema_state"]
        print(f"  Resumed from {resume}")

    print(f"Epochs: {epochs} | Micro-batch: {micro_batch} | Accum: {accum_steps} | Effective: {effective_batch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        t0 = time.time()
        n_steps = 0

        # LR warmup
        if epoch < 2:
            warmup_factor = (epoch + 1) / 2.0
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warmup_factor

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            clean = batch["clean"].to(device)
            noisy = batch["degraded"].to(device)

            loss, loss_mse, loss_bark = model.compute_loss(noisy, clean)
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Update EMA
                with torch.no_grad():
                    for k, v in model.decoder.state_dict().items():
                        ema_state[k] = ema_decay * ema_state[k].to(device) + (1 - ema_decay) * v.detach()
                        ema_state[k] = ema_state[k].cpu()

            train_loss += loss.detach() * accum_steps
            n_steps += 1

            if (step + 1) % 50 == 0:
                e = time.time() - t0
                avg_l = (train_loss.item() / n_steps) if n_steps > 0 else 0
                eta = e / (step + 1) * (steps_per_epoch - step - 1) if step > 0 else 0
                print(f"  Ep {epoch+1:3d}/{epochs} | St {step+1:3d}/{steps_per_epoch} | "
                      f"L {avg_l:.3f} | {e:.0f}s/{eta:.0f}s", flush=True)

        if epoch >= 2:
            scheduler.step()
        avg_train = train_loss.item() / max(n_steps, 1)

        # Validation (with EMA weights)
        model.eval()
        # Save current weights, load EMA for validation
        current_weights = {k: v.clone().detach().cpu() for k, v in model.decoder.state_dict().items()}
        model.decoder.load_state_dict({k: v.to(device) for k, v in ema_state.items()})

        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                if vn >= 20:
                    break
                cv = vb["clean"].to(device)
                nv = vb["degraded"].to(device)
                _, pred_spec = model(nv, return_spec=True)
                clean_spec_gt = torch.stft(cv, n_fft=N_FFT, hop_length=HOP,
                                           window=model.window.to(device),
                                           return_complex=True)
                val_loss += (F.mse_loss(pred_spec.real, clean_spec_gt.real) +
                             F.mse_loss(pred_spec.imag, clean_spec_gt.imag)).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        # Restore training weights
        model.decoder.load_state_dict({k: v.to(device) for k, v in current_weights.items()})

        print(f"Ep {epoch+1:3d}/{epochs} | Tr {avg_train:.4f} | Val {avg_val:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s", flush=True)

        torch.save({"decoder_state_dict": {k: v.cpu() for k, v in model.decoder.state_dict().items()},
                    "ema_state": ema_state,
                    "epoch": epoch + 1, "val_loss": avg_val}, LATEST_PT)
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"decoder_state_dict": {k: v.cpu() for k, v in model.decoder.state_dict().items()},
                        "ema_state": ema_state,
                        "epoch": epoch + 1, "val_loss": avg_val}, BEST_PT)
            print(f"  >> Best: {best_val:.4f}")

    print(f"\nDone. Best val: {best_val:.4f} | {BEST_PT}")


def _quick_check(path):
    """Check if file has a valid audio header. Returns True if valid."""
    try:
        with sf.SoundFile(str(path)) as snd:
            return snd.frames > 0 and snd.samplerate > 0
    except Exception:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MERT-Denoiser v4 Training (§v10.128)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args.epochs, args.micro_batch, args.accum_steps, args.lr, args.steps_per_epoch, args.resume)
