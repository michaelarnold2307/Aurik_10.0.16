#!/usr/bin/env python3
"""
Train FlowMatchingDiT for vocal enhancement (§v10.14).

Training paradigm:
  Given clean vocal y and degraded vocal x_noisy (SNR 5–12 dB):
    t       ~ U(0, 1)                          — flow time
    x_t     = (1-t) * x_noisy + t * y          — interpolated sample
    v       = y - x_noisy                      — OT velocity field
    model(x_t, t) → v̂ ≈ v                       — supervised regression

  At inference (t=0.5):  ŷ = x_0.5 + (1-0.5) * v̂ = x_0.5 + 0.5 * v̂

Data sources:
  - MUSDB18-HQ vocals  (100 train, 50 test) @ 44.1kHz stereo → mono 48kHz
  - Corpus clean files (24 reference tracks)
  - Synthetic + environmental noise at SNR 5–12 dB

Architecture: FlowMatchingDiT (18L, 768-dim, 12 heads, ~201M params)
GPU memory:  ~24 GB at batch=8 with gradient accumulation ×4
Training:    ~48–72h on RTX 4090 (500 epochs, 200 steps/epoch)

Usage:
    python scripts/train_miipher_dit.py [--epochs 500] [--batch-size 8] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import librosa

# ═══════════════════════════════════════════════════════════════════════════
# Project imports
# ═══════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.miipher_dit.dit_model import create_miipher_dit, FlowMatchingDiTExportWrapper


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SR = 48000                              # Model native sample rate
CHUNK_SEC = 4.0                         # Training chunk duration
CHUNK_SAMPLES = int(CHUNK_SEC * SR)     # 192,000 samples per chunk
PATCH_SIZE = 256                        # DiT patch size (must match model)
SNR_RANGE_DB = (5.0, 12.0)             # SNR range for degradation
GRADIENT_ACCUM_STEPS = 4               # Effective batch = B * accum
CHECKPOINT_DIR = Path("models/miipher_dit")
ONNX_OUTPUT = CHECKPOINT_DIR / "flow_matching_dit.onnx"
BEST_PT = CHECKPOINT_DIR / "checkpoint_best.pt"
LATEST_PT = CHECKPOINT_DIR / "checkpoint_latest.pt"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def _collect_vocal_files(musdb_root: Path) -> list[Path]:
    """Collect all vocal.wav files from MUSDB18-HQ train set."""
    vocals = []
    train_dir = musdb_root / "train"
    if train_dir.is_dir():
        for track_dir in sorted(train_dir.iterdir()):
            if track_dir.is_dir():
                vf = track_dir / "vocals.wav"
                if vf.is_file():
                    vocals.append(vf)
    return vocals


def _collect_corpus_clean(corpus_root: Path) -> list[Path]:
    """Collect clean reference files from corpus."""
    clean_files = []
    if corpus_root.is_dir():
        for f in sorted(corpus_root.rglob("*clean*.wav")):
            if f.is_file():
                clean_files.append(f)
    return clean_files


class MiipherDiTDataset(Dataset):
    """Streaming dataset: loads, resamples, and mixes on-the-fly.

    Each __getitem__ returns:
        y_clean:  [1, CHUNK_SAMPLES] — clean vocal waveform at 48kHz
        x_noisy:  [1, CHUNK_SAMPLES] — degraded at SNR 5–12 dB
    """

    def __init__(
        self,
        vocal_files: list[Path],
        noise_files: list[Path],
        augment_noise: bool = True,
    ):
        self.vocal_files = vocal_files
        self.noise_files = noise_files
        self.augment_noise = augment_noise

    def __len__(self) -> int:
        return len(self.vocal_files) * 50  # Oversample each vocal 50× per epoch

    def _load_audio(self, path: Path, target_sr: int = SR) -> np.ndarray:
        """Load audio, convert to mono, resample to target_sr."""
        y, sr = librosa.load(str(path), sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y.astype(np.float32)

    def _random_chunk(self, audio: np.ndarray, length: int) -> np.ndarray:
        """Extract a random chunk of `length` samples, pad if too short."""
        if len(audio) < length:
            # Pad with reflection if too short
            pad = length - len(audio)
            audio = np.pad(audio, (0, pad), mode="reflect")
        start = random.randint(0, len(audio) - length)
        return audio[start : start + length]

    def _generate_noise(self, length: int) -> np.ndarray:
        """Generate noise: either from files or synthetic (white/pink/brown)."""
        if self.noise_files and random.random() < 0.7:
            # Load noise from file
            nf = random.choice(self.noise_files)
            try:
                noise = self._load_audio(nf)
                noise = self._random_chunk(noise, length)
                # Normalise noise
                noise = noise / (np.abs(noise).max() + 1e-8)
                return noise
            except Exception:
                pass  # Fall through to synthetic

        # Synthetic noise
        noise = np.random.randn(length).astype(np.float32)
        if self.augment_noise:
            color = random.choice(["white", "pink", "brown"])
            if color == "pink":
                noise = np.cumsum(noise)
            elif color == "brown":
                noise = np.cumsum(np.cumsum(noise))
        noise = noise / (np.abs(noise).max() + 1e-8)
        return noise

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load clean vocal
        vf = self.vocal_files[idx % len(self.vocal_files)]
        clean = self._load_audio(vf)
        clean = self._random_chunk(clean, CHUNK_SAMPLES)

        # Peak-normalise clean
        peak = np.abs(clean).max() + 1e-8
        clean = clean / peak

        # Generate noise
        noise = self._generate_noise(CHUNK_SAMPLES)

        # Mix at random SNR
        snr_db = random.uniform(*SNR_RANGE_DB)
        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-8)
        noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-8)
        target_noise_rms = clean_rms / (10 ** (snr_db / 20))
        noise = noise * (target_noise_rms / (noise_rms + 1e-8))
        degraded = clean + noise

        # Normalise degraded to [-1, 1]
        degraded_peak = np.abs(degraded).max() + 1e-8
        degraded = degraded / degraded_peak
        # Apply same scaling to clean to preserve relationship
        clean = clean / degraded_peak

        return {
            "clean": torch.from_numpy(clean).unsqueeze(0),       # [1, T]
            "degraded": torch.from_numpy(degraded).unsqueeze(0), # [1, T]
        }


def _find_noise_files(corpus_root: Path) -> list[Path]:
    """Find noise files from corpus damaged dirs and Musan if available."""
    noise_files: list[Path] = []

    # Use corpus damaged files as noise sources
    if corpus_root.is_dir():
        for dmg in corpus_root.rglob("damaged"):
            if dmg.is_dir():
                for f in dmg.rglob("*.wav"):
                    if f.is_file() and "clean" not in f.stem.lower():
                        noise_files.append(f)

    # Check for Musan
    musan_dir = Path("data/noise/musan")
    if musan_dir.is_dir():
        for f in musan_dir.rglob("*.wav"):
            if f.is_file():
                noise_files.append(f)

    return noise_files


# ═══════════════════════════════════════════════════════════════════════════
# Flow Matching Loss
# ═══════════════════════════════════════════════════════════════════════════

def flow_matching_loss(
    model: nn.Module,
    clean: torch.Tensor,       # [B, 1, T]
    degraded: torch.Tensor,    # [B, 1, T]
    device: torch.device,
) -> torch.Tensor:
    """Flow matching training step.

    Samples t ~ U(0,1), builds x_t = (1-t)*degraded + t*clean,
    and trains model to predict velocity v = clean - degraded.
    """
    B = clean.shape[0]
    T = clean.shape[2]

    # Sample flow times
    t = torch.rand(B, device=device)  # [B]

    # Interpolate: x_t = (1-t)*x_0 + t*x_1
    # Where x_0 = degraded, x_1 = clean
    t_expanded = t.view(B, 1, 1)
    x_t = (1 - t_expanded) * degraded + t_expanded * clean  # [B, 1, T]

    # Target velocity field (OT path)
    v_target = clean - degraded  # [B, 1, T]

    # Model prediction
    # DiT expects [B, T, 1] input
    x_t_transposed = x_t.transpose(1, 2)  # [B, T, 1]
    v_pred = model(x_t_transposed, t)     # [B, T, 1]
    v_pred = v_pred.transpose(1, 2)       # [B, 1, T]

    # L1 loss on velocity field
    loss = F.l1_loss(v_pred, v_target)

    return loss


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Resolution STFT Loss (optional augmentation)
# ═══════════════════════════════════════════════════════════════════════════

def mr_stft_loss(pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss for waveform quality."""
    loss = torch.tensor(0.0, device=pred_wave.device)
    count = 0
    for n_fft in [512, 1024, 2048]:
        hop = n_fft // 4
        if hop < 1:
            continue
        window = torch.hann_window(n_fft, device=pred_wave.device)
        pred_spec = torch.stft(
            pred_wave.squeeze(1), n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True
        )
        target_spec = torch.stft(
            target_wave.squeeze(1), n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True
        )
        sc = (pred_spec - target_spec).abs().pow(2).sum() / (target_spec.abs().pow(2).sum() + 1e-8)
        mag = F.l1_loss(pred_spec.abs(), target_spec.abs())
        loss = loss + sc + mag
        count += 1
    return loss / max(count, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(
    epochs: int = 500,
    batch_size: int = 8,
    lr: float = 1e-4,
    steps_per_epoch: int = 200,
    use_stft_loss: bool = True,
    stft_weight: float = 0.1,
    resume: Optional[str] = None,
):
    """Main training loop for FlowMatchingDiT."""

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        scaler = torch.amp.GradScaler("cuda")
        use_amp = True
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected — training on CPU will be extremely slow!")
        scaler = None
        use_amp = False

    # Data
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    musdb_root = PROJECT_ROOT / "data" / "musdb18hq"
    corpus_root = PROJECT_ROOT / "corpus"

    vocal_files = _collect_vocal_files(musdb_root)
    noise_files = _find_noise_files(corpus_root)

    print(f"Vocal files (MUSDB18): {len(vocal_files)}")
    print(f"Noise files:           {len(noise_files)}")

    if len(vocal_files) == 0:
        print("ERROR: No vocal files found. Place MUSDB18-HQ in data/musdb18hq/")
        print("  Expected: data/musdb18hq/train/<track>/vocals.wav")
        return

    full_dataset = MiipherDiTDataset(vocal_files, noise_files)

    # 80/20 train/val split
    n_total = len(full_dataset.vocal_files)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    # Split at file level (not sample level)
    train_files = full_dataset.vocal_files[:n_train]
    val_files = full_dataset.vocal_files[n_train:]

    train_dataset = MiipherDiTDataset(train_files, noise_files)
    val_dataset = MiipherDiTDataset(val_files, noise_files)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )

    effective_batch = batch_size * GRADIENT_ACCUM_STEPS
    print(f"Batch size: {batch_size} × {GRADIENT_ACCUM_STEPS} = {effective_batch} effective")
    print(f"Train files: {n_train}, Val files: {n_val}")

    # Model
    model = create_miipher_dit(dropout=0.1)  # Dropout for training
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: FlowMatchingDiT ({n_params:.1f}M params)")

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # Create output directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training: {epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"LR: {lr} → {scheduler.eta_min} (CosineAnnealingWarmRestarts)")
    print(f"Chunk: {CHUNK_SAMPLES} samples ({CHUNK_SEC}s @ {SR}Hz)")
    print(f"SNR range: {SNR_RANGE_DB[0]}–{SNR_RANGE_DB[1]} dB")
    print(f"AMP: {use_amp}, STFT loss: {use_stft_loss} (×{stft_weight})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break

            clean = batch["clean"].to(device)          # [B, 1, T]
            degraded = batch["degraded"].to(device)    # [B, 1, T]

            # Flow matching loss
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    loss_fm = flow_matching_loss(model, clean, degraded, device)

                    if use_stft_loss:
                        # Compute enhanced output for STFT loss
                        t_infer = torch.full((clean.shape[0],), 0.5, device=device)
                        x_05 = 0.5 * degraded + 0.5 * clean
                        with torch.no_grad():
                            v_pred = model(x_05.transpose(1, 2), t_infer).transpose(1, 2)
                        enhanced = x_05 + 0.5 * v_pred
                        loss_stft = mr_stft_loss(enhanced, clean)
                        loss = loss_fm + stft_weight * loss_stft
                    else:
                        loss = loss_fm

                loss = loss / GRADIENT_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                loss_fm = flow_matching_loss(model, clean, degraded, device)
                if use_stft_loss:
                    t_infer = torch.full((clean.shape[0],), 0.5, device=device)
                    x_05 = 0.5 * degraded + 0.5 * clean
                    with torch.no_grad():
                        v_pred = model(x_05.transpose(1, 2), t_infer).transpose(1, 2)
                    enhanced = x_05 + 0.5 * v_pred
                    loss_stft = mr_stft_loss(enhanced, clean)
                    loss = loss_fm + stft_weight * loss_stft
                else:
                    loss = loss_fm
                loss = loss / GRADIENT_ACCUM_STEPS
                loss.backward()

            train_loss += loss.item() * GRADIENT_ACCUM_STEPS

            if (step + 1) % GRADIENT_ACCUM_STEPS == 0 or (step + 1) >= steps_per_epoch:
                if use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                optimizer.zero_grad()

            # Progress
            if (step + 1) % 20 == 0:
                elapsed = time.time() - t0
                steps_done = step + 1
                eta = elapsed / steps_done * (steps_per_epoch - steps_done) if steps_done > 0 else 0
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | Step {step+1:3d}/{steps_per_epoch} | "
                    f"Loss {train_loss/(step+1):.4f} | {elapsed:.0f}s elapsed, {eta:.0f}s ETA",
                    flush=True,
                )

        scheduler.step()
        avg_train_loss = train_loss / min(steps_per_epoch, len(train_loader))
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_steps >= 20:  # Limit validation to 20 batches
                    break
                clean_v = val_batch["clean"].to(device)
                degraded_v = val_batch["degraded"].to(device)
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        loss_v = flow_matching_loss(model, clean_v, degraded_v, device)
                else:
                    loss_v = flow_matching_loss(model, clean_v, degraded_v, device)
                val_loss += loss_v.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
            f"LR {current_lr:.1e} | {elapsed:.0f}s",
            flush=True,
        )

        # Save latest checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            LATEST_PT,
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                },
                BEST_PT,
            )
            print(f"  ✅ Best checkpoint saved (val_loss={best_val_loss:.4f})")

    # Final: export to ONNX
    print("\nExporting best model to ONNX...")
    _export_trained_to_onnx(str(BEST_PT), str(ONNX_OUTPUT), device)
    print(f"✅ Done. Best val_loss: {best_val_loss:.4f}")


def _export_trained_to_onnx(checkpoint_path: str, output_path: str, device: torch.device):
    """Export trained model to ONNX with dynamic axes."""
    try:
        import onnx
    except ImportError:
        print("⚠️  onnx not available — skipping export")
        return

    model = create_miipher_dit(dropout=0.0)  # No dropout for inference
    wrapper = FlowMatchingDiTExportWrapper(model)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    try:
        wrapper.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=True)

    wrapper.eval()
    wrapper.to("cpu")

    dummy_x = torch.randn(1, 48000, 1, dtype=torch.float32)
    dummy_t = torch.tensor([0.5], dtype=torch.float32)

    dynamic_axes = {
        "x": {0: "batch", 1: "time"},
        "t": {0: "batch"},
        "output": {0: "batch", 1: "time"},
    }

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_t),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x", "t"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"📦 ONNX exported: {output_path} ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FlowMatchingDiT for MIIPHER vocal enhancement (§v10.14)"
    )
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs (default: 500)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--steps-per-epoch", type=int, default=200, help="Steps per epoch (default: 200)")
    parser.add_argument("--no-stft-loss", action="store_true", help="Disable MR-STFT auxiliary loss")
    parser.add_argument("--stft-weight", type=float, default=0.1, help="STFT loss weight (default: 0.1)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_epoch=args.steps_per_epoch,
        use_stft_loss=not args.no_stft_loss,
        stft_weight=args.stft_weight,
        resume=args.resume,
    )
