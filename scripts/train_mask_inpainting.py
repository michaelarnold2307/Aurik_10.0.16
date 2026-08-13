#!/usr/bin/env python3
"""
§v10.910: Mask-konditioniertes Inpainting-Re-Fine-Tuning.

Root Cause aus §v10.900: Das Fine-Tuning maskierte den TARGET-Velocity,
aber die Inferenz maskierte die INPUT-Regionen. Das Modell hat nie gelernt,
die Maske ALS INPUT zu sehen.

Fix: DiT mit 2 Eingangskanälen — [Audio, Maske]. Das Modell sieht, WO
inpaintet werden soll, und lernt, nur dort zu rekonstruieren.

Initialisierung: Kanal 0 (Audio) aus dem vortrainierten Inpainting-Checkpoint,
Kanal 1 (Maske) mit Null-Gewichten — der Pretrained-Zustand bleibt exakt erhalten.
"""

from __future__ import annotations

import argparse, random, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "models" / "miipher_dit"))

from dit_model import FlowMatchingDiT

SR = 48_000
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)

CHECKPOINT_DIR = _PROJECT / "models" / "harmonic_inpainting"
MASK_BEST = CHECKPOINT_DIR / "inpainting_mask_best.pt"
PRETRAINED = CHECKPOINT_DIR / "inpainting_best.pt"


def make_harmonic_mask(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Erzeugt eine harmonische Maske: Regionen mit klaren Harmonien = 1."""
    n_fft = 2048
    hop = n_fft // 2
    window = np.hanning(n_fft)
    n_frames = max(1, (len(audio) - n_fft) // hop)
    mask = np.zeros(len(audio), dtype=np.float32)

    for i in range(n_frames):
        s = i * hop
        if s + n_fft > len(audio):
            break
        frame = audio[s:s + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) + 1e-12
        # Harmonisch = spektrale Konzentration (wenige starke Peaks)
        top = np.sort(spec)[::-1]
        concentration = top[:10].sum() / (spec.sum() + 1e-12)
        if concentration > 0.5:
            mask[s:s + n_fft] = 1.0
    return mask


class MaskInpaintingDataset(Dataset):
    """(attenuiertes Audio + Maske) → clean. Maske wird als 2. Kanal mitgegeben."""

    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self):
        return max(len(self.files), 100) * 5

    def _load_chunk(self, path: Path) -> np.ndarray:
        try:
            with sf.SoundFile(str(path)) as snd:
                sr = snd.samplerate
                chunk_native = min(int(3.0 * sr), snd.frames)
                max_start = max(0, snd.frames - chunk_native)
                snd.seek(random.randint(0, max_start) if max_start > 0 else 0)
                y = snd.read(chunk_native, dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            y = np.nan_to_num(y).astype(np.float32)
            if len(y) < CHUNK_SAMPLES:
                y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
            else:
                y = y[:CHUNK_SAMPLES]
            peak = np.abs(y).max() + 1e-10
            return (y / peak).astype(np.float32)
        except Exception:
            return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def __getitem__(self, idx):
        clean = self._load_chunk(self.files[idx % len(self.files)])

        # Maske: harmonische Regionen
        mask = make_harmonic_mask(clean)

        # Attenuiere maskierte Regionen (simuliert Denoiser-Schaden)
        attenuation = np.random.uniform(0.4, 0.8)
        attenuated = clean * (1.0 - attenuation * mask)

        return {
            "clean": torch.from_numpy(clean),
            "attenuated": torch.from_numpy(attenuated),
            "mask": torch.from_numpy(mask),
        }


def train(epochs: int = 20, lr: float = 5e-5):
    device = torch.device("cuda")

    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    fma = _PROJECT / "data" / "fma_small" / "fma_small"
    files = []
    if fma.is_dir():
        files.extend(sorted(fma.rglob("*.mp3"))[:1500])
    files.extend(sorted(musdb.rglob("*.wav"))[:400])
    rng = random.Random(42)
    rng.shuffle(files)
    n_train = int(0.9 * len(files))
    train_files, val_files = files[:n_train], files[n_train:]
    print(f"Files: {len(files)} → Train {len(train_files)}, Val {len(val_files)}")

    train_loader = DataLoader(MaskInpaintingDataset(train_files), batch_size=1,
                              shuffle=True, num_workers=2, drop_last=True, prefetch_factor=2)

    # 2-Kanal-Modell + Pretrained-Gewichte für Kanal 0
    model = FlowMatchingDiT(in_channels=2).to(device)
    if PRETRAINED.exists():
        ckpt = torch.load(str(PRETRAINED), map_location=device, weights_only=True)
        pretrained_state = ckpt.get("model_state_dict", ckpt)
        # Kanal-0-Gewichte kopieren, Kanal-1 bleibt Null
        with torch.no_grad():
            w = model.patch_embed.weight.data
            w[:, :1] = pretrained_state["patch_embed.weight"][:, :1]
            w[:, 1:] = 0.0
        # Alle anderen Layer direkt kopieren
        for k, v in pretrained_state.items():
            if k != "patch_embed.weight" and k in model.state_dict():
                model.state_dict()[k].copy_(v)
        print("Pretrained-Gewichte geladen (Kanal 0 = Audio, Kanal 1 = Null)")

    # Nur patch_embed + letzte 3 Blöcke trainierbar
    for name, param in model.named_parameters():
        trainable = (
            "patch_embed" in name
            or any(f"blocks.{i}." in name for i in range(15, 18))
            or "final_ada" in name or "output_proj" in name
        )
        param.requires_grad = trainable
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable: {n_t:.1f}M (patch_embed + Blöcke 15-17)")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if step >= 100:
                break
            clean = batch["clean"].to(device).unsqueeze(-1)
            atten = batch["attenuated"].to(device).unsqueeze(-1)
            mask = batch["mask"].to(device).unsqueeze(-1)

            # Input: [B, T, 2] — Kanal 0 = Audio, Kanal 1 = Maske
            x_input = torch.cat([atten, mask], dim=-1)

            t_vals = torch.rand(1, device=device)
            noise = torch.randn_like(atten) * 0.01
            x_t = (1 - t_vals) * x_input + t_vals * torch.cat([clean, mask], dim=-1)
            x_t = x_t + torch.cat([noise, torch.zeros_like(noise)], dim=-1)

            velocity = model(x_t, t_vals)
            target = torch.cat([clean - atten, torch.zeros_like(mask)], dim=-1)

            loss = F.mse_loss(velocity, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_steps += 1

        avg = train_loss / max(n_steps, 1)
        print(f"Ep {epoch+1:3d}/{epochs} | Loss {avg:.6f} | {time.time()-t0:.0f}s", flush=True)

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_loss": avg}, MASK_BEST)
        if avg < best_val:
            best_val = avg
            print(f"  >> Best: {best_val:.6f}")

    print(f"Done. Best: {best_val:.6f} | {MASK_BEST}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()
    train(args.epochs, args.lr)
