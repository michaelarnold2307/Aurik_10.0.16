#!/usr/bin/env python3
"""
Prepare MUSDB18 music data for SGMSE+ fine-tuning (§v10.16).

Creates the directory structure expected by SpecsDataModule:
  data/sgmse_musik/train/{clean,noisy}/
  data/sgmse_musik/val/{clean,noisy}/

Each clean/noisy pair has identical filenames.
Clean: MUSDB18 stems (vocals, drums, bass, other), 48kHz mono.
Noisy: clean + additive noise at random SNR 5-20 dB.

Usage:
    python scripts/prepare_sgmse_musik_data.py \
        --musdb data/musdb18hq/train \
        --output data/sgmse_musik \
        --sr 48000 --chunk 4.0 --snr 5,20
"""

import argparse
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def prepare(
    musdb_dir: str,
    output_dir: str,
    sr: int = 48000,
    chunk_sec: float = 4.0,
    snr_range: tuple[float, float] = (5.0, 20.0),
    val_split: float = 0.2,
    samples_per_file: int = 10,
):
    musdb = Path(musdb_dir)
    out = Path(output_dir)
    chunk = int(chunk_sec * sr)

    # Collect all stem files
    audio_files = sorted(
        f
        for f in musdb.rglob("*.wav")
        if f.is_file() and any(s in f.stem for s in ["vocals", "drums", "bass", "other"])
    )
    if not audio_files:
        print(f"ERROR: No stem files found in {musdb}")
        return

    random.shuffle(audio_files)
    n_val = max(1, int(len(audio_files) * val_split))
    train_files = audio_files[n_val:]
    val_files = audio_files[:n_val]

    print(f"Files: {len(audio_files)} (train={len(train_files)}, val={len(val_files)})")
    print(f"Chunk: {chunk_sec}s @ {sr}Hz = {chunk} samples")
    print(f"SNR: {snr_range[0]}-{snr_range[1]} dB")
    print(f"Samples per file: {samples_per_file}")
    print(f"Output: {out}")

    for split, files in [("train", train_files), ("val", val_files)]:
        clean_dir = out / split / "clean"
        noisy_dir = out / split / "noisy"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        for fi, f in enumerate(files):
            try:
                y, orig_sr = librosa.load(str(f), sr=None, mono=True)
                if orig_sr != sr:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
                y = y.astype(np.float32)
            except Exception as e:
                print(f"  SKIP {f.name}: {e}")
                continue

            if len(y) < chunk:
                y = np.pad(y, (0, chunk - len(y)), mode="reflect")

            for si in range(samples_per_file):
                start = random.randint(0, max(0, len(y) - chunk))
                clean = y[start : start + chunk].copy()
                peak = np.abs(clean).max() + 1e-8
                clean = clean / peak

                # Generate noise
                noise = np.random.randn(chunk).astype(np.float32)
                c = random.choice(["white", "pink", "brown"])
                if c == "pink":
                    noise = np.cumsum(noise)
                elif c == "brown":
                    noise = np.cumsum(np.cumsum(noise))
                noise = noise / (np.abs(noise).max() + 1e-8)

                snr_db = random.uniform(*snr_range)
                cr = np.sqrt(np.mean(clean**2) + 1e-8)
                nr = np.sqrt(np.mean(noise**2) + 1e-8)
                noise = noise * (cr / (10 ** (snr_db / 20))) / (nr + 1e-8)
                degraded = clean + noise

                dpeak = np.abs(degraded).max() + 1e-8
                degraded = degraded / dpeak
                clean = clean / dpeak

                fname = f"{f.stem}_{si:03d}.wav"
                sf.write(str(clean_dir / fname), clean, sr)
                sf.write(str(noisy_dir / fname), degraded, sr)

        print(f"  {split}: {len(files)} stems × {samples_per_file} = {len(files) * samples_per_file} pairs")

    # Create dataset config
    (out / "dataset_info.txt").write_text(
        f"SGMSE+ Musik Fine-Tuning Dataset\n"
        f"Source: MUSDB18-HQ ({len(audio_files)} stems)\n"
        f"Sample Rate: {sr} Hz\n"
        f"Chunk: {chunk_sec}s ({chunk} samples)\n"
        f"SNR Range: {snr_range[0]}-{snr_range[1]} dB\n"
        f"Samples per file: {samples_per_file}\n"
        f"Train files: {len(train_files)} → {len(train_files) * samples_per_file} pairs\n"
        f"Val files: {len(val_files)} → {len(val_files) * samples_per_file} pairs\n"
    )
    print(f"\nDone: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SGMSE+ music training data")
    parser.add_argument("--musdb", type=str, default="data/musdb18hq/train")
    parser.add_argument("--output", type=str, default="data/sgmse_musik")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--chunk", type=float, default=4.0)
    parser.add_argument("--snr", type=str, default="5,20", help="SNR range as lo,hi")
    parser.add_argument("--samples-per-file", type=int, default=10)
    args = parser.parse_args()

    snr_lo, snr_hi = map(float, args.snr.split(","))
    prepare(args.musdb, args.output, args.sr, args.chunk, (snr_lo, snr_hi), samples_per_file=args.samples_per_file)
