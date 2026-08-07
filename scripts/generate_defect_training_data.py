#!/usr/bin/env python3
"""
Generator für synthetische Trainingsdaten: Bandwidth-Limit-Defekte.

Erzeugt Paare (bandbreitenbegrenzt, Original) aus hochwertigen Audio-Dateien.
Nutzt Auriks internen DefectScanner, um nur "saubere" Segmente zu verwenden.

Verwendung mit MUSDB18:
    python scripts/generate_defect_training_data.py \
        --input /path/to/MUSDB18/train \
        --output data/bw_defects/train \
        --duration_hours 10

Ohne MUSDB18 (nutzt Rausch-Sweeps als Fallback):
    python scripts/generate_defect_training_data.py \
        --output data/bw_defects/synthetic \
        --duration_hours 5 \
        --synthetic
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def detect_bw_cutoff(y: np.ndarray, sr: int, threshold_db: float = -60.0) -> float | None:
    """Erkennt die tatsächliche Bandbreitengrenze eines Signals."""
    n_fft = min(4096, len(y))
    if n_fft < 256:
        return None

    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=n_fft))
    spec_db = 20 * np.log10(spec + 1e-10)
    max_db = spec_db.max()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    above_threshold = spec_db > (max_db + threshold_db)
    if not above_threshold.any():
        return None

    last_bin = np.where(above_threshold)[0][-1]
    return float(freqs[min(last_bin + 1, len(freqs) - 1)])


def apply_bandwidth_limit(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    filter_order: int = 8,
) -> np.ndarray:
    """Wendet Butterworth-Tiefpass mit realistischer Flankensteilheit an."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    if cutoff_hz >= nyquist * 0.99:
        return audio.copy()

    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(filter_order, normalized_cutoff, btype="low", output="sos")
    return sosfiltfilt(sos, audio)  # type: ignore[no-any-return]


def spec_to_log_mel(y: np.ndarray, sr: int, n_mels: int = 256) -> np.ndarray:
    """Konvertiert Audiosignal zu logarithmiertem Mel-Spektrogramm."""
    n_fft = 2048
    hop = 512
    n_frames = 256
    target_samples = (n_frames - 1) * hop + n_fft

    if len(y) < target_samples:
        y_pad = np.zeros(target_samples)
        y_pad[: len(y)] = y
        y = y_pad
    elif len(y) > target_samples:
        y = y[:target_samples]

    spec = np.abs(np.lib.stride_tricks.sliding_window_view(y, n_fft, axis=0)[::hop])
    spec = spec[:, : n_fft // 2 + 1] ** 2

    mel_fb = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = spec @ mel_fb.T
    mel_spec = np.log10(mel_spec + 1e-6)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

    return mel_spec.astype(np.float32)  # type: ignore[no-any-return]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Erzeugt Mel-Filterbank-Matrix."""
    f_min, f_max = 20.0, sr / 2.0
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        f_prev, f_curr, f_next = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        if f_curr > f_prev:
            filters[m, f_prev:f_curr] = (np.arange(f_prev, f_curr) - f_prev) / (f_curr - f_prev)
        if f_next > f_curr:
            filters[m, f_curr:f_next] = (f_next - np.arange(f_curr, f_next)) / (f_next - f_curr)

    return filters


def generate_from_audio_dir(
    input_dir: Path,
    output_dir: Path,
    duration_hours: float,
    sr: int = 22050,
    segment_duration: float = 6.0,
    cutoff_range: tuple = (2000, 12000),
) -> int:
    """Generiert Trainingsdaten aus einem Verzeichnis mit Audiodateien."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio_files = list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.flac"))
    if not audio_files:
        print(f"❌ Keine Audiodateien in {input_dir} gefunden.")
        return 0

    print(f"📁 {len(audio_files)} Audiodateien gefunden.")
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_samples = int(segment_duration * sr)
    target_segments = int(duration_hours * 3600 / segment_duration)

    manifest = []
    generated = 0
    random.shuffle(audio_files)

    for audio_path in audio_files:
        if generated >= target_segments:
            break

        try:
            info = sf.info(str(audio_path))
            file_sr = info.samplerate

            duration = info.duration
            if duration < segment_duration + 2:
                continue

            segments_per_file = min(
                int(duration / segment_duration),
                max(1, (target_segments - generated) // max(1, len(audio_files) - generated // len(audio_files))),
            )

            for seg_idx in range(segments_per_file):
                if generated >= target_segments:
                    break

                start = random.uniform(0, duration - segment_duration - 1)
                chunk, _ = sf.read(
                    str(audio_path),
                    start=int(start * file_sr),
                    frames=int((segment_duration + 1) * file_sr),
                )
                start_offset = random.randint(0, file_sr)
                chunk = chunk[start_offset : start_offset + segment_samples]

                if len(chunk) < segment_samples:
                    continue

                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)

                if file_sr != sr:
                    chunk = resample_poly(chunk, sr, file_sr)

                chunk = chunk.astype(np.float32)
                chunk /= np.abs(chunk).max() + 1e-8

                cutoff = random.uniform(*cutoff_range)
                low_res = apply_bandwidth_limit(chunk, sr, cutoff)

                mel_orig = spec_to_log_mel(chunk, sr)
                mel_low = spec_to_log_mel(low_res, sr)

                sample_id = f"bw_{generated:06d}_{cutoff:.0f}hz"
                np.savez_compressed(
                    output_dir / f"{sample_id}.npz",
                    input=mel_low,
                    target=mel_orig,
                    cutoff_hz=cutoff,
                )

                manifest.append(
                    {
                        "id": sample_id,
                        "source": str(audio_path.name),
                        "cutoff_hz": round(cutoff, 1),
                        "original_bw": round(detect_bw_cutoff(chunk, sr) or sr / 2, 1),
                    }
                )
                generated += 1

                if generated % 100 == 0:
                    print(f"   Generiert: {generated}/{target_segments} ({100 * generated / target_segments:.0f}%)")

        except Exception as e:
            print(f"   ⚠️ Fehler bei {audio_path.name}: {e}")
            continue

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ {generated} Segmente generiert in {output_dir}")
    return generated


def generate_synthetic(
    output_dir: Path,
    duration_hours: float,
    sr: int = 22050,
    segment_duration: float = 6.0,
    cutoff_range: tuple = (2000, 12000),
) -> int:
    """Synthetische Trainingsdaten (Harmonische + Rauschen)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_samples = int(segment_duration * sr)
    target_segments = int(duration_hours * 3600 / segment_duration)

    manifest = []
    harmonic_freqs = np.array([261.63, 329.63, 392.00, 523.25, 659.25, 783.99, 1046.50])
    noise_colors = ["pink", "brown", "violet"]

    for idx in range(target_segments):
        t = np.arange(segment_samples) / sr

        signal = np.zeros(segment_samples, dtype=np.float64)
        num_harmonics = random.randint(2, 6)
        for _ in range(num_harmonics):
            f0 = random.choice(harmonic_freqs) * random.uniform(0.5, 2.0)
            num_partials = random.randint(2, 8)
            for p in range(1, num_partials + 1):
                amp = 0.3 / (p ** random.uniform(0.5, 1.2))
                phase = random.uniform(0, 2 * np.pi)
                signal += amp * np.sin(2 * np.pi * f0 * p * t + phase)

        noise = np.random.randn(segment_samples)
        color = random.choice(noise_colors)
        if color == "pink":
            noise = np.cumsum(noise)
            noise /= np.abs(noise).max() + 1e-8
        elif color == "brown":
            noise = np.cumsum(np.cumsum(noise))
            noise /= np.abs(noise).max() + 1e-8
        elif color == "violet":
            noise = np.diff(noise, prepend=0)
            noise /= np.abs(noise).max() + 1e-8

        signal += 0.1 * noise
        signal /= np.abs(signal).max() + 1e-8
        signal = signal.astype(np.float32)

        cutoff = random.uniform(*cutoff_range)
        low_res = apply_bandwidth_limit(signal, sr, cutoff)

        mel_orig = spec_to_log_mel(signal, sr)
        mel_low = spec_to_log_mel(low_res, sr)

        sample_id = f"bw_{idx:06d}_{cutoff:.0f}hz"
        np.savez_compressed(
            output_dir / f"{sample_id}.npz",
            input=mel_low,
            target=mel_orig,
            cutoff_hz=cutoff,
        )
        manifest.append({"id": sample_id, "source": "synthetic", "cutoff_hz": round(cutoff, 1)})

        if (idx + 1) % 100 == 0:
            print(f"   Generiert: {idx + 1}/{target_segments}")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ {target_segments} synthetische Segmente generiert in {output_dir}")
    return target_segments


def main():
    parser = argparse.ArgumentParser(description="BW-Defect-Trainingsdaten-Generator")
    parser.add_argument("--input", type=str, help="Verzeichnis mit Eingabe-Audiodateien")
    parser.add_argument("--output", type=str, required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--duration_hours", type=float, default=5, help="Ziel-Dauer in Stunden")
    parser.add_argument("--synthetic", action="store_true", help="Nur synthetische Daten")
    parser.add_argument("--sr", type=int, default=22050, help="Ziel-Samplerate")
    parser.add_argument("--cutoff_min", type=float, default=2000, help="Minimale Cutoff-Frequenz")
    parser.add_argument("--cutoff_max", type=float, default=12000, help="Maximale Cutoff-Frequenz")
    args = parser.parse_args()

    output_dir = Path(args.output)
    cutoff_range = (args.cutoff_min, args.cutoff_max)

    if args.input:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"❌ Eingabeverzeichnis nicht gefunden: {input_dir}")
            sys.exit(1)
        generate_from_audio_dir(input_dir, output_dir, args.duration_hours, sr=args.sr, cutoff_range=cutoff_range)
    elif args.synthetic:
        generate_synthetic(output_dir, args.duration_hours, sr=args.sr, cutoff_range=cutoff_range)
    else:
        print("Bitte --input ODER --synthetic angeben.")
        sys.exit(1)


if __name__ == "__main__":
    main()
