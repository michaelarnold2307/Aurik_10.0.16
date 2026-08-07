#!/usr/bin/env python3
"""Quick Import-vs-Export guard verification for early-pipeline regressions.

Focus checks:
- End-tail silence lift (RMS delta in dB)
- Clipping risk increase (sample clip ratio and peak99)
- Brightness/harshness drift on loud frames (HF ratio delta)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

# Ensure workspace root is importable when script is launched via scripts/ path.
_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))

from backend.file_import import load_audio_file

logger = logging.getLogger("verify_import_export_guards")


_EXT_TO_MATERIAL: dict[str, str] = {
    ".wav": "tape",
    ".flac": "cd_digital",
    ".aiff": "tape",
    ".aif": "tape",
    ".mp3": "mp3_low",
    ".aac": "mp3_low",
    ".m4a": "cd_digital",
    ".ogg": "cd_digital",
    ".wma": "mp3_low",
}


_MATERIAL_THRESHOLDS: dict[str, dict[str, float]] = {
    # tail_delta_db_max, clip_delta_max, export_peak99_max, hf_delta_max
    "shellac": {"tail": 4.5, "clip": 0.0015, "peak99": 0.998, "hf": 0.12},
    "vinyl": {"tail": 3.5, "clip": 0.0012, "peak99": 0.997, "hf": 0.10},
    "reel_tape": {"tail": 3.0, "clip": 0.0010, "peak99": 0.996, "hf": 0.08},
    "tape": {"tail": 3.0, "clip": 0.0010, "peak99": 0.996, "hf": 0.08},
    "cassette": {"tail": 3.0, "clip": 0.0010, "peak99": 0.996, "hf": 0.09},
    "cd_digital": {"tail": 2.0, "clip": 0.0006, "peak99": 0.994, "hf": 0.06},
    "mp3_low": {"tail": 2.5, "clip": 0.0008, "peak99": 0.995, "hf": 0.07},
    "unknown": {"tail": 3.0, "clip": 0.0010, "peak99": 0.995, "hf": 0.08},
}


@dataclass
class SignalStats:
    sr: int
    duration_s: float
    tail_rms_dbfs: float
    peak99: float
    clip_ratio: float
    hf_ratio_loud: float


def _normalize_material(material: str | None) -> str:
    m = str(material or "unknown").strip().lower()
    aliases = {
        "reeltape": "reel_tape",
        "reel-tape": "reel_tape",
        "digital": "cd_digital",
        "cd": "cd_digital",
        "mp3": "mp3_low",
    }
    m = aliases.get(m, m)
    if m not in _MATERIAL_THRESHOLDS:
        return "unknown"
    return m


def _infer_material_from_path(path: Path) -> str:
    ext = os.path.splitext(str(path).lower())[1]
    return _EXT_TO_MATERIAL.get(ext, "unknown")


def _resolve_material(user_material: str, import_path: Path) -> str:
    if user_material and user_material.lower() != "auto":
        return _normalize_material(user_material)
    return _infer_material_from_path(import_path)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        return arr.reshape(-1)
    if arr.shape[1] == 2:
        return 0.5 * (arr[:, 0] + arr[:, 1])
    if arr.shape[0] == 2:
        return 0.5 * (arr[0, :] + arr[1, :])
    return np.mean(arr, axis=1)  # type: ignore[no-any-return]


def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    g = math.gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr // g)
    down = int(src_sr // g)
    return signal.resample_poly(audio.astype(np.float32), up, down).astype(np.float32)  # type: ignore[no-any-return]


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))
    return float(20.0 * np.log10(rms + 1e-12))


def _tail_rms_dbfs(x: np.ndarray, sr: int, tail_seconds: float) -> float:
    n = max(1, int(round(tail_seconds * sr)))
    tail = x[-n:] if len(x) > n else x
    return _rms_dbfs(tail)


def _clip_ratio(x: np.ndarray, threshold: float = 0.999) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= threshold))


def _hf_ratio_on_loud_frames(x: np.ndarray, sr: int, loud_gate_dbfs: float = -35.0) -> float:
    """HF ratio proxy: energy(2.5-8k) / energy(80-8k) on loud frames only."""
    if x.size < 2048:
        return 0.0

    nperseg = 2048
    noverlap = 1536
    freqs, _, stft = signal.stft(
        x.astype(np.float32),
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="even",
        padded=True,
    )
    mag2 = np.abs(stft) ** 2
    if mag2.size == 0:
        return 0.0

    frame_rms = np.sqrt(np.mean(mag2, axis=0) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms + 1e-12)
    loud_mask = frame_db > loud_gate_dbfs
    if not np.any(loud_mask):
        loud_mask = frame_db > (loud_gate_dbfs - 10.0)
    if not np.any(loud_mask):
        return 0.0

    hf_mask = (freqs >= 2500.0) & (freqs <= min(8000.0, 0.49 * sr))
    bb_mask = (freqs >= 80.0) & (freqs <= min(8000.0, 0.49 * sr))
    if not np.any(hf_mask) or not np.any(bb_mask):
        return 0.0

    hf_energy = float(np.mean(np.sum(mag2[hf_mask][:, loud_mask], axis=0)))
    bb_energy = float(np.mean(np.sum(mag2[bb_mask][:, loud_mask], axis=0)))
    if bb_energy <= 1e-12:
        return 0.0
    return float(hf_energy / (bb_energy + 1e-12))


def _collect_stats(audio: np.ndarray, sr: int, tail_seconds: float) -> SignalStats:
    mono = _to_mono(audio)
    duration_s = float(len(mono) / max(sr, 1))
    return SignalStats(
        sr=sr,
        duration_s=duration_s,
        tail_rms_dbfs=_tail_rms_dbfs(mono, sr, tail_seconds),
        peak99=float(np.percentile(np.abs(mono), 99.9)) if mono.size > 0 else 0.0,
        clip_ratio=_clip_ratio(mono),
        hf_ratio_loud=_hf_ratio_on_loud_frames(mono, sr),
    )


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    loaded = load_audio_file(str(path), target_sr=None, mono=False, do_carrier_analysis=False)
    if not loaded or loaded.get("audio") is None:
        raise RuntimeError(f"Could not load audio: {path}")
    audio = np.asarray(loaded["audio"], dtype=np.float32)
    sr = int(loaded.get("sr") or 0)
    if sr <= 0:
        raise RuntimeError(f"Invalid sample rate for {path}")
    return audio, sr


def run(import_path: Path, export_path: Path, tail_seconds: float, material: str = "auto") -> int:
    imp_audio, imp_sr = _load_audio(import_path)
    exp_audio, exp_sr = _load_audio(export_path)

    imp_mono = _to_mono(imp_audio)
    exp_mono = _to_mono(exp_audio)

    target_sr = imp_sr
    exp_mono_rs = _resample_if_needed(exp_mono, exp_sr, target_sr)

    # Align to common length for frame-comparable derived metrics
    n = min(len(imp_mono), len(exp_mono_rs))
    if n < 2048:
        raise RuntimeError("Audio too short for robust verification")
    imp_mono = imp_mono[:n]
    exp_mono_rs = exp_mono_rs[:n]

    imp_stats = _collect_stats(imp_mono, target_sr, tail_seconds)
    exp_stats = _collect_stats(exp_mono_rs, target_sr, tail_seconds)

    tail_delta_db = exp_stats.tail_rms_dbfs - imp_stats.tail_rms_dbfs
    clip_delta = exp_stats.clip_ratio - imp_stats.clip_ratio
    hf_delta = exp_stats.hf_ratio_loud - imp_stats.hf_ratio_loud
    peak99_delta = exp_stats.peak99 - imp_stats.peak99

    material_key = _resolve_material(material, import_path)
    thresholds = _MATERIAL_THRESHOLDS.get(material_key, _MATERIAL_THRESHOLDS["unknown"])

    tail_fail = tail_delta_db > thresholds["tail"]
    clip_fail = clip_delta > thresholds["clip"] or exp_stats.peak99 > thresholds["peak99"]
    hf_fail = hf_delta > thresholds["hf"] and exp_stats.peak99 > 0.92

    print("=== Import-vs-Export Guard Verification ===")
    print(f"Import: {import_path}")
    print(f"Export: {export_path}")
    print(f"Material: {material_key}")
    print(f"Aligned SR: {target_sr} Hz | Samples: {n}")
    print(
        "Thresholds: "
        f"tail<= {thresholds['tail']:.2f} dB, "
        f"clip_delta<= {thresholds['clip']:.6f}, "
        f"peak99<= {thresholds['peak99']:.3f}, "
        f"hf_delta<= {thresholds['hf']:.3f}"
    )
    print("")
    print("Tail Silence:")
    print(f"  import tail RMS: {imp_stats.tail_rms_dbfs:.2f} dBFS")
    print(f"  export tail RMS: {exp_stats.tail_rms_dbfs:.2f} dBFS")
    print(f"  delta: {tail_delta_db:+.2f} dB {'FAIL' if tail_fail else 'OK'}")
    print("")
    print("Peak/Clipping:")
    print(f"  import peak99: {imp_stats.peak99:.4f}")
    print(f"  export peak99: {exp_stats.peak99:.4f}")
    print(f"  peak99 delta: {peak99_delta:+.4f}")
    print(f"  import clip ratio: {imp_stats.clip_ratio:.6f}")
    print(f"  export clip ratio: {exp_stats.clip_ratio:.6f}")
    print(f"  clip ratio delta: {clip_delta:+.6f} {'FAIL' if clip_fail else 'OK'}")
    print("")
    print("HF Harshness Proxy (loud frames):")
    print(f"  import hf ratio: {imp_stats.hf_ratio_loud:.4f}")
    print(f"  export hf ratio: {exp_stats.hf_ratio_loud:.4f}")
    print(f"  delta: {hf_delta:+.4f} {'FAIL' if hf_fail else 'OK'}")
    print("")

    fail_count = int(tail_fail) + int(clip_fail) + int(hf_fail)
    if fail_count > 0:
        print(f"RESULT: FAIL ({fail_count} guard(s) violated)")
        return 2

    print("RESULT: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick guard verification: import vs export audio")
    p.add_argument("import_file", type=Path, help="Path to source/import audio")
    p.add_argument("export_file", type=Path, help="Path to restored/exported audio")
    p.add_argument(
        "--material",
        type=str,
        default="auto",
        help="Material override (auto, shellac, vinyl, reel_tape, tape, cassette, cd_digital, mp3_low)",
    )
    p.add_argument("--tail-seconds", type=float, default=5.0, help="Tail window size in seconds (default: 5.0)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    return run(args.import_file, args.export_file, tail_seconds=float(args.tail_seconds), material=str(args.material))


if __name__ == "__main__":
    raise SystemExit(main())
