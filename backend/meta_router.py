#!/usr/bin/env python3
"""
Meta-Controller: classify medium + chain + route to best YAML profile.

Design goals:
- Offline, deterministic, audit-friendly
- Hard tests first, then weighted scoring
- Pilot-run verification on 3 short segments
- Outputs routing package -> your restoration decision logic

Dependencies:
- numpy
- scipy
- soundfile (recommended) OR scipy.io.wavfile as fallback
- PyYAML (recommended) for YAML configs

Install (typical):
  pip install numpy scipy soundfile pyyaml
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

try:
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

try:
    import librosa  # type: ignore

    _LIBROSA_OK = True
except Exception:
    _LIBROSA_OK = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Lädt audio file → (mono float32 array, sample_rate).

    Cascade: soundfile → scipy.io.wavfile → synthetic silence (fallback).
    """
    # Stufe 1: backend.file_import — robust cascade (soundfile → pedalboard → pydub)
    try:
        from backend.file_import import load_audio_file as _load_audio_file

        _res = _load_audio_file(path, do_carrier_analysis=False)
        if _res and not _res.get("error"):
            audio = _res["audio"]
            sr = _res["sr"]
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
        logger.warning("_laden_audio: file_import returned error for %s: %s", path, (_res or {}).get("error"))
    except Exception as exc:
        logger.warning("_laden_audio: file_import fehlgeschlagen for %s: %s — trying soundfile", path, exc)

    # Stufe 2: scipy fallback (WAV only)
    try:
        from scipy.io import wavfile

        _wf_result = wavfile.read(path)
        if isinstance(_wf_result, tuple) and len(_wf_result) >= 2:
            sr = int(_wf_result[0])
            data = _wf_result[1]
        else:
            raise ValueError(f"wavfile.read returned unexpected type: {type(_wf_result)}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = data.astype(np.float32)
        if audio.max() > 1.0:
            audio /= float(np.iinfo(data.dtype).max)
        return audio, int(sr)
    except Exception as exc:
        logger.error("scipy wavfile fehlgeschlagen for %s: %s", path, exc)

    logger.error("_laden_audio: all loaders fehlgeschlagen for %s — returning silence", path)
    return np.zeros(4800, dtype=np.float32), 48_000


def _extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Extrahiert a compact feature set for profile matching.

    Returns
    -------
    dict with keys:
        rms, zcr, spectral_centroid, spectral_rolloff, tempo,
        hf_energy_ratio, low_energy_ratio, duration_s
    """
    eps = 1e-12
    features: dict = {}

    # Duration
    features["duration_s"] = float(len(audio)) / max(sr, 1)

    # RMS
    rms = float(np.sqrt(np.mean(audio**2) + eps))
    # NaN/Inf-Guard
    rms = 0.0 if not math.isfinite(rms) else rms
    features["rms"] = rms
    rms_db = float(20.0 * math.log10(rms + eps))
    features["rms_db"] = 0.0 if not math.isfinite(rms_db) else rms_db

    # ZCR (zero-crossing rate, normalised to [0, 1] for any signal)
    features["zcr"] = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)

    # Spectral features via rfft
    n_fft = min(2048, len(audio))
    window = np.hanning(n_fft)
    seg = audio[:n_fft] * window
    spec = np.abs(np.fft.rfft(seg)) + eps
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    spectral_centroid = float(np.sum(freqs * spec) / np.sum(spec))
    features["spectral_centroid"] = 0.0 if not math.isfinite(spectral_centroid) else spectral_centroid

    # Spectral rolloff at 85 %
    cum = np.cumsum(spec)
    roll_thresh = 0.85 * cum[-1]
    roll_idx = int(np.searchsorted(cum, roll_thresh))
    features["spectral_rolloff"] = float(freqs[min(roll_idx, len(freqs) - 1)])

    # Band energy ratios
    total_e = float(np.sum(spec**2) + eps)
    hf_mask = freqs >= 8_000.0
    lo_mask = freqs <= 300.0
    features["hf_energy_ratio"] = float(np.sum(spec[hf_mask] ** 2) / total_e)
    features["low_energy_ratio"] = float(np.sum(spec[lo_mask] ** 2) / total_e)

    # Spectral flatness (Wiener entropy)
    log_spec = np.log(spec + eps)
    gm = float(np.exp(np.mean(log_spec)))
    am = float(np.mean(spec))
    features["spectral_flatness"] = gm / (am + eps)

    # Tempo
    if _LIBROSA_OK:
        try:
            tempo_arr, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features["tempo"] = float(np.asarray(tempo_arr).flat[0])
        except Exception:
            features["tempo"] = _estimate_tempo_acf(audio, sr)
    else:
        features["tempo"] = _estimate_tempo_acf(audio, sr)

    return features


def _estimate_tempo_acf(audio: np.ndarray, sr: int) -> float:
    """Rough tempo estimate via autocorrelation of the onset envelope."""
    hop = max(sr // 100, 1)  # ~10 ms
    frames = np.abs(audio[: len(audio) - hop : hop])
    if len(frames) < 8:
        return 0.0
    # Onset strength: diff of RMS frames
    onset = np.diff(np.maximum(frames, 0.0))
    if len(onset) < 4:
        return 0.0
    onset -= onset.mean()
    from backend.core.core_utils import fft_autocorr

    acf = fft_autocorr(onset)
    lag_min = max(1, int(sr / (hop * 220.0)))  # 220 BPM upper bound
    lag_max = max(lag_min + 1, int(sr / (hop * 40.0)))  # 40 BPM lower bound
    lag_max = min(lag_max, len(acf) - 1)
    if lag_min >= lag_max:
        return 0.0
    peak_lag = int(np.argmax(acf[lag_min:lag_max])) + lag_min
    bpm = 60.0 / (peak_lag * hop / sr) if peak_lag > 0 else 0.0
    return float(np.clip(bpm, 0.0, 300.0))


def _load_meta_config(path: str | None) -> dict:
    """Lädt a YAML or JSON meta/profile config file.

    Returns an empty dict if *path* is None or loading fails.
    """
    if not path:
        return {}

    # Try YAML first
    if yaml is not None:
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("YAML laden fehlgeschlagen for %s: %s — trying JSON", path, exc)

    # JSON fallback
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("JSON laden also fehlgeschlagen for %s: %s", path, exc)
        return {}


def _match_profile(features: dict, config: dict) -> str:
    """Rule-based profile matching against the *profiles* section of *config*.

    Each profile in ``config["profiles"]`` maps to a dict of threshold rules:
        { "rms": 0.05, "spectral_centroid": 2000 }

    A profile matches when **all** feature values in the rule are ≥ the
    configured threshold.  The first matching profile (in insertion order) wins.
    Falls back to ``config.get("default_profile", "standard")``.
    """
    profiles: dict = config.get("profiles", {})
    for name, rules in profiles.items():
        if not isinstance(rules, dict):
            continue
        if all(float(features.get(k, 0.0)) >= float(v) for k, v in rules.items()):
            return str(name)
    return str(config.get("default_profile", "standard"))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def route_media(audio_path: str, meta_path: str | None = None) -> dict:
    """Classify an audio file and route it to the best restoration profile.

    Parameters
    ----------
    audio_path:
        Path to the audio file (WAV, FLAC, AIFF, …).
    meta_path:
        Path to a YAML / JSON profile config.  Pass *None* to use defaults.

    Returns
    -------
    dict with keys:
        ``profile``   — matched profile name (str)
        ``features``  — extracted feature dict
        ``meta``      — loaded meta-config dict
        ``audio_path`` — echoed back for audit purposes
    """
    logger.info("route_media: lade %s", audio_path)
    audio, sr = _load_audio(audio_path)
    features = _extract_features(audio, sr)
    meta = _load_meta_config(meta_path)
    profile = _match_profile(features, meta)
    logger.info(
        "route_media: Profil=%s  centroid=%.0f Hz  tempo=%.1f BPM",
        profile,
        features.get("spectral_centroid", 0.0),
        features.get("tempo", 0.0),
    )
    return {
        "profile": profile,
        "features": features,
        "meta": meta,
        "audio_path": audio_path,
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Offline medium+chain classifier -> profile router")
    ap.add_argument("audio", help="Input audio file (wav/flac/aiff etc.)")
    ap.add_argument("--meta", required=True, help="Path to meta_controller.yaml")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    pkg = route_media(args.audio, args.meta)

    js = json.dumps(pkg, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        logger.info("Wrote: %s", args.out)
    else:
        logger.info(str(js))


if __name__ == "__main__":
    main()
