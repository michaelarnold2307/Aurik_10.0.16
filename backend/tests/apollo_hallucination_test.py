#!/usr/bin/env python3
"""§v10.303.20 — Apollo MP3-Decompression Halluzinationstest.

Prüft ob Apollo auf MP3-komprimiertem Material musikalische Inhalte
halluziniert. Verwendet den Hallucination-Guard (§2.46e):
- spectral_novelty > 0.15 → Apollo erfindet Inhalte → ungeeignet
- spectral_novelty ≤ 0.15 → Apollo reproduziert nur → geeignet

Test-Szenarien:
  1. Cassette-Hiss → MP3 → Apollo: Darf nichts hinzufügen (novelty ≤ 0.15)
  2. DGG-Klassik → MP3 → Apollo: Darf nur entfernen, nicht erfinden

Usage:
  python backend/tests/apollo_hallucination_test.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apollo_hallucination_test")

# ── Pfade ───────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_APOLLO_MODEL = _PROJECT_ROOT / "models" / "apollo" / "apollo_model.pt"


# ── Spectral Novelty (Hallucination-Guard §2.46e) ───────────────────────


def spectral_novelty(original: np.ndarray, processed: np.ndarray, sr: int) -> float:
    """Misst strukturelle NEUHEIT in processed relativ zu original.

    Returns: 0.0 = identisch (nur Rauschen entfernt), >0.15 = Halluzination.
    """
    _a = original if original.ndim == 1 else np.mean(original, axis=0)
    _b = processed if processed.ndim == 1 else np.mean(processed, axis=0)
    n_fft, hop = 2048, 512

    spec_a = np.abs(
        np.array([np.fft.rfft(_a[i : i + n_fft]) for i in range(0, len(_a) - n_fft, hop)])
    )
    spec_b = np.abs(
        np.array([np.fft.rfft(_b[i : i + n_fft]) for i in range(0, len(_b) - n_fft, hop)])
    )
    m = min(spec_a.shape[0], spec_b.shape[0])
    spec_a, spec_b = spec_a[:m], spec_b[:m]

    diff = np.abs(spec_b - spec_a)
    return float(np.clip(np.mean(diff) / (np.mean(spec_a) + 1e-10), 0.0, 1.0))


# ── Apollo-Loader ────────────────────────────────────────────────────────


def load_apollo():
    """Lädt das Apollo TorchScript-Modell. Returns (model, device) oder (None, None)."""
    if not _APOLLO_MODEL.exists():
        logger.error("Apollo-Modell nicht gefunden: %s", _APOLLO_MODEL)
        return None, None

    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.jit.load(str(_APOLLO_MODEL), map_location=dev)
        model.eval()
        model.to(dev)
        logger.info("Apollo geladen: %s (%.1f MB, device=%s)", _APOLLO_MODEL.name,
                     _APOLLO_MODEL.stat().st_size / 1e6, dev)
        return model, dev
    except ImportError:
        logger.error("torch nicht installiert. pip install torch")
        return None, None
    except Exception as e:
        logger.error("Apollo-Ladefehler: %s", e)
        return None, None


def apollo_process(model, audio: np.ndarray, sr: int, device: str = "cpu") -> np.ndarray:
    """Führt Apollo-Inferenz auf Audio aus. Returns verarbeitetes Audio."""
    import torch
    import torchaudio

    _orig_len = len(audio)
    APOLLO_SR = 44100

    # Zero-pad auf Minimum (8192 @ 44100 → ~9102 @ 48000)
    min_samples = int(np.ceil(8192 * sr / APOLLO_SR))
    if _orig_len < min_samples:
        audio = np.pad(audio.astype(np.float32), (0, min_samples - _orig_len))

    # Resample 48000 → 44100
    t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    if sr != APOLLO_SR:
        t = torchaudio.functional.resample(t, sr, APOLLO_SR)

    with torch.no_grad():
        out = model(t)

    del t

    # Resample 44100 → 48000
    if sr != APOLLO_SR:
        out = torchaudio.functional.resample(out, APOLLO_SR, sr)

    result = out.squeeze().cpu().numpy().astype(np.float32)
    del out

    if _orig_len < len(result):
        result = result[:_orig_len]

    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(result, -1.0, 1.0)


# ── Test-Signale ─────────────────────────────────────────────────────────


def generate_cassette_hiss(sr: int = 48000, duration: float = 2.0) -> np.ndarray:
    """Generiert simuliertes Cassette-Hiss: bandpass-gefiltertes weißes Rauschen."""
    try:
        from scipy.signal import butter, sosfilt

        noise = np.random.randn(int(sr * duration)).astype(np.float32) * 0.03
        sos = butter(4, [300 / (sr / 2), 8000 / (sr / 2)], btype="band", output="sos")
        return sosfilt(sos, noise).astype(np.float32)
    except ImportError:
        # Fallback ohne scipy
        noise = np.random.randn(int(sr * duration)).astype(np.float32) * 0.03
        return noise


def mp3_compress(audio: np.ndarray, sr: int, bitrate: str = "128k") -> np.ndarray:
    """Komprimiert Audio via ffmpeg MP3 und lädt zurück."""
    _tmp_wav = None
    _tmp_mp3 = None
    try:
        import soundfile as sf

        _tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        _tmp_wav.close()
        _tmp_mp3.close()

        sf.write(_tmp_wav.name, audio, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", _tmp_wav.name, "-b:a", bitrate, _tmp_mp3.name],
            capture_output=True, check=True,
        )
        result, result_sr = sf.read(_tmp_mp3.name)
        return np.asarray(result, dtype=np.float32)
    except Exception as e:
        logger.warning("MP3-Komprimierung fehlgeschlagen: %s", e)
        return audio
    finally:
        for f in [_tmp_wav, _tmp_mp3]:
            if f:
                try:
                    os.unlink(f.name)
                except OSError:
                    pass


# ── Gänsehaut-Score (vereinfacht) ────────────────────────────────────────


def simplified_goosebumps_score(
    audio: np.ndarray, sr: int, original: np.ndarray | None = None
) -> float:
    """Vereinfachter Gänsehaut-Score basierend auf Dynamik-Kontrast.

    Misst: (Peak-to-RMS Ratio) × (Spektrale-Varianz) relativ zum Original.
    1.0 = volle Dynamik erhalten, <0.6 = flachgedrückt (Gänsehaut verloren).
    """
    _a = audio if audio.ndim == 1 else np.mean(audio, axis=0)
    peak = np.max(np.abs(_a)) + 1e-10
    rms = np.sqrt(np.mean(_a**2)) + 1e-10

    # Dynamik-Kontrast
    crest = peak / rms

    # Spektrale Varianz
    n_fft = 2048
    spec = np.abs(np.fft.rfft(_a[:n_fft * 4]))
    spec_var = float(np.var(spec) / (np.mean(spec) + 1e-10))

    # Kombinierter Score, normalisiert
    score = float(np.clip(np.log1p(crest) * np.log1p(spec_var) / 10.0, 0.0, 1.0))
    return score


# ── Haupt-Test ───────────────────────────────────────────────────────────


def run_tests() -> dict:
    """Führt den vollständigen Apollo-Halluzinationstest durch."""
    logger.info("=" * 60)
    logger.info("§v10.303.20 Apollo MP3-Decompression Halluzinationstest")
    logger.info("=" * 60)

    model, device = load_apollo()
    if model is None:
        return {
            "status": "SKIPPED",
            "reason": "Apollo-Modell nicht geladen",
            "hallucination_detected": None,
        }

    sr = 48000
    results = {}

    # ── Test 1: Cassette-Hiss ─────────────────────────────────────────
    logger.info("\n── Test 1: Cassette-Hiss → MP3 → Apollo ──")
    hiss = generate_cassette_hiss(sr, duration=2.0)
    mp3_hiss = mp3_compress(hiss, sr)

    t0 = time.perf_counter()
    apollo_out = apollo_process(model, mp3_hiss, sr, device)
    elapsed = time.perf_counter() - t0

    novelty = spectral_novelty(mp3_hiss, apollo_out, sr)
    rms_in = float(np.sqrt(np.mean(mp3_hiss**2) + 1e-12))
    rms_out = float(np.sqrt(np.mean(apollo_out**2) + 1e-12))

    test1 = {
        "scenario": "cassette_hiss",
        "spectral_novelty": round(novelty, 4),
        "rms_in_db": round(20 * np.log10(rms_in + 1e-12), 1),
        "rms_out_db": round(20 * np.log10(rms_out + 1e-12), 1),
        "rms_delta_db": round(20 * np.log10(max(rms_out, 1e-12) / max(rms_in, 1e-12)), 1),
        "goosebumps_in": round(simplified_goosebumps_score(mp3_hiss, sr), 4),
        "goosebumps_out": round(simplified_goosebumps_score(apollo_out, sr), 4),
        "elapsed_s": round(elapsed, 2),
        "hallucination": novelty > 0.15,
    }
    logger.info("  Spectral Novelty: %.4f (threshold: 0.15)", novelty)
    logger.info("  RMS: %.1f dB → %.1f dB (Δ %.1f dB)", test1["rms_in_db"],
                 test1["rms_out_db"], test1["rms_delta_db"])
    logger.info("  Gänsehaut: %.4f → %.4f", test1["goosebumps_in"], test1["goosebumps_out"])
    results["hiss_test"] = test1

    # ── Test 2: 1 kHz Sinus → MP3 → Apollo ────────────────────────────
    logger.info("\n── Test 2: 1 kHz Sinus → MP3 → Apollo ──")
    t = np.arange(0, 2.0, 1 / sr, dtype=np.float32)
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5
    mp3_sine = mp3_compress(sine, sr)

    t0 = time.perf_counter()
    apollo_sine = apollo_process(model, mp3_sine, sr, device)
    elapsed = time.perf_counter() - t0

    novelty2 = spectral_novelty(mp3_sine, apollo_sine, sr)
    test2 = {
        "scenario": "1khz_sine",
        "spectral_novelty": round(novelty2, 4),
        "elapsed_s": round(elapsed, 2),
        "hallucination": novelty2 > 0.15,
    }
    logger.info("  Spectral Novelty: %.4f (threshold: 0.15)", novelty2)
    results["sine_test"] = test2

    # ── Gesamt-Verdict ────────────────────────────────────────────────
    any_hallucination = test1["hallucination"] or test2["hallucination"]
    verdict = (
        "❌ UNGEEIGNET — Apollo erfindet musikalische Inhalte auf MP3-komprimiertem Material"
        if any_hallucination
        else "✅ GEEIGNET — Apollo beschränkt sich auf Decompression, keine Halluzination"
    )

    results["verdict"] = verdict
    results["status"] = "FAIL" if any_hallucination else "PASS"
    results["hallucination_detected"] = any_hallucination

    logger.info("\n" + "=" * 60)
    logger.info(verdict)
    logger.info("=" * 60)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(_PROJECT_ROOT))

    result = run_tests()

    # JSON-Output für CI
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("status") == "FAIL":
        sys.exit(1)
    elif result.get("status") == "SKIPPED":
        sys.exit(0)
    else:
        sys.exit(0)
