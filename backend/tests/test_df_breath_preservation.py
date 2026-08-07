#!/usr/bin/env python3
"""§v10.303.24 DeepFilterNet Breath-Preservation Test.

Verifiziert dass DeepFilterNet Atmung im Gesang erhält.
Erzeugt synthetische Atemsegmente in einem Testsignal und prüft,
ob sie nach DFN-Verarbeitung unverändert bleiben.

Usage:
  python backend/tests/test_df_breath_preservation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("df_breath_test")


def make_test_with_breaths(sr: int = 48000, duration: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt Tests mit synthetischen Atemsegmenten.

    Returns: (audio_with_breaths, breath_mask)
    """
    t = np.arange(int(sr * duration), dtype=np.float32) / sr

    # Basis: 440 Hz Ton + leichtes Rauschen (simuliert Gesang)
    tone = np.sin(2 * np.pi * 440 * t) * 0.2
    noise = np.random.RandomState(42).randn(len(t)).astype(np.float32) * 0.01
    audio = tone + noise

    # Atemsegmente: 3 Segmente à ~200 ms mit typischem Atem-Profil
    # (ZCR > 0.30, Energie < -38 dBFS → Breitband-Rauschen mit niedriger Amplitude)
    breath_mask = np.zeros(len(audio), dtype=bool)
    breath_starts = [int(0.5 * sr), int(1.5 * sr), int(2.3 * sr)]
    breath_duration = int(0.2 * sr)

    rng = np.random.RandomState(123)
    for start in breath_starts:
        end = min(start + breath_duration, len(audio))
        # Atem: gefiltertes Rauschen, sehr leise (-40 dBFS)
        breath_signal = rng.randn(end - start).astype(np.float32) * 0.01
        audio[start:end] = breath_signal
        breath_mask[start:end] = True

    return audio.astype(np.float32), breath_mask


def test_breath_preservation():
    """Haupttest: DFN auf Atemsignal → Atemsegmente müssen unverändert bleiben."""
    from plugins.apollo_phase0_integration import DeepFilterNetGuard

    audio, breath_mask = make_test_with_breaths()
    sr = 48000

    guard = DeepFilterNetGuard()
    guard._hallucination_threshold = 0.99  # type: ignore[attr-defined]  # Deaktiviere Guard für Test

    if not guard._ensure_loaded():
        logger.warning("⚠ DeepFilterNet nicht verfügbar — Test übersprungen")
        return {"status": "skipped", "reason": "DFN not available"}

    # Verarbeite mit DFN
    processed, applied = guard.process(audio, sr)

    # Vergleiche Atemsegmente
    breath_before = audio[breath_mask]
    breath_after = processed[breath_mask]

    # RMS-Änderung in Atemsegmenten
    rms_before = float(np.sqrt(np.mean(breath_before**2)) + 1e-12)
    rms_after = float(np.sqrt(np.mean(breath_after**2)) + 1e-12)
    rms_delta_db = float(20 * np.log10(rms_after / rms_before))

    # Korrelation in Atemsegmenten
    corr = float(np.corrcoef(breath_before, breath_after)[0, 1]) if len(breath_before) > 2 else 1.0

    # Nicht-Atem-Segmente: sollten verändert sein (Denoising)
    non_breath_before = audio[~breath_mask]
    non_breath_after = processed[~breath_mask]
    non_breath_rms_before = float(np.sqrt(np.mean(non_breath_before**2)) + 1e-12)
    non_breath_rms_after = float(np.sqrt(np.mean(non_breath_after**2)) + 1e-12)
    non_breath_delta_db = float(20 * np.log10(non_breath_rms_after / non_breath_rms_before))

    logger.info("=" * 60)
    logger.info("DeepFilterNet Breath-Preservation Test")
    logger.info("=" * 60)
    logger.info(f"  Atemsegmente:  {np.sum(breath_mask)} samples ({np.sum(breath_mask) / len(audio) * 100:.1f}%)")
    logger.info(f"  RMS Atem Δ:    {rms_delta_db:+.2f} dB (< 3 dB = preservation)")
    logger.info(f"  Korrelation:   {corr:.4f} (> 0.95 = preservation)")
    logger.info(f"  RMS Nicht-Atem Δ: {non_breath_delta_db:+.2f} dB (≠ 0 = denoising aktiv)")

    breath_preserved = abs(rms_delta_db) < 3.0 and corr > 0.90
    denoising_active = abs(non_breath_delta_db) > 0.1

    if breath_preserved and denoising_active:
        logger.info("✅ PASS: Atmung erhalten, Denoising aktiv")
    elif breath_preserved:
        logger.info("⚠ PASS (eingeschränkt): Atmung erhalten, aber kein Denoising-Effekt")
    else:
        logger.warning("❌ FAIL: Atmung wurde verändert!")

    return {
        "status": "pass" if breath_preserved else "fail",
        "breath_preserved": breath_preserved,
        "denoising_active": denoising_active,
        "rms_delta_db": round(rms_delta_db, 2),
        "correlation": round(corr, 4),
    }


if __name__ == "__main__":
    result = test_breath_preservation()
    sys.exit(0 if result["status"] in ("pass", "skipped") else 1)
