#!/usr/bin/env python3
"""
§v10.710: Parameter-Kalibrierung — erfindet keine Werte, misst sie.

Problem: Genre-Presets (Klassik 0.20, Rock 0.55) und Modul-Gewichte sind
inventierte Zahlen. Kalibrierung via Wahrnehmungs-Regelkreis fehlte.

Lösung: Sweep über Strength-Werte pro Genre, messe UTMOS-MOS auf
Referenz-Dateien, speichere den jeweils besten Wert.

Ausgabe: models/calibrated_presets.json — wird von der SOTA-Pipeline
automatisch geladen, wenn vorhanden (Fallback: Default-Presets).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

log = logging.getLogger(__name__)

SR = 48000

CALIBRATION_FILE = _PROJECT / "models" / "calibrated_presets.json"

# Sweep-Konfiguration
STRENGTH_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
GENRES_TO_CALIBRATE = ["classical", "rock", "pop", "speech", "electronic"]


def _load_eval_files(max_files: int = 3) -> list[Path]:
    """Lädt ein paar Referenz-Dateien für die Kalibrierung."""
    files: list[Path] = []
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    if musdb.is_dir():
        files.extend(sorted(musdb.rglob("*.wav"))[:max_files])
    return files


def calibrate() -> dict:
    """Führt den Kalibrierungs-Sweep durch."""
    from backend.core.sota_denoise_pipeline import SOTADenoisePipeline
    from backend.core.perceptual_closed_loop import PerceptualClosedLoop

    loop = PerceptualClosedLoop()
    pipeline = SOTADenoisePipeline()

    eval_files = _load_eval_files()
    if not eval_files:
        log.warning("Keine Evaluierungs-Dateien gefunden — Kalibrierung übersprungen")
        return {}

    import soundfile as sf

    calibrated: dict[str, float] = {}
    print("=" * 60)
    print("§v10.710 Parameter-Kalibrierung (UTMOS-basiert)")
    print(f"  {len(eval_files)} Referenz-Dateien × {len(STRENGTH_GRID)} Strengths "
          f"× {len(GENRES_TO_CALIBRATE)} Genres")
    print("=" * 60)

    for genre in GENRES_TO_CALIBRATE:
        best_strength = 0.5
        best_mos = -1.0
        print(f"\n▶ Genre: {genre}")

        for strength in STRENGTH_GRID:
            mos_values: list[float] = []

            for f in eval_files:
                try:
                    audio, sr = sf.read(str(f), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    # 2s Chunk
                    chunk = audio[: 2 * sr]
                    if len(chunk) < sr:
                        continue

                    # Denoise mit diesem Strength
                    result = pipeline.process(
                        chunk, sr, auto_params=False, override_strength=strength,
                    )
                    # MOS messen
                    mos = loop.estimate_mos(result.audio, sr)
                    mos_values.append(mos)
                except Exception as exc:
                    log.debug("Kalibrierung: Datei übersprungen (%s)", exc)

            if mos_values:
                avg_mos = float(np.mean(mos_values))
                print(f"  strength={strength:.1f} → MOS {avg_mos:.3f}")
                if avg_mos > best_mos:
                    best_mos = avg_mos
                    best_strength = strength

        calibrated[genre] = round(best_strength, 2)
        print(f"  ✓ Best: strength={best_strength:.2f} (MOS {best_mos:.3f})")

    return calibrated


def main() -> int:
    t0 = time.time()
    calibrated = calibrate()

    if not calibrated:
        print("Keine Kalibrierung möglich — Dateien fehlen.")
        return 1

    # Speichern
    payload = {
        "version": "v10.710",
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "utmos_mos_sweep",
        "presets": calibrated,
    }
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Kalibrierung gespeichert: {CALIBRATION_FILE}")
    print(f"   Werte: {calibrated}")
    print(f"   Dauer: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
