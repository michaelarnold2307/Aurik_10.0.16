#!/usr/bin/env python3
"""
§v10.880: Schritt-Ablation — welcher Repair-Schritt schadet wirklich?

Führt JEDEN Schritt des Plans EINZELN aus (nicht in Serie) und misst die
SNR-Änderung gegenüber der beschädigten Eingabe. Damit wird präzise
sichtbar, welche Schritte verbessern, welche schaden.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

SR = 48000


def _snr_db(reference: np.ndarray, signal: np.ndarray) -> float:
    n = np.asarray(signal) - np.asarray(reference)
    ref = np.asarray(reference)
    return float(10 * np.log10((np.mean(ref**2) + 1e-10) / (np.mean(n**2) + 1e-10)))


def _load(path: Path, chunk_s: float = 4.0) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = min(int(chunk_s * sr), len(audio))
    return audio[:n].astype(np.float32), sr


def ablate(medium: str, damaged_name: str, clean_name: str):
    from backend.core.coordinated_repair import CoordinatedRepair, RepairPlanner
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

    dmg, sr = _load(_PROJECT / "corpus" / medium / "damaged" / damaged_name)
    clean, _ = _load(_PROJECT / "corpus" / medium / "clean" / clean_name)
    min_len = min(len(dmg), len(clean))
    dmg, clean = dmg[:min_len], clean[:min_len]

    baseline = _snr_db(clean, dmg)
    print(f"\n{'=' * 70}")
    print(f"{damaged_name} — Baseline SNR: {baseline:+.2f} dB")
    print(f"{'=' * 70}")

    # Consensus + Plan
    cons = DefectConsensusPipeline()
    manifest = cons.analyze(dmg, sr, metadata={"material": medium, "is_digital": medium == "digital"})
    planner = RepairPlanner()
    plan = planner.plan(manifest, len(dmg))

    print(f"Plan ({len(plan.steps)} Schritte):")
    executor = CoordinatedRepair()

    # Schritt 0: Ketten-Ergebnis als Referenz
    full_restored, _ = executor.execute(dmg, plan, manifest, sr, material=medium)
    full_restored = np.asarray(full_restored)
    if full_restored.ndim > 1:
        full_restored = full_restored.mean(axis=0)
    full_snr = _snr_db(clean, full_restored[:min_len])
    print(f"  [Kette komplett]         {full_snr:+.2f} dB ({full_snr - baseline:+.2f})")

    # Jeden Schritt EINZELN
    for step in plan.steps:
        t0 = time.time()
        try:
            # _execute_step erwartet [C, T] — Mono als [1, T] übergeben
            single = executor._execute_step(dmg[np.newaxis, :], step, manifest, sr, 1)
            single = np.asarray(single)
            if single.ndim > 1:
                single = single.mean(axis=0)
            snr_after = _snr_db(clean, single[:min_len])
            delta = snr_after - baseline
            verdict = "✅" if delta > 0.3 else ("❌" if delta < -0.3 else "➖")
            print(
                f"  {verdict} {step.phase_id:40s} {snr_after:+7.2f} dB ({delta:+6.2f})  "
                f"strength={step.parameters.get('strength', 0):.2f}"
            )
        except Exception as exc:
            print(f"  ⚠️ {step.phase_id:40s} FEHLER: {str(exc)[:60]}")
        elapsed = time.time() - t0
        if elapsed > 5:
            print(f"       ({elapsed:.0f}s)")


def main() -> int:
    cases = [
        ("vinyl", "vinyl_blues_1950s_crackle.wav", "vinyl_blues_1950s_clean.wav"),
        ("digital", "digital_electronic_2000s_clicks.wav", "digital_electronic_2000s_clean.wav"),
    ]
    for medium, dmg, cln in cases:
        ablate(medium, dmg, cln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
