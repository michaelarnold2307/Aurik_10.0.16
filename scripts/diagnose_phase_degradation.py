#!/usr/bin/env python3
"""§v10.998-Diagnose: Welche Phase zerstört das Signal auf cassette_hiphop_1980s_hiss?

Führt die SOTA-Kette Schritt für Schritt aus und misst NACH JEDER PHASE
SNR + MSE gegen die clean-Referenz. Ehrliche Ursachen-Analyse statt Raten.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

DAMAGED = "corpus/cassette/damaged/cassette_hiphop_1980s_hiss.wav"
CLEAN = "corpus/cassette/clean/cassette_hiphop_1980s_clean.wav"
CLIP_S = 30.0  # Diagnose auf den ersten 30 s — deckt alle 8 Phasen ab

# §v10.998: phase_57 (Print-Through) hat eine katastrophale Laufzeit
# (35+ min auf 30 s Audio, 1100% CPU) — für die SNR-Isolation überspringen;
# die Laufzeit-Explosion ist bereits als eigener Befund dokumentiert.
SKIP_PHASES = {"phase_57_print_through_reduction"}


def _snr_db(signal: np.ndarray, ref: np.ndarray) -> float:
    if signal.shape != ref.shape:
        n = min(len(signal), len(ref))
        signal, ref = signal[:n], ref[:n]
    if signal.ndim == 2 and signal.shape[1] > signal.shape[0]:
        signal = signal.T
    if ref.ndim == 2 and ref.shape[1] > ref.shape[0]:
        ref = ref.T
    if signal.ndim == 2:
        signal = signal.mean(axis=0)
    if ref.ndim == 2:
        ref = ref.mean(axis=0)
    diff = np.asarray(signal, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return float(10 * np.log10((np.mean(ref**2) + 1e-12) / (np.mean(diff**2) + 1e-12)))


def main() -> int:
    damaged, sr = sf.read(DAMAGED, dtype="float32", always_2d=False)
    clean, sr2 = sf.read(CLEAN, dtype="float32", always_2d=False)
    n_clip = int(sr * CLIP_S)
    damaged, clean = damaged[:n_clip], clean[:n_clip]

    print(f"START: SNR(damaged vs clean) = {_snr_db(damaged, clean):+.2f} dB", flush=True)

    from backend.core.coordinated_repair import CoordinatedRepair, RepairPlanner
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

    t0 = time.time()
    manifest = DefectConsensusPipeline().analyze(damaged, sr)
    print(
        f"Consensus: {len(manifest.defects)} Defekte, {manifest.module_count} Module ({time.time() - t0:.0f}s)",
        flush=True,
    )
    for d in manifest.defects:
        print(f"  - {d.category} sev={d.severity:.2f} conf={d.confidence:.2f}", flush=True)

    plan = RepairPlanner().plan(manifest, len(damaged))
    print(f"Plan: {len(plan.steps)} Schritte: {' → '.join(s.phase_id for s in plan.steps)}", flush=True)

    executor = CoordinatedRepair()
    current = damaged.copy()

    # ── Schrittweise Ausführung mit SNR-Messung NACH jeder Phase ──
    # §v10.998: executor.execute() pro Einzelschritt — nutzt die kanonische
    # Kanal-Normalisierung (Time-major-Fix) statt roher _execute_step-Aufrufe.
    for step in plan.steps:
        if step.phase_id in SKIP_PHASES:
            print(f"  ⏭ {step.phase_id:<36} ÜBERSPRUNGEN SNR={_snr_db(current, clean):+.2f} dB", flush=True)
            continue
        try:
            from backend.core.coordinated_repair import RepairPlan

            single = RepairPlan(steps=[step], total_defects=1)
            out, report = executor.execute(current, single, manifest, sr)
            if out.shape != current.shape:
                print(f"  ⚠ {step.phase_id}: SHAPE-CHANGE {current.shape} → {out.shape} — übersprungen", flush=True)
                continue
            changed = not np.allclose(out, current, atol=1e-7)
            current = out
            snr_now = _snr_db(current, clean)
            guards = getattr(report, "guard_violations", {}) or {}
            print(f"  {step.phase_id:<36} changed={str(changed):<5} SNR={snr_now:+.2f} dB guards={guards}", flush=True)
        except Exception as exc:
            print(f"  ❌ {step.phase_id}: {exc}", flush=True)

    print(
        f"\nENDE: SNR(restored vs clean) = {_snr_db(current, clean):+.2f} dB "
        f"(damaged: {_snr_db(damaged, clean):+.2f} dB)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
