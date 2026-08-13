#!/usr/bin/env python3
"""§v10.998: Die ehrliche Gesamtmessung — jede Defekt-Art gegen die SOTA-Kette.

Erzeugt kontrollierte Defekt-Fälle aus einer clean-Referenz, führt die
SOTA-Kette aus (Consensus → Plan → CoordinatedRepair) und misst SNR-Delta.
Ergebnis: EINE Tabelle, die zeigt, wo Aurik wirklich steht.

Ausführung: venv_rocm python scripts/measure_all_defects.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

CLEAN = "corpus/vinyl/clean/vinyl_jazz_1960s_clean.wav"
OUT = "/tmp/defect_matrix.json"


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


def make_defects(clean: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Erzeugt alle kontrollierten Defekt-Fälle (Stereo [T, C])."""
    rng = np.random.default_rng(99)
    n = len(clean)
    defects: dict[str, np.ndarray] = {}

    # click: isolierte Impulse alle 2s
    click = clean.copy()
    for t in range(sr // 2, n - sr, sr * 2):
        click[t : t + 8] += 0.7
    defects["click"] = np.clip(click, -1, 1)

    # crackle: dichte Mikro-Impulse (~100/s, niedrig)
    crackle = clean.copy()
    n_imp = int(n / sr * 100)
    for _ in range(n_imp):
        t = int(rng.integers(0, n - 8))
        crackle[t : t + 3] += float(rng.uniform(0.05, 0.25))
    defects["crackle"] = np.clip(crackle, -1, 1)

    # hiss: weißes Rauschen -25 dB
    hiss = clean + rng.standard_normal(clean.shape).astype(np.float32) * 0.02
    defects["hiss"] = hiss

    # clipping: hartes Clipping bei ±0.4
    clipping = np.clip(clean * 2.2, -0.4, 0.4)
    defects["clipping"] = clipping

    # dropout: 50-ms-Nullstellen alle 1.5s
    dropout = clean.copy()
    for t in range(sr // 2, n - sr // 20, int(sr * 1.5)):
        dropout[t : t + sr // 20] = 0.0
    defects["dropout"] = dropout

    # wow_flutter: sinusförmige Geschwindigkeitsmodulation ±2% @ 0.5 Hz
    t_axis = np.arange(n) / sr
    mod = 1.0 + 0.02 * np.sin(2 * np.pi * 0.5 * t_axis)
    t_new = np.cumsum(mod) / sr
    t_new = t_new - t_new[0]
    t_new = t_new / t_new[-1] * (n - 1) / sr
    wow = np.stack(
        [np.interp(t_new, t_axis, clean[:, c]) for c in range(clean.shape[1])], axis=1
    ).astype(np.float32)
    defects["wow_flutter"] = wow

    # print_through: verzögerte Kopie (+100 ms, alpha 0.15)
    pt = clean.copy()
    d = sr // 10
    pt[d:] += 0.15 * clean[:-d]
    defects["print_through"] = np.clip(pt, -1, 1)

    # pre_echo: vorgezogene Kopie (−100 ms, alpha 0.1)
    pe = clean.copy()
    d = sr // 10
    pe[:-d] += 0.1 * clean[d:]
    defects["pre_echo"] = np.clip(pe, -1, 1)

    # phase_error: rechter Kanal um 60 Samples verzögert (Phasenversatz)
    ph = clean.copy()
    ph[60:, 1] = clean[:-60, 1]
    defects["phase_error"] = ph

    # distortion: Soft-Clipping (tanh-Waveshaper)
    dist = np.tanh(clean * 3.0) * 0.7
    defects["distortion"] = dist.astype(np.float32)

    # hum: 50 Hz + 3 Obertöne
    t2 = np.arange(n) / sr
    hum = clean.copy()
    for h, amp in ((50, 0.05), (100, 0.03), (150, 0.02), (200, 0.01)):
        hum += (amp * np.sin(2 * np.pi * h * t2)).astype(np.float32)[:, np.newaxis]
    defects["hum"] = hum

    return defects


def main() -> int:
    clean, sr = sf.read(CLEAN, dtype="float32", always_2d=False)
    print(f"Clean-Referenz: {clean.shape} @ {sr} Hz", flush=True)

    from backend.core.coordinated_repair import CoordinatedRepair, RepairPlanner
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

    defects = make_defects(clean, sr)
    results: list[dict] = []

    for name, damaged in defects.items():
        t0 = time.time()
        snr_in = _snr_db(damaged, clean)
        print(f"\n── {name} (SNR in: {snr_in:+.1f} dB) ──", flush=True)
        try:
            manifest = DefectConsensusPipeline().analyze(damaged, sr)
            plan = RepairPlanner().plan(manifest, len(damaged))
            executor = CoordinatedRepair()
            out, report = executor.execute(damaged, plan, manifest, sr, material="unknown")
            snr_out = _snr_db(out, clean)
            results.append({
                "defect": name,
                "snr_in_db": round(snr_in, 2),
                "snr_out_db": round(snr_out, 2),
                "snr_delta_db": round(snr_out - snr_in, 2),
                "steps": len(getattr(report, "completed_steps", []) or []),
                "failed": [f[0] for f in getattr(report, "failed_steps", []) or []],
                "guards": dict(getattr(report, "guard_violations", {}) or {}),
                "detected": [
                    f"{str(d.category).split('.')[-1]}({d.severity:.2f})"
                    for d in manifest.defects[:6]
                ],
                "seconds": round(time.time() - t0, 1),
            })
            print(f"  SNR {snr_in:+.1f} → {snr_out:+.1f} dB "
                  f"({snr_out - snr_in:+.2f} dB) · {len(getattr(report, 'completed_steps', []))} Schritte "
                  f"· Guards {getattr(report, 'guard_violations', {})}", flush=True)
        except Exception as exc:
            print(f"  ❌ {exc}", flush=True)
            results.append({"defect": name, "error": str(exc), "snr_in_db": round(snr_in, 2)})

    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
