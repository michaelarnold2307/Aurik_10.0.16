#!/usr/bin/env python3
"""B2/B3 Perzeptuelle Evidenz-Runner (§G131–§G134, §G150–§G152).

Misst pro Korpus-Datei die Compliance-Metriken — NICHT SNR/THD (§G152):
  - defect_reduction_per_type (Pre/Post pro Defekttyp, §G133/§G134)      → B2
  - musical_improvement (40 % Tech + 60 % MUSHRA/HPI, §G131/§G132)       → B3
  - strategy/metadata aus RestorationResult

Usage:
    .venv_aurik/bin/python scripts/evidence_perceptual_b2b3.py \
        --corpus corpus --out reports/perceptual_evidence_b2b3.json \
        --material cassette --limit 4

Keine MUSHRA-Werte (B1) — die liefert die Hörexperiment-Kampagne separat;
dieser Runner erzeugt die maschinelle Evidenz-Grundlage dafür.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    with sf.SoundFile(str(path)) as snd:
        sr = snd.samplerate
        audio = snd.read(dtype="float32")
    if audio.ndim == 2 and audio.shape[1] > 2:
        audio = audio.mean(axis=1)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return audio, sr


def run_one(restorer, path: Path, material: str) -> dict:
    audio, sr = _load_audio(path)
    t0 = time.time()
    result = restorer.restore(
        audio,
        sample_rate=sr,
        material_type=material,
        quality_mode="balanced",
    )
    elapsed = time.time() - t0
    out = result.audio
    meta = dict(result.metadata) if hasattr(result, "metadata") else {}

    # B2: per-Defekt-Reduktion
    dr = meta.get("defect_reduction_per_type", {})
    if not dr:
        dr_attr = getattr(result, "defect_reduction_per_type", None)
        dr = dict(dr_attr) if dr_attr else {}

    # B3: musikalische Verbesserung (§G131 — Feld heißt musical_quality_assurance)
    musical_improvement = None
    mqa = meta.get("musical_quality_assurance") or meta.get("_mqa_result", {})
    if isinstance(mqa, dict):
        musical_improvement = mqa.get("musical_improvement")
    for key in ("musical_improvement", "musical_improvement_score"):
        if key in meta:
            musical_improvement = float(meta[key])
            break

    return {
        "file": str(path),
        "material": material,
        "elapsed_s": round(elapsed, 2),
        "input_seconds": round(len(audio) / sr, 2),
        "rt_factor": round(elapsed / max(len(audio) / sr, 1e-9), 2),
        "output_finite": bool(np.all(np.isfinite(out))),
        "output_peak": float(np.max(np.abs(out))) if out.size else 0.0,
        "defect_reduction_per_type": dr,
        "musical_improvement": musical_improvement,
        "strategy_used": meta.get("strategy_used"),
        "quality_mode": meta.get("quality_mode"),
        "warnings": len(getattr(result, "warnings", []) or []),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--out", default="reports/perceptual_evidence_b2b3.json")
    p.add_argument("--material", default=None, help="nur dieses Material-Verzeichnis")
    p.add_argument("--limit", type=int, default=0, help="max. Dateien (0=alle damaged)")
    p.add_argument("--max-seconds", type=int, default=0, help="nur Dateien <= N Sekunden")
    args = p.parse_args()

    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    corpus = Path(args.corpus)
    materials = [args.material] if args.material else sorted(
        d.name for d in corpus.iterdir() if d.is_dir() and (d / "damaged").is_dir()
    )
    files: list[tuple[str, Path]] = []
    for mat in materials:
        damaged = corpus / mat / "damaged"
        for f in sorted(damaged.glob("*.wav")):
            files.append((mat, f))
    if args.limit:
        files = files[: args.limit]

    restorer = UnifiedRestorerV3()
    results = []
    for mat, f in files:
        try:
            if args.max_seconds:
                import soundfile as sf

                with sf.SoundFile(str(f)) as snd:
                    if snd.frames / snd.samplerate > args.max_seconds:
                        continue
            print(f"[B2/B3] {mat:12s} {f.name} ...", flush=True)
            results.append(run_one(restorer, f, mat))
        except Exception as exc:
            print(f"[B2/B3] FEHLER {f}: {exc}", flush=True)
            results.append({"file": str(f), "material": mat, "error": str(exc)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n[B2/B3] {len(results)} Dateien → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
