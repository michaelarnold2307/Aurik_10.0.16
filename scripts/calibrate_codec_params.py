#!/usr/bin/env python3
"""§CALIBRATE: Wissenschaftliche Parameter-Optimierung via AMRB-Sweep.

Variiert die 3 kritischen Codec-Parameter über den AMRB-VOCAL-Benchmark
und findet die optimale Kombination ohne Overfitting auf einen Song.

Nutzung:
    AURIK_AMRB_SWEEP=1 python scripts/calibrate_codec_params.py
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("calibrate")

# ── Parameter-Raster ─────────────────────────────────────────
# Jeder Parameter wird in 3 Stufen getestet (27 Kombinationen total)
PARAM_GRID = {
    "codec_avg_discount": [0.35, 0.45, 0.55],  # ×0.35 (stark), ×0.45 (aktuell), ×0.55 (mild)
    "click_iqr_codec": [8.5, 10.0, 12.0],  # IQR-Schwelle für Codec-Click-Guard
    "phase03_dsp_threshold": [0.12, 0.20, 0.30],  # Strength ≤ X → use_lightweight
}


def run_amrb_scenario(params: dict[str, float]) -> dict[str, float]:
    """Führt AMRB-06-VOCAL mit gegebenen Parametern aus und gibt Scores zurück."""
    # Setze Parameter via Environment
    env_override = {
        "AURIK_CODEC_DISCOUNT": str(params["codec_avg_discount"]),
        "AURIK_CLICK_IQR_CODEC": str(params["click_iqr_codec"]),
        "AURIK_PHASE03_DSP_THRESHOLD": str(params["phase03_dsp_threshold"]),
    }
    for k, v in env_override.items():
        os.environ[k] = v

    try:
        from benchmarks.musical_restoration_benchmark import (
            BenchmarkConfig,
            run_benchmark,
        )
        from scripts.run_amrb_v99 import dsp_restore

        config = BenchmarkConfig(
            restoration_fn=dsp_restore,
            system_name=f"calibrate_{params['codec_avg_discount']}_{params['click_iqr_codec']}_{params['phase03_dsp_threshold']}",
            n_items_per_scenario=1,
            enable_mushra_proxy=False,
        )
        report = run_benchmark(config, scenario_filter=["AMRB-06-VOCAL"])  # type: ignore[call-arg]

        return {
            "overall_score": report.overall_score,
            "n_passed": report.n_passed,
            "vocal_score": report.scenario_scores.get("AMRB-06-VOCAL", 0.0),  # type: ignore[attr-defined]
        }
    except Exception as exc:
        logger.warning("AMRB Ausfuehrung fehlgeschlagen: %s", exc)
        return {"overall_score": 0.0, "n_passed": 0, "vocal_score": 0.0}


def sweep() -> list[dict]:
    """Führt den vollständigen Parameter-Sweep durch."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))

    logger.info("§CALIBRATE: %d Parameter-Kombinationen über AMRB-06-VOCAL", len(combinations))
    logger.info("Parameter: %s", ", ".join(keys))
    logger.info("-" * 60)

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        t0 = time.perf_counter()

        scores = run_amrb_scenario(params)
        elapsed = time.perf_counter() - t0

        result = {**params, **scores, "runtime_s": round(elapsed, 1)}
        results.append(result)

        logger.info(
            "[%2d/%2d] discount=%.2f iqr=%.1f dsp_thr=%.2f → vocal=%.1f (%.1fs)",
            i + 1,
            len(combinations),
            params["codec_avg_discount"],
            params["click_iqr_codec"],
            params["phase03_dsp_threshold"],
            scores.get("vocal_score", 0),
            elapsed,
        )

    return results


def report(results: list[dict]) -> None:
    """Gibt die Top-5-Konfigurationen aus."""
    sorted_results = sorted(results, key=lambda r: r.get("vocal_score", 0), reverse=True)

    print("\n" + "=" * 60)
    print("§CALIBRATE: Top-5 Parameter-Kombinationen (AMRB-06-VOCAL)")
    print("=" * 60)
    for i, r in enumerate(sorted_results[:5]):
        print(
            f"  #{i + 1}: discount={r['codec_avg_discount']:.2f} "
            f"iqr={r['click_iqr_codec']:.0f} "
            f"dsp_thr={r['phase03_dsp_threshold']:.2f} "
            f"→ vocal={r.get('vocal_score', 0):.1f}"
        )

    best = sorted_results[0]
    print(
        f"\n  ★ Optimal: discount={best['codec_avg_discount']:.2f} "
        f"iqr={best['click_iqr_codec']:.0f} "
        f"dsp_thr={best['phase03_dsp_threshold']:.2f}"
    )

    # Speichere Ergebnisse
    out_path = Path("logs/calibrate_codec_params.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": sorted_results, "best": best, "params": list(PARAM_GRID.keys())}, f, indent=2)
    print(f"\n  Ergebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Nur 2 Stufen pro Parameter (8 Kombinationen)")
    args = ap.parse_args()

    if args.quick:
        PARAM_GRID = {k: [v[0], v[-1]] for k, v in PARAM_GRID.items()}

    data = sweep()
    report(data)
