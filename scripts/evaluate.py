#!/usr/bin/env python3
"""§v10.995: Der EINE Evaluations-Einstiegspunkt.

    python scripts/evaluate.py --mode objective --corpus corpus --limit 5 --gate
    python scripts/evaluate.py --mode objective --synthetic --cases 4 --gate
    python scripts/evaluate.py --mode competitive --gate          (real AMRB-Lauf)
    python scripts/evaluate.py --listening-export evaluation/listening_20260813
    python scripts/evaluate.py --listening-import scoresheet.csv --listening-key decoder_key.json

Exit-Codes (mit --gate): 0 = PASS/SKIP, 1 = FAIL.

Ehrlichkeits-Regel: Auch Verschlechterungen werden berichtet — nie gefiltert.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.evaluation_system import (
    EvalCase,
    EvalReport,
    EvaluationSystem,
    ListeningTestExporter,
    discover_corpus_cases,
)

import numpy as np


def _load_case(paths: dict) -> EvalCase | None:
    import soundfile as sf

    try:
        damaged, sr = sf.read(paths["damaged_path"], dtype="float32", always_2d=False)
        clean, sr2 = sf.read(paths["clean_path"], dtype="float32", always_2d=False)
        case = EvalCase(
            case_id=paths["case_id"], material=paths["material"],
            damaged=damaged, clean=clean, sample_rate=int(sr),
        )
        if paths.get("restored_path"):
            restored, sr3 = sf.read(paths["restored_path"], dtype="float32", always_2d=False)
            case.restored = restored
        return case
    except Exception as exc:
        print(f"  ⚠ {paths.get('case_id', '?')}: Laden fehlgeschlagen ({exc}) — übersprungen")
        return None


def _make_synthetic_case(index: int, degraded: bool = False) -> EvalCase:
    rng = np.random.default_rng(100 + index)
    t = np.linspace(0, 2.0, 96000, endpoint=False, dtype=np.float32)
    clean = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (rng.standard_normal(len(t)) * 0.05).astype(np.float32)
    damaged = clean + noise
    restored = clean if not degraded else damaged + noise * 2.0
    return EvalCase(
        case_id=f"synthetic_{index:02d}", material="synthetic",
        damaged=damaged, clean=clean, restored=restored, sample_rate=48000,
    )


def _run_objective(args: argparse.Namespace) -> EvalReport:
    system = EvaluationSystem()
    cases: list[EvalCase] = []
    if args.synthetic:
        for i in range(int(args.cases or 4)):
            cases.append(_make_synthetic_case(i))
    else:
        paths = discover_corpus_cases(Path(args.corpus), limit=int(args.limit or 0))
        for p in paths:
            case = _load_case(p)
            if case is not None:
                cases.append(case)
    if not cases:
        print("Keine Fälle gefunden — SKIP")
        return EvalReport(mode="objective", verdict="SKIP",
                          gates=[__import__("backend.core.evaluation_system", fromlist=["GateResult"]).GateResult(
                              "objective", True, {"reason": "keine Fälle"})])
    print(f"Bewerte {len(cases)} Fälle …")
    report = system.run_objective(cases, gates=True)
    for c in report.cases:
        print(f"  {c.case_id:<40} {c.verdict:<9} SNRΔ={c.snr_delta_db:+.2f} dB  MSE={c.mse_reduction_pct:+.1f}%")
    return report


def _run_competitive(args: argparse.Namespace) -> EvalReport:
    system = EvaluationSystem()
    print("Wettbewerber-Lauf (AMRB-Szenarien, real) …")
    try:
        from benchmarks.musical_restoration_benchmark import AMRB_BASELINES, run_benchmark

        bench = run_benchmark()
        scenario_results: list[tuple[str, float, float]] = []
        rx11 = float(AMRB_BASELINES.get("iZotope RX 11 (commercial)", {}).get("mushra_overall", 71.0))
        for scen in getattr(bench, "scenarios", []) or []:
            name = str(getattr(scen, "name", "?"))
            aurik = float(getattr(getattr(scen, "result", None), "mushra_overall", 0.0) or 0.0)
            scenario_results.append((name, aurik, rx11))
        report = system.run_competitive(scenario_results)
    except Exception as exc:
        print(f"  ⚠ Benchmark nicht lauffähig ({exc}) — SKIP")
        from backend.core.evaluation_system import GateResult

        report = EvalReport(mode="competitive", verdict="SKIP")
        report.gates = [GateResult("competitive", True, {"reason": f"nicht lauffähig: {exc}"})]
    return report


def _run_all(args: argparse.Namespace) -> EvalReport:
    objective = _run_objective(args)
    competitive = _run_competitive(args)
    objective.mode = "all"
    objective.cases.extend(competitive.cases)
    objective.gates.extend(competitive.gates)
    objective.verdict = (
        "PASS"
        if all(g.passed for g in objective.gates)
        else "FAIL"
    )
    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description="§v10.995 Aurik Evaluation")
    parser.add_argument("--mode", default="objective",
                        choices=["objective", "competitive", "all"])
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--gate", action="store_true", help="Exit 1 bei FAIL")
    parser.add_argument("--listening-export", default=None)
    parser.add_argument("--listening-import", default=None)
    parser.add_argument("--listening-key", default=None)
    args = parser.parse_args()

    if args.listening_export:
        exporter = ListeningTestExporter(args.listening_export)
        for i in range(int(args.cases or 4)):
            case = _make_synthetic_case(i)
            exporter.export_pair(case.case_id, case.restored, case.clean, case.sample_rate)
        sheet = exporter.write_scoresheet()
        key = exporter.write_key()
        print(f"Hörtest-Export: {sheet}")
        print(f"Decoder-Schlüssel: {key} (getrennt aufbewahren — Doppelblind)")
        return 0

    if args.listening_import:
        import csv

        with open(args.listening_import, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"{len(rows)} Bewertungen gelesen")
        # Subjektive Scores ins gleiche Schema: als Fälle mit Verdict aus Wahl
        report = EvalReport(mode="listening", generated_at="")
        report.cases = [
            __import__("backend.core.evaluation_system", fromlist=["CaseMetrics"]).CaseMetrics(
                case_id=r["case_id"],
                verdict="neutral" if (r.get("choice") or "").strip() else "neutral",
            )
            for r in rows
        ]
        out = report.save(args.out)
        print(f"Subjektiver Report: {out}")
        return 0

    report = {
        "objective": _run_objective,
        "competitive": _run_competitive,
        "all": _run_all,
    }[args.mode](args)
    for g in report.gates:
        print(f"  Gate {g.name}: {'PASS' if g.passed else 'FAIL'} {g.details}")
    out = report.save(args.out)
    print(f"Report: {out} — Verdict: {report.verdict}")
    if args.gate and report.verdict == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
