#!/usr/bin/env python3
"""§v10.115 Exception-Forensik Dashboard — CLI für Exception-Trends und Q-Score.

Usage:
  python scripts/forensics_dashboard.py                  # Zusammenfassung
  python scripts/forensics_dashboard.py --trends         # Trends (letzte 10 Läufe)
  python scripts/forensics_dashboard.py --patterns       # Pattern-Mining
  python scripts/forensics_dashboard.py --qscore         # Q-Score-Korrelation
  python scripts/forensics_dashboard.py --full           # Vollständiger Report
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NDJSON_PATH = REPO_ROOT / "logs" / "oom_phase_forensics.ndjson"


def load_entries() -> list[dict]:
    if not NDJSON_PATH.exists():
        return []
    entries = []
    with open(NDJSON_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize(entries: list[dict]):
    """Gesamtübersicht."""
    if not entries:
        print("⚠️  Keine Forensik-Daten gefunden.")
        print(f"   Erwartet: {NDJSON_PATH}")
        return

    total = len(entries)
    by_stage = Counter(e.get("stage", "unknown") for e in entries)
    errors = [e for e in entries if e.get("stage", "").startswith(("phase_failed", "phase_exception"))]
    phases = {e.get("phase_id", "?") for e in entries if e.get("phase_id", "").startswith("phase_")}

    print(f"\n{'=' * 70}")
    print("  🔬 Exception-Forensik Dashboard — §v10.115")
    print(f"{'=' * 70}")
    print(f"  Daten:       {NDJSON_PATH}")
    print(f"  Einträge:    {total:,}")
    print(f"  Phasen:      {len(phases)} unique")
    print(f"  Fehler:      {len(errors):,} ({len(errors) / max(total, 1) * 100:.1f}%)")
    print()

    print("  Top-10 Phasen nach Einträgen:")
    by_phase = Counter(e.get("phase_id", "?") for e in entries)
    for phase, count in by_phase.most_common(10):
        bar = "█" * min(40, count * 40 // max(1, by_phase.most_common(1)[0][1]))
        print(f"    {phase:<30} {count:>5d}  {bar}")

    if errors:
        print("\n  Fehler-Typen:")
        err_types = Counter(e.get("error", "?").split(":")[0] for e in errors)
        for etype, count in err_types.most_common(8):
            print(f"    {etype:<40} {count:>5d}")


def trends(entries: list[dict]):
    """Trend-Analyse über Zeit."""
    if not entries:
        print("⚠️  Keine Daten für Trend-Analyse.")
        return

    # Group by run (pipeline_start events)
    runs = defaultdict(list)
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                run_key = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                run_key = ts[:16]
        else:
            run_key = "unknown"
        runs[run_key].append(e)

    print(f"\n{'=' * 70}")
    print(f"  📈 Exception-Trends (letzte {min(10, len(runs))} Läufe)")
    print(f"{'=' * 70}")

    sorted_runs = sorted(runs.items(), reverse=True)[:10]
    for run_key, run_entries in sorted_runs:
        total = len(run_entries)
        errors = sum(1 for e in run_entries if "failed" in e.get("stage", "") or "exception" in e.get("stage", ""))
        error_pct = errors / max(total, 1) * 100
        bar = "█" * min(30, int(error_pct))
        status = "🟢" if error_pct < 5 else ("🟡" if error_pct < 15 else "🔴")
        print(f"  {status} {run_key}: {total:>4d} Einträge, {errors:>3d} Fehler ({error_pct:4.1f}%) {bar}")

    # Trend direction
    if len(sorted_runs) >= 2:
        last_errors = sum(
            1 for e in sorted_runs[0][1] if "failed" in e.get("stage", "") or "exception" in e.get("stage", "")
        )
        prev_errors = sum(
            1 for e in sorted_runs[1][1] if "failed" in e.get("stage", "") or "exception" in e.get("stage", "")
        )
        last_total = max(len(sorted_runs[0][1]), 1)
        prev_total = max(len(sorted_runs[1][1]), 1)
        delta = (last_errors / last_total - prev_errors / prev_total) * 100
        direction = "📉 verbessert" if delta < -2 else ("📈 verschlechtert" if delta > 2 else "➡️ stabil")
        print(f"\n  Trend: {direction} (Δ = {delta:+.1f} Prozentpunkte)")


def patterns(entries: list[dict]):
    """Pattern-Mining: entdeckt neue Bug-Klassen aus NDJSON."""
    from backend.core.exception_forensics import ExceptionAggregator, PatternMiner

    agg = ExceptionAggregator()
    for e in entries:
        agg.add_raw(e)  # type: ignore[attr-defined]

    report = agg.aggregate()
    miner = PatternMiner()  # type: ignore[call-arg]
    discoveries = miner.mine(report)  # type: ignore[attr-defined]

    print(f"\n{'=' * 70}")
    print("  🧠 Pattern-Mining — §v10.115")
    print(f"{'=' * 70}")
    print(f"  Bekannte Patterns: {len(miner.KNOWN_PATTERNS)}")  # type: ignore[attr-defined]
    print(f"  Neue Entdeckungen: {len(discoveries)}")
    print()

    for i, (pattern_name, pattern) in enumerate(discoveries.items(), 1):
        print(f"  P{7 + i}: {pattern_name}")
        print(f"       Count: {pattern.get('count', '?')}")
        print(f"       Regex: {pattern.get('regex', '?')}")
        print(f"       Suggestion: {pattern.get('suggestion', '?')}")
        print()


def qscore_correlation(entries: list[dict]):
    """Q-Score-Korrelation: misst ob Fixes den Score verbessern."""
    from backend.core.exception_forensics import QScoreMonitor  # type: ignore[attr-defined]

    monitor = QScoreMonitor()

    # Extract phase-level data
    phases = defaultdict(list)
    for e in entries:
        pid = e.get("phase_id", "?")
        if pid.startswith("phase_"):
            phases[pid].append(e)

    print(f"\n{'=' * 70}")
    print("  📊 Q-Score-Korrelation — §v10.115")
    print(f"{'=' * 70}")
    print(f"  Phasen mit Daten: {len(phases)}")
    print()

    # Top phases by error rate (potential quality impact)
    phase_quality = []
    for pid, p_entries in phases.items():
        total = len(p_entries)
        errors = sum(1 for e in p_entries if "failed" in e.get("stage", "") or "exception" in e.get("stage", ""))
        error_rate = errors / max(total, 1)
        # Estimate quality impact: phases with high error rates likely hurt quality
        quality_impact = monitor.estimate_quality_impact(pid, error_rate)
        phase_quality.append((pid, error_rate, quality_impact, total))

    phase_quality.sort(key=lambda x: x[2], reverse=True)

    print("  Top-10 Phasen nach Quality Impact:")
    print(f"  {'Phase':<30} {'Err%':>6} {'Q-Impact':>8} {'#Runs':>6}")
    print(f"  {'─' * 30} {'─' * 6} {'─' * 8} {'─' * 6}")
    for pid, rate, impact, total in phase_quality[:10]:
        bar = "🟢" if impact < 0.1 else ("🟡" if impact < 0.3 else "🔴")
        print(f"  {bar} {pid:<28} {rate * 100:>5.1f}% {impact:>7.3f} {total:>5d}")


def main():
    parser = argparse.ArgumentParser(description="Exception-Forensik Dashboard")
    parser.add_argument("--trends", action="store_true", help="Trend-Analyse")
    parser.add_argument("--patterns", action="store_true", help="Pattern-Mining")
    parser.add_argument("--qscore", action="store_true", help="Q-Score-Korrelation")
    parser.add_argument("--full", action="store_true", help="Vollständiger Report")
    args = parser.parse_args()

    entries = load_entries()

    if not entries:
        print("⚠️  Keine Forensik-Daten gefunden.")
        print(f"   Erwartet: {NDJSON_PATH}")
        print("   Führe einen Pipeline-Lauf aus, um Daten zu generieren.")
        return 1

    if args.full or (not args.trends and not args.patterns and not args.qscore):
        summarize(entries)
    if args.full or args.trends:
        trends(entries)
    if args.full or args.patterns:
        patterns(entries)
    if args.full or args.qscore:
        qscore_correlation(entries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
