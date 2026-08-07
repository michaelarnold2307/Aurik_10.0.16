#!/usr/bin/env python3
"""Monatlicher Benchmark-Trend-Report — §15.1 Punkt 1.5.

Liest alle gespeicherten Benchmark-Ergebnisse aus benchmarks/competitive/results/
und erstellt einen Trend-Report (JSON + Markdown). Detektiert automatisch
Regressionen (PQS-Delta sinkt unter Vorperiode).

Nutzung:
  python scripts/monthly_benchmark_trend.py
  python scripts/monthly_benchmark_trend.py --output reports/trend_2026-08.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_ROOT = REPO_ROOT / "benchmarks" / "competitive" / "results"


def collect_results() -> list[dict]:
    """Sammelt alle oss_summary.json Dateien chronologisch."""
    entries: list[Any] = []
    if not RESULTS_ROOT.exists():
        return entries
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        sf = d / "oss_summary.json"
        if not sf.exists():
            continue
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = data.get("summary", {})
        if not summary:
            continue
        entry = {
            "date": str(d.name)[:10],  # YYYYMMDD_HHMMSS → YYYY-MM-DD
            "dir": str(d.name),
            **summary,
        }
        entries.append(entry)
    return entries


def detect_regressions(entries: list[dict]) -> list[str]:
    """Erkennt Regressionen: PQS-Delta-Verschlechterung > 2 Punkte zum Vormonat."""
    if len(entries) < 2:
        return []
    warnings = []
    for i in range(1, len(entries)):
        prev_delta = entries[i - 1].get("mean_delta", 0)
        curr_delta = entries[i].get("mean_delta", 0)
        drop = prev_delta - curr_delta
        if drop > 2.0:
            warnings.append(
                f"REGRESSION {entries[i - 1]['date']} → {entries[i]['date']}: "
                f"PQS-Δ von {prev_delta:+.2f} auf {curr_delta:+.2f} "
                f"(Verschlechterung um {drop:+.2f} Punkte)"
            )
        # Auch prüfen: win rate drop > 15%
        prev_wr = entries[i - 1].get("wins", 0) / max(entries[i - 1].get("total", 1), 1)
        curr_wr = entries[i].get("wins", 0) / max(entries[i].get("total", 1), 1)
        wr_drop = prev_wr - curr_wr
        if wr_drop > 0.15:
            warnings.append(
                f"WIN-RATE-REGRESSION {entries[i - 1]['date']} → {entries[i]['date']}: "
                f"Win-Rate von {prev_wr:.1%} auf {curr_wr:.1%} "
                f"(−{wr_drop:.1%})"
            )
    return warnings


def generate_markdown(entries: list[dict], warnings: list[str]) -> str:
    """Erstellt einen Markdown-Trend-Report."""
    lines = [
        "# Aurik Open-Source Benchmark — Monats-Trend",
        "",
        f"**Generiert:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Datenpunkte:** {len(entries)}",
        "",
        "---",
        "",
    ]

    if warnings:
        lines.append("## ⚠️ Regression-Warnungen")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Trend-Tabelle")
    lines.append("")
    lines.append("| Datum | Szenarien | Aurik-Wins | Tools | PQS-Δ Ø | Best Δ | RT Aurik | RT Tools |")
    lines.append("|-------|-----------|------------|-------|---------|--------|----------|----------|")
    for e in entries:
        lines.append(
            f"| {e.get('date', '?')} "
            f"| {e.get('total', 0)} "
            f"| {e.get('wins', 0)} "
            f"| {e.get('losses', 0)} "
            f"| {e.get('mean_delta', 0):+.2f} "
            f"| {e.get('best_delta', 0):+.2f} "
            f"| {e.get('mean_rt_a', 0):.2f}s "
            f"| {e.get('mean_rt_t', 0):.2f}s |"
        )

    lines.extend(
        [
            "",
            "## Metriken",
            "",
            "- **PQS**: Perceptual Quality Score (0-100, höher = besser)",
            "- **PQS-Δ**: Aurik PQS minus Tool PQS (positiv = Aurik gewinnt)",
            "- **RT**: Real-Time-Faktor (Sekunden Verarbeitung pro Sekunde Audio)",
            "- **Best Δ**: Bestes Aurik-Ergebnis im Vergleich",
            "",
            "## Interpretation",
            "",
            "- PQS-Δ > 0: Aurik übertrifft das Vergleichs-Tool",
            "- PQS-Δ ≈ 0: Gleichauf (Tie-Bereich: |Δ| < 0.5)",
            "- PQS-Δ < 0: Tool übertrifft Aurik — Analyse erforderlich",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monatlicher Benchmark-Trend-Report")
    parser.add_argument("--output", help="Ausgabedatei (Markdown)", default=None)
    parser.add_argument("--json", action="store_true", help="JSON-Ausgabe")
    args = parser.parse_args()

    entries = collect_results()
    warnings = detect_regressions(entries)

    if args.json:
        data = {
            "entries": entries,
            "warnings": warnings,
            "generated": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        if args.output:
            Path(args.output).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return 1 if warnings else 0

    md = generate_markdown(entries, warnings)
    print(md)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"\nReport gespeichert: {args.output}")

    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
