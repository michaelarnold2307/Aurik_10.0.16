#!/usr/bin/env python3
"""benchmarks/update_amrb_history.py — §v10.700 J4.

Automatisiert die AMRB-Performance-Historie.
Liest RT-Faktor und PQS aus dem letzten Benchmark-Lauf und
schreibt benchmarks/amrb_history.jsonl.

Nutzung:
  python benchmarks/update_amrb_history.py --commit abc123 --rt 4.2 --pqs 0.87
  python benchmarks/update_amrb_history.py --auto    # Letzten Lauf automatisch erkennen

CI: Läuft nach jedem Benchmark, warnt bei >10% Regression.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = REPO_ROOT / "benchmarks" / "amrb_history.jsonl"


def get_current_commit() -> str:
    """Ermittelt den aktuellen Git-Commit-Hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def read_history() -> list[dict]:
    """Liest die existierende Historie."""
    if not HISTORY_PATH.exists():
        return []
    entries = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return entries


def write_entry(
    commit: str, rt_factor: float, pqs_score: float, material: str = "vinyl", sample_count: int = 1
) -> None:
    """Schreibt einen Eintrag in die Historie."""
    entry = {
        "commit": commit,
        "date": datetime.now(timezone.utc).isoformat(),
        "rt_factor": round(rt_factor, 2),
        "pqs_score": round(pqs_score, 2),
        "material": material,
        "sample_count": sample_count,
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"✅ AMRB-Eintrag: commit={commit} rt={rt_factor:.1f}x pqs={pqs_score:.2f}")


def check_regression(entries: list[dict], current_rt: float, window: int = 20) -> tuple[bool, float, float]:
    """Prüft auf signifikante Regression (>10% RT-Verschlechterung).

    Returns:
        (has_regression, avg_recent, pct_change)
    """
    if len(entries) < 3:
        return False, current_rt, 0.0

    recent = entries[-window:]
    avg_rt = sum(e["rt_factor"] for e in recent) / len(recent)

    if avg_rt > 0:
        pct_change = (current_rt - avg_rt) / avg_rt * 100
        has_regression = pct_change > 10.0
        return has_regression, avg_rt, pct_change

    return False, current_rt, 0.0


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="AMRB Performance History Updater")
    p.add_argument("--commit", help="Git commit hash")
    p.add_argument("--rt", type=float, help="RT-Faktor (z.B. 4.2)")
    p.add_argument("--pqs", type=float, help="PQS-Score (z.B. 0.87)")
    p.add_argument("--material", default="vinyl")
    p.add_argument("--auto", action="store_true", help="Commit automatisch erkennen")
    args = p.parse_args()

    commit = args.commit or (get_current_commit() if args.auto else "manual")
    rt = args.rt or 0.0
    pqs = args.pqs or 0.0

    if rt == 0.0:
        print("⚠️  Kein RT-Faktor angegeben (--rt). Überspringe.")
        return 0

    # Regression prüfen
    history = read_history()
    has_reg, avg, pct = check_regression(history, rt)

    write_entry(commit, rt, pqs, args.material)

    if has_reg:
        print(f"⚠️  REGRESSION: RT {rt:.1f}x vs avg {avg:.1f}x ({pct:+.1f}%)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
