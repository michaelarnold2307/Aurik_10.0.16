#!/usr/bin/env python3
"""scripts/perf_trend.py — §v10.700 J4: Performance-Trend-Analyse.

Liest amrb_history.jsonl und erkennt signifikante Regressionen
via linearer Regression auf den letzten 20 Einträgen.

Nutzung:
  python scripts/perf_trend.py          # Trend-Report
  python scripts/perf_trend.py --ci      # CI-Mode: Exit 1 bei Degradation
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HISTORY = Path(__file__).parent.parent / "benchmarks" / "amrb_history.jsonl"


def main():
    if not HISTORY.exists():
        print("Keine History vorhanden — tracking startet mit nächstem Benchmark-Lauf.")
        return 0

    entries = []
    with open(HISTORY) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    if len(entries) < 3:
        print(f"Trend: insufficient_data ({len(entries)} Einträge)")
        return 0

    recent = entries[-20:]
    x = np.arange(len(recent), dtype=np.float64)
    rts = [e["rt_factor"] for e in recent]
    slope = float(np.polyfit(x, rts, 1)[0])

    status = "✅" if slope < 0.1 else ("⚠️" if slope < 0.5 else "❌")
    print(f"{status} RT-Trend: {slope:+.2f}s/Eintrag (letzte {len(recent)} Läufe)")
    print(f"   Erster: RT={rts[0]:.1f}, Letzter: RT={rts[-1]:.1f}")

    return 1 if slope > 0.5 else 0


if __name__ == "__main__":
    sys.exit(main())
