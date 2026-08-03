#!/usr/bin/env python3
"""§v10.115 Post-Pipeline Forensik-Hook — automatisch nach jedem Lauf.

Verdrahtet ExceptionAggregator + PatternMiner + QualityRegressionDetector
mit der existierenden NDJSON-Pipeline.

Usage:
  # Manuell nach Pipeline-Lauf:
  python scripts/post_pipeline_forensics.py

  # In Pipeline integriert (wird von UV3._execute_pipeline am Ende aufgerufen):
  from scripts.post_pipeline_forensics import run_forensics
  run_forensics(q_score=result_quality_score)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def run_forensics(q_score: float | None = None, since: str | None = None) -> dict:
    """Führt die komplette Post-Pipeline-Forensik durch.

    Args:
        q_score: Optionaler Q-Score des abgeschlossenen Laufs.
        since: Optional ISO timestamp — nur Exceptions ab diesem Zeitpunkt.
               None = nur die letzten 24h (verhindert Akkumulation historischer Fehler).

    Returns:
        Dict mit forensics_summary, pattern_candidates, quality_trend.
    """
    from backend.core.exception_forensics import (
        ContinuousAnalyzer,
        ExceptionAggregator,
        PatternMiner,
    )
    from backend.core.quality_regression_detector import QualityRegressionDetector

    # §v10.711: Ohne since-Filter nur die letzten 24h zählen.
    # Verhindert dass 44 MB NDJSON-Historie als "570 Exceptions" gemeldet werden.
    if since is None:
        since = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)).isoformat()

    result: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forensics_summary": {},
        "pattern_candidates": [],
        "quality_trend": {},
    }

    # ── L1: Exception-Aggregation ────────────────────────────────────────
    agg = ExceptionAggregator()
    entries = agg.aggregate(since=since)  # §v10.711: Zeitgefiltert
    total = sum(e.count for e in entries)
    by_pattern = {}
    for e in entries:
        by_pattern[e.pattern_class] = by_pattern.get(e.pattern_class, 0) + e.count
    unclassified = by_pattern.get("UNCLASSIFIED", 0)
    unique = len(entries)
    top = [{"type": e.exception_type, "message": e.message[:120], "count": e.count} for e in entries[:10]]
    summary = {
        "total_exceptions": total,
        "unique_messages": unique,
        "by_pattern": by_pattern,
        "unclassified": unclassified,
        "top_exceptions": top,
    }

    logger.info(
        "§v10.115 Forensik: %d Exceptions, %d unique, %d unklassifiziert",
        summary["total_exceptions"],
        summary["unique_messages"],
        summary["unclassified"],
    )
    result["forensics_summary"] = summary

    # ── L4: Pattern-Mining ───────────────────────────────────────────────
    if summary["unclassified"] > 0:
        miner = PatternMiner(agg)
        candidates = miner.discover()

        # ── §v10.303.41 Auto-Promotion ──
        _auto_promoted = 0
        for c in candidates:
            if c.confidence >= 0.90 and c.exception_count >= 10:
                if c.regex_pattern and c.regex_pattern not in agg.KNOWN_PATTERNS:
                    agg.KNOWN_PATTERNS[c.regex_pattern] = c.temporary_id
                    _auto_promoted += 1
                    logger.info(
                        "§v10.303.41 Auto-Promotion: %s → %s (conf=%.2f, count=%d)",
                        c.regex_pattern[:60],
                        c.temporary_id,
                        c.confidence,
                        c.exception_count,
                    )
        if _auto_promoted:
            logger.info("§v10.303.41 Auto-Promotion: %d Pattern(s) in KNOWN_PATTERNS aufgenommen", _auto_promoted)

        if candidates:
            # Schreibe entdeckte Patterns für den Scanner
            pattern_feed = REPO_ROOT / "logs" / "discovered_patterns.json"
            pattern_data = {
                "generated_at": result["timestamp"],
                "source": "PatternMiner §v10.115",
                "total_exceptions_analyzed": summary["total_exceptions"],
                "patterns": [
                    {
                        "id": c.temporary_id,
                        "description": c.description,
                        "regex": c.regex_pattern,
                        "confidence": c.confidence,
                        "exception_count": c.exception_count,
                        "affected_files": c.affected_files[:5],
                        "status": "candidate",
                        "message": f"{c.temporary_id} {c.description} (→ {c.regex_pattern})",
                        "scan_roots": ["backend/core", "plugins", "Aurik10", "scripts"],
                    }
                    for c in candidates
                ],
            }
            with open(pattern_feed, "w") as f:
                json.dump(pattern_data, f, indent=2)

            logger.info(
                "§v10.115 Pattern-Miner: %d neue Pattern-Kandidaten → %s",
                len(candidates),
                pattern_feed,
            )
            result["pattern_candidates"] = [{"id": c.temporary_id, "confidence": c.confidence} for c in candidates]

    # ── L5: Q-Score-Korrelation ─────────────────────────────────────────
    qrd = QualityRegressionDetector()
    if q_score is not None:
        qrd.record(q_score)
        logger.info("§v10.115 Q-Wert aufgezeichnet: %.4f", q_score)

    comparison = qrd.compare()
    if comparison.get("status") == "ok":
        result["quality_trend"] = comparison
        if comparison.get("regression_detected"):
            logger.warning(
                "§v10.115 ⚠️ Quality Regression: ΔQ = %+.4f, ΔExc = %+d",
                comparison["q_score_delta"],
                comparison["exception_delta"],
            )

    # ── L6: Continuous Analysis ─────────────────────────────────────────
    analyzer = ContinuousAnalyzer(agg)
    incr = analyzer.analyze_new()
    result["continuous_analysis"] = incr

    # ── Logge Zusammenfassung ────────────────────────────────────────────
    top = summary.get("top_exceptions", [])
    if top:
        logger.info("§v10.115 Top-Exception: %s (%dx)", top[0]["message"][:80], top[0]["count"])

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Post-Pipeline Forensik-Hook")
    parser.add_argument("--qscore", type=float, help="Q-Score des letzten Laufs")
    parser.add_argument("--json", action="store_true", help="JSON-Output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = run_forensics(q_score=args.qscore)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        summary = result["forensics_summary"]
        print("\n🔬 §v10.115 Post-Pipeline Forensik")
        print(f"   Exceptions: {summary['total_exceptions']} total, {summary['unique_messages']} unique")
        print(f"   Unklassifiziert: {summary['unclassified']}")
        print(f"   Pattern-Kandidaten: {len(result['pattern_candidates'])}")

        qt = result.get("quality_trend", {})
        if qt.get("status") == "ok":
            print(f"   Q-Score: {qt['current_q_score']:.4f} (Δ={qt['q_score_delta']:+.4f})")
            if qt.get("regression_detected"):
                print("   ⚠️  QUALITY REGRESSION DETECTED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
