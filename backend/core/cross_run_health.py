"""§v10.17 CrossRunHealth — persistiert Health-Daten über mehrere Pipeline-Läufe.

Erkennt Performance-Regressionen und warnt bei Verschlechterung.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEALTH_DIR = Path.home() / ".aurik" / "health"
_MAX_HISTORY: int = 50


def save_run_health(run_id: str, summary: dict[str, Any]) -> None:
    """Speichert Health-Summary eines Pipeline-Laufs."""
    try:
        _HEALTH_DIR.mkdir(parents=True, exist_ok=True)
        summary["timestamp"] = time.time()
        summary["run_id"] = run_id
        path = _HEALTH_DIR / f"{run_id[:12]}_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        # Cleanup old files
        files = sorted(_HEALTH_DIR.glob("*.json"), key=os.path.getmtime)
        for old in files[:-_MAX_HISTORY]:
            old.unlink()
    except Exception as _crh_save_exc:
        logger.debug(
            "cross_Ausfuehrung_health: speichern_Ausfuehrung_summary fehlgeschlagen (unkritisch): %s", _crh_save_exc
        )


def load_run_history(limit: int = 10) -> list[dict[str, Any]]:
    """Lädt die letzten N Health-Summaries."""
    try:
        files = sorted(_HEALTH_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
        results = []
        for f in files[:limit]:
            try:
                with open(f) as fh:
                    results.append(json.load(fh))
            except Exception as _crh_file_exc:
                logger.debug(
                    "cross_Ausfuehrung_health: laden_Ausfuehrung_history file read fehlgeschlagen (unkritisch): %s",
                    _crh_file_exc,
                )
        return results
    except Exception as _crh_load_exc:
        logger.debug(
            "cross_Ausfuehrung_health: laden_Ausfuehrung_history fehlgeschlagen (unkritisch): %s", _crh_load_exc
        )
        return []


def detect_regression(current: dict[str, Any]) -> list[str]:
    """Vergleicht mit historischen Läufen und warnt bei Regression."""
    warnings: list[Any] = []
    try:
        history = load_run_history(5)
        if not history:
            return warnings
        prev_avg_retries = sum(h.get("total_retries", 0) for h in history) / len(history)
        prev_avg_dur = sum(h.get("pipeline_duration_s", 0) for h in history) / len(history)
        cur_retries = current.get("total_retries", 0)
        cur_dur = current.get("pipeline_duration_s", 0)
        if cur_retries > prev_avg_retries * 1.5 and cur_retries > 10:
            warnings.append(f"Retry-Regression: {cur_retries} (avg: {prev_avg_retries:.0f})")
        if cur_dur > prev_avg_dur * 1.5 and cur_dur > 60:
            warnings.append(f"Duration-Regression: {cur_dur:.0f}s (avg: {prev_avg_dur:.0f}s)")
    except Exception as _crh_regress_exc:
        logger.debug("cross_Ausfuehrung_health: erkennen_regression fehlgeschlagen (unkritisch): %s", _crh_regress_exc)
    return warnings
