"""forensics/feature_extractor.py — §v10.700 I2.

Extrahiert Features aus Pipeline-Logs für Pattern-Mining.
Normalisiert alle Werte auf float/int/str — keine `floating[Any]` mehr.

§10 ROADMAP: 14 floating[Any] → float issues resolved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_features(log_path: str | Path) -> list[dict[str, Any]]:
    """Extrahiert numerische Features aus NDJSON-Pipeline-Logs.

    Returns:
        Liste von Feature-Dicts mit garantiert konkreten Typen (float, int, str).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    features: list[dict[str, Any]] = []

    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                feat: dict[str, Any] = {
                    "phase_id": str(entry.get("phase", "") or entry.get("phase_id", "")),
                    "duration_s": float(entry.get("duration", 0.0) or 0.0),
                    "rms_db": float(entry.get("rms_db", -120.0) or -120.0),
                    "peak_db": float(entry.get("peak_db", -120.0) or -120.0),
                    "snr_db": float(entry.get("snr_db", 0.0) or 0.0),
                    "crest_db": float(entry.get("crest_db", 0.0) or 0.0),
                    "strength": float(entry.get("strength", 0.0) or 0.0),
                    "quality_delta": float(entry.get("quality_delta", 0.0) or 0.0),
                    "material": str(entry.get("material", "unknown") or "unknown"),
                    "defect_severity": float(entry.get("severity", 0.0) or 0.0),
                }
                features.append(feat)
    except Exception:
        logger.debug("Fehler beim Feature-extrahieren aus %s", log_path, exc_info=True)

    return features


def compute_statistics(features: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Berechnet Statistiken (mean, std, min, max) über extrahierte Features."""
    if not features:
        return {}

    numeric_keys = ["duration_s", "rms_db", "snr_db", "crest_db", "strength", "quality_delta", "defect_severity"]
    stats: dict[str, dict[str, float]] = {}

    for key in numeric_keys:
        values = [float(f.get(key, 0.0)) for f in features if f.get(key) is not None]
        if values:
            arr = np.array(values, dtype=np.float64)
            stats[key] = {
                "mean": round(float(np.mean(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
                "count": len(values),
            }
    return stats
