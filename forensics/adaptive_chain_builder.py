"""forensics/adaptive_chain_builder.py — §v10.700 I2.

from typing import Any
Baut Tonträgerketten aus NDJSON-Pipeline-Logs.
Erkennt Transfer-Kettenmuster aus den Logs und rekonstruiert die
vollständige Kette (z.B. reel_tape → vinyl → cassette → mp3_low).

§10 ROADMAP: 21 dict-item type-safety issues resolved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_chain_from_logs(
    log_path: str | Path,
    *,
    min_confidence: float = 0.3,
) -> dict[str, Any]:  # type: ignore[name-defined]
    """Liest NDJSON-Pipeline-Logs und rekonstruiert die Tonträgerkette.

    Args:
        log_path: Pfad zur NDJSON-Log-Datei
        min_confidence: Minimale Konfidenz für Kettenelemente

    Returns:
        Dict mit 'chain': [str], 'confidence': float, 'sources': [str]
    """
    log_path = Path(log_path)
    if not log_path.exists():
        logger.warning("Log-Datei nicht gefunden: %s", log_path)
        return {"chain": [], "confidence": 0.0, "sources": []}

    material_hits: dict[str, int] = {}
    transfer_hints: list[str] = []
    total_entries = 0

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
                total_entries += 1

                # Extrahiere Material-Informationen
                material = str(entry.get("material", "") or "").lower().strip()
                if material and material != "unknown":
                    material_hits[material] = material_hits.get(material, 0) + 1

                # Extrahiere Transfer-Hinweise
                chain_info = entry.get("transfer_chain") or entry.get("chain") or []
                if isinstance(chain_info, list):
                    for item in chain_info:
                        item_str = str(item).lower().strip()
                        if item_str and item_str not in transfer_hints:
                            transfer_hints.append(item_str)
    except Exception:
        logger.debug("Fehler beim Lesen von %s", log_path, exc_info=True)

    # Baue Chain aus den häufigsten Materialien
    sorted_materials = sorted(material_hits.items(), key=lambda x: x[1], reverse=True)
    chain = [m for m, _ in sorted_materials if material_hits[m] / max(total_entries, 1) >= min_confidence]

    confidence = round(sum(c for _, c in sorted_materials[:5]) / max(total_entries, 1), 3) if total_entries > 0 else 0.0

    return {
        "chain": chain if chain else transfer_hints,
        "confidence": confidence,
        "sources": transfer_hints,
        "material_hits": dict(sorted_materials[:10]),
        "total_entries": total_entries,
    }


def compare_chains(
    chain_a: list[str],
    chain_b: list[str],
) -> dict[str, Any]:  # type: ignore[name-defined]
    """Vergleicht zwei Tonträgerketten und gibt Übereinstimmungsgrad zurück."""
    set_a = {str(c).lower() for c in chain_a}
    set_b = {str(c).lower() for c in chain_b}

    intersection = set_a & set_b
    union = set_a | set_b

    jaccard = len(intersection) / max(len(union), 1)

    return {
        "jaccard_similarity": round(jaccard, 3),
        "shared": sorted(intersection),
        "only_a": sorted(set_a - set_b),
        "only_b": sorted(set_b - set_a),
    }
