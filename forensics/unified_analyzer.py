"""forensics/unified_analyzer.py — §v10.700 I2.

Vereinheitlichte Analyse über alle Log-Quellen.
Kombiniert adaptive_chain_builder + feature_extractor in einer Pipeline.

§10 ROADMAP: 14 mixed-type issues resolved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forensics.adaptive_chain_builder import build_chain_from_logs
from forensics.feature_extractor import compute_statistics, extract_features

logger = logging.getLogger(__name__)


def analyze_logs(
    log_path: str | Path,
    *,
    min_confidence: float = 0.3,
) -> dict[str, Any]:
    """Führt die vereinheitlichte Forensik-Analyse auf einer Log-Datei durch.

    Returns:
        Dict mit 'chain', 'features', 'statistics', 'summary'.
    """
    log_path = Path(log_path)

    # Chain-Analyse
    chain_result = build_chain_from_logs(log_path, min_confidence=min_confidence)

    # Feature-Extraktion
    features = extract_features(log_path)

    # Statistik
    statistics = compute_statistics(features)

    # Summary
    total_phases = len(features)
    durations = [float(f.get("duration_s", 0.0)) for f in features]
    total_duration = round(sum(durations), 2)

    return {
        "log_path": str(log_path),
        "chain": chain_result,
        "total_phases": total_phases,
        "total_duration_s": total_duration,
        "statistics": statistics,
        "summary": _build_summary(chain_result, features, statistics),
    }


def _build_summary(
    chain: dict[str, Any],
    features: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> str:
    """Erstellt eine menschenlesbare Zusammenfassung."""
    parts = []

    chain_list = chain.get("chain", [])
    if chain_list:
        parts.append(f"Tonträgerkette: {' → '.join(chain_list)} (Konfidenz: {chain.get('confidence', 0):.0%})")

    parts.append(f"Phasen analysiert: {len(features)}")

    if statistics:
        snr_stats = statistics.get("snr_db", {})
        if snr_stats:
            parts.append(f"SNR: {snr_stats.get('mean', 0):.1f} ± {snr_stats.get('std', 0):.1f} dB")

        quality_stats = statistics.get("quality_delta", {})
        if quality_stats:
            parts.append(f"Quality-Δ: {quality_stats.get('mean', 0):+.3f}")

    return " | ".join(parts)
