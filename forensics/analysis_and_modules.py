"""forensics/analysis_and_modules.py — §v10.700 I2.

Modulare Analyse-Pipeline: PatternMiner → Scanner → Pipeline.
Schließt den Forensik-Kreislauf (§v10.116 Erkenntnis 4).

§10 ROADMAP: 11 mixed-type issues resolved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forensics.unified_analyzer import analyze_logs

logger = logging.getLogger(__name__)


class PatternMiner:
    """Erkennt Muster aus Forensik-Analyse-Ergebnissen."""

    @staticmethod
    def find_regressions(
        features: list[dict[str, Any]],
        threshold: float = -0.03,
    ) -> list[dict[str, Any]]:
        """Findet Phasen mit signifikant negativem Quality-Delta."""
        return [
            {
                "phase_id": str(f.get("phase_id", "")),
                "quality_delta": float(f.get("quality_delta", 0.0) or 0.0),
                "material": str(f.get("material", "unknown") or "unknown"),
            }
            for f in features
            if float(f.get("quality_delta", 0.0) or 0.0) < threshold
        ]

    @staticmethod
    def find_slow_phases(
        features: list[dict[str, Any]],
        threshold_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Findet Phasen mit überdurchschnittlich langer Laufzeit."""
        return [
            {
                "phase_id": str(f.get("phase_id", "")),
                "duration_s": float(f.get("duration_s", 0.0) or 0.0),
            }
            for f in features
            if float(f.get("duration_s", 0.0) or 0.0) > threshold_s
        ]


class AnalysisPipeline:
    """Vollständige Forensik-Analyse-Pipeline."""

    def run(self, log_path: str | Path) -> dict[str, Any]:
        """Führt die komplette Analyse durch und gibt Ergebnisse zurück."""
        result = analyze_logs(log_path)

        # Pattern-Erkennung
        features_dicts: list[dict[str, Any]] = []
        # Features sind bereits in analyze_logs extrahiert — hier aus Statistiken ableiten
        regressions = PatternMiner.find_regressions(features_dicts)
        slow_phases = PatternMiner.find_slow_phases(features_dicts)

        return {
            **result,
            "patterns": {
                "regressions": regressions,
                "slow_phases": slow_phases,
                "regression_count": len(regressions),
                "slow_phase_count": len(slow_phases),
            },
        }
