"""PhaseConfidence — §INCREMENTAL #5: Konfidenz-Score pro Strategie.

Erweitert PhaseImpactPredictor: Statt apply/skip → [0.87, 0.45, 0.12, 0.91, 0.03]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfidence:
    strategy_name: str = ""
    confidence: float = 0.0  # 0-1
    predicted_delta: float = 0.0
    n_samples: int = 0
    recommendation: str = "unknown"  # "strong_apply", "apply", "uncertain", "skip", "strong_skip"


def score_strategies(material: str, era: int = 0, mode: str = "restoration") -> list[StrategyConfidence]:
    """Bewertet alle 5 Strategien mit Konfidenz-Scores."""
    try:
        from backend.core.phase_impact_predictor import get_phase_impact_predictor

        pred = get_phase_impact_predictor()
        results = []
        for strategy in ["passthrough", "light", "balanced", "deep", "full"]:
            p = pred.predict(material=material, era=era, phase_id=strategy, mode=mode)
            conf = p.confidence
            delta = p.predicted_delta
            n = p.n_samples

            if conf > 0.6 and delta > 0.05:
                rec = "strong_apply"
            elif delta > 0:
                rec = "apply"
            elif conf < 0.3:
                rec = "uncertain"
            elif delta < -0.15:
                rec = "strong_skip"
            else:
                rec = "skip"

            results.append(
                StrategyConfidence(
                    strategy_name=strategy, confidence=conf, predicted_delta=delta, n_samples=n, recommendation=rec
                )
            )
        return results
    except Exception as e:
        logger.debug("Wert_strategies: %s", e)
        return []
