"""
PhaseImpactPredictor — §CROWN Proaktive Phasen-Steuerung
==========================================================

Fragt VOR jeder Phase die KnowledgeBase: „Hat diese Phase bei diesem
Material und dieser Era historisch geholfen oder geschadet?"

Nutzt den PhaseImpactRecorder als Datenquelle und gibt eine
Empfehlung zurück: APPLY, REDUCE, oder SKIP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImpactPrediction:
    """Vorhersage: Wird diese Phase helfen oder schaden?"""

    phase_id: str = ""
    action: str = "apply"  # "apply", "reduce", "skip"
    predicted_delta: float = 0.0  # Erwartete Qualitätsänderung
    confidence: float = 0.0  # 0–1 (0=keine Daten, 1=sehr sicher)
    n_samples: int = 0  # Anzahl historischer Datenpunkte
    recommended_strength: float = 1.0  # Empfohlene Stärke
    reason: str = ""


class PhaseImpactPredictor:
    """Proaktiver Phasen-Berater — lernt aus jeder Restaurierung.

    Verwendung:
        predictor = PhaseImpactPredictor()
        pred = predictor.predict(material="vinyl", era=1970, phase_id="phase_19_de_esser")
        if pred.action == "skip":
            logger.info("Phase 19 wird übersprungen — historisch schädlich für Vinyl/1970er")
    """

    MIN_SAMPLES_FOR_PREDICTION: int = 3  # Mindestens 3 Datenpunkte
    SKIP_THRESHOLD: float = -0.15  # Delta < -0.15 → skip
    REDUCE_THRESHOLD: float = -0.05  # Delta < -0.05 → reduce strength
    STRONG_CONFIDENCE_N: int = 5  # N samples für confidence ≥ 0.7

    def __init__(self) -> None:
        self._cache: dict[tuple, ImpactPrediction] = {}
        self._cache_loaded: bool = False

    def predict(
        self,
        material: str,
        era: int | None,
        phase_id: str,
        mode: str = "restoration",
    ) -> ImpactPrediction:
        """Sagt vorher, ob eine Phase helfen wird.

        Args:
            material: Material-Typ (z.B. "vinyl", "shellac")
            era: Jahrzehnt (z.B. 1970, 1980)
            phase_id: Phase-ID (z.B. "phase_19_de_esser")
            mode: "restoration" oder "studio_2026"

        Returns:
            ImpactPrediction mit action, confidence, und Begründung
        """
        key = (material, era or 0, phase_id, mode)

        # Cache-Check
        if key in self._cache:
            return self._cache[key]

        # Lade historische Daten
        records = self._load_impact_data(material, era, phase_id, mode)
        n = len(records)

        if n < self.MIN_SAMPLES_FOR_PREDICTION:
            pred = ImpactPrediction(
                phase_id=phase_id,
                action="apply",
                predicted_delta=0.0,
                confidence=0.0,
                n_samples=n,
                recommended_strength=1.0,
                reason=f"Zu wenige Daten ({n} < {self.MIN_SAMPLES_FOR_PREDICTION}) — Phase normal ausführen",
            )
            self._cache[key] = pred
            return pred

        # Berechne durchschnittliches Delta
        avg_delta = sum(r.get("delta", 0.0) for r in records) / n

        # Confidence basierend auf Stichprobengröße
        confidence = min(1.0, n / self.STRONG_CONFIDENCE_N)

        recommended_strength = 1.0
        # Entscheidung
        if avg_delta <= self.SKIP_THRESHOLD:
            action = "skip"
            reason = (
                f"Phase historisch schädlich: avg Δ={avg_delta:.3f} (≤ {self.SKIP_THRESHOLD}) über {n} Restaurierungen"
            )
        elif avg_delta <= self.REDUCE_THRESHOLD:
            action = "reduce"
            # Reduziere Stärke proportional zum Delta
            recommended_strength = max(0.3, 1.0 + avg_delta * 5)
            reason = (
                f"Phase marginal: avg Δ={avg_delta:.3f} "
                f"(≤ {self.REDUCE_THRESHOLD}) über {n} Restaurierungen → Stärke {recommended_strength:.1f}"
            )
        else:
            action = "apply"
            recommended_strength = 1.0
            reason = f"Phase hilft: avg Δ={avg_delta:.3f} (> {self.REDUCE_THRESHOLD}) über {n} Restaurierungen"

        pred = ImpactPrediction(
            phase_id=phase_id,
            action=action,
            predicted_delta=avg_delta,
            confidence=confidence,
            n_samples=n,
            recommended_strength=recommended_strength,
            reason=reason,
        )
        self._cache[key] = pred

        logger.debug(
            "PhaseImpactPredictor: %s | %s → %s (Δ=%.3f, conf=%.2f, n=%d)",
            material,
            phase_id,
            action,
            avg_delta,
            confidence,
            n,
        )
        return pred

    def _load_impact_data(self, material: str, era: int | None, phase_id: str, mode: str) -> list[dict[str, Any]]:
        """Lädt historische Impact-Daten aus dem PhaseImpactRecorder."""
        try:
            from backend.core.phase_impact_recorder import get_phase_impact_recorder

            recorder = get_phase_impact_recorder()
            return recorder.query(material=material, era=era, phase_id=phase_id, mode=mode)  # type: ignore[no-any-return]
        except Exception as e:
            logger.debug("PhaseImpactPredictor: Datenquelle nicht verfügbar: %s", e)
            return []

    def invalidate_cache(self) -> None:
        """Cache leeren (nach neuen Aufzeichnungen)."""
        self._cache.clear()
        self._cache_loaded = False


# ── Singleton ─────────────────────────────────────────────────────────

_predictor: PhaseImpactPredictor | None = None


def get_phase_impact_predictor() -> PhaseImpactPredictor:
    """Thread-sicherer Singleton."""
    global _predictor
    if _predictor is None:
        _predictor = PhaseImpactPredictor()
    return _predictor
