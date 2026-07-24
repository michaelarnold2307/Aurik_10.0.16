"""
§v10.118 MetricArbiter — Qualitativer Richter über quantitative Metriken.

Löst Konflikte zwischen psychoakustischen Metriken (Goosebumps) und
qualitativen Bewertungen (HPI, VQI, artifact_freedom) nach dem Prinzip:

    "Qualitative Bewertung sticht quantitative — wenn die Restauration
     nachweislich gut ist, vertraue ihr, nicht den Artefakt-Transienten."

Regeln (priorisiert):
  1. HPI ≥ 0.80 → qualitativer Score gewinnt (Restaurat bevorzugt)
  2. artifact_freedom ≥ 0.95 → qualitativer Score gewinnt
  3. VQI < 0.70 in Vokal-Frequenzen → VQI überschreibt alles
  4. Sonst: gewichtetes Mittel (60% qualitativ, 40% quantitativ)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ArbiterDecision(Enum):
    PREFER_RESTORED = auto()  # Restaurat bevorzugen
    PREFER_ORIGINAL = auto()  # Original bevorzugen (nur bei echter Regression)
    WEIGHTED_BLEND = auto()  # Mittelwert-Entscheidung


@dataclass
class ArbiterVerdict:
    """Entscheidung des MetricArbiter."""

    decision: ArbiterDecision
    reason: str
    confidence: float  # 0.0–1.0 wie sicher die Entscheidung ist
    recommended_penalty: float  # Empfohlener Penalty für das unterlegene Signal
    details: dict[str, float] = field(default_factory=dict)


def resolve_metric_conflict(
    quantitative_score: float,  # z.B. Goosebumps-Score (0–1)
    qualitative_score: float,  # z.B. HPI (0–1)
    *,
    hpi: float | None = None,
    artifact_freedom: float | None = None,
    vqi: float | None = None,
    panns_singing: float = 0.0,
    context: dict[str, Any] | None = None,
) -> ArbiterVerdict:
    """Löst einen Konflikt zwischen quantitativer und qualitativer Metrik.

    Args:
        quantitative_score: Der quantitative Score (z.B. Goosebumps)
        qualitative_score: Der qualitative Score (z.B. HPI)
        hpi: Harmonic Preservation Index (optional)
        artifact_freedom: Artifact-Freedom-Score (optional)
        vqi: Vocal Quality Index (optional)
        panns_singing: PANNS-Singing-Confidence (0–1)
        context: Zusätzlicher Kontext (optional)

    Returns:
        ArbiterVerdict mit Entscheidung und Begründung
    """
    details: dict[str, float] = {
        "quantitative_score": round(quantitative_score, 4),
        "qualitative_score": round(qualitative_score, 4),
    }

    # ═══ Regel 1: HPI-Gate — wenn Restauration objektiv gut ist ═══
    if hpi is not None and hpi >= 0.80:
        details["hpi"] = round(hpi, 4)
        return ArbiterVerdict(
            decision=ArbiterDecision.PREFER_RESTORED,
            reason=f"HPI {hpi:.3f} ≥ 0.80 — Restauration ist objektiv gut, "
            f"quantitativer Score ({quantitative_score:.3f}) wird überschrieben",
            confidence=min(0.95, hpi),
            recommended_penalty=0.120,  # Hoher Penalty für Original (siehe §v10.113)
            details=details,
        )

    # ═══ Regel 2: Artifact-Freedom-Gate ═══
    if artifact_freedom is not None and artifact_freedom >= 0.95:
        details["artifact_freedom"] = round(artifact_freedom, 4)
        return ArbiterVerdict(
            decision=ArbiterDecision.PREFER_RESTORED,
            reason=f"Artifact-Freedom {artifact_freedom:.3f} ≥ 0.95 — "
            f"Restauration ist artefaktfrei, Original-Penalty erhöht",
            confidence=min(0.90, artifact_freedom),
            recommended_penalty=0.100,
            details=details,
        )

    # ═══ Regel 3: VQI-Vorrang bei Gesang ═══
    if vqi is not None and vqi < 0.70 and panns_singing >= 0.35:
        details["vqi"] = round(vqi, 4)
        details["panns_singing"] = round(panns_singing, 4)
        return ArbiterVerdict(
            decision=ArbiterDecision.PREFER_RESTORED,
            reason=f"VQI {vqi:.3f} < 0.70 bei panns_singing={panns_singing:.2f} — "
            "Gesangsschutz hat absolute Priorität über quantitative Metriken",
            confidence=0.85,
            recommended_penalty=0.150,
            details=details,
        )

    # ═══ Regel 4: Gewichtetes Mittel — beide Scores sind plausibel ═══
    weighted = 0.60 * qualitative_score + 0.40 * quantitative_score
    details["weighted_score"] = round(weighted, 4)

    if weighted >= 0.70:
        return ArbiterVerdict(
            decision=ArbiterDecision.WEIGHTED_BLEND,
            reason=f"Gewichtetes Mittel {weighted:.3f} ≥ 0.70 — "
            f"Restaurat wird leicht bevorzugt (60% qualitativ / 40% quantitativ)",
            confidence=0.70,
            recommended_penalty=0.030,
            details=details,
        )
    else:
        return ArbiterVerdict(
            decision=ArbiterDecision.WEIGHTED_BLEND,
            reason=f"Gewichtetes Mittel {weighted:.3f} < 0.70 — "
            "beide Scores niedrig, konservativer Penalty",
            confidence=0.50,
            recommended_penalty=0.015,
            details=details,
        )
