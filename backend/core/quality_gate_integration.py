"""
Quality Gate Integration — Preservation Metrics in Decision Logic (§G61)

Bridges today's preservation metrics into UV3's quality gate.
Single import, single method call, zero UV3 modification needed.

Usage (in UV3._classify_quality_gate_events call site):
    from backend.core.quality_gate_integration import enrich_quality_gate
    _registry = enrich_quality_gate(_registry, original_audio, restored_audio, sr)

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def enrich_quality_gate(
    registry: dict,
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
) -> dict:
    """§G61: Reichert die Quality-Gate-Registry mit Preservation-Scores an.

    Fügt Harmonic-, Transient-, Formant- und Artifact-Scores hinzu.
    Bei kritischen Werten werden A_HARD_VETO- oder B_RECOVERY-Events
    in die Registry eingetragen.

    Args:
        registry: Bestehende Quality-Gate-Registry (wird NICHT modifiziert).
        original: Vor-Restaurierung (Referenz).
        restored: Nach-Restaurierung (zu bewerten).
        sr: Abtastrate.

    Returns:
        Neue Registry mit zusätzlichen Events (original unverändert).
    """
    enriched = dict(registry)
    enriched.setdefault("preservation_scores", {})

    # Harmonic Preservation (§G46)
    try:
        from backend.core.preservation_metrics import compute_harmonic_preservation_score

        h_score = compute_harmonic_preservation_score(original, restored, sr)
        enriched["preservation_scores"]["harmonic"] = float(h_score)
        if h_score < 0.70:
            enriched.setdefault("events", []).append(
                {
                    "id": "harmonic_degradation",
                    "klasse": "B_RECOVERY_TRIGGER",
                    "focus": "harmonic_preservation",
                    "value": float(h_score),
                    "threshold": 0.70,
                    "action": "harmonic_recovery_cascade",
                }
            )
    except Exception as e:
        logger.debug("Harmonic-Wert nicht verfügbar: %s", e)

    # Transient Preservation (§G47)
    try:
        from backend.core.preservation_metrics import compute_transient_preservation_score

        t_score = compute_transient_preservation_score(original, restored, sr)
        enriched["preservation_scores"]["transient"] = float(t_score)
        if t_score < 0.65:
            enriched.setdefault("events", []).append(
                {
                    "id": "transient_degradation",
                    "klasse": "B_RECOVERY_TRIGGER",
                    "focus": "transient_preservation",
                    "value": float(t_score),
                    "threshold": 0.65,
                    "action": "transient_recovery_cascade",
                }
            )
    except Exception as e:
        logger.debug("Transient-Wert nicht verfügbar: %s", e)

    # Formant Preservation (§G48)
    try:
        from backend.core.preservation_metrics import compute_formant_preservation_score

        f_score = compute_formant_preservation_score(original, restored, sr)
        enriched["preservation_scores"]["formant"] = float(f_score)
        if f_score < 0.60:
            enriched.setdefault("events", []).append(
                {
                    "id": "formant_degradation",
                    "klasse": "A_HARD_VETO",
                    "focus": "formant_preservation",
                    "value": float(f_score),
                    "threshold": 0.60,
                    "action": "rollback_checkpoint",
                }
            )
    except Exception as e:
        logger.debug("Formant-Wert nicht verfügbar: %s", e)

    # Artifact Freedom (§G53)
    try:
        from backend.core.artifact_detector import compute_artifact_freedom_score

        a_score = compute_artifact_freedom_score(restored, sr)
        enriched["preservation_scores"]["artifact_freedom"] = float(a_score)
        if a_score < 0.80:
            enriched.setdefault("events", []).append(
                {
                    "id": "artifact_detected",
                    "klasse": "A_HARD_VETO",
                    "focus": "artifact_freedom",
                    "value": float(a_score),
                    "threshold": 0.80,
                    "action": "rollback_checkpoint",
                }
            )
    except Exception as e:
        logger.debug("Artifact-Wert nicht verfügbar: %s", e)

    # Micro-Dynamics (§G52)
    try:
        from backend.core.preservation_metrics import compute_micro_dynamics_score

        m_score = compute_micro_dynamics_score(original, restored, sr)
        enriched["preservation_scores"]["micro_dynamics"] = float(m_score)
        if m_score < 0.55:
            enriched.setdefault("events", []).append(
                {
                    "id": "micro_dynamics_degradation",
                    "klasse": "B_RECOVERY_TRIGGER",
                    "focus": "micro_dynamics",
                    "value": float(m_score),
                    "threshold": 0.55,
                    "action": "dynamics_recovery_cascade",
                }
            )
    except Exception as e:
        logger.debug("Micro-Dynamics-Wert nicht verfügbar: %s", e)

    return enriched
