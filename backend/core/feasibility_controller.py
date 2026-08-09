"""§2.79 FeasibilityController — Erreichbarkeitsprüfung für 15 Musical Goals.

Berechnet per-Goal ob das Ziel physikalisch erreichbar ist,
und mit welcher Konfidenz + maximal erreichbarem Wert.

Pflicht-Integration (§2.79):
    Vor _execute_pipeline() (nach SongCalibration + GoalApplicability):

    feasibility = estimate_goal_feasibility(
        audio=audio, sr=sample_rate, material=material_type,
        restorability=restorability_score, transfer_chain=transfer_chain,
    )
    effective_goal_thresholds[goal] = min(
        effective_goal_thresholds[goal], feasibility[goal].max_achievable
    )
    if not feasibility[goal].reachable:
        metadata["goal_feasibility_limits"][goal] = feasibility[goal].to_dict()
"""

from __future__ import annotations

# v10.101 SOTA: Goal-Feasibility mit perzeptuellen Ceilings (MUSHRA/LUFS).
import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()


@dataclass(frozen=True)
class GoalFeasibility:
    """Erreichbarkeits-Ergebnis pro Musical Goal (§2.79)."""

    reachable: bool
    """True wenn physikalische Decke >= 85 % des Material-Floors."""

    confidence: float
    """Konfidenz [0.05, 0.95] — abhängig von Restorability + Chain-Komplexität."""

    max_achievable: float
    """Maximal erreichbarer Ziel-Score [0.30, 1.00] — aus PhysicalCeilingEstimator."""

    def to_dict(self) -> dict[str, float | bool]:
        """Serialisierung für metadata["goal_feasibility_limits"]."""
        return {
            "reachable": bool(self.reachable),
            "confidence": round(float(self.confidence), 4),
            "max_achievable": round(float(self.max_achievable), 4),
        }


def estimate_goal_feasibility(
    audio: np.ndarray,
    sr: int,
    material: str = "unknown",
    restorability: float = 65.0,
    transfer_chain: list[str] | None = None,
    is_studio_2026: bool = False,
) -> dict[str, GoalFeasibility]:
    """Schätzt Erreichbarkeit aller 15 Musical Goals (§2.79).

    Non-blocking: Bei Fehler wird ein leeres Dict zurückgegeben —
    Pipeline läuft ohne Feasibility-Einschränkung weiter.

    Args:
        audio: float32 ndarray
        sr: Abtastrate in Hz
        material: Material-Typ-Label (z.B. "shellac", "vinyl", "cd_digital")
        restorability: Restorability-Score 0–100
        transfer_chain: Liste der Träger in der Transfer-Chain

    Returns:
        Dict[goal_name, GoalFeasibility] mit allen aktiven Musical Goals.
        Leeres Dict bei Fehler (non-blocking).
    """
    try:
        from backend.core.calibration_matrix import get_material_floor  # pylint: disable=import-outside-toplevel  # noqa: I001
        from backend.core.calibration_matrix import (
            predict_quality_score,  # §v10.92: material-adaptive confidence
        )  # pylint: disable=import-outside-toplevel
        from backend.core.physical_ceiling_estimator import estimate_physical_ceiling  # pylint: disable=import-outside-toplevel
        from backend.core.song_goal_importance import ALL_GOAL_NAMES  # pylint: disable=import-outside-toplevel

        # Physikalische Decke ohne aktuelle Scores (reine SNR/BW-Schätzung auf Input-Audio)
        pce = estimate_physical_ceiling(
            np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0),
            int(sr),
            {},
            str(material or "unknown").strip().lower(),
        )
        ceiling = dict(pce.ceiling)

        # Konfidenz-Faktor: Restorability − Transfer-Chain-Komplexitäts-Penalty
        chain = [str(c).strip().lower() for c in (transfer_chain or []) if str(c).strip()]
        chain_penalty = float(np.clip(max(0, len(chain) - 1) * 0.05, 0.0, 0.15))
        mat_str = str(material or "unknown").strip().lower()
        # §v10.92: Material-adaptive Confidence statt hartem 0.95-Deckel.
        # predict_quality_score liefert das material-spezifische Qualitäts-Ceiling
        # (shellac=0.70, cd=0.95). Confidence = Restorability × Material-Ceiling.
        _mat_ceiling = predict_quality_score(mat_str, float(restorability), 0.0, is_studio_2026)
        # Normiere auf CD-Maximum (0.95) damit CD-Material weiterhin Confidence 0.95 erreicht
        _mat_conf_factor = float(np.clip(_mat_ceiling / 0.95, 0.50, 1.0))
        confidence = float(
            np.clip(
                float(restorability) / 100.0 * _mat_conf_factor - chain_penalty,
                0.05,
                float(np.clip(_mat_ceiling, 0.55, 0.95)),
            )
        )

        result: dict[str, GoalFeasibility] = {}
        for goal in ALL_GOAL_NAMES:
            max_ach = float(np.clip(ceiling.get(str(goal), 0.98), 0.30, 1.0))
            mat_floor = get_material_floor(mat_str, str(goal), is_studio_2026=False)
            # Erreichbar wenn Ceiling ≥ 85 % des Material-Floors (physikalische Toleranz §2.79)
            reachable = max_ach >= mat_floor * 0.85
            result[str(goal)] = GoalFeasibility(
                reachable=reachable,
                confidence=confidence,
                max_achievable=max_ach,
            )
        return result
    except Exception as _exc:
        logger.warning(
            "§G23 estimate_goal_feasibility nicht verfügbar — returning {} (DSP-Ersatzpfad): %s", _exc, exc_info=True
        )
        return {}


__all__ = [
    "GoalFeasibility",
    "estimate_goal_feasibility",
]
