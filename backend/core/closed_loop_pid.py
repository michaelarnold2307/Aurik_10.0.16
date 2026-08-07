"""§2.63 Closed-Loop PID Controller — Messen → Nachsteuern.

PID-Regler pro Musical Goal. Vor jeder Phase:
  - Lese Goal-Error aus vorherigen Phasen-Deltas
  - Berechne P (proportional), I (integral), D (derivative)
  - Adjustiere Strength: boosten wenn Goal unter Target, dämpfen wenn über Target

Hooked in UV3._profiled_phase_call:
  1. before_phase(phase_id, pre_snapshot) → strength_multiplier
     Liest PhaseEffectCatalog → weiß welche Goals diese Phase beeinflusst
     Vergleicht aktuelle Goal-Proxies mit Targets
     Gibt Multiplier [0.5, 1.5] zurück

  2. after_phase(phase_id, post_snapshot)
     Aktualisiert PID-State mit gemessenem Delta

Leichtgewichtig: O(#goals × #phases) ~ O(15 × 40) ≈ 600 Ops pro Phase, < 1 ms.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# PID-Gains (vorsichtig kalibriert für Audio-Restoration)
KP = 0.30  # Proportional: direkte Reaktion auf Error
KI = 0.08  # Integral: akkumulierter Error (verhindert steady-state)
KD = 0.10  # Derivative: bremst bei schneller Änderung (verhindert overshoot)

# Grenzen
STRENGTH_MIN = 0.40
STRENGTH_MAX = 1.50
INTEGRAL_WINDUP = 0.30  # Anti-Windup: I-Term cap


class ClosedLoopPIDController:
    """PID-Regler für per-Goal Strength-Adjustierung während UV3-Phasen."""

    def __init__(self, goal_targets: dict[str, float] | None = None):
        self._targets: dict[str, float] = dict(goal_targets or {})
        # Per-Goal PID State
        self._integral: dict[str, float] = {}  # I: accumulated error
        self._last_error: dict[str, float] = {}  # D: previous error
        self._enabled: bool = bool(self._targets)

    # ── Public API ──────────────────────────────────────────────

    def before_phase(
        self,
        phase_id: str,
        pre_snapshot: dict[str, float],
    ) -> float:
        """Berechnet Strength-Multiplier vor Phasen-Ausführung.

        Args:
            phase_id: z.B. "phase_03_denoise"
            pre_snapshot: Aktuelle Goal-Proxies von _fast_goal_snapshot

        Returns:
            Multiplier für kwargs["strength"]. 1.0 = keine Änderung.
            > 1.0 = boosten (Goal unter Target), < 1.0 = dämpfen.
        """
        if not self._enabled:
            return 1.0

        impacted_goals = _get_phase_goal_impacts(phase_id)
        if not impacted_goals:
            return 1.0

        multipliers: list[float] = []
        for goal, impact_direction in impacted_goals.items():
            target = self._targets.get(goal)
            if target is None:
                continue

            current = float(pre_snapshot.get(goal, target))
            error = target - current  # positive = unter Target → boosten

            # P: proportional
            p_term = KP * error

            # I: integral
            i_accum = self._integral.get(goal, 0.0)
            i_accum += error * KI
            i_accum = max(-INTEGRAL_WINDUP, min(INTEGRAL_WINDUP, i_accum))
            self._integral[goal] = i_accum
            i_term = i_accum

            # D: derivative
            prev_err = self._last_error.get(goal, error)
            d_term = KD * (error - prev_err)
            self._last_error[goal] = error

            # PID-Summe
            pid = p_term + i_term + d_term

            # Wenn die Phase dieses Goal negativ beeinflusst (impact < 0),
            # dann invertieren: bei Unter-Target trotzdem NICHT boosten.
            if impact_direction < 0:
                pid = -pid

            # In Strength-Multiplier umrechnen
            # pid > 0 → boosten (Goal unter Target + Phase hilft)
            # pid < 0 → dämpfen (Goal über Target ODER Phase schadet)
            mult = 1.0 + float(pid)
            mult = max(STRENGTH_MIN, min(STRENGTH_MAX, mult))
            multipliers.append(mult)

        if not multipliers:
            return 1.0

        # Verwende den Mittelwert der Multipliers (Median löscht gegenläufige Signale)
        if len(multipliers) == 1:
            result = multipliers[0]
        else:
            result = sum(multipliers) / len(multipliers)

        return float(result)

    def after_phase(
        self,
        phase_id: str,
        post_snapshot: dict[str, float],
    ) -> None:
        """Registriert Post-Phase-Snapshot (keine State-Änderung nötig,
        da before_phase bereits I+D aktualisiert).

        Args:
            phase_id: Phase-ID (für Logging)
            post_snapshot: Goal-Proxies NACH der Phase
        """
        if not self._enabled:
            return

        # I-Term wurde bereits in before_phase akkumuliert.
        # Hier nur Logging/Debugging.
        impacted_goals = _get_phase_goal_impacts(phase_id)
        if impacted_goals and logger.isEnabledFor(logging.DEBUG):
            _deltas = []
            for goal in impacted_goals:
                target = self._targets.get(goal)
                if target is None:
                    continue
                current = float(post_snapshot.get(goal, target))
                error = target - current
                _deltas.append(f"{goal}={error:+.3f}")
            if _deltas:
                logger.debug(
                    "§2.63 PID after %s: errors=[%s] I=%s",
                    phase_id,
                    " ".join(_deltas),
                    {g: f"{v:.3f}" for g, v in self._integral.items() if abs(v) > 0.001},
                )

    def get_state(self) -> dict[str, Any]:
        """Gibt aktuellen PID-State für Debugging/Logging zurück."""
        return {
            "enabled": self._enabled,
            "targets": dict(self._targets),
            "integral": {g: round(v, 4) for g, v in self._integral.items()},
            "last_error": {g: round(v, 4) for g, v in self._last_error.items()},
        }


# ── Phase → Goal Mapping ───────────────────────────────────────


def _get_phase_goal_impacts(phase_id: str) -> dict[str, float]:
    """Liest PhaseEffectCatalog: welche Goals beeinflusst diese Phase?

    Returns:
        {goal_name: impact_direction}
        impact > 0: Phase verbessert dieses Goal
        impact < 0: Phase kann dieses Goal verschlechtern
    """
    try:
        from backend.core.phase_effect_catalog import PHASE_EFFECT_CATALOG

        profile = PHASE_EFFECT_CATALOG.get(phase_id)
        if profile is not None and hasattr(profile, "goal_impact"):
            return dict(profile.goal_impact)
    except Exception as e:
        logger.warning("closed_loop_pid.py::_get_Verarbeitungsschritt_goal_impacts Ersatzpfad: %s", e)
    return {}
