"""InnovationSuperiorityOrchestrator
=================================

Advisory-only Laufzeit-Orchestrator fuer disziplinuebergreifende
Innovationsentscheidungen in UV3.

Ziele:
- Defektbehebung, Vokalintegritaet, Timbralitaet, Zeitstruktur und Raumdarstellung
  gemeinsam als "Disziplinen" im Closed Loop bewerten.
- Pro Phase belastbare Innovationschancen (Goal-Opportunities) ermitteln.
- Den bestehenden §2.78-Rescheduler nur sanft und gebunden unterstuetzen
  (kein Gate-Override, kein harter Eingriff).

Invarianten:
- Advisory-only, keine harte Gate-Logik.
- Alle Uplifts streng begrenzt.
- Rein lokal/deterministisch, keine Netzabhaengigkeit.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.core.calibration_matrix import get_goal_recovery_phases

logger = logging.getLogger(__name__)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception as e:
        logger.warning("innovation_superiority_orchestrator.py::_to_float Ersatzpfad: %s", e)
        return float(default)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class InnovationRealtimePlan:
    """Advisory-Plan fuer den aktuellen Phase-Closed-Loop."""

    discipline_scores: dict[str, float]
    priority_goals: list[str]
    goal_confidence_uplift: dict[str, float]
    recovery_phase_hints: dict[str, str]
    innovation_intensity: float


class InnovationSuperiorityOrchestrator:
    """Erzeugt bounded Innovationshints ueber alle Aurik-Disziplinen."""

    _DISCIPLINE_GOALS: dict[str, tuple[str, ...]] = {
        "defect_intelligence": (
            "transparenz",
            "natuerlichkeit",
            "artikulation",
            "transient_energie",
        ),
        "vocal_integrity": (
            "vocal_quality",
            "formant_fidelity",
            "artikulation",
            "emotionalitaet",
        ),
        "timbral_truth": (
            "timbral_authenticity",
            "tonal_center",
            "waerme",
            "brillanz",
        ),
        "temporal_precision": (
            "micro_dynamics",
            "transient_energie",
            "groove",
        ),
        "spatial_coherence": (
            "spatial_depth",
            "separation_fidelity",
        ),
        "goal_convergence": (
            "natuerlichkeit",
            "authentizitaet",
            "transparenz",
            "emotionalitaet",
        ),
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def build_realtime_plan(
        self,
        *,
        goal_scores: dict[str, float],
        goal_targets: dict[str, float],
        goal_confidence: dict[str, float],
        pipeline_confidence: float,
        material_type: str,
        transfer_chain: list[str],
        is_studio_2026: bool,
        phase_goal_delta: dict[str, float] | None,
        phase_metadata: dict[str, Any] | None,
    ) -> InnovationRealtimePlan:
        with self._lock:
            _scores = goal_scores if isinstance(goal_scores, dict) else {}
            _targets = goal_targets if isinstance(goal_targets, dict) else {}
            _conf = goal_confidence if isinstance(goal_confidence, dict) else {}
            _delta = phase_goal_delta if isinstance(phase_goal_delta, dict) else {}
            _meta = phase_metadata if isinstance(phase_metadata, dict) else {}

            _chain_complexity = _clip01(min(len(set(transfer_chain or [])) / 4.0, 1.0))
            _epi_conf = _clip01(_to_float(_meta.get("pmgg_reconstruction_epistemic_confidence"), 0.65))
            _localized = bool(_meta.get("pmgg_reconstruction_localized", False))
            _pipeline_conf = _clip01(_to_float(pipeline_confidence, 0.7))

            _intensity = _clip01(
                0.30 + 0.25 * _chain_complexity + 0.25 * (1.0 - _pipeline_conf) + 0.20 * (1.0 - _epi_conf)
            )
            if _localized:
                _intensity = _clip01(_intensity + 0.05)

            _discipline_scores: dict[str, float] = {}
            for _discipline, _goals in self._DISCIPLINE_GOALS.items():
                _vals: list[float] = []
                for _goal in _goals:
                    _target = _to_float(_targets.get(_goal), 0.0)
                    if _target <= 0.0:
                        continue
                    _score = _to_float(_scores.get(_goal), 0.0)
                    _gap_ratio = _clip01((_target - _score) / max(_target, 1e-6))
                    _phase_trend = _to_float(_delta.get(_goal), 0.0)
                    _trend_bonus = float(np.clip(_phase_trend, -0.08, 0.08))
                    _local_conf = _clip01(_to_float(_conf.get(_goal), 0.65))
                    _vals.append(_clip01(1.0 - _gap_ratio + _trend_bonus + 0.10 * _local_conf))
                _discipline_scores[_discipline] = float(np.mean(_vals)) if _vals else 0.5

            _goal_opportunities: dict[str, float] = {}
            for _goal, _target_v in _targets.items():
                _target = _to_float(_target_v, 0.0)
                if _target <= 0.0:
                    continue
                _score = _to_float(_scores.get(_goal), 0.0)
                _gap_ratio = _clip01((_target - _score) / max(_target, 1e-6))
                _goal_conf = _clip01(_to_float(_conf.get(_goal), 0.65))
                _trend = _to_float(_delta.get(_goal), 0.0)
                _trend_penalty = float(np.clip(max(_trend, 0.0), 0.0, 0.08))
                _opportunity = _clip01(_gap_ratio * (1.0 - _goal_conf) * (0.65 + 0.35 * _intensity) - _trend_penalty)
                if _opportunity > 0.01:
                    _goal_opportunities[str(_goal)] = _opportunity

            _priority_goals = [
                _goal for _goal, _ in sorted(_goal_opportunities.items(), key=lambda kv: kv[1], reverse=True)[:3]
            ]

            _uplift: dict[str, float] = {}
            _hints: dict[str, str] = {}
            for _goal in _priority_goals:
                _opp = _goal_opportunities.get(_goal, 0.0)
                _uplift[_goal] = float(np.clip(0.015 + 0.045 * _opp, 0.0, 0.05))
                _cand = get_goal_recovery_phases(
                    _goal,
                    is_studio_2026=bool(is_studio_2026),
                    transfer_chain=list(transfer_chain or []),
                )
                if _cand:
                    _hints[_goal] = str(_cand[0])

            return InnovationRealtimePlan(
                discipline_scores={k: round(float(v), 4) for k, v in _discipline_scores.items()},
                priority_goals=list(_priority_goals),
                goal_confidence_uplift={k: round(float(v), 4) for k, v in _uplift.items()},
                recovery_phase_hints=dict(_hints),
                innovation_intensity=round(float(_intensity), 4),
            )


_ORCH_INSTANCE: InnovationSuperiorityOrchestrator | None = None
_ORCH_LOCK = threading.Lock()


def get_innovation_superiority_orchestrator() -> InnovationSuperiorityOrchestrator:
    """Thread-sicherer Singleton-Zugriff."""
    global _ORCH_INSTANCE
    if _ORCH_INSTANCE is None:
        with _ORCH_LOCK:
            if _ORCH_INSTANCE is None:
                _ORCH_INSTANCE = InnovationSuperiorityOrchestrator()
    return _ORCH_INSTANCE


__all__ = [
    "InnovationRealtimePlan",
    "InnovationSuperiorityOrchestrator",
    "get_innovation_superiority_orchestrator",
]
