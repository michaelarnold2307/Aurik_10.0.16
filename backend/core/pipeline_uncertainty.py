"""
PipelineUncertainty (§2.15 Spec)
==================================

Integrations-Wrapper um backend/core/optimization/uncertainty_quantification.py.
Quantifiziert für jede Restaurierungsoperation eine Konfidenz und passt
GP-Bounds und Pipeline-Parameter entsprechend an.

Konfidenz-Schwellwerte:
    ≥ 0.80: Defekt sicher erkannt → volle GP-Aggressivität
    0.50–0.80: GP-Bounds um 20 % konservativer, Nutzer-Hinweis aktivieren
    < 0.50:  konservative Mindest-Parameter, Musical-Goal-Schwellen +0.02

Referenz: §2.15 Aurik-9-Spec (v10.0.0)
Autor: Aurik Development Team
Datum: 20. Februar 2026
"""

from __future__ import annotations

# v10.101 SOTA: Unsicherheits-Quantifizierung mit JND-kalibrierten Schwellen.
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konfidenz-Schwellwerte (§2.15)
# ---------------------------------------------------------------------------


class UncertaintyThresholds:
    """Konfidenz-Schwellwerte für Pipeline-Steuerung."""

    HIGH: float = 0.80  # Volle GP-Aggressivität
    MEDIUM: float = 0.50  # Moderate GP-Bounds
    LOW: float = 0.00  # Sicherheitsmaximierende Parameter


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfidence:
    """Konfidenz-Ergebnis für eine Restaurierungsoperation.

    Attributes:
        confidence:        Gesamt-Konfidenz ∈ [0, 1].
        tier:              "high", "medium" oder "low".
        gp_bound_factor:   Multiplikator für GP-Bound-Reduktion (1.0 = voll, 0.8 = 20 % konservativer).
        threshold_offset:  Additions-Offset auf Musical-Goal-Schwellen (+0.02 bei low).
        user_hint:         Meldung für Nutzer (Deutsch, laienverständlich), oder "" wenn high.
        details:           Detaillierte UQ-Ergebnisse (für Log/Report).
    """

    confidence: float = 1.0
    tier: str = "high"
    gp_bound_factor: float = 1.0
    threshold_offset: float = 0.0
    user_hint: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PipelineUncertaintyEstimator:
    """Quantifiziert Restaurierungs-Konfidenz und steuert GP-Aggressivität.

    Einbindungs-Pflicht (§2.15):
        CausalDefectReasoner.reason() → plan.confidence → PipelineUncertaintyEstimator
        GPParameterOptimizer.propose() erhält confidence als Eingangs-Prior
        RestorationResult.confidence: float → UI-Anzeige als Balken

    DSP-Konfidenzberechnung (ohne ML):
        Konfidenz basiert auf DefectScanner-Scores und CausalReasoner-Posterior-Entropie:
        - Hohe DefektScore-Streuung → niedrige Konfidenz (unklar welcher Defekt)
        - Dominanter Defekt (score > 0.7) → hohe Konfidenz
        - Posterior-Entropie ≥ 0.8 → niedrige Konfidenz
    """

    def estimate(
        self,
        causal_plan,  # RestorationPlan von CausalDefectReasoner
        defect_scores: dict[str, float] | None = None,
    ) -> PipelineConfidence:
        """Schätzt Konfidenz aus CausalDefectReasoner-Ergebnis.

        Args:
            causal_plan:    RestorationPlan (hat .confidence float-Attribut).
            defect_scores:  Dict DefectType-Name → Score (optional, für DSP-Konfidenz).

        Returns:
            PipelineConfidence mit Tier + GP-Steuerungsparametern.
        """
        # Primäre Konfidenz aus CausalPlan
        plan_confidence = float(getattr(causal_plan, "confidence", 0.5)) if causal_plan is not None else 0.5
        plan_support = self._estimate_plan_support(causal_plan)

        # DSP-Konfidenz aus DefectScores (optional enhacements)
        dsp_confidence = self._estimate_dsp_confidence(defect_scores)

        # Kombinieren (evidenzgewichtete Fusion):
        # Das reine geometrische Mittel ist bei diffuser Kausalverteilung zu
        # pessimistisch und drueckt robuste DSP-Evidenz unnoetig nach unten.
        combined = float(
            np.clip(
                0.30 * plan_confidence + 0.50 * dsp_confidence + 0.20 * plan_support,
                0.0,
                1.0,
            )
        )

        # Konsens-Uplift: Wenn DSP-Evidenz sehr stark ist, soll eine niedrige
        # Einzelursachen-Posterior-Konfidenz die Pipeline nicht unter 0.60 ziehen.
        if dsp_confidence >= 0.85 and (plan_confidence >= 0.10 or plan_support >= 0.50):
            combined = float(max(combined, 0.62))

        # Tier bestimmen
        if combined >= UncertaintyThresholds.HIGH:
            tier = "high"
            gp_factor = 1.0
            threshold_offset = 0.0
            user_hint = ""
        elif combined >= UncertaintyThresholds.MEDIUM:
            tier = "medium"
            gp_factor = 0.80  # 20 % konservativer
            threshold_offset = 0.0
            user_hint = (
                "Manche Stellen sind schwer zu beurteilen — das System "
                "arbeitet vorsichtig, damit nichts verschlechtert wird."
            )
        else:
            tier = "low"
            gp_factor = 0.60  # 40 % konservativer
            threshold_offset = 0.02  # Musical Goals +0.02 verschärft
            user_hint = (
                "Die Aufnahme ist sehr schwierig. Das Ergebnis wird sorgfältig "
                "geprüft, aber möglicherweise sind Restdefekte unvermeidbar."
            )

        logger.info(
            "🔮 PipelineUncertainty: Konfidenz=%.3f (Plan=%.3f Support=%.3f DSP=%.3f) Tier=%s GP-Faktor=%.2f",
            combined,
            plan_confidence,
            plan_support,
            dsp_confidence,
            tier,
            gp_factor,
        )

        # Versuche ML-UQ-Backend (uncertainty_quantification.py)
        details = self._try_ml_uq_backend(combined)

        return PipelineConfidence(
            confidence=combined,
            tier=tier,
            gp_bound_factor=gp_factor,
            threshold_offset=threshold_offset,
            user_hint=user_hint,
            details=details,
        )

    def apply_to_gp_params(
        self,
        proposed_params: dict[str, float],
        confidence: PipelineConfidence,
        param_space: dict[str, tuple],
    ) -> dict[str, float]:
        """Skaliert GP-Parameter entsprechend Konfidenz-Tier (konservative Bounds).

        Bei Konfidenz < MEDIUM: alle Parameter werden Richtung Minimum verschoben
        (konservativer Eingriff). Bei HIGH: unveränderter Vorschlag.

        Args:
            proposed_params:   GP-vorgeschlagene Parameter.
            confidence:        PipelineConfidence.
            param_space:       Parametergrenzen {name: (min, max)}.

        Returns:
            Angepasste Parameter (immer innerhalb param_space-Grenzen).
        """
        if confidence.tier == "high":
            return proposed_params

        adjusted = {}
        for name, value in proposed_params.items():
            if name not in param_space:
                adjusted[name] = value
                continue
            lo, hi = param_space[name]
            center = (lo + hi) / 2.0
            # Wert Richtung Mitte / konservativ verschieben
            factor = confidence.gp_bound_factor
            shifted = center + factor * (value - center)
            adjusted[name] = float(np.clip(shifted, lo, hi))

        return adjusted

    def apply_threshold_offsets(
        self,
        thresholds: dict[str, float],
        confidence: PipelineConfidence,
    ) -> dict[str, float]:
        """Verschärft Musical-Goal-Schwellen bei niedriger Konfidenz.

        Args:
            thresholds:  Original-Schwellwerte {goal_name: threshold}.
            confidence:  PipelineConfidence.

        Returns:
            Angepasste Schwellwerte (niemals > 1.0).
        """
        if confidence.threshold_offset == 0.0:
            return thresholds
        return {name: float(np.clip(val + confidence.threshold_offset, 0.0, 1.0)) for name, val in thresholds.items()}

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _estimate_dsp_confidence(self, defect_scores: dict[str, float] | None) -> float:
        """Schätzt Konfidenz rein aus DefectScanner-Scores."""
        if not defect_scores:
            return 0.6  # Standard-Prior

        scores = list(defect_scores.values())
        scores = [s for s in scores if isinstance(s, (int, float)) and np.isfinite(s)]
        if not scores:
            return 0.6

        max_score = max(scores)
        np.mean(scores)

        # Hohe Dominanz eines Defekts → hohe Konfidenz
        if max_score >= 0.7:
            return min(1.0, 0.6 + max_score * 0.5)
        # Homogene Scores → niedrige Konfidenz (unklar)
        std_score = float(np.std(scores))
        if std_score < 0.1:
            return 0.35
        # Normalfall: Evidenzmenge + Verteilung berücksichtigen.
        support = float(np.clip(0.18 + 0.07 * len(scores), 0.18, 0.68))
        return float(np.clip(0.30 + 0.45 * std_score + 0.25 * support, 0.3, 0.90))

    def _estimate_plan_support(self, causal_plan: Any | None) -> float:
        """Leitet einen zusätzlichen Plan-Support aus der Posterior-Form ab."""
        if causal_plan is None:
            return 0.5

        try:
            ranked = list(getattr(causal_plan, "ranked_causes", []) or [])
            probs = [float(v) for _, v in ranked if isinstance(v, (int, float))]
            if not probs:
                return 0.5

            probs_arr = np.asarray(probs, dtype=np.float64)
            probs_arr = np.clip(probs_arr, 1e-12, 1.0)
            probs_arr /= max(float(np.sum(probs_arr)), 1e-12)

            top1 = float(probs_arr[0])
            top2 = float(probs_arr[1]) if probs_arr.size > 1 else 0.0
            margin = max(0.0, top1 - top2)
            margin_conf = float(np.clip(margin / 0.18, 0.0, 1.0))

            entropy = float(-np.sum(probs_arr * np.log(probs_arr)))
            entropy_norm = entropy / max(np.log(float(probs_arr.size)), 1e-9)
            entropy_conf = float(np.clip(1.0 - entropy_norm, 0.0, 1.0))

            top3_mass = float(np.sum(probs_arr[: min(3, probs_arr.size)]))
            top3_conf = float(np.clip(top3_mass / 0.75, 0.0, 1.0))

            return float(np.clip(0.40 * margin_conf + 0.30 * entropy_conf + 0.30 * top3_conf, 0.0, 1.0))
        except Exception as e:
            logger.warning("pipeline_uncertainty.py::_estimate_plan_support Ersatzpfad: %s", e)
            return 0.5

    def _try_ml_uq_backend(self, base_confidence: float) -> dict[str, Any]:
        """Versucht ML-basierte UQ via uncertainty_quantification.py."""
        details: dict[str, Any] = {"base_confidence": base_confidence}
        try:
            from backend.core.optimization.uncertainty_quantification import (  # type: ignore[import]
                UncertaintyQuantifier,
            )

            uq = UncertaintyQuantifier(model=None)  # type: ignore[arg-type]
            details["ml_uq_available"] = True
            details["ml_uq_class"] = type(uq).__name__
        except Exception as e:
            details["ml_uq_available"] = False
            details["ml_uq_error"] = str(e)
        return details


# ---------------------------------------------------------------------------
# Singleton (Thread-sicher, Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: PipelineUncertaintyEstimator | None = None
_lock = threading.Lock()


def get_pipeline_uncertainty_estimator() -> PipelineUncertaintyEstimator:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PipelineUncertaintyEstimator()
    return _instance


def estimate_pipeline_confidence(
    causal_plan,
    defect_scores: dict[str, float] | None = None,
) -> PipelineConfidence:
    """Convenience-Funktion: Schätzt Pipeline-Konfidenz.

    Args:
        causal_plan:    RestorationPlan (hat .confidence float-Attribut).
        defect_scores:  Dict DefectType-Name → Score (optional).

    Returns:
        PipelineConfidence mit Tier + GP-Steuerungsparametern.
    """
    return get_pipeline_uncertainty_estimator().estimate(causal_plan, defect_scores)


def estimate_goal_confidence_map(
    goal_scores: dict[str, float] | None,
    *,
    pipeline_confidence: float,
    restoration_context: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Leitet goal-spezifische Konfidenzen aus Proxy-Ensemble + Kontext ab.

    Diese Funktion ist strikt DSP-deterministisch (kein ML, kein Zufall) und
    dient als epistemische Zusatzschicht über den Goal-Proxys.
    """
    _scores = goal_scores if isinstance(goal_scores, dict) else {}
    if not _scores:
        return {}

    _pipe_conf = float(np.clip(pipeline_confidence, 0.0, 1.0))
    _ctx = restoration_context if isinstance(restoration_context, dict) else {}
    _chain = [str(v).strip().lower() for v in (_ctx.get("transfer_chain") or []) if str(v).strip()]
    _tcci = float(np.clip(_compute_transfer_chain_complexity(_chain), 0.0, 1.0))

    _group_map: dict[str, tuple[str, ...]] = {
        "vocal": ("natuerlichkeit", "authentizitaet", "artikulation", "emotionalitaet", "timbre_authentizitaet"),
        "clarity": ("transparenz", "brillanz", "separation_fidelity", "transient_energie"),
        "body": ("waerme", "bass_kraft", "tonal_center", "micro_dynamics"),
        "space": ("spatial_depth", "groove"),
    }

    _goal_group: dict[str, str] = {}
    for _group_name, _goals in _group_map.items():
        for _goal_name in _goals:
            _goal_group[_goal_name] = _group_name

    _group_stats: dict[str, tuple[float, float]] = {}
    for _group_name, _goals in _group_map.items():
        _vals = [float(np.clip(_scores.get(_goal, 0.5), 0.0, 1.0)) for _goal in _goals if _goal in _scores]
        if not _vals:
            continue
        _mean = float(np.mean(_vals))
        _std = float(np.std(_vals))
        _group_stats[_group_name] = (_mean, _std)

    _result: dict[str, float] = {}
    for _goal_name, _score_val in _scores.items():
        _goal = str(_goal_name)
        _score = float(np.clip(_score_val, 0.0, 1.0))
        _group = _goal_group.get(_goal, "clarity")
        _g_mean, _g_std = _group_stats.get(_group, (0.5, 0.20))

        # Ensemble-Agreement: geringe Intra-Group-Streuung = höhere Konfidenz.
        _agreement = float(np.clip(1.0 - (_g_std / 0.30), 0.0, 1.0))
        _deviation = float(np.clip(abs(_score - _g_mean) / 0.35, 0.0, 1.0))
        _self_consistency = float(np.clip(1.0 - _deviation, 0.0, 1.0))

        # Komplexe Transfer-Chains senken Proxy-Verlässlichkeit leicht.
        _chain_term = float(np.clip(1.0 - 0.25 * _tcci, 0.65, 1.0))

        # In Randzonen (<0.30 oder >0.90) sind Proxys oft volatiler.
        _extreme_penalty = 0.0
        if _score < 0.30:
            _extreme_penalty = float(np.clip((0.30 - _score) * 0.30, 0.0, 0.09))
        elif _score > 0.90:
            _extreme_penalty = float(np.clip((_score - 0.90) * 0.20, 0.0, 0.06))

        _conf = 0.45 * _pipe_conf + 0.30 * _agreement + 0.20 * _self_consistency + 0.05 * _chain_term - _extreme_penalty
        _result[_goal] = float(np.clip(_conf, 0.20, 0.98))

    return _result


def estimate_uncertainty_budget(
    *,
    goal_confidence: dict[str, float] | None,
    pipeline_confidence: float,
    transfer_chain: list[str] | None = None,
) -> float:
    """Schätzt ein globales Unsicherheitsbudget für geschlossene Regelkreise.

    Rückgabe in [0, 1]:
      0.0 = sehr sicher (aggressiver Recovery-Spielraum)
      1.0 = hoch unsicher (konservativer Recovery-Spielraum)
    """
    _pipe_conf = float(np.clip(pipeline_confidence, 0.0, 1.0))
    _goal_vals = [
        float(np.clip(v, 0.0, 1.0))
        for v in ((goal_confidence or {}).values())
        if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    _goal_mean = float(np.mean(_goal_vals)) if _goal_vals else _pipe_conf
    _goal_var = float(np.var(_goal_vals)) if _goal_vals else 0.0

    _chain = [str(v).strip().lower() for v in (transfer_chain or []) if str(v).strip()]
    _tcci = float(np.clip(_compute_transfer_chain_complexity(_chain), 0.0, 1.0))

    _base_uncertainty = 1.0 - (0.65 * _pipe_conf + 0.35 * _goal_mean)
    _dispersion_penalty = float(np.clip(_goal_var / 0.08, 0.0, 1.0)) * 0.20
    _chain_penalty = 0.18 * _tcci

    return float(np.clip(_base_uncertainty + _dispersion_penalty + _chain_penalty, 0.0, 1.0))


def _compute_transfer_chain_complexity(transfer_chain: list[str]) -> float:
    """Lokale, robuste TCCI-Näherung für Budget/Confidence-Helfer."""
    if not transfer_chain:
        return 0.0
    _chain = [str(v).strip().lower() for v in transfer_chain if str(v).strip()]
    if not _chain:
        return 0.0
    _length_term = float(np.clip((len(_chain) - 1) / 4.0, 0.0, 1.0))
    _unique_term = float(np.clip((len(set(_chain)) - 1) / 4.0, 0.0, 1.0))
    _analog = {"vinyl", "shellac", "tape", "reel_tape", "cassette", "wire_recording", "wax_cylinder"}
    _lossy = {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
    _mix_types = int(any(c in _analog for c in _chain)) + int(any(c in _lossy for c in _chain))
    _mix_term = 0.20 if _mix_types >= 2 else 0.0
    return float(np.clip(0.45 * _length_term + 0.35 * _unique_term + _mix_term, 0.0, 1.0))
