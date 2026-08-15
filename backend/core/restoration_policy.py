"""Zentrale Policy-Hilfen für songweite Restaurierungsentscheidungen."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _clamp(value: Any, lower: float, upper: float, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        number = fallback
    if number != number:
        number = fallback
    return float(max(lower, min(upper, number)))


def get_restoration_policy_profile(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Gibt das zentrale Policy-Profil zurück oder ein leeres Dict."""
    if not isinstance(kwargs, dict):
        return {}
    profile = kwargs.get("restoration_policy_profile")
    if isinstance(profile, dict):
        return dict(profile)
    return {}


def get_effective_song_goal_weights(kwargs: dict[str, Any] | None) -> dict[str, float] | None:
    """Gibt die effektiven Goal-Weights aus dem Policy-Profil zurück.

    Das Policy-Profil ist die normative Quelle. `song_goal_weights` bleibt nur
    Kompatibilitäts-Fallback für Legacy-Aufrufer.
    """
    if not isinstance(kwargs, dict):
        return None

    profile = get_restoration_policy_profile(kwargs)
    goal_weights = profile.get("goal_weights")
    if isinstance(goal_weights, dict) and goal_weights:
        return {str(k): float(v) for k, v in goal_weights.items() if isinstance(k, str)}

    legacy_goal_weights = kwargs.get("song_goal_weights")
    if isinstance(legacy_goal_weights, dict) and legacy_goal_weights:
        return {str(k): float(v) for k, v in legacy_goal_weights.items() if isinstance(k, str)}

    return None


def blend_denker_policy_goal_weights(
    base_goal_weights: dict[str, float] | None,
    denker_policy_input: dict[str, Any] | None,
) -> dict[str, float]:
    """Reichert Goal-Weights mit Denker-Signalen an, ohne eine zweite Steuerquelle zu schaffen.

    Die Denker liefern Wahrnehmungs- und Risiko-Hinweise. Dieser Helper verdichtet
    sie in die zentralen Policy-Weights; UV3 folgt anschließend nur noch dem
    resultierenden `restoration_policy_profile`.
    """
    weights: dict[str, float] = {
        str(k): float(v)
        for k, v in (base_goal_weights or {}).items()
        if isinstance(k, str) and isinstance(v, (int, float))
    }
    if not weights:
        weights = {
            "natuerlichkeit": 1.20,
            "authentizitaet": 1.20,
            "micro_dynamics": 1.00,
            "artikulation": 1.00,
            "waerme": 1.00,
            "brillanz": 1.00,
            "transparenz": 1.00,
        }
    if not isinstance(denker_policy_input, dict):
        return weights

    strategy = denker_policy_input.get("strategy")
    if isinstance(strategy, dict):
        listening_targets = strategy.get("listening_experience_targets")
        if isinstance(listening_targets, dict):
            for goal, factor in listening_targets.items():
                if isinstance(goal, str) and isinstance(factor, (int, float)):
                    weights[goal] = float(weights.get(goal, 1.0)) * float(max(0.85, min(1.35, factor)))

    phase_interaction = denker_policy_input.get("phase_interaction")
    if isinstance(phase_interaction, dict):
        goal_risks = phase_interaction.get("goal_risk_map")
        if isinstance(goal_risks, dict):
            for goal, risk in goal_risks.items():
                if isinstance(goal, str) and isinstance(risk, (int, float)):
                    weights[goal] = float(weights.get(goal, 1.0)) * float(1.0 + 0.20 * max(0.0, min(1.0, risk)))

    comfort = synthesize_human_hearing_comfort_profile(denker_policy_input)
    fatigue = float(comfort.get("fatigue_sensitivity", 0.0))
    transient_protection = float(comfort.get("transient_protection", 0.0))
    micro_protection = float(comfort.get("microdynamic_protection", 0.0))
    warmth_presence = float(comfort.get("warmth_presence_balance", 0.0))
    vocal_priority = float(comfort.get("vocal_comfort_priority", 0.0))
    weights["natuerlichkeit"] = float(weights.get("natuerlichkeit", 1.0)) * (1.0 + 0.16 * fatigue)
    weights["authentizitaet"] = float(weights.get("authentizitaet", 1.0)) * (1.0 + 0.12 * fatigue)
    weights["artikulation"] = float(weights.get("artikulation", 1.0)) * (1.0 + 0.12 * transient_protection)
    weights["micro_dynamics"] = float(weights.get("micro_dynamics", 1.0)) * (1.0 + 0.14 * micro_protection)
    weights["waerme"] = float(weights.get("waerme", 1.0)) * (1.0 + 0.10 * warmth_presence)
    weights["brillanz"] = float(weights.get("brillanz", 1.0)) * (1.0 + 0.08 * (1.0 - fatigue))
    if vocal_priority > 0.0:
        weights["vocal_quality"] = float(weights.get("vocal_quality", 1.0)) * (1.0 + 0.18 * vocal_priority)
        weights["formant_fidelity"] = float(weights.get("formant_fidelity", 1.0)) * (1.0 + 0.16 * vocal_priority)

    return {goal: float(max(0.65, min(1.65, value))) for goal, value in weights.items()}


def synthesize_human_hearing_comfort_profile(
    denker_policy_input: dict[str, Any] | None,
    *,
    mode: str = "restoration",
    intervention_budget: float | None = None,
) -> dict[str, float]:
    """Verdichtet Denker-Hinweise zu songindividuellen Hoerkomfort-Parametern.

    Die Rueckgabe ist reine Policy: sie fuehrt keine Audio-Operation aus und darf
    von UV3/Guards nur als zentrale Steuerquelle gelesen werden.
    """
    if not isinstance(denker_policy_input, dict):
        denker_policy_input = {}

    strategy = denker_policy_input.get("strategy")
    strategy = strategy if isinstance(strategy, dict) else {}
    phase_interaction = denker_policy_input.get("phase_interaction")
    phase_interaction = phase_interaction if isinstance(phase_interaction, dict) else {}
    signal_signature = denker_policy_input.get("signal_signature")
    if not isinstance(signal_signature, dict):
        strategy_signature = strategy.get("signal_signature")
        signal_signature = dict(strategy_signature) if isinstance(strategy_signature, dict) else {}
    else:
        signal_signature = dict(signal_signature)
    risk_map = strategy.get("human_hearing_risk_map")
    risk_map = risk_map if isinstance(risk_map, dict) else {}
    repair_risk = denker_policy_input.get("repair_risk_profile")
    repair_risk = repair_risk if isinstance(repair_risk, dict) else {}
    reconstruction_risk = denker_policy_input.get("reconstruction_risk_profile")
    reconstruction_risk = reconstruction_risk if isinstance(reconstruction_risk, dict) else {}

    crest_db = _clamp(signal_signature.get("crest_db", 12.0), 0.0, 40.0, 12.0)
    hf_ratio = _clamp(signal_signature.get("hf_ratio", 0.04), 0.0, 1.0, 0.04)
    transient_ratio = _clamp(signal_signature.get("transient_ratio", 0.006), 0.0, 0.08, 0.006)
    micro_dynamic_db = _clamp(signal_signature.get("micro_dynamic_db", 10.0), 0.0, 60.0, 10.0)
    budget = _clamp(
        intervention_budget if intervention_budget is not None else strategy.get("intervention_budget", 0.5),
        0.0,
        1.0,
        0.5,
    )

    fatigue = _clamp(
        0.55 * _clamp(risk_map.get("listening_fatigue", 0.0), 0.0, 1.0)
        + 0.25 * _clamp(repair_risk.get("musical_damage", 0.0), 0.0, 1.0)
        + 0.20 * _clamp(hf_ratio * 2.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    transient_protection = _clamp(
        0.35 + min(transient_ratio, 0.025) * 18.0 + 0.25 * _clamp(risk_map.get("transient_smear", 0.0), 0.0, 1.0),
        0.0,
        1.0,
    )
    micro_protection = _clamp(
        0.35
        + 0.35 * _clamp(risk_map.get("microdynamics_loss", 0.0), 0.0, 1.0)
        + 0.20 * _clamp(max(0.0, 14.0 - micro_dynamic_db) / 14.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    overprocessing_risk = _clamp(
        0.55 * _clamp(risk_map.get("overprocessing", 0.0), 0.0, 1.0)
        + 0.25 * _clamp(reconstruction_risk.get("hallucination", 0.0), 0.0, 1.0)
        + 0.20 * _clamp(max(0.0, crest_db - 18.0) / 18.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    dullness_risk = _clamp(max(0.0, 0.045 - hf_ratio) / 0.045, 0.0, 1.0)
    vocal_priority = _clamp(strategy.get("vocal_comfort_priority", 0.0), 0.0, 1.0)
    mode_is_studio = "studio" in str(mode).lower() or "maximum" in str(mode).lower()

    profile: dict[str, float] = {
        "fatigue_sensitivity": fatigue,
        "transient_protection": transient_protection,
        "microdynamic_protection": micro_protection,
        "overprocessing_risk": overprocessing_risk,
        "dullness_risk": dullness_risk,
        "vocal_comfort_priority": vocal_priority,
        "peak_overshoot_cap_db": _clamp(3.0 - 0.40 * fatigue - 0.30 * vocal_priority + 0.10 * budget, 2.2, 3.0),
        "hf_loss_tolerance_db": _clamp(0.95 - 0.25 * fatigue - 0.15 * vocal_priority, 0.45, 1.05),
        "hf_lift_cap_db": _clamp(
            0.65 + 0.75 * dullness_risk - 0.25 * fatigue + (0.15 if mode_is_studio else 0.0), 0.35, 1.35
        ),
        "noise_floor_relative_cap_db": _clamp(
            0.65 - 0.25 * fatigue - 0.20 * overprocessing_risk - 0.10 * vocal_priority + 0.10 * budget,
            0.20,
            1.20,
        ),
        "warmth_presence_balance": _clamp(0.45 + 0.30 * dullness_risk + 0.20 * fatigue, 0.0, 1.0),
        "dynamic_smoothing_tolerance": _clamp(
            0.25 - 0.18 * transient_protection - 0.20 * overprocessing_risk, 0.0, 0.35
        ),
        "intervention_budget": budget,
    }

    explicit = strategy.get("human_hearing_comfort_profile")
    if isinstance(explicit, dict):
        for key, value in explicit.items():
            if key in profile:
                if key.endswith("_db"):
                    if key == "peak_overshoot_cap_db":
                        lower, upper = 2.2, 3.0
                    elif key == "noise_floor_relative_cap_db":
                        lower, upper = 0.20, 1.20
                    else:
                        lower, upper = 0.0, 1.35
                    profile[key] = _clamp(value, lower, upper, profile[key])
                else:
                    profile[key] = _clamp(value, 0.0, 1.0, profile[key])

    return {str(key): round(float(value), 4) for key, value in profile.items()}


def get_human_hearing_comfort_profile(policy_profile: dict[str, Any] | None) -> dict[str, float]:
    """Liest das Komfortprofil aus dem zentralen restoration_policy_profile."""
    if not isinstance(policy_profile, dict):
        return {}
    raw = policy_profile.get("human_hearing_comfort_profile")
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, (int, float))}
