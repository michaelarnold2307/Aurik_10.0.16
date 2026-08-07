"""Zentraler Strength-Contract fuer Phasen.

Dieses Modul vereinheitlicht die Berechnung von
- phase_locality_factor
- pmgg_strength
- effective_strength

Optional kann ein Vocal-Cap fuer panns_singing aktiviert werden.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    """Konvertiert robust nach float und faengt NaN/Inf ab."""
    try:
        out = float(value)
    except Exception as e:
        logger.warning("Verarbeitungsschritt_strength_contract.py::_safe_float Ersatzpfad: %s", e)
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def resolve_phase_strength_contract(
    kwargs: dict[str, Any],
    *,
    locality_min: float = 0.35,
    locality_max: float = 1.0,
    strength_min: float = 0.0,
    strength_max: float = 1.0,
    vocal_gate_threshold: float | None = None,
    vocal_strength_cap: float | None = None,
) -> dict[str, float | bool]:
    """Berechnet den kanonischen Strength-Contract fuer eine Phase.

    Rueckgabe:
        phase_locality_factor: geclippt in [locality_min, locality_max]
        pmgg_strength: angeforderte PMGG-Staerke aus kwargs['strength']
        effective_strength: geclipptes Produkt aus PMGG und Locality
        vocal_cap_applied: True wenn Vocal-Cap aktiv gegriffen hat
    """
    _loc_raw = _safe_float(kwargs.get("phase_locality_factor", 1.0), 1.0)
    phase_locality_factor = float(np.clip(_loc_raw, locality_min, locality_max))

    pmgg_strength = _safe_float(kwargs.get("strength", 1.0), 1.0)
    effective_strength = float(np.clip(pmgg_strength * phase_locality_factor, strength_min, strength_max))

    vocal_cap_applied = False
    if vocal_gate_threshold is not None and vocal_strength_cap is not None:
        panns = _safe_float(
            kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)),
            0.0,
        )
        cap = float(np.clip(vocal_strength_cap, strength_min, strength_max))
        if panns >= float(vocal_gate_threshold) and effective_strength > cap:
            effective_strength = cap
            vocal_cap_applied = True

    return {
        "phase_locality_factor": phase_locality_factor,
        "pmgg_strength": pmgg_strength,
        "effective_strength": effective_strength,
        "vocal_cap_applied": vocal_cap_applied,
    }
