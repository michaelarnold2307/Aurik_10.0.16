"""
backend/core/adaptive_phase_order.py — Material-adaptive Phase-Reihenfolge (§v10.9)
===================================================================================

Passt die Ausführungsreihenfolge der Phasen ans Trägermedium an.
Kassette: Hiss vor Harmonics. Vinyl: Klicks vor Entrauschung.

Usage:
    from backend.core.adaptive_phase_order import reorder_phases_for_material
    ordered = reorder_phases_for_material(phase_list, "cassette")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Material-spezifische Reihenfolge-Regeln: (Phase_A, Phase_B) bedeutet "A vor B"
# Positive Priorität = Phase soll früher laufen, negativ = später
_MATERIAL_PHASE_PRIORITY: dict[str, dict[str, int]] = {
    "cassette": {
        "phase_29_tape_hiss_reduction": +5,  # Hiss VOR Harmonics
        "phase_07_harmonic_restoration": -3,  # Harmonics NACH Hiss
        "phase_22_tape_saturation": +2,  # Tape-Charakter früh
        "phase_12_wow_flutter_fix": +4,  # Gleichlauf VOR allem
        "phase_31_speed_pitch_correction": +3,  # Speed vor Restoration
    },
    "vinyl": {
        "phase_01_click_removal": +6,  # Klicks ZUERST
        "phase_09_crackle_removal": +5,  # Crackle vor Rauschen
        "phase_03_denoise": -2,  # Entrauschung NACH Klicks
        "phase_60_inner_groove_distortion_repair": +4,
        "phase_61_groove_echo_cancellation": +3,
    },
    "shellac": {
        "phase_01_click_removal": +6,
        "phase_09_crackle_removal": +5,
        "phase_04_eq_correction": +4,  # EQ VOR Restoration
        "phase_06_frequency_restoration": -2,
    },
    "reel_tape": {
        "phase_29_tape_hiss_reduction": +4,
        "phase_12_wow_flutter_fix": +5,
        "phase_31_speed_pitch_correction": +4,
        "phase_25_azimuth_correction": +3,
        "phase_07_harmonic_restoration": -3,
    },
    "cd_digital": {
        "phase_03_denoise": -5,  # Kaum Rauschen → fast ganz hinten
        "phase_01_click_removal": -3,
    },
}


def reorder_phases_for_material(
    phases: list[str],
    material: str,
) -> list[str]:
    """Ordnet Phasenliste material-adaptiv um.

    Behält die relative Reihenfolge innerhalb gleicher Priorität bei.
    Phasen ohne Eintrag behalten ihre ursprüngliche Position.

    Args:
        phases: Liste von Phase-IDs in aktueller Reihenfolge.
        material: Trägermedium (z.B. 'cassette', 'vinyl').

    Returns:
        Neu geordnete Liste.
    """
    material = str(material or "").lower().strip()
    rules = _MATERIAL_PHASE_PRIORITY.get(material, {})

    if not rules:
        return list(phases)

    # Jedem Phase-Eintrag eine Priorität zuweisen
    _phase_index: dict[str, int] = {p: i for i, p in enumerate(phases)}
    _priority: dict[str, int] = {}

    for p in phases:
        _key = p.split("_", 2)
        _base = f"{_key[0]}_{_key[1]}" if len(_key) >= 2 else p
        _prio = 0
        for rule_key, rule_prio in rules.items():
            if p == rule_key or p.startswith(rule_key[:20]) or _base == rule_key[: len(_base)]:
                _prio = rule_prio
                break
        _priority[p] = _prio

    # Nach Priorität sortieren (stabil: bei gleicher Prio bleibt Original-Reihenfolge)
    _changed = any(v != 0 for v in _priority.values())
    if not _changed:
        return list(phases)

    _ordered = sorted(phases, key=lambda p: (-_priority[p], _phase_index[p]))

    _moved = [p for p in phases if _priority.get(p, 0) != 0]
    if _moved:
        logger.info(
            "🔄 Adaptive Verarbeitungsschritt-Order (%s): %d Phasen umgeordnet — %s",
            material,
            len(_moved),
            ", ".join(_moved[:5]),
        )

    return _ordered
