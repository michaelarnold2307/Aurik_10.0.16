"""backend/core/decision_trace.py — §v10.700 G3.

from typing import Any
logger = logging.getLogger(__name__)
Decision-Trace-Backend: Erklärt PRO PHASE warum sie lief, mit welcher
Stärke, und was sie bewirkt hat.

Nutzt existierende Daten aus RestorationResult (phases_executed,
phase_gate_log, goal_priority_log, metadata) und baut daraus eine
strukturierte Erklärung pro Phase.

Der Decision-Trace wird im metadata["decision_trace"] des
RestorationResult abgelegt und ist GUI-bereit für den
„Warum?"-Dialog / Decision-Trace-Tab.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PhaseDecision:
    """Eine einzelne Phasen-Entscheidung im Decision-Trace."""

    phase_id: str
    phase_name: str = ""
    executed: bool = True
    strength: float = 0.0
    reason: str = ""
    delta_quality: float | None = None  # Δ zu vorherigem Quality-Score
    duration_ms: float = 0.0
    material: str = ""
    defect: str = ""


@dataclass
class DecisionTrace:
    """Vollständiger Decision-Trace für eine Restaurierung."""

    song_name: str = ""
    material_type: str = ""
    total_phases: int = 0
    phases_executed: int = 0
    phases_skipped: int = 0
    decisions: list[PhaseDecision] = field(default_factory=list)
    summary: str = ""


def build_decision_trace(
    phases_executed: list[str],
    phases_skipped: list[str],
    material_type: str = "",
    phase_gate_log: list[str] | None = None,
    goal_priority_log: list[str] | None = None,
    song_name: str = "",
) -> DecisionTrace:
    """Baut Decision-Trace aus existierenden Pipeline-Daten.

    Args:
        phases_executed: Liste der ausgeführten Phasen-IDs
        phases_skipped: Liste der übersprungenen Phasen-IDs
        material_type: Material-Typ (vinyl, tape, etc.)
        phase_gate_log: PMGG-Log (welche Phasen mit welcher Stärke)
        goal_priority_log: GoalPriorityProtocol-Entscheidungen
        song_name: Name des Songs

    Returns:
        DecisionTrace mit einer PhaseDecision pro Phase
    """
    trace = DecisionTrace(
        song_name=song_name,
        material_type=material_type,
        total_phases=len(phases_executed) + len(phases_skipped),
        phases_executed=len(phases_executed),
        phases_skipped=len(phases_skipped),
    )

    # Executed phases
    for phase_id in phases_executed:
        decision = PhaseDecision(
            phase_id=phase_id,
            phase_name=_phase_id_to_name(phase_id),
            executed=True,
            strength=_extract_strength(phase_id, phase_gate_log),
            reason=_build_reason(phase_id, material_type, goal_priority_log),
            material=material_type,
            defect=_extract_defect(phase_id),
        )
        trace.decisions.append(decision)

    # Skipped phases
    for phase_id in phases_skipped:
        decision = PhaseDecision(
            phase_id=phase_id,
            phase_name=_phase_id_to_name(phase_id),
            executed=False,
            reason="Phase wurde vom PMGG als nicht nötig eingestuft",
            material=material_type,
        )
        trace.decisions.append(decision)

    # Summary
    trace.summary = (
        f"{song_name}: {trace.phases_executed}/{trace.total_phases} Phasen "
        f"ausgeführt ({material_type}). "
        f"{trace.phases_skipped} Phasen übersprungen."
    )

    return trace


def _phase_id_to_name(phase_id: str) -> str:
    """Maps phase_id zu menschenlesbarem Namen."""
    names = {
        "phase_01": "Defect-Scan (Klicks)",
        "phase_02": "Defect-Scan (Rauschen)",
        "phase_03": "Spektrale Denoising",
        "phase_04": "De-Click (Wavelet)",
        "phase_05": "De-Crackle",
        "phase_09": "De-Essing",
        "phase_12": "Entzerrung (EQ)",
        "phase_23": "Stereo-Breite",
        "phase_24": "Phase-Kohärenz",
        "phase_27": "Dynamik-Kompression",
        "phase_50": "CD-Rauschprofil",
        "phase_56": "Lautheits-Normalisierung",
        "phase_60": "Dithering",
        "phase_61": "Export-Vorbereitung",
        "phase_64": "Final Export",
    }
    return names.get(phase_id, phase_id)


def _extract_strength(phase_id: str, gate_log: list[str] | None) -> float:
    """Extrahiert die angewandte Stärke aus dem PMGG-Log."""
    if not gate_log:
        return 0.5
    for line in gate_log:
        if phase_id in line and "strength" in line.lower():
            try:
                # Parse "strength=0.75" oder "strength: 0.75"
                import re

                match = re.search(r"strength[=:]\s*([\d.]+)", line, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            except (ValueError, IndexError):
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return 0.5


def _extract_defect(phase_id: str) -> str:
    """Ordnet phase_id einem Defekttyp zu."""
    defect_map = {
        "phase_01": "clicks",
        "phase_02": "noise",
        "phase_03": "broadband_noise",
        "phase_04": "clicks",
        "phase_05": "crackle",
        "phase_09": "sibilance",
        "phase_12": "frequency_imbalance",
        "phase_23": "stereo_narrow",
        "phase_24": "phase_correlation",
        "phase_27": "dynamics_flat",
        "phase_50": "digital_silence",
        "phase_56": "loudness_mismatch",
        "phase_60": "quantization_noise",
        "phase_61": "export_ready",
        "phase_64": "final_export",
    }
    return defect_map.get(phase_id, "unknown")


def _build_reason(
    phase_id: str,
    material_type: str,
    goal_log: list[str] | None,
) -> str:
    """Baut eine menschenlesbare Begründung."""
    if phase_id in ("phase_01", "phase_02", "phase_04", "phase_05"):
        return f"Defekt-Scan hat {_extract_defect(phase_id)} auf {material_type} erkannt"
    elif phase_id == "phase_03":
        return "Spektrale Rauschreduktion für bessere Transparenz"
    elif phase_id == "phase_09":
        return "Zischlaut-Reduktion für natürlichere Vokale"
    elif phase_id == "phase_12":
        return "Frequenzgang-Korrektur für ausgewogene Tonalität"
    elif phase_id == "phase_27":
        return "Sanfte Dynamik-Anpassung für mehr Lebendigkeit"
    elif phase_id in ("phase_50", "phase_60"):
        return "Authentische Trägermedium-Charakteristik"
    else:
        return f"Material-adaptive Phase für {material_type}"
