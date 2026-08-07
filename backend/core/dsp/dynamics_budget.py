"""Global Dynamics Budget — §v10.303.

Verhindert kumulative Dynamik-Kompression durch mehrere unabhängige
Dynamics-Prozessoren. P10 (Compression), P26 (Expansion), P54 (Transparent
Dynamics) und P47 (TruePeak Limiter) arbeiten auf denselben Frequenzbändern
ohne Koordination. Das Budget begrenzt die kumulative Crest-Reduktion.

Prinzip: Jede Dynamics-Phase meldet ihre Crest-Änderung an. Wenn das
kumulative Budget erschöpft ist, werden nachfolgende Dynamics-Phasen
auf minimale Stärke reduziert.

Messung → Kalibrierung → Anwendung:
  1. Vor der ersten Dynamics-Phase: Original-Crest messen
  2. Nach jeder Dynamics-Phase: Crest-Änderung buchen
  3. Vor jeder Dynamics-Phase: Verfügbares Budget prüfen
  4. Budget erschöpft → Phase läuft mit strength_cap = 0.10
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Budget-Parameter ─────────────────────────────────────────────────────────
# Maximal erlaubte Crest-Reduktion in dB (Original → verarbeitet)
# Ein typischer degraded-Song hat Crest 15 dB. Nach -3 dB Reduktion
# bleibt Crest bei 12 dB — das ist die untere Hörbarkeitsgrenze.
# Weitere Reduktion (12→11, 11→10) macht den Klang "platt".
MAX_CREST_REDUCTION_DB: float = 3.0

# Phasen, die am Dynamics-Budget teilnehmen
DYNAMICS_PHASES: frozenset[str] = frozenset(
    {
        "phase_10_compression",
        "phase_26_dynamic_range_expansion",
        "phase_54_transparent_dynamics",
        "phase_47_truepeak_limiter",
        "phase_35_multiband_compression",
        "phase_11_limiting",
    }
)

# Minimale Stärke wenn Budget erschöpft ist
BUDGET_EXHAUSTED_STRENGTH_CAP: float = 0.10


@dataclass
class DynamicsBudgetState:
    """Thread-lokaler Zustand des Dynamics-Budgets."""

    original_crest_db: float = 0.0
    cumulative_reduction_db: float = 0.0
    budget_exhausted: bool = False
    phases_processed: list[str] = field(default_factory=list)


# Thread-lokale Instanz
_state: threading.local = threading.local()


def _get_state() -> DynamicsBudgetState:
    if not hasattr(_state, "budget"):
        _state.budget = DynamicsBudgetState()
    return _state.budget  # type: ignore[no-any-return]


def reset_dynamics_budget() -> None:
    """Budget zurücksetzen (vor neuem Pipeline-Lauf)."""
    _state.budget = DynamicsBudgetState()
    logger.debug("§DYN-Grenze: zurückgesetzt")


def initialize_budget(audio: np.ndarray, sample_rate: int) -> None:
    """Original-Crest messen und Budget initialisieren."""
    state = _get_state()
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    rms = float(np.sqrt(np.mean(a**2) + 1e-12))
    peak = float(np.max(np.abs(a)))
    state.original_crest_db = float(20.0 * np.log10(peak / rms + 1e-12))
    state.cumulative_reduction_db = 0.0
    state.budget_exhausted = False
    state.phases_processed = []
    logger.debug(
        "§DYN-Grenze: initialisiert — Originalsignal_crest=%.1f dB, max_reduction=%.1f dB",
        state.original_crest_db,
        MAX_CREST_REDUCTION_DB,
    )


def get_available_strength_cap(
    phase_id: str,
    current_strength: float,
) -> float:
    """Gibt den maximal erlaubten Strength-Wert für eine Dynamics-Phase zurück.

    Wenn das Budget erschöpft ist, wird die Stärke auf
    BUDGET_EXHAUSTED_STRENGTH_CAP reduziert.

    Args:
        phase_id: Phasen-ID (z.B. "phase_10_compression")
        current_strength: Aktuell geplante Stärke (0–1)

    Returns:
        Effektive Stärke (0–1), möglicherweise reduziert.
    """
    state = _get_state()

    if phase_id not in DYNAMICS_PHASES:
        return current_strength

    if state.budget_exhausted:
        logger.info(
            "§DYN-Grenze: %s Grenze erschöpft (%.1f/%.1f dB) → strength %.2f→%.2f",
            phase_id,
            state.cumulative_reduction_db,
            MAX_CREST_REDUCTION_DB,
            current_strength,
            BUDGET_EXHAUSTED_STRENGTH_CAP,
        )
        return BUDGET_EXHAUSTED_STRENGTH_CAP

    # Projiziere: wenn diese Phase mit aktueller Stärke läuft,
    # wie viel Crest-Reduktion würde das kosten?
    # Faustregel: strength=1.0 → ~2 dB Crest-Reduktion
    projected = current_strength * 2.0
    remaining = MAX_CREST_REDUCTION_DB - state.cumulative_reduction_db

    if projected > remaining and remaining > 0.1:
        capped = float(np.clip(remaining / 2.0, 0.05, current_strength))
        logger.info(
            "§DYN-Grenze: %s projiziert %.1f dB → nur %.1f dB verfügbar → strength %.2f→%.2f",
            phase_id,
            projected,
            remaining,
            current_strength,
            capped,
        )
        return capped

    return current_strength


def report_crest_change(
    phase_id: str,
    audio_before: np.ndarray,
    audio_after: np.ndarray,
) -> None:
    """Crest-Änderung nach einer Dynamics-Phase melden.

    Args:
        phase_id: Phasen-ID
        audio_before: Audio vor der Phase
        audio_after: Audio nach der Phase
    """
    state = _get_state()

    if phase_id not in DYNAMICS_PHASES:
        return

    def _crest(a: np.ndarray) -> float:
        x = np.asarray(a, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return float(20.0 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12) + 1e-12))

    crest_before = _crest(audio_before)
    crest_after = _crest(audio_after)
    change = crest_before - crest_after  # Positiv = Reduktion

    state.cumulative_reduction_db += max(0.0, change)
    state.phases_processed.append(phase_id)

    if state.cumulative_reduction_db >= MAX_CREST_REDUCTION_DB:
        state.budget_exhausted = True
        logger.warning(
            "§DYN-Grenze: ERSCHÖPFT nach %s — kumulativ %.1f/%.1f dB "
            "(crest %.1f→%.1f dB). Nachfolgende Dynamics-Phasen auf %.2f.",
            phase_id,
            state.cumulative_reduction_db,
            MAX_CREST_REDUCTION_DB,
            state.original_crest_db,
            state.original_crest_db - state.cumulative_reduction_db,
            BUDGET_EXHAUSTED_STRENGTH_CAP,
        )
    else:
        logger.debug(
            "§DYN-Grenze: %s crest %.1f→%.1f dB (Δ=%.1f dB) — kumulativ %.1f/%.1f dB",
            phase_id,
            crest_before,
            crest_after,
            change,
            state.cumulative_reduction_db,
            MAX_CREST_REDUCTION_DB,
        )


def get_budget_summary() -> dict[str, Any]:
    """Gibt Zusammenfassung für Metadaten/Logging."""
    state = _get_state()
    return {
        "original_crest_db": round(state.original_crest_db, 2),
        "cumulative_reduction_db": round(state.cumulative_reduction_db, 2),
        "budget_exhausted": state.budget_exhausted,
        "phases_processed": list(state.phases_processed),
        "max_budget_db": MAX_CREST_REDUCTION_DB,
    }


__all__ = [
    "DynamicsBudgetState",
    "MAX_CREST_REDUCTION_DB",
    "DYNAMICS_PHASES",
    "reset_dynamics_budget",
    "initialize_budget",
    "get_available_strength_cap",
    "report_crest_change",
    "get_budget_summary",
]
