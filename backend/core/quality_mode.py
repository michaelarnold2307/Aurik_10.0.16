"""
§2.59 QualityMode-Validierung (2026-07-09)

Zentrale Validierung aller Quality-Mode-Strings.
Verhindert stille Fallbacks durch Tippfehler wie "restoraton".

Usage:
  from backend.core.quality_mode import validate_mode, QUALITY_MODES
  mode = validate_mode(user_input)  # "restoration" → "restoration"
                                     # "restoraton" → WARNING + Fallback
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── QualityMode Enum ─────────────────────────────────────────────────────


class QualityMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    MAXIMUM = "maximum"

    @property
    def is_ml_enabled(self) -> bool:
        return self in (QualityMode.QUALITY, QualityMode.MAXIMUM)


class QualityModeConfig:
    _current_mode: QualityMode = QualityMode.QUALITY
    _ml_phases_enabled: bool = True

    @classmethod
    def set_mode(cls, mode: QualityMode) -> None:
        cls._current_mode = mode

    @classmethod
    def get_mode(cls) -> QualityMode:
        return cls._current_mode

    @classmethod
    def should_use_ml(cls, phase_name: str, defect_severity: float = 0.0) -> bool:
        if not cls._ml_phases_enabled:
            return False
        if cls._current_mode == QualityMode.FAST:
            return False
        return cls._current_mode.is_ml_enabled


def is_phase_ml_enabled(phase_number: int) -> bool:
    _CRITICAL: frozenset[int] = frozenset({3, 23, 24, 29, 55, 66})
    if phase_number not in _CRITICAL:
        return False
    return QualityModeConfig._ml_phases_enabled


def log_mode_decision(phase_name: str, use_ml: bool, message: str = "") -> None:
    logger.debug(
        "ML-%s %s: %s", "ON" if use_ml else "OFF", phase_name, message or ("enabled" if use_ml else "disabled")
    )


# ──

QUALITY_MODES: frozenset[str] = frozenset(
    {
        "restoration",
        "quality",
        "maximum",
        "studio_2026",
        "balanced",
        "fast",
    }
)

MODE_ALIASES: dict[str, str] = {
    "restoration": "quality",
    "studio_2026": "maximum",
    "quality": "quality",
    "maximum": "maximum",
    "balanced": "balanced",
    "fast": "fast",
}

MODE_FALLBACK = "quality"


def validate_mode(mode: Any, fallback: str = MODE_FALLBACK) -> str:
    """Validiert und normalisiert einen Quality-Mode-String.

    Args:
        mode: Roher Mode-String vom User/API
        fallback: Fallback-Mode bei ungültiger Eingabe

    Returns:
        Kanonischer Mode-String
    """
    if mode is None or not isinstance(mode, str):
        logger.warning(
            "QualityMode: invalid type %s, Ersatzpfad to '%s'",
            type(mode).__name__ if mode is not None else "None",
            fallback,
        )
        return fallback

    mode_lower = mode.strip().lower()

    if mode_lower in MODE_ALIASES:
        canonical = MODE_ALIASES[mode_lower]
        if canonical != mode_lower:
            logger.debug("QualityMode: alias '%s' → '%s'", mode_lower, canonical)
        return canonical

    # Check partial matches for common typos
    for known in QUALITY_MODES:
        if mode_lower in known or known in mode_lower:
            logger.warning(
                "QualityMode: '%s' is not a valid Betriebsart. Did you mean '%s'? Falling back to '%s'.",
                mode,
                known,
                fallback,
            )
            return fallback

    logger.warning(
        "QualityMode: '%s' is not a valid Betriebsart. Valid: %s. Ersatzpfad to '%s'.",
        mode,
        sorted(QUALITY_MODES),
        fallback,
    )
    return fallback
