"""RollbackSanityCheck — Audio-Validierung nach Pipeline-Rollback.

Spec 18.3, G92 RELEASE_MUST.
Verhindert, dass beschädigtes Audio nach einem Rollback an Folgephasen
weitergereicht wird. Prüft auf Stille, NaN, und Null-Signal.

- rms_db < -60 dBFS → Stille nach Rollback
- NaN im Signal → numerische Korruption
- peak < 1e-6 → kein Signal

Integration: NACH jedem Rollback in unified_restorer_v3._execute_pipeline.
Bei False: nächsten Checkpoint als Quelle verwenden, nicht Rollback-Ziel.

Autor: Aurik 10 — August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Schwellwerte gemaess Spec 18.3
SILENCE_THRESHOLD_DBFS: float = -60.0
MIN_PEAK: float = 1e-6
MIN_RMS: float = 1e-8


@dataclass
class RollbackSanityResult:
    """Ergebnis der Rollback-Sanity-Prüfung."""

    passed: bool
    rms_db: float = -200.0
    peak: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    source_phase: str = ""
    failure_reason: str = ""
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rms_db": self.rms_db,
            "peak": self.peak,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "source_phase": self.source_phase,
            "failure_reason": self.failure_reason,
            "recommended_action": self.recommended_action,
        }


def validate_rollback_audio(
    audio: np.ndarray,
    source_phase: str = "unknown",
    *,
    silence_threshold_db: float = SILENCE_THRESHOLD_DBFS,
    min_peak: float = MIN_PEAK,
) -> RollbackSanityResult:
    """Prüft ob das Audio nach einem Rollback noch valide ist.

    Spec 18.3: Drei Checks — Stille, NaN, kein Signal.

    Args:
        audio: Audio-Array (float32/float64, beliebige Shape)
        source_phase: Name der Phase, die den Rollback ausgelöst hat
        silence_threshold_db: RMS-Schwelle für Stille-Erkennung (dBFS)
        min_peak: Minimaler Peak-Wert für Signal-Präsenz

    Returns:
        RollbackSanityResult mit passed=True wenn Audio in Ordnung ist.

        Bei False:
        - rms_db < -60 → Stille (Rollback hat Audio gelöscht)
        - has_nan → numerische Korruption
        - peak < 1e-6 → kein Signal (Null-Array)
    """
    arr = np.asarray(audio, dtype=np.float64)
    result = RollbackSanityResult(passed=False, source_phase=source_phase)

    # Check 1: NaN / Inf
    result.has_nan = bool(np.any(np.isnan(arr)))
    result.has_inf = bool(np.any(np.isinf(arr)))

    if result.has_nan:
        result.failure_reason = f"NaN detected in audio after rollback from {source_phase}"
        result.recommended_action = "Use PREVIOUS checkpoint (not rollback target) as audio source; NaN indicates numerical corruption during rollback"
        logger.critical("RollbackSanityCheck FAILED: %s", result.failure_reason)
        return result

    if result.has_inf:
        result.failure_reason = f"Inf detected in audio after rollback from {source_phase}"
        result.recommended_action = "Use PREVIOUS checkpoint as audio source"
        logger.critical("RollbackSanityCheck FAILED: %s", result.failure_reason)
        return result

    # Check 2: RMS (Stille-Erkennung)
    rms = float(np.sqrt(np.mean(arr**2)) + 1e-20)
    result.peak = float(np.max(np.abs(arr)))
    result.rms_db = float(20.0 * np.log10(max(rms, 1e-20)))

    if result.rms_db < silence_threshold_db:
        result.failure_reason = (
            f"Silence detected after rollback: RMS={result.rms_db:.1f} dBFS "
            f"(threshold={silence_threshold_db:.0f} dBFS)"
        )
        result.recommended_action = (
            "NEXT checkpoint (not rollback target) as audio source. "
            "The rollback produced near-silence — likely all phases reverted to zero-state audio."
        )
        logger.critical("RollbackSanityCheck FAILED: %s", result.failure_reason)
        return result

    # Check 3: Signal-Praesenz (Peak)
    if result.peak < min_peak:
        result.failure_reason = (
            f"No signal after rollback: peak={result.peak:.2e} "
            f"(threshold={min_peak})"
        )
        result.recommended_action = "NEXT checkpoint as audio source"
        logger.critical("RollbackSanityCheck FAILED: %s", result.failure_reason)
        return result

    # All checks passed
    result.passed = True
    logger.debug(
        "RollbackSanityCheck PASSED: phase=%s, rms=%.1f dBFS, peak=%.6f",
        source_phase, result.rms_db, result.peak,
    )
    return result


class RollbackSanityGuard:
    """Guard für Rollback-Sanity in der Pipeline.

    Wird NACH jedem Rollback aufgerufen. Speichert den letzten
    validen Checkpoint für den Fall dass ein Rollback fehlschlägt.

    Usage in unified_restorer_v3._execute_pipeline:
        guard = RollbackSanityGuard()
        ...
        result = guard.check(audio, "phase_19")
        if not result.passed:
            audio = guard.get_fallback_audio()
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, np.ndarray] = {}
        self._last_valid: np.ndarray | None = None
        self._failure_count: int = 0
        self._total_checks: int = 0

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def total_checks(self) -> int:
        return self._total_checks

    @property
    def failure_rate(self) -> float:
        if self._total_checks == 0:
            return 0.0
        return self._failure_count / self._total_checks

    def save_checkpoint(self, phase_id: str, audio: np.ndarray) -> None:
        """Speichert einen Checkpoint vor einem Rollback."""
        self._checkpoints[phase_id] = np.asarray(audio, dtype=np.float32).copy()
        self._last_valid = self._checkpoints[phase_id]
        logger.debug("RollbackSanityGuard: checkpoint saved for %s", phase_id)

    def check(
        self,
        audio: np.ndarray,
        source_phase: str = "unknown",
        *,
        use_fallback_on_failure: bool = True,
    ) -> tuple[RollbackSanityResult, np.ndarray]:
        """Prüft Audio nach Rollback und gibt ggf. Fallback zurück.

        Returns:
            Tuple (result, audio_to_use).
            Wenn result.passed: audio_to_use == audio (unverändert).
            Wenn nicht passed: audio_to_use == letzter valider Checkpoint.
        """
        self._total_checks += 1
        result = validate_rollback_audio(audio, source_phase)

        if not result.passed:
            self._failure_count += 1
            if use_fallback_on_failure and self._last_valid is not None:
                logger.warning(
                    "RollbackSanityGuard: using fallback checkpoint after %s failure (%d total failures)",
                    source_phase,
                    self._failure_count,
                )
                return result, self._last_valid.copy()
            logger.error(
                "RollbackSanityGuard: no fallback available for %s failure! Returning damaged audio.",
                source_phase,
            )
            return result, audio

        self._last_valid = np.asarray(audio, dtype=np.float32).copy()
        return result, audio

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_checks": self._total_checks,
            "failures": self._failure_count,
            "failure_rate": self.failure_rate,
            "checkpoints_saved": len(self._checkpoints),
        }


# Singleton
_sanity_guard: RollbackSanityGuard | None = None


def get_rollback_sanity_guard() -> RollbackSanityGuard:
    """Singleton-Zugriff auf den RollbackSanityGuard."""
    global _sanity_guard
    if _sanity_guard is None:
        _sanity_guard = RollbackSanityGuard()
    return _sanity_guard
