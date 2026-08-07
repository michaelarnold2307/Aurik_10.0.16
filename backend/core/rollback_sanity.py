"""
backend/core/rollback_sanity.py — §v10.701 Rollback Audio Integrity Check

Validiert Audio nach jedem Rollback (CIG, SFT, AFG) auf Integrität.
Verhindert, dass −92,4 dBFS Stille, NaN oder Null-Signale an
Folgephasen weitergereicht werden.

Spec: §18.3/§G92 — Rollback-Sanity-Pflicht
GEBOTE: §G92
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Minimale RMS-Schwelle in dBFS — alles darunter gilt als Stille/Schaden
_MIN_RMS_DBFS: float = -60.0
# Minimaler Peak — alles darunter gilt als Null-Signal
_MIN_PEAK: float = 1e-6


@dataclass
class RollbackSanityResult:
    """Ergebnis der Rollback-Audio-Integritätsprüfung."""

    valid: bool
    rms_dbfs: float = 0.0
    peak: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    reason: str = ""
    recommended_action: str = "use_rollback_target"  # "use_rollback_target" | "use_next_checkpoint" | "use_original"


def validate_rollback_audio(
    audio: np.ndarray,
    source_phase: str = "unknown",
    *,
    min_rms_dbfs: float = _MIN_RMS_DBFS,
    min_peak: float = _MIN_PEAK,
) -> RollbackSanityResult:
    """Prüft ob das Audio nach einem Rollback noch intakt ist.

    Args:
        audio: Das Audio-Array NACH dem Rollback (von dem aus weiterverarbeitet wird).
        source_phase: Phase-ID, die den Rollback ausgelöst hat (für Logging).
        min_rms_dbfs: RMS-Schwelle in dBFS (Standard: −60).
        min_peak: Peak-Schwelle (Standard: 1e−6).

    Returns:
        RollbackSanityResult mit Validierungsstatus und empfohlener Aktion.
    """
    arr = np.asarray(audio, dtype=np.float32)

    # Check 1: NaN/Inf
    has_nan = bool(np.any(np.isnan(arr)))
    has_inf = bool(np.any(np.isinf(arr)))
    if has_nan or has_inf:
        logger.critical(
            "§v10.701 Rollback-Sanity %s: NaN=%s Inf=%s — Checkpoint-Wiederherstellung erforderlich",
            source_phase,
            has_nan,
            has_inf,
        )
        return RollbackSanityResult(
            valid=False,
            has_nan=has_nan,
            has_inf=has_inf,
            reason=f"NaN={has_nan}, Inf={has_inf}",
            recommended_action="use_next_checkpoint",
        )

    # Check 2: RMS (Stille-Detektion)
    rms = float(np.sqrt(np.mean(arr**2)) + 1e-15)
    rms_dbfs = float(20.0 * np.log10(rms))
    if rms_dbfs < min_rms_dbfs:
        logger.critical(
            "§v10.701 Rollback-Sanity %s: Audio zerstört (RMS=%.1f dBFS < %.0f dBFS) — Checkpoint-Wiederherstellung",
            source_phase,
            rms_dbfs,
            min_rms_dbfs,
        )
        return RollbackSanityResult(
            valid=False,
            rms_dbfs=rms_dbfs,
            reason=f"RMS={rms_dbfs:.1f} dBFS < {min_rms_dbfs:.0f} dBFS",
            recommended_action="use_next_checkpoint",
        )

    # Check 3: Peak (Null-Signal-Detektion)
    peak = float(np.max(np.abs(arr)))
    if peak < min_peak:
        logger.critical(
            "§v10.701 Rollback-Sanity %s: Kein Signal (Peak=%.1e < %.1e) — Checkpoint-Wiederherstellung",
            source_phase,
            peak,
            min_peak,
        )
        return RollbackSanityResult(
            valid=False,
            rms_dbfs=rms_dbfs,
            peak=peak,
            reason=f"Peak={peak:.1e} < {min_peak:.1e}",
            recommended_action="use_next_checkpoint",
        )

    # All checks passed
    logger.debug(
        "§v10.701 Rollback-Sanity %s: Audio intakt (RMS=%.1f dBFS, Peak=%.3f)",
        source_phase,
        rms_dbfs,
        peak,
    )
    return RollbackSanityResult(
        valid=True,
        rms_dbfs=rms_dbfs,
        peak=peak,
    )
