"""§AM: PerDefectRepairVerifier — Before/After-Prüfung pro Defekt mit Auto-Retry.

Stellt sicher dass JEDE einzelne Defektreparatur verifiziert wird:
  1. Before/After-Messung an der exakten Defektposition
  2. Automatischer Retry mit reduzierter/anderer Stärke bei Misserfolg
  3. Maximal 3 Retries pro Defekt
  4. Integration mit §AF DynamicsGuard für Kontinuitäts-Check

Teamwork:
  - §AF RepairDynamicsGuard → Kontinuität nach Reparatur
  - §AG SibilanceMaxRepair → Sibilance-spezifische Verifikation
  - DefectScanner → Original-Defektdaten als Baseline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RepairVerification:
    """Ergebnis der Verifikation einer einzelnen Defektreparatur."""

    defect_type: str = ""
    defect_sample: int = 0
    before_rms: float = 0.0
    after_rms: float = 0.0
    before_peak: float = 0.0
    after_peak: float = 0.0
    rms_reduction_db: float = 0.0
    peak_reduction_db: float = 0.0
    repair_ok: bool = False
    retries_used: int = 0
    continuity_ok: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class BatchRepairReport:
    """Gesamt-Report über alle verifizierten Reparaturen."""

    total_defects: int = 0
    repaired_ok: int = 0
    retried: int = 0
    failed: int = 0
    avg_rms_reduction_db: float = 0.0
    avg_retries: float = 0.0
    verifications: list[RepairVerification] = field(default_factory=list)


class PerDefectRepairVerifier:
    """Verifiziert jede Defektreparatur einzeln mit Auto-Retry.

    Wird vom UV3-Phase-Loop nach jeder Reparatur-Phase aufgerufen.
    """

    MAX_RETRIES = 3
    MIN_RMS_REDUCTION_DB = 0.5  # Minimum Verbesserung für "repaired_ok"
    MAX_RMS_REDUCTION_DB = 30.0  # Maximum (Over-Repair-Schutz)

    def __init__(self, dynamics_guard: Any = None) -> None:
        self._dynamics = dynamics_guard

    def verify_defect(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        defect_start: int,
        defect_end: int,
        defect_type: str = "UNKNOWN",
    ) -> RepairVerification:
        """Verifiziert eine einzelne Defektreparatur.

        Args:
            audio_before/after: Audio vor/nach der Reparatur
            sr: Sample-Rate
            defect_start/end: Sample-Indizes des Defekts
            defect_type: Defekt-Typ
        """
        mono_before = np.mean(audio_before, axis=0) if audio_before.ndim == 2 else audio_before
        mono_after = np.mean(audio_after, axis=0) if audio_after.ndim == 2 else audio_after

        s0 = max(0, defect_start)
        s1 = min(len(mono_after), defect_end)
        if s1 - s0 < 2:
            return RepairVerification(defect_type=defect_type, defect_sample=s0)

        # ── Vorher/Nachher-Messung ──
        seg_before = mono_before[s0:s1]
        seg_after = mono_after[s0:s1]

        v = RepairVerification(
            defect_type=defect_type,
            defect_sample=s0,
            before_rms=float(np.sqrt(np.mean(seg_before**2) + 1e-12)),
            after_rms=float(np.sqrt(np.mean(seg_after**2) + 1e-12)),
            before_peak=float(np.max(np.abs(seg_before)) + 1e-12),
            after_peak=float(np.max(np.abs(seg_after)) + 1e-12),
        )

        if v.before_rms > 1e-10 and v.after_rms > 1e-10:
            v.rms_reduction_db = float(20.0 * np.log10(v.before_rms / v.after_rms))
        if v.before_peak > 1e-10 and v.after_peak > 1e-10:
            v.peak_reduction_db = float(20.0 * np.log10(v.before_peak / v.after_peak))

        # ── Bewertung ──
        # Für impulsive Defekte (Clicks): Peak-Reduktion ist wichtiger
        if defect_type in ("CLICKS", "CLICK_POP", "CLICK", "CRACKLE", "TRANSPORT_BUMP"):
            reduction = v.peak_reduction_db
        else:
            reduction = v.rms_reduction_db

        # Reparatur OK wenn: genug reduziert aber nicht überreduziert
        v.repair_ok = reduction >= self.MIN_RMS_REDUCTION_DB and reduction <= self.MAX_RMS_REDUCTION_DB

        if reduction < self.MIN_RMS_REDUCTION_DB:
            v.warnings.append(f"Under-repair: {reduction:.1f} dB reduction (min {self.MIN_RMS_REDUCTION_DB})")
        if reduction > self.MAX_RMS_REDUCTION_DB:
            v.warnings.append(f"Over-repair: {reduction:.1f} dB reduction (max {self.MAX_RMS_REDUCTION_DB})")

        # §AF Continuity-Check
        if self._dynamics is not None:
            try:
                ct = self._dynamics.verify_continuity(mono_after, sr, [s0, s1], channel=None)
                v.continuity_ok = ct.continuity_ok
                if not ct.continuity_ok:
                    v.warnings.append(f"Continuity violation: {ct.max_envelope_deviation_db:.1f} dB")
            except Exception as e:
                logger.warning("per_defect_repair_verifier.py::pruefen_defect Ersatzpfad: %s", e)

        return v

    def verify_batch(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        defects: list[dict],
    ) -> BatchRepairReport:
        """Verifiziert alle Defektreparaturen in einem Batch.

        Args:
            audio_before/after: Audio vor/nach
            sr: Sample-Rate
            defects: Liste von {'type': str, 'start_sample': int, 'end_sample': int, ...}
        """
        report = BatchRepairReport(total_defects=len(defects))

        for d in defects:
            s0 = d.get("start_sample", 0)
            s1 = d.get("end_sample", s0 + 100)
            dtype = d.get("type", "UNKNOWN")

            v = self.verify_defect(audio_before, audio_after, sr, s0, s1, dtype)
            v.retries_used = d.get("retries", 0)

            if v.repair_ok:
                report.repaired_ok += 1
            elif v.retries_used < self.MAX_RETRIES:
                report.retried += 1
            else:
                report.failed += 1

            report.verifications.append(v)

        if report.repaired_ok > 0:
            report.avg_rms_reduction_db = float(
                np.mean([v.rms_reduction_db for v in report.verifications if v.repair_ok])
            )
        report.avg_retries = (
            float(np.mean([v.retries_used for v in report.verifications])) if report.verifications else 0.0
        )

        return report

    def needs_retry(self, v: RepairVerification, retries_used: int) -> tuple[bool, float]:
        """Bestimmt ob und mit welcher Stärke retried werden soll.

        Returns: (should_retry, new_strength_multiplier)
        """
        if v.repair_ok:
            return False, 1.0
        if retries_used >= self.MAX_RETRIES:
            return False, 1.0

        # Reduziere Stärke pro Retry
        strength_mod = max(0.3, 1.0 - 0.25 * retries_used)

        # Bei Under-Repair: stärker
        if v.rms_reduction_db < self.MIN_RMS_REDUCTION_DB:
            strength_mod = min(1.5, 1.0 + 0.15 * retries_used)
        # Bei Over-Repair: schwächer
        elif v.rms_reduction_db > self.MAX_RMS_REDUCTION_DB:
            strength_mod = max(0.2, 1.0 - 0.30 * retries_used)

        return True, strength_mod
