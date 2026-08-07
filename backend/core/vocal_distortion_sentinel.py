"""
§2.59 Vocal Distortion Sentinel — SENSOR (2026-07-09)

Misst Gesangsqualität und schreibt Befunde in den RestorationContext.
Die Stärke-Entscheidung trifft UV3/SongCalibration auf Basis dieser Messungen.

Architektur:
  Sentinel misst → schreibt restoration_context → UV3 entscheidet Stärke → Phase führt aus

Der Sentinel ist ein SENSOR, kein AKTOR.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VocalDistortionSentinel:
    """Sensor: misst Gesangsqualität, schreibt in restoration_context."""

    def __init__(self, singing_confidence: float = 0.0) -> None:
        self._singing_conf = singing_confidence
        self._hnr_before: float | None = None
        self._hnr_after: float | None = None
        self._harmonic_restoration_active: bool = False
        self._deesser_active: bool = False

    def set_baseline_hnr(self, hnr_db: float) -> None:
        self._hnr_before = hnr_db

    def record_phase_active(self, phase_id: str) -> None:
        if "harmonic_restoration" in phase_id:
            self._harmonic_restoration_active = True
        if "de_esser" in phase_id or "deesser" in phase_id:
            self._deesser_active = True

    def measure(self, post_hnr_db: float | None = None) -> dict[str, Any]:
        """Misst und schreibt Befunde. Keine Stärke-Entscheidungen.

        Returns dict zur Integration in restoration_context.
        UV3/SongCalibration nutzt diese Messwerte für Stärke-Berechnung.
        """
        findings: dict[str, Any] = {
            "singing_confidence": self._singing_conf,
            "hnr_before_db": self._hnr_before,
            "hnr_after_db": post_hnr_db,
            "harmonic_restoration_active": self._harmonic_restoration_active,
            "deesser_active": self._deesser_active,
        }

        # Messwert 1: HNR-Veränderung
        if post_hnr_db is not None and self._hnr_before is not None:
            hnr_delta = post_hnr_db - self._hnr_before
            findings["hnr_delta_db"] = hnr_delta
            if hnr_delta < -3.0:
                logger.warning(
                    "🎤 Sentinel: HNR-Abfall %.1f dB (%.1f→%.1f) — "
                    "SongCalibration sollte Harmonic-Restoration-Stärke reduzieren",
                    hnr_delta,
                    self._hnr_before,
                    post_hnr_db,
                )
            elif hnr_delta < -1.5:
                logger.info(
                    "🎤 Sentinel: leichter HNR-Abfall %.1f dB — beobachten",
                    hnr_delta,
                )

        # Messwert 2: Fehlende Schutz-Phasen
        if self._harmonic_restoration_active and not self._deesser_active:
            if self._singing_conf >= 0.25:
                findings["missing_deesser"] = True
                logger.warning(
                    "🎤 Sentinel: Harmonic-Restoration aktiv (singing=%.2f) "
                    "aber KEIN De-Esser — PhaseInteractionDenker sollte "
                    "Verarbeitungsschritt_19_de_esser + Verarbeitungsschritt_43_ml_deesser injizieren",
                    self._singing_conf,
                )

        return findings
