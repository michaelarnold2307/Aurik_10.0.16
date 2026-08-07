"""
PerceptualQualityGates: Integration von ViSQOL (--audio), CDPAM (§4.4-konforme Musik-Metriken).
# NISQA/DNSMOS entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
Automatisches Reprocessing und Multi-Pass-Optimierung.
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


class PerceptualQualityGates:
    """Perceptual Quality Gates für Musik-Restaurierung (§4.4-konform).

    Integriert:
    - ViSQOL v3 (--audio mode, MOS 1.0–5.0)

    Verboten (§4.4+§10.2):
    - NISQA, DNSMOS, PESQ, STOI, ViSQOL --speech, CDPAM
    """

    def __init__(self, reprocess_callback: Callable[[], None]) -> None:
        """Initialisiert Perceptual Quality Gates.

        Args:
            reprocess_callback: Callback-Funktion für Reprocessing bei Failure.
        """
        # NISQA/DNSMOS/CDPAM entfernt — verboten §4.4+§10.2 (Sprach-Metriken / CDPAM)
        # Nur ViSQOL --audio (nicht --speech!) ist erlaubt.
        self.thresholds: dict[str, float] = {"ViSQOL": 3.5}
        self.reprocess_callback = reprocess_callback
        logger.info("PerceptualQualityGates initialisiert with thresholds: %s", self.thresholds)

    def evaluate(self, metrics: dict[str, float]) -> bool:
        """Evaluiert Metriken gegen Schwellwerte.

        Args:
            metrics: Dictionary mit Metrik-Namen und Werten.

        Returns:
            True wenn alle Schwellwerte erfüllt, sonst False.
        """
        # §v10.93 NaN-Guard: NaN-Origin-Logging bevor Werte stumm auf 0.0 fallen.
        # So wird sichtbar, WELCHE Metrik NaN produziert hat — nicht nur DASS das Gate fehlschlägt.
        _nan_keys = [k for k, v in metrics.items() if not np.isfinite(float(v))]
        if _nan_keys:
            logger.warning("Quality Gate: NaN/Inf in metrics: %s (values coerced to 0.0)", _nan_keys)
        metrics_clean = {k: float(np.nan_to_num(v, nan=0.0)) for k, v in metrics.items()}

        failed = [k for k, v in metrics_clean.items() if k in self.thresholds and v < self.thresholds[k]]
        if failed:
            logger.warning("Quality Gate fehlgeschlagen: %s", failed)
            self.reprocess_callback()
            return False
        logger.info("Quality Gate PASSED")
        return True
