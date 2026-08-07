"""
ConfidenceBasedProcessing: Dynamische Anpassung und Rollback-Mechanismen.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


_instance: ConfidenceBasedProcessing | None = None
_lock = threading.Lock()


def get_confidence_processor(rollback_callback: Callable[[str], None] | None = None) -> ConfidenceBasedProcessing:
    """Gibt zurück: or create ConfidenceBasedProcessing singleton.

    Args:
        rollback_callback: Callback function for rollbacks (only used on first call)

    Returns:
        ConfidenceBasedProcessing singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ConfidenceBasedProcessing(rollback_callback or (lambda x: None))
    return _instance


class ConfidenceBasedProcessing:
    def __init__(self, rollback_callback: Callable[[str], None]) -> None:
        self.rollback_callback = rollback_callback
        self.confidence_threshold = 0.5
        self.max_change = 0.2
        logger.info("ConfidenceBasedProcessing initialisiert")

    def adjust_strength(self, base_strength: float, confidence: float) -> float:
        """Adjust processing strength based on confidence.

        Args:
            base_strength: Base processing strength (0.0-1.0)
            confidence: Confidence score (0.0-1.0)

        Returns:
            Adjusted strength value (0.0-1.0)
        """
        import math

        import numpy as np

        base_strength = float(base_strength)
        confidence = float(confidence)

        if not math.isfinite(base_strength) or not math.isfinite(confidence):
            logger.warning(
                "confidence_processing: non-finite Eingabe base=%.4f conf=%.4f → Ersatzpfad 0.5",
                float(base_strength) if math.isfinite(base_strength) else float("nan"),
                float(confidence) if math.isfinite(confidence) else float("nan"),
            )
            return 0.5  # Safe fallback

        base_strength = float(np.clip(base_strength, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence < self.confidence_threshold:
            result = base_strength * confidence
        else:
            result = min(base_strength, base_strength * (1 + self.max_change))

        result = float(np.clip(result, 0.0, 1.0))
        if not math.isfinite(result):
            logger.warning(
                "confidence_processing: non-finite Ergebnis=%.4f (base=%.4f conf=%.4f) → Ersatzpfad 0.5",
                float(result) if math.isfinite(result) else float("nan"),
                float(base_strength),
                float(confidence),
            )
            return 0.5

        return result

    def check_and_rollback(self, module_name: str, confidence: float) -> None:
        """Prüft confidence and trigger rollback if needed.

        Args:
            module_name: Name of the module to potentially rollback
            confidence: Confidence score (0.0-1.0)
        """
        import math

        import numpy as np

        confidence = float(confidence)
        if not math.isfinite(confidence):
            confidence = 0.0

        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence < self.confidence_threshold:
            self.rollback_callback(module_name)
