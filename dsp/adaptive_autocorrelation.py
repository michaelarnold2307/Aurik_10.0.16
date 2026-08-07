"""
Adaptive Autocorrelation DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten (klassische DSP, SOTA-Maximum).
Verwendet numpy für die Berechnung.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import correlate as _sc_correlate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_autocorrelation"
    category: str = "autocorrelation"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveAutocorrelation:
    """
    Klassische adaptive Autokorrelationsberechnung (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, normalize: bool = True, max_lag: int | None = None):
        self.normalize = normalize
        self.max_lag = max_lag

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def autocorrelation(self, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Berechnet die Autokorrelation adaptiv mit aktuellen Parametern.
        :param y: Eingabesignal (np.ndarray)
        :return: Autokorrelationsfunktion (np.ndarray)
        """
        self.log_contract()
        max_lag = kwargs.get("max_lag", self.max_lag)
        result = _sc_correlate(y, y, mode="full", method="fft")
        mid = len(result) // 2
        result = result[mid : mid + max_lag] if max_lag is not None else result[mid:]
        if self.normalize:
            result = result / np.max(np.abs(result))
        return result  # type: ignore[no-any-return]

    def auto_optimize(self, y: np.ndarray) -> None:
        """
        Automatische Anpassung der max_lag-Parameter je nach Signal.
        :param y: Eingabesignal (np.ndarray)
        """
        self.max_lag = min(2048, len(y) // 2)
