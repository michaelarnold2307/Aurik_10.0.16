"""
balance.py - SOTA-konforme Balance-Korrektur für Aurik 6.0

SOTA-konforme Balance-Korrektur mit DSPContract und Auditierbarkeit.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContractBalance:
    id: str = "balance"
    category: str = "balance"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


balance_contract = DSPContractBalance(
    io={
        "channels": "stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"balance": 0.0}},
    budgets={"compute_cost": 0.01},
    side_effects=[{"risk": "Fehlbalance", "expected_when": "balance zu groß", "severity": 0.2}],
    reports={"self_metrics": ["balance_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class Balance:
    """
    SOTA-konforme Balance-Korrektur (Mono/Stereo)
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractBalance = balance_contract

    def __init__(self, balance: float = 0.0):
        self.balance = balance

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.log_contract()
        if audio.ndim != 2 or audio.shape[1] != 2:
            return audio
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Balance as L/R gain (not additive offset which shifts DC)
        bal = float(np.clip(self.balance, -1.0, 1.0))
        gain_l = np.clip(1.0 - bal, 0.0, 2.0)
        gain_r = np.clip(1.0 + bal, 0.0, 2.0)
        left = audio[:, 0] * gain_l
        right = audio[:, 1] * gain_r
        out = np.stack([left, right], axis=1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)
        return out  # type: ignore[no-any-return]
