import logging
from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "stereo_matrix"
    category: str = "spatial"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
stereo_matrix_contract = DSPContract(
    io={
        "channels": "stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"width": 1.0, "balance": 0.0},
        "safe_ranges": {
            "width": {"min": 0.0, "max": 2.0},
            "balance": {"min": -1.0, "max": 1.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Phasenauslöschung", "expected_when": "width > 1.5", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["stereo_width", "balance"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class StereoMatrix:
    """
    SOTA-konforme Stereo-Matrix (M/S, Balance, Breite)
    """

    def __init__(self, width: float = 1.0, balance: float = 0.0):
        self.width = width
        self.balance = balance

    def process(self, audio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        logger.debug("[DSPContract] %s", asdict(stereo_matrix_contract))
        if audio.ndim != 2 or audio.shape[1] != 2:
            return audio
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mid = (audio[:, 0] + audio[:, 1]) / 2.0
        side = (audio[:, 0] - audio[:, 1]) / 2.0 * self.width
        # Level compensation to preserve energy at different widths
        compensation = 1.0 / max(1e-9, np.sqrt(0.5 * (1.0 + self.width**2)))
        left = (mid + side) * compensation
        right = (mid - side) * compensation
        # Balance as L/R gain offset (not additive to signal)
        if abs(self.balance) > 1e-6:
            bal_gain_l = np.clip(1.0 - self.balance, 0.0, 2.0)
            bal_gain_r = np.clip(1.0 + self.balance, 0.0, 2.0)
            left = left * bal_gain_l
            right = right * bal_gain_r
        out = np.stack([left, right], axis=1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)
        return out  # type: ignore[no-any-return]
