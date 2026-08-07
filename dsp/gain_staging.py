from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "gain_staging"
    category: str = "level"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
gain_staging_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"target_lufs": -18.0},
        "safe_ranges": {"target_lufs": {"min": -30.0, "max": -8.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Übersteuerung",
            "expected_when": "target_lufs > -10.0",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["gain_applied"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
import numpy as np
import numpy.typing as npt


class GainStaging:
    """
    SOTA-konformes Gain-Staging (EBU R128, ITU-R BS.1770)
    """

    def __init__(self, target_lufs: float = -18.0):
        self.target_lufs = target_lufs

    # Audit: Contract-Infos loggen (optional)
    def log_contract(self):
        import logging

        logging.info("[DSPContract] %s", asdict(gain_staging_contract))

    def process(self, audio: npt.NDArray[np.float64], measured_lufs: float) -> npt.NDArray[np.float64]:
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))
        gain_db = self.target_lufs - measured_lufs
        # Safety: limit gain to ±24 dB to prevent extreme amplification
        gain_db = float(np.clip(gain_db, -24.0, 24.0))
        gain = 10.0 ** (gain_db / 20.0)
        self.log_contract()
        out = audio * gain
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)
        return out  # type: ignore[no-any-return]
