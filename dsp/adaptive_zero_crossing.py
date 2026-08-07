import logging

"""
adaptive_zero_crossing.py - SOTA-konformes Zero-Crossing Modul für Aurik 6.0

Dieses Modul ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_zero_crossing"
    category: str = "zero_crossing"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_zero_crossing_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"frame_length": 2048, "hop_length": 512},
        "safe_ranges": {
            "frame_length": {"min": 64, "max": 8192},
            "hop_length": {"min": 16, "max": 4096},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Fehlklassifikation",
            "expected_when": "frame_length zu klein",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["zcr_value"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveZeroCrossingRate:
    def __init__(self, frame_length=2048, hop_length=512, center=True):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_zero_crossing_contract))

    def zero_crossing_rate(self, y, **kwargs):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        frame_length = kwargs.get("frame_length", self.frame_length)
        hop_length = kwargs.get("hop_length", self.hop_length)
        center = kwargs.get("center", self.center)
        if center:
            pad = frame_length // 2
            y = np.pad(y, (pad, pad), mode="reflect")
        zcr = []
        for i in range(0, len(y) - frame_length + 1, hop_length):
            frame = y[i : i + frame_length]
            zc = np.sum(np.abs(np.diff(np.sign(frame)))) // 2
            zcr.append(zc / frame_length)
        return np.array(zcr)

    def auto_optimize(self, y):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        if len(y) < 4096:
            self.frame_length = 256
            self.hop_length = 64
        elif len(y) < 16384:
            self.frame_length = 1024
            self.hop_length = 256
        else:
            self.frame_length = 2048
            self.hop_length = 512
