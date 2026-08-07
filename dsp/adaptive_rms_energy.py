import logging

"""
adaptive_rms_energy.py - SOTA-konformes RMS/Energy-Modul für Aurik 6.0
Dieses Modul berechnet adaptiv RMS und Energie und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_rms_energy"
    category: str = "energy"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_rms_energy_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"frame_length": 2048, "hop_length": 512, "center": True},
        "safe_ranges": {
            "frame_length": {"min": 128, "max": 8192},
            "hop_length": {"min": 32, "max": 4096},
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
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Fehlmessung bei DC-Offset",
            "expected_when": "center=False",
            "severity": 0.1,
        }
    ],
    reports={"self_metrics": ["rms", "energy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveRMSEnergy:
    def __init__(self, frame_length=2048, hop_length=512, center=True):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_rms_energy_contract))

    def rms(self, y, **kwargs):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        """Berechnet den RMS-Wert adaptiv mit aktuellen Parametern."""
        frame_length = kwargs.get("frame_length", self.frame_length)
        hop_length = kwargs.get("hop_length", self.hop_length)
        center = kwargs.get("center", self.center)
        if center:
            pad = frame_length // 2
            y = np.pad(y, (pad, pad), mode="reflect")
        rms_vals = np.sqrt(np.convolve(y**2, np.ones(frame_length) / frame_length, mode="valid")[::hop_length])
        return rms_vals

    def energy(self, y, **kwargs):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        """Berechnet die Energie adaptiv mit aktuellen Parametern."""
        frame_length = kwargs.get("frame_length", self.frame_length)
        hop_length = kwargs.get("hop_length", self.hop_length)
        center = kwargs.get("center", self.center)
        if center:
            pad = frame_length // 2
            y = np.pad(y, (pad, pad), mode="reflect")
        energy_vals = np.convolve(y**2, np.ones(frame_length), mode="valid")[::hop_length]
        return energy_vals

    def auto_optimize(self, y):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        """Automatische Anpassung der Frame-Parameter je nach Signal."""
        if len(y) < 4096:
            self.frame_length = 256
            self.hop_length = 64
        elif len(y) < 16384:
            self.frame_length = 1024
            self.hop_length = 256
        else:
            self.frame_length = 2048
            self.hop_length = 512
