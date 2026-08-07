import logging

"""
adaptive_spectral_flux.py - SOTA-konformes Spectral Flux Modul für Aurik 6.0
Dieses Modul ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_spectral_flux"
    category: str = "spectral_flux"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_spectral_flux_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"n_fft": 2048, "hop_length": 512},
        "safe_ranges": {"n_fft": {"min": 256, "max": 8192}},
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
            "expected_when": "n_fft zu niedrig",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["flux_value"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveSpectralFlux:
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, center=True):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_spectral_flux_contract))

    def spectral_flux(self, y, **kwargs):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        kwargs.get("sr", self.sr)
        n_fft = kwargs.get("n_fft", self.n_fft)
        hop_length = kwargs.get("hop_length", self.hop_length)
        center = kwargs.get("center", self.center)
        if center:
            pad = n_fft // 2
            y = np.pad(y, (pad, pad), mode="reflect")
        prev_spectrum = None
        flux = []
        for i in range(0, len(y) - n_fft + 1, hop_length):
            frame = y[i : i + n_fft]
            spectrum = np.abs(np.fft.rfft(frame))
            if prev_spectrum is not None:
                flux_val = np.sqrt(np.sum((spectrum - prev_spectrum) ** 2))
                flux.append(flux_val)
            prev_spectrum = spectrum
        return np.array(flux)

    def auto_optimize(self, y, sr):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        if len(y) < 4096:
            self.n_fft = 256
            self.hop_length = 64
        elif len(y) < 16384:
            self.n_fft = 1024
            self.hop_length = 256
        else:
            self.n_fft = 2048
            self.hop_length = 512
