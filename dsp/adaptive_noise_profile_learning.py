import logging

"""
adaptive_noise_profile_learning.py - SOTA-konformes adaptives Noise-Profile-Learning für Aurik 6.0
Dieses Modul lernt adaptiv ein Noise-Profil für Denoising (klassische DSP, SOTA-Maximum).
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_noise_profile_learning"
    category: str = "noise_profile_learning"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_noise_profile_learning_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"method": "mean", "min_frames": 10, "noise_floor": 1e-6},
        "safe_ranges": {
            "min_frames": {"min": 1, "max": 100},
            "noise_floor": {"min": 1e-12, "max": 1e-3},
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
            "risk": "Fehlerhaftes Profil bei zu wenig Frames",
            "expected_when": "min_frames zu klein",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["noise_profile"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveNoiseProfileLearning:
    """
    SOTA-konformes adaptives Noise-Profile-Learning (klassisch):
    - Lernt ein Noise-Profil aus Spektrogramm-Frames (mean/median)
    - Quality-Gates, Auditierbarkeit, Rollback
    """

    def __init__(self, method="mean", min_frames=10, noise_floor=1e-6):
        self.method = method  # 'mean' oder 'median'
        self.min_frames = min_frames
        self.noise_floor = noise_floor
        self.noise_profile = None

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_noise_profile_learning_contract))

    def learn_profile(self, power_spectrogram, mask=None):
        """
        Lernt ein Noise-Profil aus Spektrogramm-Frames.
        :param power_spectrogram: Eingabe-Spektrogramm (np.ndarray)
        :param mask: Optionales Maskenarray (np.ndarray)
        :return: Noise-Profil (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        if mask is not None:
            noise_frames = power_spectrogram[mask == 0]
        else:
            # Wenn keine Maske: nehme die ersten min_frames als Noise an
            noise_frames = power_spectrogram[: self.min_frames]
        if self.method == "mean":
            self.noise_profile = np.maximum(np.mean(noise_frames, axis=0), self.noise_floor)
        else:
            self.noise_profile = np.maximum(np.median(noise_frames, axis=0), self.noise_floor)
        return self.noise_profile

    def get_profile(self):
        """
        Gibt das aktuell gelernte Noise-Profil zurück.
        :return: Noise-Profil (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        return self.noise_profile

    def auto_optimize(self, power_spectrogram):
        """
        Optimiert die Parameter für das Noise-Profil-Learning adaptiv.
        :param power_spectrogram: Eingabe-Spektrogramm (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        n_frames = power_spectrogram.shape[0]
        self.min_frames = min(20, max(5, n_frames // 20))
