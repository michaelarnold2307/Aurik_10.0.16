import logging

"""
adaptive_pyint_pitch_tracking.py - SOTA-konformes Pitch-Tracking-Modul für Aurik 6.0
Dieses Modul implementiert adaptives pYIN Pitch Tracking und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_pyint_pitch_tracking"
    category: str = "pitch_tracking"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_pyint_pitch_tracking_contract = DSPContract(
    io={
        "channels": "mono",
        "sample_rates": [16000, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"sr": 16000},
        "safe_ranges": {"sr": [8000, 96000]},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Pitch-Fehlschätzung", "expected_when": "Dummy", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["pitch_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptivePYINPitchTracking:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_pyint_pitch_tracking_contract))

    def track(self, x: np.ndarray) -> float:
        """Pitch-Tracking via vereinfachtem YIN-Algorithmus.

        YIN (de Cheveigne & Kawahara 2002):
          1. Differenzfunktion d(tau) = sum[(x_t - x_{t+tau})^2]
          2. Kumulierte normalisierte Differenzfunktion (CMND)
          3. Erstes lokales Minimum unter Schwellwert 0.1 -> Grundfrequenz
        """
        self.log_contract()
        if len(x) < 512:
            return 0.0
        # Verwende max 1024 Samples ab Anfang
        frame = x[:1024].astype(np.float64)
        n = len(frame)
        max_tau = n // 2
        min_tau = max(2, int(self.sr / 1000))  # max 1000 Hz
        max_period = min(max_tau, int(self.sr / 60))  # min 60 Hz

        # 1. Differenzfunktion
        d = np.zeros(max_tau)
        for tau in range(1, max_tau):
            diff = frame[: n - tau] - frame[tau:n]
            d[tau] = float(np.sum(diff**2))

        # 2. CMND
        cmnd = np.zeros(max_tau)
        cmnd[0] = 1.0
        running_sum = 0.0
        for tau in range(1, max_tau):
            running_sum += d[tau]
            cmnd[tau] = d[tau] * tau / (running_sum + 1e-12)

        # 3. Erstes Minimum unter Schwellwert
        threshold = 0.1
        best_tau = 0
        for tau in range(min_tau, max_period):
            if cmnd[tau] < threshold:
                # Suche lokales Minimum
                while tau + 1 < max_period and cmnd[tau + 1] < cmnd[tau]:
                    tau += 1
                best_tau = tau
                break

        if best_tau < 2:
            # Fallback: Global-Minimum
            best_tau = int(np.argmin(cmnd[min_tau:max_period])) + min_tau
        if best_tau < 2:
            return 0.0
        pitch = float(self.sr) / float(best_tau)
        return round(float(np.clip(pitch, 60.0, 1000.0)), 2)

    def auto_optimize(self, x: np.ndarray) -> None:
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        self.sr = 16000
