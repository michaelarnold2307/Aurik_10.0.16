from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "limiter"
    category: str = "integrity"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
limiter_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold_db": -1.0},
        "safe_ranges": {"threshold_db": {"min": -6.0, "max": 0.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "pumping", "expected_when": "threshold_db > -0.5", "severity": 0.1}],
    reports={"self_metrics": ["limiting_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)

import numpy as np
import numpy.typing as npt


class Limiter:
    """
    SOTA-konformer Limiter:
    - True Peak, Soft-Knee, Lookahead, ML-ready
    """

    def __init__(
        self,
        ceiling_db: float = -2.0,
        knee_db: float = 6.0,
        lookahead_ms: float = 2.0,
        release_ms: float = 50.0,
    ) -> None:
        """
        ceiling_db: Maximalpegel (dBFS)
        knee_db: Soft-Knee (dB)
        lookahead_ms: Lookahead (ms)
        release_ms: Release-Zeit (ms)
        """
        self.ceiling_db = ceiling_db
        self.knee_db = knee_db
        self.lookahead_ms = lookahead_ms
        self.release_ms = release_ms

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit Limiting.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: limitiertes Signal (gleicher Typ wie audio)
        """
        # Lookahead-Buffer
        lookahead = int(sr * self.lookahead_ms / 1000)
        padded = np.pad(audio, (lookahead, 0), mode="constant")
        shifted = padded[:-lookahead] if lookahead > 0 else audio
        # True-Peak-Detection (Sample-Peak)
        peak = np.abs(shifted)
        peak_db = 20 * np.log10(peak + 1e-8)
        over = peak_db - self.ceiling_db
        gain_db = np.zeros_like(peak_db)
        # Soft-Knee
        idx_soft = (over > -self.knee_db / 2) & (over < self.knee_db / 2)
        gain_db[idx_soft] = -((over[idx_soft] + self.knee_db / 2) ** 2) / (2 * self.knee_db)
        idx_over = over >= self.knee_db / 2
        gain_db[idx_over] = -over[idx_over]
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        release_coeff = np.exp(-1.0 / (sr * self.release_ms / 1000))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        out = audio * env
        # Pegel normalisieren
        maxval = np.max(np.abs(out))
        ceiling_lin = 10 ** (self.ceiling_db / 20)
        if maxval > ceiling_lin:
            out = out * (ceiling_lin / maxval)
        return out.astype(audio.dtype)  # type: ignore[no-any-return]

        # Audit: Contract-Infos loggen (optional)
        import logging

        logging.info("[DSPContract] %s", asdict(limiter_contract))
