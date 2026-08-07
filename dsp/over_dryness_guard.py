import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "over_dryness_guard"
    category: str = "reverb"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


over_dryness_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"min_rt60": 0.15},
        "safe_ranges": {"min_rt60": {"min": 0.05, "max": 0.5}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.0,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[],
    reports={"self_metrics": ["dryness_score"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class OverDrynessGuard:
    """
    SOTA-konformer Over-dryness Guard:
    - Überwacht, ob das Signal zu trocken (ohne Nachhall) ist und gibt Warnung/Empfehlung
    """

    def __init__(self, min_rt60: float = 0.15):
        self.min_rt60 = min_rt60

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(over_dryness_guard_contract))

    def process(self, audio: np.ndarray, sr: int, rt60: float) -> dict[str, Any]:
        """
        SOTA-Maximum: Prüft, ob das Signal zu trocken ist (RT60 < min_rt60), Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        result = {"over_dry": False, "rt60": rt60, "min_rt60": self.min_rt60, "error": None}
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für OverDrynessGuard")
            if rt60 < self.min_rt60:
                logger.warning("[QualityGate] Warnung: Signal ist zu trocken (Over-dryness), Nachhall empfohlen.")
                result["over_dry"] = True
            self._audit_log(result, sr)
            return result
        except Exception as e:
            result["error"] = str(e)  # type: ignore[assignment]
            logger.error("[OverDrynessGuard][Fehler] %s", e)
            self._audit_log(result, sr if "sr" in locals() else None)
            return result

    def _audit_log(self, result: dict[str, Any], sr: int | None = None):
        logger.info("[AuditLog][OverDrynessGuard] Ergebnis: %s | SR: %s", result, sr)
