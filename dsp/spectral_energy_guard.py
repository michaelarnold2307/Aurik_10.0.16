import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_energy_guard"
    category: str = "spectral_analysis"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


spectral_energy_guard_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"energy_min": 0.01, "energy_max": 1.0},
        "safe_ranges": {
            "energy_min": {"min": 0.0, "max": 0.5},
            "energy_max": {"min": 0.5, "max": 2.0},
        },
    },
    budgets={
        "artifact_budget": 0.0,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[],
    reports={"self_metrics": ["spectral_energy"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class SpectralEnergyGuard:
    """
    SOTA-konformer Spectral Energy Guard:
    - Überwacht die gesamte spektrale Energie als Qualitätsmaß
    """

    def __init__(self, energy_min: float = 0.01, energy_max: float = 1.0):
        self.energy_min = energy_min
        self.energy_max = energy_max

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_energy_guard_contract))

    def process(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        SOTA-Maximum: Berechnung der spektralen Gesamtenergie, Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        result = {"spectral_energy": None, "ok": False, "error": None}
        try:
            if not isinstance(audio, np.ndarray):
                raise TypeError("audio muss ein numpy.ndarray sein")
            if audio.size == 0:
                raise ValueError("audio ist leer")
            if not np.issubdtype(audio.dtype, np.floating):
                raise TypeError("audio muss float-Typ sein")
            if sr <= 0:
                raise ValueError("Sample-Rate muss > 0 sein")
            spec = np.abs(np.fft.rfft(audio))
            energy = float(np.sum(spec) / (len(spec) + 1e-8))
            ok = self.energy_min <= energy <= self.energy_max
            # Quality-Gate
            if energy < 0 or np.isnan(energy):
                logger.warning("[QualityGate] Warnung: Unplausible Energie, Rollback aktiviert.")
                result["spectral_energy"] = 0.0  # type: ignore[assignment]
                result["ok"] = False
                result["error"] = "Unplausible Energie detektiert"  # type: ignore[assignment]
                self._audit_log(result, sr)
                return result
            result["spectral_energy"] = energy  # type: ignore[assignment]
            result["ok"] = ok
            self._audit_log(result, sr)
            return result
        except Exception as e:
            result["error"] = str(e)  # type: ignore[assignment]
            logger.error("[SpectralEnergyGuard][Fehler] %s", e)
            self._audit_log(result, sr if "sr" in locals() else None)
            return result

    def _audit_log(self, result: dict[str, Any], sr: int | None = None):
        # Minimaler Audit-Log als Code, erweiterbar für zentrale Audit-Logik
        logger.info("[AuditLog][SpectralEnergyGuard] Ergebnis: %s | SR: %s", result, sr)
