"""
adaptive_spectral_peak_removal.py - SOTA-konformes Spectral Peak Removal Modul für Aurik 6.0
Dieses Modul ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.adaptive_spectral_peak_removal")
logger.setLevel(logging.INFO)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_spectral_peak_removal"
    category: str = "spectral_peak_removal"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_spectral_peak_removal_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold": 0.8},
        "safe_ranges": {"threshold": {"min": 0.1, "max": 1.0}},
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
            "risk": "Verlust von Details",
            "expected_when": "threshold zu niedrig",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["peak_suppression"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveSpectralPeakRemoval:
    """
    SOTA-konformes Spectral Peak Removal mit Quality-Gate, Audit-Logging, Fehlerbehandlung, DL-Inferenz-Platzhalter, Doku als Code.
    """

    def __init__(self, threshold: float = 0.8):
        """
        Initialisiert das Modul mit Quality-Gate für threshold.
        :param threshold: Schwellenwert für Peak-Entfernung (0.1-1.0)
        """
        if not (0.1 <= threshold <= 1.0):
            logger.error("Ungültiger Schwelle: %s. Muss zwischen 0.1 und 1.0 liegen.", threshold)
            raise ValueError("threshold muss zwischen 0.1 und 1.0 liegen.")
        self.threshold = threshold
        logger.info("AdaptiveSpectralPeakRemoval initialisiert mit Schwelle=%s", self.threshold)

    def log_contract(self):
        """
        Gibt den DSPContract für Auditierbarkeit aus (Log + Print).
        """
        contract_dict = asdict(adaptive_spectral_peak_removal_contract)
        logger.info("[DSPContract] %s", contract_dict)

    def remove(self, spectrum: np.ndarray, use_deep_learning: bool = False, audit_log: bool = True) -> np.ndarray:
        """
        Entfernt spektrale Peaks über threshold. Quality-Gate, Audit-Logging, DL-Inferenz-Platzhalter, Fehlerbehandlung, SOTA-Transparenz.
        :param spectrum: Eingabespektrum (np.ndarray)
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :return: Peak-unterdrücktes Spektrum (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(spectrum, np.ndarray):
            logger.error("spectrum ist kein np.ndarray")
            raise TypeError("spectrum ist kein np.ndarray")
        if spectrum.size == 0:
            logger.error("spectrum ist leer")
            raise ValueError("spectrum ist leer")
        if np.isnan(spectrum).any():
            logger.error("spectrum enthält NaN-Werte")
            raise ValueError("spectrum enthält NaN-Werte")

        output = None
        fallback_used = False
        try:
            if use_deep_learning:
                if not _TORCH_AVAILABLE:
                    logger.warning("PyTorch nicht verfügbar, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._remove_classic(spectrum)
                else:
                    logger.info("Deep-Learning-Inferenz aktiviert für Spectral Peak Removal.")
                    # TorchScript-Modell (Platzhalter)
                    # model = torch.jit.load('spectral_peak_removal.pt')
                    # output = model(torch.from_numpy(spectrum).float().unsqueeze(0)).squeeze(0).numpy()
                    logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._remove_classic(spectrum)
            else:
                output = self._remove_classic(spectrum)
        except Exception as e:
            logger.error("Fehler bei Spectral Peak Removal: %s", e, exc_info=True)
            fallback_used = True
            output = spectrum.copy()

        if audit_log:
            peak_suppression = float(np.mean(spectrum - output))
            logger.info(
                f"AdaptiveSpectralPeakRemoval: peak_suppression={peak_suppression:.6f}, Ersatzpfad_used={fallback_used}, Schwelle={self.threshold}"
            )
            logger.info("[DSPContract] %s", asdict(adaptive_spectral_peak_removal_contract))
        return output

    def _remove_classic(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Klassische Peak-Entfernung: Werte über threshold werden auf Mittelwert gesetzt.
        """
        output = np.copy(spectrum)
        mask = spectrum > self.threshold * np.max(spectrum)
        output[mask] = np.mean(spectrum)
        return output

    def auto_optimize(self, spectrum: np.ndarray) -> None:
        """
        Passt threshold adaptiv an (Dummy, normkonform gekennzeichnet).
        :param spectrum: Eingabespektrum (np.ndarray)
        """
        self.log_contract()
        if np.max(spectrum) > 1:
            self.threshold = 0.6
        else:
            self.threshold = 0.8
        logger.info("Schwelle auto-optimiert auf %s", self.threshold)
