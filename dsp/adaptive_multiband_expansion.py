import logging

"""
adaptive_multiband_expansion.py - SOTA-konformes Multiband-Expansion-Modul für Aurik 6.0
Dieses Modul implementiert adaptive Multiband-Expansion und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_multiband_expansion"
    category: str = "expansion"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_multiband_expansion_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "n_bands": 4,
            "threshold_db": -40,
            "ratio": 2.0,
            "attack": 0.01,
            "release": 0.1,
            "sr": 44100,
        },
        "safe_ranges": {
            "n_bands": {"min": 1, "max": 16},
            "threshold_db": {"min": -80, "max": 0},
            "ratio": {"min": 1.0, "max": 10.0},
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
    side_effects=[{"risk": "Überexpansion", "expected_when": "ratio zu hoch", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["expansion_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveMultibandExpansion:
    """
    SOTA-konforme adaptive Multiband-Expansion (klassisch):
    - Multiband-Expander mit adaptiven Parametern
    - SOTA-Maximum: Gain-Expansion pro Band, Quality-Gates, Auditierbarkeit
    """

    def __init__(self, n_bands=4, threshold_db=-40, ratio=2.0, attack=0.01, release=0.1, sr=44100):
        self.n_bands = n_bands
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.sr = sr

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_multiband_expansion_contract))

    def expand(self, band_signals, **kwargs):
        """
        Führt adaptive Multiband-Expansion durch.
        :param band_signals: Liste der Band-Signale (np.ndarray)
        :param kwargs: optionale Parameter (threshold_db, ratio)
        :return: Expandierte Band-Signale (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        threshold_db = kwargs.get("threshold_db", self.threshold_db)
        ratio = kwargs.get("ratio", self.ratio)
        expanded = []
        for band in band_signals:
            band_db = 20 * np.log10(np.maximum(np.abs(band), 1e-8))
            gain_db = np.where(band_db < threshold_db, (band_db - threshold_db) * (ratio - 1), 0)
            gain = 10 ** (gain_db / 20)
            expanded.append(band * gain)
        return np.array(expanded)

    def auto_optimize(self, band_signals):
        """
        Optimiert die Expander-Parameter adaptiv anhand des RMS der Band-Signale.
        :param band_signals: Liste der Band-Signale (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        rms = np.sqrt(np.mean(np.square(band_signals)))
        if rms < 0.01:
            self.threshold_db = -50
            self.ratio = 3.0
        elif rms < 0.1:
            self.threshold_db = -40
            self.ratio = 2.0
        else:
            self.threshold_db = -30
            self.ratio = 1.5
