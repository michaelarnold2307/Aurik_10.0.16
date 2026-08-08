"""
---
modul_name: StemProcessingDecision
aufgabe: Adaptive Entscheidungslogik für Stem-Processing
ein_ausgabe_typen:
        input: np.ndarray (Stem-Audio), dict (Policy)
        output: dict (Features, Action)
staerken: Explainable AI, Policy-Integration, Logging, adaptiv
schwaechen: Policy-Logik muss gepflegt werden
abhaengigkeiten: [numpy]
---
"""

from dataclasses import asdict, dataclass

import numpy as np
import logging
logger = logging.getLogger(__name__)


@dataclass
class StemFeatures:
    """Typed audio features extracted from a stem signal."""

    rms: float
    spectral_centroid: float
    transient: float

    # Backward-compatible dict-style access
    def get(self, key: str, default=None):
        return asdict(self).get(key, default)

    def __getitem__(self, key: str):
        return asdict(self)[key]

    def __contains__(self, key: str) -> bool:
        return key in asdict(self)


@dataclass
class StemDecisionResult:
    """Typed result of StemProcessingDecision.decide()."""

    action: str
    features: StemFeatures

    # Backward-compatible dict-style access
    def get(self, key: str, default=None):
        d = {"action": self.action, "features": self.features}
        return d.get(key, default)

    def __getitem__(self, key: str):
        d = {"action": self.action, "features": self.features}
        return d[key]

    def __contains__(self, key: str) -> bool:
        return key in {"action", "features"}

    def to_dict(self) -> dict:
        return asdict(self)


class StemProcessingDecision:
    def __init__(self, policy: dict | None = None, logger=None):
        self.policy = policy or {}
        self.logger = logger

    def decide(self, stem: np.ndarray, sr: int, features: dict | StemFeatures | None = None) -> StemDecisionResult:
        feats = (StemFeatures(**features) if isinstance(features, dict) else features) or self._analyze_features(
            stem, sr
        )
        action = self._policy_decision(feats)
        result = StemDecisionResult(action=action, features=feats)
        if self.logger:
            self.logger.log_decision(result)
        return result

    def _analyze_features(self, stem: np.ndarray, sr: int) -> StemFeatures:
        rms = float(np.sqrt(np.mean(stem**2)))
        spec_centroid = float(
            np.sum(np.abs(np.fft.rfft(stem)) * np.fft.rfftfreq(len(stem), 1 / sr))
            / (np.sum(np.abs(np.fft.rfft(stem))) + 1e-8)
        )
        transient = float(np.max(np.abs(np.diff(stem))))
        return StemFeatures(rms=rms, spectral_centroid=spec_centroid, transient=transient)

    def _policy_decision(self, feats: StemFeatures) -> str:
        if self.policy.get("goal") == "vocal_enhance" and feats.spectral_centroid > 2000:
            return "apply_vocal_enhancer"
        if self.policy.get("goal") == "drum_separation" and feats.transient > 0.2:
            return "apply_drum_separator"
        return "bypass"
