"""
ai_transient_enhancer.py - SOTA-Transienten-Enhancer für Aurik 6.0

Dieses Modul verstärkt Transienten in Audiosignalen.
Kombiniert klassische Transientenverstärkung (Envelope/Peaks) und Deep-Learning (ML-ready).
Jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "ai_transient_enhancer"
    category: str = "transient_enhancement"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
ai_transient_enhancer_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"amount": 1.5},
        "safe_ranges": {"amount": {"min": 1.0, "max": 3.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Überbetonung", "expected_when": "amount zu hoch", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["transient_enhancement_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AiTransientEnhancer:
    """
    SOTA-Transienten-Enhancer: Kombiniert klassische Transientenverstärkung (Envelope/Peaks) und Deep-Learning (ML-ready). Robust, adaptiv, Fallback-fähig.
    """

    def __init__(self, model_path: str | None = None, amount: float = 1.5):
        self.model_path = model_path
        self.model = None
        self.amount = amount

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(ai_transient_enhancer_contract))

    def enhance_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Verstärkt Transienten im Audiosignal. Zuerst klassische Envelope/Peak-Detection, dann optional Deep-Learning-Inferenz (wenn Modell geladen).
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        audio_f64 = np.asarray(audio, dtype=np.float64)
        analytic = np.asarray(hilbert(audio_f64), dtype=np.complex128)
        envelope = np.sqrt(np.square(analytic.real) + np.square(analytic.imag))

        # Transient detection: adaptive threshold using percentile
        env_median = np.median(envelope)
        env_std = np.std(envelope)
        threshold = env_median + 2.0 * env_std

        # Gradual gain curve instead of binary peak/no-peak
        # Soft gain: scales linearly from 1.0 at threshold to self.amount at 2x threshold
        gain_curve = np.ones_like(envelope)
        above = envelope > threshold
        if np.any(above):
            overshoot = (envelope[above] - threshold) / (env_std + 1e-10)
            # Gradual gain: sigmoid-like scaling capped at self.amount
            transient_gain = 1.0 + (self.amount - 1.0) * np.clip(overshoot / 2.0, 0.0, 1.0)
            gain_curve[above] = transient_gain

        # Smooth the gain curve to avoid clicks (1ms smoothing)
        smooth_len = max(1, int(sr * 0.001))
        if smooth_len > 1:
            kernel = np.ones(smooth_len) / smooth_len
            gain_curve = np.convolve(gain_curve, kernel, mode="same")

        audio_out = audio_f64 * gain_curve
        # ML-Inferenz via ONNX (wenn Modell geladen)
        if self.model is not None:
            try:
                _in_name = self.model.get_inputs()[0].name
                _inp = audio_out[np.newaxis, :].astype(np.float32)  # (1, N)
                _raw = self.model.run(None, {_in_name: _inp})[0].squeeze()
                _raw = np.nan_to_num(_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                if _raw.shape == audio_out.shape:
                    audio_out = _raw
            except Exception as _onnx_err:
                logger.warning(
                    "AiTransientEnhancer: ONNX-Inferenz fehlgeschlagen (%s) — Envelope-DSP-Ersatzpfad aktiv.",
                    _onnx_err,
                )
        return np.clip(np.nan_to_num(audio_out, nan=0.0), -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]
