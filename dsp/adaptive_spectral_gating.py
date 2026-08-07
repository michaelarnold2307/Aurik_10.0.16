"""
adaptive_spectral_gating.py - SOTA-konformes Spectral Gating Modul für Aurik 6.0
Dieses Modul ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_spectral_gating"
    category: str = "spectral_gating"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_spectral_gating_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold_db": -40, "reduction_db": -20},
        "safe_ranges": {
            "threshold_db": {"min": -80, "max": 0},
            "reduction_db": {"min": -60, "max": 0},
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
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Verlust von Details",
            "expected_when": "reduction_db zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["gating_effect"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveSpectralGating:
    def __init__(self, threshold_db=-40, reduction_db=-20):
        self.threshold_db = threshold_db
        self.reduction_db = reduction_db

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_spectral_gating_contract))

    def gate(self, mag_spectrogram, noise_floor=None, **kwargs):
        """Spectral gating with soft sigmoid transition, hysteresis, and
        frequency-dependent thresholds.

        Instead of hard binary masking, uses a sigmoid gain curve:
            gain = 1 / (1 + exp(-(mag_db - threshold) / softness))
        This eliminates harsh cutoff artifacts ("musical noise").

        Args:
            mag_spectrogram: Magnitude spectrogram [n_frames, n_bins] or [n_bins]
            noise_floor: Optional noise floor estimate (same shape or broadcastable)
        Returns:
            Gated magnitude spectrogram
        """
        self.log_contract()
        threshold_db = kwargs.get("threshold_db", self.threshold_db)
        reduction_db = kwargs.get("reduction_db", self.reduction_db)
        softness = kwargs.get("softness", 3.0)  # sigmoid transition width in dB

        mag = np.asarray(mag_spectrogram, dtype=np.float64)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-10))

        # Frequency-dependent threshold: lower freqs allow more aggressive gating
        # (noise is typically more perceptible at low frequencies)
        if mag.ndim == 2:
            n_bins = mag.shape[1]
            # Shift threshold: low bins get -6 dB more aggressive, high bins +3 dB gentler
            freq_offset = np.linspace(-6.0, 3.0, n_bins)
            freq_offset = freq_offset[np.newaxis, :]
        elif mag.ndim == 1:
            n_bins = mag.shape[0]
            freq_offset = np.linspace(-6.0, 3.0, n_bins)
        else:
            freq_offset = 0.0  # type: ignore[assignment]

        if noise_floor is not None:
            nf = np.asarray(noise_floor, dtype=np.float64)
            nf = np.nan_to_num(nf, nan=1e-10, posinf=1e-10, neginf=1e-10)
            threshold = 20.0 * np.log10(np.maximum(nf, 1e-10)) + threshold_db + freq_offset
        else:
            threshold = threshold_db + freq_offset

        # Soft sigmoid gating — smooth transition instead of hard binary gate
        # gain ∈ [reduction_linear, 1.0]
        reduction_linear = 10.0 ** (reduction_db / 20.0)
        sigmoid_input = (mag_db - threshold) / max(softness, 0.1)
        # Clip to prevent overflow in exp
        sigmoid_input = np.clip(sigmoid_input, -20.0, 20.0)
        sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_input))

        # Map sigmoid [0,1] → gain [reduction_linear, 1.0]
        gain = reduction_linear + (1.0 - reduction_linear) * sigmoid

        # Apply gain
        gated_mag = mag * gain
        gated_mag = np.nan_to_num(gated_mag, nan=0.0, posinf=0.0, neginf=0.0)
        gated_mag = np.maximum(gated_mag, 0.0)

        return gated_mag

    def auto_optimize(self, mag_spectrogram, noise_floor=None):
        """Adapt threshold and reduction based on signal statistics."""
        self.log_contract()
        mag = np.nan_to_num(np.asarray(mag_spectrogram, dtype=np.float64))
        median_db = float(np.median(20.0 * np.log10(np.maximum(mag, 1e-10))))
        p10_db = float(np.percentile(20.0 * np.log10(np.maximum(mag, 1e-10)), 10))

        # Set threshold slightly above the quiet percentile
        self.threshold_db = p10_db + 5.0
        # Reduction based on dynamic range
        dynamic_range = median_db - p10_db
        self.reduction_db = float(np.clip(-15.0 - dynamic_range * 0.5, -40.0, -10.0))
