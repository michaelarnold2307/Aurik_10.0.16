"""
adaptive_omlsa.py - SOTA-konformes OMLSA-Modul für Aurik 6.0
Dieses Modul implementiert adaptives OMLSA und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import expn

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_omlsa"
    category: str = "omlsa"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_omlsa_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"alpha": 0.98, "noise_floor": 1e-8},
        "safe_ranges": {
            "alpha": {"min": 0.8, "max": 1.0},
            "noise_floor": {"min": 1e-12, "max": 1e-3},
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
            "risk": "Fehlanpassung bei falschem Noise-Floor",
            "expected_when": "noise_floor zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["omlsa_gain"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveOMLSA:
    def __init__(self, alpha=0.98, noise_floor=1e-8):
        self.alpha = alpha
        self.noise_floor = noise_floor
        self._contract_logged = False

    def log_contract(self):
        if not self._contract_logged:
            logger.debug("[DSPContract] %s", asdict(adaptive_omlsa_contract))
            self._contract_logged = True

    def omlsa(self, noisy_mag, noise_mag, *, sr: int = 48000, **kwargs):
        """OMLSA gain with optional psychoacoustic frequency weighting.

        Args:
            noisy_mag: Noisy magnitude spectrum
            noise_mag: Noise magnitude estimate
            sr: Sample rate for psychoacoustic weighting
        """
        self.log_contract()
        alpha = kwargs.get("alpha", self.alpha)
        noise_floor = kwargs.get("noise_floor", self.noise_floor)
        psychoacoustic = kwargs.get("psychoacoustic", True)

        noisy_mag = np.nan_to_num(np.asarray(noisy_mag, dtype=np.float64))
        noise_mag = np.nan_to_num(np.asarray(noise_mag, dtype=np.float64))

        # A-priori SNR — guard against negative noise_mag
        noise_pow = noise_mag**2 + noise_floor
        noisy_pow = noisy_mag**2
        gamma = noisy_pow / noise_pow
        # Decision-directed a-priori SNR (Ephraim-Malah 1984)
        xi = alpha * noisy_pow / noise_pow + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 1e-10)  # xi must be strictly positive for stable v
        # OMLSA Gain — v := xi*gamma/(1+xi); must be strictly positive for expn(1,.)
        v = xi * gamma / (1.0 + xi)
        v = np.maximum(v, 1e-8)  # §3.1 NaN-Guard: expn(1,0)→∞, 0·∞=NaN at silence
        gain = (xi / (1.0 + xi)) * np.exp(0.5 * expn(1, v))

        # Psychoacoustic frequency weighting: gentler in sensitive bands (2-5 kHz)
        if psychoacoustic and noisy_mag.ndim >= 1:
            n_bins = noisy_mag.shape[-1] if noisy_mag.ndim >= 2 else noisy_mag.shape[0]
            freqs = np.linspace(0, sr / 2, n_bins)
            sensitivity = np.exp(-0.5 * ((np.log2(np.maximum(freqs, 20.0) / 3500.0)) ** 2) / 1.5**2)
            psy_floor = 0.01 + 0.04 * sensitivity  # higher floor in sensitive bands

            # §9.10.118 Silence-Adaptive G_floor: In quiet regions (silence
            # between phrases), the perceptual floor can be reduced because
            # there is no signal to mask musical noise artefacts — twinkling
            # remnants in silence are audibly conspicuous.  Scientific basis:
            # Fastl & Zwicker 2007 §8.3 — masking threshold vanishes in
            # absence of primary stimulus.
            frame_energy_db = -60.0  # default: non-silence
            _silence_threshold_db = kwargs.get("silence_threshold_db", -55.0)
            if noisy_mag.ndim == 2:
                # 2-D spectrogram: per-frame energy along freq axis
                _frame_pow = np.mean(noisy_mag**2, axis=-1, keepdims=True)
                _frame_db = 10.0 * np.log10(_frame_pow + 1e-12)
                # Scale floor per frame: silent frames get ×0.5 (more suppression)
                _silence_mask = (_frame_db < _silence_threshold_db).astype(np.float32)
                _floor_scale = 1.0 - 0.5 * _silence_mask  # 1.0 normal, 0.5 silence
                psy_floor = psy_floor[np.newaxis, :] * _floor_scale
            else:
                # 1-D spectrum: single-frame energy check
                frame_energy_db = float(10.0 * np.log10(np.mean(noisy_mag**2) + 1e-12))
                if frame_energy_db < _silence_threshold_db:
                    psy_floor = psy_floor * 0.5

            gain = np.maximum(gain, psy_floor)

        gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)  # §3.1 NaN/Inf-Guard
        gain = np.clip(gain, 0.0, 1.0)  # gain must not exceed 1 (no amplification)
        clean_mag = gain * noisy_mag
        return clean_mag

    def auto_optimize(self, noisy_mag, noise_mag):
        """
        Passt alpha und noise_floor adaptiv an den geschätzten SNR an.
        Hohes SNR → alpha nahe 1 (starkes Glätten), niedriger Rauschboden.
        Niedriges SNR → alpha kleiner (schnellere Adaptation), höherer Rauschboden.
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        # SNR-Schätzung: Mittlere Signal- zu Rauschmagnituden-Ratio
        snr = float(np.mean(noisy_mag) / (np.mean(noise_mag) + 1e-8))

        # alpha: 0.85 (niedriger SNR) … 0.99 (hoher SNR)
        self.alpha = float(np.clip(0.85 + 0.14 * np.tanh((snr - 5.0) / 5.0), 0.85, 0.99))

        # noise_floor: schärfer bei hohem SNR, lockerer bei niedrigem
        self.noise_floor = float(np.clip(1e-6 / (snr + 1.0), 1e-8, 1e-5))

        logger.info(
            f"adaptive_omlsa.auto_optimieren: SNR={snr:.2f} → alpha={self.alpha:.4f}, noise_floor={self.noise_floor:.2e}"
        )
