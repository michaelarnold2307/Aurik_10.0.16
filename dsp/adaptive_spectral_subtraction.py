"""
adaptive_spectral_subtraction.py - SOTA-konformes Spectral Subtraction Modul für Aurik 6.0

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

logger = logging.getLogger("aurik.dsp.adaptive_spectral_subtraction")
logger.setLevel(logging.INFO)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_spectral_subtraction"
    category: str = "spectral_subtraction"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_spectral_subtraction_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"oversubtract": 1.0, "floor": 0.01},
        "safe_ranges": {
            "oversubtract": {"min": 0.5, "max": 2.0},
            "floor": {"min": 0.0, "max": 0.1},
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
            "risk": "Musikverlust",
            "expected_when": "oversubtract zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["snr_improvement"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveSpectralSubtraction:
    """
    SOTA-konformes Spectral Subtraction mit Quality-Gate, Audit-Logging, Fehlerbehandlung, DL-Inferenz-Platzhalter, Doku als Code.
    """

    def __init__(self, oversubtract=1.0, floor=0.01):
        """
        Initialisiert das Modul mit Quality-Gate für Parameter.
        :param oversubtract: Multiplikator für Noise Estimate (0.5-2.0)
        :param floor: Minimum-Faktor für Noise-Floor (0.0-0.1)
        """
        if not (0.5 <= oversubtract <= 2.0):
            logger.error("Ungültiger oversubtract: %s. Muss zwischen 0.5 und 2.0 liegen.", oversubtract)
            raise ValueError("oversubtract muss zwischen 0.5 und 2.0 liegen.")
        if not (0.0 <= floor <= 0.1):
            logger.error("Ungültiger floor: %s. Muss zwischen 0.0 und 0.1 liegen.", floor)
            raise ValueError("floor muss zwischen 0.0 und 0.1 liegen.")
        self.oversubtract = oversubtract
        self.floor = floor
        logger.info(
            f"AdaptiveSpectralSubtraction initialisiert mit oversubtract={self.oversubtract}, floor={self.floor}"
        )

    def log_contract(self):
        """
        Gibt den DSPContract für Auditierbarkeit aus (Log + Print).
        """
        contract_dict = asdict(adaptive_spectral_subtraction_contract)
        logger.info("[DSPContract] %s", contract_dict)

    def subtract(self, noisy_mag, noise_mag, use_deep_learning: bool = False, audit_log: bool = True, **kwargs):
        """
        Führt Spectral Subtraction durch. Quality-Gate, Audit-Logging, DL-Inferenz-Platzhalter, Fehlerbehandlung, SOTA-Transparenz.
        :param noisy_mag: Magnitudenspektrum mit Störung (np.ndarray)
        :param noise_mag: Noise Estimate (np.ndarray)
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :param kwargs: Optionale Parameter (oversubtract, floor)
        :return: Clean Magnitude (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(noisy_mag, np.ndarray):
            logger.error("noisy_mag ist kein np.ndarray")
            raise TypeError("noisy_mag ist kein np.ndarray")
        if not isinstance(noise_mag, np.ndarray):
            logger.error("noise_mag ist kein np.ndarray")
            raise TypeError("noise_mag ist kein np.ndarray")
        if noisy_mag.size == 0 or noise_mag.size == 0:
            logger.error("noisy_mag oder noise_mag ist leer")
            raise ValueError("noisy_mag oder noise_mag ist leer")
        if noisy_mag.shape != noise_mag.shape:
            logger.error("Shape mismatch: noisy_mag %s, noise_mag %s", noisy_mag.shape, noise_mag.shape)
            raise ValueError(f"Shape mismatch: noisy_mag {noisy_mag.shape}, noise_mag {noise_mag.shape}")
        if np.isnan(noisy_mag).any() or np.isnan(noise_mag).any():
            logger.error("noisy_mag oder noise_mag enthält NaN-Werte")
            raise ValueError("noisy_mag oder noise_mag enthält NaN-Werte")

        oversubtract = kwargs.get("oversubtract", self.oversubtract)
        floor = kwargs.get("floor", self.floor)

        output = None
        fallback_used = False
        try:
            if use_deep_learning:
                if not _TORCH_AVAILABLE:
                    logger.warning("PyTorch nicht verfügbar, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._subtract_classic(noisy_mag, noise_mag, oversubtract, floor)
                else:
                    logger.info("Deep-Learning-Inferenz aktiviert für Spectral Subtraction.")
                    # TorchScript-Modell (Platzhalter)
                    # model = torch.jit.load('spectral_subtraction.pt')
                    # output = model(torch.from_numpy(noisy_mag).float().unsqueeze(0), torch.from_numpy(noise_mag).float().unsqueeze(0)).squeeze(0).numpy()
                    logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._subtract_classic(noisy_mag, noise_mag, oversubtract, floor)
            else:
                output = self._subtract_classic(noisy_mag, noise_mag, oversubtract, floor)
        except Exception as e:
            logger.error("Fehler bei Spectral Subtraction: %s", e, exc_info=True)
            fallback_used = True
            output = noisy_mag.copy()

        if audit_log:
            snr_improvement = (
                float(np.mean(output) / (np.mean(noise_mag) + 1e-8)) if output is not None else float("nan")
            )
            logger.info(
                f"AdaptiveSpectralSubtraction: snr_improvement={snr_improvement:.4f}, Ersatzpfad_used={fallback_used}, oversubtract={oversubtract}, floor={floor}"
            )
            logger.info("[DSPContract] %s", asdict(adaptive_spectral_subtraction_contract))
        return output

    def _subtract_classic(self, noisy_mag, noise_mag, oversubtract, floor):
        """
        MMSE-inspired parametric Wiener gain (replaces raw subtraction).

        Instead of clip(mag - noise, floor):
            gain = max(1 - oversubtract * noise_pow / noisy_pow, floor)
        This reduces musical noise artifacts significantly.
        """
        noise_pow = noise_mag**2 + 1e-10
        noisy_pow = noisy_mag**2 + 1e-10
        gain = 1.0 - oversubtract * (noise_pow / noisy_pow)
        gain = np.maximum(gain, floor)
        gain = np.minimum(gain, 1.0)
        gain = np.nan_to_num(gain, nan=floor, posinf=floor, neginf=floor)
        clean_mag = gain * noisy_mag
        return clean_mag

    def auto_optimize(self, noisy_mag, noise_mag):
        """
        Passt oversubtract und floor adaptiv an (Dummy, normkonform gekennzeichnet).
        :param noisy_mag: Magnitudenspektrum mit Störung (np.ndarray)
        :param noise_mag: Noise Estimate (np.ndarray)
        """
        self.log_contract()
        snr = np.mean(noisy_mag) / (np.mean(noise_mag) + 1e-8)
        if snr < 2:
            self.oversubtract = 1.5
            self.floor = 0.05
        elif snr < 5:
            self.oversubtract = 1.2
            self.floor = 0.02
        else:
            self.oversubtract = 1.0
            self.floor = 0.01
        logger.info("Parameter auto-optimiert: oversubtract=%s, floor=%s", self.oversubtract, self.floor)
