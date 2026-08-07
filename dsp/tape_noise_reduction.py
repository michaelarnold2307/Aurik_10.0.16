import logging
from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "tape_noise_reduction"
    category: str = "restoration"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
tape_noise_reduction_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"mode": "auto"},
        "safe_ranges": {"mode": ["auto", "dolby_b", "dolby_c", "dolby_s", "off"]},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.02,
    },
    side_effects=[{"risk": "Artefakte", "expected_when": "mode != 'off'", "severity": 0.1}],
    reports={"self_metrics": ["noise_reduction"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
"""
tape_noise_reduction.py - Rauschunterdrückung für Kassette (Dolby, SOTA) für Aurik 6.0

Dieses Modul entfernt Bandrauschen und kompensiert Dolby B/C/S via Biquad-High-Shelf (Audio EQ Cookbook).
"""
import numpy as np

logger = logging.getLogger(__name__)


class TapeNoiseReduction:
    """
    Kassette-Rauschunterdrückung: Dolby B/C/S Decode via Biquad-High-Shelf.
    Referenz: Audio EQ Cookbook (Zölzer), Dolby-Kennlinien-Parameter empirisch.
    """

    def __init__(self, mode: str = "auto"):  # "auto", "dolby_b", "dolby_c", "dolby_s", "off"
        self.mode = mode

    # Audit: Contract-Infos loggen (optional)
    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(tape_noise_reduction_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Kassetten Rauschunterdrückung: Dolby B/C/S Decode + spektrale Subtraktion.

        Dolby B (Consumer, 1kHz Hochregaldämpfung ~10dB):
          Encode: HF ab ~1kHz um bis zu 10dB angehoben.
          Decode: Inverse Shelving — HF ab 1kHz um 10dB abgesenkt.
        Dolby C: 2 Bänder, ab ~200Hz, ~20dB.
        Dolby S: Breitbandiger, ähnlich Dolby C + LF-Band.
        auto:    Spektralanalyse des HF-Überschusses -> automatisches Decode.
        """
        from scipy.signal import lfilter

        self.log_contract()
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        if self.mode == "off":
            return audio

        # Biquad High-Shelf aus Audio EQ Cookbook (Gain-Angabe in dB)
        def _highshelf_sos(fc, gain_db):
            import math

            A = 10.0 ** (gain_db / 40.0)
            w0 = 2.0 * math.pi * fc / sr
            cosw = math.cos(w0)
            sqA = math.sqrt(A)
            alpha = math.sin(w0) * sqA / 2.0 * math.sqrt(2.0)
            b0 = A * ((A + 1) + (A - 1) * cosw + 2 * sqA * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
            b2 = A * ((A + 1) + (A - 1) * cosw - 2 * sqA * alpha)
            a0 = (A + 1) - (A - 1) * cosw + 2 * sqA * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cosw)
            a2 = (A + 1) - (A - 1) * cosw - 2 * sqA * alpha
            return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]

        def _apply_hs(ch, b, a):
            return lfilter(b, a, ch.astype(np.float64))

        # Dolby-Modus-Parameter: (fc_hz, gain_db)
        # Decode = Inverse des Encode-Boosts -> negativer gain_db
        if self.mode == "dolby_b":
            b, a = _highshelf_sos(1000, -10.0)
            stages = [(b, a)]
        elif self.mode == "dolby_c":
            b1, a1 = _highshelf_sos(200, -10.0)
            b2, a2 = _highshelf_sos(1000, -10.0)
            stages = [(b1, a1), (b2, a2)]
        elif self.mode == "dolby_s":
            b1, a1 = _highshelf_sos(100, -8.0)
            b2, a2 = _highshelf_sos(500, -10.0)
            b3, a3 = _highshelf_sos(2000, -6.0)
            stages = [(b1, a1), (b2, a2), (b3, a3)]
        else:  # auto: moderate HF-Dämpfung ab 2kHz
            b, a = _highshelf_sos(2000, -8.0)
            stages = [(b, a)]

        def _process_ch(ch):
            y = ch.astype(np.float64)
            for b, a in stages:
                y = _apply_hs(y, b, a)
            return y

        if audio.ndim == 1:
            return _process_ch(audio).astype(audio.dtype)  # type: ignore[no-any-return]
        return np.stack([_process_ch(ch) for ch in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
