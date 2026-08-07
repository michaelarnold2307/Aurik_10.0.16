import logging

"""
reel_to_reel_equalizer.py - Tonband-Entzerrer für Aurik 6.0

Dieses Modul entzerrt oder simuliert typische Tonband-Kennlinien (Stub).
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DSPContract:
    name: str = "ReelToReelEqualizer"
    version: str = "1.0"
    description: str = "Bandmaschinenentzerrung nach Standard (NAB, IEC, CCIR, ...)"
    parameters: dict[str, Any] | None = None


reel_to_reel_equalizer_contract = DSPContract(parameters={"standard": "NAB"})


class ReelToReelEqualizer:
    """
    Tonband-Entzerrer (Stub):
    - Wendet verschiedene Entzerrungskurven (z.B. NAB, IEC, CCIR) auf Audiosignale an
    """

    def __init__(self, standard: str = "NAB"):  # "NAB", "IEC", "CCIR", ...
        self.standard = standard

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(reel_to_reel_equalizer_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """NAB/IEC/CCIR Tonband-Wiedergabe-Entzerrung.

        NAB (7.5 ips): \u03c41=3180\u00b5s (50Hz bass), \u03c42=50\u00b5s (3183Hz treble)
        IEC (15 ips): \u03c41=3180\u00b5s, \u03c42=35\u00b5s  (4547Hz)
        CCIR (7.5 ips): \u03c41=3180\u00b5s, \u03c42=70\u00b5s (2274Hz)
        Alle via bilinearer Transformation: 2 Shelving-Filter in Kaskade.
        """
        from scipy.signal import lfilter

        self.log_contract()
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        _tau = {
            "NAB": (3180e-6, 50e-6),
            "IEC": (3180e-6, 35e-6),
            "CCIR": (3180e-6, 70e-6),
        }
        tau_b, tau_t = _tau.get(self.standard.upper(), _tau["NAB"])

        def _1st_lp(tau):
            k = 2.0 * sr * tau
            b = np.array([1.0 / (k + 1.0), 1.0 / (k + 1.0)])
            a = np.array([1.0, (1.0 - k) / (k + 1.0)])
            return b, a

        def _1st_hp(tau):
            k = 2.0 * sr * tau
            b = np.array([k / (k + 1.0), -k / (k + 1.0)])
            a = np.array([1.0, (1.0 - k) / (k + 1.0)])
            return b, a

        b_b, a_b = _1st_hp(tau_b)  # Bass anheben (Hochpass-Charakter)
        b_t, a_t = _1st_lp(tau_t)  # Höhen dämpfen (Tiefpass-Charakter)

        def _apply(ch):
            y = lfilter(b_b, a_b, ch.astype(np.float64))
            return lfilter(b_t, a_t, y)

        if audio.ndim == 1:
            return _apply(audio).astype(audio.dtype)  # type: ignore[no-any-return]
        return np.stack([_apply(ch) for ch in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
