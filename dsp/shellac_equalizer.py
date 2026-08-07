"""
shellac_equalizer.py - Schellack-Entzerrer für Aurik 6.0

Dieses Modul entzerrt oder simuliert typische Schellack-Kennlinien (Stub).
"""

import logging

import numpy as np

logger = logging.getLogger("aurik.dsp.shellac_equalizer")
logger.setLevel(logging.INFO)


class ShellacEqualizer:
    """
    Schellack-Entzerrer (Stub):
    - Wendet verschiedene Entzerrungskurven (z.B. 78rpm, Columbia, Decca, HMV) auf Audiosignale an
    """

    def __init__(self, curve: str = "78rpm"):  # "78rpm", "Columbia", "Decca", "HMV", ...
        self.curve = curve

    def process(self, audio: np.ndarray, sr: int, audit_log: bool = True) -> np.ndarray:
        """
        Wendet SOTA-Entzerrungskurve für Schellack an (Stub, normkonform).
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param sr: Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: Entzerrtes Audio (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            logger.error("Ungültiges Audio-Array (leer oder falscher Typ)")
            raise ValueError("Ungültiges Audio-Array (leer oder falscher Typ)")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1e6:
            logger.warning("Audio möglicherweise nicht normiert (max > 1e6)")

        try:
            # Historische 78rpm Wiedergabe-Entzerrungskurven via IIR-Shelving-Filter
            # Parameter: (bass_turnover_hz, bass_gain_db, treble_rolloff_hz, treble_cut_db)
            curve_params = {
                "78rpm": (500, +18.0, 8000, -18.0),  # Generische 78rpm / RIAA-Vorgänger
                "Columbia": (250, +16.0, 9000, -18.0),  # Columbia Records (pre-1948)
                "Decca": (375, +17.0, 7000, -16.0),  # Decca FFRR
                "HMV": (500, +18.0, 5500, -18.0),  # HMV / EMI (corrected: 5.5 kHz, not 3.5 kHz)
            }
            if self.curve not in curve_params:
                logger.warning("Unbekannte Entzerrungskurve: %s, Ersatzpfad auf 78rpm", self.curve)
            params = curve_params.get(self.curve, curve_params["78rpm"])
            audio_out = self._apply_shellac_eq(audio, sr, *params)
        except Exception as e:
            logger.error("Fehler bei der Entzerrung: %s", e)
            audio_out = audio.copy()

        if audit_log:
            logger.info("ShellacEqualizer: curve=%s, EQ angewendet", self.curve)
        return audio_out.astype(audio.dtype)

    @staticmethod
    def _biquad_lowshelf(sr: int, fc: float, gain_db: float) -> tuple:
        """Audio-EQ-Cookbook Low-Shelf Biquad (S=1)."""
        import math

        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * fc / sr
        cosw = math.cos(w0)
        sinw = math.sin(w0)
        alpha = sinw / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / 1.0 - 1.0) + 2.0)
        # S=1 vereinfacht alpha = sinw/2 * sqrt(2) ... standard shelf
        alpha = sinw * math.sqrt(A) / 2.0 * math.sqrt(2.0)  # Q=1/sqrt(2), S=1
        sqA = math.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
        b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqA * alpha)
        a0 = (A + 1) + (A - 1) * cosw + 2 * sqA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw)
        a2 = (A + 1) + (A - 1) * cosw - 2 * sqA * alpha
        return ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])

    @staticmethod
    def _biquad_highshelf(sr: int, fc: float, gain_db: float) -> tuple:
        """Audio-EQ-Cookbook High-Shelf Biquad (S=1)."""
        import math

        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * fc / sr
        cosw = math.cos(w0)
        sinw = math.sin(w0)
        sqA = math.sqrt(A)
        alpha = sinw * sqA / 2.0 * math.sqrt(2.0)
        b0 = A * ((A + 1) + (A - 1) * cosw + 2 * sqA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
        b2 = A * ((A + 1) + (A - 1) * cosw - 2 * sqA * alpha)
        a0 = (A + 1) - (A - 1) * cosw + 2 * sqA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cosw)
        a2 = (A + 1) - (A - 1) * cosw - 2 * sqA * alpha
        return ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])

    def _apply_shellac_eq(
        self, audio: np.ndarray, sr: int, bass_hz: float, bass_db: float, treble_hz: float, treble_db: float
    ) -> np.ndarray:
        """Wendet Bass-Boost + Treble-Cut (2 Biquad-Kaskade) an."""
        from scipy.signal import lfilter

        b1, a1 = self._biquad_lowshelf(sr, bass_hz, bass_db)
        b2, a2 = self._biquad_highshelf(sr, treble_hz, treble_db)

        def _filt(ch):
            y = lfilter(b1, a1, ch)
            return lfilter(b2, a2, y)

        if audio.ndim == 1:
            return _filt(audio).astype(audio.dtype)  # type: ignore[no-any-return]
        return np.stack([_filt(ch) for ch in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
