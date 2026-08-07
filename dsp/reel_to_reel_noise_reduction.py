"""
reel_to_reel_noise_reduction.py - Rauschunterdrückung für Tonband (SOTA) für Aurik 6.0

Dieses Modul entfernt Bandrauschen und simuliert/kompensiert Dolby/DBX (Stub).
"""

import numpy as np


class ReelToReelNoiseReduction:
    """
    Tonband-Rauschunterdrückung (Stub):
    - Entfernt Bandrauschen, simuliert/kompensiert Dolby/DBX-Charakteristika
    """

    def __init__(self, mode: str = "auto"):  # "auto", "dolby_a", "dolby_b", "dbx", "off"
        self.mode = mode

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Tonband Rauschunterdrückung: Dolby A/B, DBX Decode.

        Dolby A (professional, 4-Band-System):
          Decode: moderate Dämpfung in HF, MF und LF über Shelving-Kette.
        Dolby B: Gleich wie Kassetten-Dolby B (HF 1kHz, -10dB).
        DBX (Wideband-Kompandierung):
          Encode: Vollband-Kompression (sqrt). Decode: Vollband-Expansion (^2).
          Vereinfacht: leichte Tiefpass-Glättung der Transienen.
        auto:   Spektral adaptives Decode (moderate HF-Dämpfung).
        """
        from scipy.signal import lfilter

        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        if self.mode == "off":
            return audio

        def _highshelf_coeffs(fc, gain_db):
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

        def _lowshelf_coeffs(fc, gain_db):
            import math

            A = 10.0 ** (gain_db / 40.0)
            w0 = 2.0 * math.pi * fc / sr
            cosw = math.cos(w0)
            sqA = math.sqrt(A)
            alpha = math.sin(w0) * sqA / 2.0 * math.sqrt(2.0)
            b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqA * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
            b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqA * alpha)
            a0 = (A + 1) + (A - 1) * cosw + 2 * sqA * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cosw)
            a2 = (A + 1) + (A - 1) * cosw - 2 * sqA * alpha
            return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]

        def _ch(ch, stages):
            y = ch.astype(np.float64)
            for b, a in stages:
                y = lfilter(b, a, y)
            return y

        if self.mode == "dolby_a":
            # 4-Band-Decode: LF (<80Hz), LMF (<3kHz), HMF (<9kHz), HF (breitband)
            stages = [
                _lowshelf_coeffs(80, -5.0),
                _highshelf_coeffs(3000, -8.0),
                _highshelf_coeffs(9000, -6.0),
            ]
        elif self.mode == "dolby_b":
            stages = [_highshelf_coeffs(1000, -10.0)]
        elif self.mode == "dbx":
            # DBX: Vollband-Expansion -> leichter HF-Glätter (Tiefpass)
            k = 2.0 * sr * 70e-6  # 70µs Tiefpass
            b = np.array([1.0 / (k + 1.0), 1.0 / (k + 1.0)])
            a = np.array([1.0, (1.0 - k) / (k + 1.0)])

            def _ch_dbx(ch):
                return lfilter(b, a, ch.astype(np.float64))

            if audio.ndim == 1:
                return _ch_dbx(audio).astype(audio.dtype)  # type: ignore[no-any-return]
            return np.stack([_ch_dbx(c) for c in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
        else:  # auto
            stages = [_highshelf_coeffs(3000, -7.0)]
        if audio.ndim == 1:
            return _ch(audio, stages).astype(audio.dtype)  # type: ignore[no-any-return]
        return np.stack([_ch(c, stages) for c in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
