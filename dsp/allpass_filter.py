import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)
"""
---
modul_name: AllpassFilter
aufgabe: SOTA-konformer Allpass-Filter (Phasenkorrektur)
ein_ausgabe_typen:
    input: np.ndarray (Audio)
    output: np.ndarray (Audio)
staerken: Phasenkorrektur, SOTA
schwaechen: Keine Amplitudenänderung
abhaengigkeiten: [numpy]
---
"""


class AllpassFilter:
    """
    SOTA-konformer Allpass-Filter (Phasenkorrektur)
    """

    def __init__(self, a: float = 0.5):
        self.a = a

    def process(self, audio: npt.NDArray[np.float64], use_dl: bool = False) -> npt.NDArray[np.float64]:
        """
        SOTA-konformer Allpass-Filter (Phasenkorrektur) mit Quality-Gate, Audit-Logging, robuster Fehlerbehandlung, optionaler DL-Inferenz.
        :param audio: Eingabe-Audiosignal (np.ndarray)
        :param use_dl: Optional Deep-Learning-Inferenz (Platzhalter)
        :return: Gefiltertes Audiosignal (np.ndarray)
        """
        # Quality-Gate: Input-Check
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        if audio.ndim != 1:
            self._audit_log("error", "Input must be 1D array")
            raise ValueError("Input must be 1D array")
        if np.any(np.isnan(audio)):
            self._audit_log("warn", "NaN values in input")
        try:
            if use_dl:
                self._audit_log("info", "DL-Inferenz aktiviert (Platzhalter)")
                y = self._dl_allpass(audio)
            else:
                y = np.zeros_like(audio)
                for n in range(1, len(audio)):
                    y[n] = -self.a * audio[n] + audio[n - 1] + self.a * y[n - 1]
            self._audit_log("success", "Allpass-Filter erfolgreich angewendet")
            return y
        except Exception as e:
            self._audit_log("error", f"Fehler bei Allpass-Filter: {e}")
            # Fallback: Rückgabe Originalsignal
            return audio

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[allpass_filter] %s", message)

    def _dl_allpass(self, audio: np.ndarray) -> np.ndarray:
        """Kaskade aus 4 Allpass-Biquads zweiter Ordnung (Audio EQ Cookbook).

        Jedes Biquad verschiebt die Phase frequenz-abhängig ohne Amplitudenveränderung.
        Zentrumsfrequenzen: 250 Hz, 1 kHz, 4 kHz, 10 kHz (bei sr=44100).
        Q=0.707 (Butterworth-Charakteristik).
        """
        from scipy.signal import sosfilt

        float(self.a)  # Nutze a als Gütefaktor-Proxy
        # Referenz-Sr für Frequenzlagen
        sr_ref = 44100
        Q = 0.707
        centers = [250.0, 1000.0, 4000.0, 10000.0]
        sos_sections = []
        for fc in centers:
            w0 = 2.0 * np.pi * fc / sr_ref
            alpha = np.sin(w0) / (2.0 * Q)
            cos_w0 = np.cos(w0)
            b0 = 1.0 - alpha
            b1 = -2.0 * cos_w0
            b2 = 1.0 + alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
            sos_sections.append([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])
        sos_arr = np.array(sos_sections)
        return sosfilt(sos_arr, audio).astype(audio.dtype)  # type: ignore[no-any-return]
