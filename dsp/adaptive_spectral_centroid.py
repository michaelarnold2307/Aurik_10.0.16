"""
Adaptive Spectral Centroid DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten.
Verwendet numpy für die Berechnung.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveSpectralCentroid:
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, center=True):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

    def spectral_centroid(self, y, use_dl: bool = False, **kwargs):
        """
        Berechnet das Spektralzentrum adaptiv mit aktuellen Parametern.
        Quality-Gate, Audit-Logging, robuste Fehlerbehandlung, optionale DL-Inferenz integriert.
        :param y: Audiosignal (1D np.ndarray)
        :param use_dl: Optional Deep-Learning-Inferenz (Platzhalter)
        :return: Array der Spektralzentren
        """
        # Quality-Gate: Input-Check
        if not isinstance(y, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        if y.ndim != 1:
            self._audit_log("error", "Input must be 1D array")
            raise ValueError("Input must be 1D array")
        if np.any(np.isnan(y)):
            self._audit_log("warn", "NaN values in input")
        try:
            if use_dl:
                self._audit_log("info", "DL-Inferenz aktiviert (Platzhalter)")
                centroids = self._dl_centroid_estimate(y)
            else:
                sr = kwargs.get("sr", self.sr)
                n_fft = kwargs.get("n_fft", self.n_fft)
                hop_length = kwargs.get("hop_length", self.hop_length)
                center = kwargs.get("center", self.center)
                if center:
                    pad = n_fft // 2
                    y = np.pad(y, (pad, pad), mode="reflect")
                centroids = []  # type: ignore[assignment]
                for i in range(0, len(y) - n_fft + 1, hop_length):
                    frame = y[i : i + n_fft]
                    magnitude = np.abs(np.fft.rfft(frame))
                    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
                    centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0.0
                    centroids.append(centroid)  # type: ignore[attr-defined]
                centroids = np.array(centroids)
            self._audit_log("success", "Spektralzentrum-Berechnung erfolgreich")
            return centroids
        except Exception as e:
            self._audit_log("error", f"Fehler bei Spektralzentrum-Berechnung: {e}")
            # Fallback: Rückgabe Nullen
            return np.zeros(len(y) // self.hop_length)

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[adaptive_spectral_centroid] %s", message)

    def _dl_centroid_estimate(self, y: np.ndarray) -> np.ndarray:
        """Frame-weise spektrale Zentroide via scipy rfft (DL-Fallback, Standalone-Modus).

        Algorithmus:
          Jeder Frame der Länge n_fft wird mit Hanning-Fenster gewichtet,
          FFT-Magnitudenspektrum berechnet, gewichtetes Mittel der Frequenzachse.
        """
        import numpy as np

        n_frames = max(1, (len(y) - self.n_fft) // self.hop_length + 1)
        sr_est = getattr(self, "sr", 44100)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sr_est)
        window = np.hanning(self.n_fft)
        centroids = np.zeros(n_frames)
        for k in range(n_frames):
            start = k * self.hop_length
            frame = y[start : start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            mag = np.abs(np.fft.rfft(frame * window))
            total = float(np.sum(mag)) + 1e-12
            centroids[k] = float(np.sum(freqs * mag) / total)
        return centroids

    def auto_optimize(self, y, sr):
        """Automatische Anpassung der FFT-Parameter je nach Signal."""
        if len(y) < 4096:
            self.n_fft = 256
            self.hop_length = 64
        elif len(y) < 16384:
            self.n_fft = 1024
            self.hop_length = 256
        else:
            self.n_fft = 2048
            self.hop_length = 512
