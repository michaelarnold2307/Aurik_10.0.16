"""
noise_profile_matcher.py - Spektrales Rauschprofil-Matching für Aurik 6.0

Klassifiziert das Hintergrundrauschen eines Audiosignals anhand spektraler Merkmale:
  1. Stille-Segmente extrahieren (unterhalb RMS-Perzentil)
  2. Mittleres Leistungsspektrum der Stille berechnen
  3. Spektral-Fingerabdruck-Vergleich gegen vordefinierte Profile

Profile:
  - 'white':    Flaches Spektrum
  - 'brown':    Abfall -6 dB/Oktave (1/f^2)
  - 'pink':     Abfall -3 dB/Oktave (1/f)
  - 'hum':      Dominante Netzbrumm-Harmonische (50/60 Hz)
  - 'hiss':     HF-betontes Spektrum (> 4 kHz dominiert)
  - 'broadband': Breitbandiges Rauschen ohne klare Charakteristik
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NoiseProfileMatcher:
    """Spektrales Rauschprofil-Matching."""

    PROFILES = ["white", "brown", "pink", "hum", "hiss", "broadband"]

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def _extract_noise_spectrum(self, audio: np.ndarray, sr: int, n_fft: int = 2048) -> np.ndarray | None:
        """Extrahiert mittleres Spektrum der leisesten 20% des Signals."""
        if audio.ndim > 1:
            audio = audio[0]  # Mono-Kanal für Analyse
        audio = audio.astype(np.float64)
        frame_len = n_fft
        hop = frame_len // 2
        frames = []
        for i in range(0, len(audio) - frame_len, hop):
            frame = audio[i : i + frame_len]
            frames.append((np.sqrt(np.mean(frame**2)), i))
        if not frames:
            return None
        frames.sort(key=lambda x: x[0])
        n_quiet = max(1, len(frames) // 5)  # unterste 20%
        spectra = []
        for _, i in frames[:n_quiet]:
            mag = np.abs(np.fft.rfft(audio[i : i + frame_len] * np.hanning(frame_len)))
            spectra.append(mag**2)
        return np.mean(spectra, axis=0)  # type: ignore[no-any-return]

    @staticmethod
    def _classify(power_spectrum: np.ndarray, sr: int) -> str:
        """Klassifiziert Rauschprofil anhand spektraler Eigenschaften."""
        n = len(power_spectrum)
        freqs = np.fft.rfftfreq(2 * (n - 1), 1.0 / sr)
        freqs[0] = 1.0  # DC vermeiden
        eps = 1e-12
        total = float(np.sum(power_spectrum)) + eps
        # Netzbrumm-Check: Energie bei 50/60 Hz (und Harmonischen)
        hum_bins = []
        for f_hum in [50, 60]:
            for k in range(1, 6):
                b = round(f_hum * k * (2 * (n - 1)) / sr)
                if 0 < b < n:
                    hum_bins.append(b)
        hum_energy = float(np.sum(power_spectrum[hum_bins])) / total if hum_bins else 0.0
        if hum_energy > 0.15:
            return "hum"
        # HF-Dominanz: >4kHz vs Gesamt
        hf_start = int(4000 * (2 * (n - 1)) / sr)
        hf_energy = float(np.sum(power_spectrum[hf_start:])) / total if hf_start < n else 0.0
        if hf_energy > 0.55:
            return "hiss"
        # Spektrale Steigung (log-log Regression)
        log_f = np.log10(freqs[1:])
        log_p = np.log10(power_spectrum[1:] + eps)
        slope, _ = np.polyfit(log_f, log_p, 1)
        # Brown/Pink/White nach Steigung
        if slope < -5.0:
            return "brown"
        elif slope < -2.0:
            return "pink"
        elif abs(slope) <= 2.0:
            return "white"
        return "broadband"

    def match_profile(self, audio: np.ndarray, sr: int) -> str | None:
        """Klassifiziert das Hintergrundrausch-Profil des Signals.

        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :return: Profilname (str) oder None bei Fehler
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
            return None
        try:
            spectrum = self._extract_noise_spectrum(audio, sr)
            if spectrum is None or np.sum(spectrum) < 1e-12:
                return None
            return self._classify(spectrum, sr)
        except Exception:
            logger.warning("noise_Profil_matcher.py::match_Profil Ersatzpfad", exc_info=True)
            return None
