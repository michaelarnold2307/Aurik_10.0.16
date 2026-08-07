"""
ML-basierter De-Esser für Aurik 10.0.0
Open Source, Eigenentwicklung — scipy/numpy (kein torch/torchaudio)

Spektrale De-Esser Implementierung:
  1. STFT -> Spektralleistung
  2. Sibilanten-Detektion: Energie in typischem Sibilanten-Band (4–8 kHz > Schwellwert)
  3. Gain-Reduktion: Envelope-Follower pro Frame auf HF-Band
  4. ISTFT -> Zeitbereich
"""

import logging

import numpy as np
import soundfile as sf
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)


class MLDeEsser:
    """Spektraler De-Esser (scipy-only, kein torch)."""

    def __init__(
        self,
        model_path: str | None = None,
        sibilant_threshold: float = 0.5,
        sibilant_low_hz: float = 4000.0,
        sibilant_high_hz: float = 12000.0,
        reduction_db: float = 6.0,
    ):
        """
        :param model_path: Ignoriert (für Kompatibilität)
        :param sibilant_threshold: Sibilanten-Score-Schwellwert (0–1)
        :param sibilant_low_hz: Untere Frequenz des Sibilanten-Bands
        :param sibilant_high_hz: Obere Frequenz des Sibilanten-Bands
        :param reduction_db: Maximale Gain-Reduktion in dB
        """
        self.model_path = model_path
        self.sibilant_threshold = sibilant_threshold
        self.sibilant_low_hz = sibilant_low_hz
        self.sibilant_high_hz = sibilant_high_hz
        self.reduction_db = reduction_db

    def predict_sibilants(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet Sibilanten-Score (0..1) anhand spektraler Energie.

        :param audio: Mono-Signal (np.ndarray)
        :param sr: Abtastrate
        :return: Sibilanten-Score
        """
        if audio.ndim > 1:
            audio = audio[0]  # Mono
        n = 2048
        mag = np.abs(np.fft.rfft(audio[:n] * np.hanning(min(n, len(audio))), n=n))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        sib_mask = (freqs >= self.sibilant_low_hz) & (freqs <= self.sibilant_high_hz)
        total_energy = float(np.sum(mag**2)) + 1e-12
        sib_energy = float(np.sum(mag[sib_mask] ** 2))
        return float(np.clip(sib_energy / total_energy * 3.0, 0.0, 1.0))

    def reduce_sibilants(self, audio_path: str, output_path: str) -> str:
        """Reduziert Sibilanten in einer Audiodatei.

        :param audio_path: Pfad zur Eingabedatei
        :param output_path: Pfad zur Ausgabedatei
        :return: output_path
        """
        import numpy as np

        from backend.file_import import load_audio_file

        _res = load_audio_file(audio_path)
        audio = np.asarray(_res["audio"], dtype=np.float64)  # type: ignore[index]
        sr = int(_res["sr"])  # type: ignore[index]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, samples) for always_2d compat
        else:
            audio = audio.T  # (channels, samples)
        out = self.process(audio, sr)
        sf.write(output_path, out.T, sr)
        return output_path

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Wendet De-Esser auf np.ndarray-Signal an (mono oder stereo).

        :param audio: Eingabesignal (channels, samples) oder (samples,)
        :param sr: Abtastrate
        :return: De-essiertes Signal
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        nperseg = 1024
        noverlap = nperseg * 3 // 4
        reduction_lin = 10.0 ** (-abs(self.reduction_db) / 20.0)
        nyq = sr / 2.0
        max(self.sibilant_low_hz / nyq, 0.001)
        min(self.sibilant_high_hz / nyq, 0.499)

        def _deess_mono(ch: np.ndarray) -> np.ndarray:
            ch = ch.astype(np.float64)
            _, _times, Zxx = stft(ch, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            freqs_stft = np.fft.rfftfreq(nperseg, 1.0 / sr)
            sib_mask = (freqs_stft >= self.sibilant_low_hz) & (freqs_stft <= self.sibilant_high_hz)
            total_energy = np.sum(np.abs(Zxx) ** 2, axis=0) + 1e-12
            sib_energy = np.sum(np.abs(Zxx[sib_mask]) ** 2, axis=0)
            sib_score = np.clip(sib_energy / total_energy * 3.0, 0.0, 1.0)
            # Gain pro Frame: 1.0 wenn kein Sibilant, reduction_lin wenn voll
            gain_per_frame = 1.0 - sib_score * (1.0 - reduction_lin)
            # Nur Sibilanten-Bins dämpfen
            Zxx_out = Zxx.copy()
            Zxx_out[sib_mask, :] *= gain_per_frame[np.newaxis, :]
            _, y_out = istft(Zxx_out, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            n = len(ch)
            if len(y_out) >= n:
                return y_out[:n]  # type: ignore[no-any-return]
            return np.pad(y_out, (0, n - len(y_out)))

        try:
            if audio.ndim == 1:
                return _deess_mono(audio).astype(audio.dtype)
            return np.stack([_deess_mono(ch) for ch in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
        except Exception:
            logger.warning("deesser_ml.py::_deess_mono Ersatzpfad", exc_info=True)
            return audio
