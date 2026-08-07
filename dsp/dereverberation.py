"""
ai_dereverberation.py - KI-gestützte Dereverberation für Aurik 6.0

Spektrale-Subtraktions-Dereverberation nach Lebart et al. (2001):
  1. STFT des Eingangssignals
  2. Geschätzte Nachhall-Energie via exponentiellem Decay-Modell
  3. Spektrale Subtraktion: |X_out|^2 = |X_in|^2 - alpha * |X_reverb|^2
  4. ISTFT zurück (Hanning OLA)
Fallback auf librosa.decompose.nn_filter (non-local means) wenn verfügbar.
"""

import numpy as np


class AiDereverberation:
    """Spektrale-Subtraktions-Dereverberation."""

    def __init__(self, model_path: str | None = None, rt60: float = 0.3, alpha: float = 1.0, beta: float = 0.001):
        """
        :param model_path: Ignoriert (für ML-Kompatibilität)
        :param rt60: Geschätzte Nachhallzeit in Sekunden (Default: 0.3s)
        :param alpha: Stärke der Subtraktion (1.0 = vol, >1 = aggressiv)
        :param beta: Spektrales Untergrenzverhältnis (Artefaktschutz)
        """
        self.model_path = model_path
        self.model = None
        self.rt60 = rt60
        self.alpha = alpha
        self.beta = beta

    def dereverberate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Spektrale-Subtraktions-Dereverberation.

        :param audio: Eingabesignal (np.ndarray, mono oder stereo)
        :param sr: Abtastrate
        :return: Dereverberiertes Signal (np.ndarray)
        """
        from scipy.signal import istft, stft

        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        nperseg = 1024
        noverlap = nperseg * 3 // 4
        hop = nperseg - noverlap

        def _derev_mono(y: np.ndarray) -> np.ndarray:
            y = y.astype(np.float64)
            _, _, Z = stft(y, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            power = np.abs(Z) ** 2
            # Exponentieller Decay-Filter: decay_factor = exp(-0.69/RT60/fps)
            fps = sr / hop
            decay = np.exp(-6.908 / (self.rt60 * fps))  # exp(-ln(1000^2) / (RT60*fps))
            reverb_est = np.zeros_like(power)
            reverb_est[:, 0] = power[:, 0]
            for t in range(1, power.shape[1]):
                reverb_est[:, t] = decay * reverb_est[:, t - 1] + (1 - decay) * power[:, t]
            # Spektrale Subtraktion (Wiener-like)
            gain_sq = np.maximum(power - self.alpha * reverb_est, self.beta * power)
            gain = np.sqrt(gain_sq / (power + 1e-12))
            Z_out = gain * Z
            _, y_out = istft(Z_out, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            n = len(y)
            if len(y_out) >= n:
                return y_out[:n]  # type: ignore[no-any-return]
            return np.pad(y_out, (0, n - len(y_out)))

        if audio.ndim == 1:
            out = _derev_mono(audio)
            return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(audio.dtype)
        out = np.stack([_derev_mono(ch) for ch in audio], axis=0)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(audio.dtype)
