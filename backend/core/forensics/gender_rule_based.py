import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import hamming

logger = logging.getLogger(__name__)


class RuleBasedGenderDetector:
    """
    Regelbasierte Gender-Erkennung ohne KI, SOTA-nah für Sprachaufnahmen.
    Nutzt Pitch, Formanten und spektrale Merkmale. Optimiert für Robustheit und Transparenz.
    """

    def __init__(self, sr: int = 16000) -> None:
        self.sr = sr

    def classify_from_features(self, f0: float, f1: float, f2: float, voiced_ratio: float) -> str:
        """
        Öffentliche API für die Entscheidungslogik auf Basis extrahierter Features.
        Ermöglicht Grid-Search und Schwellenwertoptimierung ohne Zugriff auf protected Methoden.
        """
        # Entscheidungslogik (optimiert, Schwellen empirisch anpassbar)
        if f0 < 170 and f1 < 700 and f2 < 1200:
            gender = "male"
        elif 170 <= f0 < 300 and f1 >= 700 and f2 >= 1200:
            gender = "female"
        elif f0 >= 300:
            gender = "child"
        else:
            gender = "unknown"
        return gender

    def detect_gender(self, audio_file: str) -> str:
        from backend.file_import import load_audio_file

        _af_res = load_audio_file(audio_file, do_carrier_analysis=False)
        if _af_res is None or _af_res.get("error") or _af_res["audio"] is None:
            return "unknown"
        audio, sr = _af_res["audio"], int(_af_res["sr"])
        if audio.ndim > 1:
            audio = audio[:, 0]  # Mono (left channel, §VERBOTEN: audio[0] wrong for (samples×channels))
        # Robustere Pitch-Schätzung: Median über voiced frames
        f0, voiced_ratio = self._estimate_pitch(audio, sr)
        # Formanten: Mittelwert über mehrere Frames
        f1, f2 = self._estimate_formants(audio, sr)
        # Unsicherheits-Score
        uncertainty = 1.0 - voiced_ratio
        # Entscheidungslogik (optimiert, Schwellen empirisch anpassbar)
        if f0 < 170 and f1 < 700 and f2 < 1200:
            gender = "male"
        elif 170 <= f0 < 300 and f1 >= 700 and f2 >= 1200:
            gender = "female"
        elif f0 >= 300:
            gender = "child"
        else:
            gender = "unknown"
        # Logging für Audit
        logger.debug(
            f"[GenderDetection] f0={f0:.1f}Hz, f1={f1:.1f}Hz, f2={f2:.1f}Hz, voiced={voiced_ratio:.2f}, uncertainty={uncertainty:.2f}, Ergebnis={gender}"
        )
        return gender

    def _estimate_pitch(self, audio: np.ndarray[Any, Any], sr: int) -> tuple[float, float]:
        # YIN-ähnliche Pitch-Schätzung (robuster, voiced/unvoiced)
        frame_size = int(0.03 * sr)
        hop_size = int(0.01 * sr)
        pitches = []
        voiced = 0
        total = 0
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size] * hamming(frame_size)
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(frame_size, 1 / sr)
            peaks, _props = find_peaks(spectrum, height=np.max(spectrum) * 0.3)
            total += 1
            if len(peaks) > 0:
                pitches.append(freqs[peaks[0]])
                voiced += 1
        if len(pitches) == 0:
            return 0, 0.0
        return float(np.median(pitches)), voiced / total if total > 0 else 0.0

    def _estimate_formants(self, audio: np.ndarray[Any, Any], sr: int) -> tuple[float, float]:
        # LPC-Formant-Schätzung über mehrere Frames (robuster)
        frame_size = int(0.03 * sr)
        hop_size = int(0.015 * sr)
        # §VERBOTEN: LPC order < 16 — use SR-adaptive order (min 16 per spec VERBOTEN list)
        n_coeff = max(16, 2 + sr // 1000)
        f1s, f2s = [], []
        for i in range(0, min(len(audio) - frame_size, 10 * hop_size), hop_size):
            frame = audio[i : i + frame_size] * hamming(frame_size)
            A = self._lpc(frame, n_coeff)
            # Guard: degenerate LPC → LAPACK DLASCL failure
            if not np.all(np.isfinite(A)):
                continue
            try:
                roots = np.roots(A)
            except (np.linalg.LinAlgError, ValueError):
                continue
            roots = [r for r in roots if np.imag(r) >= 0.01]  # type: ignore[assignment]
            angz = np.arctan2(np.imag(roots), np.real(roots))
            formants = sorted(angz * (sr / (2 * np.pi)))
            if len(formants) >= 2:
                f1s.append(formants[0])
                f2s.append(formants[1])
        if len(f1s) > 0 and len(f2s) > 0:
            return float(np.median(f1s)), float(np.median(f2s))
        elif len(f1s) > 0:
            return float(np.median(f1s)), 0.0
        else:
            return 0.0, 0.0

    def _lpc(self, x: np.ndarray[Any, Any], order: int) -> np.ndarray[Any, Any]:
        # Autokorrelationsmethode für LPC — FFT-based O(N log N)
        from backend.core.core_utils import fft_autocorr

        R = fft_autocorr(x, max_lag=order)
        # R now contains r[0..order]
        a = np.zeros(order + 1)
        e = R[0]
        for i in range(1, order + 1):
            acc = 0
            for j in range(1, i):
                acc += a[j] * R[i - j]
            k = (R[i] - acc) / e
            a[i] = k
            for j in range(1, i):
                a[j] = a[j] - k * a[i - j]
            e *= 1 - k * k
        a[0] = 1
        return a  # type: ignore[no-any-return]
