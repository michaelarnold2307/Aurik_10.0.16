"""processing/stem_based_processor.py

Stem-basierter DSP-Prozessor für Aurik 10.0.0.

Bietet private DSP-Methoden für Transient-Enhancement,
intelligenten Click-Removal, Bass-Enhancement und sanfte
Rauschunterdrückung (OMLSA/Minimum-Statistics + Wiener-Gain).
All workload runs on the CPU with pure numpy/scipy – no ML model required.

Spec-Referenz:
    §2.27  TransientDecoupledProcessing (HPSS-Mediannfilter)
    §4.5   Rauschunterdrückung: OMLSA/IMCRA (Cohen 2002/2003)
    §4.5   DecrackleClick: RBME + iterative Konsistenz
    §3.1   Numerische Robustheit
    §3.2   Singleton + Convenience-Pattern
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import (
    butter,
    sosfilt,
)
from scipy.signal import (
    istft as scipy_istft,
)
from scipy.signal import (
    stft as scipy_stft,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modul-Konstanten
# ---------------------------------------------------------------------------
_NPERSEG: int = 1024  # STFT-Fensterlänge (Samples)
_NOVERLAP: int = 768  # 75 % Überlappung
_G_FLOOR: float = 0.10  # Minimum-Gain (verhindert Musical Noise)
_ALPHA_NR: float = 0.95  # Temporale Smoothing der Rauschschätzung
_BASS_CROSSOVER_HZ: float = 200.0  # Low-Shelf-Grenzfrequenz
_BASS_GAIN: float = 0.20  # Anteil Bass-Addition (~+3 dB)
_HPSS_K_TIME: int = 15  # HPSS Medianfilter Zeitachse (ungerade)
_HPSS_K_FREQ: int = 11  # HPSS Medianfilter Frequenzachse (ungerade)
_CLICK_SIGMA_MULT: float = 6.0  # Schwellwert-Faktor für Click-Detektion
_CLICK_ROBUST: float = 0.6745  # MAD → σ-Skalierung (Gaußians)


# ===========================================================================
# Hauptklasse
# ===========================================================================


class StemBasedProcessor:
    """Stem-basierter Audio-Prozessor ohne ML-Abhängigkeiten.

    Bietet private DSP-Methoden für Transient-Enhancement,
    Click-Removal, Bass-Enhancement und sanfte Rauschunterdrückung.
    Die Methoden operieren direkt auf dem Vollsignal (Mono, float32),
    ohne dass ein geladenes Separation-Modell benötigt wird.

    Args:
        separation_model: Name des ML-Stem-Separation-Modells (z.B. ``"demucs_v4"``).
                         Wird nur als Label gespeichert; die privaten DSP-Methoden
                         arbeiten modellunabhängig.
    """

    def __init__(self, separation_model: str = "demucs_v4") -> None:
        self.separation_model = separation_model
        logger.debug("StemBasedProcessor init: model=%s (DSP-only Modus)", separation_model)

    # ------------------------------------------------------------------
    # Öffentliche Convenience-API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Vollständige DSP-Kette auf einem Mono-Signal.

        Reihenfolge:  Click-Removal → NR → Bass → Transienten.

        Args:
            audio: 1-D float32 normalisiert auf ``[-1, 1]``.
            sr:    Sample-Rate in Hz.

        Returns:
            Bearbeitetes Signal (gleiche Shape), normalisiert auf ``[-1, 1]``.
        """
        out = self._intelligent_click_removal(audio, sr)
        out = self._gentle_noise_reduction(out, sr)
        out = self._bass_enhancement(out, sr)
        out = self._enhance_transients(out, sr)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    # ------------------------------------------------------------------
    # Private DSP-Methoden (direkt per Unit-Test prüfbar)
    # ------------------------------------------------------------------

    def _enhance_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """HPSS-basiertes Transient-Enhancement via STFT-Medianfilter.

        Algorithmus:
            1. STFT → Betragsspektrogramm
            2. Harmonische Maske H: horizontaler Medianfilter (entlang Zeit)
            3. Perkussive Maske P: vertikaler Medianfilter (entlang Frequenz)
            4. Soft-Maske für Perkussives: mask_p = P / (P + H + ε)
            5. Boost-Faktor: 1.0 + 0.5 · mask_p  (max 1.5 × Original)
            6. ISTFT Rücksynthese mit originaler Phase

        Args:
            audio: 1-D float32 Signal ``∈ [−1, 1]``.
            sr:    Sample-Rate (für STFT-Parametrierung).

        Returns:
            Signal mit betonten Transienten, gleiche Shape, ``∈ [−1, 1]``.
        """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n < _NPERSEG:
            return audio.copy()

        # STFT Analyse
        _, _, Zxx = scipy_stft(
            audio,
            fs=float(sr),
            nperseg=_NPERSEG,
            noverlap=_NOVERLAP,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # HPSS: harmonisch = horizontal glatt, perkussiv = vertikal glatt
        k_t = min(_HPSS_K_TIME, max(3, mag.shape[1] // 8) | 1)
        k_f = min(_HPSS_K_FREQ, max(3, mag.shape[0] // 8) | 1)
        H = median_filter(mag, size=(1, k_t))  # horizontal → harmonisch
        P = median_filter(mag, size=(k_f, 1))  # vertikal   → perkussiv

        # Soft-Maske perkussiver Anteil
        eps = 1e-8
        mask_p = P / (P + H + eps)

        # Boost: Transienten-Energie anheben (Faktor 1.0-1.5)
        boost = 1.0 + 0.5 * mask_p
        mag_out = mag * boost

        # Rücksynthese mit originaler Phase
        Zxx_out = mag_out * np.exp(1j * phase)
        _, out = scipy_istft(
            Zxx_out,
            fs=float(sr),
            nperseg=_NPERSEG,
            noverlap=_NOVERLAP,
            boundary=True,
        )
        out = out.astype(np.float32)

        # Länge angleichen
        if len(out) >= n:
            out = out[:n]
        else:
            out = np.pad(out, (0, n - len(out)))

        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    def _intelligent_click_removal(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Zweite-Differenz-basierter Click-Detektor mit linearer Interpolation.

        Algorithmus:
            1. Zweite Differenz d2[i] = x[i+1] - 2·x[i] + x[i-1]
               (Clicks erzeugen dort starke Ausschläge, glattes Signal → klein)
            2. Schwellwert: 6 · MAD(d2) / 0.6745  (σ-Schätzung)
            3. Click-Positionen markieren wo |d2| > thresh
            4. Linearer Übergang zwischen nächsten unbeschädigten Samples
               (walk left/right bis erstes unmarkiertes Sample)

        Args:
            audio: 1-D float32 ``∈ [−1, 1]``.
            sr:    Sample-Rate (für API-Konsistenz, intern indirekt genutzt).

        Returns:
            Signal ohne Impuls-Klicks, gleiche Shape, ``∈ [−1, 1]``.
        """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n < 16:
            return audio.copy()

        # Zweite Differenz (Leave-Eins-Rand frei)
        d2 = np.zeros(n, dtype=np.float64)
        d2[1:-1] = audio[2:].astype(np.float64) - 2.0 * audio[1:-1].astype(np.float64) + audio[:-2].astype(np.float64)

        # Robuste σ-Schätzung via MAD (Gaussian consistency factor)
        mad_d2 = float(np.median(np.abs(d2[1:-1])))
        if mad_d2 < 1e-12:
            return audio.copy()  # Kein messbares Rauschen → kein Eingriff

        sigma_d2 = mad_d2 / _CLICK_ROBUST
        thresh = _CLICK_SIGMA_MULT * sigma_d2

        click_mask: np.ndarray = np.abs(d2) > thresh
        if not np.any(click_mask):
            return audio.copy()

        out = audio.copy()

        # Interpolation: für jede markierte Position → nächste saubere Nachbarn
        for i in np.where(click_mask)[0]:
            # Linken sauberen Nachbarn suchen
            x0 = int(i) - 1
            while x0 >= 0 and click_mask[x0]:
                x0 -= 1
            # Rechten sauberen Nachbarn suchen
            x1 = int(i) + 1
            while x1 < n and click_mask[x1]:
                x1 += 1

            if x0 >= 0 and x1 < n:
                t = (i - x0) / max(x1 - x0, 1)
                out[i] = (1.0 - t) * audio[x0] + t * audio[x1]
            elif x0 >= 0:
                out[i] = audio[x0]
            elif x1 < n:
                out[i] = audio[x1]
            else:
                out[i] = 0.0

        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0)

    def _bass_enhancement(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Sanfte Low-Shelf-Bass-Anhebung (~+3 dB unter 200 Hz).

        Algorithmus:
            1. Butterworth Low-Pass 4. Ordnung bei 200 Hz → Tiefton-Anteil
            2. ``out = audio + _BASS_GAIN · bass``
            3. Peak-Normalisierung auf 1.0 falls ``max(|out|) > 1.0``

        Args:
            audio: 1-D float32 ``∈ [−1, 1]``.
            sr:    Sample-Rate in Hz.

        Returns:
            Signal mit angehobenem Bassanteil, gleiche Shape, ``∈ [−1, 1]``.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < 32:
            return audio.copy()

        nyq = sr / 2.0
        cutoff = min(_BASS_CROSSOVER_HZ, nyq * 0.90)
        sos = butter(4, cutoff / nyq, btype="low", output="sos")
        bass = sosfilt(sos, audio.astype(np.float64)).astype(np.float32)

        out = audio + _BASS_GAIN * bass

        # §VERBOTEN: Peak-Normalisierung würde LUFS-Verhältnis zum Original verzerren.
        # TruePeak-Schutz ausschließlich via hartem Clip (letzter Schritt nach NaN-Bereinigung).
        return np.clip(np.nan_to_num(out.astype(np.float32), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    def _gentle_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """OLA-STFT Rauschunterdrückung: Minimum-Statistics + Wiener-Gain.

        Algorithmus (OMLSA-vereinfacht nach Cohen & Berdugo 2002):
            1. STFT: nperseg=1024, noverlap=768 (75 %)
            2. Rausch-PSD-Schätzung: gleitendes Minimum der Magnitude
               über ein Fenster von ``hist_len`` Frames (P5-Perzentil)
            3. SNR_post = mag[t]² / max(noise_est², ε)
            4. Wiener-Gain: G = max(G_floor, SNR_post / (SNR_post + 1))
            5. ISTFT mit modifizierter Betragsspektrum + Original-Phase

        Args:
            audio: 1-D float32 ``∈ [−1, 1]``.
            sr:    Sample-Rate in Hz.

        Returns:
            Rauschreduziertes Signal, gleiche Shape, ``∈ [−1, 1]``.
        """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n < _NPERSEG:
            return audio.copy()

        _, _, Zxx = scipy_stft(
            audio,
            fs=float(sr),
            nperseg=_NPERSEG,
            noverlap=_NOVERLAP,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        n_frames = mag.shape[1]

        # Historien-Fenstergröße (max. 50 Frames, min. 10)
        hist_len = max(10, min(50, n_frames // 4))

        # Initialisierung: Rauschboden aus P5 aller Frames (globaler Schätzwert)
        noise_est: np.ndarray = np.full(mag.shape[0], np.percentile(mag, 5), dtype=np.float64)

        mag_out = np.empty_like(mag)
        history: list[np.ndarray] = []

        for t in range(n_frames):
            history.append(mag[:, t].astype(np.float64))
            if len(history) > hist_len:
                history.pop(0)
            if len(history) >= 2:
                stack = np.stack(history, axis=1)  # [n_bins, hist_len]
                noise_est = np.percentile(stack, 5, axis=1)

            snr_post = mag[:, t].astype(np.float64) ** 2 / np.maximum(noise_est**2, 1e-10)
            G = np.maximum(_G_FLOOR, snr_post / (snr_post + 1.0))
            mag_out[:, t] = mag[:, t] * G

        Zxx_out = mag_out * np.exp(1j * phase)
        _, out = scipy_istft(
            Zxx_out,
            fs=float(sr),
            nperseg=_NPERSEG,
            noverlap=_NOVERLAP,
            boundary=True,
        )
        out = out.astype(np.float32)

        if len(out) >= n:
            out = out[:n]
        else:
            out = np.pad(out, (0, n - len(out)))

        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    def _compute_quality(self, audio: np.ndarray, sr: int) -> float:
        """SNR-basierte MOS-Qualitätsschätzung im Bereich ``[1.0, 5.0]``.

        Algorithmus:
            1. STFT-Betragsspektrogramm
            2. Signal-PSD: 95. Perzentil des Betragsspektrums
            3. Rausch-PSD: 5. Perzentil des Betragsspektrums
            4. Gesamt-SNR (dB) via Verhältnis Signal/Rausch-PSD
            5. Sigmoid-Mapping: MOS = 1 + 4 / (1 + exp(−(snr_dB − 10) / 8))
            6. Default 3.8 bei Stille (RMS < 1e-7) oder zu kurzen Signalen

        Args:
            audio: 1-D float32 Signal.
            sr:    Sample-Rate in Hz.

        Returns:
            Qualitäts-Schätzung (MOS-analog) ``∈ [1.0, 5.0]``.

        Raises:
            None — NaN/Inf-sicher.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < _NPERSEG:
            return 3.8

        # Stille-Check
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 1e-7:
            return 3.8

        _, _, Zxx = scipy_stft(
            audio,
            fs=float(sr),
            nperseg=_NPERSEG,
            noverlap=_NOVERLAP,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(Zxx)
        if mag.shape[1] < 2:
            return 3.8

        signal_psd = float(np.percentile(mag, 95)) ** 2
        noise_psd = max(float(np.percentile(mag, 5)) ** 2, 1e-12)

        snr_db = 10.0 * math.log10(max(signal_psd / noise_psd, 1e-6))

        # Sigmoid-Mapping: 0 dB → ≈ 2.5 MOS, 30 dB → ≈ 4.5 MOS
        mos = 1.0 + 4.0 / (1.0 + math.exp(-(snr_db - 10.0) / 8.0))
        return float(np.clip(mos, 1.0, 5.0))


# ---------------------------------------------------------------------------
# Modul-Level Singleton (§3.2)
# ---------------------------------------------------------------------------
import threading as _threading

_instance: StemBasedProcessor | None = None
_lock = _threading.Lock()


def get_stem_processor(separation_model: str = "demucs_v4") -> StemBasedProcessor:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking).

    Returns:
        Die globale :class:`StemBasedProcessor`-Instanz.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StemBasedProcessor(separation_model=separation_model)
    return _instance
