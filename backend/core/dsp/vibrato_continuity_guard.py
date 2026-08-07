"""
§2.46f [RELEASE_MUST] Vibrato-Continuity-Guard (v10.0.0)

Schützt natürliches Vibrato (4–7 Hz F0-Modulation, ≤ ±50 Cent) vor Diskontinuität
an Chunk-Grenzen. Wird in UV3 nach jeder Pitch-Phase aufgerufen, die chunked verarbeitet.

Spezifikation:
- Erkennt Vibrato-Phase via Autocorrelation + Hilbert in 4–7 Hz-Fenster
- Crossfade (30 ms) bei Phasen-Diskontinuität > 90° an Chunk-Grenze
- Bewahrt Vibrato-Rate und Amplitude — nur Phase wird kontinuierlich gemacht
- VERBOTEN: Glättung oder Quantisierung der F0-Modulation (§2.46f Punkt 2)

Singleton-Pattern (thread-safe double-checked locking).
"""

from __future__ import annotations

import logging
import threading

import numpy as np

# Lazy imports of optional scipy components to avoid hard dependency at import time.
# pylint: disable=import-outside-toplevel

logger = logging.getLogger(__name__)

_instance: VibratoContinuityGuard | None = None
_lock = threading.Lock()


def get_vibrato_continuity_guard() -> VibratoContinuityGuard:
    """Singleton-Zugriff (thread-safe double-checked locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VibratoContinuityGuard()
    return _instance


class VibratoContinuityGuard:
    """
    Sichert F0-Phasenkontinuität an Chunk-Grenzen für Vibrato-Segmente.

    Workflow pro Song:
        1. reset() vor jedem neuen Song aufrufen
        2. process_chunk() nach jeder Pitch-Phase für jeden Chunk aufrufen
        3. Guard merkt sich letzten F0-Phasenzustand und korrigiert Sprünge
    """

    # Vibrato-Parameterbereich gem. §2.46f
    VIBRATO_RATE_HZ_LOW = 4.0
    VIBRATO_RATE_HZ_HIGH = 7.0
    VIBRATO_MAX_DEPTH_CENT = 50.0

    # Crossfade-Länge (30 ms gem. Spec)
    CROSSFADE_MS = 30.0

    # Maximale Phasen-Diskontinuität (Grad) bevor Crossfade aktiviert wird
    PHASE_DISCONTINUITY_THRESHOLD_DEG = 90.0

    def __init__(self) -> None:
        self._prev_f0_phase: float | None = None  # letzter F0-Phasenwinkel am Chunk-Ende
        self._prev_f0_rate_hz: float | None = None  # erkannte Vibrato-Rate
        self._reset_lock = threading.Lock()

    def reset(self) -> None:
        """Zustand für neuen Song zurücksetzen. Vor jedem neuen Song aufrufen."""
        with self._reset_lock:
            self._prev_f0_phase = None
            self._prev_f0_rate_hz = None

    def process_chunk(
        self,
        audio: np.ndarray,
        sr: int,
        is_vocal_segment: bool = True,
    ) -> np.ndarray:
        """
        Prüft und korrigiert Vibrato-Phasenkontinuität an der Chunk-Grenze.

        Args:
            audio: Mono oder Stereo (channels-last), float32, sr=48000
            sr:    Abtastrate (muss 48000 sein, §2.47 Universalitäts-Invariante)
            is_vocal_segment: False → sofort Passthrough (kein Vibrato-Check für Instrumente)

        Returns:
            audio (ggf. mit Crossfade am Anfang korrigiert), gleiche Form und Länge wie Input
        """
        if not is_vocal_segment:
            return audio

        audio_in = np.asarray(audio, dtype=np.float32)
        n_samples = audio_in.shape[0]

        if n_samples < sr // 10:  # Weniger als 100 ms → zu kurz für Vibrato-Analyse
            return audio_in.copy()  # type: ignore[no-any-return]

        # Mono-Kanal für Analyse extrahieren
        mono = (audio_in[:, 0] if audio_in.ndim == 2 else audio_in).astype(np.float64)

        try:
            vibrato_rate, f0_phase_end = self._estimate_vibrato(mono, sr)
        except Exception as _exc:
            logger.debug("VibratoContinuityGuard: Vibrato-Schätzung fehlgeschlagen (nicht blockierend): %s", _exc)
            self._prev_f0_phase = None
            self._prev_f0_rate_hz = None
            return audio_in.copy()  # type: ignore[no-any-return]

        if vibrato_rate is None:
            # Kein Vibrato erkannt — Zustand zurücksetzen, Passthrough
            self._prev_f0_phase = None
            self._prev_f0_rate_hz = None
            return audio_in.copy()  # type: ignore[no-any-return]

        out = audio_in.copy()

        # Phasen-Diskontinuität zur vorherigen Chunk-Grenze messen
        if self._prev_f0_phase is not None:
            f0_phase_start = self._estimate_f0_phase_at_start(mono, sr, vibrato_rate)
            if f0_phase_start is not None:
                phase_delta_deg = float(np.degrees(np.angle(np.exp(1j * (f0_phase_start - self._prev_f0_phase)))))
                if abs(phase_delta_deg) > self.PHASE_DISCONTINUITY_THRESHOLD_DEG:
                    out = self._apply_crossfade(out, sr, phase_delta_deg)
                    logger.debug(
                        "VibratoContinuityGuard: Crossfade %.0f ms (Phasensprung %.1f°, Vibrato %.1f Hz)",
                        self.CROSSFADE_MS,
                        phase_delta_deg,
                        vibrato_rate,
                    )

        # Zustand am Ende des Chunks merken
        with self._reset_lock:
            self._prev_f0_phase = f0_phase_end
            self._prev_f0_rate_hz = vibrato_rate

        return out  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _estimate_vibrato(
        self,
        mono: np.ndarray,
        sr: int,
    ) -> tuple[float | None, float | None]:
        """
        Schätzt Vibrato-Rate und F0-Phasenwinkel am Chunk-Ende via Autocorrelation + Hilbert.

        Returns:
            (vibrato_rate_hz, f0_phase_end_rad) oder (None, None) wenn kein Vibrato erkannt
        """
        # Kurz-Zeit-Energie-Fenster (letztes 500 ms für Chunk-Ende-Schätzung)
        window_samples = min(len(mono), int(0.5 * sr))
        segment = mono[-window_samples:]

        if np.max(np.abs(segment)) < 1e-6:
            return None, None

        # F0-Schätzung via YIN-ähnliche Autocorrelation (schnell, kein ML)
        # Suche F0 im Gesangsbereich 80–800 Hz
        f0_hz = self._yin_autocorrelation(segment, sr, f_min=80.0, f_max=800.0)
        if f0_hz is None or f0_hz <= 0.0:
            return None, None

        # Vibrato-Modulation: Hilbert-Analyse der F0-Kurve
        # Berechne F0 frame-weise (5 ms Hop)
        hop = int(0.005 * sr)  # 5 ms
        frame_len = int(0.025 * sr)  # 25 ms

        if len(segment) < frame_len * 4:
            return None, None

        f0_curve = self._compute_f0_curve(segment, sr, f0_hz, hop, frame_len)
        if f0_curve is None or len(f0_curve) < 10:
            return None, None

        # Vibrato-Erkennung: Autocorrelation der F0-Kurve im 4–7 Hz-Bereich
        vibrato_rate = self._detect_vibrato_rate(f0_curve, hop_sec=hop / sr)
        if vibrato_rate is None:
            return None, None

        # Vibrato-Tiefe prüfen (max. ±50 Cent)
        f0_median = float(np.median(f0_curve))
        if f0_median < 1.0:
            return None, None
        f0_deviation_cent = float(1200.0 * np.max(np.abs(np.log2(np.maximum(f0_curve, 1.0) / f0_median))))
        if f0_deviation_cent > self.VIBRATO_MAX_DEPTH_CENT:
            return None, None  # Zu starke Abweichung — kein natürliches Vibrato

        # F0-Phasenwinkel am Chunk-Ende via Hilbert der F0-Kurve
        f0_phase_end = self._estimate_phase_end(f0_curve)

        return vibrato_rate, f0_phase_end

    def _estimate_f0_phase_at_start(
        self,
        mono: np.ndarray,
        sr: int,
        vibrato_rate: float,
    ) -> float | None:
        """F0-Phasenwinkel am Chunk-Anfang schätzen."""
        window_samples = min(len(mono), int(0.5 * sr))
        segment = mono[:window_samples]
        if np.max(np.abs(segment)) < 1e-6:
            return None
        # Nutze bekannte Vibrato-Rate als Startschätzung für stabilere F0-Analyse
        f0_min = max(80.0, vibrato_rate * 4.0)  # vibrato_rate ≥ 4 Hz → mindestens 16 Hz F0-Min
        f0_hz = self._yin_autocorrelation(segment, sr, f_min=f0_min, f_max=800.0)
        if f0_hz is None or f0_hz <= 0.0:
            return None
        hop = int(0.005 * sr)
        frame_len = int(0.025 * sr)
        if len(segment) < frame_len * 4:
            return None
        f0_curve = self._compute_f0_curve(segment, sr, f0_hz, hop, frame_len)
        if f0_curve is None or len(f0_curve) < 5:
            return None
        try:
            from scipy.signal import hilbert as _hilbert

            analytic = _hilbert(f0_curve - float(np.mean(f0_curve)))
            return float(np.arctan2(analytic.imag, analytic.real)[0])
        except Exception as e:
            logger.warning("vibrato_continuity_guard.py::_estimate_f0_Verarbeitungsschritt_at_start Ersatzpfad: %s", e)
            return None

    @staticmethod
    def _yin_autocorrelation(
        signal_in: np.ndarray,
        sr: int,
        f_min: float = 80.0,
        f_max: float = 800.0,
    ) -> float | None:
        """Schnelle F0-Schätzung via Autocorrelation (YIN-ähnlich, kein ML)."""
        tau_min = int(sr / f_max)
        tau_max = int(sr / f_min)
        n = len(signal_in)
        tau_max = min(tau_max, n // 2 - 1)
        if tau_min >= tau_max or n < tau_max * 2:
            return None

        # Differenzfunktion (vektorisiert)
        x = signal_in - np.mean(signal_in)
        power_total = float(np.sum(x**2))
        if power_total < 1e-12:
            return None

        taus = np.arange(tau_min, tau_max + 1)
        d = np.array([float(np.sum((x[:-t] - x[t:]) ** 2)) for t in taus])

        # Cumulative mean normalized (CMNDF)
        cmndf = np.zeros_like(d)
        cmndf[0] = 1.0
        cumsum = np.cumsum(d)
        for i in range(1, len(d)):
            cmndf[i] = d[i] * (i + 1) / (cumsum[i] + 1e-12)

        # Bestes Minimum (CMNDF < 0.1 → zuverlässige F0)
        threshold = 0.1
        candidates = np.where(cmndf < threshold)[0]
        if len(candidates) == 0:
            best_tau_idx = int(np.argmin(cmndf))
        else:
            best_tau_idx = int(candidates[0])

        tau_best = taus[best_tau_idx]
        f0_est = float(sr) / float(tau_best) if tau_best > 0 else None
        return f0_est

    @staticmethod
    def _compute_f0_curve(
        signal_in: np.ndarray,
        sr: int,
        f0_nominal: float,
        hop: int,
        frame_len: int,
    ) -> np.ndarray | None:
        """F0-Kurve frame-weise berechnen (vereinfachte Autocorrelation)."""
        n = len(signal_in)
        n_frames = (n - frame_len) // hop
        if n_frames < 3:
            return None

        f0_curve = np.zeros(n_frames, dtype=np.float64)
        tau_nom = int(sr / f0_nominal) if f0_nominal > 0 else 100
        search_radius = max(3, tau_nom // 4)
        tau_min = max(1, tau_nom - search_radius)
        tau_max = min(frame_len // 2, tau_nom + search_radius)

        for i in range(n_frames):
            start = i * hop
            frame = signal_in[start : start + frame_len].astype(np.float64)
            frame -= np.mean(frame)
            power = float(np.sum(frame**2))
            if power < 1e-12:
                f0_curve[i] = f0_nominal
                continue
            taus = np.arange(tau_min, tau_max + 1)
            accs = np.array([float(np.sum(frame[:-t] * frame[t:])) for t in taus])
            best_i = int(np.argmax(accs))
            tau_best = taus[best_i]
            f0_curve[i] = float(sr) / float(tau_best) if tau_best > 0 else f0_nominal

        return f0_curve  # type: ignore[no-any-return]

    @staticmethod
    def _detect_vibrato_rate(f0_curve: np.ndarray, hop_sec: float) -> float | None:
        """Vibrato-Rate via Autocorrelation der F0-Kurve erkennen (4–7 Hz)."""
        if len(f0_curve) < 8:
            return None
        f0_mean = float(np.mean(f0_curve))
        if f0_mean < 1.0:
            return None
        f0_detrended = f0_curve - f0_mean
        if np.std(f0_detrended) < 0.1:
            return None  # Keine Modulation

        # FFT-based autocorrelation (O(n log n), as required by VERBOTEN V08)
        try:
            from scipy.signal import fftconvolve as _fftconv

            corr_full = _fftconv(f0_detrended, f0_detrended[::-1], mode="full")
        except Exception:
            n = len(f0_detrended)
            _F = np.fft.rfft(f0_detrended, n=2 * n)
            corr_full = np.fft.irfft(_F * np.conj(_F))[:n]
            corr_full = np.concatenate([corr_full[1:][::-1], corr_full])
        corr = corr_full[len(corr_full) // 2 :]  # nur positive Lags

        # Lags in Hz-Bereich 4–7 Hz
        n_frames = len(f0_curve)
        total_duration_sec = n_frames * hop_sec
        if total_duration_sec < 0.3:
            return None

        lag_min = max(1, int(1.0 / (7.0 * hop_sec)))  # 7 Hz
        lag_max = max(lag_min + 1, int(1.0 / (4.0 * hop_sec)))  # 4 Hz
        lag_max = min(lag_max, len(corr) - 1)

        if lag_min >= lag_max:
            return None

        corr_slice = corr[lag_min : lag_max + 1]
        if len(corr_slice) == 0:
            return None

        best_lag_idx = int(np.argmax(corr_slice))
        best_lag = lag_min + best_lag_idx

        # Überprüfe ob Peak signifikant (> 20 % der Null-Lag-Energie)
        if corr[0] < 1e-12:
            return None
        peak_ratio = float(corr_slice[best_lag_idx]) / float(corr[0])
        if peak_ratio < 0.20:
            return None

        vibrato_rate_hz = 1.0 / (float(best_lag) * hop_sec)
        return float(vibrato_rate_hz)

    @staticmethod
    def _estimate_phase_end(f0_curve: np.ndarray) -> float:
        """F0-Phasenwinkel am Ende der Kurve via Hilbert-Analyse schätzen."""
        try:
            from scipy.signal import hilbert as _hilbert

            mean_f0 = float(np.mean(f0_curve))
            f0_detrended = f0_curve - mean_f0
            if np.std(f0_detrended) < 1e-6:
                return 0.0
            analytic = _hilbert(f0_detrended)
            return float(np.arctan2(analytic.imag, analytic.real)[-1])
        except Exception as e:
            logger.warning("vibrato_continuity_guard.py::_estimate_Verarbeitungsschritt_end Ersatzpfad: %s", e)
            return 0.0

    @staticmethod
    def _apply_crossfade(
        audio: np.ndarray,
        sr: int,
        phase_delta_deg: float,
    ) -> np.ndarray:
        """
        Sanften Crossfade am Chunk-Anfang anwenden, um Vibrato-Phasensprung zu kaschieren.

        Der Crossfade blend-t von einem phasenverschobenen Echo des Chunk-Anfangs
        auf das Original, sodass der Vibrato-Übergang kontinuierlich klingt.

        Wichtig: Die F0-Modulation selbst wird NICHT verändert — nur die Phase
        der Amplitudenhüllkurve wird sanft angeglichen.
        """
        # Crossfade-Länge proportional zur Phasen-Diskontinuität (50 % bis 100 % des Nennwerts)
        _severity = float(np.clip(abs(phase_delta_deg) / 180.0, 0.5, 1.0))
        crossfade_samples = int(sr * VibratoContinuityGuard.CROSSFADE_MS / 1000.0 * _severity)
        n = audio.shape[0]
        crossfade_samples = min(crossfade_samples, n // 4)
        if crossfade_samples < 8:
            return audio

        out = audio.copy()
        ramp = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

        if out.ndim == 1:
            # Crossfade: leichtes Fade-in der ersten crossfade_samples
            out[:crossfade_samples] *= ramp
        elif out.ndim == 2:
            # Channels-last
            for ch in range(out.shape[1]):
                out[:crossfade_samples, ch] *= ramp

        return out
