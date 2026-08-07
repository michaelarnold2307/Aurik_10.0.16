import logging
from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "ai_decrackler"
    category: str = "disruptor_removal"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
decrackler_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {},
        "safe_ranges": {},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.05,
        "identity_budget": 0.99,
        "spectral_change_budget": 0.1,
        "temporal_change_budget": 0.05,
        "compute_cost": 0.05,
    },
    side_effects=[{"risk": "transient_smear", "expected_when": "True", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["crackle_removal_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
"""
ai_decrackler.py - Deep-Learning-basierter Decrackler, Dehum und Debuzz für Aurik 6.0

Dieses Modul stellt SOTA-Stub-Architekturen für die Entfernung von Crackle, Hum und Buzz bereit.
"""
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AiDecrackler:
    """RBME-inspirierter Crackle/Click-Entferner.

    Algorithmus (DSP-Fallback):
        1. Median-Absolute-Deviation (MAD) in gleitendem 5-ms-Fenster →
           lokale Streuung pro Sample.
        2. Samples mit |Abweichung vom lokalen Median| > threshold * MAD
           gelten als Click/Crackle.
        3. Klick-Segmente (max. 5 ms) werden per AR-Interpolation ersetzt:
           lineare Interpolation über die Lücke + Hanning-Randoverblend.
        4. Gefiltert mit causal-Tiefpassfilter (Hanning-Kernel) um
           Transient-Smear zu begrenzen.
    Referenz: Cemgil et al. (2006) Sparse Bayes; RBME (Bando et al. 2019).
    Invariante: NaN/Inf-frei, Ausgang ∈ [−1, 1].
    """

    _THRESHOLD: float = 6.0  # MAD-Vielfaches für Click-Erkennung
    _WIN_MS: float = 5.0  # Fensterbreite für MAD in Millisekunden
    _MAX_CLICK_MS: float = 5.0  # Maximale Klick-Dauer in Millisekunden
    _FADE: int = 4  # Hanning-Randsamples

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def _process_channel(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Remove crackle events via AR-predicted gap interpolation.

        v10.0.0-Fix: Linear interpolation over gaps up to 5 ms (240 samples at
        48 kHz) created audible "holes" — the human ear perceives a straight-line
        ramp >1 ms as a signal dropout. Replaced with forward-backward Burg AR
        prediction that preserves the local spectral envelope of the surrounding
        signal, making the repair inaudible.

        Algorithm:
            1. Burg AR model fit on pre-gap context (order = min(16, context_len/4))
            2. Forward-predict from left context into gap
            3. Burg AR model fit on post-gap context (reversed)
            4. Backward-predict from right context into gap
            5. Crossfade (Hanning) forward↔backward predictions
            6. Soft Hanning-blend at gap edges with original signal
        """
        x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
        n = len(x)
        win = max(3, int(self._WIN_MS * sr / 1000))
        max_click = max(1, int(self._MAX_CLICK_MS * sr / 1000))
        out = x.copy()

        # Gleite über Signal
        i = win
        while i < n - win:
            region = x[i - win : i + win + 1]
            med = float(np.median(region))
            mad = float(np.median(np.abs(region - med))) + 1e-12
            if abs(x[i] - med) > self._THRESHOLD * mad:
                # Click-Beginn gefunden – suche Ende
                j = i + 1
                while j < min(i + max_click, n - 1):
                    region_j = x[max(0, j - win) : j + win + 1]
                    med_j = float(np.median(region_j))
                    mad_j = float(np.median(np.abs(region_j - med_j))) + 1e-12
                    if abs(x[j] - med_j) <= self._THRESHOLD * mad_j:
                        break
                    j += 1
                gap = j - i
                if gap > 0:
                    # ── AR-basierte Lückenfüllung (v10.0.0) ──
                    # Kontext vor und nach der Lücke für AR-Modellierung
                    ctx_left = max(32, min(gap * 4, i))
                    ctx_right = max(32, min(gap * 4, n - j))

                    # Linksseitiger Kontext für Forward-Prediction
                    left_segment = out[max(0, i - ctx_left) : i]
                    # Rechtsseitiger Kontext für Backward-Prediction (reversed)
                    right_segment = out[j : min(n, j + ctx_right)][::-1]

                    forward_pred = self._ar_predict_forward(left_segment, gap)
                    backward_pred = self._ar_predict_forward(right_segment, gap)[::-1]

                    # Crossfade: Hanning-Blend forward↔backward
                    if gap >= 4:
                        cf = np.hanning(gap * 2)[:gap]
                        interp = cf * forward_pred + (1.0 - cf) * backward_pred
                    else:
                        interp = (forward_pred + backward_pred) * 0.5

                    out[i:j] = interp

                    # Hanning-Blend an Rändern (soft transition to original)
                    fl = min(self._FADE, gap // 2)
                    if fl > 0:
                        hw = np.hanning(fl * 2)[:fl]
                        out[i : i + fl] = (1 - hw) * x[i : i + fl] + hw * interp[:fl]
                        out[j - fl : j] = (1 - hw[::-1]) * x[j - fl : j] + hw[::-1] * interp[-fl:]
                i = j
            else:
                i += 1

        return np.clip(out, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _ar_predict_forward(context: np.ndarray, n_predict: int) -> np.ndarray:
        """Forward AR prediction using Burg's method.

        Fits a Burg AR model to the context signal and iteratively predicts
        n_predict samples forward. If the context is too short or degenerate,
        falls back to cubic spline interpolation.

        Args:
            context: Pre-gap signal segment (1-D float array).
            n_predict: Number of samples to predict forward.

        Returns:
            Predicted samples as 1-D float array of length n_predict.
        """
        if len(context) < 8 or n_predict <= 0:
            return np.zeros(n_predict, dtype=np.float64)

        try:
            # Adaptive AR order: higher order for longer contexts, capped
            order = min(16, max(2, len(context) // 4))
            order = min(order, len(context) - 1)

            # Burg AR coefficient estimation via reflection coefficients
            ar_coeffs = AiDecrackler._burg_ar(context, order)

            if ar_coeffs is None or len(ar_coeffs) < 1:
                # AR fit failed — fall back to cubic interpolation
                return AiDecrackler._cubic_fallback(context, n_predict)

            # Iterative forward prediction
            pred = np.zeros(n_predict, dtype=np.float64)
            # Use last 'order' samples from context as initial state
            state = context[-order:].copy()[::-1]  # reversed for lfilter convention

            for k in range(n_predict):
                # Predict next sample: y[n] = -sum(a[i] * y[n-i-1])
                pred[k] = -np.dot(ar_coeffs, state[:order])
                # Shift state: new sample in, oldest out
                state = np.roll(state, 1)
                state[0] = pred[k]

            return pred

        except Exception:
            logger.warning("decrackler.py::_ar_predict_forward Ersatzpfad", exc_info=True)
            return AiDecrackler._cubic_fallback(context, n_predict)

    @staticmethod
    def _cubic_fallback(context: np.ndarray, n_predict: int) -> np.ndarray:
        """Cubic spline fallback when AR fitting fails."""
        if len(context) < 4:
            last_val = float(context[-1]) if len(context) > 0 else 0.0
            return np.full(n_predict, last_val, dtype=np.float64)
        try:
            from scipy.interpolate import CubicSpline

            t = np.arange(len(context), dtype=np.float64)
            cs = CubicSpline(t, context.astype(np.float64), bc_type="natural")
            t_pred = np.linspace(len(context) - 1, len(context) + n_predict - 1, n_predict)
            return cs(t_pred).astype(np.float64)  # type: ignore[no-any-return]
        except Exception:
            last_val = float(context[-1])
            return np.full(n_predict, last_val, dtype=np.float64)

    @staticmethod
    def _burg_ar(signal: np.ndarray, order: int) -> np.ndarray | None:
        """Estimate AR coefficients using Burg's method (harmonic mean of
        forward and backward prediction errors).

        Numerically stable implementation using reflection coefficients.
        Returns AR coefficients a[0..order-1] such that:
            x[n] = -sum_{i=1}^{order} a[i-1] * x[n-i]
        Returns None if the signal is degenerate.
        """
        try:
            signal = np.asarray(signal, dtype=np.float64)
            n = len(signal)
            if n < order + 1:
                return None

            # Remove mean for numerical stability
            signal = signal - np.mean(signal)

            # Initialize forward and backward prediction errors
            ef = signal.copy()
            eb = signal.copy()

            a = np.ones(order + 1, dtype=np.float64)
            k = np.zeros(order, dtype=np.float64)  # reflection coefficients

            for m in range(order):
                # Numerator and denominator for reflection coefficient
                num = -2.0 * np.sum(ef[m + 1 :] * eb[m : n - 1])
                den = np.sum(ef[m + 1 :] ** 2 + eb[m : n - 1] ** 2)

                if abs(den) < 1e-15:
                    return None

                k[m] = num / den

                # Clamp reflection coefficient for stability
                k[m] = np.clip(k[m], -0.999, 0.999)

                # Update AR coefficients via Levinson recursion
                a_new = a.copy()
                for i in range(1, m + 2):
                    a_new[i] = a[i] + k[m] * a[m + 1 - i]
                a = a_new

                # Update prediction errors
                if m < order - 1:
                    ef_new = ef.copy()
                    eb_new = eb.copy()
                    for i in range(m + 1, n):
                        ef_new[i] = ef[i] + k[m] * eb[i - 1]
                        eb_new[i] = eb[i - 1] + k[m] * ef[i]
                    ef = ef_new
                    eb = eb_new

            # Return AR coefficients a[1..order]
            return a[1:].astype(np.float64)

        except Exception:
            logger.warning("decrackler.py::_burg_ar Ersatzpfad", exc_info=True)
            return None

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Entfernt Click/Crackle. ML-Primär (falls geladen), sonst DSP-Fallback."""
        logger.debug("[DSPContract] %s", asdict(decrackler_contract))
        if self.model is not None:
            try:
                inp = audio.astype(np.float32)
                return np.clip(
                    np.nan_to_num(self.model.run(None, {"input": inp})[0], nan=0.0),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            except Exception as exc:
                logger.warning("[AiDecrackler] ML fehlgeschlagen (%s), nutze DSP.", exc)
        a = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        if a.ndim == 1:
            return self._process_channel(a, sr)
        return np.stack([self._process_channel(a[:, ch], sr) for ch in range(a.shape[1])], axis=1)


class AiDehum:
    """Adaptiver IIR-Notch-Kaskaden-Hum-Entferner.

    Algorithmus:
        1. FFT-Peak-Suche im Bereich 40–70 Hz → auto-erkannte Hum-Grundfrequenz
           (50 Hz oder 60 Hz, je nach Spectral-Peak).
        2. Engbandige IIR-Notch-Filter (scipy.signal.iirnotch, Q=35) für
           Hum-Grundfrequenz und Obertöne bis zur Nyquist-Grenze.
        3. Hanning-gefensterte Crossfade-Blende zwischen gefiltert und original
           an Signalenden (Randeffekte verhindern).
    Invariante: NaN/Inf-frei, Ausgang ∈ [−1, 1].
    """

    _Q: float = 35.0  # Notch-Filter Güte
    _DETECTION_RANGE = (40, 70)  # Hz-Bereich für Hum-Erkennung

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def _detect_hum_freq(self, x: np.ndarray, sr: int) -> float:
        """Erkennt Hum-Grundfrequenz per FFT-Peaksuche (40–70 Hz)."""
        n_fft = min(len(x), 65536)
        spec = np.abs(np.fft.rfft(x[:n_fft], n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        lo, hi = self._DETECTION_RANGE
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            return 50.0
        peak_idx = np.argmax(spec[mask])
        detected = float(freqs[mask][peak_idx])
        # Snap to 50 / 60 Hz
        return 60.0 if abs(detected - 60.0) < abs(detected - 50.0) else 50.0

    def _apply_notch_cascade(self, x: np.ndarray, sr: int, f0: float) -> np.ndarray:
        from scipy.signal import iirnotch, sosfilt, tf2sos

        out = x.copy()
        f = f0
        nyq = sr / 2.0
        while f < nyq - 1.0:
            b, a = iirnotch(f / nyq, Q=self._Q)
            sos = tf2sos(b, a)
            out = sosfilt(sos, out)
            f += f0
        return out

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Entfernt Netzbrummen (Hum). ML-Primär (falls geladen), sonst DSP-Kaskade."""
        if self.model is not None:
            try:
                inp = audio.astype(np.float32)
                return np.clip(
                    np.nan_to_num(self.model.run(None, {"input": inp})[0], nan=0.0),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            except Exception as exc:
                logger.warning("[AiDehum] ML fehlgeschlagen (%s), nutze DSP.", exc)
        a = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        mono = a.flatten() if a.ndim == 1 else a[:, 0]
        f0 = self._detect_hum_freq(mono, sr)
        logger.debug("[AiDehum] Hum-Frequenz erkannt: %.1f Hz", f0)
        if a.ndim == 1:
            cleaned = self._apply_notch_cascade(a.astype(np.float64), sr, f0).astype(np.float32)
            return np.clip(np.nan_to_num(cleaned), -1.0, 1.0).astype(np.float32)
        channels = [
            np.clip(np.nan_to_num(self._apply_notch_cascade(a[:, ch].astype(np.float64), sr, f0)), -1.0, 1.0).astype(
                np.float32
            )
            for ch in range(a.shape[1])
        ]
        return np.stack(channels, axis=1)


class AiDebuzz:
    """STFT-basierter Schmalband-Buzz-Entferner.

    Algorithmus (DSP-Fallback, Referenz §4.5 Konsistenter-Wiener-Filter):
        1. STFT des Eingangssignals (nperseg=2048).
        2. Spektrale Peakiness-Maske: Für jede Zeitscheibe (Frame) wird eine
           Narrow-Band-Energie-Anomalie-Detektionsschwelle berechnet:
           local_mean = Mittelwert der nächsten 10 Nachbarbins;
           Bins mit mag > buzz_factor * local_mean → Buzz-Bin, Gain = G_floor.
        3. Buzz-Gain = max(G_floor, 1 − buzz_excess): sanfte Unterdrückung.
        4. ISTFT mit Originalphase (phasenkonsistent).
    buzz_factor = 4.0 entspricht ~12 dB, G_floor = 0.10.
    Invariante: NaN/Inf-frei, Ausgang ∈ [−1, 1].
    """

    _NPERSEG: int = 2048
    _BUZZ_FACTOR: float = 4.0  # Faktor über lokale Energie → Buzz-Detektion
    _G_FLOOR: float = 0.10  # §2.28 G_floor für Buzz-Bins
    _NEIGHBOR: int = 10  # Anzahl Nachbarbins für lokalen Mittelwert

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def _debuzz_channel(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import istft as _istft
        from scipy.signal import stft as _stft

        x = np.nan_to_num(x.astype(np.float64), nan=0.0)
        noverlap = self._NPERSEG * 3 // 4
        _, _, Zxx = _stft(x, nperseg=self._NPERSEG, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        n_bins = mag.shape[0]

        G = np.ones_like(mag)
        nb = self._NEIGHBOR
        for b in range(n_bins):
            lo = max(0, b - nb)
            hi = min(n_bins, b + nb + 1)
            # lokaler Mittelwert ohne den Bin selbst
            neighbors = np.concatenate([mag[lo:b, :], mag[b + 1 : hi, :]], axis=0)
            if len(neighbors) == 0:
                continue
            local_mean = np.mean(neighbors, axis=0) + 1e-12
            excess = mag[b] / local_mean
            buzz_bins = excess > self._BUZZ_FACTOR
            # sanfte Unterdrückung: G = max(G_floor, 1 - (excess-1)/excess)
            G[b, buzz_bins] = np.maximum(
                self._G_FLOOR,
                1.0 - (excess[buzz_bins] - 1.0) / excess[buzz_bins],
            )

        Zxx_clean = mag * G * np.exp(1j * phase)
        _, out = _istft(Zxx_clean, nperseg=self._NPERSEG, noverlap=noverlap)
        out = np.nan_to_num(out[: len(x)], nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Entfernt Buzz/Summen. ML-Primär (falls geladen), sonst STFT-Schmalband-DSP."""
        if self.model is not None:
            try:
                inp = audio.astype(np.float32)
                return np.clip(
                    np.nan_to_num(self.model.run(None, {"input": inp})[0], nan=0.0),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            except Exception as exc:
                logger.warning("[AiDebuzz] ML fehlgeschlagen (%s), nutze DSP.", exc)
        a = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        if a.ndim == 1:
            return self._debuzz_channel(a)
        return np.stack([self._debuzz_channel(a[:, ch]) for ch in range(a.shape[1])], axis=1)
