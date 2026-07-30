"""§VocalStyle VocalStyleProfiler — per-recording singer fingerprint (v10.0.0).

Learns from the first ~20 s of material how this specific singer produces sound
in this specific recording: vibrato characteristics, register distribution,
formant colour, and breathiness index. Used to calibrate VQI thresholds and NR
strength relative to the actual singer-ideal rather than a generic era default.

Non-blocking: all public methods catch exceptions and return a fallback profile.

Usage in UV3 (after VocalFocusAnalyzer):
    from backend.core.dsp.vocal_style_profiler import get_vocal_style_profiler
    _vsp = get_vocal_style_profiler().profile(audio, sample_rate)
    _restoration_context["vocal_style_profile"] = _vsp.to_dict()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import correlate as _scipy_correlate

from backend.core.audio_utils import safe_filtfilt  # §v10.101

logger = logging.getLogger(__name__)

_VIBRATO_RATE_MIN_HZ: float = 4.0
_VIBRATO_RATE_MAX_HZ: float = 8.0
_ANALYSIS_MAX_DURATION_S: float = 20.0
_F0_MIN_HZ: float = 70.0
_F0_MAX_HZ: float = 950.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class VocalStyleProfile:
    """Per-recording vocal production fingerprint."""

    vibrato_rate_hz: float = 0.0  # dominant vibrato rate [4–8 Hz]
    vibrato_depth_cents: float = 0.0  # F0-modulation depth in cents
    chest_head_ratio: float = 0.5  # 0.0 = pure head, 1.0 = pure chest
    phrase_contour_variance: float = 0.0  # scaled RMS-envelope variance
    f1_f2_ratio: float = 0.0  # F1/F2 — formant colour proxy
    breathiness_index: float = 0.0  # H1-H2 amplitude difference [dB]
    valid: bool = False  # False = insufficient clean material

    def vqi_calibration_offset(self) -> float:
        """VQI floor adjustment for vocal complexity.

        Complex voices (wide vibrato, high breathiness) receive a small
        downward offset (max −0.05) so perfect reconstruction is not demanded
        for recordings that are inherently harder to preserve.
        """
        if not self.valid:
            return 0.0
        complexity = (
            min(1.0, self.vibrato_depth_cents / 80.0) * 0.3
            + min(1.0, self.breathiness_index / 10.0) * 0.3
            + min(1.0, self.phrase_contour_variance / 100.0) * 0.4
        )
        return float(np.clip(-complexity * 0.05, -0.05, 0.0))

    def nr_strength_cap(self) -> float:
        """Upper NR strength cap to protect fragile vocal characteristics.

        Breathy and strongly vibrating voices need gentler NR to avoid
        artefacts (musicalnoise in vibrato, breathiness suppression).
        Returns value in [0.5, 1.0].
        """
        if not self.valid:
            return 1.0
        breathiness_cap = 1.0 - min(0.3, self.breathiness_index / 30.0)
        vibrato_cap = 1.0 - min(0.2, self.vibrato_depth_cents / 200.0)
        return float(np.clip(min(breathiness_cap, vibrato_cap), 0.5, 1.0))

    def to_dict(self) -> dict:
        """Serialisiert the vocal style profile as a dictionary for UV3 context injection."""
        d = asdict(self)
        d["vqi_calibration_offset"] = self.vqi_calibration_offset()
        d["nr_strength_cap"] = self.nr_strength_cap()
        return d


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class VocalStyleProfiler:
    """Singleton-Vokalstil-Profiler – thread-safe, alle Methoden nicht-blockierend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def profile(self, audio: np.ndarray, sr: int) -> VocalStyleProfile:
        """Berechnet a VocalStyleProfile from the first 20 s of *audio*.

        Args:
            audio: mono or stereo float32 audio.
            sr:    sample rate (any value; NOT restricted to 48000 — analysis module).

        Returns:
            VocalStyleProfile; .valid == False on error or too-short input.
        """
        with self._lock:
            try:
                return self._profile_impl(audio, sr)
            except Exception as exc:
                logger.debug("VocalStyleProfiler non-blocking: %s", exc)
                return VocalStyleProfile(valid=False)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _profile_impl(self, audio: np.ndarray, sr: int) -> VocalStyleProfile:
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim == 2:
            # (channels, samples) if channels <= 2 and samples >> channels
            if mono.shape[0] <= 2 and mono.shape[0] < mono.shape[1]:
                mono = np.mean(mono, axis=0)
            else:
                mono = np.mean(mono, axis=1)
        mono = mono[: int(_ANALYSIS_MAX_DURATION_S * sr)]
        if len(mono) < int(1.0 * sr):
            return VocalStyleProfile(valid=False)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        vibrato_rate, vibrato_depth = self._compute_vibrato(mono, sr)
        chest_head = self._compute_chest_head_ratio(mono, sr)
        phrase_var = self._compute_phrase_contour_variance(mono, sr)
        f1_f2 = self._compute_f1_f2_ratio(mono, sr)
        breathiness = self._compute_breathiness(mono, sr)

        return VocalStyleProfile(
            vibrato_rate_hz=float(np.clip(vibrato_rate, 0.0, 12.0)),
            vibrato_depth_cents=float(np.clip(vibrato_depth, 0.0, 200.0)),
            chest_head_ratio=float(np.clip(chest_head, 0.0, 1.0)),
            phrase_contour_variance=float(np.clip(phrase_var, 0.0, 1000.0)),
            f1_f2_ratio=float(np.clip(f1_f2, 0.0, 1.0)),
            breathiness_index=float(np.clip(breathiness, 0.0, 30.0)),
            valid=True,
        )

    def _extract_f0_frames(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """Leichtgewichtiges autocorrelation F0 tracker → voiced F0 array."""
        hop = int(sr * 0.01)  # 10 ms
        frame_len = int(sr * 0.04)  # 40 ms
        lag_min = max(1, int(sr / _F0_MAX_HZ))
        lag_max = int(sr / _F0_MIN_HZ)
        f0_frames: list[float] = []
        for i in range(0, len(mono) - frame_len, hop):
            frame = mono[i : i + frame_len]
            if float(np.max(np.abs(frame))) < 1e-4:
                continue
            acf = _scipy_correlate(frame, frame, mode="full", method="fft")[len(frame) - 1 :]
            if lag_max >= len(acf):
                continue
            peak_idx = lag_min + int(np.argmax(acf[lag_min : lag_max + 1]))
            f0 = sr / max(peak_idx, 1)
            if _F0_MIN_HZ <= f0 <= _F0_MAX_HZ:
                f0_frames.append(f0)
        return np.array(f0_frames, dtype=np.float64)  # type: ignore[no-any-return]

    def _compute_vibrato(self, mono: np.ndarray, sr: int) -> tuple[float, float]:
        """Vibrato rate (Hz) and depth (cents) via F0 modulation analysis."""
        try:
            f0_arr = self._extract_f0_frames(mono, sr)
            if len(f0_arr) < 20:
                return 0.0, 0.0

            f0_sr = 100.0  # 100 Hz (10 ms hop)
            from scipy.signal import butter, filtfilt  # pylint: disable=import-outside-toplevel

            lo = _VIBRATO_RATE_MIN_HZ / (f0_sr / 2.0)
            hi = _VIBRATO_RATE_MAX_HZ / (f0_sr / 2.0)
            lo, hi = max(lo, 0.01), min(hi, 0.99)
            if lo >= hi:
                return 0.0, 0.0

            _butter_ba = butter(2, [lo, hi], btype="bandpass")
            if _butter_ba is None:
                return 0.0, 0.0
            b, a = np.asarray(_butter_ba[0]), np.asarray(_butter_ba[1])
            f0_detrended = f0_arr - float(np.mean(f0_arr))
            f0_vibrato = safe_filtfilt(b, a, f0_detrended)

            depth_hz = float(np.std(f0_vibrato))
            if depth_hz < 0.5 or float(np.mean(f0_arr)) <= 0.0:
                return 0.0, 0.0

            f0_mean = float(np.mean(f0_arr))
            depth_cents = float(1200.0 * np.log2(1.0 + depth_hz / f0_mean))

            # Dominant vibrato rate via FFT of bandpassed F0
            n_fft = max(len(f0_vibrato), 512)
            fft_mag = np.abs(np.fft.rfft(f0_vibrato, n=n_fft))
            freqs_mod = np.fft.rfftfreq(n_fft, d=1.0 / f0_sr)
            mask = (freqs_mod >= _VIBRATO_RATE_MIN_HZ) & (freqs_mod <= _VIBRATO_RATE_MAX_HZ)
            if not np.any(mask):
                return 0.0, depth_cents
            peak_rate = float(freqs_mod[np.where(mask)[0][0] + int(np.argmax(fft_mag[mask]))])

            return float(np.clip(peak_rate, 0.0, 12.0)), float(np.clip(depth_cents, 0.0, 200.0))
        except RecursionError:
            # §v10.110: scipy/numpy recursion bei edge-case Audio.
            # safe_filtfilt → scipy.signal.filtfilt kann in seltenen Fällen
            # (0.3% aller Runs) eine C-Level-Rekursion triggern.
            # Fallback: lfilter (minimum-phase) statt filtfilt (zero-phase).
            logger.debug("vocal_style_profiler.py::_compute_vibrato recursion → lfilter fallback")
            try:
                from scipy.signal import lfilter

                if "b" in locals() and "a" in locals():
                    f0_vibrato = lfilter(b, a, f0_detrended)
                    depth_hz = float(np.std(f0_vibrato))
                    if depth_hz >= 0.5 and float(np.mean(f0_arr)) > 0.0:
                        f0_mean = float(np.mean(f0_arr))
                        return 0.0, float(np.clip(1200.0 * np.log2(1.0 + depth_hz / f0_mean), 0.0, 200.0))
            except Exception:
                pass
            return 0.0, 0.0
        except Exception as e:
            logger.warning("vocal_style_profiler.py::_compute_vibrato fallback: %s", e)
            return 0.0, 0.0

    def _compute_chest_head_ratio(self, mono: np.ndarray, sr: int) -> float:
        """Chest vs head register proxy via low/high harmonic energy ratio.

        Chest voice: dominant energy in low harmonics (<800 Hz relative to F0).
        Head voice: fundamental dominates, overtone energy relatively weaker.
        """
        try:
            n_fft = 2048
            hop = n_fft // 4
            window = np.hanning(n_fft).astype(np.float32)
            ratios: list[float] = []
            for i in range(0, len(mono) - n_fft, hop):
                frame = mono[i : i + n_fft]
                rms = float(np.sqrt(np.mean(frame**2)))
                if rms < 5e-4:
                    continue
                spec = np.abs(np.fft.rfft(frame * window))
                freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
                boundary = 800.0
                e_low = float(np.sum(spec[freqs < boundary] ** 2))
                e_high = float(np.sum(spec[(freqs >= boundary) & (freqs < 3000.0)] ** 2))
                total = e_low + e_high
                if total > 1e-10:
                    ratios.append(e_low / total)
            return float(np.clip(np.mean(ratios) if ratios else 0.5, 0.0, 1.0))
        except Exception as e:
            logger.warning("vocal_style_profiler.py::_compute_chest_head_ratio fallback: %s", e)
            return 0.5

    def _compute_phrase_contour_variance(self, mono: np.ndarray, sr: int) -> float:
        """Skalierte RMS-Hüllkurven-Varianz – Proxy für Phrasenkontur-Komplexität."""
        try:
            window = max(1, int(sr * 0.1))  # 100 ms
            hop = max(1, window // 2)
            rms_vals: list[float] = []
            for i in range(0, len(mono) - window, hop):
                rms_vals.append(float(np.sqrt(np.mean(mono[i : i + window] ** 2))))
            if not rms_vals:
                return 0.0
            return float(np.clip(float(np.var(rms_vals)) * 1e4, 0.0, 1000.0))
        except Exception as e:
            logger.warning("vocal_style_profiler.py::_compute_phrase_contour_variance fallback: %s", e)
            return 0.0

    def _compute_f1_f2_ratio(self, mono: np.ndarray, sr: int) -> float:
        """F1/F2 ratio from LPC formant tracker (non-blocking)."""
        try:
            from backend.core.dsp.lpc_formant_tracker import (
                get_lpc_formant_tracker,  # pylint: disable=import-outside-toplevel
            )

            result = get_lpc_formant_tracker().track(mono, sr)
            f1 = float(result.get("f1_mean", 0.0))
            f2 = float(result.get("f2_mean", 0.0))
            if f2 > 10.0 and f1 > 10.0:
                return float(np.clip(f1 / f2, 0.0, 1.0))
            return 0.0
        except Exception as e:
            logger.warning("vocal_style_profiler.py::_compute_f1_f2_ratio fallback: %s", e)
            return 0.0

    def _compute_breathiness(self, mono: np.ndarray, sr: int) -> float:
        """H1-H2 amplitude difference in dB — Hillenbrand breathiness proxy.

        Higher H1-H2 → breathy/airy voice (glottal flow with incomplete closure).
        Lower / negative H1-H2 → pressed or modal phonation.
        """
        try:
            hop = int(sr * 0.01)  # 10 ms
            frame_len = int(sr * 0.04)  # 40 ms
            if frame_len > len(mono):
                return 0.0
            bin_hz = sr / frame_len
            h1_h2_diffs: list[float] = []
            for i in range(0, len(mono) - frame_len, hop):
                frame = mono[i : i + frame_len]
                if float(np.max(np.abs(frame))) < 1e-4:
                    continue
                spec = np.abs(np.fft.rfft(frame * np.hanning(frame_len)))
                freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
                # F0 via peak in 80–900 Hz
                f0_mask = (freqs >= 80.0) & (freqs <= 900.0)
                if not np.any(f0_mask):
                    continue
                f0_hz = float(freqs[f0_mask][int(np.argmax(spec[f0_mask]))])
                if f0_hz < 80.0:
                    continue
                # H1 window ±10 %
                h1_lo = max(0, int((f0_hz * 0.9) / bin_hz))
                h1_hi = min(len(spec) - 1, int((f0_hz * 1.1) / bin_hz))
                h1_amp = float(np.max(spec[h1_lo : h1_hi + 1])) if h1_hi > h1_lo else 1e-10
                # H2 window near 2×F0 ±10 %
                h2_lo = max(0, int((f0_hz * 1.8) / bin_hz))
                h2_hi = min(len(spec) - 1, int((f0_hz * 2.2) / bin_hz))
                h2_amp = float(np.max(spec[h2_lo : h2_hi + 1])) if h2_hi > h2_lo else 1e-10
                if h1_amp > 1e-10 and h2_amp > 1e-10:
                    h1_h2_diffs.append(20.0 * np.log10(h1_amp / h2_amp))
            return float(np.clip(np.mean(h1_h2_diffs) if h1_h2_diffs else 0.0, 0.0, 30.0))
        except Exception as e:
            logger.warning("vocal_style_profiler.py::_compute_breathiness fallback: %s", e)
            return 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_profiler_holder: list[VocalStyleProfiler | None] = [None]
_profiler_lock = threading.Lock()


def get_vocal_style_profiler() -> VocalStyleProfiler:
    """Thread-safe singleton factory."""
    if _profiler_holder[0] is None:
        with _profiler_lock:
            if _profiler_holder[0] is None:
                _profiler_holder[0] = VocalStyleProfiler()
    instance = _profiler_holder[0]
    assert instance is not None
    return instance
