"""natural_performance_detector.py — §2.46f Natural-Performance-Artifacts-Guard (Aurik 10.0.0)

Erkennt und schützt performancebedingte Klangereignisse vor ungewollter Entfernung.
Drei Kategorien (§2.46f):
  1. Atemgeräusche zwischen Phrasen
  2. Natürliches Vibrato / Portamento
  3. Recording-Chain-Early-Reflections

Singleton: get_natural_performance_detector()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NaturalPerformanceResult:
    """Ergebnis der Natural-Performance-Erkennung."""

    breath_zones: list[tuple[float, float]] = field(default_factory=list)  # (start_s, end_s)
    vibrato_zones: list[tuple[float, float]] = field(default_factory=list)
    early_reflection_zones: list[tuple[float, float]] = field(default_factory=list)

    def get_protected_mask(self, n_samples: int, sr: int) -> np.ndarray:
        """Bool-Maske: True = geschütztes Frame, darf nicht durch NR entfernt werden."""
        mask = np.zeros(n_samples, dtype=bool)
        for start_s, end_s in self.breath_zones + self.vibrato_zones + self.early_reflection_zones:
            i0 = max(0, int(start_s * sr))
            i1 = min(n_samples, int(np.ceil(end_s * sr)))
            if i0 < i1:
                mask[i0:i1] = True
        return mask  # type: ignore[no-any-return]

    @property
    def has_early_reflections(self) -> bool:
        """True wenn Early Reflections erkannt — Dereverb wet_mix cap auf 0.35 setzen (§2.46f/§4.5c)."""
        return len(self.early_reflection_zones) > 0


class NaturalPerformanceDetector:
    """§2.46f Natural-Performance-Artifacts-Guard.

    Erkennt drei Kategorien von Klangereignissen, die NICHT als Defekte behandelt werden dürfen.
    """

    # ── Atem-Parameter §2.46f ─────────────────────────────────────────────
    _BREATH_ENERGY_MIN_DBFS: float = -55.0
    _BREATH_ENERGY_MAX_DBFS: float = -40.0
    _BREATH_FLATNESS_MIN: float = 0.40
    _BREATH_MIN_DURATION_S: float = 0.050
    _BREATH_MAX_DURATION_S: float = 0.500

    # ── Vibrato-Parameter §2.46f ──────────────────────────────────────────
    _VIBRATO_RATE_MIN_HZ: float = 4.0
    _VIBRATO_RATE_MAX_HZ: float = 7.0
    _VIBRATO_MAX_CENTS: float = 50.0

    # ── Early-Reflection-Parameter §2.46f ─────────────────────────────────
    _ER_WINDOW_S: float = 0.050  # 0–50 ms nach Onset
    _ER_C80_MIN_DB: float = 3.0

    def detect(self, audio: np.ndarray, sr: int) -> NaturalPerformanceResult:
        """Erkennt alle drei Kategorien natürlicher Performance-Artefakte.

        Args:
            audio: float32 ndarray, mono (N,) oder stereo (N, 2), 48 kHz.
            sr: Sample-Rate (muss 48000 sein).

        Returns:
            NaturalPerformanceResult mit allen geschützten Zonen.
        """
        result = NaturalPerformanceResult()
        try:
            mono = np.asarray(audio, dtype=np.float32)
            mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
            if mono.ndim == 2:
                mono = mono.mean(axis=1)

            result.breath_zones = self._detect_breath_zones(mono, sr)
            result.vibrato_zones = self._detect_vibrato_zones(mono, sr)
            result.early_reflection_zones = self._detect_early_reflection_zones(mono, sr)

            logger.debug(
                "§2.46f NPA: breaths=%d vibrato=%d er=%d",
                len(result.breath_zones),
                len(result.vibrato_zones),
                len(result.early_reflection_zones),
            )
        except Exception as exc:
            logger.debug("§2.46f NaturalPerformanceDetector.erkennen fehlgeschlagen (nicht blockierend): %s", exc)
        return result

    def _detect_breath_zones(self, mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """§2.46f: Atemgeräusche zwischen Phrasen.

        Kriterien:
        - Energie −55 bis −40 dBFS
        - Dauer 50–500 ms
        - spectral_flatness > 0.40
        """
        frame_len = max(1, int(0.020 * sr))  # 20 ms frames
        hop = frame_len // 2
        n = len(mono)
        zones: list[tuple[float, float]] = []

        frame_starts = list(range(0, n - frame_len + 1, hop))
        if not frame_starts:
            return zones

        # Batch-Verarbeitung aller Frames
        n_frames = len(frame_starts)
        energies_db = np.full(n_frames, -120.0)
        flatness = np.zeros(n_frames)

        for i, start in enumerate(frame_starts):
            seg = mono[start : start + frame_len]
            rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
            energies_db[i] = 20.0 * np.log10(max(rms, 1e-12))

            # Spectral flatness: geometric_mean / arithmetic_mean of power spectrum
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            power = spec**2 + 1e-12
            gm = float(np.exp(np.mean(np.log(power))))
            am = float(np.mean(power))
            flatness[i] = float(np.clip(gm / (am + 1e-12), 0.0, 1.0))

        # Frame-Kandidaten: Energie in [-55, -40] UND flatness > 0.40
        candidates = (
            (energies_db >= self._BREATH_ENERGY_MIN_DBFS)
            & (energies_db <= self._BREATH_ENERGY_MAX_DBFS)
            & (flatness >= self._BREATH_FLATNESS_MIN)
        )

        # Zusammenhängende Kandidaten-Blöcke → Zonen
        zones = self._merge_frame_runs(
            candidates,
            frame_starts,
            frame_len,
            hop,
            sr,
            self._BREATH_MIN_DURATION_S,
            self._BREATH_MAX_DURATION_S,
        )
        return zones

    def _detect_vibrato_zones(self, mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """§2.46f: Natürliches Vibrato / Portamento.

        Kriterien:
        - F0-Modulation 4–7 Hz
        - Amplitude ≤ ±50 Cent
        """
        zones: list[tuple[float, float]] = []
        # Nur bei ausreichender Länge sinnvoll (mind. 500 ms)
        if len(mono) < int(0.5 * sr):
            return zones

        try:
            # F0-Extraktion via Autokorrelation (20 ms Frames, 10 ms Hop)
            frame_len = max(1, int(0.020 * sr))
            hop = max(1, int(0.010 * sr))
            f0_values: list[float] = []
            frame_times: list[float] = []

            for start in range(0, len(mono) - frame_len + 1, hop):
                seg = mono[start : start + frame_len].astype(np.float64)
                f0 = self._autocorr_f0(seg, sr, f_lo=60.0, f_hi=1200.0)
                f0_values.append(f0)
                frame_times.append(float(start) / float(sr))

            if len(f0_values) < 10:
                return zones

            f0_arr = np.array(f0_values, dtype=np.float64)
            # Nur Voiced-Frames
            voiced = f0_arr > 80.0
            if float(np.mean(voiced)) < 0.3:
                return zones

            # Modulation Rate & Amplitude im Voiced-Segment
            # Sliding-Window (500 ms) → FFT der F0-Kurve → Prüfe ob Energie im [4,7] Hz Band dominiert
            win_s = 0.500
            win_n = max(10, int(win_s / 0.010))
            for i in range(0, len(f0_arr) - win_n + 1, win_n // 2):
                seg_f0 = f0_arr[i : i + win_n]
                voiced_seg = voiced[i : i + win_n]
                if float(np.mean(voiced_seg)) < 0.5:
                    continue

                # Interpoliere unvoiced Frames
                x_full = np.arange(len(seg_f0))
                x_voiced = x_full[voiced_seg]
                if len(x_voiced) < 4:
                    continue
                seg_interp = np.interp(x_full, x_voiced, seg_f0[voiced_seg])

                # F0-Variation in Cent
                mean_f0 = float(np.mean(seg_interp))
                if mean_f0 < 80.0:
                    continue
                cents = 1200.0 * np.log2(np.clip(seg_interp / mean_f0, 0.01, 10.0))

                # FFT der Cent-Kurve (10 ms hop → 100 Hz Abtastrate)
                N = len(cents)
                fft_mag = np.abs(np.fft.rfft(cents - cents.mean()))
                freqs = np.fft.rfftfreq(N, d=0.010)  # Hz

                # Energie in Vibrato-Band [4, 7 Hz]
                vib_mask = (freqs >= self._VIBRATO_RATE_MIN_HZ) & (freqs <= self._VIBRATO_RATE_MAX_HZ)
                total_energy = float(np.sum(fft_mag**2)) + 1e-12
                vib_energy = float(np.sum(fft_mag[vib_mask] ** 2))

                # Amplitude ≤ ±50 Cent
                amplitude_cents = float(np.std(cents)) * 2.0  # ±σ approximation

                if (vib_energy / total_energy) > 0.30 and amplitude_cents <= self._VIBRATO_MAX_CENTS * 2.0:
                    t_start = frame_times[i]
                    t_end = min(
                        float(len(mono)) / float(sr),
                        frame_times[min(i + win_n - 1, len(frame_times) - 1)] + win_s,
                    )
                    zones.append((t_start, t_end))

            zones = self._merge_overlapping(zones)
        except Exception as exc:
            logger.debug("§2.46f vibrato detection fehlgeschlagen (nicht blockierend): %s", exc)

        return zones

    def _detect_early_reflection_zones(self, mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """§2.46f: Recording-Chain-Early-Reflections (0–50 ms nach Onset).

        Criterion: C80-Proxy > 3 dB in der Onset-Region.
        Erkennt Onsets via Spektralfluss, schützt +50 ms danach.
        """
        zones: list[tuple[float, float]] = []
        try:
            frame_len = max(1, int(0.020 * sr))
            hop = max(1, int(0.010 * sr))

            # Spektralfluss für Onset-Erkennung
            prev_mag = None
            flux_vals: list[float] = []
            frame_starts_list: list[int] = []

            for start in range(0, len(mono) - frame_len + 1, hop):
                seg = mono[start : start + frame_len]
                mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                if prev_mag is not None:
                    diff = np.maximum(0.0, mag - prev_mag)
                    flux_vals.append(float(np.sum(diff)))
                else:
                    flux_vals.append(0.0)
                frame_starts_list.append(start)
                prev_mag = mag

            if not flux_vals:
                return zones

            flux_arr = np.array(flux_vals)
            threshold = float(np.mean(flux_arr) + 1.5 * np.std(flux_arr))
            er_window_n = max(1, int(self._ER_WINDOW_S * sr))  # samples in 50 ms

            # Für jeden Onset: C80-Proxy prüfen
            for i, (start_samp, flux) in enumerate(zip(frame_starts_list, flux_vals)):
                if flux < threshold:
                    continue
                # 50 ms Fenster nach Onset
                onset_end = min(len(mono), start_samp + er_window_n)
                if onset_end - start_samp < 64:
                    continue

                direct_seg = mono[start_samp : start_samp + min(frame_len, er_window_n)]
                er_seg = mono[start_samp + frame_len // 2 : onset_end]

                if len(direct_seg) < 32 or len(er_seg) < 32:
                    continue

                e_direct = float(np.mean(direct_seg**2) + 1e-12)
                e_er = float(np.mean(er_seg**2) + 1e-12)
                c80_proxy_db = 10.0 * np.log10(e_direct / e_er)

                if c80_proxy_db > self._ER_C80_MIN_DB:
                    t_start = float(start_samp) / float(sr)
                    t_end = min(float(len(mono)) / float(sr), t_start + self._ER_WINDOW_S)
                    zones.append((t_start, t_end))

            zones = self._merge_overlapping(zones)
        except Exception as exc:
            logger.debug("§2.46f early reflection detection fehlgeschlagen (nicht blockierend): %s", exc)
        return zones

    @staticmethod
    def _autocorr_f0(seg: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
        """Autokorrelations-basierte F0-Schätzung für ein einzelnes Frame."""
        n = len(seg)
        if n < 64:
            return 0.0
        # Normalisierte Autokorrelation
        seg = seg - seg.mean()
        # FFT-basierte Autokorrelation (O(n log n), VERBOTEN: np.correlate mode='full' O(n²))
        from scipy.signal import fftconvolve

        ac_full = fftconvolve(seg, seg[::-1], mode="full")
        ac = ac_full[len(seg) - 1 :]
        ac_norm = ac / (ac[0] + 1e-12)

        lag_min = max(1, int(sr / f_hi))
        lag_max = min(n - 1, int(sr / f_lo))
        if lag_min >= lag_max:
            return 0.0

        peak_idx = int(np.argmax(ac_norm[lag_min:lag_max]) + lag_min)
        if ac_norm[peak_idx] > 0.25:
            return float(sr) / float(peak_idx)
        return 0.0

    @staticmethod
    def _merge_frame_runs(
        mask: np.ndarray,
        frame_starts: list[int],
        frame_len: int,
        hop: int,
        sr: int,
        min_dur: float,
        max_dur: float,
    ) -> list[tuple[float, float]]:
        """Zusammenhängende Frames zu Zonen zusammenführen, Min/Max-Dauer filtern."""
        zones: list[tuple[float, float]] = []
        in_zone = False
        zone_start = 0
        for i, active in enumerate(mask):
            if active and not in_zone:
                in_zone = True
                zone_start = frame_starts[i]
            elif not active and in_zone:
                in_zone = False
                zone_end = frame_starts[i - 1] + frame_len
                dur = float(zone_end - zone_start) / float(sr)
                if min_dur <= dur <= max_dur:
                    zones.append((float(zone_start) / float(sr), float(zone_end) / float(sr)))
        if in_zone:
            zone_end = frame_starts[-1] + frame_len
            dur = float(zone_end - zone_start) / float(sr)
            if min_dur <= dur <= max_dur:
                zones.append((float(zone_start) / float(sr), float(zone_end) / float(sr)))
        return zones

    @staticmethod
    def _merge_overlapping(zones: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Überlappende Zonen zusammenführen."""
        if not zones:
            return []
        zones = sorted(zones)
        merged: list[tuple[float, float]] = [zones[0]]
        for start, end in zones[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged


_npa_instance: NaturalPerformanceDetector | None = None
_npa_lock = threading.Lock()


def get_natural_performance_detector() -> NaturalPerformanceDetector:
    """Thread-safe Singleton (§3.x spec pattern)."""
    global _npa_instance
    if _npa_instance is None:
        with _npa_lock:
            if _npa_instance is None:
                _npa_instance = NaturalPerformanceDetector()
    return _npa_instance


__all__ = [
    "NaturalPerformanceDetector",
    "NaturalPerformanceResult",
    "get_natural_performance_detector",
]
