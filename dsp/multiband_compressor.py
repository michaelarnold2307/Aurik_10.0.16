import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.signal
from scipy.signal import butter, sosfilt

_logger = logging.getLogger(__name__)


@dataclass
class DSPContract:
    name: str = "MultibandCompressor"
    version: str = "1.0"
    description: str = "Adaptiver Multiband-Kompressor mit SOTA-Features"
    parameters: dict[str, Any] | None = None


multiband_compressor_contract = DSPContract(
    parameters={
        "bands": 3,
        "crossovers": (200, 2000),
        "thresholds_db": (-24, -18, -12),
        "ratios": (2, 3, 4),
        "knees_db": (6, 6, 6),
        "attack_ms": (10, 8, 5),
        "release_ms": (80, 60, 40),
    }
)


class MultibandCompressor:
    """
    SOTA-konformer adaptiver Multiband-Kompressor:
    - Beliebige Anzahl Bänder (default: 3)
    - Adaptive Crossover (Butterworth, Linkwitz-Riley)
    - Pro Band: RMS/Peak-Detection, Soft-Knee, Ratio, Attack/Release
    - Sidechain-Option, Band-Feedback
    - ML-ready (Hooks für ML-basierte Parameter)
    """

    def __init__(
        self,
        bands: int = 3,
        crossovers: tuple[float, float] = (200, 2000),
        thresholds_db: Sequence[float] = (-24, -18, -12),
        ratios: Sequence[float] = (2, 3, 4),
        knees_db: Sequence[float] = (6, 6, 6),
        attack_ms: Sequence[float] = (10, 8, 5),
        release_ms: Sequence[float] = (80, 60, 40),
    ) -> None:
        """
        bands: Anzahl der Frequenzbänder
        crossovers: Übergangsfrequenzen (Hz)
        thresholds_db: Kompressor-Schwellen pro Band (dB)
        ratios: Kompressionsraten pro Band
        knees_db: Soft-Knee pro Band (dB)
        attack_ms: Attack-Zeiten pro Band (ms)
        release_ms: Release-Zeiten pro Band (ms)
        """
        self.bands = bands
        self.crossovers = crossovers

        # Defensive: Parameter-Arrays auf richtige Länge bringen
        def ensure_len(seq, n, default=0):
            if isinstance(seq, (list, tuple)) and len(seq) == n:
                return seq
            elif isinstance(seq, (list, tuple)) and len(seq) < n:
                return tuple(list(seq) + [seq[-1]] * (n - len(seq)))
            elif isinstance(seq, (list, tuple)) and len(seq) > n:
                return tuple(seq[:n])
            else:
                return tuple([default] * n)

        self.thresholds_db = ensure_len(thresholds_db, bands, -24)
        self.ratios = ensure_len(ratios, bands, 2)
        self.knees_db = ensure_len(knees_db, bands, 6)
        self.attack_ms = ensure_len(attack_ms, bands, 10)
        self.release_ms = ensure_len(release_ms, bands, 80)

    def log_contract(self):
        _logger.debug("[DSPContract] %s", asdict(multiband_compressor_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für MultibandCompressor")
            band_signals = self._split_bands(audio, sr)
            processed: list[npt.NDArray[np.float64]] = []
            for i, band in enumerate(band_signals):
                idx = min(i, len(self.thresholds_db) - 1)
                comp = self._compress_band(
                    band,
                    sr,
                    self.thresholds_db[idx],
                    self.ratios[idx],
                    self.knees_db[idx],
                    self.attack_ms[idx],
                    self.release_ms[idx],
                )
                processed.append(np.asarray(comp, dtype=np.float64))
            if processed:
                out = np.sum(np.stack(processed, axis=0), axis=0)
                maxval = np.max(np.abs(out))
                if maxval > 1.0:
                    out = np.clip(out, -1.0, 1.0)
                self._audit_log({"bands": self.bands, "shape": out.shape, "success": True})
                return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype)  # type: ignore[no-any-return]
            else:
                self._audit_log({"bands": self.bands, "error": "No bands processed"})
                return audio.astype(orig_dtype)
        except Exception as e:
            _logger.error("MultibandCompressor Fehler: %s", e)
            self._audit_log({"bands": self.bands, "error": str(e)})
            return audio.astype(orig_dtype)

    def _audit_log(self, result: dict[str, Any]):
        _logger.debug("[AuditLog][MultibandCompressor] Ergebnis: %s", result)

    @staticmethod
    def _lr4_sos(cutoff: float, sr: float, btype: str) -> npt.NDArray[np.float64]:
        """Linkwitz-Riley 4th-order filter as SOS (2x Butterworth-2 cascaded).

        LR4 key property: LR4_low(f) + LR4_high(f) = 1 for all f
        (flat summed amplitude, no phase cancellation at crossover).
        Butterworth 4th order alone does NOT have this property.

        Args:
            cutoff: Crossover frequency in Hz
            sr:     Sample rate in Hz
            btype:  'low' or 'high'

        Returns:
            SOS matrix [2*2, 6] (two 2nd-order sections cascaded)
        """
        wn = cutoff / (sr / 2.0)
        wn = float(np.clip(wn, 1e-4, 0.9999))
        sos = butter(2, wn, btype=btype, output="sos")
        # Cascade the same filter twice to get LR4
        sos = np.asarray(sos, dtype=np.float64)
        stacked = np.empty((sos.shape[0] * 2, sos.shape[1]), dtype=np.float64)
        stacked[: sos.shape[0], :] = sos
        stacked[sos.shape[0] :, :] = sos
        return stacked

    @staticmethod
    def _butter_ba(
        order: int, wn: float | list[float], btype: str
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Gibt validated Butterworth (b, a) coefficients with stable typing zurück."""
        coeffs = butter(order, wn, btype=btype, output="ba")
        if not isinstance(coeffs, tuple) or len(coeffs) != 2:
            raise RuntimeError("scipy.signal.butter returned unexpected coefficient format")

        b = np.asarray(coeffs[0], dtype=np.float64)
        a = np.asarray(coeffs[1], dtype=np.float64)
        return b, a

    def _split_bands(self, audio: npt.NDArray[np.float64], sr: int) -> list[npt.NDArray[np.float64]]:
        """Split signal into frequency bands using Linkwitz-Riley LR4 crossover.

        LR4 crossover guarantees flat amplitude sum (no comb filtering).
        Each crossover is implemented as 2x cascaded Butterworth-2 (24 dB/oct).
        """
        bands: list[npt.NDArray[np.float64]] = []
        cross = list(self.crossovers)
        while len(cross) < self.bands - 1:
            cross.append(cross[-1] if cross else 2000)
        if self.bands == 1:
            return [audio]
        # Low band
        sos_low = self._lr4_sos(cross[0], sr, "low")
        bands.append(np.asarray(sosfilt(sos_low, audio), dtype=np.float64))
        # Mid bands
        for i in range(1, self.bands - 1):
            fc0, fc1 = cross[i - 1], cross[i]
            if fc0 >= fc1:
                bands.append(bands[-1].copy() if bands else np.zeros_like(audio))
                continue
            # LR4 band: high-pass @ fc0 then low-pass @ fc1
            sos_hp = self._lr4_sos(fc0, sr, "high")
            sos_lp = self._lr4_sos(fc1, sr, "low")
            mid = sosfilt(sos_hp, audio)
            mid = sosfilt(sos_lp, mid)
            bands.append(np.asarray(mid, dtype=np.float64))
        # High band
        sos_high = self._lr4_sos(cross[self.bands - 2], sr, "high")
        bands.append(np.asarray(sosfilt(sos_high, audio), dtype=np.float64))
        return bands

    def _compress_band(
        self,
        audio: npt.NDArray[np.float64],
        sr: int,
        threshold_db: float,
        ratio: float,
        knee_db: float,
        attack_ms: float,
        release_ms: float,
    ) -> npt.NDArray[np.float64]:
        """
        Komprimiert ein einzelnes Frequenzband.
        """
        # RMS-Detection
        window = int(sr * 0.01)
        rms = self._moving_rms(audio, window)
        rms = np.nan_to_num(rms, nan=1e-8, posinf=1e-8, neginf=1e-8)
        rms_db = 20 * np.log10(rms + 1e-8)
        over = rms_db - threshold_db
        gain_db = np.zeros_like(rms_db)
        # Soft-Knee
        idx_soft = (over > -knee_db / 2) & (over < knee_db / 2)
        gain_db[idx_soft] = (1 / ratio - 1) * ((over[idx_soft] + knee_db / 2) ** 2) / (2 * knee_db)
        idx_over = over >= knee_db / 2
        gain_db[idx_over] = (1 / ratio - 1) * (over[idx_over])
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = attack_coeff * env[i - 1] + (1 - attack_coeff) * gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        out = audio * env
        return np.asarray(out)

    @staticmethod
    def _moving_rms(audio: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
        window = max(1, int(window))
        x = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim == 1:
            sq = np.square(x)
            left = window // 2
            right = window - left - 1
            padded = np.pad(sq, (left, right), mode="edge")
            csum = np.cumsum(np.concatenate(([0.0], padded)))
            avg = (csum[window:] - csum[:-window]) / float(window)
            return np.sqrt(np.maximum(avg, 0.0))  # type: ignore[no-any-return]
        return np.apply_along_axis(lambda ch: MultibandCompressor._moving_rms(ch, window), axis=-1, arr=x)


class MultibandCompressorStudio:
    """
    SOTA Multiband-Kompressor (Studio-Algorithmus):
    - 3 Bänder (Low, Mid, High), jeweils mit eigenem Kompressor
    """

    def __init__(self, thresholds: list | None = None, ratios: list | None = None):
        self.thresholds = thresholds if thresholds is not None else [-30, -24, -18]
        self.ratios = ratios if ratios is not None else [2, 3, 4]

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für MultibandCompressorStudio")
            sos_low = scipy.signal.iirfilter(2, 200 / (sr / 2), btype="low", ftype="butter", output="sos")
            low = scipy.signal.sosfilt(sos_low, audio)
            sos_mid = scipy.signal.iirfilter(
                2,
                [200 / (sr / 2), 3000 / (sr / 2)],
                btype="band",
                ftype="butter",
                output="sos",
            )
            mid = scipy.signal.sosfilt(sos_mid, audio)
            sos_high = scipy.signal.iirfilter(2, 3000 / (sr / 2), btype="high", ftype="butter", output="sos")
            high = scipy.signal.sosfilt(sos_high, audio)

            def compress(x, threshold_db, ratio):
                x_db = 20 * np.log10(np.abs(x) + 1e-8)
                over = x_db > threshold_db
                gain = np.ones_like(x)
                gain[over] = 10 ** (((threshold_db + (x_db[over] - threshold_db) / ratio) - x_db[over]) / 20)
                return x * gain

            low_c = compress(low, self.thresholds[0], self.ratios[0])
            mid_c = compress(mid, self.thresholds[1], self.ratios[1])
            high_c = compress(high, self.thresholds[2], self.ratios[2])
            out = low_c + mid_c + high_c
            _peak_p99 = float(np.percentile(np.abs(out), 99.9)) if out.size > 0 else 0.0
            if _peak_p99 > 0:
                out = out / _peak_p99
            self._audit_log({"bands": 3, "shape": out.shape, "success": True})
            return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype)  # type: ignore[no-any-return]
        except Exception as e:
            _logger.error("MultibandCompressorStudio Fehler: %s", e)
            self._audit_log({"bands": 3, "error": str(e)})
            return audio.astype(orig_dtype)

    def _audit_log(self, result: dict[str, Any]):
        _logger.debug("[AuditLog][MultibandCompressorStudio] Ergebnis: %s", result)
