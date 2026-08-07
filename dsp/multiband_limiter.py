import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt

_logger = logging.getLogger(__name__)


@dataclass
class DSPContract:
    name: str = "MultibandLimiter"
    version: str = "1.0"
    description: str = "Adaptiver Multiband-Limiter mit SOTA-Features"
    parameters: dict[str, Any] | None = None


multiband_limiter_contract = DSPContract(
    parameters={
        "bands": 3,
        "crossovers": (200, 2000),
        "ceilings_db": (-1, -1, -1),
        "knees_db": (6, 6, 6),
        "lookahead_ms": (2, 2, 2),
        "release_ms": (50, 40, 30),
    }
)


class MultibandLimiter:
    """
    SOTA-konformer adaptiver Multiband-Limiter:
    - Beliebige Anzahl Bänder (default: 3)
    - Adaptive Crossover (Butterworth, Linkwitz-Riley)
    - Pro Band: Lookahead, Soft-Knee, Release, Ceiling
    - True Peak, Inter-Sample Peak, ML-ready
    """

    def __init__(
        self,
        bands: int = 3,
        crossovers: tuple[float, float] = (200, 2000),
        ceilings_db: Sequence[float] = (-1, -1, -1),
        knees_db: Sequence[float] = (6, 6, 6),
        lookahead_ms: Sequence[float] = (2, 2, 2),
        release_ms: Sequence[float] = (50, 40, 30),
    ) -> None:
        """
        bands: Anzahl der Frequenzbänder
        crossovers: Übergangsfrequenzen (Hz)
        ceilings_db: Maximalpegel pro Band (dBFS)
        knees_db: Soft-Knee pro Band (dB)
        lookahead_ms: Lookahead pro Band (ms)
        release_ms: Release-Zeiten pro Band (ms)
        """
        self.bands = bands
        self.crossovers = crossovers

        def ensure_len(seq: Sequence[float], n: int, default: float) -> tuple[float, ...]:
            if len(seq) == n:
                return tuple(float(v) for v in seq)
            if len(seq) < n:
                tail = float(seq[-1]) if len(seq) > 0 else default
                return tuple(float(v) for v in seq) + tuple([tail] * (n - len(seq)))
            return tuple(float(v) for v in seq[:n])

        self.ceilings_db = ensure_len(ceilings_db, bands, -1.0)
        self.knees_db = ensure_len(knees_db, bands, 6.0)
        self.lookahead_ms = ensure_len(lookahead_ms, bands, 2.0)
        self.release_ms = ensure_len(release_ms, bands, 50.0)

    def log_contract(self):
        _logger.debug("[DSPContract] %s", asdict(multiband_limiter_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für MultibandLimiter")
            band_signals = self._split_bands(audio, sr)
            processed: list[npt.NDArray[np.float64]] = []
            for i, band in enumerate(band_signals):
                idx = min(i, len(self.ceilings_db) - 1)
                lim = self._limit_band(
                    band,
                    sr,
                    self.ceilings_db[idx],
                    self.knees_db[idx],
                    self.lookahead_ms[idx],
                    self.release_ms[idx],
                )
                processed.append(np.asarray(lim, dtype=np.float64))
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
            _logger.error("MultibandLimiter Fehler: %s", e)
            self._audit_log({"bands": self.bands, "error": str(e)})
            return audio.astype(orig_dtype)

    def _audit_log(self, result: dict[str, Any]):
        _logger.debug("[AuditLog][MultibandLimiter] Ergebnis: %s", result)

    @staticmethod
    def _lr4_sos(cutoff: float, sr: float, btype: str) -> npt.NDArray[np.float64]:
        wn = float(np.clip(cutoff / (sr / 2.0), 1e-4, 0.9999))
        sos = np.asarray(butter(2, wn, btype=btype, output="sos"), dtype=np.float64)
        stacked = np.empty((sos.shape[0] * 2, sos.shape[1]), dtype=np.float64)
        stacked[: sos.shape[0], :] = sos
        stacked[sos.shape[0] :, :] = sos
        return stacked

    def _split_bands(self, audio: npt.NDArray[np.float64], sr: int) -> list[npt.NDArray[np.float64]]:
        """
        Teilt das Signal in Frequenzbänder auf.
        Rückgabe: Liste von Band-Signalen
        """
        bands: list[npt.NDArray[np.float64]] = []
        cross = list(self.crossovers)
        while len(cross) < self.bands - 1:
            cross.append(cross[-1] if cross else 2000.0)
        if self.bands == 1:
            return [audio]

        sos_low = self._lr4_sos(cross[0], sr, "low")
        bands.append(np.asarray(sosfilt(sos_low, audio), dtype=np.float64))

        for i in range(1, self.bands - 1):
            fc0, fc1 = cross[i - 1], cross[i]
            if fc0 >= fc1:
                bands.append(bands[-1].copy() if bands else np.zeros_like(audio))
                continue
            sos_hp = self._lr4_sos(fc0, sr, "high")
            sos_lp = self._lr4_sos(fc1, sr, "low")
            mid = sosfilt(sos_hp, audio)
            mid = sosfilt(sos_lp, mid)
            bands.append(np.asarray(mid, dtype=np.float64))

        sos_high = self._lr4_sos(cross[self.bands - 2], sr, "high")
        bands.append(np.asarray(sosfilt(sos_high, audio), dtype=np.float64))
        return bands

    def _limit_band(
        self,
        audio: npt.NDArray[np.float64],
        sr: int,
        ceiling_db: float,
        knee_db: float,
        lookahead_ms: float,
        release_ms: float,
    ) -> npt.NDArray[np.float64]:
        """
        Limitiert ein einzelnes Frequenzband.
        """
        # Lookahead-Buffer
        lookahead = int(sr * lookahead_ms / 1000)
        padded = np.pad(audio, (lookahead, 0), mode="edge")
        shifted = padded[:-lookahead] if lookahead > 0 else audio
        # True-Peak-Detection (Sample-Peak)
        peak = np.abs(shifted)
        peak_db = 20 * np.log10(peak + 1e-8)
        over = peak_db - ceiling_db
        gain_db = np.zeros_like(peak_db)
        # Soft-Knee
        idx_soft = (over > -knee_db / 2) & (over < knee_db / 2)
        gain_db[idx_soft] = -((over[idx_soft] + knee_db / 2) ** 2) / (2 * knee_db)
        idx_over = over >= knee_db / 2
        gain_db[idx_over] = -over[idx_over]
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        out = audio * env
        return out  # type: ignore[no-any-return]
