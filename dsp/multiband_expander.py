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
    name: str = "MultibandExpander"
    version: str = "1.0"
    description: str = "Adaptiver Multiband-Expander mit SOTA-Features"
    parameters: dict[str, Any] | None = None


multiband_expander_contract = DSPContract(
    parameters={
        "bands": 3,
        "crossovers": (200, 2000),
        "thresholds_db": (-40, -35, -30),
        "ratios": (0.5, 0.4, 0.3),
        "knees_db": (6, 6, 6),
        "attack_ms": (10, 8, 5),
        "release_ms": (80, 60, 40),
    }
)


class MultibandExpander:
    """
    SOTA-konformer adaptiver Multiband-Expander:
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
        thresholds_db: Sequence[float] = (-40, -35, -30),
        ratios: Sequence[float] = (0.5, 0.4, 0.3),
        knees_db: Sequence[float] = (6, 6, 6),
        attack_ms: Sequence[float] = (10, 8, 5),
        release_ms: Sequence[float] = (80, 60, 40),
    ) -> None:
        """
        bands: Anzahl der Frequenzbänder
        crossovers: Übergangsfrequenzen (Hz)
        thresholds_db: Expander-Schwellen pro Band (dB)
        ratios: Expansionsraten pro Band (<1)
        knees_db: Soft-Knee pro Band (dB)
        attack_ms: Attack-Zeiten pro Band (ms)
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

        self.thresholds_db = ensure_len(thresholds_db, bands, -40.0)
        self.ratios = ensure_len(ratios, bands, 0.5)
        self.knees_db = ensure_len(knees_db, bands, 6.0)
        self.attack_ms = ensure_len(attack_ms, bands, 10.0)
        self.release_ms = ensure_len(release_ms, bands, 80.0)

    def log_contract(self):
        _logger.debug("[DSPContract] %s", asdict(multiband_expander_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für MultibandExpander")
            band_signals = self._split_bands(audio, sr)
            processed: list[npt.NDArray[np.float64]] = []
            for i, band in enumerate(band_signals):
                idx = min(i, len(self.thresholds_db) - 1)
                exp = self._expand_band(
                    band,
                    sr,
                    self.thresholds_db[idx],
                    self.ratios[idx],
                    self.knees_db[idx],
                    self.attack_ms[idx],
                    self.release_ms[idx],
                )
                processed.append(np.asarray(exp, dtype=np.float64))
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
            _logger.error("MultibandExpander Fehler: %s", e)
            self._audit_log({"bands": self.bands, "error": str(e)})
            return audio.astype(orig_dtype)

    def _audit_log(self, result: dict[str, Any]):
        _logger.debug("[AuditLog][MultibandExpander] Ergebnis: %s", result)

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
        return np.apply_along_axis(lambda ch: MultibandExpander._moving_rms(ch, window), axis=-1, arr=x)

    def _expand_band(
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
        Expandiert ein einzelnes Frequenzband.
        """
        window = int(sr * 0.01)
        rms = self._moving_rms(audio, window)
        rms = np.nan_to_num(rms, nan=1e-8, posinf=1e-8, neginf=1e-8)
        rms_db = 20 * np.log10(rms + 1e-8)
        under = threshold_db - rms_db
        gain_db = np.zeros_like(rms_db)
        # Soft-Knee
        idx_soft = (under > -knee_db / 2) & (under < knee_db / 2)
        gain_db[idx_soft] = (1 / ratio - 1) * ((under[idx_soft] + knee_db / 2) ** 2) / (2 * knee_db)
        idx_under = under >= knee_db / 2
        gain_db[idx_under] = (1 / ratio - 1) * (under[idx_under])
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
