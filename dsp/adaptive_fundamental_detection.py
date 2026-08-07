"""
Adaptive Fundamental Detection DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Einfache Grundtonerkennung per Autokorrelation (klassische DSP, SOTA-Maximum).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.adaptive_fundamental_detection")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_fundamental_detection"
    category: str = "fundamental_detection"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveFundamentalDetection:
    """
    Klassische Grundtonerkennung per Autokorrelation (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def detect(self, x: np.ndarray, use_deep_learning: bool = False, audit_log: bool = True) -> float:
        """
        Erkennt die Grundfrequenz per Autokorrelation (SOTA-Ansatz) oder optional Deep-Learning.
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung, fallback auf klassische Methode.
        :param x: Eingabesignal (np.ndarray)
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :return: Grundfrequenz (float)
        """
        self.log_contract()
        # Quality Gate: Input-Checks
        if not isinstance(x, np.ndarray) or x.size == 0:
            logger.error("Ungültiges Eingabesignal (kein np.ndarray oder leer)")
            raise ValueError("Ungültiges Eingabesignal (kein np.ndarray oder leer)")
        if np.isnan(x).any():
            logger.error("Eingabesignal enthält NaN-Werte")
            raise ValueError("Eingabesignal enthält NaN-Werte")
        if np.max(np.abs(x)) > 1.5:
            logger.warning("Eingabesignal möglicherweise nicht normiert (max > 1.5)")

        freq = 0.0
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für Grundtonerkennung.")
                # TorchScript-Modell (Platzhalter)
                # model = torch.jit.load('fundamental_detector.pt')
                # freq = float(model(torch.from_numpy(x).float().unsqueeze(0)).item())
                logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                fallback_used = True
                freq = self._detect_classic(x)
            else:
                freq = self._detect_classic(x)
        except Exception as e:
            logger.error("Fehler bei Grundtonerkennung: %s", e)
            fallback_used = True
            freq = 0.0

        if audit_log:
            logger.info("AdaptiveFundamentalDetection: freq=%.2f Hz, Ersatzpfad_used=%s", freq, fallback_used)
        return freq

    def _detect_classic(self, x: np.ndarray) -> float:
        """Klassische Grundtonerkennung per Autokorrelation."""
        from backend.core.core_utils import fft_autocorr

        corr = fft_autocorr(x)
        corr = corr[len(corr) // 2 :]
        peak = np.argmax(corr[1:]) + 1
        freq = self.sr / peak if peak > 0 else 0.0
        return freq  # type: ignore[return-value]

    def auto_optimize(self, x: np.ndarray) -> None:
        """
        Passe effektive Analyse-Samplingrate anhand des Hochfrequenzanteils an.
        Hoher HF-Anteil (Musik, Breitband) → sr=44100.
        Niedriger HF-Anteil (Sprache, Schmalband) → sr=16000.
        :param x: Eingabesignal (np.ndarray)
        """
        mag = np.abs(np.fft.rfft(x.astype(float)))
        n_bins = len(mag)
        low_energy = float(np.sum(mag[: n_bins // 2] ** 2))
        high_energy = float(np.sum(mag[n_bins // 2 :] ** 2))
        hf_ratio = high_energy / (low_energy + high_energy + 1e-8)

        if hf_ratio > 0.25:
            self.sr = 44100  # Breitband-Inhalt → volle SR
        elif hf_ratio > 0.10:
            self.sr = 22050  # Mittlerer Bereich
        else:
            self.sr = 16000  # Sprache / schmalbandiges Signal
        logger.info("auto_optimieren (FundamentalDetection): HF-Verhaeltnis=%.3f → sr=%s", hf_ratio, self.sr)
