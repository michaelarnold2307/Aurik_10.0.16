"""
adaptive_stft.py - SOTA-konformes STFT/MelSpectrogram Modul für Aurik 6.0

Dieses Modul ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import librosa
import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.adaptive_stft")
logger.setLevel(logging.INFO)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_stft"
    category: str = "stft"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_stft_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "n_fft": 2048,
            "hop_length": 512,
            "win_length": 2048,
            "window": "hann",
        },
        "safe_ranges": {
            "n_fft": {"min": 256, "max": 8192},
            "hop_length": {"min": 64, "max": 4096},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Aliasing", "expected_when": "n_fft zu klein", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["spectral_resolution"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)

adaptive_mel_contract = DSPContract(
    id="adaptive_mel_spectrogram",
    category="mel_spectrogram",
    version="1.0.0",
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 128,
            "fmin": 0,
            "fmax": None,
        },
        "safe_ranges": {
            "n_fft": {"min": 256, "max": 8192},
            "n_mels": {"min": 16, "max": 512},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Verlust von Details",
            "expected_when": "n_mels zu klein",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["mel_resolution"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveSTFT:
    """
    SOTA-konforme STFT mit Quality-Gate, Audit-Logging, Fehlerbehandlung, DL-Inferenz-Platzhalter, Doku als Code.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
    ):
        if not (256 <= n_fft <= 8192):
            logger.error("Ungültiges n_fft: %s. Muss zwischen 256 und 8192 liegen.", n_fft)
            raise ValueError("n_fft muss zwischen 256 und 8192 liegen.")
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self._last_stft_settings: dict[str, Any] = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "center": self.center,
            "pad_mode": self.pad_mode,
        }
        logger.info(
            f"AdaptiveSTFT initialisiert mit n_fft={self.n_fft}, hop_length={self.hop_length}, win_length={self.win_length}, window={self.window}, center={self.center}, pad_Betriebsart={self.pad_mode}"
        )

    def log_contract(self):
        contract_dict = asdict(adaptive_stft_contract)
        logger.info("[DSPContract] %s", contract_dict)

    def stft(self, y, sr=None, use_deep_learning: bool = False, audit_log: bool = True, **kwargs):
        """
        Führt STFT durch. Quality-Gate, Audit-Logging, DL-Inferenz-Platzhalter, Fehlerbehandlung, SOTA-Transparenz.
        :param y: Audiosignal (np.ndarray)
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :return: STFT-Matrix (np.ndarray)
        """
        if not isinstance(y, np.ndarray):
            logger.error("y ist kein np.ndarray")
            raise TypeError("y ist kein np.ndarray")
        if y.size == 0:
            logger.error("y ist leer")
            raise ValueError("y ist leer")
        if np.isnan(y.astype(np.float64, copy=False)).any():
            logger.error("y enthält NaN-Werte")
            raise ValueError("y enthält NaN-Werte")

        n_fft = kwargs.get("n_fft", self.n_fft)
        n_fft = min(n_fft, max(64, len(y)))
        hop_length = kwargs.get("hop_length", self.hop_length)
        win_length = kwargs.get("win_length", self.win_length)
        window = kwargs.get("window", self.window)
        center = kwargs.get("center", self.center)
        pad_mode = kwargs.get("pad_mode", self.pad_mode)

        self._last_stft_settings = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "center": center,
            "pad_mode": pad_mode,
        }

        output = None
        fallback_used = False
        try:
            if use_deep_learning:
                if not _TORCH_AVAILABLE:
                    logger.warning("PyTorch nicht verfügbar, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._stft_classic(y, n_fft, hop_length, win_length, window, center, pad_mode)
                else:
                    logger.info("Deep-Learning-Inferenz aktiviert für STFT.")
                    # TorchScript-Modell (Platzhalter)
                    # model = torch.jit.load('stft.pt')
                    # output = model(torch.from_numpy(y).float().unsqueeze(0)).squeeze(0).numpy()
                    logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._stft_classic(y, n_fft, hop_length, win_length, window, center, pad_mode)
            else:
                output = self._stft_classic(y, n_fft, hop_length, win_length, window, center, pad_mode)
        except Exception as e:
            logger.error("Fehler bei STFT: %s", e, exc_info=True)
            fallback_used = True
            output = np.zeros((n_fft // 2 + 1, 1), dtype=np.complex128)

        if audit_log:
            spectral_resolution = float(n_fft) / float(sr if sr is not None else 48000)
            logger.info(
                f"AdaptiveSTFT: spectral_resolution={spectral_resolution:.4f}, Ersatzpfad_used={fallback_used}, n_fft={n_fft}"
            )
            logger.info("[DSPContract] %s", asdict(adaptive_stft_contract))
        return output

    def _stft_classic(self, y, n_fft, hop_length, win_length, window, center, pad_mode):
        return librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

    def istft(self, D, sr=None, audit_log: bool = True, **kwargs):
        """
        Führt inverse STFT durch. Quality-Gate, Audit-Logging, Fehlerbehandlung, SOTA-Transparenz.
        :param D: STFT-Matrix (np.ndarray)
        :return: Zeitsignal (np.ndarray)
        """
        if not isinstance(D, np.ndarray):
            logger.error("D ist kein np.ndarray")
            raise TypeError("D ist kein np.ndarray")
        if D.size == 0:
            logger.error("D ist leer")
            raise ValueError("D ist leer")
        if np.isnan(D).any():
            logger.error("D enthält NaN-Werte")
            raise ValueError("D enthält NaN-Werte")
        output = None
        try:
            hop_length = kwargs.get("hop_length", self._last_stft_settings.get("hop_length", self.hop_length))
            win_length = kwargs.get("win_length", self._last_stft_settings.get("win_length", self.win_length))
            window = kwargs.get("window", self._last_stft_settings.get("window", self.window))
            center = kwargs.get("center", self._last_stft_settings.get("center", self.center))
            output = librosa.istft(
                D,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                length=kwargs.get("length"),
            )
        except Exception as e:
            logger.error("Fehler bei ISTFT: %s", e, exc_info=True)
            output = np.zeros(1)
        output = np.nan_to_num(np.asarray(output), nan=0.0, posinf=0.0, neginf=0.0)
        if audit_log:
            logger.info("AdaptiveSTFT: ISTFT ausgeführt, shape=%s", output.shape if output is not None else None)
            logger.info("[DSPContract] %s", asdict(adaptive_stft_contract))
        return output

    def auto_optimize(self, y, sr):
        self.log_contract()
        if len(y) < 4096:
            self.n_fft = 512
            self.hop_length = 128
        elif len(y) < 16384:
            self.n_fft = 1024
            self.hop_length = 256
        else:
            self.n_fft = 2048
            self.hop_length = 512
        logger.info("STFT-Parameter auto-optimiert: n_fft=%s, hop_length=%s", self.n_fft, self.hop_length)


class AdaptiveMelSpectrogram:
    """
    SOTA-konformes MelSpectrogram mit Quality-Gate, Audit-Logging, Fehlerbehandlung, DL-Inferenz-Platzhalter, Doku als Code.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        n_mels=128,
        sr=22050,
        fmin=0,
        fmax=None,
        power=2.0,
    ):
        if not (256 <= n_fft <= 8192):
            logger.error("Ungültiges n_fft: %s. Muss zwischen 256 und 8192 liegen.", n_fft)
            raise ValueError("n_fft muss zwischen 256 und 8192 liegen.")
        if not (16 <= n_mels <= 512):
            logger.error("Ungültiges n_mels: %s. Muss zwischen 16 und 512 liegen.", n_mels)
            raise ValueError("n_mels muss zwischen 16 und 512 liegen.")
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        logger.info(
            f"AdaptiveMelSpectrogram initialisiert mit n_fft={self.n_fft}, hop_length={self.hop_length}, n_mels={self.n_mels}, sr={self.sr}, fmin={self.fmin}, fmax={self.fmax}, power={self.power}"
        )

    def log_contract(self):
        contract_dict = asdict(adaptive_mel_contract)
        logger.info("[DSPContract] %s", contract_dict)

    def mel_spectrogram(self, y, sr=None, use_deep_learning: bool = False, audit_log: bool = True, **kwargs):
        """
        Führt MelSpectrogram-Berechnung durch. Quality-Gate, Audit-Logging, DL-Inferenz-Platzhalter, Fehlerbehandlung, SOTA-Transparenz.
        :param y: Audiosignal (np.ndarray)
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :return: Mel-Spektrogramm (np.ndarray)
        """
        if not isinstance(y, np.ndarray):
            logger.error("y ist kein np.ndarray")
            raise TypeError("y ist kein np.ndarray")
        if y.size == 0:
            logger.error("y ist leer")
            raise ValueError("y ist leer")
        if np.isnan(y.astype(np.float64, copy=False)).any():
            logger.error("y enthält NaN-Werte")
            raise ValueError("y enthält NaN-Werte")

        n_fft = kwargs.get("n_fft", self.n_fft)
        n_mels = kwargs.get("n_mels", self.n_mels)
        output = None
        fallback_used = False
        try:
            if use_deep_learning:
                if not _TORCH_AVAILABLE:
                    logger.warning("PyTorch nicht verfügbar, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._mel_spectrogram_classic(y, sr, n_fft, n_mels, **kwargs)
                else:
                    logger.info("Deep-Learning-Inferenz aktiviert für MelSpectrogram.")
                    # TorchScript-Modell (Platzhalter)
                    # model = torch.jit.load('mel_spectrogram.pt')
                    # output = model(torch.from_numpy(y).float().unsqueeze(0)).squeeze(0).numpy()
                    logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._mel_spectrogram_classic(y, sr, n_fft, n_mels, **kwargs)
            else:
                output = self._mel_spectrogram_classic(y, sr, n_fft, n_mels, **kwargs)
        except Exception as e:
            logger.error("Fehler bei MelSpectrogram: %s", e, exc_info=True)
            fallback_used = True
            output = np.zeros((n_mels, 1))

        if audit_log:
            mel_resolution = float(n_mels)
            logger.info(
                f"AdaptiveMelSpectrogram: mel_resolution={mel_resolution:.2f}, Ersatzpfad_used={fallback_used}, n_fft={n_fft}, n_mels={n_mels}"
            )
            logger.info("[DSPContract] %s", asdict(adaptive_mel_contract))
        return output

    def _mel_spectrogram_classic(self, y, sr, n_fft, n_mels, **kwargs):
        return librosa.feature.melspectrogram(
            y=y,
            sr=sr or self.sr,
            n_fft=n_fft,
            hop_length=kwargs.get("hop_length", self.hop_length),
            n_mels=n_mels,
            fmin=kwargs.get("fmin", self.fmin),
            fmax=kwargs.get("fmax", self.fmax),
            power=kwargs.get("power", self.power),
        )

    def auto_optimize(self, y, sr):
        self.log_contract()
        if sr < 16000:
            self.n_mels = 64
        elif sr < 32000:
            self.n_mels = 128
        else:
            self.n_mels = 256
        logger.info("Mel-Parameter auto-optimiert: n_mels=%s", self.n_mels)
