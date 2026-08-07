from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "rumble_filter"
    category: str = "restoration"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
rumble_filter_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"cutoff": 20.0, "order": 4},
        "safe_ranges": {
            "cutoff": {"min": 10.0, "max": 60.0},
            "order": {"min": 1, "max": 8},
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
    side_effects=[{"risk": "Bassverlust", "expected_when": "cutoff > 40.0", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["rumble_reduction"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
"""
rumble_filter.py - Rumble-Filter (Vinyl-Tiefbass-Störungsentfernung) für Aurik 6.0

Dieses Modul entfernt tieffrequente Rumpelstörungen (Rumble) aus Audiosignalen (Stub).
"""

import logging

import numpy as np

try:
    import onnxruntime as ort
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    ort = None
from scipy.signal import butter, lfilter

from dsp._memory_budget_guard import check_budget

logger = logging.getLogger("aurik.dsp.rumble_filter")
logger.setLevel(logging.INFO)


class RumbleFilter:
    """
    SOTA-Rumble-Filter:
    - Digitaler Hochpass (Butterworth) für Rumpelstörungen
    - Deep-Learning-Inferenz (ONNX/Torch) als Option
    """

    def __init__(self, cutoff: float = 20.0, order: int = 4, model_path: str | None = None):
        self.cutoff = cutoff
        self.order = order
        self.model_path = model_path
        self.model = None
        self.backend = None
        if model_path:
            if ort is not None:
                if not check_budget("rumble_filter_onnx", 0.1):
                    logger.warning("Memory Grenze exceeded for rumble_filter ONNX — using DSP Ersatzpfad")
                else:
                    try:
                        self.model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                        self.backend = "onnx"
                    except Exception as e:
                        logger.warning("ONNX-Modell konnte nicht geladen werden: %s", e)
            elif torch is not None:
                try:
                    self.model = torch.jit.load(model_path)
                    self.backend = "torch"
                except Exception as e:
                    logger.warning("Torch-Modell konnte nicht geladen werden: %s", e)
            else:
                logger.warning("Weder ONNX noch Torch verfügbar. Nur klassischer Filter nutzbar.")

    # Audit: Contract-Infos loggen (optional)
    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(rumble_filter_contract))

    def process(self, audio: np.ndarray, sr: int, audit_log: bool = True) -> np.ndarray:
        """
        Entfernt tieffrequente Rumpelstörungen (Rumble) aus Audiosignalen.
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung, optionale DL-Inferenz, Rückfallstrategie
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param sr: Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: Gefiltertes Audio (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            logger.error("Ungültiges Audio-Array (leer oder falscher Typ)")
            raise ValueError("Ungültiges Audio-Array (leer oder falscher Typ)")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1e6:
            logger.warning("Audio möglicherweise nicht normiert (max > 1e6)")

        audio_out = None
        fallback_used = False
        try:
            self.log_contract()
            # Deep-Learning-Inferenz
            if self.model is not None and self.backend == "onnx":
                inp = audio.astype(np.float32)[None, None, :]
                try:
                    out = self.model.run(None, {self.model.get_inputs()[0].name: inp})[0]
                    audio_out = out.squeeze().astype(audio.dtype)
                except Exception as e:
                    logger.warning("ONNX-Inferenz fehlgeschlagen: %s", e)
                    fallback_used = True
            elif self.model is not None and self.backend == "torch" and torch is not None:
                try:
                    model_torch: Any = self.model
                    inp_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # type: ignore[assignment]
                    out = model_torch(inp_t).detach().cpu().numpy().squeeze()
                    audio_out = out.astype(audio.dtype)
                except Exception as e:
                    logger.warning("Torch-Inferenz fehlgeschlagen: %s", e)
                    fallback_used = True
            if audio_out is None:
                # Fallback: Digitaler Hochpass (Butterworth)
                nyq = 0.5 * sr
                norm_cutoff = self.cutoff / nyq
                b, a = butter(self.order, norm_cutoff, btype="high", output="ba")  # type: ignore[misc]
                audio_out = lfilter(b, a, audio)
                fallback_used = True
        except Exception as e:
            logger.error("Fehler beim Rumble-Filtering: %s", e)
            audio_out = audio.copy()
            fallback_used = True

        if audit_log:
            logger.info("RumbleFilter: cutoff=%s, order=%s, Ersatzpfad_used=%s", self.cutoff, self.order, fallback_used)
        return audio_out.astype(audio.dtype)
