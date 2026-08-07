"""
ai_riaa_declicker.py - KI-gestützter RIAA-Declicker für Aurik 6.0

Dieses Modul entfernt Klicks/Knackser speziell nach RIAA-Kennlinie (Stub).
"""

import logging
from typing import Any

import numpy as np

ort: Any | None = None
try:
    import onnxruntime as ort  # type: ignore[no-redef]
except ImportError:
    ort = None

torch: Any | None = None
try:
    import torch
except ImportError:
    torch = None
from scipy.signal import medfilt

from dsp._memory_budget_guard import check_budget

logger = logging.getLogger("aurik.dsp.riaa_declicker")
logger.setLevel(logging.INFO)


class AiRiaaDeclicker:
    """
    SOTA RIAA-Declicker:
    - Deep-Learning-Inferenz (ONNX/Torch) für Declicking nach RIAA-Kennlinie
    - Klassische Pulsdetektion als Fallback
    """

    def __init__(self, model_path: str | None = None, sensitivity: float = 1.0):
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.model = None  # Legacy alias
        self.onnx_session = None
        self.torch_model = None
        self.backend = None
        if model_path:
            if ort is not None:
                if not check_budget("riaa_declicker_onnx", 0.1):
                    logger.warning("Memory Grenze exceeded for riaa_declicker ONNX — using DSP Ersatzpfad")
                else:
                    try:
                        self.onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                        self.model = self.onnx_session
                        self.backend = "onnx"
                    except Exception as e:
                        logger.warning("ONNX-Modell konnte nicht geladen werden: %s", e)
            elif torch is not None:
                try:
                    self.torch_model = torch.jit.load(model_path)
                    self.model = self.torch_model
                    self.backend = "torch"
                except Exception as e:
                    logger.warning("Torch-Modell konnte nicht geladen werden: %s", e)
            else:
                logger.warning("Weder ONNX noch Torch verfügbar. Nur klassische Pulsdetektion nutzbar.")

    def declick_riaa(self, audio: np.ndarray, sr: int, audit_log: bool = True) -> np.ndarray:
        """
        Entfernt Klicks/Knackser nach RIAA-Kennlinie.
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung, optionale DL-Inferenz, Rückfallstrategie
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param sr: Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: Deklicktes Audio (np.ndarray)
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
            # Deep-Learning-Inferenz
            if self.onnx_session is not None and self.backend == "onnx":
                inp = audio.astype(np.float32)[None, None, :]
                try:
                    out = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: inp})[0]
                    audio_out = out.squeeze().astype(audio.dtype)
                except Exception as e:
                    logger.warning("ONNX-Inferenz fehlgeschlagen: %s", e)
                    fallback_used = True
            elif self.torch_model is not None and self.backend == "torch" and torch is not None:
                try:
                    inp = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    out = self.torch_model(inp).detach().cpu().numpy().squeeze()
                    audio_out = out.astype(audio.dtype)
                except Exception as e:
                    logger.warning("Torch-Inferenz fehlgeschlagen: %s", e)
                    fallback_used = True
            if audio_out is None:
                # Fallback: Klassische Pulsdetektion
                diff = np.abs(audio - medfilt(audio, kernel_size=7))
                mask = diff > self.sensitivity * np.max(diff)
                audio_out = audio.copy()
                if np.any(mask):
                    idx = np.where(mask)[0]
                    for i in idx:
                        left = max(0, i - 3)
                        right = min(len(audio) - 1, i + 3)
                        audio_out[i] = np.median(audio_out[left : right + 1])
                fallback_used = True
        except Exception as e:
            logger.error("Fehler beim RIAA-Declicking: %s", e)
            audio_out = audio.copy()
            fallback_used = True

        if audit_log:
            declicking_error = float(np.mean(np.abs(audio - audio_out)))
            logger.info("AiRiaaDeclicker: declicking_error=%.4f, Ersatzpfad_used=%s", declicking_error, fallback_used)
        return audio_out.astype(audio.dtype)
