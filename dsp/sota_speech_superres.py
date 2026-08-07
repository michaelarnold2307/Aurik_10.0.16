"""
---
modul_name: SotaSpeechSuperRes
aufgabe: SOTA-Speech/Music-Super-Resolution (DiffWave, HiFi-GAN)
ein_ausgabe_typen:
    input: np.ndarray (Audio)
    output: np.ndarray (Audio)
staerken: Deep-Learning, SOTA, flexibel
schwaechen: Modellabhängig, benötigt Modelle/Weights
abhaengigkeiten: [numpy, onnxruntime]
---
"""

import logging
import os

import numpy as np

from dsp._memory_budget_guard import check_budget

logger = logging.getLogger(__name__)

MODEL_PATH = "../../models/hifi_gan/hifi_gan.onnx"


class SotaSpeechSuperRes:
    def __init__(self, use_diffwave=True, use_hifigan=True):
        self.diffwave_session = None
        self.hifigan_session = None
        # DiffWave-ONNX laden
        try:
            import onnxruntime as ort

            diffwave_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../models/diffwave/diffwave_model.onnx")
            )
            if use_diffwave and os.path.exists(diffwave_path):
                if check_budget("speech_superres_diffwave", 0.2):
                    self.diffwave_session = ort.InferenceSession(diffwave_path, providers=["CPUExecutionProvider"])
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        # HiFi-GAN-ONNX laden
        try:
            import onnxruntime as ort

            hifigan_path = MODEL_PATH
            if use_hifigan and os.path.exists(hifigan_path):
                if check_budget("speech_superres_hifigan", 0.15):
                    self.hifigan_session = ort.InferenceSession(hifigan_path, providers=["CPUExecutionProvider"])
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    def super_resolve(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Priorität: DiffWave > HiFi-GAN > Fallback
        if self.diffwave_session is not None:
            x = audio.astype(np.float32)
            if x.ndim == 1:
                x = x[None, :]
            ort_inputs = {self.diffwave_session.get_inputs()[0].name: x}
            out = self.diffwave_session.run(None, ort_inputs)[0]
            return np.asarray(out.squeeze().astype(audio.dtype))
        if self.hifigan_session is not None:
            x = audio.astype(np.float32)
            if x.ndim == 1:
                x = x[None, :]
            ort_inputs = {self.hifigan_session.get_inputs()[0].name: x}
            out = self.hifigan_session.run(None, ort_inputs)[0]
            return np.asarray(out.squeeze().astype(audio.dtype))
        # Fallback: Identität
        return audio
