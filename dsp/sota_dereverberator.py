"""
---
modul_name: SotaDereverberator
aufgabe: SOTA-Dereverberation (DCCRN-ONNX, Conv-TasNet)
ein_ausgabe_typen:
    input: np.ndarray (Audio)
    output: np.ndarray (Audio)
staerken: Deep-Learning, SOTA, flexibel
schwaechen: Modellabhängig, benötigt Modelle/Weights
abhaengigkeiten: [numpy, onnxruntime, torch]
---
"""

import logging
import os

import numpy as np

from dsp._memory_budget_guard import check_budget

_logger = logging.getLogger(__name__)


class SotaDereverberator:
    def __init__(self, use_dccrn=True, use_conv_tasnet=True):
        self.dccrn_session = None
        self.conv_tasnet = None
        # DCCRN-ONNX laden
        try:
            import onnxruntime as ort

            dccrn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/dccrn/dccrn.onnx"))
            if use_dccrn and os.path.exists(dccrn_path):
                if check_budget("sota_dereverb_dccrn", 0.15):
                    self.dccrn_session = ort.InferenceSession(dccrn_path, providers=["CPUExecutionProvider"])
                else:
                    _logger.warning("Memory budget exceeded for dereverberator DCCRN")
        except ImportError:
            pass
        # Conv-TasNet (PyTorch) laden
        try:
            import torch

            conv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/conv-tasnet/best_1.pt"))
            if use_conv_tasnet and os.path.exists(conv_path):
                self.conv_tasnet = torch.jit.load(conv_path, map_location="cpu")
                self.conv_tasnet.eval()
        except ImportError:
            _logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    def dereverberate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Entfernt Nachhall mit DCCRN (ONNX) oder Conv-TasNet (PyTorch). Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Eingabe-Audiosignal (np.ndarray)
        :param sr: Sample-Rate
        :return: Dereverberiertes Signal (np.ndarray)
        """
        # Quality-Gate: Input-Check
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            return np.zeros(0, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim != 1:
            self._audit_log("error", "Input must be 1D array")
            return audio.astype(audio.dtype)
        try:
            # Priorität: DCCRN > Conv-TasNet > Fallback
            if self.dccrn_session is not None:
                x = audio.astype(np.float32)
                if x.ndim == 1:
                    x = x[None, :]
                ort_inputs = {self.dccrn_session.get_inputs()[0].name: x}
                out = self.dccrn_session.run(None, ort_inputs)[0]
                self._audit_log("success", "DCCRN-Inferenz erfolgreich")
                return np.asarray(out.squeeze().astype(audio.dtype))
            if self.conv_tasnet is not None:
                import torch

                x = torch.from_numpy(audio).float().unsqueeze(0)  # type: ignore[assignment]
                with torch.no_grad():
                    out = self.conv_tasnet(x)
                self._audit_log("success", "Conv-TasNet-Inferenz erfolgreich")
                return np.asarray(out.squeeze().cpu().numpy().astype(audio.dtype))
            self._audit_log("warn", "Kein DL-Modell verfügbar, Fallback Identität")
            return audio.copy().astype(audio.dtype)
        except Exception as e:
            self._audit_log("error", str(e))
            return audio.copy().astype(audio.dtype)

    def _audit_log(self, level: str, message: str) -> None:
        _logger.debug("[AUR-AUDIT][%s][sota_dereverberator] %s", level.upper(), message)
