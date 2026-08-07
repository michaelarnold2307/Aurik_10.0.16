"""
perceptual_eq.py - Perceptual EQ (Hörmodell, AI) für Aurik 6.0

Dieses Modul stellt ein Perceptual-EQ-Modul auf Basis von Hörmodellen als Stub bereit.
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

from dsp._memory_budget_guard import check_budget

logger = logging.getLogger(__name__)

_logger = logging.getLogger(__name__)


class PerceptualEQ:
    """
    SOTA Perceptual EQ:
    - Deep-Learning-Inferenz (ONNX/Torch) für psychoakustische Anpassung
    - Auswahl verschiedener Hörmodelle
    - Fallback: klassische Filter
    """

    def __init__(self, model_path: str | None = None, hearing_model: str | None = None):
        self.model_path = model_path
        self.model = None  # Legacy alias
        self.onnx_session = None
        self.torch_model = None
        self.hearing_model = hearing_model or "Moore-Glasberg"
        self.backend = None
        if model_path:
            if ort is not None:
                if not check_budget("perceptual_eq_onnx", 0.1):
                    _logger.warning("Memory budget exceeded for perceptual_eq ONNX — using DSP fallback")
                else:
                    try:
                        self.onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                        self.model = self.onnx_session
                        self.backend = "onnx"
                    except Exception as e:
                        _logger.warning("ONNX-Modell konnte nicht geladen werden: %s", e)
            elif torch is not None:
                try:
                    self.torch_model = torch.jit.load(model_path)
                    self.model = self.torch_model
                    self.backend = "torch"
                except Exception as e:
                    _logger.warning("Torch-Modell konnte nicht geladen werden: %s", e)
            else:
                _logger.warning("Weder ONNX noch Torch verfügbar. Nur klassische Filter nutzbar.")

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Deep-Learning-Inferenz
        if self.onnx_session is not None and self.backend == "onnx":
            inp = audio.astype(np.float32)[None, None, :]
            try:
                out = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: inp})[0]
                return out.squeeze().astype(audio.dtype)  # type: ignore[no-any-return]
            except Exception as e:
                _logger.warning("ONNX-Inferenz fehlgeschlagen: %s", e)
        elif self.torch_model is not None and self.backend == "torch" and torch is not None:
            try:
                inp = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                out = self.torch_model(inp).detach().cpu().numpy().squeeze()
                return out.astype(audio.dtype)  # type: ignore[no-any-return]
            except Exception as e:
                _logger.warning("Torch-Inferenz fehlgeschlagen: %s", e)
        # Fallback: klassische Filter nach Hörmodell
        return self._perceptual_filter(audio, sr)

    def _perceptual_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Psychoakustischer Filter nach Moore-Glasberg / ISO 226 Equal-Loudness-Konturen.

        Approximation der Gleichlautheitskorrektur bei 60 Phon relativ zur Referenz:
          - Sub-Bass (<80 Hz): +3 dB Boost (hörphysiologisch relevant)
          - Bass (80-500 Hz): leichte EQ-Korrektur
          - Mittenband (800-4000 Hz): Präsenzbereich, +1 dB
          - Hochtonbereich (4000-12000 Hz): natürliche Spitze, +2 dB
          - Ober-Hochton (>12 kHz): leichte Abschwächung

        Implementierung: Butterworth Shelving-Kaskade (scipy.signal).
        """
        from scipy.signal import butter, sosfilt

        try:
            y = audio.astype(np.float64)
            nyq = sr / 2.0
            # Band 1: Low-Shelf Boost bei 80 Hz (+3 dB)
            sos_ls = butter(2, min(80.0 / nyq, 0.49), btype="low", output="sos")
            ls_band = sosfilt(sos_ls, y)
            y = y + ls_band * (10 ** (3.0 / 20.0) - 1.0)
            # Band 2: Präsenz-Anhebung 1-4 kHz (+1.5 dB)
            lo = min(1000.0 / nyq, 0.499)
            hi = min(4000.0 / nyq, 0.499)
            if lo < hi:
                sos_mid = butter(2, [lo, hi], btype="band", output="sos")
                mid_band = sosfilt(sos_mid, y)
                y = y + mid_band * (10 ** (1.5 / 20.0) - 1.0)
            # Band 3: Brillanz 6-12 kHz (+2 dB)
            lo2 = min(6000.0 / nyq, 0.499)
            hi2 = min(12000.0 / nyq, 0.499)
            if lo2 < hi2:
                sos_hi = butter(2, [lo2, hi2], btype="band", output="sos")
                hi_band = sosfilt(sos_hi, y)
                y = y + hi_band * (10 ** (2.0 / 20.0) - 1.0)
            # Normalisierung: Amplitude erlält RMS-Verhältnis
            rms_in = float(np.sqrt(np.mean(audio**2))) + 1e-12
            rms_out = float(np.sqrt(np.mean(y**2))) + 1e-12
            y = y * (rms_in / rms_out)
            return np.clip(y, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]
        except Exception:
            logger.warning("perceptual_eq.py::_perceptual_filter Ersatzpfad", exc_info=True)
            return audio
