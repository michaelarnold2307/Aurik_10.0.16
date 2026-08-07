# SOTA Deep-Learning Artefakt-Detection Plugin für Aurik
# Modular, erweiterbar, API-ready
# CPU-only Policy (Section 9.5): Keine CUDA/GPU-Nutzung

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torchaudio

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    logger.debug("torch/torchaudio nicht verfügbar, DSP-Ersatzpfad aktiv")


class ArtifactDetectionPlugin:
    """
    Deep-Learning Artefakt-Detection für Audiosignale.
    - Modular: Modell kann getauscht werden
    - API-ready: detect_artifacts() für Batch/Echtzeit
    - User-Feedback: Feedback-Interface
    - CPU-only: Läuft immer auf der CPU (keine GPU-Anforderung)
    """

    _BUDGET_NAME: str = "ArtifactDetection"
    _BUDGET_SIZE_GB: float = 0.05  # ~30-50 MB TorchScript

    def __init__(self, model_path: str):
        # Pflicht: Ausschließlich CPU — keine CUDA/GPU (Section 9.5)
        self.device = "cpu"
        self.model = None
        if _TORCH_AVAILABLE:
            try:
                from backend.core.ml_memory_budget import try_allocate

                if not try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                    logger.info("ArtifactDetection: ML-Grenze erschöpft — DSP-Ersatzpfad.")
                else:
                    self.model = self._load_model(model_path)
            except ImportError:
                self.model = self._load_model(model_path)
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _self = self
                _reg_plm(
                    self._BUDGET_NAME,
                    size_gb=self._BUDGET_SIZE_GB,
                    unload_fn=lambda s=_self: setattr(s, "model", None),  # type: ignore[misc]
                )
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        elif _TORCH_AVAILABLE:
            try:
                from backend.core.ml_memory_budget import release as _release

                _release(self._BUDGET_NAME)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    def _load_model(self, path: str):
        """TorchScript-Modell laden — nur wenn torch verfügbar."""
        try:
            import os as _os

            torch.set_num_threads(_os.cpu_count() or 4)  # §2.37 CPU-Thread-Budget
            return torch.jit.load(path, map_location="cpu")  # CPU-only
        except Exception as exc:
            logger.warning("Modell konnte nicht geladen werden: %s — DSP-Ersatzpfad aktiv", exc)
            return None

    def _dsp_fallback_detect(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """DSP-Fallback-Detektion (kein ML erforderlich)."""
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio.flatten() if audio.ndim > 1 else audio
        mono = np.clip(mono, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-10)
        rms = np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(mono)))
        peak = np.nan_to_num(peak, nan=0.0, posinf=0.0, neginf=0.0)
        clipping_score = float(np.mean(np.abs(mono) > 0.98))
        clipping_score = np.nan_to_num(clipping_score, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "artifact_scores": [clipping_score, 0.0, 0.0, rms],
            "artifact_types": ["click", "clip", "hum", "noise"],
            "time_ranges": [],
            "mode": "dsp_fallback",
        }

    def detect_artifacts(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Artefakt-Detection für ein Audiosignal.

        Args:
            audio: Audiodaten (Mono/Stereo, float32)
            sr: Sample Rate (muss 48000 Hz sein)

        Returns:
            Dict mit Artefakt-Typen, Scores, Zeitbereichen
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        # DSP-Fallback wenn ML nicht verfügbar
        if not _TORCH_AVAILABLE or self.model is None:
            return self._dsp_fallback_detect(audio, sr)

        # ML-Inferenz auf CPU (CPU-only Policy)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = torch.tensor(audio, dtype=torch.float32)  # CPU (kein .to(device) nötig — default cpu)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)  # Mono
        waveform = waveform.unsqueeze(0)  # Batch
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
        # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during TorchScript inference
        _plm_ad = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_ad

            _plm_ad = _get_plm_ad()
            _plm_ad.set_active(self._BUDGET_NAME, True)
        except Exception:
            logger.warning("artifact_detection_plugin.py::erkennen_artifacts Ersatzpfad", exc_info=True)
        try:
            with torch.no_grad():
                output = self.model(mel)
        finally:
            if _plm_ad is not None:
                try:
                    _plm_ad.set_active(self._BUDGET_NAME, False)
                except Exception:
                    logger.warning("artifact_detection_plugin.py::erkennen_artifacts Ersatzpfad", exc_info=True)
        raw = np.nan_to_num(output.numpy().tolist(), nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "artifact_scores": list(raw) if not isinstance(raw, list) else raw,
            "artifact_types": ["click", "clip", "hum", "noise"],
            "time_ranges": [],
            "mode": "ml",
        }

    def feedback(self, user_feedback: dict[str, Any]) -> None:
        """Strukturiertes Feedback-Logging für kontinuierliche Modell-Verbesserung.

        Alle numerischen Werte werden auf NaN/Inf geprüft und bereinigt.
        Das Log-Entry wird als strukturierter JSON-kompatibler Dict geloggt,
        damit es für zukünftige Active-Learning-Pipelines maschinell auswertbar ist.
        """
        import math
        import time

        sanitized: dict[str, Any] = {}
        for k, v in user_feedback.items():
            if isinstance(v, float) and not math.isfinite(v):
                sanitized[k] = None  # NaN/Inf → None (JSON-sicher)
            else:
                sanitized[k] = v
        sanitized.setdefault("_timestamp", time.time())
        sanitized.setdefault("_mode", "ml" if self.model is not None else "dsp_fallback")
        logger.info("ArtifactDetectionPlugin.feedback: %s", sanitized)


# Beispiel für API-Nutzung
if __name__ == "__main__":
    from backend.file_import import load_audio_file

    _res = load_audio_file("audio_examples/example.wav")
    audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]
    plugin = ArtifactDetectionPlugin("models/artifact_detector.pt")
    result = plugin.detect_artifacts(audio, sr)
    logger.debug(result)
