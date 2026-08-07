"""
ONNX Plugin Manager for AURIK v8
================================

Central manager for all ONNX-accelerated ML model plugins.
Provides unified interface for loading, managing, and executing ONNX models.

Features:
- Load models from model_registry.json
- Automatic ONNX Runtime optimization
- Fallback to PyTorch if ONNX fails
- Performance monitoring and statistics
- Model lifecycle management (load, unload, reload)

Usage:
    from backend.core.onnx.plugin_manager import ONNXPluginManager

    # Initialize manager
    manager = ONNXPluginManager()

    # Load specific model
    manager.load_model("deepfilternet_v3_ii")

    # Process audio
    output = manager.process("deepfilternet_v3_ii", input_audio)

    # Get statistics
    stats = manager.get_statistics()
"""

import json
import logging
from pathlib import Path

import numpy as np

from backend.core.onnx.fallback import FallbackManager, ONNXModelWithFallback
from backend.core.onnx.runtime import ONNXProvider, OptimizedONNXModel

logger = logging.getLogger(__name__)


class ONNXPluginManager:
    """
    Central manager for all ONNX-accelerated ML model plugins.

    Manages the lifecycle of ONNX models:
    - Loading from registry
    - ONNX Runtime optimization
    - Fallback to PyTorch
    - Performance monitoring
    """

    def __init__(
        self, registry_path: str = "backend/core/onnx/model_registry.json", provider: ONNXProvider = ONNXProvider.CPU
    ):
        """
        Initialisiert the ONNX Plugin Manager.

        Args:
            registry_path: Path to model_registry.json
            provider: ONNX execution provider (CPU, CUDA, TensorRT)
        """
        self.registry_path = Path(registry_path)
        self.provider = provider

        # Load registry
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")

        with open(self.registry_path) as f:
            self.registry = json.load(f)

        # Model storage
        self.loaded_models: dict[str, ONNXModelWithFallback] = {}
        self.model_configs: dict[str, dict] = {}

        # Fallback manager
        self.fallback_manager = FallbackManager()

        # Statistics
        self.total_inferences = 0
        self.total_inference_time_ms = 0.0

        logger.info("ONNX Plugin Manager initialisiert with %s models", len(self.registry["models"]))

    def get_available_models(self) -> list[str]:
        """
        Gibt zurück: list of all available model IDs.

        Returns:
            List of model IDs
        """
        return list(self.registry["models"].keys())

    def get_model_config(self, model_id: str) -> dict | None:
        """
        Gibt zurück: configuration for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model configuration dict or None if not found
        """
        return self.registry["models"].get(model_id)  # type: ignore[no-any-return]

    def load_model(self, model_id: str, use_quantized: bool = False, force_reload: bool = False) -> bool:
        """
        Lädt an ONNX model from the registry.

        Args:
            model_id: Model identifier (e.g., "deepfilternet_v3_ii")
            use_quantized: If True, load INT8 quantized version
            force_reload: If True, reload even if already loaded

        Returns:
            True if loaded successfully, False otherwise
        """
        # Defensive init (z.B. wenn __init__ in Tests gemockt wurde)
        if not hasattr(self, "loaded_models"):
            self.loaded_models = {}
        if not hasattr(self, "model_configs"):
            self.model_configs = {}
        if not hasattr(self, "registry"):
            self.registry = {"models": {}}

        # Check if already loaded
        if model_id in self.loaded_models and not force_reload:
            logger.debug("Model %s already geladen", model_id)
            return True

        # Get model config
        model_config = self.get_model_config(model_id)
        if not model_config:
            logger.error("Model %s not found in registry", model_id)
            return False

        # Store config
        self.model_configs[model_id] = model_config

        # Load all ONNX models for this plugin
        onnx_models = {}
        for onnx_config in model_config["onnx_models"]:
            # Choose FP32 or INT8
            if use_quantized and onnx_config.get("quantized", False):
                model_path = Path(onnx_config["quantized_path"])
                logger.info("lade quantized model: %s", model_path)
            else:
                model_path = Path(onnx_config["path"])
                logger.info("lade FP32 model: %s", model_path)

            # Check if model exists
            if not model_path.exists():
                logger.warning("Model file not found: %s", model_path)
                continue

            try:
                # Create ONNX model
                onnx_model = OptimizedONNXModel(
                    model_path=model_path,
                    model_type=model_config["type"],
                    sample_rate=onnx_config["sample_rate"],
                    providers=[self.provider.value],
                )

                onnx_models[onnx_config["name"]] = onnx_model
                logger.info("geladen ONNX model: %s/%s", model_id, onnx_config["name"])

            except Exception as e:
                logger.error("konnte nicht laden %s/%s: %s", model_id, onnx_config["name"], e)
                continue

        if not onnx_models:
            logger.error("No ONNX models geladen for %s", model_id)
            return False

        # Multi-model-Plugins (z.B. DeepFilterNet: encoder + decoder + erb_dec)
        # werden vollständig geladen; alle Teilmodelle im Config-Dict hinterlegt.
        primary_model = next(iter(onnx_models.values()))
        if len(onnx_models) > 1:
            model_config["_all_onnx_models"] = onnx_models
            logger.info(
                "Multi-model plugin '%s': %d ONNX-Modelle geladen (%s).",
                model_id,
                len(onnx_models),
                ", ".join(onnx_models.keys()),
            )

        # PyTorch-Fallback: CPU-only (§9.5) — aus model_registry.json 'pytorch_model.path'
        pytorch_fallback = None
        pytorch_cfg = model_config.get("pytorch_model") or {}
        pytorch_path_str = pytorch_cfg.get("path", "")
        if pytorch_path_str:
            try:
                import torch as _torch

                _pt_path = Path(pytorch_path_str)
                if _pt_path.exists():
                    pytorch_fallback = _torch.jit.load(str(_pt_path), map_location="cpu")
                    pytorch_fallback.eval()
                    logger.info("PyTorch-Ersatzpfad geladen (CPU-only): %s", _pt_path.name)
                else:
                    logger.debug("PyTorch-Ersatzpfad-Pfad nicht gefunden: %s", pytorch_path_str)
            except Exception as _pt_err:
                logger.debug("PyTorch-Ersatzpfad nicht geladen (%s) — nur ONNX aktiv.", _pt_err)

        wrapped_model = ONNXModelWithFallback(
            name=model_id,
            onnx_model=primary_model,
            pytorch_model=pytorch_fallback,
            fallback_manager=self.fallback_manager,
        )

        self.loaded_models[model_id] = wrapped_model
        logger.info("erfolgreich geladen model: %s", model_id)
        return True

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_id: Model identifier

        Returns:
            True if unloaded successfully
        """
        if model_id not in self.loaded_models:
            logger.warning("Model %s not geladen", model_id)
            return False

        del self.loaded_models[model_id]
        del self.model_configs[model_id]
        logger.info("Unloaded model: %s", model_id)
        return True

    def is_loaded(self, model_id: str) -> bool:
        """
        Prüft if a model is loaded.

        Args:
            model_id: Model identifier

        Returns:
            True if loaded
        """
        return model_id in self.loaded_models

    def process(self, model_id: str, audio: np.ndarray, **kwargs) -> np.ndarray | None:
        """
        Verarbeitet audio with a loaded model.

        Args:
            model_id: Model identifier
            audio: Input audio (numpy array)
            **kwargs: Additional model-specific parameters

        Returns:
            Processed audio or None if failed
        """
        if model_id not in self.loaded_models:
            logger.error("Model %s not geladen. Call laden_model() first.", model_id)
            return None

        try:
            model = self.loaded_models[model_id]
            output = model.process(audio)

            # Update statistics
            self.total_inferences += 1
            stats = model.get_stats()
            if "onnx_inference_count" in stats:
                # Has inference stats
                pass  # Stats are self-contained in model

            return output  # type: ignore[no-any-return]

        except Exception as e:
            logger.error("Error processing with %s: %s", model_id, e)
            return None

    def get_statistics(self, model_id: str | None = None) -> dict:
        """
        Gibt zurück: performance statistics.

        Args:
            model_id: If provided, get stats for specific model. Otherwise, global stats.

        Returns:
            Statistics dictionary
        """
        if model_id:
            if model_id not in self.loaded_models:
                return {}
            return self.loaded_models[model_id].get_stats()

        # Global statistics
        stats = {
            "total_models_loaded": len(self.loaded_models),
            "total_inferences": self.total_inferences,
            "total_inference_time_ms": self.total_inference_time_ms,
            "average_inference_time_ms": (
                self.total_inference_time_ms / self.total_inferences if self.total_inferences > 0 else 0
            ),
            "fallback_stats": self.fallback_manager.get_stats(),
            "models": {},
        }

        # Per-model statistics
        for model_id, model in self.loaded_models.items():
            stats["models"][model_id] = model.get_stats()  # type: ignore[index]

        return stats

    def get_fallback_history(self, model_id: str | None = None, limit: int = 10) -> list[dict]:
        """
        Gibt zurück: fallback event history.

        Args:
            model_id: Filter by model ID
            limit: Maximum number of events to return

        Returns:
            List of fallback events
        """
        events = self.fallback_manager.get_fallback_history(limit=limit)

        if model_id:
            events = [e for e in events if e.model_name == model_id]

        return [
            {
                "timestamp": e.timestamp,
                "model_name": e.model_name,
                "reason": e.reason.value,
                "error_message": e.error_message,
                "recovered": e.recovered,
            }
            for e in events
        ]

    def load_all_models(self, use_quantized: bool = False) -> dict[str, bool]:
        """
        Lädt all models from the registry.

        Args:
            use_quantized: If True, load INT8 quantized versions

        Returns:
            Dictionary mapping model_id to load success status
        """
        results = {}

        for model_id in self.get_available_models():
            success = self.load_model(model_id, use_quantized=use_quantized)
            results[model_id] = success

        successful = sum(results.values())
        total = len(results)
        logger.info("geladen %s/%s models", successful, total)

        return results

    def unload_all_models(self) -> int:
        """
        Unload all loaded models.

        Returns:
            Number of models unloaded
        """
        count = len(self.loaded_models)
        self.loaded_models.clear()
        self.model_configs.clear()
        logger.info("Unloaded %s models", count)
        return count

    def get_model_info(self, model_id: str) -> dict | None:
        """
        Gibt zurück: detailed information about a model.

        Args:
            model_id: Model identifier

        Returns:
            Model information dictionary
        """
        config = self.get_model_config(model_id)
        if not config:
            return None

        info = {
            "model_id": model_id,
            "name": config["name"],
            "type": config["type"],
            "task": config["task"],
            "docker_image": config["docker_image"],
            "onnx_models": config["onnx_models"],
            "optimization": config["optimization"],
            "plugin": config["plugin"],
            "notes": config.get("notes", ""),
            "loaded": self.is_loaded(model_id),
        }

        if self.is_loaded(model_id):
            info["statistics"] = self.get_statistics(model_id)

        return info


# Convenience functions for common operations


def load_model(model_id: str, use_quantized: bool = False) -> ONNXPluginManager:
    """
    Quick load a single model.

    Args:
        model_id: Model identifier
        use_quantized: If True, load INT8 quantized version

    Returns:
        ONNXPluginManager instance with model loaded
    """
    manager = ONNXPluginManager()
    manager.load_model(model_id, use_quantized=use_quantized)
    return manager


def process_audio(model_id: str, audio: np.ndarray, use_quantized: bool = False) -> np.ndarray | None:
    """
    Quick process audio with a model.

    Args:
        model_id: Model identifier
        audio: Input audio
        use_quantized: If True, use INT8 quantized version

    Returns:
        Processed audio or None
    """
    manager = load_model(model_id, use_quantized=use_quantized)
    return manager.process(model_id, audio)
