"""
Integration tests for ONNX Plugin Manager
=========================================

Tests the complete plugin integration:
- Model registry loading
- ONNX model loading
- Plugin processing pipeline
- Statistics and monitoring
"""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from backend.core.onnx.plugin_manager import ONNXPluginManager, load_model


@pytest.mark.unit
class TestONNXPluginManager:
    """Test suite for ONNXPluginManager."""

    @pytest.fixture
    def mock_registry(self, tmp_path):
        """Create a mock model registry."""
        registry = {
            "models": {
                "test_model": {
                    "name": "Test Model",
                    "type": "denoising",
                    "task": "noise_reduction",
                    "base_dir": "models/test",
                    "onnx_models": [
                        {
                            "name": "main",
                            "path": str(tmp_path / "test_model.onnx"),
                            "input_shape": [1, 1, -1],
                            "output_shape": [1, 1, -1],
                            "sample_rate": 16000,
                            "opset_version": 14,
                            "quantized": False,
                            "quantized_path": str(tmp_path / "test_model_int8.onnx"),
                            "model_size_mb": 10.5,
                            "expected_speedup": 1.8,
                        }
                    ],
                    "optimization": {
                        "onnx_runtime_enabled": True,
                        "quantization_enabled": False,
                        "quantization_type": "dynamic",
                        "target_speedup": "1.5-2.0x",
                        "cpu_threads": 4,
                    },
                    "plugin": "plugins.test_plugin.TestPlugin",
                    "docker_image": "aurik-plugin-runtime:latest",
                    "notes": "Test model for integration testing",
                }
            },
            "statistics": {"total_models": 1, "total_onnx_files": 1},
        }

        registry_path = tmp_path / "model_registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        return registry_path

    @pytest.fixture
    def mock_onnx_model(self):
        """Create a mock ONNX model."""
        model = Mock()
        model.process = Mock(return_value=np.zeros((16000,)))
        model.get_statistics = Mock(return_value={"total_inferences": 1, "total_inference_time_ms": 10.5})
        return model

    def test_initialization(self, mock_registry):
        """Test ONNXPluginManager initialization."""
        manager = ONNXPluginManager(registry_path=str(mock_registry))

        assert manager.registry_path == mock_registry
        assert len(manager.registry["models"]) == 1
        assert manager.total_inferences == 0
        assert manager.total_inference_time_ms == 0.0

    def test_get_available_models(self, mock_registry):
        """Test getting available model list."""
        manager = ONNXPluginManager(registry_path=str(mock_registry))

        models = manager.get_available_models()
        assert len(models) == 1
        assert "test_model" in models

    def test_get_model_config(self, mock_registry):
        """Test getting model configuration."""
        manager = ONNXPluginManager(registry_path=str(mock_registry))

        config = manager.get_model_config("test_model")
        assert config is not None
        assert config["name"] == "Test Model"
        assert config["type"] == "denoising"

        # Test non-existent model
        config = manager.get_model_config("nonexistent")
        assert config is None

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    def test_load_model(self, mock_model_class, mock_registry, tmp_path, mock_onnx_model):
        """Test loading a model."""
        # Create dummy ONNX file
        onnx_file = tmp_path / "test_model.onnx"
        onnx_file.write_bytes(b"dummy onnx")

        # Setup mock
        mock_model_class.return_value = mock_onnx_model

        manager = ONNXPluginManager(registry_path=str(mock_registry))
        success = manager.load_model("test_model")

        assert success
        assert manager.is_loaded("test_model")
        assert "test_model" in manager.loaded_models

    def test_load_nonexistent_model(self, mock_registry):
        """Test loading non-existent model."""
        manager = ONNXPluginManager(registry_path=str(mock_registry))

        success = manager.load_model("nonexistent")
        assert not success
        assert not manager.is_loaded("nonexistent")

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    def test_unload_model(self, mock_model_class, mock_registry, tmp_path, mock_onnx_model):
        """Test unloading a model."""
        # Create dummy ONNX file
        onnx_file = tmp_path / "test_model.onnx"
        onnx_file.write_bytes(b"dummy onnx")

        # Setup mock
        mock_model_class.return_value = mock_onnx_model

        manager = ONNXPluginManager(registry_path=str(mock_registry))
        manager.load_model("test_model")

        assert manager.is_loaded("test_model")

        success = manager.unload_model("test_model")
        assert success
        assert not manager.is_loaded("test_model")

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    def test_process(self, mock_model_class, mock_registry, tmp_path, mock_onnx_model):
        """Test processing audio with a loaded model."""
        # Create dummy ONNX file
        onnx_file = tmp_path / "test_model.onnx"
        onnx_file.write_bytes(b"dummy onnx")

        # Setup mock
        mock_model_class.return_value = mock_onnx_model

        manager = ONNXPluginManager(registry_path=str(mock_registry))
        manager.load_model("test_model")

        # Process audio
        audio = np.random.randn(16000).astype(np.float32)
        output = manager.process("test_model", audio)

        assert output is not None
        assert manager.total_inferences == 1

    def test_process_without_loading(self, mock_registry):
        """Test processing without loading model first."""
        manager = ONNXPluginManager(registry_path=str(mock_registry))

        audio = np.random.randn(16000).astype(np.float32)
        output = manager.process("test_model", audio)

        assert output is None

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    def test_get_statistics_global(self, mock_model_class, mock_registry, tmp_path, mock_onnx_model):
        """Test getting global statistics."""
        # Create dummy ONNX file
        onnx_file = tmp_path / "test_model.onnx"
        onnx_file.write_bytes(b"dummy onnx")

        # Setup mock
        mock_model_class.return_value = mock_onnx_model

        manager = ONNXPluginManager(registry_path=str(mock_registry))
        manager.load_model("test_model")

        stats = manager.get_statistics()

        assert "total_models_loaded" in stats
        assert stats["total_models_loaded"] == 1
        assert "total_inferences" in stats
        assert "fallback_stats" in stats
        assert "models" in stats

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    def test_get_model_info(self, mock_model_class, mock_registry, tmp_path, mock_onnx_model):
        """Test getting model information."""
        # Create dummy ONNX file
        onnx_file = tmp_path / "test_model.onnx"
        onnx_file.write_bytes(b"dummy onnx")

        # Setup mock
        mock_model_class.return_value = mock_onnx_model

        manager = ONNXPluginManager(registry_path=str(mock_registry))

        # Get info before loading
        info = manager.get_model_info("test_model")
        assert info is not None
        assert info["model_id"] == "test_model"
        assert info["loaded"] == False

        # Get info after loading
        manager.load_model("test_model")
        info = manager.get_model_info("test_model")
        assert info["loaded"] == True  # type: ignore[index]
        assert "statistics" in info  # type: ignore[operator]

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    def test_load_all_models(self, mock_model_class, mock_registry, tmp_path, mock_onnx_model):
        """Test loading all models from registry."""
        # Create dummy ONNX file
        onnx_file = tmp_path / "test_model.onnx"
        onnx_file.write_bytes(b"dummy onnx")

        # Setup mock
        mock_model_class.return_value = mock_onnx_model

        manager = ONNXPluginManager(registry_path=str(mock_registry))
        results = manager.load_all_models()

        assert len(results) == 1
        assert results["test_model"] == True

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    def test_unload_all_models(self, mock_model_class, mock_registry, tmp_path, mock_onnx_model):
        """Test unloading all models."""
        # Create dummy ONNX file
        onnx_file = tmp_path / "test_model.onnx"
        onnx_file.write_bytes(b"dummy onnx")

        # Setup mock
        mock_model_class.return_value = mock_onnx_model

        manager = ONNXPluginManager(registry_path=str(mock_registry))
        manager.load_all_models()

        count = manager.unload_all_models()
        assert count == 1
        assert len(manager.loaded_models) == 0


class TestConvenienceFunctions:
    """Test convenience functions for quick operations."""

    @pytest.fixture
    def mock_registry(self, tmp_path):
        """Create a mock model registry."""
        registry = {
            "models": {
                "quick_test": {
                    "name": "Quick Test Model",
                    "type": "denoising",
                    "task": "noise_reduction",
                    "base_dir": "models/test",
                    "onnx_models": [
                        {
                            "name": "main",
                            "path": str(tmp_path / "quick_test.onnx"),
                            "input_shape": [1, 1, -1],
                            "output_shape": [1, 1, -1],
                            "sample_rate": 16000,
                            "opset_version": 14,
                            "quantized": False,
                            "quantized_path": str(tmp_path / "quick_test_int8.onnx"),
                            "model_size_mb": 5.0,
                            "expected_speedup": 1.5,
                        }
                    ],
                    "optimization": {},
                    "plugin": "plugins.test_plugin.TestPlugin",
                }
            }
        }

        registry_path = tmp_path / "model_registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        # Set environment variable for registry path
        import os

        os.environ["ONNX_REGISTRY_PATH"] = str(registry_path)

        return registry_path

    @patch("backend.core.onnx.plugin_manager.OptimizedONNXModel")
    @patch("backend.core.onnx.plugin_manager.ONNXPluginManager.__init__")
    def test_load_model_convenience(self, mock_init, mock_model_class):
        """Test quick load_model() convenience function."""
        mock_init.return_value = None

        # This test is complex due to patching, simplified version
        manager = load_model("test_model")
        assert isinstance(manager, ONNXPluginManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
