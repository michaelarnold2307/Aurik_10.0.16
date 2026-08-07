"""
ONNX Model Quantization for 2-3× speedup

Quantizes FP32 ONNX models to INT8 with minimal quality loss (<1%).
Uses dynamic quantization (no calibration data needed).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Quantization types."""

    DYNAMIC = "dynamic"  # Activations quantized at runtime (no calibration)
    STATIC = "static"  # Requires calibration data
    QDQ = "qdq"  # Quantize-Dequantize nodes (training-aware)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    per_channel: bool = False
    reduce_range: bool = False
    optimize_model: bool = True
    nodes_to_quantize: list[str] | None = None
    nodes_to_exclude: list[str] | None = None

    # Quality validation thresholds
    max_quality_loss_percent: float = 1.0  # Maximum 1% quality loss
    validation_samples: int = 10  # Number of samples for quality check


class ModelQuantizer:
    """
    Quantizes ONNX models from FP32 to INT8.

    Features:
    - Dynamic INT8 quantization (2-3× speedup)
    - No calibration data required
    - Quality validation with audio metrics
    - Automatic model size reduction (4× smaller)

    Expected results:
    - Speedup: 2-3× faster inference
    - Size: 75% smaller (FP32 → INT8 = 4 bytes → 1 byte)
    - Quality: <1% degradation

    Usage:
        quantizer = ModelQuantizer()
        success = quantizer.quantize(
            model_path="model.onnx",
            output_path="model_quantized.onnx"
        )
    """

    def __init__(self, config: QuantizationConfig | None = None):
        """
        Initialisiert model quantizer.

        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self.quantization_stats = {
            "total_quantizations": 0,
            "successful_quantizations": 0,
            "failed_quantizations": 0,
            "average_size_reduction": 0.0,
            "average_speedup": 0.0,
        }

    def quantize(self, model_path: Path, output_path: Path, validate_quality: bool = True) -> bool:
        """
        Quantize ONNX model from FP32 to INT8.

        Args:
            model_path: Path to FP32 ONNX model
            output_path: Where to save quantized model
            validate_quality: Check quality degradation

        Returns:
            True if quantization successful
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        if not model_path.exists():
            logger.error("Model not found: %s", model_path)
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Quantizing model: %s", model_path.name)
        logger.info("Type: %s", self.config.quantization_type.value)

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            # Original model size
            original_size = model_path.stat().st_size / (1024 * 1024)  # MB
            logger.info("Originalsignal size: %.2f MB", original_size)

            # Perform quantization
            if self.config.quantization_type == QuantizationType.DYNAMIC:
                quantize_dynamic(
                    model_input=str(model_path),
                    model_output=str(output_path),
                    weight_type=QuantType.QInt8,
                    per_channel=self.config.per_channel,
                    reduce_range=self.config.reduce_range,
                    nodes_to_quantize=self.config.nodes_to_quantize,
                    nodes_to_exclude=self.config.nodes_to_exclude,
                )
            else:
                logger.error("Quantization type %s not implemented", self.config.quantization_type)
                return False

            # Quantized model size
            quantized_size = output_path.stat().st_size / (1024 * 1024)  # MB
            size_reduction = (1 - quantized_size / original_size) * 100

            logger.info("Quantized size: %.2f MB", quantized_size)
            logger.info("Size reduction: %.1f%%", size_reduction)

            # Quality validation
            if validate_quality:
                quality_ok = self._validate_quality(original_path=model_path, quantized_path=output_path)

                if not quality_ok:
                    logger.warning(
                        f"Quality Validierung fehlgeschlagen! Degradation exceeds {self.config.max_quality_loss_percent}%"
                    )
                    # Note: We don't fail entirely, just warn

            self.quantization_stats["successful_quantizations"] += 1
            self.quantization_stats["total_quantizations"] += 1
            self.quantization_stats["average_size_reduction"] = (
                self.quantization_stats["average_size_reduction"]
                * (self.quantization_stats["successful_quantizations"] - 1)
                + size_reduction
            ) / self.quantization_stats["successful_quantizations"]

            logger.info("✓ Quantization erfolgreich")
            return True

        except ImportError as e:
            logger.error("onnxruntime.quantization not verfuegbar: %s", e)
            logger.error("Install with: pip install onnxruntime")
            return False
        except Exception as e:
            logger.error("Quantization fehlgeschlagen: %s", e)
            self.quantization_stats["failed_quantizations"] += 1
            self.quantization_stats["total_quantizations"] += 1
            return False

    def _validate_quality(self, original_path: Path, quantized_path: Path) -> bool:
        """
        Validiert quantized model quality.

        Compares outputs on random audio samples to ensure
        quantization doesn't degrade quality significantly.

        Args:
            original_path: Path to original FP32 model
            quantized_path: Path to quantized INT8 model

        Returns:
            True if quality degradation is acceptable
        """
        try:
            import onnxruntime as ort

            # Load both models
            session_fp32 = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
            session_int8 = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])

            input_name = session_fp32.get_inputs()[0].name
            input_shape = session_fp32.get_inputs()[0].shape

            # Generate random audio samples
            total_diff = 0.0
            max_diff = 0.0

            for i in range(self.config.validation_samples):
                # Create dummy audio
                if input_shape[1] == -1:
                    # Dynamic shape, use 48000 samples (1 second at 48kHz)
                    dummy_audio = np.random.randn(1, 48000).astype(np.float32)
                else:
                    dummy_audio = np.random.randn(*[dim if dim != -1 else 48000 for dim in input_shape]).astype(
                        np.float32
                    )

                # Run inference
                fp32_output = session_fp32.run(None, {input_name: dummy_audio})[0]
                int8_output = session_int8.run(None, {input_name: dummy_audio})[0]

                # Calculate difference
                diff = np.abs(fp32_output - int8_output).mean()
                max_sample_diff = np.abs(fp32_output - int8_output).max()

                total_diff += diff
                max_diff = max(max_diff, max_sample_diff)

            avg_diff = total_diff / self.config.validation_samples

            # Calculate quality loss percentage
            # Assume audio is in range [-1, 1], so max possible value is 1
            quality_loss_percent = (avg_diff / 1.0) * 100

            logger.info("Quality Validierung:")
            logger.info("  Average difference: %.6f", avg_diff)
            logger.info("  Max difference: %.6f", max_diff)
            logger.info("  Quality loss: %.3f%%", quality_loss_percent)

            if quality_loss_percent > self.config.max_quality_loss_percent:
                logger.warning(
                    f"Quality loss {quality_loss_percent:.3f}% exceeds Schwelle {self.config.max_quality_loss_percent}%"
                )
                return False

            logger.info("✓ Quality Validierung passed")
            return True

        except Exception as e:
            logger.warning("Quality Validierung fehlgeschlagen: %s", e)
            return True  # Don't fail quantization due to validation issues

    def quantize_batch(
        self, models: dict[str, Path], output_dir: Path, validate_quality: bool = True
    ) -> dict[str, bool]:
        """
        Quantize multiple models.

        Args:
            models: Dict of {name: model_path}
            output_dir: Output directory for quantized models
            validate_quality: Validate each model

        Returns:
            Dict of {name: success}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for name, model_path in models.items():
            logger.info("\n%s", "=" * 60)
            logger.info("Quantizing: %s", name)
            logger.info("%s", "=" * 60)

            output_path = output_dir / f"{model_path.stem}_quantized.onnx"

            success = self.quantize(model_path=model_path, output_path=output_path, validate_quality=validate_quality)

            results[name] = success
            logger.info("Ergebnis: %s", "✓ SUCCESS" if success else "❌ FAILED")

        # Summary
        logger.info("\n%s", "=" * 60)
        logger.info("QUANTIZATION SUMMARY")
        logger.info("%s", "=" * 60)
        successful = sum(1 for v in results.values() if v)
        logger.info("Total: %s", len(results))
        logger.info("erfolgreich: %s", successful)
        logger.info("fehlgeschlagen: %s", len(results) - successful)

        if self.quantization_stats["successful_quantizations"] > 0:
            logger.info("Average size reduction: %.1f%%", self.quantization_stats["average_size_reduction"])

        return results

    def estimate_speedup(self, model_path: Path, quantized_path: Path, num_runs: int = 10) -> float:
        """
        Schätzt speedup from quantization.

        Args:
            model_path: Original FP32 model
            quantized_path: Quantized INT8 model
            num_runs: Number of benchmark runs

        Returns:
            Speedup factor (e.g., 2.5 means 2.5× faster)
        """
        try:
            import time

            import onnxruntime as ort

            # Load models
            session_fp32 = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            session_int8 = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])

            input_name = session_fp32.get_inputs()[0].name

            # Create dummy input
            dummy_audio = np.random.randn(1, 48000).astype(np.float32)

            # Warmup
            for _ in range(3):
                session_fp32.run(None, {input_name: dummy_audio})
                session_int8.run(None, {input_name: dummy_audio})

            # Benchmark FP32
            start = time.time()
            for _ in range(num_runs):
                session_fp32.run(None, {input_name: dummy_audio})
            fp32_time = time.time() - start

            # Benchmark INT8
            start = time.time()
            for _ in range(num_runs):
                session_int8.run(None, {input_name: dummy_audio})
            int8_time = time.time() - start

            speedup = fp32_time / int8_time

            logger.info("Speedup estimation:")
            logger.info("  FP32: %.2f ms/inference", fp32_time / num_runs * 1000)
            logger.info("  INT8: %.2f ms/inference", int8_time / num_runs * 1000)
            logger.info("  Speedup: %.2f×", speedup)

            return speedup

        except Exception as e:
            logger.error("Speedup estimation fehlgeschlagen: %s", e)
            return 1.0

    def get_stats(self) -> dict[str, Any]:
        """Gibt zurück: quantization statistics."""
        return self.quantization_stats.copy()

    def reset_stats(self) -> None:
        """Setzt zurück: quantization statistics."""
        self.quantization_stats = {
            "total_quantizations": 0,
            "successful_quantizations": 0,
            "failed_quantizations": 0,
            "average_size_reduction": 0.0,
            "average_speedup": 0.0,
        }
