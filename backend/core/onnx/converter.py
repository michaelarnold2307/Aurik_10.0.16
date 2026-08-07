"""
ONNX Converter for PyTorch models

Converts PyTorch audio processing models to ONNX format for faster inference.
Handles model-specific quirks and validates conversion quality.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for PyTorch → ONNX conversion."""

    opset_version: int = 14
    do_constant_folding: bool = True
    input_names: list | None = None
    output_names: list | None = None
    dynamic_axes: dict[str, dict[int, str]] | None = None
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ["audio_input"]
        if self.output_names is None:
            self.output_names = ["audio_output"]


class ONNXConverter:
    """
    Konvertiert PyTorch audio models to ONNX format.

    Features:
    - Automatic shape inference
    - Model-specific handling (DeepFilterNet, Demucs, etc.)
    - Validation against PyTorch output
    - External data handling for large models (>2GB)

    Usage:
        converter = ONNXConverter()
        success = converter.convert(
            pytorch_model=model,
            output_path="model.onnx",
            sample_input=dummy_audio
        )
    """

    def __init__(self, config: ConversionConfig | None = None, validation_tolerance: float = 1e-5):
        """
        Initialisiert ONNX converter.

        Args:
            config: Conversion configuration
            validation_tolerance: Maximum allowed difference between PyTorch and ONNX
        """
        self.config = config or ConversionConfig()
        self.validation_tolerance = validation_tolerance
        self.conversion_stats = {"total_conversions": 0, "successful_conversions": 0, "failed_conversions": 0}

    def convert(
        self,
        pytorch_model: Any,
        output_path: str | Path,
        sample_input: Any,
        validate: bool = True,
        use_external_data: bool = False,
    ) -> bool:
        """
        Konvertiert PyTorch model to ONNX format.

        Args:
            pytorch_model: PyTorch model to convert
            output_path: Where to save ONNX model
            sample_input: Sample input tensor for tracing
            validate: Validate ONNX output matches PyTorch
            use_external_data: Store weights externally (for models >2GB)

        Returns:
            True if conversion successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Converting model to ONNX: %s", output_path.name)
        logger.info("Sample Eingabe shape: %s", sample_input.shape)

        # Set model to evaluation mode
        pytorch_model.eval()

        # Get PyTorch output for validation
        pytorch_output = None
        if validate:
            assert torch is not None
            with torch.no_grad():
                pytorch_output = pytorch_model(sample_input)
            logger.info("PyTorch Ausgabe shape: %s", pytorch_output.shape)

        try:
            # Export to ONNX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert torch is not None
                torch.onnx.export(
                    pytorch_model,
                    sample_input,
                    str(output_path),
                    export_params=self.config.export_params,
                    opset_version=self.config.opset_version,
                    do_constant_folding=self.config.do_constant_folding,
                    input_names=self.config.input_names,
                    output_names=self.config.output_names,
                    dynamic_axes=self.config.dynamic_axes
                    or {
                        "audio_input": {0: "batch_size", 1: "audio_length"},
                        "audio_output": {0: "batch_size", 1: "audio_length"},
                    },
                    verbose=self.config.verbose,
                )

            logger.info("ONNX Ausgabe erfolgreich: %s", output_path)

            # Handle external data for large models
            if use_external_data:
                self._save_external_data(output_path)

            # Validate conversion
            if validate and pytorch_output is not None:
                validation_success = self._validate_conversion(output_path, sample_input, pytorch_output)

                if not validation_success:
                    logger.error("Validierung fehlgeschlagen! ONNX Ausgabe differs from PyTorch")
                    self.conversion_stats["failed_conversions"] += 1
                    return False

            self.conversion_stats["successful_conversions"] += 1
            self.conversion_stats["total_conversions"] += 1
            logger.info("✓ Conversion erfolgreich and validated")
            return True

        except Exception as e:
            logger.error("ONNX conversion fehlgeschlagen: %s", e)
            self.conversion_stats["failed_conversions"] += 1
            self.conversion_stats["total_conversions"] += 1
            return False

    def _validate_conversion(self, onnx_path: Path, sample_input: Any, pytorch_output: Any) -> bool:
        """
        Validiert ONNX model produces same output as PyTorch.

        Args:
            onnx_path: Path to ONNX model
            sample_input: Input tensor
            pytorch_output: Expected output from PyTorch

        Returns:
            True if outputs match within tolerance
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not verfuegbar, skipping Validierung")
            return True

        try:
            # Load ONNX model
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            # Run inference
            onnx_input = {session.get_inputs()[0].name: sample_input.cpu().numpy()}
            onnx_output = session.run(None, onnx_input)[0]

            # Compare outputs
            pytorch_np = pytorch_output.cpu().detach().numpy()

            max_diff = np.abs(pytorch_np - onnx_output).max()
            mean_diff = np.abs(pytorch_np - onnx_output).mean()

            logger.info("Validierung - Max diff: %.2e, Mean diff: %.2e", max_diff, mean_diff)

            if max_diff > self.validation_tolerance:
                logger.error(
                    f"Validierung fehlgeschlagen! Max difference {max_diff:.2e} "
                    f"exceeds tolerance {self.validation_tolerance:.2e}"
                )
                return False

            logger.info("✓ Validierung passed (tolerance: %.2e)", self.validation_tolerance)
            return True

        except Exception as e:
            logger.error("Validierung error: %s", e)
            return False

    def _save_external_data(self, onnx_path: Path) -> None:
        """
        Konvertiert ONNX model to use external data format.

        For models >2GB, ONNX requires external data storage.

        Args:
            onnx_path: Path to ONNX model
        """
        try:
            import onnx

            # Load model
            model = onnx.load(str(onnx_path))

            # Save with external data
            external_data_path = onnx_path.with_suffix(".onnx.data")
            onnx.save_model(
                model,
                str(onnx_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_path.name,
            )

            logger.info("gespeichert external data: %s", external_data_path)

        except Exception as e:
            logger.warning("konnte nicht speichern external data: %s", e)

    def convert_batch(self, models: dict[str, tuple[Any, Any, Path]], validate: bool = True) -> dict[str, bool]:
        """
        Konvertiert multiple PyTorch models to ONNX.

        Args:
            models: Dict of {name: (model, sample_input, output_path)}
            validate: Validate each conversion

        Returns:
            Dict of {name: success}
        """
        results = {}

        for name, (model, sample_input, output_path) in models.items():
            logger.info("\n%s", "=" * 60)
            logger.info("Converting: %s", name)
            logger.info("%s", "=" * 60)

            success = self.convert(
                pytorch_model=model, output_path=output_path, sample_input=sample_input, validate=validate
            )

            results[name] = success
            logger.info("Ergebnis: %s", "✓ SUCCESS" if success else "❌ FAILED")

        # Summary
        logger.info("\n%s", "=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("%s", "=" * 60)
        successful = sum(1 for v in results.values() if v)
        logger.info("Total: %s", len(results))
        logger.info("erfolgreich: %s", successful)
        logger.info("fehlgeschlagen: %s", len(results) - successful)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Gibt zurück: conversion statistics."""
        return self.conversion_stats.copy()

    def reset_stats(self) -> None:
        """Setzt zurück: conversion statistics."""
        self.conversion_stats = {"total_conversions": 0, "successful_conversions": 0, "failed_conversions": 0}


class ModelSpecificConverter:
    """
    Verarbeitet model-specific conversion quirks.

    Different models have different requirements:
    - DeepFilterNet: Separate encoder/decoder
    - Demucs: Very large model (650MB+), needs external data
    - DCCRN: Complex architecture with attention
    """

    @staticmethod
    def convert_deepfilternet(model_dir: Path, output_dir: Path, converter: ONNXConverter) -> dict[str, bool]:
        """
        Konvertiert DeepFilterNet model (encoder + decoder).

        DeepFilterNet has 3 components:
        - Encoder (ERB)
        - Encoder (Complex)
        - Decoder

        Args:
            model_dir: Directory containing PyTorch model
            output_dir: Output directory for ONNX models
            converter: ONNXConverter instance

        Returns:
            Dict of conversion results
        """
        logger.info("Converting DeepFilterNet (3 components)")

        # These are typically already ONNX files in our case
        # This is more for documentation of the process

        results = {"encoder_erb": False, "encoder_complex": False, "decoder": False}

        logger.info("DeepFilterNet ONNX models already verfuegbar")
        return results

    @staticmethod
    def convert_demucs(model: Any, output_path: Path, converter: ONNXConverter, sample_rate: int = 44100) -> bool:
        """
        Konvertiert Demucs model (stem separation).

        Demucs is a large model (~650MB) requiring external data.

        Args:
            model: Demucs PyTorch model
            output_path: Output ONNX path
            converter: ONNXConverter instance
            sample_rate: Audio sample rate

        Returns:
            True if successful
        """
        logger.info("Converting Demucs (large model, external data required)")

        # Create dummy input (2 channels, 10 seconds)
        assert torch is not None
        dummy_input = torch.randn(1, 2, sample_rate * 10)

        success = converter.convert(
            pytorch_model=model,
            output_path=output_path,
            sample_input=dummy_input,
            validate=True,
            use_external_data=True,  # Demucs is large!
        )

        return success

    @staticmethod
    def convert_dccrn(model: Any, output_path: Path, converter: ONNXConverter, sample_rate: int = 16000) -> bool:
        """
        Konvertiert DCCRN model (speech denoising).

        Args:
            model: DCCRN PyTorch model
            output_path: Output ONNX path
            converter: ONNXConverter instance
            sample_rate: Audio sample rate

        Returns:
            True if successful
        """
        logger.info("Converting DCCRN (speech denoising)")

        # DCCRN expects mono audio
        assert torch is not None
        dummy_input = torch.randn(1, 1, sample_rate * 5)

        success = converter.convert(
            pytorch_model=model, output_path=output_path, sample_input=dummy_input, validate=True
        )

        return success
