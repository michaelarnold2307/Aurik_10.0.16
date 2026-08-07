#!/usr/bin/env python3
"""
ONNX Model Quantization Script for AURIK v8
===========================================

Batch quantize all ONNX models in the registry from FP32 to INT8.
Uses the ModelQuantizer from backend/core/onnx/quantizer.py.

Features:
- Batch quantization of all models in registry
- Quality validation (ensures <1% degradation)
- Automatic fallback if quality degrades too much
- Progress tracking and statistics
- Dry-run mode for testing

Usage:
    # Quantize all models
    python scripts/quantize_models.py

    # Quantize specific model
    python scripts/quantize_models.py --model deepfilternet_v3_ii

    # Dry-run (check what would be quantized)
    python scripts/quantize_models.py --dry-run

    # Custom quality threshold
    python scripts/quantize_models.py --max-quality-loss 0.5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.onnx.quantizer import (
    ModelQuantizer,
    QuantizationConfig,
    QuantizationType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelRegistryQuantizer:
    """
    Batch quantizer for all models in the ONNX model registry.

    Reads model_registry.json and quantizes all ONNX models that:
    1. Are not already quantized
    2. Have a valid quantized_path specified
    3. Pass quality validation
    """

    def __init__(
        self,
        registry_path: str = "backend/core/onnx/model_registry.json",
        max_quality_loss_percent: float = 1.0,
        dry_run: bool = False,
    ):
        """
        Initialize the registry quantizer.

        Args:
            registry_path: Path to model_registry.json
            max_quality_loss_percent: Maximum allowed quality degradation (%)
            dry_run: If True, don't actually quantize, just report what would be done
        """
        self.registry_path = Path(registry_path)
        self.max_quality_loss_percent = max_quality_loss_percent
        self.dry_run = dry_run

        # Load registry
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")

        with open(self.registry_path) as f:
            self.registry = json.load(f)

        # Initialize quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.DYNAMIC,
            per_channel=True,
            optimize_model=True,
            max_quality_loss_percent=max_quality_loss_percent,
        )
        self.quantizer = ModelQuantizer(config)

        logger.info(f"geladen registry with {len(self.registry['models'])} models")

    def get_quantizable_models(self) -> list[tuple[str, dict, dict]]:
        """
        Get list of models that can be quantized.

        Returns:
            List of (model_id, model_config, onnx_model_config) tuples
        """
        quantizable = []

        for model_id, model_config in self.registry["models"].items():
            for onnx_model in model_config["onnx_models"]:
                # Skip if already quantized
                if onnx_model.get("quantized", False):
                    logger.debug(f"Skipping {model_id}/{onnx_model['name']}: already quantized")
                    continue

                # Skip if no quantized_path
                if not onnx_model.get("quantized_path"):
                    logger.debug(f"Skipping {model_id}/{onnx_model['name']}: no quantized_path")
                    continue

                # Skip if source model doesn't exist
                model_path = Path(onnx_model["path"])
                if not model_path.exists():
                    logger.warning(f"Skipping {model_id}/{onnx_model['name']}: source model not found at {model_path}")
                    continue

                quantizable.append((model_id, model_config, onnx_model))

        return quantizable

    def quantize_model(self, model_id: str, model_config: dict, onnx_model: dict) -> tuple[bool, dict | None]:
        """
        Quantize a single ONNX model.

        Args:
            model_id: Model identifier (e.g., "deepfilternet_v3_ii")
            model_config: Model configuration from registry
            onnx_model: ONNX model configuration

        Returns:
            Tuple of (success, stats_dict)
        """
        model_path = Path(onnx_model["path"])
        quantized_path = Path(onnx_model["quantized_path"])

        logger.info(f"Quantizing {model_id}/{onnx_model['name']}...")
        logger.info(f"  Source: {model_path}")
        logger.info(f"  Target: {quantized_path}")
        logger.info(f"  Size: {onnx_model['model_size_mb']:.1f} MB")

        if self.dry_run:
            logger.info("  [DRY Ausfuehrung] Skipping actual quantization")
            return True, {
                "model_id": model_id,
                "model_name": onnx_model["name"],
                "source_size_mb": onnx_model["model_size_mb"],
                "dry_run": True,
            }

        try:
            # Quantize
            start_time = time.time()
            success = self.quantizer.quantize(model_path=str(model_path), output_path=str(quantized_path))  # type: ignore[arg-type]
            elapsed = time.time() - start_time

            if not success:
                logger.error("  ✗ Quantization fehlgeschlagen")
                return False, None

            # Get stats
            stats = self.quantizer.get_statistics()  # type: ignore[attr-defined]

            # Calculate size reduction
            quantized_size_mb = quantized_path.stat().st_size / (1024 * 1024)
            size_reduction_percent = (
                (onnx_model["model_size_mb"] - quantized_size_mb) / onnx_model["model_size_mb"] * 100
            )

            logger.info(f"  ✓ Quantized erfolgreich in {elapsed:.1f}s")
            logger.info(
                f"  Size: {onnx_model['model_size_mb']:.1f} MB → {quantized_size_mb:.1f} MB ({size_reduction_percent:.1f}% reduction)"
            )
            logger.info(f"  Expected speedup: {onnx_model['expected_speedup']:.1f}×")

            return True, {
                "model_id": model_id,
                "model_name": onnx_model["name"],
                "source_size_mb": onnx_model["model_size_mb"],
                "quantized_size_mb": quantized_size_mb,
                "size_reduction_percent": size_reduction_percent,
                "elapsed_seconds": elapsed,
                "expected_speedup": onnx_model["expected_speedup"],
                "quantization_stats": stats,
            }

        except Exception as e:
            logger.error(f"  ✗ Error during quantization: {e}")
            return False, None

    def quantize_all(self, model_filter: str | None = None) -> dict:
        """
        Quantize all quantizable models in the registry.

        Args:
            model_filter: Optional model ID filter (e.g., "deepfilternet_v3_ii")

        Returns:
            Dictionary with quantization results and statistics
        """
        quantizable = self.get_quantizable_models()

        # Apply filter
        if model_filter:
            quantizable = [(mid, mc, om) for mid, mc, om in quantizable if mid == model_filter]

        if not quantizable:
            logger.warning("No quantizable models found!")
            return {"success": False, "total": 0, "quantized": 0, "failed": 0, "results": []}

        logger.info(f"Found {len(quantizable)} models to quantize")
        if self.dry_run:
            logger.info("[DRY Ausfuehrung Betriebsart] No actual quantization will be performed")

        results = []
        success_count = 0
        failed_count = 0

        for model_id, model_config, onnx_model in quantizable:
            success, stats = self.quantize_model(model_id, model_config, onnx_model)

            if success:
                success_count += 1
            else:
                failed_count += 1

            if stats:
                results.append(stats)

        # Calculate total statistics
        total_size_reduction_mb = sum(
            r.get("source_size_mb", 0) - r.get("quantized_size_mb", 0) for r in results if not r.get("dry_run", False)
        )

        total_time = sum(r.get("elapsed_seconds", 0) for r in results)

        summary = {
            "success": failed_count == 0,
            "total": len(quantizable),
            "quantized": success_count,
            "failed": failed_count,
            "total_size_reduction_mb": total_size_reduction_mb,
            "total_time_seconds": total_time,
            "dry_run": self.dry_run,
            "results": results,
        }

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("QUANTIZATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total models: {summary['total']}")
        logger.info(f"Quantized: {summary['quantized']}")
        logger.info(f"fehlgeschlagen: {summary['failed']}")

        if not self.dry_run:
            logger.info(f"Total size reduction: {summary['total_size_reduction_mb']:.1f} MB")
            logger.info(f"Total time: {summary['total_time_seconds']:.1f}s")

        logger.info("=" * 80)

        return summary


def main():
    """Main entry point for the quantization script."""
    parser = argparse.ArgumentParser(description="Quantize ONNX models in the AURIK model registry")
    parser.add_argument(
        "--registry", type=str, default="backend/core/onnx/model_registry.json", help="Path to model_registry.json"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Quantize only specific model (e.g., 'deepfilternet_v3_ii')"
    )
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=1.0,
        help="Maximum allowed quality degradation (percent, default: 1.0)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't actually quantize)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        quantizer = ModelRegistryQuantizer(
            registry_path=args.registry, max_quality_loss_percent=args.max_quality_loss, dry_run=args.dry_run
        )

        summary = quantizer.quantize_all(model_filter=args.model)

        # Exit with error code if any quantization failed
        if not summary["success"]:
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
