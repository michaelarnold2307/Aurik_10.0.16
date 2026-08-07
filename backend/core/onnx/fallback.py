"""
Fallback Manager for ONNX → PyTorch automatic fallback

Handles graceful degradation when ONNX inference fails or is unavailable.
Ensures robust operation even when ONNX models have issues.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for fallback to PyTorch."""

    MODEL_NOT_FOUND = "model_not_found"
    ONNX_RUNTIME_ERROR = "onnx_runtime_error"
    SHAPE_MISMATCH = "shape_mismatch"
    INFERENCE_ERROR = "inference_error"
    VALIDATION_FAILED = "validation_failed"
    ONNX_DISABLED = "onnx_disabled"
    SESSION_INIT_FAILED = "session_init_failed"


@dataclass
class FallbackEvent:
    """Record of a fallback event."""

    timestamp: datetime
    model_name: str
    reason: FallbackReason
    error_message: str
    recovered: bool = False

    def __str__(self) -> str:
        status = "✓ RECOVERED" if self.recovered else "❌ ACTIVE"
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {self.model_name}: {self.reason.value} - {status}"


@dataclass
class FallbackStats:
    """Statistics for fallback events."""

    total_fallbacks: int = 0
    active_fallbacks: int = 0
    recovered_fallbacks: int = 0
    fallback_by_reason: dict[FallbackReason, int] = field(default_factory=dict)
    fallback_by_model: dict[str, int] = field(default_factory=dict)


class FallbackManager:
    """
    Manages ONNX → PyTorch fallback mechanisms.

    Features:
    - Automatic fallback on ONNX errors
    - Fallback event tracking
    - Health checks for ONNX models
    - Recovery detection
    - Detailed logging and statistics

    Usage:
        manager = FallbackManager()

        # Try ONNX first
        try:
            output = onnx_model.process(audio)
        except Exception as e:
            # Fallback to PyTorch
            manager.record_fallback(
                model_name="denoiser",
                reason=FallbackReason.INFERENCE_ERROR,
                error_message=str(e)
            )
            output = pytorch_model.process(audio)
    """

    def __init__(self, log_fallbacks: bool = True, max_fallback_history: int = 100):
        """
        Initialisiert fallback manager.

        Args:
            log_fallbacks: Log fallback events
            max_fallback_history: Maximum fallback events to retain
        """
        self.log_fallbacks = log_fallbacks
        self.max_fallback_history = max_fallback_history

        self.fallback_history: list[FallbackEvent] = []
        self.active_fallbacks: dict[str, FallbackEvent] = {}
        self.stats = FallbackStats()

    def record_fallback(self, model_name: str, reason: FallbackReason, error_message: str = "") -> None:
        """
        Record a fallback event.

        Args:
            model_name: Name of model that failed
            reason: Reason for fallback
            error_message: Error details
        """
        event = FallbackEvent(
            timestamp=datetime.now(), model_name=model_name, reason=reason, error_message=error_message
        )

        # Add to history
        self.fallback_history.append(event)
        if len(self.fallback_history) > self.max_fallback_history:
            self.fallback_history.pop(0)

        # Track active fallback
        self.active_fallbacks[model_name] = event

        # Update statistics
        self.stats.total_fallbacks += 1
        self.stats.active_fallbacks = len(self.active_fallbacks)

        if reason not in self.stats.fallback_by_reason:
            self.stats.fallback_by_reason[reason] = 0
        self.stats.fallback_by_reason[reason] += 1

        if model_name not in self.stats.fallback_by_model:
            self.stats.fallback_by_model[model_name] = 0
        self.stats.fallback_by_model[model_name] += 1

        # Log event
        if self.log_fallbacks:
            logger.warning("Ersatzpfad triggered: %s (%s) - %s", model_name, reason.value, error_message)
            logger.info("Falling back to PyTorch for %s", model_name)

    def record_recovery(self, model_name: str) -> None:
        """
        Record recovery from fallback.

        Args:
            model_name: Name of model that recovered
        """
        if model_name in self.active_fallbacks:
            event = self.active_fallbacks[model_name]
            event.recovered = True
            del self.active_fallbacks[model_name]

            self.stats.recovered_fallbacks += 1
            self.stats.active_fallbacks = len(self.active_fallbacks)

            if self.log_fallbacks:
                logger.info("✓ Wiederherstellung: %s back to ONNX", model_name)

    def is_fallback_active(self, model_name: str) -> bool:
        """
        Prüft if model is currently in fallback mode.

        Args:
            model_name: Name of model to check

        Returns:
            True if currently using fallback (PyTorch)
        """
        return model_name in self.active_fallbacks

    def get_fallback_reason(self, model_name: str) -> FallbackReason | None:
        """
        Gibt zurück: reason for active fallback.

        Args:
            model_name: Name of model

        Returns:
            FallbackReason if active, None otherwise
        """
        if model_name in self.active_fallbacks:
            return self.active_fallbacks[model_name].reason
        return None

    def health_check_onnx(self, model_name: str, onnx_inference_func: Callable, test_input: Any) -> bool:
        """
        Führt durch: health check on ONNX model.

        Attempts inference with test input to verify model is working.

        Args:
            model_name: Name of model
            onnx_inference_func: Function to run ONNX inference
            test_input: Test input for inference

        Returns:
            True if health check passed
        """
        try:
            _ = onnx_inference_func(test_input)

            # If currently in fallback, mark as recovered
            if model_name in self.active_fallbacks:
                self.record_recovery(model_name)

            return True

        except Exception as e:
            logger.warning("Health Pruefung fehlgeschlagen for %s: %s", model_name, e)

            # Record fallback if not already active
            if model_name not in self.active_fallbacks:
                self.record_fallback(
                    model_name=model_name, reason=FallbackReason.ONNX_RUNTIME_ERROR, error_message=str(e)
                )

            return False

    def get_stats(self) -> dict[str, Any]:
        """Gibt zurück: fallback statistics."""
        return {
            "total_fallbacks": self.stats.total_fallbacks,
            "active_fallbacks": self.stats.active_fallbacks,
            "recovered_fallbacks": self.stats.recovered_fallbacks,
            "fallback_by_reason": {reason.value: count for reason, count in self.stats.fallback_by_reason.items()},
            "fallback_by_model": self.stats.fallback_by_model.copy(),
            "active_fallback_models": list(self.active_fallbacks.keys()),
        }

    def get_fallback_history(self, model_name: str | None = None, limit: int = 10) -> list[FallbackEvent]:
        """
        Gibt zurück: recent fallback events.

        Args:
            model_name: Filter by model name (optional)
            limit: Maximum number of events to return

        Returns:
            List of recent fallback events
        """
        history = self.fallback_history

        if model_name:
            history = [e for e in history if e.model_name == model_name]

        return history[-limit:]

    def print_summary(self) -> None:
        """Gibt aus: fallback summary."""
        logger.debug("\n" + "=" * 60)
        logger.debug("Ersatzpfad MANAGER SUMMARY")
        logger.debug("=" * 60)

        logger.debug("Total Fallbacks: %s", self.stats.total_fallbacks)
        logger.debug("Active Fallbacks: %s", self.stats.active_fallbacks)
        logger.debug("Recovered: %s", self.stats.recovered_fallbacks)

        if self.stats.fallback_by_reason:
            logger.debug("\nFallbacks by Reason:")
            for reason, count in self.stats.fallback_by_reason.items():
                logger.debug("  %s: %s", reason.value, count)

        if self.stats.fallback_by_model:
            logger.debug("\nFallbacks by Model:")
            for model, count in self.stats.fallback_by_model.items():
                logger.debug("  %s: %s", model, count)

        if self.active_fallbacks:
            logger.debug("\nActive Fallbacks:")
            for event in self.active_fallbacks.values():
                logger.debug("  %s", event)

        logger.debug("=" * 60)

    def reset_stats(self) -> None:
        """Setzt zurück: all statistics."""
        self.stats = FallbackStats()
        self.fallback_history.clear()
        self.active_fallbacks.clear()


class ONNXModelWithFallback:
    """
    ONNX model wrapper with automatic PyTorch fallback.

    Combines ONNX inference with graceful fallback to PyTorch
    when ONNX fails or is unavailable.

    Usage:
        model = ONNXModelWithFallback(
            name="denoiser",
            onnx_model=onnx_denoiser,
            pytorch_model=pytorch_denoiser,
            fallback_manager=manager
        )

        # Automatically tries ONNX, falls back to PyTorch if needed
        output = model.process(audio)
    """

    def __init__(
        self,
        name: str,
        onnx_model: Any,
        pytorch_model: Any,
        fallback_manager: FallbackManager | None = None,
        onnx_enabled: bool = True,
    ):
        """
        Initialisiert model with fallback.

        Args:
            name: Model name
            onnx_model: ONNX model instance
            pytorch_model: PyTorch model instance
            fallback_manager: Optional fallback manager
            onnx_enabled: Enable ONNX inference
        """
        self.name = name
        self.onnx_model = onnx_model
        self.pytorch_model = pytorch_model
        self.fallback_manager = fallback_manager or FallbackManager()
        self.onnx_enabled = onnx_enabled

        self.use_onnx = onnx_enabled
        self.inference_count = 0
        self.onnx_inference_count = 0
        self.pytorch_inference_count = 0

    def process(self, audio: Any, **kwargs) -> Any:
        """
        Verarbeitet audio with automatic fallback.

        Args:
            audio: Input audio
            **kwargs: Additional processing arguments

        Returns:
            Processed audio
        """
        self.inference_count += 1

        # Determine which model to use
        if not self.use_onnx or self.fallback_manager.is_fallback_active(self.name):
            # Use PyTorch
            return self._process_pytorch(audio, **kwargs)

        # Try ONNX first
        try:
            output = self.onnx_model.process(audio, **kwargs)
            self.onnx_inference_count += 1
            return output

        except Exception as e:
            # Fallback to PyTorch
            logger.warning("ONNX inference fehlgeschlagen for %s: %s", self.name, e)
            self.fallback_manager.record_fallback(
                model_name=self.name, reason=FallbackReason.INFERENCE_ERROR, error_message=str(e)
            )
            return self._process_pytorch(audio, **kwargs)

    def _process_pytorch(self, audio: Any, **kwargs) -> Any:
        """Verarbeitet with PyTorch model."""
        self.pytorch_inference_count += 1
        return self.pytorch_model.process(audio, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        """Gibt zurück: processing statistics."""
        return {
            "name": self.name,
            "total_inferences": self.inference_count,
            "onnx_inferences": self.onnx_inference_count,
            "pytorch_inferences": self.pytorch_inference_count,
            "onnx_usage_percent": (
                (self.onnx_inference_count / self.inference_count * 100) if self.inference_count > 0 else 0
            ),
            "fallback_active": self.fallback_manager.is_fallback_active(self.name),
        }
