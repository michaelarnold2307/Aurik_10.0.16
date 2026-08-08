"""Leichtgewichtiges fallback execution guard for fault-injection validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from backend.core.pipeline_health_state import PipelineHealthState


@dataclass(frozen=True)
class FallbackExecutionResult:
    release_mode: str
    value: object
    fail_reason: str | None
    degradation_status: str = PipelineHealthState.OK.value
    fail_reasons: list[dict[str, str]] | None = None


def execute_with_fallback(
    primary: Callable[[], object],
    fallback: Callable[[], object],
) -> FallbackExecutionResult:
    """Führt aus: primary path and switch to fallback on deterministic failures.

    release_mode:
        - "primary" when primary succeeds
        - "fallback" when primary fails and fallback succeeds
        - "blocked" when both fail
    """
    try:
        value = primary()
        return FallbackExecutionResult(
            release_mode="primary",
            value=value,
            fail_reason=None,
            degradation_status=PipelineHealthState.OK.value,
            fail_reasons=[],
        )
    except Exception as primary_exc:
        primary_entry = {
            "component": "primary",
            "error_code": "PRIMARY_FAILED",
            "severity": "degraded",
            "exc_type": type(primary_exc).__name__,
            "exc_msg": str(primary_exc),
        }
        try:
            value = fallback()
            return FallbackExecutionResult(
                release_mode="fallback",
                value=value,
                fail_reason=f"primary_failed: {primary_exc}",
                degradation_status=PipelineHealthState.DEGRADED.value,
                fail_reasons=[primary_entry],
            )
        except Exception as fallback_exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            fallback_entry = {
                "component": "fallback",
                "error_code": "FALLBACK_FAILED",
                "severity": "blocked",
                "exc_type": type(fallback_exc).__name__,
                "exc_msg": str(fallback_exc),
            }
            return FallbackExecutionResult(
                release_mode="blocked",
                value=None,
                fail_reason=f"primary_failed: {primary_exc}; fallback_failed: {fallback_exc}",
                degradation_status=PipelineHealthState.BLOCKED.value,
                fail_reasons=[primary_entry, fallback_entry],
            )
