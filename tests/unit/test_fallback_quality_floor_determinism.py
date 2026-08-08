from __future__ import annotations

"""Unit-Tests fuer deterministische Fallback-Quality-Floor-Entscheidung (Paket 8).

Scope:
- Determinismus der UV3-Endentscheidung bei identischer Heavy-Fallback-Lage.
- Keine Seiteneffekte auf Eingangsdaten.
"""


from types import SimpleNamespace

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_is_deterministic_for_same_inputs() -> None:
    fallbacks = [
        {"phase": "phase_06_frequency_restoration", "fallback": "SBR"},
        {"phase": "phase_43_ml_deesser", "fallback": "OMLSA"},
    ]
    hpi = SimpleNamespace(passed=True, hpi=0.213)

    first = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=hpi,
        artifact_freedom=0.97,
    )
    second = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=hpi,
        artifact_freedom=0.97,
    )

    assert first == second
    assert first["triggered"] is True
    assert first["passed"] is True
    assert first["status"] == "passed"
    assert first["fallback_count"] == 2
    assert first["drift_warning"] is False
    assert first["drift_warning_level"] == "none"


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_does_not_mutate_input_list() -> None:
    fallbacks = [{"phase": "phase_23_spectral_repair", "fallback": "harmonic"}]
    snapshot = list(fallbacks)

    _ = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=True, hpi=0.2),
        artifact_freedom=0.99,
    )

    assert fallbacks == snapshot


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_requires_both_artifact_and_hpi() -> None:
    fallbacks = [{"phase": "phase_06_frequency_restoration", "fallback": "SBR"}]

    fail_artifact = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=True, hpi=0.33),
        artifact_freedom=0.94,
    )
    fail_hpi = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=False, hpi=-0.01),
        artifact_freedom=0.99,
    )

    assert fail_artifact["status"] == "degraded"
    assert fail_hpi["status"] == "degraded"
    assert fail_artifact["reason"] == "hpi_or_artifact_floor_failed"
    assert fail_hpi["reason"] == "hpi_or_artifact_floor_failed"


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_not_triggered_without_ml_fallbacks() -> None:
    result = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=[],
        hpi_result=SimpleNamespace(passed=False, hpi=-1.0),
        artifact_freedom=0.10,
    )

    assert result["triggered"] is False
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["fallback_count"] == 0


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_handles_non_finite_hpi_stably() -> None:
    result = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=[{"phase": "phase_43_ml_deesser", "fallback": "bypass"}],
        hpi_result=SimpleNamespace(passed=False, hpi=float("nan")),
        artifact_freedom=0.98,
    )

    assert result["hpi"] is None
    assert result["status"] == "degraded"


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_enforces_hard_limit_for_ml_fallback_count() -> None:
    fallbacks = [{"phase": f"phase_{i:02d}", "fallback": "dsp"} for i in range(1, 9)]  # §V6: logger.warning handled at call site
    result = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=True, hpi=0.91),
        artifact_freedom=0.99,
        max_ml_fallbacks_allowed=6,
    )

    assert result["triggered"] is True
    assert result["fallback_count"] == 8
    assert result["max_ml_fallbacks_allowed"] == 6
    assert result["hard_limit_exceeded"] is True
    assert result["passed"] is False
    assert result["status"] == "degraded"
    assert result["reason"] == "ml_fallback_limit_exceeded"


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_supports_stricter_vocal_profile_limit() -> None:
    fallbacks = [{"phase": f"phase_{i:02d}", "fallback": "dsp"} for i in range(1, 5)]  # §V6: logger.warning handled at call site
    result = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=True, hpi=0.91),
        artifact_freedom=0.99,
        max_ml_fallbacks_allowed=3,
    )

    assert result["fallback_count"] == 4
    assert result["max_ml_fallbacks_allowed"] == 3
    assert result["hard_limit_exceeded"] is True
    assert result["drift_warning"] is True
    assert result["drift_warning_level"] == "critical"
    assert result["status"] == "degraded"
    assert result["reason"] == "ml_fallback_limit_exceeded"


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_emits_warning_level_before_hard_limit() -> None:
    fallbacks = [{"phase": f"phase_{i:02d}", "fallback": "dsp"} for i in range(1, 5)]  # §V6: logger.warning handled at call site
    result = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=True, hpi=0.91),
        artifact_freedom=0.99,
        max_ml_fallbacks_allowed=6,
    )

    assert result["triggered"] is True
    assert result["passed"] is True
    assert result["hard_limit_exceeded"] is False
    assert result["drift_warning_threshold"] == 4
    assert result["drift_warning"] is True
    assert result["drift_warning_level"] == "warning"
