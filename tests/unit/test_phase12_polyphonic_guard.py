from __future__ import annotations

import logging

import numpy as np
import pytest

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix


@pytest.mark.unit
def test_polyphonic_estimate_is_insufficient_for_single_frame() -> None:
    pitch = np.array([220.0], dtype=np.float64)
    conf = np.array([0.95], dtype=np.float64)
    assert WowFlutterFix._polyphonic_estimate_is_insufficient(pitch, conf)


def test_polyphonic_estimate_is_insufficient_for_sparse_valid_frames() -> None:
    pitch = np.array([220.0, 0.0, 0.0, 0.0, 221.0, 0.0], dtype=np.float64)
    conf = np.array([0.16, 0.01, 0.0, 0.03, 0.14, 0.0], dtype=np.float64)
    assert WowFlutterFix._polyphonic_estimate_is_insufficient(pitch, conf)


def test_polyphonic_estimate_is_sufficient_for_stable_sequence() -> None:
    pitch = np.array([220.0, 221.0, 219.5, 220.8, 220.2, 221.1], dtype=np.float64)
    conf = np.array([0.8, 0.85, 0.78, 0.82, 0.8, 0.81], dtype=np.float64)
    assert not WowFlutterFix._polyphonic_estimate_is_insufficient(pitch, conf)


def test_polyphonic_reestimate_logs_info_not_warning(monkeypatch, caplog) -> None:
    import backend.core.hybrid.hybrid_wow_flutter as hybrid_mod
    from backend.core.defect_scanner import MaterialType

    phase = WowFlutterFix()

    class _FakePolyphonicEstimator:
        def estimate(self, mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
            return np.array([220.0], dtype=np.float64), np.array([0.95], dtype=np.float64)

    monkeypatch.setattr(hybrid_mod, "PolyphonicSpeedCurveEstimator", lambda: _FakePolyphonicEstimator())
    monkeypatch.setattr(
        phase,
        "_estimate_pitch_yin",
        lambda mono, sample_rate: (np.zeros(8, dtype=np.float64), np.zeros(8, dtype=np.float64)),
    )
    monkeypatch.setattr(phase, "_preserve_phase_loudness", lambda original, processed, material: (processed, 0.0, 0.0))

    audio = np.zeros(48000, dtype=np.float32)

    with caplog.at_level(logging.INFO):
        result = phase.process(audio, sample_rate=48000, material_type=MaterialType.VINYL, quality_mode="maximum")  # type: ignore[arg-type]

    assert result.success is True
    assert result.metrics["skipped_reason"] == "low_confidence_fallback"
    assert any(
        rec.levelno == logging.INFO and "Polyphoner Konsensus unzureichend" in rec.message for rec in caplog.records
    )
    assert not any(
        rec.levelno >= logging.WARNING and "Polyphoner Konsensus unzureichend" in rec.message for rec in caplog.records
    )
