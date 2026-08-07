from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_scanner import DefectScanner, DefectType

SR = 48_000


def _make_test_audio(seconds: float = 1.2) -> np.ndarray:
    """Create deterministic mixed-content audio for scanner coverage tests."""
    n = int(SR * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    x = 0.08 * np.sin(2.0 * np.pi * 220.0 * t)
    x += 0.04 * np.sin(2.0 * np.pi * 880.0 * t)
    x += 0.01 * np.sin(2.0 * np.pi * 50.0 * t)
    rng = np.random.default_rng(42)
    x += 0.005 * rng.standard_normal(n).astype(np.float32)
    return np.clip(x.astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_scanner_includes_gap_defect_types_with_valid_ranges() -> None:
    scanner = DefectScanner(sample_rate=SR)
    audio = _make_test_audio()

    result = scanner.scan(audio, SR)

    # Explicitly cover the previously under-referenced defect types.
    gap_types = [
        DefectType.PITCH_DRIFT,
        DefectType.REVERB_EXCESS,
        DefectType.DYNAMIC_COMPRESSION_EXCESS,
        DefectType.SOFT_SATURATION,
        DefectType.HEAD_WEAR,
        DefectType.TRANSIENT_SMEARING,
        DefectType.PRE_ECHO,
    ]

    for defect_type in gap_types:
        assert defect_type in result.scores
        score = result.scores[defect_type]
        assert 0.0 <= float(score.severity) <= 1.0
        assert 0.0 <= float(score.confidence) <= 1.0
        assert isinstance(score.locations, list)
        assert isinstance(score.metadata, dict)
        assert score.metadata.get("confidence_calibrated") is True
        assert "confidence_before_calibration" in score.metadata
        assert "confidence_after_calibration" in score.metadata
        assert "confidence_evidence_ratio" in score.metadata
        assert "confidence_material" in score.metadata
