import pytest

"""Unit-Tests: TapeSpliceRepairPhase._compute_splice_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_64_tape_splice_repair import TapeSpliceRepairPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return TapeSpliceRepairPhase._compute_splice_profile(material, qm, rest)


@pytest.mark.unit
def test_tape_more_sensitive_than_cd():
    tape = _profile("tape")
    cd = _profile("cd_digital")
    assert tape["min_splice_score"] < cd["min_splice_score"]


def test_quality_adjustment():
    base = _profile("tape", "balanced", 60.0)
    q = _profile("tape", "quality", 60.0)
    assert q["min_splice_score"] < base["min_splice_score"]
    assert q["crossfade_ms"] > base["crossfade_ms"]


def test_fast_adjustment():
    base = _profile("tape", "balanced", 60.0)
    fast = _profile("tape", "fast", 60.0)
    assert fast["min_splice_score"] > base["min_splice_score"]
    assert fast["crossfade_ms"] < base["crossfade_ms"]


def test_low_restorability_adjustment():
    high_rest = _profile("tape", "balanced", 80.0)
    low_rest = _profile("tape", "balanced", 20.0)
    assert low_rest["min_splice_score"] < high_rest["min_splice_score"]
    assert low_rest["crossfade_ms"] >= high_rest["crossfade_ms"]


def test_profile_bounds():
    for material in ["tape", "reel_tape", "cd_digital", "unknown"]:
        for qm in ["balanced", "quality", "maximum", "fast", None]:
            p = _profile(material, qm, 30.0)  # type: ignore[arg-type]
            assert 0.05 <= p["min_splice_score"] <= 0.25
            assert 6.0 <= p["crossfade_ms"] <= 30.0


def test_process_metadata_contains_profile():
    phase = TapeSpliceRepairPhase()
    audio = np.random.uniform(-0.2, 0.2, 48000).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="tape",
        defect_scores={"tape_splice_artifact": 0.2},
        strength=0.5,
    )

    assert result.success
    assert "splice_profile" in result.metadata
    assert "min_splice_score" in result.metadata
    assert "crossfade_ms" in result.metadata


def test_process_applies_defect_locations_locality(monkeypatch):
    from backend.core.phases import phase_64_tape_splice_repair as m

    phase = m.TapeSpliceRepairPhase()
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (0.25 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)

    def _fake_apply(
        audio: np.ndarray,
        sample_rate: int,
        strength: float = 0.7,
        defect_scores: dict | None = None,
        min_splice_score: float = 0.1,
        crossfade_ms: float = 15.0,
        protected_zones: list | None = None,
    ) -> np.ndarray:
        del sample_rate, strength, defect_scores, min_splice_score, crossfade_ms, protected_zones
        return (audio * 0.08).astype(np.float32)

    monkeypatch.setattr(m, "apply", _fake_apply)

    result = phase.process(
        audio,
        sample_rate=sr,
        strength=1.0,
        defect_scores={"tape_splice_artifact": 0.5},
        defect_locations={"tape_splice_artifact": [(0.20, 0.30)]},
    )
    assert result.success is True

    diff = np.abs(result.audio - audio)
    in_region = float(np.mean(diff[int(0.21 * sr) : int(0.29 * sr)]))
    out_region = float(np.mean(diff[int(0.70 * sr) : int(0.85 * sr)]))
    assert in_region > out_region * 2.0
    assert float(result.metadata.get("repair_locality_coverage", 0.0)) > 0.0
