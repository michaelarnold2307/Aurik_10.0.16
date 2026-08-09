from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_23_spectral_repair import SpectralRepair


@pytest.mark.unit
def test_admm_runtime_profile_keys_and_bounds() -> None:
    p = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=55.0,
    )

    assert set(p.keys()) == {"clip_percentile", "clip_floor", "side_clip_multiplier"}
    assert 98.8 <= p["clip_percentile"] <= 99.9
    assert 0.80 <= p["clip_floor"] <= 0.93
    assert 1.00 <= p["side_clip_multiplier"] <= 1.10


def test_low_restorability_reduces_clip_percentile() -> None:
    low = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=10.0,
    )
    high = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=90.0,
    )

    assert low["clip_percentile"] < high["clip_percentile"]


def test_fast_mode_more_conservative_than_quality() -> None:
    fast = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.CD_DIGITAL,
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.CD_DIGITAL,
        quality_mode="quality",
        restorability_score=60.0,
    )

    assert fast["clip_percentile"] > quality["clip_percentile"]
    assert fast["clip_floor"] > quality["clip_floor"]
    assert fast["side_clip_multiplier"] > quality["side_clip_multiplier"]


def test_lossy_material_raises_clip_floor_vs_analog() -> None:
    lossy = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.STREAMING,
        quality_mode="balanced",
        restorability_score=60.0,
    )
    analog = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.VINYL,
        quality_mode="balanced",
        restorability_score=60.0,
    )

    assert lossy["clip_floor"] > analog["clip_floor"]


def test_cassette_bw_ceiling_is_material_adaptive() -> None:
    assert SpectralRepair._material_bw_ceiling_hz(MaterialType.CASSETTE) == 16000.0
    assert SpectralRepair._material_bw_ceiling_hz(MaterialType.TAPE) == 15000.0
    assert SpectralRepair._material_bw_ceiling_hz(MaterialType.REEL_TAPE) == 18000.0


def test_phase23_uses_npd_singleton_accessor(monkeypatch) -> None:
    phase = SpectralRepair()
    # Keep test lightweight: this test verifies NPD accessor wiring, not MRSA quality.
    audio = np.random.uniform(-0.02, 0.02, 8_192).astype(np.float32)
    calls = {"count": 0}

    class _NpdResultStub:
        success = False

        def get_protected_mask(self, _n_samples, _sr):
            return np.zeros(_n_samples, dtype=bool)

    class _NpdStub:
        def detect(self, _audio, _sr):
            calls["count"] += 1
            return _NpdResultStub()

    monkeypatch.setattr(
        "backend.core.phases.phase_23_spectral_repair._get_phase23_npd",
        lambda: _NpdStub(),
    )
    monkeypatch.setattr(
        phase,
        "_repair_channel",
        lambda signal_in, *_, **__: np.asarray(signal_in, dtype=np.float32).copy(),
    )

    result = phase.process(
        audio,
        48_000,
        material=MaterialType.VINYL,
        quality_mode="balanced",
        restorability_score=60.0,
        strength=0.4,
    )

    assert result.success
    assert calls["count"] >= 1


def test_phase23_defect_locality_profile_is_event_adaptive() -> None:
    profile, coverage = SpectralRepair._build_defect_locality_profile(
        n_samples=48_000 * 2,
        sample_rate=48_000,
        defect_locations={"pre_echo": [(0.20, 0.30)], "codec_artifact": [(1.20, 1.30)]},
        defect_event_metadata={
            "pre_echo": {"severity": 0.95, "confidence": 0.95},
            "codec_artifact": {"severity": 0.30, "confidence": 0.65},
        },
    )

    assert profile.shape == (48_000 * 2,)
    assert 0.0 < coverage < 0.20
    pre_echo_strength = float(np.mean(profile[int(0.22 * 48_000) : int(0.28 * 48_000)]))
    codec_strength = float(np.mean(profile[int(1.22 * 48_000) : int(1.28 * 48_000)]))
    clean_strength = float(np.mean(profile[int(0.70 * 48_000) : int(0.90 * 48_000)]))
    assert pre_echo_strength > codec_strength * 1.25
    assert clean_strength < 0.02
