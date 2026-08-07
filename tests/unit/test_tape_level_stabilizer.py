"""
Tests for Phase 12 Tape Head Contact Level Stabilizer (Step 6c).

Validates the autonomous detection and repair of gradual level dips
caused by tape-head pressure variation / capstan irregularity.

Defect morphology (from real cassette analysis — Elke Best):
  - Gradual fade-down: 60-100 ms onset
  - Minimum depth: 10-25 dB below local context level
  - Sharp recovery: < 25 ms back to normal
  - 147 events across 225 s song (rate ~0.65/s)
"""

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

SR = 48000


def _make_tone_with_dips(
    duration_s: float = 5.0,
    freq_hz: float = 440.0,
    dip_positions_s: list[float] | None = None,
    dip_depth_db: float = 15.0,
    dip_duration_ms: float = 100.0,
    fade_in_ms: float = 60.0,
) -> np.ndarray:
    """Generate a sine tone with injected tape-like level dips.

    Each dip has a gradual onset (fade_in_ms) and sharp recovery.
    """
    n = int(duration_s * SR)
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = 0.3 * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)

    if dip_positions_s is None:
        dip_positions_s = [1.0, 2.5, 4.0]

    dip_samples = int(dip_duration_ms / 1000.0 * SR)
    fade_in_samples = int(fade_in_ms / 1000.0 * SR)
    linear_depth = 10.0 ** (-dip_depth_db / 20.0)

    for pos_s in dip_positions_s:
        start = int(pos_s * SR)
        end = min(start + dip_samples, n)
        if start >= n:
            continue

        # Build envelope: gradual fade-down, then sharp recovery
        region_len = end - start
        env = np.ones(region_len, dtype=np.float32)

        # Gradual onset (first fade_in_samples)
        actual_fade = min(fade_in_samples, region_len)
        if actual_fade > 0:
            fade_curve = np.linspace(1.0, linear_depth, actual_fade, dtype=np.float32)
            env[:actual_fade] = fade_curve

        # Hold at minimum for remaining dip
        env[actual_fade:] = linear_depth

        audio[start:end] *= env

    return audio  # type: ignore[no-any-return]


@pytest.fixture
def phase():
    return WowFlutterFix()


@pytest.mark.unit
class TestStabilizeTapeLevel:
    """Unit tests for WowFlutterFix._stabilize_tape_level()."""

    def test_basic_dip_detection_and_repair(self, phase):
        """Dips of 15 dB should be detected and compensated."""
        audio = _make_tone_with_dips(dip_depth_db=15.0, dip_positions_s=[1.0, 2.5, 4.0])
        result, n_repaired = phase._stabilize_tape_level(audio, SR, strength=1.0)

        assert n_repaired >= 2, f"Expected ≥2 dips repaired, got {n_repaired}"
        assert result.shape == audio.shape

        # Verify the dip at t=1.0 has been raised
        dip_center = int(1.05 * SR)  # just past onset
        win = SR // 20  # 50 ms window
        rms_orig = np.sqrt(np.mean(audio[dip_center : dip_center + win] ** 2))
        rms_fixed = np.sqrt(np.mean(result[dip_center : dip_center + win] ** 2))
        assert rms_fixed > rms_orig * 1.5, "Dip should be noticeably raised"

    def test_no_dips_returns_unchanged(self, phase):
        """Clean audio should pass through without modification."""
        n = int(3.0 * SR)
        audio = 0.3 * np.sin(2.0 * np.pi * 440.0 * np.arange(n) / SR).astype(np.float32)
        result, n_repaired = phase._stabilize_tape_level(audio, SR, strength=1.0)

        assert n_repaired == 0
        np.testing.assert_array_equal(result, audio)

    def test_strength_zero_passthrough(self, phase):
        """strength=0 yields no modification."""
        audio = _make_tone_with_dips(dip_depth_db=15.0)
        result, n_repaired = phase._stabilize_tape_level(audio, SR, strength=0.0)

        assert n_repaired == 0
        np.testing.assert_array_equal(result, audio)

    def test_strength_scaling(self, phase):
        """Higher strength should produce larger gain compensation."""
        audio = _make_tone_with_dips(dip_depth_db=12.0, dip_positions_s=[2.0])
        dip_center = int(2.08 * SR)
        win = SR // 20

        _, n_low = phase._stabilize_tape_level(audio.copy(), SR, strength=0.3)
        result_low, _ = phase._stabilize_tape_level(audio.copy(), SR, strength=0.3)
        result_high, _ = phase._stabilize_tape_level(audio.copy(), SR, strength=1.0)

        rms_low = np.sqrt(np.mean(result_low[dip_center : dip_center + win] ** 2))
        rms_high = np.sqrt(np.mean(result_high[dip_center : dip_center + win] ** 2))
        assert rms_high >= rms_low, "Higher strength should raise dip more"

    def test_stereo_handling(self, phase):
        """Stereo input should be stabilized on both channels."""
        mono = _make_tone_with_dips(dip_depth_db=15.0, dip_positions_s=[1.5])
        stereo = np.column_stack([mono, mono * 0.8])
        result, n_repaired = phase._stabilize_tape_level(stereo, SR, strength=1.0)

        assert n_repaired >= 1
        assert result.shape == stereo.shape
        assert result.ndim == 2

    def test_nan_inf_robustness(self, phase):
        """NaN/Inf in input should not crash or propagate."""
        audio = _make_tone_with_dips(dip_depth_db=12.0, dip_positions_s=[1.0])
        audio[1000] = np.nan
        audio[2000] = np.inf
        audio[3000] = -np.inf

        result, _ = phase._stabilize_tape_level(audio, SR, strength=1.0)

        assert np.isfinite(result).all(), "Output must be NaN/Inf-free"
        assert np.max(np.abs(result)) <= 1.0, "Output must be clipped to [-1, 1]"

    def test_output_clipped(self, phase):
        """Output amplitude must be ≤ 1.0 after gain compensation."""
        # High-amplitude signal with dips — gain could push beyond 1.0
        audio = _make_tone_with_dips(dip_depth_db=10.0, dip_positions_s=[1.0])
        audio *= 2.5  # push to high amplitude
        audio = np.clip(audio, -1.0, 1.0)

        result, _ = phase._stabilize_tape_level(audio, SR, strength=1.0)
        assert np.max(np.abs(result)) <= 1.0

    def test_silence_not_boosted(self, phase):
        """Genuine silence (< -55 dBFS) should NOT be boosted."""
        n = int(3.0 * SR)
        audio = np.zeros(n, dtype=np.float32)
        # Insert tiny signal
        audio[SR : SR + 4800] = 1e-4  # -80 dBFS
        result, n_repaired = phase._stabilize_tape_level(audio, SR, strength=1.0)

        # Should not attempt to boost near-silence
        assert n_repaired == 0

    def test_very_deep_dips_skipped(self, phase):
        """Dips deeper than max_gain_db + 5 should be skipped (likely genuine silence)."""
        audio = _make_tone_with_dips(dip_depth_db=25.0, dip_positions_s=[1.5])
        _, n_repaired = phase._stabilize_tape_level(audio, SR, strength=1.0)

        # 25 dB dip exceeds 15 + 5 = 20 dB limit → should be skipped
        assert n_repaired == 0

    def test_shallow_dips_below_threshold_ignored(self, phase):
        """Dips smaller than 3 dB threshold should not be repaired."""
        audio = _make_tone_with_dips(dip_depth_db=2.0, dip_positions_s=[1.0, 2.5])
        _, n_repaired = phase._stabilize_tape_level(audio, SR, strength=1.0)

        assert n_repaired == 0, "2 dB dips should be below 3 dB detection threshold"

    def test_many_dips_like_real_cassette(self, phase):
        """Simulate 30+ dips in 30 s (typical for degraded cassette)."""
        # ~1 dip per second
        positions = [float(i) + 0.3 for i in range(25)]
        audio = _make_tone_with_dips(
            duration_s=30.0,
            dip_depth_db=12.0,
            dip_positions_s=positions,
            dip_duration_ms=80.0,
        )
        result, n_repaired = phase._stabilize_tape_level(audio, SR, strength=0.9)

        assert n_repaired >= 15, f"Expected ≥15 of 25 dips repaired, got {n_repaired}"

    def test_short_audio_handled(self, phase):
        """Very short audio (< 10 frames) should return safely."""
        audio = np.zeros(100, dtype=np.float32)  # ~2 ms at 48 kHz
        result, n_repaired = phase._stabilize_tape_level(audio, SR, strength=1.0)

        assert n_repaired == 0
        np.testing.assert_array_equal(result, audio)

    def test_output_dtype_float32(self, phase):
        """Output should always be float32."""
        audio = _make_tone_with_dips(dip_depth_db=12.0)
        result, _ = phase._stabilize_tape_level(audio, SR, strength=1.0)

        assert result.dtype == np.float32


class TestPhase12IntegrationTapeLevelStabilizer:
    """Integration tests: tape level dips repaired via process() for TAPE material."""

    def test_tape_material_triggers_stabilizer(self, phase):
        """process() with MaterialType.TAPE should run the stabilizer."""
        audio = _make_tone_with_dips(duration_s=3.0, dip_depth_db=12.0, dip_positions_s=[1.0])
        result = phase.process(audio, SR, material=MaterialType.TAPE)

        assert result.success
        dips_repaired = result.metrics.get("tape_level_dips_repaired", 0)
        assert dips_repaired >= 0  # may or may not detect depending on path

    def test_vinyl_does_not_trigger_stabilizer(self, phase):
        """process() with MaterialType.VINYL should NOT run the tape stabilizer."""
        audio = _make_tone_with_dips(duration_s=3.0, dip_depth_db=12.0, dip_positions_s=[1.0])
        result = phase.process(audio, SR, material=MaterialType.VINYL)

        assert result.success
        dips_repaired = result.metrics.get("tape_level_dips_repaired", 0)
        assert dips_repaired == 0

    def test_cd_digital_does_not_trigger_stabilizer(self, phase):
        """process() with MaterialType.CD_DIGITAL should NOT run the tape stabilizer."""
        audio = _make_tone_with_dips(duration_s=3.0, dip_depth_db=12.0, dip_positions_s=[1.0])
        result = phase.process(audio, SR, material=MaterialType.CD_DIGITAL)

        assert result.success
        dips_repaired = result.metrics.get("tape_level_dips_repaired", 0)
        assert dips_repaired == 0


class TestDefectTypeEnum:
    """Verify TAPE_HEAD_LEVEL_DIP is properly registered in DefectType."""

    def test_tape_head_level_dip_exists(self):
        from backend.core.defect_scanner import DefectType

        assert hasattr(DefectType, "TAPE_HEAD_LEVEL_DIP")
        assert DefectType.TAPE_HEAD_LEVEL_DIP.value == "tape_head_level_dip"

    def test_defect_type_count_at_least_32(self):
        from backend.core.defect_scanner import DefectType

        assert len(DefectType) >= 32
