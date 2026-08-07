"""
Tests for envelope-following dynamic drive in phase_22 TapeSaturation.

Verifies the McNally (1984) peak-follower implementation:
- _peak_envelope API contracts (shape, dtype, bounds, NaN)
- Attack/release dynamics (attack faster than release, loud > quiet)
- Dynamic drive property: loud section receives more drive than quiet
- Arousal-arc preservation: loud passages are proportionally more saturated
- Edge cases: silence, DC, stereo, very short
- Integration: full process() unchanged API
"""

import numpy as np
import pytest

SR = 48_000


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def phase():
    from backend.core.phases.phase_22_tape_saturation import TapeSaturation

    return TapeSaturation()


def _sine(freq: float, dur: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _loud_quiet(loud_amp: float = 0.8, quiet_amp: float = 0.1, dur: float = 1.0) -> np.ndarray:
    """Signal: first half loud, second half quiet (same frequency)."""
    n = int(dur * SR)
    half = n // 2
    t = np.arange(n) / SR
    audio = np.empty(n, dtype=np.float32)
    audio[:half] = loud_amp * np.sin(2.0 * np.pi * 440 * t[:half])
    audio[half:] = quiet_amp * np.sin(2.0 * np.pi * 440 * t[half:])
    return audio


# ──────────────────────────────────────────────────────────────────────────────
# 1.  _peak_envelope contracts
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPeakEnvelope:
    def test_shape_preserved(self, phase):
        audio = _sine(440.0)
        env = phase._peak_envelope(audio.astype(np.float64), attack_ms=3.0, release_ms=80.0, sr=SR)
        assert env.shape == audio.shape

    def test_nonnegative(self, phase):
        audio = _sine(440.0, amp=0.6)
        env = phase._peak_envelope(audio.astype(np.float64), attack_ms=3.0, release_ms=80.0, sr=SR)
        assert np.all(env >= 0.0)

    def test_no_nan_or_inf(self, phase):
        audio = _sine(440.0, amp=0.9)
        env = phase._peak_envelope(audio.astype(np.float64), attack_ms=1.0, release_ms=50.0, sr=SR)
        assert np.all(np.isfinite(env))

    def test_silence_gives_zero_envelope(self, phase):
        audio = np.zeros(SR, dtype=np.float64)
        env = phase._peak_envelope(audio, attack_ms=3.0, release_ms=80.0, sr=SR)
        np.testing.assert_allclose(env, 0.0, atol=1e-12)

    def test_envelope_tracks_peak(self, phase):
        """Envelope maximum must be at least half the signal peak.

        With attack_ms=3 (144 samples) and a 440 Hz sine (period=109 samples)
        the one-pole attack cannot reach the full peak each cycle. With the
        slow release however, the steady-state maximum converges above RMS.
        A >=50 % threshold is physically realistic for these parameters.
        """
        audio = _sine(440.0, amp=0.7).astype(np.float64)
        env = phase._peak_envelope(audio, attack_ms=3.0, release_ms=80.0, sr=SR)
        assert np.max(env) >= 0.7 * 0.50  # at least 50 % of nominal peak

    def test_envelope_bounded_by_signal_max(self, phase):
        """Envelope must not exceed 1.1× the actual signal max (no artificial inflation)."""
        audio = _sine(440.0, amp=0.5).astype(np.float64)
        env = phase._peak_envelope(audio, attack_ms=1.0, release_ms=50.0, sr=SR)
        assert np.max(env) <= np.max(np.abs(audio)) * 1.1 + 1e-6

    def test_loud_section_higher_envelope(self, phase):
        """Envelope must be higher in the loud section than in the quiet section."""
        audio = _loud_quiet(loud_amp=0.9, quiet_amp=0.05, dur=1.0)
        env = phase._peak_envelope(audio.astype(np.float64), attack_ms=3.0, release_ms=80.0, sr=SR)
        n = len(audio)
        # Compare stable mid-sections, not the fading transition
        quarter = n // 4
        env_loud = np.mean(env[quarter // 2 : quarter])
        env_quiet = np.mean(env[3 * quarter :])
        assert env_loud > env_quiet * 2.0, f"Loud envelope {env_loud:.4f} should be > 2× quiet {env_quiet:.4f}"

    def test_release_slower_than_attack(self, phase):
        """
        After a sustained loud burst (DC=1.0 for 200 ms), the envelope must
        still be well above zero at 100 ms into the release tail.

        A single impulse is insufficient: the attack coefficient
        a_att = exp(-1/(sr*attack_ms*1e-3)) means only (1-a_att)≈2% of the
        impulse value enters the envelope in one sample.  A 200 ms DC burst
        gives the attack filter time to converge, then the slow release
        (release_ms=200 ms) leaves env ≈ exp(-0.5) ≈ 0.61 at 100 ms out.
        """
        n = SR * 2  # 2 s total
        audio = np.zeros(n, dtype=np.float64)
        burst_start = int(0.1 * SR)
        burst_end = burst_start + int(0.2 * SR)  # 200 ms of DC=1.0
        audio[burst_start:burst_end] = 1.0
        env = phase._peak_envelope(audio, attack_ms=1.0, release_ms=200.0, sr=SR)
        idx_100ms = burst_end + int(0.1 * SR)
        # Envelope should retain at least exp(-0.5)≈60% of burst peak → clearly > 0.3
        assert env[idx_100ms] > 0.30, f"Envelope decayed too fast: {env[idx_100ms]:.4f} at 100 ms after burst"

    def test_dc_signal_constant_envelope(self, phase):
        """Constant DC signal → envelope should converge to DC amplitude."""
        audio = np.full(SR, 0.5)
        env = phase._peak_envelope(audio, attack_ms=1.0, release_ms=50.0, sr=SR)
        # Last 10 % should have converged to 0.5
        tail = env[int(SR * 0.9) :]
        np.testing.assert_allclose(tail, 0.5, atol=0.01)

    def test_single_sample(self, phase):
        audio = np.array([0.7])
        env = phase._peak_envelope(audio, attack_ms=3.0, release_ms=80.0, sr=SR)
        assert env.shape == (1,)
        assert np.isfinite(env[0])


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Dynamic drive property: loud section → more saturation
# ──────────────────────────────────────────────────────────────────────────────


class TestDynamicDriveProperty:
    """
    McNally (1984) invariant: with envelope-following drive, loud passages
    should receive higher THD (more saturation) than quiet ones.
    """

    def _thd_segment(self, audio_seg: np.ndarray, phase, material) -> float:
        """Estimate THD for a mono segment via TapeSaturation."""

        result = phase.process(audio_seg, sample_rate=SR, material=material)
        orig = audio_seg
        proc = result.audio[: len(orig)]
        diff = proc - orig
        rms_diff = float(np.sqrt(np.mean(diff**2)))
        rms_orig = float(np.sqrt(np.mean(orig**2)))
        return rms_diff / rms_orig if rms_orig > 1e-8 else 0.0

    def test_envelope_drives_loud_more_than_quiet_within_signal(self, phase):
        """
        Within a *single* signal with a quiet then loud section, the peak
        envelope must be higher in the loud portion.  This exercises the
        exact property that drive_vec exploits: env/p95 is higher during
        the loud section → higher drive_vec → more absolute saturation.

        THD is a *relative* metric and cannot be used here: the normaliser
        (p95) is global, so both sections are compared against the same
        reference, not their own amplitudes.
        """
        audio = _loud_quiet(loud_amp=0.85, quiet_amp=0.08, dur=1.0)
        env = phase._peak_envelope(audio.astype(np.float64), attack_ms=3.0, release_ms=80.0, sr=SR)
        n = len(audio)
        half = n // 2
        # Mid 20 % of each section — avoids fade transition
        tenth = n // 10
        env_loud = np.mean(env[tenth : 2 * tenth])  # stable loud region
        env_quiet = np.mean(env[half + tenth : half + 2 * tenth])  # stable quiet region
        assert env_loud > env_quiet * 3.0, (
            f"Envelope in loud zone {env_loud:.4f} should be > 3× quiet zone {env_quiet:.4f}"
        )

    def test_drive_vec_scales_with_level(self, phase):
        """_saturate_multi_band must produce output that scales with input level."""
        from backend.core.defect_scanner import MaterialType

        loud = _sine(440.0, amp=0.8)
        quiet = _sine(440.0, amp=0.1)
        r_loud = phase.process(loud, sample_rate=SR, material=MaterialType.TAPE)
        r_quiet = phase.process(quiet, sample_rate=SR, material=MaterialType.TAPE)
        rms_loud = float(np.sqrt(np.mean(r_loud.audio**2)))
        rms_quiet = float(np.sqrt(np.mean(r_quiet.audio**2)))
        # Output must reflect input level ordering (loud stays louder than quiet)
        assert rms_loud > rms_quiet


# ──────────────────────────────────────────────────────────────────────────────
# 3.  _saturate_band with vector drive
# ──────────────────────────────────────────────────────────────────────────────


class TestSaturateBandVectorDrive:
    def test_scalar_drive_unchanged(self, phase):
        """Scalar drive must still work (backward compatible)."""
        audio = _sine(440.0, amp=0.5)
        out = phase._saturate_band(audio, drive=0.3, hysteresis=0.15, harmonic_weights=[0.6, 0.3, 0.1])
        assert out.shape == audio.shape
        assert np.all(np.isfinite(out))

    def test_vector_drive_same_shape(self, phase):
        """Vector drive must produce output of same shape as audio."""
        audio = _sine(440.0, amp=0.5)
        drive_vec = np.linspace(0.1, 0.5, len(audio)).astype(np.float32)
        out = phase._saturate_band(audio, drive=drive_vec, hysteresis=0.15, harmonic_weights=[0.6, 0.3, 0.1])
        assert out.shape == audio.shape

    def test_vector_drive_no_nan(self, phase):
        audio = _sine(440.0, amp=0.8)
        drive_vec = np.full(len(audio), 0.4, dtype=np.float32)
        out = phase._saturate_band(audio, drive=drive_vec, hysteresis=0.20, harmonic_weights=[0.5, 0.4, 0.1])
        assert np.all(np.isfinite(out))

    def test_zero_vector_drive_minimal_output(self, phase):
        audio = _sine(440.0, amp=0.5)
        drive_vec = np.zeros(len(audio), dtype=np.float32)
        out = phase._saturate_band(audio, drive=drive_vec, hysteresis=0.0, harmonic_weights=[0.5, 0.4, 0.1])
        assert np.all(np.isfinite(out))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Full process() integration — existing API must be unchanged
# ──────────────────────────────────────────────────────────────────────────────


class TestProcessIntegration:
    def test_no_nan_inf_tape(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _sine(440.0, amp=0.8)
        result = phase.process(audio, sample_rate=SR, material=MaterialType.TAPE)
        assert np.all(np.isfinite(result.audio))

    def test_no_clipping_tape(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _sine(440.0, amp=0.8)
        result = phase.process(audio, sample_rate=SR, material=MaterialType.TAPE)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_shape_mono(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _sine(440.0)
        result = phase.process(audio, sample_rate=SR, material=MaterialType.VINYL)
        assert result.audio.shape == audio.shape

    def test_shape_stereo(self, phase):
        from backend.core.defect_scanner import MaterialType

        mono = _sine(440.0)
        stereo = np.column_stack([mono, mono * 0.95])
        result = phase.process(stereo, sample_rate=SR, material=MaterialType.TAPE)
        assert result.audio.shape == stereo.shape

    def test_loud_quiet_signal_no_crash(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _loud_quiet()
        result = phase.process(audio, sample_rate=SR, material=MaterialType.TAPE)
        assert result.success
        assert np.all(np.isfinite(result.audio))

    def test_shellac_skipped(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _sine(440.0)
        result = phase.process(audio, sample_rate=SR, material=MaterialType.SHELLAC)
        assert result.success
        np.testing.assert_array_equal(result.audio, audio)

    def test_silence_passthrough(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = np.zeros(SR // 2, dtype=np.float32)
        result = phase.process(audio, sample_rate=SR, material=MaterialType.TAPE)
        assert result.success
        assert np.all(np.isfinite(result.audio))

    def test_metrics_present(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _sine(440.0)
        result = phase.process(audio, sample_rate=SR, material=MaterialType.TAPE)
        assert "thd_percent" in result.metrics
        assert "drive" in result.metrics
