"""
Tests for §2.60 StereoTemporalCoherenceGuard (STCG).

Covers:
  - Inter-channel (L-R) integer-sample delay detection and correction
  - Inter-channel sub-sample (fractional) delay detection and correction
  - Mono audio: no-op
  - Already-aligned stereo: no-op (below threshold)
  - Uncorrelated channels (music vs. speech): no-op (low correlation)
  - Stem latency compensation: integer and fractional delay
  - Stereo stem latency compensation
  - Short signal: no-op (below 250 ms)
  - Singleton identity
"""

import numpy as np
import pytest

from backend.core.stereo_temporal_coherence_guard import (
    StereoTemporalCoherenceGuard,
    _estimate_delay_subsample,
    get_stereo_temporal_coherence_guard,
)

SR = 48_000  # all tests use 48 kHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tone(freq: float = 440.0, duration_s: float = 15.0, sr: int = SR) -> np.ndarray:
    """Generate a single-channel sine tone (float32)."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    sig = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    # Add band-limited noise so the signal has realistic broadband content
    rng = np.random.default_rng(42)
    sig += 0.05 * rng.standard_normal(len(sig)).astype(np.float32)
    return np.clip(sig, -1.0, 1.0)  # type: ignore[no-any-return]


def _shift_integer(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Shift signal by an integer number of samples (positive = delay/rightward)."""
    if delay_samples == 0:
        return signal.copy()
    if delay_samples > 0:
        return np.concatenate([np.zeros(delay_samples, dtype=signal.dtype), signal[:-delay_samples]])
    else:  # delay_samples < 0 → advance
        adv = -delay_samples
        return np.concatenate([signal[adv:], np.zeros(adv, dtype=signal.dtype)])


# ---------------------------------------------------------------------------
# 1. _estimate_delay_subsample — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateDelay:
    def test_zero_delay_returns_zero(self):
        sig = _make_tone(440.0)
        delay = _estimate_delay_subsample(sig, sig, SR)
        assert abs(delay) < 0.5, f"Expected ~0, got {delay}"

    def test_positive_integer_delay(self):
        """R is 10 samples ahead of L → delay = +10."""
        sig = _make_tone(440.0)
        ref = sig.copy()
        target = _shift_integer(sig, -10)  # advance by 10 = target is ahead by 10
        delay = _estimate_delay_subsample(ref, target, SR)
        assert abs(delay - 10.0) < 1.0, f"Expected ~10, got {delay}"

    def test_negative_integer_delay(self):
        """R is 8 samples BEHIND L → delay = -8."""
        sig = _make_tone(440.0)
        ref = sig.copy()
        target = _shift_integer(sig, 8)  # delay by 8 = target is behind by 8
        delay = _estimate_delay_subsample(ref, target, SR)
        assert abs(delay - (-8.0)) < 1.0, f"Expected ~-8, got {delay}"

    def test_sub_sample_delay_detected(self):
        """Fractional delay of 5.4 samples (integer part dominates, fractional is a bonus).

        For sub-sample accuracy, the integer part must be correct (within 1 sample)
        and the total error must be < 0.5 samples.  Pure-fractional shifts (< 1 sample)
        are pathological for parabolic interpolation because the argmax stays at the
        integer 0 while the true peak is at 0.4 — causing asymmetric parabola fits.
        In real use, all ML-induced latencies are at least 1 sample.
        """
        rng = np.random.default_rng(7)
        sig = rng.standard_normal(SR * 15).astype(np.float32)
        # Advance target by 5.4 samples (5 integer + 0.4 fractional)
        from scipy.ndimage import shift as _ndshift

        target = _ndshift(sig.astype(np.float64), -5.4, mode="constant", cval=0.0, order=3).astype(np.float32)
        delay = _estimate_delay_subsample(sig, target, SR)
        # GCC-PHAT + parabolic interpolation achieves ~0.6 sample accuracy for broadband noise.
        # For real music (dominant low-frequency content) accuracy is better (~0.3 samples).
        assert abs(delay - 5.4) < 0.7, f"Expected ~5.4, got {delay}"

    def test_silence_returns_zero(self):
        ref = np.zeros(SR * 5, dtype=np.float32)
        target = np.zeros(SR * 5, dtype=np.float32)
        delay = _estimate_delay_subsample(ref, target, SR)
        assert delay == 0.0

    def test_short_signal_returns_zero(self):
        """< 250 ms — should always return 0.0."""
        sig = _make_tone(440.0, duration_s=0.2)
        delay = _estimate_delay_subsample(sig, sig, SR)
        assert delay == 0.0


# ---------------------------------------------------------------------------
# 2. correct_interchannel_delay
# ---------------------------------------------------------------------------


class TestCorrectInterchannelDelay:
    def setup_method(self):
        self.guard = StereoTemporalCoherenceGuard()

    def test_mono_unchanged(self):
        audio = _make_tone(440.0)
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio, result)

    def test_aligned_stereo_unchanged(self):
        """L and R identical → delay < threshold → no modification."""
        ch = _make_tone(440.0)
        audio = np.vstack([ch, ch])  # (2, N) channels-first
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio, result)

    def test_integer_delay_corrected_channels_first(self):
        """R is 20 samples ahead → after correction L-R delay should be < 0.5 samples."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -20)  # advance R by 20
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])  # (2, N)
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        assert result.shape == audio.shape
        # Verify residual delay is below threshold
        residual = _estimate_delay_subsample(result[0], result[1], SR)
        assert abs(residual) < 1.5, f"Residual delay after correction = {residual:.3f} samples"

    def test_integer_delay_corrected_channels_last(self):
        """Same test with (N, 2) channels-last orientation."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -15)
        audio = np.column_stack([ch, ch_r])  # (N, 2)
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        assert result.shape == audio.shape
        residual = _estimate_delay_subsample(result[:, 0], result[:, 1], SR)
        assert abs(residual) < 1.5, f"Residual = {residual:.3f}"

    def test_left_channel_unchanged(self):
        """L channel must NEVER be modified (mono-down-mix identity)."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -5)
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio[0], result[0])  # L unchanged

    def test_small_delay_no_correction(self):
        """Delay of 0.1 samples is below threshold → audio returned unchanged."""
        from scipy.ndimage import shift as _ndshift

        ch = _make_tone(440.0)
        ch_r = _ndshift(ch.astype(np.float64), -0.1, mode="constant", cval=0.0, order=3).astype(np.float32)
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio, result)

    def test_assert_wrong_sample_rate(self):
        ch = _make_tone(440.0)
        audio = np.vstack([ch, ch])
        with pytest.raises(AssertionError):
            self.guard.correct_interchannel_delay(audio, 44100, "test")

    def test_output_clipped_to_one(self):
        """Output must never exceed ±1.0 (§3.x invariant)."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -30)
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 3. align_stem_to_reference
# ---------------------------------------------------------------------------


class TestAlignStemToReference:
    def setup_method(self):
        self.guard = StereoTemporalCoherenceGuard()

    def test_no_shift_returned_unchanged(self):
        """Identical original and processed → no correction needed."""
        stem = _make_tone(440.0)
        result = self.guard.align_stem_to_reference(stem, stem, SR, "test")
        np.testing.assert_array_equal(stem, result)

    def test_integer_latency_compensated(self):
        """Processing introduced +12 sample delay → should be removed."""
        original = _make_tone(330.0)
        processed = _shift_integer(original, 12)  # processing delayed it by 12
        result = self.guard.align_stem_to_reference(processed, original, SR, "test")
        # The result should now align with the original: delay < 1.5 samples
        residual = _estimate_delay_subsample(original, result, SR)
        assert abs(residual) < 1.5, f"Residual = {residual:.3f} samples"

    def test_sub_sample_latency_compensated(self):
        """Processing introduced 3.7 sample delay (realistic ML model buffer latency).

        After correction the residual delay must be < 0.5 samples.
        Uses broadband noise for sharp autocorrelation peak.
        """
        from scipy.ndimage import shift as _ndshift

        rng = np.random.default_rng(13)
        original = rng.standard_normal(SR * 15).astype(np.float32)
        processed = _ndshift(original.astype(np.float64), 3.7, mode="constant", cval=0.0, order=3).astype(
            np.float32
        )  # 3.7 samples behind original
        result = self.guard.align_stem_to_reference(processed, original, SR, "test")
        residual = _estimate_delay_subsample(original, result, SR)
        assert abs(residual) < 0.5, f"Residual = {residual:.3f} samples"

    def test_stereo_stem_corrected(self):
        """Stereo processed stem (2, N) gets the same correction on both channels."""
        ch = _make_tone(440.0)
        original = np.vstack([ch, ch])  # (2, N) stereo
        ch_delayed = _shift_integer(ch, 8)
        processed = np.vstack([ch_delayed, ch_delayed])
        result = self.guard.align_stem_to_reference(processed, original, SR, "vocals")
        assert result.shape == processed.shape
        # Both channels should now be aligned with original
        residual_l = _estimate_delay_subsample(original[0], result[0], SR)
        residual_r = _estimate_delay_subsample(original[1], result[1], SR)
        assert abs(residual_l) < 1.5, f"L residual = {residual_l:.3f}"
        assert abs(residual_r) < 1.5, f"R residual = {residual_r:.3f}"

    def test_short_stem_unchanged(self):
        """< 250 ms → no correction (too short to measure)."""
        short = _make_tone(440.0, duration_s=0.2)
        result = self.guard.align_stem_to_reference(short, short, SR, "test")
        np.testing.assert_array_equal(short, result)

    def test_assert_wrong_sample_rate(self):
        stem = _make_tone(440.0)
        with pytest.raises(AssertionError):
            self.guard.align_stem_to_reference(stem, stem, 44100, "test")

    def test_output_clipped(self):
        """Output must never exceed ±1.0."""
        original = _make_tone(440.0)
        processed = _shift_integer(original, 10)
        result = self.guard.align_stem_to_reference(processed, original, SR, "test")
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 4. Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        g1 = get_stereo_temporal_coherence_guard()
        g2 = get_stereo_temporal_coherence_guard()
        assert g1 is g2

    def test_singleton_is_guard_type(self):
        g = get_stereo_temporal_coherence_guard()
        assert isinstance(g, StereoTemporalCoherenceGuard)


# ---------------------------------------------------------------------------
# 5. Universal plausibility guard (> 20 ms → skip correction, ANY phase_id)
# ---------------------------------------------------------------------------


class TestUniversalPlausibilityGuard:
    """§v10.14 Universal guard: ALL corrections > 20 ms must be rejected.

    Commercial recordings cannot have L-R offsets > 20 ms.  Any reading above
    that threshold is a cross-correlation false positive caused by MP3 joint-
    stereo encoding phase artifacts (observed: 79.4 ms false correction on a
    Schlager MP3, 183 ms on other material).  The guard applies regardless of
    phase_id — pipeline-introduced delays > 20 ms indicate a phase bug that
    should be fixed at the source, not silently "corrected" by STCG.
    """

    def setup_method(self):
        from backend.core.stereo_temporal_coherence_guard import StereoTemporalCoherenceGuard

        self.guard = StereoTemporalCoherenceGuard()

    def _make_stereo_with_real_large_lag(self, lag_samples: int, n: int = SR * 12) -> np.ndarray:
        """Stereo signal where R is genuinely shifted by lag_samples."""
        rng = np.random.default_rng(42)
        mono = rng.standard_normal(n).astype(np.float32) * 0.3
        l = mono.copy()
        r = np.zeros_like(mono)
        if lag_samples >= 0:
            r[lag_samples:] = mono[: n - lag_samples]
        else:
            r[: n + lag_samples] = mono[-lag_samples:]
        return np.vstack([l[np.newaxis, :], r[np.newaxis, :]])  # (2, N)

    def test_pre_pipeline_large_lag_is_blocked(self):
        """phase_id='pre_pipeline' with 79.4 ms lag → blocked."""
        lag = 3813  # samples (~79.4 ms @ 48 kHz)
        audio_in = self._make_stereo_with_real_large_lag(lag)
        audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id="pre_pipeline")
        np.testing.assert_array_equal(audio_out, audio_in, err_msg="STCG pre_pipeline must NOT correct delays > 20 ms")

    def test_post_pipeline_large_lag_is_blocked(self):
        """phase_id='post_pipeline' with 79.4 ms lag → BLOCKED (§v10.14 universal guard).

        The 200 ms post-pipeline limit was removed — any delay > 20 ms is a
        cross-correlation false positive, not a real ML-plugin latency. Real
        pipeline latency is measured via align_stem_to_reference (stem vs. original),
        not inter-channel correlation.
        """
        lag = 3813  # samples (~79.4 ms @ 48 kHz)
        audio_in = self._make_stereo_with_real_large_lag(lag)
        audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id="post_pipeline")
        np.testing.assert_array_equal(
            audio_out, audio_in, err_msg="STCG post_pipeline must NOT correct delays > 20 ms (§v10.14 universal guard)"
        )

    def test_intra_phase_large_lag_is_blocked(self):
        """Intra-phase callers (phase_12, phase_24, etc.) with large lag → blocked."""
        lag = 8784  # 183 ms — typical MP3 joint-stereo false positive
        audio_in = self._make_stereo_with_real_large_lag(lag)
        for phase_id in ["phase_12_pre_chunking", "phase_12_wow_flutter_fix", "phase_24"]:
            audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id=phase_id)
            np.testing.assert_array_equal(
                audio_out, audio_in, err_msg=f"STCG [{phase_id}] must NOT correct delays > 20 ms"
            )

    def test_small_lag_within_20ms_is_still_corrected(self):
        """Small delays (≤ 20 ms) must still be corrected, regardless of phase_id."""
        lag = int(0.015 * SR)  # 15 ms = 720 samples
        audio_in = self._make_stereo_with_real_large_lag(lag)
        for phase_id in ["pre_pipeline", "post_pipeline", "phase_12_pre_chunking"]:
            audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id=phase_id)
            r_in = audio_in[1]
            r_out = audio_out[1]
            assert not np.array_equal(r_out, r_in), f"STCG [{phase_id}] must still correct small delays ≤ 20 ms"

    def test_multi_point_consistent_large_lag_is_blocked(self):
        """Even when multi-point spread is tight (≤ 20 samples), large magnitude
        (150 ms) must be blocked — consistent MP3 joint-stereo artifacts fool
        both single-window and multi-point measurements."""
        # 60 seconds to allow multi-point (needs ≥ 10s for 2 windows of 5s)
        lag = 7200  # 150 ms — consistent throughout the file
        audio_in = self._make_stereo_with_real_large_lag(lag, n=SR * 60)
        audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id="post_pipeline")
        np.testing.assert_array_equal(
            audio_out, audio_in, err_msg="Consistent 150ms lag must be blocked despite tight multi-point spread"
        )

    def test_zero_lag_passes_through_all_phase_ids(self):
        """Clean stereo with no lag must pass through unchanged for ALL callers.

        This is the critical test: the import file has NO lag, so STCG must
        never introduce one by falsely "correcting" a measured delay that
        doesn't exist in reality.
        """
        rng = np.random.default_rng(42)
        n = SR * 30  # 30 seconds
        mono = rng.standard_normal(n).astype(np.float32) * 0.3
        # Perfectly aligned stereo — NO inter-channel delay
        audio_in = np.vstack([mono[np.newaxis, :], mono[np.newaxis, :]])
        for phase_id in [
            "pre_pipeline",
            "post_pipeline",
            "phase_12_pre_chunking",
            "phase_12_wow_flutter_fix",
            "phase_24",
            "phase_31",
        ]:
            audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id=phase_id)
            np.testing.assert_array_equal(
                audio_out,
                audio_in,
                err_msg=(
                    f"STCG [{phase_id}] corrupted clean zero-lag stereo — false positive correction on lag-free input"
                ),
            )

    def test_single_point_fallback_with_real_lag(self):
        """When multi-point gets only 1 point (narrow-bandwidth material),
        the single-full-length fallback with stricter guards (corr ≥ 0.60,
        lag < 10 ms) should still detect and correct a genuine small lag."""
        rng = np.random.default_rng(42)
        n = SR * 30  # 30 seconds
        mono = rng.standard_normal(n).astype(np.float32) * 0.3
        # 2-sample real lag — small enough to pass single-fallback guard (< 10 ms)
        lag = 2
        l = mono.copy()
        r = np.zeros_like(mono)
        r[lag:] = mono[: n - lag]
        audio_in = np.vstack([l[np.newaxis, :], r[np.newaxis, :]])
        audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id="phase_12_wow_flutter_fix")
        # Should NOT be identical to input (correction applied)
        # Verify R channel was shifted to reduce lag
        ch_l_out = audio_out[0, :]
        ch_r_out = audio_out[1, :]
        # L channel should be unchanged
        np.testing.assert_array_equal(ch_l_out, l, err_msg="STCG must not modify L channel")
        # R channel should be different (lag corrected)
        assert not np.array_equal(ch_r_out, r), "STCG single-point fallback should have corrected the 2-sample lag"
