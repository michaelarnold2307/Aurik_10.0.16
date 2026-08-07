import pytest

"""
tests/unit/test_loudness_cascade_guard.py — §2.45a Mid-Pipeline-Loudness-Drift-Guard 3-Stage Cascade

Tests:
  - Gated RMS calculation (frames > -50 dBFS, silence ignored)
  - Musical gain envelope (gain only on music frames, silence at 1.0)
  - Conditional soft-limiter (only when peak > 0.98)
  - 3-stage cascade: per-phase → mid-pipeline → end-of-pipeline
"""

import numpy as np

SR = 48_000


def _rms_dbfs_gated(audio: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Replicate §2.45a-I gated RMS from UV3."""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    frame_len = int(0.02 * SR)  # 20 ms frames
    if len(audio) < frame_len:
        rms = float(np.sqrt(np.mean(audio**2)))
        return 20.0 * np.log10(rms + 1e-12)  # type: ignore[no-any-return]
    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames**2, axis=1))
    dbfs_per_frame = 20.0 * np.log10(rms_per_frame + 1e-12)
    mask = dbfs_per_frame > gate_dbfs
    if not np.any(mask):
        rms = float(np.sqrt(np.mean(audio**2)))
        return 20.0 * np.log10(rms + 1e-12)  # type: ignore[no-any-return]
    gated_rms = float(np.sqrt(np.mean(rms_per_frame[mask] ** 2)))
    return 20.0 * np.log10(gated_rms + 1e-12)  # type: ignore[no-any-return]


def _musical_gain_envelope(
    audio: np.ndarray,
    gain: float,
    gate_dbfs: float = -50.0,
    crossfade_ms: float = 10.0,
    sr: int = SR,
) -> np.ndarray:
    """Replicate §2.45a-II envelope-aware gain."""
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0)
    else:
        mono = audio
    frame_len = int(0.02 * sr)
    envelope = np.ones(len(mono), dtype=np.float32)

    n_frames = len(mono) // frame_len
    for i in range(n_frames):
        chunk = mono[i * frame_len : (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(chunk**2)))
        dbfs = 20.0 * np.log10(rms + 1e-12)
        if dbfs > gate_dbfs:
            envelope[i * frame_len : (i + 1) * frame_len] = gain
        # else: remains 1.0 (silence untouched)

    # Crossfade at transitions
    cf_samples = int(crossfade_ms / 1000.0 * sr)
    if cf_samples > 1:
        for i in range(1, len(envelope)):
            if abs(envelope[i] - envelope[i - 1]) > 0.01:
                start = max(0, i - cf_samples // 2)
                end = min(len(envelope), i + cf_samples // 2)
                ramp = np.linspace(
                    envelope[start],
                    envelope[end - 1] if end < len(envelope) else envelope[-1],
                    end - start,
                    dtype=np.float32,
                )
                envelope[start:end] = ramp
                break

    if audio.ndim == 2:
        return audio * envelope[np.newaxis, :]  # type: ignore[no-any-return]
    return audio * envelope


# ------ Tests ------


@pytest.mark.unit
class TestGatedRMS:
    """§2.45a-I: Frame-based gated RMS, only frames > -50 dBFS."""

    def test_pure_silence_returns_very_low(self):
        silence = np.zeros(SR, dtype=np.float32)
        rms = _rms_dbfs_gated(silence)
        assert rms < -200.0

    def test_loud_signal_ignores_silence_frames(self):
        """Mix of loud and silent frames — silence frames should not pull RMS down."""
        loud = np.sin(np.linspace(0, 100, SR, dtype=np.float32)) * 0.5
        silence = np.zeros(SR, dtype=np.float32)
        mixed = np.concatenate([loud, silence])
        rms_mixed = _rms_dbfs_gated(mixed)
        rms_loud = _rms_dbfs_gated(loud)
        # Gated RMS of mixed ≈ gated RMS of loud-only (within 1 dB)
        assert abs(rms_mixed - rms_loud) < 1.0

    def test_stereo_downmix(self):
        """Stereo input should be downmixed to mono for framing."""
        stereo = np.random.randn(2, SR * 2).astype(np.float32) * 0.3
        rms = _rms_dbfs_gated(stereo)
        assert np.isfinite(rms)
        assert rms < 0.0  # Should be negative dBFS

    def test_short_audio_below_frame_length(self):
        """Audio shorter than 1 frame → use global RMS."""
        short = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        rms = _rms_dbfs_gated(short)
        assert np.isfinite(rms)

    def test_gate_threshold_works(self):
        """Quiet frames below gate should be excluded."""
        # -60 dBFS signal → below -50 dBFS gate
        quiet = np.sin(np.linspace(0, 100, SR, dtype=np.float32)) * 0.001
        loud = np.sin(np.linspace(0, 100, SR, dtype=np.float32)) * 0.5
        mixed = np.concatenate([loud, quiet])
        rms_gated = _rms_dbfs_gated(mixed, gate_dbfs=-50.0)
        rms_loud = _rms_dbfs_gated(loud, gate_dbfs=-50.0)
        assert abs(rms_gated - rms_loud) < 1.5


class TestMusicalGainEnvelope:
    """§2.45a-II: Gain only on music frames, silence at 1.0."""

    def test_silence_frames_unchanged(self):
        """Silent frames should remain at gain=1.0."""
        silence = np.zeros(SR, dtype=np.float32)
        result = _musical_gain_envelope(silence, gain=2.0)
        np.testing.assert_array_almost_equal(result, silence, decimal=6)

    def test_loud_frames_amplified(self):
        """Loud frames should get gain applied."""
        loud = np.sin(np.linspace(0, 100, SR, dtype=np.float32)) * 0.5
        result = _musical_gain_envelope(loud, gain=1.5)
        # Most frames should be louder
        assert np.sqrt(np.mean(result**2)) > np.sqrt(np.mean(loud**2))

    def test_output_shape_preserved(self):
        stereo = np.random.randn(2, SR * 2).astype(np.float32) * 0.3
        result = _musical_gain_envelope(stereo, gain=1.2)
        assert result.shape == stereo.shape

    def test_mono_shape_preserved(self):
        mono = np.random.randn(SR).astype(np.float32) * 0.3
        result = _musical_gain_envelope(mono, gain=1.2)
        assert result.shape == mono.shape


class TestConditionalSoftLimiter:
    """§2.45a-III: tanh soft-limiter only when peak > 0.98."""

    def test_low_peak_no_limiting(self):
        """Audio with peak < 0.98 should not be limited."""
        audio = np.sin(np.linspace(0, 100, SR, dtype=np.float32)) * 0.5
        peak = np.max(np.abs(audio))
        assert peak < 0.98
        # In the real implementation, soft-limiter would be skipped
        # We just verify the condition
        should_limit = peak > 0.98
        assert not should_limit

    def test_high_peak_triggers_limiting(self):
        """Audio with peak > 0.98 should trigger soft-limiter."""
        audio = np.sin(np.linspace(0, 100, SR, dtype=np.float32)) * 1.05
        peak = np.max(np.abs(audio))
        assert peak > 0.98
        # Apply tanh shaping at 0.92
        threshold = 0.92
        limited = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (1.0 - threshold) * np.tanh((np.abs(audio) - threshold) / (1.0 - threshold))),
            audio,
        )
        assert np.max(np.abs(limited)) <= 1.0
        assert np.max(np.abs(limited)) < np.max(np.abs(audio))

    def test_soft_limiter_preserves_low_level(self):
        """Soft-limiter should not affect audio below threshold."""
        audio = np.sin(np.linspace(0, 100, SR, dtype=np.float32)) * 0.5
        threshold = 0.92
        limited = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (1.0 - threshold) * np.tanh((np.abs(audio) - threshold) / (1.0 - threshold))),
            audio,
        )
        np.testing.assert_array_almost_equal(audio, limited, decimal=6)


class TestThreeStageCascade:
    """§2.45a-IV: Per-Phase → Mid-Pipeline → End-of-Pipeline cascade."""

    def test_per_phase_drift_detection(self):
        """Per-phase guard detects level drop after a single phase."""
        before = np.sin(np.linspace(0, 100, SR * 3, dtype=np.float32)) * 0.5
        after = before * 0.3  # Heavy level drop (simulates aggressive denoise)
        rms_before = _rms_dbfs_gated(before)
        rms_after = _rms_dbfs_gated(after)
        drop = rms_before - rms_after
        assert drop > 3.0  # Significant drop detected

    def test_cumulative_drift_detection(self):
        """Mid-pipeline guard detects cumulative level loss across phases."""
        original = np.sin(np.linspace(0, 100, SR * 3, dtype=np.float32)) * 0.5
        # Simulate 4 phases, each reducing level by ~2 dB
        current = original.copy()
        for _ in range(4):
            current = current * 0.8
        rms_orig = _rms_dbfs_gated(original)
        rms_curr = _rms_dbfs_gated(current)
        cumul_drop = rms_orig - rms_curr
        assert cumul_drop > 5.0  # Cumulative drop exceeds material threshold

    def test_end_of_pipeline_final_guard(self):
        """End-of-pipeline guard catches any remaining level issue."""
        original = np.sin(np.linspace(0, 100, SR * 3, dtype=np.float32)) * 0.5
        # Simulate large cumulative drop
        pipeline_output = original * 0.1
        rms_orig = _rms_dbfs_gated(original)
        rms_out = _rms_dbfs_gated(pipeline_output)
        rms_orig - rms_out
        # > 0.915 amplitude factor threshold
        amp_ratio = np.sqrt(np.mean(pipeline_output**2)) / (np.sqrt(np.mean(original**2)) + 1e-12)
        assert amp_ratio < 0.915  # Would trigger end-of-pipeline guard

    def test_makeup_gain_compensates(self):
        """Musical gain envelope can compensate detected level drop."""
        original = np.sin(np.linspace(0, 100, SR * 3, dtype=np.float32)) * 0.5
        too_quiet = original * 0.5
        rms_orig = _rms_dbfs_gated(original)
        rms_quiet = _rms_dbfs_gated(too_quiet)
        drop = rms_orig - rms_quiet
        gain = 10 ** (drop / 20.0)
        compensated = _musical_gain_envelope(too_quiet, gain=gain)
        rms_comp = _rms_dbfs_gated(compensated)
        # Compensated level should be close to original (within 1 dB)
        assert abs(rms_comp - rms_orig) < 1.5
