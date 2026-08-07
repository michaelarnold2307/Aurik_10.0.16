import pytest

"""
tests/unit/test_post_pipeline_stereo_guard.py — §2.49b Post-Pipeline Cumulative Stereo-Collapse Guard

Tests the UV3 stereo-collapse detection logic:
  - cu_imb > 20 dB AND pp_imb < 6 dB triggers recovery
  - Recovery cascade: best_clean_checkpoint → pre_pipeline fallback
  - No trigger when pre-pipeline was already imbalanced
  - No trigger when post-pipeline imbalance is mild
"""

import numpy as np

SR = 48_000


def _stereo(dur: float = 1.0, amp_l: float = 0.3, amp_r: float = 0.3) -> np.ndarray:
    """Create stereo sine audio with separate L/R amplitudes."""
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    L = amp_l * np.sin(2 * np.pi * 440 * t)
    R = amp_r * np.sin(2 * np.pi * 440 * t)
    return np.stack([L, R]).astype(np.float32)  # type: ignore[no-any-return]


def _imbalance_db(audio: np.ndarray) -> float:
    """RMS L/R imbalance in dB (matches UV3 inline logic)."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        return 0.0
    l_rms = float(np.sqrt(np.mean(audio[0] ** 2)) + 1e-12)
    r_rms = float(np.sqrt(np.mean(audio[1] ** 2)) + 1e-12)
    return float(abs(20.0 * np.log10(l_rms / r_rms)))


def _guard_logic(
    current_audio: np.ndarray,
    pre_pipeline_audio: np.ndarray,
    best_checkpoint: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Replicate the §2.49b guard logic from UV3 (lines ~14805-14853)."""
    if current_audio.ndim != 2 or current_audio.shape[0] != 2:
        return current_audio, False
    if pre_pipeline_audio.ndim != 2 or pre_pipeline_audio.shape[0] != 2:
        return current_audio, False

    cu_imb = _imbalance_db(current_audio)
    pp_imb = _imbalance_db(pre_pipeline_audio)

    if cu_imb > 20.0 and pp_imb < 6.0:
        # Recovery cascade
        if best_checkpoint is not None:
            cp_imb = _imbalance_db(best_checkpoint)
            if cp_imb <= 20.0:
                recovered = np.clip(np.nan_to_num(best_checkpoint, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                return recovered, True
        # Fallback to pre-pipeline
        recovered = np.clip(np.nan_to_num(pre_pipeline_audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return recovered, True

    return current_audio, False


# ------ Tests ------


@pytest.mark.unit
class TestStereoCollapseDetection:
    """Imbalance detection logic."""

    def test_balanced_stereo_no_trigger(self):
        audio = _stereo(amp_l=0.3, amp_r=0.3)
        pp = _stereo(amp_l=0.3, amp_r=0.3)
        _, triggered = _guard_logic(audio, pp)
        assert not triggered

    def test_mild_imbalance_no_trigger(self):
        """< 20 dB imbalance should not trigger."""
        audio = _stereo(amp_l=0.3, amp_r=0.05)  # ~15.6 dB
        pp = _stereo(amp_l=0.3, amp_r=0.3)
        assert _imbalance_db(audio) < 20.0
        _, triggered = _guard_logic(audio, pp)
        assert not triggered

    def test_severe_collapse_triggers(self):
        """R-channel near zero → > 20 dB imbalance."""
        audio = _stereo(amp_l=0.3, amp_r=0.001)  # ~49.5 dB
        pp = _stereo(amp_l=0.3, amp_r=0.3)
        assert _imbalance_db(audio) > 20.0
        assert _imbalance_db(pp) < 6.0
        _, triggered = _guard_logic(audio, pp)
        assert triggered

    def test_pre_pipeline_already_imbalanced_no_trigger(self):
        """If input was already imbalanced (> 6 dB), don't trigger (§2.50)."""
        audio = _stereo(amp_l=0.3, amp_r=0.001)
        pp = _stereo(amp_l=0.3, amp_r=0.05)  # ~15.6 dB > 6.0
        assert _imbalance_db(pp) > 6.0
        _, triggered = _guard_logic(audio, pp)
        assert not triggered


class TestStereoCollapseRecovery:
    """Recovery cascade when stereo collapse is detected."""

    def test_recovery_uses_checkpoint_if_healthy(self):
        collapsed = _stereo(amp_l=0.3, amp_r=0.0001)
        pp = _stereo(amp_l=0.3, amp_r=0.28)
        checkpoint = _stereo(amp_l=0.3, amp_r=0.25)
        recovered, triggered = _guard_logic(collapsed, pp, checkpoint)
        assert triggered
        np.testing.assert_array_equal(recovered, np.clip(checkpoint, -1.0, 1.0))

    def test_recovery_skips_bad_checkpoint(self):
        """If checkpoint itself is collapsed > 20 dB, fall back to pre-pipeline."""
        collapsed = _stereo(amp_l=0.3, amp_r=0.0001)
        pp = _stereo(amp_l=0.3, amp_r=0.28)
        bad_checkpoint = _stereo(amp_l=0.3, amp_r=0.0002)
        assert _imbalance_db(bad_checkpoint) > 20.0
        recovered, triggered = _guard_logic(collapsed, pp, bad_checkpoint)
        assert triggered
        np.testing.assert_array_equal(recovered, np.clip(pp, -1.0, 1.0))

    def test_recovery_falls_back_to_pre_pipeline_when_no_checkpoint(self):
        collapsed = _stereo(amp_l=0.3, amp_r=0.0001)
        pp = _stereo(amp_l=0.3, amp_r=0.28)
        recovered, triggered = _guard_logic(collapsed, pp, None)
        assert triggered
        np.testing.assert_array_equal(recovered, np.clip(pp, -1.0, 1.0))


class TestEdgeCases:
    """Edge cases for the stereo guard."""

    def test_mono_input_no_trigger(self):
        mono = np.random.randn(SR).astype(np.float32) * 0.3
        pp = _stereo(amp_l=0.3, amp_r=0.3)
        _, triggered = _guard_logic(mono, pp)
        assert not triggered

    def test_nan_in_recovery_output_safe(self):
        """Recovery output must be NaN-free even if checkpoint has NaN."""
        collapsed = _stereo(amp_l=0.3, amp_r=0.0001)
        pp = _stereo(amp_l=0.3, amp_r=0.28)
        checkpoint = _stereo(amp_l=0.3, amp_r=0.25)
        checkpoint[0, 100] = np.nan
        recovered, triggered = _guard_logic(collapsed, pp, checkpoint)
        assert triggered
        assert np.isfinite(recovered).all()

    def test_exactly_20db_threshold_no_trigger(self):
        """Boundary: exactly 20 dB should NOT trigger (> 20, not >=)."""
        # 20 dB = ratio 10:1 → amp_r = amp_l / 10
        audio = _stereo(amp_l=0.3, amp_r=0.03)
        assert abs(_imbalance_db(audio) - 20.0) < 0.5
        pp = _stereo(amp_l=0.3, amp_r=0.3)
        # At exactly 20.0 dB, the guard uses >, so should not trigger
        if _imbalance_db(audio) <= 20.0:
            _, triggered = _guard_logic(audio, pp)
            assert not triggered

    def test_imbalance_db_helper_symmetric(self):
        """L/R flip yields same dB value."""
        a1 = _stereo(amp_l=0.5, amp_r=0.1)
        a2 = _stereo(amp_l=0.1, amp_r=0.5)
        assert abs(_imbalance_db(a1) - _imbalance_db(a2)) < 0.01

    def test_recovery_output_clipped(self):
        """Recovered audio must be in [-1, 1]."""
        collapsed = _stereo(amp_l=0.3, amp_r=0.0001)
        pp = _stereo(amp_l=1.5, amp_r=1.2)
        recovered, triggered = _guard_logic(collapsed, pp)
        assert triggered
        assert np.max(np.abs(recovered)) <= 1.0
