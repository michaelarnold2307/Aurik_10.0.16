"""
tests/unit/test_cumulative_hallucination_tracker.py — §CHT-1 Unit-Tests

Tests for backend/core/dsp/cumulative_hallucination_tracker.py
"""

import numpy as np
import pytest


def _sine(freq: float = 440.0, sr: int = 48000, duration_s: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_returns_same_instance():
    from backend.core.dsp.cumulative_hallucination_tracker import get_cumulative_hallucination_tracker

    a = get_cumulative_hallucination_tracker()
    b = get_cumulative_hallucination_tracker()
    assert a is b


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_state():
    from backend.core.dsp.cumulative_hallucination_tracker import get_cumulative_hallucination_tracker

    cht = get_cumulative_hallucination_tracker()
    cht.reset("restoration")
    assert cht.cumulative_novelty == pytest.approx(0.0, abs=1e-6)
    assert cht.rollback_checkpoint is None


def test_reset_accepts_both_modes():
    from backend.core.dsp.cumulative_hallucination_tracker import get_cumulative_hallucination_tracker

    cht = get_cumulative_hallucination_tracker()
    cht.reset("restoration")
    cht.reset("studio")


# ---------------------------------------------------------------------------
# record_phase() — identical audio → ok (no novelty)
# ---------------------------------------------------------------------------


def test_identical_audio_returns_ok():
    from backend.core.dsp.cumulative_hallucination_tracker import get_cumulative_hallucination_tracker

    cht = get_cumulative_hallucination_tracker()
    cht.reset("restoration")
    audio = _sine(440.0)
    level = cht.record_phase("phase_03_denoise", audio, audio.copy(), sr=48000)
    assert level == "ok"


# ---------------------------------------------------------------------------
# record_phase() — heavily modified audio → warn or critical
# ---------------------------------------------------------------------------


def test_radically_different_audio_escalates():
    from backend.core.dsp.cumulative_hallucination_tracker import get_cumulative_hallucination_tracker

    cht = get_cumulative_hallucination_tracker()
    cht.reset("restoration")
    pre = _sine(440.0, duration_s=2.0)
    # Post is white noise — maximally different
    post = np.random.RandomState(42).randn(len(pre)).astype(np.float32) * 0.5

    # Feed many phases to push cumulative novelty high
    for i in range(6):
        level = cht.record_phase(f"phase_0{i}_test", pre, post, sr=48000)

    # After 6 phases of maximum novelty, level must be warn or critical
    assert level in ("warn", "critical")


# ---------------------------------------------------------------------------
# check_alarm()
# ---------------------------------------------------------------------------


def test_check_alarm_ok_on_fresh_reset():
    from backend.core.dsp.cumulative_hallucination_tracker import get_cumulative_hallucination_tracker

    cht = get_cumulative_hallucination_tracker()
    cht.reset("restoration")
    assert cht.check_alarm() == "ok"


# ---------------------------------------------------------------------------
# get_report()
# ---------------------------------------------------------------------------


def test_get_report_has_required_keys():
    from backend.core.dsp.cumulative_hallucination_tracker import get_cumulative_hallucination_tracker

    cht = get_cumulative_hallucination_tracker()
    cht.reset("restoration")
    audio = _sine(440.0)
    cht.record_phase("phase_03_denoise", audio, audio, sr=48000)
    report = cht.get_report()
    assert isinstance(report, dict)
    assert "cumulative_novelty" in report
    assert "alarm_level" in report
    assert "phases" in report or "phase_records" in report


# ---------------------------------------------------------------------------
# rollback_checkpoint set on first warn
# ---------------------------------------------------------------------------


def test_rollback_checkpoint_set_on_warn():
    from backend.core.dsp.cumulative_hallucination_tracker import (
        get_cumulative_hallucination_tracker,
    )

    cht = get_cumulative_hallucination_tracker()
    cht.reset("restoration")
    pre = _sine(440.0, duration_s=2.0)
    post = np.random.RandomState(7).randn(len(pre)).astype(np.float32) * 0.5

    last_level = "ok"
    for i in range(10):
        last_level = cht.record_phase(f"phase_{i:02d}_test", pre, post, sr=48000)
        if last_level in ("warn", "critical"):
            break

    if last_level in ("warn", "critical"):
        assert cht.rollback_checkpoint is not None
        assert isinstance(cht.rollback_checkpoint, int)


# ---------------------------------------------------------------------------
# Studio thresholds are higher than restoration
# ---------------------------------------------------------------------------


def test_studio_thresholds_higher():
    from backend.core.dsp.cumulative_hallucination_tracker import _THRESHOLDS

    r_warn, r_crit = _THRESHOLDS["restoration"]
    s_warn, s_crit = _THRESHOLDS["studio"]
    assert s_warn > r_warn
    assert s_crit > r_crit
