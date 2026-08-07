from __future__ import annotations

import pytest

"""Regression tests for PMGG content-integrity guard.

Guard objective:
- Catch catastrophic content collapse even when many P1/P2 goals are excluded.
- Stay inactive for mild, legitimate processing changes.
"""


import numpy as np


def _scores(m):
    return dict.fromkeys(m.FAST_GOALS_SUBSET, 0.8)


def _sine(sr: int = 48000, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_content_integrity_penalty_detects_catastrophic_drop():
    from backend.core import per_phase_musical_goals_gate as m

    ref = _sine()
    out = (ref * 0.02).astype(np.float32)
    pen, meta = m._content_integrity_penalty(ref, out)

    assert pen > 0.0
    assert float(meta.get("rms_drop_db", 0.0)) > 12.0


def test_content_integrity_penalty_ignores_mild_change():
    from backend.core import per_phase_musical_goals_gate as m

    ref = _sine()
    out = (ref * 0.95).astype(np.float32)
    pen, meta = m._content_integrity_penalty(ref, out)

    assert pen == 0.0
    assert float(meta.get("rms_drop_db", 99.0)) < 12.0


def test_run_with_retry_forces_best_effort_on_catastrophic_content_loss(monkeypatch):
    from backend.core import per_phase_musical_goals_gate as m

    gate = m.PerPhaseMusicalGoalsGate()
    audio = _sine()

    monkeypatch.setattr(gate, "_run_phase", lambda phase, a, strength, phase_kwargs=None: (a * 0.02).astype(np.float32))
    monkeypatch.setattr(m, "_measure_quick", lambda *args, **kwargs: _scores(m))
    monkeypatch.setattr(gate, "_max_regression", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(gate, "_max_regression_priority_aware", lambda *args, **kwargs: (0.0, 5))
    monkeypatch.setattr(m, "_apply_precise_metric_overrides", lambda scores, audio, sr, reference=None: scores)

    # Use a non-ML-deterministic phase with NO special exclusions so that:
    # (a) _run_phase is called during retries (DSP path — not _wet_dry_blend)
    # (b) skip_drop_check=False → RMS drop penalty stays active
    # (c) skip_corr_check=False → correlation penalty stays active
    # phase_05_rumble_filter was the original choice but is now in _LF_SUBTRACTIVE_DROP_SKIP,
    # which zeroes the drop-penalty (correct for real use, wrong for this test scaffold).
    # phase_10_transient_shaper is a plain DSP phase with no exclusions.
    _audio_out, _scores_out, action, _strength = gate._run_with_retry(
        phase=object(),
        audio=audio,
        sr=48000,
        scores_before=_scores(m),
        phase_id="phase_10_transient_shaper",
        phase_kwargs={},
        threshold=0.02,
        effective_goals=["brillanz"],
        sample_duration_s=1.0,
        initial_strength=1.0,
        defect_locations=None,
        is_studio_2026=False,
    )

    assert action.startswith("best_effort")


# ---------------------------------------------------------------------------
# §2.48 / §2.54: Timing-phase corr-bypass — phase_12 / phase_31
# ---------------------------------------------------------------------------


def _warped(sr: int = 48000, seconds: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (original, warped) pair.

    The original has strongly varying RMS envelope (AM-modulated sine).
    The warped version shuffles 10 ms blocks → envelope order changes → low corr.
    Constant-amplitude sine would yield corr=1.0 regardless of block order.
    """
    rng = np.random.default_rng(42)
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    # AM envelope: slow 0.3 Hz modulation → amplitude varies 0.05–0.45
    env = (0.05 + 0.40 * np.abs(np.sin(2.0 * np.pi * 0.3 * t))).astype(np.float32)
    sig = (env * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    # Shuffle 10 ms blocks to reorder the envelope
    block = sr // 100  # 480 samples @ 48 kHz
    n_blocks = len(sig) // block
    sig_w = sig.copy()
    idx = rng.permutation(n_blocks)
    for i_dst, i_src in enumerate(idx):
        sig_w[i_dst * block : (i_dst + 1) * block] = sig[i_src * block : (i_src + 1) * block]
    return sig, sig_w


def test_timing_phase_skip_corr_no_penalty():
    """skip_corr_check=True must yield penalty=0 even when envelope-corr is low."""
    from backend.core import per_phase_musical_goals_gate as m

    ref, out = _warped(seconds=3.0)  # AM envelope → shuffled → low corr vs original
    # Without skip: corr is low (<0.55) → penalty > 0
    pen_normal, meta_normal = m._content_integrity_penalty(ref, out, skip_corr_check=False)
    # With skip (timing phase): corr penalty is zeroed
    pen_skip, meta_skip = m._content_integrity_penalty(ref, out, skip_corr_check=True)

    # Normal path should detect low corr for shuffled AM signal
    assert meta_normal.get("corr", 1.0) < 0.55, (
        f"test pre-condition: shuffled AM-envelope corr must be low, got {meta_normal.get('corr')}"
    )
    # skip_corr path: penalty must be 0 (RMS is identical so drop_pen is also 0)
    assert pen_skip == 0.0, f"timing-phase skip must yield penalty=0, got {pen_skip}"


def test_timing_corr_exclude_set_covers_phase_12_and_31():
    """_TIMING_CORR_EXCLUDE must contain phase_12 and phase_31 IDs."""
    from backend.core import per_phase_musical_goals_gate as m

    assert "phase_12_wow_flutter_fix" in m._TIMING_CORR_EXCLUDE
    assert "phase_31_speed_pitch_correction" in m._TIMING_CORR_EXCLUDE


def test_content_integrity_still_fires_rms_drop_on_timing_phase():
    """Even with skip_corr_check, catastrophic RMS collapse must still be caught."""
    from backend.core import per_phase_musical_goals_gate as m

    ref, _ = _warped(seconds=3.0)
    out = (ref * 0.01).astype(np.float32)  # −40 dB drop
    pen, meta = m._content_integrity_penalty(ref, out, skip_corr_check=True)

    assert pen > 0.0, "RMS-drop guard must still fire even with skip_corr_check"
    assert meta.get("rms_drop_db", 0.0) > 12.0


def test_run_with_retry_passes_for_mild_change(monkeypatch):
    from backend.core import per_phase_musical_goals_gate as m

    gate = m.PerPhaseMusicalGoalsGate()
    audio = _sine()

    monkeypatch.setattr(gate, "_run_phase", lambda phase, a, strength, phase_kwargs=None: (a * 0.95).astype(np.float32))
    monkeypatch.setattr(m, "_measure_quick", lambda *args, **kwargs: _scores(m))
    monkeypatch.setattr(gate, "_max_regression", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(gate, "_max_regression_priority_aware", lambda *args, **kwargs: (0.0, 5))

    _audio_out, _scores_out, action, _strength = gate._run_with_retry(
        phase=object(),
        audio=audio,
        sr=48000,
        scores_before=_scores(m),
        phase_id="phase_03_denoise",
        phase_kwargs={},
        threshold=0.02,
        effective_goals=["brillanz"],
        sample_duration_s=1.0,
        initial_strength=1.0,
        defect_locations=None,
        is_studio_2026=False,
    )

    assert action in ("passed", "sub_threshold"), f"Expected accepting action, got '{action}'"
