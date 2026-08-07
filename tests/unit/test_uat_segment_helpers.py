from __future__ import annotations

import pytest

"""Unit tests for vocal-focused UAT segment helpers."""


import numpy as np

from tests.test_uat_acceptance_criteria import _estimate_vocal_focus_score, _select_vocal_focus_segments

SR = 48_000


def _vocal_like(duration_s: float, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False, dtype=np.float32)
    envelope = 0.45 + 0.30 * np.sin(2 * np.pi * 2.3 * t)
    harmonics = sum((1.0 / idx) * np.sin(2 * np.pi * freq * idx * t) for idx in range(1, 6))
    return np.clip(0.20 * envelope * harmonics, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_vocal_focus_score_prefers_voiced_signal_over_noise() -> None:
    rng = np.random.default_rng(42)
    vocal = _vocal_like(2.5)
    noise = rng.normal(0.0, 0.08, size=vocal.shape[0]).astype(np.float32)

    assert _estimate_vocal_focus_score(vocal, SR) > _estimate_vocal_focus_score(noise, SR)


def test_select_vocal_focus_segments_returns_ordered_non_overlapping_segments() -> None:
    audio = np.concatenate(
        [
            np.zeros(int(1.0 * SR), dtype=np.float32),
            _vocal_like(2.5),
            np.zeros(int(0.5 * SR), dtype=np.float32),
            _vocal_like(2.5, freq=180.0),
            np.zeros(int(0.5 * SR), dtype=np.float32),
            _vocal_like(2.5, freq=260.0),
        ]
    )
    stereo = np.stack([audio, audio], axis=1)

    segments = _select_vocal_focus_segments(stereo, SR, n_segments=3, segment_seconds=2.0)

    assert len(segments) == 3
    starts = [int(segment["start"]) for segment in segments]
    assert starts == sorted(starts)
    assert min(np.diff(starts)) >= int(2.0 * SR)


def test_select_vocal_focus_segments_handles_short_audio() -> None:
    audio = _vocal_like(1.0)

    segments = _select_vocal_focus_segments(audio, SR, n_segments=3, segment_seconds=2.5)

    assert len(segments) == 1
    assert int(segments[0]["start"]) == 0
    assert int(segments[0]["end"]) == len(audio)
