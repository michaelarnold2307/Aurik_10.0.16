import pytest

"""Tests for §2.35b vocal_proximity_score (compute_vocal_proximity_score).

Verifies:
- Correct return dict structure
- Good restoration → score near 1.0
- Aggressive denoising → lower breathiness_ratio
- Stereo input handling
- Edge cases (short audio, silence, vocal_segments)
"""

import numpy as np


def _make_vocal_signal(sr: int = 48000, duration_s: float = 2.0) -> np.ndarray:
    """Create a synthetic vocal-like signal with transients and pauses."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n)
    # Fundamental + formants
    sig = 0.3 * np.sin(2 * np.pi * 220 * t)
    sig += 0.15 * np.sin(2 * np.pi * 440 * t)
    sig += 0.08 * np.sin(2 * np.pi * 880 * t)
    # Plosive transients at regular intervals
    rng = np.random.RandomState(42)
    for pos_s in [0.3, 0.7, 1.2, 1.6]:
        idx = int(pos_s * sr)
        burst_len = int(0.005 * sr)
        if idx + burst_len < n:
            sig[idx : idx + burst_len] += 0.4 * rng.randn(burst_len)
    # Breath noise in a pause
    pause_start = int(0.45 * sr)
    pause_end = int(0.55 * sr)
    sig[pause_start:pause_end] = 0.002 * rng.randn(pause_end - pause_start)
    return sig.astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
class TestVocalProximityScore:
    """§2.35b compute_vocal_proximity_score unit tests."""

    def test_import(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        assert callable(compute_vocal_proximity_score)

    def test_return_dict_keys(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig = _make_vocal_signal()
        rest = orig * 0.95
        result = compute_vocal_proximity_score(orig, rest, 48000)
        assert isinstance(result, dict)
        for key in [
            "proximity_score",
            "konsonanten_transient_energy_ratio",
            "breathiness_ratio",
            "early_reflection_preservation",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_good_restoration_near_one(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig = _make_vocal_signal()
        rest = orig * 0.97 + 0.0005 * np.random.RandomState(1).randn(len(orig)).astype(np.float32)
        result = compute_vocal_proximity_score(orig, rest, 48000)
        assert result["proximity_score"] >= 0.70, f"Good restoration should score high, got {result}"

    def test_aggressive_denoising_lowers_breathiness(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig = _make_vocal_signal()
        # Aggressively remove quiet parts (breath) → breathiness drops
        rest = orig.copy()
        rest[np.abs(rest) < 0.01] = 0.0  # kill all quiet content
        result = compute_vocal_proximity_score(orig, rest, 48000)
        assert result["breathiness_ratio"] < 1.0

    def test_stereo_input(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig_mono = _make_vocal_signal()
        orig_stereo = np.column_stack([orig_mono, orig_mono * 0.9])
        rest_stereo = orig_stereo * 0.96
        result = compute_vocal_proximity_score(orig_stereo, rest_stereo, 48000)
        assert 0.0 <= result["proximity_score"] <= 1.5

    def test_short_audio_fallback(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        result = compute_vocal_proximity_score(np.zeros(100), np.zeros(100), 48000)
        assert result["proximity_score"] == 1.0  # fallback

    def test_silence_no_crash(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        sr = 48000
        silent = np.zeros(sr * 2, dtype=np.float32)
        result = compute_vocal_proximity_score(silent, silent, sr)
        assert isinstance(result["proximity_score"], float)

    def test_vocal_segments_param(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig = _make_vocal_signal()
        rest = orig * 0.95
        result = compute_vocal_proximity_score(orig, rest, 48000, vocal_segments=[(0.3, 1.5)])
        assert 0.0 <= result["proximity_score"] <= 1.5

    def test_identical_audio_high_score(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig = _make_vocal_signal()
        result = compute_vocal_proximity_score(orig, orig.copy(), 48000)
        assert result["proximity_score"] >= 0.90

    def test_values_are_finite(self):
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig = _make_vocal_signal()
        rest = orig * 0.9
        result = compute_vocal_proximity_score(orig, rest, 48000)
        for k, v in result.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_channels_first_stereo(self):
        """Handle (2, N) channels-first stereo format."""
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score

        orig_mono = _make_vocal_signal()
        orig_cf = np.stack([orig_mono, orig_mono * 0.9])  # (2, N)
        rest_cf = orig_cf * 0.95
        result = compute_vocal_proximity_score(orig_cf, rest_cf, 48000)
        assert 0.0 <= result["proximity_score"] <= 1.5
