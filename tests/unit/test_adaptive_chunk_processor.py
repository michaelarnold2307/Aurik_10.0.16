from __future__ import annotations

import pytest

"""Unit-Tests für backend/core/adaptive_chunk_processor.py — §7.6 Severity-adaptive Chunks.

≥ 20 Tests: Chunk-Größen, Crossfade, Stereo, NaN-Guard, Silence, Edge-Cases.
"""


import numpy as np

from backend.core.adaptive_chunk_processor import (
    CHUNK_HIGH_SEV_S,
    CHUNK_LOW_SEV_S,
    CHUNK_MAX_S,
    CHUNK_MED_SEV_S,
    CHUNK_MIN_S,
    CHUNK_SILENCE_S,
    ChunkProcessingResult,
    compute_chunk_size_s,
    get_adaptive_chunk_processor,
    process_in_adaptive_chunks,
)

SR = 48000


def _sine(freq: float = 440.0, secs: float = 5.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _identity_fn(audio, **kwargs):
    """Passthrough phase: returns audio unchanged."""
    return audio


def _boost_fn(audio, **kwargs):
    """Phase that boosts by 0.1 dB (trivial modification)."""
    return np.clip(audio * 1.012, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Chunk-Size Computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeChunkSize:
    def test_01_silence_returns_120s(self):
        assert compute_chunk_size_s(0.0, is_silence=True) == CHUNK_SILENCE_S

    def test_02_high_severity_returns_5s(self):
        assert compute_chunk_size_s(0.6) == CHUNK_HIGH_SEV_S
        assert compute_chunk_size_s(0.9) == CHUNK_HIGH_SEV_S

    def test_03_medium_severity_returns_15s(self):
        assert compute_chunk_size_s(0.3) == CHUNK_MED_SEV_S
        assert compute_chunk_size_s(0.5) == CHUNK_MED_SEV_S

    def test_04_low_severity_returns_60s(self):
        assert compute_chunk_size_s(0.1) == CHUNK_LOW_SEV_S
        assert compute_chunk_size_s(0.0) == CHUNK_LOW_SEV_S

    def test_05_boundary_06_is_high(self):
        assert compute_chunk_size_s(0.6) == CHUNK_HIGH_SEV_S

    def test_06_boundary_03_is_medium(self):
        assert compute_chunk_size_s(0.3) == CHUNK_MED_SEV_S

    def test_07_constants_within_spec_range(self):
        assert CHUNK_MIN_S <= CHUNK_HIGH_SEV_S <= CHUNK_MAX_S
        assert CHUNK_MIN_S <= CHUNK_MED_SEV_S <= CHUNK_MAX_S
        assert CHUNK_MIN_S <= CHUNK_LOW_SEV_S <= CHUNK_MAX_S


# ---------------------------------------------------------------------------
# 2. process_in_adaptive_chunks — Basic
# ---------------------------------------------------------------------------


class TestProcessBasic:
    def test_08_identity_preserves_audio(self):
        audio = _sine(secs=3.0)
        result = process_in_adaptive_chunks(_identity_fn, audio, SR, 0.0)
        assert isinstance(result, ChunkProcessingResult)
        np.testing.assert_allclose(result.audio, audio, atol=1e-5)

    def test_09_short_audio_single_chunk(self):
        audio = _sine(secs=2.0)
        result = process_in_adaptive_chunks(_identity_fn, audio, SR, 0.0)
        assert result.n_chunks == 1

    def test_10_long_audio_multiple_chunks_high_sev(self):
        audio = _sine(secs=30.0)
        result = process_in_adaptive_chunks(_identity_fn, audio, SR, 0.8)
        # 30s / 5s chunk = ~6 chunks
        assert result.n_chunks >= 5
        assert result.chunk_size_s == CHUNK_HIGH_SEV_S

    def test_11_output_shape_matches_input(self):
        audio = _sine(secs=15.0)
        result = process_in_adaptive_chunks(_identity_fn, audio, SR, 0.4)
        assert result.audio.shape == audio.shape

    def test_12_no_nan_in_output(self):
        audio = _sine(secs=15.0)
        result = process_in_adaptive_chunks(_boost_fn, audio, SR, 0.7)
        assert np.isfinite(result.audio).all()

    def test_13_no_clipping(self):
        audio = _sine(secs=15.0)
        result = process_in_adaptive_chunks(_boost_fn, audio, SR, 0.7)
        assert np.max(np.abs(result.audio)) <= 1.0


# ---------------------------------------------------------------------------
# 3. Stereo
# ---------------------------------------------------------------------------


class TestStereo:
    def test_14_stereo_shape_preserved(self):
        mono = _sine(secs=15.0)
        stereo = np.stack([mono, mono * 0.8])
        result = process_in_adaptive_chunks(_identity_fn, stereo, SR, 0.5)
        assert result.audio.ndim == 2
        assert result.audio.shape[0] == 2

    def test_15_stereo_no_nan(self):
        mono = _sine(secs=15.0)
        stereo = np.stack([mono, mono * 0.8])
        result = process_in_adaptive_chunks(_boost_fn, stereo, SR, 0.7)
        assert np.isfinite(result.audio).all()


# ---------------------------------------------------------------------------
# 4. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_16_silence_detection(self):
        audio = np.zeros(SR * 10, dtype=np.float32)
        result = process_in_adaptive_chunks(_identity_fn, audio, SR, 0.0)
        assert result.chunk_size_s == CHUNK_SILENCE_S

    def test_17_nan_input_safe(self):
        audio = _sine(secs=3.0)
        audio[100:200] = np.nan

        def _nan_safe_fn(a, **kw):
            return np.nan_to_num(a, nan=0.0)

        result = process_in_adaptive_chunks(_nan_safe_fn, audio, SR, 0.1)
        assert np.isfinite(result.audio).all()

    def test_18_very_short_audio(self):
        audio = np.zeros(SR, dtype=np.float32)  # 1s
        result = process_in_adaptive_chunks(_identity_fn, audio, SR, 0.9)
        assert result.n_chunks == 1

    def test_19_kwargs_forwarded(self):
        received = {}

        def _capture_fn(audio, **kwargs):
            received.update(kwargs)
            return audio

        audio = _sine(secs=3.0)
        process_in_adaptive_chunks(_capture_fn, audio, SR, 0.0, phase_kwargs={"strength": 0.5})
        assert received.get("strength") == 0.5


# ---------------------------------------------------------------------------
# 5. Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_20_singleton_identity(self):
        a = get_adaptive_chunk_processor()
        b = get_adaptive_chunk_processor()
        assert a is b

    def test_21_singleton_compute_chunk_size(self):
        proc = get_adaptive_chunk_processor()
        assert proc.compute_chunk_size(0.8) == CHUNK_HIGH_SEV_S

    def test_22_singleton_process(self):
        proc = get_adaptive_chunk_processor()
        audio = _sine(secs=3.0)
        result = proc.process(_identity_fn, audio, SR, 0.0)
        assert isinstance(result, ChunkProcessingResult)
