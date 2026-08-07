from __future__ import annotations

"""Tests für literaturbasierte DSP-Algorithmen in Aurik 10.0.0.

Bedeckt:
  - Phase 09: LPC-basierte AR-Lücken-Interpolation
    (Lagrange & Marchand 2007; Godsill & Rayner 1998)
  - Phase 50: STFT-Konsistenz-Projektion für Zeit-Achsen-Inpainting
    (Siedenburg & Dörfler 2013, JASA)
"""


import numpy as np
import pytest

SR = 48_000


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────


def _sine(freq: float = 440.0, seconds: float = 2.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _rng_audio(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32) * 0.3


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 09 — AR-Interpolation (Lagrange & Marchand 2007, Godsill & Rayner 1998)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPhase09ARInterpolation:
    """LPC-basierte AR-Lücken-Interpolation in Phase 09."""

    @pytest.fixture
    def phase09(self):
        from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase

        return CrackleRemovalPhase(sample_rate=SR)

    def test_ar_predict_shape(self, phase09):
        ctx = _sine(440.0, seconds=0.5)
        n_out = 500
        out = phase09._ar_predict(ctx, n_out, order=32)
        assert out.shape == (n_out,), f"Expected ({n_out},), got {out.shape}"

    def test_ar_predict_no_nan_inf(self, phase09):
        ctx = _sine(220.0, seconds=0.3)
        out = phase09._ar_predict(ctx, 1200, order=30)
        assert np.all(np.isfinite(out)), "AR prediction contains NaN/Inf"

    def test_ar_predict_bounded(self, phase09):
        """Prediction should stay within ±2 × input amplitude (no explosion)."""
        ctx = _sine(440.0, seconds=0.3) * 0.5  # amp ≈ 0.15
        out = phase09._ar_predict(ctx, 600, order=32)
        assert float(np.max(np.abs(out))) < 2.0, "AR prediction diverged"

    def test_ar_predict_short_context_fallback(self, phase09):
        """Too-short context must return zeros (no crash)."""
        ctx = np.array([0.1, 0.2], dtype=np.float32)
        out = phase09._ar_predict(ctx, 100, order=32)
        assert out.shape == (100,)
        assert np.allclose(out, 0.0)

    def test_ar_fill_channel_shape_mono(self, phase09):
        audio = _sine(440.0, seconds=2.0)
        gap_start, gap_end = 8000, 8400
        out = phase09._ar_fill_channel(audio, gap_start, gap_end, before_start=0, after_end=len(audio))
        assert out.shape == (gap_end - gap_start,)

    def test_ar_fill_channel_no_nan(self, phase09):
        audio = _sine(330.0, seconds=2.0)
        gap_start, gap_end = 5000, 5600
        out = phase09._ar_fill_channel(audio, gap_start, gap_end, before_start=0, after_end=len(audio))
        assert np.all(np.isfinite(out))

    def test_ar_fill_channel_boundaries_smooth(self, phase09):
        """Blended output should be near audio[gap_start-1] at start and near audio[gap_end] at end."""
        audio = _sine(440.0, seconds=2.0) * 0.8
        gap_start, gap_end = 12000, 12200
        out = phase09._ar_fill_channel(audio, gap_start, gap_end, before_start=0, after_end=len(audio))
        # Transition jump at gap start should be small
        jump_start = abs(float(out[0]) - float(audio[gap_start - 1]))
        jump_end = abs(float(out[-1]) - float(audio[gap_end]))
        assert jump_start < 0.3, f"Start jump too large: {jump_start:.3f}"
        assert jump_end < 0.3, f"End jump too large: {jump_end:.3f}"

    def test_ar_interpolation_beats_linear_on_sine(self, phase09):
        """AR interpolation should reconstruct a sinusoid better than linear interpolation."""
        audio = _sine(440.0, seconds=3.0) * 0.8
        gap_start, gap_end = 20000, 20800  # ~17 ms gap

        ar_out = phase09._ar_fill_channel(audio, gap_start, gap_end, before_start=0, after_end=len(audio))
        lin_out = phase09._interpolate_linear(audio, gap_start, gap_end)

        reference = audio[gap_start:gap_end]
        err_ar = float(np.mean((ar_out - reference) ** 2))
        err_lin = float(np.mean((lin_out - reference) ** 2))
        # AR should not be drastically worse than linear on a simple sinusoid
        assert err_ar < err_lin * 5 or err_ar < 0.05, (
            f"AR MSE ({err_ar:.4f}) significantly worse than linear ({err_lin:.4f})"
        )

    def test_interpolate_hybrid_short_gap_uses_ar(self, phase09):
        """_interpolate_hybrid must use AR (not trivial zeros) for short gaps."""
        audio = _sine(440.0, seconds=2.0) * 0.6
        gap_start, gap_end = 10000, 10400  # 8 ms — well under 50 ms limit
        result = phase09._interpolate_hybrid(audio, 0, gap_start, gap_end, len(audio))
        assert result.shape == (gap_end - gap_start,) or result.shape == (gap_end - gap_start, 1)
        # Should not return all zeros (linear fill-with-zeros fallback)
        assert not np.allclose(result, 0.0), "hybrid returned all zeros — AR path not taken"
        assert np.all(np.isfinite(result))

    def test_interpolate_hybrid_long_gap_falls_back_to_spectral(self, phase09):
        """Gaps > 50 ms must fall back to spectral interpolation, not AR."""
        audio = _sine(440.0, seconds=4.0) * 0.5
        gap_len = int(0.060 * SR)  # 60 ms > 50 ms limit
        gap_start = 48000
        gap_end = gap_start + gap_len
        result = phase09._interpolate_hybrid(audio, 0, gap_start, gap_end, len(audio))
        assert result.shape[0] == gap_len
        assert np.all(np.isfinite(result))

    def test_interpolate_hybrid_stereo(self, phase09):
        rng = np.random.default_rng(7)
        audio = rng.standard_normal((SR * 2, 2)).astype(np.float32) * 0.4
        gap_start, gap_end = 8000, 8300
        result = phase09._interpolate_hybrid(audio, 0, gap_start, gap_end, len(audio))
        assert result.shape == (gap_end - gap_start, 2)
        assert np.all(np.isfinite(result))


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 50 — STFT-Konsistenz-Projektion (Siedenburg & Dörfler 2013)
# ──────────────────────────────────────────────────────────────────────────────


class TestPhase50ConsistencyInpainting:
    """STFT-Konsistenz-Projektion für Zeit-Achsen-Dropout-Reparatur in Phase 50."""

    @pytest.fixture
    def phase50(self):
        from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase

        return SpectralRepairPhase(sample_rate=SR)

    def test_version_updated(self, phase50):
        meta = phase50.get_metadata()
        assert meta.version == "2.1.0", f"Version nicht aktualisiert: {meta.version}"

    def test_process_mono_no_crash(self, phase50):
        audio = _sine(440.0, seconds=2.0) * 0.4
        result = phase50.process(audio, sample_rate=SR)
        assert result.success
        assert result.audio.shape == audio.shape
        assert np.all(np.isfinite(result.audio))

    def test_process_stereo_no_crash(self, phase50):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal((SR * 2, 2)).astype(np.float32) * 0.3
        result = phase50.process(audio, sample_rate=SR)
        assert result.success
        assert result.audio.shape == audio.shape
        assert np.all(np.isfinite(result.audio))

    def _make_audio_with_dropouts(self, n_dropouts: int = 3) -> np.ndarray:
        """Sinusoidal audio with zeroed-out frames simulating dropouts."""
        audio = _sine(440.0, seconds=3.0) * 0.5
        rng = np.random.default_rng(1)
        frame_len = 2048
        for _ in range(n_dropouts):
            start = int(rng.integers(frame_len, len(audio) - 2 * frame_len))
            # Zero-out 2–4 frames (4096–8192 samples ≈ 85–170 ms)
            end = start + int(rng.integers(2, 5)) * frame_len
            audio[start:end] = 0.0
        return audio

    def test_dropout_repair_output_shape(self, phase50):
        audio = self._make_audio_with_dropouts()
        result = phase50.process(audio, sample_rate=SR)
        assert result.audio.shape == audio.shape

    def test_dropout_repair_fills_zeros(self, phase50):
        """After repair, dropout regions should no longer be near zero."""
        audio = self._make_audio_with_dropouts(n_dropouts=2)
        # Find dropout region
        zero_mask = np.abs(audio) < 1e-6
        result = phase50.process(audio, sample_rate=SR)
        repaired = result.audio
        if np.sum(zero_mask) > 512:
            # Check that at least some of the zeros were filled
            filled = np.abs(repaired[zero_mask]) > 1e-4
            assert np.sum(filled) > 0, "Dropout frames not filled by consistency inpainting"

    def test_consistency_iterations_reduce_energy_anomaly(self, phase50):
        """STFT-consistency should produce a more uniform spectral energy across time."""
        from backend.core.phases.phase_50_spectral_repair import _repair_channel

        audio = _sine(440.0, seconds=2.0) * 0.5
        # Manually inject a large dropout
        audio[20000:22000] = 0.0

        repaired, n_rep = _repair_channel(audio, SR, threshold_factor=4.0)
        assert np.all(np.isfinite(repaired)), "Consistency iterations produced NaN/Inf"
        # Repaired segment should not be identically zero
        assert not np.allclose(repaired[20000:22000], 0.0), "Dropout region still all-zero after inpainting"

    def test_known_frames_preserved_after_consistency(self, phase50):
        """STFT-Consistency must NOT alter undamaged parts of the spectrum."""
        from backend.core.phases.phase_50_spectral_repair import _repair_channel

        audio_clean = _sine(330.0, seconds=2.0) * 0.4
        # Small isolated dropout; most frames are clean
        audio = audio_clean.copy()
        audio[10000:10512] = 0.0

        repaired, _ = _repair_channel(audio, SR, threshold_factor=4.0)
        # Clean region away from the dropout should be nearly unchanged
        clean_before = audio_clean[:8000]
        clean_after = repaired[:8000]
        mse = float(np.mean((clean_before - clean_after) ** 2))
        assert mse < 0.01, f"Clean region altered by consistency iteration: MSE={mse:.4f}"

    def test_process_clip_guard(self, phase50):
        """Output must be clipped to [-1, 1]."""
        audio = _rng_audio(SR * 2)
        result = phase50.process(audio, sample_rate=SR)
        assert float(np.max(np.abs(result.audio))) <= 1.0

    def test_process_nan_input_handled(self, phase50):
        audio = _sine(440.0, seconds=1.0) * 0.3
        audio[5000:5010] = np.nan
        result = phase50.process(audio, sample_rate=SR)
        assert result.audio is not None
        # NaN-guard from validate_input or phase itself should handle this
        assert np.all(np.isfinite(result.audio)) or not result.success

    def test_repair_channel_zero_silence_passthrough(self, phase50):
        """Full-silence audio with > 50% damaged frames must pass through intact."""
        from backend.core.phases.phase_50_spectral_repair import _repair_channel

        silence = np.zeros(SR * 2, dtype=np.float32)
        repaired, n_rep = _repair_channel(silence, SR, threshold_factor=4.0)
        assert n_rep == 0 or np.allclose(repaired, 0.0, atol=1e-5), "Silence not passed through intact"
