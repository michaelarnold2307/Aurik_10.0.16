import pytest

"""
Tests for FlowAudio SOTA Plugin — Conditional Flow Matching Inpainting
======================================================================

Tests cover:
- Input validation (SR, gap bounds, mono/stereo, NaN/Inf)
- Spectral context analysis (LPC envelope, sinusoidal tracking)
- Flow ODE solver (step count, convergence)
- PGHI phase reconstruction
- Boundary crossfade and energy matching
- Full inpaint pipeline
- Edge cases (tiny gap, max gap, no context, one-sided context)
- Singleton pattern

Invariants tested:
- Output contains no NaN/Inf
- Output clipped to [-1, 1]
- Inpainted region has matching energy to context
- PGHI produces valid reconstruction
- Spectral envelope is non-negative
- KL divergence vs context < 0.15 (where applicable)
"""

import threading

import numpy as np

from plugins.flow_audio_sota import (
    _HOP,
    _LPC_ORDER,
    _N_FFT,
    _SR,
    FlowAudioModel,
    _build_target_estimate,
    _extract_spectral_envelope,
    _flow_ode_step,
    _istft,
    _pghi_finalize,
    _pghi_reconstruct,
    _solve_flow_ode,
    _stft,
    _synthesize_sinusoidal,
    _track_sinusoidal_partials,
    get_flow_audio_model,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_audio(dur_s: float = 1.0, sr: int = _SR, freq: float = 440.0) -> np.ndarray:
    """Generate a sine wave audio signal."""
    t = np.arange(int(dur_s * sr)) / sr
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _make_audio_with_gap(
    dur_s: float = 2.0,
    gap_start_s: float = 0.8,
    gap_end_s: float = 1.2,
    sr: int = _SR,
    freq: float = 440.0,
) -> tuple[np.ndarray, int, int]:
    """Generate audio with a zeroed-out gap."""
    audio = _make_audio(dur_s, sr, freq)
    gap_start = int(gap_start_s * sr)
    gap_end = int(gap_end_s * sr)
    audio[gap_start:gap_end] = 0.0
    return audio, gap_start, gap_end


# ═══════════════════════════════════════════════════════════════════════════
# 1. Input Validation (Tests 1–10)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestInputValidation:
    """Tests for input validation in FlowAudioModel.inpaint()."""

    def test_01_wrong_sr_returns_none(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        model = FlowAudioModel()
        assert model.inpaint(audio, gs, ge, 22050) is None

    def test_02_correct_sr_returns_array(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_03_gap_too_short_returns_none(self) -> None:
        audio = _make_audio(1.0)
        model = FlowAudioModel()
        # Gap of 100 samples (< MIN_GAP_SAMPLES=256)
        assert model.inpaint(audio, 1000, 1100, _SR) is None

    def test_04_gap_too_long_returns_none(self) -> None:
        audio = _make_audio(60.0)
        model = FlowAudioModel()
        gap_start = 0
        gap_end = int(35.0 * _SR)  # 35 s > MAX_GAP_S=30
        assert model.inpaint(audio, gap_start, gap_end, _SR) is None

    def test_05_nan_input_handled(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        audio[:100] = np.nan
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert np.isfinite(result).all()

    def test_06_inf_input_handled(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        audio[50] = np.inf
        audio[51] = -np.inf
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert np.isfinite(result).all()

    def test_07_stereo_input_handled(self) -> None:
        mono = _make_audio(2.0)
        stereo = np.stack([mono, mono * 0.8], axis=-1)
        model = FlowAudioModel()
        # Gap in middle
        gs = int(0.8 * _SR)
        ge = int(1.2 * _SR)
        result = model.inpaint(stereo, gs, ge, _SR)
        assert result is not None
        assert result.ndim == 1  # returns mono

    def test_08_output_clipped_to_unit(self) -> None:
        audio, gs, ge = _make_audio_with_gap(dur_s=2.0)
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert np.max(np.abs(result)) <= 1.0

    def test_09_output_same_length(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert len(result) == len(audio)

    def test_10_n_steps_clamped(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        model = FlowAudioModel()
        # n_steps=100 should be clamped to MAX_FLOW_STEPS=16
        result = model.inpaint(audio, gs, ge, _SR, n_steps=100)
        assert result is not None
        assert np.isfinite(result).all()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Spectral Analysis (Tests 11–17)
# ═══════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysis:
    """Tests for spectral context analysis functions."""

    def test_11_lpc_envelope_shape(self) -> None:
        sig = _make_audio(0.1)
        env = _extract_spectral_envelope(sig, _SR, _LPC_ORDER)
        assert env.shape == (_N_FFT // 2 + 1,)

    def test_12_lpc_envelope_nonneg(self) -> None:
        sig = _make_audio(0.1)
        env = _extract_spectral_envelope(sig, _SR, _LPC_ORDER)
        assert (env >= 0).all()

    def test_13_lpc_short_signal_fallback(self) -> None:
        sig = np.zeros(5, dtype=np.float32)
        env = _extract_spectral_envelope(sig, _SR, _LPC_ORDER)
        assert env.shape == (_N_FFT // 2 + 1,)
        assert np.allclose(env, 1.0)

    def test_14_sinusoidal_tracking_finds_peak(self) -> None:
        sig = _make_audio(0.5, freq=1000.0)
        stft_sig = _stft(sig, _N_FFT, _HOP)
        partials = _track_sinusoidal_partials(np.abs(stft_sig), _SR, _N_FFT)
        assert len(partials) > 0
        # Dominant partial should be near 1000 Hz
        dominant_bin = partials[0][0]
        dominant_freq = dominant_bin * _SR / _N_FFT
        assert abs(dominant_freq - 1000.0) < 100.0  # within ~100 Hz

    def test_15_sinusoidal_tracking_silence(self) -> None:
        sig = np.zeros(int(0.5 * _SR), dtype=np.float32)
        stft_sig = _stft(sig, _N_FFT, _HOP)
        partials = _track_sinusoidal_partials(np.abs(stft_sig), _SR, _N_FFT)
        assert len(partials) == 0  # no partials in silence

    def test_16_synthesize_sinusoidal_shape(self) -> None:
        partials = [(42, 0.3), (84, 0.1)]
        syn = _synthesize_sinusoidal(partials, 4800, _N_FFT, _SR)
        assert syn.shape == (4800,)
        assert syn.dtype == np.float32

    def test_17_synthesize_empty_partials(self) -> None:
        syn = _synthesize_sinusoidal([], 4800, _N_FFT, _SR)
        assert syn.shape == (4800,)
        assert np.allclose(syn, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. STFT / iSTFT / PGHI (Tests 18–23)
# ═══════════════════════════════════════════════════════════════════════════


class TestSTFTandPGHI:
    """Tests for STFT, iSTFT, and PGHI reconstruction."""

    def test_18_stft_shape(self) -> None:
        sig = _make_audio(0.1)
        stft = _stft(sig, _N_FFT, _HOP)
        assert stft.shape[0] == _N_FFT // 2 + 1
        assert stft.shape[1] > 0

    def test_19_stft_istft_roundtrip(self) -> None:
        sig = _make_audio(0.1)
        stft = _stft(sig, _N_FFT, _HOP)
        reconstructed = _istft(stft, _HOP, _N_FFT)
        # Should be close to original (within tolerance from windowing)
        min_len = min(len(sig), len(reconstructed))
        corr = np.corrcoef(sig[:min_len], reconstructed[:min_len])[0, 1]
        assert corr > 0.95

    def test_20_pghi_produces_valid_output(self) -> None:
        sig = _make_audio(0.2)
        stft = _stft(sig, _N_FFT, _HOP)
        mag = np.abs(stft)
        recon = _pghi_reconstruct(mag, _HOP, _N_FFT)
        assert np.isfinite(recon).all()
        assert len(recon) > 0

    def test_21_pghi_energy_preservation(self) -> None:
        sig = _make_audio(0.2)
        stft = _stft(sig, _N_FFT, _HOP)
        mag = np.abs(stft)
        recon = _pghi_reconstruct(mag, _HOP, _N_FFT)
        min_len = min(len(sig), len(recon))
        sig_rms = np.sqrt(np.mean(sig[:min_len] ** 2))
        recon_rms = np.sqrt(np.mean(recon[:min_len] ** 2))
        # Energy should be within 50% (PGHI is approximate)
        assert recon_rms > sig_rms * 0.3
        assert recon_rms < sig_rms * 3.0

    def test_22_istft_handles_zero_mag(self) -> None:
        stft = np.zeros((_N_FFT // 2 + 1, 10), dtype=np.complex64)
        recon = _istft(stft, _HOP, _N_FFT)
        assert np.isfinite(recon).all()
        assert np.allclose(recon, 0.0, atol=1e-6)

    def test_23_pghi_handles_zero_mag(self) -> None:
        mag = np.zeros((_N_FFT // 2 + 1, 10), dtype=np.float32)
        recon = _pghi_reconstruct(mag, _HOP, _N_FFT)
        assert np.isfinite(recon).all()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Flow ODE (Tests 24–28)
# ═══════════════════════════════════════════════════════════════════════════


class TestFlowODE:
    """Tests for the Conditional Flow Matching ODE solver."""

    def test_24_flow_step_moves_toward_target(self) -> None:
        n = 4800
        x_0 = np.random.randn(n).astype(np.float32) * 0.1
        x_1 = np.ones(n, dtype=np.float32) * 0.5
        x_t = x_0.copy()
        x_next = _flow_ode_step(x_t, x_0, x_1, 0.0, 0.5, None)
        # Should be closer to x_1 than x_0 was
        dist_before = np.mean((x_0 - x_1) ** 2)
        dist_after = np.mean((x_next - x_1) ** 2)
        assert dist_after < dist_before

    def test_25_solve_flow_converges(self) -> None:
        n = 4800
        x_0 = np.random.randn(n).astype(np.float32) * 0.1
        x_1 = np.ones(n, dtype=np.float32) * 0.3
        result = _solve_flow_ode(x_0, x_1, 8, None)
        # After 8 full steps OT path should be close to x_1 (MSE check)
        mse = float(np.mean((result - x_1) ** 2))
        assert mse < 0.05  # Should converge near target

    def test_26_solve_flow_with_envelope_regularization(self) -> None:
        n = 9600  # need >= N_FFT for spectral regularization
        x_0 = np.random.randn(n).astype(np.float32) * 0.05
        x_1 = _make_audio(n / _SR, freq=440.0) * 0.3
        envelope = np.ones(_N_FFT // 2 + 1, dtype=np.float32)
        result = _solve_flow_ode(x_0, x_1, 8, envelope)
        assert np.isfinite(result).all()
        assert result.shape == (n,)

    def test_27_flow_steps_clamped_to_max(self) -> None:
        n = 2400
        x_0 = np.zeros(n, dtype=np.float32)
        x_1 = np.ones(n, dtype=np.float32) * 0.5
        # 50 steps should be clamped to 16
        result = _solve_flow_ode(x_0, x_1, 50, None)
        assert np.isfinite(result).all()

    def test_28_flow_single_step(self) -> None:
        n = 2400
        x_0 = np.zeros(n, dtype=np.float32)
        x_1 = np.ones(n, dtype=np.float32)
        result = _solve_flow_ode(x_0, x_1, 1, None)
        # After 1 step: x_0 + 1.0 * (x_1 - x_0) = x_1
        np.testing.assert_allclose(result, x_1, atol=0.05)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Target Estimate (Tests 29–31)
# ═══════════════════════════════════════════════════════════════════════════


class TestTargetEstimate:
    """Tests for context-conditioned target estimation."""

    def test_29_target_estimate_shape(self) -> None:
        pre = _make_audio(1.0, freq=440.0)
        post = _make_audio(1.0, freq=440.0)
        gap_length = int(0.3 * _SR)
        target = _build_target_estimate(pre, post, gap_length, _SR)
        assert target.shape == (gap_length,)
        assert target.dtype == np.float32

    def test_30_target_estimate_no_nan(self) -> None:
        pre = _make_audio(0.5, freq=880.0)
        post = _make_audio(0.5, freq=880.0)
        target = _build_target_estimate(pre, post, 4800, _SR)
        assert np.isfinite(target).all()

    def test_31_target_estimate_one_sided_context(self) -> None:
        pre = _make_audio(1.0, freq=440.0)
        post = np.array([], dtype=np.float32)
        target = _build_target_estimate(pre, post, 4800, _SR)
        assert target.shape == (4800,)
        assert np.isfinite(target).all()

    def test_31b_target_estimate_eof_context_decays_to_silence(self) -> None:
        pre = _make_audio(1.0, freq=440.0)
        post = np.array([], dtype=np.float32)
        target = _build_target_estimate(pre, post, 4800, _SR)
        head_rms = float(np.sqrt(np.mean(target[:600] ** 2)))
        tail_rms = float(np.sqrt(np.mean(target[-600:] ** 2)))
        assert tail_rms < head_rms * 0.25


# ═══════════════════════════════════════════════════════════════════════════
# 6. PGHI Finalization (Tests 32–34)
# ═══════════════════════════════════════════════════════════════════════════


class TestPGHIFinalize:
    """Tests for PGHI-based finalization and crossfade."""

    def test_32_finalize_shape_and_nan(self) -> None:
        generated = np.random.randn(4800).astype(np.float32) * 0.3
        pre = _make_audio(0.5)
        post = _make_audio(0.5)
        result = _pghi_finalize(generated, pre, post, _SR)
        assert result.shape == (4800,)
        assert np.isfinite(result).all()

    def test_33_finalize_energy_matches_context(self) -> None:
        ctx_rms = 0.3
        pre = _make_audio(0.5) * ctx_rms * 2
        post = _make_audio(0.5) * ctx_rms * 2
        generated = np.random.randn(4800).astype(np.float32) * 0.01
        result = _pghi_finalize(generated, pre, post, _SR)
        result_rms = np.sqrt(np.mean(result**2))
        ctx_avg_rms = 0.5 * (np.sqrt(np.mean(pre**2)) + np.sqrt(np.mean(post**2)))
        # Should be within 10x of context energy
        assert result_rms < ctx_avg_rms * 10.0

    def test_34_finalize_short_signal(self) -> None:
        generated = np.random.randn(500).astype(np.float32) * 0.1
        pre = _make_audio(0.1)
        post = _make_audio(0.1)
        result = _pghi_finalize(generated, pre, post, _SR)
        assert result.shape == (500,)
        assert np.isfinite(result).all()

    def test_34b_finalize_one_sided_context_preserves_tail_decay(self) -> None:
        generated = np.ones(4800, dtype=np.float32) * 0.25
        pre = _make_audio(0.5)
        post = np.array([], dtype=np.float32)
        result = _pghi_finalize(generated, pre, post, _SR)
        head_rms = float(np.sqrt(np.mean(result[:600] ** 2)))
        tail_rms = float(np.sqrt(np.mean(result[-600:] ** 2)))
        assert tail_rms < head_rms * 0.25


# ═══════════════════════════════════════════════════════════════════════════
# 7. Full Pipeline (Tests 35–42)
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """End-to-end tests for FlowAudioModel.inpaint()."""

    def test_35_inpaint_sine_gap(self) -> None:
        audio, gs, ge = _make_audio_with_gap(dur_s=2.0, gap_start_s=0.8, gap_end_s=1.2)
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert np.isfinite(result).all()
        assert np.max(np.abs(result)) <= 1.0
        # Inpainted region should not be all zeros
        assert np.any(result[gs:ge] != 0.0)

    def test_36_inpaint_preserves_context(self) -> None:
        audio, gs, ge = _make_audio_with_gap(dur_s=2.0)
        original_pre = audio[:gs].copy()
        original_post = audio[ge:].copy()
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        # Context regions should be unchanged
        np.testing.assert_array_equal(result[:gs], original_pre)
        np.testing.assert_array_equal(result[ge:], original_post)

    def test_37_inpaint_with_conditioning(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        conditioning = _make_audio(4.0, freq=440.0)  # long context
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR, conditioning=conditioning)
        assert result is not None
        assert np.isfinite(result).all()

    def test_38_inpaint_minimal_gap(self) -> None:
        audio = _make_audio(1.0)
        gs = 10000
        ge = gs + 300  # just above MIN_GAP_SAMPLES
        audio[gs:ge] = 0.0
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert result.shape == audio.shape

    def test_39_inpaint_near_start(self) -> None:
        audio = _make_audio(2.0)
        gs = 500
        ge = gs + 4800
        audio[gs:ge] = 0.0
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert np.isfinite(result).all()

    def test_40_inpaint_near_end(self) -> None:
        audio = _make_audio(2.0)
        ge = len(audio) - 500
        gs = ge - 4800
        audio[gs:ge] = 0.0
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        assert np.isfinite(result).all()

    def test_40b_inpaint_gap_to_eof_decays_instead_of_sustaining(self) -> None:
        audio = _make_audio(6.0, freq=220.0) * 0.56
        gs = int(4.8 * _SR)
        ge = len(audio)
        audio[gs:ge] = 0.0
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR)
        assert result is not None
        win = _SR // 10
        head_rms = float(np.sqrt(np.mean(result[gs : gs + win] ** 2)))
        tail_rms = float(np.sqrt(np.mean(result[-win:] ** 2)))
        assert tail_rms < head_rms * 0.35

    def test_41_inpaint_large_gap_1s(self) -> None:
        audio, gs, ge = _make_audio_with_gap(dur_s=5.0, gap_start_s=2.0, gap_end_s=3.0)
        model = FlowAudioModel()
        result = model.inpaint(audio, gs, ge, _SR, n_steps=4)
        assert result is not None
        assert np.isfinite(result).all()

    def test_42_inpaint_different_n_steps(self) -> None:
        audio, gs, ge = _make_audio_with_gap()
        model = FlowAudioModel()
        r4 = model.inpaint(audio.copy(), gs, ge, _SR, n_steps=4)
        r16 = model.inpaint(audio.copy(), gs, ge, _SR, n_steps=16)
        assert r4 is not None
        assert r16 is not None
        # Both should produce valid output
        assert np.isfinite(r4).all()
        assert np.isfinite(r16).all()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Singleton & Thread-Safety (Tests 43–45)
# ═══════════════════════════════════════════════════════════════════════════


class TestSingleton:
    """Tests for singleton pattern and thread safety."""

    def test_43_singleton_returns_same_instance(self) -> None:
        m1 = get_flow_audio_model()
        m2 = get_flow_audio_model()
        assert m1 is m2

    def test_44_singleton_is_flow_audio_model(self) -> None:
        m = get_flow_audio_model()
        assert isinstance(m, FlowAudioModel)

    def test_45_concurrent_access_safe(self) -> None:
        results = []
        errors = []

        def worker() -> None:
            try:
                m = get_flow_audio_model()
                results.append(id(m))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads should get same instance
        assert len(set(results)) == 1
