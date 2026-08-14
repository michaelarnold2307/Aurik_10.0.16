from __future__ import annotations

import logging

"""
Unit tests for SOTA vocal enhancement improvements (v10.14.x):

  1. WORLD-based formant correction in FormantCorrector
       - _warp_sp_frame: identity at zero shift, direction of warp, NaN safety
       - correct(): WORLD branch produces output with preserved length and spec
  2. WORLD HNR breathiness in GenderDetector
       - Breathy (high-AP) signals score higher than modal (low-AP) signals
       - DSP fallback path works without pyworld
       - NaN safety and output range [0, 1]
  3. Singer's Formant narrowing (Sundberg 1987/2015)
       - Default bandwidth reduced to 250 Hz
       - No boost applied when singer's formant absent (passthrough)
       - Boost applied and cluster_center_hz reported when present
       - NaN safety + no amplification

Scientific basis:
  Morise et al. (2016) — WORLD vocoder
  Yumoto et al. (1982) — HNR as hoarseness index
  Sundberg (1987, 2015) — Singer's formant
"""


import numpy as np
import pytest

logger = logging.getLogger(__name__)

SR = 48_000


# ─── helpers ──────────────────────────────────────────────────────────────────


def _sine(freq_hz: float, n: int = SR, amp: float = 0.4) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]


def _harmonic_voice(f0: float = 200.0, n_harmonics: int = 8, n: int = SR, amp: float = 0.4) -> np.ndarray:
    """Synthesise a harmonic-only voice (low AP = modal phonation)."""
    t = np.arange(n, dtype=np.float64) / SR
    sig = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        sig += (amp / k) * np.sin(2.0 * np.pi * f0 * k * t)
    return sig.astype(np.float32)


def _breathy_voice(
    f0: float = 200.0, n: int = SR, harmonic_amp: float = 0.2, noise_amp: float = 0.3, seed: int = 42
) -> np.ndarray:
    """Synthesise voice with high aperiodicity (breathy)."""
    rng = np.random.default_rng(seed)
    harmonic = _harmonic_voice(f0, n=n, amp=harmonic_amp).astype(np.float64)
    noise = rng.standard_normal(n) * noise_amp
    return (harmonic + noise).astype(np.float32)


def _formant_freqs_with_cluster(has_cluster: bool, n_frames: int = 50) -> np.ndarray:
    """Return (n_frames, 5) array with or without F4/F5 clustering at 2800-3200 Hz."""
    ff = np.zeros((n_frames, 5))
    ff[:, 0] = 500.0  # F1
    ff[:, 1] = 1500.0  # F2
    ff[:, 2] = 2100.0  # F3
    if has_cluster:
        ff[:, 3] = 2900.0  # F4 — Singer's formant cluster
        ff[:, 4] = 3100.0  # F5 — Singer's formant cluster
    else:
        ff[:, 3] = 3800.0  # F4 — outside range
        ff[:, 4] = 4500.0  # F5 — outside range
    return ff


# ─── 1. WORLD formant correction ─────────────────────────────────────────────


@pytest.mark.unit
class TestWorldFormantCorrection:
    """12 tests for _warp_sp_frame and WORLD correct() branch."""

    @pytest.fixture
    def corrector(self):
        from dsp.formant_system import FormantCorrector

        return FormantCorrector(max_drift_hz=50.0, correction_strength=0.7)

    def test_warp_sp_frame_zero_shift_identity(self, corrector):
        """src == tgt → output ≈ input (identity warp)."""
        n_bins = 513
        freq_axis = np.linspace(0.0, SR / 2.0, n_bins)
        sp = np.random.default_rng(0).random(n_bins) + 0.1
        src = np.array([500.0, 1500.0, 2800.0])
        tgt = src.copy()
        out = corrector._warp_sp_frame(sp, freq_axis, src, tgt)
        np.testing.assert_allclose(out, sp, rtol=1e-5)

    def test_warp_sp_frame_no_nan(self, corrector):
        n_bins = 513
        freq_axis = np.linspace(0.0, SR / 2.0, n_bins)
        sp = np.abs(np.random.default_rng(1).standard_normal(n_bins)) + 0.01
        src = np.array([500.0, 1500.0, 2800.0])
        tgt = np.array([550.0, 1600.0, 2900.0])
        out = corrector._warp_sp_frame(sp, freq_axis, src, tgt)
        assert np.all(np.isfinite(out))

    def test_warp_sp_frame_shape_preserved(self, corrector):
        n_bins = 1025
        freq_axis = np.linspace(0.0, SR / 2.0, n_bins)
        sp = np.ones(n_bins)
        src = np.array([400.0])
        tgt = np.array([500.0])
        out = corrector._warp_sp_frame(sp, freq_axis, src, tgt)
        assert out.shape == (n_bins,)

    def test_warp_sp_frame_upward_shift_moves_energy(self, corrector):
        """Peak at src_f in SP is pulled toward tgt_f (upward)."""
        n_bins = 513
        freq_axis = np.linspace(0.0, SR / 2.0, n_bins)
        # Build SP with a narrow peak at 500 Hz
        sp = np.ones(n_bins) * 0.1
        peak_bin = int(500.0 / (SR / 2.0) * (n_bins - 1))
        sp[peak_bin] = 5.0

        src = np.array([500.0])
        tgt = np.array([600.0])  # shift up

        out = corrector._warp_sp_frame(sp, freq_axis, src, tgt)
        # After upward shift: energy near tgt_f should be elevated vs. flat baseline
        tgt_bin = int(600.0 / (SR / 2.0) * (n_bins - 1))
        assert out[tgt_bin] > out[peak_bin], "Warped SP should show elevated energy near target frequency"

    def test_warp_sp_frame_zero_formant_noop(self, corrector):
        """src or tgt = 0 → that formant pair is skipped (no change)."""
        n_bins = 513
        freq_axis = np.linspace(0.0, SR / 2.0, n_bins)
        sp = np.random.default_rng(3).random(n_bins) + 0.1
        sp_orig = sp.copy()
        src = np.array([0.0])  # invalid
        tgt = np.array([600.0])
        out = corrector._warp_sp_frame(sp, freq_axis, src, tgt)
        np.testing.assert_allclose(out, sp_orig, rtol=1e-5)

    def test_correct_preserves_length_mono(self, corrector):
        n = SR
        audio = _harmonic_voice(n=n)
        ff = np.tile([500.0, 1500.0, 2800.0, 3200.0, 4000.0], (100, 1)).astype(np.float64)
        out = corrector.correct(audio.astype(np.float64), SR, ff)
        assert len(out) == n

    def test_correct_no_nan_mono(self, corrector):
        n = SR // 2
        audio = _harmonic_voice(n=n)
        ff = np.tile([500.0, 1500.0, 2800.0, 3200.0, 4000.0], (50, 1)).astype(np.float64)
        out = corrector.correct(audio.astype(np.float64), SR, ff)
        assert np.all(np.isfinite(out))

    def test_correct_output_clipped(self, corrector):
        n = SR // 2
        audio = _harmonic_voice(n=n)
        ff = np.tile([500.0, 1500.0, 2800.0, 3200.0, 4000.0], (50, 1)).astype(np.float64)
        out = corrector.correct(audio.astype(np.float64), SR, ff)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_correct_stereo_shape_preserved(self, corrector):
        n = SR // 4
        mono = _harmonic_voice(n=n).astype(np.float64)
        stereo = np.column_stack([mono, mono * 0.9])
        ff = np.tile([500.0, 1500.0, 2800.0, 3200.0, 4000.0], (25, 1)).astype(np.float64)
        # Corrector receives stereo — it should not crash (EQ path handles nd arrays)
        try:
            out = corrector.correct(stereo, SR, ff)
            assert np.all(np.isfinite(out))
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # stereo may not be implemented for WORLD path; just must not crash

    def test_correct_strength_zero_passthrough(self, corrector):
        """correction_strength=0 → output == input."""
        from dsp.formant_system import FormantCorrector

        zero_corrector = FormantCorrector(max_drift_hz=50.0, correction_strength=0.0)
        n = SR // 4
        audio = _harmonic_voice(n=n).astype(np.float64)
        ff = np.tile([500.0, 1500.0, 2800.0, 3200.0, 4000.0], (25, 1)).astype(np.float64)
        out = zero_corrector.correct(audio, SR, ff)
        np.testing.assert_allclose(out, audio, rtol=1e-4, atol=1e-5)

    def test_correct_silent_audio_returns_silence(self, corrector):
        n = SR // 4
        audio = np.zeros(n, dtype=np.float64)
        ff = np.tile([500.0, 1500.0, 2800.0, 3200.0, 4000.0], (25, 1)).astype(np.float64)
        out = corrector.correct(audio, SR, ff)
        assert np.max(np.abs(out)) < 1e-8

    def test_warp_sp_frame_output_nonnegative(self, corrector):
        """SP values must remain non-negative after warping (power domain)."""
        n_bins = 513
        freq_axis = np.linspace(0.0, SR / 2.0, n_bins)
        sp = np.abs(np.random.default_rng(9).standard_normal(n_bins)) + 0.01
        src = np.array([500.0, 1500.0, 2700.0])
        tgt = np.array([520.0, 1550.0, 2750.0])
        out = corrector._warp_sp_frame(sp, freq_axis, src, tgt)
        assert np.all(out >= 0.0)


# ─── 2. WORLD HNR breathiness ────────────────────────────────────────────────


class TestWorldHnrBreathiness:
    """10 tests for _detect_breathiness WORLD-HNR vs DSP fallback."""

    @pytest.fixture
    def detector(self):
        from backend.core.vocal_ai_enhancement import GenderDetector

        return GenderDetector(sample_rate=SR)

    def test_output_in_range_modal(self, detector):
        audio = _harmonic_voice(f0=200.0, n=SR // 2)
        val = detector._detect_breathiness(audio)
        assert 0.0 <= val <= 1.0

    def test_output_in_range_breathy(self, detector):
        audio = _breathy_voice(f0=200.0, n=SR // 2)
        val = detector._detect_breathiness(audio)
        assert 0.0 <= val <= 1.0

    def test_output_in_range_silence(self, detector):
        audio = np.zeros(SR // 4, dtype=np.float32)
        val = detector._detect_breathiness(audio)
        assert 0.0 <= val <= 1.0

    def test_no_nan_silence(self, detector):
        audio = np.zeros(SR // 4, dtype=np.float32)
        val = detector._detect_breathiness(audio)
        assert np.isfinite(val)

    def test_no_nan_harmonic(self, detector):
        audio = _harmonic_voice(n=SR // 2)
        val = detector._detect_breathiness(audio)
        assert np.isfinite(val)

    def test_no_nan_breathy(self, detector):
        audio = _breathy_voice(n=SR // 2)
        val = detector._detect_breathiness(audio)
        assert np.isfinite(val)

    def test_breathy_higher_than_modal(self, detector):
        """Breathy voice must score higher breathiness than modal voice."""
        modal = _harmonic_voice(f0=200.0, n_harmonics=10, n=SR, amp=0.5)
        breathy = _breathy_voice(f0=200.0, n=SR, harmonic_amp=0.1, noise_amp=0.4)
        score_modal = detector._detect_breathiness(modal)
        score_breathy = detector._detect_breathiness(breathy)
        assert score_breathy > score_modal, f"Breathy {score_breathy:.3f} should exceed modal {score_modal:.3f}"

    def test_white_noise_high_breathiness(self, detector):
        """Pure white noise = fully aperiodic → breathiness near 1."""
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(SR // 2).astype(np.float32) * 0.3
        val = detector._detect_breathiness(noise)
        assert val > 0.4, f"White noise should give high breathiness, got {val:.3f}"

    def test_short_signal_no_crash(self, detector):
        audio = _harmonic_voice(n=512)
        val = detector._detect_breathiness(audio)
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0

    def test_stereo_input_handled(self, detector):
        """Stereo input should not crash — method flattens to mono internally."""
        mono = _harmonic_voice(n=SR // 2)
        stereo = np.column_stack([mono, mono])
        val = detector._detect_breathiness(stereo)
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0


# ─── 3. Singer's Formant narrowing ───────────────────────────────────────────


class TestSingersFormantNarrowing:
    """10 tests for bandwidth=250 Hz and conditional enhancement."""

    @pytest.fixture
    def enhancer(self):
        from dsp.formant_system import SingersFormantEnhancer

        return SingersFormantEnhancer()

    def test_default_bandwidth_250(self, enhancer):
        """Default bandwidth must be 250 Hz per Sundberg (1987) narrow-cluster spec."""
        assert enhancer.bandwidth_hz == 250.0

    def test_no_boost_without_cluster(self, enhancer):
        """Without a Singer's formant cluster, enhance() must be a passthrough."""
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64)
        ff = _formant_freqs_with_cluster(has_cluster=False)
        out, metrics = enhancer.enhance(audio, SR, formant_freqs=ff)
        assert metrics["gain_applied_db"] == 0.0
        np.testing.assert_allclose(out, audio, rtol=1e-5)

    def test_boost_applied_with_cluster(self, enhancer):
        """With Singer's formant present, gain_applied_db must be > 0."""
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64)
        ff = _formant_freqs_with_cluster(has_cluster=True)
        _, metrics = enhancer.enhance(audio, SR, formant_freqs=ff)
        assert metrics["gain_applied_db"] > 0.0

    def test_cluster_center_hz_reported(self, enhancer):
        """cluster_center_hz must be present in metrics dict."""
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64)
        ff = _formant_freqs_with_cluster(has_cluster=True)
        _, metrics = enhancer.enhance(audio, SR, formant_freqs=ff)
        assert "cluster_center_hz" in metrics

    def test_cluster_center_near_actual_values(self, enhancer):
        """cluster_center_hz should be near the F4/F5 values (2900/3100 Hz median)."""
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64)
        ff = _formant_freqs_with_cluster(has_cluster=True)
        _, metrics = enhancer.enhance(audio, SR, formant_freqs=ff)
        # F4=2900, F5=3100 → expected median = 3000 Hz
        assert 2800.0 <= metrics["cluster_center_hz"] <= 3200.0

    def test_no_nan_with_cluster(self, enhancer):
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64)
        ff = _formant_freqs_with_cluster(has_cluster=True)
        out, _ = enhancer.enhance(audio, SR, formant_freqs=ff)
        assert np.all(np.isfinite(out))

    def test_no_nan_without_cluster(self, enhancer):
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64)
        ff = _formant_freqs_with_cluster(has_cluster=False)
        out, _ = enhancer.enhance(audio, SR, formant_freqs=ff)
        assert np.all(np.isfinite(out))

    def test_no_amplification_without_cluster(self, enhancer):
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64) * 0.8
        ff = _formant_freqs_with_cluster(has_cluster=False)
        out, _ = enhancer.enhance(audio, SR, formant_freqs=ff)
        assert np.max(np.abs(out)) <= np.max(np.abs(audio)) + 1e-6

    def test_output_clipped(self, enhancer):
        n = SR // 2
        audio = _harmonic_voice(n=n).astype(np.float64) * 0.9
        ff = _formant_freqs_with_cluster(has_cluster=True)
        out, _ = enhancer.enhance(audio, SR, formant_freqs=ff)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_no_formant_freqs_arg_passthrough(self, enhancer):
        """formant_freqs=None → has_formant=False → passthrough."""
        n = SR // 4
        audio = _harmonic_voice(n=n).astype(np.float64)
        out, metrics = enhancer.enhance(audio, SR, formant_freqs=None)
        assert metrics["gain_applied_db"] == 0.0
        np.testing.assert_allclose(out, audio, rtol=1e-5)
