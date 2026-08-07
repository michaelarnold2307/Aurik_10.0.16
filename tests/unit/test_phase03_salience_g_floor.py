from __future__ import annotations

"""Tests for §A salience-adaptive G_floor in phase_03_denoise.

Scientific basis:
    Moore (2003) "Psychology of Hearing" §9 — masking threshold is loudn-relative.
    Loud frames: lower G_floor (residual noise masked by music energy).
    Quiet frames: higher G_floor (signal fragile; protect from over-suppression).
"""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 48_000


def _make_loud_quiet_audio(sr: int = SR, duration_s: float = 2.0) -> np.ndarray:
    """Alternating loud (0.5 RMS) and quiet (0.01 RMS) sections."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    loud = 0.50 * np.sin(2 * np.pi * 440 * t)
    quiet = 0.01 * np.sin(2 * np.pi * 440 * t)
    audio = np.where(t < duration_s / 2, loud, quiet).astype(np.float32)
    return audio


def _make_uniform_audio(sr: int = SR, duration_s: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# _compute_salience_g_floor tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeSalienceGFloor:
    """Unit tests for the static helper method."""

    @pytest.fixture
    def phase(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    def test_returns_1d_array_of_correct_length(self, phase):
        audio = _make_uniform_audio()
        n_t = 120
        hop = 512
        vec = phase._compute_salience_g_floor(audio, SR, 0.1, n_t, hop)
        assert vec.ndim == 1
        assert vec.shape[0] == n_t

    def test_dtype_float32(self, phase):
        audio = _make_uniform_audio()
        vec = phase._compute_salience_g_floor(audio, SR, 0.1, 80, 512)
        assert vec.dtype == np.float32

    def test_no_nan_or_inf(self, phase):
        audio = _make_uniform_audio()
        vec = phase._compute_salience_g_floor(audio, SR, 0.1, 100, 512)
        assert np.all(np.isfinite(vec)), "g_floor_vec must contain no NaN/Inf"

    def test_values_within_bounds(self, phase):
        """All values must be within [g_lo, g_hi] for the given g_floor_base."""
        g_floor_base = 0.1
        audio = _make_loud_quiet_audio()
        n_t, hop = 150, 512
        vec = phase._compute_salience_g_floor(audio, SR, g_floor_base, n_t, hop)
        g_lo = 0.50 * g_floor_base  # = 0.05
        g_hi = min(3.0 * g_floor_base, 0.40)  # = 0.30
        assert float(np.min(vec)) >= g_lo - 1e-4, f"min {float(np.min(vec)):.4f} < g_lo {g_lo:.4f}"
        assert float(np.max(vec)) <= g_hi + 1e-4, f"max {float(np.max(vec)):.4f} > g_hi {g_hi:.4f}"

    def test_loud_frames_lower_than_quiet_frames(self, phase):
        """Loud section → lower G_floor; quiet section → higher G_floor."""
        audio = _make_loud_quiet_audio(duration_s=4.0)  # 2 s loud, 2 s quiet
        n_t, hop = 300, 512
        vec = phase._compute_salience_g_floor(audio, SR, 0.1, n_t, hop)
        # First half → loud, second half → quiet
        half = n_t // 2
        # Allow a 10-frame grace window around the boundary (500 ms smoothing)
        grace = 10
        mean_loud = float(np.mean(vec[: half - grace]))
        mean_quiet = float(np.mean(vec[half + grace :]))
        assert mean_loud < mean_quiet, (
            f"Loud section (G_floor={mean_loud:.4f}) must be lower than quiet section (G_floor={mean_quiet:.4f})"
        )

    def test_shellac_base_respected(self, phase):
        """shellac g_floor_base = 0.30 → g_hi ≤ 0.40 (capped), g_lo ≥ 0.03."""
        audio = _make_uniform_audio()
        vec = phase._compute_salience_g_floor(audio, SR, 0.30, 80, 512)
        assert float(np.max(vec)) <= 0.40 + 1e-4, "shellac g_floor must not exceed 0.40"
        assert float(np.min(vec)) >= 0.03 - 1e-4

    def test_stereo_audio_handled(self, phase):
        """Channel-first (2, N) audio must not crash."""
        mono = _make_uniform_audio()
        stereo = np.stack([mono, mono], axis=0)  # (2, N)
        vec = phase._compute_salience_g_floor(stereo, SR, 0.1, 100, 512)
        assert vec.shape == (100,)
        assert np.all(np.isfinite(vec))

    def test_short_audio_no_crash(self, phase):
        """Very short audio (< 1 s) must not crash."""
        audio = np.zeros(4800, dtype=np.float32)
        vec = phase._compute_salience_g_floor(audio, SR, 0.1, 20, 512)
        assert vec.shape == (20,)
        assert np.all(np.isfinite(vec))

    def test_silent_audio_returns_high_floor(self, phase):
        """Completely silent audio → all LUFS ~ -80 dB → g_floor near g_hi."""
        audio = np.zeros(int(2 * SR), dtype=np.float32)
        g_floor_base = 0.1
        g_hi = min(3.0 * g_floor_base, 0.40)
        vec = phase._compute_salience_g_floor(audio, SR, g_floor_base, 100, 512)
        # Mean should be close to g_hi (quiet region mapping)
        assert float(np.mean(vec)) >= g_hi * 0.9, "Silent audio → g_floor near g_hi"


# ---------------------------------------------------------------------------
# _compute_omlsa_gain with g_floor_vec tests
# ---------------------------------------------------------------------------


class TestOmlsaGainWithVector:
    """Tests for the vector g_floor_vec path in _compute_omlsa_gain."""

    @pytest.fixture
    def phase(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    @pytest.fixture
    def dummy_stft_inputs(self):
        np.random.seed(42)
        n_freq, n_t = 1025, 60
        magnitude = np.abs(np.random.randn(n_freq, n_t).astype(np.float32)) + 1e-6
        noise_mag = np.full((n_freq, n_t), 0.05, dtype=np.float32)
        params = {"g_floor": 0.1, "strength": 0.8}
        return magnitude, noise_mag, params, n_freq, n_t

    def test_scalar_fallback_unchanged(self, phase, dummy_stft_inputs):
        """Without g_floor_vec, scalar path is used (baseline behaviour)."""
        magnitude, noise_mag, params, n_freq, n_t = dummy_stft_inputs
        G, p = phase._compute_omlsa_gain(magnitude, noise_mag, params, g_floor_vec=None)
        assert G.shape == (n_freq, n_t)
        assert np.all(np.isfinite(G))
        assert np.all(G >= 0.0) and np.all(G <= 1.0)

    def test_vector_path_valid_output(self, phase, dummy_stft_inputs):
        """With g_floor_vec, output shape and range must be valid."""
        magnitude, noise_mag, params, n_freq, n_t = dummy_stft_inputs
        g_vec = np.full(n_t, 0.1, dtype=np.float32)
        G, p = phase._compute_omlsa_gain(magnitude, noise_mag, params, g_floor_vec=g_vec)
        assert G.shape == (n_freq, n_t)
        assert np.all(np.isfinite(G))
        assert np.all(G >= 0.0) and np.all(G <= 1.0 + 1e-6)

    def test_vector_path_no_nan(self, phase, dummy_stft_inputs):
        magnitude, noise_mag, params, n_freq, n_t = dummy_stft_inputs
        g_vec = np.linspace(0.05, 0.30, n_t, dtype=np.float32)
        G, _ = phase._compute_omlsa_gain(magnitude, noise_mag, params, g_floor_vec=g_vec)
        assert not np.any(np.isnan(G)), "NaN in G_omlsa with vector g_floor"
        assert not np.any(np.isinf(G)), "Inf in G_omlsa with vector g_floor"

    def test_low_floor_frames_gain_lower_on_noise(self, phase):
        """Frames with low G_floor get more aggressive attenuation on noise-only input."""
        n_freq, n_t = 513, 60
        # Pure noise: uniform spectrum → OMLSA should produce low gains
        np.random.seed(7)
        magnitude = np.abs(np.random.randn(n_freq, n_t).astype(np.float32)) * 0.01 + 1e-5
        noise_mag = magnitude * 0.95  # SNR ≈ 1 dB → heavy suppression expected
        params = {"g_floor": 0.1, "strength": 1.0}

        g_vec_low = np.full(n_t, 0.05, dtype=np.float32)  # aggressive first half
        g_vec_high = np.full(n_t, 0.30, dtype=np.float32)  # conservative second half

        G_low, _ = phase._compute_omlsa_gain(magnitude, noise_mag, params, g_floor_vec=g_vec_low)
        G_high, _ = phase._compute_omlsa_gain(magnitude, noise_mag, params, g_floor_vec=g_vec_high)

        # Mean gain with lower floor should be <= mean gain with higher floor
        assert float(np.mean(G_low)) <= float(np.mean(G_high)) + 1e-4, (
            "Low G_floor must not produce higher mean gain than high G_floor on noise-only input"
        )

    def test_wrong_length_vec_falls_back_to_scalar(self, phase, dummy_stft_inputs):
        """g_floor_vec with wrong length → scalar fallback, no crash."""
        magnitude, noise_mag, params, n_freq, n_t = dummy_stft_inputs
        g_vec_wrong = np.full(n_t + 5, 0.1, dtype=np.float32)  # wrong length
        G, _ = phase._compute_omlsa_gain(magnitude, noise_mag, params, g_floor_vec=g_vec_wrong)
        assert G.shape == (n_freq, n_t)
        assert np.all(np.isfinite(G))

    def test_p_speech_valid(self, phase, dummy_stft_inputs):
        """p_speech must be in [0, 1] regardless of g_floor_vec."""
        magnitude, noise_mag, params, n_freq, n_t = dummy_stft_inputs
        g_vec = np.linspace(0.05, 0.35, n_t, dtype=np.float32)
        _, p = phase._compute_omlsa_gain(magnitude, noise_mag, params, g_floor_vec=g_vec)
        assert np.all(p >= 0.0) and np.all(p <= 1.0 + 1e-6)
        assert np.all(np.isfinite(p))


# ---------------------------------------------------------------------------
# § C: PGHI Gain-Gradient Phase Correction (Prusa & Holighaus 2017 §3.4)
# ---------------------------------------------------------------------------


class TestGainGradientPhaseCorrection:
    """_apply_gain_gradient_phase_correction() tests.

    Scientific basis:
        Prusa & Holighaus (2017) eq. 3.4: time-varying gain G(k,t) introduces
        an instantaneous-frequency error ∂log(G)/∂t that confuses PGHI's phase
        estimation.  Δφ = -(hop/sr) ∫ ∂logG/∂τ dτ compensates this offset.
    """

    @pytest.fixture
    def phase(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_stft(self, n_bins: int = 513, n_t: int = 80, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mag = np.abs(rng.standard_normal((n_bins, n_t))).astype(np.float32) + 1e-4
        phase = rng.uniform(-np.pi, np.pi, (n_bins, n_t)).astype(np.float32)
        return (mag * np.exp(1j * phase)).astype(np.complex64)  # type: ignore[no-any-return]

    def _make_gain(self, n_bins: int = 513, n_t: int = 80, mode: str = "flat") -> np.ndarray:  # type: ignore[return]
        if mode == "flat":
            return np.full((n_bins, n_t), 0.6, dtype=np.float32)
        if mode == "ramp":
            ramp = np.linspace(0.2, 0.9, n_t, dtype=np.float32)
            return np.tile(ramp, (n_bins, 1))
        if mode == "identity":
            return np.ones((n_bins, n_t), dtype=np.float32)

    # -----------------------------------------------------------------------
    # Shape and dtype invariants
    # -----------------------------------------------------------------------

    def test_output_shape_matches_input(self, phase):
        Zxx = self._make_stft(513, 80)
        G = self._make_gain(513, 80)
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        assert out.shape == Zxx.shape, "output shape must match STFT shape"

    def test_output_dtype_complex64(self, phase):
        Zxx = self._make_stft(513, 80)
        G = self._make_gain(513, 80)
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        assert out.dtype == np.complex64

    def test_no_nan_or_inf(self, phase):
        Zxx = self._make_stft(513, 80)
        G = self._make_gain(513, 80, mode="ramp")
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        assert np.all(np.isfinite(out)), "output must contain no NaN/Inf"

    # -----------------------------------------------------------------------
    # Magnitude invariant: |Zxx_out| == G * |Zxx_in|
    # -----------------------------------------------------------------------

    def test_magnitude_equals_gain_times_input(self, phase):
        """Phase correction must not alter the output magnitude."""
        Zxx = self._make_stft(513, 80)
        G = self._make_gain(513, 80, mode="ramp")
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        expected_mag = (G * np.abs(Zxx)).astype(np.float32)
        np.testing.assert_allclose(
            np.abs(out).astype(np.float32),
            expected_mag,
            rtol=1e-4,
            err_msg="output magnitude must equal G * |Zxx_in|",
        )

    # -----------------------------------------------------------------------
    # Identity cases
    # -----------------------------------------------------------------------

    def test_constant_gain_zero_phase_correction(self, phase):
        """Constant G → ∂logG/∂t = 0 → Δφ = 0 → phase unchanged."""
        Zxx = self._make_stft(513, 60)
        G = self._make_gain(513, 60, mode="flat")  # constant 0.6
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        # Angles must match angles of G * Zxx_in exactly (Δφ = 0)
        expected_phase = np.angle(G * Zxx)
        actual_phase = np.angle(out)
        # Wrap-around safe comparison via cos/sin
        phase_err = np.abs(np.angle(np.exp(1j * (actual_phase - expected_phase))))
        assert float(np.max(phase_err)) < 1e-3, (
            f"Constant G must produce Δφ=0; max phase error = {float(np.max(phase_err)):.6f} rad"
        )

    def test_identity_gain_passthrough(self, phase):
        """G = 1.0 everywhere → output equals input exactly (no gain, no correction)."""
        Zxx = self._make_stft(513, 60)
        G = self._make_gain(513, 60, mode="identity")
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        np.testing.assert_allclose(
            np.abs(out),
            np.abs(Zxx).astype(np.float64),
            rtol=1e-4,
        )
        phase_err = np.abs(np.angle(np.exp(1j * (np.angle(out) - np.angle(Zxx)))))
        assert float(np.max(phase_err)) < 1e-3

    # -----------------------------------------------------------------------
    # Phase correction direction
    # -----------------------------------------------------------------------

    def test_ramp_gain_introduces_phase_offset(self, phase):
        """Rising G ramp must produce a negative cumulative Δφ (as per eq. 3.4)."""
        Zxx = self._make_stft(513, 80, seed=1)
        G = self._make_gain(513, 80, mode="ramp")  # 0.2 → 0.9 (positive dlogG)
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        # For a rising ramp dlogG/dt > 0, so Δφ = -∫dlogG < 0 towards the end.
        # Mean phase correction at last 10 frames must differ from first 10 frames.
        phi_in = np.angle(Zxx)
        phi_out = np.angle(out * np.abs(Zxx) / (np.abs(out) + 1e-10))  # normalise mag
        delta = np.angle(np.exp(1j * (phi_out - phi_in)))
        assert float(np.mean(delta[:, -10:])) != pytest.approx(float(np.mean(delta[:, :10])), abs=1e-4), (
            "Rising G ramp must produce non-zero frame-varying phase correction"
        )

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_single_frame(self, phase):
        """n_t = 1: no time difference possible — output must still be valid."""
        Zxx = self._make_stft(513, 1)
        G = np.full((513, 1), 0.5, dtype=np.float32)
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        assert out.shape == (513, 1)
        assert np.all(np.isfinite(out))

    def test_zero_magnitude_bins_no_nan(self, phase):
        """Zero-magnitude bins in Zxx must not produce NaN."""
        Zxx = self._make_stft(513, 60)
        Zxx[100:110, :] = 0.0  # DC-like silent band
        G = self._make_gain(513, 60, mode="ramp")
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        assert np.all(np.isfinite(out)), "Zero-magnitude input bins must not yield NaN/Inf"

    def test_very_small_gain_no_explosion(self, phase):
        """G near zero must not explode log computation."""
        Zxx = self._make_stft(513, 60)
        G = np.full((513, 60), 1e-7, dtype=np.float32)
        out = phase._apply_gain_gradient_phase_correction(Zxx, G, hop=512, sr=SR)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# § E: Stationarity-adaptive α in IMCRA (Loizou 2013 §7.3)
# ---------------------------------------------------------------------------


class TestImcraAdaptiveAlpha:
    """_estimate_noise_imcra() tests for stationarity-adaptive smoothing.

    Scientific basis:
        Ephraim & Malah (1984): optimal α ∝ 1/|∂²P/∂t²|.
        At transient onsets the noise estimate must update faster (α=0.50)
        to avoid tracking a wrong baseline past the attack.
    """

    @pytest.fixture
    def phase(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_stft_mag(
        self, n_freq: int = 257, n_t: int = 80, seed: int = 0, onset_at: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (magnitude, times).  Optional onset_at injects an energy spike."""
        rng = np.random.default_rng(seed)
        mag = rng.uniform(0.01, 0.05, (n_freq, n_t)).astype(np.float32)
        if onset_at is not None and 0 <= onset_at < n_t:
            mag[:, onset_at] *= 20.0  # sharp transient
        dt = 0.01
        times = np.arange(n_t, dtype=np.float32) * dt
        return mag, times

    # -----------------------------------------------------------------------
    # Shape / dtype / basic invariants
    # -----------------------------------------------------------------------

    def test_output_shape(self, phase):
        mag, times = self._make_stft_mag(257, 80)
        out = phase._estimate_noise_imcra(mag, times)
        assert out.shape == mag.shape

    def test_output_positive(self, phase):
        mag, times = self._make_stft_mag(257, 80)
        out = phase._estimate_noise_imcra(mag, times)
        assert np.all(out > 0), "noise estimate must be strictly positive"

    def test_no_nan_or_inf(self, phase):
        mag, times = self._make_stft_mag(257, 80)
        out = phase._estimate_noise_imcra(mag, times)
        assert np.all(np.isfinite(out))

    def test_single_frame(self, phase):
        """n_frames=1: no smoothing loop iteration — must not crash."""
        mag = np.full((257, 1), 0.02, dtype=np.float32)
        times = np.array([0.0], dtype=np.float32)
        out = phase._estimate_noise_imcra(mag, times)
        assert out.shape == (257, 1)
        assert np.all(np.isfinite(out))

    def test_short_audio(self, phase):
        """n_frames=3 (< M): must not crash, all positive."""
        mag = np.abs(np.random.default_rng(1).standard_normal((257, 3))).astype(np.float32) + 1e-4
        times = np.array([0.0, 0.01, 0.02], dtype=np.float32)
        out = phase._estimate_noise_imcra(mag, times)
        assert np.all(np.isfinite(out)) and np.all(out > 0)

    # -----------------------------------------------------------------------
    # Adaptive-α behaviour
    # -----------------------------------------------------------------------

    def test_auto_onset_detection_runs(self, phase):
        """onset_frames=None should run auto-detection without crash."""
        mag, times = self._make_stft_mag(257, 80, onset_at=40)
        out = phase._estimate_noise_imcra(mag, times, onset_frames=None)
        assert out.shape == mag.shape
        assert np.all(np.isfinite(out))

    def test_explicit_onset_frames_accepted(self, phase):
        """Explicit onset_frames must produce valid output."""
        mag, times = self._make_stft_mag(257, 80)
        out = phase._estimate_noise_imcra(mag, times, onset_frames=np.array([10, 40, 70]))
        assert out.shape == mag.shape
        assert np.all(np.isfinite(out))

    def test_empty_onset_frames(self, phase):
        """Empty onset_frames → pure ALPHA_STAT path, must not crash."""
        mag, times = self._make_stft_mag(257, 80)
        out = phase._estimate_noise_imcra(mag, times, onset_frames=np.array([], dtype=int))
        assert np.all(np.isfinite(out))

    def test_onset_faster_convergence_after_spike(self, phase):
        """After a transient spike, adaptive α must converge faster to new level
        than a hypothetical fixed α=0.85 path.

        We simulate this by comparing the smoothed estimate 5 frames after a spike:
        adaptive (α=0.50 at onset) should show lower lag than fixed (α=0.85).
        """
        n_freq, n_t = 1, 60
        mag = np.full((n_freq, n_t), 0.01, dtype=np.float32)
        mag[:, 20] = 1.0  # sharp spike at t=20

        times = np.arange(n_t, dtype=np.float32) * 0.01

        # Adaptive run (onset at frame 20)
        out_adaptive = phase._estimate_noise_imcra(mag, times, onset_frames=np.array([20]))

        # Manual fixed-alpha reference (α=0.85 at every frame, no adaptation)
        pow_spec = mag**2
        M = max(3, int(1.5 / 0.01))
        sigma2 = np.zeros_like(pow_spec)
        window_buf = np.full((n_freq, M), np.inf)
        for t in range(n_t):
            window_buf[:, t % M] = pow_spec[:, t]
            valid = min(t + 1, M)
            sigma2[:, t] = np.min(window_buf[:, :valid], axis=1)
        sigma2 *= 1.66
        smoothed_fixed = np.zeros_like(sigma2)
        smoothed_fixed[:, 0] = sigma2[:, 0]
        for t in range(1, n_t):
            smoothed_fixed[:, t] = 0.85 * smoothed_fixed[:, t - 1] + 0.15 * sigma2[:, t]
        ref_fixed = np.sqrt(np.maximum(smoothed_fixed, 1e-10))

        # 5 frames after the onset spike (t=25): adaptive should have converged
        # CLOSER to the post-spike noise level (smaller smoothed value after spike).
        t_check = 25
        # Both should track back toward 0.01; adaptive faster → lower value
        adaptive_val = float(out_adaptive[0, t_check])
        fixed_val = float(ref_fixed[0, t_check])
        assert adaptive_val <= fixed_val + 1e-4, (
            f"Adaptive α must converge at least as fast as fixed α after onset: "
            f"adaptive={adaptive_val:.5f}, fixed={fixed_val:.5f}"
        )


# ---------------------------------------------------------------------------
# § B: ERB-rate IMCRA grouping (Glasberg & Moore 1990)
# ---------------------------------------------------------------------------


class TestErbRateImcra:
    """_compute_erb_bands() and ERB-grouped _estimate_noise_imcra() tests.

    Scientific basis:
        Glasberg & Moore (1990): ERB-rate E(f) = 21.4 × log10(4.37f/1000 + 1).
        38 perceptual bands from 100 Hz to Nyquist.  Linear STFT bins above
        ~1 kHz are redundant within one auditory critical band; pooling sigma2
        within ERB bands prevents isolated false minima from over-suppressing
        fricative bins (/s/, /f/, /ʃ/ 4–8 kHz).
    """

    @pytest.fixture
    def phase(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    # -----------------------------------------------------------------------
    # _compute_erb_bands
    # -----------------------------------------------------------------------

    def test_band_idx_shape(self, phase):
        idx = phase._compute_erb_bands(1025, 48_000)
        assert idx.shape == (1025,), "band_idx must have one entry per STFT bin"

    def test_band_idx_dtype(self, phase):
        idx = phase._compute_erb_bands(1025, 48_000)
        assert idx.dtype == np.int32

    def test_band_idx_range(self, phase):
        idx = phase._compute_erb_bands(1025, 48_000)
        assert int(idx.min()) >= 0
        assert int(idx.max()) <= 37, "must stay within 38 ERB bands (0-37)"

    def test_band_idx_monotone(self, phase):
        """Higher frequency bins must map to equal or higher band indices."""
        idx = phase._compute_erb_bands(1025, 48_000)
        assert np.all(np.diff(idx) >= 0), "ERB band indices must be non-decreasing"

    def test_multiple_bins_per_high_freq_band(self, phase):
        """Above ~1 kHz many linear bins share one ERB band — critical for grouping."""
        idx = phase._compute_erb_bands(1025, 48_000)
        counts = np.bincount(idx)
        # At 48kHz, 1025 bins → ~27 Hz/bin. ERB at 8kHz ≈ 1000Hz → ~37 bins/band.
        assert int(counts.max()) > 1, "High-frequency bands must contain >1 linear bin"

    def test_small_n_bins(self, phase):
        """n_bins=3 edge case must not crash."""
        idx = phase._compute_erb_bands(3, 48_000)
        assert idx.shape == (3,)
        assert np.all(np.isfinite(idx))

    def test_band_idx_sr_44100(self, phase):
        """Works for 44.1 kHz as well (non-48k SR)."""
        idx = phase._compute_erb_bands(1025, 44_100)
        assert idx.shape == (1025,)
        assert int(idx.max()) <= 37

    # -----------------------------------------------------------------------
    # ERB grouping in _estimate_noise_imcra (sr kwarg)
    # -----------------------------------------------------------------------

    def test_imcra_with_sr_shape(self, phase):
        """Passing sr= must not change output shape."""
        mag = np.abs(np.random.default_rng(0).standard_normal((513, 80))).astype(np.float32) + 1e-4
        times = np.arange(80, dtype=np.float32) * 0.01
        out = phase._estimate_noise_imcra(mag, times, sr=48_000)
        assert out.shape == mag.shape

    def test_imcra_with_sr_positive(self, phase):
        mag = np.abs(np.random.default_rng(1).standard_normal((513, 80))).astype(np.float32) + 1e-4
        times = np.arange(80, dtype=np.float32) * 0.01
        out = phase._estimate_noise_imcra(mag, times, sr=48_000)
        assert np.all(out > 0)

    def test_imcra_with_sr_no_nan(self, phase):
        mag = np.abs(np.random.default_rng(2).standard_normal((513, 80))).astype(np.float32) + 1e-4
        times = np.arange(80, dtype=np.float32) * 0.01
        out = phase._estimate_noise_imcra(mag, times, sr=48_000)
        assert np.all(np.isfinite(out))

    def test_erb_grouping_smooths_outlier_bins(self, phase):
        """An isolated low-energy bin must not produce a far-below-neighbour estimate.

        Without ERB grouping a single bin with near-zero power drives sigma2 to
        near-zero, causing over-suppression of that bin.  With grouping the
        estimate for that bin is raised to the ERB-band average.
        """
        n_t = 80
        # Flat noise spectrum at 0.05 everywhere except bin 300 (almost silent)
        mag = np.full((513, n_t), 0.05, dtype=np.float32)
        mag[300, :] = 1e-6  # isolated silent bin in presence band (~8 kHz @48k)

        times = np.arange(n_t, dtype=np.float32) * 0.01

        out_with_erb = phase._estimate_noise_imcra(mag, times, sr=48_000)
        # Verify bin 300 has a noise estimate close to its neighbours (within 2×), not near-zero
        neighbour_mean = float(np.mean(out_with_erb[295:300, :]))
        bin300_mean = float(np.mean(out_with_erb[300, :]))
        assert bin300_mean >= neighbour_mean * 0.3, (
            f"ERB grouping must raise outlier-bin estimate: bin300={bin300_mean:.5f}, neighbours={neighbour_mean:.5f}"
        )


# ---------------------------------------------------------------------------
# § D: Musical-Noise-Postfilter via psychoakustische Maskierungsschwelle
# ---------------------------------------------------------------------------


class TestMaskingGate:
    """_apply_masking_gate() tests.

    Scientific basis:
        Gustafsson et al. (2001) + Scalart & Filho (1996):
        G_mn(k,t) = √(min(1, E_out(k,t) / (α × P75(E_out(:,t)))))
        α = 10^(−16/10) ≈ 0.025  (simultaneous masking, Fastl & Zwicker 2007 §4.2).
    """

    @pytest.fixture
    def phase(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    def _make_uniform(
        self, n_freq: int = 513, n_t: int = 60, gain_val: float = 0.6, mag_val: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((n_freq, n_t), gain_val, dtype=np.float32),
            np.full((n_freq, n_t), mag_val, dtype=np.float32),
        )

    # -----------------------------------------------------------------------
    # Shape / dtype / basic invariants
    # -----------------------------------------------------------------------

    def test_output_shape(self, phase):
        G, M = self._make_uniform()
        out = phase._apply_masking_gate(G, M)
        assert out.shape == G.shape

    def test_output_dtype_float32(self, phase):
        G, M = self._make_uniform()
        out = phase._apply_masking_gate(G, M)
        assert out.dtype == np.float32

    def test_no_nan_or_inf(self, phase):
        G, M = self._make_uniform()
        out = phase._apply_masking_gate(G, M)
        assert np.all(np.isfinite(out))

    def test_output_in_range(self, phase):
        G, M = self._make_uniform(gain_val=0.8, mag_val=0.2)
        out = phase._apply_masking_gate(G, M)
        assert float(np.min(out)) >= 0.0
        assert float(np.max(out)) <= 1.0 + 1e-6

    def test_floor_never_below_0_1(self, phase):
        """Gate must keep gain ≥ 0.1 (−20 dB floor) — matches existing G_floor."""
        n_freq, n_t = 513, 60
        G = np.full((n_freq, n_t), 0.9, dtype=np.float32)
        # Single silent bin: output_power ≈ 0, M based on neighbours → gate → 0
        # But floor must keep gate ≥ 0.1
        M = np.full((n_freq, n_t), 0.5, dtype=np.float32)
        M[200, :] = 1e-8  # near-silent bin
        out = phase._apply_masking_gate(G, M)
        assert float(np.min(out)) >= 0.1 - 1e-5, f"Floor violated: min={float(np.min(out)):.5f}"

    # -----------------------------------------------------------------------
    # Uniform spectrum — all bins equal → all gates ≈ 1
    # -----------------------------------------------------------------------

    def test_uniform_spectrum_no_attenuation(self, phase):
        """All bins identical → each bin ≥ P75 floor-level → gate stays at 1.0."""
        G, M = self._make_uniform(gain_val=0.7, mag_val=0.15)
        out = phase._apply_masking_gate(G, M)
        # Uniform: E_out / M_threshold ≥ 1 for most bins → gate = 1 → pass-through
        np.testing.assert_allclose(
            out, G, rtol=1e-3, err_msg="Uniform spectrum: _apply_masking_gate must be near-passthrough"
        )

    # -----------------------------------------------------------------------
    # Isolated chirp attenuation
    # -----------------------------------------------------------------------

    def test_isolated_chirp_attenuated(self, phase):
        """A single isolated high-gain bin far below surrounding energy must be attenuated."""
        n_freq, n_t = 513, 60
        # Loud background at all bins
        G = np.full((n_freq, n_t), 0.8, dtype=np.float32)
        M = np.full((n_freq, n_t), 0.5, dtype=np.float32)
        # Musical noise candidate: bin 100 has very low magnitude → output_power << P75
        M[100, :] = 1e-6  # near-silent bin (magnitude ≈ 0 → output_power ≈ 0)
        out = phase._apply_masking_gate(G, M)
        chirp_out = float(np.mean(out[100, :]))
        normal_out = float(np.mean(out[200, :]))
        assert chirp_out < normal_out * 0.9, (
            f"Isolated chirp bin must be attenuated: chirp={chirp_out:.4f}, normal={normal_out:.4f}"
        )

    def test_loud_signal_preserved(self, phase):
        """Bins with output_power >> masking threshold must be near-unchanged."""
        n_freq, n_t = 513, 60
        G = np.full((n_freq, n_t), 0.8, dtype=np.float32)
        M = np.full((n_freq, n_t), 2.0, dtype=np.float32)  # very loud signal
        out = phase._apply_masking_gate(G, M)
        np.testing.assert_allclose(out, G, rtol=1e-3, err_msg="Loud bins must be preserved (gate ≈ 1.0)")

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_zero_magnitude_no_nan(self, phase):
        G = np.full((513, 60), 0.5, dtype=np.float32)
        M = np.zeros((513, 60), dtype=np.float32)  # all silent
        out = phase._apply_masking_gate(G, M)
        assert np.all(np.isfinite(out))
        assert float(np.min(out)) >= 0.1 - 1e-5

    def test_single_bin(self, phase):
        G = np.array([[0.5]], dtype=np.float32)
        M = np.array([[0.3]], dtype=np.float32)
        out = phase._apply_masking_gate(G, M)
        assert out.shape == (1, 1)
        assert np.all(np.isfinite(out))
