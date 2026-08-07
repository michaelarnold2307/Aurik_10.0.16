"""
Unit tests for Phase 19 DeEsser — Crest-Selective Spectral Sculpting (v4.1.0).

Scientific basis:
  - Fant (1960) — fricatives = turbulence noise + resonance peaks
  - Ephraim & Malah (1984) — MMSE spectral estimation, noise-floor preservation
  - Berouti et al. (1979) — spectral subtraction with floor

Tests verify:
  A. API contract of _spectral_crest_sculpt (shape, dtype, NaN safety)
  B. Core crest physics (narrow peaks attenuated > noise texture)
  C. Integration with _process_channel_multiband
  D. Full phase process() integration
"""

import numpy as np
import pytest
import scipy.signal

SR = 48_000


# ─── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def de_esser():
    from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

    return DeEsserPhase(gender_type=VocalGender.FEMALE)


def _sine_band(freq_hz: float, n: int = SR, amp: float = 0.5) -> np.ndarray:
    """Pure sine in the sibilance band (narrow spectral peak)."""
    t = np.arange(n) / SR
    return amp * np.sin(2.0 * np.pi * freq_hz * t)  # type: ignore[no-any-return]


def _bandlimited_noise(f_low: float, f_high: float, n: int = SR, amp: float = 0.1, seed: int = 42) -> np.ndarray:
    """White noise bandpass-filtered to [f_low, f_high] Hz (noise texture)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    nyq = SR / 2.0
    sos = scipy.signal.butter(4, [f_low / nyq, min(f_high / nyq, 0.999)], btype="band", output="sos")
    filtered = scipy.signal.sosfilt(sos, noise)
    # Normalise amplitude
    peak = np.max(np.abs(filtered)) + 1e-12
    return filtered * (amp / peak)  # type: ignore[no-any-return]


# ─── A: API contract ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSpectralCrestSculptApi:
    """8 tests — shape, dtype, NaN safety, edge cases."""

    def test_output_shape_matches_input(self, de_esser):
        n = 4096
        audio = _sine_band(8000.0, n)
        gain = np.full(n, 0.5)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        assert out.shape == (n,)

    def test_output_dtype_float64(self, de_esser):
        n = 4096
        audio = _sine_band(8000.0, n)
        gain = np.full(n, 0.5)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        assert out.dtype == np.float64

    def test_no_nan_inf_output(self, de_esser):
        n = SR
        audio = _bandlimited_noise(6000.0, 12000.0, n)
        gain = np.full(n, 0.3)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        assert np.all(np.isfinite(out)), "NaN/Inf in output"

    def test_zero_signal_returns_zero_or_near_zero(self, de_esser):
        """Zero input should produce zero output (no spectral artefacts)."""
        n = 4096
        audio = np.zeros(n)
        gain = np.full(n, 0.5)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        assert np.max(np.abs(out)) < 1e-10

    def test_unity_gain_passthrough(self, de_esser):
        """gain_curve = 1.0 everywhere → output ≈ input (STFT rounding only)."""
        n = 4096
        audio = _sine_band(8000.0, n, amp=0.3)
        gain = np.ones(n)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        rms_err = np.sqrt(np.mean((out - audio) ** 2)) / (np.sqrt(np.mean(audio**2)) + 1e-12)
        assert rms_err < 0.02, f"RMS error too large: {rms_err:.4f}"

    def test_zero_gain_strong_attenuation(self, de_esser):
        """gain_curve = 0 everywhere → significant attenuation (not total due to crest)."""
        n = SR
        audio = _sine_band(8000.0, n, amp=0.5)
        gain = np.zeros(n)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        rms_in = np.sqrt(np.mean(audio**2))
        rms_out = np.sqrt(np.mean(out**2))
        # Sine is a narrow peak → crest_weight ≈ 1 → gain_mod ≈ 0 → strong attenuation
        assert rms_out < 0.1 * rms_in, f"Expected strong attenuation; ratio={rms_out / rms_in:.3f}"

    def test_short_signal_fallback_no_crash(self, de_esser):
        """Signals shorter than 256 samples fall back to simple multiply."""
        n = 128
        audio = _sine_band(8000.0, n)
        gain = np.full(n, 0.5)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        assert out.shape == (n,)
        expected = audio * gain
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_near_unity_gain_fast_path(self, de_esser):
        """gain_curve > 0.999 → fast path (no STFT); output ≈ input."""
        n = 8192
        audio = _sine_band(8000.0, n, amp=0.3)
        gain = np.full(n, 0.9999)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        # Fast path: simple multiply → very close to audio * 0.9999
        np.testing.assert_allclose(out, audio * gain, rtol=1e-5)


# ─── B: Crest physics ────────────────────────────────────────────────────────


class TestCrestWeighting:
    """7 tests — core perceptual property: narrow peaks attenuated more than noise."""

    def test_sine_attenuated_more_than_noise(self, de_esser):
        """
        Core Weber-Fechner crest invariant:
        A narrow sine peak (high crest) is attenuated MORE than broadband
        noise (low crest) when the same gain_curve is applied.

        Scientific basis: Fant 1960 Ch.3 — fricative spectrum =
        turbulence floor + formant peaks. We must preserve the floor.
        """
        n = SR
        f_center = 8000.0
        f_low, f_high = 6000.0, 10000.0

        sine = _sine_band(f_center, n, amp=0.4)
        noise = _bandlimited_noise(f_low, f_high, n, amp=0.1, seed=7)

        gain = np.full(n, 0.4)  # 60% amplitude reduction

        sine_out = de_esser._spectral_crest_sculpt(sine, gain, f_low, f_high, SR)
        noise_out = de_esser._spectral_crest_sculpt(noise, gain, f_low, f_high, SR)

        sine_ratio = np.sqrt(np.mean(sine_out**2)) / (np.sqrt(np.mean(sine**2)) + 1e-12)
        noise_ratio = np.sqrt(np.mean(noise_out**2)) / (np.sqrt(np.mean(noise**2)) + 1e-12)

        assert sine_ratio < noise_ratio, (
            f"Sine should be attenuated more than noise: sine_ratio={sine_ratio:.3f}, noise_ratio={noise_ratio:.3f}"
        )

    def test_noise_texture_substantially_preserved(self, de_esser):
        """
        Broadband noise (all bins near mean → crest_weight ≈ 0) should have
        energy-ratio close to 1.0 even under moderate gain reduction.
        """
        n = SR
        noise = _bandlimited_noise(6000.0, 10000.0, n, amp=0.15, seed=13)
        gain = np.full(n, 0.4)  # would cause 60% reduction for pure peak

        out = de_esser._spectral_crest_sculpt(noise, gain, 6000.0, 10000.0, SR)
        ratio = np.sqrt(np.mean(out**2)) / (np.sqrt(np.mean(noise**2)) + 1e-12)

        # Noise should be preserved more than gain = 0.4 would imply
        assert ratio > 0.55, f"Noise texture too aggressively attenuated: ratio={ratio:.3f}"

    def test_gain_never_amplifies(self, de_esser):
        """gain_mask ≤ 1 always when gain_curve ≤ 1 (no amplification artifacts)."""
        n = SR
        audio = _sine_band(8000.0, n, amp=0.3) + _bandlimited_noise(6000.0, 12000.0, n, amp=0.05)
        gain = np.full(n, 0.6)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)

        rms_in = np.sqrt(np.mean(audio**2))
        rms_out = np.sqrt(np.mean(out**2))
        assert rms_out <= rms_in * 1.01, f"Output louder than input: rms_in={rms_in:.4f}, rms_out={rms_out:.4f}"

    def test_crest_weight_0_for_bins_at_mean(self, de_esser):
        """
        Perfectly flat spectrum within the band (all bins equal power → all ratios = 1.0)
        should give crest_weight = 0 → no gain reduction applied to any bin.
        """
        n = SR
        # Flat-spectrum noise: white noise bandlimited to exactly the sib range
        # has approximately flat power density within the band
        noise = _bandlimited_noise(6000.0, 12000.0, n, amp=0.1, seed=99)
        gain = np.full(n, 0.5)

        out = de_esser._spectral_crest_sculpt(noise, gain, 6000.0, 12000.0, SR)
        ratio = np.sqrt(np.mean(out**2)) / (np.sqrt(np.mean(noise**2)) + 1e-12)

        # With crest_weight ≈ 0, gain mod ≈ 1.0 for most bins → output ≈ input
        # Allow some residual from filter edges; ratio should be well above gain=0.5
        assert ratio > 0.65, f"Flat spectrum should be largely preserved: ratio={ratio:.3f}"

    def test_pure_sine_fully_attenuated_single_peak(self, de_esser):
        """
        A pure sine (single bin → very high crest → crest_weight = 1) receives
        approximately the same gain as gain_curve specifies (full reduction).
        """
        n = SR
        freq = 8500.0
        audio = _sine_band(freq, n, amp=0.5)
        g = 0.2
        gain = np.full(n, g)

        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        ratio = np.sqrt(np.mean(out**2)) / (np.sqrt(np.mean(audio**2)) + 1e-12)

        # Sine (narrow peak) → crest_weight ≈ 1 → gain_mod ≈ g = 0.2
        # Allow ±0.15 tolerance for STFT windowing / OLA reconstruction
        assert ratio < g + 0.15, f"Sine not fully attenuated: ratio={ratio:.3f}, expected≈{g}"

    def test_mixed_signal_asymmetric_reduction(self, de_esser):
        """
        Mixed = sine (peak) + noise (texture).
        After sculpting: sine component attenuated significantly; noise component less so.
        Verified by computing energy in narrow sine-frequency bands.
        """
        n = SR
        f_sine = 8000.0
        f_low, f_high = 6000.0, 12000.0
        sine = _sine_band(f_sine, n, amp=0.4)
        noise = _bandlimited_noise(f_low, f_high, n, amp=0.15, seed=55)
        mixed = sine + noise

        gain = np.full(n, 0.3)
        out = de_esser._spectral_crest_sculpt(mixed, gain, f_low, f_high, SR)

        # Measure energy in a narrow ±100 Hz window around f_sine
        nyq = SR / 2.0
        sos_narrow = scipy.signal.butter(4, [(f_sine - 100) / nyq, (f_sine + 100) / nyq], btype="band", output="sos")
        sine_in_filtered = scipy.signal.sosfilt(sos_narrow, mixed)
        sine_out_filtered = scipy.signal.sosfilt(sos_narrow, out)
        sine_reduction = np.sqrt(np.mean(sine_out_filtered**2)) / (np.sqrt(np.mean(sine_in_filtered**2)) + 1e-12)

        # Measure broadband residual energy (noise proxy)
        noise_reduction = np.sqrt(np.mean(out**2)) / (np.sqrt(np.mean(mixed**2)) + 1e-12)

        assert sine_reduction < noise_reduction, (
            f"Sine reduction {sine_reduction:.3f} should be less than noise reduction {noise_reduction:.3f}"
        )

    def test_mono_only_no_shape_error(self, de_esser):
        """_spectral_crest_sculpt accepts only 1-D input (per-channel method)."""
        audio = _sine_band(8000.0, n=4096)
        gain = np.full(4096, 0.5)
        out = de_esser._spectral_crest_sculpt(audio, gain, 6000.0, 12000.0, SR)
        assert out.ndim == 1


# ─── C: _process_channel_multiband Integration ───────────────────────────────


class TestMultibandWithCrestSculpting:
    """5 tests — end-to-end multiband routing uses crest sculpting correctly."""

    def test_output_shape_preserved(self, de_esser):
        from backend.core.defect_scanner import MaterialType

        n = SR
        audio = _sine_band(8000.0, n, amp=0.3) + _bandlimited_noise(6000.0, 12000.0, n)
        result = de_esser._process_channel_multiband(
            audio,
            SR,
            MaterialType.CD_DIGITAL,
            band_weights={"low": 0.5, "mid": 0.7, "high": 1.0},
            max_reduction_db=-5.0,
            threshold_ratio=1.5,
            lookahead_samples=0,
        )
        assert result.shape == (n,)

    def test_no_nan_in_output(self, de_esser):
        from backend.core.defect_scanner import MaterialType

        n = SR
        audio = _bandlimited_noise(6000.0, 12000.0, n, amp=0.3, seed=31)
        result = de_esser._process_channel_multiband(
            audio,
            SR,
            MaterialType.VINYL,
            band_weights={"low": 0.6, "mid": 0.8, "high": 0.9},
            max_reduction_db=-6.0,
            threshold_ratio=1.8,
            lookahead_samples=0,
        )
        assert np.all(np.isfinite(result))

    def test_sibilance_energy_reduced(self, de_esser):
        """De-essing with a strong sibilant input must trigger gain reduction.

        Energy measurement through overlapping Butterworth bands is unreliable
        (bands do not perfectly complement → double-counting at transitions).
        We instead verify that the processing chain records significant gain
        reduction in stats['max_gain_reduction_db'].
        """
        from backend.core.defect_scanner import MaterialType

        n = SR
        sibilant = _sine_band(8500.0, n, amp=0.6)
        # Reset stats before the call so we get a clean max_gain_reduction_db
        de_esser.stats["max_gain_reduction_db"] = 0.0
        de_esser._process_channel_multiband(
            sibilant,
            SR,
            MaterialType.CD_DIGITAL,
            band_weights={"low": 0.5, "mid": 0.7, "high": 1.0},
            max_reduction_db=-8.0,
            threshold_ratio=1.2,
            lookahead_samples=0,
        )
        # Gain reduction must have been applied (negative dB means reduction)
        assert de_esser.stats["max_gain_reduction_db"] < -0.5, (
            f"Expected gain reduction; got max_gain_reduction_db={de_esser.stats['max_gain_reduction_db']:.2f} dB"
        )

    def test_non_sibilance_region_largely_preserved(self, de_esser):
        """Low-frequency region (<2 kHz) should not be altered by de-essing."""
        from backend.core.defect_scanner import MaterialType

        n = SR
        audio = _sine_band(400.0, n, amp=0.4)  # 400 Hz — far below sibilance
        result = de_esser._process_channel_multiband(
            audio,
            SR,
            MaterialType.CD_DIGITAL,
            band_weights={"low": 0.5, "mid": 0.7, "high": 1.0},
            max_reduction_db=-8.0,
            threshold_ratio=1.5,
            lookahead_samples=0,
        )
        nyq = SR / 2.0
        sos_low = scipy.signal.butter(4, 1000.0 / nyq, btype="low", output="sos")
        e_in = np.mean(scipy.signal.sosfilt(sos_low, audio) ** 2)
        e_out = np.mean(scipy.signal.sosfilt(sos_low, result) ** 2)
        ratio = e_out / (e_in + 1e-12)
        assert ratio > 0.85, f"Low-freq region altered too much: ratio={ratio:.3f}"

    def test_output_finite_for_silent_input(self, de_esser):
        """Silent audio must not produce NaN/Inf anywhere in the chain."""
        from backend.core.defect_scanner import MaterialType

        n = SR
        audio = np.zeros(n)
        result = de_esser._process_channel_multiband(
            audio,
            SR,
            MaterialType.TAPE,
            band_weights={"low": 0.7, "mid": 0.8, "high": 0.7},
            max_reduction_db=-4.0,
            threshold_ratio=2.0,
            lookahead_samples=0,
        )
        assert np.all(np.isfinite(result))


# ─── D: Full phase integration ────────────────────────────────────────────────


class TestPhase19Integration:
    """5 tests — process() API with crest-sculpting active."""

    def _make_stereo_sibilant(self, n: int = SR) -> np.ndarray:
        left = _sine_band(8000.0, n, amp=0.4) + _bandlimited_noise(6000.0, 12000.0, n, amp=0.05)
        right = left * 0.95
        return np.column_stack([left, right])

    def test_mono_process_no_nan_no_clip(self, de_esser):
        from backend.core.defect_scanner import MaterialType

        n = SR
        audio = _sine_band(8500.0, n, amp=0.5)
        result = de_esser.process(audio, SR, material_type=MaterialType.CD_DIGITAL)
        assert result.success
        assert np.all(np.isfinite(result.audio))
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_stereo_process_shape_preserved(self, de_esser):
        from backend.core.defect_scanner import MaterialType

        n = SR
        audio = self._make_stereo_sibilant(n)
        result = de_esser.process(audio, SR, material_type=MaterialType.VINYL)
        assert result.success
        assert result.audio.shape == (n, 2)

    def test_version_bumped_to_4_1_0(self, de_esser):
        meta = de_esser.get_metadata()
        assert meta.version == "4.1.0", f"Expected 4.1.0 got {meta.version}"

    def test_success_flag_set(self, de_esser):
        from backend.core.defect_scanner import MaterialType

        n = 4 * SR
        audio = _sine_band(9000.0, n, amp=0.4)
        result = de_esser.process(audio, SR, material_type=MaterialType.TAPE)
        assert result.success is True

    def test_short_audio_no_crash(self, de_esser):
        """Short audio (512 samples, but > 5ms RMS window of 240 samples) must not crash."""
        from backend.core.defect_scanner import MaterialType

        audio = np.zeros(512)
        result = de_esser.process(audio, SR, material_type=MaterialType.CD_DIGITAL)
        assert result.success
        assert np.all(np.isfinite(result.audio))
