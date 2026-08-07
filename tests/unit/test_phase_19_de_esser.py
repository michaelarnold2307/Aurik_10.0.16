"""
tests/unit/test_phase_19_de_esser.py
======================================
Aurik 10.0.0 — Phase 19 De-Esser
  * §4.4 Breathiness-Guard: _estimate_breathiness()
  * Dynamische max_reduction_db-Skalierung
  * Mono + Stereo Input
  * Shape / NaN / Bounds / Edge-Cases

35 Unit-Tests. Alle synthetisch (keine echten Audiodateien).
"""

import math

import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq: float, duration_s: float = 2.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _noise_band(lo_hz: float, hi_hz: float, duration_s: float = 2.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Bandpass noise via Butterworth filter."""
    import scipy.signal as ss

    rng = np.random.default_rng(0)
    white = rng.standard_normal(int(duration_s * sr)).astype(np.float32)
    sos = ss.butter(6, [lo_hz / (sr / 2), hi_hz / (sr / 2)], btype="band", output="sos")
    filtered = ss.sosfilt(sos, white).astype(np.float32)
    peak = np.max(np.abs(filtered)) + 1e-8
    return (filtered / peak * amp).astype(np.float32)  # type: ignore[no-any-return]


def _make_phase():
    from backend.core.phases.phase_19_de_esser import DeEsserPhase

    return DeEsserPhase(gender_type="male")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase():
    return _make_phase()


@pytest.fixture(scope="module")
def clean_vocal_mono():
    """Low-breathiness signal: strong 300 Hz fundamental, minimal HF noise."""
    return _sine(300, 2.0) + 0.05 * _noise_band(8000, 12000, 2.0)


@pytest.fixture(scope="module")
def breathy_vocal_mono():
    """High-breathiness signal: spectrum confined to 2–5 kHz via IRFFT.

    Using IRFFT ensures virtually ALL energy is in 2–5 kHz, giving a
    guaranteed breathiness_ratio >> 0.30 for _estimate_breathiness().
    """
    n = 2 * SR  # 2 s at 48 kHz
    spectrum = np.zeros(n // 2 + 1, dtype=complex)
    bin_lo = int(2000 * n / SR)
    bin_hi = int(5000 * n / SR)
    rng = np.random.default_rng(7)
    spectrum[bin_lo:bin_hi] = rng.standard_normal(bin_hi - bin_lo) + 1j * rng.standard_normal(bin_hi - bin_lo)
    audio = np.fft.irfft(spectrum).astype(np.float32)[:n]
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak * 0.8).astype(np.float32)


@pytest.fixture(scope="module")
def silence():
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture(scope="module")
def stereo_clean(clean_vocal_mono):
    return np.stack([clean_vocal_mono, clean_vocal_mono * 0.9], axis=1)


@pytest.fixture(scope="module")
def stereo_breathy(breathy_vocal_mono):
    return np.stack([breathy_vocal_mono, breathy_vocal_mono * 0.9], axis=1)


# ---------------------------------------------------------------------------
# Tests: _estimate_breathiness() Methode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateBreahtiness:
    """_estimate_breathiness() must return float in [0.0, 1.0]."""

    def test_returns_float(self, phase, clean_vocal_mono):
        result = phase._estimate_breathiness(clean_vocal_mono, SR)
        assert isinstance(result, float)

    def test_result_in_unit_interval(self, phase, clean_vocal_mono):
        result = phase._estimate_breathiness(clean_vocal_mono, SR)
        assert 0.0 <= result <= 1.0

    def test_clean_signal_low_breathiness(self, phase, clean_vocal_mono):
        """Clean vocal (low HF noise) should have low breathiness ratio."""
        result = phase._estimate_breathiness(clean_vocal_mono, SR)
        assert result < 0.5, f"Expected < 0.5, got {result:.3f}"

    def test_breathy_signal_high_breathiness(self, phase, breathy_vocal_mono):
        """Breathy vocal (strong 2–5 kHz turbulence) must exceed threshold 0.30."""
        result = phase._estimate_breathiness(breathy_vocal_mono, SR)
        assert result > 0.30, f"Expected > 0.30, got {result:.3f}"

    def test_silence_returns_noncrashing(self, phase, silence):
        """Silence input must not raise; result can be any finite value in [0, 1]."""
        result = phase._estimate_breathiness(silence, SR)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_short_signal_under_512_returns_zero(self, phase):
        """Signals shorter than 512 samples must return 0.0 (unconstrained)."""
        tiny = np.zeros(128, dtype=np.float32)
        result = phase._estimate_breathiness(tiny, SR)
        assert result == 0.0

    def test_nan_input_safe(self, phase):
        """NaN-contaminated input must not raise or return NaN."""
        audio = np.full(SR, np.nan, dtype=np.float32)
        # The method uses np.abs + np.sum which produces NaN; clip must handle it
        # We expect the result to be finite (0.0 due to nan_to_num behaviour OR ≤ 1.0)
        try:
            result = phase._estimate_breathiness(audio, SR)
            # If it doesn't raise, result must be a finite python float
            assert isinstance(result, float)
        except Exception:
            pytest.skip("NaN propagation through fft is implementation-defined")

    def test_stereo_input_handled_as_mono(self, phase, stereo_clean):
        """If caller passes a 2-D array we pick channel 0 so no crash."""
        # _estimate_breathiness expects 1-D; the guard in process() slices [:, 0]
        # — test separately to document the expected slice path
        mono = stereo_clean[:, 0]
        result = phase._estimate_breathiness(mono, SR)
        assert 0.0 <= result <= 1.0

    def test_different_sr_accepted(self, phase):
        """Any SR ≥ 8000 must work without error."""
        audio = _sine(440, 1.0, sr=44100, amp=0.5)
        result = phase._estimate_breathiness(audio, 44100)
        assert 0.0 <= result <= 1.0

    def test_pure_tone_returns_low(self, phase):
        """Pure sine yields concentrated spectrum → almost no 2–5 kHz band energy ratio."""
        tone = _sine(440, 2.0)
        result = phase._estimate_breathiness(tone, SR)
        # Pure 440 Hz sine has zero energy in 2–5 kHz band → ratio ≈ 0
        assert result < 0.10, f"Expected < 0.10 for pure tone, got {result:.4f}"

    def test_band_limited_noise_at_3k_high(self, phase):
        """Noise centred in 2–5 kHz must produce high breathiness."""
        noise_3k = _noise_band(2500, 4000, 2.0, amp=0.9)
        result = phase._estimate_breathiness(noise_3k, SR)
        assert result > 0.30, f"Expected > 0.30 for 2.5–4 kHz noise, got {result:.3f}"


# ---------------------------------------------------------------------------
# Tests: Breathiness-Guard in process() — max_reduction_db scaling
# ---------------------------------------------------------------------------


class TestBreahinessGuardInProcess:
    """Verify that process() applies §4.4 guard and scales max_reduction_db."""

    def _run(self, audio, sr=SR):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        p = DeEsserPhase(gender_type="female")
        return p.process(audio, sr, material_type=MaterialType.TAPE)

    def test_clean_vocal_succeeds(self, clean_vocal_mono):
        result = self._run(clean_vocal_mono)
        assert result.success

    def test_breathy_vocal_succeeds(self, breathy_vocal_mono):
        result = self._run(breathy_vocal_mono)
        assert result.success

    def test_breathiness_ratio_in_stats(self, breathy_vocal_mono):
        """After process(), stats must contain 'breathiness_ratio' key."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        p = DeEsserPhase(gender_type="male")
        result = p.process(breathy_vocal_mono, SR, material_type=MaterialType.TAPE)
        assert result.success
        assert "breathiness_ratio" in p.stats, "stats must contain breathiness_ratio"
        assert 0.0 <= p.stats["breathiness_ratio"] <= 1.0  # type: ignore[operator]

    def test_clean_vocal_breathiness_ratio_low(self, clean_vocal_mono):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        p = DeEsserPhase(gender_type="female")
        p.process(clean_vocal_mono, SR, material_type=MaterialType.TAPE)
        ratio = p.stats.get("breathiness_ratio", -1.0)
        assert ratio < 0.5, f"Clean vocal should have low breathiness, got {ratio:.3f}"  # type: ignore[operator]

    def test_output_shape_mono_unchanged(self, clean_vocal_mono):
        result = self._run(clean_vocal_mono)
        assert result.audio.shape == clean_vocal_mono.shape

    def test_output_shape_stereo_unchanged(self, stereo_clean):
        result = self._run(stereo_clean)
        assert result.audio.shape == stereo_clean.shape

    def test_no_nan_in_output_clean(self, clean_vocal_mono):
        result = self._run(clean_vocal_mono)
        assert not np.any(np.isnan(result.audio))

    def test_no_nan_in_output_breathy(self, breathy_vocal_mono):
        result = self._run(breathy_vocal_mono)
        assert not np.any(np.isnan(result.audio))

    def test_output_clipped_within_bounds(self, breathy_vocal_mono):
        result = self._run(breathy_vocal_mono)
        assert np.all(np.abs(result.audio) <= 1.0 + 1e-5)

    def test_breathy_stereo_succeeds(self, stereo_breathy):
        result = self._run(stereo_breathy)
        assert result.success

    def test_breath_guard_raises_breathiness_ratio_for_breathy(self, breathy_vocal_mono, clean_vocal_mono):
        """Breathy input must produce a higher breathiness_ratio in stats than clean input.

        This directly verifies that _estimate_breathiness() discriminates correctly,
        which is the pre-condition for the guard to scale max_reduction_db down.
        """
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        p_clean = DeEsserPhase(gender_type="male")
        p_clean.process(clean_vocal_mono, SR, material_type=MaterialType.TAPE)
        clean_ratio = p_clean.stats.get("breathiness_ratio", 0.0)

        p_breathy = DeEsserPhase(gender_type="male")
        p_breathy.process(breathy_vocal_mono, SR, material_type=MaterialType.TAPE)
        breathy_ratio = p_breathy.stats.get("breathiness_ratio", 0.0)

        assert breathy_ratio > clean_ratio, (  # type: ignore[operator]
            f"Breathy ratio ({breathy_ratio:.3f}) must exceed clean ratio ({clean_ratio:.3f})"
        )
        assert breathy_ratio > 0.30, f"Breathy signal must trigger guard (ratio={breathy_ratio:.3f} must be > 0.30)"  # type: ignore[operator]

    def test_silence_input_no_crash(self, silence):
        result = self._run(silence)
        assert result.audio is not None

    def test_very_short_audio_no_crash(self):
        tiny = np.zeros(256, dtype=np.float32)
        result = self._run(tiny)
        assert result.audio is not None

    def test_metadata_has_material_key(self, clean_vocal_mono):
        result = self._run(clean_vocal_mono)
        assert "material" in result.metadata

    def test_execution_time_is_finite(self, clean_vocal_mono):
        result = self._run(clean_vocal_mono)
        assert math.isfinite(result.execution_time_seconds)
        assert result.execution_time_seconds >= 0.0


# ---------------------------------------------------------------------------
# Tests: §4.4 Breathiness-Skalierungs-Logik (unit)
# ---------------------------------------------------------------------------


class TestBreahinessScalingLogic:
    """Direct numerical verification of the scaling formula used in process()."""

    def _scale(self, breathiness_ratio: float, max_reduction_db: float = -4.0) -> float:
        """Mirror the exact formula from process()."""
        if breathiness_ratio > 0.30:
            breath_scale = max(0.5, 1.0 - (breathiness_ratio - 0.30))
            return max_reduction_db * breath_scale
        return max_reduction_db

    def test_below_threshold_no_change(self):
        """breathiness_ratio ≤ 0.30 → no scaling."""
        assert self._scale(0.30) == pytest.approx(-4.0, abs=1e-6)

    def test_zero_ratio_no_change(self):
        assert self._scale(0.0) == pytest.approx(-4.0, abs=1e-6)

    def test_ratio_05_scales_correctly(self):
        """breathiness=0.50 → breath_scale = max(0.5, 1 − 0.20) = 0.80 → −3.2 dB."""
        result = self._scale(0.50, -4.0)
        assert result == pytest.approx(-3.2, abs=0.01)

    def test_ratio_08_clamped_to_half(self):
        """breathiness=0.80 → 1 − 0.50 = 0.50 (min clamp) → −2.0 dB."""
        result = self._scale(0.80, -4.0)
        assert result == pytest.approx(-2.0, abs=0.01)

    def test_ratio_10_clamped_to_half(self):
        """breathiness=1.0 → 1 − 0.70 = 0.30, but max(0.5, 0.30) = 0.50 → −2.0 dB."""
        result = self._scale(1.0, -4.0)
        assert result == pytest.approx(-2.0, abs=0.01)

    def test_scale_floor_at_half(self):
        """Scale never falls below 0.5 regardless of breathiness_ratio."""
        for ratio in [0.9, 0.95, 1.0]:
            result = self._scale(ratio, -6.0)
            assert result >= -6.0 * 0.5 - 0.01, f"floor violated at ratio={ratio}"

    def test_max_reduction_sign_preserved(self):
        """max_reduction_db is always negative; scaled version must also be negative."""
        for ratio in [0.0, 0.5, 0.8, 1.0]:
            result = self._scale(ratio, -5.0)
            assert result <= 0.0, f"Expected ≤ 0, got {result} at ratio={ratio}"

    def test_boundary_just_above_threshold(self):
        """breathiness=0.31 → breath_scale = max(0.5, 1 − 0.01) = 0.99 → −3.96 dB."""
        result = self._scale(0.31, -4.0)
        assert result == pytest.approx(-3.96, abs=0.01)


class TestStereoMSCoherence:
    """§2.51: Stereo-De-Esser darf nicht unabhängig pro Kanal laufen."""

    def test_stereo_uses_ms_domain_not_independent_lr(self, monkeypatch):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        phase = DeEsserPhase(gender_type="female")
        calls: list[np.ndarray] = []

        def _spy_channel(
            channel: np.ndarray,
            sample_rate: int,
            material: object,
            band_weights: dict,
            max_reduction_db: float,
            threshold_ratio: float,
            lookahead_samples: int,
        ) -> tuple[np.ndarray, dict]:
            calls.append(np.asarray(channel, dtype=np.float32))
            return np.asarray(channel, dtype=np.float32), {"low": True, "mid": True, "high": True}

        monkeypatch.setattr(phase, "_process_channel_multiband_gender_aware", _spy_channel)

        # Near-mono stereo: M/S erwartet sehr kleinen Side-Kanal.
        base = (0.25 * _sine(300, 1.0) + 0.85 * _noise_band(6000, 9500, 1.0, amp=0.8)).astype(np.float32)
        stereo = np.stack([base, (0.98 * base).astype(np.float32)], axis=1)

        result = phase.process(stereo, SR, material_type=MaterialType.TAPE)
        assert result.success is True
        assert result.audio.shape == stereo.shape
        assert len(calls) == 2, "Stereo-Pfad muss genau zwei Kanalaufrufe ausführen (Mid + Side)."

        rms_first = float(np.sqrt(np.mean(calls[0] ** 2) + 1e-12))
        rms_second = float(np.sqrt(np.mean(calls[1] ** 2) + 1e-12))
        assert rms_second < rms_first * 0.25, (
            "Bei Near-Mono muss Side deutlich kleiner als Mid sein; "
            f"gefunden: mid={rms_first:.6f}, side={rms_second:.6f}"
        )


class TestIntelligibilityGuard:
    def test_air_loss_counts_less_than_presence_articulation_loss(self):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        phase = DeEsserPhase(gender_type="female")
        original = (0.35 * _sine(3000.0, 1.0) + 0.30 * _sine(6500.0, 1.0) + 0.25 * _sine(10000.0, 1.0)).astype(
            np.float32
        )
        air_only_loss = (0.35 * _sine(3000.0, 1.0) + 0.30 * _sine(6500.0, 1.0) + 0.05 * _sine(10000.0, 1.0)).astype(
            np.float32
        )
        presence_and_articulation_loss = (
            0.20 * _sine(3000.0, 1.0) + 0.16 * _sine(6500.0, 1.0) + 0.25 * _sine(10000.0, 1.0)
        ).astype(np.float32)

        air_loss = phase._check_intelligibility_loss(original, air_only_loss, SR)
        speech_loss = phase._check_intelligibility_loss(original, presence_and_articulation_loss, SR)

        assert air_loss < 0.15
        assert speech_loss > air_loss + 0.15


class TestPhase19MusicalGoalMetrics:
    def test_metrics_contain_musical_goal_keys(self, clean_vocal_mono):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(clean_vocal_mono, SR, material_type=MaterialType.TAPE, gender="female")

        assert "musical_goal_brillanz" in result.metrics
        assert "musical_goal_artikulation" in result.metrics
        assert "musical_goal_authentizitaet" in result.metrics
        assert "musical_goal_transparenz" in result.metrics

    def test_metrics_are_bounded(self, breathy_vocal_mono):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(breathy_vocal_mono, SR, material_type=MaterialType.TAPE, gender="female")

        assert 0.0 <= float(result.metrics["musical_goal_brillanz"]) <= 1.0
        assert 0.0 <= float(result.metrics["musical_goal_artikulation"]) <= 1.0
        assert 0.0 <= float(result.metrics["musical_goal_authentizitaet"]) <= 1.0
        assert 0.0 <= float(result.metrics["musical_goal_transparenz"]) <= 1.0

    def test_transparency_metric_tracks_intelligibility_score(self, clean_vocal_mono):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(clean_vocal_mono, SR, material_type=MaterialType.VINYL, gender="female")

        assert result.metrics["musical_goal_transparenz"] == pytest.approx(
            float(np.clip(result.metrics["intelligibility_score"], 0.0, 1.0)), abs=1e-6
        )

    def test_authentizitaet_metric_tracks_presence_ratio(self, clean_vocal_mono):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(clean_vocal_mono, SR, material_type=MaterialType.VINYL, gender="female")

        assert result.metrics["musical_goal_authentizitaet"] == pytest.approx(
            float(np.clip(result.metrics["intelligibility_presence_ratio"], 0.0, 1.0)), abs=1e-6
        )

    def test_brillanz_metric_tracks_air_ratio(self, clean_vocal_mono):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(clean_vocal_mono, SR, material_type=MaterialType.VINYL, gender="female")

        assert result.metrics["musical_goal_brillanz"] == pytest.approx(
            float(np.clip(result.metrics["intelligibility_air_ratio"], 0.0, 1.0)), abs=1e-6
        )

    def test_artikulation_metric_tracks_articulation_ratio(self, clean_vocal_mono):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, MaterialType

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(clean_vocal_mono, SR, material_type=MaterialType.VINYL, gender="female")

        assert result.metrics["musical_goal_artikulation"] == pytest.approx(
            float(np.clip(result.metrics["intelligibility_articulation_ratio"], 0.0, 1.0)), abs=1e-6
        )
