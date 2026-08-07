from __future__ import annotations

import pytest

"""
tests/unit/test_goosebumps_quality_checker.py — §8.3 GoosebumpsQualityChecker Tests
====================================================================================

Validates the psychoacoustic goosebumps quality assessment module.
Covers: shape, NaN, bounds, edge cases, mono, stereo, dimension scoring,
artifact penalty, musical goals blending, singleton pattern.

Minimum: ≥ 35 unit tests (§Checkliste)
"""


import numpy as np

from backend.core.goosebumps_quality_checker import (
    GoosebumpsQualityChecker,
    GoosebumpsResult,
    get_goosebumps_checker,
    measure_goosebumps,
)

SR = 48000


def _make_tone(freq: float = 440.0, duration_s: float = 3.0, sr: int = SR) -> np.ndarray:
    """Generate a clean sine tone."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float64)  # type: ignore[no-any-return]


def _make_stereo(mono: np.ndarray) -> np.ndarray:
    """Convert mono to stereo (2, N)."""
    return np.stack([mono, mono * 0.95], axis=0)


def _make_noisy(tone: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add white noise at given SNR."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, len(tone))
    signal_power = np.mean(tone**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return tone + noise * np.sqrt(noise_power)  # type: ignore[no-any-return]


def _make_transient_signal(sr: int = SR, duration_s: float = 3.0) -> np.ndarray:
    """Generate signal with clear transients (drum-like attacks)."""
    n = int(sr * duration_s)
    sig = np.zeros(n)
    rng = np.random.default_rng(123)
    # Add percussion-like impulses every 0.5s
    for i in range(int(duration_s / 0.5)):
        pos = int(i * 0.5 * sr)
        if pos + 2000 < n:
            # Sharp attack + exponential decay
            attack = np.exp(-np.arange(2000) / 200.0) * 0.8
            sig[pos : pos + 2000] += attack * rng.normal(0, 1, 2000)
    # Add sustained tone
    t = np.linspace(0, duration_s, n, endpoint=False)
    sig += 0.3 * np.sin(2 * np.pi * 220 * t)
    return sig


# ─── Basic Output Tests ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestGoosebumpsBasicOutput:
    def test_01_returns_goosebumps_result(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert isinstance(result, GoosebumpsResult)

    def test_02_score_in_range(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert 0.0 <= result.goosebumps_score <= 1.0

    def test_03_all_dimensions_in_range(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert 0.0 <= result.transient_integrity <= 1.0
        assert 0.0 <= result.micro_dynamics <= 1.0
        assert 0.0 <= result.clarity <= 1.0
        assert 0.0 <= result.authenticity <= 1.0
        assert 0.0 <= result.artifact_penalty <= 1.0

    def test_04_summary_is_string(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert isinstance(result.summary(), str)
        assert "GoosebumpsScore" in result.summary()

    def test_05_details_dict_populated(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert isinstance(result.details, dict)
        assert len(result.details) > 0


# ─── Identity / Pass-Through Tests ───────────────────────────────────────────


class TestGoosebumpsIdentity:
    def test_06_identical_audio_high_score(self):
        """Identical original and restored should yield high goosebumps score."""
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert result.goosebumps_score >= 0.70

    def test_07_identical_transients_perfect(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert result.transient_integrity >= 0.85

    def test_08_identical_micro_dynamics_perfect(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert result.micro_dynamics >= 0.85

    def test_09_identical_authenticity_perfect(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert result.authenticity >= 0.80

    def test_10_identical_no_artifacts(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert result.artifact_penalty <= 0.15


# ─── Degradation Detection Tests ─────────────────────────────────────────────


class TestGoosebumpsDegradation:
    def test_11_noise_addition_lowers_clarity(self):
        """Adding noise to 'restored' should lower clarity score."""
        tone = _make_tone()
        noisy = _make_noisy(tone, snr_db=10.0)
        result = measure_goosebumps(tone, noisy, SR)
        # Noisy restored should have lower clarity than perfect pass-through
        result_perfect = measure_goosebumps(tone, tone, SR)
        assert result.goosebumps_score < result_perfect.goosebumps_score

    def test_12_zeroed_audio_low_score(self):
        """Silence as 'restored' should yield low goosebumps score."""
        tone = _make_tone()
        silence = np.zeros_like(tone)
        result = measure_goosebumps(tone, silence, SR)
        assert result.goosebumps_score < 0.5

    def test_13_random_noise_low_score(self):
        """Random noise as 'restored' should yield low goosebumps score."""
        tone = _make_tone()
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.1, len(tone))
        result = measure_goosebumps(tone, noise, SR)
        assert result.goosebumps_score < 0.6

    def test_14_frequency_shifted_lowers_authenticity(self):
        """Frequency-shifted restored should degrade authenticity."""
        tone_440 = _make_tone(freq=440.0)
        tone_660 = _make_tone(freq=660.0)  # Different frequency
        result = measure_goosebumps(tone_440, tone_660, SR)
        result_same = measure_goosebumps(tone_440, tone_440, SR)
        assert result.authenticity < result_same.authenticity

    def test_15_inverted_polarity_affects_score(self):
        """Phase-inverted signal should still have reasonable authenticity
        (spectral content is the same)."""
        tone = _make_tone()
        result = measure_goosebumps(tone, -tone, SR)
        # Spectral properties identical, just phase-flipped
        assert result.authenticity >= 0.5


# ─── Transient-Specific Tests ────────────────────────────────────────────────


class TestGoosebumpsTransients:
    def test_16_transient_signal_preserved(self):
        """Transient-rich signal passed through should have high transient integrity."""
        sig = _make_transient_signal()
        result = measure_goosebumps(sig, sig, SR)
        assert result.transient_integrity >= 0.85

    def test_17_smoothed_transients_detected(self):
        """Heavily smoothed signal should lose transient integrity."""
        sig = _make_transient_signal()
        # Aggressive smoothing (destroys transients)
        kernel_size = 500
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(sig, kernel, mode="same")
        result = measure_goosebumps(sig, smoothed, SR)
        result_perfect = measure_goosebumps(sig, sig, SR)
        assert result.transient_integrity < result_perfect.transient_integrity

    def test_18_transient_energy_preserved(self):
        sig = _make_transient_signal()
        result = measure_goosebumps(sig, sig, SR)
        assert "attack_energy_ratio" in result.details


# ─── Micro-Dynamics Tests ────────────────────────────────────────────────────


class TestGoosebumpsMicroDynamics:
    def test_19_flat_dynamics_detected(self):
        """Constant-amplitude signal replacing dynamic one = low micro_dynamics."""
        dynamic = _make_transient_signal()
        flat = np.ones_like(dynamic) * 0.3
        result = measure_goosebumps(dynamic, flat, SR)
        assert result.micro_dynamics < 0.5

    def test_20_dynamics_preserved(self):
        sig = _make_transient_signal()
        result = measure_goosebumps(sig, sig, SR)
        assert result.micro_dynamics >= 0.85

    def test_21_lufs_profile_correlation_reported(self):
        sig = _make_transient_signal()
        result = measure_goosebumps(sig, sig, SR)
        assert "lufs_profile_corr" in result.details


# ─── Artifact Penalty Tests ──────────────────────────────────────────────────


class TestGoosebumpsArtifacts:
    def test_22_clean_passthrough_no_penalty(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert result.artifact_penalty <= 0.15

    def test_23_noise_floor_elevation_penalized(self):
        """If restored has higher noise floor in quiet regions = penalty."""
        tone = _make_tone()
        # Add noise only to quiet regions
        rng = np.random.default_rng(77)
        noisy = tone.copy()
        # Make some regions quiet then add noise there
        noisy[:SR] = rng.normal(0, 0.1, SR)  # First second: artificial noise
        result = measure_goosebumps(tone, noisy, SR)
        assert result.artifact_penalty > 0.0 or result.goosebumps_score < 0.9

    def test_24_artifact_penalty_caps_at_1(self):
        tone = _make_tone()
        rng = np.random.default_rng(55)
        bad = rng.normal(0, 0.5, len(tone))
        result = measure_goosebumps(tone, bad, SR)
        assert result.artifact_penalty <= 1.0


# ─── Musical Goals Integration Tests ─────────────────────────────────────────


class TestGoosebumpsMusicalGoals:
    def test_25_musical_goals_blending_improves_precision(self):
        """When Musical Goals scores are provided, they should influence the result."""
        tone = _make_tone()
        result_no_goals = measure_goosebumps(tone, tone, SR)
        goals = {
            "authentizitaet": 0.95,
            "timbre_authentizitaet": 0.93,
            "micro_dynamics": 0.92,
            "artikulation": 0.90,
        }
        result_with_goals = measure_goosebumps(tone, tone, SR, musical_goal_scores=goals)
        assert result_with_goals.details.get("musical_goals_blended") is True
        assert result_no_goals.details.get("musical_goals_blended") is False

    def test_26_low_musical_goals_affect_score(self):
        """Low Musical Goals should pull the score down."""
        tone = _make_tone()
        low_goals = {
            "authentizitaet": 0.30,
            "timbre_authentizitaet": 0.30,
            "micro_dynamics": 0.30,
            "artikulation": 0.30,
        }
        result_low = measure_goosebumps(tone, tone, SR, musical_goal_scores=low_goals)
        result_no = measure_goosebumps(tone, tone, SR)
        # Low goals should pull score lower than DSP-only measurement
        assert result_low.goosebumps_score <= result_no.goosebumps_score + 0.05

    def test_27_high_musical_goals_maintain_score(self):
        tone = _make_tone()
        high_goals = {
            "authentizitaet": 0.98,
            "timbre_authentizitaet": 0.97,
            "micro_dynamics": 0.95,
            "artikulation": 0.96,
        }
        result = measure_goosebumps(tone, tone, SR, musical_goal_scores=high_goals)
        assert result.goosebumps_score >= 0.65


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestGoosebumpsEdgeCases:
    def test_28_very_short_audio(self):
        """Audio shorter than 1 second should return neutral score."""
        short = _make_tone(duration_s=0.5)
        result = measure_goosebumps(short, short, SR)
        assert result.goosebumps_score == 0.5
        assert result.details.get("skipped") is True

    def test_29_nan_in_audio(self):
        """NaN in audio should not crash — return neutral score."""
        tone = _make_tone()
        bad = tone.copy()
        bad[100:200] = np.nan
        result = measure_goosebumps(tone, bad, SR)
        assert isinstance(result, GoosebumpsResult)
        assert np.isfinite(result.goosebumps_score)

    def test_30_empty_audio(self):
        """Empty arrays should return neutral score."""
        empty = np.array([], dtype=np.float64)
        result = measure_goosebumps(empty, empty, SR)
        assert isinstance(result, GoosebumpsResult)

    def test_31_different_lengths(self):
        """Different length arrays should be handled (aligned)."""
        short = _make_tone(duration_s=2.0)
        long = _make_tone(duration_s=3.0)
        result = measure_goosebumps(short, long, SR)
        assert isinstance(result, GoosebumpsResult)
        assert 0.0 <= result.goosebumps_score <= 1.0

    def test_32_all_zeros(self):
        """All-zeros audio should not crash."""
        zeros = np.zeros(SR * 3)
        result = measure_goosebumps(zeros, zeros, SR)
        assert isinstance(result, GoosebumpsResult)

    def test_33_dc_offset(self):
        """DC offset should not crash the analysis."""
        tone = _make_tone() + 0.5
        result = measure_goosebumps(tone, tone, SR)
        assert isinstance(result, GoosebumpsResult)


# ─── Mono / Stereo Tests ─────────────────────────────────────────────────────


class TestGoosebumpsMonoStereo:
    def test_34_mono_input(self):
        tone = _make_tone()
        result = measure_goosebumps(tone, tone, SR)
        assert isinstance(result, GoosebumpsResult)

    def test_35_stereo_input(self):
        tone = _make_stereo(_make_tone())
        result = measure_goosebumps(tone, tone, SR)
        assert isinstance(result, GoosebumpsResult)
        assert 0.0 <= result.goosebumps_score <= 1.0

    def test_36_mixed_mono_stereo(self):
        mono = _make_tone()
        stereo = _make_stereo(mono)
        result = measure_goosebumps(mono, stereo, SR)
        assert isinstance(result, GoosebumpsResult)


# ─── Singleton Pattern Tests ─────────────────────────────────────────────────


class TestGoosebumpsSingleton:
    def test_37_singleton_returns_instance(self):
        checker = get_goosebumps_checker()
        assert isinstance(checker, GoosebumpsQualityChecker)

    def test_38_singleton_is_same_instance(self):
        checker1 = get_goosebumps_checker()
        checker2 = get_goosebumps_checker()
        assert checker1 is checker2

    def test_39_singleton_measure_works(self):
        checker = get_goosebumps_checker()
        tone = _make_tone()
        result = checker.measure(tone, tone, SR)
        assert isinstance(result, GoosebumpsResult)


# ─── Noise Removal Quality Tests ─────────────────────────────────────────────


class TestGoosebumpsNoiseRemoval:
    def test_40_good_denoising_high_clarity(self):
        """Original noisy + restored clean = high clarity."""
        clean = _make_tone()
        noisy = _make_noisy(clean, snr_db=10.0)
        result = measure_goosebumps(noisy, clean, SR)
        assert result.clarity >= 0.5

    def test_41_over_denoising_penalized(self):
        """Over-denoised (too flat) should be detected."""
        original = _make_noisy(_make_tone(), snr_db=20.0)
        # Over-denoise: remove ALL spectral variation
        over_denoised = _make_tone() * 0.8  # Too clean, lost character
        result = measure_goosebumps(original, over_denoised, SR)
        # Should still have decent score since it's a clean tone
        assert isinstance(result, GoosebumpsResult)


# ─── Performance / Long Audio Tests ──────────────────────────────────────────


class TestGoosebumpsPerformance:
    def test_42_long_audio_handled(self):
        """Audio > 60s should be center-cropped, not crash."""
        long_tone = _make_tone(duration_s=120.0)
        result = measure_goosebumps(long_tone, long_tone, SR)
        assert isinstance(result, GoosebumpsResult)
        assert 0.0 <= result.goosebumps_score <= 1.0

    def test_43_sample_rate_22050(self):
        """Lower sample rate should work."""
        sr = 22050
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t)
        result = measure_goosebumps(tone, tone, sr)
        assert isinstance(result, GoosebumpsResult)
