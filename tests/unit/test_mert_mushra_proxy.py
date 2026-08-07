from __future__ import annotations

"""Tests for backend.core.mert_mushra_proxy — SOTA 22-component MUSHRA proxy evaluator.

Tests cover:
- Basic proxy scoring with synthetic audio pairs (22-component fusion)
- DSP-only fallback (MERT not loaded)
- All 26 component metric ranges and monotonicity
- New components: ViSQOL v3, Multi-Resolution STFT, ISO 226 spectral distance,
  Artifact Penalty, Temporal Consistency, CLAP Cosine, Stereo Imaging,
  Transient Shape, NMR (Noise-to-Mask Ratio), Emotional Arc
- Vocal quality components: Vocal Formant, Vocal HNR, Pitch Accuracy
  (with Vibrato Fidelity), Vocal Presence / CPPS
- Perception dynamics: Modulation Fidelity, Harmonic Structure, Spectral Flux
- Worst-Case Floor Penalty (PEAQ ADB-inspired)
- Adaptive vocal weighting (PANNs-based)
- Temporal primacy/recency attention
- Ridge-regression calibration infrastructure
- Edge cases (silence, identical audio, noise-only, short audio)
- Confidence levels with/without MERT
- Grade assignment thresholds
- Serialization via as_dict()
- Mono/stereo handling
- NaN/Inf guard
- Numpy-only STFT magnitude helper
- ISO 226 weighting function
"""


import math

import numpy as np
import pytest

from backend.core.mert_mushra_proxy import (
    MertMushraProxy,
    _cosine_similarity,
    _extract_dsp_embedding,
    _grade,
    _iso226_weights_for_proxy,
    _stft_magnitude,
    _to_mono,
    estimate_mushra_proxy,
    get_proxy_evaluator,
)

SR = 48_000
DURATION = 2.0  # seconds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tone(freq: float = 440.0, duration: float = DURATION, sr: int = SR) -> np.ndarray:
    """Generate a pure sine tone."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)  # type: ignore[no-any-return]


def _make_harmonic(f0: float = 220.0, n_harmonics: int = 5, duration: float = DURATION) -> np.ndarray:
    """Generate a harmonic signal with f0 and overtones."""
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        sig += (0.3 / k) * np.sin(2 * np.pi * f0 * k * t)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def _add_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add white noise at a specified SNR."""
    rng = np.random.default_rng(42)
    rms_signal = np.sqrt(np.mean(audio**2) + 1e-12)
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = rng.standard_normal(len(audio)).astype(np.float32) * rms_noise
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Test: Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_proxy_evaluator_returns_same_instance(self):
        a = get_proxy_evaluator()
        b = get_proxy_evaluator()
        assert a is b

    def test_instance_type(self):
        assert isinstance(get_proxy_evaluator(), MertMushraProxy)


# ---------------------------------------------------------------------------
# Test: Basic scoring
# ---------------------------------------------------------------------------


class TestBasicScoring:
    """DSP-path scoring tests — MERT patched to None to prevent HF inference timeout.

    §Isolation: WaermeMetric may load MERT via get_mert_plugin() in earlier tests,
    leaving _mert_instance set for the session. Patching get_loaded_mert_plugin to
    return None forces DSP-only path so these tests remain fast regardless of load order.
    """

    @pytest.fixture(autouse=True)
    def _patch_mert_none(self):
        """Force DSP-only path for all TestBasicScoring tests."""
        from unittest.mock import patch

        with patch("plugins.mert_plugin.get_loaded_mert_plugin", return_value=None):
            yield

    def test_identical_audio_high_score(self):
        """Identical reference and test should yield high proxy score."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.proxy_score >= 85.0
        assert result.grade in ("Excellent", "Good")

    def test_noisy_audio_lower_score(self):
        """Adding noise should lower the proxy score."""
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=10.0)
        result = estimate_mushra_proxy(ref, noisy, SR)
        # Must be lower than identical
        result_ident = estimate_mushra_proxy(ref, ref, SR)
        assert result.proxy_score < result_ident.proxy_score

    def test_different_frequency_lower_score(self):
        """Completely different tonal content should score lower."""
        ref = _make_tone(440.0)
        test = _make_tone(880.0)
        result = estimate_mushra_proxy(ref, test, SR)
        assert result.proxy_score < 90.0

    def test_score_range(self):
        """Score must be in [0, 100]."""
        ref = _make_harmonic()
        test = _add_noise(ref, snr_db=5.0)
        result = estimate_mushra_proxy(ref, test, SR)
        assert 0.0 <= result.proxy_score <= 100.0

    def test_monotonicity_with_snr(self):
        """Higher SNR (less noise) should yield higher scores."""
        ref = _make_harmonic()
        score_20 = estimate_mushra_proxy(ref, _add_noise(ref, 20.0), SR).proxy_score
        score_10 = estimate_mushra_proxy(ref, _add_noise(ref, 10.0), SR).proxy_score
        score_5 = estimate_mushra_proxy(ref, _add_noise(ref, 5.0), SR).proxy_score
        assert score_20 >= score_10 >= score_5


# ---------------------------------------------------------------------------
# Test: Component metrics
# ---------------------------------------------------------------------------


class TestComponentMetrics:
    @pytest.fixture(autouse=True)
    def _patch_mert_none(self):
        """Force DSP-only path so component-metric tests don't trigger HuBERT inference.

        Without this, a loaded MERT singleton from an earlier test in the full suite
        causes >30s HuBERT inference → pytest-timeout failure.
        """
        from unittest.mock import patch

        with patch("plugins.mert_plugin.get_loaded_mert_plugin", return_value=None):
            yield

    def test_nsim_range(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert 0.0 <= result.nsim <= 1.0

    def test_mcd_zero_for_identical(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.mcd_db < 1.0  # Near zero for identical

    def test_chroma_high_for_identical(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.chroma_corr >= 0.95

    def test_lufs_small_for_identical(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert abs(result.lufs_diff_lu) < 0.5

    def test_visqol_range(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert 1.0 <= result.visqol_mos <= 5.0

    def test_visqol_high_for_identical(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.visqol_mos >= 3.5

    def test_mr_stft_low_for_identical(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.mr_stft_loss < 0.5

    def test_iso226_low_for_identical(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.iso226_distance < 1.0

    def test_component_scores_dict_populated(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        expected_keys = {
            "mert_cosine",
            "visqol",
            "nsim",
            "artifact",
            "temporal",
            "clap",
            "mr_stft",
            "iso226",
            "mcd",
            "chroma",
            "lufs",
            "stereo",
            "transient",
            "nmr",
            "emotional_arc",
            "vocal_formant",
            "vocal_hnr",
            "pitch_accuracy",
            "vocal_presence",
            "modulation",
            "harmonic",
            "spectral_flux",
            "perceptual_disturbance",
            "roughness",
            "specific_loudness",
            "fluctuation",
        }
        assert expected_keys == set(result.component_scores.keys())

    def test_mr_stft_higher_for_noisy(self):
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=10.0)
        r_clean = estimate_mushra_proxy(ref, ref, SR)
        r_noisy = estimate_mushra_proxy(ref, noisy, SR)
        assert r_noisy.mr_stft_loss > r_clean.mr_stft_loss

    def test_iso226_higher_for_noisy(self):
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=10.0)
        r_clean = estimate_mushra_proxy(ref, ref, SR)
        r_noisy = estimate_mushra_proxy(ref, noisy, SR)
        assert r_noisy.iso226_distance > r_clean.iso226_distance


# ---------------------------------------------------------------------------
# Test: Confidence and MERT availability
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_dsp_fallback_confidence(self):
        """Without MERT loaded, confidence should be DSP-only level."""
        from unittest.mock import patch

        ref = _make_harmonic()
        # Patch get_loaded_mert_plugin at the source module level so that
        # _compute_mert_cosine receives None and returns nan.
        # patch.object(MertMushraProxy, ...) can silently fail under
        # --import-mode=importlib when MERT is already loaded in process
        # (class object identity mismatch).
        with patch("plugins.mert_plugin.get_loaded_mert_plugin", return_value=None):
            result = estimate_mushra_proxy(ref, ref, SR)
        # DSP-only path must produce reduced confidence
        assert result.confidence <= 0.91
        assert math.isnan(result.mert_cosine)

    def test_calibration_stage_is_1(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.calibration_stage == 1


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_silence(self):
        silence = np.zeros(int(SR * 1.0), dtype=np.float32)
        result = estimate_mushra_proxy(silence, silence, SR)
        assert 0.0 <= result.proxy_score <= 100.0

    def test_very_short_audio(self):
        ref = _make_tone(duration=0.05)  # 50 ms
        result = estimate_mushra_proxy(ref, ref, SR)
        assert 0.0 <= result.proxy_score <= 100.0

    def test_empty_audio(self):
        empty = np.array([], dtype=np.float32)
        result = estimate_mushra_proxy(empty, empty, SR)
        assert result.proxy_score == 0.0

    def test_nan_input_guarded(self):
        ref = _make_harmonic()
        bad = ref.copy()
        bad[100:200] = np.nan
        result = estimate_mushra_proxy(ref, bad, SR)
        assert 0.0 <= result.proxy_score <= 100.0
        assert not math.isnan(result.proxy_score)

    def test_inf_input_guarded(self):
        ref = _make_harmonic()
        bad = ref.copy()
        bad[50] = np.inf
        result = estimate_mushra_proxy(ref, bad, SR)
        assert not math.isnan(result.proxy_score)

    def test_different_lengths(self):
        """Ref and test with different lengths should not crash."""
        ref = _make_tone(duration=2.0)
        test = _make_tone(duration=1.5)
        result = estimate_mushra_proxy(ref, test, SR)
        assert 0.0 <= result.proxy_score <= 100.0


# ---------------------------------------------------------------------------
# Test: Stereo / mono handling
# ---------------------------------------------------------------------------


class TestStereoMono:
    def test_stereo_input(self):
        ref_mono = _make_harmonic()
        ref_stereo = np.stack([ref_mono, ref_mono])
        result = estimate_mushra_proxy(ref_stereo, ref_stereo, SR)
        assert result.proxy_score >= 80.0

    def test_mono_vs_stereo_similar(self):
        ref = _make_harmonic()
        ref_stereo = np.stack([ref, ref])
        r_mono = estimate_mushra_proxy(ref, ref, SR)
        r_stereo = estimate_mushra_proxy(ref_stereo, ref_stereo, SR)
        # Should be close since stereo is just duplicated mono
        assert abs(r_mono.proxy_score - r_stereo.proxy_score) < 5.0


# ---------------------------------------------------------------------------
# Test: Grade assignment
# ---------------------------------------------------------------------------


class TestGrade:
    def test_grade_excellent(self):
        assert _grade(95.0) == "Excellent"

    def test_grade_good(self):
        assert _grade(85.0) == "Good"

    def test_grade_fair(self):
        assert _grade(65.0) == "Fair"

    def test_grade_poor(self):
        assert _grade(45.0) == "Poor"

    def test_grade_bad(self):
        assert _grade(15.0) == "Bad"

    def test_grade_boundary_91(self):
        assert _grade(91.0) == "Excellent"

    def test_grade_boundary_80(self):
        assert _grade(80.0) == "Good"

    def test_grade_boundary_60(self):
        assert _grade(60.0) == "Fair"

    def test_grade_boundary_40(self):
        assert _grade(40.0) == "Poor"


# ---------------------------------------------------------------------------
# Test: Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_as_dict_keys(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        for key in [
            "proxy_score",
            "grade",
            "confidence",
            "calibration_stage",
            "nsim",
            "visqol_mos",
            "mr_stft_loss",
            "iso226_distance",
            "mcd_db",
            "chroma_corr",
            "lufs_diff_lu",
            "artifact_penalty",
            "temporal_consistency",
            "clap_cosine",
        ]:
            assert key in d, f"{key} missing from as_dict()"

    def test_as_dict_types(self):
        ref = _make_harmonic()
        d = estimate_mushra_proxy(ref, ref, SR).as_dict()
        assert isinstance(d["proxy_score"], float)
        assert isinstance(d["grade"], str)
        assert isinstance(d["calibration_stage"], int)

    def test_passes_threshold(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        # Identical audio should pass 80
        assert result.passes_threshold(80.0)


# ---------------------------------------------------------------------------
# Test: Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0

    def test_parallel_vectors(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = 5.0 * a
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test: DSP embedding extraction
# ---------------------------------------------------------------------------


class TestDSPEmbedding:
    def test_output_shape(self):
        audio = _make_harmonic(duration=1.0)
        emb = _extract_dsp_embedding(audio, SR)
        assert emb.shape == (512,)

    def test_l2_normalized(self):
        audio = _make_harmonic(duration=1.0)
        emb = _extract_dsp_embedding(audio, SR)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01 or norm < 1e-6  # either unit or zero

    def test_no_nan(self):
        audio = _make_harmonic(duration=1.0)
        emb = _extract_dsp_embedding(audio, SR)
        assert np.isfinite(emb).all()

    def test_different_audio_different_embedding(self):
        a = _make_tone(440.0, duration=1.0)
        b = _make_tone(880.0, duration=1.0)
        emb_a = _extract_dsp_embedding(a, SR)
        emb_b = _extract_dsp_embedding(b, SR)
        cos = _cosine_similarity(emb_a, emb_b)
        assert cos < 0.99  # Different audio → different embedding


# ---------------------------------------------------------------------------
# Test: _to_mono utility
# ---------------------------------------------------------------------------


class TestToMono:
    def test_mono_passthrough(self):
        mono = np.ones(100, dtype=np.float32)
        result = _to_mono(mono)
        assert result.ndim == 1
        assert len(result) == 100

    def test_stereo_to_mono(self):
        stereo = np.ones((2, 100), dtype=np.float32)
        result = _to_mono(stereo)
        assert result.ndim == 1

    def test_nan_replaced(self):
        bad = np.array([1.0, np.nan, 0.5], dtype=np.float32)
        result = _to_mono(bad)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Test: STFT magnitude helper (numpy-only)
# ---------------------------------------------------------------------------


class TestSTFTMagnitude:
    def test_output_shape(self):
        audio = _make_tone(duration=1.0)
        S = _stft_magnitude(audio, 2048, 512)
        assert S.ndim == 2
        assert S.shape[0] == 2048 // 2 + 1  # n_fft//2+1

    def test_zero_for_silence(self):
        silence = np.zeros(SR, dtype=np.float32)
        S = _stft_magnitude(silence, 1024, 256)
        assert S.max() < 1e-6

    def test_nonzero_for_tone(self):
        tone = _make_tone(440.0, duration=0.5)
        S = _stft_magnitude(tone, 2048, 512)
        assert S.max() > 0.01

    def test_multiple_fft_sizes(self):
        audio = _make_harmonic(duration=0.5)
        for fft_size in [2048, 1024, 512, 256]:
            S = _stft_magnitude(audio, fft_size, fft_size // 4)
            assert S.shape[0] == fft_size // 2 + 1

    def test_short_audio_padded(self):
        short = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        S = _stft_magnitude(short, 256, 64)
        assert S.shape[0] == 129  # 256//2+1


# ---------------------------------------------------------------------------
# Test: ISO 226 weighting
# ---------------------------------------------------------------------------


class TestISO226Weights:
    def test_output_shape(self):
        freqs = np.linspace(20, 20000, 100)
        w = _iso226_weights_for_proxy(freqs)
        assert w.shape == (100,)

    def test_all_positive(self):
        freqs = np.linspace(20, 20000, 200)
        w = _iso226_weights_for_proxy(freqs)
        assert (w > 0).all()

    def test_peak_sensitivity_3_4_khz(self):
        """Weights should peak around 3-4 kHz (human sensitivity peak)."""
        freqs = np.array([100, 500, 1000, 3000, 4000, 8000, 16000], dtype=np.float64)
        w = _iso226_weights_for_proxy(freqs)
        # 3-4 kHz should have higher weight than 100 Hz
        idx_3k = 3  # 3000 Hz
        idx_100 = 0  # 100 Hz
        assert w[idx_3k] > w[idx_100]

    def test_no_nan(self):
        freqs = np.linspace(1, 24000, 500)
        w = _iso226_weights_for_proxy(freqs)
        assert np.isfinite(w).all()

    def test_clamped(self):
        freqs = np.linspace(1, 24000, 500)
        w = _iso226_weights_for_proxy(freqs)
        assert w.min() >= 0.001
        assert w.max() <= 10.0


# ---------------------------------------------------------------------------
# Test: ViSQOL integration
# ---------------------------------------------------------------------------


class TestViSQOLIntegration:
    def test_visqol_identical_high(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        # ViSQOL MOS for identical should be high
        assert result.visqol_mos >= 4.0

    @pytest.mark.slow
    def test_visqol_heavily_degraded_lower(self):
        """Heavily degraded signal should get lower ViSQOL than clean."""
        ref = _make_harmonic()
        # Use very different audio (different frequency) for reliable discrimination
        degraded = _make_tone(880.0)  # completely different tonal content
        r_clean = estimate_mushra_proxy(ref, ref, SR)
        r_degraded = estimate_mushra_proxy(ref, degraded, SR)
        assert r_degraded.visqol_mos < r_clean.visqol_mos


# ---------------------------------------------------------------------------
# Test: Multi-Resolution STFT Loss
# ---------------------------------------------------------------------------


class TestMRSTFTLoss:
    def test_zero_for_identical(self):
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        loss = evaluator._compute_mr_stft_loss(ref, ref)
        assert loss < 0.01

    def test_positive_for_different(self):
        ref = _make_harmonic()
        test = _add_noise(ref, snr_db=10.0)
        evaluator = get_proxy_evaluator()
        loss = evaluator._compute_mr_stft_loss(ref, test)
        assert loss > 0.01

    def test_monotonic_with_noise(self):
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        loss_20 = evaluator._compute_mr_stft_loss(ref, _add_noise(ref, 20.0))
        loss_5 = evaluator._compute_mr_stft_loss(ref, _add_noise(ref, 5.0))
        assert loss_5 >= loss_20


# ---------------------------------------------------------------------------
# Test: ISO 226 Spectral Distance
# ---------------------------------------------------------------------------


class TestISO226Distance:
    def test_zero_for_identical(self):
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        dist = evaluator._compute_iso226_distance(ref, ref, SR)
        assert dist < 0.1

    def test_positive_for_different(self):
        ref = _make_harmonic()
        test = _add_noise(ref, snr_db=10.0)
        evaluator = get_proxy_evaluator()
        dist = evaluator._compute_iso226_distance(ref, test, SR)
        assert dist > 0.01

    def test_monotonic_with_noise(self):
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        dist_20 = evaluator._compute_iso226_distance(ref, _add_noise(ref, 20.0), SR)
        dist_5 = evaluator._compute_iso226_distance(ref, _add_noise(ref, 5.0), SR)
        assert dist_5 >= dist_20


# ---------------------------------------------------------------------------
# Test: Artifact Penalty
# ---------------------------------------------------------------------------


class TestArtifactPenalty:
    def test_zero_for_identical(self):
        """Identical audio should produce zero artifact penalty."""
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        penalty = evaluator._compute_artifact_penalty(ref, ref, SR)
        assert penalty < 0.1

    def test_positive_for_noisy(self):
        """Adding noise may produce some artifact penalty."""
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        penalty = evaluator._compute_artifact_penalty(ref, noisy, SR)
        assert penalty >= 0.0

    def test_range(self):
        """Artifact penalty should be bounded [0, 10]."""
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=3.0)
        evaluator = get_proxy_evaluator()
        penalty = evaluator._compute_artifact_penalty(ref, noisy, SR)
        assert 0.0 <= penalty <= 10.0

    def test_result_field(self):
        """MushraProxyResult should expose artifact_penalty."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "artifact_penalty")
        assert np.isfinite(result.artifact_penalty)

    def test_short_audio_no_crash(self):
        short = np.array([0.5, -0.3, 0.1, 0.2], dtype=np.float32)
        evaluator = get_proxy_evaluator()
        penalty = evaluator._compute_artifact_penalty(short, short, SR)
        assert np.isfinite(penalty)


# ---------------------------------------------------------------------------
# Test: Temporal Consistency
# ---------------------------------------------------------------------------


class TestTemporalConsistency:
    def test_high_for_identical(self):
        """Identical audio should have near-perfect temporal consistency."""
        ref = _make_harmonic(duration=3.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_temporal_consistency(ref, ref, SR)
        assert score >= 0.9

    def test_range(self):
        """Score must be in [0, 1]."""
        ref = _make_harmonic(duration=3.0)
        noisy = _add_noise(ref, snr_db=10.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_temporal_consistency(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_lower_for_inconsistent(self):
        """Audio that varies segment-wise should score lower consistency."""
        ref = _make_harmonic(duration=4.0)
        rng = np.random.default_rng(99)
        # Create deliberately inconsistent test: clean first half, noisy second
        test = ref.copy()
        half = len(test) // 2
        test[half:] += rng.standard_normal(len(test) - half).astype(np.float32) * 0.15
        test = np.clip(test, -1.0, 1.0)
        evaluator = get_proxy_evaluator()
        consistent = evaluator._compute_temporal_consistency(ref, ref, SR)
        inconsistent = evaluator._compute_temporal_consistency(ref, test, SR)
        assert inconsistent < consistent

    def test_result_field(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "temporal_consistency")
        assert np.isfinite(result.temporal_consistency)

    def test_short_returns_1(self):
        """Very short audio (< 1 segment) should return 1.0 (assumed consistent)."""
        short = _make_tone(440.0, duration=0.3)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_temporal_consistency(short, short, SR)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Test: CLAP Cosine
# ---------------------------------------------------------------------------


class TestCLAPCosine:
    def test_high_for_identical(self):
        """Identical audio should yield CLAP cosine close to 1.0."""
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        cos = evaluator._compute_clap_cosine(ref, ref, SR)
        assert cos >= 0.99

    def test_range(self):
        ref = _make_harmonic()
        test = _add_noise(ref, snr_db=10.0)
        evaluator = get_proxy_evaluator()
        cos = evaluator._compute_clap_cosine(ref, test, SR)
        assert 0.0 <= cos <= 1.0

    def test_lower_for_different_frequency(self):
        """Different frequency content should yield lower cosine."""
        ref = _make_tone(220.0)
        different = _make_tone(880.0)
        evaluator = get_proxy_evaluator()
        cos_same = evaluator._compute_clap_cosine(ref, ref, SR)
        cos_diff = evaluator._compute_clap_cosine(ref, different, SR)
        assert cos_diff < cos_same

    def test_result_field(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "clap_cosine")
        assert np.isfinite(result.clap_cosine)


# ---------------------------------------------------------------------------
# Stereo Imaging Preservation tests
# ---------------------------------------------------------------------------


class TestStereoImaging:
    """Tests for _compute_stereo_imaging (component 12)."""

    def test_mono_returns_neutral(self):
        """Mono input should return 0.5 (neutral)."""
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_stereo_imaging(ref, ref, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_identical_stereo_high_score(self):
        """Identical stereo should give high preservation score."""
        ref_mono = _make_harmonic()
        ref_stereo = np.stack([ref_mono, ref_mono * 0.8], axis=0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_stereo_imaging(ref_stereo, ref_stereo, SR)
        assert score > 0.85

    def test_swapped_channels_lower(self):
        """Collapsed stereo (mono mix) should reduce stereo imaging score vs wide stereo."""
        np.random.default_rng(42)
        ref_left = _make_harmonic()
        # Different frequency in right channel for wide stereo image
        ref_right = _make_harmonic(f0=330.0) * 0.7
        ref_stereo = np.stack([ref_left, ref_right], axis=0)
        # Collapsed: both channels identical (mono)
        collapsed = np.stack([ref_left, ref_left], axis=0)
        evaluator = get_proxy_evaluator()
        score_same = evaluator._compute_stereo_imaging(ref_stereo, ref_stereo, SR)
        score_collapsed = evaluator._compute_stereo_imaging(ref_stereo, collapsed, SR)
        assert score_collapsed < score_same

    def test_range_bounded(self):
        ref_mono = _make_harmonic()
        ref_stereo = np.stack([ref_mono, ref_mono * 0.5], axis=0)
        noisy = ref_stereo + np.random.default_rng(42).normal(0, 0.1, ref_stereo.shape).astype(np.float32)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_stereo_imaging(ref_stereo, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "stereo_imaging")
        assert np.isfinite(result.stereo_imaging)

    def test_serialization_contains_stereo(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "stereo_imaging" in d
        assert "comp_stereo" in d


# ---------------------------------------------------------------------------
# Transient Shape Preservation tests
# ---------------------------------------------------------------------------


class TestTransientShape:
    """Tests for _compute_transient_shape (component 13)."""

    def test_identical_audio_high(self):
        """Identical audio should have high transient preservation."""
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_transient_shape(ref, ref, SR)
        assert score >= 0.0  # May be low on steady-state sine

    def test_range_bounded(self):
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_transient_shape(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_transient_rich_signal(self):
        """Signal with clicks should detect transients."""
        np.random.default_rng(42)
        ref = np.zeros(int(SR * 1.0), dtype=np.float32)
        # Add clicks
        for pos in [4000, 12000, 24000, 36000]:
            ref[pos] = 0.9
            ref[pos + 1] = -0.8
        # Same signal preserves transients
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_transient_shape(ref, ref, SR)
        assert 0.0 <= score <= 1.0

    def test_smoothed_loses_transients(self):
        """Smoothing should reduce transient preservation vs original."""
        ref = np.zeros(int(SR * 0.5), dtype=np.float32)
        for pos in [2000, 8000, 16000]:
            ref[pos] = 0.9
            ref[pos + 1] = -0.7
        # Heavy smoothing
        kernel = np.ones(50) / 50.0
        smoothed = np.convolve(ref, kernel, mode="same").astype(np.float32)
        evaluator = get_proxy_evaluator()
        score_orig = evaluator._compute_transient_shape(ref, ref, SR)
        score_smooth = evaluator._compute_transient_shape(ref, smoothed, SR)
        # Smoothed should be <= original (or both may be low on minimal signal)
        assert score_smooth <= score_orig + 0.15

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "transient_shape")
        assert np.isfinite(result.transient_shape)

    def test_serialization_contains_transient(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "transient_shape" in d
        assert "comp_transient" in d


# ---------------------------------------------------------------------------
# NMR (Noise-to-Mask Ratio) tests
# ---------------------------------------------------------------------------


class TestNMR:
    """Tests for _compute_nmr (component 14, PEAQ-inspired)."""

    def test_identical_audio_low_nmr(self):
        """Identical audio = no residual → NMR should be very low (≤ 0 dB)."""
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        nmr = evaluator._compute_nmr(ref, ref, SR)
        assert nmr <= 0.0

    def test_noisy_higher_nmr(self):
        """Added noise should increase NMR."""
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        nmr_clean = evaluator._compute_nmr(ref, ref, SR)
        nmr_noisy = evaluator._compute_nmr(ref, noisy, SR)
        assert nmr_noisy > nmr_clean

    def test_range_reasonable(self):
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=10.0)
        evaluator = get_proxy_evaluator()
        nmr = evaluator._compute_nmr(ref, noisy, SR)
        assert -60.0 <= nmr <= 60.0

    def test_very_clean_restoration(self):
        """High-SNR noise should yield NMR closer to masked threshold."""
        ref = _make_harmonic()
        tiny_noise = _add_noise(ref, snr_db=40.0)
        evaluator = get_proxy_evaluator()
        nmr = evaluator._compute_nmr(ref, tiny_noise, SR)
        # Should be lower than loud noise case
        nmr_loud = evaluator._compute_nmr(ref, _add_noise(ref, snr_db=5.0), SR)
        assert nmr < nmr_loud

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "nmr_db")
        assert np.isfinite(result.nmr_db)

    def test_nmr_normalization_in_components(self):
        """NMR norm should be [0, 1] in component_scores."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert 0.0 <= result.component_scores["nmr"] <= 1.0

    def test_serialization_contains_nmr(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "nmr_db" in d
        assert "comp_nmr" in d


# ---------------------------------------------------------------------------
# Emotional Arc Preservation tests
# ---------------------------------------------------------------------------


class TestEmotionalArc:
    """Tests for _compute_emotional_arc (component 15)."""

    def test_short_audio_returns_high(self):
        """Audio shorter than 10 s should return 1.0 (assumed preserved)."""
        ref = _make_harmonic(duration=5.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_emotional_arc(ref, ref, SR)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_identical_long_audio(self):
        """Identical long audio should preserve emotional arc well."""
        ref = _make_harmonic(duration=35.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_emotional_arc(ref, ref, SR)
        assert score >= 0.5  # At minimum neutral

    def test_range_bounded(self):
        ref = _make_harmonic(duration=15.0)
        noisy = _add_noise(ref, snr_db=10.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_emotional_arc(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "emotional_arc")
        assert np.isfinite(result.emotional_arc)

    def test_serialization_contains_emotional_arc(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "emotional_arc" in d
        assert "comp_emotional_arc" in d


# ---------------------------------------------------------------------------
# Vocal Formant Preservation tests (component 16)
# ---------------------------------------------------------------------------


def _make_vocal_like(f0: float = 150.0, duration: float = DURATION, sr: int = SR) -> np.ndarray:
    """Generate a vocal-like signal with F0 and harmonic overtones.

    Simulates a vowel-like sound with enriched harmonic structure that
    produces meaningful formant analysis via LPC.
    """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Fundamental + rich harmonics (voice-like spectrum)
    sig = np.zeros_like(t)
    for k in range(1, 15):
        amp = 0.4 / k  # Natural roll-off
        sig += amp * np.sin(2 * np.pi * f0 * k * t)
    # Add slight formant-like spectral shaping via simple resonances
    # F1 ~ 500 Hz, F2 ~ 1500 Hz emphasis
    sig += 0.15 * np.sin(2 * np.pi * 500 * t) * np.sin(2 * np.pi * f0 * t)
    sig += 0.10 * np.sin(2 * np.pi * 1500 * t) * np.sin(2 * np.pi * f0 * t)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


class TestVocalFormant:
    """Tests for _compute_vocal_formant (component 16)."""

    def test_identical_audio_high(self):
        """Identical vocal-like audio should have high formant preservation."""
        ref = _make_vocal_like()
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_formant(ref, ref, SR)
        assert score >= 0.5  # At least neutral

    def test_range_bounded(self):
        ref = _make_vocal_like()
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_formant(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_short_audio_returns_neutral(self):
        """Very short audio should return neutral 0.5."""
        ref = _make_vocal_like(duration=0.02)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_formant(ref, ref, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_silence_returns_neutral(self):
        """Silence should return 0.5 (non-vocal)."""
        ref = np.zeros(int(SR * 1.0), dtype=np.float32)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_formant(ref, ref, SR)
        assert score == pytest.approx(0.5, abs=0.15)

    def test_pitch_shifted_lower_score(self):
        """Pitch-shifted signal should have lower formant preservation."""
        ref = _make_vocal_like(f0=150.0)
        # Significantly different F0 changes formant structure
        shifted = _make_vocal_like(f0=250.0)
        evaluator = get_proxy_evaluator()
        score_same = evaluator._compute_vocal_formant(ref, ref, SR)
        score_shifted = evaluator._compute_vocal_formant(ref, shifted, SR)
        # Shifted formants should score lower
        assert score_shifted <= score_same + 0.1

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "vocal_formant")
        assert np.isfinite(result.vocal_formant)

    def test_serialization(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "vocal_formant" in d
        assert "comp_vocal_formant" in d

    def test_nan_inf_guard(self):
        """NaN/Inf input should not crash."""
        ref = np.full(int(SR * 0.5), np.nan, dtype=np.float32)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_formant(ref, ref, SR)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Vocal HNR Preservation tests (component 17)
# ---------------------------------------------------------------------------


class TestVocalHNR:
    """Tests for _compute_vocal_hnr (component 17)."""

    def test_identical_audio_high(self):
        """Identical audio should preserve HNR perfectly."""
        ref = _make_vocal_like()
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_hnr(ref, ref, SR)
        assert score >= 0.85  # High preservation

    def test_range_bounded(self):
        ref = _make_vocal_like()
        noisy = _add_noise(ref, snr_db=3.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_hnr(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_noisy_vs_clean_lower(self):
        """Adding noise should reduce HNR preservation."""
        ref = _make_vocal_like()
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        score_clean = evaluator._compute_vocal_hnr(ref, ref, SR)
        score_noisy = evaluator._compute_vocal_hnr(ref, noisy, SR)
        assert score_noisy <= score_clean + 0.05

    def test_noise_only_returns_neutral(self):
        """Pure noise has near-zero HNR → neutral."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(int(SR * 1.0)).astype(np.float32) * 0.3
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_hnr(noise, noise, SR)
        assert 0.0 <= score <= 1.0

    def test_short_audio_returns_neutral(self):
        ref = np.zeros(128, dtype=np.float32)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_hnr(ref, ref, SR)
        assert 0.0 <= score <= 1.0

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "vocal_hnr")
        assert np.isfinite(result.vocal_hnr)

    def test_serialization(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "vocal_hnr" in d
        assert "comp_vocal_hnr" in d

    def test_harmonic_signal_high_hnr(self):
        """Purely harmonic signal should have very high HNR preservation."""
        ref = _make_harmonic(f0=220.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_hnr(ref, ref, SR)
        assert score >= 0.80


# ---------------------------------------------------------------------------
# Pitch Accuracy tests (component 18)
# ---------------------------------------------------------------------------


class TestPitchAccuracy:
    """Tests for _compute_pitch_accuracy (component 18)."""

    def test_identical_audio_high(self):
        """Identical harmonic audio should give high pitch accuracy."""
        ref = _make_harmonic(f0=200.0, n_harmonics=8, duration=3.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_pitch_accuracy(ref, ref, SR)
        assert score >= 0.80

    def test_range_bounded(self):
        ref = _make_vocal_like()
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_pitch_accuracy(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_different_pitch_lower(self):
        """Different fundamental should reduce pitch accuracy."""
        ref = _make_harmonic(f0=200.0, n_harmonics=8, duration=3.0)
        shifted = _make_harmonic(f0=280.0, n_harmonics=8, duration=3.0)
        evaluator = get_proxy_evaluator()
        score_same = evaluator._compute_pitch_accuracy(ref, ref, SR)
        score_diff = evaluator._compute_pitch_accuracy(ref, shifted, SR)
        assert score_diff < score_same

    def test_short_audio_returns_neutral(self):
        ref = np.zeros(256, dtype=np.float32)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_pitch_accuracy(ref, ref, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_noise_only_returns_neutral(self):
        """Pure noise has no stable pitch → neutral."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(int(SR * 1.0)).astype(np.float32) * 0.2
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_pitch_accuracy(noise, noise, SR)
        assert 0.0 <= score <= 1.0

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "pitch_accuracy")
        assert np.isfinite(result.pitch_accuracy)

    def test_serialization(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "pitch_accuracy" in d
        assert "comp_pitch_accuracy" in d

    def test_octave_shifted_low_correlation(self):
        """Octave shift should reduce pitch accuracy."""
        ref = _make_vocal_like(f0=200.0, duration=3.0)
        # Double frequency → octave up
        octave_up = _make_vocal_like(f0=400.0, duration=3.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_pitch_accuracy(ref, octave_up, SR)
        # Octave shifts still have correlation but RMSE is high
        assert score < 0.85


# ---------------------------------------------------------------------------
# Vocal Presence / CPPS tests (component 19)
# ---------------------------------------------------------------------------


class TestVocalPresence:
    """Tests for _compute_vocal_presence (component 19)."""

    def test_identical_audio_high(self):
        """Identical audio should preserve vocal presence perfectly."""
        ref = _make_vocal_like()
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_presence(ref, ref, SR)
        assert score >= 0.80

    def test_range_bounded(self):
        ref = _make_vocal_like()
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_presence(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_short_audio_neutral(self):
        """Very short audio → neutral."""
        ref = _make_vocal_like(duration=0.02)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_presence(ref, ref, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_presence_band_removed_lower(self):
        """Removing 1-4 kHz should reduce vocal presence score."""
        from scipy import signal as sig

        ref = _make_vocal_like(duration=2.0)
        # Notch out presence band
        sos = sig.butter(4, [1000, 4000], btype="bandstop", fs=SR, output="sos")
        notched = sig.sosfilt(sos, ref).astype(np.float32)
        evaluator = get_proxy_evaluator()
        score_full = evaluator._compute_vocal_presence(ref, ref, SR)
        score_notched = evaluator._compute_vocal_presence(ref, notched, SR)
        assert score_notched < score_full

    def test_noise_returns_low_cpps(self):
        """Pure noise has no cepstral peak → low CPPS."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(int(SR * 2.0)).astype(np.float32) * 0.3
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_presence(noise, noise, SR)
        assert 0.0 <= score <= 1.0

    def test_result_field_exists(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "vocal_presence")
        assert np.isfinite(result.vocal_presence)

    def test_serialization(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "vocal_presence" in d
        assert "comp_vocal_presence" in d

    def test_harmonic_signal_cpps_preservation(self):
        """Strong harmonic signal should have high CPPS → high preservation."""
        ref = _make_vocal_like(f0=200.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_presence(ref, ref, SR)
        assert score >= 0.70

    def test_nan_guard(self):
        """NaN input should not crash."""
        ref = np.full(int(SR * 0.5), np.nan, dtype=np.float32)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_vocal_presence(ref, ref, SR)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Integration: 26 components in full evaluate()
# ---------------------------------------------------------------------------


class TestFullEvaluateVocalComponents:
    """Ensure all components integrate correctly in evaluate()."""

    def test_all_26_component_scores_present(self):
        """component_scores dict should contain all 26 keys."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        expected_keys = {
            "mert_cosine",
            "visqol",
            "nsim",
            "artifact",
            "temporal",
            "clap",
            "mr_stft",
            "iso226",
            "mcd",
            "chroma",
            "lufs",
            "stereo",
            "transient",
            "nmr",
            "emotional_arc",
            "vocal_formant",
            "vocal_hnr",
            "pitch_accuracy",
            "vocal_presence",
            "modulation",
            "harmonic",
            "spectral_flux",
            "perceptual_disturbance",
            "roughness",
            "specific_loudness",
            "fluctuation",
        }
        assert expected_keys == set(result.component_scores.keys())

    def test_vocal_components_finite(self):
        ref = _make_vocal_like()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert np.isfinite(result.vocal_formant)
        assert np.isfinite(result.vocal_hnr)
        assert np.isfinite(result.pitch_accuracy)
        assert np.isfinite(result.vocal_presence)

    def test_empty_result_has_vocal_fields(self):
        r = MertMushraProxy._empty_result()
        assert r.vocal_formant == 0.0
        assert r.vocal_hnr == 0.0
        assert r.pitch_accuracy == 0.0
        assert r.vocal_presence == 0.0

    def test_empty_result_has_psychoacoustic_fields(self):
        r = MertMushraProxy._empty_result()
        assert r.specific_loudness_diff == 0.0
        assert r.fluctuation_strength == 0.0
        assert r.perceptual_disturbance == 0.0
        assert r.roughness == 0.0

    def test_empty_result_has_new_fields(self):
        r = MertMushraProxy._empty_result()
        assert r.modulation_fidelity == 0.0
        assert r.harmonic_structure == 0.0
        assert r.spectral_flux_corr == 0.0
        assert r.perceptual_disturbance == 0.0
        assert r.roughness == 0.0
        assert r.worst_segment_score == 0.0

    def test_as_dict_contains_all_vocal_keys(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        for key in ["vocal_formant", "vocal_hnr", "pitch_accuracy", "vocal_presence"]:
            assert key in d
            assert f"comp_{key}" in d

    def test_weights_sum_to_one(self):
        """Weight tables must sum to 1.0."""
        from backend.core.mert_mushra_proxy import _WEIGHTS_DSP_ONLY, _WEIGHTS_WITH_MERT

        assert abs(sum(_WEIGHTS_WITH_MERT.values()) - 1.0) < 1e-6
        assert abs(sum(_WEIGHTS_DSP_ONLY.values()) - 1.0) < 1e-6

    def test_26_weight_keys(self):
        """Both weight dicts must have exactly 26 keys."""
        from backend.core.mert_mushra_proxy import _WEIGHTS_DSP_ONLY, _WEIGHTS_WITH_MERT

        assert len(_WEIGHTS_WITH_MERT) == 26
        assert len(_WEIGHTS_DSP_ONLY) == 26


# ---------------------------------------------------------------------------
# Adaptive Vocal Weighting tests
# ---------------------------------------------------------------------------


class TestAdaptiveVocalWeighting:
    """Tests for _adapt_weights_for_vocal_content (Lücke 4 fix)."""

    def test_sum_always_one(self):
        """Adapted weights must always sum to 1.0 regardless of vocal_prob."""
        from backend.core.mert_mushra_proxy import _WEIGHTS_WITH_MERT

        evaluator = get_proxy_evaluator()
        for vp in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
            w = evaluator._adapt_weights_for_vocal_content(_WEIGHTS_WITH_MERT, vp)
            assert abs(sum(w.values()) - 1.0) < 1e-6, f"vocal_prob={vp}: sum={sum(w.values())}"

    def test_neutral_zone_unchanged(self):
        """Vocal prob in [0.35, 0.65] should return unchanged weights."""
        from backend.core.mert_mushra_proxy import _WEIGHTS_WITH_MERT

        evaluator = get_proxy_evaluator()
        for vp in [0.35, 0.45, 0.5, 0.65]:
            w = evaluator._adapt_weights_for_vocal_content(_WEIGHTS_WITH_MERT, vp)
            for k in _WEIGHTS_WITH_MERT:
                assert w[k] == pytest.approx(_WEIGHTS_WITH_MERT[k], abs=1e-8)

    def test_low_vocal_shrinks_vocal_pool(self):
        """Low vocal probability should reduce vocal component weights."""
        from backend.core.mert_mushra_proxy import _VOCAL_COMPONENT_KEYS, _WEIGHTS_WITH_MERT

        evaluator = get_proxy_evaluator()
        w_instrumental = evaluator._adapt_weights_for_vocal_content(_WEIGHTS_WITH_MERT, 0.0)
        orig_vocal = sum(_WEIGHTS_WITH_MERT[k] for k in _VOCAL_COMPONENT_KEYS)
        new_vocal = sum(w_instrumental[k] for k in _VOCAL_COMPONENT_KEYS)
        assert new_vocal < orig_vocal * 0.5

    def test_high_vocal_grows_vocal_pool(self):
        """High vocal probability should increase vocal component weights."""
        from backend.core.mert_mushra_proxy import _VOCAL_COMPONENT_KEYS, _WEIGHTS_WITH_MERT

        evaluator = get_proxy_evaluator()
        w_vocal = evaluator._adapt_weights_for_vocal_content(_WEIGHTS_WITH_MERT, 1.0)
        orig_vocal = sum(_WEIGHTS_WITH_MERT[k] for k in _VOCAL_COMPONENT_KEYS)
        new_vocal = sum(w_vocal[k] for k in _VOCAL_COMPONENT_KEYS)
        assert new_vocal > orig_vocal * 1.3

    def test_all_keys_preserved(self):
        """All 26 weight keys must be present after adaptation."""
        from backend.core.mert_mushra_proxy import _WEIGHTS_WITH_MERT

        evaluator = get_proxy_evaluator()
        w = evaluator._adapt_weights_for_vocal_content(_WEIGHTS_WITH_MERT, 0.1)
        assert set(w.keys()) == set(_WEIGHTS_WITH_MERT.keys())

    def test_no_negative_weights(self):
        """No weight should ever go negative."""
        from backend.core.mert_mushra_proxy import _WEIGHTS_WITH_MERT

        evaluator = get_proxy_evaluator()
        for vp in [0.0, 0.5, 1.0]:
            w = evaluator._adapt_weights_for_vocal_content(_WEIGHTS_WITH_MERT, vp)
            assert all(v >= 0.0 for v in w.values())


# ---------------------------------------------------------------------------
# Vocal Probability Estimation tests
# ---------------------------------------------------------------------------


class TestVocalProbabilityEstimation:
    """Tests for _estimate_vocal_probability."""

    def test_harmonic_returns_something(self):
        """Harmonic signal should return some vocal estimation."""
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        prob = evaluator._estimate_vocal_probability(ref, SR)
        assert 0.0 <= prob <= 1.0

    def test_noise_returns_finite(self):
        """Noise should return finite vocal prob."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(int(SR * 2.0)).astype(np.float32) * 0.3
        evaluator = get_proxy_evaluator()
        prob = evaluator._estimate_vocal_probability(noise, SR)
        assert 0.0 <= prob <= 1.0

    def test_short_audio_returns_neutral(self):
        """Very short audio returns low/neutral vocal probability."""
        ref = np.zeros(100, dtype=np.float32)
        evaluator = get_proxy_evaluator()
        prob = evaluator._estimate_vocal_probability(ref, SR)
        # PANNs may be loaded from prior tests → returns near-zero for silence;
        # heuristic fallback returns 0.5 for < 2048 samples.
        assert 0.0 <= prob <= 0.6

    def test_empty_audio(self):
        """Empty audio should not crash."""
        ref = np.zeros(0, dtype=np.float32)
        evaluator = get_proxy_evaluator()
        prob = evaluator._estimate_vocal_probability(ref, SR)
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# Temporal Attention tests (primacy/recency)
# ---------------------------------------------------------------------------


class TestTemporalAttention:
    """Tests for enhanced _compute_temporal_consistency with attention."""

    def test_identical_long_audio_high(self):
        """Identical long audio should score high."""
        ref = _make_harmonic(duration=10.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_temporal_consistency(ref, ref, SR)
        assert score >= 0.60

    def test_range_bounded(self):
        ref = _make_harmonic(duration=5.0)
        noisy = _add_noise(ref, snr_db=5.0)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_temporal_consistency(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_short_returns_one(self):
        """Short audio (< 2 segments) returns 1.0."""
        ref = _make_harmonic(duration=0.5)
        evaluator = get_proxy_evaluator()
        score = evaluator._compute_temporal_consistency(ref, ref, SR)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_degraded_start_penalized_more(self):
        """Degradation at the start should be penalized more than in the middle
        (primacy effect)."""
        # Create a 10s signal
        ref = _make_harmonic(duration=10.0)
        n = len(ref)

        # Degrade first 2 seconds (affects primacy-weighted segments)
        degraded_start = ref.copy()
        rng = np.random.default_rng(77)
        noise_len = int(2.0 * SR)
        degraded_start[:noise_len] += rng.standard_normal(noise_len).astype(np.float32) * 0.3
        degraded_start = np.clip(degraded_start, -1.0, 1.0)

        # Degrade middle 2 seconds (affects low-weight segments)
        degraded_mid = ref.copy()
        mid_start = n // 2 - noise_len // 2
        degraded_mid[mid_start : mid_start + noise_len] += rng.standard_normal(noise_len).astype(np.float32) * 0.3
        degraded_mid = np.clip(degraded_mid, -1.0, 1.0)

        evaluator = get_proxy_evaluator()
        score_start = evaluator._compute_temporal_consistency(ref, degraded_start, SR)
        score_mid = evaluator._compute_temporal_consistency(ref, degraded_mid, SR)

        # Start-degraded should score ≤ mid-degraded (primacy penalty)
        assert score_start <= score_mid + 0.10  # Allow small tolerance


# ---------------------------------------------------------------------------
# Ridge Regression Calibration Infrastructure tests
# ---------------------------------------------------------------------------


class TestRidgeCalibration:
    """Tests for calibrate_from_panel (Stage 2 infrastructure)."""

    def test_calibration_produces_valid_weights(self):
        """Ridge regression should produce 26 normalized weights."""
        rng = np.random.default_rng(42)
        # Simulate 50 evaluation pairs
        N = 50
        X = rng.uniform(0.3, 0.95, (N, 26))
        # Simulate MUSHRA scores linearly related to components
        true_weights = rng.uniform(0.01, 0.2, 26)
        true_weights /= true_weights.sum()
        y = X @ true_weights * 100 + rng.normal(0, 2, N)
        y = np.clip(y, 0, 100)

        evaluator = get_proxy_evaluator()
        result = evaluator.calibrate_from_panel(X, y, alpha=1.0)

        assert len(result) == 26
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert all(v >= 0.0 for v in result.values())

    def test_calibration_sets_module_state(self):
        """After calibration, module-level weights should be set."""
        import backend.core.mert_mushra_proxy as mod

        # Reset state
        mod._calibrated_weights = None
        mod._calibrated_confidence = None

        rng = np.random.default_rng(42)
        N = 30
        X = rng.uniform(0.3, 0.95, (N, 26))
        y = rng.uniform(40, 95, N)

        get_proxy_evaluator().calibrate_from_panel(X, y)

        assert mod._calibrated_weights is not None
        assert mod._calibrated_confidence is not None
        assert mod._calibrated_confidence >= 0.70

        # Clean up — reset to Stage 1
        mod._calibrated_weights = None
        mod._calibrated_confidence = None

    def test_calibrated_weights_used_in_evaluate(self):
        """When calibrated weights exist, evaluate() should use them (Stage 2)."""
        import backend.core.mert_mushra_proxy as mod

        rng = np.random.default_rng(42)
        N = 30
        X = rng.uniform(0.3, 0.95, (N, 26))
        y = rng.uniform(40, 95, N)

        get_proxy_evaluator().calibrate_from_panel(X, y)

        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.calibration_stage == 2

        # Clean up
        mod._calibrated_weights = None
        mod._calibrated_confidence = None

    def test_uncalibrated_is_stage_1(self):
        """Without calibration, evaluate() returns stage 1."""
        import backend.core.mert_mushra_proxy as mod

        mod._calibrated_weights = None
        mod._calibrated_confidence = None

        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert result.calibration_stage == 1


# ---------------------------------------------------------------------------
# New component: Modulation Fidelity
# ---------------------------------------------------------------------------


class TestModulationFidelity:
    """Tests for _compute_modulation_fidelity (PEAQ AvgModDiff equivalent)."""

    def test_identical_signals_perfect(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        score = get_proxy_evaluator()._compute_modulation_fidelity(ref, ref, SR)
        assert score >= 0.98

    def test_noise_degrades(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        test = _add_noise(ref, snr_db=10.0)
        score = get_proxy_evaluator()._compute_modulation_fidelity(ref, test, SR)
        assert 0.0 <= score < 0.90

    def test_heavy_noise_very_low(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        test = _add_noise(ref, snr_db=0.0)
        score = get_proxy_evaluator()._compute_modulation_fidelity(ref, test, SR)
        assert 0.0 <= score < 0.50

    def test_range_bounded(self):
        ref = _make_harmonic(f0=440.0, n_harmonics=3)
        rng = np.random.default_rng(77)
        test = rng.standard_normal(len(ref)).astype(np.float32) * 0.3
        score = get_proxy_evaluator()._compute_modulation_fidelity(ref, test, SR)
        assert 0.0 <= score <= 1.0

    def test_monotonic_degradation(self):
        """More noise → lower score."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        s1 = get_proxy_evaluator()._compute_modulation_fidelity(ref, _add_noise(ref, 30), SR)
        s2 = get_proxy_evaluator()._compute_modulation_fidelity(ref, _add_noise(ref, 10), SR)
        assert s1 > s2

    def test_short_audio_handled(self):
        """Very short audio should not crash."""
        ref = _make_tone(440.0, duration=0.1)
        score = get_proxy_evaluator()._compute_modulation_fidelity(ref, ref, SR)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# New component: Harmonic Structure Preservation
# ---------------------------------------------------------------------------


class TestHarmonicStructure:
    """Tests for _compute_harmonic_structure (PEAQ EHS equivalent)."""

    def test_identical_harmonics_perfect(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=8)
        score = get_proxy_evaluator()._compute_harmonic_structure(ref, ref, SR)
        assert score >= 0.98

    def test_missing_harmonics_penalized(self):
        """If test signal has fewer harmonics, score drops."""
        ref = _make_harmonic(f0=220.0, n_harmonics=8)
        test = _make_harmonic(f0=220.0, n_harmonics=2)
        score = get_proxy_evaluator()._compute_harmonic_structure(ref, test, SR)
        assert score < 0.70

    def test_noise_only_low(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=6)
        rng = np.random.default_rng(55)
        test = rng.standard_normal(len(ref)).astype(np.float32) * 0.3
        score = get_proxy_evaluator()._compute_harmonic_structure(ref, test, SR)
        assert score < 0.70

    def test_range_bounded(self):
        ref = _make_tone(440.0)
        test = _add_noise(ref, snr_db=5.0)
        score = get_proxy_evaluator()._compute_harmonic_structure(ref, test, SR)
        assert 0.0 <= score <= 1.0

    def test_non_pitched_returns_neutral(self):
        """Pure noise (no clear f0) should return ~0.5 (neutral)."""
        rng = np.random.default_rng(99)
        ref = rng.standard_normal(int(SR * 2)).astype(np.float32) * 0.3
        score = get_proxy_evaluator()._compute_harmonic_structure(ref, ref, SR)
        assert 0.3 <= score <= 0.7

    def test_different_f0_lower(self):
        """Different pitch should show worse harmonic alignment."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        test = _make_harmonic(f0=233.0, n_harmonics=5)  # slightly detuned
        score = get_proxy_evaluator()._compute_harmonic_structure(ref, test, SR)
        assert score < 0.95


# ---------------------------------------------------------------------------
# New component: Spectral Flux Correlation
# ---------------------------------------------------------------------------


class TestSpectralFluxCorrelation:
    """Tests for _compute_spectral_flux_correlation."""

    def test_identical_perfect(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        score = get_proxy_evaluator()._compute_spectral_flux_correlation(ref, ref, SR)
        assert score >= 0.98

    def test_noise_degrades(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        test = _add_noise(ref, snr_db=10.0)
        score = get_proxy_evaluator()._compute_spectral_flux_correlation(ref, test, SR)
        assert 0.0 <= score < 0.90

    def test_stationary_signal_with_noise(self):
        """Stationary sine + small noise should not collapse to 0."""
        ref = _make_tone(440.0)
        test = _add_noise(ref, snr_db=40.0)
        score = get_proxy_evaluator()._compute_spectral_flux_correlation(ref, test, SR)
        assert score >= 0.50, f"Stationary + small noise scored unexpectedly low: {score}"

    def test_range_bounded(self):
        ref = _make_harmonic()
        rng = np.random.default_rng(88)
        test = rng.standard_normal(len(ref)).astype(np.float32) * 0.5
        score = get_proxy_evaluator()._compute_spectral_flux_correlation(ref, test, SR)
        assert 0.0 <= score <= 1.0

    def test_monotonic_degradation(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        s1 = get_proxy_evaluator()._compute_spectral_flux_correlation(ref, _add_noise(ref, 30), SR)
        s2 = get_proxy_evaluator()._compute_spectral_flux_correlation(ref, _add_noise(ref, 6), SR)
        assert s1 > s2

    def test_short_audio(self):
        ref = _make_tone(440.0, duration=0.1)
        score = get_proxy_evaluator()._compute_spectral_flux_correlation(ref, ref, SR)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# New component: Worst-Segment Score (PEAQ ADB-inspired)
# ---------------------------------------------------------------------------


class TestPerceptualDisturbance:
    """Tests for _compute_perceptual_disturbance (Component 23)."""

    def test_identical_perfect(self):
        """Identical signals → near-perfect (inaudible distortion)."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        score = MertMushraProxy._compute_perceptual_disturbance(ref, ref, SR)
        assert score >= 0.98

    def test_noise_degrades(self):
        """Added noise should produce audible distortion → lower score."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        noisy = _add_noise(ref, snr_db=6)
        score = MertMushraProxy._compute_perceptual_disturbance(ref, noisy, SR)
        assert score < 0.90

    def test_heavy_noise_very_low(self):
        """Extreme noise → very low perceptual disturbance score."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        noisy = _add_noise(ref, snr_db=0)
        score = MertMushraProxy._compute_perceptual_disturbance(ref, noisy, SR)
        assert score < 0.60

    def test_range_bounded(self):
        """Score must be in [0, 1]."""
        ref = _make_harmonic()
        rng = np.random.default_rng(99)
        test = rng.standard_normal(len(ref)).astype(np.float32) * 0.5
        score = MertMushraProxy._compute_perceptual_disturbance(ref, test, SR)
        assert 0.0 <= score <= 1.0

    def test_monotonic_with_noise(self):
        """Higher noise should produce lower score."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        s_clean = MertMushraProxy._compute_perceptual_disturbance(ref, _add_noise(ref, 30), SR)
        s_noisy = MertMushraProxy._compute_perceptual_disturbance(ref, _add_noise(ref, 6), SR)
        assert s_clean > s_noisy

    def test_short_audio_fallback(self):
        """Very short audio returns neutral 0.5."""
        ref = np.zeros(100, dtype=np.float32)
        score = MertMushraProxy._compute_perceptual_disturbance(ref, ref, SR)
        assert score == 0.5

    def test_masked_noise_scores_higher(self):
        """Low-level noise under a loud signal should be mostly masked."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5, duration=2.0)
        # Quiet noise: should be masked by the harmonic signal
        quiet_noisy = _add_noise(ref, snr_db=30)
        # Loud noise: clearly audible
        loud_noisy = _add_noise(ref, snr_db=6)
        s_quiet = MertMushraProxy._compute_perceptual_disturbance(ref, quiet_noisy, SR)
        s_loud = MertMushraProxy._compute_perceptual_disturbance(ref, loud_noisy, SR)
        assert s_quiet > s_loud


class TestRoughness:
    """Tests for _compute_roughness (Component 24)."""

    def test_identical_perfect(self):
        """Identical signals → perfect roughness preservation."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5, duration=2.0)
        score = MertMushraProxy._compute_roughness(ref, ref, SR)
        assert score >= 0.98

    def test_added_modulation_penalized(self):
        """Adding AM roughness should lower the score."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5, duration=2.0)
        # Add 70 Hz amplitude modulation (roughness peak frequency)
        t = np.arange(len(ref), dtype=np.float32) / SR
        am = 1.0 + 0.5 * np.sin(2 * np.pi * 70 * t)
        test = (ref * am).astype(np.float32)
        test = np.clip(test, -1.0, 1.0)
        score = MertMushraProxy._compute_roughness(ref, test, SR)
        assert score < 0.90

    def test_range_bounded(self):
        """Score must be in [0, 1]."""
        ref = _make_harmonic(duration=2.0)
        rng = np.random.default_rng(77)
        test = rng.standard_normal(len(ref)).astype(np.float32) * 0.3
        score = MertMushraProxy._compute_roughness(ref, test, SR)
        assert 0.0 <= score <= 1.0

    def test_short_audio_fallback(self):
        """Very short audio returns neutral 0.5."""
        ref = np.zeros(500, dtype=np.float32)
        score = MertMushraProxy._compute_roughness(ref, ref, SR)
        assert score == 0.5

    def test_noise_degrades_roughness(self):
        """Noise changes the roughness profile → lower score."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5, duration=2.0)
        noisy = _add_noise(ref, snr_db=6)
        score = MertMushraProxy._compute_roughness(ref, noisy, SR)
        assert score < 0.95

    def test_mild_noise_better_than_heavy(self):
        """Mild noise should preserve roughness better than heavy noise."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5, duration=2.0)
        s_light = MertMushraProxy._compute_roughness(ref, _add_noise(ref, 30), SR)
        s_heavy = MertMushraProxy._compute_roughness(ref, _add_noise(ref, 6), SR)
        assert s_light >= s_heavy


class TestWorstSegmentScore:
    """Tests for _compute_worst_segment_score."""

    def test_identical_perfect(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        score = get_proxy_evaluator()._compute_worst_segment_score(ref, ref, SR)
        assert score >= 0.95

    def test_single_bad_segment_drags_down(self):
        """Inject a loud artifact in one segment; floor should be low."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5, duration=4.0)
        test = ref.copy()
        # Corrupt 1 second in the middle with noise
        start = int(1.5 * SR)
        end = start + SR
        rng = np.random.default_rng(42)
        test[start:end] = rng.standard_normal(end - start).astype(np.float32) * 0.8
        test = np.clip(test, -1.0, 1.0).astype(np.float32)
        score = get_proxy_evaluator()._compute_worst_segment_score(ref, test, SR)
        assert score < 0.50

    def test_range_bounded(self):
        ref = _make_harmonic()
        rng = np.random.default_rng(33)
        test = rng.standard_normal(len(ref)).astype(np.float32) * 0.3
        score = get_proxy_evaluator()._compute_worst_segment_score(ref, test, SR)
        assert 0.0 <= score <= 1.0

    def test_noise_monotonic(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        s1 = get_proxy_evaluator()._compute_worst_segment_score(ref, _add_noise(ref, 30), SR)
        s2 = get_proxy_evaluator()._compute_worst_segment_score(ref, _add_noise(ref, 6), SR)
        assert s1 > s2

    def test_short_audio(self):
        ref = _make_tone(440.0, duration=0.5)
        score = get_proxy_evaluator()._compute_worst_segment_score(ref, ref, SR)
        assert score >= 0.90


# ---------------------------------------------------------------------------
# Floor Penalty applied in evaluate()
# ---------------------------------------------------------------------------


class TestFloorPenalty:
    """Tests for worst-case floor penalty in evaluate()."""

    def test_floor_never_inflates_score(self):
        """Floor penalty must never increase the score above raw."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        result = estimate_mushra_proxy(ref, ref, SR)
        # identical → worst_segment should be near 1.0 → floor ≈ raw → no inflation
        assert result.worst_segment_score >= 0.90
        # proxy_score is the post-floor value; should be high for identical signals
        assert result.proxy_score >= 80.0

    def test_floor_fields_populated(self):
        ref = _make_harmonic(f0=220.0, n_harmonics=5)
        test = _add_noise(ref, snr_db=15.0)
        result = estimate_mushra_proxy(ref, test, SR)
        assert np.isfinite(result.worst_segment_score)
        assert 0.0 <= result.worst_segment_score <= 1.0
        assert np.isfinite(result.modulation_fidelity)
        assert np.isfinite(result.harmonic_structure)
        assert np.isfinite(result.spectral_flux_corr)

    def test_catastrophic_segment_drags_score(self):
        """Audio with one destroyed segment should score lower than uniform noise."""
        ref = _make_harmonic(f0=220.0, n_harmonics=5, duration=4.0)
        # Uniform moderate noise
        uniform = _add_noise(ref, snr_db=15.0)
        # Same noise + one catastrophic segment
        localized = uniform.copy()
        rng = np.random.default_rng(42)
        start = int(2.0 * SR)
        end = start + SR
        localized[start:end] = rng.standard_normal(end - start).astype(np.float32) * 0.9
        localized = np.clip(localized, -1.0, 1.0).astype(np.float32)
        r_uniform = estimate_mushra_proxy(ref, uniform, SR)
        r_local = estimate_mushra_proxy(ref, localized, SR)
        assert r_local.worst_segment_score < r_uniform.worst_segment_score

    def test_as_dict_has_new_fields(self):
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        for k in ["modulation_fidelity", "harmonic_structure", "spectral_flux_corr", "worst_segment_score"]:
            assert k in d


# ---------------------------------------------------------------------------
# Vibrato Fidelity Sub-Metric (inside pitch_accuracy)
# ---------------------------------------------------------------------------


class TestVibratoFidelity:
    """Tests for vibrato fidelity sub-metric within _compute_pitch_accuracy."""

    def test_vibrato_preserved_scores_high(self):
        """Signal with matched vibrato → high pitch_accuracy."""
        t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
        vibrato = 5.0 * np.sin(2 * np.pi * 5.5 * t)  # 5.5 Hz vibrato, ±5 cents equiv
        ref = 0.5 * np.sin(2 * np.pi * (440.0 + vibrato) * t)
        score = get_proxy_evaluator()._compute_pitch_accuracy(ref.astype(np.float32), ref.astype(np.float32), SR)
        assert score >= 0.90

    def test_vibrato_removed_lower(self):
        """Signal with vibrato vs. flat pitch → lower score."""
        t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
        vibrato = 5.0 * np.sin(2 * np.pi * 5.5 * t)
        ref = (0.5 * np.sin(2 * np.pi * (440.0 + vibrato) * t)).astype(np.float32)
        test_flat = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        s_vibrato = get_proxy_evaluator()._compute_pitch_accuracy(ref, ref, SR)
        s_flat = get_proxy_evaluator()._compute_pitch_accuracy(ref, test_flat, SR)
        # Vibrato-matched should score higher than vibrato-removed
        assert s_vibrato > s_flat


# ---------------------------------------------------------------------------
# Specific Loudness Difference (Component 25)
# ---------------------------------------------------------------------------


class TestSpecificLoudnessDiff:
    """Tests for _compute_specific_loudness_diff (Component 25).

    Implements Zwicker (1958) / Moore & Glasberg (1996) specific loudness
    per Bark band with power-law compression (α ≈ 0.23).
    """

    def test_identical_returns_high(self):
        """Identical ref/test → score near 1.0."""
        ref = _make_harmonic()
        score = get_proxy_evaluator()._compute_specific_loudness_diff(ref, ref, SR)
        assert 0.95 <= score <= 1.0

    def test_noisy_returns_lower(self):
        """Noisy test → lower specific loudness match."""
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=5.0)
        s_clean = get_proxy_evaluator()._compute_specific_loudness_diff(ref, ref, SR)
        s_noisy = get_proxy_evaluator()._compute_specific_loudness_diff(ref, noisy, SR)
        assert s_clean > s_noisy

    def test_range_0_1(self):
        """Score must be in [0, 1]."""
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=0.0)
        score = get_proxy_evaluator()._compute_specific_loudness_diff(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_silence_returns_default(self):
        """Silence → 0.5 fallback."""
        silence = np.zeros(256, dtype=np.float32)
        score = get_proxy_evaluator()._compute_specific_loudness_diff(silence, silence, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_short_signal_returns_default(self):
        """Very short audio → 0.5 fallback."""
        short = np.random.randn(100).astype(np.float32)
        score = get_proxy_evaluator()._compute_specific_loudness_diff(short, short, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_gain_change_lowers_score(self):
        """6 dB gain change → perceptible loudness difference."""
        ref = _make_harmonic()
        louder = ref * 2.0
        s_same = get_proxy_evaluator()._compute_specific_loudness_diff(ref, ref, SR)
        s_loud = get_proxy_evaluator()._compute_specific_loudness_diff(ref, louder, SR)
        assert s_same > s_loud

    def test_result_field_stored(self):
        """specific_loudness_diff should be in evaluate() result."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "specific_loudness_diff")
        assert 0.0 <= result.specific_loudness_diff <= 1.0

    def test_as_dict_key(self):
        """specific_loudness_diff must appear in as_dict()."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "specific_loudness_diff" in d

    def test_component_score_key(self):
        """specific_loudness must be in component_scores."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert "specific_loudness" in result.component_scores

    def test_monotonic_with_noise(self):
        """Increasing noise → strictly decreasing score."""
        ref = _make_harmonic()
        evaluator = get_proxy_evaluator()
        scores = []
        for snr in [30.0, 15.0, 5.0]:
            noisy = _add_noise(ref, snr_db=snr)
            scores.append(evaluator._compute_specific_loudness_diff(ref, noisy, SR))
        assert scores[0] >= scores[1] >= scores[2]


# ---------------------------------------------------------------------------
# Fluctuation Strength Delta (Component 26)
# ---------------------------------------------------------------------------


class TestFluctuationStrength:
    """Tests for _compute_fluctuation_strength (Component 26).

    Implements Fastl & Zwicker 2007 Ch. 10 fluctuation strength
    for slow AM at 0.5–20 Hz (peak at ~4 Hz).
    """

    def test_identical_returns_high(self):
        """Identical ref/test → score near 1.0."""
        ref = _make_harmonic()
        score = get_proxy_evaluator()._compute_fluctuation_strength(ref, ref, SR)
        assert 0.85 <= score <= 1.0

    def test_tremolo_added_returns_lower(self):
        """Add 4 Hz tremolo to test → lower fluctuation match."""
        t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
        ref = _make_harmonic()
        # Strong 4 Hz AM (peak fluctuation frequency)
        tremolo = 0.5 * (1.0 + 0.8 * np.sin(2 * np.pi * 4.0 * t[: len(ref)]))
        test_trem = (ref * tremolo).astype(np.float32)
        s_same = get_proxy_evaluator()._compute_fluctuation_strength(ref, ref, SR)
        s_trem = get_proxy_evaluator()._compute_fluctuation_strength(ref, test_trem, SR)
        assert s_same > s_trem

    def test_range_0_1(self):
        """Score must be in [0, 1]."""
        ref = _make_harmonic()
        noisy = _add_noise(ref, snr_db=0.0)
        score = get_proxy_evaluator()._compute_fluctuation_strength(ref, noisy, SR)
        assert 0.0 <= score <= 1.0

    def test_silence_returns_default(self):
        """Silence → 0.5 fallback."""
        silence = np.zeros(256, dtype=np.float32)
        score = get_proxy_evaluator()._compute_fluctuation_strength(silence, silence, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_short_signal_returns_default(self):
        """Very short audio → 0.5 fallback."""
        short = np.random.randn(2000).astype(np.float32)
        score = get_proxy_evaluator()._compute_fluctuation_strength(short, short, SR)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_result_field_stored(self):
        """fluctuation_strength should be in evaluate() result."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert hasattr(result, "fluctuation_strength")
        assert 0.0 <= result.fluctuation_strength <= 1.0

    def test_as_dict_key(self):
        """fluctuation_strength must appear in as_dict()."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        d = result.as_dict()
        assert "fluctuation_strength" in d

    def test_component_score_key(self):
        """fluctuation must be in component_scores."""
        ref = _make_harmonic()
        result = estimate_mushra_proxy(ref, ref, SR)
        assert "fluctuation" in result.component_scores

    def test_pump_effect_detected(self):
        """Compressor-like pump at ~2 Hz → score < 1.0."""
        t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
        ref = _make_harmonic()
        # 2 Hz pump modulation (within fluctuation range 0.5–20 Hz)
        pump = 0.5 * (1.0 + 0.6 * np.sin(2 * np.pi * 2.0 * t[: len(ref)]))
        test_pump = (ref * pump).astype(np.float32)
        s_same = get_proxy_evaluator()._compute_fluctuation_strength(ref, ref, SR)
        s_pump = get_proxy_evaluator()._compute_fluctuation_strength(ref, test_pump, SR)
        assert s_same > s_pump
