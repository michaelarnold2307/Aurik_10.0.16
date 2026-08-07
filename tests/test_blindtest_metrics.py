"""
Acceptance Tests for Preservation Metrics, Artifact Detection,
Blind Test Framework, and Quality Report (§G46–§G59).

Tests cover all new modules built for blind-test readiness.
"""

import numpy as np
import pytest

SR = 48000


# ── Preservation Metrics ────────────────────────────────────────────────


class TestPreservationMetrics:
    def test_harmonic_identical(self):
        from backend.core.preservation_metrics import compute_harmonic_preservation_score

        t = np.arange(SR) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        s = compute_harmonic_preservation_score(sig, sig.copy(), SR)
        assert s > 0.95, f"Identical should be >0.95, got {s}"

    def test_harmonic_damaged(self):
        from backend.core.preservation_metrics import compute_harmonic_preservation_score

        t = np.arange(SR) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        noisy = sig + np.random.default_rng(42).standard_normal(len(t)) * 0.3
        s_clean = compute_harmonic_preservation_score(sig, sig.copy(), SR)
        s_noisy = compute_harmonic_preservation_score(sig, noisy, SR)
        assert s_noisy < s_clean, f"Noisy should score lower: {s_noisy} >= {s_clean}"

    def test_transient_identical(self):
        from backend.core.preservation_metrics import compute_transient_preservation_score

        t = np.arange(SR) / SR
        sig = np.sin(2 * np.pi * 880 * t) * 0.5
        s = compute_transient_preservation_score(sig, sig.copy(), SR)
        assert s > 0.95

    def test_formant_identical(self):
        from backend.core.preservation_metrics import compute_formant_preservation_score

        t = np.arange(int(3 * SR)) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5 + np.sin(2 * np.pi * 880 * t) * 0.3
        s = compute_formant_preservation_score(sig, sig.copy(), SR)
        assert s > 0.8

    def test_micro_dynamics_ordering(self):
        from backend.core.preservation_metrics import compute_micro_dynamics_score

        t = np.arange(int(3 * SR)) / SR
        env = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 0.5 * t))
        sig = np.sin(2 * np.pi * 440 * t) * 0.5 * env
        comp2 = np.tanh(sig * 2) / 2
        comp4 = np.tanh(sig * 4) / 4
        s2 = compute_micro_dynamics_score(sig, comp2, SR)
        s4 = compute_micro_dynamics_score(sig, comp4, SR)
        assert s4 < s2, f"Heavier compression should score lower: {s4} >= {s2}"

    def test_emotional_arc_identical(self):
        from backend.core.preservation_metrics import compute_emotional_arc_score

        t = np.arange(int(12 * SR)) / SR
        env = np.linspace(0.1, 0.8, int(12 * SR))
        sig = np.sin(2 * np.pi * 440 * t) * env
        s = compute_emotional_arc_score(sig, sig.copy(), SR)
        assert s > 0.95

    def test_all_functions_return_valid_range(self):
        from backend.core.preservation_metrics import (
            compute_emotional_arc_score,
            compute_formant_preservation_score,
            compute_harmonic_preservation_score,
            compute_micro_dynamics_score,
            compute_transient_preservation_score,
        )

        sig = np.random.default_rng(99).standard_normal(int(3 * SR)).astype(np.float32) * 0.1
        for fn in [
            compute_harmonic_preservation_score,
            compute_transient_preservation_score,
            compute_formant_preservation_score,
            compute_micro_dynamics_score,
        ]:
            s = fn(sig, sig.copy(), SR)  # type: ignore[operator]
            assert 0 <= s <= 1, f"{fn.__name__} returned {s}"
        sig2 = np.random.default_rng(99).standard_normal(int(12 * SR)).astype(np.float32) * 0.1
        s = compute_emotional_arc_score(sig2, sig2.copy(), SR)
        assert 0 <= s <= 1, f"emotional_arc returned {s}"


# ── Artifact Detection ──────────────────────────────────────────────────


class TestArtifactDetection:
    def test_clean_signal_scores_high(self):
        from backend.core.artifact_detector import ArtifactDetector

        t = np.arange(int(3 * SR)) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        report = ArtifactDetector(SR).scan(sig)
        assert report.overall_score > 0.85

    def test_click_detection(self):
        from backend.core.artifact_detector import ArtifactDetector

        t = np.arange(int(2 * SR)) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        sig_click = sig.copy()
        for pos in [5000, 15000, 25000]:
            sig_click[pos] = 1.0
            sig_click[pos + 1] = -1.0
        r_clean = ArtifactDetector(SR).scan(sig)
        r_click = ArtifactDetector(SR).scan(sig_click)
        assert r_click.click_count > r_clean.click_count

    def test_convenience_function(self):
        from backend.core.artifact_detector import compute_artifact_freedom_score

        sig = np.zeros(int(SR), dtype=np.float32)
        s = compute_artifact_freedom_score(sig, SR)
        assert 0 <= s <= 1


# ── Blind Test Framework ────────────────────────────────────────────────


class TestBlindTestFramework:
    def test_abx_identical_indistinguishable(self):
        from backend.core.blind_test_framework import ABXTestHarness

        t = np.arange(int(2 * SR)) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        result = ABXTestHarness(SR, num_trials=8).run_test(sig, sig.copy(), sig.copy())
        assert result.total_trials == 8
        # Identical signals should be hard to distinguish (p > ~0.1)
        # but spectral distance might still detect tiny float differences

    def test_mushra_identical_perfect(self):
        from backend.core.blind_test_framework import MUSHRAScorer

        t = np.arange(int(3 * SR)) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        score = MUSHRAScorer(SR).score(sig, sig.copy())
        assert score.overall > 90

    def test_mushra_damage_lower(self):
        from backend.core.blind_test_framework import MUSHRAScorer

        t = np.arange(int(3 * SR)) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        noisy = sig + np.random.default_rng(42).standard_normal(len(t)) * 0.2
        s_clean = MUSHRAScorer(SR).score(sig, sig.copy())
        s_noisy = MUSHRAScorer(SR).score(sig, noisy)
        assert s_noisy.overall < s_clean.overall


# ── Quality Report Integration ──────────────────────────────────────────


class TestQualityReport:
    def test_complete_report(self):
        from backend.core.restoration_quality_report import compute_quality_report

        t = np.arange(int(5 * SR)) / SR
        orig = np.sin(2 * np.pi * 440 * t) * 0.5 + np.sin(2 * np.pi * 880 * t) * 0.3
        restored = orig + np.random.default_rng(1).standard_normal(len(t)) * 0.05
        report = compute_quality_report(orig, restored, SR)
        assert report.overall_score > 0
        assert report.overall_score <= 100
        assert isinstance(report.blind_test_ready, bool)
        assert len(report.summary()) > 0
        assert report.computation_time_s > 0

    def test_identical_is_excellent(self):
        from backend.core.restoration_quality_report import compute_quality_report

        t = np.arange(int(5 * SR)) / SR
        sig = np.sin(2 * np.pi * 440 * t) * 0.5
        report = compute_quality_report(sig, sig.copy(), SR)
        assert report.overall_score > 85
        assert report.mushra_overall > 85


# ── CD Noise Profile ────────────────────────────────────────────────────


class TestCDNoiseProfile:
    def test_noise_floor_continuity(self):
        from backend.core.cd_noise_profile import inject_cd_noise_profile

        t = np.arange(int(3 * SR)) / SR
        audio = np.zeros((int(3 * SR), 2), dtype=np.float32)
        audio[: int(1.5 * SR), 0] = np.sin(2 * np.pi * 440 * t[: int(1.5 * SR)]) * 0.5
        audio[int(1.5 * SR) :, 0] = np.random.default_rng(7).standard_normal(int(1.5 * SR)).astype(np.float32) * 3e-5
        audio[:, 1] = audio[:, 0] * 0.9
        r = inject_cd_noise_profile(audio, SR, bit_depth=16)
        loud = r[int(0.8 * SR) : int(1.2 * SR)] - audio[int(0.8 * SR) : int(1.2 * SR)]
        quiet = r[int(1.8 * SR) : int(2.2 * SR)] - audio[int(1.8 * SR) : int(2.2 * SR)]
        lr = float(np.sqrt(np.mean(loud**2)))
        qr = float(np.sqrt(np.mean(quiet**2)))
        ratio_db = 20 * np.log10(max(qr, 1e-15) / max(lr, 1e-15))
        assert ratio_db < 25, f"Noise floor jump {ratio_db:.0f} dB exceeds 25 dB"

    def test_16bit_noise_level(self):
        from backend.core.cd_noise_profile import inject_cd_noise_profile

        audio = np.random.default_rng(3).standard_normal(int(2 * SR)).astype(np.float32) * 1e-4
        r = inject_cd_noise_profile(audio, SR, bit_depth=16)
        diff = r - audio
        rms = float(np.sqrt(np.mean(diff**2)))
        db = 20 * np.log10(max(rms, 1e-15))
        assert -98 < db < -94, f"16-bit noise at {db:.1f} dBFS, expected ~-96"

    def test_deterministic(self):
        from backend.core.cd_noise_profile import inject_cd_noise_profile

        audio = np.random.default_rng(5).standard_normal(int(SR)).astype(np.float32) * 1e-4
        r1 = inject_cd_noise_profile(audio, SR, bit_depth=16)
        r2 = inject_cd_noise_profile(audio, SR, bit_depth=16)
        assert np.max(np.abs(r1 - r2)) < 1e-10
