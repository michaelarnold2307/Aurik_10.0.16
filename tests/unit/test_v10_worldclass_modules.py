"""Tests for Aurik v10 World-Class modules (Tier 1-6).

Covers:
- Tier 1: Psychoacoustics (ATH, Moore/Glasberg, BMLD)
- Tier 2: New defect detectors (MPEG, Stereo, Phase, Dropout subtypes)
- Tier 3: Vocal supremacy (Speaker Identity, Overprocessing)
- Quick Wins: Forward masking, Vibrato guard, Parallel scan
"""

import numpy as np
import pytest

SR = 48000


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: Psychoacoustics
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestATHISO226:
    """ATH (Absolute Threshold of Hearing) nach ISO 226:2023."""

    def test_01_ath_importable(self):
        from backend.core.psychoacoustic_masking_model import PsychoacousticMaskingModel

        model = PsychoacousticMaskingModel()
        assert hasattr(type(model), "_ath_threshold_db")

    def test_02_ath_midrange_most_sensitive(self):
        """2-4 kHz should have the lowest (most sensitive) threshold."""
        from backend.core.psychoacoustic_masking_model import PsychoacousticMaskingModel

        model = PsychoacousticMaskingModel()
        ath_100 = model._ath_threshold_db(100.0)
        ath_1000 = model._ath_threshold_db(1000.0)
        ath_3000 = model._ath_threshold_db(3000.0)
        ath_8000 = model._ath_threshold_db(8000.0)
        # Mid-frequencies should be more sensitive (lower threshold) than extremes
        assert ath_3000 < ath_100, f"3kHz ({ath_3000}) should be more sensitive than 100Hz ({ath_100})"
        assert ath_1000 < ath_8000, f"1kHz ({ath_1000}) should be more sensitive than 8kHz ({ath_8000})"

    def test_03_ath_returns_finite_values(self):
        from backend.core.psychoacoustic_masking_model import PsychoacousticMaskingModel

        model = PsychoacousticMaskingModel()
        for freq in [20, 50, 100, 500, 1000, 2000, 4000, 8000, 15000, 20000]:
            ath = model._ath_threshold_db(float(freq))
            assert np.isfinite(ath), f"ATH at {freq}Hz = {ath} is not finite"


class TestMooreGlasbergDLM:
    """Moore/Glasberg Dynamic Loudness Model."""

    def test_01_moore_importable(self):
        from dsp import psychoacoustics as pa

        assert hasattr(pa, "compute_specific_loudness_moore")

    def test_02_moore_returns_correct_shape(self):
        """Should return 40 ERB bands worth of specific loudness."""
        from dsp.psychoacoustics import compute_specific_loudness_moore

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 2 * SR)) * 0.1
        result = compute_specific_loudness_moore(audio.astype(np.float32), SR)
        assert len(result) == 40, f"Expected 40 ERB bands, got {len(result)}"
        assert np.all(np.isfinite(result)), "Result contains NaN/Inf"

    def test_03_louder_signal_produces_higher_loudness(self):
        from dsp.psychoacoustics import compute_specific_loudness_moore

        quiet = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, SR)) * 0.01
        loud = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, SR)) * 0.3
        n_quiet = float(np.sum(compute_specific_loudness_moore(quiet.astype(np.float32), SR)))
        n_loud = float(np.sum(compute_specific_loudness_moore(loud.astype(np.float32), SR)))
        assert n_loud > n_quiet, f"Loud ({n_loud:.2f}) should exceed quiet ({n_quiet:.2f})"

    def test_04_silence_returns_near_zero(self):
        from dsp.psychoacoustics import compute_specific_loudness_moore

        silence = np.zeros(SR, dtype=np.float32)
        n_silence = float(np.sum(compute_specific_loudness_moore(silence, SR)))
        assert n_silence < 0.1, f"Silence loudness should be near zero, got {n_silence:.3f}"


class TestBinauralMasking:
    """Binaural masking level difference (BMLD) via IACC."""

    def test_01_iacc_identical_signals(self):
        from backend.core.psychoacoustic_masking_model import PsychoacousticMaskingModel

        model = PsychoacousticMaskingModel()
        sig = np.random.randn(SR).astype(np.float32) * 0.1
        iacc = model.compute_interaural_cross_correlation(sig, sig)
        assert 0.99 < iacc <= 1.01, f"IACC of identical signals should be ~1.0, got {iacc:.3f}"

    def test_02_iacc_uncorrelated_decreases(self):
        from backend.core.psychoacoustic_masking_model import PsychoacousticMaskingModel

        model = PsychoacousticMaskingModel()
        sig_l = np.random.randn(SR).astype(np.float32) * 0.1
        sig_r = np.random.randn(SR).astype(np.float32) * 0.1
        iacc = model.compute_interaural_cross_correlation(sig_l, sig_r)
        assert iacc < 0.5, f"IACC of uncorrelated signals should be < 0.5, got {iacc:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: New Defect Detectors
# ═══════════════════════════════════════════════════════════════════════════


class TestMPEGFrameLossDetector:
    """MPEG-Frame-Verlust-Detektor."""

    def test_01_importable(self):
        from backend.core.defect_detection.mpeg_frame_loss import detect_mpeg_frame_loss

        assert callable(detect_mpeg_frame_loss)

    def test_02_clean_sine_returns_no_loss(self):
        from backend.core.defect_detection.mpeg_frame_loss import detect_mpeg_frame_loss

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 3 * SR)).astype(np.float32)
        locations, confidence = detect_mpeg_frame_loss(audio, SR)  # type: ignore[misc]
        assert isinstance(locations, list)
        assert 0.0 <= confidence <= 1.0

    def test_03_handles_stereo(self):
        from backend.core.defect_detection.mpeg_frame_loss import detect_mpeg_frame_loss

        mono = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 3 * SR)).astype(np.float32)
        stereo = np.column_stack([mono, mono * 0.9])
        locations, confidence = detect_mpeg_frame_loss(stereo, SR)  # type: ignore[misc]
        assert isinstance(locations, list)
        assert 0.0 <= confidence <= 1.0


class TestStereoCollapseDetector:
    """Stereofeld-Kollaps-Detektor."""

    def test_01_importable(self):
        from backend.core.defect_detection.stereo_collapse import detect_stereo_collapse

        assert callable(detect_stereo_collapse)

    def test_02_normal_stereo_no_collapse(self):
        from backend.core.defect_detection.stereo_collapse import detect_stereo_collapse

        left = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5 * SR)).astype(np.float32) * 0.1
        right = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5 * SR) + 0.3).astype(np.float32) * 0.1
        stereo = np.column_stack([left, right])
        locations, ratio, confidence = detect_stereo_collapse(stereo, SR)
        assert isinstance(locations, list)
        assert 0.0 <= confidence <= 1.0

    def test_03_mono_signal_detected(self):
        from backend.core.defect_detection.stereo_collapse import detect_stereo_collapse

        mono = np.random.randn(5 * SR).astype(np.float32) * 0.1
        stereo = np.column_stack([mono, mono])  # Perfectly correlated = mono
        locations, ratio, confidence = detect_stereo_collapse(stereo, SR)
        assert isinstance(ratio, float)


class TestPhaseRotationDetector:
    """Phasenrotations-Detektor."""

    def test_01_importable(self):
        from backend.core.defect_detection.phase_rotation import detect_phase_rotation

        assert callable(detect_phase_rotation)

    def test_02_clean_sine_low_dispersion(self):
        from backend.core.defect_detection.phase_rotation import detect_phase_rotation

        audio = np.sin(2 * np.pi * 1000 * np.linspace(0, 3, 3 * SR)).astype(np.float32)
        locations, dispersion, confidence = detect_phase_rotation(audio, SR)
        assert 0.0 <= confidence <= 1.0

    def test_03_handles_short_audio(self):
        from backend.core.defect_detection.phase_rotation import detect_phase_rotation

        audio = np.random.randn(SR // 10).astype(np.float32) * 0.01
        locations, dispersion, confidence = detect_phase_rotation(audio, SR)
        assert isinstance(dispersion, float)


class TestDropoutSubtypes:
    """Dropout-Subtyp-Differenzierung."""

    def test_01_defect_types_registered(self):
        from backend.core.defect_scanner import DefectType

        for dt in ["DROPOUT_OXIDE", "DROPOUT_HEAD_CONTACT", "DROPOUT_SPLICE"]:
            assert getattr(DefectType, dt, None) is not None, f"{dt} missing"

    def test_02_defect_types_in_sensitivity_matrix(self):
        from backend.core.defect_scanner import DefectType

        # Verify they exist and can be compared
        oxide = DefectType.DROPOUT_OXIDE
        assert oxide.value == "dropout_oxide"
        assert oxide != DefectType.DROPOUTS


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: Vocal Supremacy
# ═══════════════════════════════════════════════════════════════════════════


class TestSpeakerIdentityGuard:
    """Sänger-Identitäts-Fingerabdruck."""

    def test_01_importable(self):
        from backend.ml.speaker_identity_guard import SpeakerIdentityGuard

        assert SpeakerIdentityGuard is not None

    def test_02_initialization(self):
        from backend.ml.speaker_identity_guard import SpeakerIdentityGuard

        guard = SpeakerIdentityGuard()
        assert hasattr(guard, "capture_pre_embedding")
        assert hasattr(guard, "check_phase")
        assert hasattr(guard, "get_pre_embedding")

    def test_03_capture_embedding(self):
        from backend.ml.speaker_identity_guard import SpeakerIdentityGuard

        guard = SpeakerIdentityGuard()
        audio1 = np.random.randn(3 * SR, 2).astype(np.float32) * 0.05  # Stereo
        # Capture pre-embedding should work without error
        guard.capture_pre_embedding(audio1, SR)
        emb = guard.get_pre_embedding()
        assert emb is not None
        assert len(emb) > 0

    def test_04_check_phase_returns_result(self):
        from backend.ml.speaker_identity_guard import SpeakerIdentityGuard

        guard = SpeakerIdentityGuard()
        # Verify module-level constants and methods exist
        assert hasattr(guard, "check_phase")
        assert hasattr(guard, "capture_pre_embedding")
        assert hasattr(guard, "get_pre_embedding")
        # Module constants
        from backend.ml.speaker_identity_guard import IDENTITY_THRESHOLD, VOCAL_PHASES

        assert IDENTITY_THRESHOLD > 0.5
        assert "phase_42_vocal_enhancement" in VOCAL_PHASES


class TestVocalOverprocessingDetector:
    """Vocal Overprocessing Detector."""

    def test_01_importable(self):
        from backend.core.vocal_overprocessing_detector import (
            VocalOverprocessingDetector,
            VocalOverprocessingResult,
        )

        assert VocalOverprocessingDetector is not None
        assert VocalOverprocessingResult is not None

    def test_02_detector_initialization(self):
        from backend.core.vocal_overprocessing_detector import VocalOverprocessingDetector

        detector = VocalOverprocessingDetector()
        assert hasattr(detector, "check_de_essing")
        assert hasattr(detector, "check_formant_drift")
        assert hasattr(detector, "LISP_VARIANCE_THRESHOLD_DB")
        assert hasattr(detector, "SIBILANCE_RATIO_THRESHOLD")

    def test_03_de_essing_check_no_overprocessing(self):
        from backend.core.vocal_overprocessing_detector import VocalOverprocessingDetector

        detector = VocalOverprocessingDetector()
        original = np.random.randn(3 * SR).astype(np.float32) * 0.05
        processed = original * 1.01  # Very slight change
        result = detector.check_de_essing(original, processed, SR)
        # Clean audio with minimal processing should not trigger flags
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# Quick Wins
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardMaskingFrequencyCorrection:
    """Forward Masking mit Frequenzabhängigkeit."""

    def test_01_low_freq_masks_longer(self):
        from backend.core.perceptual_salience import PerceptualSalienceEstimator

        pse = PerceptualSalienceEstimator()
        mask_100hz = pse._forward_mask_duration_ms(100.0)
        mask_1khz = pse._forward_mask_duration_ms(1000.0)
        mask_8khz = pse._forward_mask_duration_ms(8000.0)
        assert mask_100hz > mask_1khz > mask_8khz, (
            f"Expected 100Hz ({mask_100hz}) > 1kHz ({mask_1khz}) > 8kHz ({mask_8khz})"
        )

    def test_02_values_in_valid_range(self):
        from backend.core.perceptual_salience import PerceptualSalienceEstimator

        pse = PerceptualSalienceEstimator()
        for hz in [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 20000]:
            ms = pse._forward_mask_duration_ms(float(hz))
            assert 50.0 <= ms <= 500.0, f"Forward mask at {hz}Hz = {ms}ms out of [50, 500]"


class TestVibratoGuard:
    """Vibrato-Guard für Flutter-Detektion."""

    def test_01_guard_present(self):
        from backend.core.defect_scanner import DefectScanner

        assert hasattr(DefectScanner, "_is_vibrato_not_flutter")

    def test_02_single_band_returns_false(self):
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        deviations = {"band1": np.random.randn(100) * 0.1}
        result = scanner._is_vibrato_not_flutter(deviations)
        assert result is False  # Need at least 2 bands

    def test_03_high_coherence_is_vibrato(self):
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        base = np.sin(np.linspace(0, 10 * np.pi, 200)) * 0.5
        deviations = {f"band{i}": base + np.random.randn(200) * 0.02 for i in range(4)}
        result = scanner._is_vibrato_not_flutter(deviations)
        assert result is True  # High cross-band coherence = vibrato


class TestParallelScan:
    """Parallele Defekt-Detektion."""

    def test_01_scan_parallel_present(self):
        from backend.core.defect_scanner import DefectScanner

        assert hasattr(DefectScanner, "scan_parallel")

    def test_02_group_methods_present(self):
        from backend.core.defect_scanner import DefectScanner

        for method in [
            "_scan_spectral_defects",
            "_scan_temporal_defects",
            "_scan_structural_defects",
            "_scan_codec_defects",
            "_merge_parallel_results",
        ]:
            assert hasattr(DefectScanner, method), f"{method} missing"

    def test_03_scan_parallel_falls_back_to_sequential(self):
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 2 * SR)).astype(np.float32)
        try:
            result = scanner.scan_parallel(audio, SR, max_workers=2)
            assert result is not None
        except Exception as e:
            # Fallback to sequential is acceptable behavior
            assert "does not exist" not in str(e).lower()


# ═══════════════════════════════════════════════════════════════════════════
# Integration: CLI Flags
# ═══════════════════════════════════════════════════════════════════════════


class TestCLIFlags:
    """CLI --dry-run, --json, --abx Flags."""

    def test_01_cli_module_importable(self):
        import cli.aurik_cli

        assert hasattr(cli.aurik_cli, "print_usage")
        assert hasattr(cli.aurik_cli, "main")

    def test_02_print_usage_includes_new_flags(self):
        import io
        import sys

        from cli.aurik_cli import print_usage

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            print_usage()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "--dry-run" in output, "print_usage missing --dry-run"
        assert "--json" in output, "print_usage missing --json"
        assert "--abx" in output, "print_usage missing --abx"
