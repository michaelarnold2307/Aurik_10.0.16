from __future__ import annotations

"""
tests/unit/test_gap_reconstructor.py
======================================
Unit-Tests für GapReconstructor (Differenzierer #4 — Semantische Lückenfüllung).

Testet:
  - Lückenerkennung (verschiedene Dauern + Amplituden)
  - Methoden: Linear, AR, Spektral
  - Bidirektionale AR-Prädiktion (Burg-Algorithmus)
  - Stereo-Handling
  - Material-Hints
  - Edge Cases
"""


import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.gap_reconstructor import (
    GapReconstructionResult,
    GapReconstructor,
    GapReconstructorConfig,
    _ar_predict_backward,
    _ar_predict_forward,
    _burg_ar,
    _db_to_linear,
    _rms,
    _spectral_interp,
)

# ===========================================================================
# Fixtures
# ===========================================================================

SR = 44100


def _sine(freq: float = 440.0, dur_s: float = 0.5, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _insert_gap(audio: np.ndarray, start_ms: float, dur_ms: float, sr: int = SR) -> np.ndarray:
    """Fügt eine exakte Stille-Lücke in ein Signal ein."""
    a = audio.copy()
    start = int(start_ms * sr / 1000)
    end = int((start_ms + dur_ms) * sr / 1000)
    a[start:end] = 0.0
    return a


@pytest.fixture(scope="module")
def sine_audio():
    return _sine(440.0, 0.25)  # 0.25s statt 1.0s — 4× schneller


@pytest.fixture(scope="module")
def stereo_sine():
    left = _sine(440.0, 0.25)
    right = _sine(880.0, 0.25)
    return np.stack([left, right], axis=1)


# ===========================================================================
# Test: Hilfsfunktionen
# ===========================================================================


@pytest.mark.unit
class TestHelpers:
    def test_db_to_linear_0db(self):
        assert abs(_db_to_linear(0.0) - 1.0) < 1e-6

    def test_db_to_linear_minus20db(self):
        assert abs(_db_to_linear(-20.0) - 0.1) < 1e-4

    def test_rms_zero(self):
        assert _rms(np.zeros(100)) == 0.0

    def test_rms_sine(self):
        t = np.linspace(0, 1, 44100)
        x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        r = _rms(x)
        assert 0.69 < r < 0.72  # RMS von sin = 1/√2 ≈ 0.707

    def test_burg_ar_length(self):
        x = np.random.randn(200).astype(np.float64)
        coeffs = _burg_ar(x, order=10)
        assert len(coeffs) == 10

    def test_burg_ar_order_cap(self):
        """Ordnung wird auf len(x)-1 gecapped."""
        x = np.random.randn(5).astype(np.float64)
        coeffs = _burg_ar(x, order=100)
        assert len(coeffs) <= 4

    def test_ar_forward_prediction_length(self):
        x = np.random.randn(50).astype(np.float32)
        coeffs = _burg_ar(x.astype(np.float64), 8).astype(np.float32)
        pred = _ar_predict_forward(x, coeffs, 20)
        assert len(pred) == 20

    def test_ar_backward_prediction_length(self):
        x = np.random.randn(50).astype(np.float32)
        coeffs = _burg_ar(x.astype(np.float64), 8).astype(np.float32)
        pred = _ar_predict_backward(x, coeffs, 20)
        assert len(pred) == 20

    def test_spectral_interp_length(self):
        pre = np.random.randn(1024).astype(np.float32)
        post = np.random.randn(1024).astype(np.float32)
        patch = _spectral_interp(pre, post, 512, SR)
        assert len(patch) == 512


# ===========================================================================
# Test: Konfiguration
# ===========================================================================


class TestConfig:
    def test_default_config(self):
        cfg = GapReconstructorConfig()
        assert cfg.silence_threshold_db < 0
        assert cfg.min_gap_duration_ms > 0
        assert cfg.max_gap_duration_ms > cfg.min_gap_duration_ms

    def test_custom_config(self):
        cfg = GapReconstructorConfig(silence_threshold_db=-60.0, min_gap_duration_ms=1.0)
        assert cfg.silence_threshold_db == -60.0
        assert cfg.min_gap_duration_ms == 1.0


# ===========================================================================
# Test: Lückenerkennung
# ===========================================================================


class TestGapDetection:
    def test_detect_single_gap(self):
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, start_ms=200.0, dur_ms=10.0)
        recon = GapReconstructor()
        gaps = recon.detect_only(audio_gap, SR)
        # Mindestens eine Lücke erkannt
        assert len(gaps) >= 1

    def test_detect_no_gap(self):
        audio = _sine(440.0, 1.0)  # kein Aussetzer
        recon = GapReconstructor()
        gaps = recon.detect_only(audio, SR)
        assert len(gaps) == 0

    def test_gap_duration_ms_approx(self):
        dur_ms = 20.0
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, start_ms=300.0, dur_ms=dur_ms)
        recon = GapReconstructor()
        gaps = recon.detect_only(audio_gap, SR)
        assert any(abs(g.duration_ms - dur_ms) < 2.0 for g in gaps)

    def test_gap_below_min_not_detected(self):
        """Sehr kurze Lücken (< min_gap_duration_ms) werden ignoriert."""
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, start_ms=300.0, dur_ms=0.1)
        cfg = GapReconstructorConfig(min_gap_duration_ms=0.5)
        recon = GapReconstructor(cfg)
        gaps = recon.detect_only(audio_gap, SR)
        assert len(gaps) == 0

    def test_gap_above_max_not_detected(self):
        """Zu große Lücken (> max_gap_duration_ms) werden ignoriert."""
        audio = np.zeros(SR, dtype=np.float32)  # nur Stille → über max
        cfg = GapReconstructorConfig(max_gap_duration_ms=100.0)
        recon = GapReconstructor(cfg)
        gaps = recon.detect_only(audio, SR)
        assert len(gaps) == 0

    def test_multiple_gaps_detected(self):
        audio = _sine(440.0, 2.0)
        audio = _insert_gap(audio, 200.0, 10.0)
        audio = _insert_gap(audio, 800.0, 15.0)
        audio = _insert_gap(audio, 1500.0, 20.0)
        recon = GapReconstructor()
        gaps = recon.detect_only(audio, SR)
        assert len(gaps) >= 3

    def test_stereo_gap_detected_per_channel(self):
        stereo = np.random.randn(SR, 2).astype(np.float32) * 0.5
        stereo[int(0.3 * SR) : int(0.3 * SR) + 441, 0] = 0.0  # Kanal 0 Lücke 10ms
        recon = GapReconstructor()
        gaps = recon.detect_only(stereo, SR)
        assert any(g.channel == 0 for g in gaps)


# ===========================================================================
# Test: Reparatur-Methoden
# ===========================================================================


class TestLinearRepair:
    def test_linear_method_selected_for_short_gap(self):
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, 300.0, 1.0)  # 1ms → linear
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        linear_repairs = [g for g in result.gap_details if g.method_used == "linear"]
        assert len(linear_repairs) >= 1

    def test_linear_repair_fills_gap(self):
        audio = _sine(440.0, 1.0)
        start_ms, dur_ms = 300.0, 1.0
        audio_gap = _insert_gap(audio, start_ms, dur_ms)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        start_s = int(start_ms * SR / 1000)
        end_s = int((start_ms + dur_ms) * SR / 1000)
        # Repaired region should no longer be silent
        assert _rms(result.audio[start_s:end_s]) > 0.001

    def test_linear_output_no_nan(self):
        audio = _sine(440.0, 0.5)
        audio_gap = _insert_gap(audio, 100.0, 1.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        assert not np.any(np.isnan(result.audio))


class TestARRepair:
    def test_ar_method_selected_for_medium_gap(self):
        audio = _sine(440.0, 2.0)
        audio_gap = _insert_gap(audio, 500.0, 20.0)  # 20ms → AR
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        ar_repairs = [g for g in result.gap_details if g.method_used == "ar"]
        assert len(ar_repairs) >= 1

    def test_ar_repair_not_silent(self):
        audio = _sine(440.0, 2.0)
        start_ms, dur_ms = 500.0, 20.0
        audio_gap = _insert_gap(audio, start_ms, dur_ms)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        start_s = int(start_ms * SR / 1000)
        end_s = int((start_ms + dur_ms) * SR / 1000)
        assert _rms(result.audio[start_s:end_s]) > 0.001

    def test_ar_repair_no_nan_no_inf(self):
        audio = _sine(440.0, 2.0)
        audio_gap = _insert_gap(audio, 500.0, 30.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        assert not np.any(np.isnan(result.audio))
        assert not np.any(np.isinf(result.audio))

    def test_ar_output_clipped_to_one(self):
        audio = _sine(440.0, 2.0)
        audio_gap = _insert_gap(audio, 500.0, 25.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_ar_stabilize_flag(self):
        cfg = GapReconstructorConfig(ar_stabilize=True)
        recon = GapReconstructor(cfg)
        audio = _sine(440.0, 2.0)
        audio_gap = _insert_gap(audio, 500.0, 20.0)
        result = recon.reconstruct(audio_gap, SR)
        assert not np.any(np.isnan(result.audio))
        assert not np.any(np.isinf(result.audio))


class TestSpectralRepair:
    def test_spectral_method_selected_for_long_gap(self):
        audio = _sine(440.0, 3.0)
        audio_gap = _insert_gap(audio, 1000.0, 100.0)  # 100ms → spektral
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        spectral_repairs = [g for g in result.gap_details if g.method_used == "spectral"]
        assert len(spectral_repairs) >= 1

    def test_spectral_repair_not_silent(self):
        audio = _sine(440.0, 3.0)
        start_ms, dur_ms = 1000.0, 100.0
        audio_gap = _insert_gap(audio, start_ms, dur_ms)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        start_s = int(start_ms * SR / 1000)
        end_s = int((start_ms + dur_ms) * SR / 1000)
        assert _rms(result.audio[start_s:end_s]) > 0.0001

    def test_spectral_repair_no_nan(self):
        audio = _sine(440.0, 3.0)
        audio_gap = _insert_gap(audio, 1000.0, 80.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        assert not np.any(np.isnan(result.audio))


# ===========================================================================
# Test: GapReconstructionResult
# ===========================================================================


class TestResult:
    def test_result_type(self):
        audio = _sine(440.0, 1.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert isinstance(result, GapReconstructionResult)

    def test_result_audio_shape_preserved_mono(self):
        audio = _sine(440.0, 1.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert result.audio.shape == audio.shape

    def test_result_audio_shape_preserved_stereo(self, stereo_sine):
        recon = GapReconstructor()
        result = recon.reconstruct(stereo_sine, SR)
        assert result.audio.shape == stereo_sine.shape

    def test_gaps_found_counted(self):
        audio = _sine(440.0, 2.0)
        audio = _insert_gap(audio, 300.0, 15.0)
        audio = _insert_gap(audio, 900.0, 20.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert result.gaps_found >= 2

    def test_gaps_repaired_le_found(self):
        audio = _sine(440.0, 1.0)
        audio = _insert_gap(audio, 200.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert result.gaps_repaired <= result.gaps_found

    def test_repair_rate_range(self):
        audio = _sine(440.0, 1.0)
        audio = _insert_gap(audio, 200.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert 0.0 <= result.repair_rate <= 1.0

    def test_total_repaired_ms_nonneg(self):
        audio = _sine(440.0, 1.0)
        audio = _insert_gap(audio, 200.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert result.total_repaired_ms >= 0.0

    def test_processing_time_ms_positive(self):
        audio = _sine(440.0, 1.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert result.processing_time_ms >= 0.0

    def test_summary_string(self):
        audio = _sine(440.0, 1.0)
        audio = _insert_gap(audio, 200.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        s = result.summary()
        assert isinstance(s, str)
        assert "GapReconstructor" in s
        assert "Lücken" in s

    def test_no_gaps_scenario(self):
        audio = _sine(440.0, 0.5)  # kein Dropout
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert result.gaps_found == 0
        assert result.gaps_repaired == 0
        np.testing.assert_array_equal(result.audio, audio)


# ===========================================================================
# Test: Material-Hints
# ===========================================================================


class TestMaterialHints:
    def test_vinyl_hint_accepted(self):
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, 300.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR, material_hint="vinyl")
        assert isinstance(result, GapReconstructionResult)

    def test_tape_hint_accepted(self):
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, 300.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR, material_hint="tape")
        assert isinstance(result, GapReconstructionResult)

    def test_shellac_hint_accepted(self):
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, 300.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR, material_hint="shellac")
        assert isinstance(result, GapReconstructionResult)

    def test_unknown_hint_fallback(self):
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, 300.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR, material_hint="cd_rom")  # unbekannt
        assert isinstance(result, GapReconstructionResult)


# ===========================================================================
# Test: Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_all_silent_audio(self):
        audio = np.zeros(SR, dtype=np.float32)  # vollständige Stille
        cfg = GapReconstructorConfig(max_gap_duration_ms=50.0)  # capped
        recon = GapReconstructor(cfg)
        result = recon.reconstruct(audio, SR)
        assert isinstance(result, GapReconstructionResult)

    def test_very_short_audio(self):
        audio = np.random.randn(100).astype(np.float32) * 0.1
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert result.audio.shape == (100,)

    def test_gap_at_start(self):
        audio = _sine(440.0, 1.0)
        audio[:220] = 0.0  # ~5ms Stille am Anfang
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert isinstance(result, GapReconstructionResult)
        assert not np.any(np.isnan(result.audio))

    def test_gap_at_end(self):
        audio = _sine(440.0, 1.0)
        audio[-220:] = 0.0  # ~5ms Stille am Ende
        recon = GapReconstructor()
        result = recon.reconstruct(audio, SR)
        assert isinstance(result, GapReconstructionResult)
        assert not np.any(np.isnan(result.audio))

    def test_output_dtype_float32(self):
        audio = _sine(440.0, 1.0)
        audio_gap = _insert_gap(audio, 300.0, 10.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        assert result.audio.dtype == np.float32

    def test_noise_signal_no_crash(self):
        rng = np.random.default_rng(99)
        audio = rng.normal(0, 0.3, SR).astype(np.float32)
        audio_gap = _insert_gap(audio, 200.0, 25.0)
        recon = GapReconstructor()
        result = recon.reconstruct(audio_gap, SR)
        assert not np.any(np.isnan(result.audio))
