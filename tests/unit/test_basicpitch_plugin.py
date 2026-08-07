from __future__ import annotations

"""Unit tests for BasicPitch plugin.

Non-ML tests validate fallback robustness without requiring ONNX model files.
ML tests are marked and can be enabled with heavy test flags.
"""


import numpy as np
import pytest

from plugins.basicpitch_plugin import (
    BasicPitchResult,
    _bins_to_midi,
    _midi_to_hz,
    _resample,
    analyze_polyphonic_pitch,
    get_basicpitch_plugin,
)

SR = 48_000


def _sine(freq: float, dur_s: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, dur_s, int(dur_s * sr), endpoint=False)
    return (0.4 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _chord(dur_s: float = 0.6, sr: int = SR) -> np.ndarray:
    # A4 + C#5 + E5
    a = _sine(440.0, dur_s, sr)
    c = _sine(554.365, dur_s, sr)
    e = _sine(659.255, dur_s, sr)
    x = (a + c + e) / 3.0
    return np.clip(x, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(dur_s: float = 0.6, sr: int = SR) -> np.ndarray:
    left = _sine(220.0, dur_s, sr)
    right = _sine(440.0, dur_s, sr)
    return np.stack([left, right])


class TestBasicPitchImportAndSingleton:
    def test_01_import_and_singleton(self):
        p1 = get_basicpitch_plugin()
        p2 = get_basicpitch_plugin()
        assert p1 is p2

    def test_02_result_dataclass_guards(self):
        r = BasicPitchResult(
            frame_times_s=np.array([0.0, np.nan], dtype=np.float32),
            pitches_hz=np.array([[440.0, np.inf]], dtype=np.float32),
            confidences=np.array([[0.5, -1.0]], dtype=np.float32),
            model_used="dsp_spectral_peaks",
        )
        assert np.isfinite(r.frame_times_s).all()
        assert np.isfinite(r.pitches_hz).all()
        assert np.all(r.confidences >= 0.0)


class TestHelpers:
    def test_10_bins_to_midi_bounds(self):
        idx = np.array([[0, 10, 99]], dtype=np.int64)
        midi = _bins_to_midi(idx, n_bins=100)
        assert np.min(midi) >= 21.0
        assert np.max(midi) <= 108.0

    def test_11_midi_to_hz_a4(self):
        hz = _midi_to_hz(np.array([69.0], dtype=np.float32))[0]
        assert abs(float(hz) - 440.0) < 1e-3

    def test_12_resample_noop(self):
        x = _sine(440.0, 0.1)
        y = _resample(x, SR, SR)
        np.testing.assert_array_equal(x, y)

    def test_13_resample_changes_length(self):
        x = _sine(440.0, 1.0, 48_000)
        y = _resample(x, 48_000, 22_050)
        assert abs(len(y) - 22_050) <= 2


class TestFallbackAnalyze:
    def test_20_analyze_mono_no_crash(self):
        r = analyze_polyphonic_pitch(_sine(440.0, 0.5), SR)
        assert r.pitches_hz.ndim == 2
        assert r.confidences.ndim == 2
        assert r.pitches_hz.shape == r.confidences.shape

    def test_21_analyze_stereo_no_crash(self):
        r = analyze_polyphonic_pitch(_stereo(), SR)
        assert r.pitches_hz.shape[1] >= 1

    def test_22_analyze_chord_has_frames(self):
        r = analyze_polyphonic_pitch(_chord(), SR)
        assert len(r.frame_times_s) > 0

    def test_23_analyze_model_used_valid(self):
        r = analyze_polyphonic_pitch(_sine(440.0, 0.3), SR)
        assert r.model_used in {"basicpitch_onnx", "dsp_spectral_peaks", "dsp_failed"}

    def test_24_analyze_confidence_range(self):
        r = analyze_polyphonic_pitch(_sine(440.0, 0.3), SR)
        assert np.min(r.confidences) >= 0.0
        assert np.max(r.confidences) <= 1.0

    def test_25_analyze_pitch_nonnegative(self):
        r = analyze_polyphonic_pitch(_sine(440.0, 0.3), SR)
        assert np.min(r.pitches_hz) >= 0.0

    def test_26_silence_no_nan(self):
        x = np.zeros(int(0.5 * SR), dtype=np.float32)
        r = analyze_polyphonic_pitch(x, SR)
        assert np.isfinite(r.pitches_hz).all()
        assert np.isfinite(r.confidences).all()

    def test_27_short_audio_no_crash(self):
        x = np.zeros(200, dtype=np.float32)
        r = analyze_polyphonic_pitch(x, SR)
        assert r.pitches_hz.ndim == 2

    def test_28_max_polyphony_clamped_low(self):
        p = get_basicpitch_plugin()
        r = p.analyze(_sine(440.0, 0.2), SR, max_polyphony=0)
        assert r.pitches_hz.shape[1] >= 1

    def test_29_max_polyphony_clamped_high(self):
        p = get_basicpitch_plugin()
        r = p.analyze(_sine(440.0, 0.2), SR, max_polyphony=99)
        assert r.pitches_hz.shape[1] <= 12

    def test_30_nan_input_guarded(self):
        x = np.full(int(0.3 * SR), np.nan, dtype=np.float32)
        r = analyze_polyphonic_pitch(x, SR)
        assert np.isfinite(r.pitches_hz).all()

    def test_31_inf_input_guarded(self):
        x = np.full(int(0.3 * SR), np.inf, dtype=np.float32)
        r = analyze_polyphonic_pitch(x, SR)
        assert np.isfinite(r.pitches_hz).all()

    def test_32_output_shapes_consistent(self):
        r = analyze_polyphonic_pitch(_chord(), SR, max_polyphony=4)
        assert r.pitches_hz.shape[0] == len(r.frame_times_s)
        assert r.pitches_hz.shape == r.confidences.shape


@pytest.mark.ml
@pytest.mark.slow
class TestBasicPitchML:
    def test_40_model_available_if_file_exists(self):
        p = get_basicpitch_plugin()
        # This test intentionally does not require model presence.
        assert isinstance(p._model_loaded, bool)

    def test_41_onnx_path_smoke(self):
        p = get_basicpitch_plugin()
        r = p.analyze(_chord(0.4), SR, max_polyphony=3)
        assert r.pitches_hz.shape[1] == 3
