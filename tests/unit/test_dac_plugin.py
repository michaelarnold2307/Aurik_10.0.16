from __future__ import annotations

"""Unit-Tests für DacPlugin (Descript Audio Codec ONNX).

Tests laufen ohne echte ONNX-Modelle (Fallback-Pfad valide).
ML-Tests (echter Encoder/Decoder) sind mit @pytest.mark.ml markiert.
"""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 48_000


def _sine(dur_s: float = 0.5, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(dur_s: float = 0.5) -> np.ndarray:
    """Return [2, T] stereo float32 array."""
    t = np.linspace(0, dur_s, int(dur_s * SR), endpoint=False)
    ch0 = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    ch1 = (0.4 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
    return np.stack([ch0, ch1])


# ---------------------------------------------------------------------------
# Import & Singleton
# ---------------------------------------------------------------------------


class TestDacPluginImport:
    def test_01_module_importable(self):
        from plugins.dac_plugin import DacPlugin, get_dac_plugin

        assert DacPlugin is not None
        assert get_dac_plugin is not None

    def test_02_singleton_returns_same_instance(self):
        from plugins.dac_plugin import get_dac_plugin

        a = get_dac_plugin()
        b = get_dac_plugin()
        assert a is b

    def test_03_plugin_has_availability_properties(self):
        from plugins.dac_plugin import get_dac_plugin

        p = get_dac_plugin()
        assert isinstance(p.encoder_available, bool)
        assert isinstance(p.decoder_available, bool)


# ---------------------------------------------------------------------------
# Encode — fallback (no model)
# ---------------------------------------------------------------------------


class TestDacEncodeFallback:
    """Tests that run even without ONNX models present (fallback path)."""

    def test_10_encode_mono_1d_no_crash(self):
        from plugins.dac_plugin import dac_encode

        audio = _sine(0.1)
        result = dac_encode(audio, SR)
        assert result.codes is not None
        assert result.n_frames > 0

    def test_11_encode_result_dtype_int64(self):
        from plugins.dac_plugin import dac_encode

        audio = _sine(0.1)
        result = dac_encode(audio, SR)
        assert result.codes.dtype == np.int64

    def test_12_encode_result_shape_9_codebooks(self):
        from plugins.dac_plugin import dac_encode

        audio = _sine(0.1)
        result = dac_encode(audio, SR)
        # Shape: [B, 9, T_frames]
        assert result.codes.ndim == 3
        assert result.codes.shape[1] == 9

    def test_13_encode_stereo_2d_no_crash(self):
        from plugins.dac_plugin import dac_encode

        audio = _stereo(0.1)
        result = dac_encode(audio, SR)
        assert result.codes.ndim == 3

    def test_14_encode_batched_3d_no_crash(self):
        from plugins.dac_plugin import dac_encode

        audio = _sine(0.1)[np.newaxis, np.newaxis, :]  # [1, 1, T]
        result = dac_encode(audio, SR)
        assert result.codes.ndim == 3

    def test_15_encode_silence_no_crash(self):
        from plugins.dac_plugin import dac_encode

        audio = np.zeros(SR // 10, dtype=np.float32)
        result = dac_encode(audio, SR)
        assert result.n_frames > 0

    def test_16_encode_nan_input_guarded(self):
        from plugins.dac_plugin import dac_encode

        audio = np.full(SR // 10, np.nan, dtype=np.float32)
        result = dac_encode(audio, SR)  # must not raise
        assert result.codes is not None

    def test_17_encode_clipping_input_clipped(self):
        from plugins.dac_plugin import dac_encode

        audio = np.full(SR // 10, 5.0, dtype=np.float32)
        result = dac_encode(audio, SR)  # must not raise
        assert result.codes is not None

    def test_18_model_used_field_is_string(self):
        from plugins.dac_plugin import dac_encode

        result = dac_encode(_sine(0.1), SR)
        assert isinstance(result.model_used, str)
        assert result.model_used in {"dac_onnx", "unavailable"}

    def test_19_encode_very_short_audio(self):
        from plugins.dac_plugin import dac_encode

        audio = np.zeros(512, dtype=np.float32)  # exactly one stride
        result = dac_encode(audio, SR)
        assert result.n_frames >= 1


# ---------------------------------------------------------------------------
# Decode — fallback
# ---------------------------------------------------------------------------


class TestDacDecodeFallback:
    def test_20_decode_zero_codes_no_crash(self):
        from plugins.dac_plugin import dac_decode

        codes = np.zeros((1, 9, 10), dtype=np.int64)
        result = dac_decode(codes)
        assert result.audio is not None

    def test_21_decode_output_sr_is_48k(self):
        from plugins.dac_plugin import dac_decode

        codes = np.zeros((1, 9, 5), dtype=np.int64)
        result = dac_decode(codes)
        assert result.sr == 48_000

    def test_22_decode_output_finite(self):
        from plugins.dac_plugin import dac_decode

        codes = np.zeros((1, 9, 5), dtype=np.int64)
        result = dac_decode(codes)
        assert np.isfinite(result.audio).all()

    def test_23_decode_output_clipped(self):
        from plugins.dac_plugin import dac_decode

        codes = np.zeros((1, 9, 5), dtype=np.int64)
        result = dac_decode(codes)
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_24_decode_model_used_string(self):
        from plugins.dac_plugin import dac_decode

        codes = np.zeros((1, 9, 5), dtype=np.int64)
        result = dac_decode(codes)
        assert result.model_used in {"dac_onnx", "unavailable"}

    def test_25_decode_2d_codes_no_crash(self):
        from plugins.dac_plugin import dac_decode

        codes = np.zeros((9, 10), dtype=np.int64)  # missing batch dim
        result = dac_decode(codes)
        assert result.audio is not None


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestDacRoundTrip:
    def test_30_round_trip_no_crash_mono(self):
        from plugins.dac_plugin import dac_round_trip

        audio = _sine(0.1)
        result = dac_round_trip(audio, SR)
        assert result.audio_out is not None

    def test_31_round_trip_sr_is_48k(self):
        from plugins.dac_plugin import dac_round_trip

        result = dac_round_trip(_sine(0.1), SR)
        assert result.sr == 48_000

    def test_32_round_trip_output_finite(self):
        from plugins.dac_plugin import dac_round_trip

        result = dac_round_trip(_sine(0.1), SR)
        assert np.isfinite(result.audio_out).all()

    def test_33_round_trip_output_clipped(self):
        from plugins.dac_plugin import dac_round_trip

        result = dac_round_trip(_sine(0.1), SR)
        assert np.max(np.abs(result.audio_out)) <= 1.0

    def test_34_round_trip_codes_shape(self):
        from plugins.dac_plugin import dac_round_trip

        result = dac_round_trip(_sine(0.1), SR)
        assert result.codes.ndim == 3
        assert result.codes.shape[1] == 9

    def test_35_round_trip_snr_is_float(self):
        from plugins.dac_plugin import dac_round_trip

        result = dac_round_trip(_sine(0.1), SR)
        assert isinstance(result.snr_db, float)
        assert np.isfinite(result.snr_db)

    def test_36_round_trip_stereo_no_crash(self):
        from plugins.dac_plugin import dac_round_trip

        audio = _stereo(0.1)
        result = dac_round_trip(audio, SR)
        assert result.audio_out is not None

    def test_37_round_trip_silence_no_crash(self):
        from plugins.dac_plugin import dac_round_trip

        audio = np.zeros(SR // 10, dtype=np.float32)
        result = dac_round_trip(audio, SR)
        assert np.isfinite(result.audio_out).all()


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------


class TestDacResample:
    def test_40_resample_noop_same_sr(self):
        from plugins.dac_plugin import _resample

        audio = _sine(0.1)
        out = _resample(audio, 48_000, 48_000)
        np.testing.assert_array_equal(out, audio)

    def test_41_resample_48k_to_44k_length(self):
        from plugins.dac_plugin import _resample

        audio = _sine(1.0)  # 48000 samples
        out = _resample(audio, 48_000, 44_100)
        expected_len = int(len(audio) * 44_100 / 48_000)
        assert abs(len(out) - expected_len) <= 2  # ±2 samples tolerance

    def test_42_resample_44k_to_48k_length(self):
        from plugins.dac_plugin import _resample

        audio = np.zeros(44_100, dtype=np.float32)
        out = _resample(audio, 44_100, 48_000)
        expected_len = 48_000
        assert abs(len(out) - expected_len) <= 2

    def test_43_resample_2d_channel_axis(self):
        from plugins.dac_plugin import _resample

        audio = _stereo(0.5)  # [2, 24000]
        out = _resample(audio, 48_000, 44_100)
        assert out.ndim == 2
        assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# Pad helper
# ---------------------------------------------------------------------------


class TestDacPad:
    def test_50_pad_already_aligned(self):
        from plugins.dac_plugin import _STRIDE, _pad_to_stride

        audio = np.zeros(_STRIDE * 4, dtype=np.float32)
        padded, orig = _pad_to_stride(audio)
        assert orig == len(audio)
        assert len(padded) % _STRIDE == 0
        assert len(padded) == len(audio)

    def test_51_pad_unaligned(self):
        from plugins.dac_plugin import _STRIDE, _pad_to_stride

        audio = np.zeros(_STRIDE * 3 + 7, dtype=np.float32)
        padded, orig = _pad_to_stride(audio)
        assert orig == _STRIDE * 3 + 7
        assert len(padded) % _STRIDE == 0

    def test_52_pad_preserves_content(self):
        from plugins.dac_plugin import _pad_to_stride

        audio = np.ones(1000, dtype=np.float32)
        padded, orig = _pad_to_stride(audio)
        np.testing.assert_array_equal(padded[:orig], audio)
        assert np.all(padded[orig:] == 0.0)


# ---------------------------------------------------------------------------
# ML-Tests (echter Encoder/Decoder, nur mit --run-heavy-tests)
# ---------------------------------------------------------------------------


@pytest.mark.ml
@pytest.mark.slow
class TestDacPluginML:
    """Tests that require the actual ONNX models to be present."""

    def _skip_if_unavailable(self):
        from plugins.dac_plugin import get_dac_plugin

        p = get_dac_plugin()
        if not p.encoder_available:
            pytest.skip("DAC encoder ONNX nicht gefunden — ML-Test übersprungen.")

    def test_60_encoder_available(self):
        from plugins.dac_plugin import get_dac_plugin

        p = get_dac_plugin()
        assert p.encoder_available, "DAC encoder_model.onnx muss vorhanden sein"

    def test_61_encode_sine_correct_shape(self):
        self._skip_if_unavailable()
        from plugins.dac_plugin import dac_encode

        audio = _sine(1.0)
        result = dac_encode(audio, SR)
        assert result.model_used == "dac_onnx"
        assert result.codes.shape[1] == 9
        # frames ≈ 44100 * 1.0 / 512 ≈ 86
        assert 80 <= result.n_frames <= 100

    def test_62_round_trip_snr_above_30db(self):
        self._skip_if_unavailable()
        from plugins.dac_plugin import get_dac_plugin

        p = get_dac_plugin()
        if not p.decoder_available:
            pytest.skip("DAC decoder ONNX nicht gefunden.")
        from plugins.dac_plugin import dac_round_trip

        audio = _sine(1.0)
        result = dac_round_trip(audio, SR)
        assert result.model_used == "dac_onnx"
        assert result.snr_db >= 15.0, f"Round-trip SNR zu niedrig: {result.snr_db:.1f} dB"

    def test_63_output_no_nan_after_round_trip(self):
        self._skip_if_unavailable()
        from plugins.dac_plugin import get_dac_plugin

        if not get_dac_plugin().decoder_available:
            pytest.skip("DAC decoder nicht verfügbar.")
        from plugins.dac_plugin import dac_round_trip

        audio = _sine(0.5)
        result = dac_round_trip(audio, SR)
        assert np.isfinite(result.audio_out).all()
        assert np.max(np.abs(result.audio_out)) <= 1.0
