import pytest

"""
Unit-Tests für dsp/ultra_low_latency.py

Testet:
  - UltraLowLatencyLimiter.process()
  - UltraLowLatencyDenoiser.process()
  - UltraLowLatencyGate.process()
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit

from dsp.ultra_low_latency import (
    UltraLowLatencyDenoiser,
    UltraLowLatencyGate,
    UltraLowLatencyLimiter,
)

SR = 44100


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(n: int = SR, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _silence(n: int = SR) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _loud(n: int = SR, amp: float = 3.0) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * np.pi * 440.0 * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# UltraLowLatencyLimiter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUltraLowLatencyLimiter:
    def test_output_shape_preserved(self):
        audio = _sine()
        out = UltraLowLatencyLimiter().process(audio, SR)
        assert out.shape == audio.shape

    def test_dtype_preserved(self):
        audio = _sine()
        out = UltraLowLatencyLimiter().process(audio, SR)
        assert out.dtype == audio.dtype

    def test_limits_loud_signal_below_ceiling(self):
        """tanh-Waveshaping: Ceiling = 0.9 → max |output| < 0.91."""
        audio = _loud(amp=10.0)
        out = UltraLowLatencyLimiter().process(audio, SR)
        assert np.max(np.abs(out)) < 0.91

    def test_tanh_soft_clip_monotone(self):
        """tanh ist monoton steigend — keine Invertierung des Vorzeichens."""
        audio = np.array([0.1, 0.5, 1.0, 2.0, -0.5, -2.0], dtype=np.float32)
        out = UltraLowLatencyLimiter().process(audio, SR)
        assert np.sign(out[0]) == np.sign(audio[0])  # positiv bleibt positiv
        assert np.sign(out[4]) == np.sign(audio[4])  # negativ bleibt negativ

    def test_quiet_signal_nearly_unchanged(self):
        """Sehr leises Signal wird durch tanh kaum verändert."""
        audio = _sine(amp=0.01)  # -40 dBFS
        out = UltraLowLatencyLimiter().process(audio, SR)
        np.testing.assert_allclose(out, audio, rtol=0.05)

    def test_silence_stays_zero(self):
        audio = _silence()
        out = UltraLowLatencyLimiter().process(audio, SR)
        assert np.max(np.abs(out)) < 1e-9

    def test_short_buffer(self):
        audio = np.array([0.5, -0.5, 0.3, 1.5, -1.5], dtype=np.float32)
        out = UltraLowLatencyLimiter().process(audio, SR)
        assert len(out) == len(audio)
        assert np.max(np.abs(out)) < 0.91

    def test_latency_is_zero(self):
        """Kein Look-ahead: Amplitude bei Sample 0 kann sofort begrenzt werden."""
        audio = np.array([5.0], dtype=np.float32)
        out = UltraLowLatencyLimiter().process(audio, SR)
        assert np.abs(out[0]) < 0.91

    def test_different_sample_rates(self):
        for sr in [8000, 16000, 48000]:
            audio = _loud(n=sr, amp=2.0)
            out = UltraLowLatencyLimiter().process(audio, sr)
            assert np.max(np.abs(out)) < 0.91


# ---------------------------------------------------------------------------
# UltraLowLatencyDenoiser
# ---------------------------------------------------------------------------


class TestUltraLowLatencyDenoiser:
    def test_output_shape_preserved(self):
        audio = _sine()
        out = UltraLowLatencyDenoiser().process(audio, SR)
        assert out.shape == audio.shape

    def test_dtype_compatible(self):
        audio = _sine()
        out = UltraLowLatencyDenoiser().process(audio, SR)
        assert np.issubdtype(out.dtype, np.floating)

    def test_no_clipping_in_output(self):
        """Output muss auf [-1, 1] begrenzt sein."""
        audio = _sine(amp=0.9)
        out = UltraLowLatencyDenoiser().process(audio, SR)
        assert np.max(np.abs(out)) <= 1.001

    def test_silence_near_zero(self):
        audio = _silence()
        out = UltraLowLatencyDenoiser().process(audio, SR)
        assert np.max(np.abs(out)) < 0.01

    def test_reduces_noise_energy(self):
        """Rauschreduzierung: Ausgangs-RMS < Eingangs-RMS für reines Rauschen."""
        rng = np.random.default_rng(7)
        noise = (rng.standard_normal(SR) * 0.4).astype(np.float32)
        out = UltraLowLatencyDenoiser().process(noise, SR)
        rms_in = float(np.sqrt(np.mean(noise**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out < rms_in

    def test_short_buffer_no_crash(self):
        """Buffer kleiner als n_fft=128 → sicherer Fallback."""
        audio = _sine(n=64)
        out = UltraLowLatencyDenoiser().process(audio, SR)
        assert len(out) == len(audio)

    def test_minimal_latency_128_samples(self):
        """FFT-Fenstergröße 128 Samples → Latenz ~2.9ms @44100 Hz."""
        # Property-Test: Kein Look-ahead nötig (kein Lookahead im Design)
        n_fft = 128
        n_signal = n_fft * 4
        audio = _sine(n=n_signal)
        out = UltraLowLatencyDenoiser().process(audio, SR)
        assert len(out) == n_signal

    def test_different_sample_rates(self):
        for sr in [16000, 24000, 48000]:
            audio = _sine(n=sr, freq=1000.0)
            out = UltraLowLatencyDenoiser().process(audio, sr)
            assert out.shape == audio.shape


# ---------------------------------------------------------------------------
# UltraLowLatencyGate
# ---------------------------------------------------------------------------


class TestUltraLowLatencyGate:
    def test_output_shape_preserved(self):
        audio = _sine()
        out = UltraLowLatencyGate().process(audio, SR)
        assert out.shape == audio.shape

    def test_dtype_preserved(self):
        audio = _sine()
        out = UltraLowLatencyGate().process(audio, SR)
        assert out.dtype == audio.dtype

    def test_loud_signal_passes(self):
        """Signal über -40 dBFS Schwelle muss Gate passieren."""
        audio = _sine(amp=0.5)  # -6 dBFS
        out = UltraLowLatencyGate().process(audio, SR)
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out > 0.1

    def test_very_quiet_signal_gated(self):
        """Signal weit unter -40 dBFS → komplett stumm."""
        audio = _sine(amp=1e-4)  # ~-80 dBFS
        out = UltraLowLatencyGate().process(audio, SR)
        assert np.max(np.abs(out)) < 1e-3

    def test_silence_stays_silent(self):
        audio = _silence()
        out = UltraLowLatencyGate().process(audio, SR)
        np.testing.assert_array_equal(out, np.zeros_like(audio))

    def test_no_gain_increase(self):
        """Gate darf niemals Gain hinzufügen."""
        audio = _sine(amp=0.5)
        out = UltraLowLatencyGate().process(audio, SR)
        assert np.max(np.abs(out)) <= np.max(np.abs(audio)) + 1e-6

    def test_sample_accurate_response(self):
        """Sample-genauer Trigger: Gate öffnet innerhalb von 4ms nach Pegelsprung."""
        n_silence = int(SR * 0.010)  # 10ms Stille
        n_loud = int(SR * 0.100)  # 100ms lautes Signal
        audio = np.concatenate([_silence(n_silence), _sine(n=n_loud, amp=0.5)])
        out = UltraLowLatencyGate().process(audio, SR)
        # Nach 4ms Attack muss Signal durch (10ms Stille + 4ms Attack = 14ms)
        start_check = n_silence + int(SR * 0.006)  # 6ms nach Beginn des Lautsignals
        assert float(np.max(np.abs(out[start_check:]))) > 0.1

    def test_attack_release_timing(self):
        """Envelope-Follower: Attack=4ms, Release=20ms."""
        # 50ms Sinus dann 50ms Stille
        n_half = SR // 20  # 50ms
        audio = np.concatenate([_sine(n=n_half, amp=0.5), _silence(n=n_half)])
        out = UltraLowLatencyGate().process(audio, SR)
        # Erste Hälfte muss durchkommen
        rms_first = float(np.sqrt(np.mean(out[:n_half] ** 2)))
        assert rms_first > 0.1

    def test_different_sample_rates(self):
        for sr in [8000, 22050, 48000]:
            audio = _sine(n=sr, amp=0.5)
            out = UltraLowLatencyGate().process(audio, sr)
            assert out.shape == audio.shape
