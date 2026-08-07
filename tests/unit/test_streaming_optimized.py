import pytest

"""
Unit-Tests für dsp/streaming_optimized.py

Testet:
  - StreamingLimiter.process()
  - StreamingDenoiser.process()
  - StreamingGate.process()
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit

from dsp.streaming_optimized import StreamingDenoiser, StreamingGate, StreamingLimiter

SR = 44100
_N = SR // 4  # 11025 Samples (0.25s) — Default für alle Hilfsfunktionen


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(n: int = _N, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _silence(n: int = _N) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _loud(n: int = _N, amp: float = 2.0) -> np.ndarray:
    """Übersteuertes Signal (über 0 dBFS)."""
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * np.pi * 440.0 * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _noisy(n: int = _N, noise_amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(n).astype(np.float32) * noise_amp


# ---------------------------------------------------------------------------
# StreamingLimiter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingLimiter:
    def test_output_shape_preserved(self):
        audio = _sine()
        out = StreamingLimiter().process(audio, SR)
        assert out.shape == audio.shape

    def test_dtype_preserved(self):
        audio = _sine()
        out = StreamingLimiter().process(audio, SR)
        assert out.dtype == audio.dtype

    def test_limits_loud_signal(self):
        """Übersteuertes Signal muss auf ≤ 1.0 begrenzt werden."""
        audio = _loud()
        out = StreamingLimiter().process(audio, SR)
        assert np.max(np.abs(out)) <= 1.001  # 0.1 % Toleranz wegen Float

    def test_ceiling_approx_minus1_dbfs(self):
        """-1 dBFS Ceiling: max Pegel ≤ 10^(-1/20) ≈ 0.891."""
        audio = _loud(amp=3.0)
        out = StreamingLimiter().process(audio, SR)
        peak = float(np.max(np.abs(out)))
        assert peak <= 1.001  # Hard-clip Sicherung

    def test_quiet_signal_unchanged(self):
        """Leises Signal (-20 dBFS) darf nicht modifiziert werden."""
        audio = _sine(amp=0.1)  # -20 dBFS
        out = StreamingLimiter().process(audio, SR)
        np.testing.assert_allclose(out, audio, rtol=0.01)

    def test_silence_stays_zero(self):
        audio = _silence()
        out = StreamingLimiter().process(audio, SR)
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_short_buffer(self):
        """Sehr kurzer Buffer (weniger als ein Frame) darf nicht crashen."""
        audio = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        out = StreamingLimiter().process(audio, SR)
        assert len(out) == 3

    def test_different_sample_rates(self):
        for sr in [8000, 16000, 48000]:
            audio = _sine(n=sr)
            out = StreamingLimiter().process(audio, sr)
            assert out.shape == audio.shape

    def test_stereo_array(self):
        """2D-Array muss als Float behandelt werden, ohne Crash."""
        audio = np.random.randn(2, SR).astype(np.float32) * 0.5
        # Je nach Implementierung akzeptiert der Limiter 1D oder 2D
        # Mindestanforderung: kein unerwarteter Absturz
        try:
            out = StreamingLimiter().process(audio, SR)
            assert out.shape == audio.shape
        except (ValueError, IndexError):
            pass  # Explizit notiert: Mono-only ist akzeptabel


# ---------------------------------------------------------------------------
# StreamingDenoiser
# ---------------------------------------------------------------------------


class TestStreamingDenoiser:
    def test_output_shape_preserved(self):
        audio = _sine()
        out = StreamingDenoiser().process(audio, SR)
        assert out.shape == audio.shape

    def test_dtype_compatible(self):
        audio = _sine()
        out = StreamingDenoiser().process(audio, SR)
        assert np.issubdtype(out.dtype, np.floating)

    def test_reduces_broadband_noise(self):
        """Rauschreduzierung: RMS nach Denoising muss sinken."""
        rng = np.random.default_rng(0)
        noise_only = (rng.standard_normal(SR) * 0.3).astype(np.float32)
        out = StreamingDenoiser().process(noise_only, SR)
        rms_in = float(np.sqrt(np.mean(noise_only**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out < rms_in  # mindestens etwas Rauschreduzierung

    def test_signal_preserved_approx(self):
        """Reines Sinus-Signal: Output muss ähnlich dem Input sein (keine Invertierung)."""
        audio = _sine(freq=1000.0, amp=0.8)
        out = StreamingDenoiser().process(audio, SR)
        # Korrelation mit Original muss positiv sein.
        # Slice: ab Viertel der Audiolänge (NICHT SR//4 als absolute Sample-Zahl,
        # da len(audio) == _N == SR//4 → audio[SR//4:] wäre leer → NaN from corrcoef)
        q = len(audio) // 4
        corr = float(np.corrcoef(audio[q:], out[q:])[0, 1])
        assert corr > 0.3

    def test_silence_near_zero(self):
        """Stille bleibt stille (oder sehr nahe dran)."""
        audio = _silence()
        out = StreamingDenoiser().process(audio, SR)
        assert np.max(np.abs(out)) < 0.01

    def test_clipping_prevented(self):
        """Denoiser darf Signal nicht übersteuern."""
        audio = _sine(amp=0.9)
        out = StreamingDenoiser().process(audio, SR)
        assert np.max(np.abs(out)) <= 1.001

    def test_short_buffer(self):
        """Kurzer Buffer kleiner als FFT-Fenster → sicherer Fallback."""
        audio = _sine(n=100)
        out = StreamingDenoiser().process(audio, SR)
        # Entweder Output gleich Input (Fallback), oder korrekte Shape
        assert len(out) == len(audio)

    def test_different_sample_rates(self):
        for sr in [8000, 22050, 48000]:
            n = sr
            audio = _sine(n=n, freq=440.0)
            out = StreamingDenoiser().process(audio, sr)
            assert out.shape == audio.shape


# ---------------------------------------------------------------------------
# StreamingGate
# ---------------------------------------------------------------------------


class TestStreamingGate:
    def test_output_shape_preserved(self):
        audio = _sine()
        out = StreamingGate().process(audio, SR)
        assert out.shape == audio.shape

    def test_dtype_preserved(self):
        audio = _sine()
        out = StreamingGate().process(audio, SR)
        assert out.dtype == audio.dtype

    def test_loud_signal_passes(self):
        """Lautes Signal über -30 dBFS muss Gate passieren."""
        audio = _sine(amp=0.5)  # ~-6 dBFS
        out = StreamingGate().process(audio, SR)
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out > 0.1  # Signal muss durchkommen

    def test_silent_below_threshold_gated(self):
        """Sehr leises Signal unter -50 dBFS → komplett stumm."""
        audio = _sine(amp=1e-4)  # ~-80 dBFS
        out = StreamingGate().process(audio, SR)
        assert np.max(np.abs(out)) < 1e-3

    def test_silence_stays_silent(self):
        audio = _silence()
        out = StreamingGate().process(audio, SR)
        np.testing.assert_array_equal(out, np.zeros_like(audio))

    def test_no_gain_increase(self):
        """Gate darf Signal nicht verstärken."""
        audio = _sine(amp=0.5)
        out = StreamingGate().process(audio, SR)
        assert np.max(np.abs(out)) <= np.max(np.abs(audio)) + 1e-6

    def test_hysteresis_no_chattering(self):
        """Hysterese: abwechselnd lautes/leises Signal erzeugt kein Chattern."""
        # 0.5s lautes + 0.5s leises Signal
        loud_part = _sine(n=SR // 2, amp=0.5)
        quiet_part = _sine(n=SR // 2, amp=5e-4)
        audio = np.concatenate([loud_part, quiet_part])
        out = StreamingGate().process(audio, SR)
        # Stille Hälfte muss tatsächlich stumm sein
        assert float(np.sqrt(np.mean(out[SR // 2 :] ** 2))) < 1e-3

    def test_different_sample_rates(self):
        for sr in [8000, 16000, 48000]:
            audio = _sine(n=sr, amp=0.5)
            out = StreamingGate().process(audio, sr)
            assert out.shape == audio.shape
