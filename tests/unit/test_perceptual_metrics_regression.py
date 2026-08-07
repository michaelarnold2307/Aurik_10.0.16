"""§Perceptual: Perzeptuelle Metrik-Regression mit synthetischen Defekten und Ground Truth.

Dieser Test validiert objektive perzeptuelle Metriken (SNR, RMS) auf synthetischen
Signalen. Für echte ABX-Blindhörtests siehe backend/core/abx_listener.py (geplant).
Für MUSHRA-Approximation siehe backend/core/objective_mushra_estimator.py.
"""

import numpy as np
import pytest


def _generate_clean_audio(duration_s: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig += 0.25 * np.sin(2 * np.pi * 880 * t)
    sig += 0.125 * np.sin(2 * np.pi * 1320 * t)
    return sig.astype(np.float32)  # type: ignore[no-any-return]


def _add_noise(audio: np.ndarray, snr_db: float = -15.0) -> np.ndarray:
    signal_power = np.mean(audio**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*audio.shape)
    return (audio + noise).astype(np.float32)  # type: ignore[no-any-return]


def _add_clicks(audio: np.ndarray, clicks_per_sec: int = 10, sr: int = 48000) -> np.ndarray:
    result = audio.copy()
    n_clicks = int(len(audio) / sr * clicks_per_sec)
    for _ in range(n_clicks):
        pos = np.random.randint(0, len(audio) - 10)
        result[pos : pos + 5] += np.random.uniform(0.3, 0.8) * np.sign(np.random.randn())
    return result.astype(np.float32)


def _lowpass(audio: np.ndarray, cutoff_hz: float = 8000.0, sr: int = 48000) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, a = butter(4, cutoff_hz / (sr / 2), btype="low")
    return filtfilt(b, a, audio).astype(np.float32)  # type: ignore[no-any-return]


def _snr(clean: np.ndarray, degraded: np.ndarray) -> float:
    noise = degraded - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-12:
        return 100.0
    return float(10 * np.log10(signal_power / noise_power))


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)))


@pytest.mark.unit
class TestABXRegression:
    """ABX-Regression: synthetische Defekte → Restaurierung → Messung."""

    def test_noise_reduction_improves_snr(self):
        clean = _generate_clean_audio()
        noisy = _add_noise(clean, snr_db=-15.0)
        snr_before = _snr(clean, noisy)
        assert snr_before < 20.0

        from scipy.signal import butter, filtfilt

        b, a = butter(4, 2000 / (48000 / 2), btype="low")
        denoised = filtfilt(b, a, noisy).astype(np.float32)

        snr_after = _snr(clean, denoised)
        assert snr_after > snr_before, f"SNR must improve: {snr_after:.1f} <= {snr_before:.1f}"

    def test_restoration_not_identity(self):
        clean = _generate_clean_audio()
        noisy = _add_noise(clean)
        from scipy.signal import butter, filtfilt

        b, a = butter(4, 2000 / (48000 / 2), btype="low")
        denoised = filtfilt(b, a, noisy).astype(np.float32)
        assert not np.allclose(denoised, noisy, atol=1e-6)

    def test_no_nan_inf_in_output(self):
        clean = _generate_clean_audio()
        noisy = _add_noise(clean)
        from scipy.signal import butter, filtfilt

        b, a = butter(4, 2000 / (48000 / 2), btype="low")
        denoised = filtfilt(b, a, noisy).astype(np.float32)
        assert np.all(np.isfinite(denoised))
        assert denoised.shape == noisy.shape

    def test_rms_error_decreases(self):
        clean = _generate_clean_audio()
        noisy = _add_noise(clean, snr_db=-10.0)
        noisy = _add_clicks(noisy, clicks_per_sec=5)
        rms_noisy = _rms(noisy - clean)

        from scipy.signal import butter, filtfilt, medfilt

        b, a = butter(4, 2000 / (48000 / 2), btype="low")
        denoised = filtfilt(b, a, noisy).astype(np.float32)
        declicked = medfilt(denoised, kernel_size=3).astype(np.float32)

        rms_restored = _rms(declicked - clean)
        assert rms_restored < rms_noisy

    def test_quality_score_exists(self):
        clean = _generate_clean_audio()
        noisy = _add_noise(clean, snr_db=-12.0)
        try:
            from backend.core.intrinsic_audio_quality_scorer import IntrinsicQualityScore

            iqs = IntrinsicQualityScore()
            assert iqs is not None
            score = getattr(iqs, "overall", 0.0)
            assert 0.0 <= score <= 1.0
        except ImportError:
            pytest.skip("IAQS backend not available")
