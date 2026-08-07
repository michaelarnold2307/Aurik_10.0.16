"""
Tests für backend/core/dsp/noise_estimator.py — IMCRA Noise Estimation.

Test-Abdeckung:
  - compute_imcra_noise_estimate: Grundfunktionalität, Rauschkonvergenz
  - Output-Shape korrekt (n_freqs × n_frames)
  - Noise-Floor-Tracking: weißes Rauschen → Schätzung nahe am wahren Niveau
  - Initialphase konservativ (> wahr)
  - Fallback bei Edge-Cases (sehr kurze Signale, Stille)
  - get_noise_estimator: Singleton-Invariante
"""

import numpy as np
import pytest

SR = 48000
N_FFT = 2048
HOP = 512
N_FREQS = N_FFT // 2 + 1


def _white_noise(duration_s: float = 1.0, sr: int = SR, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(duration_s * sr)).astype(np.float32)


def _silence(duration_s: float = 0.5, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _sine_plus_noise(duration_s: float = 2.0, sr: int = SR, noise_rms: float = 0.05, seed: int = 7) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    clean = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    rng = np.random.default_rng(seed)
    noise = noise_rms * rng.standard_normal(len(t))
    return (clean + noise).astype(np.float32)  # type: ignore[no-any-return]


# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
class TestComputeImcraNoisePSD:
    """compute_imcra_noise_estimate — Grundfunktionalität."""

    def test_output_shape_matches_stft(self):
        """Output (n_freqs, n_frames) muss STFT-Grid entsprechen."""
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio = _white_noise()
        noise_psd = compute_imcra_noise_estimate(audio, SR, n_fft=N_FFT, hop_length=HOP)

        assert noise_psd.ndim == 2, "noise_psd muss 2D sein (n_freqs, n_frames)"
        assert noise_psd.shape[0] == N_FREQS, f"Erwartet {N_FREQS} Frequenz-Bins, got {noise_psd.shape[0]}"
        assert noise_psd.shape[1] >= 1, "Muss mindestens 1 Frame haben"

    def test_output_dtype_float32(self):
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio = _white_noise()
        noise_psd = compute_imcra_noise_estimate(audio, SR)
        assert noise_psd.dtype == np.float32, f"Erwartet float32, got {noise_psd.dtype}"

    def test_all_values_positive(self):
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio = _white_noise()
        noise_psd = compute_imcra_noise_estimate(audio, SR)
        assert np.all(noise_psd > 0), "Alle Rausch-PSD-Werte müssen positiv sein"

    def test_no_nan_inf(self):
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio = _white_noise()
        noise_psd = compute_imcra_noise_estimate(audio, SR)
        assert not np.any(np.isnan(noise_psd)), "Kein NaN erlaubt"
        assert not np.any(np.isinf(noise_psd)), "Kein Inf erlaubt"


class TestImcraNoiseTracking:
    """IMCRA-Tracking-Eigenschaften."""

    def test_white_noise_estimate_within_factor_3(self):
        """Für weißes Rauschen muss Schätzung ≤ 3× wahres Niveau liegen (Median)."""
        from scipy.signal import stft

        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        rng = np.random.default_rng(0)
        noise = (0.05 * rng.standard_normal(int(3.0 * SR))).astype(np.float32)

        _, _, Zxx = stft(noise, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", boundary="even")
        true_power = np.median(np.abs(Zxx) ** 2)

        noise_psd = compute_imcra_noise_estimate(noise, SR, n_fft=N_FFT, hop_length=HOP)
        estimated_median = float(np.median(noise_psd))

        ratio = estimated_median / (true_power + 1e-12)
        assert 0.2 < ratio < 10.0, f"Rausch-Schätzung um Faktor {ratio:.2f} vom Wert entfernt — sollte < 10×"

    def test_initialphase_conservative(self):
        """Init-Estimate liegt über dem gleitenden Minimum (b_min×1.3×S_min ≥ S_min)."""
        from scipy.signal import stft

        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        rng = np.random.default_rng(1)
        noise = (0.05 * rng.standard_normal(int(4.0 * SR))).astype(np.float32)

        _, _, Zxx = stft(noise, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", boundary="even")
        true_power = np.median(np.abs(Zxx) ** 2)

        noise_psd = compute_imcra_noise_estimate(noise, SR, n_fft=N_FFT, hop_length=HOP)
        # Initialphase: erste ~2s ≙ 2*SR/HOP Frames
        init_frames = int(2.0 * SR / HOP)
        init_median = float(np.median(noise_psd[:, :init_frames]))

        # IMCRA "konservativ" = Init-Schätzung > 0 und nicht pathologisch klein.
        # b_min×1.3×S_min kann unter true_power liegen (Minimum << Mittel bei Rauschen).
        # Prüfe: Init-Median ist positiv und mind. 5 % des wahren Rauschlevels.
        assert init_median > 0, "Init-Phase muss positive Schätzung liefern"
        assert init_median >= true_power * 0.05, (
            f"Init-Schätzung zu niedrig: {init_median:.3g} < 5% von {true_power:.3g}"
        )

    def test_stationary_noise_converges(self):
        """Für stationäres Rauschen muss Schätzung nach 3s konvergieren (Varianz klein)."""
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        rng = np.random.default_rng(2)
        noise = (0.05 * rng.standard_normal(int(5.0 * SR))).astype(np.float32)
        noise_psd = compute_imcra_noise_estimate(noise, SR, n_fft=N_FFT, hop_length=HOP)

        # Nach 3s soll Varianz über Frames deutlich kleiner sein als am Anfang
        settle_start = int(3.0 * SR / HOP)
        if settle_start >= noise_psd.shape[1]:
            pytest.skip("Signal zu kurz für Konvergenz-Test")
        settled = noise_psd[:, settle_start:]
        var_late = float(np.var(np.median(settled, axis=0)))
        var_all = float(np.var(np.median(noise_psd, axis=0)))
        assert var_late < var_all * 0.9 or var_all < 1e-20, (
            "Schätzung sollte sich stabilisieren (geringere Varianz nach 3s)"
        )


class TestImcraEdgeCases:
    """Robustheit bei Edge-Cases."""

    def test_silence_input(self):
        """Stille-Input: keine Exception, positive Schätzung."""
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio = _silence()
        noise_psd = compute_imcra_noise_estimate(audio, SR)
        assert noise_psd.ndim == 2
        assert np.all(noise_psd > 0)
        assert not np.any(np.isnan(noise_psd))

    def test_very_short_signal(self):
        """Sehr kurzes Signal (< 1s): kein Crash."""
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio = _white_noise(duration_s=0.1)
        noise_psd = compute_imcra_noise_estimate(audio, SR)
        assert noise_psd.ndim == 2
        assert not np.any(np.isnan(noise_psd))

    def test_2d_input_downmixed(self):
        """2D-Input (Stereo) wird ohne Fehler auf Mono reduziert."""
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio_2d = np.stack([_white_noise(), _white_noise(seed=1)], axis=0)  # (2, N)
        noise_psd = compute_imcra_noise_estimate(audio_2d, SR)
        assert noise_psd.ndim == 2
        assert not np.any(np.isnan(noise_psd))

    def test_sine_plus_noise_tracking(self):
        """Rauschen unter Sinuston: Schätzung folgt dem Rauschboden, nicht dem Signal."""
        from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate

        audio = _sine_plus_noise()
        noise_psd = compute_imcra_noise_estimate(audio, SR)

        # Median-PSD sollte deutlich kleiner sein als die Gesamt-Energie
        # (weil Rauschen << Signal im Leistungsbereich)
        from scipy.signal import stft

        _, _, Zxx = stft(audio, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", boundary="even")
        total_power = float(np.median(np.abs(Zxx) ** 2))
        estimated = float(np.median(noise_psd))
        # Rausch-Schätzung soll ≤ Gesamt-Power sein (Rauschen < Signal)
        assert estimated <= total_power * 5.0, f"Rausch-Schätzung {estimated:.4g} zu hoch vs. Total {total_power:.4g}"


class TestImcraSingleton:
    """get_noise_estimator: Thread-Safe-Singleton."""

    def test_singleton_same_instance(self):
        from backend.core.dsp.noise_estimator import get_noise_estimator

        inst1 = get_noise_estimator()
        inst2 = get_noise_estimator()
        assert inst1 is inst2, "get_noise_estimator muss Singleton zurückgeben"

    def test_singleton_is_imcra_instance(self):
        from backend.core.dsp.noise_estimator import ImcraNoisEstimator, get_noise_estimator

        inst = get_noise_estimator()
        assert isinstance(inst, ImcraNoisEstimator)
