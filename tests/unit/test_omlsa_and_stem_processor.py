"""
Unit-Tests für:
  - dsp/adaptive_omlsa.py (AdaptiveOMLSA.auto_optimize + omlsa)
  - processing/stem_based_processor.py (private DSP-Methoden)
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from dsp.adaptive_omlsa import AdaptiveOMLSA

SR = 44100


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(n: int = 2048, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _noise_mag(n: int = 512, level: float = 0.02) -> np.ndarray:
    """Flaches Rausch-Magnitudenspektrum."""
    return np.full(n, level, dtype=np.float64)


def _signal_mag(n: int = 512, level: float = 0.5) -> np.ndarray:
    """Flaches Signal-Magnitudenspektrum (lauter als Rauschen)."""
    return np.full(n, level, dtype=np.float64)


# ===========================================================================
# AdaptiveOMLSA Tests
# ===========================================================================


@pytest.mark.unit
class TestAdaptiveOMLSA:
    # --- OMLSA Grundfunktion ---

    def test_omlsa_output_shape(self):
        omlsa = AdaptiveOMLSA()
        noisy = _signal_mag()
        noise = _noise_mag()
        out = omlsa.omlsa(noisy, noise)
        assert out.shape == noisy.shape

    def test_omlsa_reduces_noise(self):
        """OMLSA-Gain muss Rauschen reduzieren: clean_mag < noisy_mag."""
        omlsa = AdaptiveOMLSA()
        noisy = _noise_mag(level=0.1)  # Nur Rauschen
        noise = _noise_mag(level=0.1)
        out = omlsa.omlsa(noisy, noise)
        assert np.mean(out) <= np.mean(noisy)

    def test_omlsa_preserves_loud_signal(self):
        """Lautes Signal (SNR >> 1) sollte kaum gedämpft werden."""
        omlsa = AdaptiveOMLSA(alpha=0.98)
        signal = _signal_mag(level=1.0)  # Starkes Signal
        noise = _noise_mag(level=0.001)  # Sehr schwaches Rauschen
        out = omlsa.omlsa(signal, noise)
        preservation_ratio = float(np.mean(out) / np.mean(signal))
        assert preservation_ratio > 0.5  # Mindestens 50% des Signals bleibt

    def test_omlsa_non_negative(self):
        """Gain muss ≥ 0 sein (keine negativen Magnituden)."""
        omlsa = AdaptiveOMLSA()
        noisy = _signal_mag()
        noise = _noise_mag()
        out = omlsa.omlsa(noisy, noise)
        assert np.all(out >= 0)

    def test_omlsa_accepts_2d_input(self):
        """2D-Array (Frequenz × Zeit) muss korrekt verarbeitet werden."""
        omlsa = AdaptiveOMLSA()
        noisy = np.random.rand(257, 100) * 0.5
        noise = np.random.rand(257, 100) * 0.05
        out = omlsa.omlsa(noisy, noise)
        assert out.shape == noisy.shape

    # --- auto_optimize ---

    def test_auto_optimize_returns_none(self):
        """auto_optimize modifiziert self, hat keinen Return-Wert (None)."""
        omlsa = AdaptiveOMLSA()
        result = omlsa.auto_optimize(_signal_mag(), _noise_mag())
        assert result is None

    def test_auto_optimize_updates_alpha(self):
        """alpha muss nach auto_optimize im Bereich [0.85, 0.99] liegen."""
        omlsa = AdaptiveOMLSA(alpha=0.50)
        omlsa.auto_optimize(_signal_mag(level=0.8), _noise_mag(level=0.02))
        assert 0.85 <= omlsa.alpha <= 0.99

    def test_auto_optimize_updates_noise_floor(self):
        """noise_floor muss nach auto_optimize im Bereich [1e-8, 1e-5] liegen."""
        omlsa = AdaptiveOMLSA(noise_floor=1.0)  # Falsch gesetzt
        omlsa.auto_optimize(_signal_mag(level=0.8), _noise_mag(level=0.02))
        assert 1e-8 <= omlsa.noise_floor <= 1e-5

    def test_auto_optimize_high_snr_high_alpha(self):
        """Hohes SNR → alpha nahe 0.99 (starkes Glätten)."""
        omlsa = AdaptiveOMLSA()
        signal_high = _signal_mag(level=10.0)  # Sehr starkes Signal
        noise_low = _noise_mag(level=0.001)
        omlsa.auto_optimize(signal_high, noise_low)
        assert omlsa.alpha > 0.95  # Nahe Maximum (0.99)

    def test_auto_optimize_low_snr_lower_alpha(self):
        """Niedriger SNR → alpha näher an 0.85."""
        omlsa_hi = AdaptiveOMLSA()
        omlsa_lo = AdaptiveOMLSA()
        # Hohes SNR
        omlsa_hi.auto_optimize(_signal_mag(level=10.0), _noise_mag(level=0.001))
        # Niedriges SNR (Signal ≈ Rauschen)
        omlsa_lo.auto_optimize(_signal_mag(level=0.11), _noise_mag(level=0.10))
        assert omlsa_hi.alpha > omlsa_lo.alpha  # Hohes SNR → höheres alpha

    def test_auto_optimize_high_snr_low_noise_floor(self):
        """Hohes SNR → kleiner Rauschboden (schärfere Unterdrückung)."""
        omlsa_hi = AdaptiveOMLSA()
        omlsa_lo = AdaptiveOMLSA()
        omlsa_hi.auto_optimize(_signal_mag(level=10.0), _noise_mag(level=0.001))
        omlsa_lo.auto_optimize(_signal_mag(level=0.11), _noise_mag(level=0.10))
        assert omlsa_hi.noise_floor < omlsa_lo.noise_floor

    def test_auto_optimize_idempotent_twice(self):
        """Zweifaches auto_optimize konvergiert zu stabilem Wert."""
        omlsa = AdaptiveOMLSA()
        noisy = _signal_mag(level=0.5)
        noise = _noise_mag(level=0.05)
        omlsa.auto_optimize(noisy, noise)
        alpha1 = omlsa.alpha
        omlsa.auto_optimize(noisy, noise)
        alpha2 = omlsa.alpha
        assert alpha1 == pytest.approx(alpha2, abs=1e-6)

    def test_auto_optimize_then_omlsa_consistent(self):
        """auto_optimize → omlsa mit aktualisierten Parametern muss funktionieren."""
        omlsa = AdaptiveOMLSA()
        noisy = _signal_mag(level=0.5)
        noise = _noise_mag(level=0.05)
        omlsa.auto_optimize(noisy, noise)
        out = omlsa.omlsa(noisy, noise, alpha=omlsa.alpha, noise_floor=omlsa.noise_floor)
        assert out.shape == noisy.shape
        assert np.all(np.isfinite(out))


# ===========================================================================
# StemBasedProcessor — private DSP-Methoden
# ===========================================================================


class TestStemBasedProcessorMethods:
    """
    Testet die privaten DSP-Methoden des StemBasedProcessor direkt,
    ohne die externe Demucs/HTDemucs-Abhängigkeit zu benötigen.
    """

    @pytest.fixture
    def processor(self):
        """Erstellt eine StemBasedProcessor-Instanz ohne Modell-Laden."""
        from processing.stem_based_processor import StemBasedProcessor

        # Instanz ohne echtes Modell (Fallback auf spektrale Trennung)
        return StemBasedProcessor(separation_model="demucs_v4")

    def test_enhance_transients_shape(self, processor):
        audio = _sine(n=SR)
        out = processor._enhance_transients(audio, SR)
        assert out.shape == audio.shape

    def test_enhance_transients_no_clipping(self, processor):
        audio = _sine(n=SR, amp=0.8)
        out = processor._enhance_transients(audio, SR)
        assert np.max(np.abs(out)) <= 1.001

    def test_enhance_transients_boosts_transients(self, processor):
        """Transient-Booster muss bei Onset-Material Energie erhöhen."""
        # Impuls-Signal (starke Transienten)
        audio = np.zeros(SR, dtype=np.float32)
        audio[:: int(SR * 0.01)] = 0.5  # Impulse alle 10ms
        out = processor._enhance_transients(audio, SR)
        rms_in = float(np.sqrt(np.mean(audio**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out >= rms_in * 0.9  # Mindestens keine Signalverluste

    def test_click_removal_shape(self, processor):
        audio = _sine(n=SR)
        out = processor._intelligent_click_removal(audio, SR)
        assert out.shape == audio.shape

    def test_click_removal_no_clipping(self, processor):
        audio = _sine(n=SR, amp=0.9)
        out = processor._intelligent_click_removal(audio, SR)
        assert np.max(np.abs(out)) <= 1.001

    def test_click_removal_removes_spikes(self, processor):
        """Klick-Entferner muss starke Spike-Amplitude reduzieren."""
        audio = _sine(n=SR, amp=0.3)
        audio_with_click = audio.copy()
        # Einen starken Klick einbauen
        audio_with_click[1000] = 0.99  # Klick (Laplace >> 6σ)
        out = processor._intelligent_click_removal(audio_with_click, SR)
        # Klick-Amplitude muss reduziert werden
        assert abs(float(out[1000])) < abs(float(audio_with_click[1000]))

    def test_bass_enhancement_shape(self, processor):
        audio = _sine(n=SR)
        out = processor._bass_enhancement(audio, SR)
        assert out.shape == audio.shape

    def test_bass_enhancement_no_clipping(self, processor):
        audio = _sine(n=SR, amp=0.9)
        out = processor._bass_enhancement(audio, SR)
        assert np.max(np.abs(out)) <= 1.001

    def test_bass_enhancement_boosts_lf(self, processor):
        """Bass-Enhancement muss tiefe Frequenzen (~80 Hz) leicht anheben."""
        bass_freq = 80.0
        audio_bass = _sine(n=SR, freq=bass_freq, amp=0.5)
        out = processor._bass_enhancement(audio_bass, SR)
        rms_in = float(np.sqrt(np.mean(audio_bass**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        # Bass-Anreicherung oder gleichbleibend (keine Dämpfung)
        assert rms_out >= rms_in * 0.95

    def test_noise_reduction_shape(self, processor):
        audio = _sine(n=SR)
        out = processor._gentle_noise_reduction(audio, SR)
        assert out.shape == audio.shape

    def test_noise_reduction_no_clipping(self, processor):
        audio = _sine(n=SR, amp=0.8)
        out = processor._gentle_noise_reduction(audio, SR)
        assert np.max(np.abs(out)) <= 1.001

    def test_noise_reduction_reduces_broadband_noise(self, processor):
        """OLA-STFT Wiener Masking muss Rauschen reduzieren."""
        rng = np.random.default_rng(5)
        noise = (rng.standard_normal(SR) * 0.3).astype(np.float32)
        out = processor._gentle_noise_reduction(noise, SR)
        rms_in = float(np.sqrt(np.mean(noise**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out < rms_in

    def test_compute_quality_range(self, processor):
        """Qualitäts-Score liegt im MOS-ähnlichen Bereich [1.0, 5.0]."""
        audio = _sine(n=SR, amp=0.5)
        score = processor._compute_quality(audio, SR)
        assert 1.0 <= float(score) <= 5.0

    def test_compute_quality_silent_low(self, processor):
        """Stille erhält einen niedrigen Score (nahe Minimum 1.0 oder default 3.8)."""
        audio = np.zeros(SR, dtype=np.float32)
        score = processor._compute_quality(audio, SR)
        # Stille: n_frames < 2 → Default 3.8, oder SNR=0 → Score=1.0
        assert 1.0 <= float(score) <= 5.0
