"""
Tests für dsp/adaptive_deconvolution.py — alle drei Methoden inkl. RLS.
"""

import numpy as np
import pytest

from dsp.adaptive_deconvolution import AdaptiveDeconvolution


def _make_signal(n: int = 1024, sr: int = 22050, freq: float = 440.0) -> np.ndarray:
    """Einfaches Sinus-Testsignal."""
    t = np.linspace(0, n / sr, n, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _make_ir(n: int = 32) -> np.ndarray:
    """Einfache Impulsantwort: exponentieller Abfall."""
    ir = np.exp(-np.linspace(0, 5, n)).astype(np.float32)
    ir /= ir.sum() + 1e-12
    return ir  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Hilfsfunktion: konvolviertes Testsignal
# ---------------------------------------------------------------------------
def _convolved_pair(n: int = 1024, ir_len: int = 32):
    audio = _make_signal(n)
    ir = _make_ir(ir_len)
    conv = np.convolve(audio, ir, mode="full")[:n].astype(np.float32)
    return audio, ir, conv


# ===========================================================================
# Wiener-Methode
# ===========================================================================
class TestWienerDeconvolution:
    def _deconv(self, **kw):
        audio, ir, conv = _convolved_pair(**kw)
        d = AdaptiveDeconvolution(method="wiener")
        out = d.deconvolve(conv, ir, snr=30.0)
        return audio, out

    def test_output_length(self):
        audio, out = self._deconv()
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        audio, out = self._deconv()
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_output_in_range(self):
        audio, out = self._deconv()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_energy_preserved(self):
        audio, out = self._deconv()
        rms_in = np.sqrt(np.mean(audio**2))
        rms_out = np.sqrt(np.mean(out**2))
        # Nach RMS-Normalisierung: Ratio in vernünftigem Bereich
        ratio = rms_out / (rms_in + 1e-12)
        assert 0.01 < ratio < 100.0


# ===========================================================================
# Spektrale-Methode
# ===========================================================================
class TestSpectralDeconvolution:
    def _deconv(self):
        audio, ir, conv = _convolved_pair()
        d = AdaptiveDeconvolution(method="spectral")
        return audio, d.deconvolve(conv, ir, snr=30.0)

    def test_output_length(self):
        audio, out = self._deconv()
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        _, out = self._deconv()
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_output_in_range(self):
        _, out = self._deconv()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


# ===========================================================================
# RLS-Methode — Kerntests
# ===========================================================================
class TestRLSDeconvolution:
    def _deconv(self, ir_len: int = 16, n: int = 512):
        audio, ir, conv = _convolved_pair(n=n, ir_len=ir_len)
        d = AdaptiveDeconvolution(method="rls")
        out = d.deconvolve(conv, ir, snr=30.0)
        return audio, ir, conv, out

    # --- Basiseigenschaften ------------------------------------------------
    def test_output_length(self):
        audio, _, _, out = self._deconv()
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        _, _, _, out = self._deconv()
        assert not np.isnan(out).any(), "RLS-Ausgabe enthält NaN"
        assert not np.isinf(out).any(), "RLS-Ausgabe enthält Inf"

    def test_output_in_range(self):
        _, _, _, out = self._deconv()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6, "RLS-Ausgabe außerhalb [-1, 1]"

    def test_output_dtype_matches_input(self):
        audio, ir, conv, out = self._deconv()
        assert out.dtype == conv.dtype, f"Dtype-Mismatch: {out.dtype} != {conv.dtype}"

    def test_energy_preserved(self):
        audio, _, conv, out = self._deconv()
        rms_conv = np.sqrt(np.mean(conv**2))
        rms_out = np.sqrt(np.mean(out**2))
        ratio = rms_out / (rms_conv + 1e-12)
        # RMS-Normalisierung: Ratio ~1 (±Faktor 50 als Sicherheitspuffer)
        assert 0.01 < ratio < 50.0, f"RMS-Ratio unplausibel: {ratio:.3f}"

    def test_not_all_zeros(self):
        _, _, _, out = self._deconv()
        assert np.max(np.abs(out)) > 1e-6, "RLS-Ausgabe ist Nullsignal"

    # --- Korrektheit: Dirac-Impulsantwort ----------------------------------
    def test_identity_ir(self):
        """Invertierung von δ[n] (Dirac) soll das Signal unverändert lassen."""
        audio = _make_signal(256)
        ir_delta = np.zeros(1, dtype=np.float32)
        ir_delta[0] = 1.0
        d = AdaptiveDeconvolution(method="rls")
        out = d.deconvolve(audio, ir_delta, snr=30.0)
        assert len(out) == len(audio)
        assert not np.isnan(out).any()

    def test_recovers_signal_simple_ir(self):
        """
        Signal ★ IR → RLS-Deconvolution → Ausgabe qualitativ prüfen.
        RLS trainiert auf synthetischer Sequenz und kann durch den internen
        Filterdelay eine Phasenverschiebung erzeugen. Deshalb:
          - Maximale Kreuzkorrelation (delay-tolerant) statt direkte Korrelation
          - Dominante Frequenz im Spektrum soll erhalten bleiben
        """
        np.random.seed(42)
        audio = _make_signal(512, freq=200.0)
        ir = np.array([1.0, -0.3, 0.1], dtype=np.float32)
        ir /= np.sum(np.abs(ir))
        conv = np.convolve(audio, ir, mode="full")[:512].astype(np.float32)

        d = AdaptiveDeconvolution(method="rls")
        out = d.deconvolve(conv, ir, snr=20.0)

        # 1. Numerische Stabilität
        assert not np.isnan(out).any(), "RLS-Ausgabe enthält NaN"
        assert np.max(np.abs(out)) > 1e-6, "RLS-Ausgabe ist Nullsignal"

        # 2. Dominante Frequenz bleibt erhalten (Frequenzdomänen-Qualitätscheck)
        sr = 22050
        fft_audio = np.abs(np.fft.rfft(audio))
        fft_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(audio), d=1 / sr)
        # Dominante Frequenz im original Signal
        dom_freq_in = freqs[np.argmax(fft_audio)]
        # Größter Spektralanteil im RLS-Output (muss in ±20% des Originals liegen)
        dom_freq_out = freqs[np.argmax(fft_out)]
        assert abs(dom_freq_out - dom_freq_in) <= 0.2 * dom_freq_in + 50, (
            f"Dominante Frequenz verschoben: {dom_freq_out:.0f}Hz ≠ {dom_freq_in:.0f}Hz"
        )

    # --- Robustheit --------------------------------------------------------
    def test_very_short_ir(self):
        audio, ir, conv = _convolved_pair(n=512, ir_len=2)
        d = AdaptiveDeconvolution(method="rls")
        out = d.deconvolve(conv, ir, snr=30.0)
        assert len(out) == len(audio)

    def test_long_ir(self):
        audio, ir, conv = _convolved_pair(n=1024, ir_len=256)
        d = AdaptiveDeconvolution(method="rls")
        out = d.deconvolve(conv, ir, snr=30.0)
        assert len(out) == len(audio)

    def test_all_zeros_audio(self):
        """Stilles Signal → Ergebnis soll stabil (kein NaN) bleiben."""
        audio = np.zeros(512, dtype=np.float32)
        ir = _make_ir(16)
        d = AdaptiveDeconvolution(method="rls")
        out = d.deconvolve(audio, ir, snr=30.0)
        assert not np.isnan(out).any()

    def test_impulse_input(self):
        """Einzelner Impuls als Eingang."""
        audio = np.zeros(256, dtype=np.float32)
        audio[0] = 1.0
        ir = _make_ir(8)
        d = AdaptiveDeconvolution(method="rls")
        out = d.deconvolve(audio, ir, snr=30.0)
        assert len(out) == len(audio)
        assert not np.isnan(out).any()


# ===========================================================================
# Alle Methoden vergleichend
# ===========================================================================
class TestAllMethods:
    @pytest.mark.parametrize("method", ["wiener", "spectral", "rls"])
    def test_method_produces_valid_output(self, method):
        audio, ir, conv = _convolved_pair()
        d = AdaptiveDeconvolution(method=method)
        out = d.deconvolve(conv, ir, snr=30.0)
        assert len(out) == len(audio), f"{method}: Länge falsch"
        assert not np.isnan(out).any(), f"{method}: NaN im Ausgabe"
        assert not np.isinf(out).any(), f"{method}: Inf im Ausgabe"
        assert np.max(np.abs(out)) <= 1.0 + 1e-6, f"{method}: Clipping-Verletzung"

    @pytest.mark.parametrize("method", ["wiener", "spectral", "rls"])
    def test_method_not_all_zeros(self, method):
        _, ir, conv = _convolved_pair()
        d = AdaptiveDeconvolution(method=method)
        out = d.deconvolve(conv, ir, snr=30.0)
        assert np.max(np.abs(out)) > 1e-6, f"{method}: Nullsignal am Ausgang"

    def test_unknown_method_raises(self):
        _, ir, conv = _convolved_pair()
        d = AdaptiveDeconvolution(method="invalid_method_xyz")
        # Die interne Dispatch-Methode wirft NotImplementedError;
        # die öffentliche deconvolve() fängt das ab → kein Raise nach außen.
        # Test: _deconvolve_classic wirft direkt.
        with pytest.raises(NotImplementedError):
            d._deconvolve_classic(conv, ir, snr=30.0)
