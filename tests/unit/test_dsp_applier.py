"""
Tests für backend/core/regulator/_dsp_applier.py
EQ, Compressor, Limiter, Enhancer und apply_dsp_chain.
"""

import numpy as np
import pytest

from backend.core.regulator._dsp_applier import (
    apply_dsp_chain,
    compressor,
    enhancer,
    eq,
    limiter,
)

SR = 44100


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _sine(freq: float = 440.0, n: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(n: int = SR, seed: int = 0, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.uniform(-amplitude, amplitude, n)).astype(np.float32)


def _assert_valid(out: np.ndarray, audio: np.ndarray, tag: str = ""):
    assert len(out) == len(audio), f"{tag}: Länge falsch"
    assert not np.isnan(out).any(), f"{tag}: NaN im Ausgang"
    assert not np.isinf(out).any(), f"{tag}: Inf im Ausgang"
    assert out.dtype == audio.dtype, f"{tag}: Dtype-Mismatch"
    assert np.max(np.abs(out)) <= 1.0 + 1e-5, f"{tag}: Amplitude > 1.0"


# ===========================================================================
# EQ
# ===========================================================================
class TestEQ:
    def test_no_bands_passthrough(self):
        audio = _sine()
        out = eq(audio, SR, params={"bands": []})
        np.testing.assert_array_equal(out, audio)

    def test_empty_params_passthrough(self):
        audio = _sine()
        out = eq(audio, SR, params={})
        np.testing.assert_array_equal(out, audio)

    def test_boost_1khz_valid_output(self):
        audio = _white_noise()
        out = eq(audio, SR, params={"bands": [{"freq": 1000, "gain_db": 6.0, "q": 1.0}]})
        _assert_valid(out, audio, "EQ-Boost")

    def test_cut_200hz_valid_output(self):
        audio = _white_noise()
        out = eq(audio, SR, params={"bands": [{"freq": 200, "gain_db": -6.0, "q": 0.7}]})
        _assert_valid(out, audio, "EQ-Cut")

    def test_multiple_bands(self):
        audio = _white_noise(seed=3)
        out = eq(
            audio,
            SR,
            params={
                "bands": [
                    {"freq": 120, "gain_db": -3.0, "q": 0.7},
                    {"freq": 1000, "gain_db": 2.0, "q": 1.5},
                    {"freq": 8000, "gain_db": 4.0, "q": 2.0},
                ]
            },
        )
        _assert_valid(out, audio, "EQ-MultiBand")

    def test_zero_gain_passthrough(self):
        audio = _white_noise()
        out = eq(audio, SR, params={"bands": [{"freq": 1000, "gain_db": 0.0, "q": 1.0}]})
        np.testing.assert_array_equal(out, audio)

    def test_boost_increases_energy(self):
        """Ein Boost bei 1 kHz soll die Spektralenergie im 1-kHz-Band erhöhen."""
        audio = _white_noise(seed=5)
        out = eq(audio, SR, params={"bands": [{"freq": 1000, "gain_db": 6.0, "q": 1.0}]})
        # Prüfe Energie im Boost-Band (800-1200 Hz)
        fft_in = np.abs(np.fft.rfft(audio.astype(np.float64)))
        fft_out = np.abs(np.fft.rfft(out.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(audio), d=1 / SR)
        band_mask = (freqs >= 800) & (freqs <= 1200)
        energy_in = fft_in[band_mask].sum()
        energy_out = fft_out[band_mask].sum()
        assert energy_out >= energy_in * 1.5, (
            f"1-kHz-Boost hat Bandenergie nicht erhöht: {energy_out:.1f} < {energy_in * 1.5:.1f}"
        )

    def test_cut_reduces_energy(self):
        """Ein Cut soll die Spektralenergie im entsprechenden Band reduzieren."""
        audio = _white_noise(seed=7)
        out = eq(audio, SR, params={"bands": [{"freq": 1000, "gain_db": -12.0, "q": 1.0}]})
        fft_in = np.abs(np.fft.rfft(audio.astype(np.float64)))
        fft_out = np.abs(np.fft.rfft(out.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(audio), d=1 / SR)
        band_mask = (freqs >= 800) & (freqs <= 1200)
        energy_in = fft_in[band_mask].sum()
        energy_out = fft_out[band_mask].sum()
        assert energy_out <= energy_in * 0.8, (
            f"Cut hat Bandenergie nicht reduziert: {energy_out:.1f} > {energy_in * 0.8:.1f}"
        )

    def test_invalid_freq_ignored(self):
        """Frequenzen außerhalb [0, nyq] sollen ignoriert werden."""
        audio = _sine()
        out = eq(
            audio,
            SR,
            params={
                "bands": [
                    {"freq": -100, "gain_db": 6.0, "q": 1.0},
                    {"freq": SR, "gain_db": 6.0, "q": 1.0},  # = Nyquist
                ]
            },
        )
        np.testing.assert_array_equal(out, audio)

    def test_short_audio(self):
        audio = _sine(n=64)
        out = eq(audio, SR, params={"bands": [{"freq": 440, "gain_db": 3.0, "q": 1.0}]})
        _assert_valid(out, audio, "EQ-Short")


# ===========================================================================
# Compressor
# ===========================================================================
class TestCompressor:
    def _run(self, audio, **kw):
        return compressor(audio, SR, kw)

    def test_passthrough_below_threshold(self):
        """Stilles Signal weit unter Threshold → nahezu unverändert."""
        audio = _sine(amplitude=0.01)
        out = self._run(audio, threshold_db=-20.0, ratio=4.0)
        _assert_valid(out, audio, "Comp-Silent")
        np.testing.assert_allclose(out, audio, atol=1e-3)

    def test_reduces_loud_signal(self):
        """Lautes Signal über Threshold soll gedämpft sein."""
        audio = _sine(amplitude=0.9)
        out = self._run(audio, threshold_db=-20.0, ratio=8.0)
        rms_in = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        rms_out = np.sqrt(np.mean(out.astype(np.float64) ** 2))
        assert rms_out < rms_in, f"Kompressor hat nicht reduziert: {rms_out:.4f} >= {rms_in:.4f}"

    def test_output_in_range(self):
        audio = _white_noise(amplitude=0.95)
        out = self._run(audio, threshold_db=-30.0, ratio=10.0, makeup_db=6.0)
        _assert_valid(out, audio, "Comp-Range")

    def test_no_nan_inf(self):
        audio = _white_noise(seed=42)
        out = self._run(audio, threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=80.0)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_dtype_preserved(self):
        audio = _sine().astype(np.float32)
        out = self._run(audio, threshold_db=-20.0)
        assert out.dtype == np.float32

    def test_makeup_gain_increases_output(self):
        audio = _sine(amplitude=0.1)
        out_no_makeup = self._run(audio, threshold_db=-40.0, ratio=2.0, makeup_db=0.0)
        out_makeup = self._run(audio, threshold_db=-40.0, ratio=2.0, makeup_db=12.0)
        rms_no = np.sqrt(np.mean(out_no_makeup.astype(np.float64) ** 2))
        rms_mk = np.sqrt(np.mean(out_makeup.astype(np.float64) ** 2))
        assert rms_mk > rms_no, "Makeup-Gain hat keine Wirkung"

    @pytest.mark.parametrize("ratio", [1.0, 2.0, 4.0, 8.0, 100.0])
    def test_various_ratios(self, ratio):
        audio = _white_noise()
        out = self._run(audio, threshold_db=-20.0, ratio=ratio)
        _assert_valid(out, audio, f"Comp-Ratio-{ratio}")

    def test_silence_stays_silent(self):
        audio = np.zeros(SR, dtype=np.float32)
        out = self._run(audio, threshold_db=-60.0, ratio=4.0)
        assert np.max(np.abs(out)) < 1e-6


# ===========================================================================
# Limiter
# ===========================================================================
class TestLimiter:
    def _run(self, audio, **kw):
        return limiter(audio, SR, kw)

    def test_ceiling_enforced(self):
        """Ausgabe darf niemals über ceiling_db gehen."""
        audio = _sine(amplitude=0.95)
        ceiling_db = -6.0
        ceiling_lin = 10 ** (ceiling_db / 20.0)
        out = self._run(audio, ceiling_db=ceiling_db)
        assert np.max(np.abs(out)) <= ceiling_lin + 1e-4, (
            f"Ceiling verletzt: max={np.max(np.abs(out)):.4f} > {ceiling_lin:.4f}"
        )

    def test_output_in_range(self):
        audio = _white_noise(amplitude=1.0)
        out = self._run(audio, ceiling_db=-1.0)
        _assert_valid(out, audio, "Limiter-Range")

    def test_no_nan_inf(self):
        audio = _white_noise(amplitude=0.8)
        out = self._run(audio, ceiling_db=-3.0)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_quiet_signal_passthrough(self):
        """Sehr leises Signal wird nicht verändert."""
        audio = _sine(amplitude=0.001)
        out = self._run(audio, ceiling_db=-1.0)
        np.testing.assert_allclose(out, audio, atol=1e-3)

    def test_dtype_preserved(self):
        audio = _sine()
        out = self._run(audio, ceiling_db=-1.0)
        assert out.dtype == audio.dtype

    @pytest.mark.parametrize("ceiling_db", [-1.0, -3.0, -6.0, -12.0])
    def test_various_ceilings(self, ceiling_db):
        audio = _white_noise(amplitude=0.9)
        out = self._run(audio, ceiling_db=ceiling_db)
        ceiling_lin = 10 ** (ceiling_db / 20.0)
        assert np.max(np.abs(out)) <= ceiling_lin + 1e-4


# ===========================================================================
# Enhancer
# ===========================================================================
class TestEnhancer:
    def _run(self, audio, **kw):
        return enhancer(audio, SR, kw)

    def test_output_length(self):
        audio = _white_noise()
        out = self._run(audio)
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        audio = _white_noise()
        out = self._run(audio)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_output_in_range(self):
        audio = _white_noise(amplitude=0.7)
        out = self._run(audio, drive=1.0, mix=0.5)
        _assert_valid(out, audio, "Enhancer-Range")

    def test_zero_mix_passthrough(self):
        """mix=0.0 → Signal soll unverändert bleiben."""
        audio = _white_noise(seed=9)
        out = self._run(audio, mix=0.0, drive=0.3)
        np.testing.assert_allclose(out, audio, atol=1e-4)

    def test_dtype_preserved(self):
        audio = _white_noise()
        out = self._run(audio)
        assert out.dtype == audio.dtype

    def test_adds_harmonics(self):
        """
        Enhancer soll im Spektrum Obertöne hinzufügen.
        Sinuseingang (1 kHz): Nach dem Enhancer mehr Energie bei 2 kHz+ erwartet.
        """
        audio = _sine(freq=1000.0, amplitude=0.5)
        out_enh = self._run(audio, drive=1.0, mix=1.0, freq_hz=500.0)
        # FFT: Prüfe ob Energie bei 2 kHz und darüber gestiegen ist
        fft_in = np.abs(np.fft.rfft(audio.astype(np.float64)))
        fft_out = np.abs(np.fft.rfft(out_enh.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(audio), d=1 / SR)
        hf_in = fft_in[freqs > 2000].sum()
        hf_out = fft_out[freqs > 2000].sum()
        assert hf_out >= hf_in * 0.9, f"Enhancer fügte keine Harmonischen hinzu: {hf_out:.1f} < {hf_in:.1f}"

    @pytest.mark.parametrize("drive,mix", [(0.0, 0.3), (0.5, 0.2), (1.0, 0.5)])
    def test_various_params(self, drive, mix):
        audio = _white_noise(amplitude=0.5)
        out = self._run(audio, drive=drive, mix=mix)
        _assert_valid(out, audio, f"Enhancer-d{drive}-m{mix}")


# ===========================================================================
# apply_dsp_chain
# ===========================================================================
class TestApplyDSPChain:
    def test_empty_chain_passthrough(self):
        audio = _white_noise()
        out = apply_dsp_chain(audio, SR, chain=[])
        np.testing.assert_array_equal(out, audio)

    def test_single_effect(self):
        audio = _white_noise()
        chain = [{"type": "limiter", "params": {"ceiling_db": -3.0}}]
        out = apply_dsp_chain(audio, SR, chain)
        assert len(out) == len(audio)
        assert not np.isnan(out).any()

    def test_full_mastering_chain(self):
        """EQ → Compressor → Limiter"""
        audio = _white_noise(amplitude=0.7)
        chain = [
            {"type": "eq", "params": {"bands": [{"freq": 200, "gain_db": -3.0, "q": 0.7}]}},
            {"type": "compressor", "params": {"threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0}},
            {"type": "limiter", "params": {"ceiling_db": -1.0}},
        ]
        out = apply_dsp_chain(audio, SR, chain)
        _assert_valid(out, audio, "MasteringChain")

    def test_unknown_effect_ignored(self):
        """Unbekannter Effekttyp soll das Audio nicht verändern."""
        audio = _white_noise()
        out = apply_dsp_chain(audio, SR, chain=[{"type": "unknown_xyz", "params": {}}])
        np.testing.assert_array_equal(out, audio)

    def test_missing_type_ignored(self):
        audio = _white_noise()
        out = apply_dsp_chain(audio, SR, chain=[{"params": {}}])
        np.testing.assert_array_equal(out, audio)

    def test_chain_with_enhancer(self):
        audio = _white_noise(amplitude=0.4)
        chain = [
            {"type": "enhancer", "params": {"drive": 0.5, "mix": 0.3}},
            {"type": "limiter", "params": {"ceiling_db": -1.0}},
        ]
        out = apply_dsp_chain(audio, SR, chain)
        _assert_valid(out, audio, "Enhance+Limit")

    def test_output_ceiling_after_chain(self):
        """Kette endet mit Limiter → Ceiling muss gehalten werden."""
        audio = _white_noise(amplitude=1.0, seed=10)
        ceiling_db = -3.0
        ceiling_lin = 10 ** (ceiling_db / 20.0)
        chain = [
            {"type": "eq", "params": {"bands": [{"freq": 1000, "gain_db": 12.0, "q": 0.5}]}},
            {"type": "limiter", "params": {"ceiling_db": ceiling_db}},
        ]
        out = apply_dsp_chain(audio, SR, chain)
        assert np.max(np.abs(out)) <= ceiling_lin + 1e-4
