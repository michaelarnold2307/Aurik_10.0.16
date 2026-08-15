import numpy as np
import pytest

from plugins.deepfilternet_v3_ii_plugin import _DF_ORDER, DeepFilterNetV3IIPlugin


@pytest.mark.unit
def test_deepfilternet_v3_ii_plugin_aurik90():
    np.random.randn(16000).astype(np.float32)
    plugin = DeepFilterNetV3IIPlugin()
    assert isinstance(plugin, DeepFilterNetV3IIPlugin)


def test_apply_df_filter_vectorized_correctness():
    """Verify vectorized _apply_df_filter matches scalar reference implementation."""
    rng = np.random.RandomState(42)
    S, n_bins = 50, 96
    spec_cx = (rng.randn(481, S) + 1j * rng.randn(481, S)).astype(np.complex64)
    coefs = rng.randn(S, n_bins, _DF_ORDER).astype(np.float32)
    alpha = rng.rand(1, S, 1).astype(np.float32)

    # Scalar reference implementation (old code)
    result_ref = spec_cx.copy()
    for t in range(S):
        for b in range(n_bins):
            acc = complex(0.0)
            for k in range(_DF_ORDER):
                t_past = max(0, t - k)
                c = coefs[t, b, k]
                acc += c * spec_cx[b, t_past]
            blend = float(alpha[0, t, 0])
            result_ref[b, t] = blend * acc + (1 - blend) * spec_cx[b, t]

    # Vectorized implementation
    plugin = DeepFilterNetV3IIPlugin()
    result_vec = plugin._apply_df_filter(spec_cx.copy(), coefs, alpha)

    # Check numerical equivalence (allow small float tolerance)
    np.testing.assert_allclose(
        result_vec[:n_bins, :],
        result_ref[:n_bins, :],
        rtol=1e-4,
        atol=1e-6,
        err_msg="Vectorized _apply_df_filter deviates from scalar reference",
    )
    # Bins above n_bins must be unchanged
    np.testing.assert_array_equal(result_vec[n_bins:, :], spec_cx[n_bins:, :])


def test_compute_features_shape():
    """Verify _compute_features returns correct shapes for typical audio."""
    plugin = DeepFilterNetV3IIPlugin()
    mono = np.random.randn(48000).astype(np.float32)  # 1 second @ 48kHz
    feat_erb, feat_spec, spec_cx = plugin._compute_features(mono)
    assert feat_erb.ndim == 4 and feat_erb.shape[0] == 1 and feat_erb.shape[1] == 1
    assert feat_erb.shape[3] == 32  # n_erb
    assert feat_spec.ndim == 4 and feat_spec.shape[1] == 2
    assert feat_spec.shape[3] == 96  # DF_BINS
    assert spec_cx.shape[0] == 481  # N_FFT//2+1
    # S dimension must match
    assert feat_erb.shape[2] == spec_cx.shape[1]
    assert feat_spec.shape[2] == spec_cx.shape[1]


def test_compute_features_nan_free():
    """Verify _compute_features produces no NaN/Inf."""
    plugin = DeepFilterNetV3IIPlugin()
    mono = np.random.randn(96000).astype(np.float32)  # 2s
    feat_erb, feat_spec, spec_cx = plugin._compute_features(mono)
    assert np.all(np.isfinite(feat_erb))
    assert np.all(np.isfinite(feat_spec))
    assert np.all(np.isfinite(spec_cx))


def test_enhance_omlsa_fallback_short():
    """Verify OMLSA fallback works for short audio without ONNX models."""
    plugin = DeepFilterNetV3IIPlugin()
    plugin._enc = None  # Force OMLSA fallback
    audio = np.random.randn(48000).astype(np.float32) * 0.3  # 1s mono
    out = plugin.enhance(audio, sr=48000)
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


def test_enhance_stereo_omlsa_fallback():
    """Verify OMLSA fallback works for stereo audio."""
    plugin = DeepFilterNetV3IIPlugin()
    plugin._enc = None  # Force OMLSA fallback
    audio = np.random.randn(48000, 2).astype(np.float32) * 0.3
    out = plugin.enhance(audio, sr=48000)
    assert out.ndim == 2 and out.shape[1] == 2
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


def test_omlsa_fallback_uses_dry_signal_when_secondary_snr_is_high(monkeypatch):
    def _raise_primary(_mono, _sr):
        raise RuntimeError("omlsa failed")

    monkeypatch.setattr(DeepFilterNetV3IIPlugin, "_omlsa_primary_fallback", _raise_primary)
    monkeypatch.setattr(DeepFilterNetV3IIPlugin, "_estimate_input_snr_db", lambda _mono: 40.0)

    audio = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48_000, endpoint=False))).astype(np.float32)
    out = DeepFilterNetV3IIPlugin._omlsa_fallback(audio, 48_000)

    np.testing.assert_allclose(out, audio, atol=1e-6)


def test_omlsa_fallback_uses_spectral_gating_when_secondary_snr_is_low(monkeypatch):
    def _raise_primary(_mono, _sr):
        raise RuntimeError("omlsa failed")

    monkeypatch.setattr(DeepFilterNetV3IIPlugin, "_omlsa_primary_fallback", _raise_primary)
    monkeypatch.setattr(DeepFilterNetV3IIPlugin, "_estimate_input_snr_db", lambda _mono: 10.0)
    monkeypatch.setattr(
        DeepFilterNetV3IIPlugin,
        "_spectral_gating_fallback",
        lambda mono, _sr: np.zeros_like(mono, dtype=np.float32),
    )

    audio = np.random.randn(48_000).astype(np.float32) * 0.25
    out = DeepFilterNetV3IIPlugin._omlsa_fallback(audio, 48_000)

    assert out.shape == audio.shape
    assert np.all(out == 0.0)


# ── §0j (dsp.instructions.md): energy_bias wirkt auf die NR-Entscheidungsgrenze ──


def test_apply_energy_bias_to_gain_identity():
    """§0j: bias=0.0 → identische Gain-Maske (v10.15/v10.19-Default)."""
    gain = np.random.RandomState(0).uniform(0.0, 1.0, (64, 32)).astype(np.float32)
    out = DeepFilterNetV3IIPlugin._apply_energy_bias_to_gain(gain, 0.0)
    np.testing.assert_array_equal(out, gain)


def test_apply_energy_bias_to_gain_negative_bias_raises_floor():
    """§0j: negativer Bias hebt den Gain-Floor (Harmonik-Schutz), nie über 1."""
    gain = np.random.RandomState(1).uniform(0.0, 0.6, (64, 32)).astype(np.float32)
    out = DeepFilterNetV3IIPlugin._apply_energy_bias_to_gain(gain, -6.0)
    assert np.all(out >= gain - 1e-6)
    assert np.all(out <= 1.0)
    floor = 1.0 - 10.0 ** (-6.0 / 20.0)
    assert np.all(out >= floor - 1e-6)
    assert np.all(np.isfinite(out))


def test_apply_energy_bias_to_gain_positive_bias_suppresses():
    """§0j: positiver Bias senkt den Gain (aggressivere NR), nie unter 0."""
    gain = np.random.RandomState(2).uniform(0.4, 1.0, (64, 32)).astype(np.float32)
    out = DeepFilterNetV3IIPlugin._apply_energy_bias_to_gain(gain, 6.0)
    assert np.all(out <= gain + 1e-6)
    assert np.all(out >= 0.0)


def test_omlsa_fallback_honors_negative_energy_bias():
    """§0j: energy_bias_db=-9 dB im OMLSA-Fallback → weniger Suppression."""
    rng = np.random.RandomState(3)
    t = np.arange(48_000) / 48_000
    audio = (0.1 * rng.randn(48_000) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out_neutral = DeepFilterNetV3IIPlugin._omlsa_primary_fallback(audio, 48_000, energy_bias_db=0.0)
    out_biased = DeepFilterNetV3IIPlugin._omlsa_primary_fallback(audio, 48_000, energy_bias_db=-9.0)
    assert np.all(np.isfinite(out_biased))
    # Vorbestehendes scipy-ISTFT-Kantenartefakt (NOLA-Warnung) ausschließen:
    # nur das Signalinnere vergleichen.
    assert np.sqrt(np.mean(out_biased[2048:-2048] ** 2)) > np.sqrt(np.mean(out_neutral[2048:-2048] ** 2))


def test_enhance_propagates_bias_into_omlsa_fallback():
    """§0j: enhance() reicht energy_bias_db an den OMLSA-Fallback durch."""
    plugin = DeepFilterNetV3IIPlugin()
    plugin._enc = None  # Force OMLSA fallback
    audio = np.random.RandomState(4).randn(48_000).astype(np.float32) * 0.3
    out_neutral = plugin.enhance(audio, sr=48_000)
    out_biased = plugin.enhance(audio, sr=48_000, energy_bias_db=-9.0)
    assert np.all(np.isfinite(out_biased))
    assert np.sqrt(np.mean(out_biased[2048:-2048] ** 2)) >= np.sqrt(np.mean(out_neutral[2048:-2048] ** 2))
    assert plugin._current_energy_bias_db == pytest.approx(-9.0)
