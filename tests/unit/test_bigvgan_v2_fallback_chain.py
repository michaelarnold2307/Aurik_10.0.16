import pytest

"""Unit tests for BigVGAN-v2 productive fallback chain."""

import types

import numpy as np


def _audio(sr: int = 48000, duration: float = 0.25) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_bigvgan_fallback_prefers_loaded_vocos(monkeypatch):
    from plugins.bigvgan_v2_plugin import BigVGANv2Plugin

    class _FakeVocos:
        model_loaded = True

        @staticmethod
        def vocode(audio: np.ndarray, sr: int, mode: str = "studio2026"):
            return types.SimpleNamespace(audio=(audio * 0.5).astype(np.float32), model_used="vocos_onnx")

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.vocos_plugin",
        types.SimpleNamespace(get_vocos_plugin=lambda: _FakeVocos()),
    )

    plugin = BigVGANv2Plugin()
    audio = _audio()
    out, model, confidence = plugin._synthesize_fallback_chain(audio, 48000)  # pylint: disable=protected-access
    assert model == "vocos_fallback"
    assert confidence >= 0.80
    np.testing.assert_allclose(out, audio * 0.5, atol=1e-7)


def test_bigvgan_fallback_rejects_silent_vocos(monkeypatch):
    from plugins.bigvgan_v2_plugin import BigVGANv2Plugin

    class _SilentVocos:
        model_loaded = True

        @staticmethod
        def vocode(audio: np.ndarray, sr: int, mode: str = "studio2026"):
            return types.SimpleNamespace(audio=np.zeros_like(audio), model_used="vocos_onnx")

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.vocos_plugin",
        types.SimpleNamespace(get_vocos_plugin=lambda: _SilentVocos()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.hifigan_plugin",
        types.SimpleNamespace(get_hifigan_plugin=lambda: types.SimpleNamespace(_session=None)),
    )

    plugin = BigVGANv2Plugin()
    audio = _audio()
    out, model, confidence = plugin._synthesize_fallback_chain(audio, 48000)  # pylint: disable=protected-access
    assert model == "phase_coherent_istft_fallback"
    assert confidence >= 0.60
    assert out.shape == audio.shape
    assert np.max(np.abs(out)) > 0.05


def test_phase_coherent_istft_fallback_preserves_length_and_peak():
    from plugins.bigvgan_v2_plugin import BigVGANv2Plugin

    plugin = BigVGANv2Plugin()
    audio = _audio(duration=0.5)
    out, model, confidence = plugin._synthesize_phase_coherent_istft_fallback(audio, 48000)  # pylint: disable=protected-access
    assert model == "phase_coherent_istft_fallback"
    assert confidence >= 0.60
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
    assert np.sqrt(np.mean(out.astype(np.float64) ** 2)) > 0.05
