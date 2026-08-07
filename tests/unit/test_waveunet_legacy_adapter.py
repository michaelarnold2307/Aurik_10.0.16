import pytest

"""Unit tests for WaveUNet legacy adapter routing."""

import numpy as np


def _audio(sr: int = 48000, duration: float = 0.2) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_waveunet_delegates_to_sota_router(monkeypatch):
    import backend.core.dsp.sota_vocal_model_router as router_mod
    from backend.core.dsp.sota_vocal_model_router import StemSeparationRouteResult
    from plugins.waveunet_plugin import WaveUNetPlugin

    class FakeRouter:
        def separate_vocal_instrumental(self, audio, sr, *, panns_singing=0.0, ctx=None):
            del sr, panns_singing, ctx
            return StemSeparationRouteResult(
                vocal=np.zeros_like(audio, dtype=np.float32),
                instrumental=np.asarray(audio, dtype=np.float32),
                success=True,
                model_used="demucs_v4",
                fallback_chain=["bs_roformer:missing"],
                metadata={"capability_status": "sota_fallback"},
            )

    monkeypatch.setattr(router_mod, "get_sota_vocal_model_router", lambda: FakeRouter())
    plugin = WaveUNetPlugin()
    vocal, instrumental = plugin.separate(_audio(), 48000)

    assert vocal.shape == instrumental.shape
    assert plugin.route_metadata["model_used"] == "demucs_v4"
    assert plugin.route_metadata["capability_status"] == "sota_fallback"


def test_waveunet_hpss_fallback_is_truthful(monkeypatch):
    import backend.core.dsp.sota_vocal_model_router as router_mod
    from plugins.waveunet_plugin import WaveUNetPlugin

    monkeypatch.setattr(
        router_mod, "get_sota_vocal_model_router", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    plugin = WaveUNetPlugin()
    vocal, instrumental = plugin.separate(_audio(), 48000)

    assert vocal.shape == instrumental.shape
    assert plugin.route_metadata["model_used"] == "hpss_dsp_fallback"
    assert plugin.route_metadata["capability_status"] == "dsp_fallback"
    assert np.isfinite(vocal).all()
    assert np.isfinite(instrumental).all()
