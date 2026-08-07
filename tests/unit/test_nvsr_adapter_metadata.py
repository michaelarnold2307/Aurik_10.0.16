import pytest

"""Unit tests for NVSR productive adapter metadata."""

import numpy as np


def _audio(sr: int = 48000, duration: float = 0.25) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (0.15 * np.sin(2.0 * np.pi * 5000.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_nvsr_reports_productive_dsp_capability_without_model(monkeypatch, tmp_path):
    import plugins.nvsr_plugin as nvsr_mod
    from plugins.nvsr_plugin import NvsrPlugin

    monkeypatch.setattr(nvsr_mod, "_NVSR_ONNX_PATH", tmp_path / "missing_nvsr.onnx")
    plugin = NvsrPlugin()
    result = plugin.process(_audio(), 48000, target_hz=16000.0, material_type="vinyl", strength=0.35)

    assert result["strategy"] == "dsp_sbr"
    assert result["capability_status"] == "dsp_productive"
    assert result["model_loaded"] is False
    assert plugin.route_metadata["strategy"] == "dsp_sbr"
    assert np.isfinite(result["audio"]).all()


def test_nvsr_passthrough_metadata_when_material_ceiling_blocks_extension():
    from plugins.nvsr_plugin import NvsrPlugin

    plugin = NvsrPlugin()
    result = plugin.process(_audio(), 48000, target_hz=16000.0, material_type="shellac", strength=0.8)

    assert result["strategy"] == "passthrough"
    assert result["strength"] == 0.0
    assert result["capability_status"] in {"dsp_productive", "sota_fallback", "sota_real"}
    assert plugin.route_metadata["strategy"] == "passthrough"
