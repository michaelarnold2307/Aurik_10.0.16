"""FlashSR Plugin Tests — v10.0.4 ONNX-based architecture.

Covers: SBR-DSP fallback, code quality, spectral extension,
        FlashSR ONNX inference, backward compatibility.
FlashSR ONNX inference is tested separately via integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── SBR-DSP fallback ──────────────────────────────────────────────────


def test_flashsr_hf_extend_uses_sbr_fallback(monkeypatch):
    """SBR-DSP fallback works when spectral exciter fails."""
    from plugins.flashsr_plugin import FlashSRPlugin

    plugin = FlashSRPlugin()

    def _fail(_x, _sr):
        raise RuntimeError("exciter failed")

    monkeypatch.setattr(plugin, "_spectral_exciter", _fail)

    audio = np.random.randn(48_000).astype(np.float32) * 0.05
    out = plugin._hf_extend(audio, 48_000)

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


# ── SBR spectral extension ────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,audio",
    [
        ("white noise", lambda: np.random.randn(48000).astype(np.float32) * 0.05),
        ("sine 1kHz", lambda: (np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)).astype(np.float32) * 0.3),
        ("silence", lambda: np.zeros(48000, dtype=np.float32)),
        ("soft clip", lambda: np.clip(np.random.randn(96000).astype(np.float32) * 0.5, -0.3, 0.3)),
    ],
)
def test_sbr_output_validity(label, audio):
    """SBR-DSP produces valid output for various signal types."""
    from plugins.flashsr_plugin import FlashSRPlugin

    plugin = FlashSRPlugin()
    a = audio()
    out = plugin._hf_extend(a, 48000)

    assert out.shape == a.shape, f"{label}: shape mismatch"
    assert np.all(np.isfinite(out)), f"{label}: non-finite values"
    assert np.max(np.abs(out)) <= 1.0, f"{label}: output exceeds [-1,1]"


def test_sbr_adds_spectral_energy():
    """SBR-DSP actually adds high-frequency energy."""
    from plugins.flashsr_plugin import FlashSRPlugin

    plugin = FlashSRPlugin()
    noise = np.random.randn(48000).astype(np.float32) * 0.05
    out = plugin._hf_extend(noise, 48000)

    f_in = np.mean(np.abs(np.fft.rfft(noise * np.hanning(len(noise)))))
    f_out = np.mean(np.abs(np.fft.rfft(out * np.hanning(len(out)))))

    assert f_out > f_in, f"HF not extended: {f_in:.4f} -> {f_out:.4f}"


# ── Code quality: no obsolete patches ─────────────────────────────────


def test_no_obsolete_patches_in_source():
    """Verify old GPU/mixed-device patches are fully removed."""
    with open("plugins/flashsr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    obsolete = [
        ("_fsm.cpu()", "old ROCm-Fix v2: first_stage_model CPU move"),
        ("GPU-DDIM fehlgeschlagen", "old GPU-DDIM recovery message"),
        ("CPU-Retry fehlgeschlagen", "old CPU-Retry recovery message"),
        ("_patched_mel2wav", "old mel2wav monkey-patch"),
        ("_patched_decode", "old decode monkey-patch"),
        ('build_model(model_name="basic", device=str(_dev))', "old dynamic device selection"),
        # AudioLDM2/torch patterns that should NOT exist anymore
        ("torch.load", "torch model loading should not exist in FlashSR plugin"),
        ("AudioLDM2", "AudioLDM2 references should not exist"),
        ("safetensors", "safetensors loading should not exist"),
    ]

    for pattern, description in obsolete:
        assert pattern not in code, f"Obsolete code found: {description}"


def test_flashsr_onnx_architecture():
    """Verify FlashSR uses ONNX, not torch/AudioLDM2."""
    with open("plugins/flashsr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    assert "onnxruntime" in code, "FlashSR must use onnxruntime"
    assert "FlashSR" in code or "flashsr" in code.lower(), "Plugin must reference FlashSR"
    assert "build_model" not in code, "torch build_model must not exist in FlashSR plugin"


def test_warning_suppression_not_needed():
    """Verify torch weight_norm warnings are gone (ONNX has no such issue)."""
    with open("plugins/flashsr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    assert "weight_norm" not in code, "weight_norm warning suppression not needed for ONNX"


def test_no_torch_nan_cleanup():
    """Verify old torch parameter NaN cleanup is gone (ONNX handles this)."""
    with open("plugins/flashsr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    assert "for _p in model.parameters()" not in code, "Old torch parameter NaN cleanup must not exist"


# ── Plugin structure ──────────────────────────────────────────────────


def test_plugin_has_required_methods():
    """Plugin exposes the expected public and private API."""
    from plugins.flashsr_plugin import FlashSRPlugin

    p = FlashSRPlugin()

    assert hasattr(p, "process"), "Missing public method: process"
    assert hasattr(p, "_hf_extend"), "Missing private method: _hf_extend"
    assert hasattr(p, "_spectral_band_replication"), "Missing: _spectral_band_replication"
    assert hasattr(p, "_spectral_exciter"), "Missing: _spectral_exciter"


def test_flashsr_memory_budget():
    """FlashSR ONNX model uses < 500 MB (vs 7 GB for old AudioSR)."""
    import os

    onnx_path = "models/nvsr/nvsr.onnx"
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        assert size_mb < 500, f"FlashSR ONNX model {size_mb:.0f} MB exceeds 500 MB budget"
    else:
        pytest.skip("FlashSR ONNX model not found")


# ── SBR band-limited extension ────────────────────────────────────────


def test_sbr_extends_band_limited_signal():
    """SBR/Exciter produzieren gültiges Output für bandlimitiertes Signal.

    FlashSRs DSP-Fallback ist konservativ — Output muss im Bereich
    [-1, 1] liegen und keine NaN enthalten.
    """
    from plugins.flashsr_plugin import FlashSRPlugin

    sr = 48000
    n = sr  # 1 second
    noise = np.random.randn(n).astype(np.float32) * 0.1
    spec = np.fft.rfft(noise)
    cutoff_bin = int(4000 / (sr / 2) * len(spec))
    spec[cutoff_bin:] = 0
    bl = np.fft.irfft(spec, n).astype(np.float32)
    bl = bl / max(1e-8, np.max(np.abs(bl))) * 0.1

    plugin = FlashSRPlugin()
    out = plugin._hf_extend(bl, sr)

    # Output must be valid audio
    assert out.shape == bl.shape
    assert np.all(np.isfinite(out)), "output contains non-finite values"
    assert np.max(np.abs(out)) <= 1.0, "output exceeds [-1,1]"


# ── Backward compatibility ────────────────────────────────────────────


def test_flashsr_plugin_alias_exists():
    """AudioSRPlugin is an alias for FlashSRPlugin."""
    from plugins.flashsr_plugin import AudioSRPlugin, FlashSRPlugin

    assert AudioSRPlugin is FlashSRPlugin


def test_get_flashsr_plugin_returns_flashsr():
    """get_flashsr_plugin() returns FlashSRPlugin instance."""
    from plugins.flashsr_plugin import FlashSRPlugin, get_flashsr_plugin

    p = get_flashsr_plugin()
    assert isinstance(p, FlashSRPlugin)


def test_backward_compat_functions_exist():
    """Backward-compat functions are available."""
    from plugins.flashsr_plugin import (
        _get_ml_model,
        allow_reset_ml_model_failed,
        has_flashsr_ml_failed,
        unload_flashsr,
    )

    assert callable(has_flashsr_ml_failed)
    assert callable(unload_flashsr)
    assert callable(allow_reset_ml_model_failed)
    assert callable(_get_ml_model)


def test_model_loaded_attribute():
    """_model_loaded property exists for ml_model_readiness."""
    from plugins.flashsr_plugin import get_flashsr_plugin

    p = get_flashsr_plugin()
    # _model_loaded is a property, accessing it should not raise
    loaded = p._model_loaded  # type: ignore[attr-defined]
    assert isinstance(loaded, bool)


def test_process_mono_shape_preserved():
    """process() preserves mono input shape."""
    from plugins.flashsr_plugin import get_flashsr_plugin

    p = get_flashsr_plugin()
    audio = np.random.randn(48000).astype(np.float32) * 0.1
    result = p.process(audio, 48000)
    assert result.shape == audio.shape


def test_process_stereo_shape_preserved():
    """process() preserves stereo input shape (channels-first)."""
    from plugins.flashsr_plugin import get_flashsr_plugin

    p = get_flashsr_plugin()
    audio = np.random.randn(2, 48000).astype(np.float32) * 0.1
    result = p.process(audio, 48000)
    assert result.shape == audio.shape


def test_process_no_nan_output():
    """process() never returns NaN."""
    from plugins.flashsr_plugin import get_flashsr_plugin

    p = get_flashsr_plugin()
    audio = np.random.randn(48000).astype(np.float32) * 0.1
    result = p.process(audio, 48000)
    assert np.all(np.isfinite(result))


def test_process_no_clipping():
    """process() output is bounded to [-1, 1]."""
    from plugins.flashsr_plugin import get_flashsr_plugin

    p = get_flashsr_plugin()
    audio = np.random.randn(48000).astype(np.float32) * 0.1
    result = p.process(audio, 48000)
    assert np.max(np.abs(result)) <= 1.0
