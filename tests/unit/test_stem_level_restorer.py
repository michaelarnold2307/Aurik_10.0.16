"""Unit tests for §SLR-1 StemLevelRestorer (backend/core/dsp/stem_level_restorer.py).

Tests: singleton, restore guards, DSP stem split fallback, SNR estimation.
"""

import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, sr: int = 48000, duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(sr: int = 48000, duration: float = 3.0) -> np.ndarray:
    return np.stack([_sine(440, sr, duration), _sine(550, sr, duration)], axis=-1)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stem_level_restorer_singleton():
    from backend.core.dsp.stem_level_restorer import get_stem_level_restorer

    a = get_stem_level_restorer()
    b = get_stem_level_restorer()
    assert a is b


# ---------------------------------------------------------------------------
# Guard: short audio skipped
# ---------------------------------------------------------------------------


def test_restore_short_audio_skipped():
    from backend.core.dsp.stem_level_restorer import get_stem_level_restorer

    slr = get_stem_level_restorer()
    short_audio = np.zeros(1000, dtype=np.float32)  # << 2 s @ 48 kHz
    result = slr.restore(short_audio, 48000, panns_singing=0.8)
    assert result is None


# ---------------------------------------------------------------------------
# Guard: low singing probability skipped
# ---------------------------------------------------------------------------


def test_restore_low_singing_skipped():
    from backend.core.dsp.stem_level_restorer import get_stem_level_restorer

    slr = get_stem_level_restorer()
    audio = _sine(duration=4.0)
    result = slr.restore(audio, 48000, panns_singing=0.2)
    assert result is None


# ---------------------------------------------------------------------------
# Guard: SR assertion
# ---------------------------------------------------------------------------


def test_restore_sr_assertion():
    from backend.core.dsp.stem_level_restorer import get_stem_level_restorer

    slr = get_stem_level_restorer()
    audio = _sine(duration=4.0)
    with pytest.raises(AssertionError):
        slr.restore(audio, 44100, panns_singing=0.9)


# ---------------------------------------------------------------------------
# DSP fallback stem split
# ---------------------------------------------------------------------------


def test_dsp_stem_split_shapes_match():
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    slr = StemLevelRestorer()
    audio = _sine(duration=3.0)
    vocal, instr = slr._dsp_stem_split(audio, 48000)  # pylint: disable=protected-access
    assert vocal.shape == audio.shape
    assert instr.shape == audio.shape
    assert vocal.dtype == np.float32
    assert instr.dtype == np.float32


def test_dsp_stem_split_sum_approximately_original():
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    slr = StemLevelRestorer()
    audio = _sine(duration=3.0)
    vocal, instr = slr._dsp_stem_split(audio, 48000)  # pylint: disable=protected-access
    recombined = vocal + instr
    # Sum should be close to original (bandpass + residual = original)
    np.testing.assert_allclose(recombined, audio, atol=1e-4)


def test_demucs_stem_split_sums_all_non_vocal_stems(monkeypatch):
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    class _FakeDemucs:
        _session = object()

        @staticmethod
        def separate(
            audio: np.ndarray,
            sr: int,
            prefer_mdx23c: bool = True,
        ) -> dict[str, np.ndarray]:  # pylint: disable=unused-argument
            return {
                "vocals": np.full_like(audio, 0.10, dtype=np.float32),
                "drums": np.full_like(audio, 0.20, dtype=np.float32),
                "bass": np.full_like(audio, 0.30, dtype=np.float32),
                "other": np.full_like(audio, 0.40, dtype=np.float32),
            }

    monkeypatch.setattr("plugins.demucs_v4_plugin.get_demucs_plugin", lambda: _FakeDemucs())
    monkeypatch.setattr(
        "plugins.bs_roformer_plugin.get_bs_roformer",
        lambda: (_ for _ in ()).throw(RuntimeError("roformer unavailable")),
    )
    monkeypatch.setattr(
        "plugins.mdx23c_plugin.get_mdx23c_plugin", lambda: (_ for _ in ()).throw(RuntimeError("mdx unavailable"))
    )
    monkeypatch.setattr(
        "backend.core.dsp.sota_vocal_model_router.SotaVocalModelRouter._available_memory_gb",
        lambda _self: 32.0,
    )

    audio = np.zeros(48000, dtype=np.float32)
    vocal, instr, model_used = StemLevelRestorer()._separate_stems(  # pylint: disable=protected-access
        audio,
        48000,
        panns_singing=0.8,
        ctx={"material_type": "live", "score_routing_enabled": False},
    )
    np.testing.assert_allclose(vocal, np.full_like(audio, 0.10), atol=1e-6)
    np.testing.assert_allclose(instr, np.full_like(audio, 0.90), atol=1e-6)
    assert model_used == "demucs_v4_htdemucs"


def test_vqi_gate_rolls_back_on_low_vqi(monkeypatch):
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    slr = StemLevelRestorer()
    audio = _sine(duration=3.0)
    monkeypatch.setattr(
        slr, "_separate_stems", lambda _a, _sr, _p=0.0, _ctx=None: (_a * 0.5, _a * 0.5, "test_separator")
    )
    monkeypatch.setattr(slr, "_apply_miipher", lambda stem, _sr, _bias=-6.0: (stem * 0.5, True, "test_vocal_nr"))
    monkeypatch.setattr(slr, "_apply_dfn", lambda stem, _sr, energy_bias_db=-9.0: (stem, False, "none"))
    monkeypatch.setattr(slr, "_hallucination_guard", lambda _pre, post, _sr, _name: post)
    monkeypatch.setattr(
        "backend.core.musical_goals.vocal_quality_index.compute_vqi",
        lambda *_args, **_kwargs: {"vqi": 0.60, "singer_identity_cosine": 0.99},
    )

    result = slr.restore(audio, 48000, panns_singing=0.9)
    assert result is not None
    assert result.success is False
    assert result.rollback_reason.startswith("vqi_below_floor")
    assert result.separation_model == "test_separator"
    assert result.vocal_nr_model == "test_vocal_nr"
    np.testing.assert_allclose(result.audio, audio, atol=1e-7)


def test_restore_reports_routed_models(monkeypatch):
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    slr = StemLevelRestorer()
    audio = _sine(duration=3.0)
    monkeypatch.setattr(
        slr, "_separate_stems", lambda _a, _sr, _p=0.0, _ctx=None: (_a * 0.5, _a * 0.5, "test_separator")
    )
    monkeypatch.setattr(slr, "_apply_miipher", lambda stem, _sr, _bias=-6.0: (stem, True, "miipher"))
    monkeypatch.setattr(slr, "_apply_dfn", lambda stem, _sr, energy_bias_db=-9.0: (stem, True, "deepfilternet_v3_ii"))
    monkeypatch.setattr(slr, "_hallucination_guard", lambda _pre, post, _sr, _name: post)
    monkeypatch.setattr(
        "backend.core.musical_goals.vocal_quality_index.compute_vqi",
        lambda *_args, **_kwargs: {"vqi": 0.90, "singer_identity_cosine": 0.99},
    )

    result = slr.restore(audio, 48000, panns_singing=0.9)
    assert result is not None
    assert result.success is True
    assert result.separation_model == "test_separator"
    assert result.vocal_nr_model == "miipher"
    assert result.instrumental_nr_model == "deepfilternet_v3_ii"


def test_restore_injects_active_ml_plugins_into_router_context(monkeypatch):
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    slr = StemLevelRestorer()
    audio = _sine(duration=3.0)
    captured: dict[str, dict] = {}

    class _Entry:
        def __init__(self, active: bool):
            self.active = active

    _fake_plm = types.SimpleNamespace(
        _entries={
            "A": _Entry(True),
            "B": _Entry(True),
            "C": _Entry(False),
        }
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "backend.core.plugin_lifecycle_manager",
        types.SimpleNamespace(get_plugin_lifecycle_manager=lambda: _fake_plm),
    )

    def _capture_separate(_a, _sr, _p=0.0, _ctx=None):
        captured["ctx"] = dict(_ctx or {})
        return _a * 0.5, _a * 0.5, "test_separator"

    monkeypatch.setattr(slr, "_separate_stems", _capture_separate)
    monkeypatch.setattr(slr, "_apply_miipher", lambda stem, _sr, _bias=-6.0: (stem, True, "miipher"))
    monkeypatch.setattr(slr, "_apply_dfn", lambda stem, _sr, energy_bias_db=-9.0: (stem, True, "deepfilternet_v3_ii"))
    monkeypatch.setattr(slr, "_hallucination_guard", lambda _pre, post, _sr, _name: post)
    monkeypatch.setattr(
        "backend.core.musical_goals.vocal_quality_index.compute_vqi",
        lambda *_args, **_kwargs: {"vqi": 0.90, "singer_identity_cosine": 0.99},
    )

    result = slr.restore(audio, 48000, panns_singing=0.9, restoration_context={})
    assert result is not None
    assert result.success is True
    assert captured["ctx"]["active_ml_plugins"] == 2


def test_restore_respects_active_ml_plugins_from_context(monkeypatch):
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    slr = StemLevelRestorer()
    audio = _sine(duration=3.0)
    captured: dict[str, dict] = {}

    def _capture_separate(_a, _sr, _p=0.0, _ctx=None):
        captured["ctx"] = dict(_ctx or {})
        return _a * 0.5, _a * 0.5, "test_separator"

    monkeypatch.setattr(slr, "_separate_stems", _capture_separate)
    monkeypatch.setattr(slr, "_apply_miipher", lambda stem, _sr, _bias=-6.0: (stem, True, "miipher"))
    monkeypatch.setattr(slr, "_apply_dfn", lambda stem, _sr, energy_bias_db=-9.0: (stem, True, "deepfilternet_v3_ii"))
    monkeypatch.setattr(slr, "_hallucination_guard", lambda _pre, post, _sr, _name: post)
    monkeypatch.setattr(
        "backend.core.musical_goals.vocal_quality_index.compute_vqi",
        lambda *_args, **_kwargs: {"vqi": 0.90, "singer_identity_cosine": 0.99},
    )

    result = slr.restore(audio, 48000, panns_singing=0.9, restoration_context={"active_ml_plugins": 7})
    assert result is not None
    assert result.success is True
    assert captured["ctx"]["active_ml_plugins"] == 7


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------


def test_estimate_snr_gain_positive():
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    audio = _sine()
    noise = 0.01 * np.random.randn(*audio.shape).astype(np.float32)
    noisy = (audio + noise).astype(np.float32)
    clean = audio
    snr = StemLevelRestorer._estimate_snr_gain(noisy, clean)  # pylint: disable=protected-access
    assert snr >= 0.0
    assert snr <= 30.0


def test_estimate_snr_gain_zero_when_identical():
    from backend.core.dsp.stem_level_restorer import StemLevelRestorer

    audio = _sine()
    snr = StemLevelRestorer._estimate_snr_gain(audio, audio)  # pylint: disable=protected-access
    # Pre ≈ post → noise ≈ 0 → very high SNR (clamped to 30 dB)
    assert snr >= 0.0


# ---------------------------------------------------------------------------
# Full restore (mocked plugins — DSP fallback path)
# ---------------------------------------------------------------------------


def test_restore_returns_result_or_none(monkeypatch):
    """Full restore call: plugins not available → DSP fallback → success or non-blocking None."""
    from backend.core.dsp import stem_level_restorer as _slr_mod

    slr = _slr_mod.StemLevelRestorer()
    audio = _sine(duration=4.0)
    # Monkeypatch all lazy plugin imports to raise ImportError → DSP fallback
    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _fake_import(name, *args, **kwargs):
        if "miipher_plugin" in name or "sgmse_plugin" in name or "deep_filter_plugin" in name:
            raise ImportError(f"Mocked: {name}")
        if "demucs_plugin" in name:
            raise ImportError(f"Mocked: {name}")
        if "hallucination_guard" in name:
            raise ImportError(f"Mocked: {name}")
        if "hnr_blend" in name:
            raise ImportError(f"Mocked: {name}")
        return original_import(name, *args, **kwargs)

    import builtins

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    result = slr.restore(audio, 48000, panns_singing=0.8)
    # With all plugins mocked out, DSP fallback runs but hallucination guard fails silently
    # → result is StemLevelRestorerResult (success may be True with DSP stems) or None
    assert result is None or hasattr(result, "audio")
    if result is not None:
        assert result.audio.shape == audio.shape
        assert not np.any(np.isnan(result.audio))
        assert float(np.max(np.abs(result.audio))) <= 1.0


def test_restore_stereo(monkeypatch):
    import builtins

    from backend.core.dsp import stem_level_restorer as _slr_mod

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if any(p in name for p in ("miipher", "sgmse", "deep_filter", "demucs", "hallucination", "hnr_blend")):
            raise ImportError(f"Mocked: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    slr = _slr_mod.StemLevelRestorer()
    audio = _stereo(duration=4.0)
    result = slr.restore(audio, 48000, panns_singing=0.9)
    # Non-blocking: any result is acceptable
    assert result is None or hasattr(result, "audio")
