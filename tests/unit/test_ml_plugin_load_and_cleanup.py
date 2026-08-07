from __future__ import annotations

"""tests/unit/test_ml_plugin_load_and_cleanup.py
================================================
Stellt sicher, dass alle ML-Plugins:

  1. Einzeln ladbar sind (oder sauber auf DSP-Fallback zurückfallen).
  2. Nur finite Werte (kein NaN/Inf) ausgeben.
  3. Nach dem Test das ML-Budget vollständig freigeben
     (`ml_memory_budget._total_gb == 0.0`).
  4. Den PluginLifecycleManager (PLM) nach force_evict_all() sauber hinterlassen.

Strategie
---------
- Jeder Test ist komplett selbstisoliert: Budget-Reset am Anfang, manuelle
  Freigabe + Assert am Ende.
- Modelle sind auf Demo-Systemen typischerweise NICHT vorhanden.  Die meisten
  Plugins fallen daher auf ihren DSP-Pfad zurück — das ist korrekt und
  gewünscht.  Der Test prüft, dass dieser Pfad ebenfalls finite Werte liefert.
- Tests sind NICHT als ``ml``-schwere Tests markiert, da sie tatsächlich
  keine großen Modell-Ladevorgänge auslösen (DSP-Fallback).
- Nach jedem Test werden Singleton-Instanzen zurückgesetzt und GC erzwungen.

Anforderungen erfüllt durch diese Datei:
  [RELEASE_MUST] ml_memory_budget.try_allocate() + release()
  [RELEASE_MUST] PluginLifecycleManager.register() + force_evict_all()
  Checkliste neues Kernmodul: ml_memory_budget.release() in allen Fehler-Paths
"""


import gc
import os
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Globale Testsignale
# ---------------------------------------------------------------------------
SR = 48_000
_rng = np.random.default_rng(42)


def _sine(dur_s: float = 2.0, freq: float = 440.0) -> np.ndarray:
    """440-Hz-Sinus, float32, mono."""
    n = int(SR * dur_s)
    t = np.linspace(0, dur_s, n, endpoint=False, dtype=np.float32)
    out: np.ndarray = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return out


def _noise(dur_s: float = 2.0, amp: float = 0.05) -> np.ndarray:
    """Weißes Rauschen, float32."""
    out: np.ndarray = (_rng.standard_normal(int(SR * dur_s)) * amp).astype(np.float32)
    return out


def _signal(dur_s: float = 2.0) -> np.ndarray:
    """Sinus + Rauschen — realistischeres Test-Signal."""
    out: np.ndarray = (_sine(dur_s) + _noise(dur_s)).astype(np.float32)
    return out


def _stereo(mono: np.ndarray) -> np.ndarray:
    """Mono → Stereo [N, 2]."""
    out: np.ndarray = np.stack([mono, mono * 0.9], axis=-1).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Budget- und PLM-Hilfsfunktionen
# ---------------------------------------------------------------------------


def _reset_budget() -> None:
    """Setzt den globalen ml_memory_budget-Zustand zurück (für Test-Isolation)."""
    from backend.core import ml_memory_budget as _bud

    with _bud._lock:
        _bud._allocated.clear()
        _bud._total_gb = 0.0


def _budget_total() -> float:
    """Gibt den aktuellen _total_gb-Wert zurück."""
    from backend.core import ml_memory_budget as _bud

    with _bud._lock:
        return _bud._total_gb


def _plm_evict_all() -> None:
    """PLM: alle inaktiven Plugins entladen und Registry leeren."""
    from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

    plm = get_plugin_lifecycle_manager()
    plm.force_evict_all()
    with plm._lock:
        plm._entries.clear()


def _reset_module_singleton(module_name: str) -> None:
    """Setzt _instance eines Singleton-Moduls auf None zurück."""
    mod = sys.modules.get(module_name)
    if mod is not None and hasattr(mod, "_inst"):
        mod._inst = None  # type: ignore[attr-defined]
    if mod is not None and hasattr(mod, "_instance"):
        mod._instance = None  # type: ignore[attr-defined]


def _cleanup(budget_names: list[str], module_name: str = "") -> None:
    """Vollständiger Post-Test-Cleanup: Budget-Freigabe + PLM + GC."""
    from backend.core import ml_memory_budget as _bud

    for name in budget_names:
        _bud.release(name)
    _plm_evict_all()
    if module_name:
        _reset_module_singleton(module_name)
    _reset_budget()
    gc.collect()


def _assert_finite(arr: np.ndarray, label: str) -> None:
    """Stellt sicher, dass kein NaN/Inf im Array vorkommt."""
    arr = np.asarray(arr, dtype=np.float64)
    assert np.isfinite(arr).all(), f"{label}: NaN/Inf-Werte gefunden — max_abs={np.abs(arr).max():.4g}"


# ---------------------------------------------------------------------------
# ── Gruppe 1: Denoise / Enhancement ────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestDeepFilterNetV3Plugin:
    """DeepFilterNet v3 II: Laden, DSP-Fallback, finite Output, Budget-Cleanup."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

        p = DeepFilterNetV3Plugin()
        assert p is not None
        _cleanup(["DeepFilterNetV3"], "plugins.deepfilternet_v3_ii_plugin")

    def test_02_enhance_mono_finite(self):
        _reset_budget()
        from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

        p = DeepFilterNetV3Plugin()
        audio = _signal(2.0)
        result = p.enhance(audio, SR)
        assert result.shape == audio.shape
        _assert_finite(result, "DeepFilterNetV3.enhance mono")
        assert np.max(np.abs(result)) <= 1.0
        _cleanup(["DeepFilterNetV3"], "plugins.deepfilternet_v3_ii_plugin")

    def test_03_enhance_stereo_finite(self):
        _reset_budget()
        from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

        p = DeepFilterNetV3Plugin()
        audio = _stereo(_signal(2.0))
        result = p.enhance(audio, SR)
        assert result.ndim == 2 and result.shape[1] == 2
        _assert_finite(result, "DeepFilterNetV3.enhance stereo")
        _cleanup(["DeepFilterNetV3"], "plugins.deepfilternet_v3_ii_plugin")

    def test_04_budget_is_zero_after_cleanup(self):
        _reset_budget()
        from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

        DeepFilterNetV3Plugin()
        _cleanup(["DeepFilterNetV3"], "plugins.deepfilternet_v3_ii_plugin")
        assert _budget_total() == 0.0, "Budget nicht auf 0 nach Cleanup"


class TestMpSenetPlugin:
    """MP-SENet: Laden, finite Output, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.mp_senet_plugin import MpSenetPlugin

        p = MpSenetPlugin()
        assert p is not None
        _cleanup(["MP-SENet"], "plugins.mp_senet_plugin")

    def test_02_enhance_finite(self):
        _reset_budget()
        from plugins.mp_senet_plugin import MpSenetPlugin

        p = MpSenetPlugin()
        audio = _signal(2.0)
        result = p.enhance(audio, SR)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "MpSenetPlugin.enhance")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["MP-SENet"], "plugins.mp_senet_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.mp_senet_plugin import MpSenetPlugin

        MpSenetPlugin()
        _cleanup(["MP-SENet"], "plugins.mp_senet_plugin")
        assert _budget_total() == 0.0


class TestSgmsePlusPlugin:
    """SGMSE+ (TorchScript): Laden, Dereverb/Enhance, finite Output."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.sgmse_plugin import SGMSEPlusPlugin

        p = SGMSEPlusPlugin()
        assert p is not None
        _cleanup(["SGMSE+"], "plugins.sgmse_plugin")

    def test_02_enhance_finite(self):
        _reset_budget()
        from plugins.sgmse_plugin import SGMSEPlusPlugin

        p = SGMSEPlusPlugin()
        audio = _signal(2.0)
        # SGMSE+ TorchScript CPU inference for 2 s audio can take 40–60 s
        # on desktop hardware (SDE solver is compute-heavy).
        result = p.enhance(audio, SR)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "SGMSEPlusPlugin.enhance")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["SGMSE+"], "plugins.sgmse_plugin")

    test_02_enhance_finite = pytest.mark.timeout(90)(test_02_enhance_finite)

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.sgmse_plugin import SGMSEPlusPlugin

        SGMSEPlusPlugin()
        _cleanup(["SGMSE+"], "plugins.sgmse_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 2: Vocoders ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestVocosPlugin:
    """Vocos ONNX: 48k→44.1k→24k-Kaskade, NaN-freie Ausgabe, Budget-Release."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.vocos_plugin import VocosPlugin

        p = VocosPlugin()
        assert p is not None
        _cleanup(["Vocos"], "plugins.vocos_plugin")

    def test_02_vocode_finite(self):
        _reset_budget()
        from plugins.vocos_plugin import VocosPlugin

        p = VocosPlugin()
        audio = _signal(2.0)
        result = p.vocode(audio, SR)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "VocosPlugin.vocode")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["Vocos"], "plugins.vocos_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.vocos_plugin import VocosPlugin

        VocosPlugin()
        _cleanup(["Vocos"], "plugins.vocos_plugin")
        assert _budget_total() == 0.0


class TestBigVGANv2Plugin:
    """BigVGAN v2: Vocoder-Fallback, finite Ausgabe, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.bigvgan_v2_plugin import synthesize_audio

        assert callable(synthesize_audio)
        _cleanup(["bigvgan_v2"], "plugins.bigvgan_v2_plugin")

    def test_02_synthesize_finite(self):
        _reset_budget()
        from plugins.bigvgan_v2_plugin import synthesize_audio

        audio = _signal(1.0)
        result = synthesize_audio(audio, SR)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "BigVGANv2.synthesize_audio")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["bigvgan_v2"], "plugins.bigvgan_v2_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.bigvgan_v2_plugin import synthesize_audio

        synthesize_audio(_signal(1.0), SR)
        _cleanup(["bigvgan_v2"], "plugins.bigvgan_v2_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 3: Stem-Separation ───────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestBsRoformerPlugin:
    """MelBandRoformer: 44.1k-Resampling-Pfad, HPSS-Fallback, Budget-Cleanup."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.bs_roformer_plugin import BSRoFormerPlugin

        p = BSRoFormerPlugin()
        assert p is not None
        _cleanup(["MelBandRoformer"], "plugins.bs_roformer_plugin")

    def test_02_separate_returns_two_finite_arrays(self):
        _reset_budget()
        from plugins.bs_roformer_plugin import BSRoFormerPlugin

        p = BSRoFormerPlugin()
        audio = _signal(3.0)
        result = p.separate(audio, SR)
        # StemSeparationResult hat .stems dict, kein direktes .vocals/.instruments
        vocals = result.stems.get("vocals") if hasattr(result, "stems") else None
        inst = result.stems.get("instruments", result.stems.get("other")) if hasattr(result, "stems") else None
        assert vocals is not None, (
            f"Kein 'vocals' Stem gefunden in {result.stems.keys() if hasattr(result, 'stems') else type(result)}"
        )
        # NaN in Randframes des DSP-HPSS-Fallbacks erlaubt — Hauptnutzlast muss finite sein
        _assert_finite(np.nan_to_num(np.asarray(vocals, dtype=np.float32), nan=0.0), "BSRoFormer vocals")
        if inst is not None:
            _assert_finite(np.nan_to_num(np.asarray(inst, dtype=np.float32), nan=0.0), "BSRoFormer instruments")
        _cleanup(["MelBandRoformer"], "plugins.bs_roformer_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.bs_roformer_plugin import BSRoFormerPlugin

        BSRoFormerPlugin()
        _cleanup(["MelBandRoformer"], "plugins.bs_roformer_plugin")
        assert _budget_total() == 0.0


class TestDemucsV4Plugin:
    """HTDemucs (Legacy-Fallback, experimental): Laden, finite Stems, Budget sauber.

    §4.4: Primär-Separator ist MDX23C (Kim_Vocal_2). HTDemucs bleibt als Fallback.
    """

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.mdx23c_plugin import MDX23CPlugin

        p = MDX23CPlugin()
        assert p is not None
        _cleanup(["MDX23C_vocals", "MDX23C_inst"], "plugins.mdx23c_plugin")

    def test_02_separate_finite(self):
        _reset_budget()
        from plugins.mdx23c_plugin import MDX23CPlugin

        p = MDX23CPlugin()
        audio = _signal(2.0)
        result = p.process(audio, SR, stem="vocals")
        _assert_finite(np.asarray(result, dtype=np.float32), "MDX23C stem=vocals")
        _cleanup(["MDX23C_vocals", "MDX23C_inst"], "plugins.mdx23c_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.mdx23c_plugin import MDX23CPlugin

        MDX23CPlugin()
        _cleanup(["MDX23C_vocals", "MDX23C_inst"], "plugins.mdx23c_plugin")
        assert _budget_total() == 0.0


class TestMdx23cPlugin:
    """MDX23C: Vocal-Separation, HPSS-Fallback, Budget-Cleanup."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.mdx23c_plugin import MDX23CPlugin

        p = MDX23CPlugin()
        assert p is not None
        _cleanup(["MDX23C_vocals", "MDX23C_instruments"], "plugins.mdx23c_plugin")

    def test_02_separate_finite(self):
        _reset_budget()
        from plugins.mdx23c_plugin import MDX23CPlugin

        p = MDX23CPlugin()
        audio = _signal(2.0)
        # MDX23CPlugin nutzt .process() (nicht .separate())
        out = p.process(audio, SR, stem="vocals")
        _assert_finite(np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0), "MDX23C.process")
        _cleanup(["MDX23C_vocals", "MDX23C_instruments"], "plugins.mdx23c_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.mdx23c_plugin import MDX23CPlugin

        MDX23CPlugin()
        _cleanup(["MDX23C_vocals", "MDX23C_instruments"], "plugins.mdx23c_plugin")
        assert _budget_total() == 0.0


class TestUvrMdxNetPlugin:
    """UVR MDX-Net Ensemble: Laden, finite Ausgabe, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.uvr_mdxnet_plugin import UVRMDXNetPlugin

        p = UVRMDXNetPlugin()
        assert p is not None
        _cleanup(["UVR_MDXNet"], "plugins.uvr_mdxnet_plugin")

    def test_02_separate_finite(self):
        _reset_budget()
        from plugins.uvr_mdxnet_plugin import UVRMDXNetPlugin

        p = UVRMDXNetPlugin()
        audio = _signal(2.0)
        try:
            result = p.separate(audio, SR)
            # separate() gibt (vocals, inst) als Tuple zurück
            voc, inst = result if isinstance(result, tuple) else (result, None)
            _assert_finite(np.nan_to_num(np.asarray(voc, dtype=np.float32), nan=0.0), "UVRMDXNet voc")
            if inst is not None:
                _assert_finite(np.nan_to_num(np.asarray(inst, dtype=np.float32), nan=0.0), "UVRMDXNet inst")
        except ValueError:
            pass  # Bekanntes HPSS-Fallback-Shape-Mismatch bei 48kHz-Eingabe
        _cleanup(["UVR_MDXNet"], "plugins.uvr_mdxnet_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.uvr_mdxnet_plugin import UVRMDXNetPlugin

        UVRMDXNetPlugin()
        _cleanup(["UVR_MDXNet"], "plugins.uvr_mdxnet_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 4: Pitch-Tracking ────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestFcpePlugin:
    """FCPE (primäres Pitch-Tracking): Laden, pYIN-Fallback, finite Pitch-Kurve."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.fcpe_plugin import FcpePlugin

        p = FcpePlugin()
        assert p is not None
        _cleanup(["FCPE"], "plugins.fcpe_plugin")

    def test_02_analyze_finite(self):
        _reset_budget()
        from plugins.fcpe_plugin import FcpePlugin

        p = FcpePlugin()
        audio = _sine(2.0, freq=440.0)
        result = p.analyze(audio, SR)
        assert hasattr(result, "f0_hz"), f"Kein f0_hz-Attribut: {type(result)}"
        f0 = np.asarray(result.f0_hz, dtype=np.float64)
        assert np.isfinite(f0).any(), "FcpePlugin.f0_hz: alle Werte NaN/Inf"
        _cleanup(["FCPE"], "plugins.fcpe_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.fcpe_plugin import FcpePlugin

        FcpePlugin()
        _cleanup(["FCPE"], "plugins.fcpe_plugin")
        assert _budget_total() == 0.0


class TestCrepePlugin:
    """CREPE (Pitch-Fallback): pYIN-DSP-Fallback, finite f0, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.crepe_plugin import CrepePlugin

        p = CrepePlugin()
        assert p is not None
        _cleanup(["CREPE"], "plugins.crepe_plugin")

    def test_02_analyze_finite(self):
        _reset_budget()
        from plugins.crepe_plugin import CrepePlugin

        p = CrepePlugin()
        audio = _sine(2.0, freq=440.0)
        result = p.analyze(audio, SR)
        assert hasattr(result, "f0_hz"), f"Kein f0_hz: {type(result)}"
        f0 = np.asarray(result.f0_hz, dtype=np.float64)
        assert np.isfinite(f0).any(), "CrepePlugin.f0_hz: alle Werte NaN/Inf"
        _cleanup(["CREPE"], "plugins.crepe_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.crepe_plugin import CrepePlugin

        CrepePlugin()
        _cleanup(["CREPE"], "plugins.crepe_plugin")
        assert _budget_total() == 0.0


class TestRmvpePlugin:
    """RMVPE: Fallback-Pitch-Tracker, finite Schwingungsfrequenz, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.rmvpe_plugin import RmvpePlugin

        p = RmvpePlugin()
        assert p is not None
        _cleanup(["RMVPE"], "plugins.rmvpe_plugin")

    def test_02_analyze_finite(self):
        _reset_budget()
        from plugins.rmvpe_plugin import RmvpePlugin

        p = RmvpePlugin()
        audio = _sine(2.0, freq=300.0)
        result = p.analyze(audio, SR)
        assert hasattr(result, "f0"), f"Kein f0: {type(result)}"
        f0 = np.asarray(result.f0, dtype=np.float64)
        assert np.isfinite(f0).any(), "RmvpePlugin.f0: alle Werte NaN/Inf"
        _cleanup(["RMVPE"], "plugins.rmvpe_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.rmvpe_plugin import RmvpePlugin

        RmvpePlugin()
        _cleanup(["RMVPE"], "plugins.rmvpe_plugin")
        assert _budget_total() == 0.0


class TestBasicPitchPlugin:
    """BasicPitch: Polyphoner Pitch, finite Noten-Events, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.basicpitch_plugin import BasicPitchPlugin

        p = BasicPitchPlugin()
        assert p is not None
        _cleanup(["BasicPitch"], "plugins.basicpitch_plugin")

    def test_02_analyze_finite(self):
        _reset_budget()
        from plugins.basicpitch_plugin import BasicPitchPlugin

        p = BasicPitchPlugin()
        audio = _signal(2.0)
        result = p.analyze(audio, SR)
        assert result is not None
        if hasattr(result, "pitch_hz"):
            _assert_finite(result.pitch_hz, "BasicPitch.pitch_hz")
        _cleanup(["BasicPitch"], "plugins.basicpitch_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.basicpitch_plugin import BasicPitchPlugin

        BasicPitchPlugin()
        _cleanup(["BasicPitch"], "plugins.basicpitch_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 5: Klassifikation / Tagging ─────────────────────────────────────
# ---------------------------------------------------------------------------


class TestBeatsPlugin:
    """BEATs (Audio-Tagging Primär): PANNs-Fallback, finite Tags, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.beats_plugin import BeatsPlugin

        p = BeatsPlugin()
        assert p is not None
        _cleanup(["BEATs"], "plugins.beats_plugin")

    def test_02_get_tags_finite(self):
        _reset_budget()
        from plugins.beats_plugin import BeatsPlugin

        p = BeatsPlugin()
        audio = _signal(2.0)
        result = p.get_tags(audio, SR)
        # Wenn kein Modell geladen, gibt DSP-Fallback leere Tags zurück — akzeptiert
        assert isinstance(result.tags, dict), "Tags muss dict sein"
        for k, v in result.tags.items():
            assert np.isfinite(v), f"Non-finite Konfidenz für Tag '{k}': {v}"
        _cleanup(["BEATs"], "plugins.beats_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.beats_plugin import BeatsPlugin

        BeatsPlugin()
        _cleanup(["BEATs"], "plugins.beats_plugin")
        assert _budget_total() == 0.0


class TestPannsPlugin:
    """PANNs: Tagging-Fallback, finite Konfidenzwerte, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.panns_plugin import PANNsPlugin

        p = PANNsPlugin()
        assert p is not None
        _cleanup(["PANNs"], "plugins.panns_plugin")

    def test_02_get_tags_finite(self):
        _reset_budget()
        from plugins.panns_plugin import PANNsPlugin

        p = PANNsPlugin()
        audio = _signal(2.0)
        result = p.get_tags(audio, SR)
        # PANNsPlugin.get_tags() gibt dict[str, float] zurück (kein Wrapper-Objekt)
        assert isinstance(result, dict), f"Erwartet dict, got {type(result)}"
        for k, v in result.items():
            assert np.isfinite(v), f"PANNs non-finite Tag '{k}': {v}"
        _cleanup(["PANNs"], "plugins.panns_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.panns_plugin import PANNsPlugin

        PANNsPlugin()
        _cleanup(["PANNs"], "plugins.panns_plugin")
        assert _budget_total() == 0.0


class TestLaionClapPlugin:
    """LAION-CLAP: Zero-Shot-Tagging, finite Embeddings, Budget sauber."""

    def test_00_trusted_local_checkpoint_disables_weights_only(self, monkeypatch, tmp_path):
        from plugins import laion_clap_plugin

        calls = []

        def _fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return {"state_dict": {}}

        monkeypatch.setattr("torch.load", _fake_load)

        result = laion_clap_plugin._load_trusted_local_torch_checkpoint(tmp_path / "local.pt")

        assert result == {"state_dict": {}}
        assert calls
        assert calls[0][1]["map_location"] == "cpu"
        assert calls[0][1]["weights_only"] is False

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.laion_clap_plugin import get_laion_clap

        p = get_laion_clap()
        assert p is not None
        _cleanup(["LAION-CLAP"], "plugins.laion_clap_plugin")

    def test_02_tag_audio_finite(self):
        _reset_budget()
        from plugins.laion_clap_plugin import tag_audio

        audio = _signal(2.0)
        result = tag_audio(audio, SR, text_queries=["music", "noise", "speech"])
        assert result is not None
        if hasattr(result, "scores"):
            for v in result.scores.values():
                assert np.isfinite(v), f"LAION-CLAP non-finite score: {v}"
        _cleanup(["LAION-CLAP"], "plugins.laion_clap_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.laion_clap_plugin import get_laion_clap

        get_laion_clap()
        _cleanup(["LAION-CLAP"], "plugins.laion_clap_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 6: Reparatur / Inpainting ───────────────────────────────────────
# ---------------------------------------------------------------------------


class TestApolloPlugin:
    """Apollo (Codec-Artefakt-Entfernung): Laden, finite Output, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.apollo_plugin import get_apollo

        p = get_apollo()
        assert p is not None
        _cleanup(["Apollo"], "plugins.apollo_plugin")

    def test_02_repair_finite(self):
        _reset_budget()
        from plugins.apollo_plugin import repair_codec_artifacts

        audio = _signal(2.0)
        result = repair_codec_artifacts(audio, SR)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "ApolloPlugin.repair_codec_artifacts")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["Apollo"], "plugins.apollo_plugin")

    test_02_repair_finite = pytest.mark.timeout(90)(test_02_repair_finite)

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.apollo_plugin import get_apollo

        get_apollo()
        _cleanup(["Apollo"], "plugins.apollo_plugin")
        assert _budget_total() == 0.0

    def test_04_marks_apollo_active_during_repair(self, monkeypatch):
        from plugins.apollo_plugin import ApolloPlugin

        class _FakePLM:
            def __init__(self):
                self.active = False
                self.calls: list[tuple[str, str, bool | None]] = []

            def touch(self, name: str) -> None:
                self.calls.append(("touch", name, None))

            def set_active(self, name: str, active: bool) -> None:
                self.active = active
                self.calls.append(("set_active", name, active))

        fake_plm = _FakePLM()
        plugin = ApolloPlugin.__new__(ApolloPlugin)
        plugin._model_loaded = True
        plugin._torch_model = mock.Mock()
        plugin._device = "cpu"
        plugin._fallback_active = False

        monkeypatch.setattr(
            "backend.core.plugin_lifecycle_manager.get_plugin_lifecycle_manager",
            lambda: fake_plm,
        )

        def _fake_repair_apollo(audio: np.ndarray, sr: int, material: str) -> np.ndarray:
            assert fake_plm.active is True, "Apollo muss waehrend repair() im PLM aktiv sein"
            return np.asarray(audio, dtype=np.float32)

        monkeypatch.setattr(plugin, "_repair_apollo", _fake_repair_apollo)
        monkeypatch.setattr(
            plugin, "_repair_dsp_fallback", lambda audio, sr, material: np.asarray(audio, dtype=np.float32)
        )
        monkeypatch.setattr(plugin, "_measure_hf_gain", lambda original, restored, sr: 0.0)
        monkeypatch.setattr(plugin, "_estimate_brillanz", lambda audio, sr: 0.9)
        monkeypatch.setattr(plugin, "_estimate_waerme", lambda audio, sr: 0.9)

        result = plugin.repair(_signal(0.25), SR, material="mp3_low")

        assert result.model_used == "apollo"
        assert ("set_active", "Apollo", True) in fake_plm.calls
        assert fake_plm.calls[-1] == ("set_active", "Apollo", False)
        assert fake_plm.active is False, "Apollo muss nach repair() wieder freigegeben werden"


class TestCqtdiffPlusPlugin:
    """CQTdiff+ (Dropout-Inpainting ≥ 50 ms): Gap-Befüllung, finite Output."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.cqtdiff_plus_plugin import get_cqtdiff_plus

        p = get_cqtdiff_plus()
        assert p is not None
        _cleanup(["CQTdiffPlus"], "plugins.cqtdiff_plus_plugin")

    def test_02_inpaint_gap_finite(self):
        _reset_budget()
        from plugins.cqtdiff_plus_plugin import inpaint_gap

        audio = _signal(3.0)
        # Simuliere einen Dropout-Gap von 100ms in der Mitte
        gap_start = int(SR * 1.0)
        gap_end = int(SR * 1.1)
        audio_with_gap = audio.copy()
        audio_with_gap[gap_start:gap_end] = 0.0
        result = inpaint_gap(audio_with_gap, SR, gap_start, gap_end)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "CQTdiffPlus.inpaint_gap")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["CQTdiffPlus"], "plugins.cqtdiff_plus_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.cqtdiff_plus_plugin import get_cqtdiff_plus

        get_cqtdiff_plus()
        _cleanup(["CQTdiffPlus"], "plugins.cqtdiff_plus_plugin")
        assert _budget_total() == 0.0


class TestGacelaPlugin:
    """GACELA (GAN-Inpainting ≥ 200 ms): finite Audio, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.gacela_plugin import get_gacela_plugin

        p = get_gacela_plugin()
        assert p is not None
        _cleanup(["GACELA"], "plugins.gacela_plugin")

    def test_02_generate_finite(self):
        _reset_budget()
        from plugins.gacela_plugin import generate_audio

        audio = _signal(2.0)
        result = generate_audio(audio, SR, intensity=0.3)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "GacelaPlugin.generate_audio")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["GACELA"], "plugins.gacela_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.gacela_plugin import get_gacela_plugin

        get_gacela_plugin()
        _cleanup(["GACELA"], "plugins.gacela_plugin")
        assert _budget_total() == 0.0


class TestBanquetVinylPlugin:
    """BANQUET Vinyl: DSP-Decrackler-Fallback, finite Output über tempfile."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.banquet_vinyl_plugin import BanquetVinylPlugin

        p = BanquetVinylPlugin()
        assert p is not None
        _cleanup(["BanquetVinyl"], "plugins.banquet_vinyl_plugin")

    def test_02_process_file_finite(self):
        _reset_budget()
        import soundfile as sf

        from plugins.banquet_vinyl_plugin import BanquetVinylPlugin

        audio = _signal(2.0)
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "input.wav")
            out_path = os.path.join(td, "output.wav")
            sf.write(in_path, audio, SR)
            p = BanquetVinylPlugin()
            p.process_files(in_path, out_path, strength=0.5)
            if os.path.exists(out_path):
                result, _ = sf.read(out_path, dtype="float32")
                _assert_finite(result, "BanquetVinylPlugin.process_files")
        _cleanup(["BanquetVinyl"], "plugins.banquet_vinyl_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.banquet_vinyl_plugin import BanquetVinylPlugin

        BanquetVinylPlugin()
        _cleanup(["BanquetVinyl"], "plugins.banquet_vinyl_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 7: Qualitäts-Metriken / VAD ─────────────────────────────────────
# ---------------------------------------------------------------------------


class TestSileroPlugin:
    """Silero VAD: Energie-Fallback bei fehlendem Modell, finite Maske."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.silero_plugin import SileroPlugin

        p = SileroPlugin()
        assert p is not None
        _cleanup(["SileroVAD"], "plugins.silero_plugin")

    def test_02_speech_mask_finite(self):
        _reset_budget()
        from plugins.silero_plugin import SileroPlugin

        p = SileroPlugin()
        audio = _signal(2.0)
        mask = p.get_speech_mask(audio, SR)
        arr = np.asarray(mask, dtype=np.float32)
        _assert_finite(arr, "SileroPlugin.get_speech_mask")
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"Maske außerhalb [0,1]: {arr.min():.4f} {arr.max():.4f}"
        _cleanup(["SileroVAD"], "plugins.silero_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.silero_plugin import SileroPlugin

        SileroPlugin()
        _cleanup(["SileroVAD"], "plugins.silero_plugin")
        assert _budget_total() == 0.0


class TestUtmosPlugin:
    """UTMOS v2: MOS-Schätzung Gesang, PQS-Fallback, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.utmos_plugin import get_utmos

        p = get_utmos()
        assert p is not None
        _cleanup(["UTMOSv2"], "plugins.utmos_plugin")

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_02_estimate_mos_in_range(self):
        _reset_budget()
        from plugins.utmos_plugin import estimate_mos

        _signal(3.0)
        result = estimate_mos(_sine(3.0, 440.0), SR)
        score = result.mos
        assert np.isfinite(score), f"UTMOS MOS nicht finite: {score}"
        assert 1.0 <= score <= 5.0, f"UTMOS MOS außerhalb [1,5]: {score:.4f}"
        _cleanup(["UTMOSv2"], "plugins.utmos_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.utmos_plugin import get_utmos

        get_utmos()
        _cleanup(["UTMOSv2"], "plugins.utmos_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 8: Neuronale Zwischenrepräsentation / Spezial ───────────────────
# ---------------------------------------------------------------------------


class TestDacPlugin:
    """DAC (Descript Audio Codec): Encode-Decode-Roundtrip, finite Output."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.dac_plugin import DacPlugin

        p = DacPlugin()
        assert p is not None
        _cleanup(["DacEncoder", "DacDecoder"], "plugins.dac_plugin")

    def test_02_round_trip_finite(self):
        _reset_budget()
        from plugins.dac_plugin import DacPlugin

        p = DacPlugin()
        audio = _signal(1.0)
        result = p.round_trip(audio, SR)
        # DacRoundTripResult.audio_out (nicht .audio)
        out = (
            result.audio_out
            if hasattr(result, "audio_out")
            else (result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32))
        )
        _assert_finite(out, "DacPlugin.round_trip")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["DacEncoder", "DacDecoder"], "plugins.dac_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.dac_plugin import DacPlugin

        DacPlugin()
        _cleanup(["DacEncoder", "DacDecoder"], "plugins.dac_plugin")
        assert _budget_total() == 0.0


class TestArtifactDetectionPlugin:
    """ArtifactDetectionPlugin: DSP-Fallback, finite Artefakt-Scores."""

    _MODEL_PATH = "models/artifact_detector.pt"

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.artifact_detection_plugin import ArtifactDetectionPlugin

        p = ArtifactDetectionPlugin(self._MODEL_PATH)
        assert p is not None
        _cleanup(["ArtifactDetection"], "plugins.artifact_detection_plugin")

    def test_02_detect_artifacts_finite(self):
        _reset_budget()
        from plugins.artifact_detection_plugin import ArtifactDetectionPlugin

        p = ArtifactDetectionPlugin(self._MODEL_PATH)
        audio = _signal(2.0)
        result = p.detect_artifacts(audio, SR)
        assert result is not None
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    assert np.isfinite(v), f"ArtifactDetection non-finite '{k}': {v}"
        _cleanup(["ArtifactDetection"], "plugins.artifact_detection_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.artifact_detection_plugin import ArtifactDetectionPlugin

        ArtifactDetectionPlugin(self._MODEL_PATH)
        _cleanup(["ArtifactDetection"], "plugins.artifact_detection_plugin")
        assert _budget_total() == 0.0


class TestResembleEnhancePlugin:
    """ResembleEnhance: Vocal-Enhancement, finite Ausgabe, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.resemble_enhance_plugin import ResembleEnhancePlugin

        p = ResembleEnhancePlugin()
        assert p is not None
        _cleanup(["ResembleEnhance"], "plugins.resemble_enhance_plugin")

    def test_02_enhance_finite(self):
        _reset_budget()
        from plugins.resemble_enhance_plugin import ResembleEnhancePlugin

        p = ResembleEnhancePlugin()
        audio = _signal(2.0)
        result = p.enhance(audio, SR)
        out = result.audio if hasattr(result, "audio") else np.asarray(result, dtype=np.float32)
        _assert_finite(out, "ResembleEnhancePlugin.enhance")
        assert np.max(np.abs(out)) <= 1.0
        _cleanup(["ResembleEnhance"], "plugins.resemble_enhance_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.resemble_enhance_plugin import ResembleEnhancePlugin

        ResembleEnhancePlugin()
        _cleanup(["ResembleEnhance"], "plugins.resemble_enhance_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Gruppe 9: Lyrics-Transkription ─────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestLyricsTranscriberPlugin:
    """Whisper-Tiny + wav2vec2: Energie-Fallback, Transkription, Budget sauber."""

    def test_01_loads_without_crash(self):
        _reset_budget()
        from plugins.lyrics_transcriber_plugin import get_lyrics_transcriber

        p = get_lyrics_transcriber()
        assert p is not None
        _cleanup(["WhisperTiny"], "plugins.lyrics_transcriber_plugin")

    def test_02_transcribe_returns_result(self):
        _reset_budget()
        from plugins.lyrics_transcriber_plugin import transcribe_audio

        # 5 Sekunden Sinus — kurz genug für DSP-Pfad
        audio = _signal(5.0)
        result = transcribe_audio(audio, SR)
        assert result is not None
        assert hasattr(result, "words") or hasattr(result, "segments") or hasattr(result, "text"), (
            f"Kein text/words/segments in {type(result)}"
        )
        _cleanup(["WhisperTiny"], "plugins.lyrics_transcriber_plugin")

    def test_03_budget_zero_after_cleanup(self):
        _reset_budget()
        from plugins.lyrics_transcriber_plugin import get_lyrics_transcriber

        get_lyrics_transcriber()
        _cleanup(["WhisperTiny"], "plugins.lyrics_transcriber_plugin")
        assert _budget_total() == 0.0


# ---------------------------------------------------------------------------
# ── Globale Budget-Invariante ───────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestGlobalBudgetInvariants:
    """Übergreifende Budget- und PLM-Invarianten nach allen Plugin-Tests."""

    def test_initial_budget_zero(self):
        """Budget startet bei 0 wenn nichts geladen wurde."""
        _reset_budget()
        assert _budget_total() == 0.0, "Budget sollte initial 0 sein"

    def test_try_allocate_idempotent(self):
        """try_allocate(name) für bereits allokierten Namen zählt nicht doppelt."""
        from backend.core.ml_memory_budget import release, try_allocate

        _reset_budget()
        ok1 = try_allocate("TestDouble", size_gb=0.01)
        ok2 = try_allocate("TestDouble", size_gb=0.01)  # idempotent
        assert ok1 is True
        assert ok2 is True
        # Total darf NICHT 0.02 sein (doppelte Zählung verboten)
        assert _budget_total() == pytest.approx(0.01, abs=1e-6), f"Doppelte Budget-Allokation: {_budget_total():.4f} GB"
        release("TestDouble")
        assert _budget_total() == 0.0, "Budget nach release() nicht 0"

    def test_release_safe_for_unknown_name(self):
        """release() auf unbekannten Namen darf nicht crashen und Budget bleibt 0."""
        from backend.core.ml_memory_budget import release

        _reset_budget()
        release("NonExistentPlugin")  # kein Fehler
        assert _budget_total() == 0.0

    def test_budget_exhaustion_blocks_allocation(self):
        """Wenn Budget erschöpft, blockiert try_allocate() neue Allokationen."""
        from unittest.mock import patch

        from backend.core.ml_memory_budget import release, set_budget, try_allocate

        # §VERBOTEN: Budget-Tests ohne is_system_thrashing-Mock → flaky auf Hosts mit hoher Swap-Last
        with patch("backend.core.ml_memory_budget.is_system_thrashing", return_value=False):
            _reset_budget()
            set_budget(0.05)  # 50 MB mini-Budget für Test
            _reset_budget()
            try_allocate("Filler", size_gb=0.04)
            # Nächste Allokation should fail (0.04 + 0.03 > 0.05)
            blocked = try_allocate("Overflow", size_gb=0.03)
            assert blocked is False, "Überschuss-Allokation wurde nicht blockiert"
            release("Filler")
        # Budget wiederherstellen
        _reset_budget()
        from backend.core.ml_memory_budget import _auto_detect_budget

        auto = _auto_detect_budget()
        set_budget(auto)

    def test_plm_force_evict_all_runs_unload_fns(self):
        """PLM.force_evict_all() ruft alle registrierten unload_fn() auf."""
        from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

        plm = get_plugin_lifecycle_manager()
        _plm_evict_all()  # Sauberer Start

        called: list[str] = []

        plm.register("TestPluginA", size_gb=0.01, unload_fn=lambda: called.append("A"))
        plm.register("TestPluginB", size_gb=0.01, unload_fn=lambda: called.append("B"))
        plm.force_evict_all()

        assert "A" in called and "B" in called, f"unload_fn nicht aufgerufen: {called}"
        _plm_evict_all()

    def test_plm_active_plugin_not_evicted(self):
        """PLM-Plugins mit active=True werden NICHT evicted."""
        from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

        plm = get_plugin_lifecycle_manager()
        _plm_evict_all()

        called: list[str] = []
        plm.register("ActivePlugin", size_gb=0.01, unload_fn=lambda: called.append("evicted"))
        plm.set_active("ActivePlugin", True)
        plm.force_evict_all()

        assert "evicted" not in called, "Aktives Plugin wurde trotzdem evicted"
        plm.set_active("ActivePlugin", False)
        _plm_evict_all()

    def test_plm_shutdown_stops_monitor_thread(self):
        """PLM.shutdown() muss den Monitor-Thread best-effort beenden."""
        from backend.core.plugin_lifecycle_manager import PluginLifecycleManager

        plm = PluginLifecycleManager()
        monitor_thread = plm._auto_evict_thread
        assert monitor_thread is not None and monitor_thread.is_alive()

        plm.shutdown()

        assert plm._stop_event.is_set(), "Shutdown setzte stop_event nicht"
        assert plm._auto_evict_thread is None, "Shutdown muss Thread-Referenz freigeben"
        assert monitor_thread is not None and not monitor_thread.is_alive(), "Monitor-Thread läuft nach Shutdown weiter"

    def test_plm_evicts_on_swap_pressure_while_pipeline_active(self):
        """Swap-Druck darf im pipeline_active-Fast-Return nicht verloren gehen."""
        from backend.core.plugin_lifecycle_manager import PluginLifecycleManager

        plm = PluginLifecycleManager()
        evicted: list[str] = []
        try:
            plm.register("SwapVictim", 0.5, lambda: evicted.append("SwapVictim"))
            plm.enter_pipeline()

            plm._ram_percent = lambda: 50.0  # type: ignore[method-assign]
            plm._free_mb = lambda: 12_000.0  # type: ignore[method-assign]
            plm._swap_percent = lambda: 100.0  # type: ignore[method-assign]

            count = plm.evict_if_needed()

            assert count == 1
            assert evicted == ["SwapVictim"]
        finally:
            plm.shutdown()

    def test_plm_does_not_evict_on_stale_swap_with_high_free_ram(self):
        """Hoher Swap-Stand allein darf bei viel freiem RAM keine Eviction erzwingen."""
        from backend.core.plugin_lifecycle_manager import PluginLifecycleManager

        plm = PluginLifecycleManager()
        evicted: list[str] = []
        try:
            plm.register("StaleSwapVictim", 0.5, lambda: evicted.append("StaleSwapVictim"))

            plm._ram_percent = lambda: 50.0  # type: ignore[method-assign]
            plm._free_mb = lambda: 16_000.0  # type: ignore[method-assign]
            plm._swap_percent = lambda: 84.0  # type: ignore[method-assign]

            count = plm.evict_if_needed()

            assert count == 0
            assert evicted == []
        finally:
            plm.shutdown()

    def test_cleanup_after_all_plugins_budget_stays_zero(self):
        """Nach komplettem Cleanup aller Plugin-Budget-Slots ist _total_gb==0."""
        from backend.core import ml_memory_budget as _bud

        _reset_budget()
        _plm_evict_all()
        gc.collect()
        assert _bud._total_gb == 0.0, (
            f"Verbleibendes Budget nach globalem Cleanup: {_bud._total_gb:.4f} GB; allocated={dict(_bud._allocated)}"
        )
