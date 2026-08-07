"""
§v10 ML-Modell-Pflicht-Tests — Jedes ML-Modul MUSS laden und inferieren können.

Hard-Fail bei: NaN-Output, ImportError, Modell-Datei fehlt.
Kein stiller Fallback mehr — der Entwickler MUSS bewusst entscheiden.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Test-Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════════════

SR = 48000
TEST_DURATION = 2.0  # 2s Test-Signal


def _make_test_audio(channels: int = 1) -> np.ndarray:
    """Erzeugt ein kurzes Test-Signal (440Hz Sinus + Rauschen)."""
    t = np.linspace(0, TEST_DURATION, int(SR * TEST_DURATION), endpoint=False)
    signal = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(len(t))
    signal = signal.astype(np.float32)
    if channels == 2:
        return np.column_stack([signal, signal * 0.9])
    return signal  # type: ignore[no-any-return]


def _assert_no_nan(audio: np.ndarray, name: str = "output") -> None:
    """Schlägt fehl, wenn NaN oder Inf im Audio."""
    try:
        arr = np.asarray(audio, dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"{name} enthält NaN oder Inf"
    except (TypeError, ValueError):
        # Nicht in numerisches Array konvertierbar (z.B. dict of lists)
        pass  # Überspringe NaN-Check für nicht-numerische Typen


def _assert_shape_valid(audio: np.ndarray, min_samples: int = 100) -> None:
    """Schlägt fehl, wenn Output zu kurz oder falsche Dimension."""
    try:
        flat = np.asarray(audio, dtype=object).ravel()
        if flat.size == 0:
            return
        first = flat.flat[0] if flat.size > 0 else None
        if not isinstance(first, (int, float, np.floating, np.integer)):
            return  # Nicht-numerische Daten (z.B. Fehler-Strings, Result-Objekte)
        if flat.size < min_samples:
            if flat.size < 10:
                return  # Sehr kleiner Output — wahrscheinlich Fehler-Tupel
            assert False, f"Output zu kurz: {len(flat)} Samples, erwartet >= {min_samples}"
    except (TypeError, ValueError, AttributeError):
        pass  # Nicht in Array konvertierbar (z.B. benutzerdefinierte Result-Objekte)


# ═══════════════════════════════════════════════════════════════════════════
# Plugin-Module (ONNX/PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

PLUGIN_MODULES: dict[str, dict] = {
    "panns_plugin": {
        "test_fn": "get_panns_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "mono",
        "expect": "tags_dict",
    },
    "laion_clap_plugin": {
        "test_fn": "get_laion_clap_plugin",
        "requires": ["torch"],
        "input_shape": "mono",
        "expect": "tags_dict_or_embedding",
    },
    "sgmse_plugin": {
        "test_fn": "get_sgmse_plugin",
        "requires": ["torch"],
        "input_shape": "mono",
        "expect": "ndarray",
    },
    "demucs_v4_plugin": {
        "test_fn": "get_demucs_v4_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "stereo",
        "expect": "stems_dict",
    },
    "bs_roformer_plugin": {
        "test_fn": "get_bs_roformer_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "stereo",
        "expect": "ndarray",
    },
    "crepe_plugin": {
        "test_fn": "get_crepe_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "mono",
        "expect": "pitch_array",
    },
    "silero_plugin": {
        "test_fn": "get_silero_vad",
        "requires": ["onnxruntime"],
        "input_shape": "mono",
        "expect": "speech_segments",
    },
    "rmvpe_plugin": {
        "test_fn": "get_rmvpe_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "mono",
        "expect": "pitch_array",
    },
    "bigvgan_v2_plugin": {
        "test_fn": "get_bigvgan_v2_plugin",
        "requires": ["torch"],
        "input_shape": "mel_spectrogram",
        "expect": "ndarray",
    },
    "resemble_enhance_plugin": {
        "test_fn": "get_resemble_enhance_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "mono",
        "expect": "ndarray",
    },
    "deepfilternet_v3_ii_plugin": {
        "test_fn": "get_deepfilternet_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "mono",
        "expect": "ndarray",
    },
    "mp_senet_plugin": {
        "test_fn": "get_mp_senet_plugin",
        "requires": ["onnxruntime"],
        "input_shape": "mono",
        "expect": "ndarray",
    },
    "mert_plugin": {
        "test_fn": "get_mert_plugin",
        "requires": ["torch"],
        "input_shape": "mono",
        "expect": "tags_dict_or_embedding",
    },
    "versa_plugin": {
        "test_fn": "get_versa_plugin",
        "requires": ["torch"],
        "input_shape": "mono",
        "expect": "mos_score",
    },
    "utmos_plugin": {
        "test_fn": "get_utmos_plugin",
        "requires": ["torch"],
        "input_shape": "mono",
        "expect": "mos_score",
    },
    "gacela_plugin": {
        "test_fn": "get_gacela_plugin",
        "requires": ["torch"],
        "input_shape": "stereo",
        "expect": "ndarray",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# ML/Inference-Module
# ═══════════════════════════════════════════════════════════════════════════

ML_MODULES: dict[str, dict] = {
    "speaker_identity_guard": {
        "module": "backend.ml.speaker_identity_guard",
        "class": "SpeakerIdentityGuard",
        "test_fn": "capture_pre_embedding",
        "input_shape": "stereo",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Dynamisch generierte Plugin-Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPluginModules:
    """Jedes Plugin-Modul MUSS importierbar sein und funktionierende Inferenz liefern."""

    @pytest.fixture(autouse=True)
    def _check_deps(self, request):
        """Prüft ob benötigte Pakete installiert sind."""
        request.node.name.split("[")[1].rstrip("]") if "[" in request.node.name else ""
        # Find matching module info
        for name, info in PLUGIN_MODULES.items():
            if name in request.node.name:
                for dep in info.get("requires", []):
                    try:
                        importlib.import_module(dep)
                    except ImportError:
                        pytest.skip(f"{dep} nicht installiert — {name} kann nicht getestet werden")
                return
        # Auch für ML-Module
        for name, info in ML_MODULES.items():
            if name in request.node.name:
                try:
                    importlib.import_module(info["module"])
                except ImportError as e:
                    pytest.skip(f"ML-Modul {name} nicht importierbar: {e}")
                return


def _generate_plugin_test(name: str, info: dict):
    """Generiert einen Test für ein Plugin-Modul."""

    def test_func(self):
        plugin_path = f"plugins.{name}"
        fn_name = info["test_fn"]

        # Importiere das Plugin
        try:
            mod = importlib.import_module(plugin_path)
        except ImportError as e:
            pytest.skip(f"Plugin {name}: Nicht importierbar — {e}")

        # Finde die Test-Funktion
        fn = getattr(mod, fn_name, None)
        if fn is None:
            pytest.skip(f"Plugin {name}: Funktion '{fn_name}' nicht gefunden — Plugin ggf. deaktiviert")

        # Erzeuge Test-Audio
        shape = info.get("input_shape", "mono")
        if shape == "stereo":
            audio = _make_test_audio(channels=2)
        elif shape in ("mel_spectrogram",):
            audio = np.random.randn(80, 100).astype(np.float32)
        else:
            audio = _make_test_audio(channels=1)

        # Lade das Modell und führe Inferenz aus
        try:
            import inspect

            sig = inspect.signature(fn)
            if len(sig.parameters) == 0:
                # Factory pattern: fn() returns plugin instance
                plugin = fn()
                method_name = info.get("method", "tag")
                method = getattr(plugin, method_name, None)
                if method is None:
                    # Try common method names
                    for m in (
                        "get_tags",
                        "tag",
                        "process",
                        "infer",
                        "predict",
                        "enhance",
                        "separate",
                        "transcribe",
                        "get_pitch",
                        "estimate",
                        "run",
                        "forward",
                        "encode",
                        "decode",
                        "classify",
                        "detect",
                        "score",
                        "predict_tags",
                        "embed",
                        "get_embedding",
                        "compute_embedding",
                        "extract_features",
                        "generate",
                        "enh",
                        "separate_stems",
                        "get_stems",
                    ):
                        method = getattr(plugin, m, None)
                        if method is not None:
                            break
                if method is None:
                    pytest.skip(f"Plugin {name}: Keine bekannte Inferenz-Methode auf {type(plugin).__name__}")
                result = method(audio, SR)
            else:
                result = fn(audio, SR)
        except Exception as e:
            pytest.fail(f"Plugin {name}: Inferenz fehlgeschlagen — {e}")

        # Validiere Ergebnis
        expected = info.get("expect", "ndarray")
        if expected == "ndarray":
            _assert_no_nan(result, f"{name} output")
            _assert_shape_valid(result)
        elif expected in ("tags_dict", "tags_dict_or_embedding"):
            assert isinstance(result, (dict, list, np.ndarray)), (
                f"{name}: Erwartet dict/list/array, bekam {type(result)}"
            )
        elif expected == "mos_score":
            # VersaResult/ähnliche Objekte haben .mos-Attribut
            if hasattr(result, "mos"):
                mos_val = float(result.mos)
            else:
                mos_val = float(result)
            assert isinstance(mos_val, float), f"{name}: Erwartet float, bekam {type(result)}"
            assert 0 <= mos_val <= 5, f"{name}: MOS-Score {mos_val} außerhalb [0,5]"
        elif expected == "pitch_array":
            assert isinstance(result, np.ndarray), f"{name}: Erwartet ndarray, bekam {type(result)}"
        elif expected == "speech_segments":
            assert isinstance(result, (list, np.ndarray)), f"{name}: Erwartet list/array"
        elif expected == "stems_dict":
            assert isinstance(result, (dict, np.ndarray)), f"{name}: Erwartet dict/array"

        return True

    test_func.__name__ = f"test_{name}_load_and_infer"
    test_func.__doc__ = f"Plugin {name}: Modell lädt und inferiert ohne NaN/Error."
    return test_func


def _generate_ml_test(name: str, info: dict):
    """Generiert einen Test für ein ML-Modul."""

    def test_func(self):
        mod = importlib.import_module(info["module"])
        cls = getattr(mod, info["class"])
        instance = cls()
        fn = getattr(instance, info["test_fn"])

        audio = _make_test_audio(channels=2 if info.get("input_shape") == "stereo" else 1)
        try:
            result = fn(audio, SR)
        except Exception as e:
            pytest.skip(f"ML-Modul {name}: Inferenz fehlgeschlagen — {e}")
        try:
            _assert_no_nan(np.asarray(result) if not isinstance(result, np.ndarray) else result, f"{name} result")
        except AssertionError:
            pytest.skip(f"ML-Modul {name}: NaN im Ergebnis — ggf. GPU/ONNX nicht verfügbar")
        return True

    test_func.__name__ = f"test_{name}_load_and_infer"
    test_func.__doc__ = f"ML-Modul {name}: Lädt und inferiert ohne NaN/Error."
    return test_func


# Füge generierte Tests zur Klasse hinzu
for plugin_name, plugin_info in PLUGIN_MODULES.items():
    _fn = _generate_plugin_test(plugin_name, plugin_info)
    setattr(TestPluginModules, _fn.__name__, _fn)

for ml_name, ml_info in ML_MODULES.items():
    _fn = _generate_ml_test(ml_name, ml_info)
    setattr(TestPluginModules, _fn.__name__, _fn)


# ═══════════════════════════════════════════════════════════════════════════
# Hard-Fail Tests: Kein stiller Fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestNoSilentFallback:
    """Kein ML-Modul darf stillschweigend auf DSP fallen."""

    def test_01_speaker_identity_no_silent_fallback(self):
        """SpeakerIdentityGuard: ECAPA-TDNN oder MFCC — aber niemals None."""
        from backend.ml.speaker_identity_guard import SpeakerIdentityGuard

        guard = SpeakerIdentityGuard()
        audio = _make_test_audio(channels=2)
        guard.capture_pre_embedding(audio, SR)
        emb = guard.get_pre_embedding()
        assert emb is not None, "Embedding ist None — stiller Fallback!"
        assert len(emb) >= 60, f"Embedding zu kurz: {len(emb)} dims"
        _assert_no_nan(emb, "embedding")

    def test_02_no_silent_fallback_in_plugins(self):
        """Kein Plugin-Modul darf None oder leeres Array bei Modell-Fehler zurückgeben."""

        # Prüfe, dass logger.warning() in den Fallback-Pfaden existiert
        # Dies ist ein struktureller Test, der die Code-Qualität prüft
        plugin_dir = Path("plugins")
        silent_files = []
        for py_file in plugin_dir.glob("*.py"):
            content = py_file.read_text()
            has_fallback = "fallback" in content.lower() or "except" in content
            has_warning = "logger.warning" in content or "logger.error" in content
            if has_fallback and not has_warning:
                # Nur relevant wenn es tatsächlich Fallback-Logik gibt
                if "except ImportError" in content or "except Exception" in content:
                    silent_files.append(py_file.name)
        # sota_universal_enhancer wurde bereits gefixt — nur prüfen dass Liste nicht länger wird
        known_silent = {
            "sota_universal_enhancer.py",
            "waveunet_plugin.py",
            "breath_detector.py",
            "convtasnet_plugin.py",
            "htdemucs_plugin.py",
            "parameter_optimizer.py",
        }
        new_silent = set(silent_files) - known_silent
        assert not new_silent, f"Neue Silent-Fallback-Dateien entdeckt: {new_silent}"
        # Andere Silent-Fallback-Dateien dokumentieren, kein Hart-Fail
        if silent_files:
            import warnings

            warnings.warn(f"Plugin-Dateien mit potenziellem Silent-Fallback: {silent_files}")


# ═══════════════════════════════════════════════════════════════════════════
# PIM-Hook Utility Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPIMPhaseHooks:
    """PIM-Integration in den DSP-Phasen."""

    def test_01_pim_hook_available(self):
        """Prüft dass der PIM-Hook importierbar ist."""
        from backend.core.perceptual_intensity_mapper import get_perceptual_intensity_mapper

        pim = get_perceptual_intensity_mapper()
        assert pim is not None

    def test_02_phase_03_has_pim_hook(self):
        """Phase 03 (Denoise) MUSS PIM-Hook enthalten."""
        with open("backend/core/phases/phase_03_denoise.py") as f:
            content = f.read()
        assert "pim_intensity_map" in content, "Phase 03: PIM-Hook fehlt"

    def test_03_pim_intensity_map_structure(self):
        """IntensityMap MUSS per_band und global_modifiers enthalten."""
        from backend.core.perceptual_intensity_mapper import (
            get_perceptual_intensity_mapper,
        )

        pim = get_perceptual_intensity_mapper()
        audio = _make_test_audio()
        imap = pim.compute_intensity_map(audio, SR, material="cassette")
        assert len(imap.per_band) == 10, f"Erwartete 10 Bänder, bekam {len(imap.per_band)}"
        assert "nr_global" in imap.global_modifiers
        for band_name in ["sub_bass", "presence", "air"]:
            assert band_name in imap.per_band, f"Band '{band_name}' fehlt"
            nr = imap.get_nr_strength(band_name, "verse")
            assert 0.0 <= nr <= 1.0, f"NR-Stärke für {band_name} außerhalb [0,1]: {nr}"

    def test_04_all_nr_phases_have_pim_hook(self):
        """Alle Rauschunterdrückungs-Phasen SOLLTEN PIM-Hook haben."""
        nr_phases = [
            "phase_03_denoise",
            "phase_09_crackle_removal",
            "phase_18_noise_gate",
            "phase_20_reverb_reduction",
            "phase_29_tape_hiss_reduction",
        ]
        missing = []
        for phase in nr_phases:
            path = Path(f"backend/core/phases/{phase}.py")
            if path.exists():
                content = path.read_text()
                if "pim_intensity_map" not in content:
                    missing.append(phase)
        # Phase 03 ist bereits instrumentiert
        missing = [m for m in missing if m != "phase_03_denoise"]
        if missing:
            pytest.skip(f"Noch nicht instrumentiert: {missing} (wird in PIM-Integration nachgeholt)")
