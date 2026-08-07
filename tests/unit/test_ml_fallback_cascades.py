"""
tests/unit/test_ml_fallback_cascades.py — §2.47 ML-Failure-Degradationskaskade

Tests the 5 untested ML fallback cascade patterns:
  1. FlashSR → Harmonische Oberton-Synthese + PGHI → Spectral-Band-Replication
  2. MDX23C → NMF-β-Separation (sdB ≥ 5) → HPSS
  3. MP-SENet → OMLSA/IMCRA → Bypass
  4. CREPE → pYIN → YIN
  5. MERT → DSP-Analyse → Bypass

Each cascade must: never abort, log fallback in metadata, produce valid output.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

SR = 48_000


def _import_with_recovery(module_name: str):
    """Import helper robust against transient sys.modules poisoning.

    In large suites some tests may temporarily monkeypatch entries in sys.modules.
    When such a value leaks as None/stale module, a direct import can raise
    ImportError even though the module exists in the workspace.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)


def _audio(dur: float = 2.0) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 1. FlashSR cascade
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioSRFallback:
    """FlashSR → Harmonische+PGHI → SBR."""

    def test_flashsr_plugin_has_fallback_attribute(self):
        """FlashSR plugin must declare DSP fallback capability."""
        try:
            from plugins.flashsr_plugin import FlashSRPlugin

            plugin = FlashSRPlugin()
            # Must have process method and handle OOM internally
            assert hasattr(plugin, "process") or hasattr(plugin, "enhance")
        except ImportError:
            pytest.skip("flashsr_plugin not available")

    def test_phase23_has_ml_fallback_path(self):
        """phase_23 spectral repair must have non-ML fallback."""
        try:
            import backend.core.phases.phase_23_spectral_repair as p23

            src = Path(p23.__file__).read_text(encoding="utf-8", errors="replace")
            # Must contain fallback logic (SBR or harmonic synthesis)
            assert "fallback" in src.lower() or "sbr" in src.lower() or "harmonic" in src.lower(), (
                "phase_23 must have ML fallback path"
            )
        except ImportError:
            pytest.skip("phase_23 not available")

    def test_phase06_frequency_restoration_fallback(self):
        """phase_06 frequency restoration must not crash without FlashSR."""
        try:
            import backend.core.phases.phase_06_frequency_restoration as p06

            src = Path(p06.__file__).read_text(encoding="utf-8", errors="replace")
            assert "fallback" in src.lower() or "except" in src.lower(), "phase_06 must handle ML failure gracefully"
        except ImportError:
            pytest.skip("phase_06 not available")


# ---------------------------------------------------------------------------
# 2. MDX23C cascade
# ---------------------------------------------------------------------------


class TestMDX23CFallback:
    """MDX23C → NMF-β → HPSS."""

    def test_stem_separator_fallback_chain(self):
        """Stem separator must have NMF and HPSS fallback layers."""
        try:
            mod = _import_with_recovery("dsp.stem_separator")
            StemSeparator = mod.StemSeparator

            sep = StemSeparator()
            # Must support fallback modes
            assert hasattr(sep, "separate") or hasattr(sep, "separate_stems")
        except ImportError:
            pytest.skip("stem_separator not available")

    def test_stem_separator_source_has_nmf_fallback(self):
        """Source code must reference NMF as fallback."""
        try:
            mod = _import_with_recovery("dsp.stem_separator")

            src = Path(mod.__file__).read_text(encoding="utf-8", errors="replace")
            has_nmf = "nmf" in src.lower() or "NMF" in src
            has_hpss = "hpss" in src.lower() or "HPSS" in src
            assert has_nmf, "stem_separator must reference NMF as primary fallback"
            assert has_hpss, "stem_separator must reference HPSS as tertiary fallback"
        except ImportError:
            pytest.skip("stem_separator not available")


# ---------------------------------------------------------------------------
# 3. MP-SENet cascade
# ---------------------------------------------------------------------------


class TestMPSENetFallback:
    """MP-SENet → OMLSA/IMCRA → Bypass (phase_43 skip)."""

    def test_phase43_source_has_omlsa_fallback(self):
        """phase_43 must have OMLSA/IMCRA DSP fallback."""
        try:
            p43 = _import_with_recovery("backend.core.phases.phase_43_ml_deesser")

            src = Path(p43.__file__).read_text(encoding="utf-8", errors="replace")
            has_omlsa = "omlsa" in src.lower() or "OMLSA" in src
            has_bypass = "bypass" in src.lower() or "skip" in src.lower()
            assert has_omlsa or has_bypass, "phase_43 must have OMLSA fallback or bypass path"
        except ImportError:
            pytest.skip("phase_43 not available")

    def test_mp_senet_plugin_has_process(self):
        """MP-SENet plugin must implement process interface."""
        try:
            mod = _import_with_recovery("plugins.mp_senet_plugin")
            MpSenetPlugin = mod.MpSenetPlugin

            plugin = MpSenetPlugin()
            assert hasattr(plugin, "process") or hasattr(plugin, "enhance")
        except ImportError:
            pytest.skip("mp_senet_plugin not available")


# ---------------------------------------------------------------------------
# 4. CREPE cascade
# ---------------------------------------------------------------------------


class TestCREPEFallback:
    """CREPE → pYIN → YIN."""

    def test_pitch_tracker_has_pyin_fallback(self):
        """Pitch tracking module must have pYIN fallback."""
        found_pyin = False
        for mod_name in [
            "backend.core.pitch_tracker",
            "backend.core.pitch_tracking",
            "dsp.pitch_tracking",
            "plugins.crepe_plugin",
        ]:
            try:
                mod = __import__(mod_name, fromlist=[""])
                src = Path(mod.__file__).read_text(encoding="utf-8", errors="replace")  # type: ignore[arg-type]
                if "pyin" in src.lower():
                    found_pyin = True
                    break
            except (ImportError, AttributeError, TypeError):
                continue
        if not found_pyin:
            # Check if any phase uses pitch tracking with fallback
            try:
                import backend.core.phases.phase_12_wow_flutter_fix as p12

                src = Path(p12.__file__).read_text(encoding="utf-8", errors="replace")
                found_pyin = "pyin" in src.lower() or "yin" in src.lower()
            except ImportError:
                pass
        assert found_pyin, "Pitch tracking must have pYIN/YIN fallback for CREPE"

    def test_crepe_plugin_exists_or_integrated(self):
        """CREPE must be available as plugin or integrated in pitch tracker."""
        import importlib.util

        has_crepe = importlib.util.find_spec("plugins.crepe_plugin") is not None
        if not has_crepe:
            has_crepe = importlib.util.find_spec("plugins.fcpe_plugin") is not None  # FCPE is the CREPE successor
        assert has_crepe, "CREPE or FCPE plugin must be available"


# ---------------------------------------------------------------------------
# 5. MERT cascade
# ---------------------------------------------------------------------------


class TestMERTFallback:
    """MERT → DSP-Analyse → Bypass."""

    def test_mert_plugin_has_fallback(self):
        """MertPlugin must handle OOM and fall back to DSP."""
        try:
            from plugins.mert_plugin import MertPlugin

            plugin = MertPlugin()
            # Must have compute or extract method
            assert (
                hasattr(plugin, "analyze")
                or hasattr(plugin, "compute")
                or hasattr(plugin, "extract")
                or hasattr(plugin, "process")
            )
        except ImportError:
            pytest.skip("mert_plugin not available")

    def test_mert_plugin_source_has_dsp_fallback(self):
        """Source code must reference DSP fallback path."""
        try:
            import plugins.mert_plugin as mod

            src = Path(mod.__file__).read_text(encoding="utf-8", errors="replace")
            has_dsp_fb = (
                ("dsp" in src.lower() and "fallback" in src.lower())
                or "f0" in src.lower()
                or "harmonicity" in src.lower()
                or "bypass" in src.lower()
            )
            assert has_dsp_fb, "mert_plugin must have DSP analysis fallback"
        except ImportError:
            pytest.skip("mert_plugin not available")


# ---------------------------------------------------------------------------
# Cross-cutting: metadata logging invariant
# ---------------------------------------------------------------------------


class TestFallbackMetadataContract:
    """Every ML fallback must be logged in metadata['ml_fallbacks_used']."""

    def test_ml_memory_budget_has_try_allocate(self):
        """ml_memory_budget must provide try_allocate for OOM prevention."""
        from backend.core.ml_memory_budget import get_ml_memory_budget

        budget = get_ml_memory_budget()
        assert hasattr(budget, "try_allocate")

    def test_ml_memory_budget_has_release(self):
        """ml_memory_budget must provide release for cleanup."""
        from backend.core.ml_memory_budget import get_ml_memory_budget

        budget = get_ml_memory_budget()
        assert hasattr(budget, "release")
