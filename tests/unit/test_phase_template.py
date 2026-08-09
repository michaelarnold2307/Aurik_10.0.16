#!/usr/bin/env python3
"""tests/unit/test_phase_template.py — §v10.700 I1: Per-Phase-Unit-Test-Template.

Kopiere diese Datei als Vorlage für jede neue Phase:
  cp tests/unit/test_phase_template.py tests/unit/test_phase_XX_name.py

Dann ersetze:
  - PhaseClass → Name der Phase-Klasse (z.B. ClickRemovalPhase)
  - PhaseImport → Import-Pfad (z.B. backend.core.phases.phase_01_click_removal)
  - phase_id → ID-String (z.B. "phase_01_click_removal")

Pflichttests pro Phase:
  1. test_process_returns_ndarray      — Output ist np.ndarray
  2. test_no_nan_inf                   — Keine NaN/Inf im Output
  3. test_strength_zero_is_passthrough — Strength=0 verändert Audio nicht
  4. test_stereo_preserved             — Stereo-Input bleibt Stereo
  5. test_mono_input_handled           — Mono-Input wird akzeptiert
"""

from __future__ import annotations

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# KONFIGURATION — pro Phase anpassen
# ═══════════════════════════════════════════════════════════════════════════

PHASE_CLASS = None  # TODO: z.B. ClickRemovalPhase
PHASE_IMPORT = ""  # TODO: z.B. "backend.core.phases.phase_01_click_removal"
PHASE_ID = ""  # TODO: z.B. "phase_01_click_removal"
SAMPLE_RATE = 48000
DURATION = 1.0  # Sekunden

# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mono_audio() -> np.ndarray:
    """Synthetisches Mono-Audio: 440 Hz Sinus mit leichten Obertönen."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sig += 0.15 * np.sin(2 * np.pi * 880.0 * t)
    sig += rng.randn(len(t)).astype(np.float32) * 0.01  # Minimales Rauschen
    return sig.astype(np.float32)  # type: ignore[no-any-return]


@pytest.fixture
def stereo_audio() -> np.ndarray:
    """Synthetisches Stereo-Audio: L/R leicht unterschiedlich."""
    mono = mono_audio()
    stereo = np.stack([mono, mono * 0.95], axis=-1)
    return stereo.astype(np.float32)  # type: ignore[no-any-return]


@pytest.fixture
def phase_instance():
    """Erzeugt eine Phase-Instanz."""
    if PHASE_CLASS is None:
        pytest.skip("PHASE_CLASS nicht konfiguriert — Template-Datei")
    # Versuche die Phase zu importieren und zu instanziieren
    import importlib

    module_path, class_name = (
        PHASE_IMPORT.rsplit(".", 1) if "." in PHASE_IMPORT else (PHASE_IMPORT, PHASE_CLASS.__name__)
    )
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name if class_name else (PHASE_CLASS.__name__ if PHASE_CLASS is not None else ""), None)
    return cls(sample_rate=SAMPLE_RATE)  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# PFLICHTTESTS — für jede Phase
# ═══════════════════════════════════════════════════════════════════════════


def test_process_returns_ndarray(phase_instance, mono_audio):
    """Output MUSS ein np.ndarray sein."""
    if PHASE_CLASS is None:
        pytest.skip("Template")
    result = phase_instance.process(mono_audio, sample_rate=SAMPLE_RATE, material_type="vinyl")
    # Phase kann ndarray oder NamedTuple/Objekt mit .audio zurückgeben
    if hasattr(result, "audio"):
        audio = result.audio
    elif isinstance(result, np.ndarray):
        audio = result
    else:
        audio = result[0] if isinstance(result, tuple) else result
    assert isinstance(audio, np.ndarray), f"Output type: {type(audio)}"
    assert audio.dtype == np.float32, f"Output dtype: {audio.dtype}"


def test_no_nan_inf(phase_instance, mono_audio):
    """Output darf KEINE NaN oder Inf enthalten."""
    if PHASE_CLASS is None:
        pytest.skip("Template")
    result = phase_instance.process(mono_audio, sample_rate=SAMPLE_RATE, material_type="vinyl")
    audio = _extract_audio(result)
    assert not np.any(np.isnan(audio)), "Output enthält NaN"
    assert not np.any(np.isinf(audio)), "Output enthält Inf"


def test_strength_zero_is_passthrough(phase_instance, mono_audio):
    """Strength=0 MUSS das Audio nahezu unverändert lassen."""
    if PHASE_CLASS is None:
        pytest.skip("Template")
    result = phase_instance.process(mono_audio, sample_rate=SAMPLE_RATE, material_type="vinyl", strength=0.0)
    audio = _extract_audio(result)
    # Gleiche Länge, ähnlicher Pegel (±3 dB)
    assert len(audio) == len(mono_audio), f"Länge geändert: {len(audio)} vs {len(mono_audio)}"
    rms_in = float(np.sqrt(np.mean(mono_audio**2)) + 1e-12)
    rms_out = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    rms_ratio = max(rms_in, rms_out) / (min(rms_in, rms_out) + 1e-12)
    assert rms_ratio < 2.0, f"RMS-Ratio: {rms_ratio:.2f} (zu große Abweichung)"


def test_stereo_preserved(phase_instance, stereo_audio):
    """Stereo-Input MUSS Stereo-Output erzeugen (2 Kanäle)."""
    if PHASE_CLASS is None:
        pytest.skip("Template")
    result = phase_instance.process(stereo_audio, sample_rate=SAMPLE_RATE, material_type="vinyl")
    audio = _extract_audio(result)
    assert audio.ndim >= 1, "Output hat keine Dimension"
    if audio.ndim == 2:
        assert audio.shape[-1] == 2, f"Stereo-Kanäle: {audio.shape[-1]} (erwartet 2)"


def test_mono_input_handled(phase_instance, mono_audio):
    """Mono-Input MUSS ohne Crash verarbeitet werden."""
    if PHASE_CLASS is None:
        pytest.skip("Template")
    result = phase_instance.process(mono_audio, sample_rate=SAMPLE_RATE, material_type="vinyl")
    audio = _extract_audio(result)
    assert audio.ndim == 1, f"Mono-Output sollte 1D sein, ist {audio.ndim}D"


# ═══════════════════════════════════════════════════════════════════════════
# HELFER
# ═══════════════════════════════════════════════════════════════════════════


def _extract_audio(result) -> np.ndarray:
    """Extrahiert np.ndarray aus verschiedenen Result-Typen."""
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "audio"):
        return result.audio  # type: ignore[no-any-return]
    if isinstance(result, tuple) and len(result) > 0:
        return result[0]  # type: ignore[no-any-return]
    return np.asarray(result)
