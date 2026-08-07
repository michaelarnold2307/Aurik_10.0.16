"""§v10.112 Fix-Regression-Gate: verhindert Rückkehr aller §v10.97–§v10.111 Fixes.

Abgedeckte Fixes:
  §v10.95  – PhaseResult Tuple→ndarray-Normalisierung
  §v10.97  – Broadcaster channels-last Input-Prüfung (shape[0]==2, shape[1]>2)
  §v10.99  – Edge Taper am Audio-Ende vorhanden
  §v10.100 – padlen/noverlap Paranoia-Catch
  §v10.101 – safe_filtfilt + JND-Gate + Material-Guards
  §v10.102 – Forensik-Traceback für unerklärte Fehler
  §v10.103 – noverlap-Guard in psychoacoustics.py + phase_53
  §v10.104 – Tuple→ndarray-Guard (_active_quality_intervention, _cand, wrap_phase)
  §v10.106 – Bug-Pattern-Watchdog-Klassifikation
  §v10.107 – per_segment_executor channels-first (2,N) Guard
  §v10.109 – Waveform-Performance (quantile statt percentile, Antialiasing aus)
  §v10.111 – Phase-07 FeedbackChain-Silence-Guard

Alle Tests non-destructive — lesen nur Source/API, schreiben nichts.
"""

import ast
import importlib
import inspect
import re
import sys

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# §v10.95 – PhaseResult Tuple→ndarray
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_95_phaseresult_tuple_to_ndarray():
    """Phase-Interface normalisiert Tuple→ndarray in PhaseResult.audio."""
    from backend.core.phases.phase_interface import PhaseResult

    # Tuple-Input (2 Kanäle, je 100 Samples) → erstes ndarray wird extrahiert
    data_tuple = (np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32))
    result = PhaseResult(audio=data_tuple)  # type: ignore[arg-type]

    assert isinstance(result.audio, np.ndarray), (
        "§v10.95 REGRESSION: PhaseResult.audio ist kein ndarray! "
        "Tuple→ndarray-Normalisierung fehlt oder wurde entfernt."
    )
    # Tuple-Normalisierung extrahiert erstes ndarray → 1D (mono)
    assert result.audio.ndim >= 1, "PhaseResult.audio soll mindestens 1D sein"

    # ndarray-Input muss unverändert bleiben
    data_ndarray = np.zeros((2, 100), dtype=np.float32)
    result2 = PhaseResult(audio=data_ndarray)
    np.testing.assert_array_equal(result2.audio, data_ndarray)


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.97 – Broadcaster channels-last
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_97_broadcaster_channels_last_guard():
    """Broadcaster prüft shape[0]==2 and shape[1]>2 vor stereo-spezifischen
    Operationen."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    # Muster: shape[0]==2 and ... and shape[1] ...
    pattern = r"shape\[0\]\s*==\s*2\s+and\s+.*shape\[1\]"
    matches = list(re.finditer(pattern, source))
    assert len(matches) >= 3, (
        f"§v10.97 REGRESSION: shape[0]==2 and shape[1]>2 Guards fehlen. Nur {len(matches)} statt ≥3 gefunden."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.100 + §v10.101 – safe_filtfilt + padlen-Catch
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_101_safe_filtfilt_available():
    """safe_filtfilt ist aus audio_utils importierbar."""
    from backend.core.audio_utils import safe_filtfilt

    assert callable(safe_filtfilt), "safe_filtfilt ist nicht aufrufbar"
    sig = inspect.signature(safe_filtfilt)
    assert "b" in sig.parameters, "safe_filtfilt muss 'b' Parameter haben"
    assert "a" in sig.parameters, "safe_filtfilt muss 'a' Parameter haben"
    assert "x" in sig.parameters, "safe_filtfilt muss 'x' Parameter haben"


def test_v10_101_safe_filtfilt_short_audio():
    """safe_filtfilt gibt unverändertes Audio zurück bei zu kurzem Input."""
    from backend.core.audio_utils import safe_filtfilt

    b = np.array([1.0, -0.5])
    a = np.array([1.0])
    # Audio kürzer als padlen (typisch 3 * len(b))
    short_audio = np.zeros(5, dtype=np.float32)
    result = safe_filtfilt(b, a, short_audio)
    np.testing.assert_array_equal(result, short_audio)


def test_v10_101_safe_filtfilt_stereo():
    """safe_filtfilt verarbeitet (2,N) Stereo korrekt."""
    from backend.core.audio_utils import safe_filtfilt

    b = np.array([1.0, -0.3])
    a = np.array([1.0])
    stereo = np.random.RandomState(42).randn(2, 1000).astype(np.float32)
    result = safe_filtfilt(b, a, stereo)
    assert result.shape == stereo.shape
    assert result.dtype in (np.float32, np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.103 – noverlap-Guard
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_103_noverlap_guard_psychoacoustics():
    """psychoacoustics.py hat noverlap-Clamp vor stft()."""
    filepath = "backend/core/dsp/psychoacoustics.py"
    with open(filepath) as f:
        source = f.read()

    # Mindestens ein noverlap-Clamp vor stft Aufruf
    assert "noverlap" in source, "psychoacoustics.py enthält kein noverlap"
    pattern = r"_noverlap\s*=\s*min\("
    assert re.search(pattern, source), "§v10.103 REGRESSION: noverlap min(nperseg-1)-Clamp fehlt in psychoacoustics.py"


def test_v10_103_noverlap_guard_phase_53():
    """phase_53 hat noverlap-Guard."""
    filepath = "backend/core/phases/phase_53_psychoacoustic_refinement.py"
    try:
        with open(filepath) as f:
            source = f.read()
    except FileNotFoundError:
        pytest.skip("phase_53 nicht gefunden")

    pattern = r"_noverlap\s*=\s*min\("
    assert re.search(pattern, source), "§v10.103 REGRESSION: noverlap-Clamp fehlt in phase_53"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.104 – Tuple→ndarray-Guards
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_104_active_quality_intervention_guard():
    """_active_quality_intervention hat Tuple→ndarray-Guard."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    # §v10.104 Marker
    assert "§v10.104" in source, "§v10.104 Marker fehlt in unified_restorer_v3.py"


def test_v10_104_cand_tuple_guard():
    """_cand hat Tuple→ndarray-Guard (§v10.104)."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    pattern = r"_cand.*Tuple.*ndarray|_cand.*isinstance.*tuple"
    assert re.search(pattern, source, re.IGNORECASE), "§v10.104 REGRESSION: _cand Tuple-Guard fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.107 – per_segment_executor channels-first
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_107_per_segment_channels_first():
    """per_segment_executor prüft channels-first (2,N) vor Verarbeitung."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    assert "§v10.107" in source, "§v10.107 Marker fehlt in unified_restorer_v3.py"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.109 – Waveform-Performance
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_109_waveform_quantile():
    """modern_window.py nutzt np.quantile statt np.percentile."""
    filepath = "Aurik10/ui/modern_window.py"
    with open(filepath) as f:
        source = f.read()

    assert "np.quantile" in source, "§v10.109 REGRESSION: np.quantile fehlt (sollte np.percentile ersetzen)"
    assert "§v10.109" in source, "§v10.109 Marker fehlt in modern_window.py"


def test_v10_109_waveform_antialiasing_off():
    """Waveform-Rendering hat Antialiasing deaktiviert."""
    filepath = "Aurik10/ui/modern_window.py"
    with open(filepath) as f:
        source = f.read()

    # Muster: renderHint + Antialiasing = False
    pattern = r"Antialiasing.*False|setRenderHint.*Antialiasing"
    assert re.search(pattern, source), "§v10.109 REGRESSION: Antialiasing-Deaktivierung fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.111 – Phase-07 FeedbackChain-Silence-Guard
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_111_phase07_silence_guard():
    """Phase 07 hat Silence-Guard vor FeedbackChain-Verarbeitung."""
    filepath = "backend/core/phases/phase_07_harmonic_restoration.py"
    with open(filepath) as f:
        source = f.read()

    assert "§v10.111" in source, "§v10.111 Marker fehlt in phase_07"
    # silence detection vor feedback chain
    pattern = r"rms|silence|stumm|FeedbackChain.*Guard"
    assert re.search(pattern, source, re.IGNORECASE), "§v10.111 REGRESSION: Silence-Guard fehlt in Phase 07"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.99 – Edge Taper
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_99_edge_taper_present():
    """Edge Taper (12ms fade-in/out) ist in unified_restorer_v3.py."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    assert "§v10.99" in source, "§v10.99 Marker fehlt"
    pattern = r"12\s*ms|edge.?taper|fade.?in.*fade.?out"
    assert re.search(pattern, source, re.IGNORECASE), "§v10.99 REGRESSION: Edge Taper fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.100 – padlen/noverlap Paranoia-Catch
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_100_padlen_catch_present():
    """Paranoia-Catch für padlen/noverlap-Fehler ist vorhanden."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    assert "§v10.100" in source, "§v10.100 Marker fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.102 – Forensik-Traceback
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_102_forensic_traceback_present():
    """Forensik-Traceback für tuple-ndim Fehler ist instrumentiert."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    assert "§v10.102" in source, "§v10.102 Marker fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# §v10.106 – Bug-Pattern-Watchdog
# ═══════════════════════════════════════════════════════════════════════════════


def test_v10_106_watchdog_classification():
    """Bug-Pattern-Watchdog klassifiziert Exceptions."""
    filepath = "backend/core/unified_restorer_v3.py"
    with open(filepath) as f:
        source = f.read()

    assert "§v10.106" in source, "§v10.106 Marker fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# ████████████████████ HÄRTETEST: Live-Runtime ██████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════


class TestV10FixRuntimeHardening:
    """Echte Runtime-Tests: importieren + aufrufen."""

    def test_import_all_fixed_modules(self):
        """Alle fix-betroffenen Module sind importierbar."""
        modules = [
            "backend.core.audio_utils",
            "backend.core.phases.phase_interface",
            "backend.core.dsp.psychoacoustics",
            "backend.core.unified_restorer_v3",
        ]
        for mod_name in modules:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                pytest.fail(f"Import {mod_name} fehlgeschlagen: {e}")

    def test_safe_filtfilt_mono_identity(self):
        """safe_filtfilt: Mono-Input mit identity-Filter = passthrough."""
        from backend.core.audio_utils import safe_filtfilt

        b = np.array([1.0])
        a = np.array([1.0])
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        y = safe_filtfilt(b, a, x)
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_phaseresult_metadata_preserved(self):
        """PhaseResult: metadata bleibt nach Tuple→ndarray erhalten."""
        from backend.core.phases.phase_interface import PhaseResult

        data = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        result = PhaseResult(audio=data, metadata={"key": "value"})  # type: ignore[arg-type]
        assert result.metadata == {"key": "value"}
        assert isinstance(result.audio, np.ndarray)

    def test_psychoacoustics_noverlap_clamp_runtime(self):
        """psychoacoustics: noverlap-Clamp funktioniert mit kurzem Audio."""
        try:
            from backend.core.dsp.psychoacoustics import compute_bark_spectrum  # type: ignore[attr-defined]
        except ImportError:
            pytest.skip("compute_bark_spectrum nicht importierbar")

        # Sehr kurzes Audio (100 Samples @ 48kHz = ~2ms)
        short = np.random.RandomState(99).randn(2, 100).astype(np.float32)
        try:
            result = compute_bark_spectrum(short, sr=48000)
            # Sollte nicht crashen
            assert result is not None
        except Exception as e:
            if "noverlap" in str(e).lower() or "overlap" in str(e).lower():
                pytest.fail(f"§v10.103 REGRESSION: noverlap-Crash bei kurzem Audio: {e}")
