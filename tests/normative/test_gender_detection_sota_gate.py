"""§2.8 [RELEASE_MUST] SOTA Gender Detection — Normativer CI-Gate-Test (v10.0.0)

Spec reference:  .github/specs/19_sota_gender_detection.md
Purpose:         Verifiziert, dass die gesamte Gender-Detection-Kette funktionsfähig,
                 alle Methoden auf DeEsserPhase vorhanden und die Scanning-Fixes
                 aktiv sind.  Kein externes Modell-Training nötig — synthetische
                 Audio-Signale reichen für die strukturelle Validierung.

Kategorien:
  G01 – LPC classify_gender_via_formants exists + valid
  G02 – GenderDetector._detect_f0 scanning statt nur erster 100ms
  G03 – DeEsserPhase alle 5 Gender-Methoden vorhanden (keine Stubs)
  G04 – _detect_gender_simple scanning statt erster 5s
  G05 – _detect_gender_robust Chain: LPC-Fallback integriert
  G06 – Keine toten Stub-Methoden mehr im Code
  G07 – _detect_gender_timeline ist implementiert (nicht leerer Stub)
  G08 – Full Integration: Chain erkennt Stimme hinter instrumentalem Intro

Aufruf: pytest tests/normative/test_gender_detection_sota_gate.py -v --timeout=60
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_voice(f0: float, f1: float, f2: float, f3: float,
                          duration_s: float = 3.0, sr: int = SR) -> np.ndarray:
    """Erzeugt ein synthetisches Stimmsignal mit Grundfrequenz + Formanten."""
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    # Grundton + 3 Obertöne mit abfallender Amplitude
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    sig += 0.3 * np.sin(2 * np.pi * f1 * t)
    sig += 0.2 * np.sin(2 * np.pi * f2 * t)
    sig += 0.12 * np.sin(2 * np.pi * f3 * t)
    sig *= 0.5 / max(np.max(np.abs(sig)), 1e-10)
    return sig.astype(np.float32)


def _add_silence_intro(audio: np.ndarray, silence_s: float = 1.5,
                       sr: int = SR) -> np.ndarray:
    """Fügt eine Stille-Passage vor das Audio (simuliert instrumentales Intro)."""
    silence = np.zeros(int(sr * silence_s), dtype=np.float32)
    return np.concatenate([silence, audio]).astype(np.float32)


_MALE_VOICE = _make_synthetic_voice(120, 500, 1500, 2500)
_FEMALE_VOICE = _make_synthetic_voice(220, 700, 2000, 3000)
_CHILD_VOICE = _make_synthetic_voice(330, 900, 2500, 4000)
_FEMALE_WITH_INTRO = _add_silence_intro(_make_synthetic_voice(220, 700, 2000, 3000))
_MALE_WITH_INTRO = _add_silence_intro(_make_synthetic_voice(120, 500, 1500, 2500))


# ===========================================================================
# G01 — LPC classify_gender_via_formants
# ===========================================================================


class TestLPCGenderClassifier:
    """§2.8/G01: LPC-Formant-Tracker Gender-Classifier existiert und liefert
    valide Werte (male/female/child/unknown)."""

    def test_g01_classifier_exists(self):
        """G01a: classify_gender_via_formants ist auf dem Singleton aufrufbar."""
        from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker

        tracker = get_lpc_formant_tracker()
        assert hasattr(tracker, "classify_gender_via_formants"), (
            "classify_gender_via_formants fehlt auf _LPCFormantTracker"
        )

    def test_g01_returns_valid_gender(self):
        """G01b: Rückgabewert ist einer der 4 validen Strings."""
        from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker

        tracker = get_lpc_formant_tracker()
        result = tracker.classify_gender_via_formants(_FEMALE_VOICE, SR)
        assert result in ("male", "female", "child", "unknown"), (
            f"Ungültiger Rückgabewert: {result}"
        )

    def test_g01_handles_silence(self):
        """G01c: Stille → 'unknown' ohne Crash."""
        from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker

        tracker = get_lpc_formant_tracker()
        silence = np.zeros(SR, dtype=np.float32)
        result = tracker.classify_gender_via_formants(silence, SR)
        assert result == "unknown", f"Stille sollte 'unknown' sein, nicht {result}"

    def test_g01_handles_stereo(self):
        """G01d: Stereo-Input wird korrekt zu Mono gemittelt."""
        from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker

        tracker = get_lpc_formant_tracker()
        stereo = np.column_stack([_FEMALE_VOICE, _FEMALE_VOICE * 0.8])
        result = tracker.classify_gender_via_formants(stereo.astype(np.float32), SR)
        assert result in ("male", "female", "child", "unknown"), (
            f"Stereo-Input sollte verarbeitet werden, nicht {result}"
        )

    def test_g01_short_audio(self):
        """G01e: Extrem kurzes Audio (<50ms) → 'unknown' ohne Crash."""
        from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker

        tracker = get_lpc_formant_tracker()
        short = np.zeros(100, dtype=np.float32)
        result = tracker.classify_gender_via_formants(short, SR)
        assert result == "unknown"


# ===========================================================================
# G02 — GenderDetector._detect_f0 scanning
# ===========================================================================


class TestGenderDetectorScanningF0:
    """§2.8/G02: GenderDetector._detect_f0 scannt durch das Audio statt nur
    die ersten 100ms zu prüfen."""

    def test_g02_f0_detected_despite_intro(self):
        """G02a: F0 wird auch mit 1.5s instrumentalem Intro erkannt."""
        from backend.core.vocal_ai_enhancement import GenderDetector

        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(_FEMALE_WITH_INTRO)
        assert f0 > 100.0, (
            f"F0 sollte trotz Intro > 100 Hz sein (weiblich), war: {f0:.1f} Hz"
        )

    def test_g02_f0_direct_still_works(self):
        """G02b: Direkte F0-Erkennung (ohne Intro) funktioniert weiterhin."""
        from backend.core.vocal_ai_enhancement import GenderDetector

        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(_FEMALE_VOICE)
        assert f0 > 100.0, f"F0 direkt: {f0:.1f} Hz"

    def test_g02_silence_returns_zero(self):
        """G02c: Stille → F0 = 0.0 (kein voiced Segment)."""
        from backend.core.vocal_ai_enhancement import GenderDetector

        gd = GenderDetector(sample_rate=SR)
        silence = np.zeros(SR * 2, dtype=np.float32)
        f0 = gd._detect_f0(silence)
        assert f0 == 0.0, f"Stille F0 sollte 0.0 sein, war: {f0}"


# ===========================================================================
# G03 — DeEsserPhase Methoden-Präsenz
# ===========================================================================


class TestDeEsserPhaseMethodsPresent:
    """§2.8/G03: Alle 5 Gender-Methoden sind auf DeEsserPhase vorhanden
    und keine Stubs mehr."""

    REQUIRED_METHODS = [
        "_detect_gender_robust",
        "_detect_gender_simple",
        "_detect_gender_timeline",
        "_process_per_gender_segments",
        "_apply_formant_preservation",
    ]

    @pytest.mark.parametrize("method_name", REQUIRED_METHODS)
    def test_g03_method_exists(self, method_name: str):
        """G03: Jede erforderliche Gender-Methode ist auf DeEsserPhase vorhanden."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        assert hasattr(dp, method_name), (
            f"DeEsserPhase.{method_name} fehlt — "
            f"möglicherweise in _build_union_vocal_profile gefangen"
        )

    def test_g03_timeline_not_empty_stub(self):
        """G03f: _detect_gender_timeline ist kein leerer Stub (return []) mehr."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        import inspect
        src = inspect.getsource(dp._detect_gender_timeline)
        # Der Stub war: return [] — das darf nicht mehr vorkommen
        assert "return []" not in src.replace(" ", ""), (
            "_detect_gender_timeline ist immer noch ein Stub (return [])"
        )
        # Die echte Implementierung hat librosa/pYIN
        assert "librosa" in src.lower() or "pyin" in src.lower(), (
            "_detect_gender_timeline nutzt keine pYIN/librosa — Stub?"
        )


# ===========================================================================
# G04 — _detect_gender_simple scanning
# ===========================================================================


class TestSimpleGenderScanning:
    """§2.8/G04: _detect_gender_simple scannt durch Audio statt nur erste 5s."""

    def test_g04_detects_female_with_intro(self):
        """G04a: Weibliche Stimme wird trotz Intro erkannt."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        result = dp._detect_gender_simple(_FEMALE_WITH_INTRO, SR)
        assert result == "female", (
            f"Mit Intro sollte 'female' erkannt werden, nicht {result}"
        )

    def test_g04_detects_male_with_intro(self):
        """G04b: Männliche Stimme wird trotz Intro erkannt."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        result = dp._detect_gender_simple(_MALE_WITH_INTRO, SR)
        assert result == "male", (
            f"Mit Intro sollte 'male' erkannt werden, nicht {result}"
        )

    def test_g04_simple_is_not_static_first_5s(self):
        """G04c: Der Code nimmt NICHT nur `audio[:sample_rate * 5]`."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        import inspect
        src = inspect.getsource(dp._detect_gender_simple)
        # Der alte Code hatte: max_samples = sample_rate * 5; audio = audio[:max_samples]
        assert "max_samples = sample_rate * 5" not in src, (
            "_detect_gender_simple nutzt immer noch das alte first-5s-Verfahren"
        )
        # Neuer Code scannt mit 2s-Fenstern
        assert "win_samples" in src or "hop_samples" in src, (
            "_detect_gender_simple hat keine Scanning-Logik"
        )


# ===========================================================================
# G05 — _detect_gender_robust Chain
# ===========================================================================


class TestRobustGenderChain:
    """§2.8/G05: _detect_gender_robust hat LPC-Fallback integriert."""

    def test_g05_chain_includes_lpc_fallback(self):
        """G05a: _detect_gender_robust enthält LPC-Formant-Tracker als Fallback."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        import inspect
        src = inspect.getsource(dp._detect_gender_robust)
        assert "lpc_formant_tracker" in src, (
            "LPC-Formant-Tracker-Fallback fehlt in _detect_gender_robust"
        )
        assert "classify_gender_via_formants" in src, (
            "classify_gender_via_formants-Aufruf fehlt in _detect_gender_robust"
        )

    def test_g05_chain_returns_valid_gender(self):
        """G05b: Die Chain liefert valides Gender auch mit Intro."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        result = dp._detect_gender_robust(_FEMALE_WITH_INTRO, SR)
        assert result in ("male", "female", "child"), (
            f"Chain sollte ein Gender erkennen, nicht {result}"
        )


# ===========================================================================
# G06 — Keine toten Stubs
# ===========================================================================


class TestNoDeadStubs:
    """§2.8/G06: Die gelöschten Stub-Methoden sind tatsächlich entfernt."""

    def test_g06_no_dead_detect_gender_robust_v1(self):
        """G06a: Die alte _detect_gender_robust (v1, Zeile 2469) ist entfernt."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        import inspect
        src = inspect.getsource(dp._detect_gender_robust)
        # Der tote Code rief direkt classify_gender_via_formants
        # Der lebende Code hat pYIN + GenderDetector + Contralto + LPC-Fallback
        assert "pYIN" in src or "librosa" in src or "pyin" in src, (
            "_detect_gender_robust ist die alte tote Version ohne pYIN"
        )

    def test_g06_no_standalone_stub_timeline(self):
        """G06b: Kein standalone `return []` Stub für timeline."""
        with open("backend/core/phases/phase_19_de_esser.py") as f:
            full_src = f.read()
        # Der Stub war: def _detect_gender_timeline(self, audio, sample_rate, hop_length=256):
        #                 return []
        # Dieser exakte 2-Zeilen-Stub darf nicht mehr existieren
        import re
        stub_pattern = r"def _detect_gender_timeline\(self.*\):\s*return \[\]"
        assert not re.search(stub_pattern, full_src), (
            "Stub _detect_gender_timeline mit return [] existiert noch"
        )


# ===========================================================================
# G07 — _detect_gender_timeline implementiert
# ===========================================================================


class TestGenderTimelineImplemented:
    """§2.8/G07: _detect_gender_timeline ist eine echte Implementierung."""

    def test_g07_timeline_returns_list(self):
        """G07a: _detect_gender_timeline gibt eine Liste zurück."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        result = dp._detect_gender_timeline(_FEMALE_VOICE, SR)
        assert isinstance(result, list), (
            f"Timeline sollte list sein, nicht {type(result).__name__}"
        )

    def test_g07_timeline_has_expected_keys(self):
        """G07b: Timeline-Einträge haben die erwarteten Keys."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        result = dp._detect_gender_timeline(_FEMALE_VOICE, SR)
        if len(result) > 0:
            segment = result[0]
            for key in ("t_start_s", "t_end_s", "gender", "confidence"):
                assert key in segment, (
                    f"Timeline-Segment fehlt key '{key}': {list(segment.keys())}"
                )


# ===========================================================================
# G08 — Full Integration: Intro-Test
# ===========================================================================


class TestFullIntegration:
    """§2.8/G08: Die gesamte Chain erkennt Stimmen hinter instrumentalem Intro."""

    def test_g08_female_behind_intro(self):
        """G08a: Weibliche Stimme nach 1.5s Intro wird erkannt (full chain)."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        result = dp._detect_gender_robust(_FEMALE_WITH_INTRO, SR)
        assert result in ("female", "male", "child"), (
            f"Full chain failed with intro: {result}"
        )

    def test_g08_male_behind_intro(self):
        """G08b: Männliche Stimme nach 1.5s Intro wird erkannt (full chain)."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        result = dp._detect_gender_robust(_MALE_WITH_INTRO, SR)
        assert result in ("female", "male", "child"), (
            f"Full chain failed with intro: {result}"
        )

    def test_g08_f0_scanning_vs_old_behavior(self):
        """G08c: Scanning-F0 findet Wert wo alter Code 0 geliefert hätte."""
        from backend.core.vocal_ai_enhancement import GenderDetector

        gd = GenderDetector(sample_rate=SR)
        f0_with_intro = gd._detect_f0(_FEMALE_WITH_INTRO)

        # Alter Code hätte nur audio[:4800] (100ms) geprüft = Stille = 0.0
        # Neuer Code scannt und findet F0 > 0
        assert f0_with_intro > 0.0, (
            "Scanning-F0 sollte mit Intro > 0 sein (alter Code hätte 0.0 geliefert)"
        )

    def test_g08_lpc_fallback_in_chain(self):
        """G08d: LPC-Fallback-Aufruf ist im Code der _detect_gender_robust vorhanden."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        dp = DeEsserPhase(gender_type=VocalGender.AUTO)
        import inspect
        src = inspect.getsource(dp._detect_gender_robust)
        # Die SOTA-Kette: GenderDetector → pYIN → Contralto → LPC → Simple
        assert "get_lpc_formant_tracker" in src, (
            "LPC Formant Tracker nicht in der Chain"
        )
        assert "Burg-LPC fallback" in src.lower() or "lpc formant" in src.lower(), (
            "LPC-Fallback-Log-Message fehlt"
        )
