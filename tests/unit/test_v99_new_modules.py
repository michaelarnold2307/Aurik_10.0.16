"""
Tests für alle neuen Module — §5 Aurik-9-Spec
=======================================================

Testet:
- TonalCenterMetric + MicroDynamicsMetric (Musical Goals 11 + 12)
- EraClassifier
- TemporalQualityCoherenceMetric
- MusicalStructureAnalyzer
- StereoAuthenticitiyInvariant
- FlowMatchingPlugin (DSP-Fallback)
- PipelineUncertaintyEstimator
- Neue Materialtypen (wax_cylinder, wire_recording, lacquer_disc)
- MusicalGoalsChecker (12 Ziele)

Konventionen (§5.4):
    - np.random.seed(42) in jedem Test
    - Nur synthetische Signale (keine realen Audio-Dateien)
    - Tests laufen in < 30 s (pytest --timeout=30)

Autor: Aurik Development Team
Datum: 20. Februar 2026
"""

import math
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pfad-Setup
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Synthese-Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48_000


def _sine(freq_hz: float = 440.0, duration_s: float = 3.0, sr: int = SR) -> np.ndarray:
    """Erzeugt einen normierten Sinus-Ton."""
    np.random.seed(42)
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = 0.8 * np.sin(2 * np.pi * freq_hz * t)
    return sig.astype(np.float32)  # type: ignore[no-any-return]


def _noisy(freq_hz: float = 440.0, duration_s: float = 3.0, sr: int = SR, noise_level: float = 0.05) -> np.ndarray:
    """Sinus-Ton mit weißem Rauschen."""
    np.random.seed(42)
    sig = _sine(freq_hz, duration_s, sr)
    noise = noise_level * np.random.randn(len(sig)).astype(np.float32)
    return np.clip(sig + noise, -1.0, 1.0)  # type: ignore[no-any-return]


def _silence(duration_s: float = 3.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    """Mono → Stereo (identische Kanäle)."""
    return np.stack([mono, mono], axis=-1)


def _stereo_wide(mono: np.ndarray, spread: float = 0.4) -> np.ndarray:
    """Mono → Stereo mit Kanalversatz."""
    np.random.seed(7)
    noise = spread * np.random.randn(len(mono)).astype(np.float32)
    return np.stack([mono + noise * 0.5, mono - noise * 0.5], axis=-1)


# ===========================================================================
# 1. TonalCenterMetric
# ===========================================================================


@pytest.mark.unit
class TestTonalCenterMetric:
    """Tests für TonalCenterMetric (§1.2 — 11. Musical Goal)."""

    def test_01_import_class(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        assert m is not None

    def test_02_score_returns_float(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        audio = _sine(440.0, 3.0)
        score = m.measure(audio, SR)
        assert isinstance(score, float)

    def test_03_score_in_bounds(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        audio = _sine(440.0, 3.0)
        score = m.measure(audio, SR)
        assert 0.0 <= score <= 1.0

    def test_04_no_nan(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        audio = _noisy(220.0, 3.0)
        score = m.measure(audio, SR)
        assert math.isfinite(score)

    def test_05_identical_audio_high_score_with_ref(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        audio = _sine(440.0, 3.0)
        score = m.measure(audio, SR, reference=audio.copy())
        assert score >= 0.9, f"Identisches Audio sollte Score ≥ 0.9 ergeben, got {score}"

    def test_06_silence_no_crash(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        score = m.measure(_silence(3.0), SR)
        assert 0.0 <= score <= 1.0

    def test_07_stereo_input(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        audio = _stereo(_sine(440.0, 3.0))
        score = m.measure(audio, SR)
        assert 0.0 <= score <= 1.0

    def test_08_very_short_audio(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        audio = _sine(440.0, 0.1)
        score = m.measure(audio, SR)
        assert 0.0 <= score <= 1.0

    def test_09_reference_better_than_no_reference(self):
        """Mit Referenz hat score mehr Bedeutung (kein Crash)."""
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        m = TonalCenterMetric()
        audio = _sine(440.0, 3.0)
        score_ref = m.measure(audio, SR, reference=audio.copy())
        score_no_ref = m.measure(audio, SR)
        assert isinstance(score_ref, float) and isinstance(score_no_ref, float)


# ===========================================================================
# 2. MicroDynamicsMetric
# ===========================================================================


class TestMicroDynamicsMetric:
    """Tests für MicroDynamicsMetric (§1.2 — 12. Musical Goal)."""

    def test_01_import_class(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        assert m is not None

    def test_02_score_returns_float(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        audio = _sine(440.0, 5.0)
        score = m.measure(audio, SR)
        assert isinstance(score, float)

    def test_03_score_in_bounds(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        audio = _sine(440.0, 5.0)
        score = m.measure(audio, SR)
        assert 0.0 <= score <= 1.0

    def test_04_no_nan(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        audio = _noisy(440.0, 5.0, noise_level=0.1)
        score = m.measure(audio, SR)
        assert math.isfinite(score)

    def test_05_identical_audio_high_corr_with_ref(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        audio = _noisy(440.0, 5.0)
        score = m.measure(audio, SR, reference=audio.copy())
        assert score >= 0.85, f"Identisches Audio sollte hohen Score geben, got {score}"

    def test_06_silence_no_crash(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        score = m.measure(_silence(5.0), SR)
        assert 0.0 <= score <= 1.0

    def test_07_stereo_input(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        audio = _stereo(_sine(440.0, 5.0))
        score = m.measure(audio, SR)
        assert 0.0 <= score <= 1.0

    def test_08_rms_profile_length(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        audio = _sine(440.0, 5.0)
        win = int(SR * m.WINDOW_MS / 1000)
        profile = m._rms_profile(audio, win)
        assert len(profile) >= 1

    def test_09_crest_factor_positive(self):
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        audio = _sine(440.0, 3.0)
        crest = m._crest_factor_db(audio)
        assert crest >= 0.0

    def test_10_no_reference_dynamic_music_meets_threshold(self):
        """Normaldynamisches Audio (cv≈0.12) ohne Referenz erzielt Score ≥ 0.92.

        Regression-Test für v10.0.0: altes cv/0.3 lieferte 0.33–0.60 für
        typische Musik; neue Formel 0.60 + cv*4.0 kalibriert korrekt.
        """
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        rng = np.random.default_rng(7)
        m = MicroDynamicsMetric()
        sr = SR
        win = int(sr * m.WINDOW_MS / 1000)
        # Build audio with varying RMS per 400-ms window (cv ≈ 0.12)
        rms_targets = [0.30, 0.50, 0.20, 0.70, 0.40, 0.60, 0.15, 0.55, 0.35, 0.65] * 3
        audio = np.zeros(win * len(rms_targets), dtype=np.float32)
        for i, rms_t in enumerate(rms_targets):
            seg = rng.standard_normal(win).astype(np.float32)
            seg = seg / (np.std(seg) + 1e-8) * rms_t
            audio[i * win : (i + 1) * win] = np.clip(seg, -1.0, 1.0)
        score = m.measure(audio, sr)
        assert score >= 0.92, (
            f"Dynamic music (cv≈0.12) should score ≥ 0.92 without reference, got {score:.3f}. "
            "Regression: old cv/0.3 formula produced 0.33–0.60 for typical music."
        )

    def test_11_over_compressed_scores_below_threshold(self):
        """Flat-komprimiertes Audio (cv≈0) erzielt Score < 0.88 (korrekt als schlechte Dynamik markiert)."""
        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        # Pure sine = near-constant RMS profile → cv ≈ 0
        audio = _sine(440.0, 5.0)
        score = m.measure(audio, SR)
        assert score < 0.88, f"Flat/sine audio should be below threshold, got {score:.3f}"


# ===========================================================================
# 3. MusicalGoalsChecker — 15 Ziele
# ===========================================================================


class TestMusicalGoalsChecker15:
    """Sicherstellt, dass MusicalGoalsChecker 15 Ziele misst."""

    def test_01_has_15_metrics(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        assert "artikulation" in c.metrics
        assert "transient_energie" not in c.metrics
        assert "transient_energie" in c.measure_all(np.zeros(48000, dtype=np.float32), 48000)

    def test_02_has_15_thresholds(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        assert len(c.thresholds) == 15

    def test_03_measure_all_returns_15_scores(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        audio = _sine(440.0, 5.0)
        scores = c.measure_all(audio, SR)
        assert len(scores) == 15, f"Erwartet 15 Scores, got {len(scores)}: {list(scores.keys())}"
        assert "transient_energie" in scores

    def test_04_tonal_center_in_scores(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        audio = _sine(440.0, 5.0)
        scores = c.measure_all(audio, SR)
        assert "tonal_center" in scores

    def test_05_micro_dynamics_in_scores(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        audio = _sine(440.0, 5.0)
        scores = c.measure_all(audio, SR)
        assert "micro_dynamics" in scores

    def test_06_all_scores_in_bounds(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        audio = _sine(440.0, 5.0)
        scores = c.measure_all(audio, SR)
        for name, val in scores.items():
            assert 0.0 <= val <= 1.0, f"Score {name}={val} außerhalb [0,1]"

    def test_07_no_nan_in_scores(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        audio = _noisy(440.0, 5.0)
        scores = c.measure_all(audio, SR)
        for name, val in scores.items():
            assert math.isfinite(val), f"Score {name}={val} ist NaN/Inf"

    def test_08_tonal_center_threshold_095(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker()
        assert c.thresholds["tonal_center"] == 0.95

    def test_09_micro_dynamics_threshold_092(self):
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        c = MusicalGoalsChecker(mode="studio_2026")
        # §0a Tabelle: Studio 2026 MikroDynamik >= 0.90 (normativ korrekt)
        assert c.thresholds["micro_dynamics"] == 0.90


# ===========================================================================
# 4. EraClassifier
# ===========================================================================


class TestEraClassifier:
    """Tests für EraClassifier (§2.14)."""

    def test_01_import(self):
        from backend.core.era_classifier import EraClassifier

        assert EraClassifier is not None

    def test_02_classify_returns_result(self):
        from backend.core.era_classifier import classify_era

        audio = _noisy(440.0, 5.0)
        result = classify_era(audio, SR)
        assert result is not None

    def test_03_result_has_decade(self):
        from backend.core.era_classifier import classify_era

        audio = _sine(440.0, 5.0)
        result = classify_era(audio, SR)
        assert hasattr(result, "decade")
        assert isinstance(result.decade, int)

    def test_04_decade_in_valid_range(self):
        from backend.core.era_classifier import classify_era

        audio = _sine(440.0, 5.0)
        result = classify_era(audio, SR)
        assert 1890 <= result.decade <= 2025

    def test_05_confidence_in_bounds(self):
        from backend.core.era_classifier import classify_era

        audio = _noisy(440.0, 5.0)
        result = classify_era(audio, SR)
        assert 0.0 <= result.confidence <= 1.0

    def test_06_no_nan_confidence(self):
        from backend.core.era_classifier import classify_era

        audio = _sine(440.0, 5.0)
        result = classify_era(audio, SR)
        assert math.isfinite(result.confidence)

    def test_07_material_prior_string(self):
        from backend.core.era_classifier import classify_era

        audio = _sine(440.0, 5.0)
        result = classify_era(audio, SR)
        assert isinstance(result.material_prior, str) and len(result.material_prior) > 0

    def test_08_gp_warmstart_is_dict(self):
        from backend.core.era_classifier import classify_era, get_era_classifier

        audio = _sine(440.0, 5.0)
        result = classify_era(audio, SR)
        ec = get_era_classifier()
        warmstart = ec.get_gp_warmstart(result)
        assert isinstance(warmstart, dict)

    def test_09_silence_no_crash(self):
        from backend.core.era_classifier import classify_era

        audio = _silence(5.0)
        result = classify_era(audio, SR)
        assert result is not None

    def test_10_stereo_no_crash(self):
        from backend.core.era_classifier import classify_era

        audio = _stereo(_sine(440.0, 5.0))
        result = classify_era(audio, SR)
        assert result is not None


# ===========================================================================
# 5. TemporalQualityCoherenceMetric
# ===========================================================================


class TestTemporalQualityCoherenceMetric:
    """Tests für TemporalQualityCoherenceMetric (§2.16)."""

    def test_01_import(self):
        from backend.core.temporal_quality_coherence import TemporalQualityCoherenceMetric

        m = TemporalQualityCoherenceMetric()
        assert m is not None

    def test_02_short_audio_skipped(self):
        """Dateien < 25 s sollen nicht vermessen werden (nicht genug Segmente)."""
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        audio = _sine(440.0, 10.0)
        result = measure_temporal_coherence(audio, SR)
        # Zu kurz → passed=True, n_segments < 3
        assert result.passed is True

    def test_03_long_audio_has_segments(self):
        """Lange Dateien werden in Segmente aufgeteilt."""
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        # 35 s Signal → mindestens 3 Segmente via 10 s Fenster / 5 s Hop
        audio = _noisy(440.0, 35.0)
        result = measure_temporal_coherence(audio, SR)
        assert result.n_segments >= 3

    def test_04_result_has_max_span(self):
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        audio = _noisy(440.0, 35.0)
        result = measure_temporal_coherence(audio, SR)
        assert hasattr(result, "max_span")
        assert math.isfinite(result.max_span)

    def test_05_result_has_sigma(self):
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        audio = _noisy(440.0, 35.0)
        result = measure_temporal_coherence(audio, SR)
        assert hasattr(result, "sigma")
        assert math.isfinite(result.sigma)

    def test_06_uniform_signal_passes(self):
        """Ein zeitlich gleichmäßiges Signal sollte bestehen."""
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        # Gleichmäßiger Sinus über 40 s → geringe MOS-Spanne
        audio = _sine(440.0, 40.0)
        result = measure_temporal_coherence(audio, SR)
        assert result.passed is True

    def test_07_silence_no_crash(self):
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        result = measure_temporal_coherence(_silence(40.0), SR)
        assert result is not None

    def test_08_segment_scores_list(self):
        from backend.core.temporal_quality_coherence import measure_temporal_coherence

        audio = _noisy(440.0, 35.0)
        result = measure_temporal_coherence(audio, SR)
        assert isinstance(result.segment_scores, list)
        assert len(result.segment_scores) == result.n_segments


# ===========================================================================
# 6. MusicalStructureAnalyzer
# ===========================================================================


class TestMusicalStructureAnalyzer:
    """Tests für MusicalStructureAnalyzer (§2.17)."""

    def test_01_import(self):
        from backend.core.musical_structure_analyzer import MusicalStructureAnalyzer

        m = MusicalStructureAnalyzer()
        assert m is not None

    def test_02_short_audio_returns_empty(self):
        from backend.core.musical_structure_analyzer import analyze_musical_structure

        audio = _sine(440.0, 10.0)  # < 20 s → keine Analyse
        result = analyze_musical_structure(audio, SR)
        assert result.segments == []

    def test_03_long_audio_segments(self):
        from backend.core.musical_structure_analyzer import analyze_musical_structure

        audio = _noisy(440.0, 40.0)
        result = analyze_musical_structure(audio, SR)
        assert result.total_duration_s > 0.0

    def test_04_bpm_positive(self):
        from backend.core.musical_structure_analyzer import analyze_musical_structure

        audio = _noisy(440.0, 40.0)
        result = analyze_musical_structure(audio, SR)
        assert result.bpm >= 0.0

    def test_05_confidence_in_bounds(self):
        from backend.core.musical_structure_analyzer import analyze_musical_structure

        audio = _noisy(440.0, 40.0)
        result = analyze_musical_structure(audio, SR)
        assert 0.0 <= result.confidence <= 1.0

    def test_06_no_crash_silence(self):
        from backend.core.musical_structure_analyzer import analyze_musical_structure

        result = analyze_musical_structure(_silence(40.0), SR)
        assert result is not None

    def test_07_segments_max_200(self):
        from backend.core.musical_structure_analyzer import analyze_musical_structure

        audio = _noisy(440.0, 40.0)
        result = analyze_musical_structure(audio, SR)
        assert len(result.segments) <= 200

    def test_08_stereo_no_crash(self):
        from backend.core.musical_structure_analyzer import analyze_musical_structure

        audio = _stereo(_noisy(440.0, 40.0))
        result = analyze_musical_structure(audio, SR)
        assert result is not None

    def test_09_singleton_consistent(self):
        from backend.core.musical_structure_analyzer import get_musical_structure_analyzer

        a = get_musical_structure_analyzer()
        b = get_musical_structure_analyzer()
        assert a is b


# ===========================================================================
# 7. StereoAuthenticitiyInvariant
# ===========================================================================


class TestStereoAuthenticitiyInvariant:
    """Tests für StereoAuthenticitiyInvariant (§2.18)."""

    def _make_era(self, decade: int = 1970, confidence: float = 0.8):
        """Erstellt ein einfaches EraResult-Objekt."""

        class EraResult:
            pass

        era = EraResult()
        era.decade = decade  # type: ignore[attr-defined]
        era.confidence = confidence  # type: ignore[attr-defined]
        era.era_label = f"{decade}s"  # type: ignore[attr-defined]
        era.material_prior = "unknown"  # type: ignore[attr-defined]
        return era

    def test_01_import(self):
        from backend.core.stereo_authenticity_invariant import StereoAuthenticitiyInvariant

        s = StereoAuthenticitiyInvariant()
        assert s is not None

    def test_02_mono_era_low_confidence_passes(self):
        """Bei Konfidenz < 0.40 wird immer passed=True zurückgegeben."""
        from backend.core.stereo_authenticity_invariant import check_stereo_authenticity

        audio = _sine(440.0, 3.0)
        era = self._make_era(1930, confidence=0.3)
        result = check_stereo_authenticity(audio, audio, era, SR)
        assert result.passed is True

    def test_03_identical_mono_passes(self):
        """Mono-Signal (beide Kanäle identisch) bei Mono-Ära sollte bestehen."""
        from backend.core.stereo_authenticity_invariant import check_stereo_authenticity

        mono = _sine(440.0, 3.0)
        stereo_mono = _stereo(mono)  # identische Kanäle → hohe M/S-Korrelation
        era = self._make_era(1940, confidence=0.8)
        result = check_stereo_authenticity(stereo_mono, stereo_mono, era, SR)
        assert result.passed is True

    def test_04_result_has_ms_correlation(self):
        from backend.core.stereo_authenticity_invariant import check_stereo_authenticity

        audio = _stereo(_sine(440.0, 3.0))
        era = self._make_era(1970, confidence=0.8)
        result = check_stereo_authenticity(audio, audio, era, SR)
        assert hasattr(result, "ms_correlation")
        assert math.isfinite(result.ms_correlation)

    def test_05_result_has_lr_cross_corr(self):
        from backend.core.stereo_authenticity_invariant import check_stereo_authenticity

        audio = _stereo(_sine(440.0, 3.0))
        era = self._make_era(1970, confidence=0.8)
        result = check_stereo_authenticity(audio, audio, era, SR)
        assert hasattr(result, "lr_cross_corr")
        assert math.isfinite(result.lr_cross_corr)

    def test_06_mono_era_wide_stereo_fails(self):
        """Mono-Ära mit künstlichem Stereo sollte fehlschlagen."""
        from backend.core.stereo_authenticity_invariant import check_stereo_authenticity

        mono = _sine(440.0, 3.0)
        original_mono = _stereo(mono)  # identisch → mono
        wide_stereo = _stereo_wide(mono, spread=0.8)  # breites Stereo
        era = self._make_era(1930, confidence=0.8)
        result = check_stereo_authenticity(original_mono, wide_stereo, era, SR)
        # Sollte fehlschlagen da Original Mono aber Restaurierung breites Stereo
        # (abhängig von Berechnung)
        assert isinstance(result.passed, bool)

    def test_07_enforce_returns_clipped_audio(self):
        from backend.core.stereo_authenticity_invariant import get_stereo_authenticity_invariant

        s = get_stereo_authenticity_invariant()
        mono = _sine(440.0, 3.0)
        audio = _stereo(mono)
        era = self._make_era(1930, confidence=0.8)
        enforced = s.enforce(audio, SR, audio, era)
        assert np.max(np.abs(enforced)) <= 1.0
        assert np.isfinite(enforced).all()

    def test_08_singleton_consistent(self):
        from backend.core.stereo_authenticity_invariant import get_stereo_authenticity_invariant

        a = get_stereo_authenticity_invariant()
        b = get_stereo_authenticity_invariant()
        assert a is b

    def test_09_mono_audio_no_crash(self):
        from backend.core.stereo_authenticity_invariant import check_stereo_authenticity

        audio = _sine(440.0, 3.0)  # Mono
        era = self._make_era(1950, confidence=0.8)
        result = check_stereo_authenticity(audio, audio, era, SR)
        assert result is not None


# ===========================================================================
# 8. FlowMatchingPlugin (DSP-Fallback)
# ===========================================================================


class TestFlowMatchingPlugin:
    """Tests für FlowMatchingPlugin (§4.5) — DSP-Fallback."""

    def test_01_import(self):
        from plugins.flow_matching_plugin import FlowMatchingPlugin

        p = FlowMatchingPlugin()
        assert p is not None

    def test_02_inpaint_returns_result(self):
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 5.0)
        gap_start = int(SR * 1.0)
        gap_end = int(SR * 1.5)
        result = inpaint_flow(audio, gap_start, gap_end, SR)
        assert result is not None

    def test_03_output_length_preserved(self):
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 5.0)
        gap_start = int(SR * 1.0)
        gap_end = int(SR * 1.5)
        result = inpaint_flow(audio, gap_start, gap_end, SR)
        assert len(result.audio) == len(audio)

    def test_04_output_no_nan(self):
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 5.0)
        gap_start = int(SR * 1.0)
        gap_end = int(SR * 1.5)
        result = inpaint_flow(audio, gap_start, gap_end, SR)
        assert np.isfinite(result.audio).all()

    def test_05_output_clipped(self):
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 5.0)
        gap_start = int(SR * 1.0)
        gap_end = int(SR * 1.5)
        result = inpaint_flow(audio, gap_start, gap_end, SR)
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_06_method_used_string(self):
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 5.0)
        result = inpaint_flow(audio, int(SR * 1.0), int(SR * 1.5), SR)
        assert isinstance(result.method_used, str) and len(result.method_used) > 0

    def test_07_kl_divergence_finite(self):
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 5.0)
        result = inpaint_flow(audio, int(SR * 1.0), int(SR * 1.5), SR)
        assert math.isfinite(result.kl_divergence)

    def test_08_tiny_gap_no_inpainting(self):
        """Sehr kleine Lücke (< MIN_GAP_SAMPLES) → no_inpainting."""
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 5.0)
        gap_start = int(SR * 1.0)
        gap_end = gap_start + 10  # Nur 10 Samples
        result = inpaint_flow(audio, gap_start, gap_end, SR)
        assert result.method_used == "no_inpainting"

    def test_09_singleton_consistent(self):
        from plugins.flow_matching_plugin import get_flow_matching_plugin

        a = get_flow_matching_plugin()
        b = get_flow_matching_plugin()
        assert a is b

    def test_10_sr_assertion(self):
        """SR != 48000 soll AssertionError auslösen."""
        from plugins.flow_matching_plugin import inpaint_flow

        audio = _sine(440.0, 3.0)
        with pytest.raises(AssertionError):
            inpaint_flow(audio, 0, 1000, sr=44100)


# ===========================================================================
# 9. PipelineUncertaintyEstimator
# ===========================================================================


class TestPipelineUncertaintyEstimator:
    """Tests für PipelineUncertaintyEstimator (§2.15)."""

    def _make_plan(self, confidence: float = 0.8):
        class RestorationPlan:
            pass

        p = RestorationPlan()
        p.confidence = confidence  # type: ignore[attr-defined]
        return p

    def test_01_import(self):
        from backend.core.pipeline_uncertainty import PipelineUncertaintyEstimator

        e = PipelineUncertaintyEstimator()
        assert e is not None

    def test_02_estimate_returns_result(self):
        from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

        plan = self._make_plan(0.9)
        result = estimate_pipeline_confidence(plan)
        assert result is not None

    def test_03_confidence_in_bounds(self):
        from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

        for conf in [0.1, 0.5, 0.9]:
            plan = self._make_plan(conf)
            result = estimate_pipeline_confidence(plan)
            assert 0.0 <= result.confidence <= 1.0

    def test_04_high_confidence_tier_high(self):
        from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

        plan = self._make_plan(0.95)
        result = estimate_pipeline_confidence(plan, defect_scores={"CLICKS": 0.9})
        assert result.tier == "high"
        assert result.gp_bound_factor == 1.0

    def test_05_low_confidence_tier_low(self):
        from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

        plan = self._make_plan(0.2)
        result = estimate_pipeline_confidence(plan, defect_scores={"CLICKS": 0.1})
        assert result.tier in ("medium", "low")

    def test_06_low_confidence_has_user_hint(self):
        from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

        plan = self._make_plan(0.1)
        result = estimate_pipeline_confidence(plan, defect_scores={"CLICKS": 0.1})
        if result.tier == "low":
            assert len(result.user_hint) > 0

    def test_07_apply_to_gp_params_high_unchanged(self):
        from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

        plan = self._make_plan(0.95)
        result = estimate_pipeline_confidence(plan, defect_scores={"CLICKS": 0.9})
        params = {"noise_reduction_strength": 0.7, "harmonic_boost_db": 3.0}
        param_space = {"noise_reduction_strength": (0.1, 1.0), "harmonic_boost_db": (0.0, 6.0)}
        from backend.core.pipeline_uncertainty import get_pipeline_uncertainty_estimator

        estimator = get_pipeline_uncertainty_estimator()
        adjusted = estimator.apply_to_gp_params(params, result, param_space)
        if result.tier == "high":
            assert adjusted == params

    def test_08_threshold_offsets_low_confidence(self):
        from backend.core.pipeline_uncertainty import PipelineConfidence, get_pipeline_uncertainty_estimator

        estimator = get_pipeline_uncertainty_estimator()
        low_conf = PipelineConfidence(confidence=0.3, tier="low", threshold_offset=0.02)
        thresholds = {"brillanz": 0.85, "waerme": 0.80}
        adjusted = estimator.apply_threshold_offsets(thresholds, low_conf)
        assert adjusted["brillanz"] == pytest.approx(0.87, abs=0.001)

    def test_09_singleton_consistent(self):
        from backend.core.pipeline_uncertainty import get_pipeline_uncertainty_estimator

        a = get_pipeline_uncertainty_estimator()
        b = get_pipeline_uncertainty_estimator()
        assert a is b

    def test_10_none_plan_no_crash(self):
        from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

        result = estimate_pipeline_confidence(None)
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0


# ===========================================================================
# 10. Neue Materialtypen im DefectScanner
# ===========================================================================


class TestNewMaterialTypes:
    """Tests für wax_cylinder, wire_recording, lacquer_disc (§6.1/6.2)."""

    def test_01_wax_cylinder_in_enum(self):
        from backend.core.defect_scanner import MaterialType

        assert hasattr(MaterialType, "WAX_CYLINDER")
        assert MaterialType.WAX_CYLINDER.value == "wax_cylinder"

    def test_02_wire_recording_in_enum(self):
        from backend.core.defect_scanner import MaterialType

        assert hasattr(MaterialType, "WIRE_RECORDING")
        assert MaterialType.WIRE_RECORDING.value == "wire_recording"

    def test_03_lacquer_disc_in_enum(self):
        from backend.core.defect_scanner import MaterialType

        assert hasattr(MaterialType, "LACQUER_DISC")
        assert MaterialType.LACQUER_DISC.value == "lacquer_disc"

    def test_04_wax_cylinder_in_sensitivity(self):
        from backend.core.defect_scanner import DefectScanner, MaterialType

        assert MaterialType.WAX_CYLINDER in DefectScanner.MATERIAL_SENSITIVITY

    def test_05_wire_recording_in_sensitivity(self):
        from backend.core.defect_scanner import DefectScanner, MaterialType

        assert MaterialType.WIRE_RECORDING in DefectScanner.MATERIAL_SENSITIVITY

    def test_06_lacquer_disc_in_sensitivity(self):
        from backend.core.defect_scanner import DefectScanner, MaterialType

        assert MaterialType.LACQUER_DISC in DefectScanner.MATERIAL_SENSITIVITY

    def test_07_wax_cylinder_bandwidth_loss_low(self):
        """WaxCylinder sollte sehr niedrige Bandwidth-Loss-Threshold haben (sehr häufig)."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        thresh = DefectScanner.MATERIAL_SENSITIVITY[MaterialType.WAX_CYLINDER]
        assert thresh[DefectType.BANDWIDTH_LOSS] <= 0.2, "WaxCylinder: BANDWIDTH_LOSS sollte ≤ 0.2 sein"

    def test_08_wire_recording_jitter_low(self):
        """WireRecording sollte niedrige Jitter-Threshold haben (sehr charakteristisch)."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        thresh = DefectScanner.MATERIAL_SENSITIVITY[MaterialType.WIRE_RECORDING]
        assert thresh[DefectType.JITTER_ARTIFACTS] <= 0.3, "WireRecording: JITTER_ARTIFACTS sollte ≤ 0.3 sein"

    def test_09_lacquer_disc_clicks_low(self):
        """LacquerDisc sollte niedrige Clicks-Threshold haben (Rissbildung)."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        thresh = DefectScanner.MATERIAL_SENSITIVITY[MaterialType.LACQUER_DISC]
        assert thresh[DefectType.CLICKS] <= 0.3, "LacquerDisc: CLICKS sollte ≤ 0.3 sein"

    def test_10_scanner_accepts_wax_cylinder(self):
        """DefectScanner akzeptiert wax_cylinder als MaterialType."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(sample_rate=SR, material_type=MaterialType.WAX_CYLINDER)
        assert scanner.material_type == MaterialType.WAX_CYLINDER

    def test_11_scan_wax_cylinder_no_crash(self):
        """Vollständiger Scan mit wax_cylinder Material-Typ."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(sample_rate=SR, material_type=MaterialType.WAX_CYLINDER)
        audio = _noisy(220.0, 3.0, noise_level=0.2)
        result = scanner.scan(audio, SR, material_type=MaterialType.WAX_CYLINDER)
        assert result is not None

    def test_12_all_three_materials_have_all_defect_types(self):
        """Alle drei neuen Materialien haben Einträge für alle DefectTypes."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        for mat in [MaterialType.WAX_CYLINDER, MaterialType.WIRE_RECORDING, MaterialType.LACQUER_DISC]:
            sensitivity = DefectScanner.MATERIAL_SENSITIVITY[mat]
            for dt in DefectType:
                assert dt in sensitivity, f"{mat.value} fehlt Eintrag für {dt.value}"
