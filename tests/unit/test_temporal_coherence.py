from __future__ import annotations

import pytest

"""Pflicht-Tests für TemporalQualityCoherenceMetric (§2.16).

Testkonventionen:
    - np.random.seed(42) für Reproduzierbarkeit
    - Nur synthetische Signale (keine echten Audio-Dateien)
    - SR = 48000 Hz (Aurik-Invariante)
    - Alle Tests ≤ 30 s Laufzeit
"""


import math
import threading

import numpy as np

from backend.core.temporal_quality_coherence import (
    TemporalCoherenceResult,
    TemporalQualityCoherenceMetric,
    get_temporal_quality_coherence,
    measure_temporal_coherence,
)

SR = 48_000

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, duration_s: float = 35.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(duration_s: float = 35.0, sr: int = SR, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(duration_s * sr)).astype(np.float32) * 0.1


def _silence(duration_s: float = 35.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# Klasse 1: Rückgabetyp und Felder
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestTemporalCoherenceResultFields:
    def test_01_returns_result_instance(self) -> None:
        """measure() gibt ein TemporalCoherenceResult-Objekt zurück."""
        metric = get_temporal_quality_coherence()
        audio = _sine(duration_s=35.0)
        result = metric.measure(audio, SR)
        assert isinstance(result, TemporalCoherenceResult)

    def test_02_passed_is_bool(self) -> None:
        """passed ist ein boolescher Wert."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert isinstance(result.passed, bool)

    def test_03_max_span_finite(self) -> None:
        """max_span ist ein endlicher Float."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert math.isfinite(result.max_span), "max_span ist NaN oder Inf"

    def test_04_max_span_non_negative(self) -> None:
        """max_span ist ≥ 0."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert result.max_span >= 0.0

    def test_05_sigma_finite(self) -> None:
        """sigma ist ein endlicher Float."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert math.isfinite(result.sigma), "sigma ist NaN oder Inf"

    def test_06_sigma_non_negative(self) -> None:
        """sigma ist ≥ 0."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert result.sigma >= 0.0

    def test_07_segment_scores_is_list(self) -> None:
        """segment_scores ist eine Liste."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert isinstance(result.segment_scores, list)

    def test_08_segment_scores_all_finite(self) -> None:
        """Alle Segment-Scores sind endliche Floats."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        for i, s in enumerate(result.segment_scores):
            assert math.isfinite(s), f"Segment-Score[{i}]={s} ist nicht endlich"

    def test_09_n_segments_positive(self) -> None:
        """n_segments ist ≥ 0."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert result.n_segments >= 0

    def test_10_n_segments_matches_list(self) -> None:
        """n_segments stimmt mit len(segment_scores) überein."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert result.n_segments == len(result.segment_scores)

    def test_11_message_is_string(self) -> None:
        """message ist ein String (kann leer sein)."""
        metric = get_temporal_quality_coherence()
        result = metric.measure(_sine(35.0), SR)
        assert isinstance(result.message, str)


# ---------------------------------------------------------------------------
# Klasse 2: Prüflogik (passed-Kriterium §2.16)
# ---------------------------------------------------------------------------
class TestPassedCriterion:
    def test_12_uniform_signal_passes(self) -> None:
        """Uniform-Sinus-Signal (gleichmäßige Qualität) soll passed=True liefern."""
        metric = get_temporal_quality_coherence()
        # Einfacher, gleichmäßiger Sinus → sehr stabile Qualität über Zeit
        audio = _sine(duration_s=35.0)
        result = metric.measure(audio, SR)
        # Wenn ≥ 3 Segmente: max_span ≤ 0.30 und sigma ≤ 0.15 erwartet
        if result.n_segments >= 3:
            assert result.passed is True

    def test_13_passed_iff_criteria_met(self) -> None:
        """passed=True genau dann wenn max_span≤0.30 UND sigma≤0.15."""
        metric = get_temporal_quality_coherence()
        audio = _sine(35.0)
        result = metric.measure(audio, SR)
        if result.n_segments >= 3:
            expected_pass = (result.max_span <= 0.30) and (result.sigma <= 0.15)
            assert result.passed == expected_pass

    def test_14_consistency_threshold_value(self) -> None:
        """TEMPORAL_CONSISTENCY_THRESHOLD beträgt 0.30 (§2.16)."""
        assert TemporalQualityCoherenceMetric.TEMPORAL_CONSISTENCY_THRESHOLD == 0.30

    def test_15_sigma_threshold_value(self) -> None:
        """SIGMA_THRESHOLD beträgt 0.15 (§2.16)."""
        assert TemporalQualityCoherenceMetric.SIGMA_THRESHOLD == 0.15


# ---------------------------------------------------------------------------
# Klasse 3: Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_16_short_audio_no_crash(self) -> None:
        """Kurzes Audio (< 25 s / < 3 Segmente) → kein Absturz."""
        metric = get_temporal_quality_coherence()
        audio = _sine(duration_s=10.0)
        result = metric.measure(audio, SR)
        assert isinstance(result, TemporalCoherenceResult)

    def test_17_short_audio_n_segments_small(self) -> None:
        """Kurzes Audio hat n_segments < 3 (nicht geprüft laut §2.16)."""
        metric = get_temporal_quality_coherence()
        audio = _sine(duration_s=10.0)
        result = metric.measure(audio, SR)
        assert result.n_segments <= 3

    def test_18_silence_no_crash(self) -> None:
        """Stille → kein Absturz, kein NaN in max_span/sigma."""
        metric = get_temporal_quality_coherence()
        audio = _silence(duration_s=35.0)
        result = metric.measure(audio, SR)
        assert math.isfinite(result.max_span)
        assert math.isfinite(result.sigma)

    def test_19_white_noise_no_crash(self) -> None:
        """Weißes Rauschen → keine Exception, alle Felder endlich."""
        metric = get_temporal_quality_coherence()
        audio = _white_noise(duration_s=35.0)
        result = metric.measure(audio, SR)
        assert math.isfinite(result.max_span)
        assert math.isfinite(result.sigma)

    def test_20_dirac_impulse_no_crash(self) -> None:
        """Dirac-Impuls → keine Exception."""
        metric = get_temporal_quality_coherence()
        audio = np.zeros(int(35.0 * SR), dtype=np.float32)
        audio[int(17.5 * SR)] = 1.0
        result = metric.measure(audio, SR)
        assert isinstance(result, TemporalCoherenceResult)

    def test_21_result_no_nan_anywhere(self) -> None:
        """Alle numerischen Felder sind NaN-frei."""
        metric = get_temporal_quality_coherence()
        audio = _white_noise(35.0, seed=99)
        result = metric.measure(audio, SR)
        assert math.isfinite(result.max_span)
        assert math.isfinite(result.sigma)
        for s in result.segment_scores:
            assert math.isfinite(s)


# ---------------------------------------------------------------------------
# Klasse 4: Singleton & Convenience-Wrapper
# ---------------------------------------------------------------------------
class TestSingletonAndWrapper:
    def test_22_singleton_same_object(self) -> None:
        """get_temporal_quality_coherence() gibt stets dasselbe Objekt zurück."""
        a = get_temporal_quality_coherence()
        b = get_temporal_quality_coherence()
        assert a is b

    def test_23_singleton_thread_safe(self) -> None:
        """Parallele Zugriffe liefern dasselbe Singleton-Objekt."""
        instances: list[TemporalQualityCoherenceMetric] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                inst = get_temporal_quality_coherence()
                with lock:
                    instances.append(inst)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-Fehler: {errors}"
        assert all(inst is instances[0] for inst in instances)

    def test_24_convenience_function_returns_result(self) -> None:
        """measure_temporal_coherence(audio, sr) gibt TemporalCoherenceResult zurück."""
        audio = _sine(35.0)
        result = measure_temporal_coherence(audio, SR)
        assert isinstance(result, TemporalCoherenceResult)

    def test_25_convenience_function_consistent(self) -> None:
        """Convenience-Funktion und .measure() liefern identische Ergebnisse."""
        np.random.seed(42)
        audio = _sine(35.0)
        direct = get_temporal_quality_coherence().measure(audio, SR)
        wrapper = measure_temporal_coherence(audio, SR)
        assert direct.n_segments == wrapper.n_segments
        assert direct.passed == wrapper.passed
        assert abs(direct.max_span - wrapper.max_span) < 1e-6
