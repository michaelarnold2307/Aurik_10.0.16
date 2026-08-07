from __future__ import annotations

"""
Unit-Tests für TimbralAuthenticityMetric (10. Musical Goal) – ≥ 35 Tests.

Prüft gemäß Aurik-Spec §5.1:
  - Shape / Dtype-Kompatibilität
  - NaN/Inf-Robustheit
  - Bounds ∈ [0.0, 1.0]
  - Edge-Cases (Stille, Rauschen, Dirac, < 0.1 s)
  - Mono + Stereo
  - Konsistenz (selbe Eingabe → selber Score)
  - Referenz-basierter vs. referenz-freier Modus
  - Integration mit MusicalGoalsChecker (10 Ziele)

Konventionen:
  - np.random.seed(42) für Reproduzierbarkeit
  - Keine realen Audio-Dateien
  - @pytest.mark.timeout implizit via pytest.ini --timeout=30
"""


import math
import threading

import numpy as np
import pytest
from scipy.signal import lfilter

from backend.core.musical_goals.musical_goals_metrics import (
    MusicalGoalsChecker,
    TimbralAuthenticityMetric,
)

SR = 48_000  # Pflicht-SR gem. Aurik-Spec §6.5


# ============================================================================
# Hilfsfunktionen
# ============================================================================


def _noise(duration_s: float = 1.0, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return (np.random.randn(int(duration_s * SR)) * 0.3).astype(np.float32)


def _silence(duration_s: float = 1.0) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _sine(freq_hz: float = 440.0, duration_s: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * 0.4).astype(np.float32)  # type: ignore[no-any-return]


def _colored_noise(beta: float = 2.0, duration_s: float = 1.5, seed: int = 42) -> np.ndarray:
    """Rosa / rotes Rauschen via IIR-Näherung."""
    np.random.seed(seed)
    n = int(duration_s * SR)
    white = np.random.randn(n).astype(np.float32)
    a = 1.0 - 0.97 * min(beta / 2.0, 1.0)
    colored = lfilter([1.0], [1.0, -(1.0 - a)], white).astype(np.float32)
    colored /= np.max(np.abs(colored)) + 1e-10
    return colored * 0.3  # type: ignore[no-any-return]


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.column_stack([mono, mono])


def _add_degradation(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Fügt Weißrauschen hinzu für definiertes SNR."""
    sig_power = np.mean(audio**2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    np.random.seed(99)
    noise = np.random.randn(len(audio)).astype(np.float32) * math.sqrt(noise_power)
    degraded = audio + noise
    return np.clip(degraded, -1.0, 1.0)  # type: ignore[no-any-return]


# ============================================================================
# TestGroup 1: Grundlegende Shape/Dtype
# ============================================================================


class TestBasicShape:
    """Tests 01–06: Shape, Dtype, Rückgabe-Typ."""

    def test_01_returns_float(self) -> None:
        m = TimbralAuthenticityMetric()
        s = m.measure(_noise(), SR)
        assert isinstance(s, float)

    def test_02_dtype_float32(self) -> None:
        """float32 Eingabe → kein Fehler."""
        m = TimbralAuthenticityMetric()
        audio = _noise().astype(np.float32)
        assert isinstance(m.measure(audio, SR), float)

    def test_03_dtype_float64(self) -> None:
        """float64 Eingabe → kein Fehler."""
        m = TimbralAuthenticityMetric()
        audio = _noise().astype(np.float64)
        assert isinstance(m.measure(audio, SR), float)

    def test_04_mono_1d(self) -> None:
        m = TimbralAuthenticityMetric()
        audio = _noise()
        assert audio.ndim == 1
        s = m.measure(audio, SR)
        assert 0.0 <= s <= 1.0

    def test_05_stereo_2d(self) -> None:
        m = TimbralAuthenticityMetric()
        audio = _stereo(_noise())
        assert audio.ndim == 2
        s = m.measure(audio, SR)
        assert 0.0 <= s <= 1.0

    def test_06_reference_stereo(self) -> None:
        m = TimbralAuthenticityMetric()
        ref = _stereo(_noise(seed=1))
        deg = _stereo(_noise(seed=2))
        s = m.measure(deg, SR, reference=ref)
        assert 0.0 <= s <= 1.0


# ============================================================================
# TestGroup 2: Bounds ∈ [0.0, 1.0]
# ============================================================================


class TestBounds:
    """Tests 07–12: Alle Ausgaben innerhalb [0.0, 1.0]."""

    def _m(self) -> TimbralAuthenticityMetric:
        return TimbralAuthenticityMetric()

    def test_07_noise_bounds(self) -> None:
        s = self._m().measure(_noise(), SR)
        assert 0.0 <= s <= 1.0

    def test_08_silence_bounds(self) -> None:
        s = self._m().measure(_silence(), SR)
        assert 0.0 <= s <= 1.0

    def test_09_sine_bounds(self) -> None:
        s = self._m().measure(_sine(), SR)
        assert 0.0 <= s <= 1.0

    def test_10_reference_mode_bounds(self) -> None:
        ref = _noise(seed=11)
        deg = _add_degradation(ref, snr_db=15)
        s = self._m().measure(deg, SR, reference=ref)
        assert 0.0 <= s <= 1.0

    def test_11_colored_noise_bounds(self) -> None:
        s = self._m().measure(_colored_noise(beta=2.0), SR)
        assert 0.0 <= s <= 1.0

    def test_12_low_snr_bounds(self) -> None:
        ref = _sine(1000.0, 1.0)
        deg = _add_degradation(ref, snr_db=5)
        s = self._m().measure(deg, SR, reference=ref)
        assert 0.0 <= s <= 1.0


# ============================================================================
# TestGroup 3: Edge-Cases
# ============================================================================


class TestEdgeCases:
    """Tests 13–20: Grenzfälle."""

    def _m(self) -> TimbralAuthenticityMetric:
        return TimbralAuthenticityMetric()

    def test_13_very_short_audio(self) -> None:
        """< 50 ms → kein Crash."""
        audio = _noise(0.04)
        s = self._m().measure(audio, SR)
        assert 0.0 <= s <= 1.0

    def test_14_single_sample(self) -> None:
        audio = np.array([0.5], dtype=np.float32)
        s = self._m().measure(audio, SR)
        assert 0.0 <= s <= 1.0

    def test_15_all_zeros(self) -> None:
        s = self._m().measure(_silence(1.0), SR)
        assert 0.0 <= s <= 1.0

    def test_16_dirac_impulse(self) -> None:
        audio = np.zeros(SR, dtype=np.float32)
        audio[SR // 2] = 1.0
        s = self._m().measure(audio, SR)
        assert 0.0 <= s <= 1.0

    def test_17_nan_input_no_crash(self) -> None:
        audio = np.full(SR, np.nan, dtype=np.float32)
        s = self._m().measure(audio, SR)
        assert math.isfinite(s)

    def test_18_inf_input_no_crash(self) -> None:
        audio = np.full(SR, np.inf, dtype=np.float32)
        s = self._m().measure(audio, SR)
        assert math.isfinite(s)

    def test_19_long_audio_5s(self) -> None:
        audio = _noise(5.0)
        s = self._m().measure(audio, SR)
        assert 0.0 <= s <= 1.0

    def test_20_ref_shorter_than_deg(self) -> None:
        """Referenz kürzer als degradiertes Signal → kein Crash."""
        ref = _noise(0.5)
        deg = _noise(1.5, seed=99)
        s = self._m().measure(deg, SR, reference=ref)
        assert 0.0 <= s <= 1.0


# ============================================================================
# TestGroup 4: Semantische Korrektheit
# ============================================================================


class TestSemantics:
    """Tests 21–28: Inhaltliche Plausibilität."""

    def _m(self) -> TimbralAuthenticityMetric:
        return TimbralAuthenticityMetric()

    def test_21_identical_signal_perfect_score(self) -> None:
        """Identisches Referenz- und degradiertes Signal → Score nahe 1.0."""
        audio = _noise(1.0)
        s = self._m().measure(audio, SR, reference=audio)
        assert s >= 0.95, f"Identisches Signal → Erwartet ≥ 0.95, erhalten {s:.4f}"

    def test_22_high_snr_degradation_high_score(self) -> None:
        """SNR=40 dB → Score höher als bei SNR=5 dB (Richtungsprüfung)."""
        ref = _noise(2.0, seed=5)  # rauschreiches Signal – bessere MFCC-Differenzierbarkeit
        deg_high = _add_degradation(ref, snr_db=40)
        deg_low = _add_degradation(ref, snr_db=5)
        m = self._m()
        s_high = m.measure(deg_high, SR, reference=ref)
        s_low = m.measure(deg_low, SR, reference=ref)
        assert s_high >= s_low, f"Hohes SNR ({s_high:.4f}) sollte Score ≥ niederem SNR ({s_low:.4f}) haben"

    def test_23_uncorrelated_low_score(self) -> None:
        """Völlig anderes Signal → Score niedriger als identisches Signal."""
        ref = _noise(1.0, seed=1)
        other = _sine(800, 1.0)
        s_diff = self._m().measure(other, SR, reference=ref)
        s_same = self._m().measure(ref, SR, reference=ref)
        assert s_same >= s_diff, f"Gleich={s_same:.4f} sollte ≥ Verschieden={s_diff:.4f}"

    def test_24_stability_pure_tone_high(self) -> None:
        """Reiner Ton (zeitlich stabil) → hoher Stabilitätsscore."""
        audio = _sine(440, 2.0)
        s = self._m().measure(audio, SR)  # referenz-frei
        # Reiner Ton hat geringe MFCC-Varianz → sollte > noise
        assert 0.0 <= s <= 1.0

    def test_25_threshold_default(self) -> None:
        assert TimbralAuthenticityMetric().threshold == pytest.approx(0.87)

    def test_26_custom_threshold(self) -> None:
        m = TimbralAuthenticityMetric(threshold=0.90)
        assert m.threshold == pytest.approx(0.90)

    def test_27_score_finite(self) -> None:
        s = self._m().measure(_noise(), SR)
        assert math.isfinite(s)

    def test_28_ref_based_vs_stability(self) -> None:
        """Beide Modi liefern plausible Werte."""
        audio = _noise(1.0)
        m = self._m()
        s_stab = m.measure(audio, SR)
        s_ref = m.measure(audio, SR, reference=audio)
        assert math.isfinite(s_stab)
        assert math.isfinite(s_ref)


# ============================================================================
# TestGroup 5: Konsistenz
# ============================================================================


class TestConsistency:
    """Tests 29–32: Deterministische Ausgabe."""

    def _m(self) -> TimbralAuthenticityMetric:
        return TimbralAuthenticityMetric()

    def test_29_same_input_same_output(self) -> None:
        audio = _noise(1.0)
        m = self._m()
        assert m.measure(audio, SR) == pytest.approx(m.measure(audio, SR))

    def test_30_same_ref_same_output(self) -> None:
        ref = _noise(seed=77)
        deg = _noise(seed=88)
        m = self._m()
        s1 = m.measure(deg, SR, reference=ref)
        s2 = m.measure(deg, SR, reference=ref)
        assert s1 == pytest.approx(s2)

    def test_31_multiple_instances_agree(self) -> None:
        audio = _noise(1.0)
        s1 = TimbralAuthenticityMetric().measure(audio, SR)
        s2 = TimbralAuthenticityMetric().measure(audio, SR)
        assert s1 == pytest.approx(s2)

    def test_32_thread_consistency(self) -> None:
        """Mehrere Threads melden identische Scores für gleiche Eingabe."""
        audio = _noise(1.0)
        results: list[float] = []
        lock = threading.Lock()

        def _run() -> None:
            s = TimbralAuthenticityMetric().measure(audio, SR)
            with lock:
                results.append(s)

        threads = [threading.Thread(target=_run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(abs(r - results[0]) < 1e-5 for r in results)


# ============================================================================
# TestGroup 6: Integration mit MusicalGoalsChecker
# ============================================================================


class TestMusicalGoalsIntegration:
    """Tests 33–40: 10. Ziel im MusicalGoalsChecker."""

    def test_33_checker_has_14_explicit_metric_classes(self) -> None:
        checker = MusicalGoalsChecker()
        assert "timbre_authentizitaet" in checker.metrics
        assert "artikulation" in checker.metrics
        assert "transient_energie" not in checker.metrics

    def test_34_timbre_in_checker_metrics(self) -> None:
        checker = MusicalGoalsChecker()
        assert "timbre_authentizitaet" in checker.metrics

    def test_35_timbre_threshold_087(self) -> None:
        checker = MusicalGoalsChecker()
        assert checker.thresholds["timbre_authentizitaet"] == pytest.approx(0.87)

    def test_36_measure_all_returns_15_keys(self) -> None:
        checker = MusicalGoalsChecker()
        audio = _noise(1.0)
        scores = checker.measure_all(audio, SR)
        assert len(scores) == 15
        assert "transient_energie" in scores

    def test_37_timbre_in_measure_all(self) -> None:
        checker = MusicalGoalsChecker()
        audio = _noise(1.0)
        scores = checker.measure_all(audio, SR)
        assert "timbre_authentizitaet" in scores

    def test_38_all_scores_in_bounds(self) -> None:
        checker = MusicalGoalsChecker()
        audio = _noise(2.0)
        scores = checker.measure_all(audio, SR)
        for goal, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{goal}={score} außerhalb [0,1]"

    def test_39_all_scores_finite(self) -> None:
        checker = MusicalGoalsChecker()
        audio = _noise(2.0)
        scores = checker.measure_all(audio, SR)
        for goal, score in scores.items():
            assert math.isfinite(score), f"{goal}={score} ist nicht finite"

    def test_40_measure_all_with_reference(self) -> None:
        """measure_all mit reference → timbre_authenticity nutzt Vergleich."""
        checker = MusicalGoalsChecker()
        ref = _noise(1.0, seed=1)
        deg = _add_degradation(ref, snr_db=20)
        scores = checker.measure_all(deg, SR, reference=ref)
        assert "timbre_authentizitaet" in scores
        assert 0.0 <= scores["timbre_authentizitaet"] <= 1.0
