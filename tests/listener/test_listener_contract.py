"""ABX/MUSHRA Listener Contract Tests. Spec 15 §15.3.4.

Testet:
- Stimulus-Zufälligkeit (ABX X-Zuweisung randomisiert)
- Session-Isolation (Sessions unabhängig)  
- Ergebnis-Aggregation (Binomialtest, Statistik)
"""
from __future__ import annotations
import pytest
import numpy as np


def _make_test_audio(duration_s: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestABXStimulusRandomness:
    """Stimulus-Zufälligkeit: X-Zuweisung muss randomisiert sein."""

    def test_x_assignment_is_binary(self):
        """Jeder Trial hat X = A oder X = B (nicht immer gleich)."""
        from backend.core.ab_comparison import ABComparison, ABBlindTest

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.9
        cmp = ABComparison(audio_a, audio_b, sr=48000)
        bt = ABBlindTest(cmp, seed=42)
        bt.start_session(n_trials=20)

        x_values = [t["x_is_a"] for t in bt.trials]
        assert len(set(x_values)) == 2, f"X should be mixed A/B, got only {set(x_values)}"

    def test_x_distribution_is_roughly_balanced(self):
        """X sollte ungefähr 50/50 A/B sein (Binomial, p≈0.5)."""
        from backend.core.ab_comparison import ABComparison, ABBlindTest

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.9
        cmp = ABComparison(audio_a, audio_b, sr=48000)
        bt = ABBlindTest(cmp, seed=42)
        bt.start_session(n_trials=100)

        count_a = sum(1 for t in bt.trials if t["x_is_a"])
        assert 35 <= count_a <= 65, f"Expected ~50 A-assignments, got {count_a}"


class TestSessionIsolation:
    """Session-Isolation: Sessions dürfen sich nicht beeinflussen."""

    def test_two_sessions_independent(self):
        """Zwei ABBlindTest-Sessions mit verschiedenem Seed erzeugen verschiedene X-Muster."""
        from backend.core.ab_comparison import ABComparison, ABBlindTest

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.9
        cmp = ABComparison(audio_a, audio_b, sr=48000)

        bt1 = ABBlindTest(cmp, seed=1)
        bt1.start_session(n_trials=10)
        x1 = [t["x_is_a"] for t in bt1.trials]

        bt2 = ABBlindTest(cmp, seed=99)
        bt2.start_session(n_trials=10)
        x2 = [t["x_is_a"] for t in bt2.trials]

        assert x1 != x2, "Sessions with different seeds should have different X patterns"

    def test_same_seed_reproduces(self):
        """Gleicher Seed = gleiches X-Muster (Reproduzierbarkeit)."""
        from backend.core.ab_comparison import ABComparison, ABBlindTest

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.9
        cmp = ABComparison(audio_a, audio_b, sr=48000)

        bt1 = ABBlindTest(cmp, seed=42)
        bt1.start_session(n_trials=10)
        x1 = [t["x_is_a"] for t in bt1.trials]

        bt2 = ABBlindTest(cmp, seed=42)
        bt2.start_session(n_trials=10)
        x2 = [t["x_is_a"] for t in bt2.trials]

        assert x1 == x2, "Same seed should produce identical X patterns"


class TestResultAggregation:
    """Ergebnis-Aggregation: Binomialtest, Preference-Statistik."""

    def test_perfect_discrimination_is_significant(self):
        """20/20 korrekt → p < 0.001 (hochsignifikant)."""
        from backend.core.ab_comparison import ABComparison, ABBlindTest

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.5  # Deutlich anders
        cmp = ABComparison(audio_a, audio_b, sr=48000)
        bt = ABBlindTest(cmp, seed=42)
        bt.start_session(n_trials=20)

        # Simuliere perfekte Diskrimination
        for t in bt.trials:
            bt.record_answer(t, chosen_a=t["x_is_a"], confidence=5)

        result = bt.get_result()
        assert result.correct == 20
        assert result.p_value < 0.001, f"Expected p<0.001 for 20/20, got {result.p_value}"
        assert result.is_significant

    def test_random_guessing_is_not_significant(self):
        """10/20 korrekt → p ≈ 0.5 (Raten)."""
        from backend.core.ab_comparison import ABComparison, ABBlindTest

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.99  # Kaum unterscheidbar
        cmp = ABComparison(audio_a, audio_b, sr=48000)
        bt = ABBlindTest(cmp, seed=42)
        bt.start_session(n_trials=20)

        # Simuliere Raten (10/20)
        n_correct = 0
        for t in bt.trials:
            # Immer "A" raten → ~50% korrekt wenn balanced
            bt.record_answer(t, chosen_a=True, confidence=1)
            if t["x_is_a"]:
                n_correct += 1

        result = bt.get_result()
        assert not result.is_significant or result.p_value >= 0.01

    def test_preference_tracking(self):
        """Preferences (A, B, none) werden korrekt gezählt."""
        from backend.core.ab_comparison import ABComparison, ABBlindTest

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.9
        cmp = ABComparison(audio_a, audio_b, sr=48000)
        bt = ABBlindTest(cmp, seed=42)
        bt.start_session(n_trials=10)

        for i, t in enumerate(bt.trials):
            pref = "a" if i < 4 else ("b" if i < 8 else "none")
            bt.record_answer(t, chosen_a=t["x_is_a"], preference=pref)

        result = bt.get_result()
        assert result.preference_a == 4
        assert result.preference_b == 4
        assert result.no_preference == 2


class TestABComparison:
    """A/B-Vergleich: Toggle, Delta, Segment."""

    def test_toggle_switches(self):
        """Toggle wechselt zwischen A und B."""
        from backend.core.ab_comparison import ABComparison

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.9
        cmp = ABComparison(audio_a, audio_b, sr=48000)

        assert cmp.current_is_a
        cmp.toggle()
        assert not cmp.current_is_a
        cmp.toggle()
        assert cmp.current_is_a

    def test_delta_computation(self):
        """Delta zwischen A und B wird korrekt berechnet."""
        from backend.core.ab_comparison import ABComparison

        audio_a = _make_test_audio()
        audio_b = audio_a * 0.5  # -6dB Unterschied
        cmp = ABComparison(audio_a, audio_b, sr=48000)
        delta = cmp.compute_delta()
        assert abs(delta.rms_delta_db - 6.0) < 2.0, f"Expected ~6dB delta, got {delta.rms_delta_db}"
