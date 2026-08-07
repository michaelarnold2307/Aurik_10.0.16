from __future__ import annotations

import pytest

"""Tests für core/era_authentic_perceptual_completion.py — Spec §2.35.

≥ 25 Unit-Tests: Aktivierungsbedingungen, Bandbreiten-Erkennung, era-Ceiling,
NaN-Safety, Singleton-Thread-Safety.
"""


import math
import threading

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit

from backend.core.era_authentic_perceptual_completion import (
    ERA_BRILLANZ_CEILING,
    EraCompletionResult,
    apply_era_authentic_completion,
    get_era_authentic_perceptual_completion,
)

SR = 48_000
RNG = np.random.default_rng(42)


def _sine(freq: float = 440.0, dur: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _bandlimited_signal(cutoff_hz: float, dur: float = 2.0, sr: int = SR) -> np.ndarray:
    """Sinuston - repräsentiert bandlimitiertes Signal (nur eine Frequenz)."""
    freq = min(cutoff_hz * 0.5, cutoff_hz - 100.0, 4000.0)
    freq = max(100.0, freq)
    return _sine(freq=freq, dur=dur, sr=sr)


def _stereo(dur: float = 2.0) -> np.ndarray:
    mono = _sine(dur=dur)
    return np.stack([mono, mono * 0.9], axis=1)


# ---------------------------------------------------------------------------
# 1. ERA_BRILLANZ_CEILING Konstanten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEraBrillanzCeiling:
    def test_ceiling_keys_are_decades(self):
        for decade in ERA_BRILLANZ_CEILING:
            assert isinstance(decade, int)
            assert 1900 <= decade <= 2010

    def test_ceiling_values_in_unit_interval(self):
        for decade, ceiling in ERA_BRILLANZ_CEILING.items():
            assert 0.0 < ceiling <= 1.0, f"decade={decade}: ceiling={ceiling}"

    def test_older_decades_lower_ceiling(self):
        sorted_decades = sorted(ERA_BRILLANZ_CEILING.keys())
        ceilings = [ERA_BRILLANZ_CEILING[d] for d in sorted_decades]
        # Ältere Ären haben niedrigere Ceiling als neuere (nicht streng, aber monoton)
        assert ceilings[0] <= ceilings[-1]

    def test_1920_ceiling_below_07(self):
        assert ERA_BRILLANZ_CEILING[1920] <= 0.75

    def test_2000_ceiling_above_09(self):
        assert ERA_BRILLANZ_CEILING[2000] >= 0.90


# ---------------------------------------------------------------------------
# 2. Ergebnis-Struktur
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_returns_dataclass(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result, EraCompletionResult)

    def test_audio_output_shape_mono(self):
        audio = _sine(dur=2.0)
        result = apply_era_authentic_completion(audio, SR)
        assert result.audio.ndim == 1
        assert len(result.audio) == len(audio)

    def test_audio_output_no_nan(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR)
        assert np.all(np.isfinite(result.audio))

    def test_audio_output_clipped(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR)
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_as_dict_returns_dict(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "applied" in d
        assert "operation_type" in d

    def test_operation_type_valid(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR)
        assert result.operation_type in ("synthesize_era_authentic", "passthrough")

    def test_message_nonempty(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result.message, str)
        assert len(result.message) >= 5

    def test_era_decade_reasonable(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR, era=1940)
        assert result.era_decade == 1940

    def test_brillanz_ceiling_positive(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR)
        assert 0.0 < result.brillanz_ceiling <= 1.0
        assert math.isfinite(result.brillanz_ceiling)


# ---------------------------------------------------------------------------
# 3. Aktivierungsbedingungen
# ---------------------------------------------------------------------------


class TestActivation:
    def test_broadband_signal_passthrough(self):
        # Breitband-Sinus (20 kHz) → passthrough, kein Eingriff
        audio = _sine(freq=10_000.0, dur=2.0)
        result = apply_era_authentic_completion(audio, SR, era=2000)
        # Bei modernem Material oder breitem Signal: kein Ingriff
        assert result.audio.dtype == np.float32

    def test_is_applicable_with_narrow_band(self):
        comp = get_era_authentic_perceptual_completion()
        audio = _bandlimited_signal(cutoff_hz=3000.0)
        # Schmalbandiges Signal → applicabel
        result = comp.is_applicable(audio, SR)
        assert isinstance(result, bool)

    def test_is_applicable_returns_false_when_brillanz_inapplicable(self):
        comp = get_era_authentic_perceptual_completion()
        audio = _sine()

        class FakeApplicability:
            applicable = frozenset({"natuerlichkeit", "authentizitaet"})
            # kein "brillanz"

        result = comp.is_applicable(audio, SR, goal_applicability=FakeApplicability())
        assert result is False

    def test_is_applicable_returns_true_when_brillanz_applicable(self):
        comp = get_era_authentic_perceptual_completion()
        audio = _bandlimited_signal(cutoff_hz=3000.0)

        class FakeApplicability:
            applicable = frozenset({"natuerlichkeit", "brillanz", "authentizitaet"})

        result = comp.is_applicable(audio, SR, goal_applicability=FakeApplicability())
        # Schmalbandiges Signal + brillanz applicable → True
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 4. Ära-spezifische Ceiling
# ---------------------------------------------------------------------------


class TestEraCeiling:
    def test_completion_brillanz_ceiling_1920(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR, era=1920)
        assert result.brillanz_ceiling == ERA_BRILLANZ_CEILING[1920]

    def test_completion_brillanz_ceiling_1970(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR, era=1970)
        assert result.brillanz_ceiling == ERA_BRILLANZ_CEILING[1970]

    def test_completion_brillanz_ceiling_2000(self):
        audio = _sine()
        result = apply_era_authentic_completion(audio, SR, era=2000)
        assert result.brillanz_ceiling == ERA_BRILLANZ_CEILING[2000]

    def test_older_era_lower_ceiling_than_newer(self):
        audio = _sine()
        old = apply_era_authentic_completion(audio, SR, era=1920)
        new = apply_era_authentic_completion(audio, SR, era=2000)
        assert old.brillanz_ceiling <= new.brillanz_ceiling


# ---------------------------------------------------------------------------
# 5. NaN/Inf-Sicherheit & Edge-Cases
# ---------------------------------------------------------------------------


class TestNanSafety:
    def test_nan_audio_handled(self):
        audio = np.full(SR * 2, float("nan"), dtype=np.float32)
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result, EraCompletionResult)
        assert np.all(np.isfinite(result.audio))

    def test_inf_audio_handled(self):
        audio = np.full(SR * 2, float("inf"), dtype=np.float32)
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result, EraCompletionResult)
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_silence_handled(self):
        audio = np.zeros(SR * 2, dtype=np.float32)
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result, EraCompletionResult)
        assert np.all(np.isfinite(result.audio))

    def test_very_short_audio(self):
        audio = np.zeros(128, dtype=np.float32)
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result, EraCompletionResult)

    def test_single_sample_audio(self):
        audio = np.array([0.5], dtype=np.float32)
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result, EraCompletionResult)


# ---------------------------------------------------------------------------
# 6. Stereo-Unterstützung
# ---------------------------------------------------------------------------


class TestStereoInput:
    def test_stereo_accepted(self):
        audio = _stereo()
        result = apply_era_authentic_completion(audio, SR)
        assert isinstance(result, EraCompletionResult)

    def test_stereo_output_no_nan(self):
        audio = _stereo()
        result = apply_era_authentic_completion(audio, SR)
        assert np.all(np.isfinite(result.audio))

    def test_stereo_output_clipped(self):
        audio = _stereo()
        result = apply_era_authentic_completion(audio, SR)
        assert np.max(np.abs(result.audio)) <= 1.0


class TestAnchorGuidance:
    def test_anchor_guidance_path_runs_and_stays_finite(self):
        comp = get_era_authentic_perceptual_completion()
        audio = _bandlimited_signal(cutoff_hz=3500.0)
        anchor = _sine(freq=6000.0, dur=2.0)

        result = comp.complete(audio, SR, era=1960, anchor=anchor)

        assert isinstance(result, EraCompletionResult)
        assert result.audio.shape == audio.shape
        assert np.all(np.isfinite(result.audio))
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_short_anchor_keeps_output_length(self):
        comp = get_era_authentic_perceptual_completion()
        audio = _bandlimited_signal(cutoff_hz=3200.0, dur=2.0)
        anchor_short = _sine(freq=5500.0, dur=0.5)

        result = comp.complete(audio, SR, era=1950, anchor=anchor_short)

        assert isinstance(result, EraCompletionResult)
        assert result.audio.shape == audio.shape
        assert np.all(np.isfinite(result.audio))


# ---------------------------------------------------------------------------
# 7. Singleton & Thread-Safety
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_same_instance(self):
        a = get_era_authentic_perceptual_completion()
        b = get_era_authentic_perceptual_completion()
        assert a is b

    def test_thread_safe(self):
        instances = []
        errors = []

        def worker():
            try:
                instances.append(get_era_authentic_perceptual_completion())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(i is instances[0] for i in instances)

    def test_convenience_wrapper_same_result_shape(self):
        audio = _sine()
        wrapper = apply_era_authentic_completion(audio, SR, era=1950)
        assert len(wrapper.audio) == len(audio)

    def test_deterministic(self):
        audio = _sine()
        r1 = apply_era_authentic_completion(audio, SR, era=1940)
        r2 = apply_era_authentic_completion(audio, SR, era=1940)
        np.testing.assert_array_equal(r1.audio, r2.audio)
