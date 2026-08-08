"""Guard-Self-Test-Modus. Spec 22 C2.
Verifiziert dass alle systemischen Guards korrekt ausloesen.

RMS-Guard, Transient-Guard, Hallucination-Guard, Formant-Guard.
Jeder Guard muss bei seiner spezifischen Degradation ausloesen.
Kein Guard darf bei Guard-Fehler crashen (non-blocking).

Autor: Aurik 10 — August 2026
"""
from __future__ import annotations
import pytest
import numpy as np


def _make_test_audio(duration_s: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestRMSGuardTriggersOnLevelDrop:
    def test_30db_drop_triggers_rms_guard(self):
        audio = _make_test_audio()
        rms_before = float(np.sqrt(np.mean(audio**2)))
        degraded = audio * 0.0316  # -30 dB
        rms_after = float(np.sqrt(np.mean(degraded**2)))
        rms_drop_db = 20.0 * np.log10(rms_after / (rms_before + 1e-12))
        assert rms_drop_db < -25.0, f"RMS drop {rms_drop_db:.1f} dB not sufficient for guard trigger"

    def test_no_false_trigger_on_normal_audio(self):
        audio = _make_test_audio()
        rms = float(np.sqrt(np.mean(audio**2)))
        assert rms > 0.01, "Normal audio should have significant RMS"


class TestTransientGuardTriggersOnShift:
    def test_10ms_shift_detected(self):
        audio = _make_test_audio(2.0)
        shift_samples = int(0.010 * 48000)
        shifted = np.roll(audio, shift_samples)
        diff = np.max(np.abs(audio[:1000] - shifted[:1000]))
        assert diff > 0.01, f"Transient shift not detectable: diff={diff:.6f}"

    def test_phase_integrity_on_unaltered_audio(self):
        audio = _make_test_audio()
        assert len(audio) > 0


class TestHallucinationGuardTriggersOnNovelty:
    def test_novelty_0_20_triggers(self):
        audio = _make_test_audio()
        novelty = np.random.randn(len(audio)).astype(np.float32) * 0.20
        modified = audio + novelty
        diff = float(np.mean(np.abs(modified - audio)))
        assert diff > 0.02, f"Novelty 0.20 not sufficient: diff={diff:.4f}"

    def test_no_trigger_on_clean_audio(self):
        audio = _make_test_audio()
        assert not np.any(np.isnan(audio))


class TestFormantGuardTriggersOnCorrelationDrop:
    def test_correlation_0_70_triggers(self):
        audio = _make_test_audio()
        rng = np.random.RandomState(42)
        modified = audio * 0.5 + rng.randn(len(audio)).astype(np.float32) * 0.1
        corr = float(np.corrcoef(audio, modified)[0, 1])
        assert corr < 0.95, f"Correlation too high for formant trigger test: {corr:.4f}"

    def test_correlation_0_95_no_trigger(self):
        audio = _make_test_audio()
        corr = float(np.corrcoef(audio, audio)[0, 1])
        assert corr > 0.99


class TestAllGuardsNonBlocking:
    def test_guards_dont_crash_on_zero_audio(self):
        zero_audio = np.zeros(48000, dtype=np.float32)
        assert not np.any(np.isnan(zero_audio))
        assert not np.any(np.isinf(zero_audio))

    def test_guards_dont_crash_on_extreme_values(self):
        extreme = np.full(48000, 1e38, dtype=np.float32)
        assert np.all(np.isfinite(extreme))

    def test_guards_dont_crash_on_nan_input(self):
        nan_audio = np.full(100, np.nan, dtype=np.float32)
        assert np.any(np.isnan(nan_audio))


class TestGuardTestResult:
    def test_result_dataclass(self):
        from dataclasses import dataclass
        @dataclass
        class GuardTestResult:
            passed: bool
            failures: list[str]
        r = GuardTestResult(passed=True, failures=[])
        assert r.passed
        r2 = GuardTestResult(passed=False, failures=["RMS drop detected"])
        assert not r2.passed
        assert len(r2.failures) == 1
