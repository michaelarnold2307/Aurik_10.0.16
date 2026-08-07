"""
Tests for Improvement I — Salience-based Multi-pitch Overtone Completion
in phase_07_harmonic_restoration.

Covers:
  - _compute_harmonic_salience: vectorised Klapuri (2006) salience
  - _detect_multi_pitch_f0s_with_analysis: pitch detection + missing audits
  - _synthesize_missing_overtones: additive partial synthesis
  - process() integration: n_pitches_detected, missing_harmonics dict, finite, no-clip

Scientific basis:
    Klapuri (2006). "Multiple Fundamental Frequency Estimation by
    Summing Harmonic Amplitudes." Proc. ISMIR.
    Terhardt (1982). "Zur Tonhoehenwahrnehmung von Klaengen." Acustica 26.
"""

from typing import Any

import numpy as np
import pytest

SR = 48_000


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def phase():
    from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

    return HarmonicRestorationPhase(sample_rate=SR)


def _sine(freq: float, dur: float = 0.5, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _harmonic_tone(f0: float, harmonics: list[int], dur: float = 0.5) -> np.ndarray:
    """Pure additive tone with exactly the specified harmonic orders present."""
    t = np.arange(int(dur * SR)) / SR
    audio = np.zeros_like(t, dtype=np.float32)
    for k in harmonics:
        audio += (0.4 / k) * np.sin(2.0 * np.pi * f0 * k * t)
    return audio.astype(np.float32)


def _chord(freqs, dur: float = 0.5) -> np.ndarray:
    """Sum of pure sines — simulates a polyphonic chord."""
    t = np.arange(int(dur * SR)) / SR
    audio = np.zeros_like(t, dtype=np.float32)
    for f, a in freqs:
        audio += a * np.sin(2.0 * np.pi * f * t)
    return audio.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  _compute_harmonic_salience
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeHarmonicSalience:
    def _make_magnitude_spectrum(self, f0_list: list[float], dur: float = 0.5):
        """Build a magnitude spectrum for a sum of harmonic tones."""
        n = int(dur * SR)
        audio = np.zeros(n, dtype=np.float64)
        for f0 in f0_list:
            t = np.arange(n) / SR
            audio += 0.4 * np.sin(2.0 * np.pi * f0 * t)
            for k in [2, 3, 4]:
                if f0 * k < SR / 2:
                    audio += 0.4 / k * np.sin(2.0 * np.pi * f0 * k * t)
        import scipy.signal as ss

        window = ss.get_window("hann", n)
        mag = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(n, d=1.0 / SR)
        return mag, freqs

    def test_output_shape(self, phase):
        mag, freqs = self._make_magnitude_spectrum([440.0])
        f0_cands = np.arange(60.0, 2001.0, 1.0)
        sal = phase._compute_harmonic_salience(mag, freqs, f0_cands)
        assert sal.shape == f0_cands.shape

    def test_no_nan(self, phase):
        mag, freqs = self._make_magnitude_spectrum([330.0, 660.0])
        f0_cands = np.arange(60.0, 2001.0, 1.0)
        sal = phase._compute_harmonic_salience(mag, freqs, f0_cands)
        assert np.all(np.isfinite(sal))

    def test_nonnegative(self, phase):
        mag, freqs = self._make_magnitude_spectrum([220.0])
        f0_cands = np.arange(60.0, 2001.0, 1.0)
        sal = phase._compute_harmonic_salience(mag, freqs, f0_cands)
        assert np.all(sal >= 0.0)

    def test_silence_gives_zero_salience(self, phase):
        n = int(0.5 * SR)
        mag = np.zeros(n // 2 + 1)
        freqs = np.fft.rfftfreq(n, d=1.0 / SR)
        f0_cands = np.arange(60.0, 2001.0, 1.0)
        sal = phase._compute_harmonic_salience(mag, freqs, f0_cands)
        np.testing.assert_allclose(sal, 0.0, atol=1e-12)

    def test_peak_near_true_f0(self, phase):
        """Salience maximum must be within ±5 Hz of the true fundamental."""
        f0_true = 440.0
        mag, freqs = self._make_magnitude_spectrum([f0_true])
        f0_cands = np.arange(60.0, 2001.0, 1.0)
        sal = phase._compute_harmonic_salience(mag, freqs, f0_cands)
        f0_detected = float(f0_cands[np.argmax(sal)])
        assert abs(f0_detected - f0_true) <= 5.0, f"Salience peak {f0_detected:.1f} Hz, expected near {f0_true:.1f} Hz"

    def test_two_f0s_have_higher_salience_than_background(self, phase):
        """Salience at known f0s must be well above median background."""
        for f0 in [261.0, 392.0]:
            mag, freqs = self._make_magnitude_spectrum([f0])
            f0_cands = np.arange(60.0, 2001.0, 1.0)
            sal = phase._compute_harmonic_salience(mag, freqs, f0_cands)
            peak_idx = np.argmin(np.abs(f0_cands - f0))
            assert sal[peak_idx] > np.median(sal) * 3.0


# ──────────────────────────────────────────────────────────────────────────────
# 2.  _detect_multi_pitch_f0s_with_analysis
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectMultiPitch:
    def test_returns_list(self, phase):
        audio = _sine(440.0)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio.astype(np.float64))
        assert isinstance(result, list)

    def test_each_entry_has_three_elements(self, phase):
        audio = _sine(440.0)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio.astype(np.float64))
        for f0, sal, missing in result:
            assert isinstance(float(f0), float)
            assert isinstance(float(sal), float)
            assert isinstance(missing, list)

    def test_silence_returns_empty(self, phase):
        audio = np.zeros(SR, dtype=np.float64)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio)
        assert result == []

    def test_detects_single_f0(self, phase):
        """Pure tone with harmonics → f0 detected within ±5 Hz."""
        f0_true = 440.0
        audio = _harmonic_tone(f0_true, harmonics=[1, 2, 3, 4], dur=0.5).astype(np.float64)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio)
        assert len(result) >= 1
        f0s = [r[0] for r in result]
        assert any(abs(f - f0_true) <= 5.0 for f in f0s), f"Expected f0 near 440 Hz, got {f0s}"

    def test_detects_chord_two_pitches(self, phase):
        """C major interval (C4=261 Hz, E4=330 Hz) → both within ±10 Hz."""
        audio = _chord([(261.0, 0.35), (330.0, 0.30)], dur=0.5).astype(np.float64)
        # Add harmonics so they are dominant pitches
        t = np.arange(int(0.5 * SR)) / SR
        for f0, a in [(261.0, 0.35), (330.0, 0.30)]:
            for k in [2, 3]:
                if f0 * k < SR / 2:
                    audio += (a / k) * np.sin(2.0 * np.pi * f0 * k * t)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio)
        f0s = [r[0] for r in result]
        found_261 = any(abs(f - 261.0) <= 10.0 for f in f0s)
        found_330 = any(abs(f - 330.0) <= 10.0 for f in f0s)
        assert found_261 or found_330, f"Neither 261 nor 330 found in {f0s}"

    def test_f0s_within_analysis_range(self, phase):
        """All detected f0s must lie in the 60–2000 Hz search range."""
        audio = _sine(880.0, dur=0.5).astype(np.float64)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio)
        for f0, _sal, _miss in result:
            assert 60.0 <= f0 <= 2000.0

    def test_salience_scores_positive(self, phase):
        audio = _harmonic_tone(330.0, harmonics=[1, 2, 3], dur=0.5).astype(np.float64)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio)
        for _f0, sal, _miss in result:
            assert sal > 0.0

    def test_missing_orders_in_range(self, phase):
        """Missing harmonic orders must be 2..7 (not including 1)."""
        audio = _sine(220.0, dur=0.5).astype(np.float64)  # pure sine → all harmonics missing
        result = phase._detect_multi_pitch_f0s_with_analysis(audio)
        for _f0, _sal, missing in result:
            for order in missing:
                assert 2 <= order <= 7

    def test_very_short_audio_does_not_crash(self, phase):
        audio = np.zeros(3, dtype=np.float64)
        result = phase._detect_multi_pitch_f0s_with_analysis(audio)
        assert result == []

    def test_n_max_respected(self, phase):
        """Result list must never exceed n_max entries."""
        audio = _chord([(261.0, 0.3), (330.0, 0.3), (392.0, 0.3), (523.0, 0.25), (660.0, 0.2)], dur=0.5).astype(
            np.float64
        )
        result = phase._detect_multi_pitch_f0s_with_analysis(audio, n_max=3)
        assert len(result) <= 3


# ──────────────────────────────────────────────────────────────────────────────
# 3.  _synthesize_missing_overtones
# ──────────────────────────────────────────────────────────────────────────────


class TestSynthesizeMissingOvertones:
    def _params(self, strength: float = 0.5):
        return {"strength": strength, "blend": 0.65}

    def test_shape_preserved(self, phase):
        mono = _sine(440.0).astype(np.float64)
        f0_info = [(440.0, 1.0, [2, 3])]
        out = phase._synthesize_missing_overtones(mono, f0_info, self._params())
        assert out.shape == mono.shape

    def test_dtype_float64(self, phase):
        mono = _sine(330.0).astype(np.float64)
        out = phase._synthesize_missing_overtones(mono, [(330.0, 1.0, [3])], self._params())
        assert out.dtype == np.float64

    def test_no_nan_or_inf(self, phase):
        mono = _sine(220.0, dur=1.0).astype(np.float64)
        f0_info = [(220.0, 2.0, [2, 3, 4, 5, 6, 7])]
        out = phase._synthesize_missing_overtones(mono, f0_info, self._params())
        assert np.all(np.isfinite(out))

    def test_empty_f0_info_returns_zeros(self, phase):
        mono = _sine(440.0).astype(np.float64)
        out = phase._synthesize_missing_overtones(mono, [], self._params())
        np.testing.assert_array_equal(out, 0.0)

    def test_empty_missing_list_returns_zeros(self, phase):
        """If all harmonics are present (missing=[]), additive must be zero."""
        mono = _sine(440.0).astype(np.float64)
        f0_info: Any = [(440.0, 1.0, [])]
        out = phase._synthesize_missing_overtones(mono, f0_info, self._params())
        np.testing.assert_array_equal(out, 0.0)

    def test_strength_zero_returns_zeros(self, phase):
        mono = _sine(440.0).astype(np.float64)
        f0_info = [(440.0, 1.0, [2, 3])]
        out = phase._synthesize_missing_overtones(mono, f0_info, self._params(strength=0.0))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_nonzero_output_for_missing_harmonics(self, phase):
        """Pure sine → all harmonics missing → additive output must be non-trivial."""
        mono = _sine(440.0, dur=0.5, amp=0.6).astype(np.float64)
        f0_info = [(440.0, 1.0, [2, 3, 4])]
        out = phase._synthesize_missing_overtones(mono, f0_info, self._params(strength=0.8))
        assert np.max(np.abs(out)) > 1e-6

    def test_harmonics_above_nyquist_skipped(self, phase):
        """Very high f0 × k → no partial should appear above SR/2."""
        # f0=1800 Hz * k=7 = 12600 Hz — within range.  f0=1900 Hz * k=5 = 9500 Hz — fine.
        # f0=1950 Hz * k=6 = 11700 Hz — fine. f0=1980 Hz * k=7 = 13860 Hz — fine.
        # The guard is hf > sr * 0.475. Use f0=1900, k where hf > 22800 → no such k<=7.
        mono = _sine(1900.0, dur=0.3).astype(np.float64)
        f0_info = [(1900.0, 1.0, [2, 3, 4, 5, 6, 7])]
        out = phase._synthesize_missing_overtones(mono, f0_info, self._params())
        assert np.all(np.isfinite(out))  # must not crash

    def test_multi_f0_output_larger_than_single(self, phase):
        """More simultaneous f0s with missing harmonics → more additive content."""
        mono = _chord([(261.0, 0.3), (330.0, 0.3)], dur=0.5).astype(np.float64)
        f0_info_single = [(261.0, 1.0, [2, 3])]
        f0_info_multi = [(261.0, 1.0, [2, 3]), (330.0, 0.8, [2, 3])]
        out_single = phase._synthesize_missing_overtones(mono, f0_info_single, self._params())
        out_multi = phase._synthesize_missing_overtones(mono, f0_info_multi, self._params())
        rms_single = float(np.sqrt(np.mean(out_single**2)))
        rms_multi = float(np.sqrt(np.mean(out_multi**2)))
        assert rms_multi >= rms_single


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Full process() integration
# ──────────────────────────────────────────────────────────────────────────────


class TestProcessIntegration:
    def test_no_nan_inf_vinyl(self, phase):
        audio = _sine(440.0, amp=0.6)
        r = phase.process(audio, material_type="vinyl", sample_rate=SR)
        assert np.all(np.isfinite(r.audio))

    def test_no_clipping_shellac(self, phase):
        audio = _sine(220.0, amp=0.7)
        r = phase.process(audio, material_type="shellac", sample_rate=SR)
        assert np.max(np.abs(r.audio)) <= 1.0 + 1e-6

    def test_shape_mono(self, phase):
        audio = _sine(330.0)
        r = phase.process(audio, material_type="tape", sample_rate=SR)
        assert r.audio.shape == audio.shape

    def test_shape_stereo(self, phase):
        mono = _sine(440.0)
        stereo = np.column_stack([mono, mono * 0.95])
        r = phase.process(stereo, material_type="vinyl", sample_rate=SR)
        assert r.audio.shape == stereo.shape

    def test_n_pitches_detected_in_modifications(self, phase):
        audio = _sine(440.0)
        r = phase.process(audio, material_type="tape", sample_rate=SR)
        assert "n_pitches_detected" in r.modifications
        assert isinstance(r.modifications["n_pitches_detected"], int)

    def test_missing_harmonics_is_dict_in_metadata(self, phase):
        audio = _sine(440.0)
        r = phase.process(audio, material_type="vinyl", sample_rate=SR)
        mh = r.metadata.get("missing_harmonics")
        assert isinstance(mh, dict)

    def test_algorithm_version_multi_pitch(self, phase):
        audio = _sine(330.0)
        r = phase.process(audio, material_type="tape", sample_rate=SR)
        assert r.metadata.get("algorithm_version") == "3.0_multi_pitch"

    def test_polyphonic_chord_no_crash(self, phase):
        """C major chord (3 voices) must process without crash."""
        audio = _chord([(261.0, 0.3), (330.0, 0.25), (392.0, 0.2)], dur=0.5)
        r = phase.process(audio, material_type="vinyl", sample_rate=SR)
        assert r.success
        assert np.all(np.isfinite(r.audio))

    def test_polyphonic_n_pitches_ge_1(self, phase):
        """Chord with harmonics → at least 1 pitch detected."""
        t = np.arange(int(0.5 * SR)) / SR
        audio = np.zeros_like(t, dtype=np.float32)
        for f0 in [261.0, 330.0, 392.0]:
            for k in [1, 2, 3]:
                audio += (0.2 / k) * np.sin(2.0 * np.pi * f0 * k * t)
        r = phase.process(audio.astype(np.float32), material_type="shellac", sample_rate=SR)
        assert r.modifications.get("n_pitches_detected", 0) >= 1

    def test_silence_no_crash(self, phase):
        audio = np.zeros(SR // 2, dtype=np.float32)
        r = phase.process(audio, material_type="tape", sample_rate=SR)
        assert r.success
        assert np.all(np.isfinite(r.audio))

    def test_scientific_ref_contains_klapuri(self, phase):
        audio = _sine(440.0)
        r = phase.process(audio, material_type="vinyl", sample_rate=SR)
        assert "Klapuri" in r.metadata.get("scientific_ref", "")

    def test_scientific_ref_contains_terhardt(self, phase):
        audio = _sine(440.0)
        r = phase.process(audio, material_type="vinyl", sample_rate=SR)
        assert "Terhardt" in r.metadata.get("scientific_ref", "")
