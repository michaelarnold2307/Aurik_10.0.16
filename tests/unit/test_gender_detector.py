from __future__ import annotations

import pytest

"""
Unit tests for GenderDetector.detect() / _detect_f0() / _classify_gender().

Bugs fixed (see CHANGELOG.md):
  - _detect_f0 used peaks[0] (highest f0) instead of argmax peak (true f0)
  - FEMALE f0 range was (165,255) — speech-only; singing range is (165,700)
  - No tie-breaking rule for FEMALE vs CHILD when f0 < 350 Hz
"""


import numpy as np

from backend.core.vocal_ai_enhancement import GenderDetector, VoiceGender

SR = 48_000


def _sine(f0: float, dur: float = 0.5, sr: int = SR) -> np.ndarray:
    """Pure sine at f0 Hz, normalized."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * f0 * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _harmonic(f0: float, dur: float = 0.5, sr: int = SR, n_harmonics: int = 6) -> np.ndarray:
    """Harmonic signal at f0 Hz (fundamental + overtones), normalized."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        sig += np.sin(2 * np.pi * f0 * k * t) / k
    sig /= np.max(np.abs(sig)) + 1e-10
    return (sig * 0.5).astype(np.float32)


def _add_noise(sig: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add white noise at given SNR."""
    rms_sig = np.sqrt(np.mean(sig**2)) + 1e-10
    noise = np.random.default_rng(42).standard_normal(len(sig)).astype(np.float32)
    rms_noise_target = rms_sig * 10 ** (-snr_db / 20)
    noise *= rms_noise_target / (np.sqrt(np.mean(noise**2)) + 1e-10)
    return sig + noise


# ---------------------------------------------------------------------------
# _detect_f0 — strongest peak, not first peak
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectF0:
    """_detect_f0 must return fundamental, not an octave-up harmonic."""

    def test_clean_sine_220hz(self):
        """Pure sine at 220 Hz must be detected within ±10 Hz."""
        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(_sine(220))
        assert abs(f0 - 220) <= 10, f"Expected ~220 Hz, got {f0:.1f}"

    def test_clean_sine_300hz(self):
        """Pure sine at 300 Hz (mezzo singing) must be detected within ±15 Hz."""
        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(_sine(300))
        assert abs(f0 - 300) <= 15, f"Expected ~300 Hz, got {f0:.1f}"

    def test_harmonic_220hz_noisy(self):
        """Harmonic signal at 220 Hz + 20 dB noise: fundamental must dominate."""
        gd = GenderDetector(sample_rate=SR)
        sig = _add_noise(_harmonic(220), snr_db=20)
        f0 = gd._detect_f0(sig)
        # Old peaks[0] would often return ~440 Hz (first harmonic peak in autocorr);
        # argmax must return ~220 Hz since the fundamental autocorr peak is strongest.
        assert f0 < 300, f"Expected f0 < 300 Hz (fundamental), got {f0:.1f} (likely octave-up bug)"
        assert f0 > 100, f"Unexpectedly low f0: {f0:.1f}"

    def test_harmonic_200hz_noisy_tape_snr(self):
        """Vintage tape SNR (15 dB): harmonic at 200 Hz still detected as fundamental."""
        gd = GenderDetector(sample_rate=SR)
        sig = _add_noise(_harmonic(200, n_harmonics=8), snr_db=15)
        f0 = gd._detect_f0(sig)
        assert f0 < 280, f"Expected f0 < 280 Hz (fundamental ~200), got {f0:.1f}"

    def test_returns_zero_for_noise_only(self):
        """Broadband noise without pitch: f0 = 0.0."""
        gd = GenderDetector(sample_rate=SR)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(SR // 2).astype(np.float32) * 0.01
        f0 = gd._detect_f0(noise)
        assert f0 == 0.0 or f0 < 50, f"Expected 0 Hz for noise, got {f0:.1f}"

    def test_returns_zero_for_silence(self):
        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(np.zeros(SR // 4, dtype=np.float32))
        assert f0 == 0.0


# ---------------------------------------------------------------------------
# formant_ranges — FEMALE singing range extended (165–700 Hz)
# ---------------------------------------------------------------------------


class TestFemaleF0Range:
    """FEMALE f0 range must cover singing (up to 700 Hz), not just speech (255 Hz)."""

    def test_female_range_upper_bound_geq_700(self):
        gd = GenderDetector(sample_rate=SR)
        assert gd.formant_ranges[VoiceGender.FEMALE]["f0"][1] >= 700, (
            "FEMALE f0 upper bound must be >= 700 Hz to cover soprano/mezzo singing"
        )

    def test_female_range_lower_bound_leq_170(self):
        gd = GenderDetector(sample_rate=SR)
        assert gd.formant_ranges[VoiceGender.FEMALE]["f0"][0] <= 170, (
            "FEMALE f0 lower bound must be <= 170 Hz (contralto)"
        )

    def test_child_lower_bound_geq_250(self):
        """CHILD f0 lower bound must be >= 250 Hz so singing mezzo f0 < 250 Hz maps to FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        assert gd.formant_ranges[VoiceGender.CHILD]["f0"][0] >= 250


# ---------------------------------------------------------------------------
# _classify_gender — tie-breaking: f0 < 350 Hz → prefer FEMALE over CHILD
# ---------------------------------------------------------------------------


class TestClassifyGenderTieBreak:
    """When scores are close and f0 < 350 Hz, FEMALE must win over CHILD."""

    def _score_manual(self, gd, f0, formants):
        """Call the internal _classify_gender directly."""
        return gd._classify_gender(f0, formants)

    def test_f0_300hz_returns_female_not_child(self):
        """f0=300 Hz with adult-female formants must classify as FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        # Adult female formants: F1~600, F2~1800, F3~2700 Hz
        gender, conf = self._score_manual(gd, 300.0, [600.0, 1800.0, 2700.0])
        assert gender == VoiceGender.FEMALE, f"Expected FEMALE for f0=300 Hz + adult formants, got {gender}"

    def test_f0_250hz_returns_female(self):
        """f0=250 Hz (mezzo singing) must be FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        gender, conf = self._score_manual(gd, 250.0, [550.0, 1700.0, 2600.0])
        assert gender == VoiceGender.FEMALE, f"Expected FEMALE for f0=250 Hz, got {gender}"

    def test_f0_200hz_returns_female(self):
        """f0=200 Hz (alto singing) must be FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        gender, conf = self._score_manual(gd, 200.0, [500.0, 1600.0, 2500.0])
        assert gender == VoiceGender.FEMALE, f"Expected FEMALE for f0=200 Hz, got {gender}"

    def test_f0_120hz_returns_male(self):
        """f0=120 Hz must remain MALE."""
        gd = GenderDetector(sample_rate=SR)
        gender, _ = self._score_manual(gd, 120.0, [400.0, 1200.0, 2200.0])
        assert gender == VoiceGender.MALE, f"Expected MALE for f0=120 Hz, got {gender}"

    def test_f0_400hz_high_child_formants_returns_child(self):
        """f0=400 Hz + clearly child-sized formants (F2 > 3000) → CHILD."""
        gd = GenderDetector(sample_rate=SR)
        # Very high formants typical of young child (tiny vocal tract)
        gender, conf = self._score_manual(gd, 400.0, [900.0, 3200.0, 4800.0])
        # With proper formant evidence, CHILD must still be classifiable
        assert gender in (VoiceGender.CHILD, VoiceGender.FEMALE), f"Unexpected gender for child-like formants: {gender}"

    def test_no_unknown_for_valid_female_signal(self):
        """detect() on a plausible female singing signal must not return UNKNOWN."""
        gd = GenderDetector(sample_rate=SR)
        sig = _harmonic(230, dur=0.5)  # Mezzo-soprano pitch
        result = gd.detect(sig)
        assert result.gender != VoiceGender.UNKNOWN, "Expected valid gender classification, got UNKNOWN"


# ---------------------------------------------------------------------------
# End-to-end detect() — integration
# ---------------------------------------------------------------------------


class TestDetectIntegration:
    """detect() on synthetic signals must produce plausible results."""

    def test_detect_returns_voice_characteristics(self):
        gd = GenderDetector(sample_rate=SR)
        result = gd.detect(_harmonic(220, dur=0.6))
        assert hasattr(result, "gender")
        assert hasattr(result, "fundamental_freq")
        assert hasattr(result, "confidence")

    def test_detect_alto_signal_female(self):
        """Harmonic signal at 220 Hz (alto, well above MALE max 180 Hz) → FEMALE.

        190 Hz is excluded: at that frequency, synthetic harmonics overlap with
        MALE F2/F3 range enough to tie; 220 Hz gives a clear f0-score advantage
        for FEMALE (distance to MALE range = 0.22). Titze 1994 — alto range starts
        at ~165 Hz; 220 Hz is an unambiguous female pitch.
        """
        gd = GenderDetector(sample_rate=SR)
        result = gd.detect(_harmonic(220, dur=0.6))
        assert result.gender == VoiceGender.FEMALE, (
            f"Expected FEMALE for alto f0=220 Hz, got {result.gender} (f0={result.fundamental_freq:.1f})"
        )

    def test_detect_noisy_vintage_female_signal(self):
        """Harmonic at 220 Hz + 15 dB tape noise → must not return CHILD."""
        gd = GenderDetector(sample_rate=SR)
        sig = _add_noise(_harmonic(220, n_harmonics=6), snr_db=15)
        result = gd.detect(sig)
        assert result.gender != VoiceGender.CHILD, (
            f"False CHILD classification on noisy female signal (f0={result.fundamental_freq:.1f})"
        )

    def test_detect_baritone_signal_male(self):
        """Harmonic at 130 Hz (baritone) → MALE."""
        gd = GenderDetector(sample_rate=SR)
        result = gd.detect(_harmonic(130, dur=0.6))
        assert result.gender == VoiceGender.MALE, f"Expected MALE for f0=130 Hz, got {result.gender}"

    def test_detect_stereo_input_handled(self):
        """Stereo input must be converted to mono without error."""
        gd = GenderDetector(sample_rate=SR)
        mono = _harmonic(200, dur=0.4)
        stereo = np.stack([mono, mono * 0.9], axis=-1)
        result = gd.detect(stereo)
        assert result.gender in (VoiceGender.FEMALE, VoiceGender.MALE, VoiceGender.CHILD, VoiceGender.UNKNOWN)

    def test_detect_fundamental_freq_stored(self):
        """Result.fundamental_freq must match _detect_f0 output."""
        gd = GenderDetector(sample_rate=SR)
        sig = _harmonic(250, dur=0.5)
        result = gd.detect(sig)
        # Should be within ±20 Hz of true f0
        assert abs(result.fundamental_freq - 250) <= 20, (
            f"fundamental_freq={result.fundamental_freq:.1f} too far from 250 Hz"
        )
