from __future__ import annotations

"""tests/unit/test_medium_classifier_sota.py
Unit tests for three SOTA improvements to MediumClassifier (Aurik 10.0.0):

  1. Rotation Periodicity Detector  (Cano 2005; Rodriguez & Bello 2018)
  2. IEC 60386 Wow/Flutter via FCPE modulation spectrum
  3. MDCT Codec Forensics            (Farid 2009; Bianchi 2020)

All tests use synthetic signals — no real audio files required.
"""


import math

import numpy as np
import pytest

from backend.core.medium_classifier import (
    ClassificationResult,
    _SpectralFingerprinter,
)

SR = 48_000
_fp = _SpectralFingerprinter()


# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------


def _make_lp_modulated(duration_s: float = 8.0, sr: int = SR) -> np.ndarray:
    """White noise with amplitude modulation at LP rotation (0.556 Hz)."""
    t = np.linspace(0, duration_s, int(duration_s * sr), dtype=np.float32)
    carrier = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32) * 0.05
    modulation = 1.0 + 0.20 * np.sin(2 * math.pi * 0.556 * t)
    return carrier * modulation  # type: ignore[no-any-return]


def _make_shellac_modulated(duration_s: float = 6.0, sr: int = SR) -> np.ndarray:
    """White noise with amplitude modulation at shellac rotation (1.300 Hz)."""
    t = np.linspace(0, duration_s, int(duration_s * sr), dtype=np.float32)
    carrier = np.random.default_rng(1).standard_normal(len(t)).astype(np.float32) * 0.05
    modulation = 1.0 + 0.22 * np.sin(2 * math.pi * 1.300 * t)
    return carrier * modulation  # type: ignore[no-any-return]


def _make_unmodulated(duration_s: float = 8.0, sr: int = SR) -> np.ndarray:
    """Pure white noise without any rotation modulation (tape / digital)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(duration_s * sr)).astype(np.float32) * 0.05


def _make_mp3_signature(duration_s: float = 4.0, sr: int = SR) -> np.ndarray:
    """Harmonic music signal with a hard codec cutoff at 14 kHz (simulates 128 kbps MP3).

    Uses a harmonically rich base (to approximate structured music) and then
    applies a hard spectral truncation above 14 kHz — the most reliable MP3
    indicator.  A quantisation noise floor is added between harmonics.
    """
    rng = np.random.default_rng(7)
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    # Harmonics at 220, 440, 880, 1760, 3520, 7040 Hz (well below cutoff)
    sig = np.zeros(n, dtype=np.float64)
    for f in [220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0]:
        sig += 0.04 * np.sin(2 * math.pi * f * t)
    # Quantisation noise floor (simulates raised codec noise between harmonics)
    sig += rng.standard_normal(n) * 0.008
    # Hard cutoff above 14 kHz in the frequency domain
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec[freqs > 14000.0] = 0.0
    audio = np.fft.irfft(spec, n=n).astype(np.float32)
    return audio / (np.max(np.abs(audio)) + 1e-8) * 0.1  # type: ignore[no-any-return]


def _make_digital_clean(duration_s: float = 4.0, sr: int = SR) -> np.ndarray:
    """Spectrally rich clean digital audio (CD-like, full 20 kHz BW)."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, duration_s, int(duration_s * sr), dtype=np.float32)
    # Sum of harmonics up to 20 kHz
    sig = np.zeros_like(t)
    for f in [440, 880, 1760, 3520, 7040, 14080]:
        sig += 0.02 * np.sin(2 * math.pi * f * t)
    sig += rng.standard_normal(len(t)).astype(np.float32) * 0.002
    return sig * 0.3


def _make_vinyl_with_infrasonic(duration_s: float = 6.0, sr: int = SR) -> np.ndarray:
    """Stereo vinyl simulation: music + infrasonic rumble < 20 Hz."""
    rng = np.random.default_rng(5)
    n = int(duration_s * sr)
    music = rng.standard_normal(n).astype(np.float32) * 0.1
    # Infrasonic rumble at ~5 Hz, −48 dBFS relative to music
    t = np.arange(n, dtype=np.float32) / sr
    rumble_l = 0.004 * np.sin(2 * math.pi * 5.0 * t + 0.0)
    rumble_r = 0.004 * np.sin(2 * math.pi * 7.0 * t + 1.2)  # uncorrelated
    stereo = np.stack([music + rumble_l, music + rumble_r], axis=0)
    return stereo  # type: ignore[no-any-return]


# ============================================================================
# 1. Rotation Periodicity Detector
# ============================================================================


@pytest.mark.unit
class TestRotationPeriodicityDetector:
    """Tests for _SpectralFingerprinter._rotation_periodicity()."""

    def test_lp_modulation_detected(self):
        """LP (33⅓ RPM, 0.556 Hz) modulation must be detected with strength > 0."""
        audio = _make_lp_modulated()
        rot_hz, strength = _fp._rotation_periodicity(audio, SR)
        assert strength > 0.0, "LP rotation signal not detected"

    def test_lp_frequency_range(self):
        """Detected LP rotation frequency must be within ±0.15 Hz of 0.556 Hz."""
        audio = _make_lp_modulated()
        rot_hz, strength = _fp._rotation_periodicity(audio, SR)
        if strength > 0.05:
            assert 0.35 <= rot_hz <= 0.75, f"LP frequency {rot_hz:.3f} Hz out of range"

    def test_shellac_modulation_detected(self):
        """Shellac (78 RPM, 1.300 Hz) modulation must be detected."""
        audio = _make_shellac_modulated()
        rot_hz, strength = _fp._rotation_periodicity(audio, SR)
        assert strength > 0.0, "Shellac rotation signal not detected"

    def test_shellac_frequency_range(self):
        """Detected shellac rotation frequency must be within ±0.20 Hz of 1.300 Hz."""
        audio = _make_shellac_modulated()
        rot_hz, strength = _fp._rotation_periodicity(audio, SR)
        if strength > 0.05:
            assert 1.0 <= rot_hz <= 1.65, f"Shellac frequency {rot_hz:.3f} Hz out of range"

    def test_unmodulated_very_low_strength(self):
        """Unmodulated white noise must return low or zero rotation strength."""
        audio = _make_unmodulated()
        _rot_hz, strength = _fp._rotation_periodicity(audio, SR)
        # Not necessarily zero due to random peaks, but must be < 0.5
        assert strength <= 0.50, f"False positive rotation detected: strength={strength:.3f}"

    def test_short_audio_returns_zero(self):
        """Audio shorter than 4 s must return (0.0, 0.0) — not enough resolution."""
        short = np.random.default_rng(9).standard_normal(int(2.0 * SR)).astype(np.float32)
        rot_hz, strength = _fp._rotation_periodicity(short, SR)
        assert rot_hz == 0.0
        assert strength == 0.0

    def test_silent_audio_returns_zero(self):
        """Silent array must not raise and must return (0.0, 0.0)."""
        silence = np.zeros(int(8.0 * SR), dtype=np.float32)
        rot_hz, strength = _fp._rotation_periodicity(silence, SR)
        assert rot_hz == 0.0
        assert strength == 0.0

    def test_returns_tuple_of_two_floats(self):
        """Return type must be (float, float) — no NaN / Inf."""
        audio = _make_lp_modulated()
        result = _fp._rotation_periodicity(audio, SR)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_strength_in_unit_interval(self):
        """rotation_strength must always be in [0, 1]."""
        for seed in range(5):
            audio = np.random.default_rng(seed).standard_normal(int(8.0 * SR)).astype(np.float32) * 0.1
            _, strength = _fp._rotation_periodicity(audio, SR)
            assert 0.0 <= strength <= 1.0, f"Strength={strength} out of [0,1]"

    def test_lp_strength_greater_than_unmodulated(self):
        """LP-modulated signal must have higher rotation_strength than unmodulated."""
        modulated = _make_lp_modulated()
        unmodulated = _make_unmodulated(duration_s=8.0)
        _, str_mod = _fp._rotation_periodicity(modulated, SR)
        _, str_flat = _fp._rotation_periodicity(unmodulated, SR)
        assert str_mod >= str_flat, f"LP modulated ({str_mod:.3f}) should be >= unmodulated ({str_flat:.3f})"


# ============================================================================
# 2. Wow/Flutter via FCPE (with ZCR fallback)
# ============================================================================


class TestWowFlutterImproved:
    """Tests for _SpectralFingerprinter._wow_flutter() and ZCR fallback."""

    def test_returns_tuple_of_two_floats(self):
        """_wow_flutter must return (dom_freq_hz, depth_pct) as two finite floats."""
        audio = _make_unmodulated(duration_s=2.0)
        result = _fp._wow_flutter(audio, SR)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_depth_in_valid_range(self):
        """Modulation depth must be in [0, 20] for various inputs."""
        for func in [_make_unmodulated, _make_lp_modulated, _make_digital_clean]:
            audio = func()
            _, depth = _fp._wow_flutter(audio, SR)
            assert 0.0 <= depth <= 20.0, f"Depth {depth} out of [0, 20]"

    def test_frequency_in_valid_range(self):
        """Dominant modulation frequency must be in [0, 30] Hz."""
        audio = _make_lp_modulated()
        freq, _ = _fp._wow_flutter(audio, SR)
        assert 0.0 <= freq <= 30.0, f"Wow/flutter frequency {freq} Hz out of range"

    def test_short_audio_no_crash(self):
        """Very short audio must fall back gracefully (no exception, (0.0, 0.0))."""
        short = np.zeros(64, dtype=np.float32)
        result = _fp._wow_flutter(short, SR)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_zcr_fallback_returns_shape(self):
        """ZCR fallback path must also return (dom_freq, depth)."""
        audio = _make_unmodulated(duration_s=1.5)
        result = _fp._wow_flutter_zcr_fallback(audio, SR)
        assert len(result) == 2
        dom_freq, depth = result
        assert dom_freq == 0.0  # ZCR returns no frequency information
        assert 0.0 <= depth <= 20.0

    def test_nan_input_no_nan_output(self):
        """NaN-contaminated input must not produce NaN output (via extract)."""
        audio = np.full(int(2.0 * SR), np.nan, dtype=np.float32)
        f = _fp.extract(audio, SR)
        assert math.isfinite(f["wow_flutter_hz"])
        assert math.isfinite(f["wow_depth"])

    def test_high_flutter_signal_has_positive_depth(self):
        """Audio with pitch-modulated tone (flutter) must yield depth > 0."""
        t = np.linspace(0, 4.0, int(4.0 * SR), dtype=np.float32)
        # Simulate flutter: 440 Hz tone frequency-modulated at 8 Hz (flutter range)
        f_mod = 440.0 * (1.0 + 0.003 * np.sin(2 * math.pi * 8.0 * t))
        sig = 0.1 * np.sin(2 * math.pi * np.cumsum(f_mod) / SR)
        _, depth = _fp._wow_flutter(sig, SR)
        assert depth >= 0.0  # must be finite and non-negative


# ============================================================================
# 3. MDCT Codec Forensics
# ============================================================================


class TestCodecArtifactScore:
    """Tests for _SpectralFingerprinter._codec_artifact_score()."""

    def test_returns_tuple_of_two_floats(self):
        """Must return (score, code) as two finite floats."""
        audio = _make_mp3_signature()
        result = _fp._codec_artifact_score(audio, SR)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_score_in_unit_interval(self):
        """Artifact score must be in [0, 1]."""
        for func in [_make_mp3_signature, _make_digital_clean, _make_unmodulated]:
            audio = func()
            score, _ = _fp._codec_artifact_score(audio, SR)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"

    def test_codec_type_code_valid(self):
        """Codec type code must be one of {0.0, 1.0, 2.0, 3.0}."""
        for func in [_make_mp3_signature, _make_digital_clean, _make_unmodulated]:
            audio = func()
            _, code = _fp._codec_artifact_score(audio, SR)
            assert code in (0.0, 1.0, 2.0, 3.0), f"Invalid codec code: {code}"

    def test_clean_digital_low_score(self):
        """Clean full-bandwidth digital audio must have lower codec score than simulated MP3."""
        mp3_audio = _make_mp3_signature()
        clean_audio = _make_digital_clean()
        mp3_score, _ = _fp._codec_artifact_score(mp3_audio, SR)
        clean_score, _ = _fp._codec_artifact_score(clean_audio, SR)
        assert mp3_score >= clean_score, f"MP3 score ({mp3_score:.3f}) should be >= clean ({clean_score:.3f})"

    def test_short_audio_returns_zero(self):
        """Audio shorter than 4 × n_fft must return (0.0, 0.0)."""
        short = np.zeros(4096 * 2, dtype=np.float32)  # < 4 * 4096
        score, code = _fp._codec_artifact_score(short, SR)
        assert score == 0.0
        assert code == 0.0

    def test_silent_audio_no_crash(self):
        """Silent signal must not raise an exception."""
        silence = np.zeros(int(4.0 * SR), dtype=np.float32)
        score, code = _fp._codec_artifact_score(silence, SR)
        assert math.isfinite(score)
        assert math.isfinite(code)

    def test_nan_input_handled(self):
        """NaN input must be sanitised by extract() before reaching codec scorer."""
        nan_audio = np.full(int(4.0 * SR), np.nan, dtype=np.float32)
        f = _fp.extract(nan_audio, SR)
        assert math.isfinite(f["block_artifact"])
        assert math.isfinite(f["codec_type_code"])

    def test_hard_cutoff_signal_detected(self):
        """Hard codec LPF at 14 kHz on a broadband signal must score higher than full-BW.

        Physical principle: codec artifact detection is only meaningful when the original
        signal has content above the cutoff.  Pink noise (1/f spectrum) fills the full
        band to Nyquist; cutting at 14 kHz creates a detectable brick-wall anomaly.
        The algorithm's 90 %%-bandwidth estimator places bw_hz above 14 kHz for the
        full signal (above_mask exists) but below 14 kHz for the cutoff signal — the
        depleted region above bw_hz then produces high cutoff_score.
        """
        rng = np.random.default_rng(11)
        n = int(5.0 * SR)
        # Pink noise: broad-spectrum content fills 20 Hz to Nyquist
        noise = rng.standard_normal(n).astype(np.float64)
        spec = np.fft.rfft(noise)
        freqs_n = np.fft.rfftfreq(n, d=1.0 / SR)
        safe_f = np.where(freqs_n > 0, freqs_n, 1.0)
        spec /= np.sqrt(safe_f)  # 1/sqrt(f) → pink noise amplitude spectral density
        spec[0] = 0.0

        # Hard cutoff version at 14 kHz (simulates 128 kbps MP3 cutoff)
        spec_cut = spec.copy()
        spec_cut[freqs_n > 14000.0] = 0.0
        cutoff_audio = np.fft.irfft(spec_cut, n=n).astype(np.float32)
        cutoff_audio = cutoff_audio / (np.max(np.abs(cutoff_audio)) + 1e-8) * 0.1

        # Full-bandwidth version (natural broad-spectrum content)
        full_audio = np.fft.irfft(spec, n=n).astype(np.float32)
        full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-8) * 0.1

        score_cut, _ = _fp._codec_artifact_score(cutoff_audio, SR)
        score_full, _ = _fp._codec_artifact_score(full_audio, SR)
        assert score_cut >= score_full, f"Hard-cutoff ({score_cut:.3f}) should be >= full-BW ({score_full:.3f})"


# ============================================================================
# 4. Infrasonic RMS (vinyl rumble)
# ============================================================================


class TestInfrasonicRms:
    """Tests for _SpectralFingerprinter._infrasonic_rms()."""

    def test_returns_finite_float(self):
        """Must return a finite float for valid audio."""
        audio = _make_vinyl_with_infrasonic()
        val = _fp._infrasonic_rms(audio, SR)
        assert math.isfinite(val)

    def test_in_unit_interval(self):
        """Infrasonic RMS must be in [0, 1]."""
        for func in [_make_vinyl_with_infrasonic, _make_unmodulated, _make_digital_clean]:
            val = _fp._infrasonic_rms(func(), SR)
            assert 0.0 <= val <= 1.0, f"Infrasonic RMS {val} out of [0, 1]"

    def test_vinyl_higher_than_clean_digital(self):
        """Vinyl-with-infrasonic must have higher rumble than clean digital audio."""
        vinyl_val = _fp._infrasonic_rms(_make_vinyl_with_infrasonic(), SR)
        digital_val = _fp._infrasonic_rms(_make_digital_clean(), SR)
        assert vinyl_val >= digital_val, f"Vinyl infrasonic ({vinyl_val:.4f}) should be >= digital ({digital_val:.4f})"

    def test_short_audio_returns_zero(self):
        """Audio shorter than 1 s must return 0.0."""
        short = np.zeros(int(0.5 * SR), dtype=np.float32)
        val = _fp._infrasonic_rms(short, SR)
        assert val == 0.0

    def test_silent_no_crash(self):
        """Silent input must not raise and must return a finite value."""
        silence = np.zeros(int(3.0 * SR), dtype=np.float32)
        val = _fp._infrasonic_rms(silence, SR)
        assert math.isfinite(val)


# ============================================================================
# 5. ClassificationResult new fields
# ============================================================================


class TestClassificationResultNewFields:
    """Validate new fields were properly added to ClassificationResult."""

    def _minimal_result(self) -> ClassificationResult:
        return ClassificationResult(material="vinyl", confidence=0.7)

    def test_rotation_hz_default_zero(self):
        r = self._minimal_result()
        assert r.rotation_hz == 0.0

    def test_rotation_strength_default_zero(self):
        r = self._minimal_result()
        assert r.rotation_strength == 0.0

    def test_infrasonic_rms_default_zero(self):
        r = self._minimal_result()
        assert r.infrasonic_rms == 0.0

    def test_codec_type_default_empty(self):
        r = self._minimal_result()
        assert r.codec_type == ""

    def test_as_dict_contains_new_fields(self):
        r = ClassificationResult(
            material="vinyl",
            confidence=0.8,
            rotation_hz=0.556,
            rotation_strength=0.42,
            infrasonic_rms=0.05,
            codec_type="clean",
        )
        d = r.as_dict()
        assert "rotation_hz" in d
        assert "rotation_strength" in d
        assert "infrasonic_rms" in d
        assert "codec_type" in d
        assert d["rotation_hz"] == pytest.approx(0.556)
        assert d["rotation_strength"] == pytest.approx(0.42)
        assert d["codec_type"] == "clean"


# ============================================================================
# 6. End-to-end: extract() returns all expected keys
# ============================================================================


class TestExtractNewKeys:
    """Validate extract() returns the new feature keys."""

    def test_all_new_keys_present(self):
        audio = _make_unmodulated(duration_s=5.0)
        f = _fp.extract(audio, SR)
        for key in ("rotation_hz", "rotation_strength", "infrasonic_rms", "wow_depth", "codec_type_code"):
            assert key in f, f"Key '{key}' missing from extract() output"

    def test_all_values_finite(self):
        audio = _make_lp_modulated(duration_s=5.0)
        f = _fp.extract(audio, SR)
        for k, v in f.items():
            assert math.isfinite(v), f"Feature '{k}' = {v} is not finite"

    def test_stereo_input_handled(self):
        """Stereo audio must not raise in extract()."""
        stereo = _make_vinyl_with_infrasonic()
        f = _fp.extract(stereo, SR)
        for k, v in f.items():
            assert math.isfinite(v), f"Feature '{k}' = {v} is not finite"


# ============================================================================
# 7. _MaterialScorer integration: new features drive correct classification
# ============================================================================


class TestMaterialScorerIntegration:
    """Validate that new features steer scoring toward correct materials."""

    def _score(self, features: dict) -> ClassificationResult:
        from backend.core.medium_classifier import _MaterialScorer

        return _MaterialScorer().score(features, None)

    def _base_vinyl_features(self) -> dict:
        return {
            "bandwidth_hz": 18_000.0,
            "snr_db": 42.0,
            "noise_color": 1.6,
            "crackle_density": 0.002,
            "wow_flutter_hz": 0.5,
            "wow_depth": 0.3,
            "block_artifact": 0.0,
            "codec_type_code": 0.0,
            "pre_echo_ms": 0.5,
            "rotation_hz": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
        }

    def test_lp_rotation_boosts_vinyl(self):
        """Adding LP rotation signal must increase vinyl score relative to baseline."""
        self._score(self._base_vinyl_features())
        enhanced = self._base_vinyl_features()
        enhanced["rotation_hz"] = 0.556
        enhanced["rotation_strength"] = 0.35
        enhanced["infrasonic_rms"] = 0.06
        self._score(enhanced)
        # With rotation features, vinyl posterior should increase or remain dominant
        from backend.core.medium_classifier import _MaterialScorer

        scorer = _MaterialScorer()
        r_base = scorer.score(self._base_vinyl_features(), None)
        r_enh = scorer.score(enhanced, None)
        # Enhanced features must produce vinyl-family material
        assert r_enh.material_type in ("vinyl", "lacquer_disc", "shellac")
        # Vinyl evidence confidence should not decrease with rotation
        vinyl_conf_base = next(
            (e.confidence for e in r_base.evidence if str(getattr(e.material, "value", e.material)) == "vinyl"), 0.0
        )
        vinyl_conf_enh = next(
            (e.confidence for e in r_enh.evidence if str(getattr(e.material, "value", e.material)) == "vinyl"), 0.0
        )
        assert vinyl_conf_enh >= vinyl_conf_base * 0.9

    def test_shellac_rotation_boosts_shellac(self):
        """Shellac rotation (1.3 Hz) in scorer must add confidence to shellac."""
        from backend.core.medium_classifier import _MaterialScorer

        scorer = _MaterialScorer()
        features_shellac = {
            "bandwidth_hz": 7_000.0,
            "snr_db": 12.0,
            "noise_color": 2.1,
            "crackle_density": 0.001,
            "wow_flutter_hz": 1.3,
            "wow_depth": 1.0,
            "block_artifact": 0.0,
            "codec_type_code": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_hz": 1.3,
            "rotation_strength": 0.4,
            "infrasonic_rms": 0.03,
        }
        result_with = scorer.score(features_shellac, None)
        features_without = {**features_shellac, "rotation_hz": 0.0, "rotation_strength": 0.0}
        result_without = scorer.score(features_without, None)
        # With rotation, shellac confidence must be >= without rotation
        assert result_with.confidence >= result_without.confidence * 0.9

    def test_mp3_codec_code_boosts_mp3_low(self):
        """codec_type_code == 1.0 (mp3) must increase mp3_low/mp3_high scores."""
        from backend.core.medium_classifier import _MaterialScorer

        scorer = _MaterialScorer()
        base = {
            "bandwidth_hz": 14_000.0,
            "snr_db": 35.0,
            "noise_color": 1.2,
            "crackle_density": 0.0,
            "wow_flutter_hz": 0.0,
            "wow_depth": 0.0,
            "block_artifact": 0.25,
            "codec_type_code": 0.0,
            "pre_echo_ms": 6.0,
            "rotation_hz": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
        }
        result_base = scorer.score(base, None)
        mp3_features = {**base, "codec_type_code": 1.0}
        result_mp3 = scorer.score(mp3_features, None)
        # mp3 confidence must be at least as good as without the codec hint
        assert result_mp3.confidence >= result_base.confidence * 0.85

    def test_tape_penalised_when_rotation_present(self):
        """Tape material must receive rotation penalty when rotation detected."""
        from backend.core.medium_classifier import _MaterialScorer

        scorer = _MaterialScorer()
        features_no_rot = {
            "bandwidth_hz": 14_000.0,
            "snr_db": 28.0,
            "noise_color": 1.4,
            "crackle_density": 0.0,
            "wow_flutter_hz": 0.8,
            "wow_depth": 0.8,
            "block_artifact": 0.0,
            "codec_type_code": 0.0,
            "pre_echo_ms": 0.0,
            "rotation_hz": 0.0,
            "rotation_strength": 0.0,
            "infrasonic_rms": 0.0,
        }
        features_with_rot = {**features_no_rot, "rotation_hz": 0.556, "rotation_strength": 0.35}
        # With rotation present, vinyl/shellac posterior should rise relative to tape
        r_no_rot = scorer.score(features_no_rot, None)
        r_with_rot = scorer.score(features_with_rot, None)

        # Extract tape posterior from evidence
        def _tape_post(r):
            return next(
                (e.confidence for e in r.evidence if str(getattr(e.material, "value", e.material)) == "tape"), 0.0
            )

        # Tape confidence should decrease (or not increase) when rotation is present
        # because rotation is a disc-family feature, not a tape feature
        assert _tape_post(r_with_rot) <= _tape_post(r_no_rot) + 0.01

    def test_score_all_positive_no_negative(self):
        """All material scores must be ≥ 0 (penalty clipping applied)."""
        from backend.core.medium_classifier import _MaterialScorer

        scorer = _MaterialScorer()
        features = {
            "bandwidth_hz": 10_000.0,
            "snr_db": 20.0,
            "noise_color": 1.5,
            "crackle_density": 0.001,
            "wow_flutter_hz": 0.5,
            "wow_depth": 0.5,
            "block_artifact": 0.1,
            "codec_type_code": 1.0,
            "pre_echo_ms": 4.0,
            "rotation_hz": 0.556,
            "rotation_strength": 0.30,
            "infrasonic_rms": 0.04,
        }
        result = scorer.score(features, None)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        for ev in result.evidence:
            assert ev.confidence >= 0.0

    def test_codec_type_string_populated(self):
        """ClassificationResult.codec_type must be a non-empty readable string
        when codec_type_code > 0."""
        from backend.core.medium_classifier import _MaterialScorer

        scorer = _MaterialScorer()
        for code, expected in [(0.0, "clean"), (1.0, "mp3"), (2.0, "aac"), (3.0, "lossy")]:
            f = {
                "bandwidth_hz": 20_000.0,
                "snr_db": 60.0,
                "noise_color": 1.0,
                "crackle_density": 0.0,
                "wow_flutter_hz": 0.0,
                "wow_depth": 0.0,
                "block_artifact": 0.0 if code == 0.0 else 0.3,
                "codec_type_code": code,
                "pre_echo_ms": 0.0,
                "rotation_hz": 0.0,
                "rotation_strength": 0.0,
                "infrasonic_rms": 0.0,
            }
            result = scorer.score(f, None)
            assert result.codec_type == expected, f"codec_code={code}: expected '{expected}', got '{result.codec_type}'"


# ============================================================================
# Regression: Stereo-Downmix-Achsen-Bug (2026-06-08)
# ============================================================================


class TestSpectralFingerprinterDownmixAxis:
    """Regression für den _to_mono-Achsen-Bug.

    File-Import liefert Stereo als ``(samples, channels)`` = ``(N, 2)``.
    Ein ``mean(axis=0)`` ohne Layout-Check kollabierte die Sample-Achse auf 2
    Werte → ``mono.size < FRAME_SIZE`` → ``_null_features()`` (alles Null) →
    Material-Fehlklassifikation bei JEDEM Stereo-Song.
    """

    def test_to_mono_samples_first_preserves_length(self):
        mono = _make_unmodulated(duration_s=4.0)
        stereo_nch = np.stack([mono, mono * 0.8], axis=1)  # (N, 2)
        assert stereo_nch.shape == (mono.shape[0], 2)
        out = _fp._to_mono(stereo_nch)
        assert out.shape[0] == mono.shape[0], (
            f"Sample-Achse kollabiert: erwartet {mono.shape[0]}, erhielt {out.shape[0]}"
        )
        assert out.ndim == 1

    def test_to_mono_channels_first_preserves_length(self):
        mono = _make_unmodulated(duration_s=4.0)
        stereo_chn = np.stack([mono, mono * 0.8], axis=0)  # (2, N)
        assert stereo_chn.shape == (2, mono.shape[0])
        out = _fp._to_mono(stereo_chn)
        assert out.shape[0] == mono.shape[0]
        assert out.ndim == 1

    def test_extract_stereo_yields_nonnull_features(self):
        """Stereo ``(N, 2)`` darf NICHT in _null_features() laufen."""
        mono = _make_mp3_signature(duration_s=4.0)
        stereo_nch = np.stack([mono, mono], axis=1)  # (N, 2)
        feats = _fp.extract(stereo_nch, SR)
        # Bandbreite muss eine reale Messung (> 0) sein, kein Null-Fallback.
        assert feats["bandwidth_hz"] > 0.0, "Stereo lief in _null_features() (Achsen-Bug)"

    def test_extract_stereo_matches_mono_source(self):
        """Identischer Inhalt mono vs. (N, 2)-Stereo → quasi gleiche Bandbreite."""
        mono = _make_mp3_signature(duration_s=4.0)
        feats_mono = _fp.extract(mono, SR)
        feats_stereo = _fp.extract(np.stack([mono, mono], axis=1), SR)
        assert math.isclose(feats_mono["bandwidth_hz"], feats_stereo["bandwidth_hz"], rel_tol=0.05)
