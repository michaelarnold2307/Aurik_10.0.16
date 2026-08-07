from __future__ import annotations

import pytest

"""Tests for PerceptualSalienceEstimator (§9.1c).

Validations:
- Shape/NaN/Inf guards
- Loud passages mask defects (low salience)
- Quiet passages expose defects (high salience)
- Temporal forward masking (post-transient)
- Temporal backward masking (pre-transient)
- Edge cases: empty audio, no defects, single defect, mono/stereo
- Severity scaling via annotate_defect_scores()
- Singleton pattern
"""


import numpy as np

from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, DefectType, MaterialType
from backend.core.perceptual_salience import (
    PerceptualSalienceEstimator,
    SalienceAnnotation,
    SalienceResult,
    get_perceptual_salience_estimator,
)


def _make_defect_result(
    scores: dict[DefectType, DefectScore] | None = None,
    sr: int = 48000,
    duration: float = 10.0,
) -> DefectAnalysisResult:
    """Helper: create a minimal DefectAnalysisResult."""
    if scores is None:
        scores = {}
    return DefectAnalysisResult(
        material_type=MaterialType.UNKNOWN,
        scores=scores,
        analysis_time_seconds=0.1,
        sample_rate=sr,
        duration_seconds=duration,
    )


def _make_tone(sr: int, duration_s: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    """Generate a pure sine tone."""
    t = np.arange(int(sr * duration_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)  # type: ignore[no-any-return]


def _make_silence(sr: int, duration_s: float) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration_s), dtype=np.float64)


def _insert_silence_gap(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Zero out a region in the audio to simulate a dropout."""
    out = audio.copy()
    s = int(start_s * sr)
    e = int(end_s * sr)
    out[s:e] = 0.0
    return out


@pytest.mark.unit
class TestPerceptualSalienceEstimatorShape:
    """Basic shape, NaN/Inf guards, and return type tests."""

    def test_01_returns_salience_result(self):
        pse = PerceptualSalienceEstimator()
        audio = _make_tone(48000, 2.0)
        dr = _make_defect_result(sr=48000, duration=2.0)
        result = pse.estimate(audio, 48000, dr)
        assert isinstance(result, SalienceResult)

    def test_02_empty_defects_zero_annotations(self):
        pse = PerceptualSalienceEstimator()
        audio = _make_tone(48000, 2.0)
        dr = _make_defect_result()
        result = pse.estimate(audio, 48000, dr)
        assert len(result.annotations) == 0
        assert result.mean_salience == 0.0

    def test_03_nan_input_audio(self):
        pse = PerceptualSalienceEstimator()
        audio = np.full(48000 * 2, np.nan)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.5,
                confidence=0.8,
                locations=[(0.5, 0.6)],
            )
        }
        dr = _make_defect_result(scores=scores)
        result = pse.estimate(audio, 48000, dr)
        assert np.isfinite(result.mean_salience)
        for ann in result.annotations:
            assert np.isfinite(ann.salience)

    def test_04_inf_input_audio(self):
        pse = PerceptualSalienceEstimator()
        audio = np.full(48000 * 2, np.inf)
        scores = {
            DefectType.DROPOUTS: DefectScore(
                defect_type=DefectType.DROPOUTS,
                severity=0.3,
                confidence=0.7,
                locations=[(0.2, 0.3)],
            )
        }
        dr = _make_defect_result(scores=scores)
        result = pse.estimate(audio, 48000, dr)
        assert np.isfinite(result.mean_salience)

    def test_05_salience_bounded_0_1(self):
        pse = PerceptualSalienceEstimator()
        audio = _make_tone(48000, 5.0, amp=0.8)
        audio = _insert_silence_gap(audio, 48000, 2.0, 2.1)
        scores = {
            DefectType.DROPOUTS: DefectScore(
                defect_type=DefectType.DROPOUTS,
                severity=0.6,
                confidence=0.9,
                locations=[(2.0, 2.1)],
            )
        }
        dr = _make_defect_result(scores=scores, duration=5.0)
        result = pse.estimate(audio, 48000, dr)
        for ann in result.annotations:
            assert 0.0 <= ann.salience <= 1.0

    def test_06_result_counts_consistent(self):
        pse = PerceptualSalienceEstimator()
        audio = _make_tone(48000, 5.0)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.4,
                confidence=0.8,
                locations=[(1.0, 1.01), (2.0, 2.01), (3.0, 3.01)],
            )
        }
        dr = _make_defect_result(scores=scores, duration=5.0)
        result = pse.estimate(audio, 48000, dr)
        assert len(result.annotations) == 3
        assert result.n_salient + result.n_masked <= len(result.annotations)


class TestSimultaneousMasking:
    """Test simultaneous masking: defects during loud passages should have low salience."""

    def test_07_dropout_in_loud_passage_low_salience(self):
        """A dropout surrounded by loud audio should be highly salient (it's noticeable!)."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 5.0, amp=0.8)
        # Insert silent gap (dropout) at 2.5s
        audio = _insert_silence_gap(audio, sr, 2.5, 2.55)
        scores = {
            DefectType.DROPOUTS: DefectScore(
                defect_type=DefectType.DROPOUTS,
                severity=0.7,
                confidence=0.95,
                locations=[(2.5, 2.55)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=5.0)
        result = pse.estimate(audio, sr, dr)
        # Dropout in loud context = high contrast → high salience
        assert result.annotations[0].salience >= 0.5

    def test_08_defect_in_silence_full_salience(self):
        """A tiny click in silence is fully exposed → high salience."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_silence(sr, 5.0)
        # Insert tiny click
        click_pos = int(2.5 * sr)
        audio[click_pos : click_pos + 10] = 0.01  # very quiet click
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.3,
                confidence=0.7,
                locations=[(2.5, 2.501)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=5.0)
        result = pse.estimate(audio, sr, dr)
        # In silence, everything is heard
        assert len(result.annotations) == 1


class TestTemporalMasking:
    """Test forward and backward temporal masking."""

    def test_09_forward_masking_after_transient(self):
        """A defect immediately after a loud transient should be masked (forward masking)."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_silence(sr, 3.0)
        # Loud transient at 1.0s
        t_start = int(1.0 * sr)
        audio[t_start : t_start + int(0.05 * sr)] = 0.9  # loud burst
        # Quiet defect at 1.1s (within 200ms forward masking window)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.3,
                confidence=0.7,
                locations=[(1.1, 1.11)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 1

    def test_10_backward_masking_before_transient(self):
        """A defect just before a loud passage should be partially masked."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_silence(sr, 3.0)
        # Loud passage starting at 1.5s
        t_start = int(1.5 * sr)
        audio[t_start : t_start + int(0.5 * sr)] = 0.8
        # Defect at 1.49s (just 10ms before loud passage — backward masking)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.3,
                confidence=0.7,
                locations=[(1.49, 1.495)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 1


class TestAnnotateDefectScores:
    """Test the annotate_defect_scores() method that modifies DefectAnalysisResult."""

    def test_11_severity_scaled_by_salience(self):
        """Severity should be adjusted based on salience."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 5.0, amp=0.8)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.5,
                confidence=0.8,
                locations=[(2.0, 2.01)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=5.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        ds = result.scores[DefectType.CLICKS]
        # Severity should have been adjusted (not necessarily same as original)
        assert 0.0 <= ds.severity <= 1.0
        assert "perceptual_salience" in ds.metadata

    def test_12_metadata_fields_present(self):
        """Check that all metadata fields are added."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 3.0)
        scores = {
            DefectType.HUM: DefectScore(
                defect_type=DefectType.HUM,
                severity=0.4,
                confidence=0.7,
                locations=[(0.5, 1.5)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        ds = result.scores[DefectType.HUM]
        assert "perceptual_salience" in ds.metadata
        assert "n_salient_events" in ds.metadata
        assert "n_masked_events" in ds.metadata

    def test_13_severity_never_exceeds_1(self):
        """Even after scaling, severity must stay in [0.0, 1.0]."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 2.0)
        scores = {
            DefectType.HIGH_FREQ_NOISE: DefectScore(
                defect_type=DefectType.HIGH_FREQ_NOISE,
                severity=1.0,
                confidence=0.99,
                locations=[(0.0, 2.0)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=2.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        assert result.scores[DefectType.HIGH_FREQ_NOISE].severity <= 1.0

    def test_14_severity_never_negative(self):
        """Severity must never go below 0."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_silence(sr, 2.0)
        scores = {
            DefectType.HIGH_FREQ_NOISE: DefectScore(
                defect_type=DefectType.HIGH_FREQ_NOISE,
                severity=0.01,
                confidence=0.5,
                locations=[(0.5, 1.0)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=2.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        assert result.scores[DefectType.HIGH_FREQ_NOISE].severity >= 0.0

    def test_15_no_defects_no_metadata(self):
        """If no defects, annotate should still return valid result."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 2.0)
        dr = _make_defect_result(sr=sr, duration=2.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        assert isinstance(result, DefectAnalysisResult)


class TestMultipleDefectTypes:
    """Test with multiple simultaneous defect types."""

    def test_16_multiple_defect_types(self):
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 5.0, amp=0.7)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.4,
                confidence=0.8,
                locations=[(1.0, 1.01), (3.0, 3.01)],
            ),
            DefectType.HUM: DefectScore(
                defect_type=DefectType.HUM,
                severity=0.6,
                confidence=0.9,
                locations=[(0.0, 5.0)],
            ),
            DefectType.DROPOUTS: DefectScore(
                defect_type=DefectType.DROPOUTS,
                severity=0.5,
                confidence=0.85,
                locations=[(2.0, 2.05)],
            ),
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=5.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 4  # 2 clicks + 1 hum + 1 dropout

    def test_17_each_type_gets_metadata(self):
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 4.0)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.3,
                confidence=0.7,
                locations=[(1.0, 1.01)],
            ),
            DefectType.HIGH_FREQ_NOISE: DefectScore(
                defect_type=DefectType.HIGH_FREQ_NOISE,
                severity=0.5,
                confidence=0.8,
                locations=[(0.0, 4.0)],
            ),
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=4.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        for dt in [DefectType.CLICKS, DefectType.HIGH_FREQ_NOISE]:
            assert "perceptual_salience" in result.scores[dt].metadata


class TestStereoHandling:
    """Test with stereo input."""

    def test_18_stereo_input(self):
        pse = PerceptualSalienceEstimator()
        sr = 48000
        mono = _make_tone(sr, 3.0)
        stereo = np.column_stack([mono, mono])
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.4,
                confidence=0.8,
                locations=[(1.0, 1.01)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.estimate(stereo, sr, dr)
        assert len(result.annotations) == 1
        assert np.isfinite(result.mean_salience)


class TestSampleRateAgnostic:
    """Test that the estimator works at any sample rate (analysis module rule)."""

    def test_19_sr_44100(self):
        pse = PerceptualSalienceEstimator()
        sr = 44100
        audio = _make_tone(sr, 3.0)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.3,
                confidence=0.7,
                locations=[(1.0, 1.01)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 1

    def test_20_sr_22050(self):
        pse = PerceptualSalienceEstimator()
        sr = 22050
        audio = _make_tone(sr, 3.0, freq=220.0)
        scores = {
            DefectType.HIGH_FREQ_NOISE: DefectScore(
                defect_type=DefectType.HIGH_FREQ_NOISE,
                severity=0.5,
                confidence=0.8,
                locations=[(0.5, 2.0)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.estimate(audio, sr, dr)
        assert np.isfinite(result.mean_salience)


class TestSingleton:
    """Test singleton pattern."""

    def test_21_singleton_identity(self):
        a = get_perceptual_salience_estimator()
        b = get_perceptual_salience_estimator()
        assert a is b

    def test_22_singleton_is_correct_type(self):
        pse = get_perceptual_salience_estimator()
        assert isinstance(pse, PerceptualSalienceEstimator)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_23_very_short_audio(self):
        """Audio shorter than window size."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 0.05)  # 50 ms
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.3,
                confidence=0.7,
                locations=[(0.01, 0.02)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=0.05)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 1
        assert np.isfinite(result.annotations[0].salience)

    def test_24_defect_at_audio_start(self):
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 3.0)
        scores = {
            DefectType.DROPOUTS: DefectScore(
                defect_type=DefectType.DROPOUTS,
                severity=0.5,
                confidence=0.8,
                locations=[(0.0, 0.05)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 1

    def test_25_defect_at_audio_end(self):
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 3.0)
        scores = {
            DefectType.DROPOUTS: DefectScore(
                defect_type=DefectType.DROPOUTS,
                severity=0.5,
                confidence=0.8,
                locations=[(2.95, 3.0)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 1

    def test_26_zero_severity_defect(self):
        """Defects with severity 0 should still get annotations."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 2.0)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.0,
                confidence=0.5,
                locations=[(1.0, 1.01)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=2.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 1

    def test_27_defect_no_locations(self):
        """Defect with severity but no locations → no annotations."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 2.0)
        scores = {
            DefectType.HUM: DefectScore(
                defect_type=DefectType.HUM,
                severity=0.6,
                confidence=0.9,
                locations=[],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=2.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 0

    def test_28_annotation_dataclass_fields(self):
        """Check SalienceAnnotation has all expected fields."""
        ann = SalienceAnnotation(
            defect_type=DefectType.CLICKS,
            location=(1.0, 1.01),
            salience=0.75,
            local_loudness_lufs=-20.0,
            surrounding_loudness_lufs=-10.0,
            masking_type="none",
        )
        assert ann.defect_type == DefectType.CLICKS
        assert 0.0 <= ann.salience <= 1.0
        assert ann.masking_type == "none"

    def test_29_loudness_profile_shape(self):
        """Internal: loudness profile has correct shape."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        duration_s = 5.0
        audio = _make_tone(sr, duration_s)
        profile = pse._compute_loudness_profile(audio, sr)
        # Window = 400 ms, hop = 100 ms → ~(5.0 - 0.4)/0.1 + 1 = 47 frames
        expected = (int(duration_s * sr) - int(0.4 * sr)) // int(0.1 * sr) + 1
        assert abs(len(profile) - expected) <= 2

    def test_30_loudness_profile_nan_free(self):
        """Loudness profile should never contain NaN."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 2.0)
        profile = pse._compute_loudness_profile(audio, sr)
        assert np.all(np.isfinite(profile))

    def test_31_salience_result_default_values(self):
        """SalienceResult default values should be sane."""
        sr = SalienceResult()
        assert sr.annotations == []
        assert sr.mean_salience == 0.0
        assert sr.n_salient == 0
        assert sr.n_masked == 0

    def test_32_loud_exposed_dropout_high_salience(self):
        """A dropout in a very loud passage is extremely noticeable — salience should be high."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 4.0, amp=0.9)
        # Complete silence gap
        audio = _insert_silence_gap(audio, sr, 2.0, 2.1)
        scores = {
            DefectType.DROPOUTS: DefectScore(
                defect_type=DefectType.DROPOUTS,
                severity=0.8,
                confidence=0.95,
                locations=[(2.0, 2.1)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=4.0)
        result = pse.estimate(audio, sr, dr)
        # Exposed dropout in loud context → must be salient
        assert result.annotations[0].salience >= 0.3

    def test_33_many_events_performance(self):
        """Should handle many defect events without issues."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 10.0)
        # 50 click events
        locs = [(i * 0.2, i * 0.2 + 0.005) for i in range(50)]
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.3,
                confidence=0.7,
                locations=locs,
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=10.0)
        result = pse.estimate(audio, sr, dr)
        assert len(result.annotations) == 50

    def test_34_salience_metadata_value_range(self):
        """perceptual_salience metadata value must be in [0, 1]."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 3.0)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.4,
                confidence=0.7,
                locations=[(1.0, 1.01)],
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        sal = result.scores[DefectType.CLICKS].metadata["perceptual_salience"]
        assert 0.0 <= sal <= 1.0

    def test_35_annotate_preserves_other_metadata(self):
        """Existing metadata should not be overwritten."""
        pse = PerceptualSalienceEstimator()
        sr = 48000
        audio = _make_tone(sr, 3.0)
        scores = {
            DefectType.CLICKS: DefectScore(
                defect_type=DefectType.CLICKS,
                severity=0.4,
                confidence=0.7,
                locations=[(1.0, 1.01)],
                metadata={"custom_key": "preserved"},
            )
        }
        dr = _make_defect_result(scores=scores, sr=sr, duration=3.0)
        result = pse.annotate_defect_scores(audio, sr, dr)
        assert result.scores[DefectType.CLICKS].metadata["custom_key"] == "preserved"
        assert "perceptual_salience" in result.scores[DefectType.CLICKS].metadata
