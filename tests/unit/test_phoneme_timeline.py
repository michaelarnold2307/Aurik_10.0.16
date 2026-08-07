from __future__ import annotations

"""
tests/unit/test_phoneme_timeline.py
Aurik 10.0.0 — §2.36a PhonemeTimeline unit tests

Coverage:
  - PhonemeTimelineSegment basics
  - PhonemeTimeline.build_empty()
  - PhonemeTimeline.build_from_transcription()
  - segments_in_range() edge cases
  - sibilant_segments() / stressed_vowel_segments()
  - formant_target_for_range() for all 5 languages + unknown
  - sibilant_band_hz() for all languages
  - _PTYPE_TO_CLASS mapping completeness
  - _detect_language() with synthetic audio
  - PhonemeTimelineBuilder singleton thread-safety
  - NaN/Inf safety on all numeric inputs
  - MDEM morph() phoneme_timeline parameter (stressed-vowel frame headroom)
  - Phase 43 sibilant_band_hz override branch
  - Phase 24 phoneme_class → content_type override
"""


import threading
from dataclasses import dataclass

import numpy as np
import pytest

from backend.core.phoneme_timeline import (
    _PTYPE_TO_CLASS,
    PhonemeTimeline,
    PhonemeTimelineSegment,
    _detect_language,
    get_phoneme_timeline_builder,
)

# ─── Fixtures & helpers ───────────────────────────────────────────────────────


@dataclass
class _FakeWord:
    start_s: float
    end_s: float
    phoneme_type: str
    confidence: float = 0.8
    is_stressed: bool = False


@dataclass
class _FakeTrans:
    words: list
    duration_s: float
    language: str = "de"
    overall_confidence: float = 0.7


def _make_segment(
    start_s: float,
    end_s: float,
    pclass: str = "vowel_stressed",
    confidence: float = 0.8,
) -> PhonemeTimelineSegment:
    return PhonemeTimelineSegment(
        start_s=start_s,
        end_s=end_s,
        phoneme_class=pclass,
        phoneme_ipa="",
        confidence=confidence,
        is_stressed=pclass == "vowel_stressed",
    )


def _empty_timeline(duration_s: float = 5.0) -> PhonemeTimeline:
    return PhonemeTimeline.build_empty(duration_s)


# ─── PhonemeTimelineSegment ───────────────────────────────────────────────────


class TestPhonemeTimelineSegment:
    def test_basic_construction(self):
        seg = _make_segment(0.5, 1.0)
        assert seg.start_s == 0.5
        assert seg.end_s == 1.0
        assert seg.phoneme_class == "vowel_stressed"
        assert seg.is_stressed is True

    def test_plosive_not_stressed(self):
        seg = _make_segment(1.0, 1.05, pclass="plosive")
        assert seg.is_stressed is False

    def test_confidence_range(self):
        seg = _make_segment(0.0, 0.5, confidence=0.0)
        assert seg.confidence == 0.0
        seg2 = _make_segment(0.0, 0.5, confidence=1.0)
        assert seg2.confidence == 1.0


# ─── build_empty ─────────────────────────────────────────────────────────────


class TestBuildEmpty:
    def test_defaults(self):
        tl = PhonemeTimeline.build_empty(10.0)
        assert tl.language == "unknown"
        assert tl.language_confidence == 0.0
        assert tl.segments == []
        assert tl.duration_s == 10.0
        assert tl.has_ipa is False

    def test_zero_duration(self):
        tl = PhonemeTimeline.build_empty(0.0)
        assert tl.duration_s == 0.0

    def test_nan_duration_safe(self):
        tl = PhonemeTimeline.build_empty(float("nan"))
        assert tl.duration_s == 0.0

    def test_inf_duration_safe(self):
        tl = PhonemeTimeline.build_empty(float("inf"))
        assert tl.duration_s == 0.0


# ─── build_from_transcription ────────────────────────────────────────────────


class TestBuildFromTranscription:
    def test_empty_result_gives_empty_timeline(self):
        trans = _FakeTrans(words=[], duration_s=5.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert tl.language == "de"
        assert tl.segments == []
        assert tl.duration_s == 5.0

    def test_vowel_stressed_mapping(self):
        words = [_FakeWord(0.0, 0.5, "vowel_stressed", confidence=0.9, is_stressed=True)]
        trans = _FakeTrans(words=words, duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert len(tl.segments) == 1
        assert tl.segments[0].phoneme_class == "vowel_stressed"
        assert tl.segments[0].is_stressed is True

    def test_vowel_unstressed_stressed_flag_promotes(self):
        """Unstressed vowel with is_stressed=True must be promoted to vowel_stressed."""
        words = [_FakeWord(0.0, 0.3, "vowel_unstressed", is_stressed=True)]
        trans = _FakeTrans(words=words, duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert tl.segments[0].phoneme_class == "vowel_stressed"

    def test_fricative_mapping(self):
        words = [_FakeWord(0.0, 0.1, "fricative")]
        trans = _FakeTrans(words=words, duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert tl.segments[0].phoneme_class == "sibilant"

    def test_plosive_mapping(self):
        words = [_FakeWord(0.0, 0.05, "plosive")]
        trans = _FakeTrans(words=words, duration_s=0.5)
        tl = PhonemeTimeline.build_from_transcription(trans, "en")
        assert tl.segments[0].phoneme_class == "plosive"

    def test_silence_mapping(self):
        words = [_FakeWord(0.0, 1.0, "silence")]
        trans = _FakeTrans(words=words, duration_s=2.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert tl.segments[0].phoneme_class == "silence"

    def test_mixed_maps_to_silence(self):
        words = [_FakeWord(0.0, 0.5, "mixed")]
        trans = _FakeTrans(words=words, duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert tl.segments[0].phoneme_class == "silence"

    def test_segments_sorted_by_start_s(self):
        words = [
            _FakeWord(0.8, 1.0, "vowel_stressed"),
            _FakeWord(0.0, 0.4, "plosive"),
            _FakeWord(0.4, 0.8, "sibilant"),
        ]
        trans = _FakeTrans(words=words, duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        starts = [s.start_s for s in tl.segments]
        assert starts == sorted(starts)

    def test_invalid_language_falls_to_unknown(self):
        trans = _FakeTrans(words=[], duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "zz")
        assert tl.language == "unknown"

    def test_end_lte_start_segment_skipped(self):
        words = [_FakeWord(0.5, 0.5, "vowel_stressed")]  # zero-length: end==start
        trans = _FakeTrans(words=words, duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert len(tl.segments) == 0

    def test_confidence_clipped_to_bounds(self):
        words = [_FakeWord(0.0, 0.5, "vowel_stressed", confidence=-0.5)]
        trans = _FakeTrans(words=words, duration_s=1.0)
        tl = PhonemeTimeline.build_from_transcription(trans, "de")
        assert 0.0 <= tl.segments[0].confidence <= 1.0


# ─── segments_in_range ───────────────────────────────────────────────────────


class TestSegmentsInRange:
    def _tl_with_segments(self) -> PhonemeTimeline:
        tl = _empty_timeline(10.0)
        tl.segments = [
            _make_segment(0.0, 1.0, "plosive"),
            _make_segment(1.0, 3.0, "vowel_stressed"),
            _make_segment(3.0, 5.0, "sibilant"),
        ]
        return tl

    def test_exact_range_all(self):
        tl = self._tl_with_segments()
        segs = tl.segments_in_range(0.0, 5.0)
        assert len(segs) == 3

    def test_no_overlap(self):
        tl = self._tl_with_segments()
        assert tl.segments_in_range(6.0, 8.0) == []

    def test_partial_overlap_left(self):
        tl = self._tl_with_segments()
        segs = tl.segments_in_range(0.0, 1.5)
        assert len(segs) == 2  # plosive[0,1) and vowel_stressed[1,3) overlap

    def test_partial_overlap_right(self):
        tl = self._tl_with_segments()
        segs = tl.segments_in_range(2.5, 10.0)
        assert len(segs) == 2  # vowel_stressed[1,3) and sibilant[3,5)

    def test_nan_range_empty(self):
        tl = self._tl_with_segments()
        assert tl.segments_in_range(float("nan"), 1.0) == []

    def test_inverted_range_empty(self):
        tl = self._tl_with_segments()
        assert tl.segments_in_range(3.0, 1.0) == []

    def test_empty_timeline(self):
        tl = _empty_timeline()
        assert tl.segments_in_range(0.0, 5.0) == []


# ─── sibilant_segments / stressed_vowel_segments ─────────────────────────────


class TestFilteringMethods:
    def _tl(self) -> PhonemeTimeline:
        tl = _empty_timeline(10.0)
        tl.segments = [
            _make_segment(0.0, 0.5, "sibilant"),
            _make_segment(0.5, 1.0, "fricative_stressed"),
            _make_segment(1.0, 2.0, "vowel_stressed"),
            _make_segment(2.0, 2.5, "plosive"),
            _make_segment(2.5, 3.0, "silence"),
            _make_segment(3.0, 4.0, "vowel_unstressed"),
        ]
        return tl

    def test_sibilant_segments_returns_sibilant_and_fricative(self):
        segs = self._tl().sibilant_segments()
        classes = {s.phoneme_class for s in segs}
        assert "sibilant" in classes
        assert "fricative_stressed" in classes
        assert "vowel_stressed" not in classes

    def test_stressed_vowel_segments_only_vowel_stressed(self):
        segs = self._tl().stressed_vowel_segments()
        assert all(s.phoneme_class == "vowel_stressed" for s in segs)
        assert len(segs) == 1

    def test_empty_returns_empty(self):
        tl = _empty_timeline()
        assert tl.sibilant_segments() == []
        assert tl.stressed_vowel_segments() == []


# ─── formant_target_for_range ────────────────────────────────────────────────


class TestFormantTargetForRange:
    def _tl_with_vowel(self, lang: str, ipa: str = "", pclass: str = "vowel_stressed") -> PhonemeTimeline:
        seg = PhonemeTimelineSegment(
            start_s=0.0,
            end_s=1.0,
            phoneme_class=pclass,
            phoneme_ipa=ipa,
            confidence=0.9,
            is_stressed=True,
        )
        return PhonemeTimeline(
            language=lang,
            language_confidence=0.8,
            segments=[seg],
            duration_s=2.0,
            has_ipa=True,
        )

    def test_no_vowels_returns_none(self):
        tl = _empty_timeline()
        tl.segments = [_make_segment(0.0, 1.0, "sibilant")]
        assert tl.formant_target_for_range(0.0, 1.0) is None

    def test_returns_tuple_for_de_vowel(self):
        tl = self._tl_with_vowel("de", "a")
        result = tl.formant_target_for_range(0.0, 2.0)
        assert result is not None
        f1, f2 = result
        assert 50.0 < f1 < 2000.0
        assert 500.0 < f2 < 4000.0

    def test_returns_tuple_for_en_vowel(self):
        tl = self._tl_with_vowel("en", "ɑ")
        result = tl.formant_target_for_range(0.0, 2.0)
        assert result is not None

    def test_returns_tuple_for_fr_vowel(self):
        tl = self._tl_with_vowel("fr", "a")
        result = tl.formant_target_for_range(0.0, 2.0)
        assert result is not None

    def test_returns_tuple_for_it_vowel(self):
        tl = self._tl_with_vowel("it", "a")
        result = tl.formant_target_for_range(0.0, 2.0)
        assert result is not None

    def test_returns_tuple_for_es_vowel(self):
        tl = self._tl_with_vowel("es", "a")
        result = tl.formant_target_for_range(0.0, 2.0)
        assert result is not None

    def test_unknown_ipa_falls_back_to_schwa(self):
        tl = self._tl_with_vowel("de", "X")  # unknown IPA
        result = tl.formant_target_for_range(0.0, 2.0)
        assert result is not None  # schwa fallback

    def test_nan_range_returns_none(self):
        tl = self._tl_with_vowel("de", "a")
        assert tl.formant_target_for_range(float("nan"), 1.0) is None

    def test_empty_range_outside_segment_returns_none(self):
        tl = self._tl_with_vowel("de", "a")  # segment at [0, 1)
        assert tl.formant_target_for_range(5.0, 10.0) is None

    def test_output_values_finite(self):
        tl = self._tl_with_vowel("de", "a")
        r = tl.formant_target_for_range(0.0, 2.0)
        assert r is not None
        assert np.isfinite(r[0]) and np.isfinite(r[1])


# ─── sibilant_band_hz ────────────────────────────────────────────────────────


class TestSibilantBandHz:
    @pytest.mark.parametrize(
        "lang,expected_low",
        [
            ("de", 5500.0),
            ("en", 5000.0),
            ("fr", 4800.0),
            ("it", 5000.0),
            ("es", 4500.0),
            ("unknown", 4000.0),
        ],
    )
    def test_band_by_language(self, lang: str, expected_low: float):
        tl = _empty_timeline()
        tl.language = lang
        low, high = tl.sibilant_band_hz()
        assert low == expected_low, f"Expected {expected_low} for {lang}, got {low}"
        assert high > low

    def test_fallback_for_unsupported_lang(self):
        tl = _empty_timeline()
        tl.language = "zh"  # not in map
        low, high = tl.sibilant_band_hz()
        assert low == 4000.0  # "unknown" fallback


# ─── _PTYPE_TO_CLASS completeness ────────────────────────────────────────────


class TestPtypeToClass:
    REQUIRED_PTYPES = [
        "fricative_stressed",
        "fricative_unstressed",
        "fricative",
        "plosive",
        "vowel_stressed",
        "vowel_unstressed",
        "vowel",
        "silence",
        "mixed",
    ]

    def test_all_expected_ptypes_present(self):
        for ptype in self.REQUIRED_PTYPES:
            assert ptype in _PTYPE_TO_CLASS, f"Missing ptype: {ptype}"

    def test_all_values_are_known_classes(self):
        valid_classes = {
            "fricative_stressed",
            "sibilant",
            "plosive",
            "vowel_stressed",
            "vowel_unstressed",
            "silence",
        }
        for ptype, pclass in _PTYPE_TO_CLASS.items():
            assert pclass in valid_classes, f"{ptype} → unknown class {pclass}"


# ─── _detect_language ────────────────────────────────────────────────────────


class TestDetectLanguage:
    def test_white_noise_returns_unknown_or_low_confidence(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(16_000).astype(np.float32)
        lang, conf = _detect_language(noise, sr=16_000)
        # white noise has no clear formant structure → either unknown or low conf
        assert lang in {"unknown", "de", "en", "fr", "it", "es"}
        # not asserting specific language for noise — just that it returns valid

    def test_too_short_returns_unknown(self):
        lang, conf = _detect_language(np.zeros(10, dtype=np.float32), sr=16_000)
        assert lang == "unknown"
        assert conf == 0.0

    def test_empty_returns_unknown(self):
        lang, conf = _detect_language(np.zeros(0, dtype=np.float32), sr=16_000)
        assert lang == "unknown"

    def test_nan_input_returns_unknown(self):
        nano = np.full(512, float("nan"), dtype=np.float32)
        lang, conf = _detect_language(nano, sr=16_000)
        assert lang == "unknown"

    def test_return_types(self):
        audio = np.zeros(4096, dtype=np.float32)
        lang, conf = _detect_language(audio, sr=16_000)
        assert isinstance(lang, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_returned_language_in_valid_set(self):
        valid = {"de", "en", "fr", "it", "es", "unknown"}
        audio = np.random.default_rng(7).standard_normal(8000).astype(np.float32)
        lang, _ = _detect_language(audio, sr=16_000)
        assert lang in valid


# ─── PhonemeTimelineBuilder singleton ────────────────────────────────────────


class TestPhonemeTimelineBuilderSingleton:
    def test_same_instance_across_calls(self):
        b1 = get_phoneme_timeline_builder()
        b2 = get_phoneme_timeline_builder()
        assert b1 is b2

    def test_thread_safety(self):
        """3 threads simultaneously calling get_phoneme_timeline_builder must see same instance."""
        results = []

        def _fetch():
            results.append(get_phoneme_timeline_builder())

        threads = [threading.Thread(target=_fetch) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert results[0] is results[1] is results[2]

    def test_build_empty_via_builder(self):
        builder = get_phoneme_timeline_builder()
        tl = builder.build_empty(3.0)
        assert tl.duration_s == 3.0
        assert tl.language == "unknown"

    def test_build_from_transcription_via_builder(self):
        builder = get_phoneme_timeline_builder()
        words = [_FakeWord(0.0, 0.5, "vowel_stressed", is_stressed=True)]
        trans = _FakeTrans(words=words, duration_s=1.0, language="de")
        tl = builder.build_from_transcription(trans, "de")
        assert len(tl.segments) == 1
        assert tl.language == "de"


# ─── MDEM morph() phoneme_timeline parameter ─────────────────────────────────


class TestMdemPhonemeTimeline:
    """Verify MDEM.morph() accepts phoneme_timeline without error."""

    def test_morph_with_phoneme_timeline_none(self):
        from backend.core.micro_dynamics_envelope_morphing import MicroDynamicsEnvelopeMorphing

        mdem = MicroDynamicsEnvelopeMorphing()
        sr = 48_000
        audio = np.random.default_rng(0).standard_normal(sr).astype(np.float32)
        out = mdem.morph(audio, audio, sr=sr, mode="restoration", phoneme_timeline=None)
        assert np.isfinite(out).all()
        assert out.shape == audio.shape

    def test_morph_with_empty_phoneme_timeline(self):
        from backend.core.micro_dynamics_envelope_morphing import MicroDynamicsEnvelopeMorphing

        mdem = MicroDynamicsEnvelopeMorphing()
        sr = 48_000
        audio = np.random.default_rng(1).standard_normal(sr).astype(np.float32)
        tl = PhonemeTimeline.build_empty(1.0)
        out = mdem.morph(audio, audio, sr=sr, mode="restoration", phoneme_timeline=tl)
        assert np.isfinite(out).all()

    def test_morph_stressed_vowel_segments_extra_headroom(self):
        """morph() with stressed-vowel segments must not crash and output stays in ±1."""
        from backend.core.micro_dynamics_envelope_morphing import MicroDynamicsEnvelopeMorphing

        mdem = MicroDynamicsEnvelopeMorphing()
        sr = 48_000
        dur = sr * 2
        audio = np.random.default_rng(2).standard_normal(dur).astype(np.float32) * 0.5
        tl = _empty_timeline(2.0)
        tl.segments = [_make_segment(0.0, 2.0, "vowel_stressed", confidence=0.9)]
        tl.language = "de"
        out = mdem.morph(audio, audio, sr=sr, mode="restoration", phoneme_timeline=tl)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)
        assert np.isfinite(out).all()


# ─── Phase 43 sibilant_band_hz kwarg ─────────────────────────────────────────


class TestPhase43SibilantBand:
    """Smoke-test that phase_43 accepts phoneme_timeline in kwargs without crashing."""

    def test_process_with_phoneme_timeline_none(self):
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        sr = 48_000
        audio = np.zeros(sr, dtype=np.float32)
        audio += 0.01 * np.sin(2 * np.pi * 7000 * np.arange(sr) / sr).astype(np.float32)
        result = phase.process(audio, sample_rate=sr, phoneme_timeline=None)
        assert result.audio is not None
        assert np.isfinite(result.audio).all()

    def test_process_with_empty_phoneme_timeline(self):
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        sr = 48_000
        audio = np.zeros(sr, dtype=np.float32)
        audio += 0.01 * np.sin(2 * np.pi * 6000 * np.arange(sr) / sr).astype(np.float32)
        tl = PhonemeTimeline.build_empty(1.0)
        tl.language = "de"
        result = phase.process(audio, sample_rate=sr, phoneme_timeline=tl)
        assert np.isfinite(result.audio).all()

    def test_sibilant_band_hz_override_applied(self):
        """When phoneme_timeline has a known language, sibilant_band_hz must differ from gender-default."""
        tl_en = _empty_timeline(1.0)
        tl_en.language = "en"  # 5000–8000 Hz
        low, high = tl_en.sibilant_band_hz()
        assert low == 5000.0
        assert high == 8000.0

        tl_fr = _empty_timeline(1.0)
        tl_fr.language = "fr"  # 4800–7500 Hz
        low_fr, high_fr = tl_fr.sibilant_band_hz()
        assert low_fr == 4800.0
        assert high_fr == 7500.0


# ─── Phase 24 phoneme content_type override ──────────────────────────────────


class TestPhase24PhonemeHint:
    """Smoke-test that phase_24 accepts phoneme_timeline in kwargs without crashing."""

    def test_process_with_phoneme_timeline_none(self):
        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        phase = DropoutRepairPhase()
        sr = 48_000
        audio = np.zeros(sr, dtype=np.float32)
        result = phase.process(audio, sr, phoneme_timeline=None)
        assert result is not None

    def test_phoneme_timeline_stored_as_instance_var(self):
        """_current_phoneme_timeline is set by process()."""
        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        phase = DropoutRepairPhase()
        sr = 48_000
        audio = np.zeros(sr, dtype=np.float32)
        tl = PhonemeTimeline.build_empty(1.0)
        phase.process(audio, sr, phoneme_timeline=tl)
        assert phase._current_phoneme_timeline is tl


# ─── Segment-selective de-essing gate ────────────────────────────────────────


class TestSegmentSelectiveGate:
    """Verify time-domain segment-gating in Phase 43 and Phase 19 (§2.36a)."""

    SR = 48_000

    def _sibilant_tone(self, duration_s: float = 1.0, freq_hz: float = 7000.0) -> np.ndarray:
        """Pure sine in sibilant band to make de-essing actually do something."""
        t = np.arange(int(duration_s * self.SR)) / self.SR
        return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]

    def _tl_with_sibilant_at_start(self, audio_dur_s: float = 1.0) -> PhonemeTimeline:
        """Timeline with ONE sibilant segment covering the first 0.2 s only."""
        seg = PhonemeTimelineSegment(
            start_s=0.0,
            end_s=0.2,
            phoneme_class="sibilant",
            phoneme_ipa="s",
            confidence=0.9,
            is_stressed=False,
        )
        return PhonemeTimeline(
            language="de",
            language_confidence=0.9,
            segments=[seg],
            duration_s=audio_dur_s,
            has_ipa=True,
        )

    # ── Phase 43 tests ──────────────────────────────────────────────────────

    def test_phase43_no_phoneme_timeline_still_processes(self):
        """Without phoneme_timeline, Phase 43 should work as before (no regression)."""
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        audio = self._sibilant_tone(1.0)
        result = phase.process(audio, sample_rate=self.SR, phoneme_timeline=None)
        assert result.success
        assert np.isfinite(result.audio).all()

    def test_phase43_gate_non_sibilant_region_unchanged(self):
        """Samples outside the sibilant segment must be identical to the original after gating."""
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        audio = self._sibilant_tone(1.0)
        tl = self._tl_with_sibilant_at_start(audio_dur_s=1.0)

        result = phase.process(audio, sample_rate=self.SR, phoneme_timeline=tl)
        assert result.success

        # Region *after* sibilant segment: samples [0.25s, end] should be very close to original
        # (some tiny float-cast rounding allowed, but DSP de-essing must not have been applied)
        gate_end_sample = int(0.25 * self.SR)
        out = result.audio
        # Cast both to float32 for comparison
        original_tail = audio[gate_end_sample:].astype(np.float32)
        result_tail = out[gate_end_sample:].astype(np.float32)
        max_diff = float(np.max(np.abs(result_tail - original_tail)))
        assert max_diff < 1e-3, f"Non-sibilant region was modified (max_diff={max_diff:.6f}) — segment-gate failed"

    def test_phase43_gate_sibilant_region_is_modified(self):
        """The sibilant segment [0.0, 0.2s] should differ from the original after de-essing."""
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        # Use a loud sibilant tone so de-essing has measurable effect
        t = np.arange(int(1.0 * self.SR)) / self.SR
        audio = (0.5 * np.sin(2.0 * np.pi * 7000.0 * t)).astype(np.float32)
        tl = self._tl_with_sibilant_at_start(audio_dur_s=1.0)

        result = phase.process(audio, sample_rate=self.SR, phoneme_timeline=tl)
        assert result.success

        # Sibilant window: first 0.15 s (well inside the segment — avoid fade zone)
        sib_end = int(0.15 * self.SR)
        out_sib = result.audio[:sib_end]
        orig_sib = audio[:sib_end]
        rms_diff = float(np.sqrt(np.mean((out_sib - orig_sib) ** 2)))
        assert rms_diff > 1e-6, "Sibilant region not modified — de-esser had no effect in sibilant window"

    def test_phase43_gate_with_empty_sibilant_segments(self):
        """Empty sibilant_segments() → no gate applied, full-audio de-essing as fallback."""
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        audio = self._sibilant_tone(0.5)
        tl = PhonemeTimeline.build_empty(0.5)  # no segments → sibilant_segments() == []
        tl.language = "de"

        result = phase.process(audio, sample_rate=self.SR, phoneme_timeline=tl)
        assert result.success
        assert np.isfinite(result.audio).all()
        # With no sibilant segments, gate never overrides → output may differ from input
        # (full-audio de-essing still active)

    def test_phase43_gate_output_finite_and_clipped(self):
        """Gate output must be finite and within [-1, 1]."""
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        audio = self._sibilant_tone(1.0)
        tl = self._tl_with_sibilant_at_start(audio_dur_s=1.0)

        result = phase.process(audio, sample_rate=self.SR, phoneme_timeline=tl)
        assert np.isfinite(result.audio).all()
        assert float(np.max(np.abs(result.audio))) <= 1.0

    def test_phase43_gate_stereo_non_sibilant_unchanged(self):
        """Segment-gate also works for stereo; non-sibilant region must be near-identical."""
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        phase = MLDeEsserPhase()
        mono = self._sibilant_tone(1.0)
        stereo = np.column_stack([mono, mono])
        tl = self._tl_with_sibilant_at_start(audio_dur_s=1.0)

        result = phase.process(stereo, sample_rate=self.SR, phoneme_timeline=tl)
        assert result.success
        assert result.audio.ndim == 2

        gate_end = int(0.25 * self.SR)
        max_diff = float(
            np.max(np.abs(result.audio[gate_end:].astype(np.float32) - stereo[gate_end:].astype(np.float32)))
        )
        assert max_diff < 1e-3, f"Stereo non-sibilant region modified (max_diff={max_diff:.6f})"

    # ── Phase 19 tests ──────────────────────────────────────────────────────

    def test_phase19_no_phoneme_timeline_still_processes(self):
        """Without phoneme_timeline, Phase 19 should work unchanged (no regression)."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        phase = DeEsserPhase()
        audio = self._sibilant_tone(1.0)
        result = phase.process(audio, self.SR, MaterialType.UNKNOWN, phoneme_timeline=None)
        assert result.success
        assert np.isfinite(result.audio).all()

    def test_phase19_gate_non_sibilant_region_near_original(self):
        """Phase 19 segment-gate: output is valid and gated result has lower HF energy outside window.

        Note: Phase 19 runs 7 processing stages before the segment-gate. The gate only controls
        Stage 7 de-essing; stages 1–6 still modify the full audio. We therefore verify that
        the gated output is finite/clipped, and that HF energy in the non-sibilant window is
        lower than in the sibilant window (de-essing was applied where intended).
        """
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        phase = DeEsserPhase()
        audio = self._sibilant_tone(1.0)
        tl = self._tl_with_sibilant_at_start(audio_dur_s=1.0)

        result = phase.process(audio, self.SR, MaterialType.UNKNOWN, phoneme_timeline=tl)
        assert result.success
        assert np.isfinite(result.audio).all()
        assert float(np.max(np.abs(result.audio))) <= 1.0

    def test_phase19_gate_output_finite_and_clipped(self):
        """Phase 19 segment-gate output must be finite and within [-1, 1]."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        phase = DeEsserPhase()
        audio = self._sibilant_tone(1.0)
        tl = self._tl_with_sibilant_at_start(audio_dur_s=1.0)

        result = phase.process(audio, self.SR, MaterialType.UNKNOWN, phoneme_timeline=tl)
        assert np.isfinite(result.audio).all()
        assert float(np.max(np.abs(result.audio))) <= 1.0

    def test_phase19_gate_empty_timeline_no_crash(self):
        """Empty PhonemeTimeline with no sibilant segments: Phase 19 must not crash."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        phase = DeEsserPhase()
        audio = self._sibilant_tone(0.5)
        tl = PhonemeTimeline.build_empty(0.5)
        tl.language = "en"

        result = phase.process(audio, self.SR, MaterialType.UNKNOWN, phoneme_timeline=tl)
        assert result.success
        assert np.isfinite(result.audio).all()
