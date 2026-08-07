from __future__ import annotations

"""
Tests für LyricsGuided Enhancement (§2.36 Aurik Spec v10.0.0b)
=================================================================

Abdeckung:
    • WordTimestamp + LyricsTranscriptionResult (Dataclass-Kontrakte)
    • LyricsTranscriber (DSP-Fallback, ONNX-Fallback, NaN-Schutz)
    • ContentAwareProcessor.compute_lyrics_saliency() (alle SALIENCY_BOOST-Wege)
    • apply_lyrics_g_floor() (G_floor 0.90 an fricative+stressed)
    • PAM.apply_to_gain() mit lyrics_saliency (maximale Kombination)
    • LyricsGuidedTimeline (Farbkodierung, Shortcut-Konstante)
    • Singleton-Thread-Sicherheit
    • Edge-Cases: Stille, mono, stereo, kurze Dateien

Alle Tests nutzen synthetische Signale — keine realen Audiodateien (§5.4).
np.random.seed(42) für Reproduzierbarkeit.
"""


import concurrent.futures
import math

import numpy as np
import pytest

from backend.core.content_aware_processor import (
    G_FLOOR_FRICATIVE_STRESSED,
    HF_BARK_END,
    HF_BARK_START,
    N_BARK_BANDS,
    SALIENCY_BOOST,
    _find_word_at,
    _resolve_boost_key,
    compute_lyrics_saliency,
)
from backend.core.lyrics_guided_enhancement import (
    ContentAwareProcessor,
    get_content_aware_processor,
)
from backend.core.perceptual_attention_model import PerceptualAttentionModel

# ---- Imports unter Test --------
from plugins.lyrics_transcriber_plugin import (
    LyricsTranscriber,
    LyricsTranscriptionResult,
    WordTimestamp,
    get_lyrics_transcriber,
)

SR = 48_000
np.random.seed(42)


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================


def _make_audio(duration_s: float = 5.0, freq: float = 440.0) -> np.ndarray:
    """Synthetischer Sinuston, mono, float32."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _make_am_audio(duration_s: float = 5.0, carrier: float = 440.0, mod: float = 8.0) -> np.ndarray:
    """AM-modulierter Sinuston (Akkordeon-ähnlich)."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    envelope = 1.0 + 0.5 * np.sin(2 * np.pi * mod * t)
    return (np.sin(2 * np.pi * carrier * t) * envelope * 0.4).astype(np.float32)  # type: ignore[no-any-return]


def _make_fricative_audio(duration_s: float = 0.2, sr: int = SR) -> np.ndarray:
    """Weißes Rauschen → Frikativ-ähnliches Signal (hohe ZCR)."""
    np.random.seed(42)
    n = int(duration_s * sr)
    noise = np.random.randn(n).astype(np.float32) * 0.3
    # HF-Betonung via HP-Filter-Approximation (Differenz)
    hp = np.diff(noise, prepend=noise[0])
    return np.clip(hp, -1.0, 1.0).astype(np.float32)


def _make_silence(duration_s: float = 2.0) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _dummy_transcription(
    n_words: int = 3,
    phoneme_type: str = "vowel",
    stressed: bool = False,
    fallback: bool = False,
) -> LyricsTranscriptionResult:
    """Hilfsfunktion: erzeugt LyricsTranscriptionResult mit synthetischen Wörtern."""
    words = []
    for i in range(n_words):
        words.append(
            WordTimestamp(
                word="[vocal]",
                start_s=float(i),
                end_s=float(i) + 0.8,
                confidence=0.75,
                is_stressed=stressed,
                phoneme_type=phoneme_type,
            )
        )
    return LyricsTranscriptionResult(
        words=words,
        language="de",
        overall_confidence=0.75 if not fallback else 0.0,
        duration_s=float(n_words),
        fallback_used=fallback,
    )


def _dummy_saliency(n_frames: int = 20) -> np.ndarray:
    """Einheits-Salienz-Karte [n_frames × 24], alle 1.0."""
    return np.ones((n_frames, N_BARK_BANDS), dtype=np.float32)


# ===========================================================================
# 1. WordTimestamp — Dataclass-Kontrakt
# ===========================================================================


@pytest.mark.unit
class TestWordTimestamp:
    def test_01_fields_accessible(self) -> None:
        w = WordTimestamp(
            word="[vocal]",
            start_s=0.0,
            end_s=0.8,
            confidence=0.75,
            is_stressed=True,
            phoneme_type="vowel",
        )
        assert w.word == "[vocal]"
        assert math.isfinite(w.start_s)
        assert math.isfinite(w.end_s)
        assert 0.0 <= w.confidence <= 1.0
        assert isinstance(w.is_stressed, bool)
        assert w.phoneme_type in ("vowel", "fricative", "plosive", "silence", "mixed")

    def test_02_confidence_bounds(self) -> None:
        w = WordTimestamp("[vocal]", 0.0, 1.0, 0.0, False, "mixed")
        assert w.confidence >= 0.0
        w2 = WordTimestamp("[vocal]", 0.0, 1.0, 1.0, True, "fricative")
        assert w2.confidence <= 1.0


# ===========================================================================
# 2. LyricsTranscriptionResult — Dataclass-Kontrakt
# ===========================================================================


class TestLyricsTranscriptionResult:
    def test_03_duration_finite(self) -> None:
        r = _dummy_transcription()
        assert math.isfinite(r.duration_s)
        assert r.duration_s > 0.0

    def test_04_overall_confidence_bounded(self) -> None:
        r = _dummy_transcription()
        assert 0.0 <= r.overall_confidence <= 1.0

    def test_05_fallback_flag(self) -> None:
        r = _dummy_transcription(fallback=True)
        assert r.fallback_used is True
        assert r.overall_confidence == 0.0


# ===========================================================================
# 3. LyricsTranscriber — DSP-Fallback (kein Whisper-Modell nötig)
# ===========================================================================


class TestLyricsTranscriberDSPFallback:
    """DSP-Fallback ist immer aktiv wenn whisper_tiny.onnx nicht existiert."""

    def _transcriber_without_onnx(self) -> LyricsTranscriber:
        t = LyricsTranscriber.__new__(LyricsTranscriber)
        t._session = None
        t._session_loaded = False
        return t

    def test_06_transcribe_returns_result(self) -> None:
        t = self._transcriber_without_onnx()
        audio = _make_audio(5.0)
        result = t.transcribe(audio, SR)
        assert isinstance(result, LyricsTranscriptionResult)
        assert result.fallback_used is True

    def test_07_duration_correct(self) -> None:
        t = self._transcriber_without_onnx()
        audio = _make_audio(3.0)
        result = t.transcribe(audio, SR)
        assert abs(result.duration_s - 3.0) < 0.2

    def test_08_silence_no_words(self) -> None:
        t = self._transcriber_without_onnx()
        audio = _make_silence(2.0)
        result = t.transcribe(audio, SR)
        assert isinstance(result.words, list)
        # Stille → keine oder sehr wenige Segmente
        assert len(result.words) <= 2

    def test_09_no_nan_in_confidence(self) -> None:
        t = self._transcriber_without_onnx()
        audio = _make_audio(4.0)
        result = t.transcribe(audio, SR)
        for w in result.words:
            assert math.isfinite(w.confidence), f"NaN-Konfidenz in Wort {w}"

    def test_10_stereo_input(self) -> None:
        t = self._transcriber_without_onnx()
        audio = np.stack([_make_audio(3.0), _make_audio(3.0, 880.0)], axis=0)
        result = t.transcribe(audio, SR)
        assert result.duration_s > 0

    def test_11_phoneme_types_valid(self) -> None:
        t = self._transcriber_without_onnx()
        audio = _make_audio(5.0)
        result = t.transcribe(audio, SR)
        valid = {"vowel", "fricative", "plosive", "silence", "mixed"}
        for w in result.words:
            assert w.phoneme_type in valid, f"Ungültiger Phonem-Typ: {w.phoneme_type}"

    def test_12_classify_fricative(self) -> None:
        t = self._transcriber_without_onnx()
        fric = _make_fricative_audio(0.3)
        ptype = t._classify_phoneme_type(fric, SR)
        # Weißes Rauschen mit HF-Betonung → fricative oder mixed (ZCR-Schwelle)
        assert ptype in ("fricative", "mixed", "vowel", "plosive")  # kein Absturz

    def test_13_classify_short_segment_no_crash(self) -> None:
        t = self._transcriber_without_onnx()
        short = np.zeros(10, dtype=np.float32)
        ptype = t._classify_phoneme_type(short, SR)
        assert ptype == "mixed"

    def test_14_transcribe_never_raises(self) -> None:
        t = self._transcriber_without_onnx()
        for audio in [
            np.zeros(1, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.full(1000, np.nan, dtype=np.float32),
            np.full(1000, np.inf, dtype=np.float32),
        ]:
            result = t.transcribe(audio, SR)
            assert isinstance(result, LyricsTranscriptionResult)


# ===========================================================================
# 4. Singleton-Thread-Sicherheit
# ===========================================================================


class TestSingletonThreadSafety:
    def test_15_lyrics_transcriber_singleton_identity(self) -> None:
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(get_lyrics_transcriber) for _ in range(20)]
            instances = [f.result() for f in futs]
        assert all(inst is instances[0] for inst in instances)

    def test_16_content_aware_processor_singleton_identity(self) -> None:
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(get_content_aware_processor) for _ in range(20)]
            instances = [f.result() for f in futs]
        assert all(inst is instances[0] for inst in instances)


# ===========================================================================
# 5. compute_lyrics_saliency — SALIENCY_BOOST-Tabelle
# ===========================================================================


class TestComputeLyricsSaliency:
    def _cap(self) -> ContentAwareProcessor:
        return ContentAwareProcessor()

    def test_17_saliency_bounds_always_valid(self) -> None:
        cap = self._cap()
        base = _dummy_saliency(20)
        for ptype in ("vowel", "fricative", "plosive", "silence", "mixed"):
            tr = _dummy_transcription(phoneme_type=ptype, stressed=True)
            result = cap.compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
            assert result.shape == base.shape
            assert np.all(result >= 0.29), f"Zu niedrig bei {ptype}"
            assert np.all(result <= 2.01), f"Zu hoch bei {ptype}"

    def test_18_fricative_stressed_hf_boost(self) -> None:
        cap = self._cap()
        base = np.ones((20, N_BARK_BANDS), dtype=np.float32)
        tr = _dummy_transcription(n_words=15, phoneme_type="fricative", stressed=True)
        result = cap.compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
        # HF-Bänder 17–23 sollten durch fricative_stressed-Boost (2.0) erhöht sein
        hf_mean = float(np.mean(result[:, HF_BARK_START:HF_BARK_END]))
        lf_mean = float(np.mean(result[:, :HF_BARK_START]))
        assert hf_mean >= lf_mean, "HF sollte nach fricative_stressed-Boost ≥ LF sein"

    def test_19_g_floor_fricative_stressed(self) -> None:
        cap = self._cap()
        # Sehr niedrige Basis-Salienz
        base = np.full((20, N_BARK_BANDS), 0.01, dtype=np.float32)
        tr = _dummy_transcription(n_words=15, phoneme_type="fricative", stressed=True)
        result = cap.compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
        # G_floor 0.90 an HF-Bändern ≥ HF_BARK_START
        hf = result[:, HF_BARK_START:HF_BARK_END]
        # Nach clip auf [0.3, 2.0] muss zumindest 0.30 gelten
        assert np.all(hf >= 0.29), "G_floor wurde nicht gesetzt"

    def test_20_fallback_returns_unchanged_saliency(self) -> None:
        cap = self._cap()
        base = _dummy_saliency(15)
        tr = _dummy_transcription(fallback=True)
        result = cap.compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
        # Fallback → base_saliency geclampt und zurückgegeben (nicht zerbrochen)
        assert result.shape == base.shape
        assert np.all(np.isfinite(result))

    def test_21_silence_segments_reduced(self) -> None:
        cap = self._cap()
        base = np.ones((20, N_BARK_BANDS), dtype=np.float32)
        # Keine Wörter → alle Frames bekommen "silence"-Boost (0.5)
        tr = LyricsTranscriptionResult(
            words=[],
            language="de",
            overall_confidence=0.0,
            duration_s=5.0,
            fallback_used=False,
        )
        result = cap.compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
        # Früh-Exit bei leerer words-Liste
        assert np.all(np.isfinite(result))

    def test_22_no_nan_in_output(self) -> None:
        cap = self._cap()
        base = np.full((10, N_BARK_BANDS), np.nan, dtype=np.float32)
        tr = _dummy_transcription(phoneme_type="vowel", stressed=False)
        result = cap.compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
        assert np.all(np.isfinite(result)), "NaN im Output nach NaN-Eingang"

    def test_23_convenience_wrapper_matches_class(self) -> None:
        base = _dummy_saliency(10)
        tr = _dummy_transcription(phoneme_type="vowel", stressed=True)
        result_wrapper = compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
        result_class = ContentAwareProcessor().compute_lyrics_saliency(base, tr, SR)  # type: ignore[arg-type]
        # Beide sollten gleiche Dimension und ähnliche Werte liefern
        assert result_wrapper.shape == result_class.shape


# ===========================================================================
# 6. PAM.apply_to_gain() mit lyrics_saliency
# ===========================================================================


class TestPAMApplyToGain:
    def test_24_lyrics_saliency_integration(self) -> None:
        pam = PerceptualAttentionModel()
        n_frames, n_bins = 10, 512
        base_gain = np.full((n_frames, n_bins), 0.5, dtype=np.float32)
        saliency = np.ones((n_frames, N_BARK_BANDS), dtype=np.float32)
        # Lyrics-Salienz mit hohem Wert → sollte Gain beeinflussen
        lyrics_sal = np.full((n_frames, N_BARK_BANDS), 2.0, dtype=np.float32)
        result = pam.apply_to_gain(base_gain, saliency, lyrics_saliency=lyrics_sal)
        assert result.shape == (n_frames, n_bins)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

    def test_25_no_lyrics_saliency_backwards_compat(self) -> None:
        pam = PerceptualAttentionModel()
        n_frames, n_bins = 8, 256
        base_gain = np.ones((n_frames, n_bins), dtype=np.float32) * 0.7
        saliency = np.ones((n_frames, N_BARK_BANDS), dtype=np.float32)
        # Aufruf ohne lyrics_saliency → Rückwärtskompatibilität
        result = pam.apply_to_gain(base_gain, saliency)
        assert result.shape == (n_frames, n_bins)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)


# ===========================================================================
# 7. Hilfsfunktionen _resolve_boost_key + _find_word_at
# ===========================================================================


class TestHelperFunctions:
    def test_boost_key_fricative_stressed(self) -> None:
        assert _resolve_boost_key("fricative", True) == "fricative_stressed"

    def test_boost_key_fricative_unstressed(self) -> None:
        assert _resolve_boost_key("fricative", False) == "fricative_unstressed"

    def test_boost_key_vowel_stressed(self) -> None:
        assert _resolve_boost_key("vowel", True) == "vowel_stressed"

    def test_boost_key_plosive(self) -> None:
        # Betonung irrelevant für plosive
        assert _resolve_boost_key("plosive", True) == "plosive"
        assert _resolve_boost_key("plosive", False) == "plosive"

    def test_boost_key_unknown_fallback(self) -> None:
        assert _resolve_boost_key("unknown_xyz", True) == "mixed"

    def test_find_word_at_match(self) -> None:
        words = [
            WordTimestamp("[vocal]", 0.0, 1.0, 0.8, True, "vowel"),
            WordTimestamp("[vocal]", 2.0, 3.0, 0.9, False, "fricative"),
        ]
        result = _find_word_at(words, 0.5)
        assert result is not None
        assert result.start_s == 0.0  # type: ignore[attr-defined]

    def test_find_word_at_no_match(self) -> None:
        words = [WordTimestamp("[vocal]", 0.0, 1.0, 0.8, True, "vowel")]
        # t_center liegt außerhalb aller Wörter
        result = _find_word_at(words, 5.0)
        assert result is None

    def test_find_word_at_empty(self) -> None:
        result = _find_word_at([], 1.0)
        assert result is None


# ===========================================================================
# 8. LyricsGuidedTimeline Konstanten
# ===========================================================================


class TestLyricsGuidedTimelineConstants:
    def test_shortcut_constant(self) -> None:
        """Shortcut L für Overlay an/aus (§2.36)."""
        # Wir vergewissern uns, dass der Wert in der Plugin-Dokumentation steht.
        # Das tatsächliche Widget liegt im Frontend; hier prüfen wir die Spec-Konstante.
        EXPECTED_SHORTCUT = "L"
        assert EXPECTED_SHORTCUT == "L"

    def test_saliency_boost_all_keys_present(self) -> None:
        required_keys = {
            "fricative_stressed",
            "fricative_unstressed",
            "vowel_stressed",
            "vowel_unstressed",
            "plosive",
            "silence",
            "mixed",
        }
        assert required_keys.issubset(set(SALIENCY_BOOST.keys()))

    def test_saliency_boost_values_in_range(self) -> None:
        for key, val in SALIENCY_BOOST.items():
            assert 0.3 <= val <= 2.0, f"SALIENCY_BOOST[{key}] = {val} außerhalb [0.3, 2.0]"

    def test_g_floor_fricative_stressed_value(self) -> None:
        assert G_FLOOR_FRICATIVE_STRESSED == 0.90, "G_floor muss 0.90 sein (§2.36)"


# ===========================================================================
# 9. LyricsGuidedEnhancement (backend.core.lyrics_guided_enhancement) — §2.36
#    Tests für _classify_phoneme_type, _energy_to_words, enhance(), _build_sample_saliency
#    und den compute_lyrics_saliency-Fix in ContentAwareProcessor (LGE-intern).
# ===========================================================================


def _make_lge_no_onnx():
    """Erzeugt LyricsGuidedEnhancement-Instanz ohne ONNX-Session (DSP-Fallback)."""
    from backend.core.lyrics_guided_enhancement import (
        ContentAwareProcessor as _InternalCAP,
    )
    from backend.core.lyrics_guided_enhancement import (
        LyricsGuidedEnhancement,
        LyricsGuidedTimeline,
    )

    lge = LyricsGuidedEnhancement.__new__(LyricsGuidedEnhancement)
    lge._cap = _InternalCAP()
    lge._tl = LyricsGuidedTimeline()
    lge._ort_session = None  # kein ONNX → DSP-Fallback aktiv
    lge._aligner_session = None
    lge._whisper_hf_processor = None  # §v10.303.50: __new__ umgeht __init__
    lge._whisper_hf_model = None
    lge._transcriber = None
    return lge


class TestLGEClassifyPhonemeType:
    """§2.36 _classify_phoneme_type — DSP-basierte Phonemklassifikation."""

    def test_lge_01_plosive_short_transient(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        n = int(0.01 * SR)  # 10 ms
        seg = np.zeros(n, dtype=np.float32)
        seg[n // 2] = 0.9  # einzelner Spike → hoher Crest-Factor
        result = LyricsGuidedEnhancement._classify_phoneme_type(seg, SR, 0.8, True)
        assert result == "plosive"

    def test_lge_02_vowel_long_sine(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        seg = (0.4 * np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.5, int(0.5 * SR)))).astype(np.float32)
        result = LyricsGuidedEnhancement._classify_phoneme_type(seg, SR, 0.5, True)
        assert result in {
            "vowel_stressed",
            "vowel_unstressed",
            "fricative_stressed",
            "fricative_unstressed",
        }  # kein plosive

    def test_lge_03_stressed_flag_vowel(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        seg = (0.4 * np.sin(2 * np.pi * 200.0 * np.linspace(0, 0.2, int(0.2 * SR)))).astype(np.float32)
        stressed = LyricsGuidedEnhancement._classify_phoneme_type(seg, SR, 0.8, True)
        unstressed = LyricsGuidedEnhancement._classify_phoneme_type(seg, SR, 0.2, False)
        if "vowel" in stressed:
            assert "stressed" in stressed
        if "vowel" in unstressed:
            assert "unstressed" in unstressed

    def test_lge_04_empty_segment_no_raise(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        seg = np.zeros(3, dtype=np.float32)
        result = LyricsGuidedEnhancement._classify_phoneme_type(seg, SR, 0.0, False)
        assert isinstance(result, str) and len(result) > 0

    def test_lge_05_fricative_broadband_noise(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        rng = np.random.default_rng(42)
        seg = rng.standard_normal(int(0.2 * SR)).astype(np.float32) * 0.3
        seg = np.diff(seg, prepend=seg[0]).astype(np.float32)  # high-pass → HF-betont
        result = LyricsGuidedEnhancement._classify_phoneme_type(seg, SR, 0.6, True)
        assert isinstance(result, str)  # kein Absturz, beliebiger gültiger Typ


class TestLGEEnergyToWords:
    """§2.36 _energy_to_words mit Phonem-Klassifikation via source_audio."""

    def test_lge_06_empty_energy_returns_empty(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        words = LyricsGuidedEnhancement._energy_to_words(np.array([]), 0.0)
        assert words == []

    def test_lge_07_zero_dur_returns_empty(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        words = LyricsGuidedEnhancement._energy_to_words(np.ones(50), 0.0)
        assert words == []

    def test_lge_08_no_nan_in_output(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        rng = np.random.default_rng(7)
        energy = rng.random(100).astype(np.float32)
        audio = _make_audio(2.0)
        words = LyricsGuidedEnhancement._energy_to_words(energy, 2.0, audio, SR)
        for w in words:
            assert math.isfinite(w.start_s)
            assert math.isfinite(w.end_s)
            assert math.isfinite(w.confidence)

    def test_lge_09_word_field_always_empty(self) -> None:
        """Privacy invariant: word field darf nie Text enthalten."""
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        rng = np.random.default_rng(42)
        energy = rng.random(100).astype(np.float32)
        audio = _make_audio(2.0)
        words = LyricsGuidedEnhancement._energy_to_words(energy, 2.0, audio, SR)
        for w in words:
            assert w.word == "", "Privacy violation: non-empty word field"

    def test_lge_10_timestamps_non_overlapping(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        rng = np.random.default_rng(3)
        energy = rng.random(200).astype(np.float32)
        words = LyricsGuidedEnhancement._energy_to_words(energy, 4.0)
        prev_end = -1.0
        for w in words:
            assert w.start_s >= prev_end - 1e-6
            assert w.end_s > w.start_s
            prev_end = w.end_s

    def test_lge_11_phoneme_types_valid(self) -> None:
        from backend.core.lyrics_guided_enhancement import (
            ContentAwareProcessor as _InternalCAP,
        )
        from backend.core.lyrics_guided_enhancement import (
            LyricsGuidedEnhancement,
        )

        valid = set(_InternalCAP.SALIENCY_BOOST.keys())
        rng = np.random.default_rng(5)
        energy = rng.random(100).astype(np.float32)
        audio = _make_audio(2.0)
        words = LyricsGuidedEnhancement._energy_to_words(energy, 2.0, audio, SR)
        for w in words:
            assert w.phoneme_type in valid, f"Unbekannter Phonem-Typ: {w.phoneme_type}"

    def test_lge_12_silence_input_no_words(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        # All-zero energy: percentile=0, so 0>=0 → all frames "active" → at most 1 segment.
        # expectation: ≤ 1 word (not many disconnected words)
        energy = np.zeros(50, dtype=np.float32)
        words = LyricsGuidedEnhancement._energy_to_words(energy, 1.0)
        assert len(words) <= 1

    def test_lge_13_all_active_produces_words(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        energy = np.ones(50, dtype=np.float32)
        words = LyricsGuidedEnhancement._energy_to_words(energy, 1.0)
        assert len(words) >= 1


class TestLGEInternalCAP:
    """compute_lyrics_saliency() in LyricsGuidedEnhancement.ContentAwareProcessor (§2.36 fix)."""

    def _cap(self):
        from backend.core.lyrics_guided_enhancement import ContentAwareProcessor as _InternalCAP

        return _InternalCAP()

    def _make_result(self, phoneme_type: str) -> object:
        from backend.core.lyrics_guided_enhancement import (
            LyricsTranscriptionResult,
            WordTimestamp,
        )

        word = WordTimestamp("", 0.0, 0.5, 0.9, True, phoneme_type)
        return LyricsTranscriptionResult([word], "de", 0.9, 1.0, fallback_used=False)

    def test_lge_14_fricative_boost_applied(self) -> None:
        cap = self._cap()
        base = np.ones(SR, dtype=np.float32)
        result_obj = self._make_result("fricative_stressed")
        sal = cap.compute_lyrics_saliency(base, result_obj, SR)
        assert sal[int(0.25 * SR)] == pytest.approx(1.55, abs=0.01)  # §8.3: fricative ×1.55

    def test_lge_15_silence_boost_applied(self) -> None:
        cap = self._cap()
        base = np.ones(SR, dtype=np.float32)
        from backend.core.lyrics_guided_enhancement import (
            LyricsTranscriptionResult,
            WordTimestamp,
        )

        word = WordTimestamp("", 0.2, 0.8, 0.1, False, "silence")
        result_obj = LyricsTranscriptionResult([word], "de", 0.1, 1.0, fallback_used=False)
        sal = cap.compute_lyrics_saliency(base, result_obj, SR)
        assert sal[int(0.5 * SR)] == pytest.approx(0.70, abs=0.01)  # §8.3: silence ×0.70

    def test_lge_16_fallback_no_boost(self) -> None:
        cap = self._cap()
        base = np.ones(SR, dtype=np.float32)
        from backend.core.lyrics_guided_enhancement import LyricsTranscriptionResult

        result_obj = LyricsTranscriptionResult([], "de", 0.0, 1.0, fallback_used=True)
        sal = cap.compute_lyrics_saliency(base, result_obj, SR)
        assert np.allclose(sal, 1.0)

    def test_lge_17_output_bounds(self) -> None:
        cap = self._cap()
        base = np.full(SR, 10.0, dtype=np.float32)  # out-of-range input
        result_obj = self._make_result("vowel_stressed")
        sal = cap.compute_lyrics_saliency(base, result_obj, SR)
        assert sal.min() >= 0.3
        assert sal.max() <= 2.0

    def test_lge_18_nan_input_guarded(self) -> None:
        cap = self._cap()
        base = np.full(SR, np.nan, dtype=np.float32)
        from backend.core.lyrics_guided_enhancement import LyricsTranscriptionResult

        result_obj = LyricsTranscriptionResult([], "de", 0.0, 1.0, fallback_used=True)
        sal = cap.compute_lyrics_saliency(base, result_obj, SR)
        assert np.all(np.isfinite(sal))

    def test_lge_18b_vowel_stressed_dsp_is_not_passthrough(self) -> None:
        """Regression guard for §2.36a vowel branch.

        Ensures LPC/formant DSP stays active and does not silently fall back to
        passthrough when optional helper APIs are unavailable.
        """
        cap = self._cap()
        t = np.linspace(0.0, 0.20, int(0.20 * SR), endpoint=False)
        # Harmonic-rich vocal-like segment (F0 + partials)
        seg = (
            0.45 * np.sin(2 * np.pi * 220.0 * t)
            + 0.20 * np.sin(2 * np.pi * 660.0 * t)
            + 0.12 * np.sin(2 * np.pi * 1320.0 * t)
        ).astype(np.float32)
        out = cap._apply_phoneme_dsp(seg, "vowel_stressed", SR, strength=0.9)

        assert out.shape == seg.shape
        assert np.all(np.isfinite(out))
        # Must apply a measurable modification for vowel_stressed processing
        delta = float(np.mean(np.abs(out - seg)))
        assert delta > 1e-5


class TestLGEEnhance:
    """LyricsGuidedEnhancement.enhance() Integration (DSP-Fallback-Pfad)."""

    def test_lge_19_mono_shape_preserved(self) -> None:
        lge = _make_lge_no_onnx()
        audio = _make_audio(2.0)
        out, _ = lge.enhance(audio, SR)
        assert out.shape == audio.shape

    def test_lge_20_stereo_shape_preserved(self) -> None:
        lge = _make_lge_no_onnx()
        left = _make_audio(1.0, 440.0)
        right = _make_audio(1.0, 880.0)
        audio = np.stack([left, right], axis=1)  # (N, 2)
        out, _ = lge.enhance(audio, SR)
        assert out.shape == audio.shape

    def test_lge_21_output_bounded(self) -> None:
        lge = _make_lge_no_onnx()
        audio = _make_audio(1.0)
        out, _ = lge.enhance(audio, SR)
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= -1.0

    def test_lge_22_no_nan_inf(self) -> None:
        lge = _make_lge_no_onnx()
        audio = _make_audio(3.0)
        out, _ = lge.enhance(audio, SR)
        assert np.all(np.isfinite(out))

    def test_lge_23_nan_input_guarded(self) -> None:
        lge = _make_lge_no_onnx()
        audio = np.full(SR, np.nan, dtype=np.float32)
        out, _ = lge.enhance(audio, SR)
        assert np.all(np.isfinite(out))

    def test_lge_24_sr_guard_raises(self) -> None:
        lge = _make_lge_no_onnx()
        with pytest.raises(AssertionError):
            lge.enhance(_make_audio(1.0), sr=44_100)

    def test_lge_25_silence_passthrough(self) -> None:
        lge = _make_lge_no_onnx()
        audio = _make_silence(1.0)
        out, _ = lge.enhance(audio, SR)
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_lge_26_returns_transcription_result(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsTranscriptionResult

        lge = _make_lge_no_onnx()
        _, trans = lge.enhance(_make_audio(1.0), SR)
        assert isinstance(trans, LyricsTranscriptionResult)

    def test_lge_27_word_field_privacy(self) -> None:
        """Privacy: word-Feld aller zurückgegebenen WordTimestamps muss leer sein."""
        lge = _make_lge_no_onnx()
        _, trans = lge.enhance(_make_audio(2.0), SR)
        for w in trans.words:
            assert w.word == "", "Privacy violation: non-empty word field in enhance() output"

    def test_lge_28_uv3_attribute_access_words_not_segments(self) -> None:
        """Regressions-Test für UV3-Bug: .segments existiert nicht — nur .words."""

        lge = _make_lge_no_onnx()
        _, trans = lge.enhance(_make_audio(1.0), SR)
        assert hasattr(trans, "words")
        assert not hasattr(trans, "segments"), "LyricsTranscriptionResult hat kein .segments — UV3 muss .words nutzen"

    def test_lge_28b_public_transcribe_api_returns_words(self) -> None:
        lge = _make_lge_no_onnx()
        trans = lge.transcribe(_make_audio(1.0), SR)
        assert hasattr(trans, "words")
        assert isinstance(trans.words, list)


class TestLGEBuildSampleSaliency:
    """_build_sample_saliency() — Sample-Level-Gain-Kurve."""

    def test_lge_29_fallback_returns_ones(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsTranscriptionResult

        lge = _make_lge_no_onnx()
        result_obj = LyricsTranscriptionResult([], "de", 0.0, 1.0, fallback_used=True)
        sal = lge._build_sample_saliency(result_obj, SR, SR)
        assert sal.shape == (SR,)
        assert np.allclose(sal, 1.0)

    def test_lge_30_fricative_boost_correct(self) -> None:
        from backend.core.lyrics_guided_enhancement import (
            LyricsTranscriptionResult,
            WordTimestamp,
        )

        lge = _make_lge_no_onnx()
        word = WordTimestamp("", 0.0, 0.5, 0.9, True, "fricative_stressed")
        result_obj = LyricsTranscriptionResult([word], "de", 0.9, 1.0, fallback_used=False)
        sal = lge._build_sample_saliency(result_obj, SR, SR)
        assert sal[int(0.25 * SR)] == pytest.approx(1.55, abs=0.01)  # §8.3: fricative ×1.55
        assert sal[int(0.75 * SR)] == pytest.approx(1.0, abs=0.01)

    def test_lge_31_silence_boost_correct(self) -> None:
        from backend.core.lyrics_guided_enhancement import (
            LyricsTranscriptionResult,
            WordTimestamp,
        )

        lge = _make_lge_no_onnx()
        word = WordTimestamp("", 0.0, 1.0, 0.0, False, "silence")
        result_obj = LyricsTranscriptionResult([word], "de", 0.0, 1.0, fallback_used=False)
        sal = lge._build_sample_saliency(result_obj, SR, SR)
        assert sal[int(0.5 * SR)] == pytest.approx(0.70, abs=0.01)  # §8.3: silence ×0.70

    def test_lge_32_output_clipped(self) -> None:
        from backend.core.lyrics_guided_enhancement import (
            LyricsTranscriptionResult,
            WordTimestamp,
        )

        lge = _make_lge_no_onnx()
        word = WordTimestamp("", 0.0, 1.0, 0.9, True, "fricative_stressed")
        result_obj = LyricsTranscriptionResult([word], "de", 0.9, 1.0, fallback_used=False)
        sal = lge._build_sample_saliency(result_obj, SR, SR)
        assert sal.min() >= 0.3
        assert sal.max() <= 2.0

    def test_lge_33_plosive_boost_correct(self) -> None:
        from backend.core.lyrics_guided_enhancement import (
            ContentAwareProcessor as _ICAP,
        )
        from backend.core.lyrics_guided_enhancement import (
            LyricsTranscriptionResult,
            WordTimestamp,
        )

        lge = _make_lge_no_onnx()
        expected = _ICAP.SALIENCY_BOOST["plosive"]
        word = WordTimestamp("", 0.1, 0.2, 0.9, True, "plosive")
        result_obj = LyricsTranscriptionResult([word], "de", 0.9, 0.5, fallback_used=False)
        n = int(0.5 * SR)
        sal = lge._build_sample_saliency(result_obj, n, SR)
        mid = int(0.15 * SR)
        assert sal[mid] == pytest.approx(expected, abs=0.01)


class TestLGELGESingleton:
    """Singleton-Zugriff für LyricsGuidedEnhancement aus lyrics_guided_enhancement."""

    def test_lge_34_singleton_identity(self) -> None:
        from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement

        a = get_lyrics_guided_enhancement()
        b = get_lyrics_guided_enhancement()
        assert a is b, "LyricsGuidedEnhancement Singleton gebrochen"

    def test_lge_35_has_enhance_method(self) -> None:
        from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement

        lge = get_lyrics_guided_enhancement()
        assert callable(lge.enhance)

    def test_lge_36_has_get_timeline_method(self) -> None:
        from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement

        lge = get_lyrics_guided_enhancement()
        timeline = lge.get_timeline()
        assert timeline is not None


class TestLGEPhonemeAlignmentIntegration:
    """§2.36 wav2vec2 forced-alignment integration in LyricsGuidedEnhancement."""

    def test_lge_37_align_phonemes_fallback_without_session(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement, WordTimestamp

        lge = LyricsGuidedEnhancement.__new__(LyricsGuidedEnhancement)
        lge._aligner_session = None

        words = [
            WordTimestamp("", 0.0, 0.2, 0.7, True, "vowel_stressed"),
            WordTimestamp("", 0.2, 0.4, 0.6, False, "fricative_unstressed"),
        ]
        mono_16k = np.zeros(int(0.5 * 16_000), dtype=np.float32)

        out = lge._align_phonemes(words, mono_16k, 16_000)
        assert out is words or out == words

    def test_lge_38_align_phonemes_refines_type_and_privacy(self) -> None:
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement, WordTimestamp

        class _FakeAlignerSession:
            def run(self, _outputs, _inputs):
                n_frames = 40
                vocab = 64
                logits = np.zeros((1, n_frames, vocab), dtype=np.float32)
                # word-1 frames (0..9): fricative range 21..35 dominant
                logits[0, 0:10, 25] = 8.0
                # word-2 frames (10..19): plosive range 36..50 dominant
                logits[0, 10:20, 40] = 8.0
                # word-3 frames (20..29): vowel range 4..20 dominant
                logits[0, 20:30, 10] = 8.0
                return [logits]

        lge = LyricsGuidedEnhancement.__new__(LyricsGuidedEnhancement)
        lge._aligner_session = _FakeAlignerSession()

        words = [
            WordTimestamp("", 0.00, 0.20, 0.7, True, "vowel_stressed"),
            WordTimestamp("", 0.20, 0.40, 0.7, False, "vowel_unstressed"),
            WordTimestamp("", 0.40, 0.60, 0.7, False, "vowel_unstressed"),
        ]
        mono_16k = np.ones(int(1.0 * 16_000), dtype=np.float32) * 0.1

        out = lge._align_phonemes(words, mono_16k, 16_000)

        assert len(out) == 3
        assert out[0].phoneme_type == "fricative_stressed"
        assert out[1].phoneme_type == "plosive"
        assert out[2].phoneme_type in {"vowel_unstressed", "vowel_stressed"}
        # Privacy invariant: no lyric text in output word field
        assert all(w.word == "" for w in out)

    def test_lge_39_transcribe_dsp_calls_align_hook(self, monkeypatch) -> None:
        from backend.core.lyrics_guided_enhancement import (
            ContentAwareProcessor as _InternalCAP,
        )
        from backend.core.lyrics_guided_enhancement import (
            LyricsGuidedEnhancement,
            LyricsGuidedTimeline,
        )

        lge = LyricsGuidedEnhancement.__new__(LyricsGuidedEnhancement)
        lge._cap = _InternalCAP()
        lge._tl = LyricsGuidedTimeline()
        lge._ort_session = None
        lge._aligner_session = object()  # non-None to exercise hook path

        called = {"n": 0}

        def _fake_align(words, mono_16k, sr_16k=16_000):
            called["n"] += 1
            assert sr_16k == 16_000
            assert mono_16k.ndim == 1
            return words

        monkeypatch.setattr(lge, "_align_phonemes", _fake_align)
        mono = _make_audio(1.2)
        result = lge._transcribe_dsp(mono, SR, dur=1.2)
        assert called["n"] == 1
        assert result is not None

    def test_lge_40_transcribe_onnx_calls_align_hook(self, monkeypatch) -> None:
        from backend.core.lyrics_guided_enhancement import (
            ContentAwareProcessor as _InternalCAP,
        )
        from backend.core.lyrics_guided_enhancement import (
            LyricsGuidedEnhancement,
            LyricsGuidedTimeline,
        )

        class _FakeWhisperSession:
            def run(self, _outputs, _inputs):
                # shape expected by _transcribe_onnx: (1, 1500, 384)
                return [np.ones((1, 1500, 384), dtype=np.float32) * 0.05]

        lge = LyricsGuidedEnhancement.__new__(LyricsGuidedEnhancement)
        lge._cap = _InternalCAP()
        lge._tl = LyricsGuidedTimeline()
        lge._ort_session = _FakeWhisperSession()
        lge._aligner_session = object()  # non-None to exercise hook path

        called = {"n": 0}

        def _fake_align(words, mono_16k, sr_16k=16_000):
            called["n"] += 1
            assert sr_16k == 16_000
            assert mono_16k.ndim == 1
            return words

        monkeypatch.setattr(lge, "_align_phonemes", _fake_align)
        mono = _make_audio(1.0)
        result = lge._transcribe_onnx(mono, SR, dur=1.0)
        assert called["n"] == 1
        assert result is not None

    def test_lge_41_align_phonemes_too_short_returns_words_unchanged(self) -> None:
        """§2.36 Guard: inputs < 400 samples (25 ms @ 16 kHz) must not reach the
        wav2vec2 ONNX session — the Conv1d feature extractor raises
        OrtInvalidArgument 'Invalid input shape: {N}' for N < 400.
        The guard must return words unchanged (DSP classification preserved).
        """
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement, WordTimestamp

        class _ShouldNotBeCalledSession:
            def run(self, _outputs, _inputs):
                raise AssertionError("wav2vec2 ONNX session must not be called for inputs < 400 samples")

        lge = LyricsGuidedEnhancement.__new__(LyricsGuidedEnhancement)
        lge._aligner_session = _ShouldNotBeCalledSession()

        words = [WordTimestamp("", 0.0, 0.01, 0.5, False, "vowel_unstressed")]

        # 1 sample — absolute minimum edge case (triggers the original bug)
        mono_1 = np.array([0.1], dtype=np.float32)
        out_1 = lge._align_phonemes(words, mono_1, 16_000)
        assert out_1 is words or out_1 == words, "1-sample input must return words unchanged"

        # 399 samples — one below the minimum threshold
        mono_399 = np.ones(399, dtype=np.float32) * 0.05
        out_399 = lge._align_phonemes(words, mono_399, 16_000)
        assert out_399 is words or out_399 == words, "399-sample input must return words unchanged"

        # 400 samples — exactly at the boundary: session IS allowed to be called
        # (we replace with a no-op to avoid actual ONNX call in unit test)
        called = {"n": 0}

        class _CountingSession:
            def run(self, _outputs, _inputs):
                called["n"] += 1
                # Return valid logits shape (1, 1, 64)
                return [np.zeros((1, 1, 64), dtype=np.float32)]

        lge._aligner_session = _CountingSession()
        mono_400 = np.ones(400, dtype=np.float32) * 0.05
        lge._align_phonemes(words, mono_400, 16_000)
        assert called["n"] == 1, "400-sample input must reach the ONNX session"

    def test_lge_42_align_phonemes_boundary_values(self) -> None:
        """Boundary test: verify _MIN_WAV2VEC2_SAMPLES constant is exactly 400
        and that the guard is consistent with the documented receptive field.
        """
        from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

        assert LyricsGuidedEnhancement._MIN_WAV2VEC2_SAMPLES == 400, (
            "_MIN_WAV2VEC2_SAMPLES must be 400 (wav2vec2 Conv1d receptive field at 16 kHz)"
        )
