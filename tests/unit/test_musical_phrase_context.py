from __future__ import annotations

import pytest

"""Pflicht-Tests für MusicalPhraseContextExtractor (§2.12).

Testkonventionen:
    - np.random.seed(42) für Reproduzierbarkeit
    - Nur synthetische Signale (keine echten Audio-Dateien)
    - SR = 48000 Hz (Aurik-Invariante)
    - Alle Tests ≤ 30 s Laufzeit
"""


import threading

import numpy as np

# ---------------------------------------------------------------------------
# Imports unter Test
# ---------------------------------------------------------------------------
from backend.core.musical_phrase_context import (
    MusicalPhraseContextExtractor,
    PhraseBoundary,
    PhraseContext,
    get_phrase_extractor,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
SR = 48_000


def _sine(freq: float = 440.0, duration_s: float = 30.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(duration_s: float = 30.0, sr: int = SR, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(duration_s * sr)).astype(np.float32) * 0.1


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono * 0.9], axis=1)


def _make_gap(audio_len: int, gap_s: float = 0.2, sr: int = SR) -> tuple[int, int]:
    """Gibt (gap_start, gap_end) in Samples zurück (mittig)"""
    center = audio_len // 2
    half = int(gap_s * sr // 2)
    return center - half, center + half


# ---------------------------------------------------------------------------
# Testklasse: Grundlegende Korrektheit
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPhraseContextBasic:
    def test_01_returns_phrase_context(self) -> None:
        """extract_context gibt eine PhraseContext-Instanz zurück."""
        np.random.seed(42)
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)

    def test_02_audio_context_not_empty(self) -> None:
        """audio_context enthält Audio-Daten (> 0 Samples)."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert ctx.audio_context is not None
        assert len(ctx.audio_context) > 0

    def test_03_chroma_mean_shape(self) -> None:
        """chroma_mean hat Länge 12 (Chroma-Klassen)."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert ctx.chroma_mean.shape == (12,)

    def test_04_chroma_mean_no_nan(self) -> None:
        """chroma_mean ist NaN- und Inf-frei."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert np.all(np.isfinite(ctx.chroma_mean))

    def test_05_tempo_in_valid_range(self) -> None:
        """tempo_bpm liegt im validen Bereich [40, 240]."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert 40.0 <= ctx.tempo_bpm <= 240.0, f"tempo_bpm={ctx.tempo_bpm} außerhalb [40,240]"

    def test_06_beat_positions_are_list(self) -> None:
        """beat_positions ist eine Liste (kann leer sein)."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx.beat_positions, list)

    def test_07_phrase_start_before_end(self) -> None:
        """phrase_start_s < phrase_end_s."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert ctx.phrase_start_s < ctx.phrase_end_s

    def test_08_gap_positions_match(self) -> None:
        """gap_start_s und gap_end_s entsprechen den übergebenen Sample-Positionen."""
        np.random.seed(42)
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        expected_start = gs / SR
        expected_end = ge / SR
        # Toleranz: ±10 ms
        assert abs(ctx.gap_start_s - expected_start) < 0.01
        assert abs(ctx.gap_end_s - expected_end) < 0.01

    def test_09_is_fallback_bool(self) -> None:
        """is_fallback ist ein boolescher Wert."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx.is_fallback, bool)

    def test_10_audio_context_dtype_float32(self) -> None:
        """audio_context hat dtype float32."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert ctx.audio_context.dtype == np.float32

    def test_11_audio_context_no_nan(self) -> None:
        """audio_context ist NaN- und Inf-frei."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert np.all(np.isfinite(ctx.audio_context))


# ---------------------------------------------------------------------------
# Testklasse: Edge Cases
# ---------------------------------------------------------------------------
class TestPhraseContextEdgeCases:
    def test_12_short_audio_no_crash(self) -> None:
        """Kurzes Audio (< 8 s) → kein Absturz, is_fallback=True erwartet."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=5.0)
        gs, ge = int(2.0 * SR), int(2.2 * SR)
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)
        # Kurzes Audio → Fallback erwartet
        assert ctx.is_fallback is True

    def test_13_white_noise_no_crash(self) -> None:
        """Weißes Rauschen liefert eine valide PhraseContext-Instanz."""
        extractor = get_phrase_extractor()
        audio = _white_noise(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)
        assert np.all(np.isfinite(ctx.chroma_mean))

    def test_14_silence_no_crash(self) -> None:
        """Stille liefert eine valide PhraseContext-Instanz ohne Exception."""
        extractor = get_phrase_extractor()
        audio = np.zeros(int(30.0 * SR), dtype=np.float32)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)

    def test_15_gap_at_start(self) -> None:
        """Lücke am Anfang des Audios → kein Absturz."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = 0, int(0.3 * SR)
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)

    def test_16_gap_at_end(self) -> None:
        """Lücke am Ende des Audios → kein Absturz."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        n = len(audio)
        gs, ge = n - int(0.3 * SR), n - 1
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)

    def test_17_stereo_input_no_crash(self) -> None:
        """Stereo-Eingabe (2-channel) → kein Absturz."""
        extractor = get_phrase_extractor()
        audio = _stereo(_sine(duration_s=30.0))
        gs, ge = _make_gap(audio.shape[0])
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)

    def test_18_dirac_impulse(self) -> None:
        """Dirac-Impuls als Eingabe → keine NaN, kein Absturz."""
        extractor = get_phrase_extractor()
        audio = np.zeros(int(30.0 * SR), dtype=np.float32)
        audio[int(15.0 * SR)] = 1.0
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)
        assert np.all(np.isfinite(ctx.chroma_mean))

    def test_19_long_gap_uses_adjacent_phrase(self) -> None:
        """Sehr große Lücke (> 50 % der Phrase) → kein Absturz, Fallback aktiv."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        # Lücke = halbe Datei
        gs = int(10.0 * SR)
        ge = int(20.0 * SR)
        ctx = extractor.extract_context(audio, SR, gs, ge)
        assert isinstance(ctx, PhraseContext)


# ---------------------------------------------------------------------------
# Testklasse: Singleton & condition_inpainting
# ---------------------------------------------------------------------------
class TestPhraseContextSingleton:
    def test_20_singleton_same_object(self) -> None:
        """get_phrase_extractor() gibt stets dasselbe Objekt zurück."""
        a = get_phrase_extractor()
        b = get_phrase_extractor()
        assert a is b

    def test_21_singleton_thread_safe(self) -> None:
        """Parallele Zugriffe liefern dasselbe Singleton-Objekt."""
        instances: list[MusicalPhraseContextExtractor] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                inst = get_phrase_extractor()
                with lock:
                    instances.append(inst)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-Fehler: {errors}"
        assert all(inst is instances[0] for inst in instances)

    def test_22_condition_inpainting_returns_array(self) -> None:
        """condition_inpainting gibt ein np.ndarray zurück."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        gap_audio = audio[gs:ge]
        result = extractor.condition_inpainting(gap_audio, ctx)
        assert isinstance(result, np.ndarray)

    def test_23_condition_inpainting_no_nan(self) -> None:
        """Ausgabe von condition_inpainting ist NaN- und Inf-frei."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        gap_audio = audio[gs:ge]
        result = extractor.condition_inpainting(gap_audio, ctx)
        assert np.all(np.isfinite(result))

    def test_24_condition_inpainting_correct_length(self) -> None:
        """condition_inpainting-Ausgabe hat dieselbe Länge wie der Gap-Input."""
        extractor = get_phrase_extractor()
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx = extractor.extract_context(audio, SR, gs, ge)
        gap_audio = audio[gs:ge]
        result = extractor.condition_inpainting(gap_audio, ctx)
        assert len(result) == len(gap_audio)


# ---------------------------------------------------------------------------
# Testklasse: PhraseContext und PhraseBoundary Datenklassen
# ---------------------------------------------------------------------------
class TestDataclasses:
    def test_25_phrase_context_as_dict(self) -> None:
        """PhraseContext.as_dict() liefert das erwartete Dictionary."""
        ctx = PhraseContext(
            audio_context=np.zeros(100, dtype=np.float32),
            chroma_mean=np.zeros(12, dtype=np.float32),
            tempo_bpm=120.0,
            beat_positions=[0, 1000, 2000],
            phrase_start_s=0.0,
            phrase_end_s=5.0,
            gap_start_s=2.0,
            gap_end_s=2.5,
            is_fallback=False,
        )
        d = ctx.as_dict()
        assert "phrase_start_s" in d
        assert "phrase_end_s" in d
        assert "tempo_bpm" in d
        assert d["is_fallback"] is False

    def test_26_phrase_boundary_fields(self) -> None:
        """PhraseBoundary hat die Felder sample_pos, cause und strength."""
        pb = PhraseBoundary(sample_pos=48000, cause="harmonic", strength=0.75)
        assert pb.sample_pos == 48000
        assert pb.cause == "harmonic"
        assert 0.0 <= pb.strength <= 1.0

    def test_27_phrase_context_is_fallback_default_false(self) -> None:
        """PhraseContext.is_fallback hat Default False."""
        ctx = PhraseContext(
            audio_context=np.zeros(100, dtype=np.float32),
            chroma_mean=np.zeros(12, dtype=np.float32),
            tempo_bpm=100.0,
            beat_positions=[],
            phrase_start_s=0.0,
            phrase_end_s=1.0,
            gap_start_s=0.4,
            gap_end_s=0.6,
        )
        assert ctx.is_fallback is False

    def test_28_repeated_calls_consistent(self) -> None:
        """Zwei Aufrufe mit identischer Eingabe sind numerisch konsistent."""
        extractor = get_phrase_extractor()
        np.random.seed(42)
        audio = _sine(duration_s=30.0)
        gs, ge = _make_gap(len(audio))
        ctx_a = extractor.extract_context(audio, SR, gs, ge)
        ctx_b = extractor.extract_context(audio, SR, gs, ge)
        np.testing.assert_array_equal(ctx_a.audio_context, ctx_b.audio_context)
        np.testing.assert_array_equal(ctx_a.chroma_mean, ctx_b.chroma_mean)
        assert ctx_a.tempo_bpm == ctx_b.tempo_bpm
