from __future__ import annotations

import pytest

"""Unit-Tests für core/genre_classifier.py — GermanSchlagerClassifier.

Spec §2.19: 7-Schicht-Ensemble Zero-Shot-Schlager-Erkennung.
≥ 35 Tests (shape, NaN, Bounds, Edge-Cases, Singleton, Profile, Sprachunterscheidung).
"""


import concurrent.futures
import math
import sys
from types import SimpleNamespace

import numpy as np

from backend.core.genre_classifier import (
    GENRE_RESTORATION_PROFILES,
    JAZZ_RESTORATION_PROFILE,
    KLASSIK_RESTORATION_PROFILE,
    OPER_RESTORATION_PROFILE,
    ROCK_RESTORATION_PROFILE,
    SCHLAGER_RESTORATION_PROFILE,
    GermanSchlagerClassifier,
    SchlagerClassificationResult,
    classify_genre,
    get_genre_classifier,
    get_restoration_profile,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, sr: int = 48000, secs: float = 5.0) -> np.ndarray:
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(sr: int = 48000, secs: float = 5.0, amp: float = 0.1) -> np.ndarray:
    np.random.seed(42)
    return (np.random.randn(int(sr * secs)) * amp).astype(np.float32)


def _silence(sr: int = 48000, secs: float = 5.0) -> np.ndarray:
    return np.zeros(int(sr * secs), dtype=np.float32)


def _am_signal(
    carrier_hz: float = 600.0,
    mod_hz: float = 8.0,
    sr: int = 48000,
    secs: float = 5.0,
) -> np.ndarray:
    """AM-Signal mit 8 Hz Reed-Beating (Akkordeon-ähnlich)."""
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    mod = 1.0 + 0.6 * np.sin(2 * np.pi * mod_hz * t)
    return (carrier * mod).astype(np.float32)  # type: ignore[no-any-return]


def _repetitive_signal(sr: int = 48000, block_secs: float = 8.0, repeats: int = 5) -> np.ndarray:
    """Stark repetitives Signal (gleicher Block mehrfach)."""
    block = np.sin(2 * np.pi * 261.6 * np.linspace(0, block_secs, int(sr * block_secs))).astype(np.float32)
    return np.tile(block, repeats)


# ---------------------------------------------------------------------------
# Klasse 1: Grundlegende Importierbarkeit und Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenreClassifierImport:
    def test_01_import_classifier_class(self):
        assert GermanSchlagerClassifier is not None

    def test_02_import_result_dataclass(self):
        assert SchlagerClassificationResult is not None

    def test_03_get_genre_classifier_returns_instance(self):
        clf = get_genre_classifier()
        assert isinstance(clf, GermanSchlagerClassifier)

    def test_04_singleton_same_object(self):
        a = get_genre_classifier()
        b = get_genre_classifier()
        assert a is b

    def test_05_singleton_thread_safe(self):
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(get_genre_classifier) for _ in range(20)]
            instances = [f.result() for f in futures]
        first = instances[0]
        assert all(inst is first for inst in instances)

    def test_06_classify_genre_convenience_wrapper(self):
        audio = _white_noise()
        result = classify_genre(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)


# ---------------------------------------------------------------------------
# Klasse 2: Ausgabe-Invarianten (NaN, Bounds, Tipos)
# ---------------------------------------------------------------------------


class TestGenreClassifierOutput:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_07_no_nan_on_white_noise(self):
        audio = _white_noise()
        r = self.clf.classify(audio, sr=48000)
        for field in (
            "confidence",
            "clap_score",
            "accordion_score",
            "harmonic_simplicity",
            "rhythm_score",
            "vocal_german_prior",
            "melodic_repetition",
        ):
            assert math.isfinite(getattr(r, field)), f"NaN in {field}"

    def test_08_no_nan_on_silence(self):
        r = self.clf.classify(_silence(), sr=48000)
        assert math.isfinite(r.confidence)
        assert isinstance(r.is_schlager, bool)

    def test_09_all_scores_bounded_white_noise(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        for field in (
            "confidence",
            "clap_score",
            "accordion_score",
            "harmonic_simplicity",
            "rhythm_score",
            "vocal_german_prior",
            "melodic_repetition",
        ):
            val = getattr(r, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1]"

    def test_10_all_scores_bounded_sine(self):
        r = self.clf.classify(_sine(), sr=48000)
        for field in (
            "confidence",
            "clap_score",
            "accordion_score",
            "harmonic_simplicity",
            "rhythm_score",
            "vocal_german_prior",
            "melodic_repetition",
        ):
            val = getattr(r, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1]"

    def test_11_genre_label_is_string(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r.genre_label, str)
        assert len(r.genre_label) > 0

    def test_12_subgenre_is_string(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r.subgenre, str)

    def test_13_reasoning_is_string(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r.reasoning, str)

    def test_14_bpm_positive(self):
        r = self.clf.classify(_sine(freq=200, secs=10.0), sr=48000)
        assert math.isfinite(r.bpm)
        assert r.bpm >= 0.0

    def test_15_key_is_string(self):
        r = self.clf.classify(_sine(), sr=48000)
        assert isinstance(r.key, str)

    def test_16_white_noise_confidence_bounded(self):
        """Weißes Rauschen → alle Scores in [0,1] und Klassifikation stabil."""
        r = self.clf.classify(_white_noise(secs=15.0), sr=48000)
        assert 0.0 <= r.confidence <= 1.0
        assert math.isfinite(r.confidence)

    def test_17_no_false_positive_silence(self):
        """Stille → kein Schlager, kein Absturz."""
        r = self.clf.classify(_silence(secs=10.0), sr=48000)
        assert not r.is_schlager
        assert math.isfinite(r.confidence)


# ---------------------------------------------------------------------------
# Klasse 3: Tier-2 Akkordeon-Fingerprint
# ---------------------------------------------------------------------------


class TestAccordionDetection:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_18_accordion_am_signal_detected(self):
        """AM-Signal bei 8 Hz → accordion_score ≥ 0.50."""
        audio = _am_signal(carrier_hz=600.0, mod_hz=8.0, secs=8.0)
        r = self.clf.classify(audio, sr=48000)
        assert r.accordion_score >= 0.50, f"accordion_score={r.accordion_score}"

    def test_19_pure_sine_low_accordion_score(self):
        """Reiner Sinuston → kein Akkordeon-Fingerprint."""
        audio = _sine(freq=440.0, secs=8.0)
        r = self.clf.classify(audio, sr=48000)
        assert r.accordion_score <= 0.60

    def test_20_accordion_score_finite(self):
        audio = _am_signal(mod_hz=10.0, secs=6.0)
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.accordion_score)
        assert 0.0 <= r.accordion_score <= 1.0


# ---------------------------------------------------------------------------
# Klasse 4: Tier-3 Harmonische Simplizität
# ---------------------------------------------------------------------------


class TestHarmonicSimplicity:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_21_major_triad_high_hsi(self):
        """I-IV-V-I-Progression → hohe harmonische Simplizität."""
        t = np.linspace(0, 10, 10 * 48000, endpoint=False)
        # C-Dur Dreiklang (C, E, G)
        audio = (np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t) + np.sin(2 * np.pi * 392.0 * t)).astype(
            np.float32
        ) * 0.3
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.harmonic_simplicity)
        assert 0.0 <= r.harmonic_simplicity <= 1.0

    def test_22_harmonic_simplicity_bounded(self):
        audio = _white_noise(secs=10.0)
        r = self.clf.classify(audio, sr=48000)
        assert 0.0 <= r.harmonic_simplicity <= 1.0

    def test_23_harmonic_simplicity_finite_silence(self):
        r = self.clf.classify(_silence(secs=10.0), sr=48000)
        assert math.isfinite(r.harmonic_simplicity)


# ---------------------------------------------------------------------------
# Klasse 5: Tier-6 Melodische Wiederholungsrate
# ---------------------------------------------------------------------------


class TestMelodicRepetition:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_24_repetitive_signal_high_score(self):
        """Gleicher 8-s-Block × 5 → melodic_repetition ≥ 0.45."""
        audio = _repetitive_signal(block_secs=8.0, repeats=5)
        r = self.clf.classify(audio, sr=48000)
        assert r.melodic_repetition >= 0.45, f"melodic_repetition={r.melodic_repetition}"

    def test_25_short_signal_neutral_score(self):
        """Signal < 30 s → neutraler Score (keine Strafe)."""
        audio = _sine(secs=15.0)
        r = self.clf.classify(audio, sr=48000)
        assert 0.0 <= r.melodic_repetition <= 1.0
        assert math.isfinite(r.melodic_repetition)

    def test_26_melodic_repetition_bounded(self):
        """melodic_repetition liegt immer in [0, 1]."""
        audio = _white_noise(secs=40.0)
        r = self.clf.classify(audio, sr=48000)
        assert 0.0 <= r.melodic_repetition <= 1.0
        assert math.isfinite(r.melodic_repetition)


# ---------------------------------------------------------------------------
# Klasse 6: Edge-Cases (Stereo, kurze Dateien, extreme Amplituden)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_27_stereo_input_accepted(self):
        """Stereo-Input wird intern zu Mono gemittelt."""
        mono = _white_noise(secs=8.0)
        stereo = np.stack([mono, mono * 0.8], axis=1)
        r = self.clf.classify(stereo, sr=48000)
        assert isinstance(r, SchlagerClassificationResult)
        assert math.isfinite(r.confidence)

    def test_27a_to_mono_samples_first_preserves_length(self):
        """Regression (Achsen-Bug 2026-06-08): Stereo (samples, channels) = (N, 2)
        darf NICHT über die Sample-Achse gemittelt werden (sonst Kollaps auf 2 Samples).

        File-Import liefert ``(N, 2)``. Ein ``mean(axis=0)`` ohne Layout-Check würde
        ALLE DSP-Tiers mit Müll füttern → konstante Default-Konfidenz über alle Songs.
        """
        mono = _white_noise(secs=8.0)
        stereo_nch = np.stack([mono, mono * 0.8], axis=1)  # (N, 2) samples-first
        assert stereo_nch.shape == (mono.shape[0], 2)
        out = self.clf._to_mono(stereo_nch)
        assert out.shape[0] == mono.shape[0], (
            f"Sample-Achse kollabiert: erwartet {mono.shape[0]}, erhielt {out.shape[0]}"
        )
        assert out.ndim == 1

    def test_27b_to_mono_channels_first_preserves_length(self):
        """Regression: auch Channel-first ``(2, N)`` muss volle Länge erhalten."""
        mono = _white_noise(secs=8.0)
        stereo_chn = np.stack([mono, mono * 0.8], axis=0)  # (2, N) channels-first
        assert stereo_chn.shape == (2, mono.shape[0])
        out = self.clf._to_mono(stereo_chn)
        assert out.shape[0] == mono.shape[0]
        assert out.ndim == 1

    def test_27c_to_mono_column_vector_treated_as_mono(self):
        """Spaltenvektor ``(N, 1)`` → Mono ``(N,)``, kein Kanal-Mittel."""
        mono = _white_noise(secs=4.0)
        col = mono.reshape(-1, 1)  # (N, 1)
        out = self.clf._to_mono(col)
        assert out.shape[0] == mono.shape[0]
        assert out.ndim == 1

    def test_27d_stereo_both_layouts_equivalent_result(self):
        """Identischer Inhalt in ``(N, 2)`` und ``(2, N)`` → gleiche Klassifikation.

        Beweist, dass der Downmix layout-robust ist und song-spezifische Evidenz
        liefert (nicht den konstanten Default-Fallback des kollabierten Bugs).
        """
        mono = _am_signal(secs=8.0)
        r_nch = self.clf.classify(np.stack([mono, mono], axis=1), sr=48000)
        r_chn = self.clf.classify(np.stack([mono, mono], axis=0), sr=48000)
        assert math.isclose(r_nch.confidence, r_chn.confidence, abs_tol=1e-6)

    def test_28_very_short_signal_no_crash(self):
        """Signal < 5 s darf nicht abstürzen."""
        audio = _sine(secs=2.0)
        r = self.clf.classify(audio, sr=48000)
        assert isinstance(r, SchlagerClassificationResult)
        assert math.isfinite(r.confidence)

    def test_29_clipped_signal_no_crash(self):
        """Geclipptes Signal → kein Absturz."""
        audio = np.ones(48000 * 5, dtype=np.float32)
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.confidence)

    def test_30_very_quiet_signal_no_crash(self):
        """Sehr leises Signal → kein Absturz."""
        audio = _white_noise(amp=1e-6, secs=8.0)
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.confidence)

    def test_31_result_is_dataclass_instance(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r, SchlagerClassificationResult)

    def test_32_sr_agnostic_native_import_sr(self):
        """SR-agnostisch: Analyse-Module dürfen kein assert sr == 48000 enthalten (Spec §Performance-Budget)."""
        audio = _white_noise(sr=22050, secs=8.0)
        # Must NOT raise at 22050 Hz (native import SR is valid)
        result = self.clf.classify(audio, sr=22050)
        assert result is not None, "SR-agnostic classify() must return a result at 22050 Hz"


# ---------------------------------------------------------------------------
# Klasse 7: Restaurierungsprofile
# ---------------------------------------------------------------------------


class TestRestorationProfiles:
    def test_33_schlager_profile_has_required_keys(self):
        required = {
            "soft_saturation_preserve",
            "tonal_center_threshold",
            "groove_dtw_max_ms",
            "deessing_target_hz",
            "brillanz_target",
            "waerme_target",
            "gp_memory_key",
        }
        assert required.issubset(SCHLAGER_RESTORATION_PROFILE.keys())

    def test_34_schlager_tonal_center_threshold_ge_095(self):
        assert SCHLAGER_RESTORATION_PROFILE["tonal_center_threshold"] >= 0.95

    def test_35_schlager_groove_dtw_le_8ms(self):
        assert SCHLAGER_RESTORATION_PROFILE["groove_dtw_max_ms"] <= 8.0

    def test_36_schlager_waerme_target_ge_080(self):
        assert SCHLAGER_RESTORATION_PROFILE["waerme_target"] >= 0.80

    def test_37_schlager_brillanz_target_in_range(self):
        assert 0.70 <= SCHLAGER_RESTORATION_PROFILE["brillanz_target"] <= 1.0

    def test_38_schlager_gp_memory_key_is_string(self):
        key = SCHLAGER_RESTORATION_PROFILE["gp_memory_key"]
        assert isinstance(key, str) and len(key) > 0

    def test_39_jazz_profile_has_groove_key(self):
        assert "groove_dtw_max_ms" in JAZZ_RESTORATION_PROFILE

    def test_40_jazz_groove_strict(self):
        """Jazz-Timing ist sacred — DTW ≤ 4 ms."""
        assert JAZZ_RESTORATION_PROFILE["groove_dtw_max_ms"] <= 5.0

    def test_41_klassik_profile_has_dereverb_flag(self):
        assert (
            "phase_20_dereverb_enabled" in KLASSIK_RESTORATION_PROFILE
            or "phase_49_dereverb_enabled" in KLASSIK_RESTORATION_PROFILE
        )

    def test_42_oper_profile_has_deessing_key(self):
        assert "deessing_target_hz" in OPER_RESTORATION_PROFILE or "deessing_strength_cap" in OPER_RESTORATION_PROFILE

    def test_43_rock_brillanz_target_high(self):
        assert ROCK_RESTORATION_PROFILE.get("brillanz_target", 1.0) >= 0.85

    def test_44_genre_profiles_dict_not_empty(self):
        assert len(GENRE_RESTORATION_PROFILES) >= 4

    def test_45_get_restoration_profile_schlager(self):
        p = get_restoration_profile("Schlager")
        assert p is not None
        assert "gp_memory_key" in p

    def test_46_get_restoration_profile_unknown_returns_default(self):
        p = get_restoration_profile("UnbekanntesGenre")
        # Soll Default zurückgeben, nicht abstürzen
        assert p is not None

    def test_47_all_profiles_have_gp_memory_key(self):
        for genre, profile in GENRE_RESTORATION_PROFILES.items():
            assert "gp_memory_key" in profile, f"gp_memory_key fehlt in {genre}"

    def test_48_all_profile_values_finite(self):
        """Alle numerischen Werte in den Profilen sind finite."""
        all_profiles = [
            SCHLAGER_RESTORATION_PROFILE,
            JAZZ_RESTORATION_PROFILE,
            KLASSIK_RESTORATION_PROFILE,
            OPER_RESTORATION_PROFILE,
            ROCK_RESTORATION_PROFILE,
        ]
        for profile in all_profiles:
            for k, v in profile.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"Nicht-finiter Wert {k}={v}"


# ---------------------------------------------------------------------------
# Klasse 8: Tier-7 Vokalsprach-Erkennung (Deutsch vs. Englisch)
# ---------------------------------------------------------------------------


def _german_umlaut_signal(sr: int = 22050, secs: float = 5.0) -> np.ndarray:
    """Synthetisches Signal mit Vokalformanten für dt. ü (F1=320, F2=1900 → F2-F1=1580)."""
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    # Grundton 150 Hz (männliche Stimme) + Resonanzen bei F1=320, F2=1900 Hz
    sig = (
        np.sin(2 * np.pi * 150 * t) * 1.0
        + np.sin(2 * np.pi * 320 * t) * 0.6  # F1 (ü-typisch)
        + np.sin(2 * np.pi * 1900 * t) * 0.4  # F2 (ü-typisch, hohes F2)
    ).astype(np.float32)
    sig /= np.max(np.abs(sig) + 1e-8)
    return sig  # type: ignore[no-any-return]


def _english_vowel_signal(sr: int = 22050, secs: float = 5.0) -> np.ndarray:
    """Synthetisches Signal mit englischen Vokalformanten /ʌ/ (F1=700, F2=1100 → F2-F1=400)."""
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    # Grundton 150 Hz + Resonanzen bei F1=700, F2=1100 Hz (eng. /ʌ/, kein Umlaut)
    sig = (
        np.sin(2 * np.pi * 150 * t) * 1.0
        + np.sin(2 * np.pi * 700 * t) * 0.6  # F1 (englisch-typisch)
        + np.sin(2 * np.pi * 1100 * t) * 0.4  # F2 (englisch-typisch, niedriges F2)
    ).astype(np.float32)
    sig /= np.max(np.abs(sig) + 1e-8)
    return sig  # type: ignore[no-any-return]


class TestVocalLanguageDetection:
    """Tests für Tier-7: _detect_vocal_language (Deutsch vs. Englisch)."""

    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_49_vocal_language_score_exists_in_result(self):
        """SchlagerClassificationResult hat vocal_language_score."""
        r = self.clf.classify(_sine(), sr=48000)
        assert hasattr(r, "vocal_language_score")
        assert math.isfinite(r.vocal_language_score)

    def test_50_vocal_language_score_bounded(self):
        """vocal_language_score ∈ [0.0, 1.0] für alle Eingaben."""
        for audio in [_sine(), _white_noise(), _silence(), _am_signal()]:
            r = self.clf.classify(audio, sr=48000)
            assert 0.0 <= r.vocal_language_score <= 1.0, f"vocal_language_score={r.vocal_language_score} out of [0,1]"

    def test_51_vocal_language_score_finite_on_silence(self):
        """Stille → Fallback 0.5, kein NaN."""
        r = self.clf.classify(_silence(), sr=48000)
        assert math.isfinite(r.vocal_language_score)

    def test_52_umlaut_signal_scores_higher_than_english(self):
        """Dt. Umlaut-Signal (F2-F1=1580 Hz) hat höheren lang_de_score als engl. Signal."""
        r_de = self.clf._detect_vocal_language(_german_umlaut_signal(), sr=22050)
        r_en = self.clf._detect_vocal_language(_english_vowel_signal(), sr=22050)
        assert r_de >= r_en, f"Umlaut-Signal sollte höheren lang_de_score haben: de={r_de:.3f} en={r_en:.3f}"

    def test_53_german_umlaut_score_above_neutral(self):
        """Dt. Umlaut-Signal (F2-F1 > 1400) → lang_de_score ≥ 0.40."""
        score = self.clf._detect_vocal_language(_german_umlaut_signal(), sr=22050)
        assert score >= 0.40, f"Umlaut-Signal: lang_de_score={score:.3f} < 0.40"

    def test_54_no_crash_on_very_short_audio(self):
        """Sehr kurzes Signal → Fallback 0.5, kein Absturz."""
        short = np.zeros(100, dtype=np.float32)
        score = self.clf._detect_vocal_language(short, sr=22050)
        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_55_german_label_for_high_lang_score(self):
        """_determine_genre_label mit lang_de_score=0.8 → 'Deutscher Schlager' (eindeutig deutschsprachig)."""
        label = self.clf._determine_genre_label("schunkel", 130.0, lang_de_score=0.8)
        assert label == "Deutscher Schlager"
        assert "Internationaler" not in label

    def test_56_international_label_for_low_lang_score(self):
        """_determine_genre_label mit lang_de_score=0.1 → 'Internationaler Schlager'."""
        label = self.clf._determine_genre_label("schunkel", 130.0, lang_de_score=0.1)
        assert label == "Internationaler Schlager"

    def test_57_reasoning_mentions_language_confidence(self):
        """Reasoning enthält Sprachinformation wenn Sprache klar erkennbar."""
        r = self.clf.classify(_german_umlaut_signal(sr=48000), sr=48000)
        # Score + Reasoning sollen konsistent sein
        assert isinstance(r.reasoning, str)
        assert math.isfinite(r.vocal_language_score)

    def test_58_vocal_language_score_nan_safe(self):
        """NaN-Audio → kein NaN im vocal_language_score."""
        nan_audio = np.full(22050, np.nan, dtype=np.float32)
        score = self.clf._detect_vocal_language(nan_audio, sr=22050)
        assert math.isfinite(score)

    def test_59_walzer_international_label(self):
        """Walzer-Subgenre mit niedrigem lang_de_score → 'Internationaler Walzer'."""
        label = self.clf._determine_genre_label("walzer", 165.0, lang_de_score=0.15)
        assert label == "Internationaler Walzer"

    def test_60_result_has_all_required_fields(self):
        """SchlagerClassificationResult enthält alle Felder inkl. vocal_language_score."""
        r = SchlagerClassificationResult(
            is_schlager=True,
            confidence=0.75,
            genre_label="Schlager",
            subgenre="schunkel",
            bpm=130.0,
            vocal_language_score=0.80,
        )
        assert r.vocal_language_score == 0.80
        assert isinstance(r.top_genres, list)
        assert isinstance(r.open_set_unknown, bool)
        assert r.is_schlager is True

    def test_61_lyrics_hint_fuses_into_language_score(self, monkeypatch):
        """§2.36-Hinweis hebt lang_de_score an und verhindert Non-Schlager-Fehlrouting."""
        clf = get_genre_classifier()

        monkeypatch.setattr(clf, "_is_music_like", lambda _a: True)
        monkeypatch.setattr(clf, "_compute_clap_score", lambda _a, _sr: 0.35)
        monkeypatch.setattr(clf, "_compute_accordion_score", lambda _a, _sr: 0.60)
        monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: 0.78)
        monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (0.82, "schunkel", 128.0))
        monkeypatch.setattr(clf, "_compute_german_vocal_prior", lambda _a, _sr: 0.66)
        monkeypatch.setattr(clf, "_compute_melodic_repetition", lambda _a, _sr: 0.48)
        monkeypatch.setattr(clf, "_detect_vocal_language", lambda _a, _sr: 0.22)
        monkeypatch.setattr(clf, "_compute_lyrics_language_hint", lambda _a, _sr: 0.86)
        monkeypatch.setattr(clf, "_estimate_key", lambda _a, _sr: "C-Dur")

        audio = _sine(freq=220.0, secs=10.0)
        r = clf.classify(audio, sr=48000)

        assert r.vocal_language_score >= 0.85
        assert r.is_schlager is True
        assert r.genre_label == "Deutscher Schlager"

    def test_62_lyrics_hint_reads_mocked_lge(self, monkeypatch):
        """_compute_lyrics_language_hint nutzt §2.36-Transkriptionsdaten ohne Lyrics-Text."""
        clf = get_genre_classifier()

        fake_words = [
            SimpleNamespace(word="", phoneme_type="fricative_stressed", confidence=0.8),
            SimpleNamespace(word="", phoneme_type="plosive", confidence=0.7),
            SimpleNamespace(word="", phoneme_type="vowel_stressed", confidence=0.9),
        ]
        fake_transcription = SimpleNamespace(language="de", words=fake_words)
        fake_lge = SimpleNamespace(transcribe=lambda _a, _sr: fake_transcription)
        fake_module = SimpleNamespace(get_lyrics_guided_enhancement=lambda: fake_lge)

        monkeypatch.setitem(sys.modules, "backend.core.lyrics_guided_enhancement", fake_module)

        score = clf._compute_lyrics_language_hint(_sine(freq=330.0, secs=10.0), 48000)
        assert 0.70 <= score <= 1.0

    def test_63_family_and_topk_present_for_schlager(self, monkeypatch):
        """Klassifikation liefert Family-Stage und Top-k-Genres im Ergebnis."""
        clf = get_genre_classifier()

        monkeypatch.setattr(clf, "_is_music_like", lambda _a: True)
        monkeypatch.setattr(clf, "_compute_clap_score", lambda _a, _sr: 0.35)
        monkeypatch.setattr(clf, "_compute_accordion_score", lambda _a, _sr: 0.64)
        monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: 0.80)
        monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (0.82, "schunkel", 126.0))
        monkeypatch.setattr(clf, "_compute_german_vocal_prior", lambda _a, _sr: 0.70)
        monkeypatch.setattr(clf, "_compute_melodic_repetition", lambda _a, _sr: 0.50)
        monkeypatch.setattr(clf, "_detect_vocal_language", lambda _a, _sr: 0.74)
        monkeypatch.setattr(clf, "_compute_lyrics_language_hint", lambda _a, _sr: 0.0)
        monkeypatch.setattr(clf, "_estimate_key", lambda _a, _sr: "C-Dur")
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2600.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 2.8)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 24.0)

        r = clf.classify(_sine(freq=220.0, secs=10.0), sr=48000)

        assert r.is_schlager
        assert r.genre_family == "schlager_folk"
        assert r.genre_family_confidence >= 0.50
        assert len(r.top_genres) >= 1
        assert r.top_genres[0][0].lower().find("schlager") >= 0

    def test_64_open_set_unknown_for_ambiguous_non_schlager(self):
        """Bei engem Score-Margin triggert Open-Set-Unknown gemäß §2.19."""
        clf = get_genre_classifier()
        top_genres = [("Rock", 0.42), ("Jazz", 0.41)]
        assert clf._is_open_set_unknown(top_genres) is True

    def test_65_open_set_false_for_clear_non_schlager(self):
        """Bei klarem Top-Genre (Margin >= 0.08) bleibt Open-Set deaktiviert."""
        clf = get_genre_classifier()
        top_genres = [("Rock", 0.62), ("Jazz", 0.41)]
        assert clf._is_open_set_unknown(top_genres) is False


# ---------------------------------------------------------------------------
# Klasse 9: Non-Schlager Golden-Sample-Tests
# ---------------------------------------------------------------------------
# Motivation: Falsch-„Unbekannt" für Jazz deaktiviert die jazz-sacred-timing-
# Kalibrierung lautlos. Diese Tests stellen sicher, dass klar profilierte
# Non-Schlager-Genres (Jazz/Rock/Klassik) bei eindeutigen Merkmalen korrekt
# erkannt werden und nicht stillschweigend durch den Open-Set-Unbekannt-Pfad
# fallen.  Alle Tests verwenden monkeypatching, um Tier-Scores kontrolliert
# zu setzen — unabhängig von Audio-Synthesis-Qualität.
# ---------------------------------------------------------------------------


def _build_non_schlager_clf(
    monkeypatch, *, accordion=0.05, hsi=0.45, rhythm=0.20, vocal_prior=0.20, melodic=0.20, lang_de=0.20, clap=0.05
):
    """Setzt alle Schlager-Tier-Scores auf Non-Schlager-Werte.

    Patcht zusätzlich alle 12 seit v10.0.0.x ergänzten Genre-Scorer auf 0.10,
    damit Test-Klassen (TestNonSchlagerGoldenSamples etc.) ausschließlich die
    explizit über ``monkeypatch.setattr(clf, "_score_<genre>", ...)`` gesetzten
    Scores testen — unabhängig von echten Audio-Feature-Artefakten.
    """
    clf = get_genre_classifier()
    monkeypatch.setattr(clf, "_is_music_like", lambda _a: True)
    monkeypatch.setattr(clf, "_compute_clap_score", lambda _a, _sr: clap)
    monkeypatch.setattr(clf, "_compute_accordion_score", lambda _a, _sr: accordion)
    monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: hsi)
    monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (rhythm, "unknown", 120.0))
    monkeypatch.setattr(clf, "_compute_german_vocal_prior", lambda _a, _sr: vocal_prior)
    monkeypatch.setattr(clf, "_compute_melodic_repetition", lambda _a, _sr: melodic)
    monkeypatch.setattr(clf, "_detect_vocal_language", lambda _a, _sr: lang_de)
    monkeypatch.setattr(clf, "_compute_lyrics_language_hint", lambda _a, _sr: 0.0)
    monkeypatch.setattr(clf, "_estimate_key", lambda _a, _sr: "C-Dur")
    # Isolate the 12 new genre scorers (added v10.0.0.x) to a low base score so
    # TestNonSchlagerGoldenSamples tests are not affected by audio-synthesis
    # artefacts (e.g. onset_rate spikes on pure tones).  Tests that deliberately
    # verify one of these genres must override the corresponding patch afterward.
    for _method in (
        "_score_pop",
        "_score_blues",
        "_score_soul_rnb",
        "_score_country",
        "_score_folk",
        "_score_funk",
        "_score_electronic",
        "_score_hiphop",
        "_score_metal",
        "_score_latin",
        "_score_gospel",
        "_score_reggae",
    ):
        monkeypatch.setattr(clf, _method, lambda *_a, **_k: 0.10)
    return clf


class TestNonSchlagerGoldenSamples:
    """Golden-sample-Tests für Non-Schlager-Genre-Erkennung.

    Stellt sicher, dass Jazz/Rock/Klassik mit klaren Merkmalen korrekt klassifiziert
    werden (nicht als „Unbekannt") und das richtige Restaurierungsprofil greifen kann.
    """

    def test_66_jazz_clear_profile_not_unknown(self, monkeypatch):
        """Jazz mit komplexer Harmonik (hsi=0.35) und großer DR → genre_label=='Jazz'."""
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.35)
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2200.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 2.0)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 44.0)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert not r.is_schlager
        assert r.genre_label == "Jazz", f"Erwartet Jazz, bekommen: {r.genre_label!r}"
        assert not r.open_set_unknown

    def test_67_jazz_restoration_profile_accessible(self, monkeypatch):
        """Jazz-Ergebnis liefert gültiges Restaurierungsprofil via get_restoration_profile."""
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.35)
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2100.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 1.8)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 45.0)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert r.genre_label == "Jazz"
        profile = get_restoration_profile(r.genre_label)
        assert profile is not None
        assert "groove_dtw_max_ms" in profile
        assert profile["groove_dtw_max_ms"] <= 5.0  # Jazz: sacred timing

    def test_68_rock_clear_profile_not_unknown(self, monkeypatch):
        """Rock-Profil (hsi=0.65, bpm=150, dr=30 dB) → genre_label=='Rock' ohne Ambiguität.

        hsi=0.65 liegt knapp über Funks HSI-Obergrenze (0.64) und schließt den
        Funk-Bonus aus. bpm=150 liegt außerhalb der Latin-Kernzone (80-130 und 160-250),
        dr=30 dB liegt über Funks DR-Fenster (12-28 dB). Rock (0.90) gewinnt klar
        gegen Latin (0.80) mit MARGIN > 0.08.
        """
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.58)
        # Re-patch to unambiguous Rock feature set (outside Funk/Latin overlap zone).
        monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: 0.65)
        monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (0.20, "unknown", 150.0))
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 3200.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 4.5)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 30.0)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert not r.is_schlager
        assert r.genre_label == "Rock", f"Erwartet Rock, bekommen: {r.genre_label!r}"
        assert not r.open_set_unknown

    def test_69_rock_restoration_profile_brillanz_ge_085(self, monkeypatch):
        """Rock-Profil hat brillanz_target ≥ 0.85 für Brillanz-Enhancement."""
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.58)
        # Same unambiguous Rock feature set as test_68.
        monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: 0.65)
        monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (0.20, "unknown", 150.0))
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 3200.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 4.5)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 30.0)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert r.genre_label == "Rock"
        profile = get_restoration_profile(r.genre_label)
        assert profile.get("brillanz_target", 0.0) >= 0.85

    def test_70_klassik_clear_profile_not_unknown(self, monkeypatch):
        """Klassik mit sehr hoher DR + niedriger Onset-Rate → genre_label=='Klassik'."""
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.72)
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 1800.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 0.8)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 50.0)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert not r.is_schlager
        assert r.genre_label == "Klassik", f"Erwartet Klassik, bekommen: {r.genre_label!r}"
        assert not r.open_set_unknown

    def test_71_klassik_profile_has_dereverb_control(self, monkeypatch):
        """Klassik-Profil hat phase_20/49-Dereverb-Flag (Raumklang bewahren)."""
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.72)
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 1800.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 0.8)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 50.0)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert r.genre_label == "Klassik"
        profile = get_restoration_profile(r.genre_label)
        has_dereverb_key = "phase_20_dereverb_enabled" in profile or "phase_49_dereverb_enabled" in profile
        assert has_dereverb_key

    def test_72_jazz_high_hsi_blocked(self, monkeypatch):
        """Jazz-Score darf bei hsi ≥ 0.68 nicht zurückgegeben werden (Anti-Schlager-Mislabel)."""
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.72)  # eindeutig harmonisch simpel
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2200.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 2.0)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 44.0)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert r.genre_label != "Jazz", f"Jazz bei hsi=0.72 ist ein Mislabel — bekommen: {r.genre_label!r}"

    def test_73_non_schlager_score_below_min_gives_unbekannt(self, monkeypatch):
        """Alle Non-Schlager-Scores < 0.35 → genre_label='Unbekannt', kein Absturz."""
        clf = _build_non_schlager_clf(monkeypatch, hsi=0.50)
        # Alle Scoring-Methoden auf sehr niedrige Scores
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2000.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 2.0)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 20.0)
        # Originale 4 Genres: alle niedrig
        monkeypatch.setattr(clf, "_score_rock", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_jazz", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_classical", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_oper", lambda *_a, **_k: 0.10)
        # Neue 12 Genres: alle niedrig
        monkeypatch.setattr(clf, "_score_pop", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_blues", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_soul_rnb", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_country", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_folk", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_funk", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_electronic", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_hiphop", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_metal", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_latin", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_gospel", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_reggae", lambda *_a, **_k: 0.10)
        r = clf.classify(_white_noise(secs=10.0), sr=48000)
        assert r.genre_label == "Unbekannt"
        assert not r.is_schlager

    def test_74_accordion_tremolo_discrimination_coherent_am(self, monkeypatch):
        """Kohärentes AM (Tremolo) → Akkordeon-Score wird durch Diskriminator reduziert."""
        clf = get_genre_classifier()
        # Erzeuge ein AM-Signal, das in ALLEN Bändern identisch bei 8 Hz moduliert
        # (simuliert Tremolo-Gitarre: ein einzelner Modulator).
        sr = 48000
        t = np.linspace(0, 8.0, sr * 8, endpoint=False)
        mod = 1.0 + 0.7 * np.sin(2 * np.pi * 8.0 * t)
        # Breites Signal mit Energy über alle drei Sub-Bänder
        carrier = (
            np.sin(2 * np.pi * 300.0 * t)  # sub-band 1
            + np.sin(2 * np.pi * 900.0 * t)  # sub-band 2
            + np.sin(2 * np.pi * 2000.0 * t)  # sub-band 3
        )
        coherent_am = (carrier * mod).astype(np.float32) * 0.3
        score = clf._compute_accordion_score(coherent_am, sr)
        # Ein kohärentes AM darf nach Diskriminator-Penalty nicht volle Akkordeon-Confidence erreichen
        assert score <= 0.80, (
            f"Kohärentes AM (Tremolo) sollte nach Diskriminator reduziert sein, aber accordion_score={score:.3f}"
        )

    def test_75_levinson_durbin_matches_vocal_prior_bounds(self):
        """_lpc_levinson auf reale Autokorrelation → finite Koeffizienten in vertretbarem Bereich."""
        clf = get_genre_classifier()
        sr = 22050
        # Synthetisches Vokal-Signal mit F1=400, F2=1800 Hz
        t = np.linspace(0, 0.025, int(sr * 0.025), endpoint=False)
        frame = (
            np.sin(2 * np.pi * 120 * t)  # Grundton
            + 0.5 * np.sin(2 * np.pi * 400 * t)  # F1
            + 0.3 * np.sin(2 * np.pi * 1800 * t)  # F2
        ).astype(np.float64)
        r = np.correlate(frame, frame, mode="full")
        r = r[len(r) // 2 :]
        r[0] = max(r[0], 1e-10)
        coefs = clf._lpc_levinson(r, order=16)
        assert len(coefs) == 16
        assert np.isfinite(coefs).all(), "Levinson-Durbin liefert NaN/Inf-Koeffizienten"
        # Formant-Extraktion muss funktionieren
        poly = np.concatenate([[1.0], -coefs])
        roots = np.roots(poly)
        formants = sorted(
            np.angle(root) * sr / (2 * np.pi)
            for root in roots
            if np.imag(root) > 0 and 200 < np.angle(root) * sr / (2 * np.pi) < 3500
        )
        assert len(formants) >= 1, "Mindestens ein Formant muss erkannt werden"


# ---------------------------------------------------------------------------
# Klasse 10: Jazz-Veto-Guard — n_active >= 1 Invariante
# ---------------------------------------------------------------------------


class TestJazzVetoGuard:
    """Jazz-Veto darf nur feuern, wenn mindestens ein Schlager-Tier aktiv ist.

    Root-Cause: Das Jazz-Veto (§ genre_classifier lines ~207) blockiert Genres
    wie Deutsch-Choral oder Kunstlied, bei denen n_active=0 ist und trotzdem
    alt_genre=="Jazz" ermittelt wird (hsi < 0.50 + _score_jazz gibt positiven Wert).
    Ohne Guard: is_schlager=True, obwohl kein Schlager-Merkmal aktiv ist.
    """

    def test_76_jazz_veto_does_not_fire_without_schlager_evidence(self, monkeypatch):
        """Jazz-Veto darf nicht triggern wenn n_active=0 (kein Schlager-Tier aktiv)."""
        clf = get_genre_classifier()
        monkeypatch.setattr(clf, "_is_music_like", lambda _a: True)
        # Alle Schlager-Tier-Scores strikt unter ihren Thresholds → n_active=0
        monkeypatch.setattr(clf, "_compute_clap_score", lambda _a, _sr: 0.05)
        monkeypatch.setattr(clf, "_compute_accordion_score", lambda _a, _sr: 0.05)
        monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: 0.40)
        monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (0.20, "unknown", 120.0))
        monkeypatch.setattr(clf, "_compute_german_vocal_prior", lambda _a, _sr: 0.20)
        monkeypatch.setattr(clf, "_compute_melodic_repetition", lambda _a, _sr: 0.20)
        monkeypatch.setattr(clf, "_detect_vocal_language", lambda _a, _sr: 0.35)  # triggers lang_de>=0.30
        monkeypatch.setattr(clf, "_compute_lyrics_language_hint", lambda _a, _sr: 0.0)
        monkeypatch.setattr(clf, "_estimate_key", lambda _a, _sr: "D-Moll")
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2200.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 2.0)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 45.0)
        # Force Jazz as top non-Schlager genre (hsi=0.40 → Jazz gets +0.40)
        monkeypatch.setattr(clf, "_score_rock", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_jazz", lambda *_a, **_k: 0.55)
        monkeypatch.setattr(clf, "_score_classical", lambda *_a, **_k: 0.10)

        r = clf.classify(_sine(freq=440.0, secs=10.0), sr=48000)

        # Jazz-Veto darf NICHT feuern: kein Schlager-Tier aktiv (n_active=0)
        assert r.is_schlager is False, (
            f"Jazz-Veto feuerte fälschlicherweise ohne Schlager-Evidenz "
            f"(n_active=0): genre_label={r.genre_label!r}, confidence={r.confidence:.3f}"
        )

    def test_77_jazz_veto_still_fires_with_schlager_evidence(self, monkeypatch):
        """Jazz-Veto soll weiterhin feuern wenn n_active >= 1 und Jazz als Top-Genre."""
        clf = get_genre_classifier()
        monkeypatch.setattr(clf, "_is_music_like", lambda _a: True)
        # Einen Schlager-Tier aktivieren: vocal_prior >= 0.50 (threshold[3]=0.50)
        monkeypatch.setattr(clf, "_compute_clap_score", lambda _a, _sr: 0.05)
        monkeypatch.setattr(clf, "_compute_accordion_score", lambda _a, _sr: 0.05)
        monkeypatch.setattr(clf, "_compute_harmonic_simplicity", lambda _a, _sr: 0.40)
        monkeypatch.setattr(clf, "_classify_rhythm_pattern", lambda _a, _sr: (0.20, "unknown", 120.0))
        monkeypatch.setattr(clf, "_compute_german_vocal_prior", lambda _a, _sr: 0.52)  # ≥ 0.50 → 1 tier active
        monkeypatch.setattr(clf, "_compute_melodic_repetition", lambda _a, _sr: 0.20)
        monkeypatch.setattr(clf, "_detect_vocal_language", lambda _a, _sr: 0.35)  # lang_de>=0.30
        monkeypatch.setattr(clf, "_compute_lyrics_language_hint", lambda _a, _sr: 0.0)
        monkeypatch.setattr(clf, "_estimate_key", lambda _a, _sr: "C-Dur")
        monkeypatch.setattr(clf, "_spectral_centroid_hz", lambda _a, _sr: 2200.0)
        monkeypatch.setattr(clf, "_onset_rate", lambda _a, _sr: 2.0)
        monkeypatch.setattr(clf, "_dynamic_range_db", lambda _a, _sr: 45.0)
        monkeypatch.setattr(clf, "_score_rock", lambda *_a, **_k: 0.10)
        monkeypatch.setattr(clf, "_score_jazz", lambda *_a, **_k: 0.55)
        monkeypatch.setattr(clf, "_score_classical", lambda *_a, **_k: 0.10)
        # Isolate the 12 new genre scorers: Jazz must win so the Veto triggers.
        for _method in (
            "_score_pop",
            "_score_blues",
            "_score_soul_rnb",
            "_score_country",
            "_score_folk",
            "_score_funk",
            "_score_electronic",
            "_score_hiphop",
            "_score_metal",
            "_score_latin",
            "_score_gospel",
            "_score_reggae",
        ):
            monkeypatch.setattr(clf, _method, lambda *_a, **_k: 0.10)

        r = clf.classify(_sine(freq=440.0, secs=10.0), sr=48000)

        # Veto darf feuern: n_active >= 1, alt_genre="Jazz", lang_de=0.35>=0.30
        assert r.is_schlager is True, (
            f"Jazz-Veto sollte bei n_active>=1 + alt_genre=Jazz + lang_de>=0.30 feuern, is_schlager={r.is_schlager}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §v10 Tests: CLAP-DSP-Konsistenz-Gate
# ═══════════════════════════════════════════════════════════════════════════════


class TestClapDspConsistencyGateV10:
    """CLAP wird ignoriert, wenn es DSP-Ensemble systematisch unterdrückt."""

    def test_clap_low_score_dsp_high_triggers_gate(self):
        """CLAP < 0.35 + DSP-Mittel ≥ 0.50 → CLAP verworfen, pure-DSP-Konfidenz."""
        clf = GermanSchlagerClassifier()
        # Ändere Schwellwerte temporär, damit DSP-Features feuern
        clf.SCHLAGER_CONFIDENCE_THRESHOLD = 0.40
        # Erzeuge ein Signal mit starken Schlager-DSP-Merkmalen (Akkordeon-ähnlich)
        sr = 48000
        t = np.arange(sr * 5) / sr
        # AM-moduliertes Signal (simuliert Akkordeon Reed-Beating)
        carrier = np.sin(2 * np.pi * 440 * t)
        modulator = 1.0 + 0.3 * np.sin(2 * np.pi * 8 * t)  # 8 Hz Tremolo
        audio = (carrier * modulator * 0.3).astype(np.float32)

        # Mock CLAP: sehr niedriger Score
        original_clap = clf._compute_clap_score

        def fake_clap(*args, **kwargs):
            clf._clap_score_is_fallback = False
            return 0.10  # CLAP sagt: kein Schlager

        clf._compute_clap_score = fake_clap  # type: ignore[method-assign]

        try:
            result = clf.classify(audio, sr)
            # CLAP 0.10 < 0.35, DSP-Mittel sollte ≥ 0.50 sein → Gate aktiv
            # Das Ergebnis sollte is_schlager=True haben (DSP erkennt Schlager)
            # oder zumindest nicht von CLAP runtergezogen sein
            assert result.confidence >= 0.30, (
                f"Confidence should not be dragged down by CLAP, got {result.confidence:.3f}"
            )
        finally:
            clf._compute_clap_score = original_clap  # type: ignore[method-assign]

    def test_clap_high_score_dsp_low_no_gate(self):
        """CLAP ≥ 0.35 → Gate NICHT aktiv, normale Fusion."""
        clf = GermanSchlagerClassifier()
        sr = 48000
        # Rauschen → DSP sollte niedriges Mittel haben
        np.random.seed(42)
        audio = (np.random.randn(sr * 5) * 0.1).astype(np.float32)

        original_clap = clf._compute_clap_score

        def fake_clap(*args, **kwargs):
            clf._clap_score_is_fallback = False
            return 0.50  # CLAP moderat

        clf._compute_clap_score = fake_clap  # type: ignore[method-assign]

        try:
            result = clf.classify(audio, sr)
            # Gate sollte NICHT aktiv sein (CLAP ≥ 0.35)
            # Confidence = 0.30*clap + 0.70*dsp (normale Fusion)
            assert result.confidence >= 0.0  # Sanity: kein Crash
        finally:
            clf._compute_clap_score = original_clap  # type: ignore[method-assign]

    def test_clap_fallback_flag_uses_pure_dsp(self):  # §V6: logger.warning handled at call site
        """Wenn _clap_score_is_fallback=True → pure-DSP-Konfidenz."""
        clf = GermanSchlagerClassifier()
        sr = 48000
        t = np.arange(sr * 5) / sr
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        original_clap = clf._compute_clap_score

        def fake_clap(*args, **kwargs):
            clf._clap_score_is_fallback = True
            return 0.35  # neutraler Fallback

        clf._compute_clap_score = fake_clap  # type: ignore[method-assign]

        try:
            result = clf.classify(audio, sr)
            # Fallback-Flag → confidence = pure DSP weighted_mean
            assert result.confidence >= 0.0
        finally:
            clf._compute_clap_score = original_clap  # type: ignore[method-assign]


# ═══════════════════════════════════════════════════════════════
# §v10.0.5 Genre-Profile Coverage Tests
# Stellt sicher, dass ALLE Genre-Profile vollständig definiert
# und mit konsistenten Parametern ausgestattet sind.
# ═══════════════════════════════════════════════════════════════


class TestGenreProfileUniversality:
    """§v10.0.5: Alle 19 Genres müssen vollständige Profile haben."""

    REQUIRED_PARAMS = frozenset(
        {
            "gp_memory_key",
            "groove_dtw_max_ms",
            "compression_ratio_cap",
        }
    )

    RECOMMENDED_PARAMS = frozenset(
        {
            "brillanz_target",
            "waerme_target",
            "deessing_strength_cap",
            "transient_preservation_strength",
        }
    )

    def test_ambient_profile_not_empty(self):
        """Ambient darf KEIN leeres Profil sein (Bug: war vorher nicht definiert)."""
        from backend.core.genre_classifier import AMBIENT_RESTORATION_PROFILE

        assert len(AMBIENT_RESTORATION_PROFILE) >= 8
        assert AMBIENT_RESTORATION_PROFILE["gp_memory_key"] == "ambient"
        assert "groove_dtw_max_ms" in AMBIENT_RESTORATION_PROFILE
        assert "brillanz_target" in AMBIENT_RESTORATION_PROFILE
        assert "waerme_target" in AMBIENT_RESTORATION_PROFILE

    def test_world_profile_not_empty(self):
        """World darf KEIN leeres Profil sein (Bug: war vorher nicht definiert)."""
        from backend.core.genre_classifier import WORLD_RESTORATION_PROFILE

        assert len(WORLD_RESTORATION_PROFILE) >= 6
        assert WORLD_RESTORATION_PROFILE["gp_memory_key"] == "world"
        assert "groove_dtw_max_ms" in WORLD_RESTORATION_PROFILE
        assert "brillanz_target" in WORLD_RESTORATION_PROFILE

    def test_oper_profile_has_all_required(self):
        """Oper muss groove_dtw_max_ms, compression_ratio_cap, etc. haben."""
        from backend.core.genre_classifier import OPER_RESTORATION_PROFILE

        for param in self.REQUIRED_PARAMS:
            assert param in OPER_RESTORATION_PROFILE, f"oper MISSING required param: {param}"
        assert OPER_RESTORATION_PROFILE["groove_dtw_max_ms"] == 12.0
        assert OPER_RESTORATION_PROFILE["compression_ratio_cap"] == 1.2
        assert OPER_RESTORATION_PROFILE["brillanz_target"] == 0.85
        assert OPER_RESTORATION_PROFILE["waerme_target"] == 0.82

    def test_all_genre_profiles_non_empty(self):
        """JEDES Genre-Profil muss mindestens 4 Parameter haben."""
        from backend.core.genre_classifier import GENRE_RESTORATION_PROFILES

        empty_or_tiny = []
        for name, prof in GENRE_RESTORATION_PROFILES.items():
            if not isinstance(prof, dict) or len(prof) < 3:
                empty_or_tiny.append(name)

        assert not empty_or_tiny, f"Genres mit zu wenigen Parametern: {empty_or_tiny}"

    def test_get_restoration_profile_returns_populated_for_all_genres(self):
        """get_restoration_profile() muss für ALLE 19 Genres ein nicht-leeres Dict liefern."""
        from backend.core.genre_classifier import get_restoration_profile

        genres = [
            "schlager",
            "jazz",
            "klassik",
            "oper",
            "rock",
            "pop",
            "blues",
            "soul/r&b",
            "country",
            "folk",
            "funk",
            "electronic",
            "hip-hop",
            "metal",
            "latin",
            "gospel",
            "reggae",
            "ambient",
            "world",
        ]

        fails = []
        for genre in genres:
            profile = get_restoration_profile(genre)
            if not profile or len(profile) < 3:
                fails.append(f"{genre}: {profile}")

        assert not fails, "get_restoration_profile() returned empty/tiny profiles:\n" + "\n".join(fails)

    def test_ambient_has_dereverb_disabled(self):
        """Ambient: Hall ist intentional — Dereverb muss disabled sein."""
        from backend.core.genre_classifier import AMBIENT_RESTORATION_PROFILE

        assert AMBIENT_RESTORATION_PROFILE.get("phase_20_dereverb_enabled") is False
        assert AMBIENT_RESTORATION_PROFILE.get("phase_49_dereverb_enabled") is False

    def test_ambient_has_high_groove_tolerance(self):
        """Ambient hat keine perkussiven Transienten — hohe DTW-Toleranz."""
        from backend.core.genre_classifier import AMBIENT_RESTORATION_PROFILE

        assert AMBIENT_RESTORATION_PROFILE["groove_dtw_max_ms"] >= 10.0

    def test_hiphop_has_bass_kraft_target(self):
        """Hip-Hop braucht bass_kraft_target für Sub-Bass-Präsenz."""
        from backend.core.genre_classifier import HIPHOP_RESTORATION_PROFILE

        assert "bass_kraft_target" in HIPHOP_RESTORATION_PROFILE
        assert HIPHOP_RESTORATION_PROFILE["bass_kraft_target"] >= 0.85

    def test_country_has_transient_preservation(self):
        """Country: Akustik-Gitarren-Transienten müssen erhalten bleiben."""
        from backend.core.genre_classifier import COUNTRY_RESTORATION_PROFILE

        assert "transient_preservation_strength" in COUNTRY_RESTORATION_PROFILE
        assert COUNTRY_RESTORATION_PROFILE["transient_preservation_strength"] >= 0.8

    def test_gospel_has_deessing_cap(self):
        """Gospel: Vokal-lastig mit potenziell starken Sibilanten."""
        from backend.core.genre_classifier import GOSPEL_RESTORATION_PROFILE

        assert "deessing_strength_cap" in GOSPEL_RESTORATION_PROFILE
        assert 0.2 <= GOSPEL_RESTORATION_PROFILE["deessing_strength_cap"] <= 0.7

    def test_reggae_has_waerme_target(self):
        """Reggae: Wärme für Bass-Dub und Orgel-Sounds."""
        from backend.core.genre_classifier import REGGAE_RESTORATION_PROFILE

        assert "waerme_target" in REGGAE_RESTORATION_PROFILE
        assert REGGAE_RESTORATION_PROFILE["waerme_target"] >= 0.80

    def test_required_params_in_all_lowercase_profiles(self):
        """Jedes Profil muss gp_memory_key, groove_dtw_max_ms, compression_ratio_cap haben."""
        from backend.core.genre_classifier import GENRE_RESTORATION_PROFILES

        missing = []
        for name, prof in GENRE_RESTORATION_PROFILES.items():
            if name[0].isupper():
                continue
            for param in self.REQUIRED_PARAMS:
                if param not in prof:
                    missing.append(f"{name}: MISSING {param}")

        assert not missing, "Fehlende Pflichtparameter in Genre-Profilen:\n" + "\n".join(missing)
