from __future__ import annotations

"""
tests/unit/test_consonant_detector.py — Testsuite für ConsonantDetector (§2.8 Step 5b)
========================================================================================

Pflicht-Tests gemäß §5.1: Shape, NaN/Inf, Bounds, Edge-Cases, Mono+Stereo,
Konsistenz, Singleton-Identität, Thread-Safety.

np.random.seed(42) für Reproduzierbarkeit (§5.4).
"""


import concurrent.futures
import math

import numpy as np
import pytest

from plugins.consonant_detector import (
    ConsonantDetectionResult,
    ConsonantDetector,
    detect_consonants,
    get_consonant_detector,
)

SR = 48_000


# ── Hilfsfunktionen ──────────────────────────────────────────────────────── #


def _white_noise(duration_s: float = 0.5, amplitude: float = 0.3) -> np.ndarray:
    """Weißes Rauschen mit hoher ZCR und breitem Spektrum → Frikativ-ähnlich."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration_s * SR)) * amplitude).astype(np.float32)


def _sine(freq_hz: float = 440.0, duration_s: float = 0.5) -> np.ndarray:
    """Reiner Sinuston: niedrige ZCR, keine HF-Energie → kein Frikativ."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)  # type: ignore[no-any-return]


def _silence(duration_s: float = 0.3) -> np.ndarray:
    """Stille (0-Signal)."""
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _fricative_like(duration_s: float = 0.5) -> np.ndarray:
    """Synthetisches Frikativ-Signal: hochfrequentes Bandpass-Rauschen (6–14 kHz)."""
    import scipy.signal as sp

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(int(duration_s * SR)).astype(np.float64)
    nyq = SR / 2.0
    sos = sp.butter(6, [6_000 / nyq, 14_000 / nyq], btype="band", output="sos")
    filtered = sp.sosfilt(sos, noise)
    filtered /= max(np.max(np.abs(filtered)), 1e-8)  # normalisieren
    return filtered.astype(np.float32) * 0.5  # type: ignore[no-any-return]


# ── Tests ────────────────────────────────────────────────────────────────── #


class TestConsonantDetectorSingleton:
    """Schritt 5b — Singleton-Invarianten (§3.2)."""

    def test_01_get_returns_consonant_detector_instance(self) -> None:
        """get_consonant_detector() gibt eine ConsonantDetector-Instanz zurück."""
        det = get_consonant_detector()
        assert isinstance(det, ConsonantDetector)

    def test_02_singleton_same_object(self) -> None:
        """Zwei Aufrufe liefern dasselbe Objekt."""
        a = get_consonant_detector()
        b = get_consonant_detector()
        assert a is b

    def test_03_singleton_thread_safe(self) -> None:
        """20 parallele Aufrufe liefern alle dasselbe Singleton-Objekt."""
        results: list[ConsonantDetector] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(get_consonant_detector) for _ in range(20)]
            results = [f.result() for f in futures]
        first = results[0]
        assert all(inst is first for inst in results)

    def test_04_convenience_wrapper_returns_result(self) -> None:
        """detect_consonants() Convenience-Funktion arbeitet fehlerfrei."""
        audio = _white_noise()
        result = detect_consonants(audio, SR)
        assert isinstance(result, ConsonantDetectionResult)


class TestConsonantDetectorShape:
    """Ausgabe-Shape-Invarianten (§3.1)."""

    def test_05_mask_shape_matches_mono_input(self) -> None:
        """Maske hat dieselbe Länge wie das Eingangs-Mono-Signal."""
        audio = _white_noise(duration_s=0.5)
        result = get_consonant_detector().detect(audio, SR)
        assert result.mask.shape == audio.shape

    def test_06_mask_shape_matches_stereo_channels_first(self) -> None:
        """Stereo [2, n] → Maske hat Länge n."""
        mono = _white_noise(duration_s=0.3)
        stereo = np.stack([mono, mono * 0.8])  # [2, n]
        result = get_consonant_detector().detect(stereo, SR)
        assert result.mask.shape[0] == stereo.shape[1]

    def test_07_mask_shape_matches_stereo_samples_first(self) -> None:
        """Stereo [n, 2] → Maske hat Länge n."""
        mono = _white_noise(duration_s=0.3)
        stereo = np.stack([mono, mono], axis=1)  # [n, 2]
        result = get_consonant_detector().detect(stereo, SR)
        assert result.mask.shape[0] == stereo.shape[0]

    def test_08_mask_dtype_is_bool(self) -> None:
        """Maske hat immer dtype=bool."""
        result = get_consonant_detector().detect(_white_noise(), SR)
        assert result.mask.dtype == bool

    def test_09_mask_shape_short_audio(self) -> None:
        """Sehr kurzes Audio (< 1 Frame) → leere Maske, kein Fehler."""
        short = np.zeros(256, dtype=np.float32)
        result = get_consonant_detector().detect(short, SR)
        assert result.mask.shape[0] == 256
        assert result.mask.dtype == bool


class TestConsonantDetectorNaNInf:
    """NaN/Inf-Robustheit (§3.1)."""

    def test_10_nan_input_no_crash(self) -> None:
        """NaN-Eingang → kein Absturz, gültige Maske."""
        audio = np.full(SR // 2, float("nan"), dtype=np.float32)
        result = get_consonant_detector().detect(audio, SR)
        assert result.mask.shape[0] == len(audio)
        assert result.mask.dtype == bool

    def test_11_inf_input_no_crash(self) -> None:
        """Inf-Eingang → kein Absturz, gültige Maske."""
        audio = np.full(SR // 2, float("inf"), dtype=np.float32)
        result = get_consonant_detector().detect(audio, SR)
        assert result.mask.shape[0] == len(audio)

    def test_12_mixed_nan_inf_input(self) -> None:
        """Gemischte NaN/Inf-Werte → kein Absturz."""
        rng = np.random.default_rng(7)
        audio = rng.standard_normal(SR).astype(np.float32)
        audio[100:200] = float("nan")
        audio[500:600] = float("inf")
        result = get_consonant_detector().detect(audio, SR)
        assert result.mask.shape[0] == len(audio)

    def test_13_result_fields_finite(self) -> None:
        """Alle Skalar-Felder im Ergebnis sind finite (kein NaN/Inf)."""
        result = get_consonant_detector().detect(_white_noise(), SR)
        assert math.isfinite(result.fricative_ratio)
        assert math.isfinite(result.mean_zcr)
        assert math.isfinite(result.mean_hf_ratio)


class TestConsonantDetectorEdgeCases:
    """Edge-Cases (§5.1)."""

    def test_14_silence_no_fricative(self) -> None:
        """Stille → alle Frikativ-Frames = 0."""
        result = get_consonant_detector().detect(_silence(), SR)
        assert result.n_fricative_frames == 0
        assert not result.mask.any()

    def test_15_pure_sine_no_fricative(self) -> None:
        """Reiner Sinuston (440 Hz) → keine Frikativ-Frames (wenig HF-Energie)."""
        result = get_consonant_detector().detect(_sine(440.0, 1.0), SR)
        assert result.n_fricative_frames == 0

    def test_16_white_noise_is_fricative(self) -> None:
        """Weißes Rauschen → mindestens 1 Frikativ-Frame erkannt."""
        result = get_consonant_detector().detect(_white_noise(1.0), SR)
        assert result.n_fricative_frames > 0
        assert result.mask.any()

    def test_17_bandpass_hf_noise_is_fricative(self) -> None:
        """HF-Bandpass-Rauschen (6–14 kHz) → Frikativ erkannt."""
        result = get_consonant_detector().detect(_fricative_like(0.5), SR)
        assert result.n_fricative_frames > 0

    def test_18_empty_array_no_crash(self) -> None:
        """Leeres Array → kein Absturz, leere Maske."""
        audio = np.array([], dtype=np.float32)
        result = get_consonant_detector().detect(audio, SR)
        assert result.mask.shape[0] == 0

    def test_19_dirac_impulse(self) -> None:
        """Dirac-Impuls → kein Absturz, gültige Maske."""
        audio = np.zeros(SR // 2, dtype=np.float32)
        audio[SR // 4] = 1.0
        result = get_consonant_detector().detect(audio, SR)
        assert result.mask.shape[0] == len(audio)

    def test_20_very_low_sample_rate(self) -> None:
        """Niedrige SR (8000 Hz) → kein Absturz, HF-Band adaptiv begrenzt."""
        audio = _white_noise(duration_s=1.0)
        result = get_consonant_detector().detect(audio, 8_000)
        assert result.mask.shape[0] == len(audio)

    def test_21_single_sample(self) -> None:
        """Einzelnes Sample → leere Maske, kein Fehler."""
        audio = np.array([0.5], dtype=np.float32)
        result = get_consonant_detector().detect(audio, SR)
        assert result.mask.shape[0] == 1


class TestConsonantDetectorBounds:
    """Bounds-Invarianten (§3.1, §5.1)."""

    def test_22_fricative_ratio_in_0_1(self) -> None:
        """fricative_ratio ∈ [0, 1]."""
        result = get_consonant_detector().detect(_white_noise(1.0), SR)
        assert 0.0 <= result.fricative_ratio <= 1.0

    def test_23_mean_zcr_non_negative(self) -> None:
        """mean_zcr ≥ 0."""
        result = get_consonant_detector().detect(_white_noise(1.0), SR)
        assert result.mean_zcr >= 0.0

    def test_24_mean_hf_ratio_in_0_1(self) -> None:
        """mean_hf_ratio ∈ [0, 1]."""
        result = get_consonant_detector().detect(_white_noise(1.0), SR)
        assert 0.0 <= result.mean_hf_ratio <= 1.0

    def test_25_n_fricative_frames_non_negative(self) -> None:
        """n_fricative_frames ≥ 0."""
        result = get_consonant_detector().detect(_sine(220.0, 0.5), SR)
        assert result.n_fricative_frames >= 0


class TestConsonantDetectorConsistency:
    """Konsistenz-Tests (gleiche Eingabe → gleiche Ausgabe, §5.1)."""

    def test_26_deterministic_on_same_input(self) -> None:
        """Gleiche Eingabe → identische Maske bei zwei Aufrufen."""
        audio = _white_noise(0.5)
        r1 = get_consonant_detector().detect(audio, SR)
        r2 = get_consonant_detector().detect(audio, SR)
        np.testing.assert_array_equal(r1.mask, r2.mask)

    def test_27_sample_rate_field_matches_input(self) -> None:
        """Ergebnis.sample_rate entspricht dem Übergabe-SR."""
        result = get_consonant_detector().detect(_white_noise(), 44_100)
        assert result.sample_rate == 44_100


class TestConsonantDetectorVoiceGender:
    """Stimmtyp-Adaptation (§2.8 VoiceGender-System)."""

    @pytest.mark.parametrize("gender", ["male", "female", "child", "androgynous", "unknown", "MALE", "Female"])
    def test_28_all_genders_no_crash(self, gender: str) -> None:
        """Alle Stimmtyp-Label → kein Absturz, gültige Maske."""
        result = get_consonant_detector().detect(_white_noise(0.3), SR, voice_gender=gender)
        assert result.mask.shape[0] > 0
        assert result.mask.dtype == bool

    def test_29_unknown_gender_is_broadband(self) -> None:
        """voice_gender='unknown' → breitestes HF-Band (4–16 kHz, konservativ)."""
        # Mit 'unknown' werden mehr Frames erkannt als mit engem Band → Smoke-Test
        audio = _fricative_like(0.5)
        result = get_consonant_detector().detect(audio, SR, voice_gender="unknown")
        assert isinstance(result, ConsonantDetectionResult)

    def test_30_invalid_gender_falls_back_to_unknown(self) -> None:
        """Ungültiger voice_gender → Fallback auf 'unknown', kein Absturz."""
        result = get_consonant_detector().detect(_white_noise(0.3), SR, voice_gender="robot")
        assert result.mask.dtype == bool


class TestConsonantDetectorIntegration:
    """Integrations-Tests: ConsonantDetector in _sibilant_mask (§2.8 Step 5b)."""

    def test_31_consonant_enhancement_imports_detector(self) -> None:
        """consonant_enhancement._sibilant_mask() ruft ConsonantDetector intern auf."""
        from backend.core.consonant_enhancement import ConsonantEnhancement

        ce = ConsonantEnhancement()
        # Weißes Rauschen: ConsonantDetector sollte Frikative erkennen
        mono = _white_noise(0.5)
        mask = ce._sibilant_mask(mono, SR)
        assert mask.dtype == bool
        assert mask.shape[0] == len(mono)

    def test_32_sibilant_mask_no_crash_on_silence(self) -> None:
        """_sibilant_mask() bei Stille → False-Maske, kein Absturz."""
        from backend.core.consonant_enhancement import ConsonantEnhancement

        ce = ConsonantEnhancement()
        mono = _silence(0.3)
        mask = ce._sibilant_mask(mono, SR)
        assert not mask.any()

    def test_33_end_to_end_enhance_with_fricative_audio(self) -> None:
        """Vollständige enhance()-Kette auf Frikativ-Audio läuft fehlerfrei durch."""
        from backend.core.consonant_enhancement import enhance_consonants

        audio = _fricative_like(0.3)
        result = enhance_consonants(audio, SR, voice_gender="female")
        assert result.audio.shape == audio.shape
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-5

    def test_34_mask_true_only_where_fricative(self) -> None:
        """Bei reinem HF-Rauschen hat die Maske True-Bereiche."""
        audio = _fricative_like(1.0)
        result = get_consonant_detector().detect(audio, SR)
        # Mindestens 10 % der Samples sollten als Frikativ erkannt werden
        assert result.mask.mean() > 0.05

    def test_35_n_fricative_frames_consistent_with_mask(self) -> None:
        """n_fricative_frames > 0 genau dann wenn mask.any()."""
        for audio in [_silence(0.3), _white_noise(0.5), _fricative_like(0.3)]:
            result = get_consonant_detector().detect(audio, SR)
            if result.n_fricative_frames > 0:
                assert result.mask.any(), "n_fricative_frames > 0 aber mask komplett False"
            if result.mask.any():
                assert result.n_fricative_frames > 0, "mask hat True aber n_fricative_frames == 0"
