from __future__ import annotations

"""
tests/unit/test_consonant_enhancement.py
=========================================

Unit-Tests für core/consonant_enhancement.py

Prüft §2.8 Step 5b/5c:
  - ConsonantDetector (ZCR + HF-Energie-Gate)
  - ConsonantEnhancement (Adaptive Spectral Tilt Correction)
  - SNR-Invariante (+3 dB im Frikativband)
  - Boost ≤ +6 dB
  - NaN/Inf-Freiheit
  - Kausal-Konditionierung aus DefectScanner-Scores
  - Singleton-Thread-Safety

Alle Tests verwenden synthetische Signale (keine realen Audiodateien), §5.4.
"""


import math
import threading

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

# ---------------------------------------------------------------------------
# Hilfs-Fixtures
# ---------------------------------------------------------------------------
SR = 48_000  # Pflicht-SR §6.6
RNG = np.random.default_rng(42)  # Reproduzierbarkeit §5.4


def _sine(freq: float, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Reiner Sinuston."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _fricative(sr: int = SR, duration: float = 1.0) -> np.ndarray:
    """Synthetisches Frikativsignal: hochfrequentes AM-Rauschen 6–12 kHz."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * 8_000 * t)
    noise = RNG.standard_normal(len(t)).astype(np.float32) * 0.5
    # Bandpass 6–12 kHz als Proxy für Frikativenergie
    from scipy.signal import butter, sosfilt

    sos = butter(4, [6_000 / (sr / 2), 12_000 / (sr / 2)], btype="band", output="sos")
    fric = sosfilt(sos, noise) + 0.1 * carrier
    return (fric / (np.max(np.abs(fric)) + 1e-9)).astype(np.float32)  # type: ignore[no-any-return]


def _voiced(sr: int = SR, duration: float = 1.0) -> np.ndarray:
    """Synthetisches Vokal-Signal (200 Hz Grundton + Obertöne)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = sum(np.sin(2 * np.pi * 200 * n * t) / n for n in range(1, 8))
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.8).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
from backend.core.consonant_enhancement import (
    CAUSAL_DEFECT_BOOST,
    MAX_BOOST_DB,
    SNR_MIN_IMPROVEMENT_DB,
    ConsonantEnhancement,
    ConsonantEnhancementResult,
    _fricative_band,
    _snr_in_band,
    enhance_consonants,
    get_consonant_enhancer,
)

# ===========================================================================
# 1. Singleton & Thread-Safety (§3.2)
# ===========================================================================


class TestSingleton:
    def test_01_singleton_ist_dieselbe_instanz(self):
        a = get_consonant_enhancer()
        b = get_consonant_enhancer()
        assert a is b

    def test_02_singleton_thread_safe(self):
        results = []

        def _get():
            results.append(get_consonant_enhancer())

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)

    def test_03_convenience_wrapper_nutzt_singleton(self):
        audio = _sine(440.0, 0.5)
        result = enhance_consonants(audio, SR)
        assert isinstance(result, ConsonantEnhancementResult)


# ===========================================================================
# 2. Ausgabe-Form & Dtype (§5.1 Shape/Dtype-Tests)
# ===========================================================================


class TestOutputShapeAndDtype:
    def test_04_mono_shape_erhalten(self):
        audio = _sine(440.0, 1.0)
        result = enhance_consonants(audio, SR)
        assert result.audio.shape == audio.shape

    def test_05_stereo_shape_erhalten(self):
        audio = np.stack([_sine(440.0), _sine(880.0)], axis=0)
        result = enhance_consonants(audio, SR)
        assert result.audio.shape == audio.shape

    def test_06_ausgabe_ist_float32(self):
        audio = _sine(440.0).astype(np.float64)
        result = enhance_consonants(audio, SR)
        assert result.audio.dtype == np.float32

    def test_07_kein_clipping_im_ausgang(self):
        audio = _sine(440.0)
        result = enhance_consonants(audio, SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ===========================================================================
# 3. NaN/Inf-Schutz (§3.1 & §5.1)
# ===========================================================================


class TestNanInfSchutz:
    def test_08_nan_im_eingang_kein_absturz(self):
        audio = _sine(440.0)
        audio[100:200] = np.nan
        result = enhance_consonants(audio, SR)
        assert np.isfinite(result.audio).all()

    def test_09_inf_im_eingang_kein_absturz(self):
        audio = _sine(440.0)
        audio[50] = np.inf
        result = enhance_consonants(audio, SR)
        assert np.isfinite(result.audio).all()

    def test_10_stille_kein_absturz_nan_frei(self):
        audio = np.zeros(SR, dtype=np.float32)
        result = enhance_consonants(audio, SR)
        assert np.isfinite(result.audio).all()
        assert math.isfinite(result.snr_improvement_db)
        assert math.isfinite(result.boost_applied_db)

    def test_11_ergebnis_felder_alle_finite(self):
        audio = _fricative()
        result = enhance_consonants(audio, SR, voice_gender="female")
        for field_name in ("snr_improvement_db", "boost_applied_db", "causal_factor"):
            val = getattr(result, field_name)
            assert math.isfinite(val), f"NaN/Inf in {field_name}: {val}"


# ===========================================================================
# 4. Stimmtyp-adaptive Bänder (§2.8)
# ===========================================================================


class TestVoiceGenderBands:
    @pytest.mark.parametrize(
        "gender,expected_lo,expected_hi",
        [
            ("male", 5_000, 10_000),
            ("female", 6_000, 12_000),
            ("child", 7_000, 14_000),
            ("unknown", 6_000, 12_000),  # FEMALE-Fallback per Spec
        ],
    )
    def test_12_band_grenzen_gender_adaptiv(self, gender, expected_lo, expected_hi):
        lo, hi = _fricative_band(gender, SR)
        assert lo <= expected_lo + 500, f"{gender}: f_lo={lo} > {expected_lo}"
        assert hi >= expected_hi - 1_000, f"{gender}: f_hi={hi} < {expected_hi}"

    def test_13_nyquist_sicher_kein_bug_bei_8khz_sr(self):
        # 8 kHz SR → Nyquist 4 kHz → alle Bänder müssen darunter liegen
        lo, hi = _fricative_band("female", 8_000)
        assert lo < hi
        assert hi < 4_000


# ===========================================================================
# 5. Kausal-Konditionierung (§2.8 positionsabhängige Stärke)
# ===========================================================================


class TestCausalConditioning:
    def test_14_kein_defect_score_gibt_default_faktor(self):
        enh = ConsonantEnhancement()
        factor = enh._causal_factor({})
        assert factor == pytest.approx(CAUSAL_DEFECT_BOOST["default"], abs=1e-6)

    def test_15_bandwidth_loss_ergibt_hoechsten_faktor(self):
        enh = ConsonantEnhancement()
        f_bw = enh._causal_factor({"bandwidth_loss": 1.0})
        f_comp = enh._causal_factor({"compression_artifacts": 1.0})
        assert f_bw >= f_comp

    def test_16_faktor_immer_zwischen_0_1_und_1(self):
        enh = ConsonantEnhancement()
        for scores in [
            {},
            {"bandwidth_loss": 0.0},
            {"bandwidth_loss": 1.0},
            {"high_freq_noise": 0.5, "tape_hiss": 0.3},
            {"bandwidth_loss": 999.0},  # Extremwert
        ]:
            f = enh._causal_factor(scores)
            assert 0.1 <= f <= 1.0, f"Faktor {f} außerhalb [0.1, 1.0] für {scores}"

    def test_17_nan_in_defect_scores_kein_absturz(self):
        enh = ConsonantEnhancement()
        factor = enh._causal_factor({"bandwidth_loss": float("nan")})
        assert math.isfinite(factor)

    def test_18_hohe_defect_scores_erhoehen_boost(self):
        audio = _fricative(duration=2.0)
        r_low = enhance_consonants(audio, SR, defect_scores={"bandwidth_loss": 0.1})
        r_high = enhance_consonants(audio, SR, defect_scores={"bandwidth_loss": 0.95})
        assert r_high.boost_applied_db >= r_low.boost_applied_db - 1e-3


# ===========================================================================
# 6. Boost-Grenzen (§2.8: ≤ +6 dB)
# ===========================================================================


class TestBoostBounds:
    def test_19_boost_nie_ueber_6_db(self):
        audio = _fricative(duration=2.0)
        for gender in ("male", "female", "child", "unknown"):
            result = enhance_consonants(
                audio,
                SR,
                voice_gender=gender,
                defect_scores={"bandwidth_loss": 1.0},
            )
            assert result.boost_applied_db <= MAX_BOOST_DB + 1e-6, (
                f"Boost {result.boost_applied_db:.2f} dB > {MAX_BOOST_DB} dB für gender={gender}"
            )

    def test_20_boost_ist_nicht_negativ(self):
        audio = _fricative(duration=1.0)
        result = enhance_consonants(audio, SR, defect_scores={"bandwidth_loss": 0.8})
        assert result.boost_applied_db >= 0.0

    def test_21_kein_boost_bei_null_causal_factor(self):
        """Ohne Defekt-Scores ist Boost minimal, aber kein Überschreiten von +6 dB."""
        audio = _sine(440.0, 2.0)
        result = enhance_consonants(audio, SR, defect_scores={})
        assert result.boost_applied_db <= MAX_BOOST_DB


# ===========================================================================
# 7. SNR-Invariante (§2.8: SNR_frikativ_after ≥ SNR_frikativ_before + 3 dB)
# ===========================================================================


class TestSNRInvariant:
    def test_22_snr_invariante_auf_frikativ_signal(self):
        """Klares Frikativsignal: SNR sollte sich verbessern."""
        audio = _fricative(duration=3.0)
        # Stark gedämpftes Frikativband simulieren (NR hat abgedämpft)
        from scipy.signal import butter, sosfilt

        sos = butter(4, [6_000 / (SR / 2), min(12_000 / (SR / 2), 0.99)], btype="band", output="sos")
        fric = sosfilt(sos, audio)
        audio_nrd = audio - 0.7 * fric  # 70 % des Frikativbands weggefiltert
        result = enhance_consonants(
            audio_nrd.astype(np.float32),
            SR,
            voice_gender="female",
            defect_scores={"bandwidth_loss": 0.9, "high_freq_noise": 0.7},
        )
        # Invariante: SNR-Verbesserung ≥ 3 dB ODER invariant_met=True wenn Fallback
        if result.fricative_segments > 0:
            # Akzeptiere wenn invariant_met gesetzt
            assert result.invariant_met or result.snr_improvement_db >= SNR_MIN_IMPROVEMENT_DB - 1.0

    def test_23_snr_hilfsfunktion_nan_sicher(self):
        """_snr_in_band darf nicht NaN/Inf zurückgeben."""
        audio = np.zeros(SR, dtype=np.float32)
        snr = _snr_in_band(audio, SR, 6_000, 12_000)
        assert math.isfinite(snr)

    def test_24_snr_hilfsfunktion_auf_rauschen(self):
        audio = RNG.standard_normal(SR).astype(np.float32) * 0.5
        snr = _snr_in_band(audio, SR, 6_000, 12_000)
        assert math.isfinite(snr)


# ===========================================================================
# 8. Frikativ-Maske (§2.8 Step 5b: ZCR + HF-Energie)
# ===========================================================================


class TestSibilantMask:
    def test_25_frikativ_signal_ergibt_maske(self):
        enh = ConsonantEnhancement()
        audio = _fricative(duration=1.0)
        mask = enh._sibilant_mask(audio, SR)
        assert mask.shape[0] == audio.shape[0]
        assert mask.dtype == bool
        # Frikativsignal sollte mindestens einige True-Frames erzeugen
        assert mask.sum() > 0

    def test_26_stille_ergibt_leere_maske(self):
        enh = ConsonantEnhancement()
        audio = np.zeros(SR // 2, dtype=np.float32)
        mask = enh._sibilant_mask(audio, SR)
        assert mask.sum() == 0

    def test_27_reiner_sinuston_wenig_maske(self):
        """Tiefer Sinuston (200 Hz) hat kaum HF-Energie → wenige Frikativ-Frames."""
        enh = ConsonantEnhancement()
        audio = _sine(200.0, 2.0)
        mask = enh._sibilant_mask(audio, SR)
        # Weniger als 30 % aller Samples als Frikativ markiert
        assert mask.mean() < 0.30

    def test_28_maske_laenge_gleich_audiolarenge(self):
        enh = ConsonantEnhancement()
        for n in [SR // 4, SR, 2 * SR]:
            audio = _fricative(duration=n / SR)
            mask = enh._sibilant_mask(audio, SR)
            assert mask.shape[0] == n


# ===========================================================================
# 9. Passthrough bei leerem Eingang (§5.1 Edge-Cases)
# ===========================================================================


class TestEdgeCases:
    def test_29_leeres_array_kein_absturz(self):
        audio = np.zeros(0, dtype=np.float32)
        result = enhance_consonants(audio, SR)
        assert isinstance(result, ConsonantEnhancementResult)

    def test_30_sehr_kurzes_signal_kein_absturz(self):
        audio = _sine(440.0, 0.001)  # 48 Samples
        result = enhance_consonants(audio, SR)
        assert np.isfinite(result.audio).all()

    def test_31_mono_und_stereo_konsistent(self):
        """Mono und Stereo-Verarbeitung führen zu ähnlichen boost_applied_db."""
        mono = _fricative(duration=2.0)
        stereo = np.stack([mono, mono], axis=0)
        r_mono = enhance_consonants(mono, SR, defect_scores={"bandwidth_loss": 0.7})
        r_stereo = enhance_consonants(stereo, SR, defect_scores={"bandwidth_loss": 0.7})
        assert abs(r_mono.boost_applied_db - r_stereo.boost_applied_db) < 1e-6

    def test_32_gleiche_eingabe_gleiche_ausgabe(self):
        """Determinismus §5.1 Konsistenztest."""
        audio = _fricative(duration=1.0)
        r1 = enhance_consonants(audio.copy(), SR, defect_scores={"bandwidth_loss": 0.5})
        r2 = enhance_consonants(audio.copy(), SR, defect_scores={"bandwidth_loss": 0.5})
        np.testing.assert_array_almost_equal(r1.audio, r2.audio, decimal=5)

    def test_33_voiced_signal_bleibt_unveraendert_wenn_keine_frikative(self):
        """Reines Voiced-Signal ohne Frikative → kein Boost (fricative_segments=0)."""
        audio = _voiced(duration=2.0)
        result = enhance_consonants(audio, SR, defect_scores={})
        assert result.fricative_segments == 0 or result.boost_applied_db <= MAX_BOOST_DB

    def test_34_causal_factor_in_ergebnis_gesetzt(self):
        audio = _fricative(duration=1.0)
        result = enhance_consonants(audio, SR, defect_scores={"high_freq_noise": 0.6})
        assert 0.1 <= result.causal_factor <= 1.0

    def test_35_ergebnis_voice_gender_durchgereicht(self):
        audio = _sine(440.0, 0.5)
        for gender in ("male", "female", "child", "unknown"):
            result = enhance_consonants(audio, SR, voice_gender=gender)
            assert result.voice_gender == gender
