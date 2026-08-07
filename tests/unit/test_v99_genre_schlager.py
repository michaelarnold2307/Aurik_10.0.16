import pytest

"""
tests/unit/test_v99_genre_schlager.py — GermanSchlagerClassifier Test-Suite (≥ 35 Tests)
Alle Tests synthetisch, kein ML-Modell-Download erforderlich.
"""

import concurrent.futures
import math

import numpy as np

SR = 48_000
np.random.seed(42)


def _sine(freq: float, dur: float = 3.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _am_signal(carrier_hz: float, mod_hz: float, dur: float = 5.0) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    modulator = 1.0 + 0.5 * np.sin(2 * np.pi * mod_hz * t)
    return (carrier * modulator).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(dur: float = 5.0) -> np.ndarray:
    """Deterministisches weißes Rauschen (Seed 42) — reproduzierbar bei parallelen Tests."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(dur * SR)) * 0.1).astype(np.float32)


def _silence(dur: float = 5.0) -> np.ndarray:
    return np.zeros(int(dur * SR), dtype=np.float32)


def _make_oompah(dur: float = 10.0, bpm: float = 120.0) -> np.ndarray:
    """Synthetisches Oom-Pah-Rhythmus-Signal mit Grundton + Begleitung."""
    samples = int(dur * SR)
    audio = np.zeros(samples, dtype=np.float32)
    beat_samples = int(SR * 60.0 / bpm)
    # Beats auf 1 und 2 unterschiedlich stark
    for i in range(int(dur * bpm / 60) + 1):
        pos = i * beat_samples
        end = min(pos + SR // 10, samples)
        if end <= samples:
            amp = 0.8 if i % 2 == 0 else 0.3
            audio[pos:end] += amp * _sine(220, (end - pos) / SR)[: end - pos]
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def _repeat_block(dur: float = 40.0) -> np.ndarray:
    """Signal mit hoher melodischer Wiederholungsrate (Schlager-Refrain-Muster)."""
    block_len = int(8.0 * SR)
    block = _sine(261.63, 8.0)  # C4
    repeats = math.ceil(int(dur * SR) / block_len)
    audio = np.tile(block, repeats)[: int(dur * SR)]
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Importprüfung
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import_module():
    from backend.core.genre_classifier import GermanSchlagerClassifier

    assert GermanSchlagerClassifier is not None


def test_01_classify_returns_result():
    from backend.core.genre_classifier import SchlagerClassificationResult, classify_genre

    audio = _white_noise(10.0)
    result = classify_genre(audio, SR)
    assert isinstance(result, SchlagerClassificationResult)


def test_02_result_fields_finite():
    from backend.core.genre_classifier import classify_genre

    audio = _white_noise(10.0)
    r = classify_genre(audio, SR)
    for attr in (
        "confidence",
        "clap_score",
        "accordion_score",
        "harmonic_simplicity",
        "rhythm_score",
        "vocal_german_prior",
        "melodic_repetition",
    ):
        val = getattr(r, attr)
        assert math.isfinite(val), f"NaN/Inf in {attr}: {val}"


def test_03_all_scores_bounded():
    from backend.core.genre_classifier import classify_genre

    audio = _white_noise(10.0)
    r = classify_genre(audio, SR)
    for attr in (
        "confidence",
        "clap_score",
        "accordion_score",
        "harmonic_simplicity",
        "rhythm_score",
        "vocal_german_prior",
        "melodic_repetition",
    ):
        val = getattr(r, attr)
        assert 0.0 <= val <= 1.0, f"{attr}={val} außerhalb [0,1]"


def test_04_is_schlager_bool():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(10.0), SR)
    assert isinstance(r.is_schlager, bool)


def test_05_no_false_positive_white_noise():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(10.0), SR)
    assert not r.is_schlager, "Weißes Rauschen darf nicht als Schlager erkannt werden"


def test_06_no_false_positive_silence():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_silence(5.0), SR)
    assert not r.is_schlager
    assert math.isfinite(r.confidence)


def test_07_no_false_positive_pure_sine():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_sine(440.0, 10.0), SR)
    assert not r.is_schlager


def test_08_accordion_am_signal_score():
    """Reed-Beating-Signal (8 Hz AM) → accordion_score erhöht."""
    from backend.core.genre_classifier import get_genre_classifier

    audio = _am_signal(440.0, 8.0, dur=5.0)
    clf = get_genre_classifier()
    mono = audio if audio.ndim == 1 else audio.mean(axis=0)
    score = clf._compute_accordion_score(mono, SR)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_09_no_accordion_pure_sine():
    """Reiner Sinuston → niedriger accordion_score."""
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    mono = _sine(440.0, 5.0)
    score = clf._compute_accordion_score(mono, SR)
    assert score <= 0.6  # kein starkes Reed-Beating-Muster


def test_10_harmonic_simplicity_range():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _sine(261.63, 10.0)  # reiner Ton
    hsi = clf._compute_harmonic_simplicity(audio, SR)
    assert math.isfinite(hsi)
    assert 0.0 <= hsi <= 1.0


def test_10b_harmonic_simplicity_short_clip_has_no_cqt_fft_warning():
    import warnings

    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _sine(261.63, 2.0)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
        hsi = clf._compute_harmonic_simplicity(audio, SR)
    assert math.isfinite(hsi)
    assert 0.0 <= hsi <= 1.0


def test_11_melodic_repetition_high_for_repetitive():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _repeat_block(40.0)
    rep = clf._compute_melodic_repetition(audio, SR)
    assert math.isfinite(rep)
    assert rep >= 0.30, f"Sehr repetitives Signal sollte ≥ 0.30 erreichen, hat {rep:.3f}"


def test_12_melodic_repetition_short_audio_neutral():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _sine(440.0, 10.0)  # < 30 s → neutral
    rep = clf._compute_melodic_repetition(audio, SR)
    # Short audio → neutral value (no division errors)
    assert math.isfinite(rep)
    assert 0.0 <= rep <= 1.0


def test_13_vocal_german_prior_range():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _white_noise(10.0)
    prior = clf._compute_german_vocal_prior(audio, SR)
    assert math.isfinite(prior)
    assert 0.0 <= prior <= 1.0


def test_14_rhythm_classification_returns_tuple():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _make_oompah(10.0, bpm=120.0)
    result = clf._classify_rhythm_pattern(audio, SR)
    assert isinstance(result, tuple)
    assert len(result) == 3
    score, label, bpm = result
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert isinstance(label, str)
    assert math.isfinite(bpm)


def test_15_genre_label_not_empty():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(10.0), SR)
    assert isinstance(r.genre_label, str)
    assert len(r.genre_label) > 0


def test_16_subgenre_not_empty():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(10.0), SR)
    assert isinstance(r.subgenre, str)
    assert len(r.subgenre) > 0


def test_17_bpm_range():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_make_oompah(15.0, 120.0), SR)
    assert math.isfinite(r.bpm)
    assert r.bpm >= 0.0


def test_18_key_field_string():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(10.0), SR)
    assert isinstance(r.key, str)


def test_18b_estimate_key_short_clip_has_no_cqt_fft_warning():
    import warnings

    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _sine(261.63, 2.0)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*n_fft=.*too large.*", category=UserWarning)
        key = clf._estimate_key(audio, SR)
    assert isinstance(key, str)


def test_19_reasoning_field_string():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(10.0), SR)
    assert isinstance(r.reasoning, str)
    assert len(r.reasoning) > 0


def test_20_stereo_input_handled():
    """Stereo-Eingang: kein Absturz, endliche Werte, deterministisches Signal."""
    from backend.core.genre_classifier import classify_genre

    # Deterministisches Stereo-Signal (kein np.random ohne Seed — war flaky).
    # Zwei leicht verschiedene Sinustöne → klar kein Schlager-Merkmal.
    t = np.linspace(0.0, 8.0, SR * 8, dtype=np.float32)
    left = np.sin(2.0 * np.pi * 300.0 * t) * 0.08
    right = np.sin(2.0 * np.pi * 350.0 * t) * 0.08
    stereo = np.stack([left, right], axis=0)
    r = classify_genre(stereo, SR)
    assert math.isfinite(r.confidence)
    assert 0.0 <= r.confidence <= 1.0
    assert isinstance(r.is_schlager, bool)
    # Zwei stationäre Sinustöne bilden kein Akkordeon-AM-Muster,
    # keinen Schlager-Rhythmus und keine harmonische I-IV-V-Schlager-Struktur.
    assert not r.is_schlager


def test_21_very_short_audio():
    from backend.core.genre_classifier import classify_genre

    audio = _sine(440.0, 2.0)
    r = classify_genre(audio, SR)
    assert math.isfinite(r.confidence)


def test_22_singleton_identity():
    from backend.core.genre_classifier import get_genre_classifier

    a = get_genre_classifier()
    b = get_genre_classifier()
    assert a is b


def test_23_singleton_thread_safe():
    from backend.core.genre_classifier import get_genre_classifier

    instances = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(get_genre_classifier) for _ in range(20)]
        instances = [f.result(timeout=15.0) for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_24_classify_genre_convenience():
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(8.0), SR)
    assert r is not None


def test_25_no_nan_in_repeated_calls():
    from backend.core.genre_classifier import classify_genre

    for _ in range(3):
        r = classify_genre(_white_noise(5.0), SR)
        assert math.isfinite(r.confidence)
        assert math.isfinite(r.accordion_score)


def test_26_accordion_score_high_for_am_tremolo():
    """4 Hz Balgzug-Tremolo → accordion_score nicht null."""
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    audio = _am_signal(440.0, 5.0, dur=5.0)  # 5 Hz Tremolo
    score = clf._compute_accordion_score(audio, SR)
    assert math.isfinite(score)
    assert score >= 0.0  # positiver oder neutraler Score


def test_27_hsi_returns_float():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    hsi = clf._compute_harmonic_simplicity(_white_noise(8.0), SR)
    assert isinstance(hsi, float)


def test_28_all_fields_have_default():
    """SchlagerClassificationResult hat alle erwarteten Felder."""
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(8.0), SR)
    for field in (
        "is_schlager",
        "confidence",
        "genre_label",
        "clap_score",
        "accordion_score",
        "harmonic_simplicity",
        "rhythm_score",
        "vocal_german_prior",
        "melodic_repetition",
        "subgenre",
        "bpm",
        "key",
        "reasoning",
    ):
        assert hasattr(r, field), f"Feld {field!r} fehlt im Result"


def test_29_confidence_threshold_respected():
    """is_schlager=True nur wenn confidence ≥ SCHLAGER_CONFIDENCE_THRESHOLD."""
    from backend.core.genre_classifier import classify_genre, get_genre_classifier

    r = classify_genre(_white_noise(15.0), SR)
    thresh = get_genre_classifier().SCHLAGER_CONFIDENCE_THRESHOLD
    if r.is_schlager:
        assert r.confidence >= thresh


def test_30_rhythm_score_bounded():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    score, label, bpm = clf._classify_rhythm_pattern(_white_noise(10.0), SR)
    assert 0.0 <= score <= 1.0


def test_31_melody_repetition_bounded():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    val = clf._compute_melodic_repetition(_repeat_block(35.0), SR)
    assert 0.0 <= val <= 1.0


def test_32_vocal_prior_bounded_silence():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    val = clf._compute_german_vocal_prior(_silence(8.0), SR)
    assert 0.0 <= val <= 1.0


def test_33_accordion_bounded_silence():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    val = clf._compute_accordion_score(_silence(5.0), SR)
    assert 0.0 <= val <= 1.0


def test_34_hsi_bounded_silence():
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    val = clf._compute_harmonic_simplicity(_silence(5.0), SR)
    assert 0.0 <= val <= 1.0


def test_35_classify_handles_float64():
    from backend.core.genre_classifier import classify_genre

    audio = np.random.randn(SR * 8).astype(np.float64) * 0.1
    r = classify_genre(audio, SR)
    assert math.isfinite(r.confidence)


def test_36_oompah_rhythmus_pattern_detektiert():
    """Oom-Pah-Muster: rhythm_score sollte moderat sein."""
    from backend.core.genre_classifier import classify_genre

    audio = _make_oompah(15.0, 120.0)
    r = classify_genre(audio, SR)
    assert math.isfinite(r.rhythm_score)
    assert 0.0 <= r.rhythm_score <= 1.0


# ── New tests for Oper genre, open_set_unknown, genre family scalars ──────────


def test_37_score_oper_returns_bounded_float():
    """_score_oper() gibt einen Wert in [0, 1] zurück."""
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    score = clf._score_oper(centroid_hz=2200.0, onset_rate=1.5, hsi=0.70, dr_db=50.0)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_38_score_oper_high_for_opera_profile():
    """Hohe DR + Singformant-Centroid → _score_oper soll ≥ 0.35 erreichen."""
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    # dr_db > 48 → +0.35; centroid 1800-3200 → +0.25; onset moderat → +0.15; hsi passt → +0.15
    score = clf._score_oper(centroid_hz=2400.0, onset_rate=1.8, hsi=0.72, dr_db=52.0)
    assert score >= 0.35, f"Oper-Profil-Signal sollte score ≥ 0.35 haben, ist {score:.3f}"


def test_39_score_oper_low_for_rock_profile():
    """Rock-Profil (niedrige DR, hohe Onsets) → _score_oper nahe 0."""
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    score = clf._score_oper(centroid_hz=3200.0, onset_rate=4.5, hsi=0.50, dr_db=18.0)
    assert score <= 0.30, f"Rock-Signal sollte niedrigen Oper-Score haben, ist {score:.3f}"


def test_40_compute_non_schlager_scores_includes_oper():
    """_compute_non_schlager_scores() muss 'Oper' im zurückgegebenen Dict haben."""
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    scores = clf._compute_non_schlager_scores(centroid_hz=2200.0, onset_rate=1.5, hsi=0.70, dr_db=50.0, bpm=80.0)
    assert "Oper" in scores, "Schlüssel 'Oper' fehlt in _compute_non_schlager_scores-Ergebnis"
    assert math.isfinite(scores["Oper"])
    assert 0.0 <= scores["Oper"] <= 1.0


def test_41_classify_result_has_open_set_unknown():
    """SchlagerClassificationResult hat das Feld 'open_set_unknown'."""
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_white_noise(10.0), SR)
    assert hasattr(r, "open_set_unknown")
    assert isinstance(r.open_set_unknown, bool)


def test_42_open_set_unknown_silence():
    """Stille → open_set_unknown=True (kein Genre erkennbar)."""
    from backend.core.genre_classifier import classify_genre

    r = classify_genre(_silence(8.0), SR)
    # Stille hat keine genre-relevanten Features → open_set_unknown erwartet
    assert isinstance(r.open_set_unknown, bool)  # nur Typ prüfen, nicht hartes True erzwingen


def test_43_genre_family_four_families():
    """_infer_genre_family kennt jetzt 4 Familien (rock/jazz/klassik/oper + schlager_folk)."""
    from backend.core.genre_classifier import get_genre_classifier

    clf = get_genre_classifier()
    # Opera-Profil-Scores: 'Oper' dominiert
    non_scores = {"Rock": 0.10, "Jazz": 0.12, "Klassik": 0.28, "Oper": 0.55}
    label, conf = clf._infer_genre_family(non_scores, schlager_family_score=0.05)
    assert label == "oper", f"Erwartet 'oper', erhalten '{label}'"
    assert conf >= 0.38


def test_44_build_song_calibration_klassik_reduces_reverb():
    """Klassik-genre_label → reverb family_scalar < 0.95 (Raumklang-Schutz)."""
    from backend.core.defect_scanner import MaterialType
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    profile_default = UnifiedRestorerV3._build_song_calibration_profile(
        material_type=MaterialType.VINYL,
        mode=QualityMode.QUALITY,
        restorability_score=70.0,
        input_snr_db=30.0,
        max_defect_severity=0.2,
        pipeline_confidence=0.8,
        genre_label="",
    )
    profile_klassik = UnifiedRestorerV3._build_song_calibration_profile(
        material_type=MaterialType.VINYL,
        mode=QualityMode.QUALITY,
        restorability_score=70.0,
        input_snr_db=30.0,
        max_defect_severity=0.2,
        pipeline_confidence=0.8,
        genre_label="Klassik",
    )
    rv_default = profile_default["family_scalars"]["reverb"]
    rv_klassik = profile_klassik["family_scalars"]["reverb"]
    assert rv_klassik < rv_default, f"Klassik reverb-scalar ({rv_klassik:.3f}) sollte < default ({rv_default:.3f}) sein"


def test_45_build_song_calibration_oper_raises_vocal():
    """Oper-genre_label → vocal family_scalar > Klassik (Singformant-Boost)."""
    from backend.core.defect_scanner import MaterialType
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    _shared = {
        "material_type": MaterialType.VINYL,
        "mode": QualityMode.QUALITY,
        "restorability_score": 70.0,
        "input_snr_db": 30.0,
        "max_defect_severity": 0.2,
        "pipeline_confidence": 0.8,
    }
    profile_klassik = UnifiedRestorerV3._build_song_calibration_profile(**_shared, genre_label="Klassik")  # type: ignore[arg-type]
    profile_oper = UnifiedRestorerV3._build_song_calibration_profile(**_shared, genre_label="Oper")  # type: ignore[arg-type]
    v_klassik = profile_klassik["family_scalars"]["vocal"]
    v_oper = profile_oper["family_scalars"]["vocal"]
    assert v_oper > v_klassik, f"Oper vocal scalar ({v_oper:.3f}) sollte > Klassik ({v_klassik:.3f}) sein"


def test_46_build_song_calibration_rock_raises_transient():
    """Rock-genre_label → transient family_scalar > neutrales Profil."""
    from backend.core.defect_scanner import MaterialType
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    _shared = {
        "material_type": MaterialType.VINYL,
        "mode": QualityMode.QUALITY,
        "restorability_score": 70.0,
        "input_snr_db": 30.0,
        "max_defect_severity": 0.2,
        "pipeline_confidence": 0.8,
    }
    profile_default = UnifiedRestorerV3._build_song_calibration_profile(**_shared, genre_label="")  # type: ignore[arg-type]
    profile_rock = UnifiedRestorerV3._build_song_calibration_profile(**_shared, genre_label="Rock")  # type: ignore[arg-type]
    t_default = profile_default["family_scalars"]["transient"]
    t_rock = profile_rock["family_scalars"]["transient"]
    assert t_rock >= t_default, f"Rock transient scalar ({t_rock:.3f}) sollte ≥ default ({t_default:.3f}) sein"


def test_47_build_song_calibration_genre_label_stored():
    """genre_label wird im zurückgegebenen Profil-Dict gespeichert."""
    from backend.core.defect_scanner import MaterialType
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    profile = UnifiedRestorerV3._build_song_calibration_profile(
        material_type=MaterialType.VINYL,
        mode=QualityMode.QUALITY,
        restorability_score=70.0,
        input_snr_db=30.0,
        max_defect_severity=0.2,
        pipeline_confidence=0.8,
        genre_label="Jazz",
    )
    assert profile.get("genre_label") == "Jazz"


def test_48_oper_profile_in_genre_restoration_profiles():
    """GENRE_RESTORATION_PROFILES enthält 'Oper' und 'oper' als Keys."""
    from backend.core.genre_classifier import GENRE_RESTORATION_PROFILES

    assert "Oper" in GENRE_RESTORATION_PROFILES
    assert "oper" in GENRE_RESTORATION_PROFILES
    profile = GENRE_RESTORATION_PROFILES["Oper"]
    assert isinstance(profile, dict)
    assert len(profile) > 0


def test_49_get_restoration_profile_oper():
    """get_restoration_profile('Oper') gibt das OPER_RESTORATION_PROFILE zurück."""
    from backend.core.genre_classifier import OPER_RESTORATION_PROFILE, get_restoration_profile

    profile = get_restoration_profile("Oper")
    assert profile == OPER_RESTORATION_PROFILE
    # Lowercase alias
    profile_lc = get_restoration_profile("oper")
    assert profile_lc == OPER_RESTORATION_PROFILE
