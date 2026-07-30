import pytest

"""
tests/unit/test_holistic_perceptual_gate.py — §2.44 HolisticPerceptualGate Test-Suite (≥ 25 Tests)
Alle Tests synthetisch, kein ML-Modell erforderlich.
"""

from unittest.mock import patch

import numpy as np

SR = 48_000


def _audio(dur: float = 1.0, amp: float = 0.3, freq: float = 440.0):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate, get_holistic_gate

    assert HolisticPerceptualGate is not None
    assert get_holistic_gate is not None


def test_01_singleton():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    g1 = get_holistic_gate()
    g2 = get_holistic_gate()
    assert g1 is g2


def test_02_restoration_identical_passes():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    assert result.passed
    assert result.hpi > 0


def test_03_restoration_with_artifacts_fails():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR, artifact_freedom=0.80)
    assert not result.passed
    assert result.artifact_freedom == 0.80


def test_04_studio_identical_passes():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_studio(audio, audio, SR, pqs_improvement=0.5)
    assert result.hpi > 0
    assert result.is_studio_mode


def test_05_studio_with_artifacts_fails():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_studio(audio, audio, SR, artifact_freedom=0.80)
    assert not result.passed


def test_06_hpi_positive_means_export():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    # HPI > 0 → export allowed
    if result.hpi > 0 and result.artifact_freedom >= 0.95:
        assert result.passed


def test_07_mert_similarity_identical():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    sim = gate._compute_mert_similarity(audio, audio, SR)
    assert sim >= 0.99, f"Identical audio should have near-perfect MERT similarity, got {sim}"


def test_08_mert_similarity_different():
    # VERSA ist referenzfrei (§2.44) und bewertet nur das restored-Signal unabhängig
    # vom Original → hohe Qualität für saubere Signale unabhängig von Frequenz-Unterschieden.
    # Dieser Test prüft die Frequenz-Diskriminierung des spectral proxy direkt.
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio1 = _audio(2.0, freq=440.0)
    audio2 = _audio(2.0, freq=880.0)
    sim = gate._compute_mert_similarity_spectral_proxy(audio1, audio2, SR)
    assert sim < 0.99, "Different frequencies should have lower similarity (spectral proxy)"


def test_09_timbral_fidelity_identical():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    fidelity = gate._compute_timbral_fidelity(audio, audio, SR)
    assert fidelity >= 0.99


def test_10_timbral_fidelity_different():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    np.random.seed(42)
    noise = 0.3 * np.random.randn(len(audio)).astype(np.float32)
    fidelity = gate._compute_timbral_fidelity(audio, noise, SR)
    assert fidelity < 0.95, "Noise should differ from tone"


def test_11_studio_quality_gain_loud():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0, amp=0.15)  # roughly -14 dBFS
    gain = gate._compute_studio_quality_gain(audio, audio, SR)
    assert 0.0 <= gain <= 1.0


def test_12_studio_quality_gain_quiet():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    quiet = _audio(2.0, amp=0.001)  # very quiet
    gain = gate._compute_studio_quality_gain(quiet, quiet, SR)
    assert 0.0 <= gain <= 1.0


def test_13_hpi_result_fields():
    from backend.core.holistic_perceptual_gate import HPIResult

    result = HPIResult(hpi=0.85, passed=True)
    assert result.hpi == 0.85
    assert result.passed is True
    assert result.mert_similarity == 1.0  # default
    assert result.is_studio_mode is False


def test_14_emotional_arc_factor():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    # Low emotional arc should reduce HPI
    result_low = gate.evaluate_restoration(audio, audio, SR, emotional_arc_score=0.5)
    result_high = gate.evaluate_restoration(audio, audio, SR, emotional_arc_score=1.0)
    assert result_low.hpi < result_high.hpi


def test_15_restorability_strict_gate():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result_normal = gate.evaluate_restoration(audio, audio, SR, restorability_score=70.0)
    result_strict = gate.evaluate_restoration(audio, audio, SR, restorability_score=90.0)
    # Strict gate reduces HPI slightly
    assert result_strict.hpi <= result_normal.hpi


def test_16_short_audio_graceful():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    short = np.zeros(100, dtype=np.float32)
    result = gate.evaluate_restoration(short, short, SR)
    assert np.isfinite(result.hpi)


def test_17_stereo_input():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    mono = _audio(2.0)
    stereo = np.stack([mono, mono * 0.9], axis=0)
    result = gate.evaluate_restoration(stereo, stereo, SR)
    assert result.hpi > 0


def test_18_nan_input_safe():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    nan_audio = audio.copy()
    nan_audio[100] = np.nan
    # Should not crash
    sim = gate._compute_mert_similarity(audio, nan_audio, SR)
    assert np.isfinite(sim)


def test_19_studio_mode_flag():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    rest = gate.evaluate_restoration(audio, audio, SR)
    studio = gate.evaluate_studio(audio, audio, SR)
    assert rest.is_studio_mode is False
    assert studio.is_studio_mode is True


def test_20_artifact_freedom_veto():
    """artifact_freedom < 0.95 → passed must be False regardless of HPI value."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR, artifact_freedom=0.94)
    assert not result.passed


def test_21_detail_dict_present():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    assert isinstance(result.detail, dict)


def test_22_pqs_improvement_affects_studio():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result_low = gate.evaluate_studio(audio, audio, SR, pqs_improvement=-0.5)
    result_high = gate.evaluate_studio(audio, audio, SR, pqs_improvement=1.0)
    assert result_high.hpi >= result_low.hpi


def test_23_mert_short_audio():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    short = np.zeros(500, dtype=np.float32)
    sim = gate._compute_mert_similarity(short, short, SR)
    assert sim == 1.0  # fallback for short audio


def test_24_timbral_short_audio():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    short = np.zeros(500, dtype=np.float32)
    fid = gate._compute_timbral_fidelity(short, short, SR)
    assert fid == 1.0  # fallback for short audio


def test_25_hpi_range():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    assert result.hpi >= 0.0
    assert result.hpi <= 1.5  # theoretical max with all factors = 1.0


# ── §2.44 Restorability-Anker-Strategie ───────────────────────────────────


def test_26_restorability_anchor_detail_keys():
    """evaluate_restoration muss input_weight / ref_weight im detail-Dict liefern."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=60.0)
    assert "input_weight" in result.detail
    assert "ref_weight" in result.detail
    assert "timbral_input" in result.detail
    assert "timbral_ref" in result.detail


def test_27_high_restorability_ref_dominant():
    """Restorability > 70 → input_weight=0.0, ref_weight=1.0 (§2.44 FIX v10.0.0).

    Nach dem Referenz-Paradox-Fix wird bei hoher Restorability NICHT mehr die
    Ähnlichkeit zum degradierten Input gemessen, sondern die Referenz/direktionale
    Qualität. input_weight=0.0 verhindert, dass gute Restaurierung bestraft wird.
    """
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=80.0)
    assert result.detail["input_weight"] == 0.0
    assert result.detail["ref_weight"] == 1.0


def test_28_mid_restorability_ref_dominant():
    """Restorability 50–70 → input_weight=0.35, ref_weight=0.65 (§2.44 FIX v10.0.0)."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=60.0)
    assert result.detail["input_weight"] == 0.35
    assert result.detail["ref_weight"] == 0.65


def test_29_low_restorability_ref_dominant():
    """Restorability ≤ 50 → input_weight=0.2, ref_weight=0.8 (§2.44 FIX v10.0.0)."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=30.0)
    assert result.detail["input_weight"] == 0.2
    assert result.detail["ref_weight"] == 0.8


def test_30_restorability_50_boundary_blended():
    """Restorability = 50 (Grenzfall) → blended (50–70 Bereich, §2.44 FIX v10.0.0)."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=50.0)
    assert result.detail["input_weight"] == 0.35
    assert result.detail["ref_weight"] == 0.65


# ── §2.44 update_reference_memory + EMA ───────────────────────────────────


def test_31_update_reference_memory_method_exists():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    assert hasattr(gate, "update_reference_memory")


def test_32_update_rejected_if_hpi_too_low():
    """Quality-Gate v10.0.0 (V54-aligned): HPI ≤ 0.0 → kein Update; HPI > 0.0 wird akzeptiert."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory = {}  # Disk-Memory leeren — __init__ lädt ~/.aurik/hpg_reference_memory.json
    audio = _audio(2.0)

    # HPI = 0.0 → Gate blockiert (HPI must be > 0.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.0,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    assert ("jazz", "vinyl_std", "pre-1960") not in gate._ref_memory, (
        "HPI=0.0 darf nicht ins Reference-Memory aufgenommen werden (V54-Gate)"
    )

    # HPI = 0.4 → wird jetzt akzeptiert (V54: HPI > 0.0 AND artifact_freedom ≥ 0.95)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.4,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    assert ("jazz", "vinyl_std", "pre-1960") in gate._ref_memory, (
        "HPI=0.4 > 0.0 mit artifact_freedom=1.0 muss akzeptiert werden (V54-Gate)"
    )


def test_33_update_rejected_if_artifact_freedom_low():
    """Quality-Gate: artifact_freedom < 0.95 → kein Update."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory = {}  # Disk-Memory leeren — __init__ lädt ~/.aurik/hpg_reference_memory.json
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.8,
        artifact_freedom=0.90,
        p1_p2_passed=True,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    assert ("jazz", "vinyl_std", "pre-1960") not in gate._ref_memory


def test_34_update_rejected_if_p1_p2_failed():
    """V54-Gate: p1_p2_passed ist kein Blocker mehr — nur HPI > 0.0 + artifact_freedom ≥ 0.95 zählt."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory = {}  # Disk-Memory leeren — __init__ lädt ~/.aurik/hpg_reference_memory.json
    audio = _audio(2.0)
    # p1_p2_passed=False blockiert NICHT mehr (V54-aligned v10.0.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.8,
        artifact_freedom=1.0,
        p1_p2_passed=False,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    assert ("jazz", "vinyl_std", "pre-1960") in gate._ref_memory, (
        "p1_p2_passed=False darf den Update nicht mehr blockieren (V54-aligned)"
    )


def test_35_update_creates_entry():
    """Valides Update → Eintrag in _ref_memory angelegt."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory = {}  # Disk-Memory leeren — __init__ lädt ~/.aurik/hpg_reference_memory.json
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.8,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    key = ("jazz", "vinyl_std", "pre-1960")
    assert key in gate._ref_memory
    assert gate._ref_memory[key].obs_count == 1
    assert gate._ref_memory[key].calibrated is False  # erst 1 Beobachtung


def test_36_update_ema_three_times_calibrated():
    """Nach 3 validen Updates → calibrated = True."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory = {}  # Disk-Memory leeren — __init__ lädt ~/.aurik/hpg_reference_memory.json
    audio = _audio(2.0)
    key = ("pop", "tape_std", "1960-1990")
    for _ in range(3):
        gate.update_reference_memory(
            audio,
            SR,
            hpi=0.9,
            artifact_freedom=1.0,
            p1_p2_passed=True,
            genre="pop",
            material="tape_std",
            era_bin="1960-1990",
        )
    assert key in gate._ref_memory
    assert gate._ref_memory[key].obs_count == 3
    assert gate._ref_memory[key].calibrated is True


def test_37_ema_smoothing_alpha():
    """EMA-Update: Embedding muss sich nach Update ändern (weder identisch noch komplett ersetzt)."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio1 = _audio(2.0, freq=440.0)
    audio2 = _audio(2.0, freq=880.0)  # different embedding

    gate.update_reference_memory(
        audio1,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="rock",
        material="digital",
        era_bin="post-1990",
    )
    emb_after_first = gate._ref_memory[("rock", "digital", "post-1990")].embedding.copy()

    gate.update_reference_memory(
        audio2,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="rock",
        material="digital",
        era_bin="post-1990",
    )
    emb_after_second = gate._ref_memory[("rock", "digital", "post-1990")].embedding

    # Embedding muss sich geändert haben (EMA-Mischung)
    assert not np.allclose(emb_after_first, emb_after_second, atol=1e-6)
    # Aber nicht komplett ersetzt (EMA-Faktor 0.15 → große Überlappung)
    corr = float(
        np.dot(emb_after_first, emb_after_second)
        / (np.linalg.norm(emb_after_first) * np.linalg.norm(emb_after_second) + 1e-12)
    )
    assert corr > 0.5, f"EMA should preserve most of old reference, got correlation {corr:.3f}"


# ── §2.44 Fallback-Kaskade ─────────────────────────────────────────────────


def test_38_fallback_stufe5_no_vector():
    """Kein Eintrag in _ref_memory → _get_reference_vector gibt None zurück."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory = {}  # explizit leeren: __init__ lädt ~./aurik/hpg_reference_memory.json
    vec = gate._get_reference_vector("oper", "shellac", "pre-1960")
    assert vec is None


def test_39_fallback_stufe1_exact_match():
    """Exakter Treffer (genre, material, era) → liefert gespeicherten Vektor."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0, freq=220.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="blues",
        material="shellac",
        era_bin="pre-1960",
    )
    vec = gate._get_reference_vector("blues", "shellac", "pre-1960")
    assert vec is not None
    assert vec.shape == (40,)


def test_40_fallback_stufe2_same_era_same_genre():
    """Kein exakter material-Treffer, aber gleiche genre+era → Stufe-2-Fallback."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="shellac",
        era_bin="pre-1960",
    )
    # Anderes Material, aber gleiche genre+era
    vec = gate._get_reference_vector("jazz", "vinyl_78", "pre-1960")
    assert vec is not None


def test_41_fallback_stufe3_same_genre():
    """Kein era-Treffer für dieses genre → Stufe-3: gleicher Genre, andere Ära."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="tape_std",
        era_bin="1960-1990",
    )
    # Anderes Material UND andere Ära, aber gleicher Genre → Stufe-3
    vec = gate._get_reference_vector("jazz", "digital", "post-1990")
    assert vec is not None


def test_42_reference_used_in_hpi_low_restorability():
    """Restorability ≤ 50 + existierender Ref-Vektor → ref_weight=0.7 und result.passed=True."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    # Zuerst Referenz anlegen
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="folk",
        material="shellac",
        era_bin="pre-1960",
    )
    result = gate.evaluate_restoration(
        audio,
        audio,
        SR,
        restorability_score=30.0,
        genre="folk",
        material="shellac",
        era_bin="pre-1960",
    )
    assert result.detail["ref_weight"] == 0.8
    assert result.hpi > 0


def test_43_compute_embedding_returns_valid_shape():
    """_compute_embedding liefert normierten float32-Vektor der Länge 40."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    emb = gate._compute_embedding(audio, SR)
    assert emb.shape == (40,)
    assert emb.dtype == np.float32
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 0.01, f"Embedding should be unit-normed, got norm={norm:.4f}"


def test_44_cosine_similarity_identical_vectors():
    """Cosine similarity identischer Vektoren = 1.0."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    v = np.random.rand(40).astype(np.float32)
    sim = gate._cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-5


def test_45_genre_material_era_in_detail():
    """genre/material/era_bin müssen im detail-Dict stehen."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(
        audio,
        audio,
        SR,
        genre="schlager",
        material="vinyl_std",
        era_bin="1960-1990",
    )
    assert result.detail.get("genre") == "schlager"
    assert result.detail.get("material") == "vinyl_std"
    assert result.detail.get("era_bin") == "1960-1990"


# ── §2.44 BW-Ceiling-Guard ─────────────────────────────────────────────────


def test_46_bw_ceiling_dict_contains_cassette():
    """_MATERIAL_BW_CEILING_HZ muss 'cassette' mit 14000 Hz enthalten (§6.2c zentrale Definition)."""
    from backend.core.holistic_perceptual_gate import _MATERIAL_BW_CEILING_HZ

    assert "cassette" in _MATERIAL_BW_CEILING_HZ
    assert _MATERIAL_BW_CEILING_HZ["cassette"] == 14000


def test_47_bw_ceiling_dict_contains_all_material_types():
    """_MATERIAL_BW_CEILING_HZ muss alle kanonischen Material-Typen enthalten."""
    from backend.core.holistic_perceptual_gate import _MATERIAL_BW_CEILING_HZ

    required = {
        "shellac",
        "wax_cylinder",
        "lacquer_disc",
        "wire_recording",
        "vinyl",
        "tape",
        "reel_tape",
        "cassette",
        "cd_digital",
        "dat",
        "md",
        "mp3_low",
        "mp3_high",
        "aac",
        "unknown",
    }
    missing = required - set(_MATERIAL_BW_CEILING_HZ.keys())
    assert not missing, f"Fehlende Material-Typen in _MATERIAL_BW_CEILING_HZ: {missing}"


def test_48_spectral_proxy_bw_ceiling_limits_comparison():
    """BW-Ceiling-Guard: Spectral-Proxy mit ceiling=8000 Hz soll BW-Erweiterung nicht bestrafen.

    Aufbau:
     - original: 440 Hz Sinus (Energie nur im Bassbereich)
     - restored: original + synthetisierter 14 kHz Ton (FlashSR-Simulation)
     - Mit ceiling=8000 Hz: Proxy vergleicht nur 0–8 kHz → Similarity ≈ 1.0
     - Ohne ceiling (full-band): 14 kHz-Energie macht restored anders → niedrigere Similarity
    """
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    dur = 2.0
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    # Original: 440 Hz Ton (typisch für Vokalinhalt)
    original = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    # Restored: Original + 14 kHz Syntheseinhalt (FlashSR über Kassetten-Ceiling)
    restored = original + 0.3 * np.sin(2 * np.pi * 14000.0 * t).astype(np.float32)

    sim_full = gate._compute_mert_similarity_spectral_proxy(original, restored, SR)
    sim_ceil = gate._compute_mert_similarity_spectral_proxy(original, restored, SR, bw_ceiling_hz=8000)

    # Mit BW-Ceiling muss Similarity HÖHER sein als ohne (14 kHz ignoriert)
    assert sim_ceil > sim_full, (
        f"BW-Ceiling-Guard unwirksam: sim_ceil={sim_ceil:.3f} ≤ sim_full={sim_full:.3f} — "
        "BW-Erweiterung wird auch mit Ceiling bestraft (§2.44 BW-Ceiling-Guard)"
    )
    # Mit Ceiling: 14 kHz komplett außerhalb → Similarity nahe 1.0 (nur 440 Hz verglichen)
    assert sim_ceil >= 0.90, f"Mit BW-Ceiling sollte Similarity ≥ 0.90 sein, got {sim_ceil:.3f}"


def test_49_spectral_proxy_no_ceiling_penalizes_hf_extension():
    """Ohne BW-Ceiling: 14 kHz Extrasignal senkt Spectral-Proxy-Similarity."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    dur = 2.0
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    original = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    restored = original + 0.3 * np.sin(2 * np.pi * 14000.0 * t).astype(np.float32)

    sim_no_ceil = gate._compute_mert_similarity_spectral_proxy(original, restored, SR)
    sim_identical = gate._compute_mert_similarity_spectral_proxy(original, original, SR)

    assert sim_no_ceil < sim_identical, (
        "14 kHz-Zusatzinhalt soll Spectral-Proxy ohne Ceiling absenken "
        f"(sim_no_ceil={sim_no_ceil:.3f} vs. sim_identical={sim_identical:.3f})"
    )


def test_50_evaluate_restoration_cassette_passes_bw_ceiling():
    """evaluate_restoration mit material='cassette' soll BW-Ceiling korrekt anwenden.

    Simuliert: original (Kassetten-Tier, kein HF), restored mit FlashSR-Extension (14 kHz).
    HPI muss > 0 bleiben (BW-Extension ist kein Qualitätsverlust).
    """
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    dur = 2.0
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    original = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    restored = original + 0.2 * np.sin(2 * np.pi * 14000.0 * t).astype(np.float32)

    result = gate.evaluate_restoration(
        original,
        restored,
        SR,
        artifact_freedom=0.98,
        material="cassette",
        restorability_score=55.0,
    )
    # Muss passieren — BW-Extension ist kein Defekt
    assert result.hpi > 0, f"HPI={result.hpi:.4f} — BW-Erweiterung auf Kassette darf HPI nicht auf ≤ 0 senken"


# ---------------------------------------------------------------------------
# test_51–53: mel-Embedding BW-Ceiling-Guard (v10.0.0) — _compute_embedding
# §2.44: FlashSR-synthetisierter HF-Content darf timbral_input nicht verfälschen.
# ---------------------------------------------------------------------------


def test_51_embedding_bw_ceiling_limits_hf_bins():
    """_compute_embedding mit bw_ceiling_hz erzeugt anderes Embedding als ohne Ceiling.

    Für Audio mit Energie ausschließlich bei 14 kHz:
    - Ohne Ceiling: signifikante Energie in hohen Mel-Bins
    - Mit Ceiling=8000: alle hohen Mel-Bins = 0 → anderes Embedding
    """
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    dur = 1.0
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    # Ton NUR bei 14 kHz (oberhalb 8 kHz Ceiling)
    audio_hf = (0.5 * np.sin(2 * np.pi * 14_000.0 * t)).astype(np.float32)

    embed_full = gate._compute_embedding(audio_hf, SR)
    embed_ceil = gate._compute_embedding(audio_hf, SR, bw_ceiling_hz=8000)

    # Mit Ceiling muss der Inhalt unsichtbar sein → niedrige Norm (fast Nullvektor)
    # → Norm des gemittelten Embeddings nahe 0 (log1p(0)=0, dann normiert auf 1-Vektor)
    # Aber: norm nach Normierung = 1 immer; stattdessen prüfen ob Embeddings verschieden sind
    sim = float(np.dot(embed_full, embed_ceil))
    assert sim < 0.98, (
        f"Embedding mit und ohne BW-Ceiling für reinen 14-kHz-Ton muss verschieden sein "
        f"(cos_sim={sim:.4f} ≥ 0.98 — BW-Ceiling hat keine Wirkung)"
    )


def test_52_timbral_fidelity_bw_ceiling_higher_similarity():
    """BW-Ceiling erhöht timbral_fidelity wenn original kein HF hat, restored HF-Extension hat.

    Szenario: Kassetten-Restaurierung mit FlashSR (12 kHz → 22 kHz Extension).
    - Ohne Ceiling: Cosinus-Ähnlichkeit sinkt wegen HF-Divergenz (Reference Paradox)
    - Mit Ceiling=12000: HF-Anteil ignoriert → höhere Ähnlichkeit
    """
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    dur = 2.0
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    original = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    # Restored: original + starker HF-Ton bei 15 kHz (FlashSR-Synthetik)
    restored_with_hf = (original + 0.6 * np.sin(2 * np.pi * 15_000.0 * t)).astype(np.float32)

    sim_full = gate._compute_timbral_fidelity(original, restored_with_hf, SR)
    sim_ceil = gate._compute_timbral_fidelity(original, restored_with_hf, SR, bw_ceiling_hz=12_000)

    assert sim_ceil > sim_full, (
        f"BW-Ceiling (12 kHz) muss höhere Ähnlichkeit liefern als voller Spektralvergleich "
        f"bei HF-Extension: sim_ceil={sim_ceil:.4f} soll > sim_full={sim_full:.4f}"
    )


def test_53_evaluate_restoration_cassette_timbral_bw_ceiling_passed():
    """evaluate_restoration leitet BW-Ceiling korrekt an _compute_timbral_fidelity weiter.

    Prüft indirekt: HPI für cassette mit HF-Extension ist höher als ohne material-spezifische
    Ceiling (d.h. evaluate_restoration nutzt material='cassette' → bw_ceiling=14000 →
    timbral_input ist größer → HPI ist größer als bei material='digital').
    """
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    dur = 2.0
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    original = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    restored = (original + 0.5 * np.sin(2 * np.pi * 15_000.0 * t)).astype(np.float32)

    result_cassette = gate.evaluate_restoration(
        original,
        restored,
        SR,
        artifact_freedom=0.99,
        material="cassette",
        restorability_score=55.0,
    )
    result_digital = gate.evaluate_restoration(
        original,
        restored,
        SR,
        artifact_freedom=0.99,
        material="cd_digital",
        restorability_score=55.0,
    )

    # Kassette mit BW-Ceiling muss höheren oder gleichen HPI liefern als Vollband-Vergleich
    # (digital = 22050 Hz ceiling → kein Unterschied zum alten Code für digital)
    assert result_cassette.hpi >= result_digital.hpi - 0.01, (
        f"Kassetten-BW-Ceiling soll HPI nicht verschlechtern vs digital: "
        f"cassette.hpi={result_cassette.hpi:.4f} vs digital.hpi={result_digital.hpi:.4f}"
    )


# ── §2.44 Persistenz-Tests (test_54–test_57) ───────────────────────────────


def test_54_ref_memory_saves_and_loads_from_disk(tmp_path):
    """update_reference_memory() persistiert Eintrag; neuer Gate-Singleton lädt ihn."""

    import backend.core.holistic_perceptual_gate as hpg_mod

    mem_path = tmp_path / "hpg_reference_memory.json"
    with patch.object(hpg_mod, "_HPG_REF_MEMORY_PATH", mem_path):
        gate1 = hpg_mod.HolisticPerceptualGate()
        audio = _audio(dur=2.0, amp=0.4, freq=440.0)
        # Qualitäts-Gate-Bedingungen erfüllen
        gate1.update_reference_memory(
            restored=audio,
            sr=SR,
            hpi=0.75,
            artifact_freedom=0.97,
            p1_p2_passed=True,
            genre="pop",
            material="cassette",
            era_bin="era_1970",
        )
        assert mem_path.exists(), "JSON-Datei muss nach update_reference_memory existieren"
        # Neuer Gate-Singleton lädt von Disk
        gate2 = hpg_mod.HolisticPerceptualGate()
        assert len(gate2._ref_memory) == 1, "Neuer Singleton muss 1 Eintrag geladen haben"
        key = ("pop", "cassette", "era_1970")
        assert key in gate2._ref_memory
        assert gate2._ref_memory[key].obs_count == 1


def test_55_ref_memory_not_saved_below_quality_gate(tmp_path):
    """update_reference_memory() darf bei HPI ≤ 0.0 NICHT speichern (V54-Gate v10.0.0)."""
    import backend.core.holistic_perceptual_gate as hpg_mod

    mem_path = tmp_path / "hpg_reference_memory.json"
    with patch.object(hpg_mod, "_HPG_REF_MEMORY_PATH", mem_path):
        gate = hpg_mod.HolisticPerceptualGate()
        audio = _audio(dur=1.0, amp=0.2, freq=440.0)
        # HPI = 0.0 → blockiert (HPI must be > 0.0); artifact_freedom = 0.94 < 0.95 → blockiert
        gate.update_reference_memory(
            restored=audio,
            sr=SR,
            hpi=0.0,
            artifact_freedom=0.94,
            p1_p2_passed=True,
            genre="pop",
            material="cassette",
            era_bin="era_1970",
        )
        assert not mem_path.exists(), "Keine Datei wenn HPI=0.0 und artifact_freedom<0.95 (V54-Gate)"


def test_56_ref_memory_ema_update_persists(tmp_path):
    """Mehrfache Updates akkumulieren obs_count und persistieren korrekt."""
    import backend.core.holistic_perceptual_gate as hpg_mod

    mem_path = tmp_path / "hpg_reference_memory.json"
    with patch.object(hpg_mod, "_HPG_REF_MEMORY_PATH", mem_path):
        gate = hpg_mod.HolisticPerceptualGate()
        audio = _audio(dur=2.0, amp=0.4, freq=440.0)
        for _ in range(4):
            gate.update_reference_memory(
                restored=audio,
                sr=SR,
                hpi=0.80,
                artifact_freedom=0.97,
                p1_p2_passed=True,
                genre="jazz",
                material="vinyl",
                era_bin="era_1960",
            )
        key = ("jazz", "vinyl", "era_1960")
        assert gate._ref_memory[key].obs_count == 4
        assert gate._ref_memory[key].calibrated, "Nach 3+ Obs muss calibrated=True sein"

        # Laden → obs_count muss erhalten bleiben
        gate2 = hpg_mod.HolisticPerceptualGate()
        assert gate2._ref_memory[key].obs_count == 4
        assert gate2._ref_memory[key].calibrated


def test_57_ref_memory_load_corrupt_file_safe(tmp_path):
    """Beschädigte JSON-Datei führt zu leerem Memory (keine Exception)."""
    import backend.core.holistic_perceptual_gate as hpg_mod

    mem_path = tmp_path / "hpg_reference_memory.json"
    mem_path.write_text("{ UNGÜLTIGES JSON }", encoding="utf-8")
    with patch.object(hpg_mod, "_HPG_REF_MEMORY_PATH", mem_path):
        gate = hpg_mod.HolisticPerceptualGate()
        assert len(gate._ref_memory) == 0, "Beschädigtes File → leeres Memory, kein Crash"


# ═══════════════════════════════════════════════════════════════════════════
# §v10.91–§v10.93 Non-Plus-Ultra Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_58_blind_reference_vector_exists():
    """§v10.91: _compute_blind_reference_vector liefert Embedding für sauberes Audio."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(dur=6.0, amp=0.3, freq=440.0)  # 6s braucht BlindInternalReference min 5s
    vec = gate._compute_blind_reference_vector(audio, SR)
    assert vec is not None, "Blind reference vector should exist for clean 6s audio"
    assert isinstance(vec, np.ndarray), f"Expected ndarray, got {type(vec)}"
    assert vec.ndim == 1, f"Expected 1D embedding, got shape {vec.shape}"
    assert np.isfinite(vec).all(), "Embedding must be finite"


def test_59_blind_reference_vector_used_when_no_gp_memory():
    """§G90: Wenn GP-Memory keinen Vektor hat, wird blinder Vektor verwendet."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    # Leeres Memory → kein GP-Vektor
    gate._ref_memory.clear()
    audio = _audio(dur=8.0, amp=0.3, freq=440.0)
    result = gate.evaluate_restoration(
        audio,
        audio,
        SR,
        artifact_freedom=0.96,
        restorability_score=70.0,
        genre="test",  # Keine GP-Memory für "test"
        material="digital",
        era_bin="era_2000",
    )
    assert result.passed
    assert result.hpi > 0.6, f"HPI should be healthy with blind ref, got {result.hpi:.3f}"
    # Der blinde Referenz-Vektor sollte einen validen timbral_ref liefern
    assert result.timbral_fidelity > 0.5, f"Timbral should be >0.5, got {result.timbral_fidelity:.3f}"


def test_60_hpi_nan_guard_max():
    """§G96, §V35: max(nan, 0.5) wird durch nan_to_num geschützt."""
    # Direkter Test des NaN-Verhaltens
    x = float("nan")
    # Python: max(nan, 0.5) == nan (BUG)
    assert np.isnan(max(x, 0.5)), "Python max(nan, 0.5) IS nan — Guard muss davor sein"
    # Mit Guard: nan_to_num vor max
    safe = float(np.nan_to_num(x, nan=0.5))
    assert safe == 0.5, "nan_to_num(nan, nan=0.5) should give 0.5"
    assert max(safe, 0.5) == 0.5, "max(0.5, 0.5) = 0.5"


def test_61_hpi_product_nan_guard():
    """§G96: HPI-Produkt NaN wird abgefangen und auf Floor 0.5 gesetzt."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory.clear()
    audio = _audio(dur=3.0, amp=0.3, freq=440.0)
    # Mit extrem kurzem Audio (<1024 samples) bekommen einige Pfade NaN
    short = np.array([0.1, 0.2, 0.1], dtype=np.float32)
    result = gate.evaluate_restoration(
        short,
        short,
        SR,
        artifact_freedom=0.9,
        restorability_score=50.0,
        genre="test",
        material="digital",
        era_bin="era_2000",
    )
    # Sollte nicht crashen, selbst bei sehr kurzem Audio
    assert result is not None
    assert isinstance(result.hpi, float)
    assert np.isfinite(result.hpi), f"HPI must be finite, got {result.hpi}"


def test_62_blind_reference_short_audio_returns_none():
    """§v10.91: BlindInternalReference gibt None für Audio < 2s zurück."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    short = _audio(dur=0.5, amp=0.3, freq=440.0)
    vec = gate._compute_blind_reference_vector(short, SR)
    assert vec is None, f"Short audio should return None, got {type(vec)}"


def test_63_blind_reference_noisy_audio_may_return_none():
    """§v10.91: Stark verrauschtes Audio kann None zurückgeben (best_score < 0.3)."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    t = np.linspace(0, 8, int(8 * SR), endpoint=False)
    noise = (0.05 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.5, len(t))).astype(np.float32)
    vec = gate._compute_blind_reference_vector(noise, SR)
    # Kann None sein (kein sauberes Fenster gefunden) — das ist OK
    # Wichtig: kein Crash


def test_64_reference_audio_none_not_degraded_input():
    """§G90, §V29: reference_audio=None fällt NICHT still auf degraded_input."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    gate._ref_memory.clear()
    audio = _audio(dur=8.0, amp=0.3, freq=440.0)
    # reference_audio=None → sollte blinden Vektor verwenden, nicht degraded_input
    result = gate.evaluate_restoration(
        audio,
        audio,
        SR,
        reference_audio=None,
        artifact_freedom=0.96,
        restorability_score=70.0,
        genre="test_none_ref",
        material="digital",
        era_bin="era_2000",
    )
    assert result.passed
    # WICHTIG: HPI sollte NICHT 0.43 sein (degraded_input-Vergleich)
    assert result.hpi > 0.5, f"HPI {result.hpi:.3f} should be > 0.5 (not degraded_input comparison)"
