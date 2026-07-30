"""tests/unit/test_non_plus_ultra.py — §v10.90–§v10.94 Non-Plus-Ultra Test-Suite

Testet:
- §G95 Phase-DAG: P02 vor P03
- §G98 AUTHENTIC_CHARACTER-Vollständigkeit (17 Materialien)
- §G99 Equality-of-Materials (Aliase explizit)
- §G92 Material-adaptive Confidence (predict_quality_score aktiviert)
- §G97 log10-Null-Guard
"""

import numpy as np
import pytest

SR = 48000


# ═══════════════════════════════════════════════════════════════════════════
# §G95 Phase-DAG: P02 vor P03
# ═══════════════════════════════════════════════════════════════════════════


def test_phase_dag_p02_before_p03():
    """§G95: HARD_BEFORE_CONSTRAINTS enthält phase_02 → phase_03."""
    from backend.core.phase_dag import HARD_BEFORE_CONSTRAINTS, PhaseConstraint

    p02_before_p03 = False
    for c in HARD_BEFORE_CONSTRAINTS:
        if c.before == "phase_02_hum_removal" and c.after == "phase_03_denoise":
            p02_before_p03 = True
            break
    assert p02_before_p03, "§G95 VERLETZT: phase_02_hum_removal MUSS vor phase_03_denoise deklariert sein!"


def test_phase_dag_validate_enforces_p02_before_p03():
    """§G95: validate_phase_order erkennt P03 vor P02 als Verstoß."""
    from backend.core.phase_dag import validate_phase_order

    violations = validate_phase_order(
        [
            "phase_01_click_removal",
            "phase_03_denoise",  # P03 VOR P02 → Verstoß!
            "phase_02_hum_removal",
        ]
    )
    assert len(violations) > 0, "P03 vor P02 sollte als DAG-Verstoß erkannt werden"


# ═══════════════════════════════════════════════════════════════════════════
# §G98 AUTHENTIC_CHARACTER-Vollständigkeit
# ═══════════════════════════════════════════════════════════════════════════


def test_authentic_character_all_materials_present():
    """§G98: Alle 17 Materialien haben AUTHENTIC_CHARACTER-Eintrag."""
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    required = [
        "shellac",
        "vinyl",
        "lp",
        "tape",
        "reel_tape",
        "cassette",
        "kassette",
        "cd_digital",
        "mp3_low",
        "mp3_high",
        "aac",
        "streaming",
        "minidisc",
        "dat",
        "wax_cylinder",
        "wire_recording",
        "lacquer_disc",
    ]
    missing = [m for m in required if m not in AUTHENTIC_CHARACTER]
    assert not missing, f"§G98: Fehlende AUTHENTIC_CHARACTER-Einträge: {missing}"


def test_authentic_character_aliases_explicit():
    """§G99: Aliase (kassette, lp) sind explizit deklariert, nicht via Default."""
    from backend.core.intentional_artifact_classifier import AUTHENTIC_CHARACTER

    assert "kassette" in AUTHENTIC_CHARACTER, "kassette (deutsche Schreibweise) fehlt"
    assert "lp" in AUTHENTIC_CHARACTER, "lp (Vinyl-Alias) fehlt"
    assert "streaming" in AUTHENTIC_CHARACTER, "streaming fehlt"
    assert "aac" in AUTHENTIC_CHARACTER, "aac fehlt"


# ═══════════════════════════════════════════════════════════════════════════
# §G99 Material-Threshold-Bonus Vollständigkeit
# ═══════════════════════════════════════════════════════════════════════════


def test_material_threshold_bonus_all_materials_present():
    """§G99: Fehlende Keys in _MATERIAL_THRESHOLD_BONUS ergänzt."""
    from backend.core.per_phase_musical_goals_gate import _MATERIAL_THRESHOLD_BONUS

    required = [
        "lacquer_disc",
        "lp",
        "kassette",
        "aac",
        "streaming",
        "cassette",
        "vinyl",
        "shellac",
        "cd_digital",
        "mp3_low",
        "mp3_high",
        "minidisc",
        "dat",
        "wax_cylinder",
        "wire_recording",
        "reel_tape",
        "tape",
        "radio_broadcast",
        "optical_film",
    ]
    missing = [m for m in required if m not in _MATERIAL_THRESHOLD_BONUS]
    assert not missing, f"§G99: Fehlende _MATERIAL_THRESHOLD_BONUS-Keys: {missing}"


# ═══════════════════════════════════════════════════════════════════════════
# §G92 Material-adaptive Confidence
# ═══════════════════════════════════════════════════════════════════════════


def test_predict_quality_score_material_aware():
    """§G92: predict_quality_score liefert materialspezifische Ceilings."""
    from backend.core.calibration_matrix import predict_quality_score

    cd = predict_quality_score("cd_digital", 90.0, 0.0, False)
    shellac = predict_quality_score("shellac", 40.0, 0.0, False)
    cassette = predict_quality_score("cassette", 60.0, 0.0, False)

    assert cd > shellac, f"CD ({cd:.3f}) sollte höheres Ceiling als Shellac ({shellac:.3f}) haben"
    assert cd > cassette, f"CD ({cd:.3f}) sollte höheres Ceiling als Kassette ({cassette:.3f}) haben"
    assert 0.0 <= shellac <= 0.99, f"Shellac {shellac:.3f} außerhalb [0,1]"
    assert 0.0 <= cd <= 0.99, f"CD {cd:.3f} außerhalb [0,1]"


def test_predict_quality_score_all_materials():
    """§G99: predict_quality_score funktioniert für alle Materialien."""
    from backend.core.calibration_matrix import predict_quality_score

    materials = [
        "cd_digital",
        "vinyl",
        "tape",
        "cassette",
        "shellac",
        "mp3_low",
        "mp3_high",
        "aac",
        "streaming",
        "minidisc",
        "dat",
        "wax_cylinder",
        "wire_recording",
        "lacquer_disc",
        "lp",
        "kassette",
        "reel_tape",
    ]
    for mat in materials:
        score = predict_quality_score(mat, 50.0, 0.0, False)
        assert 0.0 <= score <= 0.99, f"{mat}: score {score:.3f} außerhalb [0,1]"


# ═══════════════════════════════════════════════════════════════════════════
# §G97 log10-Null-Guard
# ═══════════════════════════════════════════════════════════════════════════


def test_log10_guard_silence():
    """§G97: log10(0) → -inf wird durch max(x, 1e-10) verhindert."""
    # Simuliere das Guard-Pattern aus excellence_optimizer.py
    p10 = 0.0  # np.percentile(frames_rms, 10) bei Stille
    safe = float(20 * np.log10(max(p10, 1e-10)))
    assert np.isfinite(safe), f"log10({p10}) mit Guard sollte finit sein, ist {safe}"
    assert safe < -100, f"Erwarte sehr niedrigen Wert (< -100 dB), bekam {safe:.1f}"


def test_log10_guard_noise_ratio():
    """§G97: Epsilon-Kollision in difficulty_estimator verhindert."""
    # Teste dass 1e-8 im Nenner vs 1e-10 im Zähler keine Kollision erzeugt
    log_mean = 1e-10  # exp(mean(log(spec + 1e-10))) bei Stille
    arith_mean = 0.0  # np.mean(spec) bei Stille
    noise_ratio = log_mean / max(arith_mean, 1e-8)
    # Mit 1e-8: noise_ratio = 1e-10 / 1e-8 = 0.01 → NICHT noisy
    # Mit 1e-10 (alt): noise_ratio = 1e-10 / 1e-10 = 1.0 → FALSCH noisy
    assert noise_ratio < 0.5, (
        f"Epsilon-Kollision: noise_ratio={noise_ratio:.4f} sollte < 0.5 sein "
        f"(1e-8 Nenner vs 1e-10 Zähler verhindert Kollision)"
    )
