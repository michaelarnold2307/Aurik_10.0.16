from __future__ import annotations

"""Tests für backend/core/musical_goals/transient_energy_metric.py (§1.4.6)."""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stable_audio(n: int = 48000) -> np.ndarray:
    """Kontinuierliches Sinus-Signal ohne klare Onsets."""
    t = np.linspace(0, 1.0, n, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t)  # type: ignore[no-any-return]


def _make_audio_with_onsets(sr: int = 48000, n_onsets: int = 5) -> np.ndarray:
    """Signal mit klar definierten Impulsen (Onsets)."""
    audio = np.zeros(sr, dtype=np.float32)
    for i in range(n_onsets):
        onset = int((i + 0.5) / n_onsets * sr)
        audio[onset : onset + int(0.002 * sr)] = 0.8  # 2ms Impuls
    return audio


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_singleton():
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    a = get_transient_energy_metric()
    b = get_transient_energy_metric()
    assert a is b


# ---------------------------------------------------------------------------
# Material-Floor-Lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "material,expected_min",
    [
        ("shellac", 0.70),
        ("vinyl", 0.75),
        ("cd_digital", 0.80),
        ("mp3_low", 0.77),
        ("unknown", 0.77),
    ],
)
def test_material_floor_lookup(material, expected_min):
    from backend.core.musical_goals.transient_energy_metric import (
        get_transient_energy_material_floor,
    )

    floor = get_transient_energy_material_floor(material)
    assert floor >= expected_min, f"Material '{material}' floor {floor} < erwartetes Minimum {expected_min}"
    assert floor <= 1.0


# ---------------------------------------------------------------------------
# PHASE_GOAL_EXCLUSIONS_TRANSIENT_ENERGIE
# ---------------------------------------------------------------------------


def test_phase_exclusions_set_has_required_phases():
    from backend.core.musical_goals.transient_energy_metric import (
        PHASE_GOAL_EXCLUSIONS_TRANSIENT_ENERGIE,
    )

    assert "phase_18_nmf_separation" in PHASE_GOAL_EXCLUSIONS_TRANSIENT_ENERGIE
    # phase_26 in irgendeiner Variante (transient_shaper oder dynamic_range_expansion)
    assert any("phase_26" in p for p in PHASE_GOAL_EXCLUSIONS_TRANSIENT_ENERGIE)


def test_phase_exclusions_is_frozenset():
    from backend.core.musical_goals.transient_energy_metric import (
        PHASE_GOAL_EXCLUSIONS_TRANSIENT_ENERGIE,
    )

    assert isinstance(PHASE_GOAL_EXCLUSIONS_TRANSIENT_ENERGIE, frozenset)


# ---------------------------------------------------------------------------
# measure_transient_energy: stille Eingabe
# ---------------------------------------------------------------------------


def test_silence_is_valid_false():
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    silence = np.zeros(sr, dtype=np.float32)
    result = get_transient_energy_metric().measure_transient_energy(audio_input=silence, audio_restored=silence, sr=sr)
    # Stille → keine Onsets → is_valid=False (< MIN_ONSETS)
    assert result["is_valid"] is False
    # Score sollte im Fallback 1.0 sein
    assert result["transient_energy_score"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# measure_transient_energy: perfekte Bewahrung
# ---------------------------------------------------------------------------


def test_perfect_preservation_score_one():
    """Wenn input == restored → alle Ratios=1.0 → Score=1.0."""
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    audio = _make_audio_with_onsets(sr, n_onsets=8)
    result = get_transient_energy_metric().measure_transient_energy(
        audio_input=audio, audio_restored=audio.copy(), sr=sr
    )
    if result["is_valid"]:
        assert result["transient_energy_score"] >= 0.95, (
            f"Perfekte Bewahrung sollte Score ≈ 1.0 geben, bekam {result['transient_energy_score']}"
        )


# ---------------------------------------------------------------------------
# measure_transient_energy: stark gedämpfter Output
# ---------------------------------------------------------------------------


def test_attenuated_restored_lower_score():
    """restored = input * 0.1 (−20 dB) → Score deutlich < 1.0."""
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    audio = _make_audio_with_onsets(sr, n_onsets=8)
    attenuated = audio * 0.1
    result = get_transient_energy_metric().measure_transient_energy(audio_input=audio, audio_restored=attenuated, sr=sr)
    if result["is_valid"]:
        assert result["transient_energy_score"] < 0.8, (
            f"Stark gedämpfter Restored sollte niedrigen Score geben, bekam {result['transient_energy_score']}"
        )


# ---------------------------------------------------------------------------
# measure_transient_energy: Output-Dict-Struktur
# ---------------------------------------------------------------------------


def test_result_dict_structure():
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    audio = _make_audio_with_onsets(sr, n_onsets=5)
    result = get_transient_energy_metric().measure_transient_energy(
        audio_input=audio, audio_restored=audio.copy(), sr=sr
    )
    required_keys = {
        "transient_energy_score",
        "per_onset_ratios",
        "onset_positions_samples",
        "n_onsets_detected",
        "material_floor",
        "is_valid",
    }
    for key in required_keys:
        assert key in result, f"Pflichtfeld '{key}' fehlt im Ergebnis-Dict"
    assert 0.0 <= result["transient_energy_score"] <= 1.0
    assert isinstance(result["per_onset_ratios"], list)
    assert isinstance(result["n_onsets_detected"], int)
    assert isinstance(result["is_valid"], bool)


# ---------------------------------------------------------------------------
# blend_onset_regions: Shape-Erhaltung
# ---------------------------------------------------------------------------


def test_blend_onset_regions_shape():
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    audio_orig = _make_audio_with_onsets(sr, n_onsets=4)
    audio_proc = audio_orig * 0.5
    onset_samples = [int(i / 4 * sr) for i in range(4)]

    blended = get_transient_energy_metric().blend_onset_regions(
        audio_original=audio_orig,
        audio_processed=audio_proc,
        onset_samples=onset_samples,
        sr=sr,
        blend_factor=0.5,
    )
    assert blended.shape == audio_orig.shape, (
        f"blend_onset_regions: Shape-Verlust: {audio_orig.shape} → {blended.shape}"
    )


def test_blend_onset_regions_no_nan():
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    audio = _make_audio_with_onsets(sr)
    blended = get_transient_energy_metric().blend_onset_regions(
        audio_original=audio,
        audio_processed=audio * 0.7,
        onset_samples=[int(0.2 * sr), int(0.5 * sr), int(0.8 * sr)],
        sr=sr,
        blend_factor=0.3,
    )
    assert not np.any(np.isnan(blended))
    assert not np.any(np.isinf(blended))


def test_blend_onset_regions_clipped():
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    audio = _make_audio_with_onsets(sr) * 0.9
    blended = get_transient_energy_metric().blend_onset_regions(
        audio_original=audio,
        audio_processed=audio,
        onset_samples=[int(0.3 * sr)],
        sr=sr,
        blend_factor=1.0,
    )
    assert np.max(np.abs(blended)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Stereo-Kompatibilität
# ---------------------------------------------------------------------------


def test_measure_stereo_input():
    from backend.core.musical_goals.transient_energy_metric import get_transient_energy_metric

    sr = 48000
    audio = np.stack([_make_audio_with_onsets(sr), _make_audio_with_onsets(sr)], axis=0)  # (2, T)
    result = get_transient_energy_metric().measure_transient_energy(
        audio_input=audio, audio_restored=audio.copy(), sr=sr
    )
    assert "transient_energy_score" in result
    assert not np.isnan(result["transient_energy_score"])
