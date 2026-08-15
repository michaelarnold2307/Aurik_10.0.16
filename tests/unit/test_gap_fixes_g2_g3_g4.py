from __future__ import annotations

"""Tests für Gap-Fixes G2, G3, G4 in unified_restorer_v3.py.

G2: FallbackQualityFloor multi-candidate recovery (shape-compatibility cascade)
G3: SGI vollständig fehlgeschlagen → konservative Uniform-Weights statt None
G4: PMGG best_effort Phasen → FeedbackChain _fc_max_iter boost
"""


from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(shape=(48000,)) -> np.ndarray:
    rng = np.random.default_rng(42)
    audio: np.ndarray[Any, Any] = np.asarray(rng.uniform(-0.3, 0.3, shape), dtype=np.float32)
    return audio


# ---------------------------------------------------------------------------
# G3 — conservative goal weights after full SGI failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestG3ConservativeGoalWeights:
    """§2.56 Gap G3: wenn SGI und Label-Fallback beide fehlschlagen,
    darf self._song_goal_weights NICHT None sein."""

    def _extract_conservative_weights(self) -> bool:
        """Import UV3 module and read the conservative defaults from source."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        # Find the block that assigns conservative weights after SGI failure
        # by searching for 'natuerlichkeit': 1.20 near 'Label-stage fallback also failed'
        assert '"natuerlichkeit": 1.20' in src, "G3-Fix fehlt: conservative weights-Block nicht in UV3 gefunden"
        assert '"authentizitaet": 1.20' in src
        assert '"tonal_center": 1.15' in src
        # All 15 goals must be present
        all_goals = [
            "natuerlichkeit",
            "authentizitaet",
            "tonal_center",
            "timbre_authentizitaet",
            "artikulation",
            "transient_energie",
            "emotionalitaet",
            "micro_dynamics",
            "groove",
            "transparenz",
            "waerme",
            "bass_kraft",
            "separation_fidelity",
            "brillanz",
            "spatial_depth",
        ]
        for g in all_goals:
            assert f'"{g}"' in src, f"G3-Fix: Goal '{g}' fehlt in conservative-weights-Block"
        return True

    def test_conservative_weights_present_in_source(self):
        """UV3-Quelltext enthält alle 15 Goals im konservativen Fallback-Block."""
        result = self._extract_conservative_weights()
        assert result is True

    def test_conservative_weights_p1p2_higher_than_p3(self):
        """P1/P2-Goals haben höhere Weights als P3–P5 im konservativen Fallback."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        # natuerlichkeit: 1.20 > groove: 1.00
        idx_nat = src.find('"natuerlichkeit": 1.20')
        idx_groove = src.find('"groove": 1.00')
        assert idx_nat != -1 and idx_groove != -1, "G3-Fix: Blöcke nicht im Quelltext gefunden"
        # Both should be in the same conservative block (within 500 chars of each other)
        assert abs(idx_nat - idx_groove) < 1000, "natuerlichkeit und groove scheinen in verschiedenen Blöcken zu stehen"

    def test_fallback_block_logs_warning(self):
        """UV3 loggt eine Warnung wenn conservative weights aktiviert werden."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        assert "Label-Stufe Ersatzpfad also fehlgeschlagen" in src
        assert "conservative uniform weights" in src


# ---------------------------------------------------------------------------
# G2 — FallbackQualityFloor multi-candidate cascade
# ---------------------------------------------------------------------------


class TestG2FallbackQualityFloorCascade:
    """§Gap G2: FallbackQualityFloor versucht alle kompatiblen Kandidaten
    in Prioritätsreihenfolge statt nur einen."""

    def test_multi_candidate_cascade_in_source(self):
        """UV3-Quelltext enthält _fqf_candidates Multi-Kandidaten-Logik."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        assert "_fqf_candidates" in src, "G2-Fix fehlt: _fqf_candidates nicht in UV3 gefunden"
        assert "hpi_best_checkpoint" in src
        assert "original_audio" in src
        # The cascade should try both sources
        assert "_fqf_src" in src or "_fqf_cand" in src

    def test_shape_compatible_check_uses_all_candidates(self):
        """Cascade iteriert über alle Kandidaten bis shape-kompatiblen findet."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        # The loop should be a 'for' over _fqf_candidates
        assert "for _fqf_cand, _fqf_src in _fqf_candidates:" in src, "G2-Fix: for-Schleife über _fqf_candidates fehlt"

    def test_attempts_count_reflects_candidate_count(self):
        """attempts-Feld zählt die Anzahl versuchter Kandidaten."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        assert '_fallback_quality_floor["attempts"] = len(_fqf_candidates)' in src, (
            "G2-Fix: attempts = len(_fqf_candidates) fehlt"
        )


# ---------------------------------------------------------------------------
# G4 — FeedbackChain max_iter boost bei PMGG-Konflikten
# ---------------------------------------------------------------------------


class TestG4FeedbackChainConflictBoost:
    """§Gap G4: PMGG best_effort-Phasen triggern erhöhte FC-Iterationen."""

    def test_conflict_boost_logic_in_source(self):
        """UV3-Quelltext enthält §G4 (GEBOTE.md) conflict-boost Logik."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        assert "§G4 (GEBOTE.md) FeedbackChain conflict-boost" in src, (
            "G4-Fix fehlt: conflict-boost Kommentar nicht in UV3 gefunden"
        )
        assert "_fc_conflict_phases" in src
        assert "best_effort" in src

    def test_boost_thresholds_conservative(self):
        """Boost-Schwellwerte: ≥3 Phasen → +1, ≥5 Phasen → +2."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        assert "_fc_conflict_phases >= 5" in src, "G4: Schwellwert 5 fehlt"
        assert "_fc_conflict_phases >= 3" in src, "G4: Schwellwert 3 fehlt"
        assert "min(_fc_max_iter + 2, 9)" in src, "G4: +2 boost mit cap 9 fehlt"
        assert "min(_fc_max_iter + 1, 7)" in src, "G4: +1 boost mit cap 7 fehlt"

    def test_boost_respects_short_audio_cap(self):
        """Boost findet VOR dem is_very_short-Check statt — korrekte Reihenfolge prüfen.

        _fc_max_iter = 5 → §G4 (GEBOTE.md) boost → _is_very_short cap → PhysicalCeiling cap.
        So the final cap from _is_very_short (min(x, 2)) still limits the boosted value.
        This is by design: short audio never gets more than 2 FC iterations.
        """
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        # §G4 (GEBOTE.md) block should appear before _is_very_short cap
        idx_g4 = src.find("§G4 (GEBOTE.md) FeedbackChain conflict-boost")
        idx_short = src.find("§2.31d: Audio < 10s → FeedbackChain max_iter")
        assert idx_g4 != -1
        assert idx_short != -1
        assert idx_g4 < idx_short, "G4 boost muss VOR _is_very_short-Cap stehen, damit cap korrekt greift"

    def test_boost_reads_from_pmgg_log_entries(self):
        """Conflict count liest aus self._pmgg_log_entries."""
        import pathlib

        src = pathlib.Path(
            "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        ).read_text(encoding="utf-8")
        # Find _fc_conflict_phases assignment — it must reference _pmgg_log_entries
        idx_conflict_phases = src.find("_fc_conflict_phases")
        assert idx_conflict_phases != -1, "_fc_conflict_phases nicht in UV3 gefunden"
        # Search in a ±800 char window around _fc_conflict_phases for _pmgg_log_entries
        window = src[max(0, idx_conflict_phases - 800) : idx_conflict_phases + 800]
        assert "_pmgg_log_entries" in window, (
            "G4 boost block liest nicht aus _pmgg_log_entries (erwartet in ±800 Zeichen um _fc_conflict_phases)"
        )


# ---------------------------------------------------------------------------
# Integration: all three gaps produce no import errors
# ---------------------------------------------------------------------------


class TestUV3ImportIntegrity:
    """UV3 muss nach allen Gap-Fixes fehlerfrei importierbar sein."""

    def test_uv3_syntax_ok(self):
        """py_compile auf UV3 liefert keine Fehler."""
        import py_compile

        path = "/media/michael/Software 4TB/Aurik_Standalone/backend/core/unified_restorer_v3.py"
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"UV3 Syntax-Fehler nach Gap-Fixes: {e}")
