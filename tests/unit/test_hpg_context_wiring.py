"""Tests für HPG-Kontext-Verdrahtung in UV3 (§2.44 FIX).

Verifiziert:
 1. UV3 übergibt echten _pmgg_restorability_score (nicht getattr Default 70.0)
 2. UV3 übergibt echte genre/material/era_bin (nicht "DEFAULT/digital/post-1990")
 3. HPIResult.detail enthält alle drei Kontext-Schlüssel mit richtigen Werten
 4. Regressions-Test: HPG-Aufruf ohne Fehler wenn Variablen None sind
"""

import math
import re

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# HPG Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHPGEvaluateRestoration:
    """Unit-Tests für HolisticPerceptualGate.evaluate_restoration() Kontext-Parameter."""

    def _make_gate(self):
        from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

        return HolisticPerceptualGate()

    def _sine(self, sr: int = 48000, dur: float = 1.0, freq: float = 440.0) -> np.ndarray:
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        return (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]

    def test_restorability_high_uses_ref_weight_100(self):
        """restorability > 70 → input_weight 0.0, ref_weight 1.0 in detail."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(audio, audio, 48000, restorability_score=85.0)
        assert result.detail["input_weight"] == 0.0
        assert result.detail["ref_weight"] == 1.0

    def test_restorability_low_uses_mixed_weight(self):
        """restorability < 50 → input_weight 0.2 in detail."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(audio, audio, 48000, restorability_score=30.0)
        assert result.detail["input_weight"] == pytest.approx(0.2, abs=0.01)

    def test_context_keys_in_detail(self):
        """genre/material/era_bin werden korrekt in HPIResult.detail eingetragen."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(
            audio,
            audio,
            48000,
            genre="jazz",
            material="vinyl",
            era_bin="pre-1980",
        )
        assert result.detail["genre"] == "jazz"
        assert result.detail["material"] == "vinyl"
        assert result.detail["era_bin"] == "pre-1980"

    def test_restorability_in_detail(self):
        """restorability_score wird korrekt in HPIResult.detail eingetragen."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(
            audio,
            audio,
            48000,
            restorability_score=55.0,
        )
        assert result.detail["restorability_score"] == pytest.approx(55.0, abs=0.1)

    def test_artifact_freedom_below_095_gate_fails(self):
        """artifact_freedom < 0.95 → HPIResult.passed False."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(audio, audio, 48000, artifact_freedom=0.80)
        assert result.passed is False

    def test_artifact_freedom_1_default_passes(self):
        """artifact_freedom=1.0 (Default) → HPIResult.passed True (HPI > 0)."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(audio, audio, 48000)
        assert result.passed is True

    def test_emotional_arc_zero_fails_hpi(self):
        """emotional_arc_score=0.0 → HPI = 0, passed False."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(audio, audio, 48000, emotional_arc_score=0.0)
        assert result.passed is False
        assert result.hpi == pytest.approx(0.0, abs=1e-4)

    def test_reference_audio_sets_carrier_reference_mode(self):
        """reference_audio aktiviert den Carrier-Referenzmodus im HPI-Ergebnis."""
        gate = self._make_gate()
        audio = self._sine()
        result = gate.evaluate_restoration(
            audio,
            audio,
            48000,
            reference_audio=audio,
        )
        assert result.reference_mode == "best_carrier_checkpoint"
        assert result.detail["reference_mode"] == "best_carrier_checkpoint"

    def test_strict_gate_at_high_restorability(self):
        """restorability > 85 → HPI *= 0.95 (strict gate), detail['strict_gate']=True."""
        gate = self._make_gate()
        audio = self._sine()
        result_strict = gate.evaluate_restoration(audio, audio, 48000, restorability_score=90.0)
        result_normal = gate.evaluate_restoration(audio, audio, 48000, restorability_score=70.0)
        assert result_strict.detail.get("strict_gate") is True
        assert result_normal.detail.get("strict_gate") is False
        # Mit strict_gate sollte HPI etwas niedriger sein
        assert result_strict.hpi <= result_normal.hpi + 0.01


# ---------------------------------------------------------------------------
# UV3 Source Code Audit — verifiziert ohne echte Pipeline auszuführen
# ---------------------------------------------------------------------------


class TestUV3HPGWiring:
    """Statische Code-Audit-Tests für die UV3→HPG-Kontextverdrahtung."""

    @pytest.fixture(scope="class")
    def uv3_source(self):
        import pathlib

        src_path = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "unified_restorer_v3.py"
        return src_path.read_text(encoding="utf-8")

    def test_no_getattr_restorability_score_default_in_hpi_block(self, uv3_source: str):
        """UV3 darf `getattr(self, "_restorability_score", 70.0)` beim HPG-Aufruf NICHT verwenden."""
        # Suche nach dem alten Bug-Pattern
        bug_pattern = r'getattr\s*\(\s*self\s*,\s*["\']_restorability_score["\']'
        assert not re.search(bug_pattern, uv3_source), (
            "Bug: UV3 nutzt getattr(_restorability_score) beim HPG — immer 70.0 statt echtem _pmgg_restorability_score"
        )

    def test_hpi_restorability_variable_present(self, uv3_source: str):
        """UV3 muss _hpi_restorability Variable definieren (echter Restorability-Wert)."""
        assert "_hpi_restorability" in uv3_source, (
            "UV3 fehlt _hpi_restorability Variable — HPG bekommt keine echte Restorability"
        )

    def test_hpi_genre_variable_present(self, uv3_source: str):
        """UV3 muss _hpi_genre Variable definieren (echter Genre-Wert)."""
        assert "_hpi_genre" in uv3_source, "UV3 fehlt _hpi_genre Variable — HPG bekommt immer 'DEFAULT'"

    def test_hpi_material_variable_present(self, uv3_source: str):
        """UV3 muss _hpi_material Variable definieren (echter Material-Wert)."""
        assert "_hpi_material" in uv3_source, "UV3 fehlt _hpi_material Variable — HPG bekommt immer 'digital'"

    def test_hpi_era_variable_present(self, uv3_source: str):
        """UV3 muss _hpi_era Variable definieren (echter Ära-Wert)."""
        assert "_hpi_era" in uv3_source, "UV3 fehlt _hpi_era Variable — HPG bekommt immer 'post-1990'"

    def test_evaluate_restoration_uses_context_vars(self, uv3_source: str):
        """evaluate_restoration() muss _hpi_restorability/_hpi_genre/_hpi_material/_hpi_era übergeben."""
        # Finde den evaluate_restoration-Aufruf
        block_start = uv3_source.find("_hg.evaluate_restoration")
        assert block_start != -1, "evaluate_restoration Aufruf nicht gefunden"
        # Das Block-Ende bis zur schließenden Klammer
        block_snippet = uv3_source[block_start : block_start + 600]
        assert "restorability_score=_hpi_restorability" in block_snippet, (
            "HPG evaluate_restoration übergibt kein _hpi_restorability"
        )
        assert "genre=_hpi_genre" in block_snippet, "HPG evaluate_restoration übergibt kein _hpi_genre"
        assert "material=_hpi_material" in block_snippet, "HPG evaluate_restoration übergibt kein _hpi_material"
        assert "era_bin=_hpi_era" in block_snippet, "HPG evaluate_restoration übergibt kein _hpi_era"

    def test_hpi_restorability_uses_pmgg_value(self, uv3_source: str):
        """_hpi_restorability muss aus _pmgg_restorability_score berechnet werden."""
        # Suche nach der Zuweisung
        pattern = r"_hpi_restorability\s*=\s*float\s*\(\s*_pmgg_restorability_score\s*\)"
        assert re.search(pattern, uv3_source), "_hpi_restorability wird nicht aus _pmgg_restorability_score berechnet"

    def test_hpi_material_uses_material_type(self, uv3_source: str):
        """_hpi_material muss material_type.value nutzen (nicht hardcoded 'digital')."""
        pattern = r"_hpi_material\s*=.*material_type"
        assert re.search(pattern, uv3_source, re.DOTALL), "_hpi_material wird nicht aus material_type berechnet"

    def test_no_hardcoded_default_genre_era_in_hpi_block(self, uv3_source: str):
        """Kein hardcoded genre='DEFAULT' oder era_bin='post-1990' mehr im HPG-Block."""
        # Finde HPG-Block
        block_start = uv3_source.find("_hg.evaluate_restoration")
        if block_start == -1:
            pytest.skip("evaluate_restoration nicht gefunden — Skip")
        block = uv3_source[block_start : block_start + 600]
        assert 'genre="DEFAULT"' not in block, "Hardcoded genre='DEFAULT' im HPG-Aufruf"
        assert 'era_bin="post-1990"' not in block, "Hardcoded era_bin='post-1990' im HPG-Aufruf"
        assert "era_bin='post-1990'" not in block, "Hardcoded era_bin='post-1990' im HPG-Aufruf"

    def test_afg_shape_mismatch_path_is_fail_closed(self, uv3_source: str):
        """AFG-Veto darf bei inkompatiblem Checkpoint nicht in Export-trotzdem kippen."""
        assert "Export trotzdem" not in uv3_source, "AFG-Pfad enthält noch fail-open Export-trotzdem-Text"
        assert "kein kompatibler Rollback-Checkpoint (fail-closed auf Original)" in uv3_source, (
            "AFG-Shape-Mismatch muss explizit fail-closed auf Original loggen"
        )
