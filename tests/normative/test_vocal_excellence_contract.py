from __future__ import annotations

"""[RELEASE_MUST] Normativer Gesangsexzellenz-Contract.

Dieser CI-Test bündelt die wichtigsten Gesangs-Invarianten aus Spec 01/§2.35d,
§2.35c, §2.36 und §0a in einem kompakten Guard:

1. VQI-Schwellen und Gewichtung bleiben stabil.
2. Singer-Identity-Rollback-Schutz bleibt dokumentiert.
3. Die Vocal-Chain-Reihenfolge bleibt kanonisch.
4. Restoration trennt Stem-Enhancement strikt von phonem-bewusster Steuerung.

Der Test prüft bewusst Code- und Spec-Artefakte gemeinsam, um Drift zwischen
normativer Aussage und tatsächlicher Pipeline-Verdrahtung früh zu erkennen.
"""


from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_01 = _ROOT / ".github" / "specs" / "01_musical_goals.md"
_VQI_CODE = _ROOT / "backend" / "core" / "musical_goals" / "vocal_quality_index.py"
_UV3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"
_MUSICAL_GOALS_INSTR = _ROOT / ".github" / "instructions" / "musical_goals.instructions.md"
_SECTION_0A_GUARD = _ROOT / "tests" / "normative" / "test_section_0a_restoration_guard.py"
_PANNS = _ROOT / "plugins" / "panns_plugin.py"
_PHASE65 = _ROOT / "backend" / "core" / "phases" / "phase_65_vocal_naturalness_restoration.py"


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestVocalExcellenceSpec:
    def test_spec_declares_vocal_excellence_contract(self) -> None:
        content = _SPEC_01.read_text(encoding="utf-8")

        assert "§2.35d [RELEASE_MUST] Gesangsexzellenz-Contract" in content
        assert (
            "phase_19_de_esser -> phase_42_vocal_enhancement -> phase_43_ml_deesser -> phase_58_lyrics_guided_enhancement"
            in content
        )
        assert "phase_42_vocal_enhancement` bleibt in `restoration` verboten" in content
        assert "phase_58_lyrics_guided_enhancement` bleibt bei erkannter Stimme dennoch Pflicht" in content


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestVocalExcellenceThresholds:
    def test_vqi_constants_remain_at_documented_contract_values(self) -> None:
        content = _VQI_CODE.read_text(encoding="utf-8")

        assert "VQI_WORLD_CLASS = 0.88" in content
        assert "VQI_PROFESSIONAL = 0.82" in content
        assert "VQI_THRESHOLD = 0.72" in content

    def test_vqi_weights_still_sum_to_one(self) -> None:
        from backend.core.musical_goals import vocal_quality_index as vqi

        total = vqi._W_SINGER_ID + vqi._W_FORMANT + vqi._W_ARTICULATION + vqi._W_PROXIMITY + vqi._W_SIBILANCE
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_singer_identity_rollback_threshold_remains_documented(self) -> None:
        content = _MUSICAL_GOALS_INSTR.read_text(encoding="utf-8")

        assert "cos_sim < 0.92 → Rollback letzter Vokal-Phase" in content

    def test_vocal_metric_activation_thresholds_remain_conservative(self) -> None:
        panns = _PANNS.read_text(encoding="utf-8")
        spec = _SPEC_01.read_text(encoding="utf-8")

        assert "threshold ≥ 0.40" in panns
        assert "panns_singing_confidence >= 0.35" in spec


@pytest.mark.normative
@pytest.mark.timeout(10)
class TestVocalExcellencePipelineWiring:
    def test_uv3_passes_song_structure_vocal_segments_into_vqi(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "for seg in (self._ssa_segments or [])" in content
        assert 'if bool(getattr(seg, "has_vocals", False)) and float(seg.end_s) > float(seg.start_s)' in content
        assert "vocal_segments=_vqi_segments or None" in content

    def test_uv3_accumulates_vqi_fields_before_final_export(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert 'self._phase_metadata_accumulator["vqi"] = _vqi_score' in content
        assert 'self._phase_metadata_accumulator["singer_identity_cosine"] = float(' in content
        assert 'self._phase_metadata_accumulator["singer_id_dsp_fallback"] = bool(' in content
        assert 'self._phase_metadata_accumulator["vqi_tier"] = _vqi_result.get("vqi_tier", "unknown")' in content

    def test_uv3_exports_final_vocal_metadata_fields(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"vqi": (self._phase_metadata_accumulator or {}).get("vqi")' in content
        assert (
            '"singer_identity_cosine": (self._phase_metadata_accumulator or {}).get("singer_identity_cosine")'
            in content
        )
        assert (
            '"singer_id_dsp_fallback": (self._phase_metadata_accumulator or {}).get("singer_id_dsp_fallback")'  # §V6 (copilot-instructions.md): logger.warning handled at call site
            in content
        )
        assert '"vqi_tier": (self._phase_metadata_accumulator or {}).get("vqi_tier")' in content

    def test_uv3_keeps_canonical_vocal_chain_order(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '_move_before("phase_19_de_esser", "phase_42_vocal_enhancement")' in content
        assert '_move_before("phase_19_de_esser", "phase_43_ml_deesser")' in content
        assert '_move_before("phase_42_vocal_enhancement", "phase_58_lyrics_guided_enhancement")' in content
        assert '_move_before("phase_43_ml_deesser", "phase_58_lyrics_guided_enhancement")' in content

    def test_uv3_keeps_lyrics_guided_enhancement_for_detected_vocals(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "if vocals_detected:" in content
        assert 'selected.append("phase_58_lyrics_guided_enhancement")' in content

    def test_restoration_guard_keeps_mode_split_between_phase_42_and_phase_58(self) -> None:
        content = _SECTION_0A_GUARD.read_text(encoding="utf-8")

        assert '"phase_42_vocal_enhancement"' in content
        assert '"phase_58_lyrics_guided_enhancement": "Lyrics-Guided: §2.36 PFLICHT auch in Restoration' in content

    def test_restoration_routes_low_vqi_to_phase65_not_phase42(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "phase_65_vocal_naturalness_restoration" in content
        assert "_vqi_score < 0.74" in content
        assert "from backend.core.phases.phase_65_vocal_naturalness_restoration" in content
        assert "# §0a: phase_42_vocal_enhancement ist in Restoration VERBOTEN." in content

    def test_restoration_routes_vocal_import_naturalness_deficit_to_phase65(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '_gbc_goal == "natuerlichkeit"' in content
        assert 'float(getattr(self, "_panns_singing", 0.0)) >= 0.25' in content
        assert '"phase_65_vocal_naturalness_restoration",' in content

    def test_low_singmos_triggers_phase65_recovery_not_warning_only(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "_singmos_needs_phase65 = _singmos_val < 2.5" in content
        assert "singmos_phase65_recovery" in content
        assert "§G4 (GEBOTE.md) SingMOS Phase_65-Recovery" in content

    def test_import_temporal_coherence_failure_is_repair_driver_not_export_judgment(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "Import-TQC" in content
        assert "import_temporal_coherence" in content
        assert "import_temporal_coherence_recovery_phases" in content
        assert '"phase_12_wow_flutter_fix"' in content
        assert '"phase_14_phase_correction"' in content

    def test_vocal_analog_restoration_removes_phase17_mastering_polish_preflight(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"phase_17_mastering_polish" in _sel_set_prerisk' in content
        assert '_sel_set_prerisk.remove("phase_17_mastering_polish")' in content
        assert "NOVELTY_CRIT/HNR_DROP/ECHO" in content
        assert "preflight_risk_removed_phases" in content
        assert "Preflight-Risk-Guard hatte Phase entfernt" in content

    def test_phase03_skips_resemble_second_pass_after_vocal_primary_on_cassette(self) -> None:
        content = (_ROOT / "backend/core/phases/phase_03_denoise.py").read_text(encoding="utf-8")

        assert "_skip_ml_hybrid_after_vocal_primary" in content
        assert "_miipher_applied" in content
        assert 'material_type in ("cassette", "tape", "reel_tape", "mp3_low")' in content
        assert "konservative OMLSA/DSP-Restglättung statt Resemble-Zweitpass" in content

    def test_phase65_keeps_vocal_naturalness_guards_active(self) -> None:
        content = _PHASE65.read_text(encoding="utf-8")

        assert "apply_hnr_blend" in content
        assert "_HNR_DELTA_THRESHOLD: float = 2.5" in content
        assert "_FORMANT_MAX_BOOST_DB: float = 1.0" in content
        assert "vibrato_zone_cap_applied" in content
        assert "passaggio" in content.lower()

    def test_phase65_is_protected_from_wall_time_budget_pressure(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert '"phase_65",' in content
        assert '"phase_65_vocal_naturalness_restoration",' in content
        assert 'phase_65_vocal_naturalness_restoration",  # §0p VQI-/Naturalness-Recovery' in content

    def test_phase_coalitions_are_restoration_safe_and_vocal_aware(self) -> None:
        content = _UV3.read_text(encoding="utf-8")

        assert "_PHASE_COALITIONS" in content
        assert "_RESTORATION_FORBIDDEN_COALITION_PHASES" in content
        assert "get_active_phase_coalitions" in content
        assert "phase_65_vocal_naturalness_restoration" in content
        assert "phase_coalitions=_active_phase_coalitions" in content
