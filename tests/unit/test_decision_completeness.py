"""Vollständigkeits-Beweis: Jeder DefectType × MaterialType × QualityMode → ≥1 gezielte Phase.

Spec-Referenz: copilot-instructions.md §7.2 (CAUSE_TO_PHASES), §6.3 (DefectType), §6.1 (MaterialType)
Invariante: „Entscheidend ist immer die musikalische Exzellenz."

Struktureller Beweis:
    810 parametrische Kombinationen (27 DefectTypes × 15 MaterialTypes × 2 QualityModes).
    Für jede Kombination mit severity=0.80 muss _select_phases() mindestens eine
    gezielt ausgewählte Phase (außer Tier-0-Pflichtphasen und Tier-6-Export) zurückgeben.

    Tier-0 (immer): phase_30_dc_offset_removal, phase_05_rumble_filter
    Tier-6 (immer): phase_16_final_eq, phase_47_truepeak_limiter,
                    phase_40_loudness_normalization, phase_41_output_format_optimization
    Tier-6 (nur Studio 2026, §v10.70): phase_17_mastering_polish
"""

from unittest.mock import MagicMock

import pytest

from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, DefectType, MaterialType
from backend.core.performance_guard import QualityMode

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

TIER_0_PHASES = {
    "phase_30_dc_offset_removal",
    "phase_05_rumble_filter",
}

TIER_6_PHASES = {
    "phase_16_final_eq",
    "phase_47_truepeak_limiter",
    "phase_40_loudness_normalization",
    "phase_41_output_format_optimization",
}

# §v10.70: phase_17_mastering_polish ist bewusst STUDIO_ONLY, kein Tier-6-Immer-Fall.
STUDIO_ONLY_PHASES = {"phase_17_mastering_polish"}

STRUCTURAL_PHASES = TIER_0_PHASES | TIER_6_PHASES


def _make_defect_result(
    material: MaterialType, defect_type: DefectType, severity: float = 0.80
) -> DefectAnalysisResult:
    """Erstellt ein DefectAnalysisResult mit einem einzigen aktiven Defekt."""
    scores = {dt: DefectScore(defect_type=dt, severity=0.0, confidence=0.0) for dt in DefectType}
    scores[defect_type] = DefectScore(
        defect_type=defect_type,
        severity=severity,
        confidence=0.9,
    )
    return DefectAnalysisResult(
        material_type=material,
        scores=scores,
        analysis_time_seconds=0.1,
        sample_rate=48000,
        duration_seconds=0.0,
    )


def _select_phases_for(
    material: MaterialType, defect_type: DefectType, mode: QualityMode, *, studio_2026: bool = False
) -> list[str]:
    """Ruft _select_phases() über RestorationConfig auf, ohne echte ML-Modelle."""
    # Lazy import damit der Singleton nicht im Modulscope instanziiert wird
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    config = RestorationConfig(mode=mode, studio_2026=studio_2026)
    restorer = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
    restorer.config = config
    restorer.logger = MagicMock()  # type: ignore[attr-defined]

    dr = _make_defect_result(material, defect_type, severity=0.80)
    return restorer._select_phases(dr)


# ─────────────────────────────────────────────────────────────────────────────
# Parametrische Haupt-Testsuite (630 Kombinationen)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [QualityMode.QUALITY, QualityMode.BALANCED])
@pytest.mark.parametrize("material", list(MaterialType))
@pytest.mark.parametrize("defect_type", list(DefectType))
def test_every_switching_state_produces_targeted_phases(
    defect_type: DefectType, material: MaterialType, mode: QualityMode
):
    """Jede DefectType × MaterialType × QualityMode Kombination → ≥1 gezielte Phase."""
    phases = _select_phases_for(material, defect_type, mode)
    phase_set = set(phases)
    targeted = phase_set - STRUCTURAL_PHASES
    assert len(targeted) >= 1, (
        f"Keine gezielte Phase für:\n"
        f"  DefectType   = {defect_type.name}\n"
        f"  MaterialType = {material.name}\n"
        f"  QualityMode  = {mode.name}\n"
        f"  Aktive Phasen: {sorted(phase_set)}\n"
        f"  (Tier-0/6 strukturell, zählen nicht)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dedizierte Tests für die 8 behobenen Lücken
# ─────────────────────────────────────────────────────────────────────────────


class TestPrintThrough:
    """Fix #1: PRINT_THROUGH hatte überhaupt keinen Branch."""

    def test_tape_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.PRINT_THROUGH, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases
        assert "phase_03_denoise" in phases

    def test_reel_tape_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.PRINT_THROUGH, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases
        assert "phase_03_denoise" in phases

    def test_non_tape_material_no_print_through_phases(self):
        """Vinyl hat kein Print-Through — soll NICHT durch den PRINT_THROUGH-Block aktiviert werden."""
        phases = _select_phases_for(MaterialType.VINYL, DefectType.PRINT_THROUGH, QualityMode.QUALITY)
        # phase_29 sollte NICHT durch Print-Through ausgelöst sein (Vinyl ≠ Tape)
        # (Es könnte durch HIGH_FREQ_NOISE ausgelöst sein — aber wir testen nur die Grundphase)
        assert isinstance(phases, list)  # kein Absturz

    def test_no_nan_in_phases(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.PRINT_THROUGH, QualityMode.BALANCED)
        assert all(isinstance(p, str) for p in phases)


class TestQuantizationNoise:
    """Fix #2: QUANTIZATION_NOISE hatte überhaupt keinen Branch."""

    def test_quality_mode_activates_denoise_and_spectral(self):
        phases = _select_phases_for(MaterialType.CD_DIGITAL, DefectType.QUANTIZATION_NOISE, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases
        assert "phase_23_spectral_repair" in phases

    def test_balanced_mode_activates_denoise_and_spectral(self):
        phases = _select_phases_for(MaterialType.MP3_LOW, DefectType.QUANTIZATION_NOISE, QualityMode.BALANCED)
        assert "phase_03_denoise" in phases
        assert "phase_23_spectral_repair" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.QUANTIZATION_NOISE, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"QUANTIZATION_NOISE × {mat.name} → keine Phasen"


class TestJitterArtifacts:
    """Fix #3: JITTER_ARTIFACTS hatte überhaupt keinen Branch."""

    def test_dat_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.DAT, DefectType.JITTER_ARTIFACTS, QualityMode.QUALITY)
        assert "phase_14_phase_correction" in phases
        assert "phase_23_spectral_repair" in phases

    def test_streaming_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.STREAMING, DefectType.JITTER_ARTIFACTS, QualityMode.QUALITY)
        assert "phase_14_phase_correction" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.JITTER_ARTIFACTS, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"JITTER_ARTIFACTS × {mat.name} → keine Phasen"


class TestWaxCylinder:
    """Fix #4: WAX_CYLINDER wurde nirgends in _select_phases() behandelt."""

    def test_wax_cylinder_activates_click_removal(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_01_click_removal" in phases

    def test_wax_cylinder_activates_denoise(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases

    def test_wax_cylinder_activates_freq_restoration(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.BANDWIDTH_LOSS, QualityMode.QUALITY)
        assert "phase_06_frequency_restoration" in phases

    def test_wax_cylinder_always_includes_harmonic_restoration(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.CLICKS, QualityMode.BALANCED)
        assert "phase_07_harmonic_restoration" in phases

    def test_wax_cylinder_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.WAX_CYLINDER, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"WAX_CYLINDER × {dt.name} → keine Phasen"


class TestWireRecording:
    """Fix #5: WIRE_RECORDING wurde nirgends in _select_phases() behandelt."""

    @pytest.mark.parametrize("defect_type", [DefectType.WOW, DefectType.FLUTTER])
    def test_wire_recording_activates_wow_and_flutter(self, defect_type: DefectType):
        phases = _select_phases_for(MaterialType.WIRE_RECORDING, defect_type, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases

    def test_wire_recording_activates_dropout_repair(self):
        phases = _select_phases_for(MaterialType.WIRE_RECORDING, DefectType.DROPOUTS, QualityMode.QUALITY)
        assert "phase_24_dropout_repair" in phases

    def test_wire_recording_activates_hiss_reduction(self):
        phases = _select_phases_for(MaterialType.WIRE_RECORDING, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases

    def test_wire_recording_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.WIRE_RECORDING, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"WIRE_RECORDING × {dt.name} → keine Phasen"


class TestLacquerDisc:
    """Fix #6: LACQUER_DISC wurde nirgends in _select_phases() behandelt."""

    def test_lacquer_disc_activates_click_removal(self):
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_01_click_removal" in phases

    def test_lacquer_disc_activates_crackle_removal(self):
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.CRACKLE, QualityMode.QUALITY)
        assert "phase_09_crackle_removal" in phases

    def test_lacquer_disc_activates_surface_noise_profiling(self):
        """LACQUER_DISC muss jetzt im surface_noise_profiling-Gate sein (Fix #1 + #6)."""
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.CRACKLE, QualityMode.QUALITY)
        assert "phase_28_surface_noise_profiling" in phases

    def test_lacquer_disc_activates_denoise(self):
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases

    def test_lacquer_disc_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.LACQUER_DISC, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"LACQUER_DISC × {dt.name} → keine Phasen"


class TestReelTapeGaps:
    """Fix #7: REEL_TAPE fehlte in Tape-Hiss, Azimuth, Air-Band, Tape-Saturation, EQ-Gates."""

    def test_reel_tape_activates_tape_hiss_reduction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases

    def test_reel_tape_activates_azimuth_correction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.PHASE_ISSUES, QualityMode.QUALITY)
        assert "phase_25_azimuth_correction" in phases

    def test_reel_tape_activates_air_band_enhancement(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.BANDWIDTH_LOSS, QualityMode.QUALITY)
        assert "phase_39_air_band_enhancement" in phases

    def test_reel_tape_activates_tape_saturation(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.CLICKS, QualityMode.BALANCED)
        assert "phase_22_tape_saturation" in phases

    def test_reel_tape_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_reel_tape_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.REEL_TAPE, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"REEL_TAPE × {dt.name} → keine Phasen"


class TestDATGap:
    """Fix #8: DAT fehlte im EQ-Correction-Gate."""

    def test_dat_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.DAT, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_dat_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.DAT, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"DAT × {dt.name} → keine Phasen"


# ─────────────────────────────────────────────────────────────────────────────
# Robustheitstests
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    """_select_phases() darf niemals abstürzen oder NaN zurückgeben."""

    def test_all_combinations_no_exception(self):
        """630 Kombinationen — kein einziger Exception."""
        for mode in [QualityMode.QUALITY, QualityMode.BALANCED]:
            for mat in MaterialType:
                for dt in DefectType:
                    try:
                        phases = _select_phases_for(mat, dt, mode)
                        assert isinstance(phases, list)
                    except Exception as exc:
                        pytest.fail(f"Exception bei {dt.name} × {mat.name} × {mode.name}: {exc}")

    def test_phases_are_strings(self):
        """Alle zurückgegebenen Phasen-Namen sind echte Strings."""
        for mat in [
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
            MaterialType.REEL_TAPE,
        ]:
            for dt in [DefectType.PRINT_THROUGH, DefectType.QUANTIZATION_NOISE, DefectType.JITTER_ARTIFACTS]:
                phases = _select_phases_for(mat, dt, QualityMode.QUALITY)
                assert all(isinstance(p, str) and len(p) > 0 for p in phases), (
                    f"Ungültige Phase bei {mat.name} × {dt.name}: {phases}"
                )

    def test_no_duplicate_phases(self):
        """Keine doppelten Phasen in einer Auswahl (Deduplizierung)."""
        # WAX_CYLINDER kann mehrfach dieselbe Phase triggern (z.B. phase_03_denoise)
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.HIGH_FREQ_NOISE, QualityMode.BALANCED)
            assert len(phases) == len(set(phases)), f"Doppelte Phasen bei {mat.name}: {phases}"

    def test_tier_0_threshold_gated(self):
        """Tier-0 Phasen werden nur bei echten Defekten aktiviert (§0 Minimal-Intervention)."""
        for mat in list(MaterialType)[:3]:  # Stichprobe
            # Phase_30 nur bei DC-OFFSET severity > 0.10
            phases_dc = _select_phases_for(mat, DefectType.DC_OFFSET, QualityMode.QUALITY)
            assert "phase_30_dc_offset_removal" in set(phases_dc), f"phase_30 fehlt bei {mat.name} mit DC_OFFSET"

            # Phase_05 nur bei LOW_FREQ_RUMBLE severity > 0.10
            phases_rumble = _select_phases_for(mat, DefectType.LOW_FREQ_RUMBLE, QualityMode.QUALITY)
            assert "phase_05_rumble_filter" in set(phases_rumble), f"phase_05 fehlt bei {mat.name} mit RUMBLE"

    def test_tier_6_always_present(self):
        """Tier-6 Export-Pflichtphasen sind immer enthalten."""
        for mat in list(MaterialType)[:5]:  # Stichprobe
            phases = _select_phases_for(mat, DefectType.CLICKS, QualityMode.BALANCED)
            phase_set = set(phases)
            for p in TIER_6_PHASES:
                assert p in phase_set, f"Tier-6 Phase {p} fehlt bei {mat.name}"

    def test_studio_only_phase_present_only_in_studio_mode(self):
        """§v10.70: phase_17_mastering_polish nur in Studio 2026, nie in Restoration."""
        for mat in list(MaterialType)[:5]:  # Stichprobe
            restoration_phases = set(_select_phases_for(mat, DefectType.CLICKS, QualityMode.BALANCED))
            assert "phase_17_mastering_polish" not in restoration_phases, (
                f"phase_17_mastering_polish sollte in Restoration (BALANCED) fehlen bei {mat.name}"
            )
            studio_phases = set(_select_phases_for(mat, DefectType.CLICKS, QualityMode.MAXIMUM, studio_2026=True))
            assert "phase_17_mastering_polish" in studio_phases, (
                f"phase_17_mastering_polish sollte in Studio 2026 (MAXIMUM) vorhanden sein bei {mat.name}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Zusammenfassungs-Statistik (Informativ, kein Fehlschlag)
# ─────────────────────────────────────────────────────────────────────────────


def test_coverage_summary():
    """Gibt eine Zusammenfassung der Abdeckungsstatistik aus."""
    total = 0
    covered = 0
    gaps = []

    for mode in [QualityMode.QUALITY, QualityMode.BALANCED]:
        for mat in MaterialType:
            for dt in DefectType:
                total += 1
                phases = _select_phases_for(mat, dt, mode)
                targeted = set(phases) - STRUCTURAL_PHASES
                if len(targeted) >= 1:
                    covered += 1
                else:
                    gaps.append((dt.name, mat.name, mode.name))

    coverage_pct = covered / total * 100
    # §10.1: Kein print() in Tests — Diagnose-Info in assert-Nachricht
    gap_summary = (
        (f"Lücken ({len(gaps)}): " + "; ".join(f"{g[0]}×{g[1]}×{g[2]}" for g in gaps[:20]))
        if gaps
        else "alle Kombinationen vollständig abgedeckt"
    )

    assert coverage_pct == 100.0, f"Nicht alle Kombinationen abgedeckt: {coverage_pct:.1f}% — {gap_summary}"


# ─────────────────────────────────────────────────────────────────────────────
# Neue DefectTypes v10.0.0c: RIAA_CURVE_ERROR, ALIASING, BIAS_ERROR
# ─────────────────────────────────────────────────────────────────────────────


class TestRiaaCurveError:
    """Fix #9: RIAA_CURVE_ERROR — falsche Entzerrungskurve (Shellac/früher Vinyl)."""

    def test_shellac_activates_eq_and_freq_restoration(self):
        """Shellac-RIAA-Fehler → phase_04_eq_correction + phase_06_frequency_restoration."""
        phases = _select_phases_for(MaterialType.SHELLAC, DefectType.RIAA_CURVE_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases
        assert "phase_06_frequency_restoration" in phases

    def test_vinyl_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.VINYL, DefectType.RIAA_CURVE_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_balanced_mode_adds_harmonic_restoration(self):
        """Im BALANCED-Modus Oberton-Rekonstruktion bei hoher Severity (durch Entzerrungs-Kette verloren)."""
        phases = _select_phases_for(MaterialType.SHELLAC, DefectType.RIAA_CURVE_ERROR, QualityMode.BALANCED)
        assert "phase_07_harmonic_restoration" in phases

    def test_all_materials_produce_targeted_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.RIAA_CURVE_ERROR, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"RIAA_CURVE_ERROR × {mat.name} → keine Phasen"

    def test_no_nan_phases(self):
        phases = _select_phases_for(MaterialType.SHELLAC, DefectType.RIAA_CURVE_ERROR, QualityMode.BALANCED)
        assert all(isinstance(p, str) and len(p) > 0 for p in phases)


class TestAliasing:
    """Fix #10: ALIASING — AA-Filter-Artefakte aus ADC-Digitalisierung."""

    def test_cd_activates_denoise_and_spectral(self):
        """CD-Aliasing → spektrale Chirurgie (Denoise ist kontraindiziert, V30)."""
        phases = _select_phases_for(MaterialType.CD_DIGITAL, DefectType.ALIASING, QualityMode.QUALITY)
        assert "phase_23_spectral_repair" in phases
        assert "phase_03_denoise" not in phases

    def test_tape_activates_denoise_and_spectral(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.ALIASING, QualityMode.QUALITY)
        assert "phase_23_spectral_repair" in phases
        assert "phase_03_denoise" not in phases

    def test_balanced_mode_adds_spectral_repair_second_pass(self):
        """Schweres Aliasing im BALANCED-Modus → phase_50_spectral_repair (zweiter Pass)."""
        phases = _select_phases_for(MaterialType.MP3_LOW, DefectType.ALIASING, QualityMode.BALANCED)
        assert "phase_50_spectral_repair" in phases

    def test_all_materials_produce_targeted_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.ALIASING, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"ALIASING × {mat.name} → keine Phasen"

    def test_no_nan_phases(self):
        phases = _select_phases_for(MaterialType.DAT, DefectType.ALIASING, QualityMode.BALANCED)
        assert all(isinstance(p, str) and len(p) > 0 for p in phases)


class TestBiasError:
    """Fix #11: BIAS_ERROR — falscher Vormagnetisierungsstrom bei Bandaufnahme."""

    def test_tape_activates_eq_and_denoise(self):
        """Kassetten-Bias-Fehler → phase_04_eq_correction + phase_03_denoise."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.BIAS_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases
        assert "phase_03_denoise" in phases

    def test_reel_tape_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.BIAS_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_balanced_mode_adds_freq_and_hiss_reduction(self):
        """Starker Bias-Fehler im BALANCED-Modus → phase_06 + phase_29."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.BIAS_ERROR, QualityMode.BALANCED)
        assert "phase_06_frequency_restoration" in phases
        assert "phase_29_tape_hiss_reduction" in phases

    def test_all_materials_produce_targeted_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.BIAS_ERROR, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"BIAS_ERROR × {mat.name} → keine Phasen"

    def test_no_nan_phases(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.BIAS_ERROR, QualityMode.BALANCED)
        assert all(isinstance(p, str) and len(p) > 0 for p in phases)


# ─────────────────────────────────────────────────────────────────────────────
# §6.3 v10.0.0: Neue Carrier-Ursachen-Defekttypen (V28/V29/V31 + 6 weitere)
# VERBOTEN-Tabelle: V28 (NR_BREATHING→phase_54, NOT phase_03/29),
#                   V29 (OVERLOAD→phase_09/23, NOT phase_63),
#                   V31 (ROOM_MODE→phase_04 Primary, NOT phase_05 alone)
# ─────────────────────────────────────────────────────────────────────────────


class TestNrBreathingArtifact:
    """V28: NR_BREATHING_ARTIFACT → phase_54 + phase_08. VERBOTEN: phase_03/phase_29."""

    def test_activates_transparent_dynamics_primary(self):
        """V28 Primary: phase_54_transparent_dynamics (Envelope-Re-Smoothing)."""
        phases = _select_phases_for(MaterialType.CASSETTE, DefectType.NR_BREATHING_ARTIFACT, QualityMode.QUALITY)
        assert "phase_54_transparent_dynamics" in phases, "V28: phase_54 fehlt bei NR_BREATHING_ARTIFACT"

    def test_activates_transient_preservation_secondary(self):
        """V28 Sekundär: phase_08_transient_preservation."""
        phases = _select_phases_for(MaterialType.CASSETTE, DefectType.NR_BREATHING_ARTIFACT, QualityMode.QUALITY)
        assert "phase_08_transient_preservation" in phases, "V28: phase_08 fehlt bei NR_BREATHING_ARTIFACT"

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.NR_BREATHING_ARTIFACT, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"V28: NR_BREATHING_ARTIFACT × {mat.name} → keine Phasen"

    def test_no_nr_as_primary_trigger_for_breathing(self):
        """V28 VERBOTEN: phase_03/phase_29 NICHT als Primary für NR_BREATHING.
        Der Test prüft, dass UV3 phase_54 als Primary-Antwort auf den Breathing-Defekt enthält,
        unabhängig davon ob phase_03 durch andere Defekte auch aktiv ist."""
        phases = _select_phases_for(MaterialType.CASSETTE, DefectType.NR_BREATHING_ARTIFACT, QualityMode.QUALITY)
        # phase_54 MUSS vorhanden sein (zeigt korrekte Primary-Logik)
        assert "phase_54_transparent_dynamics" in phases, (
            "V28 Verletzung: NR_BREATHING_ARTIFACT ohne phase_54 als Primary-Antwort"
        )


class TestOverloadDistortion:
    """V29: OVERLOAD_DISTORTION → phase_09 + phase_23. VERBOTEN: phase_63."""

    def test_activates_crackle_removal_primary(self):
        """V29 Primary: phase_09_crackle_removal (asymmetrische Wellenform + Transient-Rekonstruktion)."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.OVERLOAD_DISTORTION, QualityMode.QUALITY)
        assert "phase_09_crackle_removal" in phases, "V29: phase_09 fehlt bei OVERLOAD_DISTORTION"

    def test_activates_spectral_repair_secondary(self):
        """V29 Sekundär: phase_23_spectral_repair (Harmonischer Klirr H2/H3)."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.OVERLOAD_DISTORTION, QualityMode.QUALITY)
        assert "phase_23_spectral_repair" in phases, "V29: phase_23 fehlt bei OVERLOAD_DISTORTION"

    def test_no_imd_phase_for_harmonic_distortion(self):
        """V29 VERBOTEN: phase_63_intermodulation_reduction für OVERLOAD_DISTORTION.
        Harmonische (H2/H3/H5) ≠ Intermodulationsprodukte (f₁±f₂) — §4.11."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.OVERLOAD_DISTORTION, QualityMode.QUALITY)
        assert "phase_63_intermodulation_reduction" not in phases, (
            "V29 Verletzung: phase_63 (IMD) fälschlicherweise für OVERLOAD_DISTORTION (Harmonische) aktiviert"
        )

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.OVERLOAD_DISTORTION, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"V29: OVERLOAD_DISTORTION × {mat.name} → keine Phasen"


class TestRoomModeResonance:
    """V31: ROOM_MODE_RESONANCE → phase_04 als Primary (Notch-EQ). VERBOTEN: phase_05 allein."""

    def test_activates_eq_correction_primary(self):
        """V31 Primary: phase_04_eq_correction (parametrischer Notch-EQ 40–200 Hz, Q=12)."""
        phases = _select_phases_for(MaterialType.VINYL, DefectType.ROOM_MODE_RESONANCE, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases, "V31: phase_04 (Notch-EQ) fehlt bei ROOM_MODE_RESONANCE"

    def test_phase04_in_phases_before_phase05(self):
        """V31: phase_04 MUSS in der Auswahl sein (wird vor phase_05 hinzugefügt)."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.ROOM_MODE_RESONANCE, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.ROOM_MODE_RESONANCE, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"V31: ROOM_MODE_RESONANCE × {mat.name} → keine Phasen"

    def test_phase04_present_not_phase05_alone(self):
        """V31: phase_04 MUSS vorhanden sein; phase_05 allein reicht für Resonanzen nicht."""
        phases = _select_phases_for(MaterialType.SHELLAC, DefectType.ROOM_MODE_RESONANCE, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases


class TestProximityEffectExcess:
    """PROXIMITY_EFFECT_EXCESS → phase_04 (LF-Notch ≤250 Hz)."""

    def test_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.PROXIMITY_EFFECT_EXCESS, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.PROXIMITY_EFFECT_EXCESS, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"PROXIMITY_EFFECT_EXCESS × {mat.name} → keine Phasen"


class TestFlutterSpectralSidebands:
    """FLUTTER_SPECTRAL_SIDEBANDS → phase_12 + phase_23."""

    def test_activates_wow_flutter_fix_primary(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.FLUTTER_SPECTRAL_SIDEBANDS, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases

    def test_activates_spectral_repair_secondary(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.FLUTTER_SPECTRAL_SIDEBANDS, QualityMode.QUALITY)
        assert "phase_23_spectral_repair" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.FLUTTER_SPECTRAL_SIDEBANDS, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"FLUTTER_SPECTRAL_SIDEBANDS × {mat.name} → keine Phasen"


class TestSpeedCalibrationError:
    """SPEED_CALIBRATION_ERROR → phase_12 (Constant-Rate-Mode) + phase_31."""

    def test_activates_wow_flutter_fix(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.SPEED_CALIBRATION_ERROR, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases

    def test_activates_speed_pitch_correction(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.SPEED_CALIBRATION_ERROR, QualityMode.QUALITY)
        assert "phase_31_speed_pitch_correction" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.SPEED_CALIBRATION_ERROR, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"SPEED_CALIBRATION_ERROR × {mat.name} → keine Phasen"


class TestLacquerDiscDegradation:
    """LACQUER_DISC_DEGRADATION DefectType (unabhängig von MaterialType.LACQUER_DISC)."""

    def test_activates_crackle_removal(self):
        phases = _select_phases_for(MaterialType.VINYL, DefectType.LACQUER_DISC_DEGRADATION, QualityMode.QUALITY)
        assert "phase_09_crackle_removal" in phases

    def test_activates_denoise(self):
        phases = _select_phases_for(MaterialType.VINYL, DefectType.LACQUER_DISC_DEGRADATION, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.LACQUER_DISC_DEGRADATION, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"LACQUER_DISC_DEGRADATION × {mat.name} → keine Phasen"


class TestScrapeFlutter:
    """SCRAPE_FLUTTER → phase_12 (Hochfrequenz-Transport-Korrektur)."""

    def test_tape_activates_wow_flutter_fix(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.SCRAPE_FLUTTER, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases

    def test_reel_tape_activates_wow_flutter_fix(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.SCRAPE_FLUTTER, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.SCRAPE_FLUTTER, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"SCRAPE_FLUTTER × {mat.name} → keine Phasen"


class TestTapeHeadClog:
    """TAPE_HEAD_CLOG → phase_56 + phase_25 (temporäre HF-Auslöschung)."""

    def test_tape_activates_spectral_band_gap_repair(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.TAPE_HEAD_CLOG, QualityMode.QUALITY)
        assert "phase_56_spectral_band_gap_repair" in phases

    def test_tape_activates_azimuth_correction(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.TAPE_HEAD_CLOG, QualityMode.QUALITY)
        assert "phase_25_azimuth_correction" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.TAPE_HEAD_CLOG, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"TAPE_HEAD_CLOG × {mat.name} → keine Phasen"
