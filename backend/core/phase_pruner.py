"""
§AC: Intelligent Phase Pruning — nur hörbare Verbesserungen ausführen.

Nicht alle 39–64 Phasen bringen für jede Aufnahme einen hörbaren Gewinn.
Phase Pruning analysiert das Audio, die Defekte und das Material und
entscheidet pro Phase: ausführen, überspringen oder mit Minimal-Stärke.

Kriterien:
1. Defekt nicht vorhanden → Phase skip (z.B. kein Hum → phase_02 skip)
2. Defekt unterhalb psychoakustischer Hörschwelle → Phase skip
3. Material/Gerre schließt Phase aus → Phase skip
4. Phase nur bei bestimmten Defekt-Kombinationen nötig → check
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Phase → Defekt-Präsenz-Requirements ──────────────────────────────────────
# §2.59 Fix (2026-07-09): Namen auf tatsächliche DefectType.values() abgestimmt.
# Alle 66 Phasen sind vollständig erfasst. Phasen mit [] sind immer aktiv.
# Der Match ist Substring-basiert (d in defect), d.h. "rumble" matcht "low_freq_rumble".
_PHASE_DEFECT_REQUIREMENTS: dict[str, list[str]] = {
    # --- Grundreinigung ---
    "phase_01_click_removal": ["clicks"],
    "phase_02_hum_removal": ["hum", "motor_interference"],
    "phase_03_denoise": ["high_freq_noise", "modulation_noise", "quantization_noise", "crackle", "aliasing"],
    "phase_04_eq_correction": [],  # immer
    "phase_05_rumble_filter": ["low_freq_rumble"],
    # --- Frequenz / Harmonik / Transienten ---
    "phase_06_frequency_restoration": ["clipping", "bandwidth_loss"],
    "phase_07_harmonic_restoration": ["clipping", "overload_distortion"],
    "phase_08_transient_preservation": ["clicks", "transient_smearing"],
    "phase_09_crackle_removal": ["crackle"],
    # --- Dynamik / Limiting ---
    "phase_10_compression": ["dynamic_compression_excess"],
    "phase_11_limiting": ["clipping"],
    # --- Transport / Wow & Flutter ---
    "phase_12_wow_flutter_fix": [
        "wow",
        "flutter",
        "multiband_wow_flutter",
        "scrape_flutter",
        "speed_calibration_error",
        "transport_bump",
        "pitch_drift",
    ],
    # --- Stereo / Phase ---
    "phase_13_stereo_enhancement": [],  # fast immer
    "phase_14_phase_correction": ["azimuth_error", "phase_issues"],
    "phase_15_stereo_balance": ["stereo_imbalance"],
    # --- Mastering / EQ ---
    "phase_16_final_eq": [],  # immer
    "phase_17_mastering_polish": [],  # immer
    "phase_18_noise_gate": ["high_freq_noise", "modulation_noise"],
    # --- De-Essing / Reverb ---
    "phase_19_de_esser": ["sibilance"],
    "phase_20_reverb_reduction": ["reverb_excess"],
    # --- Exciter / Saturation ---
    "phase_21_exciter": [],  # kreativ, immer
    "phase_22_tape_saturation": ["soft_saturation"],
    # --- Reparatur ---
    "phase_23_spectral_repair": ["dropouts", "bandwidth_loss"],
    "phase_24_dropout_repair": ["dropouts", "dropout_oxide", "dropout_head_contact", "dropout_splice"],
    "phase_25_azimuth_correction": ["azimuth_error"],
    "phase_26_dynamic_range_expansion": ["dynamic_compression_excess"],
    "phase_27_click_pop_removal": ["clicks"],
    "phase_28_surface_noise_profiling": ["crackle"],
    # --- Tape / analoge Defekte ---
    "phase_29_tape_hiss_reduction": ["modulation_noise", "high_freq_noise"],
    "phase_30_dc_offset_removal": ["dc_offset"],
    "phase_31_speed_pitch_correction": ["speed_calibration_error", "pitch_drift"],
    # --- Stereo-Erweiterung ---
    "phase_32_mono_to_stereo": ["stereo_field_collapse"],
    "phase_33_stereo_width_limiter": ["stereo_imbalance"],
    "phase_34_mid_side_processing": [],  # immer
    # --- Dynamik-Processing ---
    "phase_35_multiband_compression": ["dynamic_compression_excess"],
    "phase_36_transient_shaper": [],  # immer
    # --- Bass / Präsenz / Air ---
    "phase_37_bass_enhancement": [],  # immer
    "phase_38_presence_boost": [],  # immer
    "phase_39_air_band_enhancement": ["bandwidth_loss"],
    # --- Loudness / Output ---
    "phase_40_loudness_normalization": [],  # immer
    "phase_41_output_format_optimization": [],  # immer
    # --- Vocal ---
    "phase_42_vocal_enhancement": ["vocal_harshness"],
    "phase_43_ml_deesser": ["sibilance"],
    # --- Instrument-spezifisch ---
    "phase_44_guitar_enhancement": [],  # bedarfsgesteuert
    "phase_45_brass_enhancement": [],  # bedarfsgesteuert
    "phase_46_spatial_enhancement": [],  # bedarfsgesteuert
    # --- Limiter / Stereo / Dereverb ---
    "phase_47_truepeak_limiter": [],  # immer
    "phase_48_stereo_width_enhancer": [],  # immer
    "phase_49_advanced_dereverb": ["reverb_excess"],
    "phase_50_spectral_repair": ["dropouts", "bandwidth_loss"],
    # --- Instrument-spezifisch II ---
    "phase_51_drums_enhancement": [],  # bedarfsgesteuert
    "phase_52_piano_restoration": [],  # bedarfsgesteuert
    # --- Semantic / Dynamics ---
    "phase_53_semantic_audio": [],  # immer
    "phase_54_transparent_dynamics": [],  # immer
    # --- Inpainting / Reparatur ---
    "phase_55_diffusion_inpainting": ["dropouts", "bandwidth_loss", "mpeg_frame_loss"],
    "phase_56_spectral_band_gap_repair": ["bandwidth_loss"],
    # --- Fortgeschrittene analoge Defekte ---
    "phase_57_print_through_reduction": ["print_through"],
    "phase_58_lyrics_guided_enhancement": [],  # content-abhängig
    "phase_59_modulation_noise_reduction": ["modulation_noise"],
    "phase_60_inner_groove_distortion_repair": ["inner_groove_distortion"],
    "phase_61_groove_echo_cancellation": ["groove_echo"],
    "phase_62_crosstalk_cancellation": ["crosstalk"],
    "phase_63_intermodulation_reduction": ["intermodulation_distortion"],
    "phase_64_tape_splice_repair": ["tape_splice_artifact"],
    # --- Vocal / Stem ---
    "phase_65_vocal_naturalness_restoration": ["vocal_harshness"],
    "phase_66_stem_targeted_nr": ["high_freq_noise", "modulation_noise"],
}

# ── Material-spezifische Skip-Phasen ──────────────────────────────────────────
# §2.59: Alle Digital-Materialien teilen dieselben Analog-Phasen-Skips.
# Bei digitalen Quellen gibt es keinen mechanischen Transport, keine Nadel,
# keinen Magnetkopf → diese Phasen sind physikalisch sinnlos.
_ANALOG_ONLY_PHASES: list[str] = [
    "phase_02_hum_removal",
    "phase_05_rumble_filter",
    "phase_09_crackle_removal",
    "phase_12_wow_flutter_fix",
    "phase_22_tape_saturation",
    "phase_25_azimuth_correction",
    "phase_28_surface_noise_profiling",
    "phase_29_tape_hiss_reduction",
    "phase_57_print_through_reduction",
    "phase_60_inner_groove_distortion_repair",
    "phase_61_groove_echo_cancellation",
    "phase_64_tape_splice_repair",
]

_MATERIAL_SKIP_PHASES: dict[str, list[str]] = dict.fromkeys(
    ("aac", "cd_digital", "dat", "minidisc", "mp3_high", "mp3_low", "streaming"), _ANALOG_ONLY_PHASES
)


@dataclass
class PruningResult:
    """Ergebnis der Phase-Pruning-Analyse."""

    kept_phases: list[str] = field(default_factory=list)
    skipped_phases: list[str] = field(default_factory=list)
    reduced_phases: dict[str, float] = field(default_factory=dict)  # phase → reduzierte Stärke
    reasons: dict[str, str] = field(default_factory=dict)
    reduction_pct: float = 0.0


class IntelligentPhasePruner:
    """Analysiert und reduziert den Phasenplan auf hörbar notwendige Phasen."""

    def __init__(self) -> None:
        pass

    def prune(
        self,
        phases: list[str],
        defect_types: list[str] | None = None,
        material: str = "unknown",
        defect_severities: dict[str, float] | None = None,
        audio_duration_s: float = 0.0,
        restoration_context: dict[str, Any] | None = None,
    ) -> PruningResult:
        """Reduziert den Phasenplan auf das Wesentliche.

        Args:
            phases: Vollständiger PID-Phasenplan
            defect_types: Detektierte Defekt-Typen (lowercase)
            material: Material-Typ
            defect_severities: Defekt-Schweregrade (0–1)
            audio_duration_s: Audio-Dauer in Sekunden
        """
        defects_lower = [d.lower() for d in (defect_types or [])]
        sevs = defect_severities or {}
        result = PruningResult()

        # Material-spezifische Skips
        material_skips = set(_MATERIAL_SKIP_PHASES.get(material, []))

        # §2.59: Kontext-abhängige Entscheidungen
        _ctx = restoration_context or {}
        _era = _ctx.get("decade")
        _genre = _ctx.get("genre_label", "")
        _is_vintage = _era is not None and _era <= 1980
        _ctx.get("vocal_detected", False)

        logger.debug(
            "PhasePruner: pruning %d phases | material=%s era=%s genre=%s defects=%s",
            len(phases),
            material,
            _era,
            _genre,
            sorted(defects_lower)[:25] if defects_lower else "[]",
        )
        for phase_id in phases:
            # 1. Material-basierter Skip
            if phase_id in material_skips:
                result.skipped_phases.append(phase_id)
                result.reasons[phase_id] = f"Material {material} benötigt diese Phase nicht"
                continue

            # 2. Defekt-Präsenz-Check
            required = _PHASE_DEFECT_REQUIREMENTS.get(phase_id, [])

            # §2.59: Chirurgische Phasen werden NIE geprunt
            _ctx = restoration_context or {}
            _surgical_defects = _ctx.get("surgical_defect_types", [])
            if _surgical_defects and required:
                # Prüfe ob diese Phase chirurgische Defekte behandelt
                _surgical_match = any(req in defect for req in required for defect in _surgical_defects)
                if _surgical_match:
                    result.kept_phases.append(phase_id)
                    logger.debug("PhasePruner KEEP %s: surgical defect protection", phase_id)
                    continue
            if required:
                matching_defects = [d for d in required if any(d in defect for defect in defects_lower)]
                if not matching_defects:
                    # Defekt nicht vorhanden → Skip
                    result.skipped_phases.append(phase_id)
                    result.reasons[phase_id] = f"Kein {'/'.join(required)} detektiert"
                    logger.debug(
                        "PhasePruner ueberspringen %s: no defect match (needs=%s, have=%s)",
                        phase_id,
                        required,
                        sorted(defects_lower)[:15],
                    )
                    continue

                # 3. Psychoakustische Hörschwelle: sehr schwache Defekte → reduzierte Stärke
                min_sev = min(sevs.get(d, 1.0) for d in matching_defects if d in sevs) if sevs else 1.0
                if min_sev < 0.15:
                    result.reduced_phases[phase_id] = max(0.1, min_sev * 2.0)
                    result.reasons[phase_id] = f"Defekt sehr schwach (sev={min_sev:.2f})"

            # Phase wird behalten
            result.kept_phases.append(phase_id)

        result.reduction_pct = (1.0 - len(result.kept_phases) / max(1, len(phases))) * 100.0

        return result
