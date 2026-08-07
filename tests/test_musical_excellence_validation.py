from typing import Any

import pytest

"""
Musikalische Exzellenz - Validierung aller 42 Phasen
====================================================

Überprüft, ob alle Phasen die 7 musikalischen Ziele erreichen:

1. BRILLANZ       - HF-Klarheit, Air-Band (12-20 kHz)
2. WÄRME          - Harmonische Fülle, LF-Richness (80-400 Hz)
3. NATÜRLICHKEIT  - Keine Artefakte, organischer Klang
4. AUTHENTIZITÄT  - Material-Treue, Charakter erhalten
5. EMOTIONALITÄT  - Dynamik, Transienten, Expression
6. TRANSPARENZ    - Keine Verschleierung, Detail-Auflösung
7. BASS-KRAFT     - LF-Fundament, Sub-Bass (20-80 Hz)

Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.defect_scanner import MaterialType
from backend.core.phases import (
    AirBandEnhancement,
    AzimuthCorrectionPhaseV2,
    BassEnhancement,
    ClickPopRemoval,
    ClickRemovalPhase,
    CompressionPhase,
    CrackleRemovalPhase,
    DCOffsetRemoval,
    DeEsserPhase,
    DenoisePhase,
    DropoutRepairPhase,
    DynamicRangeExpansion,
    EQCorrectionPhase,
    Exciter,
    FinalEQ,
    FrequencyRestorationPhase,
    HarmonicRestorationPhase,
    HumRemovalPhase,
    LimitingPhase,
    LoudnessNormalizationPhase,
    MasteringPolishPhase,
    MidSideProcessing,
    MonoToStereoPhaseV2,
    MultibandCompressionPhase,
    NoiseGate,
    OutputFormatOptimization,
    PhaseCorrection,
    PresenceBoost,
    ReverbReduction,
    RumbleFilterPhase,
    SpectralRepair,
    SpeedPitchCorrectionPhase,
    StereoBalancePhaseV2,
    StereoEnhancementPhaseV2,
    StereoWidthLimiterPhaseV2,
    SurfaceNoiseProfiling,
    TapeHissReductionPhase,
    TapeSaturation,
    TransientPreservationPhase,
    TransientShaper,
    VocalEnhancement,
    WowFlutterFix,
)

# ==================== Musikalische Ziele ====================

MUSICAL_GOALS = {
    "brillanz": {
        "name": "Brillanz (HF-Klarheit)",
        "frequency_range": (12000, 20000),
        "target": "Klare Höhen, Air-Band-Präsenz",
        "critical_phases": [6, 7, 16, 21, 38, 39],  # Frequency/Harmonic Restoration, EQ, Exciter, Presence, Air
    },
    "waerme": {
        "name": "Wärme (Harmonische Fülle)",
        "frequency_range": (80, 400),
        "target": "Harmonische Richness, Körper",
        "critical_phases": [4, 6, 7, 16, 22, 37],  # EQ, Freq/Harm Restoration, Tape Saturation, Bass Enhancement
    },
    "natuerlichkeit": {
        "name": "Natürlichkeit (Keine Artefakte)",
        "frequency_range": None,
        "target": "Organischer Klang, keine digitalen Artefakte",
        "critical_phases": [1, 2, 3, 9, 27, 28],  # Click/Hum/Denoise/Crackle Removal, Surface Noise
    },
    "authentizitaet": {
        "name": "Authentizität (Material-Treue)",
        "frequency_range": None,
        "target": "Charakter des Originals erhalten",
        "critical_phases": [8, 13, 22, 25, 28, 31],  # Transient Preservation, Stereo, Tape, Azimuth, Surface
    },
    "emotionalitaet": {
        "name": "Emotionalität (Dynamik & Expression)",
        "frequency_range": None,
        "target": "Dynamischer Ausdruck, Transienten",
        "critical_phases": [
            8,
            10,
            11,
            26,
            36,
        ],  # Transient Preservation, Compression, Limiting, Range Expansion, Shaper
    },
    "transparenz": {
        "name": "Transparenz (Detail-Auflösung)",
        "frequency_range": None,
        "target": "Keine Verschleierung, klare Trennung",
        "critical_phases": [
            3,
            13,
            14,
            20,
            23,
            34,
        ],  # Denoise, Stereo Enhancement, Phase Correction, Reverb, Spectral, M/S
    },
    "basskraft": {
        "name": "Bass-Kraft (LF-Fundament)",
        "frequency_range": (20, 80),
        "target": "Solides Sub-Bass-Fundament",
        "critical_phases": [5, 16, 30, 37],  # Rumble Filter, Final EQ, DC Offset, Bass Enhancement
    },
}


# Phase-Kategorien nach musikalischen Zielen
PHASE_MUSICAL_MAPPING = {
    # Defect Removal (Natürlichkeit)
    1: ["natuerlichkeit"],
    2: ["natuerlichkeit"],
    3: ["natuerlichkeit", "transparenz"],
    4: ["waerme", "brillanz"],
    5: ["basskraft", "natuerlichkeit"],
    6: ["brillanz", "waerme"],
    7: ["brillanz", "waerme"],
    8: ["authentizitaet", "emotionalitaet"],
    9: ["natuerlichkeit"],
    # Dynamics & Spatial (Emotionalität, Transparenz)
    10: ["emotionalitaet"],
    11: ["emotionalitaet"],
    12: ["authentizitaet"],
    13: ["transparenz", "authentizitaet"],
    14: ["transparenz"],
    15: ["transparenz"],
    16: ["brillanz", "waerme", "basskraft"],
    17: ["brillanz", "waerme"],
    18: ["natuerlichkeit"],
    19: ["brillanz", "natuerlichkeit"],
    # Advanced Restoration (Authentizität)
    20: ["transparenz"],
    21: ["brillanz"],
    22: ["waerme", "authentizitaet"],
    23: ["transparenz", "natuerlichkeit"],
    24: ["natuerlichkeit"],
    25: ["authentizitaet"],
    26: ["emotionalitaet"],
    27: ["natuerlichkeit"],
    28: ["natuerlichkeit", "authentizitaet"],
    29: ["natuerlichkeit"],
    # Format & Enhancement
    30: ["basskraft", "natuerlichkeit"],
    31: ["authentizitaet"],
    32: ["transparenz"],
    33: ["transparenz"],
    34: ["transparenz"],
    35: ["emotionalitaet"],
    36: ["emotionalitaet", "authentizitaet"],
    37: ["basskraft", "waerme"],
    38: ["brillanz"],
    39: ["brillanz"],
    40: ["emotionalitaet"],
    41: ["transparenz"],
    42: ["brillanz", "natuerlichkeit"],
}


def generate_test_audio_with_character(duration=2.0, sample_rate=44100, character="neutral"):
    """Generate test audio with specific musical characteristics."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base: Musical content (A4 = 440 Hz fundamental)
    fundamental = 440.0
    audio = 0.3 * np.sin(2 * np.pi * fundamental * t)

    # Add harmonics for warmth and brilliance
    harmonics = [2, 3, 4, 5, 6, 7, 8]
    for h in harmonics:
        amplitude = 0.3 / h  # Natural harmonic decay
        audio += amplitude * np.sin(2 * np.pi * fundamental * h * t)

    # Add transients for emotionality (drum hits)
    if character in ["dynamic", "neutral"]:
        for hit_time in [0.5, 1.0, 1.5]:
            hit_pos = int(hit_time * sample_rate)
            if hit_pos < len(audio):
                transient = np.exp(-50 * np.linspace(0, 0.05, int(0.05 * sample_rate)))
                audio[hit_pos : hit_pos + len(transient)] += 0.8 * transient

    # Add character-specific content
    if character == "with_defects":
        # Add clicks (natürlichkeit test)
        for i in range(5):
            pos = int(sample_rate * (i * 0.4 + 0.1))
            if pos < len(audio) - 1:
                audio[pos] += 0.9
                audio[pos + 1] -= 0.9
        # Add hum (natürlichkeit test)
        audio += 0.1 * np.sin(2 * np.pi * 60 * t)
        # Add noise (natürlichkeit test)
        audio += np.random.normal(0, 0.03, len(audio))

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9

    return audio.astype(np.float32)


def measure_musical_goal(audio_before, audio_after, goal_key, sample_rate=44100):
    """
    Measure how well a phase achieves a musical goal.

    Returns:
        score (0.0-1.0): 1.0 = perfect, 0.0 = failure
        details: dict with measurements
    """
    goal = MUSICAL_GOALS[goal_key]
    details = {}

    if audio_before.ndim == 2:
        audio_before = np.mean(audio_before, axis=1)
    if audio_after.ndim == 2:
        audio_after = np.mean(audio_after, axis=1)

    # Frequency-based goals
    if goal["frequency_range"]:  # type: ignore[index]
        from scipy import signal as sp_signal

        f_low, f_high = goal["frequency_range"]  # type: ignore[index]
        nyquist = sample_rate / 2.0

        # Band energy before/after
        sos = sp_signal.butter(4, [f_low / nyquist, min(f_high, nyquist * 0.95) / nyquist], btype="band", output="sos")
        band_before = sp_signal.sosfilt(sos, audio_before)
        band_after = sp_signal.sosfilt(sos, audio_after)

        energy_before = np.sqrt(np.mean(band_before**2))
        energy_after = np.sqrt(np.mean(band_after**2))

        if goal_key in ["brillanz", "basskraft", "waerme"]:
            # Goal: Enhance or preserve energy
            if energy_before > 1e-6:
                enhancement = energy_after / energy_before
                score = min(1.0, enhancement)  # Enhancement is good
            else:
                score = 1.0 if energy_after > 1e-6 else 0.5

            details["energy_before"] = float(energy_before)
            details["energy_after"] = float(energy_after)
            details["enhancement"] = float(energy_after / (energy_before + 1e-9))
        else:
            score = 1.0

    # Artifact-based goals (natürlichkeit)
    elif goal_key == "natuerlichkeit":
        # Measure noise/artifact reduction
        # High-frequency noise should be reduced
        from scipy import signal as sp_signal

        # Noise estimation (HF content above 8 kHz)
        nyquist = sample_rate / 2.0
        sos = sp_signal.butter(4, 8000 / nyquist, btype="high", output="sos")
        hf_before = sp_signal.sosfilt(sos, audio_before)
        hf_after = sp_signal.sosfilt(sos, audio_after)

        noise_before = np.std(hf_before)
        noise_after = np.std(hf_after)

        if noise_before > 1e-6:
            noise_reduction = 1.0 - (noise_after / noise_before)
            score = max(0.0, min(1.0, 0.5 + noise_reduction))  # 0.5 baseline, +0.5 for reduction
        else:
            score = 1.0

        details["noise_before"] = float(noise_before)
        details["noise_after"] = float(noise_after)
        details["noise_reduction"] = float(noise_reduction) if noise_before > 1e-6 else 0.0

    # Dynamic-based goals (emotionalität)
    elif goal_key == "emotionalitaet":
        # Measure dynamic range preservation
        rms_before = np.sqrt(np.mean(audio_before**2))
        rms_after = np.sqrt(np.mean(audio_after**2))
        peak_before = np.max(np.abs(audio_before))
        peak_after = np.max(np.abs(audio_after))

        crest_before = peak_before / (rms_before + 1e-9)
        crest_after = peak_after / (rms_after + 1e-9)

        # Good: Preserve or slightly improve crest factor
        crest_ratio = crest_after / (crest_before + 1e-9)
        if 0.8 <= crest_ratio <= 1.2:
            score = 1.0
        elif 0.6 <= crest_ratio <= 1.5:
            score = 0.8
        else:
            score = 0.5

        details["crest_before"] = float(crest_before)
        details["crest_after"] = float(crest_after)
        details["crest_preservation"] = float(crest_ratio)

    # Transparency/Authenticity
    elif goal_key in ["transparenz", "authentizitaet"]:
        # Measure correlation (should stay high = authentic/transparent)
        # Truncate to same length
        min_len = min(len(audio_before), len(audio_after))
        corr = np.corrcoef(audio_before[:min_len], audio_after[:min_len])[0, 1]
        score = max(0.0, corr)  # 1.0 = perfect correlation

        details["correlation"] = float(corr)
        details["interpretation"] = "high_correlation_good" if corr > 0.8 else "low_correlation_bad"  # type: ignore[assignment]

    else:
        # Default: Passthrough is good
        score = 1.0
        details["method"] = "default_passthrough"  # type: ignore[assignment]

    return score, details


def validate_all_phases_for_musical_excellence():
    """Validate all 42 phases against 7 musical goals."""

    print("\n" + "=" * 80)
    print(" " * 15 + "MUSIKALISCHE EXZELLENZ - 42-PHASEN VALIDIERUNG")
    print("=" * 80 + "\n")

    print("7 Musikalische Ziele:")
    for goal_key, goal in MUSICAL_GOALS.items():
        print(f"  • {goal['name']}: {goal['target']}")  # type: ignore[index]
    print()

    # Test setup
    sample_rate = 44100

    # Results tracking
    phase_results = {}
    goal_coverage: Any = {key: [] for key in MUSICAL_GOALS}

    # All phases to test
    phases = [
        (1, ClickRemovalPhase),
        (2, HumRemovalPhase),
        (3, DenoisePhase),
        (4, EQCorrectionPhase),
        (5, RumbleFilterPhase),
        (6, FrequencyRestorationPhase),
        (7, HarmonicRestorationPhase),
        (8, TransientPreservationPhase),
        (9, CrackleRemovalPhase),
        (10, CompressionPhase),
        (11, LimitingPhase),
        (12, WowFlutterFix),
        (13, StereoEnhancementPhaseV2),
        (14, PhaseCorrection),
        (15, StereoBalancePhaseV2),
        (16, FinalEQ),
        (17, MasteringPolishPhase),
        (18, NoiseGate),
        (19, DeEsserPhase),
        (20, ReverbReduction),
        (21, Exciter),
        (22, TapeSaturation),
        (23, SpectralRepair),
        (24, DropoutRepairPhase),
        (25, AzimuthCorrectionPhaseV2),
        (26, DynamicRangeExpansion),
        (27, ClickPopRemoval),
        (28, SurfaceNoiseProfiling),
        (29, TapeHissReductionPhase),
        (30, DCOffsetRemoval),
        (31, SpeedPitchCorrectionPhase),
        (32, MonoToStereoPhaseV2),
        (33, StereoWidthLimiterPhaseV2),
        (34, MidSideProcessing),
        (35, MultibandCompressionPhase),
        (36, TransientShaper),
        (37, BassEnhancement),
        (38, PresenceBoost),
        (39, AirBandEnhancement),
        (40, LoudnessNormalizationPhase),
        (41, OutputFormatOptimization),
        (42, VocalEnhancement),
    ]

    print(f"Testing {len(phases)} phases...\n")

    for phase_num, phase_class in phases:
        phase_name = f"Phase {phase_num:02d}"

        try:
            phase = phase_class()

            # Get musical goals for this phase
            goals_for_phase = PHASE_MUSICAL_MAPPING.get(phase_num, [])

            if not goals_for_phase:
                print(f"{phase_name}: No specific musical goals assigned (utility phase)")
                continue

            # Generate appropriate test audio
            if "natuerlichkeit" in goals_for_phase:
                audio = generate_test_audio_with_character(character="with_defects")
            else:
                audio = generate_test_audio_with_character(character="neutral")

            # Process
            try:
                processed = phase.process(audio, sample_rate, MaterialType.CD_DIGITAL)  # type: ignore[arg-type]
            except TypeError:
                processed = phase.process(audio, sample_rate)

            # Extract audio from PhaseResult if needed
            if hasattr(processed, "audio"):
                processed_audio = processed.audio
            elif isinstance(processed, tuple):
                processed_audio = processed[0]
            else:
                processed_audio = processed  # type: ignore[assignment]

            # Measure each goal
            phase_scores = {}
            for goal_key in goals_for_phase:
                score, details = measure_musical_goal(audio, processed_audio, goal_key, sample_rate)
                phase_scores[goal_key] = score
                goal_coverage[goal_key].append((phase_num, score))

            # Report
            avg_score = np.mean(list(phase_scores.values()))
            status = "✓" if avg_score > 0.7 else "⚠" if avg_score > 0.5 else "✗"

            goals_str = ", ".join(
                [f"{MUSICAL_GOALS[g]['name'].split('(')[0].strip()}: {phase_scores[g]:.2f}" for g in goals_for_phase]  # type: ignore[index]
            )

            print(f"{status} {phase_name}: {goals_str} (Ø {avg_score:.2f})")

            phase_results[phase_num] = {"goals": goals_for_phase, "scores": phase_scores, "avg_score": avg_score}

        except Exception as e:
            print(f"✗ {phase_name}: ERROR - {str(e)[:60]}")

    # Summary
    print("\n" + "=" * 80)
    print("MUSIKALISCHE ZIELE - COVERAGE & SCORES")
    print("=" * 80 + "\n")

    for goal_key, goal in MUSICAL_GOALS.items():
        phases_for_goal = goal_coverage[goal_key]

        if phases_for_goal:
            avg_score = np.mean([score for _, score in phases_for_goal])
            phase_list = [f"{p}" for p, _ in phases_for_goal]
            status = "✅" if avg_score > 0.7 else "⚠️" if avg_score > 0.5 else "❌"

            print(f"{status} {goal['name']}")  # type: ignore[index]
            print(f"   Phasen: {', '.join(phase_list)}")
            print(f"   Durchschnitt: {avg_score:.2f}")
            print(f"   Critical Phases: {goal['critical_phases']}")  # type: ignore[index]
            print()

    # Overall assessment
    all_scores = [r["avg_score"] for r in phase_results.values()]
    overall_avg = np.mean(all_scores) if all_scores else 0.0

    print("=" * 80)
    print(f"GESAMTBEWERTUNG MUSIKALISCHE EXZELLENZ: {overall_avg:.2f}")

    if overall_avg >= 0.8:
        print("✅ EXZELLENT - Alle musikalischen Ziele werden hervorragend erreicht!")
    elif overall_avg >= 0.7:
        print("🟢 GUT - Musikalische Ziele werden solid erreicht")
    elif overall_avg >= 0.6:
        print("🟡 BEFRIEDIGEND - Verbesserungen bei einigen Phasen empfohlen")
    else:
        print("🔴 KRITISCH - Musikalische Exzellenz gefährdet, Optimierung erforderlich!")

    print("=" * 80 + "\n")

    return phase_results, goal_coverage


if __name__ == "__main__":
    results, coverage = validate_all_phases_for_musical_excellence()
