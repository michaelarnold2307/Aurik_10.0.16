"""
Phase Skipping Logic für AURIK

Intelligentes Überspringen unnötiger Processing-Phasen basierend auf:
- DefectAnalysis (welche Defekte sind vorhanden?)
- SemanticProfile (welcher Content?)

Ermöglicht 20-40% Speedup für saubere Inputs ohne Qualitätsverlust.

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-08
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from backend.core.defect_analysis import DefectAnalysis, SourceMedium

logger = logging.getLogger(__name__)


class ProcessingPhase(Enum):
    """Verarbeitet phases in UnifiedRestorerV2."""

    PHASE_1_FORENSIC = "phase_1_forensic"
    PHASE_2_DENOISE = "phase_2_denoise"
    PHASE_3_ENHANCE = "phase_3_enhance"
    PHASE_4_DECLIP = "phase_4_declip"
    PHASE_5_CLICK_REMOVAL = "phase_5_click_removal"
    PHASE_6_DEHUM = "phase_6_dehum"
    PHASE_7_DROPOUT_REPAIR = "phase_7_dropout_repair"
    PHASE_8_SPECTRAL_REPAIR = "phase_8_spectral_repair"
    PHASE_9_DYNAMICS = "phase_9_dynamics"
    PHASE_10_FINALIZE = "phase_10_finalize"


@dataclass
class PhaseSkipDecision:
    """Decision whether to skip a phase."""

    phase: ProcessingPhase
    skip: bool
    reason: str
    confidence: float  # 0-1


class PhaseSkipper:
    """Decides which processing phases can be skipped."""

    def __init__(self, conservative: bool = False, min_confidence: float = 0.8):
        """
        Initialisiert phase skipper.

        Args:
            conservative: If True, skip fewer phases (safer)
            min_confidence: Minimum confidence to skip a phase
        """
        self.conservative = conservative
        self.min_confidence = min_confidence

    def analyze_pipeline(
        self, defect_analysis: DefectAnalysis, semantic_profile: Any | None = None
    ) -> dict[ProcessingPhase, PhaseSkipDecision]:
        """
        Analysiert, welche Phasen übersprungen werden können.

        Args:
            defect_analysis: Defect analysis of audio
            semantic_profile: Optional semantic profile (for advanced logic)

        Returns:
            Dictionary mapping each phase to skip decision
        """
        decisions = {}

        # Phase 1: Forensic Analysis - NEVER skip (needed for insight)
        decisions[ProcessingPhase.PHASE_1_FORENSIC] = PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_1_FORENSIC,
            skip=False,
            reason="Forensic analysis always needed for insights",
            confidence=1.0,
        )

        # Phase 2: Denoise
        decisions[ProcessingPhase.PHASE_2_DENOISE] = self._should_skip_denoise(defect_analysis)

        # Phase 3: Enhance - Rarely skip (quality improvement)
        decisions[ProcessingPhase.PHASE_3_ENHANCE] = PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_3_ENHANCE, skip=False, reason="Enhancement usually beneficial", confidence=0.7
        )

        # Phase 4: Declipping
        decisions[ProcessingPhase.PHASE_4_DECLIP] = self._should_skip_declip(defect_analysis)

        # Phase 5: Click Removal
        decisions[ProcessingPhase.PHASE_5_CLICK_REMOVAL] = self._should_skip_click_removal(defect_analysis)

        # Phase 6: Dehum
        decisions[ProcessingPhase.PHASE_6_DEHUM] = self._should_skip_dehum(defect_analysis)

        # Phase 7: Dropout Repair
        decisions[ProcessingPhase.PHASE_7_DROPOUT_REPAIR] = self._should_skip_dropout_repair(defect_analysis)

        # Phase 8: Spectral Repair
        decisions[ProcessingPhase.PHASE_8_SPECTRAL_REPAIR] = self._should_skip_spectral_repair(defect_analysis)

        # Phase 9: Dynamics - Rarely skip (mastering stage)
        decisions[ProcessingPhase.PHASE_9_DYNAMICS] = PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_9_DYNAMICS,
            skip=False,
            reason="Dynamics processing usually beneficial",
            confidence=0.7,
        )

        # Phase 10: Finalize - NEVER skip (final stage)
        decisions[ProcessingPhase.PHASE_10_FINALIZE] = PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_10_FINALIZE, skip=False, reason="Finalization always required", confidence=1.0
        )

        return decisions

    def _should_skip_denoise(self, defect_analysis: DefectAnalysis) -> PhaseSkipDecision:
        """Decide whether to skip denoising."""
        # Skip if noise floor is very low
        if defect_analysis.noise_floor_db < -60.0 and not defect_analysis.has_hiss:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_2_DENOISE,
                skip=not self.conservative,
                reason=f"Very low noise floor ({defect_analysis.noise_floor_db:.1f} dB)",
                confidence=0.9,
            )

        # Skip if digital source with low noise
        if defect_analysis.medium == SourceMedium.DIGITAL and defect_analysis.noise_floor_db < -50.0:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_2_DENOISE,
                skip=not self.conservative,
                reason="Clean digital source",
                confidence=0.8,
            )

        return PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_2_DENOISE, skip=False, reason="Denoising needed", confidence=1.0
        )

    def _should_skip_declip(self, defect_analysis: DefectAnalysis) -> PhaseSkipDecision:
        """Decide whether to skip declipping."""
        # Skip if clipping is minimal (<0.5%)
        if defect_analysis.clipping_percentage < 0.5:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_4_DECLIP,
                skip=True,
                reason=f"Minimal clipping ({defect_analysis.clipping_percentage:.2f}%)",
                confidence=0.95,
            )

        # Skip if clipping is light (<1%) and conservative mode is off
        if defect_analysis.clipping_percentage < 1.0 and not self.conservative:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_4_DECLIP,
                skip=True,
                reason=f"Light clipping ({defect_analysis.clipping_percentage:.2f}%)",
                confidence=0.85,
            )

        return PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_4_DECLIP,
            skip=False,
            reason=f"Significant clipping ({defect_analysis.clipping_percentage:.1f}%)",
            confidence=1.0,
        )

    def _should_skip_click_removal(self, defect_analysis: DefectAnalysis) -> PhaseSkipDecision:
        """Decide whether to skip click removal."""
        # Always process vinyl/shellac (may have undetected clicks)
        if defect_analysis.medium in [SourceMedium.VINYL, SourceMedium.SHELLAC]:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_5_CLICK_REMOVAL,
                skip=False,
                reason=f"Vinyl/shellac source (medium={defect_analysis.medium.value})",
                confidence=1.0,
            )

        # Skip if no clicks detected and not analog medium
        if defect_analysis.click_count == 0:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_5_CLICK_REMOVAL,
                skip=True,
                reason="No clicks detected, not vinyl/shellac",
                confidence=0.9,
            )

        # Skip if very few clicks (<0.1/sec) and not conservative
        if defect_analysis.click_density < 0.1 and not self.conservative:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_5_CLICK_REMOVAL,
                skip=True,
                reason=f"Very few clicks ({defect_analysis.click_count})",
                confidence=0.8,
            )

        return PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_5_CLICK_REMOVAL,
            skip=False,
            reason=f"Clicks detected ({defect_analysis.click_count})",
            confidence=1.0,
        )

    def _should_skip_dehum(self, defect_analysis: DefectAnalysis) -> PhaseSkipDecision:
        """Decide whether to skip dehum."""
        # Skip if no hum detected
        if not defect_analysis.has_hum:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_6_DEHUM, skip=True, reason="No hum detected", confidence=0.9
            )

        return PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_6_DEHUM, skip=False, reason="Hum detected (50/60 Hz)", confidence=1.0
        )

    def _should_skip_dropout_repair(self, defect_analysis: DefectAnalysis) -> PhaseSkipDecision:
        """Decide whether to skip dropout repair."""
        # Always process tape sources (may have undetected dropouts)
        if defect_analysis.medium in [SourceMedium.CASSETTE, SourceMedium.REEL_TAPE, SourceMedium.DAT]:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_7_DROPOUT_REPAIR,
                skip=False,
                reason=f"Tape source (medium={defect_analysis.medium.value})",
                confidence=1.0,
            )

        # Skip if no dropouts detected
        if defect_analysis.dropout_count == 0:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_7_DROPOUT_REPAIR,
                skip=True,
                reason="No dropouts detected, not tape source",
                confidence=0.9,
            )

        return PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_7_DROPOUT_REPAIR,
            skip=False,
            reason=f"Dropouts detected ({defect_analysis.dropout_count})",
            confidence=1.0,
        )

    def _should_skip_spectral_repair(self, defect_analysis: DefectAnalysis) -> PhaseSkipDecision:
        """Decide whether to skip spectral repair."""
        # Skip if no spectral artifacts
        if not defect_analysis.has_aliasing and not defect_analysis.has_codec_artifacts:
            return PhaseSkipDecision(
                phase=ProcessingPhase.PHASE_8_SPECTRAL_REPAIR,
                skip=True,
                reason="No spectral artifacts detected",
                confidence=0.85,
            )

        return PhaseSkipDecision(
            phase=ProcessingPhase.PHASE_8_SPECTRAL_REPAIR,
            skip=False,
            reason="Spectral artifacts detected",
            confidence=0.9,
        )

    def get_skippable_phases(
        self, defect_analysis: DefectAnalysis, semantic_profile: Any | None = None
    ) -> list[ProcessingPhase]:
        """
        Gibt zurück: list of phases that can be skipped.

        Args:
            defect_analysis: Defect analysis
            semantic_profile: Optional semantic profile

        Returns:
            List of skippable phases
        """
        decisions = self.analyze_pipeline(defect_analysis, semantic_profile)

        skippable = [
            phase
            for phase, decision in decisions.items()
            if decision.skip and decision.confidence >= self.min_confidence
        ]

        return skippable

    def estimate_speedup(self, defect_analysis: DefectAnalysis, semantic_profile: Any | None = None) -> float:
        """
        Schätzt processing time speedup.

        Args:
            defect_analysis: Defect analysis
            semantic_profile: Optional semantic profile

        Returns:
            Speedup factor (e.g., 1.3 = 30% faster)
        """
        skippable = self.get_skippable_phases(defect_analysis, semantic_profile)

        # Each phase takes roughly equal time (simplified model)
        total_phases = len(ProcessingPhase)
        skipped_phases = len(skippable)

        if skipped_phases == 0:
            return 1.0  # No speedup

        # Speedup = 1 / (remaining_work_percentage)
        remaining_percentage = (total_phases - skipped_phases) / total_phases
        speedup = 1.0 / remaining_percentage

        return speedup

    def generate_report(self, defect_analysis: DefectAnalysis, semantic_profile: Any | None = None) -> str:
        """
        Generiert human-readable phase skipping report.

        Args:
            defect_analysis: Defect analysis
            semantic_profile: Optional semantic profile

        Returns:
            Formatted report string
        """
        decisions = self.analyze_pipeline(defect_analysis, semantic_profile)
        speedup = self.estimate_speedup(defect_analysis, semantic_profile)

        report = []
        report.append("=" * 70)
        report.append("AURIK PHASE SKIPPING ANALYSIS")
        report.append("=" * 70)
        report.append(f"\nDefect Analysis: {defect_analysis}")
        report.append(f"\nEstimated Speedup: {speedup:.1f}x ({(speedup - 1) * 100:.0f}% faster)")
        report.append("\n" + "=" * 70)
        report.append("PHASE DECISIONS:")
        report.append("=" * 70)

        for phase in ProcessingPhase:
            decision = decisions[phase]
            status = "SKIP" if decision.skip else "PROCESS"
            report.append(f"\n{phase.value}:")
            report.append(f"  Status:     {status}")
            report.append(f"  Reason:     {decision.reason}")
            report.append(f"  Confidence: {decision.confidence:.1%}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


if __name__ == "__main__":
    # Demo
    logger.debug("AURIK Verarbeitungsschritt Skipping Logic")
    logger.debug("=" * 60)
    logger.debug("\nEnables 20-40%% speedup for clean inputs by skipping")
    logger.debug("unnecessary processing phases.")
    logger.debug("\nPhases that can be uebersprungen:")
    logger.debug("  • Declipping (if <0.5%% clipped)")
    logger.debug("  • Click removal (if no clicks, not vinyl)")
    logger.debug("  • Dehum (if no 50/60 Hz hum)")
    logger.debug("  • Dropout repair (if no dropouts, not tape)")
    logger.debug("  • Spectral repair (if no artifacts)")
    logger.debug("  • Denoising (if very low noise floor)")
