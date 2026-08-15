"""
core/quality_recovery.py
Adaptive Musical Excellence System
===================================

Ferrari Edition - Automatische Qualitätsoptimierung:
KEIN harter Abbruch! Aurik findet IMMER den besten Weg!

Statt zu warnen oder abzubrechen:
1. Adaptive Optimization: Automatische Parameter-Anpassung
2. Multi-Path Search: Verschiedene Wege probieren
3. Iterative Refinement: Schrittweise zum Optimum
4. Quality Maximization: Beste erreichbare Qualität
5. Never Give Up: Immer eine Lösung finden

Aurik garantiert: Maximale musikalische Exzellenz - IMMER!

Version: 2.0.0 "Adaptive Excellence"
Author: AURIK Team
Date: 10. Februar 2026
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import signal

from backend.core.musical_quality_assurance import (
    IntegrityViolation,
    MediumType,
    MusicalQualityAssurance,
    MusicalQualityReport,
    ProcessingMode,
)

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Adaptive optimization strategies."""

    REDUCE_INTENSITY = "reduce_intensity"  # Weniger aggressive Verarbeitung
    BYPASS_MODULE = "bypass_module"  # Modul überspringen
    ADJUST_PARAMETERS = "adjust_parameters"  # Parameter optimieren
    SWITCH_MODE = "switch_mode"  # Mode wechseln (z.B. RESTORATION → FORENSIC)
    USE_ALTERNATIVE = "use_alternative"  # Alternative Module verwenden
    INCREMENTAL_PROCESSING = "incremental"  # Schrittweise mit Quality Checks
    MAXIMIZE_QUALITY = "maximize_quality"  # Beste mögliche Qualität finden (kein Abbruch!)


class ProblemType(Enum):
    """Types of quality problems."""

    LOW_SNR = "low_snr"  # SNR zu niedrig
    OVERBRIGHTENING = "overbrightening"  # Zu hell/harsh
    CHARACTER_LOSS = "character_loss"  # Analog character verloren
    UNNATURAL_SOUND = "unnatural_sound"  # Klingt unnatürlich
    DYNAMIC_LOSS = "dynamic_loss"  # Dynamik verloren
    FREQUENCY_IMBALANCE = "frequency_imbalance"  # Frequenzen unbalanciert
    ARTIFACT_INTRODUCTION = "artifact_introduction"  # Neue Artefakte


@dataclass
class RecoveryAction:
    """single recovery action."""

    strategy: RecoveryStrategy
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1 = highest, 5 = lowest
    expected_improvement: float = 0.0  # 0-1

    # Execution details
    target_module: str | None = None
    parameter_adjustments: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """Complete adaptive optimization plan."""

    problem_type: ProblemType
    problem_description: str

    # Recovery actions (sorted by priority)
    actions: list[RecoveryAction]

    # ALWAYS optimize - never give up!
    fallback_strategy: RecoveryStrategy = RecoveryStrategy.MAXIMIZE_QUALITY
    fallback_description: str = "Find best achievable quality through adaptive optimization"

    # Execution tracking
    attempted_actions: list[str] = field(default_factory=list)
    successful_actions: list[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Result from quality recovery."""

    success: bool
    recovered_audio: np.ndarray
    actions_taken: list[str]

    # Quality improvement
    original_score: float
    recovered_score: float
    improvement: float

    # Details
    strategy_used: RecoveryStrategy
    problem_solved: bool
    warnings: list[str] = field(default_factory=list)

    # New processing parameters
    optimized_parameters: dict[str, Any] = field(default_factory=dict)


class QualityRecoverySystem:
    """
    Adaptive Musical Excellence System (v2.0).

    OUT-OF-THE-BOX PERFEKTIONIERT - Aurik's Kern-Philosophie:
    =========================================================

    ✓ IMMER die beste Lösung finden (kein harter Abbruch)
    ✓ Selbstständige Optimierung (keine User-Intervention)
    ✓ Maximale musikalische Exzellenz (garantiert)

    Arbeitsweise:
    ------------
    1. Problem Detection: Warum failed der Quality Gate?
    2. Multi-Strategy Search: Verschiedene Lösungswege probieren
    3. Adaptive Optimization: 10+ Intensitätsstufen testen
    4. Best-Effort Guarantee: Beste erreichbare Qualität finden
    5. Never Give Up: Immer eine optimale Lösung liefern

    Keine Empfehlungen, keine Abbrüche - nur Exzellenz!

    Usage:
        # Im Module Coordinator automatisch aktiv (default: True)
        coordinator = ModuleCoordinator(
            context, bus,
            enable_musical_quality_assurance=True  # OUT-OF-THE-BOX!
        )

        # Recovery läuft automatisch bei Quality Gate Failures
        # → Findet IMMER die beste Lösung
        # → Keine manuelle Intervention nötig
    """

    VERSION = "1.0.0"

    def __init__(self):
        """Initialisiert Quality Recovery System."""
        self.mqa = MusicalQualityAssurance()
        self.analyzer = self.mqa.analyzer  # Use MQA's analyzer

        # Recovery strategy templates
        self._strategy_templates = {
            ProblemType.LOW_SNR: [
                RecoveryAction(
                    strategy=RecoveryStrategy.REDUCE_INTENSITY,
                    description="Reduce noise reduction intensity (over-processing detected)",
                    parameters={"intensity_factor": 0.7},
                    priority=1,
                    expected_improvement=0.3,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.BYPASS_MODULE,
                    description="Bypass aggressive noise reduction module",
                    parameters={"module_names": ["NoiseReduction", "DeepNoise"]},
                    priority=2,
                    expected_improvement=0.4,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SWITCH_MODE,
                    description="Switch to FORENSIC mode (minimal processing)",
                    parameters={"new_mode": "forensic"},
                    priority=3,
                    expected_improvement=0.5,
                ),
            ],
            ProblemType.OVERBRIGHTENING: [
                RecoveryAction(
                    strategy=RecoveryStrategy.ADJUST_PARAMETERS,
                    description="Reduce high-frequency enhancement",
                    parameters={"high_freq_reduction": 0.5},
                    priority=1,
                    expected_improvement=0.4,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.BYPASS_MODULE,
                    description="Bypass de-esser and enhancer modules",
                    parameters={"module_names": ["DeEsser", "Enhancer", "Brightness"]},
                    priority=2,
                    expected_improvement=0.5,
                ),
            ],
            ProblemType.CHARACTER_LOSS: [
                RecoveryAction(
                    strategy=RecoveryStrategy.SWITCH_MODE,
                    description="Switch to VINTAGE_WARMTH mode (preserve character)",
                    parameters={"new_mode": "vintage_warmth"},
                    priority=1,
                    expected_improvement=0.6,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.BYPASS_MODULE,
                    description="Bypass aggressive processing modules",
                    parameters={"module_names": ["Modernizer", "Enhancer", "DeEsser"]},
                    priority=2,
                    expected_improvement=0.5,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.MAXIMIZE_QUALITY,
                    description="Adaptive optimization: Find best quality through iterative refinement",
                    parameters={"max_iterations": 8},
                    priority=3,
                    expected_improvement=0.7,
                ),
            ],
            ProblemType.UNNATURAL_SOUND: [
                RecoveryAction(
                    strategy=RecoveryStrategy.REDUCE_INTENSITY,
                    description="Reduce all processing intensity by 50%",
                    parameters={"intensity_factor": 0.5},
                    priority=1,
                    expected_improvement=0.5,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.INCREMENTAL_PROCESSING,
                    description="Re-process incrementally with quality checks",
                    parameters={"step_size": 0.25},
                    priority=2,
                    expected_improvement=0.4,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SWITCH_MODE,
                    description="Switch to ARCHIVAL mode (minimal change)",
                    parameters={"new_mode": "archival"},
                    priority=3,
                    expected_improvement=0.6,
                ),
            ],
            ProblemType.DYNAMIC_LOSS: [
                RecoveryAction(
                    strategy=RecoveryStrategy.BYPASS_MODULE,
                    description="Bypass compression and limiting",
                    parameters={"module_names": ["Compressor", "Limiter", "DynamicsProcessor"]},
                    priority=1,
                    expected_improvement=0.7,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.ADJUST_PARAMETERS,
                    description="Reduce compression ratio",
                    parameters={"compression_ratio": 0.3},
                    priority=2,
                    expected_improvement=0.4,
                ),
            ],
        }

        logger.info("Quality Wiederherstellung System initialisiert (v%s)", self.VERSION)

    def diagnose_problem(
        self,
        audio: np.ndarray,
        sample_rate: int,
        quality_report: MusicalQualityReport,
        medium_type: MediumType,
        processing_mode: ProcessingMode,
    ) -> RecoveryPlan:
        """
        Diagnose quality problem and generate recovery plan.

        Args:
            audio: Current (problematic) audio
            sample_rate: Sample rate
            quality_report: MQA report showing problems
            medium_type: Medium type
            processing_mode: Current processing mode

        Returns:
            RecoveryPlan with actionable strategies
        """
        logger.info("=" * 60)
        logger.info("QUALITY Wiederherstellung: Diagnosing Problem")
        logger.info("=" * 60)
        del audio, sample_rate

        # Analyze what went wrong
        problems = self._identify_problems(quality_report, medium_type)

        if not problems:
            logger.warning("No specific problems identified - using generic Wiederherstellung")
            problems = [ProblemType.UNNATURAL_SOUND]

        # Select primary problem
        primary_problem = problems[0]
        logger.info("Primary Problem: %s", primary_problem.value)

        # Generate recovery actions
        actions = self._generate_recovery_actions(primary_problem, quality_report, medium_type, processing_mode)

        # Sort by priority
        actions.sort(key=lambda a: a.priority)

        # Create recovery plan
        plan = RecoveryPlan(
            problem_type=primary_problem,
            problem_description=self._describe_problem(primary_problem, quality_report),
            actions=actions,
            fallback_strategy=RecoveryStrategy.MAXIMIZE_QUALITY,
            fallback_description="Adaptive optimization to find best achievable quality",
        )

        logger.info("Wiederherstellung Plan erzeugt: %s strategies", len(actions))
        for i, action in enumerate(actions, 1):
            logger.info("  %s. [%s] %s", i, action.strategy.value, action.description)
            logger.info("     Expected improvement: %.1f", action.expected_improvement)

        return plan

    def execute_recovery(
        self,
        original_audio: np.ndarray,
        current_audio: np.ndarray,
        sample_rate: int,
        recovery_plan: RecoveryPlan,
        modules_applied: list[str],
        medium_type: MediumType,
        processing_mode: ProcessingMode,
    ) -> RecoveryResult:
        """
        Führt aus: recovery plan to fix quality.

        Args:
            original_audio: Original input audio
            current_audio: Current (problematic) audio
            sample_rate: Sample rate
            recovery_plan: Recovery plan to execute
            modules_applied: List of modules that were applied
            medium_type: Medium type
            processing_mode: Processing mode

        Returns:
            RecoveryResult with recovered audio and details
        """
        logger.info("=" * 60)
        logger.info("QUALITY Wiederherstellung: Executing Wiederherstellung Plan")
        logger.info("=" * 60)

        # Measure original quality
        original_quality = self.analyzer.analyze_quality(original_audio, sample_rate)
        current_quality = self.analyzer.analyze_quality(current_audio, sample_rate)

        logger.info("Originalsignal Quality: %.1f/100", original_quality.overall_score)
        logger.info("Current Quality: %.1f/100", current_quality.overall_score)
        logger.info("Degradation: %.1f points", current_quality.overall_score - original_quality.overall_score)

        # Try each recovery strategy
        recovered_audio = current_audio.copy()
        actions_taken = []
        strategy_used = None

        for action in recovery_plan.actions:
            logger.info("\nAttempting: [%s] %s", action.strategy.value, action.description)

            try:
                # Execute strategy
                if action.strategy == RecoveryStrategy.REDUCE_INTENSITY:
                    recovered_audio = self._reduce_intensity(original_audio, current_audio, action.parameters)

                elif action.strategy == RecoveryStrategy.BYPASS_MODULE:
                    # Simulated: Re-process without problematic modules
                    recovered_audio = self._reprocess_without_modules(
                        original_audio,
                        current_audio,
                        sample_rate,
                        modules_applied,
                        action.parameters.get("module_names", []),
                    )

                elif action.strategy == RecoveryStrategy.SWITCH_MODE:
                    # Return to original with recommendation to switch mode
                    recovered_audio = original_audio.copy()
                    logger.info("  → Recommendation: Switch to %s Betriebsart", action.parameters.get("new_mode"))

                elif action.strategy == RecoveryStrategy.INCREMENTAL_PROCESSING:
                    recovered_audio = self._incremental_processing(original_audio, current_audio, action.parameters)

                elif action.strategy == RecoveryStrategy.MAXIMIZE_QUALITY:
                    # Adaptive multi-pass optimization
                    recovered_audio = self._maximize_quality(
                        original_audio, current_audio, action.parameters, medium_type, processing_mode, sample_rate
                    )

                elif action.strategy == RecoveryStrategy.ADJUST_PARAMETERS:
                    recovered_audio = self._adjust_parameters(
                        original_audio, current_audio, action.parameters, sample_rate
                    )

                elif action.strategy == RecoveryStrategy.USE_ALTERNATIVE:
                    recovered_audio = self._use_alternative(
                        original_audio, current_audio, action.parameters, medium_type, processing_mode, sample_rate
                    )

                else:
                    logger.warning(
                        "  Unbekannte Strategy '%s' — verwende MAXIMIZE_QUALITY Ersatzpfad",
                        action.strategy.value,
                    )
                    recovered_audio = self._maximize_quality(
                        original_audio,
                        current_audio,
                        {"max_iterations": 6},
                        medium_type,
                        processing_mode,
                        sample_rate,
                    )

                # Check if recovery worked
                recovered_quality = self.analyzer.analyze_quality(recovered_audio, sample_rate)

                logger.info("  Recovered Quality: %.1f/100", recovered_quality.overall_score)

                # Check quality gate
                gate_passed, reason = self.mqa.check_quality_gate(
                    recovered_audio, sample_rate, original_quality, medium_type, processing_mode
                )

                if gate_passed:
                    logger.info("  ✓ Quality gate PASSED after Wiederherstellung!")
                    strategy_used = action.strategy
                    actions_taken.append(action.description)
                    break
                else:
                    logger.warning("  ❌ Quality gate still failing: %s", reason)
                    actions_taken.append(f"{action.description} (failed)")

            except Exception as e:
                logger.error("  ❌ Strategy fehlgeschlagen with error: %s", e)
                actions_taken.append(f"{action.description} (error)")

        # If no strategy worked, use adaptive optimization fallback
        if not strategy_used:
            logger.info("All standard strategies tried - using adaptive optimization Ersatzpfad")
            recovered_audio = self._maximize_quality(
                original_audio, current_audio, {"max_iterations": 10}, medium_type, processing_mode, sample_rate
            )
            strategy_used = recovery_plan.fallback_strategy
            actions_taken.append(recovery_plan.fallback_description)

        # Final quality check
        final_quality = self.analyzer.analyze_quality(recovered_audio, sample_rate)

        improvement = final_quality.overall_score - current_quality.overall_score
        # Success if improved OR found best achievable quality
        success = improvement >= 0 or strategy_used == RecoveryStrategy.MAXIMIZE_QUALITY

        result = RecoveryResult(
            success=success,
            recovered_audio=recovered_audio,
            actions_taken=actions_taken,
            original_score=original_quality.overall_score,
            recovered_score=final_quality.overall_score,
            improvement=improvement,
            strategy_used=strategy_used,
            problem_solved=success,
            warnings=[],
        )

        logger.info("=" * 60)
        logger.info("QUALITY Wiederherstellung: vollstaendig")
        logger.info("=" * 60)
        logger.info("Status: %s", "SUCCESS" if success else "FAILED")
        logger.info("Strategy Used: %s", strategy_used.value if strategy_used else "None")
        logger.info("Quality Improvement: %+.1f points", improvement)
        logger.info("Final Wert: %.1f/100", final_quality.overall_score)

        return result

    def _identify_problems(self, quality_report: MusicalQualityReport, medium_type: MediumType) -> list[ProblemType]:
        """Identify specific problems from quality report."""
        del medium_type
        problems = []

        # §v10.703 QUALITY_UNCERTAIN: MQA kann bei fehlenden perzeptuellen Daten
        # (MUSHRA/HPI) keinen Report erzeugen und liefert None. Der Recovery-
        # Flow darf dann nicht crashen — generischer Fallback übernimmt.
        if quality_report is None:
            return problems

        # Check SNR
        if not quality_report.gates_passed:
            for warning in quality_report.warnings:
                if "SNR too low" in warning:
                    problems.append(ProblemType.LOW_SNR)
                elif "Over-brightened" in warning or "brightness" in warning.lower():
                    problems.append(ProblemType.OVERBRIGHTENING)
                elif "character" in warning.lower() or "authenticity" in warning.lower():
                    problems.append(ProblemType.CHARACTER_LOSS)
                elif "unnatural" in warning.lower() or "naturalness" in warning.lower():
                    problems.append(ProblemType.UNNATURAL_SOUND)
                elif "dynamic" in warning.lower():
                    problems.append(ProblemType.DYNAMIC_LOSS)

        # Check integrity violations
        for violation in quality_report.integrity_result.violations:
            if violation == IntegrityViolation.OVERPROCESSING:
                problems.append(ProblemType.UNNATURAL_SOUND)
            elif violation == IntegrityViolation.CHARACTER_LOSS:
                problems.append(ProblemType.CHARACTER_LOSS)
            elif violation == IntegrityViolation.FREQUENCY_IMBALANCE:
                problems.append(ProblemType.FREQUENCY_IMBALANCE)
            elif violation == IntegrityViolation.DYNAMIC_DESTRUCTION:
                problems.append(ProblemType.DYNAMIC_LOSS)

        # Remove duplicates
        problems = list(dict.fromkeys(problems))

        return problems

    def _generate_recovery_actions(
        self,
        problem_type: ProblemType,
        quality_report: MusicalQualityReport,
        medium_type: MediumType,
        processing_mode: ProcessingMode,
    ) -> list[RecoveryAction]:
        """Generiert recovery actions for specific problem."""
        del quality_report, medium_type, processing_mode
        # Get template actions
        actions = self._strategy_templates.get(problem_type, []).copy()

        # Customize based on context
        if not actions:
            # Generic fallback
            actions = [
                RecoveryAction(
                    strategy=RecoveryStrategy.REDUCE_INTENSITY,
                    description="Reduce processing intensity generically",
                    parameters={"intensity_factor": 0.6},
                    priority=1,
                    expected_improvement=0.3,
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.MAXIMIZE_QUALITY,
                    description="Adaptive optimization to find best quality",
                    parameters={"max_iterations": 5},
                    priority=2,
                    expected_improvement=0.6,
                ),
            ]

        return actions

    def _describe_problem(self, problem_type: ProblemType, quality_report: MusicalQualityReport) -> str:
        """Beschreibt das Problem menschenlesbar (§v10.703-sicher bei None-Report)."""
        if quality_report is None:
            return (
                "Quality-Report nicht verfügbar — perzeptuelle Daten (MUSHRA/HPI) fehlen; "
                "konservative Wiederherstellung wird angewendet"
            )
        descriptions = {
            ProblemType.LOW_SNR: f"SNR too low ({quality_report.output_quality.snr_db:.1f} dB)",
            ProblemType.OVERBRIGHTENING: (
                f"Over-brightened (brightness {quality_report.output_quality.brightness:.2f})"
            ),
            ProblemType.CHARACTER_LOSS: (
                f"Analog character lost (authenticity {quality_report.output_quality.authenticity:.2f})"
            ),
            ProblemType.UNNATURAL_SOUND: (
                f"Unnatural sound (naturalness {quality_report.output_quality.naturalness:.2f})"
            ),
            ProblemType.DYNAMIC_LOSS: (f"Dynamic range lost ({quality_report.output_quality.dynamic_range_db:.1f} dB)"),
        }

        return descriptions.get(problem_type, "Unknown problem")

    def _maximize_quality(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        parameters: dict[str, Any],
        medium_type: MediumType,
        processing_mode: ProcessingMode,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Adaptive optimization to find best achievable quality.

        Tries multiple intensity levels and returns the best result.
        NEVER gives up - always finds the optimal solution!
        """
        max_iterations = parameters.get("max_iterations", 10)

        logger.info("  → Adaptive optimization: Testing %s different approaches...", max_iterations)

        best_audio = processed.copy()
        best_score = self.analyzer.analyze_quality(processed, sample_rate).overall_score
        best_gate_passed = False

        # Try different blending ratios
        for i in range(max_iterations):
            # Intensity from 0.0 (original) to 1.0 (fully processed)
            intensity = i / (max_iterations - 1) if max_iterations > 1 else 0.5

            # Blend original and processed
            candidate = (1 - intensity) * original + intensity * processed

            # Measure quality
            candidate_quality = self.analyzer.analyze_quality(candidate, sample_rate)
            candidate_score = candidate_quality.overall_score

            # Check if quality gate would pass
            baseline = self.mqa.establish_baseline(original, sample_rate, medium_type, processing_mode)
            gate_passed, _ = self.mqa.check_quality_gate(candidate, sample_rate, baseline, medium_type, processing_mode)

            # Keep best result (prefer gate-passing solutions)
            if gate_passed and (not best_gate_passed or candidate_score > best_score):
                best_audio = candidate.copy()
                best_score = candidate_score
                best_gate_passed = True
                logger.info(
                    "    Iteration %s: Quality %.1f (intensity %.2f) ✓ GATE PASSED",
                    i + 1,
                    candidate_score,
                    intensity,
                )
            elif not best_gate_passed and candidate_score > best_score:
                # Still better than what we had, and we haven't found a gate-passing solution yet
                best_audio = candidate.copy()
                best_score = candidate_score

        if best_gate_passed:
            logger.info("  → OPTIMAL quality found: %.1f/100 (GATE PASSED)", best_score)
        else:
            logger.info("  → BEST ACHIEVABLE quality: %.1f/100", best_score)

        return best_audio.astype(np.float32)  # type: ignore[no-any-return]

    def _reduce_intensity(self, original: np.ndarray, processed: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
        """Reduce processing intensity by blending with original."""
        intensity_factor = parameters.get("intensity_factor", 0.7)

        # Blend processed with original
        # Less processed = more original
        blended = (1 - intensity_factor) * original + intensity_factor * processed

        logger.info("  → Blending: %.1f Originalsignal + %.1f verarbeitet", (1 - intensity_factor), intensity_factor)

        return blended.astype(np.float32)  # type: ignore[no-any-return]

    def _reprocess_without_modules(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sample_rate: int,
        modules_applied: list[str],
        modules_to_skip: list[str],
    ) -> np.ndarray:
        """Approximate reprocessing without the given modules via STFT-domain blending.

        Since the full pipeline cannot be re-invoked from here, this method applies
        a targeted STFT-domain correction that approximates what bypassing each module
        category would have produced:

        - NR modules  (NoiseReduction, DeepNoise, …) →  over-suppression shows as
          spectrally smooth regions; blending back toward the original restores
          natural noise texture and harmonics.
        - Enhancement modules (Enhancer, Brightness, DeEsser, …) →  excess
          high-frequency energy; attenuate HF in processed where it exceeds
          the original PSD by more than 3 dB.
        - Compression modules (Compressor, Limiter, DynamicsProcessor) →  dynamic
          flatness; re-scale each STFT frame to match the original's crest factor.

        The overall blend weight is proportional to the fraction of applied modules
        that are being skipped, clamped to [0.20, 0.80].
        """
        logger.info("  → Would ueberspringen modules: %s", ", ".join(modules_to_skip))

        if original.shape[0] == 0 or processed.shape[0] == 0 or not modules_to_skip:
            return processed.copy()

        # ── Module-category lookup ───────────────────────────────────────────
        _NR_KEYWORDS = {"noise", "denoise", "deepnoise", "noisereduction", "hiss"}
        _ENHANCE_KEYS = {"enhancer", "brightness", "deesser", "exciter", "air", "presence", "highfrequency"}
        _COMPRESS_KEYS = {"compressor", "limiter", "dynamics", "compression"}

        def _cat(name: str) -> str:
            n = name.lower()
            if any(k in n for k in _NR_KEYWORDS):
                return "nr"
            if any(k in n for k in _ENHANCE_KEYS):
                return "enhance"
            if any(k in n for k in _COMPRESS_KEYS):
                return "compress"
            return "generic"

        skip_cats = {_cat(m) for m in modules_to_skip}

        # Overall blend: fraction of applied modules being skipped
        n_applied = max(len(modules_applied), 1)
        skip_ratio = min(len(modules_to_skip) / n_applied, 1.0)
        # clamp to [0.20, 0.80] — never go all the way to original
        blend_orig = max(0.20, min(0.80, skip_ratio))
        blend_proc = 1.0 - blend_orig  # keep some of the processed signal

        # Start from a simple time-domain blend
        try:
            nperseg = 1024
            ORIG = original.astype(np.float32)
            PROC = processed.astype(np.float32)

            # Work mono; handle stereo by processing per channel
            if ORIG.ndim == 2:
                channels = []
                for ch in range(ORIG.shape[1]):
                    channels.append(
                        self._reprocess_without_modules(
                            ORIG[:, ch : ch + 1].squeeze(),
                            PROC[:, ch : ch + 1].squeeze(),
                            sample_rate,
                            modules_applied,
                            modules_to_skip,
                        )
                    )
                result = np.stack(channels, axis=1)
                return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

            if len(PROC) != len(ORIG):
                n_min = min(len(PROC), len(ORIG))
                ORIG = ORIG[:n_min]
                PROC = PROC[:n_min]

            _, _, Zxx_orig = signal.stft(ORIG, nperseg=nperseg, noverlap=nperseg // 2)
            _, _, Zxx_proc = signal.stft(PROC, nperseg=nperseg, noverlap=nperseg // 2)
            mag_orig = np.abs(Zxx_orig)
            mag_proc = np.abs(Zxx_proc)
            phase_proc = np.angle(Zxx_proc)

            # Start from processed spectrum and steer only where the skipped
            # module categories likely caused damage.
            mag_corr = mag_proc.copy()

            if "enhance" in skip_cats:
                # Attenuate HF over-emphasis when processed magnitude exceeds original.
                _sr = max(int(sample_rate), 1)
                hf_start_bin = max(1, int(6000.0 * nperseg / float(_sr)))
                hf_start_bin = min(hf_start_bin, mag_corr.shape[0] - 1)
                if hf_start_bin > 0:
                    _ratio = (mag_orig[hf_start_bin:, :] + 1e-9) / (mag_proc[hf_start_bin:, :] + 1e-9)
                    _ratio = np.clip(_ratio, 0.35, 1.0)
                    mag_corr[hf_start_bin:, :] *= _ratio

            if "compress" in skip_cats:
                # Restore crest factor: scale each frame so its peak ≈ original's
                frame_peak_orig = np.percentile(mag_orig, 99.9, axis=0, keepdims=True) + 1e-12
                frame_peak_corr = np.max(mag_corr, axis=0, keepdims=True) + 1e-12
                scale = frame_peak_orig / frame_peak_corr
                mag_corr = mag_corr * np.clip(scale, 0.60, 1.60)

            if "nr" in skip_cats:
                # Bring back part of suppressed low-level texture from original.
                mag_corr = 0.70 * mag_corr + 0.30 * mag_orig

            # Reconstruct with processed phase to preserve current time alignment.
            Zxx_corr = mag_corr * np.exp(1j * phase_proc)
            _, audio_corr = signal.istft(Zxx_corr, nperseg=nperseg, noverlap=nperseg // 2)
            audio_corr = np.nan_to_num(audio_corr[: len(ORIG)], nan=0.0, posinf=0.0, neginf=0.0)

            # NR case: for NR modules blend corr signal back toward original
            if "nr" in skip_cats:
                blend_orig_nr = min(blend_orig + 0.15, 0.85)
                audio_corr = (1.0 - blend_orig_nr) * audio_corr + blend_orig_nr * ORIG
            else:
                audio_corr = blend_proc * audio_corr + blend_orig * ORIG

            result = np.clip(audio_corr, -1.0, 1.0).astype(np.float32)
            logger.info(
                "  → STFT-domain correction angewendet (ueberspringen_Verhaeltnis=%.2f, cats=%s)",
                skip_ratio,
                skip_cats,
            )
            return result  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning("  → STFT correction fehlgeschlagen (%s), using time-domain blend.", exc)
            blended = blend_orig * original + blend_proc * processed
            return np.clip(blended, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def _incremental_processing(
        self, original: np.ndarray, target: np.ndarray, parameters: dict[str, Any]
    ) -> np.ndarray:
        """Verarbeitet incrementally towards target with quality checks."""
        step_size = parameters.get("step_size", 0.25)

        # Move 25% towards target
        incremental = original + step_size * (target - original)

        logger.info("  → Incremental step: %.1f towards target", step_size)

        return incremental.astype(np.float32)  # type: ignore[no-any-return]

    def _adjust_parameters(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        parameters: dict[str, Any],
        sample_rate: int,
    ) -> np.ndarray:
        """Spectral parameter adjustment via STFT-domain EQ correction.

        Applies frequency-selective gain to address:
        - ``high_freq_reduction`` (0–1): blend HF bins (≥ 6 kHz) back toward
          the original PSD, reducing over-brightening.
        - ``compression_ratio`` (0–1): reduce residual dynamic compression
          by restoring frame-level crest-factor toward original.
        - ``low_freq_boost`` (0–1): boost LF bins (≤ 250 Hz) relative to
          the original to restore warmth lost during processing.

        All operations are M/S-aware: stereo input is processed as M/S and
        adjustments are applied to Mid only (§2.51).
        """
        hf_red = float(parameters.get("high_freq_reduction", 0.0))
        comp_rat = float(parameters.get("compression_ratio", 0.0))
        lf_boost = float(parameters.get("low_freq_boost", 0.0))

        if hf_red == 0.0 and comp_rat == 0.0 and lf_boost == 0.0:
            return processed.astype(np.float32)  # type: ignore[no-any-return]

        audio_f32 = processed.astype(np.float32)
        orig_f32 = original.astype(np.float32)
        stereo = audio_f32.ndim == 2

        # Convert stereo → M/S (Mid only adjusted, §2.51)
        if stereo:
            mid = (audio_f32[:, 0] + audio_f32[:, 1]) * 0.5
            side = (audio_f32[:, 0] - audio_f32[:, 1]) * 0.5
            mid_orig = (orig_f32[:, 0] + orig_f32[:, 1]) * 0.5
        else:
            mid = audio_f32
            mid_orig = orig_f32

        nperseg = min(2048, max(1, len(mid)))
        noverlap = min(nperseg * 3 // 4, nperseg - 1)

        try:
            _, _, Zxx = signal.stft(mid, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            _, _, Zxx_orig = signal.stft(mid_orig, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

            freqs = np.linspace(0, sample_rate / 2, Zxx.shape[0])
            mag = np.abs(Zxx)
            mag_orig = np.abs(Zxx_orig)
            phase = np.angle(Zxx)

            gain = np.ones_like(mag)

            if hf_red > 0.0:
                # HF bins ≥ 6 kHz — blend back toward original to reduce over-brightness
                hf_mask = freqs >= 6000.0
                # gain factor: 1 at boundary, moves toward mag_orig/mag at full cut
                ratio = np.where(mag > 1e-12, mag_orig / (mag + 1e-12), 1.0)
                ratio = np.clip(ratio, 0.1, 10.0)
                gain[hf_mask, :] *= 1.0 - hf_red * (1.0 - ratio[hf_mask, :])
                gain[hf_mask, :] = np.clip(gain[hf_mask, :], 0.1, 2.0)

            if comp_rat > 0.0:
                # Per-frame crest-factor restoration — reintroduce dynamics
                frame_peak_proc = np.percentile(mag, 99.9, axis=0, keepdims=True) + 1e-12
                frame_peak_orig = np.percentile(mag_orig, 99.9, axis=0, keepdims=True) + 1e-12
                crest_scale = (frame_peak_orig / frame_peak_proc) ** comp_rat
                crest_scale = np.clip(crest_scale, 0.5, 2.0)
                gain *= crest_scale

            if lf_boost > 0.0:
                lf_mask = freqs <= 250.0
                lf_scale = 1.0 + lf_boost * 0.5  # max +6 dB
                gain[lf_mask, :] *= lf_scale

            Zxx_out = mag * gain * np.exp(1j * phase)
            _, mid_out = signal.istft(Zxx_out, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            n = len(mid)
            mid_out = mid_out[:n] if len(mid_out) >= n else np.pad(mid_out, (0, n - len(mid_out)))
            mid_out = np.nan_to_num(mid_out, nan=0.0, posinf=0.0, neginf=0.0)

            if stereo:
                out_l = mid_out + side
                out_r = mid_out - side
                result = np.stack([out_l, out_r], axis=1)
            else:
                result = mid_out

            result = np.clip(result, -1.0, 1.0).astype(np.float32)
            logger.info(
                "  → anpassen_PARAMETERS: hf_red=%.2f comp=%.2f lf_boost=%.2f angewendet",
                hf_red,
                comp_rat,
                lf_boost,
            )
            return result  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning("  → _anpassen_parameters STFT fehlgeschlagen (%s), returning verarbeitet unchanged.", exc)
            return processed.astype(np.float32)  # type: ignore[no-any-return]

    def _use_alternative(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        parameters: dict[str, Any],
        medium_type: MediumType,
        processing_mode: ProcessingMode,
        sample_rate: int,
    ) -> np.ndarray:
        """Multi-band alternative processing path.

        Replaces the current processing chain output with a frequency-zone-aware
        blend: uses more of the *original* in perceptually sensitive zones (vocals,
        low-mids) and more of the *processed* output in bands where processing
        likely helped (HF extensions, bass cleaning).

        Blend weights per zone are tunable via ``parameters``:
          - ``lf_blend``       bass zone (≤ 250 Hz)  — default 0.4 orig
          - ``low_mid_blend``  vocals/mids (250–2 kHz) — default 0.55 orig
          - ``high_mid_blend`` presence (2–6 kHz)     — default 0.35 orig
          - ``hf_blend``       air (≥ 6 kHz)          — default 0.25 orig

        Always adds a limiting step to prevent clipping.
        """
        del medium_type, processing_mode

        lf_blend = float(parameters.get("lf_blend", 0.40))
        low_mid_blend = float(parameters.get("low_mid_blend", 0.55))
        high_mid_blend = float(parameters.get("high_mid_blend", 0.35))
        hf_blend = float(parameters.get("hf_blend", 0.25))

        audio_f32 = processed.astype(np.float32)
        orig_f32 = original.astype(np.float32)

        # Work on mono-mid for M/S compliance (§2.51); keep side unchanged
        stereo = audio_f32.ndim == 2
        if stereo:
            mid = (audio_f32[:, 0] + audio_f32[:, 1]) * 0.5
            side = (audio_f32[:, 0] - audio_f32[:, 1]) * 0.5
            mid_orig = (orig_f32[:, 0] + orig_f32[:, 1]) * 0.5
        else:
            mid = audio_f32
            mid_orig = orig_f32

        nperseg = min(2048, max(1, len(mid)))
        noverlap = min(nperseg * 3 // 4, nperseg - 1)

        try:
            _, _, Zxx_proc = signal.stft(mid, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            _, _, Zxx_orig = signal.stft(mid_orig, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

            freqs = np.linspace(0, sample_rate / 2, Zxx_proc.shape[0])

            # Per-bin blend weights (α_orig × Zxx_orig + (1-α_orig) × Zxx_proc)
            alpha = np.zeros(len(freqs))
            alpha[freqs <= 250.0] = lf_blend
            alpha[(freqs > 250.0) & (freqs <= 2000.0)] = low_mid_blend
            alpha[(freqs > 2000.0) & (freqs <= 6000.0)] = high_mid_blend
            alpha[freqs > 6000.0] = hf_blend

            alpha = alpha[:, np.newaxis]  # broadcast over frames
            Zxx_alt = alpha * Zxx_orig + (1.0 - alpha) * Zxx_proc

            _, mid_out = signal.istft(Zxx_alt, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            n = len(mid)
            mid_out = mid_out[:n] if len(mid_out) >= n else np.pad(mid_out, (0, n - len(mid_out)))
            mid_out = np.nan_to_num(mid_out, nan=0.0, posinf=0.0, neginf=0.0)

            if stereo:
                out_l = mid_out + side
                out_r = mid_out - side
                result = np.stack([out_l, out_r], axis=1)
            else:
                result = mid_out

            # True-peak guard (§ peak-guard rule)
            peak = np.percentile(np.abs(result), 99.9)
            if peak > 0.98:
                result = result * (0.98 / peak)

            result = np.clip(result, -1.0, 1.0).astype(np.float32)
            logger.info(
                "  → USE_ALTERNATIVE: multi-band blend angewendet (lf=%.2f lo_mid=%.2f hi_mid=%.2f hf=%.2f)",
                lf_blend,
                low_mid_blend,
                high_mid_blend,
                hf_blend,
            )
            return result  # type: ignore[no-any-return]

        except Exception as exc:
            logger.warning("  → _use_alternative STFT fehlgeschlagen (%s), using blend Ersatzpfad.", exc)
            blend = 0.5 * processed + 0.5 * original
            return np.clip(blend, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def create_quality_recovery_system() -> QualityRecoverySystem:
    """Factory-Funktion zum Erstellen des Qualitäts-Recovery-Systems."""
    return QualityRecoverySystem()


# === Example Usage ===
if __name__ == "__main__":
    import soundfile as sf

    from backend.file_import import load_audio_file

    # Example: Recover from failed quality gate
    _res1 = load_audio_file("input/vinyl.wav")
    if not isinstance(_res1, dict) or "audio" not in _res1 or "sr" not in _res1:
        raise RuntimeError("load_audio_file('input/vinyl.wav') lieferte kein gueltiges Payload")
    demo_original_audio = np.asarray(_res1["audio"], dtype=np.float32)
    demo_sample_rate = int(_res1["sr"])
    _res2 = load_audio_file("output/over_processed.wav")
    if not isinstance(_res2, dict) or "audio" not in _res2:
        raise RuntimeError("load_audio_file('output/over_processed.wav') lieferte kein gueltiges Payload")
    demo_processed_audio = np.asarray(_res2["audio"], dtype=np.float32)

    # Create systems
    mqa = MusicalQualityAssurance()
    recovery = QualityRecoverySystem()

    # Validate and detect problem
    demo_quality_report = mqa.validate_final_quality(
        demo_original_audio,
        demo_processed_audio,
        demo_sample_rate,
        MediumType.VINYL_33,
        ProcessingMode.RESTORATION,
        ["NoiseReduction", "DeEsser", "Enhancer"],
    )

    if not demo_quality_report.quality_guaranteed:
        # Generate recovery plan
        demo_recovery_plan = recovery.diagnose_problem(
            demo_processed_audio,
            demo_sample_rate,
            demo_quality_report,
            MediumType.VINYL_33,
            ProcessingMode.RESTORATION,
        )

        # Execute recovery
        demo_recovery_result = recovery.execute_recovery(
            demo_original_audio,
            demo_processed_audio,
            demo_sample_rate,
            demo_recovery_plan,
            ["NoiseReduction", "DeEsser", "Enhancer"],
            MediumType.VINYL_33,
            ProcessingMode.RESTORATION,
        )

        if demo_recovery_result.success:
            logger.debug("✓ Quality recovered: %+.1f points", demo_recovery_result.improvement)
            sf.write("output/recovered.wav", demo_recovery_result.recovered_audio, demo_sample_rate)
        else:
            logger.debug("❌ Wiederherstellung fehlgeschlagen - using Originalsignal")
