"""
AURIK v8 Automatic Reprocessing Engine
=======================================

Automatic reprocessing with parameter tuning on Quality Gate failures.

Component 4.6: Auto-Reprocessing with Fallback Strategies
Impact: +0.5 Punkt - Automatic recovery from quality failures

HIPS Compliance:
- Requirement 1: Explizite Verantwortung (System decides retry strategy)
- Requirement 2: Kontextbewusstsein (Adapts strategy to signal characteristics)
- Requirement 4: Reversibilität (Can fall back to original or intermediate results)
- Requirement 6: Auditierbarkeit (Full logging of attempts and decisions)

Problem:
Ohne Automatic Reprocessing führen Quality Gate Failures zu:
- User Intervention (violates "NO user decisions" policy)
- Suboptimal results (rollback to original without trying alternatives)
- Manual parameter tuning (expensive and inconsistent)

Solution:
3-Tier Fallback Strategy:
1. Parameter Reduction (50% → 75% → 90% intensity)
2. Alternative Processing Chains (e.g., DSP-only vs ML-heavy)
3. Hybrid Approaches (blend original + processed)

Quelle: Excellence Roadmap Strategy #2 - Perceptual Quality Gates
Autor: AI Team
Datum: 8. Februar 2026
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ReprocessingStrategy(Enum):
    """Reprocessing strategy types."""

    PARAMETER_REDUCTION = "parameter_reduction"  # Reduce processing intensity
    ALTERNATIVE_CHAIN = "alternative_chain"  # Try different processing chain
    HYBRID_BLEND = "hybrid_blend"  # Blend original + processed
    PARTIAL_ROLLBACK = "partial_rollback"  # Rollback selected modules only
    FORENSIC_GUIDED = "forensic_guided"  # Use forensics to guide processing


@dataclass
class ReprocessingAttempt:
    """Record of a single reprocessing attempt."""

    attempt_number: int
    strategy: ReprocessingStrategy
    parameters: dict[str, Any]
    quality_scores: dict[str, float]
    improvements: dict[str, float]
    violations: dict[str, dict[str, float]]
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str | None = None


@dataclass
class ReprocessingResult:
    """Result of automatic reprocessing."""

    success: bool
    best_audio: np.ndarray
    best_quality_scores: dict[str, float]
    attempts: list[ReprocessingAttempt]
    total_attempts: int
    strategy_used: ReprocessingStrategy
    final_decision: str  # "reprocessed", "fallback_blend", "rollback_original"
    improvements_achieved: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AutoReprocessingEngine:
    """
    Automatic reprocessing engine with intelligent fallback strategies.

    When Quality Gates fail, this engine:
    1. Analyzes failure mode (which metrics failed, by how much)
    2. Selects optimal retry strategy based on failure characteristics
    3. Executes retry with adjusted parameters
    4. Validates improvement with Quality Gates
    5. Repeats with fallback strategies if needed
    6. Returns best result after max_attempts

    Strategies:
    - Parameter Reduction: Reduce processing intensity (common for over-processing)
    - Alternative Chain: Try different DSP/ML modules (common for artifacts)
    - Hybrid Blend: Mix original + processed (preserves naturalness)
    - Partial Rollback: Rollback only problematic modules (surgical fix)
    - Forensic Guided: Use signal forensics to guide processing (context-aware)

    Attributes:
        max_attempts: Maximum reprocessing attempts (default: 5)
        min_improvement: Minimum improvement threshold to accept result (default: 0.02)
        enable_hybrid_fallback: Allow hybrid blending as last resort (default: True)
        enable_forensic_guidance: Use signal forensics for strategy selection (default: True)
    """

    def __init__(
        self,
        max_attempts: int = 5,
        min_improvement: float = 0.02,
        enable_hybrid_fallback: bool = True,
        enable_forensic_guidance: bool = True,
    ) -> None:
        """
        Initialisiert Auto-Reprocessing Engine.

        Args:
            max_attempts: Maximum retry attempts
            min_improvement: Minimum improvement to accept result (0.02 = 2%)
            enable_hybrid_fallback: If True, allow original + processed blending
            enable_forensic_guidance: If True, use forensics for strategy selection
        """
        self.max_attempts = max_attempts
        self.min_improvement = min_improvement
        self.enable_hybrid_fallback = enable_hybrid_fallback
        self.enable_forensic_guidance = enable_forensic_guidance

        logger.info(
            f"AutoReprocessingEngine initialisiert: max_attempts={max_attempts}, min_improvement={min_improvement}"
        )

    def reprocess_on_failure(
        self,
        original: np.ndarray,
        failed_processed: np.ndarray,
        sr: int,
        processing_function: Callable[[np.ndarray, int, dict[str, Any]], np.ndarray],
        quality_validator: Callable[
            [np.ndarray, np.ndarray, int], tuple[bool, dict[str, float], dict[str, dict[str, float]]]
        ],
        baseline_scores: dict[str, float],
        initial_violations: dict[str, dict[str, float]],
        context: dict[str, Any] | None = None,
    ) -> ReprocessingResult:
        """
        Automatic reprocessing with intelligent fallback strategies.

        Args:
            original: Original audio
            failed_processed: Processed audio that failed Quality Gates
            sr: Sample rate
            processing_function: Function to reprocess audio
                Signature: (audio, sr, params) -> processed_audio
            quality_validator: Function to validate quality
                Signature: (original, processed, sr) -> (passed, scores, violations)
            baseline_scores: Baseline quality scores from original
            initial_violations: Quality violations from initial processing
            context: Optional context (medium_type, genre, forensics, etc.)

        Returns:
            ReprocessingResult with best audio and attempt history

        Example:
            >>> engine = AutoReprocessingEngine(max_attempts=5)
            >>> result = engine.reprocess_on_failure(
            ...     original, failed_processed, sr,
            ...     processing_function=my_processor,
            ...     quality_validator=quality_gate.validate,
            ...     baseline_scores=baseline,
            ...     initial_violations=violations
            ... )
            >>> if result.success:
            ...     audio = result.best_audio
            >>> else:
            ...     audio = original  # Ultimate fallback
        """
        context = context or {}
        attempts: list[ReprocessingAttempt] = []

        logger.info("Reprocessing gestartet: %s violations erkannt", len(initial_violations))

        # Track best result so far
        best_audio = failed_processed
        best_scores = {}
        best_strategy = ReprocessingStrategy.PARAMETER_REDUCTION

        # Analyze failure mode to select strategies
        strategies = self._select_strategies(initial_violations, baseline_scores, context)

        logger.info("Selected strategies: %s", [s.value for s in strategies])

        # Try strategies sequentially
        for attempt_num in range(1, self.max_attempts + 1):
            if attempt_num > len(strategies):
                # Out of strategies, use hybrid fallback if enabled
                if self.enable_hybrid_fallback and attempt_num == self.max_attempts:
                    strategy = ReprocessingStrategy.HYBRID_BLEND
                else:
                    logger.warning("Out of strategies at Versuch %s", attempt_num)
                    break
            else:
                strategy = strategies[attempt_num - 1]

            logger.info("Versuch %s/%s: Strategy=%s", attempt_num, self.max_attempts, strategy.value)

            # Execute strategy
            try:
                reprocessed, params = self._execute_strategy(
                    strategy, original, failed_processed, sr, processing_function, attempt_num, context
                )
            except Exception as e:
                logger.error("Strategy %s fehlgeschlagen: %s", strategy.value, e)
                attempts.append(
                    ReprocessingAttempt(
                        attempt_number=attempt_num,
                        strategy=strategy,
                        parameters={},
                        quality_scores={},
                        improvements={},
                        violations={},
                        success=False,
                        notes=f"Exception: {e!s}",
                    )
                )
                continue

            # Validate quality
            passed, scores, violations = quality_validator(original, reprocessed, sr)

            # Calculate improvements
            improvements = {goal: scores[goal] - baseline_scores.get(goal, 0.0) for goal in scores}

            # Record attempt
            attempt = ReprocessingAttempt(
                attempt_number=attempt_num,
                strategy=strategy,
                parameters=params,
                quality_scores=scores,
                improvements=improvements,
                violations=violations,
                success=passed,
                notes=f"Violations: {len(violations)}" if not passed else "All gates passed",
            )
            attempts.append(attempt)

            logger.info(
                f"Versuch {attempt_num}: "
                f"passed={passed}, violations={len(violations)}, "
                f"avg_improvement={np.mean(list(improvements.values())):.3f}"
            )

            # Check if this is best so far
            avg_improvement = np.mean(list(improvements.values()))
            if avg_improvement > self.min_improvement:
                best_audio = reprocessed
                best_scores = scores
                best_strategy = strategy

            # Success? Stop retrying
            if passed:
                logger.info("✓ Reprocessing succeeded at Versuch %s", attempt_num)
                return ReprocessingResult(
                    success=True,
                    best_audio=reprocessed,
                    best_quality_scores=scores,
                    attempts=attempts,
                    total_attempts=attempt_num,
                    strategy_used=strategy,
                    final_decision="reprocessed",
                    improvements_achieved=improvements,
                )

        # All attempts exhausted
        if len(best_scores) > 0:
            # We have some improvements, return best result
            avg_improvement = np.mean([best_scores[g] - baseline_scores.get(g, 0.0) for g in best_scores])

            logger.warning(
                "Reprocessing did not pass all gates but achieved "
                f"improvements: avg={avg_improvement:.3f}, "
                f"strategy={best_strategy.value}"
            )

            return ReprocessingResult(
                success=False,
                best_audio=best_audio,
                best_quality_scores=best_scores,
                attempts=attempts,
                total_attempts=len(attempts),
                strategy_used=best_strategy,
                final_decision="partial_improvement",
                improvements_achieved={g: best_scores[g] - baseline_scores.get(g, 0.0) for g in best_scores},
            )
        else:
            # Complete failure, rollback to original
            logger.error("All reprocessing attempts fehlgeschlagen, rolling back to Originalsignal")

            return ReprocessingResult(
                success=False,
                best_audio=original,
                best_quality_scores=baseline_scores,
                attempts=attempts,
                total_attempts=len(attempts),
                strategy_used=ReprocessingStrategy.PARAMETER_REDUCTION,  # Default
                final_decision="rollback_original",
                improvements_achieved={},
            )

    def _select_strategies(
        self, violations: dict[str, dict[str, float]], baseline: dict[str, float], context: dict[str, Any]
    ) -> list[ReprocessingStrategy]:
        """
        Wählt aus: optimal retry strategies based on failure characteristics.

        Strategy Selection Heuristics:
        - Many violations + low baseline → FORENSIC_GUIDED (signal too degraded)
        - Brillanz violations → PARAMETER_REDUCTION (over-processing HF)
        - Bass-kraft violations → ALTERNATIVE_CHAIN (bass processing artifacts)
        - Multiple degradations → PARTIAL_ROLLBACK (some modules failed)
        - All else → PARAMETER_REDUCTION → HYBRID_BLEND
        """
        strategies = []

        # Count violations by severity
        critical_violations = sum(1 for v in violations.values() if v["achieved"] < 0.70)

        # Analyze which goals failed
        failed_goals = list(violations.keys())

        # Strategy 1: Forensic-Guided (if signal is extremely degraded)
        if self.enable_forensic_guidance and critical_violations >= 3 and sum(baseline.values()) / len(baseline) < 0.50:
            strategies.append(ReprocessingStrategy.FORENSIC_GUIDED)

        # Strategy 2: Parameter Reduction (most common - over-processing)
        strategies.append(ReprocessingStrategy.PARAMETER_REDUCTION)

        # Strategy 3: Alternative Chain (if specific goals failed)
        if "brillanz" in failed_goals or "transparenz" in failed_goals:
            strategies.append(ReprocessingStrategy.ALTERNATIVE_CHAIN)

        # Strategy 4: Partial Rollback (if many goals degraded)
        if len(failed_goals) >= 4:
            strategies.append(ReprocessingStrategy.PARTIAL_ROLLBACK)

        # Strategy 5: Hybrid Blend (always as last resort if enabled)
        if self.enable_hybrid_fallback:
            strategies.append(ReprocessingStrategy.HYBRID_BLEND)

        return strategies[: self.max_attempts]

    def _execute_strategy(
        self,
        strategy: ReprocessingStrategy,
        original: np.ndarray,
        processed: np.ndarray,
        sr: int,
        processing_function: Callable,
        attempt_num: int,
        context: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Führt aus: specific reprocessing strategy.

        Returns:
            (reprocessed_audio, parameters_used)
        """
        if strategy == ReprocessingStrategy.PARAMETER_REDUCTION:
            return self._strategy_parameter_reduction(original, sr, processing_function, attempt_num, context)

        elif strategy == ReprocessingStrategy.ALTERNATIVE_CHAIN:
            return self._strategy_alternative_chain(original, sr, processing_function, context)

        elif strategy == ReprocessingStrategy.HYBRID_BLEND:
            return self._strategy_hybrid_blend(original, processed, sr, attempt_num, context)

        elif strategy == ReprocessingStrategy.PARTIAL_ROLLBACK:
            return self._strategy_partial_rollback(original, processed, sr, context)

        elif strategy == ReprocessingStrategy.FORENSIC_GUIDED:
            return self._strategy_forensic_guided(original, sr, processing_function, context)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _strategy_parameter_reduction(
        self, original: np.ndarray, sr: int, processing_function: Callable, attempt_num: int, context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reduce processing intensity progressively."""
        # Progressive intensity reduction: 50% → 75% → 90%
        intensity_levels = [0.50, 0.75, 0.90]
        intensity = intensity_levels[min(attempt_num - 1, len(intensity_levels) - 1)]

        params = {**context, "intensity": intensity}
        reprocessed = processing_function(original, sr, params)

        logger.info("Parameter reduction: intensity=%.2f", intensity)

        return reprocessed, params

    def _strategy_alternative_chain(
        self, original: np.ndarray, sr: int, processing_function: Callable, context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Try alternative processing chain (e.g., DSP-only instead of ML-heavy)."""
        params = {**context, "prefer_dsp": True, "ml_weight": 0.3}
        reprocessed = processing_function(original, sr, params)

        logger.info("Alternative chain: DSP-preferred Betriebsart")

        return reprocessed, params

    def _strategy_hybrid_blend(
        self, original: np.ndarray, processed: np.ndarray, sr: int, attempt_num: int, context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Mischt original + processed to preserve naturalness."""
        # Progressive blend: 70/30 → 60/40 → 50/50
        blend_ratios = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
        proc_weight, orig_weight = blend_ratios[min(attempt_num - 1, len(blend_ratios) - 1)]

        # Ensure same length
        min_len = min(len(original), len(processed))
        blended = proc_weight * processed[:min_len] + orig_weight * original[:min_len]

        params = {"processed_weight": proc_weight, "original_weight": orig_weight}

        logger.info("Hybrid blend: %.0f verarbeitet + %.0f Originalsignal", proc_weight, orig_weight)

        return blended, params

    def _strategy_partial_rollback(
        self, original: np.ndarray, processed: np.ndarray, sr: int, context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Rollback problematic processing modules only (surgical fix)."""
        # Simplified: 80% processed + 20% original (frequency-dependent blend)
        # In full implementation, this would selectively rollback specific modules

        min_len = min(len(original), len(processed))
        partial = 0.8 * processed[:min_len] + 0.2 * original[:min_len]

        params = {"rollback_mode": "partial", "blend_ratio": 0.8}

        logger.info("Partial rollback: 80%% verarbeitet + 20%% Originalsignal")

        return partial, params

    def _strategy_forensic_guided(
        self, original: np.ndarray, sr: int, processing_function: Callable, context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Use signal forensics to guide processing parameters."""
        # Extract forensics if available
        forensics = context.get("forensics", {})

        # Adjust parameters based on forensics
        params = {**context}

        # If signal is vinyl, reduce declick intensity
        if forensics.get("medium_type") == "vinyl":
            params["declick_intensity"] = 0.3

        # If signal is tape, prioritize wow/flutter
        elif forensics.get("medium_type") == "tape":
            params["dewow_priority"] = "high"

        # If high noise, use gentler denoising
        if forensics.get("noise_level", 0) > 0.7:
            params["denoise_intensity"] = 0.5

        reprocessed = processing_function(original, sr, params)

        logger.info("Forensic-guided: %s", params)

        return reprocessed, params


# ============================================================================
# Utility Functions
# ============================================================================


def create_reprocessing_report(result: ReprocessingResult, output_path: Path) -> None:
    """
    Export reprocessing report to JSON for auditing.

    HIPS Requirement 6: Auditierbarkeit
    """
    import json

    report = {
        "success": result.success,
        "final_decision": result.final_decision,
        "strategy_used": result.strategy_used.value,
        "total_attempts": result.total_attempts,
        "best_quality_scores": result.best_quality_scores,
        "improvements_achieved": result.improvements_achieved,
        "timestamp": result.timestamp,
        "attempts": [
            {
                "attempt_number": a.attempt_number,
                "strategy": a.strategy.value,
                "parameters": a.parameters,
                "quality_scores": a.quality_scores,
                "improvements": a.improvements,
                "violations": dict(a.violations.items()),
                "success": a.success,
                "notes": a.notes,
                "timestamp": a.timestamp,
            }
            for a in result.attempts
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Reprocessing report exported to %s", output_path)
