"""
AURIK v8 Musical Goals Monitor: Pre-Validation & Continuous Monitoring
=======================================================================

Provides:
1. Pre-validation: Estimate predicted musical goals BEFORE processing
2. Continuous monitoring: Track goals during processing
3. Epistemic uncertainty assessment: Confidence in predictions
4. Impact prediction: Estimate effect of processing steps

Architecture:
- Pre-validate: audio + processing params → predicted goals + confidence
- Monitor: Track goals at multiple checkpoints during pipeline
- Alert: Warn if goals predicted to violate thresholds
- Report: Comprehensive audit trail of all predictions and actual values

Quelle: Finalisierungs_Roadmap.md - Component 0.5
Autor: AI Team
Datum: 8. Februar 2026
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreValidationResult:
    """Result of pre-validation (before processing)."""

    predicted_goals: dict[str, float]  # Predicted scores after processing
    confidence: float  # Epistemic confidence (0.0 - 1.0)
    epistemic_uncertainty: float  # Uncertainty in predictions
    recommendations: list[str]  # Recommendations based on predictions
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringCheckpoint:
    """Checkpoint während der Verarbeitung."""

    step_name: str
    timestamp: datetime
    current_goals: dict[str, float]
    violations: list[str]
    confidence: float


@dataclass
class MonitoringReport:
    """Full monitoring report."""

    pre_validation: PreValidationResult
    checkpoints: list[MonitoringCheckpoint]
    post_validation: dict[str, float] | None  # Actual final goals
    violations: list[str]
    recommendations: list[str]


class MusicalGoalsMonitor:
    """
    Überwacht musical goals throughout processing pipeline.

    Key capabilities:
    1. Pre-validate: Predict goals before processing runs
    2. Checkpoint: Track goals at intermediate steps
    3. Alert: Warn if violations predicted/detected
    4. Report: Comprehensive audit trail

    Example:
        >>> from backend.core.musical_goals import MusicalGoalsChecker
        >>>
        >>> monitor = MusicalGoalsMonitor()
        >>> checker = MusicalGoalsChecker()
        >>>
        >>> # Pre-validation
        >>> pre_result = monitor.pre_validate(
        ...     original_audio=audio,
        ...     sr=48000,
        ...     processing_config={'algorithm': 'DeepFilterNet', 'strength': 0.8}
        ... )
        >>> logger.debug(f"Predicted brillanz: {pre_result.predicted_goals['brillanz']:.3f}")
        logger.debug("Confidence: %.3f", pre_result.confidence)
        >>>
        >>> # Monitoring checkpoint during processing
        >>> monitor.add_checkpoint(
        ...     step_name='Noise Reduction',
        ...     audio=intermediate_audio,
        ...     sr=48000
        ... )
        >>>
        >>> # Final validation
        >>> final_goals = checker.measure_all(processed_audio, sr=48000)
        >>> report = monitor.finalize(final_goals)
        logger.debug("Violations: %s", report.violations)
    """

    def __init__(self) -> None:
        """Initialisiert monitor."""
        # Import Musical Goals Checker
        try:
            from backend.core.musical_goals import MusicalGoalsChecker
        except ModuleNotFoundError:
            try:
                from .musical_goals_metrics import MusicalGoalsChecker
            except ImportError:
                from musical_goals_metrics import MusicalGoalsChecker  # type: ignore[no-redef]
        self.goals_checker = MusicalGoalsChecker()

        # Monitoring state
        self.pre_validation_result: PreValidationResult | None = None
        self.checkpoints: list[MonitoringCheckpoint] = []
        self.thresholds: dict[str, float] | None = None

    def pre_validate(
        self,
        original_audio: np.ndarray,
        sr: int,
        processing_config: dict[str, Any],
        thresholds: dict[str, float] | None = None,
    ) -> PreValidationResult:
        """
        Pre-validate: Predict musical goals after processing.

        This is a PREDICTIVE model that estimates how processing will affect goals.

        Args:
            original_audio: Original audio signal
            sr: Sample rate
            processing_config: Processing configuration (algorithm, strength, etc.)
            thresholds: Optional custom thresholds

        Returns:
            PreValidationResult with predicted goals and confidence
        """
        # Measure current goals
        current_goals = self.goals_checker.measure_all(original_audio, sr)

        # Use thresholds or defaults
        if thresholds:
            self.thresholds = thresholds
        else:
            self.thresholds = self.goals_checker.thresholds

        # PREDICTIVE MODEL: Estimate impact of processing
        # This is a SIMPLIFIED model - in production, use ML-based prediction
        predicted_goals = self._predict_impact(current_goals=current_goals, processing_config=processing_config)

        # Epistemic uncertainty (confidence in prediction)
        # Higher uncertainty for:
        # - Aggressive processing (high strength)
        # - Unknown algorithms
        # - Low current scores (already degraded)
        uncertainty = self._estimate_uncertainty(current_goals=current_goals, processing_config=processing_config)
        confidence = 1.0 - uncertainty

        # Check for predicted violations
        recommendations = []
        for goal_name, predicted_score in predicted_goals.items():
            threshold = self.thresholds[goal_name]
            if predicted_score < threshold:
                recommendations.append(
                    f"WARNING: {goal_name} predicted to drop below threshold "
                    f"({predicted_score:.3f} < {threshold:.2f}). "
                    "Consider reducing processing strength."
                )

        if not recommendations:
            recommendations.append("All goals predicted to be preserved.")

        result = PreValidationResult(
            predicted_goals=predicted_goals,
            confidence=confidence,
            epistemic_uncertainty=uncertainty,
            recommendations=recommendations,
        )

        self.pre_validation_result = result

        logger.info(
            f"Pre-Validierung vollstaendig: confidence={confidence:.3f}, "
            f"{len([r for r in recommendations if 'WARNING' in r])} warnings"
        )

        return result

    def add_checkpoint(self, step_name: str, audio: np.ndarray, sr: int, confidence: float = 0.80) -> None:
        """
        Fügt hinzu: monitoring checkpoint during processing.

        Args:
            step_name: Name of processing step
            audio: Current audio state
            sr: Sample rate
            confidence: Current epistemic confidence
        """
        # Measure current goals
        current_goals = self.goals_checker.measure_all(audio, sr)

        # Check for violations
        violations = []
        if self.thresholds:
            for goal_name, score in current_goals.items():
                if goal_name in self.thresholds and score < self.thresholds[goal_name]:
                    violations.append(goal_name)

        checkpoint = MonitoringCheckpoint(
            step_name=step_name,
            timestamp=datetime.now(),
            current_goals=current_goals,
            violations=violations,
            confidence=confidence,
        )

        self.checkpoints.append(checkpoint)

        if violations:
            logger.warning(
                "Checkpoint '%s': %s violations erkannt - %s", step_name, len(violations), ", ".join(violations)
            )
        else:
            logger.info("Checkpoint '%s': All goals OK", step_name)

    def finalize(self, final_goals: dict[str, float]) -> MonitoringReport:
        """
        Finalize monitoring with actual final goals.

        Args:
            final_goals: Actual measured goals after processing

        Returns:
            MonitoringReport with full audit trail
        """
        # Check final violations
        violations = []
        if self.thresholds:
            for goal_name, score in final_goals.items():
                if goal_name in self.thresholds and score < self.thresholds[goal_name]:
                    violations.append(goal_name)

        # Recommendations based on final state
        recommendations = []
        if violations:
            recommendations.append(
                f"FINAL VIOLATIONS: {len(violations)} goal(s) violated - "
                f"{', '.join(violations)}. Consider rollback or parameter adjustment."
            )
        else:
            recommendations.append("All goals successfully preserved.")

        # Compare predictions vs. actuals
        if self.pre_validation_result:
            for goal_name in final_goals:
                if goal_name in self.pre_validation_result.predicted_goals:
                    predicted = self.pre_validation_result.predicted_goals[goal_name]
                    actual = final_goals[goal_name]
                    error = abs(predicted - actual)
                    if error > 0.10:
                        recommendations.append(
                            f"Large prediction error for {goal_name}: "
                            f"predicted={predicted:.3f}, actual={actual:.3f}, "
                            f"error={error:.3f}. Consider recalibrating predictor."
                        )

        report = MonitoringReport(
            pre_validation=self.pre_validation_result,  # type: ignore[arg-type]
            checkpoints=self.checkpoints,
            post_validation=final_goals,
            violations=violations,
            recommendations=recommendations,
        )

        logger.info("Monitoring finalized: %s checkpoints, %s final violations", len(self.checkpoints), len(violations))

        return report

    def reset(self) -> None:
        """Setzt zurück: monitoring state for new processing run."""
        self.pre_validation_result = None
        self.checkpoints = []
        self.thresholds = None

    def _predict_impact(self, current_goals: dict[str, float], processing_config: dict[str, Any]) -> dict[str, float]:
        """
        Predict impact of processing on musical goals.

        SIMPLIFIED MODEL - in production, use ML-based prediction.

        Args:
            current_goals: Current measured goals
            processing_config: Processing configuration

        Returns:
            Predicted goals after processing
        """
        # Extract processing parameters
        algorithm = processing_config.get("algorithm", "unknown")
        strength = processing_config.get("strength", 0.5)

        # Impact factors (algorithm-specific)
        # Negative impact increases with strength
        impact_factors = {
            "DeepFilterNet": {
                "bass_kraft": -0.02 * strength,  # Slight bass loss
                "brillanz": 0.01 * strength,  # Slight HF boost
                "waerme": -0.03 * strength,  # Mid loss
                "natuerlichkeit": -0.05 * strength,  # Artifacts
                "authentizitaet": -0.04 * strength,  # Voice change
                "emotionalitaet": -0.03 * strength,  # Dynamics loss
                "transparenz": 0.02 * strength,  # Clarity gain
            },
            "ResembleEnhance": {
                "bass_kraft": 0.01 * strength,
                "brillanz": 0.03 * strength,
                "waerme": 0.02 * strength,
                "natuerlichkeit": -0.02 * strength,
                "authentizitaet": -0.03 * strength,
                "emotionalitaet": 0.01 * strength,
                "transparenz": 0.03 * strength,
            },
            # Default (unknown algorithm)
            "unknown": dict.fromkeys(current_goals, -0.05 * strength),
        }

        # Get algorithm-specific impacts
        impacts = impact_factors.get(algorithm, impact_factors["unknown"])

        # Apply impacts
        predicted = {}
        for goal_name, current_score in current_goals.items():
            impact = impacts.get(goal_name, -0.03 * strength)
            predicted[goal_name] = current_score + impact
            # Clip to [0, 1]
            predicted[goal_name] = min(1.0, max(0.0, predicted[goal_name]))

        return predicted

    def _estimate_uncertainty(self, current_goals: dict[str, float], processing_config: dict[str, Any]) -> float:
        """
        Schätzt epistemic uncertainty in prediction.

        Args:
            current_goals: Current measured goals
            processing_config: Processing configuration

        Returns:
            Uncertainty (0.0 - 1.0, higher = more uncertain)
        """
        # Factors increasing uncertainty:
        # 1. High processing strength
        strength = processing_config.get("strength", 0.5)
        strength_uncertainty = strength * 0.3  # Max 0.3 for strength=1.0

        # 2. Unknown algorithm
        algorithm = processing_config.get("algorithm", "unknown")
        algorithm_uncertainty = 0.2 if algorithm == "unknown" else 0.0

        # 3. Low current scores (already degraded)
        mean_score = np.mean(list(current_goals.values()))
        degradation_uncertainty = max(0.0, (0.80 - mean_score) * 0.5)

        # Combined uncertainty (capped at 0.50)
        total_uncertainty = min(0.50, strength_uncertainty + algorithm_uncertainty + degradation_uncertainty)

        return total_uncertainty  # type: ignore[no-any-return]


if __name__ == "__main__":
    # Test Musical Goals Monitor
    logger.debug("=== AURIK v8 Musical Goals Monitor Test ===\n")

    from musical_goals_metrics import MusicalGoalsChecker

    # Create test audio
    sr = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (
        0.3 * np.sin(2 * np.pi * 100 * t)
        + 0.3 * np.sin(2 * np.pi * 500 * t)
        + 0.2 * np.sin(2 * np.pi * 2000 * t)
        + 0.2 * np.sin(2 * np.pi * 8000 * t)
    )

    # Initialize monitor
    monitor = MusicalGoalsMonitor()

    # Pre-validation
    logger.debug("1. Pre-Validierung:")
    pre_result = monitor.pre_validate(
        original_audio=audio, sr=sr, processing_config={"algorithm": "DeepFilterNet", "strength": 0.8}
    )
    logger.debug("   Confidence: %.3f", pre_result.confidence)
    logger.debug("   Uncertainty: %.3f", pre_result.epistemic_uncertainty)
    logger.debug("   Predicted goals:")
    for goal, score in pre_result.predicted_goals.items():
        logger.debug("      %s: %.3f", goal, score)
    logger.debug("   Recommendations:")
    for rec in pre_result.recommendations:
        logger.debug("      - %s", rec)

    # Checkpoint 1
    logger.debug("\n2. Checkpoint 1 (Noise Reduction):")
    monitor.add_checkpoint("Noise Reduction", audio, sr, confidence=0.85)

    # Checkpoint 2
    logger.debug("3. Checkpoint 2 (Enhancement):")
    monitor.add_checkpoint("Enhancement", audio, sr, confidence=0.80)

    # Final validation
    logger.debug("\n4. Final Validierung:")
    checker = MusicalGoalsChecker()
    final_goals = checker.measure_all(audio, sr)
    report = monitor.finalize(final_goals)

    logger.debug("   Checkpoints: %s", len(report.checkpoints))
    logger.debug("   Violations: %s", report.violations if report.violations else "None")
    logger.debug("   Recommendations:")
    for rec in report.recommendations:
        logger.debug("      - %s", rec)

    logger.debug("\n=== Test vollstaendig ===")
