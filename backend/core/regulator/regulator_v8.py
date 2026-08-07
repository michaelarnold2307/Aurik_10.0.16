"""
AURIK v8 Regulator: Musical Goals Violation Handling & Adaptive Parameters
===========================================================================

Detects Musical Goal violations and adaptively adjusts processing parameters.
Makes hard stop decisions when critical thresholds are violated.

Architecture:
- Pre-validation: Check predicted musical goals before processing
- Parameter Adaptation: Adjust strength/aggressiveness based on violations
- Hard Stop: Reject processing if predicted score < critical threshold
- Post-validation: Verify actual results match predictions

Quelle: Finalisierungs_Roadmap.md - Component 0.4
Autor: AI Team
Datum: 8. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of Musical Goal violations."""

    MINOR = "minor"  # < threshold but > hard_stop
    CRITICAL = "critical"  # < hard_stop → reject processing
    SEVERE = "severe"  # Multiple goals violated


class DecisionType(Enum):
    """Regulator decisions."""

    ALLOW = "allow"  # Proceed with current parameters
    ADJUST_DOWN = "adjust_down"  # Reduce aggressiveness
    ADJUST_UP = "adjust_up"  # Increase aggressiveness
    HARD_STOP = "hard_stop"  # Reject processing


@dataclass
class ViolationReport:
    """Report of Musical Goal violations."""

    violated_goals: list[str]
    violation_type: ViolationType
    severities: dict[str, float]  # goal_name → (threshold - score)
    recommendation: str


@dataclass
class RegulatorDecision:
    """Decision from Regulator."""

    decision: DecisionType
    allowed: bool
    parameter_adjustments: dict[str, float]  # parameter_name → new_value
    reasoning: str
    violations: ViolationReport | None


class RegulatorV8:
    """
    v8 Regulator for Musical Goals preservation and adaptive control.

    Workflow:
    1. Pre-validate: Check predicted musical goals vs. thresholds
    2. Decide: Allow / Adjust / Hard Stop
    3. Adjust: Modify processing parameters if needed
    4. Post-validate: Check actual results

    Example:
        >>> regulator = RegulatorV8()
        >>>
        >>> # Pre-validation
        >>> decision = regulator.pre_validate(
        ...     predicted_goals={'brillanz': 0.82, 'waerme': 0.75, ...},
        ...     thresholds={'brillanz': 0.85, 'waerme': 0.80, ...}
        ... )
        >>>
        >>> if decision.decision == DecisionType.HARD_STOP:
        logger.debug("Processing rejected: %s", decision.reasoning)
        >>> elif decision.decision == DecisionType.ADJUST_DOWN:
        ...     # Apply parameter adjustments
        ...     adjusted_params = decision.parameter_adjustments
        logger.debug("Reducing aggressiveness: %s", adjusted_params)
    """

    # Hard stop thresholds (from conduct_rules.yaml)
    HARD_STOP_THRESHOLDS = {
        "bass_kraft": 0.70,
        "brillanz": 0.70,
        "waerme": 0.65,
        "natuerlichkeit": 0.75,
        "authentizitaet": 0.70,
        "emotionalitaet": 0.70,
        "transparenz": 0.75,
    }

    # Parameter adjustment strategies
    # Maps violation severity to parameter multipliers
    ADJUSTMENT_STRATEGIES = {
        "minor": {
            "strength": 0.9,  # Reduce by 10%
            "aggressiveness": 0.85,  # Reduce by 15%
            "threshold": 0.9,  # Reduce by 10%
        },
        "critical": {
            "strength": 0.7,  # Reduce by 30%
            "aggressiveness": 0.6,  # Reduce by 40%
            "threshold": 0.7,  # Reduce by 30%
        },
    }

    def __init__(self):
        """Initialisiert Regulator."""
        self.stats = {
            "allow_count": 0,
            "adjust_down_count": 0,
            "adjust_up_count": 0,
            "hard_stop_count": 0,
        }

    def pre_validate(
        self,
        predicted_goals: dict[str, float],
        thresholds: dict[str, float],
        current_parameters: dict[str, float] | None = None,
    ) -> RegulatorDecision:
        """
        Pre-validate predicted musical goals before processing.

        Args:
            predicted_goals: Predicted scores for all 7 musical goals
            thresholds: Zone/medium-specific thresholds
            current_parameters: Current processing parameters (for adjustment)

        Returns:
            RegulatorDecision with decision type and parameter adjustments
        """
        if current_parameters is None:
            current_parameters = {}

        # Check for violations
        violations = []
        violation_severities = {}
        hard_stop_required = False

        for goal_name, predicted_score in predicted_goals.items():
            if goal_name not in thresholds:
                logger.warning("No Schwelle for goal '%s', skipping", goal_name)
                continue

            threshold = thresholds[goal_name]
            hard_stop_threshold = self.HARD_STOP_THRESHOLDS.get(goal_name, 0.70)

            if predicted_score < hard_stop_threshold:
                # CRITICAL: Below hard stop
                violations.append(goal_name)
                violation_severities[goal_name] = threshold - predicted_score
                hard_stop_required = True
                logger.error(
                    f"HARD STOP: {goal_name} predicted Wert {predicted_score:.3f} "
                    f"< hard stop Schwelle {hard_stop_threshold:.2f}"
                )
            elif predicted_score < threshold:
                # MINOR: Below threshold but above hard stop
                violations.append(goal_name)
                violation_severities[goal_name] = threshold - predicted_score
                logger.warning(
                    f"Violation: {goal_name} predicted Wert {predicted_score:.3f} < Schwelle {threshold:.2f}"
                )

        # Determine violation type
        if hard_stop_required:
            violation_type = ViolationType.CRITICAL
        elif len(violations) >= 3:
            violation_type = ViolationType.SEVERE
        elif len(violations) > 0:
            violation_type = ViolationType.MINOR
        else:
            violation_type = None

        # Make decision
        if hard_stop_required:
            decision = self._make_hard_stop_decision(violations, violation_severities)
            self.stats["hard_stop_count"] += 1
        elif violation_type == ViolationType.SEVERE:
            decision = self._make_adjust_down_decision(
                violations,
                violation_severities,
                current_parameters,
                severity="critical",
            )
            self.stats["adjust_down_count"] += 1
        elif violation_type == ViolationType.MINOR:
            decision = self._make_adjust_down_decision(
                violations, violation_severities, current_parameters, severity="minor"
            )
            self.stats["adjust_down_count"] += 1
        else:
            decision = self._make_allow_decision()
            self.stats["allow_count"] += 1

        return decision

    def post_validate(
        self,
        original_goals: dict[str, float],
        processed_goals: dict[str, float],
        thresholds: dict[str, float],
    ) -> RegulatorDecision:
        """
        Post-validate actual musical goals after processing.

        Args:
            original_goals: Original scores
            processed_goals: Processed scores
            thresholds: Zone/medium-specific thresholds

        Returns:
            RegulatorDecision (usually ALLOW or HARD_STOP for rollback)
        """
        # Check for violations in processed audio
        violations = []
        violation_severities = {}
        hard_stop_required = False

        for goal_name, processed_score in processed_goals.items():
            if goal_name not in thresholds:
                continue

            threshold = thresholds[goal_name]
            hard_stop_threshold = self.HARD_STOP_THRESHOLDS.get(goal_name, 0.70)

            if processed_score < hard_stop_threshold:
                violations.append(goal_name)
                violation_severities[goal_name] = threshold - processed_score
                hard_stop_required = True
                logger.error(
                    f"POST-Validierung Fehlschlag: {goal_name} actual Wert {processed_score:.3f} "
                    f"< hard stop Schwelle {hard_stop_threshold:.2f}"
                )
            elif processed_score < threshold:
                violations.append(goal_name)
                violation_severities[goal_name] = threshold - processed_score
                logger.warning(
                    f"Post-Validierung violation: {goal_name} actual Wert {processed_score:.3f} "
                    f"< Schwelle {threshold:.2f}"
                )

        if hard_stop_required:
            # ROLLBACK REQUIRED
            decision = RegulatorDecision(
                decision=DecisionType.HARD_STOP,
                allowed=False,
                parameter_adjustments={},
                reasoning=(
                    f"Post-validation failed: {len(violations)} goal(s) violated. "
                    f"Violated: {', '.join(violations)}. ROLLBACK REQUIRED."
                ),
                violations=ViolationReport(
                    violated_goals=violations,
                    violation_type=ViolationType.CRITICAL,
                    severities=violation_severities,
                    recommendation="Rollback to original audio",
                ),
            )
            self.stats["hard_stop_count"] += 1
        else:
            # Accept (even if minor violations)
            decision = RegulatorDecision(
                decision=DecisionType.ALLOW,
                allowed=True,
                parameter_adjustments={},
                reasoning=(
                    f"Post-validation passed. {len(violations)} minor violations (acceptable)."
                    if violations
                    else "Post-validation passed. All goals preserved."
                ),
                violations=None,
            )
            self.stats["allow_count"] += 1

        return decision

    def _make_hard_stop_decision(self, violations: list[str], severities: dict[str, float]) -> RegulatorDecision:
        """Erstellt hard stop decision."""
        return RegulatorDecision(
            decision=DecisionType.HARD_STOP,
            allowed=False,
            parameter_adjustments={},
            reasoning=(
                f"Processing rejected: {len(violations)} critical violation(s). "
                f"Goals below hard stop threshold: {', '.join(violations)}. "
                "Recommendation: Skip processing or use gentler algorithm."
            ),
            violations=ViolationReport(
                violated_goals=violations,
                violation_type=ViolationType.CRITICAL,
                severities=severities,
                recommendation="Skip processing or use gentler algorithm",
            ),
        )

    def _make_adjust_down_decision(
        self,
        violations: list[str],
        severities: dict[str, float],
        current_parameters: dict[str, float],
        severity: str,
    ) -> RegulatorDecision:
        """Erstellt adjust down decision with parameter modifications."""
        # Get adjustment multipliers
        multipliers = self.ADJUSTMENT_STRATEGIES[severity]

        # Adjust parameters
        adjusted_params = {}
        for param_name, current_value in current_parameters.items():
            if param_name in multipliers:
                adjusted_params[param_name] = current_value * multipliers[param_name]
            else:
                # Default: reduce by 20% for unknown parameters
                adjusted_params[param_name] = current_value * 0.8

        # If no current parameters, provide defaults
        if not adjusted_params:
            adjusted_params = {
                "strength": 0.8 if severity == "minor" else 0.6,
                "aggressiveness": 0.7 if severity == "minor" else 0.5,
            }

        violation_type = ViolationType.SEVERE if severity == "critical" else ViolationType.MINOR

        return RegulatorDecision(
            decision=DecisionType.ADJUST_DOWN,
            allowed=True,
            parameter_adjustments=adjusted_params,
            reasoning=(
                f"Reducing aggressiveness due to {len(violations)} predicted violation(s). "
                f"Violated goals: {', '.join(violations)}. "
                f"Severity: {severity}. Adjusted parameters: "
                f"{', '.join(f'{k}={v:.2f}' for k, v in adjusted_params.items())}"
            ),
            violations=ViolationReport(
                violated_goals=violations,
                violation_type=violation_type,
                severities=severities,
                recommendation=f"Reduce strength/aggressiveness ({severity} violations)",
            ),
        )

    def _make_allow_decision(self) -> RegulatorDecision:
        """Erstellt allow decision."""
        return RegulatorDecision(
            decision=DecisionType.ALLOW,
            allowed=True,
            parameter_adjustments={},
            reasoning="All musical goals predicted to be preserved. Processing allowed.",
            violations=None,
        )

    def get_statistics(self) -> dict[str, int]:
        """Gibt zurück: regulator decision statistics."""
        return self.stats.copy()

    def reset_statistics(self):
        """Setzt zurück: regulator statistics."""
        self.stats = {
            "allow_count": 0,
            "adjust_down_count": 0,
            "adjust_up_count": 0,
            "hard_stop_count": 0,
        }


if __name__ == "__main__":
    # Test Regulator v8
    logger.debug("=== AURIK v8 Regulator Test ===\n")

    regulator = RegulatorV8()

    # Define thresholds (Zone B, vinyl)
    thresholds = {
        "bass_kraft": 0.935,
        "brillanz": 0.935,
        "waerme": 0.880,
        "natuerlichkeit": 0.990,
        "authentizitaet": 0.968,
        "emotionalitaet": 0.957,
        "transparenz": 0.979,
    }

    # Test Case 1: All goals meet thresholds → ALLOW
    logger.debug("Test 1: All goals above thresholds")
    predicted_goals_ok = {
        "bass_kraft": 0.95,
        "brillanz": 0.94,
        "waerme": 0.89,
        "natuerlichkeit": 1.00,
        "authentizitaet": 0.97,
        "emotionalitaet": 0.96,
        "transparenz": 0.98,
    }
    decision1 = regulator.pre_validate(predicted_goals_ok, thresholds)
    logger.debug("   Decision: %s", decision1.decision.value)
    logger.debug("   Allowed: %s", decision1.allowed)
    logger.debug("   Reasoning: %s\n", decision1.reasoning)

    # Test Case 2: Minor violation → ADJUST_DOWN
    logger.debug("Test 2: Minor violation (brillanz slightly below)")
    predicted_goals_minor = predicted_goals_ok.copy()
    predicted_goals_minor["brillanz"] = 0.92  # Below 0.935
    decision2 = regulator.pre_validate(
        predicted_goals_minor,
        thresholds,
        current_parameters={"strength": 1.0, "aggressiveness": 0.8},
    )
    logger.debug("   Decision: %s", decision2.decision.value)
    logger.debug("   Allowed: %s", decision2.allowed)
    logger.debug("   Adjustments: %s", decision2.parameter_adjustments)
    logger.debug("   Reasoning: %s\n", decision2.reasoning)

    # Test Case 3: Critical violation → HARD_STOP
    logger.debug("Test 3: Critical violation (natuerlichkeit below hard stop)")
    predicted_goals_critical = predicted_goals_ok.copy()
    predicted_goals_critical["natuerlichkeit"] = 0.72  # Below 0.75 hard stop
    decision3 = regulator.pre_validate(predicted_goals_critical, thresholds)
    logger.debug("   Decision: %s", decision3.decision.value)
    logger.debug("   Allowed: %s", decision3.allowed)
    logger.debug("   Reasoning: %s\n", decision3.reasoning)

    # Test Case 4: Post-validation (all ok)
    logger.debug("Test 4: Post-Validierung (all goals preserved)")
    original_goals = predicted_goals_ok
    processed_goals = predicted_goals_ok.copy()
    decision4 = regulator.post_validate(original_goals, processed_goals, thresholds)
    logger.debug("   Decision: %s", decision4.decision.value)
    logger.debug("   Allowed: %s", decision4.allowed)
    logger.debug("   Reasoning: %s\n", decision4.reasoning)

    # Statistics
    logger.debug("Statistics:")
    stats = regulator.get_statistics()
    logger.debug("   Allow: %s", stats["allow_count"])
    logger.debug("   anpassen Down: %s", stats["adjust_down_count"])
    logger.debug("   Hard Stop: %s", stats["hard_stop_count"])

    logger.debug("\n=== Test vollstaendig ===")
