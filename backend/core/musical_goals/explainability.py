"""
Goal Explainability & User Transparency System

Component 4.4: Explainability & User Transparency
Impact: +0.5 Punkte - User versteht Musical Goals Achievement/Failure

Provides user-friendly explanations of Musical Goals achievement:
- Step-by-step processing attribution (which step affected which goal)
- Natural language explanations (user-friendly, not technical)
- Goal trajectory visualization (Pre → During → Post)
- Comparison with baselines and targets
- Actionable recommendations for improvements

Problem:
Users don't understand WHY Musical Goals were achieved or why they failed.
Without transparency, users cannot debug or optimize their processing workflow.

Solution:
GoalExplainer generates comprehensive, user-friendly explanations that show:
1. How each processing step affected each goal
2. Overall goal trajectory from original to final
3. Which goals were achieved, which failed, and why
4. Recommendations for improving failed goals

Author: AI Team
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
from backend.core.musical_goals.processing_modes import PROCESSING_MODE_CONFIGS, ProcessingMode

logger = logging.getLogger(__name__)


class GoalChangeType(Enum):
    """Type of goal change."""

    IMPROVED = "improved"  # Score increased
    DEGRADED = "degraded"  # Score decreased
    MAINTAINED = "maintained"  # Score stayed similar (±0.02)
    ACHIEVED = "achieved"  # Reached target threshold
    FAILED = "failed"  # Below target threshold


@dataclass
class ProcessingStepImpact:
    """
    Impact of a single processing step on musical goals.

    Attributes:
        step_name: Name of processing step (e.g., "DeHum", "EQ", "Limiter")
        step_index: Order in processing chain
        goal_changes: Dict mapping goal name to score delta
        overall_impact: Total impact across all goals
        positive_impacts: Goals that improved
        negative_impacts: Goals that degraded
        explanation: User-friendly explanation of this step's impact
    """

    step_name: str
    step_index: int
    goal_changes: dict[str, float]
    overall_impact: float
    positive_impacts: list[str]
    negative_impacts: list[str]
    explanation: str


@dataclass
class GoalTrajectory:
    """
    Trajectory of a single goal through processing chain.

    Attributes:
        goal_name: Name of the goal
        initial_score: Score before any processing
        final_score: Score after all processing
        target_score: Target threshold for this goal
        step_scores: Score after each processing step
        step_deltas: Delta contributed by each step
        achieved: Whether target was achieved
        change_type: Type of overall change
        explanation: User-friendly explanation of trajectory
    """

    goal_name: str
    initial_score: float
    final_score: float
    target_score: float
    step_scores: list[float]
    step_deltas: list[float]
    achieved: bool
    change_type: GoalChangeType
    explanation: str


@dataclass
class GoalExplanation:
    """
    Complete explanation of Musical Goals achievement.

    Attributes:
        mode: Processing mode used
        overall_success: Whether all goals were achieved
        achieved_goals: List of achieved goal names
        failed_goals: List of failed goal names
        goal_trajectories: Trajectory for each goal
        step_impacts: Impact of each processing step
        summary: High-level summary for user
        recommendations: Actionable recommendations
        details: Additional details for debugging
    """

    mode: ProcessingMode
    overall_success: bool
    achieved_goals: list[str]
    failed_goals: list[str]
    goal_trajectories: dict[str, GoalTrajectory]
    step_impacts: list[ProcessingStepImpact]
    summary: str
    recommendations: list[str]
    details: dict[str, Any] = field(default_factory=dict)


class GoalExplainer:
    """
    Generiert user-friendly explanations of Musical Goals achievement.

    This system provides transparency into how processing affected goals:
    1. Step attribution: Which steps improved/degraded which goals
    2. Trajectory visualization: How goals changed through processing
    3. Natural language: User-friendly explanations, not technical jargon
    4. Recommendations: Actionable advice for improving failed goals

    Example:
        >>> explainer = GoalExplainer()
        >>>
        >>> # Track goals through processing chain
        >>> explainer.start_tracking(original_audio, sr, mode=ProcessingMode.RESTORATION)
        >>>
        >>> # After each step
        >>> explainer.record_step("DeHum", audio_after_dehum, sr)
        >>> explainer.record_step("EQ", audio_after_eq, sr)
        >>> explainer.record_step("Limiter", audio_after_limiter, sr)
        >>>
        >>> # Generate explanation
        >>> explanation = explainer.generate_explanation()
        >>> logger.debug(explanation.summary)
        >>> for rec in explanation.recommendations:
        logger.debug("  - %s", rec)
    """

    def __init__(self, checker: MusicalGoalsChecker | None = None) -> None:
        """
        Initialisiert goal explainer.

        Args:
            checker: Optional MusicalGoalsChecker instance
        """
        self.checker = checker or MusicalGoalsChecker()

        # Tracking state
        self.is_tracking = False
        self.mode: ProcessingMode | None = None
        self.original_audio: np.ndarray | None = None
        self.sr: int | None = None
        self.step_history: list[dict] = []
        self.goal_history: list[dict[str, float]] = []

        # User-friendly goal names
        self.goal_display_names = {
            "brillanz": "Brilliance",
            "waerme": "Warmth",
            "natuerlichkeit": "Naturalness",
            "authentizitaet": "Authenticity",
            "emotionalitaet": "Emotionality",
            "transparenz": "Transparency",
            "bass_kraft": "Bass Power",
            "bass-kraft": "Bass Power",
        }

        # Impact descriptions for common processing steps
        self.step_descriptions = {
            "dehum": "Hum Removal",
            "dehiss": "Hiss Reduction",
            "declick": "Click Removal",
            "declip": "Clipping Restoration",
            "decrackle": "Crackle Removal",
            "denoise": "Noise Reduction",
            "eq": "Equalization",
            "compressor": "Dynamic Compression",
            "limiter": "Peak Limiting",
            "reverb": "Reverb Enhancement",
            "stereo_widener": "Stereo Widening",
            "pitch_correction": "Pitch Correction",
            "restoration": "General Restoration",
        }

    # =========================================================================
    # Tracking Methods
    # =========================================================================

    def start_tracking(
        self, original_audio: np.ndarray, sr: int, mode: ProcessingMode = ProcessingMode.RESTORATION
    ) -> None:
        """
        Start tracking Musical Goals through processing chain.

        Args:
            original_audio: Original audio before processing
            sr: Sample rate
            mode: Processing mode (determines goal targets)
        """
        self.is_tracking = True
        self.mode = mode
        self.original_audio = original_audio.copy()
        self.sr = sr
        self.step_history = []
        self.goal_history = []

        # Measure initial goals
        initial_scores = self.checker.measure_all(original_audio, sr)
        self.goal_history.append(
            {"step": "Original", "scores": initial_scores, "audio_hash": hash(original_audio.tobytes())}  # type: ignore[dict-item]
        )

        logger.info("gestartet goal tracking in %s Betriebsart", mode.value)
        logger.info("Initial scores: %s", initial_scores)

    def record_step(
        self, step_name: str, processed_audio: np.ndarray, sr: int, step_params: dict | None = None
    ) -> dict[str, float]:
        """
        Record impact of a processing step on goals.

        Args:
            step_name: Name of the processing step
            processed_audio: Audio after this step
            sr: Sample rate
            step_params: Optional parameters used in this step

        Returns:
            Dict of current goal scores
        """
        if not self.is_tracking:
            raise RuntimeError("Not tracking! Call start_tracking() first.")

        # Measure goals after this step
        current_scores = self.checker.measure_all(processed_audio, sr, reference=self.original_audio)

        # Record step
        self.step_history.append(
            {"name": step_name, "params": step_params or {}, "audio_hash": hash(processed_audio.tobytes())}
        )

        self.goal_history.append(
            {"step": step_name, "scores": current_scores, "audio_hash": hash(processed_audio.tobytes())}  # type: ignore[dict-item]
        )

        logger.debug("Recorded step '%s': %s", step_name, current_scores)

        return current_scores

    def stop_tracking(self) -> None:
        """Stop tracking."""
        self.is_tracking = False
        logger.info("gestoppt goal tracking")

    # =========================================================================
    # Explanation Generation
    # =========================================================================

    def generate_explanation(self) -> GoalExplanation:
        """
        Generiert comprehensive explanation of Musical Goals achievement.

        Returns:
            GoalExplanation with all details
        """
        if len(self.goal_history) < 2:
            raise RuntimeError("Need at least original + 1 processed step to explain")

        _mode = self.mode or ProcessingMode.RESTORATION
        mode_config = (
            PROCESSING_MODE_CONFIGS[_mode]
            if _mode in PROCESSING_MODE_CONFIGS
            else PROCESSING_MODE_CONFIGS[ProcessingMode.RESTORATION]
        )

        # Build goal trajectories
        goal_trajectories = self._build_goal_trajectories(mode_config)

        # Calculate step impacts
        step_impacts = self._calculate_step_impacts()

        # Determine achieved/failed goals
        achieved_goals = []
        failed_goals = []
        for goal_name, trajectory in goal_trajectories.items():
            if trajectory.achieved:
                achieved_goals.append(goal_name)
            else:
                failed_goals.append(goal_name)

        # Overall success
        overall_success = len(failed_goals) == 0

        # Generate summary
        summary = self._generate_summary(achieved_goals, failed_goals, goal_trajectories)

        # Generate recommendations
        recommendations = self._generate_recommendations(failed_goals, goal_trajectories, step_impacts)

        # Compile details
        details = {
            "mode": (self.mode or ProcessingMode.RESTORATION).value,
            "num_steps": len(self.step_history),
            "initial_scores": self.goal_history[0]["scores"],
            "final_scores": self.goal_history[-1]["scores"],
            "targets": mode_config.musical_goals,
        }

        return GoalExplanation(
            mode=self.mode,  # type: ignore[arg-type]
            overall_success=overall_success,
            achieved_goals=achieved_goals,
            failed_goals=failed_goals,
            goal_trajectories=goal_trajectories,
            step_impacts=step_impacts,
            summary=summary,
            recommendations=recommendations,
            details=details,
        )

    def _build_goal_trajectories(self, mode_config) -> dict[str, GoalTrajectory]:
        """Erstellt trajectory for each goal."""
        trajectories = {}

        # Get all goal names
        initial_scores = self.goal_history[0]["scores"]
        goal_names = list(initial_scores.keys())  # type: ignore[attr-defined]

        for goal_name in goal_names:
            # Extract scores through processing chain
            step_scores = []
            for entry in self.goal_history:
                step_scores.append(entry["scores"][goal_name])  # type: ignore[index]

            # Calculate deltas
            step_deltas = []
            for i in range(1, len(step_scores)):
                delta = step_scores[i] - step_scores[i - 1]
                step_deltas.append(delta)

            initial = step_scores[0]
            final = step_scores[-1]
            target = mode_config.musical_goals.get(goal_name, 0.85)

            # Determine achievement (explicit bool cast for numpy types)
            achieved = bool(final >= target)

            # Determine change type
            change_delta = final - initial
            if achieved:
                change_type = GoalChangeType.ACHIEVED
            elif abs(change_delta) < 0.02:
                change_type = GoalChangeType.MAINTAINED
            elif change_delta > 0:
                change_type = GoalChangeType.IMPROVED
            else:
                change_type = GoalChangeType.DEGRADED

            # Generate explanation
            explanation = self._explain_trajectory(
                goal_name, initial, final, target, step_deltas, change_type, achieved
            )

            trajectories[goal_name] = GoalTrajectory(
                goal_name=goal_name,
                initial_score=initial,
                final_score=final,
                target_score=target,
                step_scores=step_scores,
                step_deltas=step_deltas,
                achieved=achieved,
                change_type=change_type,
                explanation=explanation,
            )

        return trajectories

    def _calculate_step_impacts(self) -> list[ProcessingStepImpact]:
        """Calculate impact of each processing step."""
        impacts = []

        for i, step in enumerate(self.step_history):
            step_name = step["name"]

            # Get scores before and after this step
            scores_before = self.goal_history[i]["scores"]
            scores_after = self.goal_history[i + 1]["scores"]

            # Calculate changes
            goal_changes = {}
            positive = []
            negative = []
            total_impact = 0.0

            for goal_name in scores_before.keys():  # type: ignore[attr-defined]
                delta = scores_after[goal_name] - scores_before[goal_name]  # type: ignore[index]
                goal_changes[goal_name] = delta
                total_impact += abs(delta)

                if delta > 0.01:
                    positive.append(goal_name)
                elif delta < -0.01:
                    negative.append(goal_name)

            # Generate explanation
            explanation = self._explain_step_impact(step_name, goal_changes, positive, negative)

            impacts.append(
                ProcessingStepImpact(
                    step_name=step_name,
                    step_index=i,
                    goal_changes=goal_changes,
                    overall_impact=total_impact,
                    positive_impacts=positive,
                    negative_impacts=negative,
                    explanation=explanation,
                )
            )

        return impacts

    def _explain_trajectory(
        self,
        goal_name: str,
        initial: float,
        final: float,
        target: float,
        deltas: list[float],
        change_type: GoalChangeType,
        achieved: bool,
    ) -> str:
        """Generiert user-friendly trajectory explanation."""
        display_name = self.goal_display_names.get(goal_name, goal_name.title())

        delta = final - initial
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"

        status = "✅" if achieved else "❌"

        explanation = f"{display_name}: {initial:.2f} → {final:.2f} ({delta_str}) {status}\n"
        explanation += f"  Target: {target:.2f}, "

        if achieved:
            explanation += f"Achieved! (+{(final - target):.2f} above target)"
        else:
            explanation += f"Not achieved ({(target - final):.2f} below target)"

        # Summarize major contributors
        if len(deltas) > 0:
            max_positive = max(deltas)
            max_negative = min(deltas)

            if max_positive > 0.02:
                idx = deltas.index(max_positive)
                step_name = self.step_history[idx]["name"]
                explanation += f"\n  Best step: {step_name} (+{max_positive:.2f})"

            if max_negative < -0.02:
                idx = deltas.index(max_negative)
                step_name = self.step_history[idx]["name"]
                explanation += f"\n  Worst step: {step_name} ({max_negative:.2f})"

        return explanation

    def _explain_step_impact(
        self, step_name: str, goal_changes: dict[str, float], positive: list[str], negative: list[str]
    ) -> str:
        """Generiert user-friendly step impact explanation."""
        display_name = self.step_descriptions.get(step_name.lower(), step_name)

        explanation = f"{display_name}:"

        if len(positive) == 0 and len(negative) == 0:
            explanation += " No significant impact"
            return explanation

        if len(positive) > 0:
            explanation += "\n  Improved: "
            improvements = []
            for goal in positive:
                display = self.goal_display_names.get(goal, goal.title())
                delta = goal_changes[goal]
                improvements.append(f"{display} (+{delta:.2f})")
            explanation += ", ".join(improvements)

        if len(negative) > 0:
            explanation += "\n  Degraded: "
            degradations = []
            for goal in negative:
                display = self.goal_display_names.get(goal, goal.title())
                delta = goal_changes[goal]
                degradations.append(f"{display} ({delta:.2f})")
            explanation += ", ".join(degradations)

        return explanation

    def _generate_summary(self, achieved: list[str], failed: list[str], trajectories: dict[str, GoalTrajectory]) -> str:
        """Generiert high-level summary."""
        total = len(achieved) + len(failed)

        summary = f"Musical Goals Achievement: {len(achieved)}/{total} goals achieved\n\n"

        if len(achieved) == total:
            summary += "🎉 SUCCESS! All Musical Goals achieved!\n"
        elif len(achieved) > total / 2:
            summary += f"⚠️ PARTIAL SUCCESS: {len(failed)} goal(s) not achieved\n"
        else:
            summary += f"❌ FAILURE: {len(failed)} goal(s) not achieved\n"

        summary += "\nAchieved Goals:\n"
        for goal_name in achieved:
            display = self.goal_display_names.get(goal_name, goal_name.title())
            traj = trajectories[goal_name]
            summary += f"  ✅ {display}: {traj.final_score:.2f} (target: {traj.target_score:.2f})\n"

        if len(failed) > 0:
            summary += "\nFailed Goals:\n"
            for goal_name in failed:
                display = self.goal_display_names.get(goal_name, goal_name.title())
                traj = trajectories[goal_name]
                gap = traj.target_score - traj.final_score
                summary += f"  ❌ {display}: {traj.final_score:.2f} (target: {traj.target_score:.2f}, gap: {gap:.2f})\n"

        return summary

    def _generate_recommendations(
        self, failed: list[str], trajectories: dict[str, GoalTrajectory], step_impacts: list[ProcessingStepImpact]
    ) -> list[str]:
        """Generiert actionable recommendations."""
        recommendations = []

        if len(failed) == 0:
            recommendations.append("All goals achieved! No changes needed.")
            return recommendations

        # Analyze each failed goal
        for goal_name in failed:
            display = self.goal_display_names.get(goal_name, goal_name.title())
            traj = trajectories[goal_name]
            gap = traj.target_score - traj.final_score

            # Find steps that degraded this goal
            degrading_steps = []
            for impact in step_impacts:
                if goal_name in impact.negative_impacts:
                    degrading_steps.append(impact.step_name)

            # Find steps that improved this goal (could be strengthened)
            improving_steps = []
            for impact in step_impacts:
                if goal_name in impact.positive_impacts:
                    improving_steps.append((impact.step_name, impact.goal_changes[goal_name]))

            # Generate recommendation
            if len(degrading_steps) > 0:
                steps_str = ", ".join(degrading_steps)
                recommendations.append(
                    f"{display}: Consider reducing strength of {steps_str} (degraded by {abs(sum(s.goal_changes[goal_name] for s in step_impacts if goal_name in s.negative_impacts)):.2f})"
                )

            if len(improving_steps) > 0:
                # Sort by impact
                improving_steps.sort(key=lambda x: x[1], reverse=True)
                best_step, best_delta = improving_steps[0]
                recommendations.append(
                    f"{display}: Consider increasing strength of {best_step} (currently improved by {best_delta:.2f}, need {gap:.2f} more)"
                )

            # Goal-specific recommendations
            if goal_name == "brillanz":
                recommendations.append(
                    f"{display}: Try adding high-frequency enhancement or reducing low-pass filtering"
                )
            elif goal_name == "waerme":
                recommendations.append(f"{display}: Try adding mid-frequency warmth or reducing excessive denoising")
            elif goal_name in ["bass_kraft", "bass-kraft"]:
                recommendations.append(f"{display}: Try bass enhancement or reducing high-pass filtering")
            elif goal_name == "transparenz":
                recommendations.append(f"{display}: Try reducing reverb or improving spectral clarity")

        # General recommendations
        if len(failed) > 3:
            recommendations.append("Consider switching to a more conservative processing mode (FORENSIC or ARCHIVAL)")

        return recommendations

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def explain_simple(
        self, original: np.ndarray, processed: np.ndarray, sr: int, mode: ProcessingMode = ProcessingMode.RESTORATION
    ) -> str:
        """
        Einfaches one-shot explanation without step tracking.

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate
            mode: Processing mode

        Returns:
            Simple text explanation
        """
        # Measure goals
        initial_scores = self.checker.measure_all(original, sr)
        final_scores = self.checker.measure_all(processed, sr, reference=original)

        mode_config = (
            PROCESSING_MODE_CONFIGS[mode]
            if mode in PROCESSING_MODE_CONFIGS
            else PROCESSING_MODE_CONFIGS[ProcessingMode.RESTORATION]
        )
        targets = mode_config.musical_goals

        # Build explanation
        explanation = f"Musical Goals Analysis ({mode.value} mode):\n\n"

        achieved = []
        failed = []

        for goal_name in initial_scores:
            display = self.goal_display_names.get(goal_name, goal_name.title())
            initial = initial_scores[goal_name]
            final = final_scores[goal_name]
            target = targets.get(goal_name, 0.85)
            delta = final - initial

            status = "✅" if final >= target else "❌"
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"

            explanation += f"{status} {display}: {initial:.2f} → {final:.2f} ({delta_str})\n"
            explanation += f"   Target: {target:.2f}\n"

            if final >= target:
                achieved.append(goal_name)
            else:
                failed.append(goal_name)

        explanation += f"\nResult: {len(achieved)}/{len(achieved) + len(failed)} goals achieved\n"

        if len(failed) > 0:
            explanation += "\nFailed goals: " + ", ".join(
                [str(self.goal_display_names.get(g) or g.title()) for g in failed]
            )

        return explanation


if __name__ == "__main__":
    # Example usage
    logger.debug("=== GoalExplainer Example ===\n")

    # Create test signal
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Original audio
    original = (
        0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 2000 * t)
    )

    # Simulate processing steps
    after_denoise = original + np.random.normal(0, 0.01, len(original))  # Slight noise
    after_eq = after_denoise * 1.1  # Boost
    final = after_eq * 0.95  # Slight limiting

    # Initialize explainer
    explainer = GoalExplainer()

    # Track through processing
    logger.debug("Starting goal tracking...")
    explainer.start_tracking(original, sr, mode=ProcessingMode.RESTORATION)
    explainer.record_step("Denoise", after_denoise, sr)
    explainer.record_step("EQ", after_eq, sr)
    explainer.record_step("Limiter", final, sr)

    # Generate explanation
    logger.debug("\nGenerating explanation...")
    explanation = explainer.generate_explanation()

    logger.debug("\n" + "=" * 60)
    logger.debug(explanation.summary)
    logger.debug("=" * 60)

    logger.debug("\nRecommendations:")
    for rec in explanation.recommendations:
        logger.debug("  • %s", rec)

    logger.debug("\n" + "=" * 60)

    # Detailed trajectory example
    logger.debug("\nDetailed Goal Trajectories:")
    for goal_name, traj in explanation.goal_trajectories.items():
        logger.debug("\n%s", traj.explanation)

    logger.debug("\n=== Test vollstaendig ===")
