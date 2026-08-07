"""
Processing Modes for User Control over Musical Goals Priorities

Component 4.2: Processing Modes - Magic Button Edition
Impact: +2.0 Punkte - User Control über Ästhetik-Prioritäten

Provides 2 distinct Magic Button modes with different Musical Goals priorities.
FORENSIC ist kein Mode, sondern eine fest integrierte Pipeline-Komponente.

Nur 2 user-wählbare Modi:
- RESTORATION: Balanced restoration, authenticity preservation
- STUDIO_2026: Modern production, maximum brilliance

Version: 2.0 (Magic Button Edition)
Date: 2026-02-13
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Single source of truth: ProcessingMode-Enum kommt aus core.processing_modes.
# Alle Imports von backend.core.musical_goals.processing_modes.ProcessingMode
# bleiben gültig, da das Enum hier re-exportiert wird.
from backend.core.processing_modes import ProcessingMode


@dataclass
class ProcessingModeConfig:
    """
    Configuration for a specific processing mode.

    Attributes:
        mode: ProcessingMode enum value
        name: Human-readable name
        description: What this mode optimizes for
        musical_goals: Target values for each Musical Goal (0.7-1.0)
        goal_weights: Importance weights for goal prioritization
        processing_params: Default processing parameters
        quality_thresholds: Minimum acceptable Musical Goals scores
    """

    mode: ProcessingMode
    name: str
    description: str
    musical_goals: dict[str, float]
    goal_weights: dict[str, float]
    processing_params: dict[str, Any]
    quality_thresholds: dict[str, float]

    def get_prioritized_goals(self) -> list[tuple[str, float]]:
        """Gibt goals sorted by weight (highest first) zurück."""
        return sorted(self.goal_weights.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# Processing Mode Configurations
# =============================================================================

PROCESSING_MODE_CONFIGS: dict[ProcessingMode, ProcessingModeConfig] = {
    # =========================================================================
    # RESTORATION MODE
    # =========================================================================
    ProcessingMode.RESTORATION: ProcessingModeConfig(
        mode=ProcessingMode.RESTORATION,
        name="Restoration",
        description=(
            "Balanced restoration with maximum defect removal and neutral aesthetics. "
            "Suitable for general-purpose audio restoration where the goal is to "
            "remove defects while maintaining original character."
        ),
        # Musical Goals: Balanced across all 7 goals
        musical_goals={
            "brillanz": 0.85,  # Clear highs
            "waerme": 0.85,  # Balanced warmth
            "natuerlichkeit": 0.88,  # Natural timbre
            "authentizitaet": 0.88,  # Authentic to original
            "emotionalitaet": 0.85,  # Preserve emotion
            "transparenz": 0.88,  # Clean separation
            "bass-kraft": 0.85,  # Solid low-end
        },
        # Goal Weights: All relatively equal, slight preference for authenticity
        goal_weights={
            "natuerlichkeit": 1.0,
            "authentizitaet": 1.0,
            "transparenz": 0.95,
            "brillanz": 0.90,
            "waerme": 0.90,
            "emotionalitaet": 0.90,
            "bass-kraft": 0.85,
        },
        # Processing Parameters
        processing_params={
            "denoise_strength": 0.65,
            "declick_threshold": 0.70,
            "dehum_intensity": 0.75,
            "eq_adjustment": "subtle",
            "compression_ratio": 2.0,
            "reverb_preservation": 0.85,
            "transient_preservation": 0.90,
        },
        # Quality Thresholds: Moderate requirements
        quality_thresholds={
            "brillanz": 0.75,
            "waerme": 0.75,
            "natuerlichkeit": 0.80,
            "authentizitaet": 0.80,
            "emotionalitaet": 0.75,
            "transparenz": 0.80,
            "bass-kraft": 0.75,
        },
    ),
    # =========================================================================
    # STUDIO_2026 MODE
    # =========================================================================
    ProcessingMode.STUDIO_2026: ProcessingModeConfig(
        mode=ProcessingMode.STUDIO_2026,
        name="Studio 2026",
        description=(
            "Modern studio production sound with aggressive optimization. "
            "Prioritizes clarity, brilliance, and transparency. Suitable for "
            "commercial releases that need competitive loudness and polish."
        ),
        # Musical Goals: High brilliance, clarity, bass
        musical_goals={
            "brillanz": 0.95,  # Maximum clarity
            "waerme": 0.80,  # Less critical
            "natuerlichkeit": 0.82,  # Can sacrifice some
            "authentizitaet": 0.80,  # Can sacrifice some
            "emotionalitaet": 0.88,  # Still important
            "transparenz": 0.95,  # Maximum separation
            "bass-kraft": 0.92,  # Solid modern bass
        },
        # Goal Weights: Brilliance + Transparency >> Warmth/Authenticity
        goal_weights={
            "brillanz": 1.0,
            "transparenz": 1.0,
            "bass-kraft": 0.95,
            "emotionalitaet": 0.90,
            "waerme": 0.70,
            "natuerlichkeit": 0.70,
            "authentizitaet": 0.65,
        },
        # Processing Parameters: Aggressive
        processing_params={
            "denoise_strength": 0.80,
            "declick_threshold": 0.75,
            "dehum_intensity": 0.85,
            "eq_adjustment": "modern",
            "compression_ratio": 3.0,
            "reverb_preservation": 0.70,
            "transient_preservation": 0.95,
            "saturation": 0.15,
            "stereo_width": 1.15,
        },
        # Quality Thresholds: High for priority goals
        quality_thresholds={
            "brillanz": 0.88,
            "waerme": 0.70,
            "natuerlichkeit": 0.75,
            "authentizitaet": 0.75,
            "emotionalitaet": 0.80,
            "transparenz": 0.88,
            "bass-kraft": 0.85,
        },
    ),
}


# =============================================================================
# Mode Selection & Management
# =============================================================================


class ProcessingModeManager:
    """
    Manages processing mode selection and configuration.

    Provides methods to select a mode, get its configuration,
    and validate Musical Goals against mode-specific thresholds.
    """

    def __init__(self, mode: ProcessingMode = ProcessingMode.RESTORATION):
        """
        Initialisiert with a default processing mode.

        Args:
            mode: Initial processing mode
        """
        self.current_mode = mode
        self.config = PROCESSING_MODE_CONFIGS[mode]
        logger.info("ProcessingModeManager initialisiert with Betriebsart: %s", mode.value)

    def set_mode(self, mode: ProcessingMode) -> None:
        """Change processing mode."""
        logger.info("Switching from %s → %s", self.current_mode.value, mode.value)
        self.current_mode = mode
        self.config = PROCESSING_MODE_CONFIGS[mode]

    def get_config(self, mode: ProcessingMode | None = None) -> ProcessingModeConfig:
        """Gibt zurück: configuration for specified mode (or current if None)."""
        target_mode = mode if mode is not None else self.current_mode
        return PROCESSING_MODE_CONFIGS[target_mode]

    def get_musical_goals_for_mode(self, mode: ProcessingMode | None = None) -> dict[str, float]:
        """Gibt zurück: target Musical Goals for specified mode."""
        config = self.get_config(mode)
        return config.musical_goals.copy()

    def get_processing_params_for_mode(self, mode: ProcessingMode | None = None) -> dict[str, Any]:
        """Gibt zurück: processing parameters for specified mode."""
        config = self.get_config(mode)
        return config.processing_params.copy()

    def validate_goals_against_mode(
        self, achieved_goals: dict[str, float], mode: ProcessingMode | None = None
    ) -> dict[str, Any]:
        """
        Validiert achieved goal scores against mode thresholds.

        Args:
            achieved_goals: Measured Musical Goals scores
            mode: Mode to validate against (or current if None)

        Returns:
            Dict with validation results:
            {
                'passed': bool,
                'violations': List[str],
                'scores': Dict[goal_name, dict],  # target, achieved, passed
                'overall_score': float
            }
        """
        config = self.get_config(mode)
        thresholds = config.quality_thresholds

        violations = []
        scores = {}
        total_weighted_diff = 0.0
        total_weight = 0.0

        for goal_name, threshold in thresholds.items():
            achieved = achieved_goals.get(goal_name, 0.0)
            target = config.musical_goals[goal_name]
            weight = config.goal_weights[goal_name]
            passed = achieved >= threshold

            if not passed:
                violations.append(f"{goal_name}: {achieved:.2f} < {threshold:.2f} (target: {target:.2f})")

            scores[goal_name] = {
                "target": target,
                "achieved": achieved,
                "threshold": threshold,
                "passed": passed,
                "weight": weight,
                "diff": achieved - target,
            }

            # Weighted score
            total_weighted_diff += weight * (achieved - target)
            total_weight += weight

        overall_score = 1.0 + (total_weighted_diff / total_weight) if total_weight > 0 else 1.0
        overall_score = max(0.0, min(1.0, overall_score))

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "scores": scores,
            "overall_score": overall_score,
            "mode": config.mode.value,
            "mode_name": config.name,
        }

    def compare_modes(self, achieved_goals: dict[str, float]) -> list[dict[str, Any]]:
        """
        Compare achieved goals against all processing modes.

        Useful for recommending the best mode for given input material.

        Args:
            achieved_goals: Measured Musical Goals scores

        Returns:
            List of validation results for each mode, sorted by overall_score
        """
        results = []

        for mode in ProcessingMode:
            validation = self.validate_goals_against_mode(achieved_goals, mode)
            results.append(validation)

        # Sort by overall_score (best first)
        results.sort(key=lambda x: x["overall_score"], reverse=True)

        return results

    @staticmethod
    def get_mode_summary() -> str:
        """Gibt human-readable summary of all modes zurück."""
        lines = ["Available Processing Modes:\n"]

        for mode in ProcessingMode:
            config = PROCESSING_MODE_CONFIGS[mode]
            top_goals = config.get_prioritized_goals()[:3]
            goals_str = ", ".join([f"{g}: {w:.2f}" for g, w in top_goals])

            lines.append(f"\n{mode.value.upper():20} | {config.name}")
            lines.append(f"{'':20} | {config.description[:60]}...")
            lines.append(f"{'':20} | Top priorities: {goals_str}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_mode_from_string(mode_str: str) -> ProcessingMode:
    """
    Konvertiert string to ProcessingMode enum.

    Args:
        mode_str: Mode name (case-insensitive)

    Returns:
        ProcessingMode enum value

    Raises:
        ValueError: If mode string is invalid
    """
    mode_str = mode_str.lower().strip()

    for mode in ProcessingMode:
        if mode.value == mode_str:
            return mode

    raise ValueError(f"Unknown processing mode: '{mode_str}'. Valid modes: {[m.value for m in ProcessingMode]}")


def get_recommended_mode_for_medium(medium: str) -> ProcessingMode:
    """
    Recommend processing mode based on source medium.

    Args:
        medium: Source medium (e.g., 'vinyl', 'tape', 'digital', etc.)

    Returns:
        Recommended ProcessingMode
    """
    medium = medium.lower()

    recommendations = {
        "vinyl": ProcessingMode.RESTORATION,
        "tape": ProcessingMode.RESTORATION,
        "cassette": ProcessingMode.RESTORATION,
        "shellac": ProcessingMode.RESTORATION,
        "acetate": ProcessingMode.RESTORATION,
        "78rpm": ProcessingMode.RESTORATION,
        "digital": ProcessingMode.STUDIO_2026,
        "cd": ProcessingMode.STUDIO_2026,
        "dat": ProcessingMode.STUDIO_2026,
        "mp3": ProcessingMode.RESTORATION,
        "speech": ProcessingMode.RESTORATION,
        "interview": ProcessingMode.RESTORATION,
        "field_recording": ProcessingMode.RESTORATION,
    }

    return recommendations.get(medium, ProcessingMode.RESTORATION)


def print_mode_comparison(achieved_goals: dict[str, float]) -> None:
    """
    Gibt aus: comparison of all modes against achieved goals.

    Args:
        achieved_goals: Measured Musical Goals scores
    """
    manager = ProcessingModeManager()
    results = manager.compare_modes(achieved_goals)

    logger.debug("\n" + "=" * 80)
    logger.debug("PROCESSING Betriebsart COMPARISON")
    logger.debug("=" * 80)

    for i, result in enumerate(results, 1):
        passed_str = "✅ PASS" if result["passed"] else "❌ FAIL"
        logger.debug("\n%s. %s | Wert: %.2f | %s", i, result["mode_name"], result["overall_score"], passed_str)

        if result["violations"]:
            logger.debug("   Violations: %s", len(result["violations"]))
            for violation in result["violations"][:3]:
                logger.debug("   - %s", violation)
