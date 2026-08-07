"""
Reference-Based Learning for Musical Goals

Component 0.9.8: Reference-Based Learning
Impact: +0.5 Punkte - Personalization & Continuous Improvement

Enables learning user preferences from reference tracks and A/B test feedback.

Key Features:
- Analyze reference tracks to extract Musical Goals
- Learn from user A/B test choices
- Build personalized goal profiles
- Continuous improvement from feedback
- Confidence-weighted learning

Architecture:
    Reference Track → Goal Analysis → User Preference Profile
    A/B Tests → User Feedback → Profile Update
    Profile → Personalized Goal Targets
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Learning strategy for preference updates"""

    CONSERVATIVE = "conservative"  # Slow learning, high stability
    BALANCED = "balanced"  # Medium learning rate
    AGGRESSIVE = "aggressive"  # Fast learning, less stable


@dataclass
class ReferenceTrack:
    """
    Reference track with analyzed Musical Goals.

    Attributes:
        audio_path: Path to reference audio
        analyzed_goals: Extracted goal values
        metadata: Track info (title, artist, genre, etc.)
        confidence: Analysis confidence (0-1)
        timestamp: When analyzed
    """

    audio_path: str
    analyzed_goals: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serialisiert to dict."""
        return {
            "audio_path": self.audio_path,
            "analyzed_goals": self.analyzed_goals,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceTrack":
        """Deserialize from dict"""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data_copy)


@dataclass
class ABTestResult:
    """
    A/B test result with user choice.

    Attributes:
        variant_a_goals: Goal values for variant A
        variant_b_goals: Goal values for variant B
        user_choice: "A" or "B" (which user preferred)
        audio_id: ID of audio being tested
        confidence: User confidence in choice (0-1)
        timestamp: When test was performed
        context: Additional context
    """

    variant_a_goals: dict[str, float]
    variant_b_goals: dict[str, float]
    user_choice: str  # "A" or "B"
    audio_id: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)

    def get_preferred_goals(self) -> dict[str, float]:
        """Gibt zurück: goals of preferred variant."""
        if self.user_choice == "A":
            return self.variant_a_goals
        else:
            return self.variant_b_goals

    def get_rejected_goals(self) -> dict[str, float]:
        """Gibt zurück: goals of rejected variant."""
        if self.user_choice == "A":
            return self.variant_b_goals
        else:
            return self.variant_a_goals

    def to_dict(self) -> dict:
        """Serialisiert to dict."""
        return {
            "variant_a_goals": self.variant_a_goals,
            "variant_b_goals": self.variant_b_goals,
            "user_choice": self.user_choice,
            "audio_id": self.audio_id,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ABTestResult":
        """Deserialize from dict"""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data_copy)


@dataclass
class UserPreferenceProfile:
    """
    User's learned preference profile.

    Attributes:
        user_id: User identifier
        learned_goals: Learned target goal values
        goal_weights: Importance weights for each goal
        confidence: Overall confidence in profile (0-1)
        n_references: Number of reference tracks analyzed
        n_ab_tests: Number of A/B tests completed
        learning_history: History of updates
        created_at: When profile was created
        updated_at: Last update time
    """

    user_id: str
    learned_goals: dict[str, float]
    goal_weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    n_references: int = 0
    n_ab_tests: int = 0
    learning_history: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_weighted_goals(self) -> dict[str, float]:
        """
        Gibt zurück: goals weighted by importance.

        Returns goals scaled by their learned importance weights.
        """
        if not self.goal_weights:
            return self.learned_goals.copy()

        weighted = {}
        for goal_name, value in self.learned_goals.items():
            weight = self.goal_weights.get(goal_name, 1.0)
            weighted[goal_name] = value * weight

        return weighted

    def is_reliable(self, min_samples: int = 5, min_confidence: float = 0.60) -> bool:
        """
        Prüft if profile is reliable enough to use.

        Args:
            min_samples: Minimum number of samples (references + AB tests)
            min_confidence: Minimum confidence threshold

        Returns:
            True if profile is reliable
        """
        total_samples = self.n_references + self.n_ab_tests
        return total_samples >= min_samples and self.confidence >= min_confidence

    def to_dict(self) -> dict:
        """Serialisiert to dict."""
        return {
            "user_id": self.user_id,
            "learned_goals": self.learned_goals,
            "goal_weights": self.goal_weights,
            "confidence": self.confidence,
            "n_references": self.n_references,
            "n_ab_tests": self.n_ab_tests,
            "learning_history": self.learning_history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserPreferenceProfile":
        """Deserialize from dict"""
        data_copy = data.copy()
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        data_copy["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data_copy)


class ReferenceLearner:
    """
    Learns user preferences from reference tracks and A/B tests.

    Implements continuous learning with confidence-weighted updates.

    Usage:
        learner = ReferenceLearner(
            user_id="user_123",
            strategy=LearningStrategy.BALANCED
        )

        # Analyze reference track
        reference = learner.analyze_reference_track(
            audio,
            sr,
            metadata={"title": "My Favorite Song"}
        )

        # Learn from A/B test
        ab_result = ABTestResult(
            variant_a_goals={"bass-kraft": 0.85, ...},
            variant_b_goals={"bass-kraft": 0.75, ...},
            user_choice="A",
            audio_id="track_456"
        )
        learner.learn_from_ab_test(ab_result)

        # Get personalized goals
        personalized = learner.adapt_goals_to_preference(base_goals)
    """

    def __init__(
        self,
        user_id: str,
        strategy: LearningStrategy = LearningStrategy.BALANCED,
        goal_names: list[str] | None = None,
        base_goals: dict[str, float] | None = None,
        profile_path: str | None = None,
    ) -> None:
        """
        Initialisiert reference learner.

        Args:
            user_id: User identifier
            strategy: Learning strategy (conservative/balanced/aggressive)
            goal_names: List of goal names to learn
            base_goals: Default goal values
            profile_path: Path to save/load profile
        """
        self.user_id = user_id
        self.strategy = strategy

        # Default goal names
        if goal_names is None:
            goal_names = [
                "bass-kraft",
                "brillanz",
                "transparenz",
                "natürlichkeit",
                "emotionalität",
                "wärme",
                "authentizität",
            ]
        self.goal_names = goal_names

        # Default base goals
        if base_goals is None:
            base_goals = dict.fromkeys(goal_names, 0.85)
        self.base_goals = base_goals

        self.profile_path = profile_path

        # Try to load existing profile
        if profile_path and Path(profile_path).exists():
            self.profile = self._load_profile(profile_path)
            logger.info("geladen existing Profil for %s", user_id)
        else:
            # Create new profile
            self.profile = UserPreferenceProfile(
                user_id=user_id, learned_goals=base_goals.copy(), goal_weights=dict.fromkeys(goal_names, 1.0)
            )
            logger.info("erstellt new Profil for %s", user_id)

        # Learning rate based on strategy
        self.learning_rates = {
            LearningStrategy.CONSERVATIVE: 0.05,
            LearningStrategy.BALANCED: 0.15,
            LearningStrategy.AGGRESSIVE: 0.30,
        }

        logger.info(
            f"ReferenceLearner initialisiert: user={user_id}, "
            f"strategy={strategy.value}, "
            f"samples={self.profile.n_references + self.profile.n_ab_tests}"
        )

    def get_learning_rate(self) -> float:
        """Gibt zurück: learning rate for current strategy."""
        return self.learning_rates[self.strategy]

    def analyze_reference_track(
        self,
        audio: np.ndarray,
        sr: int,
        goals_calculator: Any,  # Musical goals calculator
        metadata: dict | None = None,
        analysis_confidence: float = 1.0,
    ) -> ReferenceTrack:
        """
        Analysiert die Referenzspur zur Extraktion der Musical Goals.

        Args:
            audio: Audio data
            sr: Sample rate
            goals_calculator: Calculator with calculate_all_goals(audio, sr) method
            metadata: Track metadata
            analysis_confidence: Confidence in analysis (0-1)

        Returns:
            ReferenceTrack with analyzed goals
        """
        # Calculate goals for reference track
        analyzed_goals = goals_calculator.calculate_all_goals(audio, sr)

        reference = ReferenceTrack(
            audio_path="reference",  # Would be actual path in production
            analyzed_goals=analyzed_goals,
            metadata=metadata or {},
            confidence=analysis_confidence,
        )

        # Update profile with reference
        self._update_from_reference(reference)

        logger.info("Analyzed Referenz track: %s goals, confidence=%.2f", len(analyzed_goals), analysis_confidence)

        return reference

    def _update_from_reference(self, reference: ReferenceTrack) -> None:
        """Aktualisiert preference profile from reference track."""
        learning_rate = self.get_learning_rate()

        # Weight learning rate by analysis confidence
        effective_lr = learning_rate * reference.confidence

        # Update each goal with weighted average
        for goal_name in self.goal_names:
            if goal_name in reference.analyzed_goals:
                current_value = self.profile.learned_goals[goal_name]
                reference_value = reference.analyzed_goals[goal_name]

                # Exponential moving average
                new_value = (1 - effective_lr) * current_value + effective_lr * reference_value

                self.profile.learned_goals[goal_name] = float(new_value)

        # Update metadata
        self.profile.n_references += 1
        self.profile.updated_at = datetime.now()

        # Update confidence (more samples = higher confidence)
        total_samples = self.profile.n_references + self.profile.n_ab_tests
        self.profile.confidence = min(1.0, total_samples / 20.0)  # Saturates at 20 samples

        # Log update
        self.profile.learning_history.append(
            {
                "type": "reference",
                "timestamp": datetime.now().isoformat(),
                "confidence": reference.confidence,
                "goals": reference.analyzed_goals,
            }
        )

        # Save if path specified
        if self.profile_path:
            self._save_profile(self.profile_path)

    def learn_from_ab_test(self, ab_result: ABTestResult) -> None:
        """
        Learn from A/B test result.

        Args:
            ab_result: A/B test result with user choice
        """
        learning_rate = self.get_learning_rate()

        # Weight learning rate by user confidence
        effective_lr = learning_rate * ab_result.confidence

        preferred_goals = ab_result.get_preferred_goals()
        rejected_goals = ab_result.get_rejected_goals()

        # Update each goal
        for goal_name in self.goal_names:
            if goal_name not in preferred_goals or goal_name not in rejected_goals:
                continue

            current_value = self.profile.learned_goals[goal_name]
            preferred_value = preferred_goals[goal_name]
            rejected_value = rejected_goals[goal_name]

            # Move towards preferred, away from rejected
            # Difference indicates importance
            difference = abs(preferred_value - rejected_value)

            if difference > 0.01:  # Significant difference
                # Update goal value
                new_value = (1 - effective_lr) * current_value + effective_lr * preferred_value
                self.profile.learned_goals[goal_name] = float(new_value)

                # Update goal weight (importance)
                current_weight = self.profile.goal_weights.get(goal_name, 1.0)
                # Goals with larger differences are more important to user
                importance_boost = min(0.1, difference * 0.5)
                new_weight = min(2.0, current_weight + importance_boost)
                self.profile.goal_weights[goal_name] = float(new_weight)

        # Update metadata
        self.profile.n_ab_tests += 1
        self.profile.updated_at = datetime.now()

        # Update confidence
        total_samples = self.profile.n_references + self.profile.n_ab_tests
        self.profile.confidence = min(1.0, total_samples / 20.0)

        # Log update
        self.profile.learning_history.append(
            {
                "type": "ab_test",
                "timestamp": datetime.now().isoformat(),
                "confidence": ab_result.confidence,
                "choice": ab_result.user_choice,
                "preferred_goals": preferred_goals,
                "rejected_goals": rejected_goals,
            }
        )

        # Save if path specified
        if self.profile_path:
            self._save_profile(self.profile_path)

        logger.info(
            f"Learned from A/B test: choice={ab_result.user_choice}, "
            f"confidence={ab_result.confidence:.2f}, "
            f"Profil_confidence={self.profile.confidence:.2f}"
        )

    def adapt_goals_to_preference(
        self, base_goals: dict[str, float], adaptation_strength: float = 0.5
    ) -> dict[str, float]:
        """
        Adapt base goals to user preferences.

        Args:
            base_goals: Base goal values
            adaptation_strength: How much to adapt (0-1)
                0 = no adaptation (use base)
                1 = full adaptation (use learned)

        Returns:
            Adapted goal values
        """
        # Scale adaptation by profile confidence
        effective_strength = adaptation_strength * self.profile.confidence

        adapted = {}
        for goal_name in self.goal_names:
            base_value = base_goals.get(goal_name, 0.85)
            learned_value = self.profile.learned_goals.get(goal_name, base_value)

            # Weighted average
            adapted_value = (1 - effective_strength) * base_value + effective_strength * learned_value

            # Apply importance weight
            weight = self.profile.goal_weights.get(goal_name, 1.0)
            # Weight > 1.0 means important to user, boost it slightly
            if weight > 1.0:
                boost = min(0.05, (weight - 1.0) * 0.05)
                adapted_value = min(1.0, adapted_value + boost)

            adapted[goal_name] = float(np.clip(adapted_value, 0.7, 1.0))

        logger.debug("Adapted goals: strength=%.2f, confidence=%.2f", effective_strength, self.profile.confidence)

        return adapted

    def get_confidence(self) -> float:
        """Gibt zurück: confidence in learned preferences."""
        return self.profile.confidence

    def get_goal_importances(self) -> dict[str, float]:
        """
        Gibt zurück: learned importance of each goal.

        Returns goal weights normalized to 0-1 range.
        """
        if not self.profile.goal_weights:
            return dict.fromkeys(self.goal_names, 1.0)

        weights = self.profile.goal_weights
        max_weight = max(weights.values())

        # Normalize to 0-1
        if max_weight > 0:
            normalized = {name: weight / max_weight for name, weight in weights.items()}
        else:
            normalized = dict.fromkeys(self.goal_names, 1.0)

        return normalized

    def reset_profile(self):
        """Setzt zurück: profile to base goals (start learning fresh)."""
        self.profile = UserPreferenceProfile(
            user_id=self.user_id,
            learned_goals=self.base_goals.copy(),
            goal_weights=dict.fromkeys(self.goal_names, 1.0),
        )

        if self.profile_path:
            self._save_profile(self.profile_path)

        logger.info("zurueckgesetzt Profil for %s", self.user_id)

    def _save_profile(self, path: str):
        """Speichert profile to file."""
        try:
            with open(path, "w") as f:
                json.dump(self.profile.to_dict(), f, indent=2)
            logger.debug("gespeichert Profil to %s", path)
        except Exception as e:
            logger.error("konnte nicht speichern Profil: %s", e)

    def _load_profile(self, path: str) -> UserPreferenceProfile:
        """Lädt profile from file."""
        try:
            with open(path) as f:
                data = json.load(f)
            return UserPreferenceProfile.from_dict(data)
        except Exception as e:
            logger.error("konnte nicht laden Profil: %s", e)
            # Return new profile on error
            return UserPreferenceProfile(
                user_id=self.user_id,
                learned_goals=self.base_goals.copy(),
                goal_weights=dict.fromkeys(self.goal_names, 1.0),
            )


# Convenience functions


def quick_preference_adaptation(
    base_goals: dict[str, float], reference_goals: dict[str, float], adaptation_strength: float = 0.3
) -> dict[str, float]:
    """
    Quick adaptation of base goals towards reference.

    Args:
        base_goals: Current goal values
        reference_goals: Reference goal values
        adaptation_strength: How much to adapt (0-1)

    Returns:
        Adapted goals
    """
    adapted = {}
    for goal_name, base_value in base_goals.items():
        if goal_name in reference_goals:
            reference_value = reference_goals[goal_name]
            adapted_value = (1 - adaptation_strength) * base_value + adaptation_strength * reference_value
            adapted[goal_name] = float(np.clip(adapted_value, 0.7, 1.0))
        else:
            adapted[goal_name] = base_value

    return adapted


def compare_variants(variant_a: dict[str, float], variant_b: dict[str, float]) -> dict[str, float]:
    """
    Compare two goal variants to help user decide.

    Returns differences (B - A) for each goal.
    Positive = B is higher, Negative = A is higher
    """
    differences = {}
    for goal_name in variant_a:
        if goal_name in variant_b:
            diff = variant_b[goal_name] - variant_a[goal_name]
            differences[goal_name] = float(diff)

    return differences


def get_learning_summary(profile: UserPreferenceProfile) -> str:
    """
    Gibt zurück: human-readable summary of learning progress.

    Args:
        profile: User preference profile

    Returns:
        Summary string
    """
    lines = [
        f"User {profile.user_id} Preference Profile:",
        "=" * 50,
        f"Confidence: {profile.confidence:.2f}",
        f"References: {profile.n_references}",
        f"A/B Tests: {profile.n_ab_tests}",
        f"Total Samples: {profile.n_references + profile.n_ab_tests}",
        f"Reliable: {'Yes' if profile.is_reliable() else 'No'}",
        "",
        "Learned Goals:",
    ]

    for goal_name, value in profile.learned_goals.items():
        weight = profile.goal_weights.get(goal_name, 1.0)
        importance = "⭐" * int(weight) if weight > 1 else ""
        lines.append(f"  {goal_name:20s}: {value:.3f} {importance}")

    return "\n".join(lines)
