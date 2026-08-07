"""
AURIK v8 Rollback Mechanism: Snapshot Management & Rollback
============================================================

Provides undo/rollback functionality for audio processing pipeline.

Key capabilities:
1. Snapshot: Save audio state before critical operations
2. Rollback: Restore previous state if violations detected
3. History: Maintain history of snapshots for multi-level rollback
4. Audit: Track all rollback decisions

Architecture:
- Before each processing step: create_snapshot()
- After processing: validate results
- If violations: rollback_to_snapshot()
- Maintain max N snapshots (configurable, default 5)

Quelle: Finalisierungs_Roadmap.md - Component 0.6
Autor: AI Team
Datum: 8. Februar 2026
"""

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSnapshot:
    """Snapshot of audio state at a point in time."""

    name: str
    timestamp: datetime
    audio: np.ndarray  # Audio data
    sr: int  # Sample rate
    musical_goals: dict[str, float]  # Musical goals scores at snapshot time
    metadata: dict = field(default_factory=dict)  # Additional metadata


@dataclass
class RollbackDecision:
    """Decision to rollback."""

    rolled_back: bool
    from_snapshot: str | None
    to_snapshot: str | None
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class RollbackManager:
    """
    Manages audio snapshots and rollback functionality.

    Example:
        >>> from backend.core.musical_goals import MusicalGoalsChecker
        >>>
        >>> manager = RollbackManager(max_snapshots=5)
        >>> checker = MusicalGoalsChecker()
        >>>
        >>> # Create initial snapshot
        >>> original_goals = checker.measure_all(audio, sr=48000)
        >>> manager.create_snapshot('original', audio, sr=48000, original_goals)
        >>>
        >>> # Process audio...
        >>> processed_audio = some_processing(audio)
        >>>
        >>> # Before committing, create checkpoint
        >>> processed_goals = checker.measure_all(processed_audio, sr=48000)
        >>> manager.create_snapshot('after_processing', processed_audio, sr=48000, processed_goals)
        >>>
        >>> # Check for violations
        >>> if has_violations(processed_goals):
        >>>     # Rollback to original
        >>>     audio, sr, goals = manager.rollback_to_snapshot('original')
        logger.debug("Rolled back: %s", goals)
    """

    def __init__(self, max_snapshots: int = 5):
        """
        Initialisiert Rollback Manager.

        Args:
            max_snapshots: Maximum number of snapshots to keep (default: 5)
        """
        self.max_snapshots = max_snapshots
        self.snapshots: list[AudioSnapshot] = []
        self.rollback_history: list[RollbackDecision] = []

    def create_snapshot(
        self, name: str, audio: np.ndarray, sr: int, musical_goals: dict[str, float], metadata: dict | None = None
    ):
        """
        Erstellt a snapshot of current audio state.

        Args:
            name: Snapshot name (e.g., 'original', 'after_noise_reduction')
            audio: Audio data
            sr: Sample rate
            musical_goals: Musical goals scores at this point
            metadata: Optional additional metadata
        """
        # Create deep copy to prevent unwanted modifications
        audio_copy = copy.deepcopy(audio)
        goals_copy = copy.deepcopy(musical_goals)
        metadata_copy = copy.deepcopy(metadata) if metadata else {}

        snapshot = AudioSnapshot(
            name=name,
            timestamp=datetime.now(),
            audio=audio_copy,
            sr=sr,
            musical_goals=goals_copy,
            metadata=metadata_copy,
        )

        # Add snapshot
        self.snapshots.append(snapshot)

        # Enforce max snapshots limit (keep most recent)
        if len(self.snapshots) > self.max_snapshots:
            removed = self.snapshots.pop(0)  # Remove oldest
            logger.info("Removed oldest snapshot '%s' (max limit reached)", removed.name)

        logger.info("erstellt snapshot '%s': %s/%s snapshots", name, len(self.snapshots), self.max_snapshots)

    def get_snapshot(self, name: str) -> AudioSnapshot | None:
        """
        Gibt zurück: snapshot by name.

        Args:
            name: Snapshot name

        Returns:
            AudioSnapshot if found, None otherwise
        """
        for snapshot in self.snapshots:
            if snapshot.name == name:
                return snapshot
        return None

    def rollback_to_snapshot(
        self, name: str, reason: str = "Manual rollback"
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        """
        Rollback to a specific snapshot.

        Args:
            name: Snapshot name to rollback to
            reason: Reason for rollback

        Returns:
            Tuple of (audio, sr, musical_goals) from snapshot

        Raises:
            ValueError: If snapshot not found
        """
        snapshot = self.get_snapshot(name)

        if snapshot is None:
            available = [s.name for s in self.snapshots]
            raise ValueError(f"Snapshot '{name}' not found. Available: {', '.join(available) if available else 'none'}")

        # Record rollback decision
        current_name = self.snapshots[-1].name if self.snapshots else None
        decision = RollbackDecision(rolled_back=True, from_snapshot=current_name, to_snapshot=name, reason=reason)
        self.rollback_history.append(decision)

        logger.warning("Rolling back from '%s' to '%s': %s", current_name, name, reason)

        # Return deep copies to prevent unwanted modifications
        return (copy.deepcopy(snapshot.audio), snapshot.sr, copy.deepcopy(snapshot.musical_goals))

    def rollback_to_latest(self, reason: str = "Rollback to latest") -> tuple[np.ndarray, int, dict[str, float]]:
        """
        Rollback to the most recent snapshot.

        Args:
            reason: Reason for rollback

        Returns:
            Tuple of (audio, sr, musical_goals) from latest snapshot

        Raises:
            ValueError: If no snapshots available
        """
        if not self.snapshots:
            raise ValueError("No snapshots available for rollback")

        latest = self.snapshots[-1]
        return self.rollback_to_snapshot(latest.name, reason=reason)

    def rollback_to_index(
        self, index: int, reason: str = "Rollback by index"
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        """
        Rollback to snapshot at specific index.

        Args:
            index: Index in snapshots list (0 = oldest, -1 = latest)
            reason: Reason for rollback

        Returns:
            Tuple of (audio, sr, musical_goals) from snapshot

        Raises:
            IndexError: If index out of range
        """
        if not self.snapshots:
            raise ValueError("No snapshots available for rollback")

        try:
            snapshot = self.snapshots[index]
            return self.rollback_to_snapshot(snapshot.name, reason=reason)
        except IndexError as e:
            raise IndexError(f"Snapshot index {index} out of range (available: 0 to {len(self.snapshots) - 1})") from e

    def list_snapshots(self) -> list[dict]:
        """
        Listet auf: all available snapshots.

        Returns:
            List of snapshot summaries
        """
        return [
            {
                "name": s.name,
                "timestamp": s.timestamp.isoformat(),
                "sr": s.sr,
                "audio_shape": s.audio.shape,
                "musical_goals": s.musical_goals,
                "metadata": s.metadata,
            }
            for s in self.snapshots
        ]

    def get_rollback_history(self) -> list[dict]:
        """
        Gibt zurück: history of all rollback decisions.

        Returns:
            List of rollback decisions
        """
        return [
            {
                "rolled_back": d.rolled_back,
                "from_snapshot": d.from_snapshot,
                "to_snapshot": d.to_snapshot,
                "reason": d.reason,
                "timestamp": d.timestamp.isoformat(),
            }
            for d in self.rollback_history
        ]

    def clear_snapshots(self):
        """Löscht all snapshots (use with caution!)."""
        count = len(self.snapshots)
        self.snapshots = []
        logger.warning("geleert all %s snapshots", count)

    def clear_history(self):
        """Löscht rollback history."""
        count = len(self.rollback_history)
        self.rollback_history = []
        logger.info("geleert rollback history (%s entries)", count)


if __name__ == "__main__":
    # Test Rollback Manager
    logger.debug("=== AURIK v8 Rollback Manager Test ===\n")

    # Create test audio
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio_original = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine

    # Initialize manager
    manager = RollbackManager(max_snapshots=3)

    # Create snapshots
    logger.debug("1. Creating snapshots:")

    # Snapshot 1: Original
    goals_1 = {
        "bass_kraft": 0.85,
        "brillanz": 0.87,
        "waerme": 0.82,
        "natuerlichkeit": 0.92,
        "authentizitaet": 0.90,
        "emotionalitaet": 0.88,
        "transparenz": 0.91,
    }
    manager.create_snapshot("original", audio_original, sr, goals_1)
    logger.debug("   erstellt 'Originalsignal'")

    # Snapshot 2: After noise reduction (slightly degraded)
    audio_nr = audio_original * 0.95
    goals_2 = goals_1.copy()
    goals_2["natuerlichkeit"] = 0.88
    manager.create_snapshot("after_noise_reduction", audio_nr, sr, goals_2)
    logger.debug("   erstellt 'after_noise_reduction'")

    # Snapshot 3: After enhancement (further degraded)
    audio_enh = audio_nr * 0.90
    goals_3 = goals_2.copy()
    goals_3["authentizitaet"] = 0.75  # Below threshold!
    manager.create_snapshot("after_enhancement", audio_enh, sr, goals_3)
    logger.debug("   erstellt 'after_enhancement'")

    # List snapshots
    logger.debug("\n2. List snapshots:")
    for i, snapshot_info in enumerate(manager.list_snapshots()):
        logger.debug("   [%s] %s - %s", i, snapshot_info["name"], snapshot_info["timestamp"])
        logger.debug("       Musical goals: %s", snapshot_info["musical_goals"])

    # Rollback to 'after_noise_reduction' (violations detected)
    logger.debug("\n3. Rollback (violation erkannt in 'after_enhancement'):")
    audio_restored, sr_restored, goals_restored = manager.rollback_to_snapshot(
        "after_noise_reduction", reason="Critical violation: authentizitaet < 0.88"
    )
    logger.debug("   Rolled back to 'after_noise_reduction'")
    logger.debug("   wiederhergestellt musical goals: %s", goals_restored)

    # Verify audio restored correctly
    logger.debug("   Audio wiederhergestellt correctly: %s", np.allclose(audio_restored, audio_nr))

    # Rollback history
    logger.debug("\n4. Rollback history:")
    for entry in manager.get_rollback_history():
        logger.debug("   %s → %s: %s", entry["from_snapshot"], entry["to_snapshot"], entry["reason"])

    logger.debug("\n=== Test vollstaendig ===")
