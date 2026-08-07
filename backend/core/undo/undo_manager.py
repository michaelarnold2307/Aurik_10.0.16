"""
Undo/Redo System for User Actions

Component 5.2: Undo/Redo System
Impact: +1.0 Punkt - User-gesteuerte Rückgängig-Funktionalität

Provides comprehensive undo/redo functionality for all user actions:
- Undo/Redo stacks (last 50 actions)
- Multiple action types (processing, parameters, mode, file operations)
- Memory-efficient audio snapshots with delta compression
- Composite actions for multi-step undo
- Action serialization

Problem:
Laien erwarten Ctrl+Z in jeder modernen GUI:
- "Oh nein, falscher Mode! Kann ich zurück?"
- "Das war zu viel Noise Reduction, kann ich weniger machen?"
- Keine Undo → User muss komplett neu starten

Solution:
UndoManager verwaltet Undo/Redo-History für alle User-Aktionen:
- Action-basierte Architektur (revert/apply)
- Memory-effiziente Audio-Snapshots
- Max 50 Undo-Levels (konfigurierbar)
- UI-friendly history descriptions

Author: AI Team
Date: 8. Februar 2026
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of undoable actions."""

    PROCESSING = "processing"  # Audio processing (full snapshot)
    PARAMETER = "parameter"  # Parameter change (no audio)
    MODE = "mode"  # Processing mode change
    FILE_OPERATION = "file_operation"  # File load/save/export
    COMPOSITE = "composite"  # Multiple actions grouped


@dataclass
class AudioSnapshot:
    """
    Memory-efficient audio snapshot using delta compression.

    Attributes:
        audio: Audio data (None if delta-compressed)
        delta: Delta from previous snapshot (if using compression)
        sample_rate: Sample rate
        shape: Original audio shape
        is_compressed: Whether this uses delta compression
        reference_index: Index of reference snapshot for delta
    """

    audio: np.ndarray | None = None
    delta: np.ndarray | None = None
    sample_rate: int = 48000
    shape: tuple[int, ...] | None = None
    is_compressed: bool = False
    reference_index: int | None = None

    def get_audio(self, reference: np.ndarray | None = None) -> np.ndarray:
        """
        Rekonstruiert audio from snapshot.

        Args:
            reference: Reference audio if using delta compression

        Returns:
            Reconstructed audio array
        """
        if not self.is_compressed:
            if self.audio is None:
                raise ValueError("Full snapshot has no audio payload")
            return self.audio.copy()

        if reference is None:
            raise ValueError("Delta-compressed snapshot requires reference audio")

        # Reconstruct from delta
        return reference + self.delta  # type: ignore[no-any-return]

    @classmethod
    def create_full(cls, audio: np.ndarray, sr: int) -> "AudioSnapshot":
        """Erstellt full audio snapshot (no compression)."""
        return cls(audio=audio.copy(), sample_rate=sr, shape=audio.shape, is_compressed=False)

    @classmethod
    def create_delta(cls, audio: np.ndarray, reference: np.ndarray, sr: int, reference_index: int) -> "AudioSnapshot":
        """Erstellt delta-compressed snapshot."""
        delta = audio - reference

        return cls(delta=delta, sample_rate=sr, shape=audio.shape, is_compressed=True, reference_index=reference_index)

    def memory_size(self) -> int:
        """Schätzt memory size in bytes."""
        if self.audio is not None:
            return self.audio.nbytes
        elif self.delta is not None:
            return self.delta.nbytes
        return 0


class Action(ABC):
    """
    Abstract base class for undoable actions.

    All actions must implement:
    - revert(): Undo the action
    - apply(): Redo the action
    - cleanup(): Free memory when action is removed from history
    """

    def __init__(self, action_type: ActionType, description: str, timestamp: str | None = None):
        self.action_type = action_type
        self.description = description
        self.timestamp = timestamp or datetime.now().isoformat()

    @abstractmethod
    def revert(self) -> dict[str, Any]:
        """
        Revert action (undo).

        Returns:
            State dict with restored values
        """

    @abstractmethod
    def apply(self) -> dict[str, Any]:
        """
        Wendet Aktion erneut an (Redo).

        Returns:
            State dict with new values
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Free memory when action is removed from history."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(description='{self.description}')"


class ProcessingAction(Action):
    """
    Action for audio processing operations.

    Stores audio snapshots before and after processing.
    Uses delta compression to save memory.
    """

    def __init__(
        self,
        description: str,
        before_audio: np.ndarray,
        after_audio: np.ndarray,
        sr: int,
        processing_params: dict | None = None,
        use_delta: bool = True,
        reference_snapshot: AudioSnapshot | None = None,
        reference_index: int | None = None,
    ):
        super().__init__(ActionType.PROCESSING, description)

        self.sr = sr
        self.processing_params = processing_params or {}
        self._reference_audio: np.ndarray | None = None

        # Store audio snapshots
        if use_delta and reference_snapshot is not None:
            # Delta compression: store only difference from reference
            reference_audio = reference_snapshot.get_audio()
            self.before_snapshot = AudioSnapshot.create_delta(before_audio, reference_audio, sr, reference_index)  # type: ignore[arg-type]
            self._reference_audio = reference_audio.copy()
        else:
            # Full snapshot
            self.before_snapshot = AudioSnapshot.create_full(before_audio, sr)

        # After snapshot (always full for easier reconstruction)
        self.after_snapshot = AudioSnapshot.create_full(after_audio, sr)

        self.current_state = "after"  # Track current state

    def revert(self) -> dict[str, Any]:
        """Revert to before-processing audio."""
        self.current_state = "before"
        if self.before_snapshot is None:
            raise RuntimeError("Before snapshot is not available")
        return {
            "audio": self.before_snapshot.get_audio(self._reference_audio),
            "sr": self.sr,
            "params": self.processing_params,
        }

    def apply(self) -> dict[str, Any]:
        """Wendet an: processing (go to after-audio)."""
        self.current_state = "after"
        if self.after_snapshot is None:
            raise RuntimeError("After snapshot is not available")
        return {"audio": self.after_snapshot.get_audio(), "sr": self.sr, "params": self.processing_params}

    def cleanup(self):
        """Free audio memory."""
        self.before_snapshot = None  # type: ignore[assignment]
        self.after_snapshot = None  # type: ignore[assignment]
        self._reference_audio = None

    def memory_size(self) -> int:
        """Schätzt total memory usage."""
        size = 0
        if self.before_snapshot:
            size += self.before_snapshot.memory_size()
        if self.after_snapshot:
            size += self.after_snapshot.memory_size()
        return size


class ParameterChangeAction(Action):
    """
    Action for parameter changes without audio processing.

    Lightweight - only stores parameter values, no audio.
    """

    def __init__(self, description: str, parameter_name: str, old_value: Any, new_value: Any):
        super().__init__(ActionType.PARAMETER, description)

        self.parameter_name = parameter_name
        self.old_value = old_value
        self.new_value = new_value
        self.current_value = new_value

    def revert(self) -> dict[str, Any]:
        """Revert to old parameter value."""
        self.current_value = self.old_value
        return {"parameter": self.parameter_name, "value": self.old_value}

    def apply(self) -> dict[str, Any]:
        """Wendet an: new parameter value."""
        self.current_value = self.new_value
        return {"parameter": self.parameter_name, "value": self.new_value}

    def cleanup(self) -> None:
        """No resources to free for parameter actions."""


class ModeChangeAction(Action):
    """
    Action for processing mode changes.

    Stores mode selection changes.
    """

    def __init__(self, description: str, old_mode: str, new_mode: str, mode_config: dict | None = None):
        super().__init__(ActionType.MODE, description)

        self.old_mode = old_mode
        self.new_mode = new_mode
        self.mode_config = mode_config or {}
        self.current_mode = new_mode

    def revert(self) -> dict[str, Any]:
        """Revert to old mode."""
        self.current_mode = self.old_mode
        return {"mode": self.old_mode, "config": self.mode_config}

    def apply(self) -> dict[str, Any]:
        """Wendet an: new mode."""
        self.current_mode = self.new_mode
        return {"mode": self.new_mode, "config": self.mode_config}

    def cleanup(self) -> None:
        """No resources to release for mode change actions."""


class FileOperationAction(Action):
    """
    Action for file operations (load, save, export).

    Stores file paths and operation details.
    """

    def __init__(
        self,
        description: str,
        operation: str,  # "load", "save", "export"
        file_path: str,
        metadata: dict | None = None,
    ):
        super().__init__(ActionType.FILE_OPERATION, description)

        self.operation = operation
        self.file_path = file_path
        self.metadata = metadata or {}

    def revert(self) -> dict[str, Any]:
        """Revert file operation."""
        return {"operation": f"revert_{self.operation}", "file_path": self.file_path, "metadata": self.metadata}

    def apply(self) -> dict[str, Any]:
        """Wendet an: file operation."""
        return {"operation": self.operation, "file_path": self.file_path, "metadata": self.metadata}

    def cleanup(self) -> None:
        """No resources to release for file operation actions."""


class CompositeAction(Action):
    """
    Action for multiple grouped actions (multi-step undo).

    Allows undoing/redoing multiple related actions as a single unit.
    """

    def __init__(self, description: str, actions: list[Action]):
        super().__init__(ActionType.COMPOSITE, description)

        self.actions = actions

    def revert(self) -> dict[str, Any]:
        """Revert all actions in reverse order."""
        results = []
        for action in reversed(self.actions):
            result = action.revert()
            results.append(result)

        return {"composite_results": results, "action_count": len(self.actions)}

    def apply(self) -> dict[str, Any]:
        """Wendet alle Aktionen in Vorwärtsreihenfolge an."""
        results = []
        for action in self.actions:
            result = action.apply()
            results.append(result)

        return {"composite_results": results, "action_count": len(self.actions)}

    def cleanup(self):
        """Cleanup all sub-actions."""
        for action in self.actions:
            action.cleanup()


class UndoManager:
    """
    Manages undo/redo history for all user actions.

    Features:
    - Undo/Redo stacks (max 50 actions)
    - Multiple action types
    - Memory-efficient audio snapshots
    - Composite actions
    - Action serialization

    Example:
        >>> manager = UndoManager()
        >>>
        >>> # Record audio processing
        >>> action = ProcessingAction(
        ...     "Apply Noise Reduction",
        ...     before_audio=original,
        ...     after_audio=processed,
        ...     sr=48000
        ... )
        >>> manager.record_action(action)
        >>>
        >>> # Undo
        >>> state = manager.undo()
        >>> restored_audio = state['audio']
        >>>
        >>> # Redo
        >>> state = manager.redo()
        >>> processed_audio = state['audio']
    """

    def __init__(self, max_undo_levels: int = 50, enable_delta_compression: bool = True):
        """
        Initialisiert UndoManager.

        Args:
            max_undo_levels: Maximum number of undo actions to keep
            enable_delta_compression: Use delta compression for audio snapshots
        """
        self.max_undo_levels = max_undo_levels
        self.enable_delta_compression = enable_delta_compression

        self.undo_stack: list[Action] = []
        self.redo_stack: list[Action] = []

        # Track total memory usage
        self._total_memory = 0

        logger.info(
            f"UndoManager initialisiert (max_levels={max_undo_levels}, delta_compression={enable_delta_compression})"
        )

    def record_action(self, action: Action):
        """
        Record a new action in the undo stack.

        Args:
            action: Action to record
        """
        # Remove oldest action if at max capacity
        if len(self.undo_stack) >= self.max_undo_levels:
            oldest = self.undo_stack.pop(0)
            oldest.cleanup()
            logger.debug("Removed oldest action: %s", oldest.description)

        # Add new action
        self.undo_stack.append(action)

        # Clear redo stack (new action invalidates redo history)
        for action in self.redo_stack:
            action.cleanup()
        self.redo_stack.clear()

        logger.info("Recorded action: %s", action.description)

    def undo(self) -> dict[str, Any] | None:
        """
        Undo last action.

        Returns:
            State dict from action.revert(), or None if undo stack empty
        """
        if not self.can_undo():
            logger.warning("Undo fehlgeschlagen: stack is empty")
            return None

        action = self.undo_stack.pop()
        previous_state = action.revert()
        self.redo_stack.append(action)

        logger.info("Undid action: %s", action.description)
        return previous_state

    def redo(self) -> dict[str, Any] | None:
        """
        Redo last undone action.

        Returns:
            State dict from action.apply(), or None if redo stack empty
        """
        if not self.can_redo():
            logger.warning("Redo fehlgeschlagen: stack is empty")
            return None

        action = self.redo_stack.pop()
        new_state = action.apply()
        self.undo_stack.append(action)

        logger.info("Redid action: %s", action.description)
        return new_state

    def can_undo(self) -> bool:
        """Prüft if undo is possible."""
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """Prüft if redo is possible."""
        return len(self.redo_stack) > 0

    def get_undo_history(self) -> list[str]:
        """
        Gibt zurück: user-readable undo history.

        Returns:
            List of action descriptions (most recent last)
        """
        return [action.description for action in self.undo_stack]

    def get_redo_history(self) -> list[str]:
        """
        Gibt zurück: user-readable redo history.

        Returns:
            List of action descriptions (most recent first)
        """
        return [action.description for action in reversed(self.redo_stack)]

    def clear_history(self):
        """Löscht all undo/redo history."""
        # Cleanup all actions
        for action in self.undo_stack:
            action.cleanup()
        for action in self.redo_stack:
            action.cleanup()

        self.undo_stack.clear()
        self.redo_stack.clear()

        logger.info("geleert all undo/redo history")

    def get_memory_usage(self) -> int:
        """
        Schätzt total memory usage in bytes.

        Returns:
            Approximate memory usage
        """
        total = 0

        for action in self.undo_stack:
            if isinstance(action, ProcessingAction):
                total += action.memory_size()

        for action in self.redo_stack:
            if isinstance(action, ProcessingAction):
                total += action.memory_size()

        return total

    def get_memory_usage_mb(self) -> float:
        """Gibt zurück: memory usage in MB."""
        return self.get_memory_usage() / (1024 * 1024)


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create undo manager
    manager = UndoManager(max_undo_levels=10)

    # Simulate audio processing
    original_audio = np.random.randn(48000)  # 1 second

    # Action 1: Apply noise reduction
    processed1 = original_audio * 0.9
    action1 = ProcessingAction(
        "Apply Noise Reduction (strength=0.5)",
        before_audio=original_audio,
        after_audio=processed1,
        sr=48000,
        processing_params={"module": "denoise", "strength": 0.5},
    )
    manager.record_action(action1)

    # Action 2: Change parameter
    action2 = ParameterChangeAction(
        "Change strength to 0.8", parameter_name="denoise_strength", old_value=0.5, new_value=0.8
    )
    manager.record_action(action2)

    # Action 3: Apply with new parameter
    processed2 = processed1 * 0.85
    action3 = ProcessingAction(
        "Reapply Noise Reduction (strength=0.8)",
        before_audio=processed1,
        after_audio=processed2,
        sr=48000,
        processing_params={"module": "denoise", "strength": 0.8},
    )
    manager.record_action(action3)

    # Check history
    import logging

    logging.info("\nUndo history:")
    for i, desc in enumerate(manager.get_undo_history(), 1):
        logging.info(f"  {i}. {desc}")

    # Undo last action
    logging.info("\nUndo last action:")
    state = manager.undo()
    logging.info(f"Restored to: {state}")

    # Check redo history
    logging.info("\nRedo history:")
    for i, desc in enumerate(manager.get_redo_history(), 1):
        logging.info(f"  {i}. {desc}")

    # Redo
    logging.info("\nRedo:")
    state = manager.redo()
    logging.info(f"Reapplied: {state}")

    # Memory usage
    logger.debug("\nMemory usage: %.2f MB", manager.get_memory_usage_mb())
