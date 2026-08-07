"""
ContextAwareGoalOptimizer: Adaptive, selbstlernende Optimierung nach individuellen musikalischen Zielen in Echtzeit.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


_instance: ContextAwareGoalOptimizer | None = None
_lock_singleton = threading.Lock()


def get_goal_optimizer(
    get_context: Callable[[], dict[str, Any]] | None = None,
    feedback_callback: Callable[[dict[str, float]], None] | None = None,
) -> ContextAwareGoalOptimizer:
    """Gibt zurück: or create ContextAwareGoalOptimizer singleton.

    Args:
        get_context: Context getter function (only used on first call)
        feedback_callback: Feedback callback (only used on first call)

    Returns:
        ContextAwareGoalOptimizer singleton instance
    """
    global _instance
    if _instance is None:
        with _lock_singleton:
            if _instance is None:
                _instance = ContextAwareGoalOptimizer(
                    get_context or (lambda: {}), feedback_callback or (lambda x: None)
                )
    return _instance


class ContextAwareGoalOptimizer:
    def __init__(
        self, get_context: Callable[[], dict[str, Any]], feedback_callback: Callable[[dict[str, float]], None]
    ) -> None:
        self.get_context = get_context
        self.feedback_callback = feedback_callback
        self.goal_weights: dict[str, float] = {}
        self.lock = threading.RLock()
        logger.info("ContextAwareGoalOptimizer initialisiert")

    def set_goal_weights(self, weights: dict[str, float]) -> None:
        with self.lock:
            self.goal_weights = weights.copy()

    def optimize(self, current_metrics: dict[str, float]) -> dict[str, float]:
        context = self.get_context()
        adjustments = {}
        with self.lock:
            for goal, target in self.goal_weights.items():
                value = current_metrics.get(goal, 0.0)
                delta = target - value
                # Kontextbewusste Anpassung (z. B. Genre, User, Material)
                factor = context.get("adaptivity", 1.0)
                adjustments[goal] = value + delta * factor
        self.feedback_callback(adjustments)
        return adjustments
