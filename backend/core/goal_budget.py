"""§J: Goal-Budget pro Phase — Pre-Allokation + laufende Abbuchung.

Jede Phase bekommt einen Plan, wie viel sie zu jedem Musical Goal beitragen
darf.  Das Budget wird vor Pipeline-Start aus Genre-Profil und Material-
Restorability berechnet.  Nach jeder Phase wird das erreichte Delta vom
Budget abgezogen — spätere Phasen dosieren entsprechend schwächer, um
kumulatives Over-Processing zu vermeiden.

Single entry point::

    budget = GoalBudget(targets, material_key="cassette")
    budget.get_limit(goal)       → max improvement this goal still needs
    budget.record_delta(goal, d) → deduct achieved delta
    budget.fraction_left(goal)   → 0.0–1.0, wie viel Budget noch übrig ist
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Material-adaptive Start-Budgets: wie viel Verbesserung pro Goal maximal nötig.
# Werte sind Deltas in [0, 1] — 0.3 = das Goal kann um max 0.3 Score-Punkte steigen.
_DEFAULT_GOAL_BUDGET: dict[str, float] = {
    "waerme": 0.25,
    "brillanz": 0.20,
    "emotionalitaet": 0.15,
    "groove": 0.08,
    "artikulation": 0.10,
    "natuerlichkeit": 0.12,
    "authentizitaet": 0.05,
    "durchschlagskraft": 0.15,
    "raeumlichkeit": 0.18,
    "transparenz": 0.20,
    "dynamik": 0.10,
    "stimmklarheit": 0.18,
    "bassdefinition": 0.15,
    "mikrodynamik": 0.08,
    "klangbalance": 0.15,
}

# Hochgradig beschädigte Materialien brauchen grössere Budgets.
_MATERIAL_BUDGET_SCALE: dict[str, float] = {
    "wax_cylinder": 1.8,
    "shellac": 1.5,
    "lacquer_disc": 1.4,
    "wire_recording": 1.6,
    "vinyl": 1.2,
    "tape": 1.15,
    "reel_tape": 1.1,
    "cassette": 1.3,
    "cd_digital": 0.8,
    "streaming": 0.7,
    "mp3_low": 1.0,
    "mp3_high": 0.75,
    "dat": 0.7,
    "aac": 0.6,
    "minidisc": 0.9,
    "unknown": 1.0,
}


class GoalBudget:
    """Lebendes Budget — wird pro Phase aktualisiert."""

    def __init__(
        self,
        initial_targets: dict[str, float] | None = None,
        material_key: str = "unknown",
    ) -> None:
        scale = _MATERIAL_BUDGET_SCALE.get(material_key, 1.0)
        self._budget: dict[str, float] = {}
        self._spent: dict[str, float] = {}
        for goal, base in (_DEFAULT_GOAL_BUDGET if initial_targets is None else initial_targets).items():
            self._budget[goal] = round(base * scale, 4)
            self._spent[goal] = 0.0

    def get_limit(self, goal: str) -> float:
        """Maximale noch verfügbare Verbesserung für dieses Goal."""
        return max(0.0, self._budget.get(goal, 0.0) - self._spent.get(goal, 0.0))

    def fraction_left(self, goal: str) -> float:
        """0.0–1.0 Anteil des Budgets, der noch verfügbar ist."""
        b = self._budget.get(goal, 0.0)
        if b <= 0:
            return 0.0
        return max(0.0, min(1.0, self.get_limit(goal) / b))

    def record_delta(self, goal: str, delta: float) -> None:
        """Bucht einen erreichten Delta-Wert vom Budget ab."""
        if goal in self._spent:
            self._spent[goal] = round(self._spent[goal] + max(0.0, delta), 4)

    def is_exhausted(self, goal: str) -> bool:
        """True wenn Budget für dieses Goal aufgebraucht ist."""
        return self.get_limit(goal) <= 0.001

    def strength_modifier(self, goal: str) -> float:
        """0.0–1.0 Multiplikator für Phasen-Stärke basierend auf Restbudget."""
        fl = self.fraction_left(goal)
        if fl >= 0.9:
            return 1.0
        if fl <= 0.05:
            return 0.1
        return fl

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": dict(self._budget),
            "spent": dict(self._spent),
            "remaining": {g: self.get_limit(g) for g in self._budget},
        }


def create_goal_budget(
    material_key: str = "unknown",
    genre_key: str = "",
) -> GoalBudget:
    """Fabrik: erstellt GoalBudget mit material- und genre-angepassten Targets."""
    # §H-Integration: genre-spezifische Startziele
    targets = dict(_DEFAULT_GOAL_BUDGET)
    if genre_key:
        try:
            from backend.core.genre_goal_profile import get_genre_profile

            profile = get_genre_profile(genre_key)
            # Genre-Ziele überschreiben Defaults
            for goal, weight in profile.weights.items():
                if goal in targets:
                    targets[goal] = round(0.30 * weight / 2.0, 3)  # weight 2.0 → 0.30 Budget
        except Exception as e:
            logger.warning("goal_Grenze.py::erstellen_goal_Grenze Ersatzpfad: %s", e)
    return GoalBudget(targets, material_key=material_key)
