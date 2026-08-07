"""§AA: Multi-Objective Phase Parameter Optimizer (GP-basiert, Pareto-optimal).

Nutzt einen leichtgewichtigen evolutionären Algorithmus um die optimale
Kombination von Phase-Stärken zu finden, die ALLE 15 Musical Goals
gleichzeitig maximiert — nicht nur eines auf Kosten anderer.

Funktionsweise:
1. PID-Phasenplan → Individuen (Vektoren von Phase-Stärken)
2. Fitness = gewichtete Summe der PMGG-Goal-Scores nach Mini-Restore
3. Pareto-Front: Nicht-dominierte Lösungen (kein Goal wird schlechter)
4. Beste Lösung → als initial_strength-Hints an UV3 übergeben
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Ergebnis der Multi-Objective-Optimierung."""

    best_strengths: dict[str, float] = field(default_factory=dict)
    pareto_front: list[dict[str, Any]] = field(default_factory=list)
    fitness_history: list[float] = field(default_factory=list)
    generations: int = 0
    improvement_pct: float = 0.0


class MultiObjectiveOptimizer:
    """GP-basierte Multi-Objective-Optimierung für Phase-Stärken.

    Population = 30 Individuen, 15 Generationen, Elitism + Crossover + Mutation.
    """

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 15,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
    ) -> None:
        self._pop_size = population_size
        self._generations = generations
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate

    def optimize(
        self,
        phases: list[str],
        evaluate_fn: Callable,  # type: ignore[name-defined]  # (strengths: dict) -> dict[str, float]
        initial_strengths: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Führt die Optimierung durch.

        Args:
            phases: Liste der Phase-IDs
            evaluate_fn: Funktion die (strengths_dict) -> goal_scores_dict zurückgibt
            initial_strengths: Optionale Start-Stärken (aus PID/Budget)
        """
        if len(phases) <= 1:
            return OptimizationResult(best_strengths=initial_strengths or {})

        # Initialisierung
        population: list[dict[str, float]] = []
        base = initial_strengths or {}
        for i in range(self._pop_size):
            if i == 0 and base:
                population.append(dict(base))
            else:
                individual = {}
                for ph in phases:
                    individual[ph] = float(np.clip(base.get(ph, 0.5) + random.uniform(-0.25, 0.25), 0.1, 1.0))
                population.append(individual)

        # Evaluations-Cache
        cache: dict[str, dict[str, float]] = {}

        def _cached_evaluate(individual: dict) -> dict[str, float]:
            key = str(sorted(individual.items()))
            if key not in cache:
                cache[key] = evaluate_fn(individual)  # type: ignore[misc]
            return cache[key]

        # Fitness: gewichteter Goal-Mittelwert
        def _fitness(scores: dict[str, float]) -> float:
            if not scores:
                return 0.0
            # P1-Goals doppelt gewichten
            weights = {
                g: (2.0 if g in ("waerme", "brillanz", "emotionalitaet", "natuerlichkeit") else 1.0) for g in scores
            }
            return float(np.average(list(scores.values()), weights=[weights.get(g, 1.0) for g in scores]))

        best_individual = population[0]
        best_fitness = -999.0
        fitness_history: list[float] = []
        pareto: list[dict[str, Any]] = []

        for gen in range(self._generations):
            # Evaluierung
            scored = []
            for ind in population:
                scores = _cached_evaluate(ind)
                fit = _fitness(scores)
                scored.append((fit, ind, scores))

            scored.sort(key=lambda x: x[0], reverse=True)
            current_best = scored[0]
            fitness_history.append(current_best[0])

            if current_best[0] > best_fitness:
                best_fitness = current_best[0]
                best_individual = dict(current_best[1])

            # Pareto-Front aktualisieren
            for _, ind, scores in scored[:5]:
                dominated = False
                for existing in pareto:
                    if all(scores.get(g, 0) <= existing["scores"].get(g, 0) for g in scores):
                        dominated = True
                        break
                if not dominated:
                    pareto.append({"strengths": dict(ind), "scores": dict(scores)})
            pareto = pareto[-10:]  # Max 10 behalten

            # Selektion + Crossover + Mutation
            elites = [ind for _, ind, _ in scored[:5]]
            new_pop = list(elites)

            while len(new_pop) < self._pop_size:
                # Tournament selection
                a = random.choice(scored[:15])
                b = random.choice(scored[:15])
                parent1 = a[1] if a[0] > b[0] else b[1]

                c = random.choice(scored[:15])
                d = random.choice(scored[:15])
                parent2 = c[1] if c[0] > d[0] else d[1]

                # Crossover
                child: dict[str, float] = {}
                for ph in phases:
                    if random.random() < self._crossover_rate:
                        child[ph] = parent1.get(ph, 0.5)
                    else:
                        child[ph] = parent2.get(ph, 0.5)

                # Mutation
                for ph in phases:
                    if random.random() < self._mutation_rate:
                        child[ph] = float(np.clip(child[ph] + random.gauss(0, 0.1), 0.1, 1.0))

                new_pop.append(child)

            population = new_pop

        improvement = ((best_fitness - fitness_history[0]) / max(abs(fitness_history[0]), 1e-6)) * 100
        logger.info(
            "§AA GP-Optimizer: %d Gen → fitness=%.3f (%.0f%% improve), Pareto=%d",
            self._generations,
            best_fitness,
            improvement,
            len(pareto),
        )

        return OptimizationResult(
            best_strengths=best_individual,
            pareto_front=pareto,
            fitness_history=fitness_history,
            generations=self._generations,
            improvement_pct=improvement,
        )
