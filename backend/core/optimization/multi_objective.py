"""
Multi-Objective Optimization für Aurik 10.0.0

Implementiert Multi-Objective Optimization mit Pareto-Front-Berechnung:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
- Pareto-Front Visualisierung
- Trade-off Analyse (Quality vs. Speed vs. Authenticity)

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

import json
import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_plt = None


def _get_matplotlib_pyplot():
    """Lazily import matplotlib only for visualization paths."""
    global _plt
    if _plt is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not verfuegbar, visualization will be deaktiviert")
            return None
        _plt = plt
    return _plt


@dataclass
class Individual:
    """
    Individual in multi-objective optimization population.

    Represents a configuration with multiple objectives.
    """

    parameters: dict[str, Any]
    objectives: dict[str, float] = field(default_factory=dict)
    rank: int = -1  # Pareto rank
    crowding_distance: float = 0.0
    dominated_count: int = 0
    dominates: list["Individual"] = field(default_factory=list)

    def dominates_other(self, other: "Individual") -> bool:
        """Prüft if this individual dominates another (all objectives better or equal, at least one strictly better)."""
        if not self.objectives or not other.objectives:
            return False

        better_in_all = all(
            self.objectives.get(obj, float("inf")) <= other.objectives.get(obj, float("inf")) for obj in self.objectives
        )
        strictly_better_in_one = any(
            self.objectives.get(obj, float("inf")) < other.objectives.get(obj, float("inf")) for obj in self.objectives
        )

        return better_in_all and strictly_better_in_one


@dataclass
class ObjectiveFunction:
    """Objective function definition."""

    name: str
    evaluate: Callable[[dict[str, Any]], float]
    minimize: bool = True  # True for minimization, False for maximization


class NSGAII:
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II

    Multi-objective optimization algorithm that maintains a Pareto front.
    """

    def __init__(
        self,
        objectives: list[ObjectiveFunction],
        parameter_space: dict[str, tuple[Any, Any]],
        population_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
    ):
        """
        Initialisiert NSGA-II.

        Args:
            objectives: List of objective functions to optimize
            parameter_space: Dict mapping parameter names to (min, max) ranges
            population_size: Size of population
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
        """
        self.objectives = objectives
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.population: list[Individual] = []
        self.pareto_front: list[Individual] = []

        logger.info("NSGA-II initialisiert: %s objectives, pop_size=%s", len(objectives), population_size)

    def initialize_population(self) -> list[Individual]:
        """Initialisiert random population."""
        population = []

        for _ in range(self.population_size):
            parameters = {}
            for param_name, (param_min, param_max) in self.parameter_space.items():
                if isinstance(param_min, int) and isinstance(param_max, int):
                    parameters[param_name] = np.random.randint(param_min, param_max + 1)
                else:
                    parameters[param_name] = np.random.uniform(param_min, param_max)

            population.append(Individual(parameters=parameters))

        logger.info("initialisiert population of %s individuals", len(population))
        return population

    def evaluate_population(self, population: list[Individual]) -> None:
        """Bewertet all objectives for all individuals."""
        for individual in population:
            for objective in self.objectives:
                value = objective.evaluate(individual.parameters)

                # Convert to minimization problem
                if not objective.minimize:
                    value = -value

                individual.objectives[objective.name] = value

    def fast_non_dominated_sort(self, population: list[Individual]) -> list[list[Individual]]:
        """
        Fast non-dominated sorting algorithm.

        Returns list of fronts, where front[0] is the Pareto front.
        """
        # Reset domination info
        for ind in population:
            ind.dominated_count = 0
            ind.dominates = []

        # Calculate domination relationships
        for i, p in enumerate(population):
            for q in population[i + 1 :]:
                if p.dominates_other(q):
                    p.dominates.append(q)
                    q.dominated_count += 1
                elif q.dominates_other(p):
                    q.dominates.append(p)
                    p.dominated_count += 1

        # Assign ranks
        fronts = []
        current_front = [ind for ind in population if ind.dominated_count == 0]

        rank = 0
        while current_front:
            for ind in current_front:
                ind.rank = rank

            fronts.append(current_front)

            next_front = []
            for p in current_front:
                for q in p.dominates:
                    q.dominated_count -= 1
                    if q.dominated_count == 0:
                        next_front.append(q)

            current_front = next_front
            rank += 1

        return fronts

    def calculate_crowding_distance(self, front: list[Individual]):
        """Calculate crowding distance for individuals in a front."""
        n = len(front)
        if n == 0:
            return

        # Initialize
        for ind in front:
            ind.crowding_distance = 0.0

        # For each objective
        for obj_name in self.objectives[0].name if self.objectives else []:
            # Sort by objective value
            front_sorted = sorted(front, key=lambda x: x.objectives.get(obj_name, float("inf")))

            # Boundary points have infinite distance
            front_sorted[0].crowding_distance = float("inf")
            front_sorted[-1].crowding_distance = float("inf")

            # Calculate crowding distance
            obj_min = front_sorted[0].objectives.get(obj_name, 0)
            obj_max = front_sorted[-1].objectives.get(obj_name, 1)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                distance = (
                    front_sorted[i + 1].objectives.get(obj_name, 0) - front_sorted[i - 1].objectives.get(obj_name, 0)
                ) / obj_range
                # NaN/Inf-Guard (§3.1)
                distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
                front_sorted[i].crowding_distance += float(distance)

    def tournament_selection(self, population: list[Individual]) -> Individual:
        """Wählt aus: individual using binary tournament selection."""
        i1, i2 = np.random.choice(len(population), 2, replace=False)
        ind1, ind2 = population[i1], population[i2]

        # Prefer lower rank
        if ind1.rank < ind2.rank:
            return deepcopy(ind1)  # type: ignore[no-any-return]
        elif ind2.rank < ind1.rank:
            return deepcopy(ind2)  # type: ignore[no-any-return]

        # If same rank, prefer higher crowding distance
        if ind1.crowding_distance > ind2.crowding_distance:
            return deepcopy(ind1)  # type: ignore[no-any-return]
        else:
            return deepcopy(ind2)  # type: ignore[no-any-return]

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """Simulated binary crossover (SBX)."""
        if np.random.random() > self.crossover_prob:
            return deepcopy(parent1), deepcopy(parent2)

        child1_params = {}
        child2_params = {}

        eta = 20  # Distribution index

        for param_name in parent1.parameters:
            p1_val = parent1.parameters[param_name]
            p2_val = parent2.parameters[param_name]

            param_min, param_max = self.parameter_space[param_name]

            if np.random.random() < 0.5:
                beta = (2 * np.random.random()) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - np.random.random()))) ** (1 / (eta + 1))

            c1 = 0.5 * ((p1_val + p2_val) - beta * abs(p2_val - p1_val))
            c2 = 0.5 * ((p1_val + p2_val) + beta * abs(p2_val - p1_val))

            # Clip to bounds
            c1 = np.clip(c1, param_min, param_max)
            c2 = np.clip(c2, param_min, param_max)
            # NaN/Inf-Guard (§3.1)
            c1 = float(np.nan_to_num(c1, nan=(param_min + param_max) / 2))
            c2 = float(np.nan_to_num(c2, nan=(param_min + param_max) / 2))
            child1_params[param_name] = c1
            child2_params[param_name] = c2

        return Individual(parameters=child1_params), Individual(parameters=child2_params)

    def mutate(self, individual: Individual) -> Individual:
        """Polynomial mutation."""
        if np.random.random() > self.mutation_prob:
            return individual

        eta = 20  # Distribution index

        for param_name, value in individual.parameters.items():
            if np.random.random() < 1.0 / len(individual.parameters):
                param_min, param_max = self.parameter_space[param_name]

                delta = param_max - param_min
                rand = np.random.random()

                delta_q = (2 * rand) ** (1 / (eta + 1)) - 1 if rand < 0.5 else 1 - (2 * (1 - rand)) ** (1 / (eta + 1))

                mutated_value = value + delta_q * delta
                mutated_value = np.clip(mutated_value, param_min, param_max)
                # NaN/Inf-Guard (§3.1)
                mutated_value = float(np.nan_to_num(mutated_value, nan=value))
                individual.parameters[param_name] = mutated_value

        return individual

    def optimize(self) -> list[Individual]:
        """
        Führt aus: NSGA-II optimization.

        Returns:
            Pareto front (list of non-dominated individuals)
        """
        logger.info("Starte NSGA-II optimization for %s generations...", self.n_generations)

        # Initialize population
        self.population = self.initialize_population()
        self.evaluate_population(self.population)

        for generation in range(self.n_generations):
            # Create offspring
            offspring: list[Individual] = []

            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                offspring.extend([child1, child2])

            offspring = offspring[: self.population_size]

            # Evaluate offspring
            self.evaluate_population(offspring)

            # Combine parent and offspring
            combined = self.population + offspring

            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined)

            # Calculate crowding distance for each front
            for front in fronts:
                self.calculate_crowding_distance(front)

            # Select next generation
            self.population = []
            for front in fronts:
                if len(self.population) + len(front) <= self.population_size:
                    self.population.extend(front)
                else:
                    # Sort by crowding distance and fill remaining slots
                    front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.population_size - len(self.population)
                    self.population.extend(front_sorted[:remaining])
                    break

            # Log progress
            if generation % 10 == 0:
                pareto_size = len(fronts[0]) if fronts else 0
                logger.info("Generation %s: Pareto front size = %s", generation, pareto_size)

        # Extract final Pareto front
        final_fronts = self.fast_non_dominated_sort(self.population)
        self.pareto_front = final_fronts[0] if final_fronts else []

        logger.info("Optimization abgeschlossen! Final Pareto front has %s solutions", len(self.pareto_front))

        return self.pareto_front

    def visualize_pareto_front(self, save_path: Path | None = None) -> None:
        """Visualize Pareto front (2D or 3D)."""
        plt = _get_matplotlib_pyplot()
        if plt is None:
            logger.warning("matplotlib not verfuegbar, cannot visualize Pareto front")
            return

        if not self.pareto_front:
            logger.warning("No Pareto front to visualize")
            return

        n_objectives = len(self.objectives)

        if n_objectives == 2:
            # 2D plot
            obj1_name = self.objectives[0].name
            obj2_name = self.objectives[1].name

            obj1_vals = [ind.objectives[obj1_name] for ind in self.pareto_front]
            obj2_vals = [ind.objectives[obj2_name] for ind in self.pareto_front]

            plt.figure(figsize=(10, 6))
            plt.scatter(obj1_vals, obj2_vals, c="blue", s=50, alpha=0.6)
            plt.xlabel(obj1_name)
            plt.ylabel(obj2_name)
            plt.title("Pareto Front")
            plt.grid(True)

        elif n_objectives == 3:
            # 3D plot
            try:
                pass
            except ImportError:
                logger.warning("mpl_toolkits.mplot3d not verfuegbar, cannot erstellen 3D plot")
                return

            obj1_name = self.objectives[0].name
            obj2_name = self.objectives[1].name
            obj3_name = self.objectives[2].name

            obj1_vals = [ind.objectives[obj1_name] for ind in self.pareto_front]
            obj2_vals = [ind.objectives[obj2_name] for ind in self.pareto_front]
            obj3_vals = [ind.objectives[obj3_name] for ind in self.pareto_front]

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(obj1_vals, obj2_vals, obj3_vals, c="blue", s=50, alpha=0.6)
            ax.set_xlabel(obj1_name)
            ax.set_ylabel(obj2_name)
            ax.set_zlabel(obj3_name)
            ax.set_title("Pareto Front (3D)")

        else:
            logger.warning("Cannot visualize %s objectives", n_objectives)
            return

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Pareto front gespeichert to %s", save_path)
        else:
            plt.show()

    def select_solution(self, preferences: dict[str, float] | None = None) -> Individual:
        """
        Wählt aus: a solution from Pareto front based on preferences.

        Args:
            preferences: Dict mapping objective names to weights (higher = more important)

        Returns:
            Selected individual
        """
        if not self.pareto_front:
            raise ValueError("No Pareto front available")

        if preferences is None:
            # Equal weights
            preferences = {obj.name: 1.0 for obj in self.objectives}

        # Normalize objectives to [0, 1]
        normalized_objs: dict[str, list[float]] = {obj.name: [] for obj in self.objectives}

        for obj in self.objectives:
            values = [ind.objectives[obj.name] for ind in self.pareto_front]
            min_val = min(values)
            max_val = max(values)

            if max_val - min_val == 0:
                normalized_objs[obj.name] = [0.0] * len(values)
            else:
                normalized_objs[obj.name] = [(v - min_val) / (max_val - min_val) for v in values]

        # Calculate weighted score
        best_score = float("inf")
        best_individual = None

        for i, individual in enumerate(self.pareto_front):
            score = sum(preferences.get(obj.name, 1.0) * normalized_objs[obj.name][i] for obj in self.objectives)

            if score < best_score:
                best_score = score
                best_individual = individual

        logger.info("Selected solution with Wert %.4f", best_score)
        return best_individual  # type: ignore[return-value]

    def save_pareto_front(self, path: Path):
        """Speichert Pareto front to JSON."""
        pareto_data = []

        for ind in self.pareto_front:
            pareto_data.append(
                {
                    "parameters": ind.parameters,
                    "objectives": ind.objectives,
                    "rank": ind.rank,
                    "crowding_distance": ind.crowding_distance,
                }
            )

        with open(path, "w") as f:
            json.dump(pareto_data, f, indent=2)

        logger.info("Pareto front gespeichert to %s", path)


# Example: Audio Restoration Multi-Objective Optimization
def create_audio_restoration_moo() -> NSGAII:
    """
    Erstellt multi-objective optimizer for audio restoration.

    Objectives:
    1. Audio Quality (maximize)
    2. Processing Speed (minimize)
    3. Authenticity Preservation (maximize)
    """

    # Define objectives
    def evaluate_quality(params: dict[str, Any]) -> float:
        """Simulate quality evaluation (lower is better for minimization)."""
        # In practice, this would run actual audio processing and compute PESQ/POLQA
        noise_reduction = params.get("noise_reduction", 0.5)
        declipping = params.get("declipping", 0.5)

        # Simulate quality score (higher noise_reduction and declipping = better quality)
        quality = -(noise_reduction * 0.6 + declipping * 0.4)  # Negative for minimization
        return quality  # type: ignore[no-any-return]

    def evaluate_speed(params: dict[str, Any]) -> float:
        """Simulate speed evaluation (minimize processing time)."""
        model_complexity = params.get("model_complexity", 0.5)
        n_iterations = params.get("n_iterations", 10)

        # More complexity and iterations = slower
        processing_time = model_complexity * 100 + n_iterations * 2
        return processing_time  # type: ignore[no-any-return]

    def evaluate_authenticity(params: dict[str, Any]) -> float:
        """Simulate authenticity preservation (maximize, so negate for minimization)."""
        noise_reduction = params.get("noise_reduction", 0.5)
        eq_strength = params.get("eq_strength", 0.5)

        # Too much processing reduces authenticity
        authenticity = -(1.0 - noise_reduction * 0.4 - eq_strength * 0.3)
        return authenticity  # type: ignore[no-any-return]

    objectives = [
        ObjectiveFunction("quality", evaluate_quality, minimize=True),
        ObjectiveFunction("speed", evaluate_speed, minimize=True),
        ObjectiveFunction("authenticity", evaluate_authenticity, minimize=True),
    ]

    # Parameter space
    parameter_space = {
        "noise_reduction": (0.0, 1.0),
        "declipping": (0.0, 1.0),
        "model_complexity": (0.1, 1.0),
        "n_iterations": (5, 50),
        "eq_strength": (0.0, 1.0),
    }

    # Create optimizer
    optimizer = NSGAII(
        objectives=objectives,
        parameter_space=parameter_space,
        population_size=50,
        n_generations=30,
        crossover_prob=0.9,
        mutation_prob=0.1,
    )

    return optimizer


# Example usage
if __name__ == "__main__":
    # Create multi-objective optimizer for audio restoration
    optimizer = create_audio_restoration_moo()

    # Run optimization
    pareto_front = optimizer.optimize()

    logger.debug("\nPareto front has %s solutions:", len(pareto_front))
    for i, ind in enumerate(pareto_front[:5]):  # Show first 5
        logger.debug("\nSolution %s:", i + 1)
        logger.debug("  Parameters: %s", ind.parameters)
        logger.debug("  Objectives: %s", ind.objectives)

    # Select solution with preference for quality
    preferences = {
        "quality": 2.0,  # High priority
        "speed": 1.0,  # Medium priority
        "authenticity": 1.5,  # Medium-high priority
    }

    selected = optimizer.select_solution(preferences)
    logger.debug("\nSelected solution:")
    logger.debug("  Parameters: %s", selected.parameters)
    logger.debug("  Objectives: %s", selected.objectives)

    # Visualize (if matplotlib available)
    try:
        optimizer.visualize_pareto_front()
    except Exception as e:
        logger.debug("Visualization uebersprungen: %s", e)
