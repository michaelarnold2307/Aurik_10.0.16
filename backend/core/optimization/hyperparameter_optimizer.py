"""
Hyperparameter Optimizer für Aurik 10.0.0

Automatische Optimierung aller kritischen Parameter mittels Bayesian Optimization (Optuna).

Material-spezifische Hyperparameter-Tuning für:
- Vinyl
- Tape (Shellac, Cassette, Reel-to-Reel)
- Digital
- Live Recording
- MP3/Lossy

Autor: Aurik Backend-Team
Version: 8.1
Datum: 14. Februar 2026
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import optuna  # type: ignore[reportMissingImports]
    from optuna.pruners import MedianPruner  # type: ignore[reportMissingImports]
    from optuna.samplers import TPESampler  # type: ignore[reportMissingImports]
except ImportError:
    optuna = None  # type: ignore[assignment]
    MedianPruner = None  # type: ignore[assignment]
    TPESampler = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Configuration for a hyperparameter search space."""

    # DeepFilterNet parameters
    dfn_attenuation_limit: float = 6.0
    dfn_post_filter_beta: float = 0.02
    dfn_min_db_thresh: float = -10.0
    dfn_max_db_erb_thresh: float = -10.0

    # Demucs parameters
    demucs_shifts: int = 1
    demucs_overlap: float = 0.25
    demucs_split: bool = True

    # EQ parameters
    eq_bass_gain: float = 0.0
    eq_mid_gain: float = 0.0
    eq_treble_gain: float = 0.0
    eq_presence_gain: float = 0.0

    # Compressor parameters
    comp_threshold_db: float = -20.0
    comp_ratio: float = 4.0
    comp_attack_ms: float = 5.0
    comp_release_ms: float = 100.0
    comp_knee_db: float = 6.0

    # Limiter parameters
    limiter_threshold_db: float = -0.5
    limiter_release_ms: float = 50.0

    # De-esser parameters
    deesser_frequency: float = 6000.0
    deesser_threshold_db: float = -15.0
    deesser_ratio: float = 3.0

    # Stereo enhancement
    stereo_width: float = 1.0
    stereo_bass_mono: bool = True

    # Reverb removal
    reverb_reduction: float = 0.5

    # Musical goals weights
    goal_brillanz_weight: float = 1.0
    goal_waerme_weight: float = 1.0
    goal_natuerlichkeit_weight: float = 1.0
    goal_authentizitaet_weight: float = 1.0
    goal_emotionalitaet_weight: float = 1.0
    goal_transparenz_weight: float = 1.0


class MaterialSpecificOptimizer:
    """
    Optimiert hyperparameters for specific source materials using Bayesian Optimization.
    """

    def __init__(
        self, material_type: str, storage_path: Path | None = None, n_trials: int = 100, n_jobs: int = 4
    ) -> None:
        """
        Initialisiert optimizer for specific material type.

        Args:
            material_type: One of ['vinyl', 'tape_shellac', 'tape_cassette',
                          'tape_reel', 'digital', 'live', 'mp3']
            storage_path: Path to store optimization results
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
        """
        self.material_type = material_type
        self.storage_path = storage_path or Path(f"optimization/{material_type}")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.n_trials = n_trials
        self.n_jobs = n_jobs

        # Optuna study
        self.study = None

        # Best parameters
        self.best_params = None

        logger.info("MaterialSpecificOptimizer initialisiert for %s", material_type)
        logger.info("  Storage: %s", self.storage_path)
        logger.info("  Trials: %s, Jobs: %s", n_trials, n_jobs)

    @staticmethod
    def _ensure_optuna_available() -> None:
        """Stellt sicher: Optuna is available before using optimization features."""
        if optuna is None or MedianPruner is None or TPESampler is None:
            raise RuntimeError(
                "Optuna ist nicht installiert. Bitte 'optuna' installieren, um Hyperparameter-Optimierung zu nutzen."
            )

    def define_search_space(self, trial: Any) -> HyperparameterConfig:
        """
        Define hyperparameter search space based on material type.

        Args:
            trial: Optuna trial object

        Returns:
            HyperparameterConfig with suggested values
        """
        # Base search space (common to all materials)
        config = HyperparameterConfig()

        # DeepFilterNet
        config.dfn_attenuation_limit = trial.suggest_float("dfn_attenuation_limit", 3.0, 12.0)
        config.dfn_post_filter_beta = trial.suggest_float("dfn_post_filter_beta", 0.01, 0.1, log=True)
        config.dfn_min_db_thresh = trial.suggest_float("dfn_min_db_thresh", -15.0, -5.0)
        config.dfn_max_db_erb_thresh = trial.suggest_float("dfn_max_db_erb_thresh", -15.0, -5.0)

        # Demucs
        config.demucs_shifts = trial.suggest_int("demucs_shifts", 0, 4)
        config.demucs_overlap = trial.suggest_float("demucs_overlap", 0.1, 0.5)
        config.demucs_split = trial.suggest_categorical("demucs_split", [True, False])

        # Material-specific adjustments
        if self.material_type == "vinyl":
            # Vinyl: More treble restoration, rumble filtering
            config.eq_bass_gain = trial.suggest_float("eq_bass_gain", -6.0, 0.0)
            config.eq_treble_gain = trial.suggest_float("eq_treble_gain", 0.0, 6.0)
            config.reverb_reduction = trial.suggest_float("reverb_reduction", 0.0, 0.3)

        elif self.material_type in ["tape_shellac", "tape_cassette", "tape_reel"]:
            # Tape: Handle hiss, high-frequency loss, wow/flutter
            config.eq_treble_gain = trial.suggest_float("eq_treble_gain", 0.0, 8.0)
            config.dfn_attenuation_limit = trial.suggest_float("dfn_attenuation_limit", 6.0, 15.0)

        elif self.material_type == "digital":
            # Digital artifacts: Quantization noise, clipping
            config.eq_bass_gain = trial.suggest_float("eq_bass_gain", -3.0, 3.0)
            config.eq_treble_gain = trial.suggest_float("eq_treble_gain", -3.0, 3.0)

        elif self.material_type == "live":
            # Live recording: Audience noise, room acoustics
            config.reverb_reduction = trial.suggest_float("reverb_reduction", 0.3, 0.8)
            config.dfn_attenuation_limit = trial.suggest_float("dfn_attenuation_limit", 8.0, 15.0)

        elif self.material_type == "mp3":
            # MP3 artifacts: Pre-echo, birdies
            config.eq_presence_gain = trial.suggest_float("eq_presence_gain", -2.0, 4.0)
            config.stereo_width = trial.suggest_float("stereo_width", 0.8, 1.2)

        # EQ (common parameters)
        config.eq_mid_gain = trial.suggest_float("eq_mid_gain", -3.0, 3.0)
        config.eq_presence_gain = trial.suggest_float("eq_presence_gain", -3.0, 6.0)

        # Compressor
        config.comp_threshold_db = trial.suggest_float("comp_threshold_db", -30.0, -10.0)
        config.comp_ratio = trial.suggest_float("comp_ratio", 1.5, 8.0)
        config.comp_attack_ms = trial.suggest_float("comp_attack_ms", 1.0, 20.0)
        config.comp_release_ms = trial.suggest_float("comp_release_ms", 50.0, 300.0)
        config.comp_knee_db = trial.suggest_float("comp_knee_db", 0.0, 12.0)

        # Limiter
        config.limiter_threshold_db = trial.suggest_float("limiter_threshold_db", -1.0, -0.1)
        config.limiter_release_ms = trial.suggest_float("limiter_release_ms", 20.0, 100.0)

        # De-esser
        config.deesser_frequency = trial.suggest_float("deesser_frequency", 4000.0, 8000.0)
        config.deesser_threshold_db = trial.suggest_float("deesser_threshold_db", -20.0, -10.0)
        config.deesser_ratio = trial.suggest_float("deesser_ratio", 2.0, 6.0)

        # Stereo
        config.stereo_width = trial.suggest_float("stereo_width", 0.7, 1.3)
        config.stereo_bass_mono = trial.suggest_categorical("stereo_bass_mono", [True, False])

        # Musical goals weights
        config.goal_brillanz_weight = trial.suggest_float("goal_brillanz_weight", 0.5, 1.5)
        config.goal_waerme_weight = trial.suggest_float("goal_waerme_weight", 0.5, 1.5)
        config.goal_natuerlichkeit_weight = trial.suggest_float("goal_natuerlichkeit_weight", 0.8, 1.2)
        config.goal_authentizitaet_weight = trial.suggest_float("goal_authentizitaet_weight", 0.8, 1.2)
        config.goal_emotionalitaet_weight = trial.suggest_float("goal_emotionalitaet_weight", 0.7, 1.3)
        config.goal_transparenz_weight = trial.suggest_float("goal_transparenz_weight", 0.7, 1.3)

        return config

    def objective_function(
        self, trial: Any, evaluation_dataset: list[tuple[np.ndarray, np.ndarray]], process_function: Callable
    ) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial
            evaluation_dataset: List of (input_audio, reference_audio) pairs
            process_function: Function that processes audio with given config

        Returns:
            Objective value (lower is better)
        """
        self._ensure_optuna_available()
        optuna_mod = optuna
        if optuna_mod is None:
            raise RuntimeError(
                "Optuna ist nicht installiert. Bitte 'optuna' installieren, um Hyperparameter-Optimierung zu nutzen."
            )

        # Get hyperparameter configuration
        config = self.define_search_space(trial)

        # Evaluate on dataset
        scores = []

        for input_audio, reference_audio in evaluation_dataset:
            try:
                # Process audio with current hyperparameters
                output_audio = process_function(input_audio, config)

                # Compute quality metrics
                score = self.compute_quality_score(output_audio, reference_audio)
                scores.append(score)

                # Report intermediate value for pruning
                trial.report(np.mean(scores), len(scores))

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna_mod.TrialPruned()

            except Exception as e:
                logger.warning("Trial %s fehlgeschlagen on sample: %s", trial.number, e)
                # Return high penalty for failed trials
                return 1e6

        # Return average score (lower is better, so negate quality metrics)
        avg_score = np.mean(scores)
        # NaN/Inf-Guard (§3.1)
        avg_score = np.nan_to_num(avg_score, nan=1e6, posinf=1e6, neginf=-1e6)
        return float(-avg_score)  # Negate because we want to maximize quality

    def compute_quality_score(self, output_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """
        Berechnet quality score for optimization.

        Combines multiple objective metrics:
        - PESQ
        - VISQOL
        - SI-SDR
        - Musical Goals

        Args:
            output_audio: Processed audio
            reference_audio: Reference audio

        Returns:
            Combined quality score (0-1, higher is better)
        """
        try:
            # Import quality metrics (PESQ+SI-SDR entfernt — §4.4+§10.2: Sprach-Metriken verboten)
            # compute_si_sdr entfernt — verboten §4.4+§10.2 (SI-SDR Sprach-/Trennungs-Metrik)
            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
            from plugins.visqol_plugin import score_audio as compute_visqol

            # Compute objective metrics
            visqol_score = compute_visqol(reference_audio, output_audio, sr=48000)
            # si_sdr entfernt — verboten §4.4+§10.2 (SI-SDR kein Musik-Äquivalent)

            # Compute musical goals
            checker = MusicalGoalsChecker()
            goals_result = checker.evaluate_musical_goals(output_audio, sr=48000)  # type: ignore[attr-defined]
            goals_score = goals_result["overall_score"]

            # Weighted combination (total = 1.0; SI-SDR-Gewicht auf Musical Goals umverteilt §10.2)
            quality_score = (
                0.35 * (visqol_score / 5.0)  # ViSQOL v3 audio mode — §4.4-erlaubt
                + 0.65 * goals_score  # Musical Goals §1.2 — SI-SDR-Anteil integriert
            )
            # NaN/Inf-Guard (§3.1)
            quality_score = np.nan_to_num(quality_score, nan=0.0, posinf=1.0, neginf=0.0)
            return float(np.clip(quality_score, 0.0, 1.0))

        except Exception as e:
            logger.error("konnte nicht berechnen quality Wert: %s", e)
            return 0.0

    def optimize(
        self,
        evaluation_dataset: list[tuple[np.ndarray, np.ndarray]],
        process_function: Callable,
        study_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Führt aus: optimization.

        Args:
            evaluation_dataset: Dataset for evaluation
            process_function: Function to process audio
            study_name: Name for the study

        Returns:
            Best hyperparameters and metrics
        """
        self._ensure_optuna_available()
        optuna_mod = optuna
        sampler_cls = TPESampler
        pruner_cls = MedianPruner
        if optuna_mod is None or sampler_cls is None or pruner_cls is None:
            raise RuntimeError(
                "Optuna ist nicht installiert. Bitte 'optuna' installieren, um Hyperparameter-Optimierung zu nutzen."
            )

        study_name = study_name or f"aurik_8_0_{self.material_type}_{int(time.time())}"

        # Create study
        sampler = sampler_cls(seed=42)
        pruner = pruner_cls(n_startup_trials=10, n_warmup_steps=5)

        self.study = optuna_mod.create_study(
            study_name=study_name,
            direction="minimize",  # Minimize negative quality score
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{self.storage_path}/optuna.db",
            load_if_exists=True,
        )

        logger.info("Starte optimization: %s", study_name)
        logger.info("  Dataset size: %s", len(evaluation_dataset))
        logger.info("  Trials: %s", self.n_trials)

        # Optimize
        self.study.optimize(  # type: ignore[attr-defined]
            lambda trial: self.objective_function(trial, evaluation_dataset, process_function),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        # Get best parameters
        self.best_params = self.study.best_params  # type: ignore[attr-defined]
        best_value = -self.study.best_value  # type: ignore[attr-defined]  # Negate back to get quality score

        logger.info("Optimization abgeschlossen!")
        logger.info("  Best quality Wert: %.4f", best_value)
        logger.info("  Best parameters: %s", self.best_params)

        # Save results
        self.save_best_parameters()
        self.save_optimization_report()

        return {
            "best_params": self.best_params,
            "best_score": best_value,
            "n_trials": len(self.study.trials),  # type: ignore[attr-defined]
            "study_name": study_name,
        }

    def save_best_parameters(self) -> None:
        """Speichert best parameters to file."""
        output_path = self.storage_path / f"best_params_{self.material_type}.yaml"

        with open(output_path, "w") as f:
            yaml.dump(self.best_params, f, default_flow_style=False, sort_keys=True)

        logger.info("Best parameters gespeichert: %s", output_path)

    def save_optimization_report(self) -> None:
        """Speichert detailed optimization report."""
        if self.study is None:
            logger.warning("No study verfuegbar for report generation")
            return

        report = {
            "material_type": self.material_type,
            "n_trials": len(self.study.trials),
            "best_value": float(-self.study.best_value),
            "best_params": self.best_params,
            "best_trial": self.study.best_trial.number,
            "optimization_history": [],
        }

        # Add trial history
        for trial in self.study.trials:
            if str(trial.state) == "TrialState.COMPLETE":
                report["optimization_history"].append(
                    {
                        "trial": trial.number,
                        "value": float(-trial.value),
                        "params": trial.params,
                        "duration": trial.duration.total_seconds() if trial.duration else None,
                    }
                )

        output_path = self.storage_path / f"optimization_report_{self.material_type}.json"

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Optimization report gespeichert: %s", output_path)

        # Generate plots (if optuna visualization is available)
        try:
            import optuna.visualization as vis  # type: ignore[reportMissingImports]

            # Optimization history plot
            fig = vis.plot_optimization_history(self.study)
            fig.write_html(self.storage_path / f"optimization_history_{self.material_type}.html")

            # Parameter importance plot
            fig = vis.plot_param_importances(self.study)
            fig.write_html(self.storage_path / f"param_importances_{self.material_type}.html")

            # Slice plot
            fig = vis.plot_slice(self.study)
            fig.write_html(self.storage_path / f"slice_plot_{self.material_type}.html")

            logger.info("Visualization plots erzeugt")

        except ImportError:
            logger.warning("Optuna visualization not verfuegbar, skipping plots")

    def load_best_parameters(self) -> dict[str, Any] | None:
        """Lädt best parameters from file."""
        params_path = self.storage_path / f"best_params_{self.material_type}.yaml"

        if not params_path.exists():
            logger.warning("No gespeichert parameters found: %s", params_path)
            return None

        with open(params_path) as f:
            params = yaml.safe_load(f)

        logger.info("Best parameters geladen: %s", params_path)

        return params  # type: ignore[no-any-return]


class MultiMaterialOptimizer:
    """
    Optimiert hyperparameters for all material types in parallel.
    """

    def __init__(self, storage_path: Path | None = None, n_trials_per_material: int = 100) -> None:
        self.storage_path = storage_path or Path("optimization")
        self.n_trials_per_material = n_trials_per_material

        self.material_types = ["vinyl", "tape_shellac", "tape_cassette", "tape_reel", "digital", "live", "mp3"]

        self.optimizers = {
            material: MaterialSpecificOptimizer(
                material_type=material, storage_path=self.storage_path / material, n_trials=n_trials_per_material
            )
            for material in self.material_types
        }

        logger.info("MultiMaterialOptimizer initialisiert for %s materials", len(self.material_types))

    def optimize_all(
        self, datasets: dict[str, list[tuple[np.ndarray, np.ndarray]]], process_functions: dict[str, Callable]
    ) -> dict[str, dict[str, Any]]:
        """
        Optimiert all materials.

        Args:
            datasets: Dict mapping material_type -> evaluation dataset
            process_functions: Dict mapping material_type -> processing function

        Returns:
            Dict of optimization results per material
        """
        results = {}

        for material in self.material_types:
            logger.info("\n%s", "=" * 80)
            logger.info("Optimizing %s", material)
            logger.info("%s\n", "=" * 80)

            optimizer = self.optimizers[material]
            dataset = datasets.get(material, [])
            process_func = process_functions.get(material)

            if not dataset or process_func is None:
                logger.warning("Skipping %s: no dataset or verarbeiten function", material)
                continue

            result = optimizer.optimize(evaluation_dataset=dataset, process_function=process_func)

            results[material] = result

        # Generate summary report
        self.generate_summary_report(results)

        return results

    def generate_summary_report(self, results: dict[str, dict[str, Any]]) -> np.ndarray:  # type: ignore[return]
        """Generiert summary report for all materials."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "materials": results,
            "overall_stats": {
                "total_trials": sum(r["n_trials"] for r in results.values()),
                "avg_quality_score": np.mean([r["best_score"] for r in results.values()]),
                "best_material": max(results.items(), key=lambda x: x[1]["best_score"])[0],
                "worst_material": min(results.items(), key=lambda x: x[1]["best_score"])[0],
            },
        }

        output_path = self.storage_path / "optimization_summary.json"

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\nOptimization summary gespeichert: %s", output_path)
        logger.info("Average quality Wert: %.4f", summary["overall_stats"]["avg_quality_score"])  # type: ignore[index]
        logger.info("Best material: %s", summary["overall_stats"]["best_material"])  # type: ignore[index]


# Example usage
if __name__ == "__main__":
    # Example: Optimize for vinyl
    optimizer = MaterialSpecificOptimizer(material_type="vinyl", n_trials=50)

    # Dummy dataset (in practice, load real audio files)
    dummy_dataset = [(np.random.randn(48000 * 2), np.random.randn(48000 * 2)) for _ in range(10)]

    # Dummy process function (in practice, use actual Aurik pipeline)
    def dummy_process(audio, config) -> np.ndarray:
        # Simulate processing
        return audio * 0.9  # type: ignore[no-any-return]

    # Run optimization
    results = optimizer.optimize(evaluation_dataset=dummy_dataset, process_function=dummy_process)

    logger.debug("\nOptimization abgeschlossen!")
    logger.debug("Best Wert: %.4f", results["best_score"])
    logger.debug("Best params: %s", results["best_params"])
