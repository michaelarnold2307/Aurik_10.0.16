"""
Optimization Integration für Aurik 10.0.0 Adaptive Pipeline

Integriert die neuen Optimierungs-Features:
Phase 1:
1. Perceptual Loss zur Qualitätsbewertung
2. E2E-optimierte DSP-Parameter
3. Material-spezifische Hyperparameter

Phase 2-4:
4. Neural Architecture Search (NAS)
5. Advanced Ensemble Strategies
6. Multi-Objective Optimization
7. Uncertainty Quantification
8. Automated Data Augmentation

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
import torch.nn as nn
import yaml

from backend.core.optimization.advanced_ensemble import AdvancedEnsemble, EnsembleMember
from backend.core.optimization.automated_augmentation import AutoAugment, RandAugment
from backend.core.optimization.multi_objective import NSGAII, create_audio_restoration_moo

# Phase 2-4 Imports
from backend.core.optimization.neural_architecture_search import AudioNASNetwork

# Phase 1 Imports
from backend.core.optimization.perceptual_loss import CombinedPerceptualLoss
from backend.core.optimization.uncertainty_quantification import UncertaintyQuantifier

logger = logging.getLogger(__name__)


class OptimizationIntegration:
    """
    Integration Layer zwischen Optimization Framework und Adaptive Pipeline.

    Ermöglicht:
    - Laden von optimierten Hyperparametern pro Material
    - Perceptual Loss-basierte Qualitätsbewertung
    - Dynamische Parameter-Anpassung
    """

    def __init__(
        self,
        optimization_base_path: Path | None = None,
        sr: int = 48000,
        device: str = "cpu",  # §9.5: Aurik 10.0.0 — ausschließlich CPU, kein CUDA/ROCm/Metal
    ) -> None:
        """
        Initialisiert optimization integration.

        Args:
            optimization_base_path: Base path to optimization results
            sr: Sample rate
            device: Device for torch operations
        """
        self.optimization_base_path = optimization_base_path or Path("optimization")
        self.sr = sr
        self.device = device

        # Phase 1: Load perceptual loss (for quality assessment)
        self.perceptual_loss = CombinedPerceptualLoss(sr=sr).to(device)

        # Cache for material-specific parameters
        self.material_params_cache: dict[str, dict[str, Any]] = {}

        # Load all available optimized parameters
        self._load_all_material_parameters()

        # Phase 2-4: Initialize advanced optimization components (lazy loading)
        self._nas_network_cache: dict[str, AudioNASNetwork] = {}
        self._ensemble_cache: dict[str, AdvancedEnsemble] = {}
        self._uncertainty_quantifier_cache: dict[str, UncertaintyQuantifier] = {}
        self._augmentation_cache: dict[str, RandAugment | AutoAugment] = {}
        self._moo_optimizer: NSGAII | None = None

        logger.info("OptimizationIntegration initialisiert (v8.2)")
        logger.info("  Optimization path: %s", self.optimization_base_path)
        logger.info("  geladen parameters for %s materials", len(self.material_params_cache))
        logger.info("  Verarbeitungsschritt 1: Perceptual Loss, E2E Optimization, Hyperparameter Optimization")
        logger.info("  Verarbeitungsschritt 2-4: NAS, Ensemble, MOO, Uncertainty, Augmentation verfuegbar")

    def _load_all_material_parameters(self) -> bool:
        """Lädt optimized parameters for all available materials."""
        material_types = ["vinyl", "tape_shellac", "tape_cassette", "tape_reel", "digital", "live", "mp3"]
        loaded_any = False

        for material in material_types:
            params = self._load_material_parameters(material)
            if params:
                self.material_params_cache[material] = params
                logger.info("  geladen optimiert parameters for: %s", material)
                loaded_any = True

        return loaded_any

    def _load_material_parameters(self, material_type: str) -> dict[str, Any] | None:
        """Lädt optimized parameters for specific material."""
        params_path = self.optimization_base_path / material_type / f"best_params_{material_type}.yaml"

        if not params_path.exists():
            logger.debug("No optimiert parameters found for %s: %s", material_type, params_path)
            return None

        try:
            with open(params_path, encoding="utf-8") as f:
                params = yaml.safe_load(f)
            return params  # type: ignore[no-any-return]
        except Exception as e:
            logger.error("konnte nicht laden parameters for %s: %s", material_type, e)
            return None

    def get_optimized_parameters(self, material_type: str, fallback_to_defaults: bool = True) -> dict[str, Any]:
        """
        Gibt zurück: optimized parameters for material type.

        Args:
            material_type: Material type (vinyl, tape_shellac, etc.)
            fallback_to_defaults: If True, return defaults if optimized params not found

        Returns:
            Dictionary with optimized parameters
        """
        # Check cache
        if material_type in self.material_params_cache:
            return self.material_params_cache[material_type].copy()

        # Try to load
        params = self._load_material_parameters(material_type)
        if params:
            self.material_params_cache[material_type] = params
            return params.copy()

        # Fallback to defaults
        if fallback_to_defaults:
            logger.warning("Using default parameters for %s", material_type)
            return self._get_default_parameters()

        return {}

    def _get_default_parameters(self) -> dict[str, Any]:
        """Gibt zurück: default parameter set."""
        return {
            # DeepFilterNet
            "dfn_attenuation_limit": 6.0,
            "dfn_post_filter_beta": 0.02,
            "dfn_min_db_thresh": -10.0,
            "dfn_max_db_erb_thresh": -10.0,
            # Demucs
            "demucs_shifts": 1,
            "demucs_overlap": 0.25,
            "demucs_split": True,
            # EQ
            "eq_bass_gain": 0.0,
            "eq_mid_gain": 0.0,
            "eq_treble_gain": 0.0,
            "eq_presence_gain": 0.0,
            # Compressor
            "comp_threshold_db": -20.0,
            "comp_ratio": 4.0,
            "comp_attack_ms": 5.0,
            "comp_release_ms": 100.0,
            "comp_knee_db": 6.0,
            # Limiter
            "limiter_threshold_db": -0.5,
            "limiter_release_ms": 50.0,
            # De-esser
            "deesser_frequency": 6000.0,
            "deesser_threshold_db": -15.0,
            "deesser_ratio": 3.0,
            # Stereo
            "stereo_width": 1.0,
            "stereo_bass_mono": True,
            # Reverb
            "reverb_reduction": 0.5,
            # Musical goals weights
            "goal_brillanz_weight": 1.0,
            "goal_waerme_weight": 1.0,
            "goal_natuerlichkeit_weight": 1.0,
            "goal_authentizitaet_weight": 1.0,
            "goal_emotionalitaet_weight": 1.0,
            "goal_transparenz_weight": 1.0,
        }

    def apply_optimized_parameters_to_context(self, context: dict[str, Any], material_type: str) -> dict[str, Any]:
        """
        Wendet an: optimized parameters to processing context.

        Args:
            context: Processing context dictionary
            material_type: Detected material type

        Returns:
            Updated context with optimized parameters
        """
        # Get optimized parameters
        params = self.get_optimized_parameters(material_type)

        # Create optimized_params section in context
        if "optimized_params" not in context:
            context["optimized_params"] = {}

        context["optimized_params"].update(params)

        # Update specific processor configurations
        self._update_deepfilternet_config(context, params)
        self._update_demucs_config(context, params)
        self._update_eq_config(context, params)
        self._update_compressor_config(context, params)
        self._update_musical_goals_weights(context, params)

        logger.info("angewendet optimiert parameters for material: %s", material_type)

        return context

    def _update_deepfilternet_config(self, context: dict[str, Any], params: dict[str, Any]) -> None:
        """Aktualisiert DeepFilterNet configuration."""
        if "ml_config" not in context:
            context["ml_config"] = {}

        if "deepfilternet" not in context["ml_config"]:
            context["ml_config"]["deepfilternet"] = {}

        dfn_config = context["ml_config"]["deepfilternet"]

        dfn_config["attenuation_limit"] = params.get("dfn_attenuation_limit", 6.0)
        dfn_config["post_filter_beta"] = params.get("dfn_post_filter_beta", 0.02)
        dfn_config["min_db_thresh"] = params.get("dfn_min_db_thresh", -10.0)
        dfn_config["max_db_erb_thresh"] = params.get("dfn_max_db_erb_thresh", -10.0)

    def _update_demucs_config(self, context: dict[str, Any], params: dict[str, Any]) -> None:
        """Aktualisiert Demucs configuration."""
        if "ml_config" not in context:
            context["ml_config"] = {}

        if "demucs" not in context["ml_config"]:
            context["ml_config"]["demucs"] = {}

        demucs_config = context["ml_config"]["demucs"]

        demucs_config["shifts"] = params.get("demucs_shifts", 1)
        demucs_config["overlap"] = params.get("demucs_overlap", 0.25)
        demucs_config["split"] = params.get("demucs_split", True)

    def _update_eq_config(self, context: dict[str, Any], params: dict[str, Any]) -> None:
        """Aktualisiert EQ configuration."""
        if "dsp_config" not in context:
            context["dsp_config"] = {}

        if "eq" not in context["dsp_config"]:
            context["dsp_config"]["eq"] = {}

        eq_config = context["dsp_config"]["eq"]

        eq_config["bass_gain"] = params.get("eq_bass_gain", 0.0)
        eq_config["mid_gain"] = params.get("eq_mid_gain", 0.0)
        eq_config["treble_gain"] = params.get("eq_treble_gain", 0.0)
        eq_config["presence_gain"] = params.get("eq_presence_gain", 0.0)

    def _update_compressor_config(self, context: dict[str, Any], params: dict[str, Any]):
        """Aktualisiert compressor configuration."""
        if "dsp_config" not in context:
            context["dsp_config"] = {}

        if "compressor" not in context["dsp_config"]:
            context["dsp_config"]["compressor"] = {}

        comp_config = context["dsp_config"]["compressor"]

        comp_config["threshold_db"] = params.get("comp_threshold_db", -20.0)
        comp_config["ratio"] = params.get("comp_ratio", 4.0)
        comp_config["attack_ms"] = params.get("comp_attack_ms", 5.0)
        comp_config["release_ms"] = params.get("comp_release_ms", 100.0)
        comp_config["knee_db"] = params.get("comp_knee_db", 6.0)

    def _update_musical_goals_weights(self, context: dict[str, Any], params: dict[str, Any]) -> bool:  # type: ignore[return]
        """Aktualisiert musical goals weights."""
        if "musical_goals" not in context:
            context["musical_goals"] = {}

        if "weights" not in context["musical_goals"]:
            context["musical_goals"]["weights"] = {}

        weights = context["musical_goals"]["weights"]

        weights["brillanz"] = params.get("goal_brillanz_weight", 1.0)
        weights["waerme"] = params.get("goal_waerme_weight", 1.0)
        weights["natuerlichkeit"] = params.get("goal_natuerlichkeit_weight", 1.0)
        weights["authentizitaet"] = params.get("goal_authentizitaet_weight", 1.0)
        weights["emotionalitaet"] = params.get("goal_emotionalitaet_weight", 1.0)
        weights["transparenz"] = params.get("goal_transparenz_weight", 1.0)

    def compute_perceptual_quality(
        self, output_audio: np.ndarray, reference_audio: np.ndarray | None = None, return_details: bool = False
    ) -> float | tuple[float, dict[str, float]]:
        """
        Berechnet perceptual quality score.

        Args:
            output_audio: Processed audio
            reference_audio: Optional reference for comparison
            return_details: If True, return detailed breakdown

        Returns:
            Quality score (0-1) or (score, details)
        """
        if reference_audio is None:
            # No reference: use self-consistency metrics
            logger.warning("No Referenz audio provided, quality assessment limited")
            return 0.5 if not return_details else (0.5, {"warning": "no_reference"})  # type: ignore[return-value,dict-item]

        # Convert to torch tensors (robust against array subclasses/views).
        output_np = np.asarray(output_audio, dtype=np.float32)
        reference_np = np.asarray(reference_audio, dtype=np.float32)
        output_tensor = torch.as_tensor(output_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        reference_tensor = (
            torch.as_tensor(reference_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        )

        # Ensure same length
        min_len = min(output_tensor.shape[-1], reference_tensor.shape[-1])
        output_tensor = output_tensor[..., :min_len]
        reference_tensor = reference_tensor[..., :min_len]

        # Compute perceptual loss
        with torch.no_grad():
            loss, details = self.perceptual_loss(output_tensor, reference_tensor, return_details=True)

        # Convert loss to quality score (inverse relationship)
        # Lower loss = higher quality
        quality_score = 1.0 / (1.0 + loss.item())

        if return_details:
            details["quality_score"] = quality_score
            return quality_score, details

        return quality_score  # type: ignore[no-any-return]

    def recommend_processing_strategy(self, context: dict[str, Any], material_type: str) -> dict[str, Any]:
        """
        Recommend processing strategy based on material and optimized parameters.

        Args:
            context: Processing context
            material_type: Detected material type

        Returns:
            Recommended processing strategy
        """
        params = self.get_optimized_parameters(material_type)
        detected_artifacts = context.get("detected_artifacts", []) if isinstance(context, dict) else []

        strategy = {
            "material_type": material_type,
            "recommended_models": [],
            "recommended_dsp_chain": [],
            "processing_order": [],
            "quality_target": "high",
        }

        # Material-specific model recommendations
        if material_type == "vinyl":
            strategy["recommended_models"] = [
                "deepfilternet_v3",  # For clicks/pops
                "banquet_vinyl",  # Vinyl-specific restoration
                "demucs_v4",  # For complex artifacts
            ]
            strategy["recommended_dsp_chain"] = [
                "highpass_rumble",
                "declicker",
                "dehisser",
                "eq_riaa",
                "stereo_enhancer",
            ]

        elif "tape" in material_type:
            strategy["recommended_models"] = [
                "deepfilternet_v3",  # For tape hiss
                "mp_senet",  # Advanced denoising (§4.4: MP-SENet 2023 ersetzt FullSubNet+)
                "flashsr",  # For high-frequency restoration
            ]
            strategy["recommended_dsp_chain"] = [
                "dehisser",
                "eq_tape_compensation",
                "wow_flutter_correction",
                "compressor",
            ]

        elif material_type == "digital":
            strategy["recommended_models"] = [
                "mp_senet",
                "mdx23c",
            ]  # Digital artifacts — §4.4: MP-SENet 2023 ersetzt DCCRN
            strategy["recommended_dsp_chain"] = ["dithering_removal", "eq_digital_correction", "limiter"]

        elif material_type == "live":
            strategy["recommended_models"] = [
                "demucs_v4",  # Separate audience/performance
                "mp_senet",  # Background noise (§4.4: MP-SENet 2023 ersetzt FullSubNet+)
                "gacela",  # Reverb handling
            ]
            strategy["recommended_dsp_chain"] = [
                "crowd_reduction",
                "reverb_reduction",
                "eq_live_compensation",
                "compressor",
            ]

        elif material_type == "mp3":
            strategy["recommended_models"] = [
                "flashsr",  # Restore lost frequencies
                "resemble_enhance",  # General enhancement
            ]
            strategy["recommended_dsp_chain"] = ["eq_mp3_compensation", "stereo_enhancer", "harmonic_enhancer"]

        # Add optimized parameters to strategy
        strategy["optimized_params"] = params  # type: ignore[assignment]

        logger.info("Processing strategy recommended for %s", material_type)
        logger.info("  Models: %s", strategy["recommended_models"])
        logger.info("  DSP chain: %s", strategy["recommended_dsp_chain"])
        if detected_artifacts:
            logger.debug("  Context artifacts: %s", detected_artifacts)

        return strategy

    # =========================================================================
    # Phase 2-4: Advanced Optimization Methods
    # =========================================================================

    def get_nas_network(
        self,
        material_type: str,
        input_channels: int = 1,
        initial_channels: int = 16,
        n_cells: int = 4,
        n_nodes: int = 4,
        force_new: bool = False,
    ) -> AudioNASNetwork:
        """
        Gibt zurück: or create Neural Architecture Search network for material type.

        Args:
            material_type: Material type (vinyl, tape, etc.)
            input_channels: Input audio channels
            initial_channels: Initial network channels
            n_cells: Number of DARTS cells
            n_nodes: Nodes per cell
            force_new: Force creation of new network

        Returns:
            AudioNASNetwork instance
        """
        cache_key = f"{material_type}_{input_channels}_{initial_channels}_{n_cells}_{n_nodes}"

        if not force_new and cache_key in self._nas_network_cache:
            logger.debug("Using zwischengespeichert NAS network for %s", material_type)
            return self._nas_network_cache[cache_key]

        # Create new network
        network = AudioNASNetwork(
            in_channels=input_channels, init_channels=initial_channels, n_cells=n_cells, n_nodes=n_nodes
        ).to(self.device)

        # Try to load pretrained weights if available
        weights_path = self.optimization_base_path / material_type / f"nas_network_{material_type}.pth"
        if weights_path.exists():
            try:
                network.load_state_dict(
                    torch.load(  # nosec B614 — interner Checkpoint aus models/
                        weights_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )
                logger.info("geladen pretrained NAS network for %s", material_type)
            except Exception as e:
                logger.warning("konnte nicht laden NAS weights for %s: %s", material_type, e)

        self._nas_network_cache[cache_key] = network
        logger.info("erstellt NAS network for %s: %s cells, %s nodes", material_type, n_cells, n_nodes)

        return network  # type: ignore[no-any-return]

    def create_ensemble(
        self,
        members: list[nn.Module],
        strategy: str = "stacking",
        material_type: str | None = None,
        meta_features_dim: int = 4,
        output_dim: int = 1,
    ) -> AdvancedEnsemble:
        """
        Erstellt advanced ensemble from model members.

        Args:
            members: List of model members
            strategy: Ensemble strategy ('stacking', 'weighted', 'dynamic', 'moe')
            material_type: Optional material type for caching
            meta_features_dim: Dimension of meta features
            output_dim: Output dimension

        Returns:
            AdvancedEnsemble instance
        """
        cache_key = (
            f"{material_type}_{strategy}_{len(members)}_{meta_features_dim}_{output_dim}" if material_type else None
        )

        if cache_key and cache_key in self._ensemble_cache:
            logger.debug("Using zwischengespeichert ensemble for %s", material_type)
            return self._ensemble_cache[cache_key]

        # Wrap members if needed
        wrapped_members = []
        for i, model in enumerate(members):
            if isinstance(model, EnsembleMember):
                wrapped_members.append(model)
            else:
                wrapped_members.append(EnsembleMember(model=model, name=f"member_{i}", weight=1.0 / len(members)))  # type: ignore[arg-type]

        # Create ensemble
        ensemble = AdvancedEnsemble(ensemble_members=wrapped_members, strategy=strategy, device=self.device)  # type: ignore[arg-type]

        if cache_key:
            self._ensemble_cache[cache_key] = ensemble

        logger.info(
            "erstellt %s ensemble with %s members (meta_dim=%s, out_dim=%s)",
            strategy,
            len(members),
            meta_features_dim,
            output_dim,
        )

        return ensemble

    def get_multi_objective_optimizer(
        self, material_type: str, population_size: int = 50, n_objectives: int = 3
    ) -> NSGAII:
        """
        Gibt zurück: Multi-Objective Optimizer for material type.

        Args:
            material_type: Material type
            population_size: Population size
            n_objectives: Number of objectives

        Returns:
            NSGAII optimizer instance
        """
        if self._moo_optimizer is None:
            self._moo_optimizer = create_audio_restoration_moo()
            logger.info("erstellt Multi-Objective Optimizer (generic audio restoration)")
        logger.debug(
            "MOO request: material=%s pop=%s objectives=%s",
            material_type,
            population_size,
            n_objectives,
        )

        return self._moo_optimizer

    def get_uncertainty_quantifier(
        self,
        model: nn.Module,
        method: str = "mc_dropout",
        material_type: str | None = None,
        n_samples: int = 20,
        force_new: bool = False,
    ) -> UncertaintyQuantifier:
        """
        Gibt zurück: Uncertainty Quantifier for model.

        Args:
            model: Neural network model
            method: Uncertainty method ('mc_dropout', 'bayesian', 'ensemble')
            material_type: Optional material type for caching
            n_samples: Number of samples for uncertainty estimation
            force_new: Force creation of new quantifier

        Returns:
            UncertaintyQuantifier instance
        """
        cache_key = f"{material_type}_{method}_{n_samples}" if material_type else None

        if not force_new and cache_key and cache_key in self._uncertainty_quantifier_cache:
            logger.debug("Using zwischengespeichert uncertainty quantifier for %s", material_type)
            return self._uncertainty_quantifier_cache[cache_key]

        # Create quantifier
        quantifier = UncertaintyQuantifier(model=model, method=method, n_samples=n_samples, device=self.device)

        if cache_key:
            self._uncertainty_quantifier_cache[cache_key] = quantifier

        logger.info("erstellt %s uncertainty quantifier", method)

        return quantifier

    def get_augmentation_policy(
        self,
        material_type: str,
        strategy: str = "rand",
        n_ops: int = 2,
        magnitude: float = 0.5,
        force_new: bool = False,
    ) -> RandAugment | AutoAugment:
        """
        Gibt zurück: or create data augmentation policy for material type.

        Args:
            material_type: Material type
            strategy: Augmentation strategy ('rand' or 'auto')
            n_ops: Number of operations
            magnitude: Magnitude of augmentations
            force_new: Force creation of new policy

        Returns:
            RandAugment or AutoAugment instance
        """
        cache_key = f"{material_type}_{strategy}_{n_ops}_{magnitude}"

        if not force_new and cache_key in self._augmentation_cache:
            logger.debug("Using zwischengespeichert augmentation policy for %s", material_type)
            return self._augmentation_cache[cache_key]

        # Create augmentation policy
        if strategy == "rand":
            policy = RandAugment(n_ops=n_ops, magnitude=magnitude, material_type=material_type)
        elif strategy == "auto":
            policy = AutoAugment(n_policies=5, n_ops_per_policy=n_ops, material_type=material_type)  # type: ignore[assignment]

            # Try to load pretrained policies
            policy_path = self.optimization_base_path / material_type / f"augmentation_policy_{material_type}.json"
            if policy_path.exists():
                try:
                    policy.load_policies(str(policy_path))  # type: ignore[attr-defined]
                    logger.info("geladen pretrained augmentation policy for %s", material_type)
                except Exception as e:
                    logger.warning("konnte nicht laden augmentation policy: %s", e)
        else:
            raise ValueError(f"Unknown augmentation strategy: {strategy}")

        self._augmentation_cache[cache_key] = policy
        logger.info("erstellt %s augmentation policy for %s", strategy, material_type)

        return policy


# Singleton (Thread-safe Double-Checked Locking — §3.x Aurik Spec)
_optimization_integration_state: dict[str, OptimizationIntegration | None] = {"instance": None}
_optimization_integration_lock = threading.Lock()


def get_optimization_integration(
    optimization_base_path: Path | None = None, sr: int = 48000
) -> OptimizationIntegration:
    """
    Gibt zurück: or create OptimizationIntegration singleton (Double-Checked Locking).

    Args:
        optimization_base_path: Base path to optimization results
        sr: Sample rate

    Returns:
        OptimizationIntegration instance
    """
    _instance = _optimization_integration_state["instance"]
    if _instance is None:
        with _optimization_integration_lock:
            _instance = _optimization_integration_state["instance"]
            if _instance is None:
                _instance = OptimizationIntegration(optimization_base_path=optimization_base_path, sr=sr)
                _optimization_integration_state["instance"] = _instance
    return _instance


# Example usage
if __name__ == "__main__":
    # Initialize integration
    integration = OptimizationIntegration()

    # Get optimized parameters for vinyl
    vinyl_params = integration.get_optimized_parameters("vinyl")
    logger.debug("Vinyl parameters: %s", vinyl_params)

    # Apply to context
    demo_context = {"material_type": "vinyl", "detected_artifacts": ["clicks", "pops", "rumble"]}

    demo_context = integration.apply_optimized_parameters_to_context(demo_context, "vinyl")
    logger.debug("\nUpdated context: %s", demo_context)

    # Recommend strategy
    demo_strategy = integration.recommend_processing_strategy(demo_context, "vinyl")
    logger.debug("\nRecommended strategy: %s", demo_strategy)
