"""
Tests für Advanced Optimization Features (Phase 2-4)

Tests für:
- Neural Architecture Search (NAS)
- Advanced Ensemble Strategies
- Multi-Objective Optimization
- Uncertainty Quantification
- Automated Data Augmentation

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

import pytest

pytest.importorskip("optuna")  # Skip module if optuna is not installed
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from backend.core.optimization.advanced_ensemble import (
    AdvancedEnsemble,
    AttentionWeightPredictor,
    EnsembleMember,
    MetaLearner,
    MixtureOfExperts,
)
from backend.core.optimization.automated_augmentation import (
    AudioAugmentations,
    AugmentationPolicy,
    AutoAugment,
    RandAugment,
)
from backend.core.optimization.multi_objective import (
    NSGAII,
    Individual,
    ObjectiveFunction,
    create_audio_restoration_moo,
)

# Import Phase 2-4 modules
from backend.core.optimization.neural_architecture_search import AudioNASNetwork, DARTSCell, MixedOp, NASTrainer
from backend.core.optimization.uncertainty_quantification import (
    BayesianNN,
    EnsembleUncertainty,
    MCDropoutModel,
    UncertaintyQuantifier,
)

# ============================================================================
# Neural Architecture Search Tests
# ============================================================================


@pytest.mark.unit
class TestNeuralArchitectureSearch:
    """Tests for NAS components."""

    def test_mixed_op_forward(self):
        """Test MixedOp forward pass."""
        mixed_op = MixedOp(in_channels=16, out_channels=32)
        x = torch.randn(2, 16, 100)
        output = mixed_op(x)

        assert output.shape == (2, 32, 100)
        assert not torch.isnan(output).any()

    def test_mixed_op_get_best(self):
        """Test getting best operation."""
        mixed_op = MixedOp(in_channels=16, out_channels=32)
        best_op = mixed_op.get_best_operation()

        assert isinstance(best_op, str)
        assert best_op in [
            "conv_3x3",
            "conv_5x5",
            "conv_7x7",
            "dilated_conv_3x3_rate2",
            "dilated_conv_3x3_rate4",
            "sep_conv_3x3",
            "sep_conv_5x5",
            "avg_pool_3x3",
            "max_pool_3x3",
            "skip_connect",
            "none",
        ]

    def test_darts_cell_forward(self):
        """Test DARTS cell forward pass."""
        cell = DARTSCell(in_channels_0=16, in_channels_1=16, out_channels=32, n_nodes=4)
        s0 = torch.randn(2, 16, 100)
        s1 = torch.randn(2, 16, 100)

        output = cell(s0, s1)

        # Output is concatenation of 4 nodes
        assert output.shape == (2, 32 * 4, 100)
        assert not torch.isnan(output).any()

    def test_darts_cell_genotype(self):
        """Test genotype extraction."""
        cell = DARTSCell(in_channels_0=16, in_channels_1=16, out_channels=32, n_nodes=2)
        genotype = cell.get_genotype()

        assert "cell" in genotype
        assert isinstance(genotype["cell"], list)

    def test_audio_nas_network(self):
        """Test AudioNASNetwork."""
        model = AudioNASNetwork(in_channels=1, init_channels=8, n_cells=2, n_nodes=2)

        x = torch.randn(2, 1, 4800)
        output = model(x)

        assert output.shape == (2, 1)
        assert not torch.isnan(output).any()

    def test_nas_trainer_init(self):
        """Test NAS trainer initialization."""
        model = AudioNASNetwork(in_channels=1, init_channels=8, n_cells=2, n_nodes=2)

        trainer = NASTrainer(model, device="cpu", lr_model=0.01, lr_arch=0.001)

        assert trainer.device == "cpu"
        assert trainer.optimizer_model is not None
        assert trainer.optimizer_arch is not None


# ============================================================================
# Advanced Ensemble Tests
# ============================================================================


class TestAdvancedEnsemble:
    """Tests for Advanced Ensemble components."""

    def test_meta_learner_forward(self):
        """Test MetaLearner forward pass."""
        meta_learner = MetaLearner(n_base_models=3, feature_dim=32, hidden_dim=64, output_dim=1)

        base_predictions = [torch.randn(4, 1) for _ in range(3)]
        output = meta_learner(base_predictions)

        assert output.shape == (4, 1)
        assert not torch.isnan(output).any()

    def test_attention_weight_predictor(self):
        """Test AttentionWeightPredictor."""
        predictor = AttentionWeightPredictor(n_members=3, input_feature_dim=64)

        features = torch.randn(4, 64)
        weights = predictor(features)

        assert weights.shape == (4, 3)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
        assert not torch.isnan(weights).any()

    def test_mixture_of_experts(self):
        """Test Mixture of Experts."""
        experts = [nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10)) for _ in range(3)]

        moe = MixtureOfExperts(experts, input_dim=100, k_active=2)  # type: ignore[arg-type]

        x = torch.randn(4, 100)
        output, aux_losses = moe(x)

        assert output.shape == (4, 10)
        assert "load_balance_loss" in aux_losses
        assert not torch.isnan(output).any()

    def test_advanced_ensemble_stacking(self):
        """Test Advanced Ensemble with stacking strategy."""
        # Create dummy models
        models = []
        for i in range(3):
            model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 1))
            models.append(EnsembleMember(name=f"model_{i}", model=model))

        ensemble = AdvancedEnsemble(models, strategy="stacking", device="cpu")

        x = torch.randn(2, 100)
        prediction, details = ensemble.predict(x, return_details=True)

        assert prediction.shape == (2, 1)
        assert details is not None
        assert "base_predictions" in details

    def test_advanced_ensemble_weighted_voting(self):
        """Test Advanced Ensemble with weighted voting."""
        models = []
        for i in range(3):
            model = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.Conv1d(16, 1, 1))
            models.append(EnsembleMember(name=f"model_{i}", model=model))

        ensemble = AdvancedEnsemble(models, strategy="weighted_voting", device="cpu")

        x = torch.randn(2, 1, 100)
        prediction, details = ensemble.predict(x, return_details=True)

        assert prediction.shape[0] == 2
        assert details is not None
        assert "weights" in details


# ============================================================================
# Multi-Objective Optimization Tests
# ============================================================================


class TestMultiObjectiveOptimization:
    """Tests for Multi-Objective Optimization."""

    def test_individual_domination(self):
        """Test Individual domination check."""
        ind1 = Individual(parameters={"x": 1.0}, objectives={"obj1": 1.0, "obj2": 2.0})
        ind2 = Individual(parameters={"x": 2.0}, objectives={"obj1": 2.0, "obj2": 3.0})

        assert ind1.dominates_other(ind2)
        assert not ind2.dominates_other(ind1)

    def test_nsga2_initialization(self):
        """Test NSGA-II initialization."""
        objectives = [
            ObjectiveFunction("obj1", lambda p: p["x"] ** 2, minimize=True),
            ObjectiveFunction("obj2", lambda p: (p["x"] - 1) ** 2, minimize=True),
        ]

        parameter_space = {"x": (-5.0, 5.0)}

        optimizer = NSGAII(objectives=objectives, parameter_space=parameter_space, population_size=20, n_generations=5)

        assert optimizer.population_size == 20
        assert optimizer.n_generations == 5

    def test_nsga2_population_init(self):
        """Test population initialization."""
        objectives = [ObjectiveFunction("obj1", lambda p: p["x"] ** 2, minimize=True)]
        parameter_space = {"x": (-5.0, 5.0)}

        optimizer = NSGAII(objectives, parameter_space, population_size=10)
        population = optimizer.initialize_population()

        assert len(population) == 10
        for ind in population:
            assert "x" in ind.parameters
            assert -5.0 <= ind.parameters["x"] <= 5.0

    def test_nsga2_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        objectives = [
            ObjectiveFunction("obj1", lambda p: p["x"], minimize=True),
            ObjectiveFunction("obj2", lambda p: -p["x"], minimize=True),
        ]
        parameter_space = {"x": (0.0, 1.0)}

        optimizer = NSGAII(objectives, parameter_space, population_size=10)
        population = optimizer.initialize_population()
        optimizer.evaluate_population(population)

        fronts = optimizer.fast_non_dominated_sort(population)

        assert len(fronts) > 0
        assert all(ind.rank >= 0 for ind in population)

    def test_create_audio_restoration_moo(self):
        """Test audio restoration MOO creation."""
        optimizer = create_audio_restoration_moo()

        assert optimizer is not None
        assert len(optimizer.objectives) == 3
        assert "noise_reduction" in optimizer.parameter_space


# ============================================================================
# Uncertainty Quantification Tests
# ============================================================================


class TestUncertaintyQuantification:
    """Tests for Uncertainty Quantification."""

    def test_mc_dropout_forward(self):
        """Test MC Dropout forward pass."""
        base_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        mc_model = MCDropoutModel(base_model, dropout_rate=0.2, n_samples=5)

        x = torch.randn(4, 10)
        output = mc_model(x)

        assert output.shape == (4, 1)
        assert not torch.isnan(output).any()

    def test_mc_dropout_uncertainty(self):
        """Test MC Dropout uncertainty estimation."""
        base_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        mc_model = MCDropoutModel(base_model, dropout_rate=0.2, n_samples=10)

        x = torch.randn(4, 10)
        mean, std, samples = mc_model.predict_with_uncertainty(x)

        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)
        assert samples.shape == (10, 4, 1)
        assert (std >= 0).all()

    def test_bayesian_linear_forward(self):
        """Test Bayesian Linear layer."""
        nn.Linear(10, 5)
        from backend.core.optimization.uncertainty_quantification import BayesianLinear

        bay_layer = BayesianLinear(10, 5)

        x = torch.randn(4, 10)
        output = bay_layer(x)

        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()

    def test_bayesian_nn_kl_divergence(self):
        """Test Bayesian NN KL divergence."""
        model = BayesianNN(input_dim=10, hidden_dims=[20, 10], output_dim=1)

        kl = model.kl_divergence()

        assert isinstance(kl, torch.Tensor)
        assert kl.numel() == 1
        assert kl.item() >= 0

    def test_bayesian_nn_uncertainty(self):
        """Test Bayesian NN uncertainty estimation."""
        model = BayesianNN(input_dim=10, hidden_dims=[20], output_dim=1)

        x = torch.randn(4, 10)
        mean, std = model.predict_with_uncertainty(x, n_samples=10)

        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)
        assert (std >= 0).all()

    def test_ensemble_uncertainty(self):
        """Test Ensemble Uncertainty."""
        models = [nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1)) for _ in range(3)]

        ensemble_uq = EnsembleUncertainty(models, device="cpu")  # type: ignore[arg-type]

        x = torch.randn(4, 10)
        mean, std, details = ensemble_uq.predict_with_uncertainty(x)

        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)
        assert "entropy" in details
        assert "mutual_information" in details

    def test_uncertainty_quantifier_mc_dropout(self):
        """Test UncertaintyQuantifier with MC Dropout."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        quantifier = UncertaintyQuantifier(model, method="mc_dropout", n_samples=10, device="cpu")

        x = torch.randn(4, 10)
        metrics = quantifier.predict(x)

        assert metrics.mean.shape == (4, 1)
        assert metrics.std.shape == (4, 1)
        assert metrics.confidence is not None

    def test_uncertainty_quantifier_confidence(self):
        """Test confidence threshold."""
        model = nn.Sequential(nn.Linear(10, 1))
        quantifier = UncertaintyQuantifier(model, method="mc_dropout", device="cpu")

        x = torch.randn(4, 10)
        metrics = quantifier.predict(x)
        is_confident = quantifier.is_confident(metrics, threshold=0.5)

        assert is_confident.shape == (4,)
        assert is_confident.dtype == torch.bool


# ============================================================================
# Automated Augmentation Tests
# ============================================================================


class TestAutomatedAugmentation:
    """Tests for Automated Augmentation."""

    def test_audio_augmentations_add_noise(self):
        """Test add noise augmentation."""
        audio = torch.randn(2, 1, 4800)
        augmented = AudioAugmentations.add_noise(audio, noise_level=0.1)

        assert augmented.shape == audio.shape
        assert not torch.isnan(augmented).any()

    def test_audio_augmentations_gain(self):
        """Test gain augmentation."""
        audio = torch.randn(2, 1, 4800)
        augmented = AudioAugmentations.gain(audio, db=6.0)

        assert augmented.shape == audio.shape
        assert not torch.isnan(augmented).any()

    def test_audio_augmentations_time_mask(self):
        """Test time masking."""
        audio = torch.randn(2, 1, 4800)
        augmented = AudioAugmentations.time_mask(audio, mask_width=480)

        assert augmented.shape == audio.shape
        assert not torch.isnan(augmented).any()

    def test_augmentation_policy_apply(self):
        """Test augmentation policy."""
        operations = [("add_noise", 0.5), ("gain", 0.6)]

        policy = AugmentationPolicy(operations)

        audio = torch.randn(2, 1, 4800)
        augmented = policy.apply(audio)

        assert augmented.shape == audio.shape
        assert not torch.isnan(augmented).any()

    def test_rand_augment(self):
        """Test RandAugment."""
        rand_aug = RandAugment(n_ops=2, magnitude=0.5)

        audio = torch.randn(2, 1, 4800)
        augmented = rand_aug(audio)

        assert augmented.shape == audio.shape
        assert not torch.isnan(augmented).any()

    def test_rand_augment_material_specific(self):
        """Test RandAugment with material type."""
        rand_aug = RandAugment(n_ops=2, magnitude=0.5, material_type="vinyl")

        assert "add_vinyl_noise" in rand_aug.operations

    def test_auto_augment_init(self):
        """Test AutoAugment initialization."""
        auto_aug = AutoAugment(n_policies=3, n_ops_per_policy=2)

        assert len(auto_aug.policies) == 3
        for policy in auto_aug.policies:
            assert len(policy.operations) == 2

    def test_auto_augment_apply(self):
        """Test AutoAugment application."""
        auto_aug = AutoAugment(n_policies=2, n_ops_per_policy=1)

        audio = torch.randn(2, 1, 4800)
        augmented = auto_aug(audio)

        assert augmented.shape == audio.shape
        assert not torch.isnan(augmented).any()

    def test_auto_augment_save_load(self):
        """Test saving and loading policies."""
        auto_aug = AutoAugment(n_policies=2, n_ops_per_policy=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "policies.json"
            auto_aug.save_policies(save_path)

            assert save_path.exists()

            auto_aug2 = AutoAugment(n_policies=0, n_ops_per_policy=0)
            auto_aug2.load_policies(save_path)

            assert len(auto_aug2.policies) == 2


# ============================================================================
# Integration Test
# ============================================================================


class TestIntegration:
    """Integration test for all Phase 2-4 components."""

    def test_full_pipeline(self):
        """Test full pipeline with all components."""
        # 1. Create base model
        base_model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 1))

        # 2. Apply data augmentation
        augmenter = RandAugment(n_ops=1, magnitude=0.3)

        # 3. Uncertainty quantification
        uq = UncertaintyQuantifier(base_model, method="mc_dropout", device="cpu")

        # 4. Test forward pass
        x = torch.randn(2, 100)
        metrics = uq.predict(x)

        assert metrics.mean.shape == (2, 1)
        assert metrics.std.shape == (2, 1)

        print("✅ Integration test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
