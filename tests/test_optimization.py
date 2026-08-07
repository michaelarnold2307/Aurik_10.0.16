"""
Unit Tests für Backend Optimization Framework

Testet alle Komponenten des Optimization-Systems.

Autor: Aurik Backend-Team
Version: 8.1
Datum: 14. Februar 2026
"""

import pytest

pytest.importorskip("optuna")  # Skip module if optuna is not installed
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.core.optimization.e2e_optimizer import (
    DifferentiableCompressor,
    DifferentiableEQ,
    DifferentiableNoiseGate,
    E2EOptimizationFramework,
)
from backend.core.optimization.hyperparameter_optimizer import HyperparameterConfig, MaterialSpecificOptimizer
from backend.core.optimization.optimization_integration import OptimizationIntegration
from backend.core.optimization.perceptual_loss import (
    CombinedPerceptualLoss,
    MultiResolutionSTFTLoss,
    MusicalFeatureLoss,
    PANNsPerceptualLoss,
    PsychoacousticMaskingLoss,
)


class TestPerceptualLoss:
    """Test perceptual loss functions."""

    @pytest.fixture
    def audio_pair(self):
        """Create dummy audio pair for testing."""
        batch_size = 2
        channels = 1
        samples = 48000  # 1 second at 48kHz

        output = torch.randn(batch_size, channels, samples)
        target = torch.randn(batch_size, channels, samples)

        return output, target

    def test_multi_resolution_stft_loss(self, audio_pair):
        """Test Multi-Resolution STFT Loss."""
        output, target = audio_pair

        loss_fn = MultiResolutionSTFTLoss()
        loss, details = loss_fn(output, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert "total_sc_loss" in details
        assert "total_mag_loss" in details

    def test_panns_perceptual_loss(self, audio_pair):
        """Test PANNs Perceptual Loss."""
        output, target = audio_pair

        loss_fn = PANNsPerceptualLoss()
        loss, details = loss_fn(output, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert len(details) > 0

    def test_psychoacoustic_masking_loss(self, audio_pair):
        """Test Psychoacoustic Masking Loss."""
        output, target = audio_pair

        loss_fn = PsychoacousticMaskingLoss(sr=48000)
        loss, details = loss_fn(output, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert "psychoacoustic_loss" in details

    def test_musical_feature_loss(self, audio_pair):
        """Test Musical Feature Loss."""
        output, target = audio_pair

        loss_fn = MusicalFeatureLoss(sr=48000)
        loss, details = loss_fn(output, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert "harmonic_loss" in details
        assert "rhythmic_loss" in details
        assert "timbral_loss" in details

    def test_combined_perceptual_loss(self, audio_pair):
        """Test Combined Perceptual Loss."""
        output, target = audio_pair

        loss_fn = CombinedPerceptualLoss(sr=48000)
        loss, details = loss_fn(output, target, return_details=True)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert "total_perceptual_loss" in details

        # Check that all components are present
        assert any("stft" in k for k in details.keys())
        assert any("musical" in k for k in details.keys())


class TestDifferentiableDSP:
    """Test differentiable DSP modules."""

    @pytest.fixture
    def audio_batch(self):
        """Create audio batch for testing."""
        batch_size = 2
        channels = 1
        samples = 48000

        audio = torch.randn(batch_size, channels, samples)
        return audio

    def test_differentiable_eq(self, audio_batch):
        """Test Differentiable EQ."""
        eq = DifferentiableEQ(sr=48000, n_bands=10)

        output = eq(audio_batch)

        assert output.shape == audio_batch.shape
        assert torch.isfinite(output).all()

        # Test gradient flow
        loss = output.mean()
        loss.backward()

        assert eq.log_frequencies.grad is not None
        assert eq.gains_db.grad is not None

    def test_differentiable_compressor(self, audio_batch):
        """Test Differentiable Compressor."""
        compressor = DifferentiableCompressor(sr=48000)

        output = compressor(audio_batch)

        assert output.shape == audio_batch.shape
        assert torch.isfinite(output).all()

        # Test gradient flow
        loss = output.mean()
        loss.backward()

        assert compressor.threshold_db.grad is not None
        assert compressor.ratio.grad is not None

    def test_differentiable_noise_gate(self, audio_batch):
        """Test Differentiable Noise Gate."""
        gate = DifferentiableNoiseGate(sr=48000)

        output = gate(audio_batch)

        assert output.shape == audio_batch.shape
        assert torch.isfinite(output).all()

        # Test gradient flow
        loss = output.mean()
        loss.backward()

        assert gate.threshold_db.grad is not None


class TestE2EOptimization:
    """Test End-to-End Optimization Framework."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_framework_initialization(self, temp_checkpoint_dir):
        """Test E2E framework initialization."""
        framework = E2EOptimizationFramework(sr=48000, device="cpu", checkpoint_dir=temp_checkpoint_dir)

        assert framework.sr == 48000
        assert framework.device == "cpu"
        assert framework.checkpoint_dir == temp_checkpoint_dir

    def test_forward_pass(self, temp_checkpoint_dir):
        """Test forward pass through framework."""
        framework = E2EOptimizationFramework(sr=48000, device="cpu", checkpoint_dir=temp_checkpoint_dir)

        batch_size = 2
        channels = 1
        samples = 48000

        audio = torch.randn(batch_size, channels, samples)
        output = framework.forward_pass(audio)

        assert output.shape == audio.shape
        assert torch.isfinite(output).all()

    def test_training_step(self, temp_checkpoint_dir):
        """Test training step."""
        framework = E2EOptimizationFramework(sr=48000, device="cpu", checkpoint_dir=temp_checkpoint_dir)
        framework.setup_optimizer(learning_rate=1e-4)

        batch_size = 2
        channels = 1
        samples = 48000

        input_audio = torch.randn(batch_size, channels, samples)
        target_audio = torch.randn(batch_size, channels, samples)

        losses = framework.training_step(input_audio, target_audio)

        assert "total_perceptual_loss" in losses
        assert losses["total_perceptual_loss"] > 0

    def test_export_parameters(self, temp_checkpoint_dir):
        """Test parameter export."""
        framework = E2EOptimizationFramework(sr=48000, device="cpu", checkpoint_dir=temp_checkpoint_dir)

        params = framework.export_optimized_parameters()

        assert "eq" in params
        assert "compressor" in params
        assert "gate" in params

        assert "frequencies" in params["eq"]
        assert "gains_db" in params["eq"]
        assert len(params["eq"]["frequencies"]) == 10


class TestHyperparameterOptimization:
    """Test Hyperparameter Optimization."""

    def test_hyperparameter_config(self):
        """Test HyperparameterConfig dataclass."""
        config = HyperparameterConfig()

        assert config.dfn_attenuation_limit == 6.0
        assert config.comp_threshold_db == -20.0
        assert config.eq_bass_gain == 0.0

    @pytest.mark.slow
    def test_material_specific_optimizer_init(self):
        """Test MaterialSpecificOptimizer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = MaterialSpecificOptimizer(
                material_type="vinyl", storage_path=Path(tmpdir), n_trials=5, n_jobs=1
            )

            assert optimizer.material_type == "vinyl"
            assert optimizer.n_trials == 5


class TestOptimizationIntegration:
    """Test Optimization Integration."""

    @pytest.fixture
    def temp_optimization_dir(self):
        """Create temporary optimization directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            opt_dir = Path(tmpdir) / "optimization"
            opt_dir.mkdir()

            # Create dummy parameter file for vinyl
            vinyl_dir = opt_dir / "vinyl"
            vinyl_dir.mkdir()

            params = {"dfn_attenuation_limit": 7.5, "eq_bass_gain": -3.0, "comp_threshold_db": -22.0}

            import yaml

            with open(vinyl_dir / "best_params_vinyl.yaml", "w") as f:
                yaml.dump(params, f)

            yield opt_dir

    def test_integration_initialization(self, temp_optimization_dir):
        """Test OptimizationIntegration initialization."""
        integration = OptimizationIntegration(optimization_base_path=temp_optimization_dir, sr=48000)

        assert integration.sr == 48000
        assert len(integration.material_params_cache) > 0

    def test_get_optimized_parameters(self, temp_optimization_dir):
        """Test getting optimized parameters."""
        integration = OptimizationIntegration(optimization_base_path=temp_optimization_dir, sr=48000)

        params = integration.get_optimized_parameters("vinyl")

        assert params is not None
        assert "dfn_attenuation_limit" in params
        assert params["dfn_attenuation_limit"] == 7.5

    def test_apply_optimized_parameters_to_context(self, temp_optimization_dir):
        """Test applying parameters to context."""
        integration = OptimizationIntegration(optimization_base_path=temp_optimization_dir, sr=48000)

        context = {"material_type": "vinyl", "detected_artifacts": ["clicks", "pops"]}

        context = integration.apply_optimized_parameters_to_context(context, "vinyl")

        assert "optimized_params" in context
        assert "ml_config" in context
        assert "dsp_config" in context

    def test_compute_perceptual_quality(self, temp_optimization_dir):
        """Test perceptual quality computation."""
        integration = OptimizationIntegration(optimization_base_path=temp_optimization_dir, sr=48000, device="cpu")

        # Create dummy audio
        output_audio = np.random.randn(48000)
        reference_audio = np.random.randn(48000)

        quality_score = integration.compute_perceptual_quality(output_audio, reference_audio, return_details=False)

        assert 0.0 <= quality_score <= 1.0  # type: ignore[operator]

    def test_recommend_processing_strategy(self, temp_optimization_dir):
        """Test processing strategy recommendation."""
        integration = OptimizationIntegration(optimization_base_path=temp_optimization_dir, sr=48000)

        context = {"material_type": "vinyl"}

        strategy = integration.recommend_processing_strategy(context, "vinyl")

        assert "material_type" in strategy
        assert "recommended_models" in strategy
        assert "recommended_dsp_chain" in strategy
        assert len(strategy["recommended_models"]) > 0


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for optimization components."""

    def test_perceptual_loss_speed(self, benchmark):
        """Benchmark perceptual loss computation."""
        loss_fn = CombinedPerceptualLoss(sr=48000)

        batch_size = 4
        channels = 1
        samples = 48000

        output = torch.randn(batch_size, channels, samples)
        target = torch.randn(batch_size, channels, samples)

        def compute_loss():
            loss = loss_fn(output, target, return_details=False)
            return loss

        benchmark(compute_loss)

        # Should be reasonably fast (<1.5s for 1 second of audio, allows for system load)
        assert benchmark.stats["mean"] < 1.5

    def test_differentiable_eq_speed(self, benchmark):
        """Benchmark differentiable EQ."""
        eq = DifferentiableEQ(sr=48000, n_bands=10)

        batch_size = 4
        channels = 1
        samples = 48000

        audio = torch.randn(batch_size, channels, samples)

        result = benchmark(lambda: eq(audio))

        # Should be fast
        assert benchmark.stats["mean"] < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
