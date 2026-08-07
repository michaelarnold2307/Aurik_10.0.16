"""
Uncertainty Quantification für Aurik 10.0.0

Implementiert verschiedene Methoden zur Unsicherheitsquantifizierung:
- Monte Carlo Dropout
- Bayesian Neural Networks (Variational Inference)
- Ensemble-basierte Unsicherheit
- Calibration (Temperature Scaling, Platt Scaling)
- Confidence Estimation

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

import logging
from dataclasses import dataclass

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Monte Carlo Dropout
# ============================================================================


class MCDropoutModel(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.

    Keeps dropout active during inference to sample from approximate posterior.
    """

    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.2, n_samples: int = 20):
        """
        Initialisiert MC Dropout model.

        Args:
            base_model: Base neural network
            dropout_rate: Dropout probability
            n_samples: Number of MC samples for uncertainty estimation
        """
        super().__init__()

        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples

        # Add dropout layers if not already present
        self._add_dropout_layers()

        logger.info("MCDropoutModel initialisiert: dropout=%s, n_samples=%s", dropout_rate, n_samples)

    def _add_dropout_layers(self):
        """Recursively add dropout layers after each activation."""
        for name, module in self.base_model.named_children():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                setattr(self.base_model, name, nn.Sequential(module, nn.Dropout(p=self.dropout_rate)))
            elif isinstance(module, nn.Sequential):
                # Recursively process sequential modules
                new_layers = []
                for layer in module:
                    new_layers.append(layer)
                    if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                        new_layers.append(nn.Dropout(p=self.dropout_rate))
                setattr(self.base_model, name, nn.Sequential(*new_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.base_model(x)  # type: ignore[no-any-return]

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.

        Args:
            x: Input tensor

        Returns:
            mean: Mean prediction
            std: Standard deviation (aleatoric + epistemic uncertainty)
            samples: All MC samples
        """
        # Enable dropout during inference
        self.train()

        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                prediction = self.base_model(x)
                samples.append(prediction)

        samples = torch.stack(samples, dim=0)  # type: ignore[assignment]  # [n_samples, batch, ...]

        mean = samples.mean(dim=0)  # type: ignore[attr-defined]
        std = samples.std(dim=0)  # type: ignore[attr-defined]

        # Switch back to eval mode
        self.eval()

        return mean, std, samples  # type: ignore[return-value]


# ============================================================================
# Bayesian Neural Network (Variational Inference)
# ============================================================================


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight uncertainty.

    Uses Bayes by Backprop (variational inference).
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        # Prior distribution (Gaussian)
        self.prior_std = prior_std

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        """Initialisiert parameters."""
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -5)  # Small initial variance
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_logvar, -5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick."""
        # Sample weights from posterior
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_eps * weight_std

        # Sample bias from posterior
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_eps * bias_std

        return F.linear(x, weight, bias)  # pylint: disable=not-callable

    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate KL divergence between posterior and prior.

        KL[q(w|θ) || p(w)] where p(w) is N(0, prior_std^2)
        """
        # KL for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_var) / (self.prior_std**2) - torch.log(weight_var / (self.prior_std**2)) - 1
        )

        # KL for bias
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_var) / (self.prior_std**2) - torch.log(bias_var / (self.prior_std**2)) - 1
        )

        return weight_kl + bias_kl


class BayesianNN(nn.Module):
    """
    Bayesian Neural Network for uncertainty quantification.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, prior_std: float = 1.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_std))
            layers.append(nn.ReLU())  # type: ignore[arg-type]
            prev_dim = hidden_dim

        layers.append(BayesianLinear(prev_dim, output_dim, prior_std))

        self.layers = nn.ModuleList(layers)

        logger.info("BayesianNN initialisiert: %s -> %s -> %s", input_dim, hidden_dims, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all Bayesian layers."""
        kl = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()  # type: ignore[assignment]
        return kl  # type: ignore[return-value]

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of samples from posterior

        Returns:
            mean: Mean prediction
            std: Standard deviation
        """
        samples = []

        for _ in range(n_samples):
            prediction = self.forward(x)
            samples.append(prediction)

        samples = torch.stack(samples, dim=0)  # type: ignore[assignment]
        mean = samples.mean(dim=0)  # type: ignore[attr-defined]
        std = samples.std(dim=0)  # type: ignore[attr-defined]

        return mean, std


# ============================================================================
# Ensemble-based Uncertainty
# ============================================================================


class EnsembleUncertainty:
    """
    Schätzt uncertainty using model ensemble.

    Combines predictions from multiple models to quantify epistemic uncertainty.
    """

    def __init__(self, models: list[nn.Module], device: str = "cpu"):
        """
        Initialisiert ensemble uncertainty estimator.

        Args:
            models: List of trained models
            device: Device to use (CPU-only Policy, Section 9.5)
        """
        self.models = [model.to(device) for model in models]
        self.device = device
        self.n_models = len(models)

        logger.info("EnsembleUncertainty initialisiert: %s models", self.n_models)

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Predict with uncertainty estimation.

        Args:
            x: Input tensor

        Returns:
            mean: Mean prediction across ensemble
            std: Standard deviation (epistemic uncertainty)
            details: Additional uncertainty metrics
        """
        x = x.to(self.device)

        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # type: ignore[assignment]  # [n_models, batch, ...]

        mean = predictions.mean(dim=0)  # type: ignore[attr-defined]
        std = predictions.std(dim=0)  # type: ignore[attr-defined]

        # Additional metrics
        entropy = self._calculate_entropy(predictions)  # type: ignore[arg-type]
        mutual_info = self._calculate_mutual_information(predictions)  # type: ignore[arg-type]

        details = {"entropy": entropy, "mutual_information": mutual_info, "predictions": predictions}

        return mean, std, details  # type: ignore[return-value]

    def _calculate_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate predictive entropy."""
        predictions.mean(dim=0)

        # For regression, approximate with Gaussian entropy
        variance = predictions.var(dim=0)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * variance)

        return entropy

    def _calculate_mutual_information(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate mutual information (epistemic uncertainty)."""
        # MI = Entropy[E[p(y|x,D)]] - E[Entropy[p(y|x,w)]]
        # For regression, this simplifies to variance of predictions
        mutual_info = predictions.var(dim=0)

        return mutual_info


# ============================================================================
# Calibration
# ============================================================================


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for confidence calibration.

    Scales logits by learned temperature to calibrate probabilities.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Skaliert logits by temperature."""
        return logits / self.temperature  # type: ignore[no-any-return]

    def calibrate(
        self,
        model: nn.Module,
        val_loader,
        max_iter: int = 50,
        device: str = "cpu",  # CPU-only Policy (Section 9.5)
    ):
        """
        Learn optimal temperature on validation set.

        Args:
            model: Model to calibrate
            val_loader: Validation data loader
            max_iter: Maximum optimization iterations
            device: Device to use
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                all_logits.append(logits)
                all_labels.append(labels.to(device))

        all_logits = torch.cat(all_logits, dim=0)  # type: ignore[assignment]
        all_labels = torch.cat(all_labels, dim=0)  # type: ignore[assignment]

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss() -> float:
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)  # type: ignore[arg-type]
            loss = F.cross_entropy(scaled_logits, all_labels)  # type: ignore[arg-type]
            loss.backward()
            return float(loss.item())

        optimizer.step(closure=eval_loss)

        logger.info("Kalibrierung abgeschlossen: temperature = %.4f", self.temperature.item())


@dataclass
class UncertaintyMetrics:
    """Data class for uncertainty metrics."""

    mean: torch.Tensor
    std: torch.Tensor
    confidence: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    mutual_information: torch.Tensor | None = None


class UncertaintyQuantifier:
    """
    Einheitliches Interface für Unsicherheitsquantifizierung.

    Supports multiple methods: MC Dropout, Bayesian NN, Ensemble.
    """

    def __init__(
        self,
        model: nn.Module | list[nn.Module],
        method: str = "mc_dropout",
        n_samples: int = 20,
        device: str = "cpu",  # CPU-only Policy (Section 9.5) — kein CUDA
    ):
        """
        Initialisiert uncertainty quantifier.

        Args:
            model: Base model or list of models (for ensemble)
            method: "mc_dropout", "bayesian", or "ensemble"
            n_samples: Number of samples for uncertainty estimation
            device: Device to use
        """
        if isinstance(model, list):
            if not model:
                raise ValueError("Model list for uncertainty quantification must not be empty")
            self.base_model = model[0].to(device)
        else:
            self.base_model = model.to(device)
        self.method = method
        self.n_samples = n_samples
        self.device = device
        self.uq_model: MCDropoutModel | EnsembleUncertainty | BayesianNN

        if method == "mc_dropout":
            if isinstance(model, list):
                logger.warning("MC Dropout expects a single model, using first model from list")
                base_model = model[0]
            else:
                base_model = model
            self.uq_model = MCDropoutModel(base_model, dropout_rate=0.2, n_samples=n_samples)
        elif method == "bayesian":
            if isinstance(model, list):
                logger.warning("Bayesian method expects a single BayesianNN model, using first model from list")
                base_model = model[0]
            else:
                base_model = model
            if not isinstance(base_model, BayesianNN):
                raise ValueError("For method='bayesian', model must be an instance of BayesianNN")
            self.uq_model = base_model.to(device)
        elif method == "ensemble":
            if isinstance(model, list):
                ensemble_models = model
            else:
                logger.warning("Ensemble method requires list of models, using single model")
                ensemble_models = [model]
            self.uq_model = EnsembleUncertainty(ensemble_models, device=device)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info("UncertaintyQuantifier initialisiert: method=%s", method)

    def predict(self, x: torch.Tensor, return_samples: bool = False) -> UncertaintyMetrics:
        """
        Make prediction with uncertainty quantification.

        Args:
            x: Input tensor
            return_samples: Whether to return all samples

        Returns:
            UncertaintyMetrics object
        """
        x = x.to(self.device)

        if self.method == "mc_dropout":
            uq_model = self.uq_model
            if not isinstance(uq_model, MCDropoutModel):
                raise RuntimeError("Internal model/method mismatch for mc_dropout")
            mean, std, _samples = uq_model.predict_with_uncertainty(x)

            # Confidence based on inverse std
            confidence = 1.0 / (1.0 + std)

            metrics = UncertaintyMetrics(mean=mean, std=std, confidence=confidence)

        elif self.method == "ensemble":
            uq_model = self.uq_model
            if not isinstance(uq_model, EnsembleUncertainty):
                raise RuntimeError("Internal model/method mismatch for ensemble")
            mean, std, details = uq_model.predict_with_uncertainty(x)  # type: ignore[misc]

            metrics = UncertaintyMetrics(
                mean=mean, std=std, entropy=details["entropy"], mutual_information=details["mutual_information"]
            )

        elif self.method == "bayesian":
            uq_model = self.uq_model
            if not isinstance(uq_model, BayesianNN):
                raise RuntimeError("Internal model/method mismatch for bayesian")
            mean, std = uq_model.predict_with_uncertainty(x, self.n_samples)

            metrics = UncertaintyMetrics(mean=mean, std=std)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return metrics

    def is_confident(self, metrics: UncertaintyMetrics, threshold: float = 0.8) -> torch.Tensor:
        """
        Prüft if prediction is confident based on threshold.

        Args:
            metrics: Uncertainty metrics
            threshold: Confidence threshold

        Returns:
            Boolean tensor indicating confident predictions
        """
        if metrics.confidence is not None:
            confidence = metrics.confidence.squeeze(-1) if metrics.confidence.dim() > 1 else metrics.confidence
            return confidence > threshold
        else:
            # Use inverse std as confidence proxy
            confidence = 1.0 / (1.0 + metrics.std)
            confidence = confidence.squeeze(-1) if confidence.dim() > 1 else confidence
            return confidence > threshold  # type: ignore[no-any-return]


# Example usage
if __name__ == "__main__":
    # Create simple model
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    # Test MC Dropout
    logger.debug("=== MC Dropout ===")
    mc_dropout = MCDropoutModel(model, dropout_rate=0.2, n_samples=10)

    x = torch.randn(4, 128)
    mean, std, samples = mc_dropout.predict_with_uncertainty(x)

    logger.debug("Mean shape: %s", mean.shape)
    logger.debug("Std shape: %s", std.shape)
    logger.debug("Samples shape: %s", samples.shape)
    logger.debug("Mean std: %.4f", std.mean().item())

    # Test Bayesian NN
    logger.debug("\n=== Bayesian NN ===")
    bayesian_model = BayesianNN(128, [64, 32], 1)

    mean, std = bayesian_model.predict_with_uncertainty(x, n_samples=10)
    kl = bayesian_model.kl_divergence()

    logger.debug("Mean shape: %s", mean.shape)
    logger.debug("Std shape: %s", std.shape)
    logger.debug("KL divergence: %.4f", kl.item())

    # Test UncertaintyQuantifier
    logger.debug("\n=== UncertaintyQuantifier ===")
    quantifier = UncertaintyQuantifier(model, method="mc_dropout", device="cpu")

    metrics = quantifier.predict(x)
    is_confident = quantifier.is_confident(metrics, threshold=0.8)

    logger.debug("Mean: %s", metrics.mean.shape)
    logger.debug("Std: %s", metrics.std.shape)
    logger.debug("Confidence: %s", metrics.confidence.shape if metrics.confidence is not None else "N/A")
    logger.debug("Confident predictions: %s/%s", is_confident.sum().item(), len(is_confident))
