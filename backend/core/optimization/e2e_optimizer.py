"""
End-to-End Optimizer für Aurik 10.0.0

Ermöglicht Joint Training über die gesamte Processing-Pipeline hinweg.
Macht kritische DSP-Module differenzierbar für Gradient-basierte Optimierung.

Autor: Aurik Backend-Team
Version: 8.1
Datum: 14. Februar 2026
"""

import logging
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
import torch.nn.functional as F

from backend.core.optimization.perceptual_loss import CombinedPerceptualLoss

logger = logging.getLogger(__name__)


class DifferentiableEQ(nn.Module):
    """
    Differentiable Parametric EQ for gradient-based optimization.

    Implements biquad filters with learnable parameters.
    """

    def __init__(self, sr: int = 48000, n_bands: int = 10, freq_range: tuple[float, float] = (20.0, 20000.0)):
        super().__init__()

        self.sr = sr
        self.n_bands = n_bands

        # Learnable parameters: frequency, gain, Q for each band
        log_freq_min = np.log(freq_range[0])
        log_freq_max = np.log(freq_range[1])

        # Initialize frequencies logarithmically spaced
        log_freqs = np.linspace(log_freq_min, log_freq_max, n_bands)
        init_freqs = np.exp(log_freqs)

        self.log_frequencies = nn.Parameter(torch.log(torch.tensor(init_freqs, dtype=torch.float32)))
        self.gains_db = nn.Parameter(torch.zeros(n_bands))
        self.log_q_factors = nn.Parameter(torch.log(torch.ones(n_bands) * 0.707))  # Default Q = 0.707

    def compute_biquad_coefficients(
        self, frequency: torch.Tensor, gain_db: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Berechnet biquad filter coefficients for peaking EQ.

        Returns:
            b_coeffs: Numerator coefficients [b0, b1, b2]
            a_coeffs: Denominator coefficients [a0, a1, a2]
        """
        A = torch.pow(10.0, gain_db / 40.0)
        omega = 2.0 * np.pi * frequency / self.sr
        sin_omega = torch.sin(omega)
        cos_omega = torch.cos(omega)
        alpha = sin_omega / (2.0 * q)

        # Peaking EQ coefficients
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / A

        # Normalize
        b_coeffs = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
        a_coeffs = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)

        return b_coeffs, a_coeffs

    def apply_biquad(self, audio: torch.Tensor, b_coeffs: torch.Tensor, a_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Wendet Biquad-Filter via Zeitbereichsfaltung an (differenzierbar).

        Note: For efficiency, this uses a simplified implementation.
        Full IIR filtering would require custom CUDA kernels for backprop.
        """
        # Simplified: Use frequency-domain filtering (differentiable)
        audio_fft = torch.fft.rfft(audio, dim=-1)  # pylint: disable=not-callable

        # Compute frequency response
        freqs = torch.fft.rfftfreq(audio.shape[-1], d=1.0 / self.sr, device=audio.device)  # pylint: disable=not-callable
        omega = 2.0 * np.pi * freqs
        z = torch.exp(1j * omega)

        # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
        b0, b1, b2 = b_coeffs[..., 0], b_coeffs[..., 1], b_coeffs[..., 2]
        a0, a1, a2 = a_coeffs[..., 0], a_coeffs[..., 1], a_coeffs[..., 2]

        # Expand dimensions for broadcasting
        z = z.view(1, 1, -1)
        b0 = b0.view(-1, 1, 1)
        b1 = b1.view(-1, 1, 1)
        b2 = b2.view(-1, 1, 1)
        a0 = a0.view(-1, 1, 1)
        a1 = a1.view(-1, 1, 1)
        a2 = a2.view(-1, 1, 1)

        numerator = b0 + b1 / z + b2 / (z**2)
        denominator = a0 + a1 / z + a2 / (z**2)
        H = numerator / denominator

        # Apply filter in frequency domain
        audio_fft_filtered = audio_fft * H.squeeze(1)

        # Convert back to time domain
        audio_filtered = torch.fft.irfft(audio_fft_filtered, n=audio.shape[-1], dim=-1)  # pylint: disable=not-callable

        return audio_filtered  # type: ignore[no-any-return]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Wendet an: parametric EQ to audio.

        Args:
            audio: Input audio [batch, channels, samples]

        Returns:
            Processed audio [batch, channels, samples]
        """
        output = audio

        # Apply each EQ band sequentially
        for i in range(self.n_bands):
            frequency = torch.exp(self.log_frequencies[i])
            gain_db = self.gains_db[i]
            q = torch.exp(self.log_q_factors[i])

            b_coeffs, a_coeffs = self.compute_biquad_coefficients(frequency, gain_db, q)

            # Apply to all batch items and channels
            output = self.apply_biquad(output, b_coeffs, a_coeffs)

        return output


class DifferentiableCompressor(nn.Module):
    """
    Differentiable Dynamic Range Compressor.

    Uses smooth approximations for threshold and ratio characteristics.
    """

    def __init__(self, sr: int = 48000, attack_ms: float = 5.0, release_ms: float = 100.0) -> None:
        super().__init__()

        self.sr = sr

        # Learnable parameters
        self.threshold_db = nn.Parameter(torch.tensor(-20.0))
        self.ratio = nn.Parameter(torch.tensor(4.0))
        self.knee_db = nn.Parameter(torch.tensor(6.0))
        self.makeup_gain_db = nn.Parameter(torch.tensor(0.0))

        # Attack/release coefficients
        self.attack_coeff = 1.0 - np.exp(-1.0 / (attack_ms * sr / 1000.0))
        self.release_coeff = 1.0 - np.exp(-1.0 / (release_ms * sr / 1000.0))

    def compute_gain_reduction(self, level_db: torch.Tensor) -> torch.Tensor:
        """
        Berechnet gain reduction using smooth knee compression.

        Uses a smooth approximation instead of hard thresholding.
        """
        threshold = self.threshold_db
        ratio = torch.clamp(self.ratio, min=1.0, max=20.0)
        knee = torch.clamp(self.knee_db, min=0.0, max=20.0)

        # Smooth knee function using tanh approximation
        # Above threshold: apply ratio
        # Below threshold: unity gain
        # In knee region: smooth transition

        # Distance from threshold
        delta = level_db - threshold

        # Smooth knee region using soft clipping
        knee_start = -knee / 2.0
        knee_end = knee / 2.0

        # Piece-wise smooth function
        # Below knee: no compression
        below_knee = delta < knee_start
        # Above knee: full compression
        above_knee = delta > knee_end
        # In knee: smooth transition
        in_knee = ~below_knee & ~above_knee

        gain_reduction = torch.zeros_like(level_db)

        # Above knee: apply ratio
        gain_reduction[above_knee] = (delta[above_knee] - knee_end) * (1.0 - 1.0 / ratio)

        # In knee: smooth interpolation
        if in_knee.any():
            knee_delta = delta[in_knee] - knee_start
            knee_width = knee
            knee_position = knee_delta / knee_width  # 0 to 1

            # Smooth hermite interpolation
            smooth_factor = knee_position**2 * (3.0 - 2.0 * knee_position)
            max_reduction = knee / 2.0 * (1.0 - 1.0 / ratio)
            gain_reduction[in_knee] = smooth_factor * max_reduction

        return -gain_reduction

    def envelope_follower(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Glättet envelope follower with attack/release.

        Uses exponential moving average (differentiable).
        """
        # Compute instantaneous level in dB
        level = torch.abs(audio)
        level_db = 20.0 * torch.log10(level + 1e-8)

        # Smooth with EMA (approximation of attack/release)
        # For differentiability, use learnable smoothing
        kernel_size = 1024
        kernel = torch.exp(-torch.arange(kernel_size, dtype=torch.float32, device=audio.device) * 0.01)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        # Handle channel dimension: [batch, channels, samples] -> process each channel
        batch_size, n_channels, n_samples = level_db.shape
        level_db_flat = level_db.view(batch_size * n_channels, 1, n_samples)

        # Pad and convolve
        level_db_padded = F.pad(level_db_flat, (kernel_size - 1, 0), mode="replicate")
        level_db_smooth_flat = F.conv1d(level_db_padded, kernel, padding=0)

        # Restore shape
        level_db_smooth = level_db_smooth_flat.view(batch_size, n_channels, -1)

        return level_db_smooth

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Wendet an: dynamic range compression.

        Args:
            audio: Input audio [batch, channels, samples]

        Returns:
            Compressed audio [batch, channels, samples]
        """
        # Get smooth envelope
        level_db = self.envelope_follower(audio)

        # Compute gain reduction
        gain_reduction_db = self.compute_gain_reduction(level_db)

        # Add makeup gain
        total_gain_db = gain_reduction_db + self.makeup_gain_db

        # Convert to linear gain
        gain_linear = torch.pow(10.0, total_gain_db / 20.0)

        # Apply gain (gain_linear already has shape [batch, channels, samples])
        output = audio * gain_linear

        return output


class DifferentiableNoiseGate(nn.Module):
    """
    Differentiable Noise Gate.

    Uses smooth gating function instead of hard threshold.
    """

    def __init__(self, sr: int = 48000) -> None:
        super().__init__()

        self.sr = sr

        # Learnable parameters
        self.threshold_db = nn.Parameter(torch.tensor(-40.0))
        self.range_db = nn.Parameter(torch.tensor(60.0))
        self.attack_ms = nn.Parameter(torch.tensor(1.0))
        self.release_ms = nn.Parameter(torch.tensor(100.0))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Wendet an: noise gate to audio.

        Args:
            audio: Input audio [batch, channels, samples]

        Returns:
            Gated audio [batch, channels, samples]
        """
        # Compute envelope
        level = torch.abs(audio)
        level_db = 20.0 * torch.log10(level + 1e-8)

        # Smooth envelope
        kernel_size = 512
        kernel = torch.exp(-torch.arange(kernel_size, dtype=torch.float32, device=audio.device) * 0.01)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        # Handle channel dimension: [batch, channels, samples] -> process each channel
        batch_size, n_channels, n_samples = level_db.shape
        level_db_flat = level_db.view(batch_size * n_channels, 1, n_samples)

        level_db_padded = F.pad(level_db_flat, (kernel_size - 1, 0), mode="replicate")  # pylint: disable=not-callable
        level_db_smooth_flat = F.conv1d(level_db_padded, kernel, padding=0)  # pylint: disable=not-callable

        # Restore shape
        level_db_smooth = level_db_smooth_flat.view(batch_size, n_channels, -1)

        # Compute gate gain using sigmoid (smooth approximation)
        # Above threshold: gain = 1.0
        # Below threshold: gain = exp(-range_db/20)
        threshold = self.threshold_db
        range_db = torch.clamp(self.range_db, min=0.0, max=80.0)

        # Smooth gate function
        gate_steepness = 10.0  # Controls transition smoothness
        gate_db = torch.sigmoid((level_db_smooth - threshold) * gate_steepness) * range_db
        gate_linear = torch.pow(10.0, -gate_db / 20.0)

        # Apply gate (gate_linear already has shape [batch, channels, samples])
        output = audio * gate_linear

        return output


class E2EOptimizationFramework:
    """
    End-to-End Optimization Framework for Aurik 10.0.0.

    Coordinates joint training of:
    1. ML Models (DeepFilterNet, Demucs, etc.)
    2. Differentiable DSP modules
    3. Processing pipeline parameters

    Uses Combined Perceptual Loss + Musical Goals as training objectives.
    """

    def __init__(
        self,
        sr: int = 48000,
        device: str = "cpu",  # §9.5: Aurik 10.0.0 — ausschließlich CPU, kein CUDA/ROCm/Metal
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.sr = sr
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints/e2e_optimization")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize differentiable modules
        self.diff_eq = DifferentiableEQ(sr=sr).to(device)
        self.diff_compressor = DifferentiableCompressor(sr=sr).to(device)
        self.diff_gate = DifferentiableNoiseGate(sr=sr).to(device)

        # Loss function
        self.perceptual_loss = CombinedPerceptualLoss(sr=sr).to(device)

        # Collect all trainable parameters
        self.trainable_modules = nn.ModuleDict(
            {"eq": self.diff_eq, "compressor": self.diff_compressor, "gate": self.diff_gate}
        )

        # Optimizer (will be set up during training)
        self.optimizer = None

        logger.info("E2EOptimizationFramework initialisiert on %s", device)
        logger.info("  Sample rate: %s Hz", sr)
        logger.info("  Checkpoint dir: %s", self.checkpoint_dir)

    def setup_optimizer(self, learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """Richtet Optimizer für das Training ein."""
        self.optimizer = torch.optim.AdamW(  # type: ignore[assignment]
            self.trainable_modules.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        logger.info("Optimizer konfiguriert: AdamW(lr=%s, wd=%s)", learning_rate, weight_decay)

    def _require_optimizer(self) -> torch.optim.Optimizer:
        """Gibt initialized optimizer or raise a clear error zurück."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call setup_optimizer() before training/checkpointing.")
        return self.optimizer

    def forward_pass(
        self, audio: torch.Tensor, enable_eq: bool = True, enable_compressor: bool = True, enable_gate: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through differentiable pipeline.

        Args:
            audio: Input audio [batch, channels, samples]
            enable_eq: Whether to apply EQ
            enable_compressor: Whether to apply compression
            enable_gate: Whether to apply noise gate

        Returns:
            Processed audio [batch, channels, samples]
        """
        output = audio

        if enable_gate:
            output = self.diff_gate(output)

        if enable_eq:
            output = self.diff_eq(output)

        if enable_compressor:
            output = self.diff_compressor(output)

        return output

    def training_step(self, input_audio: torch.Tensor, target_audio: torch.Tensor) -> dict[str, float]:
        """
        Single training step.

        Args:
            input_audio: Degraded audio [batch, channels, samples]
            target_audio: Clean reference audio [batch, channels, samples]

        Returns:
            Dictionary with loss values
        """
        optimizer = self._require_optimizer()
        optimizer.zero_grad()

        # Forward pass
        output_audio = self.forward_pass(input_audio)

        # Compute loss
        loss, loss_details = self.perceptual_loss(output_audio, target_audio, return_details=True)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.trainable_modules.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        return loss_details  # type: ignore[no-any-return]

    def train_epoch(self, dataloader: Any, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: PyTorch DataLoader with (input, target) pairs
            epoch: Current epoch number

        Returns:
            Average metrics for the epoch
        """
        self.trainable_modules.train()

        epoch_losses = []

        for batch_idx, (input_audio, target_audio) in enumerate(dataloader):
            input_audio = input_audio.to(self.device)
            target_audio = target_audio.to(self.device)

            # Training step
            losses = self.training_step(input_audio, target_audio)

            epoch_losses.append(losses)

            if batch_idx % 100 == 0:
                logger.info("Epoch %s, Batch %s: Loss = %.4f", epoch, batch_idx, losses["total_perceptual_loss"])

        # Average losses
        avg_losses = {}
        for key in epoch_losses[0]:
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])

        return avg_losses  # type: ignore[return-value]

    def validate(self, dataloader: Any) -> dict[str, float]:
        """
        Validiert on validation set.

        Args:
            dataloader: Validation DataLoader

        Returns:
            Validation metrics
        """
        self.trainable_modules.eval()

        val_losses = []

        with torch.no_grad():
            for input_audio, target_audio in dataloader:
                input_audio = input_audio.to(self.device)
                target_audio = target_audio.to(self.device)

                # Forward pass
                output_audio = self.forward_pass(input_audio)

                # Compute loss
                _loss, loss_details = self.perceptual_loss(output_audio, target_audio, return_details=True)

                val_losses.append(loss_details)

        # Average losses
        avg_losses = {}
        for key in val_losses[0]:
            avg_losses[key] = np.mean([l[key] for l in val_losses])

        return avg_losses  # type: ignore[return-value]

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]):
        """Speichert training checkpoint."""
        optimizer = self._require_optimizer()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.trainable_modules.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)

        logger.info("Checkpoint gespeichert: %s", checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Lädt training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)  # nosec B614 — interner Tensor-Checkpoint aus models/

        self.trainable_modules.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info("Checkpoint geladen: %s", checkpoint_path)
        logger.info("  Epoch: %s", checkpoint["epoch"])
        logger.info("  Metrics: %s", checkpoint["metrics"])

        return checkpoint["epoch"], checkpoint["metrics"]

    def export_optimized_parameters(self) -> dict[str, Any]:
        """
        Export optimized parameters for integration into Aurik pipeline.

        Returns:
            Dictionary with optimized DSP parameters
        """
        self.trainable_modules.eval()

        # Extract EQ parameters
        eq_params = {
            "frequencies": torch.exp(self.diff_eq.log_frequencies).detach().cpu().numpy().tolist(),  # pylint: disable=not-callable
            "gains_db": self.diff_eq.gains_db.detach().cpu().numpy().tolist(),
            "q_factors": torch.exp(self.diff_eq.log_q_factors).detach().cpu().numpy().tolist(),
        }

        # Extract compressor parameters
        comp_params = {
            "threshold_db": self.diff_compressor.threshold_db.item(),
            "ratio": self.diff_compressor.ratio.item(),
            "knee_db": self.diff_compressor.knee_db.item(),
            "makeup_gain_db": self.diff_compressor.makeup_gain_db.item(),
        }

        # Extract gate parameters
        gate_params = {
            "threshold_db": self.diff_gate.threshold_db.item(),
            "range_db": self.diff_gate.range_db.item(),
            "attack_ms": self.diff_gate.attack_ms.item(),
            "release_ms": self.diff_gate.release_ms.item(),
        }

        params = {"eq": eq_params, "compressor": comp_params, "gate": gate_params}

        logger.info("optimiert parameters exported")

        return params


# Example usage
if __name__ == "__main__":
    # Initialize framework
    framework = E2EOptimizationFramework(sr=48000, device="cpu")  # §9.5: CPU-only
    framework.setup_optimizer(learning_rate=1e-4)

    # Test forward pass
    batch_size = 2
    channels = 1
    samples = 48000 * 2  # 2 seconds

    input_audio = torch.randn(batch_size, channels, samples)  # §9.5: kein device="cuda"
    target_audio = torch.randn(batch_size, channels, samples)  # §9.5: CPU

    # Training step
    losses = framework.training_step(input_audio, target_audio)

    logger.debug("Training step abgeschlossen")
    logger.debug("Losses:")
    for key, value in losses.items():
        logger.debug("  %s: %.4f", key, value)

    # Export parameters
    params = framework.export_optimized_parameters()
    logger.debug("\noptimiert Parameters:")
    logger.debug("  EQ bands: %s", len(params["eq"]["frequencies"]))
    logger.debug("  Compressor Verhaeltnis: %.2f", params["compressor"]["ratio"])
    logger.debug("  Gate Schwelle: %.1f dB", params["gate"]["threshold_db"])
