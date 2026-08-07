"""
Automated Data Augmentation für Aurik 10.0.0

Implementiert automatische Datenaugmentierung für Audio:
- AutoAugment für Audio
- RandAugment
- Material-spezifische Augmentierungsstrategien
- Learned Augmentation Policies
- Augmentation Consistency Training

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class AugmentationOp:
    """Single augmentation operation."""

    name: str
    transform: Callable[[torch.Tensor], torch.Tensor]
    magnitude_range: tuple[float, float]


class AudioAugmentations:
    """
    Sammlung von Audio-Augmentierungsoperationen.

    Includes time-domain, frequency-domain, and material-specific augmentations.
    """

    @staticmethod
    def time_stretch(audio: torch.Tensor, rate: float) -> torch.Tensor:
        """Time stretching without pitch change."""
        # Simple implementation using interpolation
        length = audio.size(-1)
        new_length = int(length / rate)

        stretched = F.interpolate(audio.unsqueeze(1), size=new_length, mode="linear", align_corners=False).squeeze(1)

        # Pad or trim to original length
        if stretched.size(-1) < length:
            padding = length - stretched.size(-1)
            stretched = F.pad(stretched, (0, padding))
        else:
            stretched = stretched[..., :length]

        return stretched  # type: ignore[no-any-return]

    @staticmethod
    def pitch_shift(audio: torch.Tensor, n_steps: int, sr: int = 48000) -> torch.Tensor:
        """Pitch shifting."""
        # Approximate pitch shift using time stretching + resampling
        rate = 2 ** (n_steps / 12.0)

        # Time stretch
        stretched = AudioAugmentations.time_stretch(audio, rate)

        # Resample back to original tempo
        resampled = F.interpolate(
            stretched.unsqueeze(1), size=audio.size(-1), mode="linear", align_corners=False
        ).squeeze(1)

        return resampled  # type: ignore[no-any-return]

    @staticmethod
    def add_noise(audio: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Fügt hinzu: Gaussian noise."""
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    @staticmethod
    def gain(audio: torch.Tensor, db: float) -> torch.Tensor:
        """Wendet an: gain in dB."""
        gain_linear = 10 ** (db / 20.0)
        return audio * gain_linear

    @staticmethod
    def eq_filter(audio: torch.Tensor, cutoff_freq: float, gain_db: float, sr: int = 48000) -> torch.Tensor:
        """Einfaches EQ filter (highpass/lowpass)."""
        # Simplified implementation using spectral filtering
        # In practice, use proper filter design

        # FFT
        spec = torch.fft.rfft(audio, dim=-1)  # pylint: disable=not-callable
        freqs = torch.fft.rfftfreq(audio.size(-1), d=1 / sr)  # pylint: disable=not-callable

        # Create filter
        if gain_db > 0:
            # Highpass
            mask = (freqs > cutoff_freq).float()
        else:
            # Lowpass
            mask = (freqs < cutoff_freq).float()

        gain_linear = 10 ** (abs(gain_db) / 20.0)
        mask = mask * (gain_linear - 1) + 1

        # Apply filter
        filtered_spec = spec * mask.to(spec.device)

        # IFFT
        filtered = torch.fft.irfft(filtered_spec, n=audio.size(-1), dim=-1)  # pylint: disable=not-callable

        return filtered  # type: ignore[no-any-return]

    @staticmethod
    def time_mask(audio: torch.Tensor, mask_width: int) -> torch.Tensor:
        """Mask random time segment."""
        batch_size = audio.size(0)
        length = audio.size(-1)

        masked = audio.clone()

        for i in range(batch_size):
            start = random.randint(0, max(0, length - mask_width))
            masked[i, :, start : start + mask_width] = 0

        return masked

    @staticmethod
    def add_vinyl_noise(audio: torch.Tensor, intensity: float) -> torch.Tensor:
        """Fügt hinzu: vinyl-specific noise (clicks, pops, surface noise)."""
        # Surface noise (pink noise)
        pink_noise = torch.randn_like(audio)
        # Approximate pink noise with lowpass filter
        pink_noise = F.avg_pool1d(pink_noise.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)  # pylint: disable=not-callable

        # Clicks and pops (random impulses)
        clicks = torch.zeros_like(audio)
        n_clicks = int(audio.size(-1) * intensity * 0.001)

        for i in range(audio.size(0)):
            click_positions = np.random.randint(0, audio.size(-1), n_clicks)
            clicks[i, :, click_positions] = np.random.randn(len(click_positions)) * 0.5  # type: ignore[assignment]

        # Combine
        augmented = audio + pink_noise * intensity * 0.1 + clicks

        return augmented

    @staticmethod
    def add_tape_noise(audio: torch.Tensor, intensity: float) -> torch.Tensor:
        """Fügt hinzu: tape-specific noise (hiss, wow, flutter)."""
        # Tape hiss (high-frequency noise)
        hiss = torch.randn_like(audio) * intensity * 0.05

        # Wow and flutter (slow pitch variations)
        length = audio.size(-1)
        t = torch.linspace(0, 1, length)

        # Low-frequency modulation (wow)
        wow_freq = 0.5 + np.random.rand() * 1.5  # 0.5-2 Hz
        wow = torch.sin(2 * np.pi * wow_freq * t) * intensity * 0.002

        # High-frequency modulation (flutter)
        flutter_freq = 5 + np.random.rand() * 15  # 5-20 Hz
        flutter = torch.sin(2 * np.pi * flutter_freq * t) * intensity * 0.001

        modulation = 1 + wow + flutter

        # Apply modulation (simple amplitude modulation, not true pitch modulation)
        augmented = audio * modulation.to(audio.device)

        # Add hiss
        augmented = augmented + hiss

        return augmented

    @staticmethod
    def add_mp3_artifacts(audio: torch.Tensor, quality: float) -> torch.Tensor:
        """Simulate MP3 compression artifacts."""
        # Simplified: add quantization noise in frequency domain
        spec = torch.fft.rfft(audio, dim=-1)  # pylint: disable=not-callable

        # Quantize based on quality (lower quality = more quantization)
        quantization_step = (1.0 - quality) * 0.1

        quantized_real = torch.round(spec.real / quantization_step) * quantization_step
        quantized_imag = torch.round(spec.imag / quantization_step) * quantization_step

        quantized_spec = torch.complex(quantized_real, quantized_imag)

        augmented = torch.fft.irfft(quantized_spec, n=audio.size(-1), dim=-1)  # pylint: disable=not-callable

        return augmented  # type: ignore[no-any-return]

    @staticmethod
    def dynamic_range_compression(audio: torch.Tensor, threshold: float, ratio: float) -> torch.Tensor:
        """Wendet an: dynamic range compression."""
        # Compute envelope
        envelope = torch.abs(audio)

        # Apply compression above threshold
        compressed = torch.where(envelope > threshold, threshold + (envelope - threshold) / ratio, envelope)

        # Maintain phase
        compressed = compressed * torch.sign(audio)

        return compressed


class AugmentationPolicy:
    """
    Augmentation policy containing sequence of operations.
    """

    def __init__(self, operations: list[tuple[str, float]], material_type: str | None = None):
        """
        Initialisiert augmentation policy.

        Args:
            operations: List of (operation_name, magnitude) tuples
            material_type: Optional material specialization
        """
        self.operations = operations
        self.material_type = material_type
        self.audio_aug = AudioAugmentations()

        # Map operation names to functions
        self.op_map = {
            "time_stretch": lambda a, m: self.audio_aug.time_stretch(a, 0.8 + m * 0.4),
            "pitch_shift": lambda a, m: self.audio_aug.pitch_shift(a, int((m - 0.5) * 4)),
            "add_noise": lambda a, m: self.audio_aug.add_noise(a, m * 0.1),
            "gain": lambda a, m: self.audio_aug.gain(a, (m - 0.5) * 20),
            "eq_filter": lambda a, m: self.audio_aug.eq_filter(a, 1000 + m * 5000, (m - 0.5) * 6),
            "time_mask": lambda a, m: self.audio_aug.time_mask(a, int(m * 4800)),
            "add_vinyl_noise": lambda a, m: self.audio_aug.add_vinyl_noise(a, m),
            "add_tape_noise": lambda a, m: self.audio_aug.add_tape_noise(a, m),
            "add_mp3_artifacts": lambda a, m: self.audio_aug.add_mp3_artifacts(a, m),
            "dynamic_range_compression": lambda a, m: self.audio_aug.dynamic_range_compression(a, 0.3, 2 + m * 8),
        }

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Wendet Augmentierungsrichtlinie auf Audio an."""
        augmented = audio

        for op_name, magnitude in self.operations:
            if op_name in self.op_map:
                try:
                    augmented = self.op_map[op_name](augmented, magnitude)
                except Exception as e:
                    logger.warning("konnte nicht anwenden %s: %s", op_name, e)

        return augmented


class RandAugment:
    """
    RandAugment for audio.

    Randomly selects N operations with random magnitudes.
    """

    def __init__(self, n_ops: int = 2, magnitude: float = 0.5, material_type: str | None = None) -> None:
        """
        Initialisiert RandAugment.

        Args:
            n_ops: Number of operations to apply
            magnitude: Magnitude of augmentations (0-1)
            material_type: Optional material specialization
        """
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.material_type = material_type

        # Available operations
        self.operations = ["time_stretch", "pitch_shift", "add_noise", "gain", "eq_filter", "time_mask"]

        # Add material-specific operations
        if material_type == "vinyl":
            self.operations.append("add_vinyl_noise")
        elif material_type in ["tape_shellac", "tape_cassette", "tape_reel"]:
            self.operations.append("add_tape_noise")
        elif material_type == "mp3":
            self.operations.append("add_mp3_artifacts")

        logger.info("RandAugment initialisiert: n_ops=%s, magnitude=%s, material=%s", n_ops, magnitude, material_type)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Wendet an: random augmentations."""
        # Select random operations
        selected_ops = random.sample(self.operations, min(self.n_ops, len(self.operations)))

        # Create policy
        policy_ops = [(op, self.magnitude) for op in selected_ops]
        policy = AugmentationPolicy(policy_ops, self.material_type)

        return policy.apply(audio)


class AutoAugment:
    """
    AutoAugment for audio.

    Learns optimal augmentation policies through search.
    """

    def __init__(self, n_policies: int = 5, n_ops_per_policy: int = 2, material_type: str | None = None) -> None:
        """
        Initialisiert AutoAugment.

        Args:
            n_policies: Number of augmentation policies
            n_ops_per_policy: Number of operations per policy
            material_type: Optional material specialization
        """
        self.n_policies = n_policies
        self.n_ops_per_policy = n_ops_per_policy
        self.material_type = material_type

        # Initialize random policies
        self.policies: list = self._initialize_policies() or []

        logger.info("AutoAugment initialisiert: %s policies, %s ops per policy", n_policies, n_ops_per_policy)

    def _initialize_policies(self) -> list[AugmentationPolicy]:
        """Initialisiert random policies."""
        operations = ["time_stretch", "pitch_shift", "add_noise", "gain", "eq_filter", "time_mask"]

        # Add material-specific operations
        if self.material_type == "vinyl":
            operations.append("add_vinyl_noise")
        elif self.material_type in ["tape_shellac", "tape_cassette", "tape_reel"]:
            operations.append("add_tape_noise")
        elif self.material_type == "mp3":
            operations.append("add_mp3_artifacts")

        policies = []

        for _ in range(self.n_policies):
            policy_ops = []
            for _ in range(self.n_ops_per_policy):
                op = random.choice(operations)
                magnitude = random.random()
                policy_ops.append((op, magnitude))

            policies.append(AugmentationPolicy(policy_ops, self.material_type))

        return policies

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Wendet an: random policy."""
        policy = random.choice(self.policies)
        return policy.apply(audio)  # type: ignore[no-any-return]

    def search_policies(
        self, model: nn.Module, train_loader, val_loader, n_iterations: int = 50, device: str = "cpu"
    ) -> None:  # §9.5 CPU-only
        """
        Search for optimal augmentation policies.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            n_iterations: Number of search iterations
            device: Device to use
        """
        logger.info("Searching for optimal augmentation policies (%s iterations)...", n_iterations)

        best_policies = None
        best_val_loss = float("inf")

        for iteration in range(n_iterations):
            # Generate new policies
            self.policies = self._initialize_policies()

            # Train model with these policies
            model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Apply augmentation
                augmented_x = self(batch_x)

                # Forward pass (without updating model, just evaluate)
                with torch.no_grad():
                    pred = model(augmented_x)
                    loss = F.mse_loss(pred, batch_y)
                    train_losses.append(loss.item())

            # Evaluate on validation set
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    pred = model(batch_x)
                    loss = F.mse_loss(pred, batch_y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            # Update best policies
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss  # type: ignore[assignment]
                best_policies = [AugmentationPolicy(p.operations, p.material_type) for p in self.policies]

            if iteration % 10 == 0:
                logger.info("Iteration %s: Val Loss = %.4f, Best = %.4f", iteration, avg_val_loss, best_val_loss)

        self.policies = best_policies  # type: ignore[assignment]
        logger.info("Policy search abgeschlossen! Best val loss: %.4f", best_val_loss)

    def save_policies(self, path: Path) -> None:
        """Speichert learned policies."""
        policies_data = []

        for policy in self.policies:
            policies_data.append({"operations": policy.operations, "material_type": policy.material_type})

        with open(path, "w") as f:
            json.dump(policies_data, f, indent=2)

        logger.info("Policies gespeichert to %s", path)

    def load_policies(self, path: Path) -> None:
        """Lädt policies from file."""
        with open(path) as f:
            policies_data = json.load(f)

        self.policies = [AugmentationPolicy(p["operations"], p.get("material_type")) for p in policies_data]

        logger.info("geladen %s policies from %s", len(self.policies), path)


class ConsistencyTraining:
    """
    Consistency Training with augmentations.

    Enforces model consistency across different augmentations of same input.
    """

    def __init__(
        self, model: nn.Module, augmentation: Callable, consistency_weight: float = 1.0, device: str = "cpu"
    ) -> None:  # §9.5 CPU-only
        """
        Initialisiert consistency training.

        Args:
            model: Model to train
            augmentation: Augmentation function
            consistency_weight: Weight for consistency loss
            device: Device to use (always 'cpu' per §9.5 CPU-only policy)
        """
        self.model = model.to(device)
        self.augmentation = augmentation
        self.consistency_weight = consistency_weight
        self.device = device

        logger.info("ConsistencyTraining initialisiert: consistency_weight=%s", consistency_weight)

    def train_step(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> dict[str, float]:
        """
        Single training step with consistency loss.

        Args:
            batch_x: Input batch
            batch_y: Target batch
            optimizer: Optimizer

        Returns:
            Dict with loss values
        """
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Standard prediction
        pred = self.model(batch_x)
        task_loss = F.mse_loss(pred, batch_y)

        # Augmented prediction
        augmented_x = self.augmentation(batch_x)
        augmented_pred = self.model(augmented_x)

        # Consistency loss (predictions should be similar)
        consistency_loss = F.mse_loss(pred, augmented_pred)

        # Total loss
        total_loss = task_loss + self.consistency_weight * consistency_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "consistency_loss": consistency_loss.item(),
        }


# Example usage
if __name__ == "__main__":
    # Test RandAugment
    logger.debug("=== RandAugment ===")
    rand_augment = RandAugment(n_ops=2, magnitude=0.5, material_type="vinyl")

    audio = torch.randn(2, 1, 48000)
    augmented = rand_augment(audio)

    logger.debug("Originalsignal shape: %s", audio.shape)
    logger.debug("Augmented shape: %s", augmented.shape)

    # Test AutoAugment
    logger.debug("\n=== AutoAugment ===")
    auto_augment = AutoAugment(n_policies=3, n_ops_per_policy=2, material_type="tape_cassette")

    augmented = auto_augment(audio)
    logger.debug("Augmented shape: %s", augmented.shape)

    # Show policies
    logger.debug("\nPolicies:")
    for i, policy in enumerate(auto_augment.policies):
        logger.debug("  Policy %s: %s", i + 1, policy.operations)
