"""
Neural Architecture Search (NAS) für Aurik 10.0.0

Automatische Suche nach optimalen Netzwerk-Architekturen für:
- DeepFilterNet Varianten
- Demucs Varianten
- Custom Audio Enhancement Networks

Verwendet DARTS (Differentiable Architecture Search) und RL-basierte NAS.

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

import json
import logging
from pathlib import Path
from typing import Any, cast

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
# DARTS (Differentiable Architecture Search)
# ============================================================================


class MixedOp(nn.Module):
    """
    Mixed Operation für DARTS.

    Kombiniert mehrere Operationen mit learnable Weights (Architecture Parameters).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, operations: list[str] | None = None):
        super().__init__()

        # Standard operations for audio processing
        if operations is None:
            operations = [
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

        self.operations = nn.ModuleDict()

        for op_name in operations:
            self.operations[op_name] = self._create_operation(op_name, in_channels, out_channels, stride)

        # Architecture parameters (alphas)
        self.alpha = nn.Parameter(torch.randn(len(operations)))

    def _create_operation(self, op_name: str, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Erstellt operation based on name."""

        if op_name == "none":
            return Zero(stride, in_channels, out_channels)

        elif op_name == "skip_connect":
            if stride == 1 and in_channels == out_channels:
                return nn.Identity()
            else:
                return nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels)
                )

        elif op_name == "conv_3x3":
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_name == "conv_5x5":
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_name == "conv_7x7":
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_name == "dilated_conv_3x3_rate2":
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=2, dilation=2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_name == "dilated_conv_3x3_rate4":
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=4, dilation=4, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_name == "sep_conv_3x3":
            return SepConv(in_channels, out_channels, 3, stride, 1)

        elif op_name == "sep_conv_5x5":
            return SepConv(in_channels, out_channels, 5, stride, 2)

        elif op_name == "avg_pool_3x3":
            if in_channels != out_channels:
                return nn.Sequential(
                    nn.AvgPool1d(3, stride=stride, padding=1),
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                return nn.AvgPool1d(3, stride=stride, padding=1)

        elif op_name == "max_pool_3x3":
            if in_channels != out_channels:
                return nn.Sequential(
                    nn.MaxPool1d(3, stride=stride, padding=1),
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                return nn.MaxPool1d(3, stride=stride, padding=1)

        else:
            raise ValueError(f"Unknown operation: {op_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed operations."""
        # Softmax over architecture parameters
        weights = F.softmax(self.alpha, dim=0)

        # Weighted sum of operations
        op_outputs = [w * cast(nn.Module, op)(x) for w, op in zip(weights, self.operations.values())]
        output = torch.stack(op_outputs, dim=0).sum(dim=0)

        return output

    def get_best_operation(self) -> str:
        """Gibt zurück: operation with highest weight."""
        weights = F.softmax(self.alpha, dim=0)
        best_idx = int(torch.argmax(weights).item())
        return list(self.operations.keys())[best_idx]  # type: ignore[no-any-return]


class Zero(nn.Module):
    """Zero operation (no connection)."""

    def __init__(self, stride, in_channels, out_channels):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # Create output with correct shape
        n, _c, l = x.shape
        out_length = l if self.stride == 1 else (l + self.stride - 1) // self.stride

        return torch.zeros(n, self.out_channels, out_length, device=x.device, dtype=x.dtype)


class SepConv(nn.Module):
    """Separable Convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv1d(
                in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False
            ),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class DARTSCell(nn.Module):
    """
    DARTS Cell für Audio Processing.

    Enthält mehrere MixedOps mit learnable connections.
    """

    def __init__(self, in_channels_0: int, in_channels_1: int, out_channels: int, n_nodes: int = 4):
        super().__init__()

        self.n_nodes = n_nodes
        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.out_channels = out_channels

        # Create mixed operations for all edges
        self.ops = nn.ModuleList()
        self._typed_ops: list[MixedOp] = []

        # Each node connects to all previous nodes
        # After preprocessing, all states have out_channels
        for i in range(n_nodes):
            for j in range(i + 2):  # +2 for input nodes
                mixed_op = MixedOp(out_channels, out_channels)
                self.ops.append(mixed_op)
                self._typed_ops.append(mixed_op)

        # Preprocessing for inputs (may have different channel counts)
        self.preprocess0 = nn.Sequential(
            nn.Conv1d(in_channels_0, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels)
        )
        self.preprocess1 = nn.Sequential(
            nn.Conv1d(in_channels_1, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels)
        )

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            s0: Previous previous state
            s1: Previous state
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self.n_nodes):
            # Sum of all edges to this node
            node_sum = sum(
                self._typed_ops[offset + j](states[j])
                for j in range(i + 2)  # Each node i has i+2 incoming edges (from 2 inputs + i previous nodes)
            )
            states.append(node_sum)
            offset += i + 2  # Move offset by number of edges to this node

        # Concatenate all intermediate nodes
        return torch.cat(states[2:], dim=1)

    def get_genotype(self) -> dict[str, list[tuple[str, int]]]:
        """Extrahiert the best architecture (genotype)."""
        gene: list[tuple[str, int]] = []
        offset = 0

        for i in range(self.n_nodes):
            edges = []
            for j in range(i + 2):
                op = self._typed_ops[offset + j]
                best_op = op.get_best_operation()
                if best_op != "none":
                    edges.append((best_op, j))
            offset += i + 2

            # Keep top-2 edges per node
            edges = sorted(
                edges,
                key=lambda x: F.softmax(self._typed_ops[offset - (i + 2) + x[1]].alpha, dim=0).max().item(),
                reverse=True,
            )[:2]
            gene.extend(edges)

        return {"cell": gene}


class AudioNASNetwork(nn.Module):
    """
    Complete NAS Network for Audio Enhancement.

    Uses DARTS cells to search for optimal architecture.
    """

    def __init__(self, in_channels: int = 1, init_channels: int = 16, n_cells: int = 8, n_nodes: int = 4) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.init_channels = init_channels
        self.n_cells = n_cells

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, 3, padding=1, bias=False), nn.BatchNorm1d(init_channels)
        )

        # Cells
        self.cells = nn.ModuleList()
        self._typed_cells: list[DARTSCell] = []
        channels = init_channels
        prev_channels_0 = init_channels
        prev_channels_1 = init_channels

        for i in range(n_cells):
            # Downsample every 1/3 of cells
            if i in [n_cells // 3, 2 * n_cells // 3]:
                channels *= 2

            cell = DARTSCell(prev_channels_0, prev_channels_1, channels, n_nodes)
            self.cells.append(cell)
            self._typed_cells.append(cell)

            # After cell: output is concat of n_nodes, each with 'channels'
            cell_output_channels = channels * n_nodes
            prev_channels_0 = prev_channels_1  # Old s1
            prev_channels_1 = cell_output_channels  # New cell output

        # Output
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels * n_nodes, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # Enhancement factor
        )

        logger.info("AudioNASNetwork initialisiert: %s cells, %s nodes per cell", n_cells, n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through NAS network."""
        s0 = s1 = self.stem(x)

        for cell in self._typed_cells:
            s0, s1 = s1, cell(s0, s1)

        # Global pooling + classifier
        out: torch.Tensor = cast(torch.Tensor, self.output(s1))

        return out

    def get_genotype(self) -> dict[str, Any]:
        """Extrahiert complete architecture genotype."""
        genotype: dict[str, Any] = {"cells": []}

        for i, cell in enumerate(self._typed_cells):
            genotype["cells"].append({"cell_id": i, "structure": cell.get_genotype()})

        return genotype

    def derive_discrete_architecture(self) -> nn.Module:
        """Derive discrete architecture from searched structure."""
        genotype = self.get_genotype()

        # Build discrete network based on genotype
        # This would be used for final training after NAS
        logger.info("Deriving discrete architecture from NAS...")
        logger.info("Genotype: %s", genotype)

        # For now, return self (in practice, create DiscreteAudioNetwork)
        return self


class NASTrainer:
    """
    Trainer for Neural Architecture Search.

    Implements bilevel optimization:
    - Lower level: Optimize network weights
    - Upper level: Optimize architecture parameters
    """

    def __init__(
        self,
        model: AudioNASNetwork,
        device: str = "cpu",  # CPU-only Policy (Section 9.5) — kein CUDA
        lr_model: float = 0.025,
        lr_arch: float = 3e-4,
        weight_decay: float = 3e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device

        # Separate optimizers for model weights and architecture parameters
        self.optimizer_model = torch.optim.SGD(
            self.model.parameters(), lr=lr_model, momentum=0.9, weight_decay=weight_decay
        )

        # Collect architecture parameters (alphas)
        arch_params = []
        for cell in self.model._typed_cells:
            for op in cell._typed_ops:
                arch_params.append(op.alpha)

        self.optimizer_arch = torch.optim.Adam(arch_params, lr=lr_arch, betas=(0.5, 0.999), weight_decay=1e-3)

        logger.info("NASTrainer initialisiert on %s", device)

    def train_step(
        self, train_data: torch.Tensor, train_target: torch.Tensor, val_data: torch.Tensor, val_target: torch.Tensor
    ) -> dict[str, float]:
        """
        Single training step with bilevel optimization.

        Args:
            train_data: Training input
            train_target: Training target
            val_data: Validation input
            val_target: Validation target
        """
        # Step 1: Update architecture parameters on validation set
        self.optimizer_arch.zero_grad()
        val_pred = self.model(val_data)
        val_loss = F.mse_loss(val_pred, val_target)
        val_loss.backward()
        self.optimizer_arch.step()

        # Step 2: Update model weights on training set
        self.optimizer_model.zero_grad()
        train_pred = self.model(train_data)
        train_loss = F.mse_loss(train_pred, train_target)
        train_loss.backward()
        self.optimizer_model.step()

        return {"train_loss": train_loss.item(), "val_loss": val_loss.item()}

    def search(self, train_loader, val_loader, epochs: int = 50) -> dict[str, Any]:
        """
        Führt aus: architecture search.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of search epochs
        """
        logger.info("Starte NAS for %s epochs...", epochs)

        for epoch in range(epochs):
            epoch_losses = []

            for (train_data, train_target), (val_data, val_target) in zip(train_loader, val_loader):
                train_data = train_data.to(self.device)
                train_target = train_target.to(self.device)
                val_data = val_data.to(self.device)
                val_target = val_target.to(self.device)

                losses = self.train_step(train_data, train_target, val_data, val_target)
                epoch_losses.append(losses)

            avg_train_loss = np.mean([l["train_loss"] for l in epoch_losses])
            avg_val_loss = np.mean([l["val_loss"] for l in epoch_losses])

            if epoch % 10 == 0:
                logger.info("Epoch %s: Train Loss = %.4f, Val Loss = %.4f", epoch, avg_train_loss, avg_val_loss)

        # Extract final architecture
        genotype = self.model.get_genotype()

        logger.info("NAS abgeschlossen!")
        logger.info("Final architecture: %s", genotype)

        return {"genotype": genotype, "final_train_loss": avg_train_loss, "final_val_loss": avg_val_loss}

    def save_architecture(self, path: Path):
        """Speichert discovered architecture."""
        genotype = self.model.get_genotype()

        with open(path, "w") as f:
            json.dump(genotype, f, indent=2)

        logger.info("Architecture gespeichert to %s", path)

    def load_architecture(self, path: Path) -> dict[str, Any]:
        """Lädt architecture from file."""
        with open(path) as f:
            genotype = json.load(f)

        logger.info("Architecture geladen from %s", path)
        return genotype  # type: ignore[no-any-return]


# Example usage
if __name__ == "__main__":
    # Create NAS network
    model = AudioNASNetwork(in_channels=1, init_channels=16, n_cells=8, n_nodes=4)

    # Create trainer
    trainer = NASTrainer(model, device="cpu")

    # Dummy data
    train_data = torch.randn(4, 1, 48000)
    train_target = torch.randn(4, 1)
    val_data = torch.randn(4, 1, 48000)
    val_target = torch.randn(4, 1)

    # Single training step
    losses = trainer.train_step(train_data, train_target, val_data, val_target)
    logger.debug("Losses: %s", losses)

    # Get architecture
    genotype = model.get_genotype()
    logger.debug("Genotype: %s", genotype)
