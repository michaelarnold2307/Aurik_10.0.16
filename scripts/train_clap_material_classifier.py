#!/usr/bin/env python3
"""
§v10.19 Train CLAP-Material-Classifier Head.

Trains a lightweight 2-layer MLP on frozen CLAP embeddings to classify
audio into 16 canonical material types.

Pipeline:
  1. DatasetGenerator produces synthetic audio samples with DSP degradation
  2. LAION-CLAP extracts 512-dim embeddings (cached to .npy)
  3. Classifier head (Linear→ReLU→Dropout→Linear→Softmax) trained 100 epochs
  4. Model saved to models/forensics/clap_material_head.npz

Usage:
    python scripts/train_clap_material_classifier.py [--epochs 100] [--samples 10000] [--batch-size 128]

Dependencies:
    - .venv_rocm (GPU) or .venv_aurik (CPU) with PyTorch
    - LAION-CLAP plugin loaded (automatic via laion_clap_plugin)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Check PyTorch availability ──────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available — training requires PyTorch.", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# Model definition
# ═══════════════════════════════════════════════════════════════════════════

from backend.core.forensics.clap_material_classifier import MATERIAL_CLASSES


class ClapMaterialHead(nn.Module):
    """2-layer MLP on frozen CLAP embeddings."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 16, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        # Kaiming init (ReLU)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Training data generation
# ═══════════════════════════════════════════════════════════════════════════


def generate_training_data(num_samples: int = 10_000, cache_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate labeled CLAP embeddings via DSP degradation chain.

    Returns:
        X: (num_samples, 512) CLAP embeddings
        y: (num_samples,) integer labels [0..15]
    """
    from backend.core.forensics.dataset_generator import DatasetGenerator, SyntheticSample
    from backend.core.forensics.signatures import MediaType
    from plugins.laion_clap_plugin import get_laion_clap

    cache_dir = Path(cache_dir or "models/forensics/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"clap_embeddings_{num_samples}.npz"

    # Try cached embeddings first
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(str(cache_file))
        return data["X"], data["y"]

    print(f"Generating {num_samples} synthetic samples with CLAP embeddings...")
    generator = DatasetGenerator()
    clap = get_laion_clap()

    # Map MediaType to our class index
    _MEDIA_TO_IDX: dict[str, int] = {
        "VINYL_LP_STEREO": 0,  # vinyl
        "VINYL_MONO": 0,
        "TAPE_15IPS": 1,  # reel_tape
        "TAPE_7_5IPS": 1,
        "TAPE_3_75IPS": 1,
        "CASSETTE_TYPE_I": 2,  # cassette
        "CASSETTE_TYPE_II": 2,
        "CASSETTE_TYPE_IV": 2,
        "SHELLAC_78": 3,  # shellac
        "WAX_CYLINDER": 4,  # wax_cylinder
        "WIRE_RECORDING": 5,  # wire_recording
        "LACQUER_DISC": 6,  # lacquer_disc
        "CD_STANDARD": 8,  # cd_digital
        "DAT": 9,  # dat
        "MINIDISC": 10,  # minidisc
        "MP3_320": 11,  # mp3_high
        "MP3_128": 12,  # mp3_low
        "AAC_256": 13,  # aac
        "STREAMING": 14,  # streaming
    }

    samples_per_media = max(1, num_samples // len(_MEDIA_TO_IDX))
    X_list, y_list = [], []

    for media_name, class_idx in _MEDIA_TO_IDX.items():
        try:
            media_type = getattr(MediaType, media_name)
        except AttributeError:
            continue

        dataset = generator.generate_medium_dataset(
            media_types=[media_type],
            num_samples_per_medium=samples_per_media,
        )
        for sample in dataset.get("samples", []):
            audio = sample.audio.astype(np.float32)
            sr = getattr(sample, "sample_rate", 48000)
            # Extract CLAP embedding
            try:
                tagged = clap.tag(audio, sr)
                emb = getattr(tagged, "embedding", None)
                if emb is not None and len(emb) == 512:
                    X_list.append(emb.astype(np.float32))
                    y_list.append(class_idx)
            except Exception:
                continue

        print(f"  {media_name}: {samples_per_media} samples → {len([y for y in y_list if y == class_idx])} embeddings")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Cache embeddings
    np.savez_compressed(str(cache_file), X=X, y=y)
    print(f"Cached {len(X)} embeddings to {cache_file}")

    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════


def train(
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_samples: int = 10_000,
    output_path: str | None = None,
) -> None:
    if not HAS_TORCH:
        print("ERROR: PyTorch required for training. Install: pip install torch", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate data
    X, y = generate_training_data(num_samples)
    print(f"Training data: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")

    # Train/val split (80/20)
    n = len(X)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]

    # Model
    model = ClapMaterialHead(input_dim=512, hidden_dim=256, num_classes=16, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict | None = None

    print(f"\nTraining: {epochs} epochs, batch={batch_size}, lr={lr}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        perm = np.random.permutation(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            x_batch = torch.from_numpy(X_train[batch_idx]).to(device)
            y_batch = torch.from_numpy(y_train[batch_idx]).to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            x_val_t = torch.from_numpy(X_val).to(device)
            y_val_t = torch.from_numpy(y_val).to(device)
            logits_val = model(x_val_t)
            val_loss = criterion(logits_val, y_val_t).item()
            val_pred = logits_val.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {total_loss / max(1, n_train // batch_size):.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}"
            )

    # Save best model
    output = Path(output_path or "models/forensics/clap_material_head.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **best_state)
    print(f"\nModel saved to {output} (val_acc={best_val_acc:.4f})")

    # Quick material-wise accuracy
    model.eval()
    with torch.no_grad():
        x_all = torch.from_numpy(X_val).to(device)
        y_all = torch.from_numpy(y_val).to(device)
        preds = model(x_all).argmax(dim=1)
        for cls_idx, name in enumerate(MATERIAL_CLASSES):
            mask = y_all == cls_idx
            if mask.sum() > 0:
                acc = (preds[mask] == cls_idx).float().mean().item()
                print(f"  {name:<18s}: {acc:.3f} ({mask.sum().item()} samples)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLAP Material Classifier Head (§v10.19)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="models/forensics/clap_material_head.npz")
    parser.add_argument("--cache-dir", type=str, default="models/forensics/cache")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.samples,
        output_path=args.output,
    )
