"""
§v10.19 CLAP-Material-Classifier — Semantic material detection via CLAP embeddings.

Architecture:
  CLAP Encoder (512-dim, frozen, via laion_clap_plugin)
       ↓
  Linear(512→256) + ReLU + Dropout(0.3)
       ↓
  Linear(256→16)  ← 16 Material-Klassen
       ↓
  Softmax → P(vinyl)=0.78, P(reel_tape)=0.15, ...

Compliance (§6.8, §2.19.1):
  - CLAP is NEVER the sole decision-maker. Physical inference has priority.
  - DSP fallback is always available and authoritative on conflict.
  - Consensus weight: 0.15 in the 4-source Multi-Factor Consensus Check.
  - Source tag: "clap_material_v1" for audit trail.

Training:
  - 10,000 synthetic 4s-chunks from DatasetGenerator (DSP degradation chain)
  - CLAP embeddings extracted once and cached
  - Classifier head trained for 100 epochs (~10 min on GPU)
  - Saved to models/forensics/clap_material_head.pt

Usage:
    classifier = ClapMaterialClassifier()
    if classifier.is_trained:
        probs = classifier.predict(clap_embedding)  # dict[str, float]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Material-Ausgangsklassen (16) ──────────────────────────────────────
# Mapped to Aurik's canonical material names (§6.7)
MATERIAL_CLASSES: list[str] = [
    "vinyl",
    "reel_tape",
    "cassette",
    "shellac",
    "wax_cylinder",
    "wire_recording",
    "lacquer_disc",
    "tape",
    "cd_digital",
    "dat",
    "minidisc",
    "mp3_high",
    "mp3_low",
    "aac",
    "streaming",
    "unknown",
]

# Mapping from CLAP material_tags (9 classes) to canonical 16
# CLAP tags that don't map 1:1 (live_recording, studio_recording, broadcast)
# are not used — they're production metadata, not physical carriers.
CLAP_TAG_TO_CANONICAL: dict[str, str] = {
    "vinyl": "vinyl",
    "tape": "reel_tape",
    "shellac": "shellac",
    "digital": "cd_digital",
    "mp3": "mp3_low",
    "aac": "aac",
}


class ClapMaterialClassifier:
    """Lightweight NN head on frozen CLAP embeddings for material classification.

    This is NOT a standalone classifier. It always operates within the
    4-source consensus framework. Physical inference has veto power (§6.8).
    """

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._weights: dict[str, np.ndarray] = {}
        self._is_trained = False
        if self._model_path and self._model_path.exists():
            self._load(self._model_path)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def num_classes(self) -> int:
        return len(MATERIAL_CLASSES)

    # ── Forward pass (numpy, no PyTorch dependency at inference) ───────

    def predict(self, embedding: np.ndarray) -> dict[str, float]:
        """Predict material probabilities from a 512-dim CLAP embedding.

        Args:
            embedding: L2-normalized 512-dim CLAP audio embedding.

        Returns:
            Dict mapping material name → probability [0,1], summing to ~1.0.
        """
        if not self._is_trained:
            return self._untrained_fallback(embedding)

        # Layer 1: Linear(512→256) + ReLU + Dropout (eval mode: scale output)
        w1 = self._weights["fc1.weight"]  # [256, 512]
        b1 = self._weights["fc1.bias"]  # [256]
        x = embedding @ w1.T + b1
        x = np.maximum(0, x)  # ReLU
        x = x * 0.7  # Dropout(0.3) eval scaling

        # Layer 2: Linear(256→16) + Softmax
        w2 = self._weights["fc2.weight"]  # [16, 256]
        b2 = self._weights["fc2.bias"]  # [16]
        logits = x @ w2.T + b2

        # Stable softmax
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()

        return {MATERIAL_CLASSES[i]: float(probs[i]) for i in range(len(MATERIAL_CLASSES))}

    def predict_top(
        self, embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Return top-k materials with probabilities."""
        probs = self.predict(embedding)
        return sorted(probs.items(), key=lambda x: -x[1])[:top_k]

    def _untrained_fallback(self, embedding: np.ndarray) -> dict[str, float]:
        """Fallback when model not trained: uniform distribution.

        This should NEVER be used in production (consensus framework
        requires is_trained=True to activate the CLAP source).
        """
        return {m: 1.0 / len(MATERIAL_CLASSES) for m in MATERIAL_CLASSES}

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        """Load trained weights from .npz file. Handles both named keys
        ('fc1.weight') and Sequential keys ('0.weight', '3.weight')."""
        import numpy as _np

        data = _np.load(str(path))
        raw = {k: data[k] for k in data.files}

        # Normalize Sequential keys if present
        if "fc1.weight" not in raw:
            _fc1_w = None
            _fc1_b = None
            _fc2_w = None
            _fc2_b = None
            for k in raw:
                if k.startswith("0."):
                    if "weight" in k:
                        _fc1_w = raw[k]
                    elif "bias" in k:
                        _fc1_b = raw[k]
                elif k[0].isdigit():
                    if "weight" in k:
                        _fc2_w = raw[k]
                    elif "bias" in k:
                        _fc2_b = raw[k]
            if _fc1_w is not None and _fc2_w is not None:
                self._weights = {
                    "fc1.weight": _fc1_w,
                    "fc1.bias": _fc1_b if _fc1_b is not None else _np.zeros(_fc1_w.shape[0], dtype=_np.float32),
                    "fc2.weight": _fc2_w,
                    "fc2.bias": _fc2_b if _fc2_b is not None else _np.zeros(_fc2_w.shape[0], dtype=_np.float32),
                }
                self._is_trained = True
                logger.info("ClapMaterialClassifier: geladen aus %s (Sequential keys)", path)
                return

        self._weights = raw
        self._is_trained = True
        logger.info("ClapMaterialClassifier: geladen aus %s (%d Parameter)", path, len(self._weights))

    def save(self, path: str | Path) -> None:
        """Save trained weights to .npz file."""
        np.savez_compressed(str(path), **self._weights)
        logger.info("ClapMaterialClassifier: gespeichert nach %s", path)


# ── Integration helper ──────────────────────────────────────────────────

def get_clap_material_classifier(
    model_path: Optional[str | Path] = None,
) -> ClapMaterialClassifier:
    """Factory for ClapMaterialClassifier with default model path."""
    if model_path is None:
        from pathlib import Path as _Path

        model_path = _Path(__file__).resolve().parent.parent.parent.parent / "models" / "forensics" / "clap_material_head.npz"
    return ClapMaterialClassifier(model_path)


def map_clap_tags_to_canonical(material_tags: dict[str, float]) -> dict[str, float]:
    """Map CLAP's 9 material_tags to Aurik's 16 canonical material names.

    CLAP output:  {"vinyl": 0.78, "tape": 0.15, "mp3": 0.05, "digital": 0.02}
    Canonical:    {"vinyl": 0.78, "reel_tape": 0.15, "mp3_low": 0.05, "cd_digital": 0.02}
    """
    canonical: dict[str, float] = {}
    for clap_tag, conf in material_tags.items():
        canon = CLAP_TAG_TO_CANONICAL.get(clap_tag)
        if canon is not None:
            canonical[canon] = max(canonical.get(canon, 0.0), float(conf))
    return canonical
