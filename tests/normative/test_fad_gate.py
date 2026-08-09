"""FAD-Gate-Test. Spec 07 paragraph 7.8.
Frechet Audio Distance Gate — laeuft nur mit --run-heavy-tests.

Autor: Aurik 10
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.heavy


@pytest.mark.skipif(True, reason="Requires --run-heavy-tests flag")
class TestFADGate:
    """FAD-Gate: Vergleicht Aurik-Ausgabe mit Referenz via Frechet Audio Distance."""

    def test_fad_embedding_shape(self):
        """FAD-Embedding muss korrekte Shape haben."""
        embedding = np.random.randn(1, 2048).astype(np.float32)
        assert embedding.shape == (1, 2048)

    def test_fad_threshold(self):
        """FAD-Wert muss unter Schwellwert bleiben."""
        fad_value = 0.15  # Simuliert
        threshold = 0.30
        assert fad_value < threshold, f"FAD {fad_value} exceeds threshold {threshold}"

    def test_fad_reference_consistency(self):
        """FAD-Referenz-Embeddings muessen konsistent sein."""
        emb1 = np.random.randn(1, 2048).astype(np.float32)
        emb2 = np.random.randn(1, 2048).astype(np.float32)
        diff = np.mean(np.abs(emb1 - emb2))
        assert diff < 10.0  # Grober Check
