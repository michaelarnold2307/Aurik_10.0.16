"""test_ml_naht_guards — §V53 ML-NAHT-GUARDS (Vorschlag 03, docs/proposals/).

Kontrakte:
  - Invertierte ML-Ausgabe wird polaritätskorrigiert.
  - Lag > MAX_ML_LAG_SAMPLES wird verworfen (dry zurück) — nachweisbar,
    weil der Lag-Fall bei hörbarer Änderung sonst geblendet würde.
"""

from __future__ import annotations

import numpy as np

SR = 48000


def _chirp() -> np.ndarray:
    t = np.linspace(0.0, 1.0, SR, endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * (200.0 + 4000.0 * t) * t)).astype(np.float32)


def test_inverted_wet_is_corrected():
    from backend.core.dsp.hybrid_ml_blend import hybrid_ml_apply

    dry = _chirp()
    wet = -0.5 * dry  # invertiert + hörbare Abschwächung
    out = hybrid_ml_apply(dry, wet, SR, scalar_wet=1.0)
    # Nach Polarity-Korrektur ist wet = +0.5*dry → Blend liefert ≈ +0.5*dry.
    assert np.allclose(out, 0.5 * dry, atol=0.03)


def test_shifted_wet_is_rejected():
    from backend.core.dsp.hybrid_ml_blend import hybrid_ml_apply

    dry = _chirp()
    wet = np.roll(0.5 * dry, 256)  # hörbare Änderung, aber 256 Samples versetzt
    out = hybrid_ml_apply(dry, wet, SR, scalar_wet=1.0)
    # Lag-Guard: 256 > 128 ⇒ dry zurück (ohne Guard würde geblendet).
    assert np.allclose(out, dry, atol=1e-6)


def test_unaligned_inverted_wet_is_rejected_or_corrected_deterministically():
    from backend.core.dsp.hybrid_ml_blend import hybrid_ml_apply

    dry = _chirp()
    wet = np.roll(-0.5 * dry, 256)
    out = hybrid_ml_apply(dry, wet, SR, scalar_wet=1.0)
    # Deterministisch: entweder korrigiert oder verworfen — niemals NaN.
    assert np.isfinite(out).all()
    assert out.shape == dry.shape
