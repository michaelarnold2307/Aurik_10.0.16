"""
Depth-adaptive quality thresholds (§v10.18).

Problem: Static artifact_freedom >= 0.95 rejects valid results for deep transfer chains.
Depth-4 cassettes never achieve AF 0.95 → memory stays empty → HPI falls back
to default → quality assessment underestimates good restorations.

Solution: Depth-dependent AF threshold modulated by transfer chain depth and
restorability score. Same calibration pattern as SFT novelty (§G71 (GEBOTE.md)).

Also includes: VERSA degradation-aware fallback for SNR < 10 dB material.

Usage:
    from backend.core.depth_adaptive_quality import (
        depth_adaptive_af_threshold,
        get_quality_fallback_mode,
    )
"""


def _resolve_transfer_chain_depth(value: int | None) -> int:
    """§G86 (GEBOTE.md): Default nur aus CalibrationContext."""
    from backend.core.defect_to_audibility import _resolve_transfer_chain_depth as _resolve

    return _resolve(value)


def depth_adaptive_af_threshold(
    transfer_chain_depth: int | None = None,
    restorability_score: float = 50.0,
) -> float:
    """Compute the minimum acceptable artifact_freedom for a given restoration context.

    Transfer chain depth: number of degradation steps in the signal chain.
      depth 1 (Studio/Digital):    0.95  (strict — near-perfect expected)
      depth 2 (Vinyl/Tape):        0.85
      depth 3 (Cassette 2-step):   0.75
      depth 4+ (Cassette multi):   0.60  (tolerant — much novelty expected)

    Restorability modulation (rs = restorability_score 0-100):
      rs >= 90 (excellent):  +0.05  (higher expectation)
      rs >= 60 (good):       0.00
      rs >= 30 (fair):       -0.05
      rs < 30 (poor):        -0.10  (lower expectation)

    Returns threshold in [0.50, 0.95].
    """
    transfer_chain_depth = _resolve_transfer_chain_depth(transfer_chain_depth)
    # Base threshold by depth
    if transfer_chain_depth <= 1:
        base = 0.95
    elif transfer_chain_depth == 2:
        base = 0.85
    elif transfer_chain_depth == 3:
        base = 0.75
    else:
        base = 0.60

    # Restorability modulation
    if restorability_score >= 90:
        mod = 0.05
    elif restorability_score >= 60:
        mod = 0.00
    elif restorability_score >= 30:
        mod = -0.05
    else:
        mod = -0.10

    threshold = base + mod
    return max(0.50, min(0.95, threshold))


def get_quality_fallback_mode(
    snr_db: float,
    material: str = "digital",
) -> str:
    """Determine whether VERSA or MERT-based fallback should be used for quality scoring.

    VERSA (SingMOS Pro) was trained on modern material and produces unreliable MOS
    values for heavily degraded audio (SNR < 10 dB, shellac, etc.).

    Returns:
        "versa"   — Use VERSA SingMOS Pro (modern material, SNR >= 10 dB)
        "mert"    — Use MERT MUSHRA proxy (degraded material, SNR < 10 dB)
        "dsp"     — Use DSP-only metrics (extreme degradation, SNR < 5 dB)
    """
    if snr_db >= 10.0:
        return "versa"

    # Shellac and very old material always use MERT
    if material in ("shellac", "wax_cylinder"):
        return "mert"

    if snr_db >= 5.0:
        return "mert"

    return "dsp"
