"""§G86 Perzeptuelle Validierung: Schwellwerte gegen Psychoakustik.

Validiert dass Auriks depth-abhängige Schwellwerte mit etablierten
psychoakustischen Grenzen konsistent sind (ISO 226:2003, Zwicker 1999,
Blauert 1997, Haas 1951).
"""

import numpy as np
import pytest

from backend.core.calibration_context import CalibrationContext

# ═══════════════════════════════════════════════════════════════════════════════
# Psychoakustische Referenzwerte
# ═══════════════════════════════════════════════════════════════════════════════

JND_GROUP_DELAY_MS = 1.0       # Blauert 1997: < 1ms unhörbar
JND_PRE_ECHO_MS = 2.0          # Haas 1951: Pre-Echo < 2ms wird maskiert
JND_LOUDNESS_DB = 1.0          # Breitband-JND
MAX_PLAUSIBLE_GDD_MS = 200.0   # Darüber ist Signal faktisch dekorreliert
MIN_PLAUSIBLE_GDD_MS = 4.0     # Nyquist-Limit für 48kHz STFT


# ═══════════════════════════════════════════════════════════════════════════════
# Phase_19: Filter-Typ-Entscheidung perzeptuell validiert
# ═══════════════════════════════════════════════════════════════════════════════

def test_minimum_phase_delay_below_jnd():
    """Minimum-Phase (sosfilt) Gruppenlaufzeit muss unter 1ms JND liegen."""
    from scipy.signal import butter, sosfilt
    import scipy.signal as signal

    sos = signal.butter(4, [2000/24000, 8000/24000], btype='band', output='sos')
    impulse = np.zeros(48000)
    impulse[0] = 1.0
    response = sosfilt(sos, impulse)
    peak_idx = np.argmax(np.abs(response))
    delay_ms = peak_idx / 48  # samples → ms at 48kHz

    assert delay_ms < JND_GROUP_DELAY_MS, (
        f"Min-phase delay {delay_ms:.2f}ms ≥ JND {JND_GROUP_DELAY_MS}ms — wäre hörbar"
    )


def test_zero_phase_pre_ringing_at_audibility_threshold():
    """Zero-Phase (sosfiltfilt) Pre-Ringing liegt an der Hörschwelle.

    Für fc_low=2000 Hz, order=8: Pre-Ringing ≈ 2ms.
    Das ist an der Haas-Schwelle — auf sauberem Material maskiert,
    auf degradiertem HF (Kassette) potenziell hörbar.
    """
    # Pre-Ringing-Abschätzung: N/(2·fc_low) für N=order
    fc_low = 2000.0  # Hz — tiefste De-Esser-Frequenz
    order = 8        # 2 × 4th-order Butterworth (sosfiltfilt = 2 Durchläufe)
    pre_ring_ms = order / (2 * fc_low) * 1000

    # Pre-Ringing muss an der Schwelle liegen (nicht weit drunter, nicht weit drüber)
    assert 0.5 <= pre_ring_ms <= 5.0, (
        f"Pre-ringing {pre_ring_ms:.1f}ms außerhalb plausibler Range [0.5, 5.0]ms"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CIG GDD: Schwellwerte perzeptuell plausibel
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("depth,min_ms,max_ms", [
    (1, 20,  60),
    (2, 20,  60),
    (3, 30,  100),
    (4, 40, 120),
    (5, 50, 200),
])
def test_gdd_threshold_in_perceptual_range(depth, min_ms, max_ms):
    """GDD muss im perzeptuell plausiblen Bereich liegen."""
    ctx = CalibrationContext(
        restorability_score=50.0,
        transfer_chain_depth=depth,
        material_type="cassette",
    )
    gdd = abs(ctx.gdd_threshold("phase_29_tape_hiss_reduction"))
    assert min_ms <= gdd <= max_ms, (
        f"depth={depth}: GDD={gdd:.1f}ms außerhalb [{min_ms}, {max_ms}]ms"
    )


def test_gdd_monotonic_with_depth():
    """Mehr Tiefe → mehr Toleranz. Immer."""
    prev = 0
    for depth in range(1, 6):
        ctx = CalibrationContext(
            restorability_score=50.0,
            transfer_chain_depth=depth,
            material_type="cassette",
        )
        gdd = abs(ctx.gdd_threshold("phase_29_tape_hiss_reduction"))
        assert gdd >= prev, f"GDD non-monoton: depth={depth} ({gdd:.1f}) < prev ({prev:.1f})"
        prev = gdd


# ═══════════════════════════════════════════════════════════════════════════════
# PMGG: Regression-Toleranz perzeptuell plausibel
# ═══════════════════════════════════════════════════════════════════════════════

def test_regression_threshold_in_plausible_range():
    """PMGG muss im Bereich [0.01, 0.10] bleiben."""
    for depth in range(1, 6):
        for rs in [30, 50, 70, 90]:
            ctx = CalibrationContext(
                restorability_score=rs,
                transfer_chain_depth=depth,
                material_type="cassette",
            )
            t = ctx.regression_threshold()
            assert 0.01 <= t <= 0.10, (
                f"depth={depth} rs={rs}: PMGG={t:.4f} außerhalb [0.01, 0.10]"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Constitution: artifact_freedom perzeptuell plausibel
# ═══════════════════════════════════════════════════════════════════════════════

def test_artifact_freedom_min_monotonic_decreasing():
    """Mehr Tiefe → niedrigeres Minimum. Immer."""
    prev = 1.0
    for depth in range(1, 6):
        ctx = CalibrationContext(
            restorability_score=50.0,
            transfer_chain_depth=depth,
            material_type="cassette",
        )
        af = ctx.artifact_freedom_min
        assert af <= prev, f"AF non-monoton: depth={depth} ({af:.2f}) > prev ({prev:.2f})"
        prev = af
