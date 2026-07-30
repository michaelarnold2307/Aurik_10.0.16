"""§G124 Integrationstest: Pipeline-Simulation mit depth=4.

Simuliert eine Kassetten-Pipeline (chain_depth=4) und prüft dass:
- CIG keinen False-Rollback auslöst (GDD-Schwelle 58.5ms > typischer Drift)
- Constitution kein False-Veto auslöst (af-Schwelle 0.70)
- Phase_19 minimum-phase Filter verwendet (safe_sosfiltfilt)
- CalibrationContext korrekt gesetzt und abrufbar ist
- PMGG depth-korrekt skaliert
- Alle depth-abhängigen Schwellwerte monoton mit depth steigen
"""

import numpy as np
import pytest

SR = 48000


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture: CalibrationContext für depth=4 (Kassette)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _set_depth4_context():
    """Setzt CalibrationContext auf depth=4 Kassette vor jedem Test."""
    from backend.core.calibration_context import (
        CalibrationContext,
        set_calibration_context,
    )
    ctx = CalibrationContext(
        restorability_score=64.0,
        transfer_chain_depth=4,
        material_type="cassette",
        snr_db=14.3,
        bandwidth_hz=12000.0,
    )
    set_calibration_context(ctx)
    yield ctx
    # Cleanup: auf None zurücksetzen


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: CalibrationContext ist erreichbar
# ═══════════════════════════════════════════════════════════════════════════════

def test_calibration_context_is_set(_set_depth4_context):
    """§G122: CalibrationContext muss nach set_ erreichbar sein."""
    from backend.core.calibration_context import get_calibration_context

    ctx = get_calibration_context()
    assert ctx is not None, "CalibrationContext nicht gesetzt"
    assert ctx.transfer_chain_depth == 4
    assert ctx.material_type == "cassette"
    assert ctx.is_deep_chain is True
    assert ctx.chain_factor == 1.50
    assert ctx.artifact_freedom_min == 0.70


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: CIG GDD — kein False-Rollback bei depth=4
# ═══════════════════════════════════════════════════════════════════════════════

def test_cig_no_false_rollback_depth4():
    """CIG GDD-Schwelle muss hoch genug sein, dass typischer Kassetten-Drift
    keinen Rollback auslöst."""
    from backend.core.cumulative_interaction_guard import (
        InteractionGuardState,
        CumulativeInteractionGuard,
    )

    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.transfer_chain_depth = 4
    state.restorability_score = 64.0
    state.material_type = "cassette"

    # Phase_29 (tape_hiss_reduction) — spectral subtraction auf Kassette
    gdd_threshold = guard._compute_gdd_threshold("phase_29_tape_hiss_reduction", state)

    # Typischer Drift aus realem Log: -39.86 ms
    typical_drift = 39.86

    assert abs(gdd_threshold) > typical_drift, (
        f"GDD-Schwelle {abs(gdd_threshold):.1f}ms zu niedrig "
        f"für typischen Drift {typical_drift:.1f}ms → False-Rollback"
    )

    # Auch Phase_03 (denoise) prüfen
    gdd_03 = guard._compute_gdd_threshold("phase_03_denoise", state)
    assert abs(gdd_03) >= 50.0, f"Phase_03 GDD={abs(gdd_03):.1f}ms zu niedrig"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Constitution — kein False-Veto bei depth=4
# ═══════════════════════════════════════════════════════════════════════════════

def test_constitution_no_false_veto_depth4():
    """Constitution darf bei depth=4 und af=0.75 kein Veto auslösen."""
    from backend.core.spec_constitution import get_constitution

    const = get_constitution()

    # af=0.75 ist realistisch nach Kassetten-Restaurierung
    result = const.check_paragraph_zero(
        None, 48000,
        artifact_freedom=0.75,
        hpi=0.6,
        chain_depth=4,
    )
    veto = [v for v in result if "VETO" in v and "artifact_freedom" in v]
    assert not veto, f"False-Veto bei depth=4, af=0.75: {veto}"


def test_constitution_correctly_vetoes_below_threshold():
    """Unter 0.70 MUSS ein Veto kommen — auch bei depth=4."""
    from backend.core.spec_constitution import get_constitution

    const = get_constitution()
    result = const.check_paragraph_zero(
        None, 48000,
        artifact_freedom=0.65,
        hpi=0.6,
        chain_depth=4,
    )
    veto = [v for v in result if "VETO" in v and "artifact_freedom" in v]
    assert veto, "Kein Veto obwohl af=0.65 < 0.70"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Phase_19 — minimum-phase Filter bei depth=4
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase19_uses_minimum_phase_at_depth4():
    """safe_sosfiltfilt muss bei depth=4 sosfilt (nicht sosfiltfilt) aufrufen."""
    from backend.core.audio_utils import safe_sosfiltfilt
    from scipy.signal import butter

    sos = butter(4, [2000 / 24000, 8000 / 24000], btype="band", output="sos")
    sig = np.random.RandomState(123).randn(48000).astype(np.float32)

    # Monkey-patch zum Prüfen, welche Funktion aufgerufen wird
    import scipy.signal as signal
    calls = []

    orig_sosfilt = signal.sosfilt
    orig_sosfiltfilt = signal.sosfiltfilt

    def _mock_sosfilt(*a, **kw):
        calls.append("sosfilt")
        return orig_sosfilt(*a, **kw)

    def _mock_sosfiltfilt(*a, **kw):
        calls.append("sosfiltfilt")
        return orig_sosfiltfilt(*a, **kw)

    try:
        signal.sosfilt = _mock_sosfilt
        signal.sosfiltfilt = _mock_sosfiltfilt

        calls.clear()
        safe_sosfiltfilt(sos, sig, chain_depth=4)
        assert "sosfilt" in calls, f"depth=4 sollte sosfilt rufen, rief: {calls}"
        assert "sosfiltfilt" not in calls, "depth=4 darf NICHT sosfiltfilt rufen"

        calls.clear()
        safe_sosfiltfilt(sos, sig, chain_depth=1)
        assert "sosfiltfilt" in calls, f"depth=1 sollte sosfiltfilt rufen, rief: {calls}"

    finally:
        signal.sosfilt = orig_sosfilt
        signal.sosfiltfilt = orig_sosfiltfilt


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: PMGG — depth=4 korrekt propagiert
# ═══════════════════════════════════════════════════════════════════════════════

def test_pmgg_receives_correct_depth():
    """PMGG _get_adaptive_threshold muss mit depth=4 skalieren."""
    from backend.core.per_phase_musical_goals_gate import _get_adaptive_threshold

    t1 = _get_adaptive_threshold(64.0, "cassette", 1)
    t4 = _get_adaptive_threshold(64.0, "cassette", 4)

    assert t4 > t1, f"depth=4 ({t4:.4f}) muss > depth=1 ({t1:.4f})"
    assert t4 >= 0.060, f"depth=4 threshold ({t4:.4f}) zu niedrig für Kassette"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Alle depth-Stufen monoton
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_thresholds_monotonic_with_depth():
    """Jeder depth-abhängige Schwellwert muss mit depth steigen (oder gleich bleiben)."""
    from backend.core.cumulative_interaction_guard import (
        InteractionGuardState,
        CumulativeInteractionGuard,
    )
    from backend.core.per_phase_musical_goals_gate import _get_adaptive_threshold
    from backend.core.spec_constitution import get_constitution

    guard = CumulativeInteractionGuard()
    const = get_constitution()

    prev_gdd = 0.0
    prev_pmgg = 0.0
    prev_af_min = 1.0

    for depth in range(1, 6):
        state = InteractionGuardState()
        state.transfer_chain_depth = depth
        state.restorability_score = 50.0
        state.material_type = "cassette"

        gdd = abs(guard._compute_gdd_threshold("phase_29_tape_hiss_reduction", state))
        pmgg = _get_adaptive_threshold(50.0, "cassette", depth)

        assert gdd >= prev_gdd, f"GDD non-monoton: depth={depth} ({gdd:.1f}) < prev ({prev_gdd:.1f})"
        assert pmgg >= prev_pmgg, f"PMGG non-monoton: depth={depth} ({pmgg:.4f}) < prev ({prev_pmgg:.4f})"

        prev_gdd = gdd
        prev_pmgg = pmgg


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: UNset-Sentinel schützt vor stillschweigenden Defaults
# ═══════════════════════════════════════════════════════════════════════════════

def test_unset_sentinel_raises_on_int():
    """UNSET darf nicht als int verwendbar sein."""
    from backend.core.calibration_context import UNSET

    with pytest.raises(RuntimeError, match="UNSET"):
        int(UNSET)


def test_unset_sentinel_raises_on_float():
    """UNSET darf nicht als float verwendbar sein."""
    from backend.core.calibration_context import UNSET

    with pytest.raises(RuntimeError, match="UNSET"):
        float(UNSET)


def test_calibration_context_rejects_unset_depth():
    """CalibrationContext darf nicht mit UNSET transfer_chain_depth erstellt werden."""
    from backend.core.calibration_context import CalibrationContext, UNSET

    with pytest.raises(ValueError, match="UNSET"):
        CalibrationContext(
            restorability_score=50.0,
            transfer_chain_depth=UNSET,
            material_type="cassette",
        )
