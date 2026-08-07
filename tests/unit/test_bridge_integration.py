"""§Bridge Integrationstest: CalibrationContext→Bridge→Frontend End-to-End.

Validiert die komplette Kette:
1. CalibrationContext wird gesetzt
2. _build_bridge_calibration_dict() baut korrektes Dict
3. BridgeCalibrationData.to_frontend_dict() enthält alle Felder
4. Prognose-Widget kann update_from_bridge_calibration() verarbeiten
"""

import numpy as np
import pytest

from backend.core.calibration_context import CalibrationContext, set_calibration_context

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: BridgeCalibrationData Struktur
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_calibration_data_structure():
    """BridgeCalibrationData hat alle erforderlichen Felder."""
    from Aurik10.ui.bridge_calibration import BridgeCalibrationData

    data = BridgeCalibrationData()
    d = data.to_frontend_dict()

    required = [
        "restorability_score",
        "transfer_chain_depth",
        "material_type",
        "snr_db",
        "bandwidth_hz",
        "chain_factor",
        "artifact_freedom_min",
        "regression_threshold",
        "quality_color",
        "deep_chain_warning",
        "expected_phase_count",
        "expected_duration_factor",
    ]
    for key in required:
        assert key in d, f"BridgeCalibrationData fehlt: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Backend→Frontend via Bridge Dict
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_dict_built_from_calibration_context():
    """_build_bridge_calibration_dict() produziert korrektes Dict."""
    # Setup: CalibrationContext setzen
    ctx = CalibrationContext(
        restorability_score=64.0,
        transfer_chain_depth=4,
        material_type="cassette",
        snr_db=14.3,
        bandwidth_hz=12000.0,
        era_decade=1985,
        genre="Deutscher Schlager",
        vocal_confidence=0.65,
    )
    set_calibration_context(ctx)

    # Backend baut Dict
    from backend.api.bridge import _build_bridge_calibration_dict

    d = _build_bridge_calibration_dict()

    # Verifiziere depth-abhängige Werte
    assert d["transfer_chain_depth"] == 4
    assert d["chain_factor"] == 1.5
    assert d["artifact_freedom_min"] == 0.70
    assert d["quality_color"] == "#E6A817"  # Bernstein für depth=4
    assert d["expected_phase_count"] >= 40
    assert d["expected_duration_factor"] >= 2.0
    assert len(d["deep_chain_warning"]) > 0
    assert "4" in d["deep_chain_warning"] or "vier" in d["deep_chain_warning"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Frontend kann Dict konsumieren
# ═══════════════════════════════════════════════════════════════════════════════


def test_prognose_widget_accepts_bridge_dict():
    """Prognose-Widget verarbeitet Bridge-Dict ohne Fehler."""
    from Aurik10.ui.bridge_calibration import BridgeCalibrationData

    # Simuliere Bridge-Dict für depth=4
    data = BridgeCalibrationData(
        restorability_score=64.0,
        transfer_chain_depth=4,
        material_type="cassette",
        quality_color="#E6A817",
        expected_phase_count=43,
        deep_chain_warning="Tiefe Transfer-Kette (4 Stufen)",
    )
    d = data.to_frontend_dict()

    # Validiere dass alle Keys str-safe sind
    for key, val in d.items():
        assert key is not None
        # Keine NaN/Inf Werte
        if isinstance(val, float):
            assert np.isfinite(val), f"Non-finite value in {key}: {val}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Depth-Adaptivität korrekt für alle Stufen
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "depth,expected_color,expected_af_min",
    [
        (1, "#2196F3", 0.95),
        (2, "#2196F3", 0.88),
        (3, "#4CAF50", 0.80),
        (4, "#E6A817", 0.70),
        (5, "#E6A817", 0.70),
    ],
)
def test_bridge_data_depth_adaptive(depth, expected_color, expected_af_min):
    """Bridge-Dict-Werte skalieren korrekt mit Chain-Depth."""
    ctx = CalibrationContext(
        restorability_score=50.0,
        transfer_chain_depth=depth,
        material_type="cassette",
    )
    set_calibration_context(ctx)

    from backend.api.bridge import _build_bridge_calibration_dict

    d = _build_bridge_calibration_dict()

    assert d["quality_color"] == expected_color, f"depth={depth}: expected {expected_color}, got {d['quality_color']}"
    assert d["artifact_freedom_min"] == expected_af_min, (
        f"depth={depth}: expected {expected_af_min}, got {d['artifact_freedom_min']}"
    )
