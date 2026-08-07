"""§Bridge: Einheitliche CalibrationContext-Datenstruktur für Backend→Frontend.

DEFINIERT die Bridge-Schnittstelle für ALLE Kalibrierungsdaten.
Backend füllt dieses Dict, Frontend konsumiert es via Bridge-Callback.
Liegt im backend/api/ um §V18 (Bridge-Bypass-Verbot) einzuhalten —
kein Aurik10.ui.* Import aus dem Backend nötig.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# BridgeCalibrationData — die EINZIGE Datenstruktur für Backend→Frontend
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BridgeCalibrationData:
    """§Bridge: Vollständige Kalibrierungsdaten für das Frontend.

    Wird vom Backend (bridge.py) aus dem CalibrationContext befüllt
    und via Callback an das Frontend übergeben.

    KEIN Frontend-Modul darf backend.* direkt importieren.
    KEIN Backend-Modul darf Aurik10.ui.* direkt importieren.
    """

    # ── Kern-Messwerte (aus CalibrationContext) ──
    restorability_score: float = 50.0
    transfer_chain_depth: int = 1
    material_type: str = "unknown"
    snr_db: float = 30.0
    bandwidth_hz: float = 20000.0
    era_decade: int = 1980
    genre: str = "unknown"
    vocal_confidence: float = 0.0

    # ── Abgeleitete Werte (aus CalibratedConstants) ──
    chain_factor: float = 1.0
    artifact_freedom_min: float = 0.95
    regression_threshold: float = 0.05
    gdd_spectral_ms: float = 10.0
    echo_corr_threshold: float = 0.35
    hg_base_threshold: float = 0.15
    min_phase_strength: float = 0.35
    use_minimum_phase_filter: bool = False
    deesser_depth_factor: float = 1.0

    # ── Tiefenabhängige UI-Hinweise ──
    quality_color: str = "#2196F3"
    expected_phase_count: int = 25
    expected_duration_factor: float = 1.0
    deep_chain_warning: str = ""

    # ── Metadaten ──
    calibration_timestamp: float = 0.0
    bridge_version: str = "1.0"

    def to_frontend_dict(self) -> dict[str, Any]:
        """Gibt ein Dict zurück, das das Frontend via Bridge konsumieren kann."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Builder-Funktion: AUSSCHLIESSLICH in backend/api/bridge.py
# (bridge.py importiert bridge_calibration, nicht umgekehrt)
# ═══════════════════════════════════════════════════════════════════════════════
