"""UI-Schwellwerte — zentrale Konstanten für das Frontend.

§V25-Äquivalent für die GUI: JEDE dimensionale/zeitliche Konstante
MUSS aus diesem Modul bezogen werden. Keine Magic Numbers in
modern_window.py oder anderen UI-Dateien.

Analog zu CalibratedConstants für das Backend.
"""

from __future__ import annotations

from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# Layout & Dimensionen (alle in Pixeln)
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 800
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600

SIDEBAR_WIDTH = 280
SIDEBAR_COLLAPSED_WIDTH = 48

PROGRESS_BAR_HEIGHT = 6
STATUS_BAR_HEIGHT = 24

PHASE_LIST_ROW_HEIGHT = 28
PHASE_LIST_ICON_SIZE = 16

# ═══════════════════════════════════════════════════════════════════════════════
# Timing (alle in Millisekunden)
# ═══════════════════════════════════════════════════════════════════════════════

POLL_INTERVAL_MS = 50  # Pipeline-Status-Polling
HEARTBEAT_INTERVAL_MS = 1000  # Watchdog-Check-Intervall
ANIMATION_DURATION_MS = 300  # Übergangs-Animationen
TOOLTIP_DELAY_MS = 500
DEBOUNCE_MS = 200  # Input-Debounce für Suchfelder

# ═══════════════════════════════════════════════════════════════════════════════
# Spacing & Padding
# ═══════════════════════════════════════════════════════════════════════════════

PADDING_TIGHT = 4
PADDING_NORMAL = 8
PADDING_WIDE = 12
PADDING_SECTION = 16

SPACING_TIGHT = 2
SPACING_NORMAL = 6
SPACING_WIDE = 10

MARGIN_CONTENT = 8

# ═══════════════════════════════════════════════════════════════════════════════
# Farben / Opacity
# ═══════════════════════════════════════════════════════════════════════════════

OPACITY_DISABLED = 0.4
OPACITY_HOVER = 0.85
OPACITY_ACTIVE = 1.0
OPACITY_OVERLAY = 0.92

# ═══════════════════════════════════════════════════════════════════════════════
# Depth-abhängige UI-Werte (aus CalibrationContext)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DepthAwareUI:
    """UI-Werte die von der Transfer-Chain-Tiefe abhängen."""

    chain_depth: int = 1
    material_confidence: float = 1.0  # §v10.303.9

    @property
    def confidence_multiplier(self) -> float:
        """§v10.303.9: Bei niedriger Material-Confidence weniger Phasen."""
        if self.material_confidence < 0.25:
            return 0.40  # 60% weniger Phasen
        elif self.material_confidence < 0.35:
            return 0.55  # 45% weniger
        elif self.material_confidence < 0.50:
            return 0.75
        return 1.0

    @property
    def quality_color(self) -> str:
        """Eingeschränkte Qualitätsfarbe für tiefe Ketten."""
        if self.chain_depth >= 4:
            return "#E6A817"  # Bernstein — erwartete Einschränkungen
        elif self.chain_depth >= 3:
            return "#4CAF50"  # Grün — moderate Qualität
        return "#2196F3"  # Blau — Studio-Qualität

    @property
    def expected_duration_factor(self) -> float:
        """Erwartete längere Dauer für tiefe Ketten."""
        if self.chain_depth >= 4:
            return 2.5  # 2.5× länger als depth=1
        elif self.chain_depth >= 3:
            return 1.8
        return 1.0

    @property
    def phase_count_estimate(self) -> int:
        """Geschätzte Phasenanzahl — Confidence-bewusst."""
        if self.chain_depth >= 4:
            _base = 43
        elif self.chain_depth >= 3:
            _base = 35
        else:
            _base = 25  # Studio-Master
        return max(12, int(_base * self.confidence_multiplier))

    @property
    def progress_warning_threshold_pct(self) -> int:
        """Warnschwelle für Fortschritt in Prozent."""
        if self.chain_depth >= 4:
            return 75  # Längere Pipeline — später warnen
        return 85


def DEPTH_AWARE_UI_FACTORY(chain_label: str) -> DepthAwareUI:
    """Erzeugt DepthAwareUI aus einem Chain-Label wie 'reel_tape → vinyl → cassette → mp3_low'."""
    stages = chain_label.count(" → ") + 1 if chain_label else 1
    return DepthAwareUI(chain_depth=stages)
