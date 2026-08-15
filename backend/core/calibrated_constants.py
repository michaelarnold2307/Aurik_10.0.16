"""§V25/§G76 (GEBOTE.md)/§G77 (GEBOTE.md) Kalibrierte Konstanten — Single Source of Truth.

JEDER Schwellwert, Floor, Cap, Blend-Faktor in Aurik MUSS
AUSSCHLIESSLICH aus diesem Modul bezogen werden.

Keine numerische Konstante in irgendeinem anderen Modul.
Keine diskreten Buckets. Keine Lookup-Tabellen.

Alle Werte werden als kontinuierliche Funktionen aus dem
CalibrationContext abgeleitet (§G77 (GEBOTE.md)).

Verwendung:
    from backend.core.calibrated_constants import get_constants
    const = get_constants(ctx)
    threshold = const.regression_threshold
    gdd_ms = const.gdd_spectral_ms("phase_29")

Migration aus hartcodierten Konstanten:
    VORHER:  REGRESSION_THRESHOLD_GOOD = 0.035
    NACHHER: const.regression_threshold  (aus CalibrationContext)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from backend.core.calibration_context import CalibrationContext, get_calibration_context

# ═══════════════════════════════════════════════════════════════════════════════
# Physikalische Konstanten (AUSNAHME von §V25 — physikalische Wahrheiten)
# ═══════════════════════════════════════════════════════════════════════════════

DIGITAL_BLACK_DBFS = -60.0  # §V25-Ausnahme: physikalische Konstante
TRUE_PEAK_CEILING_DBTP = -0.3  # ITU-R BS.1770
NYQUIST_FACTOR = 0.95  # Sicherheitsabstand zu Nyquist
MIN_SIGNAL_LENGTH_S = 0.1  # 100ms Minimum für sinnvolle Analyse


# ═══════════════════════════════════════════════════════════════════════════════
# CalibratedConstants — alle Schwellwerte an EINEM Ort
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True, frozen=True)
class CalibratedConstants:
    """Alle kalibrierten Schwellwerte für EINE Pipeline-Instanz.

    Wird EINMAL pro Pipeline-Lauf aus dem CalibrationContext erzeugt.
    Alle Module rufen get_constants() statt eigener Konstanten.
    """

    # ── Basis-Werte aus Context ──
    restorability_score: float
    transfer_chain_depth: int
    material_type: str
    snr_db: float
    bandwidth_hz: float

    # ═══════════════════════════════════════════════════════════════════════
    # PMGG: Regression Threshold
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def regression_threshold(self) -> float:
        """§2.29 Material- und Restorability-adaptiver REGRESSION_THRESHOLD."""
        rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        if rs >= 70.0:
            base = 0.035
        elif rs >= 40.0:
            base = 0.050
        else:
            base = 0.065
        bonus = _MATERIAL_THRESHOLD_BONUS.get(str(self.material_type).lower(), 0.003)
        depth = max(1, int(self.transfer_chain_depth))
        depth_bonus = max(0, depth - 2) * 0.008
        return float(np.clip(base + bonus + depth_bonus, 0.012, 0.070))

    # ═══════════════════════════════════════════════════════════════════════
    # CIG: GDD Schwellen
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def gdd_base_ms(self) -> float:
        """Basis-GDD-Schwelle für nicht-spektrale Phasen."""
        return _GDD_BASE_MS

    @property
    def gdd_spectral_base_ms(self) -> float:
        """Basis-GDD-Schwelle für spectral-subtraction Phasen."""
        return _GDD_SPECTRAL_BASE_MS

    @property
    def gdd_restorability_factor(self) -> float:
        """Restorability-Faktor für GDD-Schwellen."""
        rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        if rs < 70.0:
            return 1.0 + min(1.5, (70.0 - rs) / 20.0)
        return 1.0

    @property
    def gdd_material_factor(self) -> float:
        """Material-Faktor: Analog-Träger bekommen 3.0× für spectral-sub."""
        if str(self.material_type).lower() in _ANALOG_MATERIALS:
            return 3.0
        return 1.0

    @property
    def gdd_analog_factor(self) -> float:
        """Zusätzlicher Analog-Faktor für nicht-spektrale Phasen."""
        if str(self.material_type).lower() in _ANALOG_MATERIALS:
            return 1.4
        return 1.0

    @property
    def chain_factor(self) -> float:
        """§v10.120 Depth-adaptiver Chain-Faktor (identisch zu gdd_chain_factor)."""
        depth = max(1, int(self.transfer_chain_depth))
        return 1.0 + max(0, depth - 2) * 0.25

    @property
    def gdd_chain_factor(self) -> float:
        """§v10.120 Depth-adaptiver GDD-Chain-Faktor (Alias)."""
        return self.chain_factor

    def gdd_spectral_ms(self, phase_id: str = "") -> float:
        """GDD-Schwelle für spectral-subtraction Phasen (phase_03, phase_29, …)."""
        return (
            self.gdd_spectral_base_ms * self.gdd_restorability_factor * self.gdd_material_factor * self.gdd_chain_factor
        )

    def gdd_general_ms(self) -> float:
        """GDD-Schwelle für allgemeine STFT-Phasen."""
        return self.gdd_base_ms * self.gdd_restorability_factor * self.gdd_analog_factor * self.gdd_chain_factor

    # ═══════════════════════════════════════════════════════════════════════
    # Constitution: artifact_freedom
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def artifact_freedom_min(self) -> float:
        """§v10.119 Depth-adaptiver artifact_freedom-Mindestwert."""
        depth = max(1, int(self.transfer_chain_depth))
        if depth >= 4:
            return 0.70
        elif depth == 3:
            return 0.80
        elif depth == 2:
            return 0.88
        return 0.95

    # ═══════════════════════════════════════════════════════════════════════
    # SFT: Novelty / Hallucination-Guard
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def hg_base_threshold(self) -> float:
        """§v10.122 Depth-adaptiver Hallucination-Guard Basis-Schwellwert."""
        depth = max(1, int(self.transfer_chain_depth))
        if depth >= 5:
            return 0.55
        elif depth == 4:
            return 0.40
        elif depth == 3:
            return 0.30
        elif depth == 2:
            return 0.22
        return 0.15

    @property
    def novelty_crit_scale(self) -> float:
        """§v10.41 Restorability- und Depth-adaptiver NOVELTY_CRIT-Skalierungsfaktor."""
        depth = max(1, int(self.transfer_chain_depth))
        rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        base = 0.15 + depth * 0.10
        if rs >= 70:
            base *= 0.7
        elif rs <= 30:
            base *= 1.4
        return float(np.clip(base, 0.15, 0.65))

    # ═══════════════════════════════════════════════════════════════════════
    # Joint-Calibration
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def min_phase_strength(self) -> float:
        """§G71 (GEBOTE.md) Adaptive min_strength aus Restorability."""
        rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        if rs >= 90:
            return 0.20
        elif rs >= 60:
            return 0.35
        elif rs >= 30:
            return 0.40
        return 0.45

    @property
    def depth_boost(self) -> float:
        """§v10.58 Depth-Boost für extreme Ketten."""
        depth = max(1, int(self.transfer_chain_depth))
        return float(np.clip(1.0 + (depth - 1) * 0.12, 1.0, 1.50))

    # ═══════════════════════════════════════════════════════════════════════
    # Phase_19 De-Esser
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def use_minimum_phase_filter(self) -> bool:
        """§v10.131: True wenn depth≥4 → minimum-phase für De-Esser."""
        return max(1, int(self.transfer_chain_depth)) >= 4

    @property
    def deesser_depth_factor(self) -> float:
        """§v10.120 Depth-Faktor für De-Essing-Stärke."""
        depth = max(1, int(self.transfer_chain_depth))
        if depth >= 4:
            return float(np.clip(1.0 - (depth - 3) * 0.15, 0.55, 1.0))
        return 1.0

    # ═══════════════════════════════════════════════════════════════════════
    # SFT: Echo / HNR / Wet-Ceilings (migriert aus signal_flow_tracer Globals)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def echo_corr_threshold(self) -> float:
        """§v10.131 Depth-adaptive Echo-Korrelations-Schwelle.

        depth=1-2: 0.35 (Studio-Master, strikt)
        depth=3:   0.45 (moderate chain)
        depth≥4:   0.55 (Kassette — harmonische Resonanz, kein Artefakt)
        """
        depth = max(1, int(self.transfer_chain_depth))
        if depth >= 4:
            return 0.55
        elif depth >= 3:
            return 0.45
        return 0.35

    @property
    def hnr_warn_db(self) -> float:
        """Harmonics-to-Noise-Ratio Warn-Schwelle in dB."""
        depth = max(1, int(self.transfer_chain_depth))
        vocal_conf = float(getattr(self, "vocal_confidence", 0.0) or 0.0)
        base = 2.0 if vocal_conf > 0.3 else 3.0
        return float(base + max(0, depth - 2) * 0.5)

    @property
    def hnr_crit_db(self) -> float:
        """Harmonics-to-Noise-Ratio Kritische Schwelle in dB."""
        depth = max(1, int(self.transfer_chain_depth))
        return float(4.0 + max(0, depth - 2) * 1.0)

    @property
    def sft_wet_ceiling_nonrepair(self) -> float:
        """§G71 (GEBOTE.md) SFT-Wet-Ceiling für Nicht-Reparatur-Phasen."""
        depth = max(1, int(self.transfer_chain_depth))
        return float(np.clip(0.72 + (depth - 1) * 0.05, 0.65, 0.90))

    @property
    def sft_wet_ceiling_repair(self) -> float:
        """§G71 (GEBOTE.md) SFT-Wet-Ceiling für Reparatur-Phasen."""
        depth = max(1, int(self.transfer_chain_depth))
        return float(np.clip(0.82 + (depth - 1) * 0.05, 0.75, 0.95))

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline-Kalibrierung
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def crest_tolerance_db(self) -> float:
        """Kalibrierte Crest-Toleranz."""
        rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        depth = max(1, int(self.transfer_chain_depth))
        base = 6.0 + (100.0 - rs) * 0.06
        return float(np.clip(base + depth * 0.5, 4.0, 18.0))

    @property
    def early_abort_phase_pct(self) -> float:
        """Anteil der Phasen, ab dem Early-Abort greift."""
        rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        depth = max(1, int(self.transfer_chain_depth))
        if rs >= 80:
            return 0.60
        elif rs >= 50:
            return 0.75
        return 0.85 + depth * 0.02

    # ═══════════════════════════════════════════════════════════════════════
    # CIG / Groove / Onset (Komplettierung)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def drift_tolerance(self) -> float:
        """§2.54 Adaptive Drift-Toleranz für CIG."""
        rs = float(np.clip(self.restorability_score, 0.0, 100.0))
        depth = max(1, int(self.transfer_chain_depth))
        base = 0.04 if rs >= 70 else (0.06 if rs >= 40 else 0.08)
        return float(np.clip(base + max(0, depth - 2) * 0.015, 0.04, 0.12))

    @property
    def max_rollbacks(self) -> int:
        """§2.54 Adaptive max Rollbacks."""
        depth = max(1, int(self.transfer_chain_depth))
        return max(5, 3 + depth)

    @property
    def onset_preservation_min(self) -> float:
        """§v10.131 Depth-adaptive minimale Onset-Erhaltung."""
        depth = max(1, int(self.transfer_chain_depth))
        if depth >= 4:
            return 0.65
        elif depth >= 3:
            return 0.80
        return 0.95

    @property
    def snr_depth_factor(self) -> float:
        """§v10.131 Depth-Faktor für SNR-Gates."""
        depth = max(1, int(self.transfer_chain_depth))
        return float(np.clip(1.0 - max(0, depth - 3) * 0.15, 0.50, 1.0))

    # ═══════════════════════════════════════════════════════════════════════
    # Convenience
    # ═══════════════════════════════════════════════════════════════════════

    def to_dict(self) -> dict[str, Any]:
        """Alle Werte als Dict für Logging."""
        return {
            "regression_threshold": round(self.regression_threshold, 4),
            "gdd_spectral_ms": round(self.gdd_spectral_ms(), 1),
            "gdd_general_ms": round(self.gdd_general_ms(), 1),
            "artifact_freedom_min": round(self.artifact_freedom_min, 2),
            "hg_base_threshold": round(self.hg_base_threshold, 2),
            "min_phase_strength": round(self.min_phase_strength, 2),
            "depth_boost": round(self.depth_boost, 2),
            "crest_tolerance_db": round(self.crest_tolerance_db, 1),
            "use_minimum_phase_filter": self.use_minimum_phase_filter,
        }

    @classmethod
    def from_context(cls, ctx: CalibrationContext | None = None) -> CalibratedConstants:
        """Erzeugt CalibratedConstants aus CalibrationContext.

        Wenn kein Context übergeben wird, wird get_calibration_context() versucht.
        """
        if ctx is None:
            ctx = get_calibration_context()
        if ctx is None:
            # Fallback für Tests ohne Pipeline-Kontext
            ctx = CalibrationContext(
                restorability_score=50.0,
                transfer_chain_depth=1,
                material_type="unknown",
            )
        return cls(
            restorability_score=float(ctx.restorability_score),
            transfer_chain_depth=int(ctx.transfer_chain_depth),
            material_type=str(ctx.material_type),
            snr_db=float(getattr(ctx, "snr_db", 30.0)),
            bandwidth_hz=float(getattr(ctx, "bandwidth_hz", 20000.0)),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Interne Lookup-Tabellen (NUR HIER — nirgendwo sonst im Code)
# ═══════════════════════════════════════════════════════════════════════════════

_MATERIAL_THRESHOLD_BONUS: dict[str, float] = {
    "vinyl": 0.005,
    "shellac": 0.008,
    "tape": 0.004,
    "cassette": 0.006,
    "reel_tape": 0.004,
    "wax_cylinder": 0.010,
    "wire_recording": 0.006,
    "lacquer_disc": 0.007,
}

_ANALOG_MATERIALS: frozenset[str] = frozenset(
    {
        "vinyl",
        "shellac",
        "tape",
        "cassette",
        "reel_tape",
        "wax_cylinder",
        "wire_recording",
        "lacquer_disc",
    }
)

# Basis-GDD-Schwellen (physikalisch begründet, NICHT empirisch geraten)
_GDD_BASE_MS = 5.0  # 5ms: minimale STFT-Gruppenlaufzeit bei 48kHz/2048
_GDD_SPECTRAL_BASE_MS = 10.0  # 10ms: NR-Phasen entfernen Rausch-Phaseninhalt

# ═══════════════════════════════════════════════════════════════════════════════
# §v10.998: Schutznetz-Kalibrierung — kontinuierliche Funktionen statt
# hartcodierter Schwellwerte (§V25–§V28 Kalibrierungs-Hoheit)
# ═══════════════════════════════════════════════════════════════════════════════


def _rs01(cal: CalibratedConstants) -> float:
    """Restorability 0–100 auf 0.0–1.0 (kontinuierlich)."""
    return float(np.clip(cal.restorability_score, 0.0, 100.0) / 100.0)


class _ProtectionCalibration:
    """§v10.998: Alle Schutznetz-Schwellwerte — EIN zentraler Ort.

    Jede Schwelle ist eine kontinuierliche Funktion des CalibrationContext
    (restorability_score, transfer_chain_depth, material_type) — KEINE
    diskreten Stützstellen, KEINE hartcodierten Werte im Code (§V25–§V28).
    """

    def __init__(self, cal: CalibratedConstants):
        self._cal = cal

    @property
    def defect_severity_min(self) -> float:
        """Planner-Gate: Unterhalb dieser Schwere triggert ein Befund keine Phase.

        Gutes Material (hohe Restorability) braucht konservativere Befunde:
        kontinuierlich von 0.03 (rs=0) bis 0.08 (rs=100).
        """
        rs01 = _rs01(self._cal)
        return float(0.03 + 0.05 * rs01)

    @property
    def repair_strength_floor(self) -> float:
        """Stärke-Boden: erkannte Defekte müssen WIRKEN können.

        Tiefe Ketten (mehr Vorschädigung) erlauben mehr Eingriff:
        0.20 + depth×0.05, geklammert auf [0.20, 0.45].
        """
        depth = max(1, int(self._cal.transfer_chain_depth))
        return float(np.clip(0.20 + depth * 0.05, 0.20, 0.45))

    @property
    def repair_confidence_floor(self) -> float:
        """Confidence-Schwelle, ab der der Stärke-Boden greift.

        Schlechtes Material toleriert unsicherere Befunde:
        0.35 − 0.15×rs01, geklammert auf [0.20, 0.35].
        """
        return float(np.clip(0.35 - 0.15 * _rs01(self._cal), 0.20, 0.35))

    @property
    def energy_collapse_ratio(self) -> float:
        """RMS-Ausgang < Verhältnis×RMS-Eingang → Revert.

        Konservativer auf gutem Material: 0.20 + 0.15×rs01 (0.20…0.35).
        """
        return float(0.20 + 0.15 * _rs01(self._cal))

    @property
    def spectral_damage_step_db(self) -> float:
        """Per-Schritt-Spektral-Guard: Median der Band-Abweichung > Schwelle → Revert.

        Gutes Material verträgt weniger: 10.0 − 4.0×rs01 (6.0…10.0 dB).
        """
        return float(10.0 - 4.0 * _rs01(self._cal))

    @property
    def spectral_band_delta_db(self) -> float:
        """Kumulativer Guard: Band-Abweichung pro Band.

        Konservativer auf gutem Material: 5.0 − 2.0×rs01 (3.0…5.0 dB).
        """
        return float(5.0 - 2.0 * _rs01(self._cal))

    @property
    def spectral_bands_over_min(self) -> int:
        """Kumulativer Guard: Anzahl Bänder über der Schwelle → Revert.

        Tiefe Ketten erlauben mehr verteilte Änderung: 5 + depth (5…9).
        """
        return int(5 + max(1, int(self._cal.transfer_chain_depth)))

    @property
    def localized_change_fraction(self) -> float:
        """Lokalitäts-Kriterium: < Anteil der Samples trägt 90% der Änderung."""
        depth = max(1, int(self._cal.transfer_chain_depth))
        return float(np.clip(0.08 + depth * 0.01, 0.08, 0.15))

    @property
    def sibilance_confidence_min(self) -> float:
        """De-Esser-Gate: minimale Sibilanz-Confidence.

        Analog-Material (echte Zischlaute) toleriert weniger: 0.45 − 0.10×rs01
        minus 0.05 für Analogträger.
        """
        val = 0.45 - 0.10 * _rs01(self._cal)
        if str(self._cal.material_type).lower() in _ANALOG_MATERIALS:
            val -= 0.05
        return float(np.clip(val, 0.30, 0.45))

    @property
    def hum_budget_factor(self) -> float:
        """Hum-Budget: max. Entfernung = Faktor × gemessene Hum-Energie.

        Konservativer auf gutem Material: 1.8 − 0.6×rs01 (1.2…1.8).
        """
        return float(1.8 - 0.6 * _rs01(self._cal))

    @property
    def hum_musical_depth(self) -> float:
        """Notch-Tiefe, wenn Musik im Band liegt.

        Schlechtes Material verträgt mehr: 0.15 + 0.15×(1−rs01) (0.15…0.30).
        """
        return float(0.15 + 0.15 * (1.0 - _rs01(self._cal)))

    @property
    def hum_dominance_ratio(self) -> float:
        """Band-Dominanz: Schmalband/Umgebung > Ratio → Hum dominiert.

        Kontinuierlich 0.65…0.80 (tiefe Ketten: lockerer).
        """
        depth = max(1, int(self._cal.transfer_chain_depth))
        return float(np.clip(0.65 + depth * 0.03, 0.65, 0.80))


_protection_cache: dict[int, _ProtectionCalibration] = {}


def get_protection_calibration(cal: CalibratedConstants | None = None) -> _ProtectionCalibration:
    """§v10.998: Zentraler Zugriff auf die Schutznetz-Kalibrierung.

    Alle Guards (CoordinatedRepair, Planner, Phasen) beziehen ihre
    Schwellwerte AUSSCHLIESSLICH hieraus — §V27 kein Kalibrierungs-Silo.
    """
    if cal is None:
        cal = get_constants()
    return _ProtectionCalibration(cal)


# ═══════════════════════════════════════════════════════════════════════════════
# Thread-lokaler Cache
# ═══════════════════════════════════════════════════════════════════════════════

_constants_cache: threading.local = threading.local()


def get_constants(ctx: CalibrationContext | None = None) -> CalibratedConstants:
    """Gibt die kalibrierten Konstanten für den aktuellen Pipeline-Lauf zurück.

    Beim ersten Aufruf wird aus dem CalibrationContext erzeugt und gecached.
    """
    cached = getattr(_constants_cache, "value", None)
    if cached is not None and ctx is None:
        return cached  # type: ignore[no-any-return]
    const = CalibratedConstants.from_context(ctx)
    _constants_cache.value = const
    return const


# Lazy import
import numpy as np
