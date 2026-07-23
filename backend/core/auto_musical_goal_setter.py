"""
core/auto_musical_goal_setter.py
Automatic Musical Goal Setter (AMGS)
======================================

Formuliert vollautomatisch die musikalischen Ziele für die Restaurierung.
Der Nutzer gibt NUR den Modus an (RESTORATION | STUDIO_2026).
Alle Zielwerte werden intern aus Material, Defekt-Profil und Baseline-Qualität
abgeleitet — kein Nutzereingriff erforderlich.

Ziel-Dimensionen:
  - target_snr_db         : Ziel-Signal-Rausch-Abstand in dB
  - target_authenticity   : Originalcharakter-Erhalt (0–1)
  - target_naturalness    : Natürlichkeit des Klangs (0–1)
  - target_clarity        : Transparenz / Intelligibility (0–1)
  - target_warmth         : Tonale Wärme (0–1)
  - target_brightness     : Hochtonpräsenz (0–1)
  - target_dynamic_range  : Dynamikumfang in dB
  - target_lufs           : Ziel-Lautheitsnorm (EBU R128)
  - denoise_strength      : Rauschunterdrückungsstärke (automatisch)
  - declip_strength       : Clipping-Reparatur-Stärke (automatisch)
  - click_sensitivity     : Knack/Klick-Erkennung (automatisch)
  - preserve_character    : Analogcharakter erhalten?

Architektur:
  1. ModeGoalTemplate     – Basiswerte je Modus (RESTORATION / STUDIO_2026)
  2. MaterialAdjuster     – Korrekturen je erkanntem Medium
  3. DefectScaler         – Stärkanpassung basierend auf Defekt-Severity
  4. QualityCalibrator    – Feinjustierung basierend auf Eingangsqualität
  5. GoalValidator        – Sicherheitscheck: Keine widersprüchlichen Ziele

Author: Aurik Development Team
Version: 1.0.0 "Zero-Intervention Goal Intelligence"
Date: 2026-02-17
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

from backend.core.defect_scanner import DefectAnalysisResult, DefectType, MaterialType
from backend.core.defect_scanner import MaterialType  # §v10.113
from backend.core.processing_modes import ProcessingMode
from backend.core.quality_prediction import QualityEstimate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ziel-Profil (Output)
# ---------------------------------------------------------------------------


@dataclass
class MusicalGoalProfile:
    """
    Vollständiges musikalisches Zielprofil — vollautomatisch berechnet.

    Alle Felder sind normiert:
      float-Werte: 0.0 (Min) bis 1.0 (Max) sofern nicht anders angegeben.
      dB-Werte: physikalische dB-Einheit.
    """

    mode: str
    """Gewählter Nutzer-Modus (restoration | studio_2026)."""

    material: str
    """Automatisch erkanntes Quellmaterial."""

    # === Qualitäts-Ziele ===
    target_snr_db: float
    """Ziel-SNR in dB nach Verarbeitung."""

    target_dynamic_range_db: float
    """Ziel-Dynamikumfang in dB (EBU R128 Loudness Range)."""

    target_lufs: float
    """Ziel-Lautheit in LUFS (EBU R128 Integriert)."""

    # === Klangcharakter-Ziele ===
    target_authenticity: float
    """Originalcharakter-Erhalt (0=vollständig verändert, 1=unberührt)."""

    target_naturalness: float
    """Natürlichkeit des Klangs (0=künstlich, 1=perfekt natürlich)."""

    target_clarity: float
    """Transparenz/Klarheit (0=undurchsichtig, 1=kristallklar)."""

    target_warmth: float
    """Tonale Wärme (0=kalt/analytisch, 1=warm/analog)."""

    target_brightness: float
    """Hochtonpräsenz/Air (0=dunkel, 1=strahlend)."""

    # === Processing-Stärken (für Varianten-Builder) ===
    denoise_strength: float
    """Richtlinie für Rauschunterdrückungsstärke (0–1)."""

    declip_strength: float
    """Richtlinie für Clipping-Reparatur (0–1)."""

    click_sensitivity: float
    """Richtlinie für Klick/Knack-Entfernung (0–1)."""

    dereverb_strength: float
    """Richtlinie für Nachhall-Entfernung (0–1)."""

    # === Erhaltungs-Flags ===
    preserve_character: bool
    """Analogcharakter (Tape-Saturation, Vinyl-Wärme) erhalten?"""

    preserve_breaths: bool
    """Atemgeräusche bei Gesang erhalten?"""

    preserve_room_tone: bool
    """Natürliches Raumambiente erhalten?"""

    # === Metadaten ===
    confidence: float
    """Konfidenz dieser Zielformulierung (0–1)."""

    rationale: str
    """Kurze Begründung der Zielwahl (für Audit-Log)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert das Zielprofil als Dictionary für Logging, Audit und Telemetrie."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Mode-Templates
# ---------------------------------------------------------------------------

_RESTORATION_BASE = {
    "target_snr_db": 42.0,
    "target_dynamic_range_db": 14.0,
    "target_lufs": -16.0,
    "target_authenticity": 0.90,
    "target_naturalness": 0.88,
    "target_clarity": 0.80,
    "target_warmth": 0.75,
    "target_brightness": 0.55,
    "denoise_strength": 0.30,
    "declip_strength": 0.45,
    "click_sensitivity": 0.55,
    "dereverb_strength": 0.15,
    "preserve_character": True,
    "preserve_breaths": True,
    "preserve_room_tone": True,
}

_STUDIO_2026_BASE = {
    "target_snr_db": 58.0,
    "target_dynamic_range_db": 9.0,
    "target_lufs": -14.0,
    "target_authenticity": 0.88,
    "target_naturalness": 0.90,
    "target_clarity": 0.95,
    "target_warmth": 0.60,
    "target_brightness": 0.88,
    "denoise_strength": 0.70,
    "declip_strength": 0.75,
    "click_sensitivity": 0.80,
    "dereverb_strength": 0.45,
    "preserve_character": False,
    "preserve_breaths": True,
    "preserve_room_tone": False,
}


# ---------------------------------------------------------------------------
# Material-Korrektursätze
# ---------------------------------------------------------------------------

# Jedes Material hat Deltas auf die Basis-Targets.
# Positive Werte erhöhen den Zielwert, negative senken ihn.
# Die Werte sind auf die material-spezifischen Ziele aus Spec §6.2 abgestimmt.
_MATERIAL_ADJUSTMENTS: dict[MaterialType, dict[str, float]] = {
    MaterialType.WAX_CYLINDER: {
        "target_snr_db": -8.0,
        "target_authenticity": +0.10,
        "target_naturalness": -0.04,
        "target_clarity": -0.08,
        "target_warmth": +0.10,
        "target_brightness": -0.20,
        "denoise_strength": +0.28,
        "declip_strength": -0.12,
        "click_sensitivity": +0.20,
        "dereverb_strength": -0.05,
    },
    MaterialType.SHELLAC: {
        "target_snr_db": -5.0,  # Shellac ist inhärent rauschreich — realistischere Erwartung
        "target_authenticity": +0.08,
        "target_warmth": +0.12,
        "target_brightness": -0.10,
        "denoise_strength": +0.20,
        "click_sensitivity": +0.25,
        "declip_strength": -0.10,
    },
    MaterialType.LACQUER_DISC: {
        "target_snr_db": -6.0,
        "target_authenticity": +0.08,
        "target_naturalness": -0.03,
        "target_clarity": -0.05,
        "target_warmth": +0.10,
        "target_brightness": -0.14,
        "denoise_strength": +0.22,
        "declip_strength": -0.08,
        "click_sensitivity": +0.24,
        "dereverb_strength": -0.02,
    },
    MaterialType.WIRE_RECORDING: {
        "target_snr_db": -7.0,
        "target_authenticity": +0.08,
        "target_naturalness": -0.04,
        "target_clarity": -0.07,
        "target_warmth": +0.06,
        "target_brightness": -0.16,
        "denoise_strength": +0.24,
        "declip_strength": +0.05,
        "click_sensitivity": +0.14,
        "dereverb_strength": -0.04,
    },
    MaterialType.VINYL: {
        "target_snr_db": -2.0,
        "target_authenticity": +0.05,
        "target_warmth": +0.08,
        "target_brightness": -0.05,
        "denoise_strength": +0.10,
        "click_sensitivity": +0.15,
    },
    MaterialType.TAPE: {
        "target_snr_db": -3.0,
        "target_authenticity": +0.06,
        "target_warmth": +0.10,
        "denoise_strength": +0.12,
        "dereverb_strength": +0.10,
    },
    MaterialType.REEL_TAPE: {
        "target_snr_db": 0.0,
        "target_authenticity": +0.04,
        "target_warmth": +0.06,
        "denoise_strength": +0.08,
    },
    MaterialType.CASSETTE: {
        "target_snr_db": -2.0,
        "target_authenticity": +0.05,
        "target_warmth": +0.08,
        "target_brightness": -0.04,
        "denoise_strength": +0.10,
        "declip_strength": +0.05,
        "click_sensitivity": +0.08,
        "dereverb_strength": +0.06,
    },
    MaterialType.DAT: {
        "target_snr_db": +4.0,
        "target_authenticity": -0.02,
        "target_naturalness": +0.04,
        "target_clarity": +0.04,
        "target_brightness": +0.04,
        "denoise_strength": -0.05,
        "click_sensitivity": -0.08,
    },
    MaterialType.CD_DIGITAL: {
        "target_snr_db": +5.0,
        "target_authenticity": -0.05,
        "target_naturalness": +0.05,
        "target_clarity": +0.04,
        "target_brightness": +0.05,
        "denoise_strength": -0.10,
        "click_sensitivity": -0.10,
    },
    MaterialType.MP3_LOW: {
        "target_snr_db": -4.0,
        "target_authenticity": -0.06,
        "target_naturalness": -0.04,
        "target_clarity": -0.05,
        "target_brightness": -0.10,
        "denoise_strength": +0.05,
        "declip_strength": +0.20,  # Starke Codec-Reparatur
        "dereverb_strength": -0.05,
    },
    MaterialType.MP3_HIGH: {
        "target_snr_db": +1.0,
        "target_authenticity": -0.03,
        "target_clarity": +0.03,
        "target_brightness": +0.02,
        "denoise_strength": -0.05,
        "declip_strength": +0.10,
    },
    MaterialType.AAC: {
        "target_snr_db": +2.0,
        "target_authenticity": -0.04,
        "target_naturalness": +0.02,
        "target_clarity": +0.04,
        "target_brightness": +0.05,
        "denoise_strength": -0.05,
        "declip_strength": +0.12,
    },
    MaterialType.MINIDISC: {
        "target_snr_db": -2.0,
        "target_authenticity": -0.03,
        "target_clarity": +0.02,
        "declip_strength": +0.15,
        "target_brightness": -0.05,
    },
    MaterialType.STREAMING: {
        "target_snr_db": +3.0,
        "target_authenticity": -0.08,
        "target_clarity": +0.05,
        "target_brightness": +0.08,
        "denoise_strength": -0.05,
        "declip_strength": +0.10,  # Codec-Artefakte reparieren
    },
    MaterialType.UNKNOWN: {},  # Keine Anpassung
}


# §6.2 Referenzwerte für material-spezifische Mindest-MOS-Erwartung.
_MATERIAL_PQS_TARGETS: dict[MaterialType, float] = {
    MaterialType.WAX_CYLINDER: 3.5,
    MaterialType.SHELLAC: 3.8,
    MaterialType.LACQUER_DISC: 3.7,
    MaterialType.WIRE_RECORDING: 3.6,
    MaterialType.VINYL: 4.0,
    MaterialType.TAPE: 4.2,
    MaterialType.REEL_TAPE: 4.3,
    MaterialType.CASSETTE: 4.1,
    MaterialType.DAT: 4.5,
    MaterialType.CD_DIGITAL: 4.5,
    MaterialType.MP3_LOW: 3.9,
    MaterialType.MP3_HIGH: 4.2,
    MaterialType.AAC: 4.2,
    MaterialType.MINIDISC: 4.0,
    MaterialType.STREAMING: 4.1,  # §6.2: Dropouts, Codec-Artefakte, Bitrate-Varianz
    MaterialType.UNKNOWN: 3.8,
}


# ---------------------------------------------------------------------------
# AutoMusicalGoalSetter
# ---------------------------------------------------------------------------


class AutoMusicalGoalSetter:
    """
    Berechnet vollautomatisch das musikalische Zielprofil.

    Keine Nutzereingabe jenseits des Modus (RESTORATION | STUDIO_2026).
    """

    def __init__(self, mode: ProcessingMode):
        self.mode = mode
        self._base = dict(_RESTORATION_BASE) if mode == ProcessingMode.RESTORATION else dict(_STUDIO_2026_BASE)

    def compute_goals(
        self,
        defect_result: DefectAnalysisResult,
        quality_estimate: QualityEstimate,
    ) -> MusicalGoalProfile:
        """
        Berechnet das vollständige musikalische Zielprofil.

        1. Start mit Modus-Basiswerten
        2. Material-Korrekturen anwenden
        3. Defekt-Severity skalieren
        4. Eingangsqualität einbeziehen
        5. Validierung (Konsistenz)

        Args:
            defect_result: Ergebnis der DefectScanner-Analyse.
            quality_estimate: Baseline-Qualitätsschätzung.

        Returns:
            MusicalGoalProfile — vollständig und konsistent.
        """
        params = dict(self._base)
        material = defect_result.material_type
        rationale_parts: list[str] = [
            f"Modus={self.mode.value}",
            f"Material={material.value}",
        ]

        # ----------------------------------------------------------------
        # Schritt 1: Material-Korrekturen
        # ----------------------------------------------------------------
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        adjustments = _MATERIAL_ADJUSTMENTS.get(_mk, {})
        for key, delta in adjustments.items():
            if key in params and isinstance(params[key], float):
                params[key] = params[key] + delta
        if adjustments:
            rationale_parts.append(f"Material-Korrekturen: {list(adjustments.keys())}")
        target_mos = _MATERIAL_PQS_TARGETS.get(_mk, _MATERIAL_PQS_TARGETS[MaterialType.UNKNOWN])
        rationale_parts.append(f"PQS-Ziel(material)≥{target_mos:.1f}")

        # ----------------------------------------------------------------
        # Schritt 2: Defekt-Severity skalieren
        # ----------------------------------------------------------------
        top_defects = defect_result.get_top_defects(n=5)
        for defect_score in top_defects:
            s = defect_score.severity
            defect_type = defect_score.defect_type

            if defect_type == DefectType.CLICKS and s > 0.3:
                params["click_sensitivity"] = min(1.0, params["click_sensitivity"] + s * 0.3)
                rationale_parts.append(f"Clicks-Severity={s:.2f}→click_sensitivity erhöht")

            elif defect_type == DefectType.HIGH_FREQ_NOISE and s > 0.3:
                params["denoise_strength"] = min(1.0, params["denoise_strength"] + s * 0.25)
                rationale_parts.append(f"HF-Noise-Severity={s:.2f}→denoise_strength erhöht")

            elif defect_type in (DefectType.DIGITAL_ARTIFACTS, DefectType.COMPRESSION_ARTIFACTS) and s > 0.3:
                params["declip_strength"] = min(1.0, params["declip_strength"] + s * 0.2)
                rationale_parts.append(f"Codec-Artefakt-Severity={s:.2f}→declip_strength erhöht")

            elif defect_type in (DefectType.WOW, DefectType.FLUTTER) and s > 0.4:
                # Wow/Flutter → vorsichtiger, kein Over-Processing
                params["denoise_strength"] = max(0.1, params["denoise_strength"] - 0.05)
                rationale_parts.append(f"Wow/Flutter-Severity={s:.2f}→denoise_strength gesenkt")

            elif defect_type == DefectType.DROPOUTS and s > 0.3:
                params["declip_strength"] = min(1.0, params["declip_strength"] + s * 0.15)
                rationale_parts.append(f"Dropouts-Severity={s:.2f}→declip_strength erhöht")

            elif defect_type == DefectType.HUM and s > 0.3:
                # Brummen: unabhängig vom Modus bekämpfen
                rationale_parts.append(f"Hum-Severity={s:.2f}→Hum-Entfernung aktiv")

        # ----------------------------------------------------------------
        # Schritt 3: Eingangsqualität einbeziehen
        # ----------------------------------------------------------------
        q = quality_estimate.overall_score  # 0–100
        confidence = quality_estimate.confidence

        if q < 30.0:
            # Sehr schlechte Eingangsqualität → stärkere Eingriffe nötig
            params["denoise_strength"] = min(1.0, params["denoise_strength"] * 1.25)
            params["declip_strength"] = min(1.0, params["declip_strength"] * 1.20)
            rationale_parts.append(f"Low-Quality-Input={q:.1f}→Stärken erhöht")
        elif q > 75.0:
            # Bereits gute Qualität → konservativer vorgehen
            params["denoise_strength"] = max(0.05, params["denoise_strength"] * 0.75)
            params["click_sensitivity"] = max(0.1, params["click_sensitivity"] * 0.80)
            rationale_parts.append(f"High-Quality-Input={q:.1f}→Konservativer Ansatz")

        # Bei RESTORATION: Authentizität niemals unter 0.70 senken
        if self.mode == ProcessingMode.RESTORATION:
            params["target_authenticity"] = max(0.70, params["target_authenticity"])

        # ----------------------------------------------------------------
        # Schritt 4: Klemmen auf valide Bereiche
        # ----------------------------------------------------------------
        float_params_range = {
            "target_authenticity": (0.0, 1.0),
            "target_naturalness": (0.0, 1.0),
            "target_clarity": (0.0, 1.0),
            "target_warmth": (0.0, 1.0),
            "target_brightness": (0.0, 1.0),
            "denoise_strength": (0.0, 1.0),
            "declip_strength": (0.0, 1.0),
            "click_sensitivity": (0.0, 1.0),
            "dereverb_strength": (0.0, 1.0),
        }
        for param, (lo, hi) in float_params_range.items():
            if param in params:
                val = float(params[param])
                if not math.isfinite(val):
                    val = (lo + hi) / 2.0
                params[param] = float(max(lo, min(hi, val)))

        # SNR-Ziele: physikalisch sinnvoll
        params["target_snr_db"] = float(
            max(
                20.0,
                min(80.0, float(params["target_snr_db"]) if math.isfinite(float(params["target_snr_db"])) else 42.0),
            )
        )
        params["target_dynamic_range_db"] = float(
            max(
                4.0,
                min(
                    30.0,
                    (
                        float(params["target_dynamic_range_db"])
                        if math.isfinite(float(params["target_dynamic_range_db"]))
                        else 14.0
                    ),
                ),
            )
        )
        params["target_lufs"] = float(
            max(
                -23.0, min(-8.0, float(params["target_lufs"]) if math.isfinite(float(params["target_lufs"])) else -16.0)
            )
        )

        # ----------------------------------------------------------------
        # Schritt 5: Widerspruchsprüfung
        # ----------------------------------------------------------------
        # Hohe Authentizität + aggressives De-Noising widersprechen sich
        if params["target_authenticity"] > 0.85 and params["denoise_strength"] > 0.70:
            params["denoise_strength"] = 0.70
            rationale_parts.append("Conflict-Resolution: denoise auf 0.70 wegen hoher Authenticity")

        # Studio_2026 + preserve_character wäre widersprüchlich
        if self.mode == ProcessingMode.STUDIO_2026 and params.get("preserve_character", False):
            params["preserve_character"] = False
            rationale_parts.append("Conflict-Resolution: preserve_character deaktiviert (Studio_2026)")

        logger.debug("Musikalische Ziele: %s", params)

        return MusicalGoalProfile(
            mode=self.mode.value,
            material=material.value,
            target_snr_db=params["target_snr_db"],
            target_dynamic_range_db=params["target_dynamic_range_db"],
            target_lufs=params["target_lufs"],
            target_authenticity=params["target_authenticity"],
            target_naturalness=params["target_naturalness"],
            target_clarity=params["target_clarity"],
            target_warmth=params["target_warmth"],
            target_brightness=params["target_brightness"],
            denoise_strength=params["denoise_strength"],
            declip_strength=params["declip_strength"],
            click_sensitivity=params["click_sensitivity"],
            dereverb_strength=params["dereverb_strength"],
            preserve_character=bool(params["preserve_character"]),
            preserve_breaths=bool(params["preserve_breaths"]),
            preserve_room_tone=bool(params["preserve_room_tone"]),
            confidence=float(confidence),
            rationale=" | ".join(rationale_parts),
        )
