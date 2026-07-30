"""
absolute_quality_gate.py — §v10.210 Absolute Quality Gate

Ersetzt den DSP-Proxy-Vergleich gegen das degradierte Original durch
MUSHRA-basierte absolute Qualitätsbewertung gegen synthetisierte Referenz-Anker.

Prinzip: „Vergleiche gegen das ZIEL, nicht gegen den DEFEKT."

Die MUSHRA-Referenz ist NICHT das degradierte Original, sondern ein
era/genre/material-kalibrierter synthetischer Anker, der repräsentiert,
wie die Aufnahme OHNE Defekte klingen würde.

Integration:
  - PMGG ruft nach jeder Phase absolute_quality_delta() auf
  - Wenn MUSHRA (absolut) steigt, wird DSP-Proxy-Rollback OVERRULED
  - Referenz-Anker aus ReferenceAnchorSynthesizer oder CalibrationMatrix

Author: Aurik 10 Development
Version: 1.0.0 — §v10.210
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Qualitäts-Anker pro Material-Klasse (MUSHRA-Skala 0-100)
# Basierend auf physikalischen Limits des Trägermediums
MATERIAL_ABSOLUTE_QUALITY_ANCHOR: dict[str, float] = {
    "cd_digital": 95.0,
    "dat": 92.0,
    "minidisc": 85.0,
    "streaming": 88.0,
    "mp3_high": 80.0,
    "aac": 82.0,
    "reel_tape": 78.0,
    "vinyl": 75.0,
    "lp": 75.0,
    "tape": 68.0,
    "cassette": 62.0,
    "mp3_low": 55.0,
    "shellac": 45.0,
    "wax_cylinder": 35.0,
    "wire_recording": 40.0,
    "radio_broadcast": 60.0,
    "optical_film": 50.0,
    "lacquer_disc": 48.0,
    "unknown": 70.0,
}


@dataclass
class AbsoluteQualityDelta:
    """Ergebnis der absoluten Qualitätsdifferenz-Berechnung."""

    absolute_mushra: float  # MUSHRA-Score gegen synthetischen Anker
    absolute_mushra_before: float  # Vorheriger MUSHRA-Score
    absolute_delta: float  # Verbesserung (positiv = besser)
    proxy_regression: float  # DSP-Proxy-Regression (negativ = schlechter)
    should_override_rollback: bool
    quality_anchor: float  # Material-spezifischer Qualitäts-Anker
    reason: str = ""

    def summary(self) -> str:
        return (
            f"AbsMUSHRA={self.absolute_mushra:.1f} "
            f"Δ={self.absolute_delta:+.1f} "
            f"ProxyReg={self.proxy_regression:+.3f} "
            f"Override={self.should_override_rollback}"
        )


def compute_absolute_quality_delta(
    *,
    reference_anchor: np.ndarray | None = None,
    audio_before: np.ndarray | None = None,
    audio_after: np.ndarray,
    sr: int = 48000,
    proxy_regression: float = 0.0,
    material_type: str = "unknown",
    transfer_chain_depth: int = 1,
    restorability_score: float = 50.0,
    absolute_mushra_before: float | None = None,
) -> AbsoluteQualityDelta:
    """Berechnet die absolute Qualitätsdifferenz via MUSHRA.

    Args:
        reference_anchor: Synthetischer Referenz-Anker (wenn None → degradiertes Original)
        audio_before: Audio VOR der Phase (für Δ-Berechnung)
        audio_after: Audio NACH der Phase
        sr: Sample-Rate
        proxy_regression: DSP-Proxy-Regression (negativ = PMGG sieht Verschlechterung)
        material_type: Trägermaterial
        transfer_chain_depth: Tiefe der Transfer-Kette
        restorability_score: Restorability-Score
        absolute_mushra_before: Vorheriger absoluter MUSHRA-Score (überspringt Re-Calculation)

    Returns:
        AbsoluteQualityDelta mit Entscheidung ob Rollback overruled werden soll.
    """
    quality_anchor = MATERIAL_ABSOLUTE_QUALITY_ANCHOR.get(
        str(material_type).lower(), 70.0
    )

    # Wenn kein Referenz-Anker verfügbar ist, kann keine absolute Bewertung erfolgen
    if reference_anchor is None and audio_before is None:
        return AbsoluteQualityDelta(
            absolute_mushra=50.0,
            absolute_mushra_before=absolute_mushra_before or 50.0,
            absolute_delta=0.0,
            proxy_regression=proxy_regression,
            should_override_rollback=False,
            quality_anchor=quality_anchor,
            reason="Kein Referenz-Anker verfügbar — keine absolute Bewertung möglich",
        )

    # Mono für MUSHRA
    ref_mono = reference_anchor if reference_anchor is not None else audio_before
    if ref_mono is not None and ref_mono.ndim == 2:
        ref_mono = ref_mono.mean(axis=-1)
    after_mono = audio_after if audio_after.ndim == 1 else audio_after.mean(axis=-1)

    if ref_mono is None:
        return AbsoluteQualityDelta(
            absolute_mushra=50.0,
            absolute_mushra_before=absolute_mushra_before or 50.0,
            absolute_delta=0.0,
            proxy_regression=proxy_regression,
            should_override_rollback=False,
            quality_anchor=quality_anchor,
            reason="Referenz-Audio ist None",
        )

    # MUSHRA-Berechnung
    absolute_mushra_after = 50.0
    try:
        from backend.core.mushra_evaluator import evaluate_mushra

        result = evaluate_mushra(ref_mono, after_mono, sr, compute_anchor=True)
        absolute_mushra_after = float(result.mushra_score)
    except Exception as e:
        logger.debug("AbsoluteQualityGate MUSHRA non-blocking: %s", e)
        absolute_mushra_after = absolute_mushra_before or 50.0

    if absolute_mushra_before is None:
        # Vorher-Wert berechnen wenn nicht übergeben
        if audio_before is not None:
            before_mono = audio_before if audio_before.ndim == 1 else audio_before.mean(axis=-1)
            try:
                from backend.core.mushra_evaluator import evaluate_mushra
                before_result = evaluate_mushra(ref_mono, before_mono, sr, compute_anchor=True)
                absolute_mushra_before = float(before_result.mushra_score)
            except Exception:
                absolute_mushra_before = absolute_mushra_after

    absolute_mushra_before = absolute_mushra_before or absolute_mushra_after
    absolute_delta = absolute_mushra_after - absolute_mushra_before

    # Tiefen-adaptive Toleranz: bei depth≥4 ist schon eine kleine Verbesserung signifikant
    depth = max(1, int(transfer_chain_depth))
    min_improvement = -1.0 if depth >= 4 else 0.5  # Bei depth≥4: selbst Stagnation ist OK

    # Entscheidung: Rollback overrulen wenn absolute Qualität nicht gesunken ist
    # und DSP-Proxy-Regression ein False-Positive sein könnte
    should_override = (
        absolute_delta >= min_improvement
        and proxy_regression < -0.01  # PMGG sieht Verschlechterung
        and absolute_mushra_after >= quality_anchor * 0.5  # Mindestens 50% des Material-Ankers
    )

    reason_parts = []
    if should_override:
        reason_parts.append(
            f"ABSOLUTE Qualität steigt ({absolute_delta:+.1f} MUSHRA) "
            f"trotz Proxy-Regression ({proxy_regression:+.3f})"
        )
        reason_parts.append(f"Rollback OVERRULED — vertraue absolutem Qualitäts-Modell")
    else:
        if absolute_delta < min_improvement:
            reason_parts.append(
                f"Absolute Qualität sinkt ({absolute_delta:+.1f}) — Rollback gerechtfertigt"
            )
        else:
            reason_parts.append(
                f"Absolute Qualität stabil ({absolute_delta:+.1f}) — keine Override-Entscheidung"
            )

    return AbsoluteQualityDelta(
        absolute_mushra=absolute_mushra_after,
        absolute_mushra_before=absolute_mushra_before,
        absolute_delta=absolute_delta,
        proxy_regression=proxy_regression,
        should_override_rollback=should_override,
        quality_anchor=quality_anchor,
        reason="; ".join(reason_parts),
    )


def get_material_quality_anchor(material_type: str) -> float:
    """Gibt den absoluten Qualitäts-Anker für ein Trägermaterial zurück (MUSHRA 0-100)."""
    return MATERIAL_ABSOLUTE_QUALITY_ANCHOR.get(str(material_type).lower(), 70.0)
