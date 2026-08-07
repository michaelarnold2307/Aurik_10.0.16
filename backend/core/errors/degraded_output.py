"""Degradierte Ausgabe — Dataclass für Graceful Degradation bei Phasenfehlern.

§15.8: Wenn eine Phase fehlschlägt, liefert die Pipeline ein ``DegradedOutput``
statt abzustürzen. Die aufrufende Phase entscheidet anhand der Warnings und
des ``_is_degraded``-Flags, ob sie weitermachen kann.

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DegradedOutput:
    """Ergebnis einer fehlgeschlagenen Phase mit Graceful Degradation.

    Attributes:
        audio:            Das (ggf. unverändert durchgereichte) Audio.
        sample_rate:      Abtastrate (aus Input übernommen).
        warnings:         Liste aller Warnungen die aufgetreten sind.
        metrics:          Optionale Metriken (leer wenn Phase komplett fehlschlug).
        phase_name:       Name der Phase die degradiert wurde.
        original_error:   Originaler Exception-Text (None wenn kein Fehler).
        _is_degraded:     True wenn die Phase fehlgeschlagen ist und degradiert wurde.
    """

    audio: np.ndarray
    sample_rate: int
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    phase_name: str = ""
    original_error: str | None = None
    _is_degraded: bool = False

    @property
    def is_degraded(self) -> bool:
        """True wenn die Phase fehlgeschlagen ist und degradiert wurde."""
        return self._is_degraded

    @property
    def warning_summary(self) -> str:
        """Kurze Zusammenfassung aller Warnungen."""
        if not self.warnings:
            return f"[{self.phase_name}] Keine Warnungen"
        return (
            f"[{self.phase_name}] {len(self.warnings)} Warnung(en): "
            + "; ".join(self.warnings[:5])
            + ("..." if len(self.warnings) > 5 else "")
        )

    def to_dict(self) -> dict:
        """Serialisierung für Logging/Persistenz."""
        return {
            "phase_name": self.phase_name,
            "sample_rate": self.sample_rate,
            "audio_shape": list(self.audio.shape) if hasattr(self.audio, "shape") else None,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "original_error": self.original_error,
            "is_degraded": self._is_degraded,
        }
