"""Defekt-Karte — Vorher/Nachher-Defekt-Reduktion. Spec v10.206 §4.

Berechnet Defekt-Statistiken für GUI-Visualisierung:
- Defekt-Anzahl vorher/nachher pro Typ
- Defekt-Reduktionsrate
- Heatmap-Daten für Wellenform-Overlay
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DefectCount:
    """Defekt-Zähler für einen Typ."""
    defect_type: str
    count_before: int = 0
    count_after: int = 0
    severity_before: float = 0.0
    severity_after: float = 0.0

    @property
    def reduction(self) -> int:
        return self.count_before - self.count_after

    @property
    def reduction_pct(self) -> float:
        if self.count_before > 0:
            return (self.reduction / self.count_before) * 100.0
        return 0.0

    @property
    def is_fully_fixed(self) -> bool:
        return self.count_after == 0 and self.count_before > 0


@dataclass
class DefectMap:
    """Vorher/Nachher-Defekt-Karte."""
    defects: list[DefectCount] = field(default_factory=list)
    total_before: int = 0
    total_after: int = 0
    audio_duration_s: float = 0.0
    defect_positions_before: list[dict[str, Any]] = field(default_factory=list)
    defect_positions_after: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_reduction_pct(self) -> float:
        if self.total_before > 0:
            return ((self.total_before - self.total_after) / self.total_before) * 100.0
        return 0.0

    @property
    def fully_fixed_types(self) -> list[str]:
        return [d.defect_type for d in self.defects if d.is_fully_fixed]

    @property
    def remaining_types(self) -> list[str]:
        return [d.defect_type for d in self.defects if d.count_after > 0]

    @classmethod
    def from_defect_lists(
        cls,
        defects_before: list[dict[str, Any]],
        defects_after: list[dict[str, Any]],
        audio_duration_s: float = 0.0,
    ) -> DefectMap:
        """Erstellt DefectMap aus zwei Defekt-Listen (vorher/nachher)."""
        def count_by_type(defects):
            counts: dict[str, int] = {}
            sevs: dict[str, list[float]] = {}
            for d in defects:
                dt = d.get("type", d.get("defect_type", "unknown"))
                counts[dt] = counts.get(dt, 0) + 1
                sev = d.get("severity", d.get("score", 0.5))
                sevs.setdefault(dt, []).append(float(sev))
            return counts, sevs

        before_counts, before_sevs = count_by_type(defects_before)
        after_counts, after_sevs = count_by_type(defects_after)

        all_types = set(before_counts.keys()) | set(after_counts.keys())
        defect_list = []
        for dt in sorted(all_types):
            defect_list.append(DefectCount(
                defect_type=dt,
                count_before=before_counts.get(dt, 0),
                count_after=after_counts.get(dt, 0),
                severity_before=sum(before_sevs.get(dt, [0])) / max(1, len(before_sevs.get(dt, [0]))),
                severity_after=sum(after_sevs.get(dt, [0])) / max(1, len(after_sevs.get(dt, [0]))),
            ))

        return cls(
            defects=defect_list,
            total_before=len(defects_before),
            total_after=len(defects_after),
            audio_duration_s=audio_duration_s,
            defect_positions_before=defects_before,
            defect_positions_after=defects_after,
        )

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "defects_before": self.total_before,
                "defects_after": self.total_after,
                "reduction": f"{self.total_reduction_pct:.0f}%",
                "fully_fixed_types": len(self.fully_fixed_types),
                "remaining_types": len(self.remaining_types),
                "defects_per_second_before": round(self.total_before / max(1, self.audio_duration_s), 1),
                "defects_per_second_after": round(self.total_after / max(1, self.audio_duration_s), 1),
            },
            "per_type": [
                {
                    "type": d.defect_type,
                    "before": d.count_before,
                    "after": d.count_after,
                    "reduction_pct": round(d.reduction_pct, 0),
                    "severity_before": round(d.severity_before, 2),
                    "severity_after": round(d.severity_after, 2),
                    "fully_fixed": d.is_fully_fixed,
                }
                for d in sorted(self.defects, key=lambda d: -d.count_before)
            ],
            "heatmap_positions_before": [
                {"time_s": p.get("start_s", 0), "type": p.get("type", "unknown")}
                for p in self.defect_positions_before[:200]
            ],
            "heatmap_positions_after": [
                {"time_s": p.get("start_s", 0), "type": p.get("type", "unknown")}
                for p in self.defect_positions_after[:200]
            ],
        }
