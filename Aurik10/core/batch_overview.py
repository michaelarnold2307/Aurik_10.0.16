"""Batch-Übersicht — Tabellarische Batch-Ergebnisse + Statistik. Spec v10.206 §9.

Bietet:
- Batch-Ergebnis-Tabelle (Qualität, Dauer, Modus pro Song)
- Batch-Statistik (Durchschnitt, Bester, Schlechtester)
- Batch-Filter (nur fehlgeschlagene, nur verbesserte)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchTrackInfo:
    """Einzelner Track im Batch."""

    file_path: str
    mode: str = "restoration"
    quality_before: float = 0.0
    quality_after: float = 0.0
    duration_s: float = 0.0
    processing_time_s: float = 0.0
    success: bool = True
    error: str = ""
    material: str = "unknown"
    rt_factor: float = 0.0

    @property
    def improvement(self) -> float:
        return self.quality_after - self.quality_before

    @property
    def improvement_pct(self) -> float:
        if self.quality_before > 0:
            return (self.improvement / self.quality_before) * 100.0
        return 0.0


@dataclass
class BatchOverview:
    """Batch-Ergebnis-Übersicht."""

    tracks: list[BatchTrackInfo] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def total(self) -> int:
        return len(self.tracks)

    @property
    def successful(self) -> int:
        return sum(1 for t in self.tracks if t.success)

    @property
    def failed(self) -> int:
        return self.total - self.successful

    @property
    def avg_quality_before(self) -> float:
        success = [t for t in self.tracks if t.success]
        if not success:
            return 0.0
        return sum(t.quality_before for t in success) / len(success)

    @property
    def avg_quality_after(self) -> float:
        success = [t for t in self.tracks if t.success]
        if not success:
            return 0.0
        return sum(t.quality_after for t in success) / len(success)

    @property
    def avg_improvement(self) -> float:
        success = [t for t in self.tracks if t.success]
        if not success:
            return 0.0
        return sum(t.improvement for t in success) / len(success)

    @property
    def best_track(self) -> BatchTrackInfo | None:
        success = [t for t in self.tracks if t.success]
        if not success:
            return None
        return max(success, key=lambda t: t.improvement)

    @property
    def worst_track(self) -> BatchTrackInfo | None:
        success = [t for t in self.tracks if t.success]
        if not success:
            return None
        return min(success, key=lambda t: t.improvement)

    @property
    def total_time_s(self) -> float:
        if self.finished_at > 0 and self.started_at > 0:
            return self.finished_at - self.started_at
        return 0.0

    def filter_successful(self) -> list[BatchTrackInfo]:
        return [t for t in self.tracks if t.success]

    def filter_failed(self) -> list[BatchTrackInfo]:
        return [t for t in self.tracks if not t.success]

    def filter_improved(self, min_pct: float = 1.0) -> list[BatchTrackInfo]:
        return [t for t in self.tracks if t.success and t.improvement_pct >= min_pct]

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total": self.total,
                "successful": self.successful,
                "failed": self.failed,
                "avg_quality_before": round(self.avg_quality_before, 1),
                "avg_quality_after": round(self.avg_quality_after, 1),
                "avg_improvement": f"{self.avg_improvement:+.1f}",
                "total_time": f"{self.total_time_s:.0f}s",
            },
            "best": self.best_track.file_path if self.best_track else None,
            "worst": self.worst_track.file_path if self.worst_track else None,
            "tracks": [
                {
                    "file": t.file_path,
                    "mode": t.mode,
                    "quality_before": round(t.quality_before, 1),
                    "quality_after": round(t.quality_after, 1),
                    "improvement": f"{t.improvement:+.1f}",
                    "duration": f"{t.duration_s:.0f}s",
                    "rt": f"{t.rt_factor:.1f}x",
                    "success": t.success,
                    "error": t.error[:100] if t.error else "",
                }
                for t in self.tracks
            ],
        }
