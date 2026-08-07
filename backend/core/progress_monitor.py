"""Progress Monitor — Echtzeit-Fortschritts-Callback für die Pipeline.

§Echtzeit-Monitoring: Callback-basiertes Fortschritts-System, das während
der Pipeline-Verarbeitung Events emittiert — für GUI-Fortschrittsbalken,
Phase-Status-Updates und Echtzeit-Feedback an den Nutzer.

Architektur:
    - ``ProgressMonitor``: Singleton, sammelt Callbacks
    - ``PhaseProgress``: Dataclass für Phasen-Status
    - ``PipelineProgress``: Dataclass für Pipeline-Gesamtstatus
    - Callbacks: sync (Logger) und async (GUI via Queue/Thread)

Usage::

    from backend.core.progress_monitor import get_progress_monitor

    monitor = get_progress_monitor()
    monitor.on_phase_start("phase_03_denoise", total_phases=68)
    # ... Phase läuft ...
    monitor.on_phase_end("phase_03_denoise", quality_estimate=0.95)
    monitor.on_pipeline_complete()

GUI-Integration::

    def gui_callback(event: dict) -> None:
        # Sende Event via WebSocket/SSE an GUI
        websocket.send(json.dumps(event))

    monitor = get_progress_monitor()
    monitor.subscribe(gui_callback)

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhaseProgress:
    """Status einer einzelnen Phase während der Verarbeitung."""

    phase_name: str
    phase_id: str
    status: str  # "pending" | "running" | "completed" | "failed" | "skipped"
    started_at: float | None = None
    completed_at: float | None = None
    duration_s: float | None = None
    quality_estimate: float = 1.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class PipelineProgress:
    """Gesamtstatus der Pipeline."""

    total_phases: int
    completed_phases: int = 0
    failed_phases: int = 0
    skipped_phases: int = 0
    current_phase: str = ""
    started_at: float = 0.0
    elapsed_s: float = 0.0
    estimated_remaining_s: float = 0.0
    progress_pct: float = 0.0
    phases: dict[str, PhaseProgress] = field(default_factory=dict)


# ── Callback-Typ ────────────────────────────────────────────────────────────
ProgressCallback = Callable[[dict[str, Any]], None]


class ProgressMonitor:
    """Singleton: Echtzeit-Fortschritts-Monitor für die Aurik-Pipeline.

    Sammelt Fortschrittsereignisse und verteilt sie an registrierte
    Callbacks (Logger, GUI-WebSocket, SSE-Stream).

    Events:
        - ``phase_start``: Phase beginnt
        - ``phase_progress``: Innerhalb einer Phase (optional, granular)
        - ``phase_end``: Phase abgeschlossen
        - ``pipeline_complete``: Alle Phasen durchlaufen
        - ``pipeline_error``: Pipeline abgebrochen
    """

    _instance: ProgressMonitor | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._callbacks: list[ProgressCallback] = []
        self._pipeline: PipelineProgress | None = None
        self._pipeline_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ProgressMonitor:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Subscription ────────────────────────────────────────────────────────

    def subscribe(self, callback: ProgressCallback) -> None:
        """Callback registrieren (sync oder async)."""
        self._callbacks.append(callback)

    def unsubscribe(self, callback: ProgressCallback) -> None:
        """Callback entfernen."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # ── Pipeline-Lifecycle ───────────────────────────────────────────────────

    def on_pipeline_start(
        self,
        total_phases: int,
        audio_duration_s: float = 0.0,
        material: str = "unknown",
        metadata: dict | None = None,
    ) -> None:
        """Pipeline-Start signalisieren."""
        with self._pipeline_lock:
            self._pipeline = PipelineProgress(
                total_phases=total_phases,
                started_at=time.monotonic(),
            )

        self._emit(
            {
                "event": "pipeline_start",
                "total_phases": total_phases,
                "audio_duration_s": audio_duration_s,
                "material": material,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
        )

    def on_phase_start(self, phase_name: str, phase_id: str = "") -> None:
        """Phase-Beginn signalisieren."""
        pid = phase_id or phase_name
        phase = PhaseProgress(
            phase_name=phase_name,
            phase_id=pid,
            status="running",
            started_at=time.monotonic(),
        )

        with self._pipeline_lock:
            if self._pipeline:
                self._pipeline.current_phase = phase_name
                self._pipeline.phases[pid] = phase

        self._emit(
            {
                "event": "phase_start",
                "phase_name": phase_name,
                "phase_id": pid,
                "timestamp": time.time(),
            }
        )

    def on_phase_progress(
        self,
        phase_name: str,
        progress_pct: float = 0.0,
        detail: str = "",
    ) -> None:
        """Granularen Fortschritt INNERHALB einer Phase melden (optional)."""
        self._emit(
            {
                "event": "phase_progress",
                "phase_name": phase_name,
                "progress_pct": progress_pct,
                "detail": detail,
                "timestamp": time.time(),
            }
        )

    def on_phase_end(
        self,
        phase_name: str,
        phase_id: str = "",
        quality_estimate: float = 1.0,
        warnings: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        """Phase-Ende signalisieren."""
        pid = phase_id or phase_name
        status = "failed" if error else "completed"

        with self._pipeline_lock:
            pipeline = self._pipeline
            if pipeline:
                phase = pipeline.phases.get(pid)
                if phase:
                    phase.status = status
                    phase.completed_at = time.monotonic()
                    phase.duration_s = phase.completed_at - (phase.started_at or phase.completed_at)
                    phase.quality_estimate = quality_estimate
                    phase.warnings = warnings or []
                    phase.error = error

                if status == "completed":
                    pipeline.completed_phases += 1
                elif status == "failed":
                    pipeline.failed_phases += 1

                # Fortschritt berechnen
                done = pipeline.completed_phases + pipeline.failed_phases
                pipeline.progress_pct = (done / max(pipeline.total_phases, 1)) * 100
                pipeline.elapsed_s = time.monotonic() - pipeline.started_at

                if done > 0:
                    avg_phase_s = pipeline.elapsed_s / done
                    remaining = pipeline.total_phases - done
                    pipeline.estimated_remaining_s = avg_phase_s * remaining

        self._emit(
            {
                "event": "phase_end",
                "phase_name": phase_name,
                "phase_id": pid,
                "status": status,
                "quality_estimate": quality_estimate,
                "warnings": warnings or [],
                "error": error,
                "timestamp": time.time(),
            }
        )

    def on_pipeline_complete(self, output_path: str = "") -> None:
        """Pipeline erfolgreich abgeschlossen."""
        with self._pipeline_lock:
            if self._pipeline:
                self._pipeline.elapsed_s = time.monotonic() - self._pipeline.started_at

        self._emit(
            {
                "event": "pipeline_complete",
                "output_path": output_path,
                "progress": self.get_progress(),
                "timestamp": time.time(),
            }
        )

    def on_pipeline_error(self, error: str) -> None:
        """Pipeline mit Fehler abgebrochen."""
        self._emit(
            {
                "event": "pipeline_error",
                "error": error,
                "progress": self.get_progress(),
                "timestamp": time.time(),
            }
        )

    # ── Status-Abfrage ───────────────────────────────────────────────────────

    def get_progress(self) -> dict[str, Any] | None:
        """Aktuellen Pipeline-Fortschritt abrufen (für Polling)."""
        with self._pipeline_lock:
            pipeline = self._pipeline
            if pipeline is None:
                return None

            return {
                "total_phases": pipeline.total_phases,
                "completed_phases": pipeline.completed_phases,
                "failed_phases": pipeline.failed_phases,
                "skipped_phases": pipeline.skipped_phases,
                "current_phase": pipeline.current_phase,
                "progress_pct": round(pipeline.progress_pct, 1),
                "elapsed_s": round(pipeline.elapsed_s, 1),
                "estimated_remaining_s": round(pipeline.estimated_remaining_s, 1),
                "phases": {
                    pid: {
                        "name": p.phase_name,
                        "status": p.status,
                        "duration_s": round(p.duration_s, 3) if p.duration_s else None,
                        "quality_estimate": p.quality_estimate,
                    }
                    for pid, p in pipeline.phases.items()
                },
            }

    def get_current_phase(self) -> str:
        """Name der aktuell laufenden Phase."""
        with self._pipeline_lock:
            if self._pipeline:
                return self._pipeline.current_phase
            return ""

    def reset(self) -> None:
        """Pipeline-Status zurücksetzen (für neuen Durchlauf)."""
        with self._pipeline_lock:
            self._pipeline = None

    # ── Private ──────────────────────────────────────────────────────────────

    def _emit(self, event: dict[str, Any]) -> None:
        """Event an alle registrierten Callbacks verteilen."""
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.warning("Progress-Callback fehlgeschlagen: %s", exc)

        # Default: Loggen
        event_type = event.get("event", "unknown")
        if event_type == "phase_end":
            logger.info(
                "Verarbeitungsschritt %s: %s (quality=%.2f)",
                event.get("phase_name", "?"),
                event.get("status", "?"),
                event.get("quality_estimate", 0.0),
            )
        elif event_type == "pipeline_complete":
            progress = event.get("progress", {})
            logger.info(
                "Pipeline abgeschlossen: %d/%d Phasen in %.1fs",
                progress.get("completed_phases", 0),
                progress.get("total_phases", 0),
                progress.get("elapsed_s", 0.0),
            )


# ── Convenience ─────────────────────────────────────────────────────────────


def get_progress_monitor() -> ProgressMonitor:
    """Singleton-Instanz des ProgressMonitor."""
    return ProgressMonitor.get_instance()


__all__ = [
    "ProgressMonitor",
    "PhaseProgress",
    "PipelineProgress",
    "get_progress_monitor",
]
