"""Pipeline-Prozess — UV3-Restaurierung in eigenem Systemprozess.

Entkoppelt die rechenintensive Pipeline vollständig vom Qt-GUI-Thread.
Audio-Updates laufen via SharedMemory-Ringpuffer, Status via Multiprocessing-Pipe.

Architektur:
  PipelineProcess (multiprocessing.Process)
  ├── Lädt Denker/UV3-Pipeline im Kindprozess
  ├── Empfängt Aufträge via Input-Pipe (pickle serialisiert)
  ├── Schreibt Audio-Frames nach jeder Phase in SharedAudioRing
  └── Sendet Status-Updates via Output-Pipe

  GUI-Thread
  ├── Pollt SharedAudioRing via QTimer (60 fps)
  ├── Liest Status-Updates via Output-Pipe
  └── Bleibt vollständig responsiv — kein GIL-Sharing

Sicherheit:
  - Keine Qt-Objekte im Kindprozess (reiner Python-Code)
  - Alle Audio-Daten via SharedMemory (kein Pickle)
  - Pipe für leichte Statusmeldungen (JSON-serialisiert, <1KB)
"""

from __future__ import annotations

import atexit
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Aurik10.ipc.shared_audio import AudioFrame, SharedAudioRing

logger = logging.getLogger(__name__)

logger = logging.getLogger("aurik.ipc.pipeline")


# ═══════════════════════════════════════════════════════════════════════
# Status-Datentypen (leichtgewichtig, JSON-serialisierbar)
# ═══════════════════════════════════════════════════════════════════════


class PipelineState(Enum):
    IDLE = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class PipelineStatus:
    """Statusmeldung vom Kindprozess an die GUI (via Pipe)."""

    state: str = "idle"  # PipelineState.name
    progress_pct: float = 0.0
    current_phase: str = ""
    narrative: str = ""  # §v10.118: laienverständlicher Erzähltext vom PhaseProgressNarrator
    phase_index: int = 0
    total_phases: int = 0
    mos_estimate: float = 0.0
    error: str = ""
    timestamp: float = 0.0

    # ── §v10.14 P1: Live-Metriken während des Laufs ──
    live_mushra: float = 0.0  # MUSHRA-Score (0-100), per-phase update
    live_hpi: float = 0.0  # HPI-Score (0-1)
    live_vqi: float = 0.0  # Vocal Quality Index (0-1)
    live_guardian_reverted: bool = False  # Guardian hat verworfen
    live_guardian_reason: str = ""  # Guardian-Begründung

    # ── §v10.201 Ergebnis-Felder (nur bei state∈{completed,warning,failed}) ──
    result_quality: float = 0.0  # quality_estimate * 100
    result_reverted: bool = False  # do_no_harm.reverted
    result_revert_reason: str = ""  # Guardian-Begründung
    result_mushra: float = 0.0  # MUSHRA-Score (0-100)
    result_hpi: float = 0.0  # HPI-Score (0-1)
    result_phases_done: int = 0  # Tatsächliche Phasenzahl
    result_warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = {
            "state": self.state,
            "progress_pct": self.progress_pct,
            "current_phase": self.current_phase,
            "narrative": self.narrative,  # §v10.118: Erzähltext für GUI
            "phase_index": self.phase_index,
            "total_phases": self.total_phases,
            "mos_estimate": self.mos_estimate,
            "error": self.error,
            "timestamp": time.monotonic(),
            # §v10.14 P1: Live-Metriken
            "live_mushra": self.live_mushra,
            "live_hpi": self.live_hpi,
            "live_vqi": self.live_vqi,
            "live_guardian_reverted": self.live_guardian_reverted,
            "live_guardian_reason": self.live_guardian_reason,
            # §v10.201 Ergebnis-Felder
            "result_quality": self.result_quality,
            "result_reverted": self.result_reverted,
            "result_revert_reason": self.result_revert_reason,
            "result_mushra": self.result_mushra,
            "result_hpi": self.result_hpi,
            "result_phases_done": self.result_phases_done,
            "result_warnings": self.result_warnings,
        }
        return json.dumps(d, ensure_ascii=False)


@dataclass
class PipelineJob:
    """Auftrag vom GUI-Thread an den Kindprozess (via Pipe)."""

    input_file: str
    output_file: str
    mode: str = "restoration"
    material: str = "vinyl"
    variant: str = "balanced"
    quality_target: float = 80.0
    settings: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Worker-Funktion (läuft im Kindprozess)
# ═══════════════════════════════════════════════════════════════════════


def _pipeline_worker(
    input_pipe: mp.connection.Connection,
    output_pipe: mp.connection.Connection,
    ring_name: str,
    ready_event: mp.synchronize.Event,
) -> None:
    """Hauptfunktion des Kindprozesses. Rekonstruiert Denker/UV3 und
    verarbeitet Aufträge aus der Input-Pipe.

    Args:
        input_pipe: Empfängt PipelineJob-Dicts
        output_pipe: Sendet PipelineStatus-JSON-Strings
        ring_name: Name des SharedMemory-Segments für SharedAudioRing
        ready_event: Wird gesetzt, wenn Initialisierung abgeschlossen
    """
    # ── Signalhandler für sauberes Herunterfahren ──────────────────
    _shutdown_requested = False

    def _on_sigterm(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
        logger.info("Pipeline-Worker: SIGTERM empfangen, beende...")

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, lambda *_: None)  # ignoriere Ctrl-C im Kind

    logger.info("Pipeline-Worker gestartet (PID=%d)", mp.current_process().pid)

    try:
        # ── SharedAudioRing öffnen ────────────────────────────────
        ring = SharedAudioRing(name=ring_name, create=False)

        # ── Denker/UV3 importieren und initialisieren ──────────────
        # Der Kindprozess muss die gesamte Pipeline eigenständig laden
        # (ONNX-Sessions, PMGG-Modelle, etc.)
        denker = None
        try:
            # Bridge-Import für Singletons im Kindprozess
            from Aurik10.bridge import (
                get_aurik_denker_instance as _bridge_denker,
            )

            denker = _bridge_denker()
            if denker is None:
                raise RuntimeError("Denker-Singleton nicht verfügbar")

            # ROCm warmup (falls im Kindprozess nötig)
            from Aurik10.bridge import warmup_rocm as _bridge_warmup

            if _bridge_warmup is not None:
                _bridge_warmup()

            logger.info("Pipeline-Worker: Denker initialisiert")
        except Exception as e:
            logger.error("Pipeline-Worker: Denker-Init fehlgeschlagen: %s", e)
            output_pipe.send(
                PipelineStatus(
                    state="failed",
                    error=f"Backend-Init fehlgeschlagen: {e}",
                ).to_json()
            )
            return

        # ── Bereitschaft signalisieren ─────────────────────────────
        ready_event.set()
        logger.info("Pipeline-Worker: bereit")

        # ── Hauptschleife ──────────────────────────────────────────
        while not _shutdown_requested:
            # Poll input pipe (non-blocking mit Timeout)
            if not input_pipe.poll(0.1):
                continue

            try:
                job_dict = input_pipe.recv()
            except EOFError:
                logger.info("Pipeline-Worker: Eingabe-Pipe geschlossen")
                break

            if job_dict is None or job_dict.get("_command") == "shutdown":
                logger.info("Pipeline-Worker: Herunterfahren-Kommando empfangen")
                break

            if job_dict.get("_command") == "ping":
                output_pipe.send(PipelineStatus(state="idle").to_json())
                continue

            # ── Auftrag verarbeiten ────────────────────────────────
            job = PipelineJob(**{k: v for k, v in job_dict.items() if not k.startswith("_")})
            logger.info("Pipeline-Worker: Verarbeite %s", job.input_file)

            try:
                _run_restoration_job(denker, ring, job, output_pipe)
            except Exception as e:
                logger.error("Pipeline-Worker: Job fehlgeschlagen: %s", e)
                traceback.print_exc()
                output_pipe.send(
                    PipelineStatus(
                        state="failed",
                        error=str(e),
                    ).to_json()
                )

    except Exception as e:
        logger.error("Pipeline-Worker: Kritischer Fehler: %s", e)
        traceback.print_exc()
        try:
            output_pipe.send(
                PipelineStatus(
                    state="failed",
                    error=f"Worker-Absturz: {e}",
                ).to_json()
            )
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            pass
    finally:
        logger.info("Pipeline-Worker beendet (PID=%d)", mp.current_process().pid)


def _run_restoration_job(
    denker,  # AurikDenker instance
    ring: SharedAudioRing,
    job: PipelineJob,
    output_pipe: mp.connection.Connection,
) -> None:
    """Führt einen einzelnen Restaurierungsauftrag aus.

    Schreibt Audio-Updates nach jeder Phase in den SharedAudioRing
    und Statusmeldungen in die Output-Pipe.

    Args:
        denker: Initialisierte AurikDenker-Instanz
        ring: SharedAudioRing für Live-Audio
        job: Der auszuführende Auftrag
        output_pipe: Pipe für Statusmeldungen
    """

    # ── Audio laden ────────────────────────────────────────────────
    output_pipe.send(
        PipelineStatus(
            state="running",
            progress_pct=0.0,
            current_phase="audio_loading",
            phase_index=0,
            total_phases=0,  # §v10.202: wird vom ersten echten Progress-Callback gesetzt
        ).to_json()
    )

    audio, sr = _load_audio_file(job.input_file)
    if audio is None or audio.size == 0:
        output_pipe.send(
            PipelineStatus(
                state="failed",
                error="Audio konnte nicht geladen werden",
            ).to_json()
        )
        return

    # ── Audio-Callback für denker.denke() ──────────────────────────
    # Nach jeder UV3-Phase wird dieser Callback mit dem aktuellen
    # Audio-Zustand aufgerufen.
    def _phase_callback(phase_audio: np.ndarray, phase_sr: int, phase_id: str) -> None:
        """Wird nach jeder UV3-Phase aufgerufen."""
        try:
            # In SharedAudioRing schreiben (lock-free, kopiert Daten)
            ring.write(phase_audio, phase_sr, phase_id)
        except Exception as e:
            logger.debug("SharedAudioRing write fehlgeschlagen: %s", e)

    def _progress_callback(
        pct: float, phase_name: str, phase_idx: int = 0, total: int = 0, metrics: dict | None = None
    ) -> None:
        """Wird bei Fortschrittsänderungen aufgerufen.

        §v10.14 P1: Akzeptiert optionalen metrics-Dict mit Live-Qualitätsdaten
        (mushra, hpi, vqi, guardian_reverted).
        """
        _narrative = str(phase_name) if phase_name else ""
        _phase_short = _narrative[:60] + "…" if len(_narrative) > 60 else _narrative
        _m = metrics or {}
        output_pipe.send(
            PipelineStatus(
                state="running",
                progress_pct=pct,
                current_phase=_phase_short,
                narrative=_narrative,
                phase_index=int(phase_idx) if isinstance(phase_idx, (int, float)) else 0,
                total_phases=int(total) if isinstance(total, (int, float)) else 0,
                # §v10.14 P1: Live-Metriken aus optionalem metrics-Dict
                live_mushra=float(_m.get("mushra", 0.0)),
                live_hpi=float(_m.get("hpi", 0.0)),
                live_vqi=float(_m.get("vqi", 0.0)),
                live_guardian_reverted=bool(_m.get("guardian_reverted", False)),
                live_guardian_reason=str(_m.get("guardian_reason", "")),
            ).to_json()
        )

    # ── Denker ausführen ───────────────────────────────────────────
    output_pipe.send(
        PipelineStatus(
            state="running",
            progress_pct=0.0,
            current_phase="restoration_start",
            phase_index=0,
            total_phases=0,  # §v10.202: dynamisch — erster Progress-Callback setzt korrekte Zahl
        ).to_json()
    )

    try:
        # Defect-Scan-Ergebnis vorbereiten (falls vorhanden)
        cached_defect = job.settings.get("cached_defect_result")

        result = denker.denke(
            audio=audio,
            sr=sr,
            mode=job.mode,
            material=job.material,
            variant=job.variant,
            quality_target=job.quality_target,
            input_path=job.input_file,
            output_path=job.output_file,
            progress_callback=_progress_callback,
            audio_update_callback=_phase_callback,
            cached_defect_result=cached_defect,
        )

        # ── Letztes Audio-Frame pushen ─────────────────────────────
        if result is not None and hasattr(result, "audio"):
            ring.write(
                result.audio,
                getattr(result, "sr", 48000),
                "final",
            )

        # §v10.201: Ergebnis-Metadaten aus RestorationResult extrahieren
        _rmeta = getattr(result, "metadata", {}) or {}
        _dnh = _rmeta.get("do_no_harm", {}) or {}
        _uqm = _rmeta.get("uqm", {}) or {}
        _mushra = (_rmeta.get("mushra") or {}).get("mushra_score", 0.0)
        _q = float(getattr(result, "quality_estimate", 0.0) or 0.0)
        _reverted = bool(_dnh.get("reverted", False))

        output_pipe.send(
            PipelineStatus(
                state="completed" if not _reverted else "warning",
                progress_pct=100.0,
                result_quality=round(_q * 100, 1),
                result_reverted=_reverted,
                result_revert_reason=str(_dnh.get("reason", "")),
                result_mushra=float(_mushra),
                result_hpi=float(_rmeta.get("hpi_score", 0.0)),
                result_phases_done=int(_rmeta.get("phases_total", 0)),
                result_warnings=list(getattr(result, "warnings", []) or []),
                error=str(_dnh.get("reason", "")) if _reverted else "",
            ).to_json()
        )

    except Exception as e:
        logger.error("Denker.denke() fehlgeschlagen: %s", e)
        raise


def _load_audio_file(path: str) -> tuple[np.ndarray | None, int]:
    """Lädt eine Audiodatei im Kindprozess.

    Returns:
        Tuple[audio, sample_rate] oder (None, 0) bei Fehler.
    """
    p = Path(path)
    if not p.exists():
        logger.error("Audiodatei nicht gefunden: %s", path)
        return None, 0

    _max_load_retries = 3
    for _load_attempt in range(_max_load_retries):
        try:
            import soundfile as sf

            audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
            # Normalisieren
            if audio.ndim == 1 or audio.ndim == 2:
                audio = audio.astype(np.float32)
            # §V08: np.percentile(99.9) statt np.max(abs()) — robust gegen
            # einzelne Ausreißer-Samples (Clicks/Pops).
            _peak = float(np.percentile(np.abs(audio), 99.9))
            if _peak > 0:
                audio = audio / _peak * 0.95
            return audio, int(sr)
        except Exception as e:
            if _load_attempt < _max_load_retries - 1:
                logger.debug("soundfile.read Wiederholung %d/%d: %s", _load_attempt + 1, _max_load_retries, e)
                continue
            logger.error("soundfile.read fehlgeschlagen: %s", e)
            return None, 0
    return None, 0


# ═══════════════════════════════════════════════════════════════════════
# PipelineProcess — GUI-seitige Steuerklasse
# ═══════════════════════════════════════════════════════════════════════


class PipelineProcess:
    """Steuert den Pipeline-Kindprozess vom GUI-Thread aus.

    Verwendung:
        pipeline = PipelineProcess()
        pipeline.start()
        pipeline.submit(job)
        while pipeline.poll():
            status = pipeline.latest_status
            # Update GUI...
        pipeline.stop()

    Der GUI-Thread sollte einen QTimer (50-100ms) nutzen, um
    pipeline.poll() regelmäßig aufzurufen und Status-Aktualisierungen
    zu empfangen.

    Audio-Daten werden NICHT über diese Klasse gelesen — dafür
    SharedAudioRing direkt mit ring.try_pop() pollen.
    """

    def __init__(self):
        self._process: mp.Process | None = None
        self._parent_pipe: mp.connection.Connection | None = None
        self._child_pipe: mp.connection.Connection | None = None
        self._ready_event = mp.Event()
        self._ring: SharedAudioRing | None = None
        self._ring_name: str = ""
        self.latest_status = PipelineStatus()
        self._status_queue: list[PipelineStatus] = []
        self._shutting_down = False

    @property
    def ring(self) -> SharedAudioRing | None:
        """SharedAudioRing für Live-Audio-Preview."""
        return self._ring

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def start(self, ring_frame_count: int = 10, ring_max_duration_s: float = 4.0) -> bool:
        """Startet den Pipeline-Kindprozess und initialisiert SharedAudioRing.

        Args:
            ring_frame_count: Anzahl der Audio-Frames im Ringpuffer
            ring_max_duration_s: Maximale Dauer pro Frame in Sekunden

        Returns:
            True bei Erfolg, False bei Fehler.
        """
        if self._process is not None:
            logger.warning("PipelineProcess läuft bereits")
            return False

        try:
            # ── SharedAudioRing erstellen ──────────────────────────
            self._ring = SharedAudioRing(
                name=f"aurik_pipeline_{os.getpid()}_{int(time.monotonic() * 1000)}",
                create=True,
            )
            self._ring_name = self._ring._name

            # ── Pipes erstellen ─────────────────────────────────────
            self._parent_pipe, self._child_pipe = mp.Pipe(duplex=False)

            # ── Kindprozess starten ─────────────────────────────────
            self._ready_event.clear()
            self._process = mp.Process(
                target=_pipeline_worker,
                args=(
                    self._child_pipe,
                    self._parent_pipe,
                    self._ring_name,
                    self._ready_event,
                ),
                name="AurikPipeline",
                daemon=False,
            )
            self._process.start()

            # ── Auf Bereitschaft warten (mit Timeout) ──────────────
            logger.info("Warte auf Pipeline-Worker-Bereitschaft...")
            if not self._ready_event.wait(timeout=30.0):
                logger.error("Pipeline-Worker-Zeitlimit nach 30s")
                self.stop()
                return False

            logger.info("PipelineProcess gestartet (PID=%d)", self._process.pid)

            # Cleanup bei Prozess-Ende registrieren
            atexit.register(self._cleanup)

            return True

        except Exception as e:
            logger.error("PipelineProcess.start() fehlgeschlagen: %s", e)
            self.stop()
            return False

    def submit(self, job: PipelineJob) -> bool:
        """Sendet einen Auftrag an den Kindprozess.

        Args:
            job: Der auszuführende PipelineJob

        Returns:
            True wenn gesendet, False bei Fehler.
        """
        if not self.is_running:
            logger.error("PipelineProcess: Kindprozess läuft nicht")
            return False

        try:
            self._child_pipe.send(  # type: ignore[union-attr]
                {
                    "input_file": job.input_file,
                    "output_file": job.output_file,
                    "mode": job.mode,
                    "material": job.material,
                    "variant": job.variant,
                    "quality_target": job.quality_target,
                    "settings": job.settings,
                }
            )
            return True
        except Exception as e:
            logger.error("PipelineProcess.submit() fehlgeschlagen: %s", e)
            return False

    def poll(self) -> bool:
        """Pollt die Output-Pipe auf neue Statusmeldungen.

        Sollte vom GUI-Thread via QTimer alle 50-100ms aufgerufen werden.

        Returns:
            True wenn neue Statusmeldungen verfügbar sind.
        """
        if self._parent_pipe is None:
            return False

        had_updates = False
        try:
            while self._parent_pipe.poll():
                raw = self._parent_pipe.recv()
                try:
                    data = json.loads(raw)
                    status = PipelineStatus(
                        state=data.get("state", "idle"),
                        progress_pct=data.get("progress_pct", 0.0),
                        current_phase=data.get("current_phase", ""),
                        narrative=data.get(
                            "narrative", data.get("current_phase", "")
                        ),  # §v10.118: Erzähltext, Fallback auf phase
                        phase_index=data.get("phase_index", 0),
                        total_phases=data.get("total_phases", 0),
                        mos_estimate=data.get("mos_estimate", 0.0),
                        error=data.get("error", ""),
                        timestamp=data.get("timestamp", 0.0),
                        # §v10.201 Ergebnis-Felder — ohne diese sieht die GUI nach
                        # Abschluss nie Qualität/MUSHRA/HPI/Revert-Grund/Warnungen.
                        result_quality=data.get("result_quality", 0.0),
                        result_reverted=data.get("result_reverted", False),
                        result_revert_reason=data.get("result_revert_reason", ""),
                        result_mushra=data.get("result_mushra", 0.0),
                        result_hpi=data.get("result_hpi", 0.0),
                        result_phases_done=data.get("result_phases_done", 0),
                        result_warnings=data.get("result_warnings", []),
                    )
                    self.latest_status = status
                    self._status_queue.append(status)
                    had_updates = True
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("Ungültige Statusmeldung: %s", e)
        except (EOFError, BrokenPipeError):
            logger.info("PipelineProcess: Ausgabe-Pipe geschlossen")
        except Exception as e:
            logger.debug("PipelineProcess.poll() Fehler: %s", e)

        return had_updates

    def drain_status(self) -> list[PipelineStatus]:
        """Gibt alle kumulierten Statusmeldungen zurück und leert die Queue."""
        result = self._status_queue[:]
        self._status_queue.clear()
        return result

    def stop(self, timeout: float = 5.0) -> None:
        """Stoppt den Kindprozess und räumt Ressourcen auf.

        Args:
            timeout: Maximale Wartezeit in Sekunden für sauberes Beenden.
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        try:
            # ── Shutdown-Kommando senden ────────────────────────────
            if self._child_pipe is not None:
                try:
                    self._child_pipe.send({"_command": "shutdown"})
                except (BrokenPipeError, OSError):
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

            # ── Auf Prozess-Ende warten ─────────────────────────────
            if self._process is not None:
                self._process.join(timeout=timeout)
                if self._process.is_alive():
                    logger.warning("Pipeline-Worker reagiert nicht, erzwinge Terminierung")
                    self._process.terminate()
                    self._process.join(timeout=2.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=1.0)

        except Exception as e:
            logger.error("PipelineProcess.stop() Fehler: %s", e)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Räumt Prozess, Pipes und SharedMemory auf."""
        # Pipes schließen
        for pipe in [self._parent_pipe, self._child_pipe]:
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        self._parent_pipe = None
        self._child_pipe = None
        self._process = None

        # SharedAudioRing aufräumen
        if self._ring is not None:
            try:
                self._ring.close()
                self._ring.unlink()
            except Exception as e:
                logger.debug("SharedAudioRing cleanup: %s", e)
            self._ring = None

        self._ready_event.clear()
        self._shutting_down = False

    def is_alive(self) -> bool:
        """Prüft ob der Kindprozess noch läuft."""
        return self._process is not None and self._process.is_alive()

    def request_cancel(self) -> None:
        """Fordert Abbruch des aktuellen Auftrags an (nicht des Prozesses)."""
        if self._child_pipe is not None:
            try:
                self._child_pipe.send({"_command": "cancel"})
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
