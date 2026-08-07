#!/usr/bin/env python3
"""
Aurik 10.0.0 — Integriertes Qualitäts-Monitoring & Deep-Analysis Ökosystem
=======================================================================

Orchestriert:
1. PyQt5 Frontend (GUI für Restoration)
2. Kontinuierliche Tiefenanalyse (Phase-weise Quality Checkpoints)
3. Echtzeit-Pegelexplosion-Monitor (kontinuierliche Export-Überwachung)
4. Automatische Code-Fixes bei kritischen Problemen

Bietet ein einheitliches Dashboard für kompletten Restaurierungs-Prozess.

Usage:
    python scripts/orchestrate_quality_monitoring.py [--audio <path>] [--no-gui] [--headless]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

_WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Konfiguration für Orchestrator."""

    audio_path: str | None = None
    start_gui: bool = True
    start_analyzer: bool = True
    start_pegelexplosion_monitor: bool = True
    analysis_output_dir: str = "analysis_results"
    headless: bool = False
    verbose: bool = False


class QualityMonitoringOrchestrator:
    """Orchestriert alle Quality-Monitoring-Komponenten."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.processes: dict[str, subprocess.Popen] = {}
        self.shutdown_event = Event()
        self._status_file = _WORKSPACE_ROOT / "temp_repro" / "monitoring_status.json"
        self._status_file.parent.mkdir(parents=True, exist_ok=True)

    def start_all(self) -> int:  # type: ignore[return]
        """Startet alle Komponenten."""
        logger.info("=" * 100)
        logger.info("AURIK 9 — QUALITÄTS-MONITORING ÖKOSYSTEM")
        logger.info("=" * 100)
        logger.info(f"Audio: {self.config.audio_path}")
        logger.info(f"GUI: {'✓' if self.config.start_gui else '✗'}")
        logger.info(f"Tiefenanalyse: {'✓' if self.config.start_analyzer else '✗'}")
        logger.info(f"Pegelexplosion-Monitor: {'✓' if self.config.start_pegelexplosion_monitor else '✗'}")
        logger.info("")

        try:
            # 1. Pegelexplosion-Monitor starten (Hintergrund)
            if self.config.start_pegelexplosion_monitor:
                self._start_pegelexplosion_monitor()
                time.sleep(1)

            # 2. GUI starten
            if self.config.start_gui:
                self._start_gui()
                time.sleep(2)

            # 3. Tiefenanalyse starten
            if self.config.start_analyzer:
                self._start_analyzer()
                time.sleep(1)

            # 4. Monitoring-Loop
            self._run_monitoring_loop()

        except KeyboardInterrupt:
            logger.info("\nBenutzer unterbrochen...")
        finally:
            self.shutdown_all()

    def _start_gui(self) -> None:
        """Startet PyQt5-Frontend."""
        logger.info("Starte PyQt5-Frontend...")
        gui_script = _WORKSPACE_ROOT / "run_aurik.sh"
        if not gui_script.exists():
            logger.warning(f"Ausfuehrung_aurik.sh nicht gefunden: {gui_script}")
            return

        try:
            proc = subprocess.Popen(
                ["bash", str(gui_script)],
                cwd=str(_WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.processes["gui"] = proc
            logger.info(f"✓ GUI gestartet (PID: {proc.pid})")
        except Exception as e:
            logger.error(f"✗ GUI-Start fehlgeschlagen: {e}")

    def _start_analyzer(self) -> None:
        """Startet kontinuierliche Tiefenanalyse."""
        if not self.config.audio_path:
            logger.warning("Keine Audio-Datei für Analyzer verfügbar")
            return

        logger.info("Starte Tiefenanalyse...")
        analyzer_script = _WORKSPACE_ROOT / "scripts" / "continuous_deep_analysis.py"

        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(analyzer_script),
                    "--audio",
                    self.config.audio_path,
                    "--mode",
                    "restoration",
                    "--output-dir",
                    self.config.analysis_output_dir,
                ],
                cwd=str(_WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.processes["analyzer"] = proc
            logger.info(f"✓ Analyzer gestartet (PID: {proc.pid})")
        except Exception as e:
            logger.error(f"✗ Analyzer-Start fehlgeschlagen: {e}")

    def _start_pegelexplosion_monitor(self) -> None:
        """Startet Pegelexplosion-Monitor."""
        logger.info("Starte Pegelexplosion-Monitor...")
        monitor_script = _WORKSPACE_ROOT / "scripts" / "pegelexplosion_monitor.py"

        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(monitor_script),
                    "--watch-dir",
                    "output_audio",
                    "--interval",
                    "2",
                ],
                cwd=str(_WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.processes["pegelexplosion_monitor"] = proc
            logger.info(f"✓ Pegelexplosion-Monitor gestartet (PID: {proc.pid})")
        except Exception as e:
            logger.error(f"✗ Monitor-Start fehlgeschlagen: {e}")

    def _run_monitoring_loop(self) -> None:
        """Laufen lässt die Überwachungs-Schleife."""
        logger.info("\n" + "=" * 100)
        logger.info("MONITORING AKTIV")
        logger.info("=" * 100)

        # Output-Streaming Threads
        threads = []
        for name, proc in self.processes.items():
            if proc and proc.stdout:
                thread = Thread(target=self._stream_process_output, args=(name, proc), daemon=True)
                thread.start()
                threads.append(thread)

        # Prozess-Status-Loop
        while not self.shutdown_event.is_set():
            time.sleep(2)

            # Status prüfen
            for name, proc in list(self.processes.items()):
                if proc and proc.poll() is not None:
                    ret = proc.returncode
                    if name == "gui" and ret != 0:
                        logger.warning(f"GUI beendet mit Code {ret}")
                        self.shutdown_event.set()
                        break
                    elif name == "analyzer" and ret == 0:
                        logger.info("✓ Analyzer erfolgreich abgeschlossen")
                        self.shutdown_event.set()
                        break
                    elif name == "analyzer" and ret != 0:
                        logger.error(f"✗ Analyzer fehlgeschlagen mit Code {ret}")
                        self.shutdown_event.set()
                        break
                    elif ret != 0:
                        logger.warning(f"{name} beendet mit Code {ret}")

            # Status-Datei aktualisieren
            self._update_status_file()

    def _stream_process_output(self, name: str, proc: subprocess.Popen) -> None:
        """Streamt Process-Output."""
        try:
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    if line:
                        logger.info(f"[{name.upper()}] {line.rstrip()}")
        except Exception as e:
            logger.debug(f"{name} Ausgabe stream beendet: {e}")

    def _update_status_file(self) -> None:
        """Aktualisiert Status-Datei."""
        status = {
            "timestamp": time.time(),
            "processes": {
                name: {"pid": proc.pid, "running": proc.poll() is None} for name, proc in self.processes.items() if proc
            },
        }
        try:
            with open(self._status_file, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.debug(f"Status-Datei-Aktualisierung fehlgeschlagen: {e}")

    def shutdown_all(self) -> int:
        """Fährt alle Komponenten herunter."""
        logger.info("\n" + "=" * 100)
        logger.info("FAHRE ALLE KOMPONENTEN HERUNTER")
        logger.info("=" * 100)

        exit_code = 0
        for name, proc in self.processes.items():
            if proc:
                try:
                    logger.info(f"Beende {name}...")
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"  Zeitlimit bei {name}, force-kill...")
                    proc.kill()
                except Exception as e:
                    logger.error(f"  Fehler bei {name}: {e}")
                    exit_code = 1

        logger.info("=" * 100)
        logger.info("Monitoring beendet")
        logger.info("=" * 100)

        # Final status
        self._update_status_file()

        return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Aurik 10.0.0 — Integriertes Qualitäts-Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Vollständiges Monitoring: GUI + Tiefenanalyse + Pegelexplosion-Monitor
  python scripts/orchestrate_quality_monitoring.py --audio my_song.mp3

  # Nur Tiefenanalyse + Monitor (kein GUI)
  python scripts/orchestrate_quality_monitoring.py --audio my_song.mp3 --no-gui

  # Headless-Modus (nur CLI-Output)
  python scripts/orchestrate_quality_monitoring.py --audio my_song.mp3 --headless
        """,
    )
    parser.add_argument("--audio", type=str, default=None, help="Audio-Datei-Pfad")
    parser.add_argument("--no-gui", action="store_true", help="GUI nicht starten")
    parser.add_argument("--headless", action="store_true", help="Nur CLI (impliziert --no-gui)")
    parser.add_argument("--verbose", action="store_true", help="Verbose Logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("orchestrator_runtime.log"),
        ],
    )

    # Standard-Audio
    audio_path = args.audio
    if not audio_path:
        candidates = [
            "test_audio/Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
            "test_audio/tape/*.mp3",
            "test_audio/vinyl/*.mp3",
        ]
        for cand in candidates:
            if "*" not in cand:
                if Path(cand).exists():
                    audio_path = cand
                    break
            else:
                matches = list(Path(cand).parent.glob(Path(cand).name))
                if matches:
                    audio_path = str(matches[0])
                    break

    # Config
    config = OrchestratorConfig(
        audio_path=audio_path,
        start_gui=not (args.no_gui or args.headless),
        start_analyzer=True,
        start_pegelexplosion_monitor=True,
        headless=args.headless,
        verbose=args.verbose,
    )

    # Orchestrator
    orchestrator = QualityMonitoringOrchestrator(config)
    exit_code = orchestrator.start_all()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
