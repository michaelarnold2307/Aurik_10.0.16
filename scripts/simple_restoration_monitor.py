#!/usr/bin/env python3
"""
Aurik 10.0.0 — Vereinfachter Restaurierungs-Monitor
===============================================

Direkter Monitor ohne komplexe Orchestrierung.
Startet Frontend + Überwacht Exports auf Pegelexplosionen.

Usage:
    python scripts/simple_restoration_monitor.py --audio <path>
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

_WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE_ROOT))

logger = logging.getLogger(__name__)


class SimpleRestorationMonitor:
    """Vereinfachter Monitor: Frontend + Export-Überwachung."""

    def __init__(self, audio_path: str, headless: bool = False):
        self.audio_path = audio_path
        self.headless = headless
        self.gui_process = None
        self.gui_pid: int | None = None
        self.gui_pid_owned = False
        self.monitor_process = None
        self.shutdown_event = Event()

    def _read_gui_pid_file(self) -> int | None:
        """Read detached GUI PID written by run_aurik.sh in VS Code terminals."""
        pid_file = _WORKSPACE_ROOT / "temp_repro" / "aurik_gui.pid"
        try:
            if not pid_file.exists():
                return None
            value = pid_file.read_text(encoding="utf-8").strip()
            return int(value) if value else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _find_running_gui_pid() -> int | None:
        """Find an already running Aurik GUI process."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "Aurik10/main.py"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    return int(line)
        except (OSError, ValueError):
            return None
        return None

    @staticmethod
    def _pid_running(pid: int | None) -> bool:
        if pid is None or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def start(self) -> int:  # type: ignore[return]
        """Startet Monitor und Frontend."""
        logger.info("=" * 100)
        logger.info("AURIK 9 — VEREINFACHTER RESTAURIERUNGS-MONITOR")
        logger.info("=" * 100)
        logger.info(f"Audio:    {self.audio_path}")
        logger.info("Betriebsart:     Restoration (Standard)")
        logger.info(f"Headless: {self.headless}")
        logger.info("")

        try:
            # Frontend starten (wenn nicht headless)
            if not self.headless:
                self._start_gui()
                time.sleep(2)

            # Export-Monitor starten
            self._start_export_monitor()
            time.sleep(1)

            # Überwachungs-Loop
            self._run_monitoring()

        except KeyboardInterrupt:
            logger.info("\nBenutzer unterbrochen...")
        finally:
            self._shutdown()

    def _start_gui(self) -> None:
        """Startet PyQt5-Frontend."""
        logger.info("Starte Frontend...")
        gui_script = _WORKSPACE_ROOT / "run_aurik.sh"
        pid_file = _WORKSPACE_ROOT / "temp_repro" / "aurik_gui.pid"
        pid_file_mtime_before = pid_file.stat().st_mtime if pid_file.exists() else None
        pre_existing_gui_pid = self._find_running_gui_pid()

        try:
            self.gui_process = subprocess.Popen(  # type: ignore[assignment]
                ["bash", str(gui_script)],
                cwd=str(_WORKSPACE_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"✓ Frontend gestartet (PID: {self.gui_process.pid})")  # type: ignore[attr-defined]

            # In VS Code detached run_aurik.sh exits immediately after writing the real GUI PID.
            # Wait briefly for the pid file and switch monitoring to the actual GUI process.
            for _ in range(20):
                if self.gui_process.poll() is None:  # type: ignore[attr-defined]
                    break
                time.sleep(0.1)

            pid_file_mtime_after = pid_file.stat().st_mtime if pid_file.exists() else None
            post_launch_gui_pid = self._find_running_gui_pid()
            if self.gui_process.poll() is not None and pid_file_mtime_after != pid_file_mtime_before:  # type: ignore[attr-defined]
                self.gui_pid = self._read_gui_pid_file()
                self.gui_pid_owned = self._pid_running(self.gui_pid)
                if self.gui_pid_owned:
                    logger.info(f"✓ Frontend detached aktiv (GUI-PID: {self.gui_pid})")
            elif self.gui_process.poll() is not None and self._pid_running(post_launch_gui_pid):  # type: ignore[attr-defined]
                self.gui_pid = post_launch_gui_pid
                self.gui_pid_owned = bool(
                    post_launch_gui_pid is not None and post_launch_gui_pid != pre_existing_gui_pid
                )
                if self.gui_pid_owned:
                    logger.info(f"✓ Frontend aktiv (GUI-PID: {self.gui_pid})")
                else:
                    logger.info(f"✓ Bereits laufendes Frontend erkannt (GUI-PID: {self.gui_pid})")
        except Exception as e:
            logger.error(f"✗ Frontend-Start fehlgeschlagen: {e}")

    def _start_export_monitor(self) -> None:
        """Startet Export-Monitor im Hintergrund."""
        logger.info("Starte Ausgabe-Monitor...")
        monitor_script = _WORKSPACE_ROOT / "scripts" / "pegelexplosion_monitor.py"

        try:
            self.monitor_process = subprocess.Popen(  # type: ignore[assignment]
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
            logger.info(f"✓ Ausgabe-Monitor gestartet (PID: {self.monitor_process.pid})")  # type: ignore[attr-defined]

            # Output-Streaming in Thread
            Thread(target=self._stream_monitor_output, daemon=True).start()
        except Exception as e:
            logger.error(f"✗ Monitor-Start fehlgeschlagen: {e}")

    def _stream_monitor_output(self) -> None:
        """Streamt Monitor-Output."""
        if not self.monitor_process or not self.monitor_process.stdout:
            return

        try:
            for line in iter(self.monitor_process.stdout.readline, ""):
                if line:
                    logger.info(f"[MONITOR] {line.rstrip()}")
        except Exception as e:
            logger.debug(f"Monitor Ausgabe stream beendet: {e}")

    def _run_monitoring(self) -> None:
        """Monitoring-Loop."""
        logger.info("")
        logger.info("=" * 100)
        logger.info("MONITORING AKTIV — Frontend läuft")
        logger.info("=" * 100)
        logger.info("Beobachte Ausgabe-Verzeichnis auf Pegelexplosionen...")
        logger.info("Drücke Ctrl+C zum Beenden")
        logger.info("")

        while not self.shutdown_event.is_set():
            # GUI prüfen
            if self.gui_pid_owned or self.gui_pid is not None:
                if not self._pid_running(self.gui_pid):
                    logger.info("✓ Frontend beendet")
                    self.shutdown_event.set()
                    break
            elif self.gui_process:
                if self.gui_process.poll() is not None:
                    _recovered_pid = self._read_gui_pid_file() or self._find_running_gui_pid()
                    if self._pid_running(_recovered_pid):
                        self.gui_pid = _recovered_pid
                        self.gui_pid_owned = False
                        logger.info(f"✓ Bereits laufendes/detached Frontend übernommen (GUI-PID: {self.gui_pid})")
                    else:
                        logger.info("✓ Frontend beendet")
                        self.shutdown_event.set()
                        break

            # Monitor prüfen
            if self.monitor_process:
                if self.monitor_process.poll() is not None:
                    logger.warning("✗ Monitor beendet unerwartet")
                    break

            time.sleep(1)

    def _shutdown(self) -> int:
        """Fährt alles herunter."""
        logger.info("")
        logger.info("=" * 100)
        logger.info("FAHRE PROZESSE HERUNTER")
        logger.info("=" * 100)

        # Frontend
        if self.gui_pid_owned and self._pid_running(self.gui_pid):
            try:
                logger.info("Beende Frontend...")
                os.kill(self.gui_pid, signal.SIGTERM)  # type: ignore[arg-type]
                for _ in range(50):
                    if not self._pid_running(self.gui_pid):
                        break
                    time.sleep(0.1)
                if self._pid_running(self.gui_pid):
                    os.kill(self.gui_pid, signal.SIGKILL)  # type: ignore[arg-type]
                    logger.warning("✗ Frontend force-kill")
                else:
                    logger.info("✓ Frontend beendet")
            except Exception as e:
                logger.error(f"Frontend-Fehler: {e}")
        elif self.gui_process:
            try:
                logger.info("Beende Frontend...")
                self.gui_process.terminate()
                self.gui_process.wait(timeout=5)
                logger.info("✓ Frontend beendet")
            except subprocess.TimeoutExpired:
                self.gui_process.kill()
                logger.warning("✗ Frontend force-kill")
            except Exception as e:
                logger.error(f"Frontend-Fehler: {e}")

        # Monitor
        if self.monitor_process:
            try:
                logger.info("Beende Ausgabe-Monitor...")
                self.monitor_process.terminate()
                self.monitor_process.wait(timeout=5)
                logger.info("✓ Monitor beendet")
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
                logger.warning("✗ Monitor force-kill")
            except Exception as e:
                logger.error(f"Monitor-Fehler: {e}")

        logger.info("")
        logger.info("=" * 100)
        logger.info("Monitoring beendet")
        logger.info("")
        logger.info("Ergebnisse:")
        logger.info("  - Audio-Exports:  Ausgabe_audio/")
        logger.info("  - Analyse-Logs:   *.log")
        logger.info("=" * 100)
        logger.info("")

        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Aurik 10.0.0 — Vereinfachter Restaurierungs-Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard: Frontend + Export-Monitor
  python scripts/simple_restoration_monitor.py --audio my_song.mp3

  # Headless (nur Monitor, kein GUI)
  python scripts/simple_restoration_monitor.py --audio my_song.mp3 --headless

  # Mit Standard-Elke-Best-Audio
  python scripts/simple_restoration_monitor.py
        """,
    )
    parser.add_argument("--audio", type=str, default=None, help="Audio-Datei-Pfad")
    parser.add_argument("--headless", action="store_true", help="Nur Monitor, kein GUI")
    parser.add_argument("--verbose", action="store_true", help="Verbose Logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("simple_monitor_runtime.log"),
        ],
    )

    # Audio suchen
    audio_path = args.audio
    if not audio_path:
        default_path = "test_audio/Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
        if Path(default_path).exists():
            audio_path = default_path
        else:
            matches = list(Path("test_audio").glob("*.mp3"))
            if matches:
                audio_path = str(matches[0])
            else:
                logger.error("✗ Keine Audio-Datei gefunden")
                return 1

    # Monitor
    monitor = SimpleRestorationMonitor(audio_path, headless=args.headless)
    return monitor.start()


if __name__ == "__main__":
    sys.exit(main())
