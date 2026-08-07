#!/usr/bin/env python3
"""
Aurik 10.0.0 — Frontend Launcher mit kontinuierlichem Qualitäts-Monitoring
======================================================================

Startet das PyQt5-Frontend und führt kontinuierliche Tiefenanalyse
parallel durch. Bei kritischen Qualitätsproblemen werden automatische
Code-Fixes vorgeschlagen und implementiert.

Usage:
    python scripts/frontend_with_analysis.py [--audio <path>] [--no-gui] [--analysis-only]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

# Setup paths
_WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE_ROOT))

logger = logging.getLogger(__name__)


class FrontendWithAnalysisSupervisor:
    """Startet Frontend + Analyzer gleichzeitig mit gegenseitiger Überwachung."""

    def __init__(self, audio_path: str | None = None, no_gui: bool = False):
        self.audio_path = audio_path or self._find_default_audio()
        self.no_gui = no_gui
        self.gui_process: subprocess.Popen | None = None
        self.analyzer_process: subprocess.Popen | None = None
        self._shutdown = False

    def _find_default_audio(self) -> str:
        """Findet eine Standard-Audio-Datei."""
        candidates = [
            Path("test_audio") / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
            Path("test_audio") / "tape" / "*.mp3",
            Path("test_audio") / "vinyl" / "*.mp3",
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
            # Glob
            if "*" in str(cand):
                matches = list(Path(cand).parent.glob(Path(cand).name))
                if matches:
                    return str(matches[0])
        return ""

    def start_gui(self) -> bool:
        """Startet das PyQt5-Frontend."""
        if self.no_gui:
            logger.info("GUI deaktiviert (--no-gui)")
            return True

        logger.info("Starte PyQt5-Frontend...")
        try:
            # Starte GUI über run_aurik.sh
            gui_script = _WORKSPACE_ROOT / "run_aurik.sh"
            if not gui_script.exists():
                logger.warning(f"Ausfuehrung_aurik.sh nicht gefunden: {gui_script}")
                return False

            self.gui_process = subprocess.Popen(
                ["bash", str(gui_script)],
                cwd=str(_WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(f"✓ GUI-Prozess gestartet (PID: {self.gui_process.pid})")
            return True
        except Exception as e:
            logger.error(f"✗ GUI-Start fehlgeschlagen: {e}")
            return False

    def start_analyzer(self) -> bool:
        """Startet kontinuierliche Tiefenanalyse."""
        if not self.audio_path:
            logger.warning("Keine Audio-Datei für Analyzer verfügbar")
            return False

        logger.info(f"Starte Tiefenanalyse für: {self.audio_path}")
        try:
            analyzer_script = _WORKSPACE_ROOT / "scripts" / "continuous_deep_analysis.py"
            if not analyzer_script.exists():
                logger.error(f"Analyzer-Script nicht gefunden: {analyzer_script}")
                return False

            self.analyzer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(analyzer_script),
                    "--audio",
                    self.audio_path,
                    "--realtime",
                    "--output-dir",
                    "analysis_results",
                ],
                cwd=str(_WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            logger.info(f"✓ Analyzer-Prozess gestartet (PID: {self.analyzer_process.pid})")
            return True
        except Exception as e:
            logger.error(f"✗ Analyzer-Start fehlgeschlagen: {e}")
            return False

    def monitor_processes(self) -> None:
        """Überwacht Prozesse und zeigt Output."""
        logger.info("=" * 80)
        logger.info("AURIK 9 — FRONTEND + TIEFENANALYSE")
        logger.info("=" * 80)
        logger.info(f"Audio: {self.audio_path}")
        logger.info("")

        # Analyzer-Output streamen
        if self.analyzer_process and self.analyzer_process.stdout:

            def stream_analyzer_output():
                try:
                    while not self._shutdown:
                        line = self.analyzer_process.stdout.readline()  # type: ignore[union-attr]
                        if not line:
                            break
                        logger.info(f"[ANALYZER] {line.rstrip()}")
                except Exception as e:
                    logger.debug(f"Analyzer Ausgabe stream beendet: {e}")

            analyzer_thread = Thread(target=stream_analyzer_output, daemon=True)
            analyzer_thread.start()

        # Prozess-Überwachungs-Loop
        while not self._shutdown:
            time.sleep(1)

            # GUI-Status prüfen
            if self.gui_process:
                ret = self.gui_process.poll()
                if ret is not None and ret != 0:
                    logger.warning(f"GUI-Prozess beendet mit Code {ret}")
                    self._shutdown = True

            # Analyzer-Status prüfen
            if self.analyzer_process:
                ret = self.analyzer_process.poll()
                if ret is not None:
                    if ret == 0:
                        logger.info("✓ Analyzer erfolgreich abgeschlossen")
                    else:
                        logger.error(f"Analyzer beendet mit Code {ret}")
                    self._shutdown = True

    def run(self) -> int:
        """Startet beide Prozesse und überwacht sie."""
        try:
            # GUI starten
            if not self.start_gui():
                logger.warning("GUI-Start fehlgeschlagen, fahre nur mit Analyzer fort")

            # Analyzer starten
            if not self.start_analyzer():
                if self.gui_process:
                    self.gui_process.terminate()
                return 1

            # Überwachen
            self.monitor_processes()

            # Cleanup
            return self.shutdown()

        except KeyboardInterrupt:
            logger.info("\nBenutzer unterbrochen, fahre herunter...")
            return self.shutdown()

    def shutdown(self) -> int:
        """Fährt beide Prozesse sauber herunter."""
        self._shutdown = True
        exit_code = 0

        if self.gui_process:
            try:
                self.gui_process.terminate()
                self.gui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gui_process.kill()
            except Exception as e:
                logger.error(f"GUI-Herunterfahren Fehler: {e}")
                exit_code = 1

        if self.analyzer_process:
            try:
                self.analyzer_process.terminate()
                self.analyzer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.analyzer_process.kill()
            except Exception as e:
                logger.error(f"Analyzer-Herunterfahren Fehler: {e}")
                exit_code = 1

        logger.info("=" * 80)
        logger.info("Aurik beendet")
        logger.info("=" * 80)
        return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Aurik 10.0.0 Frontend mit kontinuierlicher Tiefenanalyse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard: GUI + Analyzer für test_audio
  python scripts/frontend_with_analysis.py

  # Nur Analyzer (kein GUI)
  python scripts/frontend_with_analysis.py --no-gui

  # Custom Audio + nur Analyzer
  python scripts/frontend_with_analysis.py --audio my_song.mp3 --no-gui --analysis-only
        """,
    )
    parser.add_argument("--audio", type=str, default=None, help="Pfad zur Audio-Datei")
    parser.add_argument("--no-gui", action="store_true", help="GUI nicht starten")
    parser.add_argument("--analysis-only", action="store_true", help="Nur Analyzer starten (impliziert --no-gui)")

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("frontend_analysis_runtime.log"),
        ],
    )

    # Supervisor starten
    no_gui = args.no_gui or args.analysis_only
    supervisor = FrontendWithAnalysisSupervisor(audio_path=args.audio, no_gui=no_gui)
    exit_code = supervisor.run()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
