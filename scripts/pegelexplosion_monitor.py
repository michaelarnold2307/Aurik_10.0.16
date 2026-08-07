#!/usr/bin/env python3
"""
Aurik 10.0.0 — Echtzeit-Pegelexplosion-Monitor
===========================================

Überwacht kontinuierlich die output_audio/-Verzeichnisse und prüft
jede neue Export-Datei auf Pegelexplosionen. Zeigt Warnungen an und
schlägt automatisch Fixes vor.

Läuft als separater Prozess parallel zum Frontend.

Usage:
    python scripts/pegelexplosion_monitor.py [--watch-dir output_audio] [--interval 2]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from threading import Event

# Setup paths
_WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE_ROOT))

logger = logging.getLogger(__name__)


class RealtimePegelexplosionMonitor:
    """Überwacht Export-Verzeichnis auf Pegelexplosionen."""

    def __init__(self, watch_dir: str = "output_audio", interval_s: float = 2.0):
        self.watch_dir = Path(watch_dir)
        self.interval_s = interval_s
        self.seen_files: set[str] = set()
        self.shutdown_event = Event()
        self._import_pegelexplosion_detector()

    def _import_pegelexplosion_detector(self) -> None:
        """Lädt den Detektor."""
        try:
            from scripts.pegelexplosion_detector import PegelexplosionDetector

            self.detector = PegelexplosionDetector()
        except ImportError as e:
            logger.error(f"Pegelexplosion-Detektor nicht verfügbar: {e}")
            self.detector = None  # type: ignore[assignment]

    def monitor_loop(self) -> None:
        """Hauptüberwachungs-Schleife."""
        logger.info(f"Starte Echtzeit-Monitor für: {self.watch_dir}")
        if not self.watch_dir.exists():
            self.watch_dir.mkdir(parents=True, exist_ok=True)

        while not self.shutdown_event.is_set():
            try:
                self._check_for_new_files()
                time.sleep(self.interval_s)
            except Exception as e:
                logger.error(f"Monitor-Fehler: {e}")
                time.sleep(1)

        logger.info("Monitor beendet")

    def _check_for_new_files(self) -> None:
        """Prüft auf neue Audio-Dateien."""
        if not self.watch_dir.exists():
            return

        for audio_file in self.watch_dir.glob("*.wav"):
            file_key = str(audio_file)
            if file_key in self.seen_files:
                continue

            self.seen_files.add(file_key)
            self._analyze_file(audio_file)

    def _analyze_file(self, audio_path: Path) -> None:
        """Analysiert eine Audio-Datei auf Pegelexplosionen."""
        if not self.detector:
            return

        logger.info(f"Analysiere: {audio_path.name}")

        # §v10.50 Retry-Loop: transient WAV read errors (race condition beim Schreiben)
        _max_retries = 3
        _retry_delay_s = 0.5
        for _attempt in range(_max_retries):
            try:
                from backend.file_import import load_audio_file

                result = load_audio_file(str(audio_path))
                if result is None or result.get("error"):
                    _err = result.get("error") if result else "None"
                    if _attempt < _max_retries - 1 and "unpack" in str(_err).lower():
                        logger.debug(
                            "  ⏳ laden Wiederholung %d/%d nach transientem Fehler: %s",
                            _attempt + 1,
                            _max_retries,
                            _err,
                        )
                        time.sleep(_retry_delay_s * (_attempt + 1))
                        continue
                    logger.warning(f"  ✗ laden fehlgeschlagen: {_err}")
                    return
                audio = result["audio"]
                sr = result["sr"]
                logger.debug(f"  Geladen: {len(audio) / sr:.1f}s @ {sr} Hz")
                break
            except Exception as e:
                if _attempt < _max_retries - 1 and "unpack" in str(e).lower():
                    logger.debug(
                        "  ⏳ laden Wiederholung %d/%d nach transientem Fehler: %s", _attempt + 1, _max_retries, e
                    )
                    time.sleep(_retry_delay_s * (_attempt + 1))
                    continue
                logger.warning(f"  ✗ laden fehlgeschlagen: {e}")
                return

        # Pegelexplosion-Analyse
        findings = self.detector.analyze_audio_for_spikes(
            audio,
            sr,
            phase_id=audio_path.stem,
        )

        # Bericht
        if findings.has_spike:
            self._report_spike_found(audio_path, findings)
        else:
            logger.info("  ✓ Keine Pegelexplosionen erkannt")

    def _report_spike_found(self, audio_path: Path, findings) -> None:
        """Gibt Bericht über gefundene Spikes."""
        severity_emoji = {
            "none": "✓",
            "minor": "⚠",
            "moderate": "⚠⚠",
            "critical": "🚨",
        }.get(findings.severity, "?")

        logger.warning(f"\n  {severity_emoji} PEGELEXPLOSION ERKANNT ({findings.severity})")
        logger.warning(f"  Audio: {audio_path.name}")
        logger.warning(f"  Spikes: {len(findings.spike_locations)} @ {findings.spike_locations}")
        logger.warning(f"  Magnituden: {findings.spike_magnitudes} dB")

        if findings.fade_out_spike:
            logger.warning("  ⚠ Fade-Out-Region betroffen")
        if findings.intro_spike:
            logger.warning("  ⚠ Intro-Region betroffen")
        if findings.quiet_zone_spike:
            logger.warning("  ⚠ Stille-Zone betroffen")

        if findings.probable_cause:
            logger.warning(f"  Wahrscheinliche Ursache: {findings.probable_cause}")
            if findings.recommendation:
                logger.warning(f"  Empfehlung: {findings.recommendation}")

        # Fixes anzeigen
        fixes = self.detector.suggest_fixes(findings)
        if fixes:
            logger.warning("\n  Suggested Code Fixes:")
            for fix in fixes:
                logger.warning(f"  {fix}")

        # Alert bei kritisch
        if findings.severity == "critical":
            logger.error(f"\n🚨 KRITISCHE PEGELEXPLOSION: {audio_path.name}")
            logger.error("   Sofortiges Eingreifen empfohlen!")

    def shutdown(self) -> None:
        """Fährt Monitor herunter."""
        self.shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(
        description="Aurik 10.0.0 — Echtzeit-Pegelexplosion-Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard: Überwache output_audio/ mit 2s Intervall
  python scripts/pegelexplosion_monitor.py

  # Custom Verzeichnis + schnellere Checks
  python scripts/pegelexplosion_monitor.py --watch-dir output_audio --interval 1
        """,
    )
    parser.add_argument("--watch-dir", type=str, default="output_audio", help="Zu überwachendes Verzeichnis")
    parser.add_argument("--interval", type=float, default=2.0, help="Check-Intervall in Sekunden")
    parser.add_argument("--verbose", action="store_true", help="Verbose Logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pegelexplosion_monitor.log"),
        ],
    )

    # Monitor starten
    monitor = RealtimePegelexplosionMonitor(watch_dir=args.watch_dir, interval_s=args.interval)

    def signal_handler(sig, frame):
        logger.info("Herunterfahren signalisiert...")
        monitor.shutdown()

    import signal

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Monitor-Loop
    monitor.monitor_loop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
