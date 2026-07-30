#!/usr/bin/env python3
"""§v10.52 Bulk-Fix: Übersetzt bestehende englische Log-Meldungen nach Deutsch.

Sicher: Nur logger-Meldungen werden ersetzt. Kein Code, keine Variablen.
Führt einen Dry-Run aus (--dry) oder wendet Änderungen an (ohne Flag).

Autor: Aurik 10 — 19. Juli 2026
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Mapping: englisches Muster → deutsche Entsprechung
# Reihenfolge ist wichtig — spezifischere Muster zuerst
REPLACEMENTS: list[tuple[str, str]] = [
    # Allgemeine Status-Meldungen
    ("initialized successfully", "erfolgreich initialisiert"),
    ("initialized", "initialisiert"),
    ("Initialized", "Initialisiert"),
    ("successfully", "erfolgreich"),
    ("Successfully", "Erfolgreich"),
    ("completed successfully", "erfolgreich abgeschlossen"),
    ("completed", "abgeschlossen"),
    ("Completed", "Abgeschlossen"),
    ("finished", "abgeschlossen"),
    ("Finished", "Abgeschlossen"),
    ("started", "gestartet"),
    ("Started", "Gestartet"),
    ("starting", "starte"),
    ("Starting", "Starte"),
    # Fehler/Probleme
    ("failed to", "konnte nicht"),
    ("Failed to", "Konnte nicht"),
    ("failed:", "fehlgeschlagen:"),
    ("Failed:", "Fehlgeschlagen:"),
    ("failed", "fehlgeschlagen"),
    ("Failed", "Fehlgeschlagen"),
    ("error:", "Fehler:"),
    ("Error:", "Fehler:"),
    ("exception:", "Ausnahme:"),
    ("Exception:", "Ausnahme:"),
    ("non-blocking", "nicht-blockierend"),
    ("non-critical", "nicht-kritisch"),
    ("fallback", "Fallback"),
    ("Fallback", "Fallback"),
    ("ignoring", "ignoriert"),
    ("Ignoring", "Ignoriert"),
    # Lade/Speicher-Operationen
    ("loading", "lade"),
    ("Loading", "Lade"),
    ("loaded", "geladen"),
    ("Loaded", "Geladen"),
    ("saving", "speichere"),
    ("Saving", "Speichere"),
    ("saved", "gespeichert"),
    ("Saved", "Gespeichert"),
    ("creating", "erstelle"),
    ("Creating", "Erstelle"),
    ("created", "erstellt"),
    ("Created", "Erstellt"),
    ("writing", "schreibe"),
    ("Writing", "Schreibe"),
    ("reading", "lese"),
    ("Reading", "Lese"),
    # Verarbeitung
    ("processing", "verarbeite"),
    ("Processing", "Verarbeite"),
    ("processed", "verarbeitet"),
    ("Processed", "Verarbeitet"),
    ("applying", "wende an"),
    ("Applying", "Wende an"),
    ("applied", "angewendet"),
    ("Applied", "Angewendet"),
    ("executing", "führe aus"),
    ("Executing", "Führe aus"),
    ("executed", "ausgeführt"),
    ("Executed", "Ausgeführt"),
    ("skipping", "überspringe"),
    ("Skipping", "Überspringe"),
    ("skipped", "übersprungen"),
    ("Skipped", "Übersprungen"),
    ("running", "führe aus"),
    ("Running", "Führe aus"),
    ("updating", "aktualisiere"),
    ("Updating", "Aktualisiere"),
    ("updated", "aktualisiert"),
    ("Updated", "Aktualisiert"),
    ("checking", "prüfe"),
    ("Checking", "Prüfe"),
    ("validating", "validiere"),
    ("Validating", "Validiere"),
    ("computing", "berechne"),
    ("Computing", "Berechne"),
    ("extracting", "extrahiere"),
    ("Extracting", "Extrahiere"),
    ("detecting", "erkenne"),
    ("Detecting", "Erkenne"),
    ("detected", "erkannt"),
    ("Detected", "Erkannt"),
    ("generating", "generiere"),
    ("Generating", "Generiere"),
    ("generated", "generiert"),
    ("Generated", "Generiert"),
    # Status
    ("available", "verfügbar"),
    ("Available", "Verfügbar"),
    ("unavailable", "nicht verfügbar"),
    ("Unavailable", "Nicht verfügbar"),
    ("enabled", "aktiviert"),
    ("Enabled", "Aktiviert"),
    ("disabled", "deaktiviert"),
    ("Disabled", "Deaktiviert"),
    ("configured", "konfiguriert"),
    ("Configured", "Konfiguriert"),
    ("active", "aktiv"),
    ("Active", "Aktiv"),
    ("inactive", "inaktiv"),
    ("Inactive", "Inaktiv"),
    ("ready", "bereit"),
    ("Ready", "Bereit"),
    ("pending", "ausstehend"),
    ("Pending", "Ausstehend"),
    # Kalibrierung
    ("calibrated", "kalibriert"),
    ("Calibrated", "Kalibriert"),
    ("calibration", "Kalibrierung"),
    ("Calibration", "Kalibrierung"),
    ("threshold", "Schwelle"),
    ("Threshold", "Schwelle"),
    ("mode:", "Modus:"),
    ("Mode:", "Modus:"),
    ("profile:", "Profil:"),
    ("Profile:", "Profil:"),
    ("session", "Sitzung"),
    ("Session", "Sitzung"),
    ("budget", "Budget"),
    ("Budget", "Budget"),
    ("cache", "Cache"),
    ("Cache", "Cache"),
    ("cached", "gecached"),
    ("Cached", "Gecached"),
]


def _is_logger_line(line: str) -> bool:
    """Prüft ob eine Zeile einen Logger-Aufruf enthält."""
    return bool(re.search(r"logger\.(debug|info|warning|error)\s*\(", line))


def fix_file(filepath: Path, dry_run: bool = True) -> int:
    """Ersetzt englische Log-Meldungen in einer Datei. Gibt Anzahl Änderungen zurück."""
    try:
        with open(filepath) as f:
            original = f.read()
    except Exception:
        return 0

    lines = original.split("\n")
    changed = 0

    for i, line in enumerate(lines):
        if not _is_logger_line(line):
            continue

        new_line = line
        for en, de in REPLACEMENTS:
            # Nur innerhalb von String-Literalen ersetzen
            # Sucht nach: logger.xxx("...text..."  oder  logger.xxx('...text...'
            m = re.search(r'(logger\.\w+\s*\(\s*["\'])(.*?)(["\'])', new_line)
            if m:
                prefix = m.group(1)
                msg = m.group(2)
                suffix = m.group(3)
                new_msg = msg
                # Ersetze ganze Wörter (nicht Teilwörter)
                pattern = re.compile(r"\b" + re.escape(en) + r"\b")
                new_msg = pattern.sub(de, new_msg)
                if new_msg != msg:
                    new_line = prefix + new_msg + suffix + new_line[m.end() :]
                    # Re-scan the rest of the line
                    continue

        if new_line != line:
            lines[i] = new_line
            changed += 1

    if changed > 0 and not dry_run:
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    return changed


def main() -> int:
    dry_run = "--dry" in sys.argv
    mode = "DRY-RUN" if dry_run else "FIX"
    print(f"=== §v10.52 Bulk-Fix: Englische Logs → Deutsch ({mode}) ===\n")

    total_files = 0
    total_changes = 0

    for root_dir in ["backend", "denker", "Aurik10"]:
        root = _PROJECT_ROOT / root_dir
        if not root.exists():
            continue
        for filepath in sorted(root.rglob("*.py")):
            if "test" in str(filepath) or "__pycache__" in str(filepath):
                continue
            changes = fix_file(filepath, dry_run=dry_run)
            if changes > 0:
                rel = filepath.relative_to(_PROJECT_ROOT)
                print(f"  {changes:3d} Änderungen: {rel}")
                total_files += 1
                total_changes += changes

    print(f"\n{'=' * 60}")
    print(f"Dateien mit englischen Logs: {total_files}")
    print(f"Einzelne Log-Meldungen gefixt: {total_changes}")

    if dry_run:
        print("\n📋 DRY-RUN — keine Änderungen vorgenommen.")
        print("   Zum Anwenden: python scripts/pre_commit_language_fix.py")
        return 0 if total_changes == 0 else 1
    else:
        print(f"\n✅ {total_changes} Log-Meldungen in {total_files} Dateien übersetzt.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
