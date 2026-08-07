#!/usr/bin/env python3
"""
Aurik Live-Pipeline-Monitor
============================
Verfolgt `logs/aurik_backend.log` in Echtzeit und meldet kritische Ereignisse:

  - 🔴 PEGELEXPLOSION:  WaveformPlausibilityGuard Notfallkorrektur
  - 🟠 STEREO LAG:      §2.51a Warnstufe / Hard-Fail
  - 🔴 STEREO FAIL:     §2.51a Hard-Fail (Rollback ausgelöst)
  - 🟡 STCG KORREKTUR:  Interchannel-Delay korrigiert (INFO)
  - 🔵 PHASE START/END: Phasenfortschritt mit Zeitstempel
  - 🟣 PMGG ROLLBACK:   Phase zurückgerollt
  - 🔴 RMS-DROP:        Loudness-Verlust > 5 dB in einer Phase
  - 📊 STATISTIKEN:     Am Ende jedes Runs

Verwendung:
  .venv_aurik/bin/python scripts/live_pipeline_monitor.py [--log logs/aurik_backend.log] [--run-once]

  Mit --run-once: wartet auf einen abgeschlossenen Run und zeigt die Zusammenfassung.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── ANSI Farben ──────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_ORANGE = "\033[93m"
_YELLOW = "\033[33m"
_GREEN = "\033[92m"
_BLUE = "\033[94m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"
_DIM = "\033[2m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ── Pattern-Definitionen ─────────────────────────────────────────────────────


@dataclass
class Pattern:
    regex: re.Pattern[str]
    label: str
    color: str
    severity: str  # "CRITICAL", "WARNING", "INFO", "DEBUG"
    extract: list[str] | None = None  # Gruppen aus dem Regex


PATTERNS: list[Pattern] = [
    # ── Pegelexplosionen ─────────────────────────────────────────────────────
    Pattern(
        re.compile(
            r"WaveformPlausibilityGuard.*Quiet-Zone-Notfallkorrektur.*"
            r"quiet_ratio=([\d.]+).*windows=(\d+).*max=([+-]?[\d.]+)\s*dB.*thr=([\d.]+)\s*dB.*mat=(\w+)",
            re.IGNORECASE,
        ),
        label="🔴 PEGELEXPLOSION (WPG)",
        color=_RED + _BOLD,
        severity="CRITICAL",
        extract=["quiet_ratio", "windows", "max_dB", "thr_dB", "material"],
    ),
    Pattern(
        re.compile(
            r"§2\.30c WPG:.*?(\d+)\s+Explosions-Fenster korrigiert.*?max=([+-]?[\d.]+)\s*dB",
            re.IGNORECASE,
        ),
        label="🔴 WPG KORREKTUR",
        color=_RED,
        severity="CRITICAL",
        extract=["windows_corrected", "max_dB"],
    ),
    # ── apply_musical_gain_envelope Pegelexplosion ───────────────────────────
    Pattern(
        re.compile(
            r"apply_musical_gain_envelope.*gate_dbfs=-5[0-9]",
            re.IGNORECASE,
        ),
        label="🔴 VERBOTEN: gate_dbfs<-36",
        color=_RED + _BOLD,
        severity="CRITICAL",
    ),
    # ── L/R-Interchannel-Delay ───────────────────────────────────────────────
    Pattern(
        re.compile(
            r"§2\.51a Stereo-Hard-Fail.*?→.*?Rollback",
            re.IGNORECASE,
        ),
        label="🔴 STEREO HARD-FAIL (Rollback)",
        color=_RED + _BOLD,
        severity="CRITICAL",
    ),
    Pattern(
        re.compile(
            r"§2\.51a Stereo-Warnstufe:\s*(.+)",
            re.IGNORECASE,
        ),
        label="🟠 STEREO WARNSTUFE",
        color=_ORANGE,
        severity="WARNING",
        extract=["reasons"],
    ),
    # ── STCG Korrekturen ─────────────────────────────────────────────────────
    Pattern(
        re.compile(
            r"STCG\s*\[([^\]]+)\]:\s*L-R delay=([\d.]+)\s*samples\s*\(([\d.]+)\s*ms\)\s*[—-]\s*correcting",
            re.IGNORECASE,
        ),
        label="🟡 STCG L/R-Korrektur",
        color=_YELLOW,
        severity="INFO",
        extract=["phase", "samples", "ms"],
    ),
    Pattern(
        re.compile(
            r"STCG\s*\[([^\]]+)\]:\s*L-R delay=([\d.]+)\s*samples\s*\(([\d.]+)\s*ms\)\s*[—-]\s*within threshold",
            re.IGNORECASE,
        ),
        label="🟢 STCG OK (kein Lag)",
        color=_GREEN,
        severity="INFO",
        extract=["phase", "samples", "ms"],
    ),
    # ── PMGG Rollbacks ───────────────────────────────────────────────────────
    Pattern(
        re.compile(
            r"PMGG.*?rollback.*?phase[_\s]([\w]+)",
            re.IGNORECASE,
        ),
        label="🟣 PMGG ROLLBACK",
        color=_MAGENTA,
        severity="WARNING",
        extract=["phase"],
    ),
    Pattern(
        re.compile(
            r"phase.*?rollback.*?consecutive.*?(\d+)",
            re.IGNORECASE,
        ),
        label="🟣 ROLLBACK (konsekutiv)",
        color=_MAGENTA,
        severity="WARNING",
        extract=["count"],
    ),
    # ── RMS-Drops ────────────────────────────────────────────────────────────
    Pattern(
        re.compile(
            r"rms_drop(?:_db)?[=:\s]+([+-]?[\d.]+)",
            re.IGNORECASE,
        ),
        label="🔵 RMS-DROP",
        color=_BLUE,
        severity="INFO",
        extract=["db"],
    ),
    # ── Pipeline-Start/-Ende ─────────────────────────────────────────────────
    Pattern(
        re.compile(
            r"(?:Starting|Starte|UV3|restore\(\))\s*.*?(?:pipeline|phase_plan).*?(?:phases|Phasen).*?(\d+)",
            re.IGNORECASE,
        ),
        label="🔵 PIPELINE START",
        color=_BLUE,
        severity="INFO",
        extract=["n_phases"],
    ),
    Pattern(
        re.compile(
            r"HPI.*?(?:score|index)[=:\s]+([\d.]+).*?(?:timbral|MERT)[=:\s]+([\d.]+)",
            re.IGNORECASE,
        ),
        label="📊 HPI SCORE",
        color=_CYAN,
        severity="INFO",
        extract=["score", "timbral"],
    ),
    Pattern(
        re.compile(
            r"(?:export|Export|saved|gespeichert).*?(?:output|\.wav|\.flac|\.mp3)",
            re.IGNORECASE,
        ),
        label="✅ EXPORT",
        color=_GREEN + _BOLD,
        severity="INFO",
    ),
    # ── Critical Errors ──────────────────────────────────────────────────────
    Pattern(
        re.compile(
            r"(?:CRITICAL|ERROR|Exception|Traceback|OOM|OutOfMemory|RuntimeError)",
            re.IGNORECASE,
        ),
        label="🔴 ERROR/EXCEPTION",
        color=_RED + _BOLD,
        severity="CRITICAL",
    ),
]

# ── RMS-Drop Schwellwert ──────────────────────────────────────────────────────
_RMS_DROP_ALERT_DB = -5.0  # unter diesem Wert wird ein RMS-Drop als kritisch markiert


# ── Run-Statistik ────────────────────────────────────────────────────────────


@dataclass
class RunStats:
    start_time: str = ""
    end_time: str = ""
    pegelexplosion_count: int = 0
    stereo_warnings: int = 0
    stereo_hardfails: int = 0
    stcg_corrections: int = 0
    pmgg_rollbacks: int = 0
    rms_drops: list[float] = field(default_factory=list)
    phases_seen: set[str] = field(default_factory=set)
    last_hpi: str | None = None
    exported: bool = False
    lines_processed: int = 0

    def summary(self) -> str:
        lines: list[str] = [
            "",
            _c(_BOLD, "═" * 60),
            _c(_BOLD + _CYAN, f"  AURIK RUN ZUSAMMENFASSUNG  {self.start_time} → {self.end_time}"),
            _c(_BOLD, "═" * 60),
        ]
        status_color = _GREEN if self.pegelexplosion_count == 0 and self.stereo_hardfails == 0 else _RED
        lines.append(
            f"  Status:            {_c(status_color, '✅ SAUBER' if self.pegelexplosion_count == 0 and self.stereo_hardfails == 0 else '⚠️  PROBLEME ERKANNT')}"
        )
        lines.append(
            f"  Pegelexplosionen:  {_c(_RED if self.pegelexplosion_count > 0 else _GREEN, str(self.pegelexplosion_count))}"
        )
        lines.append(
            f"  Stereo-Warnungen:  {_c(_ORANGE if self.stereo_warnings > 0 else _GREEN, str(self.stereo_warnings))}"
        )
        lines.append(
            f"  Stereo-Hard-Fails: {_c(_RED if self.stereo_hardfails > 0 else _GREEN, str(self.stereo_hardfails))}"
        )
        lines.append(f"  STCG-Korrekturen:  {_c(_YELLOW, str(self.stcg_corrections))}")
        lines.append(
            f"  PMGG-Rollbacks:    {_c(_MAGENTA if self.pmgg_rollbacks > 0 else _DIM, str(self.pmgg_rollbacks))}"
        )
        if self.rms_drops:
            worst = min(self.rms_drops)
            lines.append(
                f"  Schlechtster RMS:  {_c(_RED if worst < _RMS_DROP_ALERT_DB else _YELLOW, f'{worst:.1f} dB')}"
            )
        if self.last_hpi:
            lines.append(f"  HPI:               {_c(_CYAN, self.last_hpi)}")
        lines.append(
            f"  Export:            {_c(_GREEN if self.exported else _ORANGE, '✅ Ja' if self.exported else '⏳ noch nicht')}"
        )
        lines.append(_c(_BOLD, "═" * 60))
        lines.append("")
        return "\n".join(lines)


# ── Monitor-Logik ─────────────────────────────────────────────────────────────


def _parse_log_timestamp(line: str) -> str | None:
    m = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]?\d*)", line)
    return m.group(1) if m else None


def _alert(stats: RunStats, pattern: Pattern, match: re.Match[str], line: str) -> None:
    """Gibt einen formatierten Alert aus und aktualisiert Statistiken."""
    ts = _parse_log_timestamp(line) or _ts()
    groups = match.groups() if match.lastindex else ()

    # Statistik-Updates
    lbl = pattern.label
    if "PEGELEXPLOSION" in lbl or "WPG KORREKTUR" in lbl:
        stats.pegelexplosion_count += 1
    elif "STEREO HARD-FAIL" in lbl:
        stats.stereo_hardfails += 1
    elif "STEREO WARNSTUFE" in lbl:
        stats.stereo_warnings += 1
    elif "STCG L/R-Korrektur" in lbl:
        stats.stcg_corrections += 1
    elif "PMGG ROLLBACK" in lbl or "ROLLBACK (konsek" in lbl:
        stats.pmgg_rollbacks += 1
    elif "RMS-DROP" in lbl and groups:
        try:
            val = float(groups[0])
            stats.rms_drops.append(val)
            if val >= _RMS_DROP_ALERT_DB:
                return  # RMS-Drop unter 5 dB = kein Alert
        except ValueError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    elif "HPI SCORE" in lbl and groups:
        stats.last_hpi = f"score={groups[0]}, timbral={groups[1]}"
    elif "EXPORT" in lbl:
        stats.exported = True

    # Ausgabe
    detail_parts: list[str] = []
    if pattern.extract and groups:
        for name, val in zip(pattern.extract, groups):  # type: ignore[assignment]
            detail_parts.append(f"{name}={val}")
    detail = "  " + " | ".join(detail_parts) if detail_parts else ""

    prefix = _c(pattern.color, f"[{ts}] {pattern.label}")
    print(f"{prefix}{_c(_DIM, detail)}", flush=True)


def _process_line(line: str, stats: RunStats) -> None:
    stats.lines_processed += 1
    stripped = line.strip()
    if not stripped:
        return

    # Pipeline-Start erkennen
    ts_str = _parse_log_timestamp(stripped)
    if ts_str and not stats.start_time:
        stats.start_time = ts_str

    for pattern in PATTERNS:
        m = pattern.regex.search(stripped)
        if m:
            _alert(stats, pattern, m, stripped)
            # Nur erstes passendes Pattern pro Zeile
            break


def monitor(
    log_path: Path,
    run_once: bool = False,
    tail_lines: int = 0,
    quiet: bool = False,
) -> None:
    """Endlos-Tail des Logs mit Echtzeit-Alerts."""
    if not quiet:
        print(_c(_BOLD + _CYAN, f"\n{'=' * 60}"))
        print(_c(_BOLD + _CYAN, "  Aurik Live-Pipeline-Monitor"))
        print(_c(_BOLD + _CYAN, f"  Log: {log_path}"))
        print(_c(_BOLD + _CYAN, f"{'=' * 60}\n"))
        print(_c(_DIM, "Warte auf Log-Einträge … (Ctrl+C zum Beenden)\n"))

    stats = RunStats(start_time=_ts())

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        # Bei tail_lines=0 → ans Ende springen
        if tail_lines == 0:
            fh.seek(0, 2)  # EOF
        elif tail_lines > 0:
            # Letzte N Zeilen lesen
            all_lines = fh.readlines()
            for ln in all_lines[-tail_lines:]:
                _process_line(ln, stats)

        run_ended = False
        while True:
            line = fh.readline()
            if not line:
                if run_once and run_ended:
                    break
                time.sleep(0.05)
                continue

            _process_line(line, stats)

            # Run-Ende-Erkennung
            if re.search(r"(?:export|Export|saved|gespeichert).*?(?:\.wav|\.flac|\.mp3)", line, re.IGNORECASE):
                stats.end_time = _parse_log_timestamp(line.strip()) or _ts()
                run_ended = True
                if not quiet:
                    print(stats.summary())
                if run_once:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aurik Live-Pipeline-Monitor — Echtzeit-Pegelexplosion + Stereo-Lag-Tracking"
    )
    parser.add_argument(
        "--log",
        default="logs/aurik_backend.log",
        help="Pfad zur Log-Datei (default: logs/aurik_backend.log)",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Beendet nach dem nächsten abgeschlossenen Run",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        metavar="N",
        help="Letzte N Zeilen vor Echtzeit-Monitor verarbeiten (0 = nur neue Zeilen)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Verarbeitet die gesamte Log-Datei und gibt die Zusammenfassung aus",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Keine Begrüßungs-Header",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        # Relative zum Workspace-Root
        log_path = Path(__file__).parent.parent / log_path
    if not log_path.exists():
        print(f"[ERROR] Log-Datei nicht gefunden: {log_path}", file=sys.stderr)
        sys.exit(1)

    if args.replay:
        # Gesamte Datei verarbeiten
        stats = RunStats(start_time="(replay)")
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                _process_line(line, stats)
        stats.end_time = _ts()
        print(stats.summary())
        return

    try:
        monitor(log_path, run_once=args.run_once, tail_lines=args.tail, quiet=args.quiet)
    except KeyboardInterrupt:
        print(_c(_DIM, "\n[Monitor beendet]"), flush=True)


if __name__ == "__main__":
    main()
