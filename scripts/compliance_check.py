#!/usr/bin/env python3
"""
Aurik 10.0.0 — Compliance-Check (VERBOTEN-Regeln aus copilot-instructions.md)
=========================================================================
Prüft alle Python-Quelldateien auf verbotene Muster aus der Ruleset-Tabelle.
Exit 0 = clean, Exit 1 = Verstöße gefunden.

Verwendung:
    python scripts/compliance_check.py                  # alle SRC_DIRS
    python scripts/compliance_check.py backend/core/    # nur ein Verzeichnis
    python scripts/compliance_check.py --fix-fstrings   # auch f-string Logger prüfen
"""

import argparse
import io as _io
import logging
import re
import sys
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zieldirectories (Produktion) — Tests, Scripts, Models werden nie geprüft
# ---------------------------------------------------------------------------
DEFAULT_SRC_DIRS = [
    "backend",
    "core",
    "dsp",
    "plugins",
    "denker",
    "Aurik10",
]

EXCLUDE_DIRS = {
    "models",
    ".venv_aurik",
    "build",
    "dist",
    "__pycache__",
    "output_audio",
    "sessions",
    "logs",
    "data",
    "golden_samples",
    "tests",
    "scripts",
    "benchmarks",
    "audit",
}

# ---------------------------------------------------------------------------
# Regel-Definition
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    """Compliance-Regel (VERBOTEN-Eintrag) mit Regex-Pattern und Schwere."""

    id: str
    description: str
    # Regex-Pattern (compiled), bei Match = Verletzung
    pattern: re.Pattern
    # Wenn True: Match in String- und Kommentar-Kontext NICHT zählen
    skip_in_strings: bool = True
    # Pfade die für diese Regel explizit ausgenommen sind (relativ oder Substring)
    allow_in: list[str] = field(default_factory=list)
    # Schwere: "error" blockiert CI, "warning" nur Hinweis
    severity: str = "error"


RULES: list[Rule] = [
    # R01 — print() in Produktion
    Rule(
        id="R01",
        description="print() verboten — logger.info() verwenden",
        pattern=re.compile(r"\bprint\s*\("),
        allow_in=["__main__", "aurik_cli"],  # CLI-Einstiegspunkte ausgenommen
        severity="error",
    ),
    # R02 — sf.read() direkt
    Rule(
        id="R02",
        description="sf.read() verboten — load_audio_file() aus backend.file_import verwenden",
        pattern=re.compile(r"\bsf\.read\s*\((?!.*BytesIO)"),
        allow_in=["backend/file_import.py"],  # kanonische Implementierung
        severity="error",
    ),
    # R03 — soundfile.read() direkt
    Rule(
        id="R03",
        description="soundfile.read() verboten — load_audio_file() verwenden",
        pattern=re.compile(r"\bsoundfile\.read\s*\((?!.*BytesIO)"),
        allow_in=["backend/file_import.py"],
        severity="error",
    ),
    # R04 — librosa.load() direkt
    Rule(
        id="R04",
        description="librosa.load() verboten — load_audio_file() verwenden",
        pattern=re.compile(r"\blibrosa\.load\s*\((?!.*BytesIO)"),
        allow_in=["backend/file_import.py"],
        severity="error",
    ),
    # R05 — CUDA map_location ohne ml_device_manager
    Rule(
        id="R05",
        description='map_location="cuda" ohne ml_device_manager — get_torch_device() verwenden',
        pattern=re.compile(r'map_location\s*=\s*["\']cuda["\']'),
        allow_in=["backend/core/ml_device_manager.py"],
        severity="error",
    ),
    # R06 — .to("cuda") / .cuda() ohne ml_device_manager
    Rule(
        id="R06",
        description='.to("cuda") / .cuda() ohne ml_device_manager — get_torch_device() verwenden',
        pattern=re.compile(r'\.to\s*\(\s*["\']cuda["\']\s*\)|\.cuda\s*\(\s*\)'),
        allow_in=["backend/core/ml_device_manager.py"],
        severity="error",
    ),
    # R07 — pesq() Metrik
    Rule(
        id="R07",
        description="pesq() verboten — PQS-MOS / VERSA / SingMOS verwenden",
        pattern=re.compile(r"\bpesq\s*\("),
        severity="error",
    ),
    # R08 — dnsmos / nisqa
    Rule(
        id="R08",
        description="dnsmos() / nisqa() verboten — PQS-MOS / VERSA / SingMOS verwenden",
        pattern=re.compile(r"\b(dnsmos|nisqa)\s*[\(\.]"),
        severity="error",
    ),
    # R09 — scipy.signal.wiener() primär eingesetzt
    Rule(
        id="R09",
        description="scipy.signal.wiener() als primärer Denoiser verboten — OMLSA/DeepFilterNet verwenden",
        pattern=re.compile(r"\bscipy\.signal\.wiener\s*\("),
        severity="warning",  # Nur Warning: legitim als Fallback-Fallback
    ),
    # R10 — plm.try_allocate (falsche Budget-API)
    Rule(
        id="R10",
        description="plm.try_allocate() verboten — ml_memory_budget.try_allocate() verwenden",
        pattern=re.compile(r"\bplm\.try_allocate\s*\("),
        severity="error",
    ),
    # R11 — Backend importiert Aurik10 (Architektur-Verletzung)
    Rule(
        id="R11",
        description="Backend darf nicht aus Aurik10 importieren (Architektur-Trennung)",
        pattern=re.compile(r"from\s+Aurik10[.\s]|import\s+Aurik10"),
        allow_in=["Aurik10/", "tests/", "scripts/"],
        severity="error",
    ),
    # R12 — griffinlim als letzter Rekonstruktions-Schritt
    Rule(
        id="R12",
        description="griffinlim() als Phase-Endschritt verboten — PGHI/Vocos verwenden",
        pattern=re.compile(r"\bgriffin_?lim\s*\("),
        allow_in=["dsp/pghi.py", "dsp/phase_reconstruction.py"],  # dort als Fallback OK
        severity="warning",
    ),
    # R13 — RMS/Peak Normalisierung (statt LUFS)
    Rule(
        id="R13",
        description="RMS/Peak-Normalisierung verboten — LUFS ITU-R BS.1770-5 verwenden",
        pattern=re.compile(r"normalize_rms\s*\(|normalize_peak\s*\(|rms_normalize\s*\(|peak_normalize\s*\("),
        severity="warning",
    ),
    # R15 — MediumClassifier.classify_medium() (statt MediumDetector.detect())
    Rule(
        id="R15",
        description=(
            "MediumClassifier.classify_medium() verboten — "
            "MediumDetector.detect(audio, sr, file_ext=...) verwenden (§6.7)"
        ),
        pattern=re.compile(r"\bMediumClassifier\(\)|\bmedium_classifier\s*\.\s*classify_medium\s*\("),
        allow_in=["backend/core/medium_classifier.py"],  # Implementierungs-Datei ausgeschlossen
        severity="error",
    ),
    # R16 — LPC-Ordnung < 10 explizit gesetzt
    # BUG-FIX v10.0.0: Formant-Analyse auf schmalem Vokalband (200–3400 Hz) benötigt
    # lpc_order=14 (ITU-T P.501: 7 Polpaare für F1–F3). Breitband-LPC (0–24 kHz)
    # verwendet 30–40. Schwelle auf 10 gesenkt um beide Anwendungsfälle zu erlauben.
    Rule(
        id="R16",
        description=(
            "LPC-Ordnung < 10 verboten — für Breitband-LPC (0–24 kHz) Ord. 30–40 verwenden; "
            "Schmalband-Formant-Analyse (ITU-T P.501) min. 12"
        ),
        pattern=re.compile(r"\blpc_order\s*=\s*[1-9]\b|order\s*=\s*[1-9]\b.*lpc"),
        severity="error",
    ),
    # R17 — SongCal-Bounds 0.0 / 2.0 (falsche Clip-Grenzen)
    Rule(
        id="R17",
        description="SongCal np.clip(scalar, 0.0, 2.0) verboten — global_scalar∈[0.50,1.50], family_scalar∈[0.30,1.80]",
        pattern=re.compile(
            r"np\.clip\s*\(.*scalar.*,\s*0\.0\s*,\s*2\.0\s*\)|np\.clip\s*\(.*,\s*0\.0\s*,\s*2\.0\s*\).*scalar"
        ),
        allow_in=["backend/core/song_calibration"],
        severity="warning",
    ),
    # R18 — warnings.warn() in Produktionscode (Meldungen landen in stderr, nie im App-Log)
    Rule(
        id="R18",
        description="warnings.warn() verboten — logger.warning(...) verwenden (DeprecationWarning-Aufrufe ausgenommen)",
        pattern=re.compile(r"\bwarnings\.warn\s*\((?!.*DeprecationWarning)(?!.*FutureWarning)(?!.*PendingDeprecation)"),
        allow_in=[
            "scripts/",
            "tests/",
            "audit/",
            "benchmarks/",
            "backend/core/ab_test_manager.py",  # DeprecationWarning (multiline)
            "backend/core/evaluation/quality_control.py",  # DeprecationWarning (module-level)
            "dsp/gpu_pipeline.py",
        ],  # DeprecationWarning (module-level)
        severity="warning",
    ),
    # R14 — f-string in logger-Aufrufen (Performance + Lazy-Evaluation)
    Rule(
        id="R14",
        description='logger.*(f"...") — %-Formatierung verwenden: logger.info("%s", val)',
        pattern=re.compile(r'\blogger\.(debug|info|warning|error|critical)\s*\(\s*f["\']'),
        severity="warning",
    ),
]

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """Einzelner Compliance-Verstoß mit Datei, Zeilennummer und Regelreferenz."""

    rule_id: str
    severity: str
    file: str
    line: int
    text: str
    description: str


def _get_string_literal_lines(source: str) -> set[int]:
    """Gibt die Zeilennummern zurück, die vollständig innerhalb von String-Literalen
    (Docstrings, mehrzeilige Strings) liegen — mittels tokenize."""
    string_lines: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(_io.StringIO(source).readline)
        for tok_type, _, tok_start, tok_end, _tok_line in tokens:
            if tok_type == tokenize.STRING:
                for lineno in range(tok_start[0], tok_end[0] + 1):
                    string_lines.add(lineno)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return string_lines


def _is_in_comment_or_string(line: str, match_start: int) -> bool:
    """Heuristic: true wenn Match in einem Inline-Kommentar (#...) liegt."""
    before = line[:match_start]
    # Alles nach # ist Kommentar (außer innerhalb eines Strings — Näherung)
    if "#" in before:
        # Prüfe ob # innerhalb eines String-Literals liegt (vereinfacht)
        single = before.count("'") % 2
        double = before.count('"') % 2
        if not single and not double:
            return True  # # ist außerhalb von Strings → Kommentar
    return False


def _should_skip_file(path: Path, allow_in: list[str]) -> bool:
    """Gibt True zurück, wenn der Pfad in der Ausnahmeliste (allow_in) enthalten ist."""
    path_str = path.as_posix()
    return any(allowed in path_str for allowed in allow_in)


def scan_file(path: Path, rules: list[Rule], check_fstrings: bool = True) -> list[Violation]:
    """Scannt eine einzelne Python-Datei auf VERBOTEN-Regelverstöße und gibt alle Violations zurück."""
    violations: list[Violation] = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return violations

    lines = content.splitlines()
    path_str = path.as_posix()

    # Zeilen, die in String-Literalen liegen (Docstrings etc.) → nicht prüfen
    string_lines = _get_string_literal_lines(content)

    for rule in rules:
        if rule.id == "R14" and not check_fstrings:
            continue
        if _should_skip_file(path, rule.allow_in):
            continue

        # Für R11: nur in backend/ prüfen
        if rule.id == "R11" and "backend/" not in path_str:
            continue

        for lineno, line in enumerate(lines, start=1):
            m = rule.pattern.search(line)
            if not m:
                continue
            # Kommentar-Zeilen und Doctest-Prompts überspringen
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith(">>>"):
                continue
            # Zeilen innerhalb von String-Literalen / Docstrings überspringen
            if lineno in string_lines:
                continue
            # Docstrings / Kommentar nach Code (Heuristik)
            if rule.skip_in_strings and _is_in_comment_or_string(line, m.start()):
                continue
            violations.append(
                Violation(
                    rule_id=rule.id,
                    severity=rule.severity,
                    file=path_str,
                    line=lineno,
                    text=line.rstrip(),
                    description=rule.description,
                )
            )

    return violations


def collect_python_files(roots: list[Path]) -> list[Path]:
    """Sammelt alle Python-Dateien in den Wurzelverzeichnissen (ohne EXCLUDE_DIRS)."""
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        for p in root.rglob("*.py"):
            # Ausschlüsse
            parts = set(p.parts)
            if parts & EXCLUDE_DIRS:
                continue
            files.append(p)
    return sorted(files)


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
GREEN = "\033[0;32m"
BOLD = "\033[1m"


def report(violations: list[Violation], show_warnings: bool = True) -> int:
    """Gibt Verstöße farbig aus und gibt den Exit-Code zurück (0 = clean, 1 = Errors)."""
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]

    if not violations:
        print(f"{GREEN}{BOLD}✅ Compliance-Check bestanden — keine Verstöße.{RESET}")
        return 0

    if errors:
        print(f"\n{RED}{BOLD}❌ COMPLIANCE ERRORS ({len(errors)}){RESET}")
        for v in errors:
            print(f"  {RED}[{v.rule_id}] {v.file}:{v.line}{RESET}")
            print(f"        {v.description}")
            print(f"        {BOLD}{v.text.strip()}{RESET}")

    if warnings and show_warnings:
        print(f"\n{YELLOW}{BOLD}⚠  COMPLIANCE WARNINGS ({len(warnings)}){RESET}")
        for v in warnings:
            print(f"  {YELLOW}[{v.rule_id}] {v.file}:{v.line}{RESET}")
            print(f"        {v.description}")
            print(f"        {v.text.strip()}{RESET}")

    print()
    print(f"  Errors  : {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print()

    if errors:
        print(f"{RED}CI-Gate: FAILED — {len(errors)} Fehler müssen behoben werden.{RESET}")
        return 1

    print(f"{YELLOW}CI-Gate: PASSED (nur Warnings — non-blocking){RESET}")
    return 0


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Einstiegspunkt: Parst Argumente, scannt Quelldateien und gibt den Exit-Code zurück."""
    parser = argparse.ArgumentParser(description="Aurik 10.0.0 Compliance-Check (VERBOTEN-Regeln)")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Zu prüfende Verzeichnisse/Dateien (default: alle SRC_DIRS)",
    )
    parser.add_argument(
        "--fix-fstrings",
        action="store_true",
        help="Auch f-string Logger (R14) als Warnings ausgeben",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Nur Errors ausgeben, Warnings unterdrücken",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Nur Zusammenfassung, keine Details",
    )
    args = parser.parse_args(argv)

    # Pfade auflösen
    if args.paths:
        roots = [Path(p) for p in args.paths]
    else:
        base = Path(__file__).parent.parent  # Workspace-Root
        roots = [base / d for d in DEFAULT_SRC_DIRS]

    files = collect_python_files(roots)
    if not files:
        print(f"{YELLOW}Keine Python-Dateien gefunden in: {roots}{RESET}")
        return 0

    print(f"Aurik 10.0.0 Compliance-Check — {len(files)} Dateien")
    print("=" * 60)

    all_violations: list[Violation] = []
    for f in files:
        all_violations.extend(scan_file(f, RULES, check_fstrings=args.fix_fstrings))

    if args.summary:
        errors = sum(1 for v in all_violations if v.severity == "error")
        warnings = sum(1 for v in all_violations if v.severity == "warning")
        print(f"Errors: {errors}  Warnings: {warnings}")
        return 1 if errors else 0

    return report(all_violations, show_warnings=not args.errors_only)


if __name__ == "__main__":
    sys.exit(main())
