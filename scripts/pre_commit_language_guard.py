#!/usr/bin/env python3
"""§v10.51 Pre-Commit Sprache-Guard — erzwingt deutsche Log-Meldungen.

Prüft vor jedem Commit:
  - logger.info/warning/error/debug Meldungen auf englische Wörter
  - Blockiert Commits mit NEUEN englischen Log-Meldungen
  - Bestehende englische Meldungen sind in .language_guard_whitelist.txt erfasst
  - Erzwingt einheitlich deutsche Terminal-Ausgabe während der Restaurierung

Exit 0 = sauber, Exit 1 = neue englische Log-Meldung gefunden.

Autor: Aurik 10 — 19. Juli 2026
"""

from __future__ import annotations

import ast
import hashlib
import io
import logging
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WHITELIST_PATH = _PROJECT_ROOT / ".language_guard_whitelist.txt"


def _load_whitelist() -> set[str]:
    """Lädt die Whitelist bestehender EN-Meldungen."""
    if not _WHITELIST_PATH.exists():
        return set()
    try:
        with open(_WHITELIST_PATH) as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


_WHITELIST: set[str] = _load_whitelist()

# Deutsche Fachbegriffe, die in Logs OK sind (keine false positives)
_GERMAN_TECH_TERMS: set[str] = {
    "debug",
    "info",
    "warning",
    "error",  # logger method names
    "ok",
    "pass",  # status
}

# Englische Wörter, die in Log-Meldungen NICHT vorkommen dürfen
# (Groß-/Kleinschreibung wird ignoriert)
_ENGLISH_FORBIDDEN: list[str] = [
    "non-blocking",
    "non-critical",
    "fallback",
    "calibrated",
    "calibration",
    "threshold",
    "thresh",
    "session",
    "capture",
    "record",
    "failed",
    "failure",
    "skipped",
    "skip",
    "executed",
    "execute",
    "applied",
    "apply",
    "completed",
    "complete",
    "started",
    "finished",
    "error",  # in Log-Text, nicht als Logger-Methode
    "warning",  # in Log-Text
    "update",
    "updating",
    "recovery",
    "recover",
    "loaded",
    "loading",
    "load",
    "saved",
    "save",
    "saving",
    "created",
    "create",
    "initialized",
    "initialize",
    "detected",
    "detect",
    "processed",
    "process",
    "generated",
    "generate",
    "configured",
    "configure",
    "enabled",
    "disabled",
    "available",
    "unavailable",
    "successful",
    "successfully",
    "failed to",
    "unable to",
    "trying to",
    "attempt",
    "retry",
    "timeout",
    "cached",
    "cache",
    "cleared",
    "clear",
    "reset",
    "aborted",
    "abort",
    "terminated",
    "terminate",
    "shutdown",
    "startup",
    "initializing",
    "finalize",
    "validate",
    "validation",
    "verifying",
    "verify",
    "checking",
    "check",
    "computing",
    "compute",
    "analyzing",
    "analysis",
    "extracting",
    "extract",
    "importing",
    "export",
    "reading",
    "writing",
    "fetching",
    "fetch",
    "sending",
    "send",
    "receiving",
    "receive",
    "connecting",
    "connection",
    "disconnect",
    "pending",
    "running",
    "run",
    "stopping",
    "stopped",
    "restart",
    "stage",
    "phase",
    "mode",
    "profile",
    "profiling",
    "budget",
    "ratio",
    "score",
    "result",
    "output",
    "input",
    "reference",
    "original",
    "restored",
    "restore",
    "enhanced",
    "enhance",
    "optimized",
    "optimize",
    "adjusted",
    "adjust",
    "normalized",
    "normalize",
]

# Ausnahmen: Zeilen die trotz englischer Wörter OK sind
# (z.B. Code-Kommentare, Spec-Referenzen, technische IDs)
_EXEMPT_PATTERNS: list[str] = [
    r"#\s*§",  # Spec-Referenzen in Kommentaren
    r"#\s*noqa",  # Ruff-Unterdrueckungskommentare
    r"#\s*pylint:",  # pylint-Direktiven
    r"#\s*type:",  # type-Kommentare
    r"logger\.(debug|info|warning|error)\($",  # Logger-Aufruf ohne String
    r"exc_info=True",  # Logger-Parameter
    r"stack_info=True",  # Logger-Parameter
]

_AUTO_FIX_REPLACEMENTS: list[tuple[str, str]] = [
    ("failed to", "konnte nicht"),
    ("unable to", "konnte nicht"),
    ("trying to", "versuche"),
    ("non-blocking", "nicht blockierend"),
    ("non-critical", "unkritisch"),
    ("fallback", "Ersatzpfad"),
    ("calibrated", "kalibriert"),
    ("calibration", "Kalibrierung"),
    ("threshold", "Schwelle"),
    ("thresh", "Schwelle"),
    ("session", "Sitzung"),
    ("capture", "Erfassung"),
    ("record", "aufzeichnen"),
    ("failed", "fehlgeschlagen"),
    ("failure", "Fehlschlag"),
    ("skipped", "uebersprungen"),
    ("skip", "ueberspringen"),
    ("executed", "ausgefuehrt"),
    ("execute", "ausfuehren"),
    ("applied", "angewendet"),
    ("apply", "anwenden"),
    ("completed", "abgeschlossen"),
    ("complete", "vollstaendig"),
    ("started", "gestartet"),
    ("finished", "beendet"),
    ("update", "Aktualisierung"),
    ("updating", "aktualisiere"),
    ("recovery", "Wiederherstellung"),
    ("recover", "wiederherstellen"),
    ("loaded", "geladen"),
    ("loading", "lade"),
    ("load", "laden"),
    ("saved", "gespeichert"),
    ("save", "speichern"),
    ("saving", "speichere"),
    ("created", "erstellt"),
    ("create", "erstellen"),
    ("initialized", "initialisiert"),
    ("initialize", "initialisieren"),
    ("detected", "erkannt"),
    ("detect", "erkennen"),
    ("processed", "verarbeitet"),
    ("process", "verarbeiten"),
    ("generated", "erzeugt"),
    ("generate", "erzeugen"),
    ("configured", "konfiguriert"),
    ("configure", "konfigurieren"),
    ("enabled", "aktiviert"),
    ("disabled", "deaktiviert"),
    ("available", "verfuegbar"),
    ("unavailable", "nicht verfuegbar"),
    ("successful", "erfolgreich"),
    ("successfully", "erfolgreich"),
    ("attempt", "Versuch"),
    ("retry", "Wiederholung"),
    ("timeout", "Zeitlimit"),
    ("cached", "zwischengespeichert"),
    ("cache", "Zwischenspeicher"),
    ("cleared", "geleert"),
    ("clear", "leeren"),
    ("reset", "zurueckgesetzt"),
    ("aborted", "abgebrochen"),
    ("abort", "abbrechen"),
    ("terminated", "beendet"),
    ("terminate", "beenden"),
    ("shutdown", "Herunterfahren"),
    ("startup", "Start"),
    ("initializing", "initialisiere"),
    ("finalize", "abschliessen"),
    ("validate", "validieren"),
    ("validation", "Validierung"),
    ("verifying", "pruefe"),
    ("verify", "pruefen"),
    ("checking", "pruefe"),
    ("check", "Pruefung"),
    ("computing", "berechne"),
    ("compute", "berechnen"),
    ("analyzing", "analysiere"),
    ("analysis", "Analyse"),
    ("extracting", "extrahiere"),
    ("extract", "extrahieren"),
    ("importing", "importiere"),
    ("export", "Ausgabe"),
    ("reading", "lese"),
    ("writing", "schreibe"),
    ("fetching", "hole"),
    ("fetch", "holen"),
    ("sending", "sende"),
    ("send", "senden"),
    ("receiving", "empfange"),
    ("receive", "empfangen"),
    ("connecting", "verbinde"),
    ("connection", "Verbindung"),
    ("disconnect", "trennen"),
    ("pending", "ausstehend"),
    ("running", "laeuft"),
    ("run", "Ausfuehrung"),
    ("stopping", "stoppe"),
    ("stopped", "gestoppt"),
    ("restart", "Neustart"),
    ("stage", "Stufe"),
    ("phase", "Verarbeitungsschritt"),
    ("mode", "Betriebsart"),
    ("profile", "Profil"),
    ("profiling", "Messung"),
    ("budget", "Grenze"),
    ("ratio", "Verhaeltnis"),
    ("score", "Wert"),
    ("result", "Ergebnis"),
    ("output", "Ausgabe"),
    ("input", "Eingabe"),
    ("reference", "Referenz"),
    ("original", "Originalsignal"),
    ("restored", "wiederhergestellt"),
    ("restore", "wiederherstellen"),
    ("enhanced", "verbessert"),
    ("enhance", "verbessern"),
    ("optimized", "optimiert"),
    ("optimize", "optimieren"),
    ("adjusted", "angepasst"),
    ("adjust", "anpassen"),
    ("normalized", "normalisiert"),
    ("normalize", "normalisieren"),
]


def get_changed_files() -> list[Path]:
    """Ermittelt git-staged .py-Dateien."""
    files: list[Path] = []
    for cmd in [
        ["git", "-C", str(_PROJECT_ROOT), "diff", "--cached", "--name-only"],
        ["git", "-C", str(_PROJECT_ROOT), "diff", "--name-only"],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.endswith(".py") and not line.startswith(("tests/", "benchmarks/")):
                    fp = _PROJECT_ROOT / line
                    if fp.exists():
                        files.append(fp)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return files


def _is_exempt(line: str) -> bool:
    """Prüft ob eine Zeile von der Sprach-Prüfung ausgenommen ist."""
    for pat in _EXEMPT_PATTERNS:
        if re.search(pat, line):
            return True
    return False


def _contains_english_word(text: str) -> tuple[bool, str]:
    """Prüft ob ein Text englische Wörter enthält. Gibt (True, wort) zurück."""
    words = re.findall(r"[a-zA-ZäöüÄÖÜß]{3,}", text.lower())
    for word in words:
        if word in _GERMAN_TECH_TERMS:
            continue
        if word in _ENGLISH_FORBIDDEN:
            return True, word
    return False, ""


def _replace_forbidden_terms(text: str) -> str:
    """Ersetzt verbotene englische Log-Begriffe in sichtbarem Log-Text."""
    translated = text
    word_letters = "A-Za-zäöüÄÖÜß"
    for term, replacement in _AUTO_FIX_REPLACEMENTS:
        pattern = re.compile(
            rf"(^|\\[abfnrtv]|[^{word_letters}])({re.escape(term)})(?![{word_letters}])", re.IGNORECASE
        )
        translated = pattern.sub(lambda match: f"{match.group(1)}{replacement}", translated)
    return translated


def _replace_fstring_literals_only(text: str) -> str:
    """Ersetzt nur sichtbare f-String-Textteile, nie Ausdruecke in {...}."""
    chunks: list[str] = []
    literal: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""
        if depth == 0:
            if char == "{" and next_char == "{":
                literal.append("{{")
                i += 2
                continue
            if char == "}" and next_char == "}":
                literal.append("}}")
                i += 2
                continue
            if char == "{":
                chunks.append(_replace_forbidden_terms("".join(literal)))
                literal.clear()
                chunks.append(char)
                depth = 1
            else:
                literal.append(char)
        else:
            chunks.append(char)
            if char == "{" and next_char != "{":
                depth += 1
            elif char == "}" and next_char != "}":
                depth = max(0, depth - 1)
        i += 1
    chunks.append(_replace_forbidden_terms("".join(literal)))
    return "".join(chunks)


def _translate_string_token(token_text: str) -> str:
    """Uebersetzt ein Python-Stringliteral, Prefix/Quotes bleiben erhalten."""
    match = re.match(r"(?is)^([rubf]*)('''|\"\"\"|'|\")", token_text)
    if not match:
        return token_text
    prefix = match.group(1)
    quote = match.group(2)
    if not token_text.endswith(quote):
        return token_text
    body = token_text[match.end() : -len(quote)]
    if "f" in prefix.lower():
        body = _replace_fstring_literals_only(body)
    else:
        body = _replace_forbidden_terms(body)
    return f"{prefix}{quote}{body}{quote}"


def _translate_source_segment(segment: str) -> str:
    """Uebersetzt String-Tokens in einem AST-Source-Segment ohne Whitespace-Drift."""
    lines = segment.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def _offset(position: tuple[int, int]) -> int:
        line, col = position
        return line_offsets[line - 1] + col

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(segment).readline))
    except tokenize.TokenError:
        return segment

    parts: list[str] = []
    cursor = 0
    for token in tokens:
        if token.type == tokenize.ENDMARKER:
            continue
        start = _offset(token.start)
        end = _offset(token.end)
        parts.append(segment[cursor:start])
        token_text = token.string
        if token.type == tokenize.STRING:
            token_text = _translate_string_token(token_text)
        parts.append(token_text)
        cursor = end
    parts.append(segment[cursor:])
    return "".join(parts)


def _logger_message_nodes(tree: ast.AST) -> list[ast.AST]:
    """Liefert erste String-Argumente von logger.*-Aufrufen."""
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "logger":
            continue
        if node.func.attr not in ("debug", "info", "warning", "error"):
            continue
        first_arg = node.args[0]
        if (isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)) or isinstance(
            first_arg, ast.JoinedStr
        ):
            nodes.append(first_arg)
    return nodes


def fix_file(filepath: Path) -> int:
    """Uebersetzt englische logger.*-Meldungen in einer Datei."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return 0

    source_lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in source_lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def _offset(line: int, byte_col: int) -> int:
        line_text = source_lines[line - 1]
        char_col = len(line_text.encode("utf-8")[:byte_col].decode("utf-8"))
        return line_offsets[line - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for node in _logger_message_nodes(tree):
        if not hasattr(node, "end_lineno") or not hasattr(node, "end_col_offset"):
            continue
        start = _offset(node.lineno, node.col_offset)
        end = _offset(node.end_lineno, node.end_col_offset)  # type: ignore[arg-type]
        segment = source[start:end]
        translated = _translate_source_segment(segment)
        if translated != segment:
            replacements.append((start, end, translated))

    if not replacements:
        return 0

    updated = source
    for start, end, translated in sorted(replacements, reverse=True):
        updated = updated[:start] + translated + updated[end:]
    if updated != source:
        filepath.write_text(updated, encoding="utf-8")
    return len(replacements)


class LogLanguageVisitor(ast.NodeVisitor):
    """AST-Visitor der englische Log-Meldungen findet."""

    def __init__(self, source_lines: list[str]) -> None:
        self.source_lines = source_lines
        self.violations: list[tuple[int, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Prüft logger.info/warning/error/debug Aufrufe auf englische Meldungen."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "logger":
                if node.func.attr in ("debug", "info", "warning", "error"):
                    if node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                            msg = first_arg.value
                            line_no = node.lineno
                            source_line = self.source_lines[line_no - 1] if line_no <= len(self.source_lines) else ""
                            if _is_exempt(source_line):
                                self.generic_visit(node)
                                return
                            is_en, word = _contains_english_word(msg)
                            if is_en:
                                self.violations.append(
                                    (
                                        line_no,
                                        "EN",
                                        f'"{msg[:60]}..." → enthält "{word}"',
                                    )
                                )
                        # Auch f-Strings prüfen
                        elif isinstance(first_arg, ast.JoinedStr):
                            # Extrahiere Text aus f-String
                            parts = []
                            for val in first_arg.values:
                                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                                    parts.append(val.value)
                            msg = "".join(parts)
                            line_no = node.lineno
                            source_line = self.source_lines[line_no - 1] if line_no <= len(self.source_lines) else ""
                            if _is_exempt(source_line):
                                self.generic_visit(node)
                                return
                            is_en, word = _contains_english_word(msg)
                            if is_en:
                                self.violations.append(
                                    (
                                        line_no,
                                        "EN",
                                        f'f"...{msg[:40]}..." → enthält "{word}"',
                                    )
                                )
        self.generic_visit(node)


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Prüft eine Datei auf englische Log-Meldungen."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    visitor = LogLanguageVisitor(source.split("\n"))
    visitor.visit(tree)
    return visitor.violations


def main() -> int:
    """Haupteinstiegspunkt."""
    fix_mode = "--fix" in sys.argv
    all_mode = "--all" in sys.argv or fix_mode

    if all_mode:
        # Scan all Python files
        files: list[Path] = []
        for root, dirs, fns in os.walk(str(_PROJECT_ROOT / "backend")):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "tests", ".git")]
            for fn in fns:
                if fn.endswith(".py"):
                    files.append(Path(root) / fn)
        for root, dirs, fns in os.walk(str(_PROJECT_ROOT / "denker")):
            for fn in fns:
                if fn.endswith(".py"):
                    files.append(Path(root) / fn)
        for root, dirs, fns in os.walk(str(_PROJECT_ROOT / "Aurik10")):
            for fn in fns:
                if fn.endswith(".py"):
                    files.append(Path(root) / fn)
    else:
        files = get_changed_files()

    if not files:
        print("✅ Sprache-Guard: Keine .py-Dateien zum Prüfen")
        return 0

    if fix_mode:
        changed_files = 0
        changed_messages = 0
        for fp in sorted(set(files)):
            fixed = fix_file(fp)
            if fixed:
                changed_files += 1
                changed_messages += fixed
        print(f"🔧 Sprache-Guard --fix: {changed_messages} Log-Meldungen in {changed_files} Dateien uebersetzt")

    total = 0
    new_violations = 0
    for fp in sorted(set(files)):
        violations = check_file(fp)
        if violations:
            rel = fp.relative_to(_PROJECT_ROOT)
            shown = False
            for line, rule, desc in violations:
                total += 1
                # Prüfe Whitelist: bestehende Verletzungen werden nicht blockiert
                vkey = f"{rel}:{line}"
                vhash = hashlib.sha256(vkey.encode()).hexdigest()[:16]
                if vhash in _WHITELIST:
                    continue
                if not shown:
                    print(f"\n─── {rel} ───")
                    shown = True
                violation_line = f"  L{line}: [{rule}] {desc}"
                print(violation_line)
                new_violations += 1

    print(f"\n{'=' * 60}")
    print(f"Geprüft: {len(set(files))} Dateien, {total} englische Log-Meldungen ({new_violations} neu)")

    if new_violations == 0:
        if total > 0:
            print(f"✅ Alle {total} Meldungen sind in der Whitelist — keine neuen EN-Meldungen")
        else:
            print("✅ Alle Log-Meldungen auf Deutsch")
        return 0
    else:
        print(f"❌ {new_violations} NEUE englische Log-Meldungen — Commit blockiert")
        print("   → Alle neuen Log-Meldungen MÜSSEN auf Deutsch sein")
        print("   → logger.info('Processing...')  →  logger.info('Verarbeite...')")
        print("   → Bestehende Meldungen wurden in .language_guard_whitelist.txt erfasst")
        return 1


if __name__ == "__main__":
    sys.exit(main())
