#!/usr/bin/env python3
"""Aurik SOTA Bug-Prevention Hook (§v10.105)

Fängt die 6 Bug-Klassen aus der Exception-Forensik (Juli 2026) PROAKTIV ab,
BEVOR sie in die Pipeline gelangen.

Gefundene Anti-Patterns (aus 460 analysierten Exceptions):
  P1: shape[0] <= shape[1] — falsche Kanal-Detection (→ Broadcast-Crash)
  P2: filtfilt( ohne Längen-Guard     (→ padlen-Crash)
  P3: stft( ohne noverlap-Clamp       (→ noverlap-Crash)
  P4: os.* ohne import os            (→ UnboundLocalError)
  P5: np.asarray(Tuple) in __post_init__ (→ inhomogeneous-Crash)
  P6: MaterialType-Enum als String    (→ KeyError in Dict-Lookups)
"""

import ast
import os
import re
import sys
from pathlib import Path

# ── Konfiguration ──────────────────────────────────────────────────────────

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".venv", ".venv_aurik", "build", "dist",
    "models", "output_audio", "sessions", "logs", "data",
    "golden_samples", "chain_templates", "configs", ".eggs",
    "tests",  # Tests dürfen Anti-Patterns für negative Tests enthalten
}

EXCLUDE_FILES = {
    "setup.py", "conftest.py",
    # Fixer-Scripts beschreiben Anti-Patterns (nicht nutzen sie)
    "fix_p6_material_lookups.py",
    "fix_p6_v2.py",
}

MIN_SEVERITY = "warning"  # "error" stoppt Commit, "warning" warnt nur

# §v10.115: Continuous Analysis — Scanner lädt neue Patterns aus Exception-Forensik
_PATTERN_FEED_PATH = Path(__file__).resolve().parents[3] / "logs" / "discovered_patterns.json"

# ── P1: shape[0] <= shape[1] Anti-Pattern ─────────────────────────────────

def check_shape_anti_pattern(filepath: str, source: str) -> list[str]:
    """Findet `audio.shape[0] <= audio.shape[1]` ohne `shape[1] > 2`-Check."""
    issues = []
    # Regex: shape[0] <= shape[1] aber NICHT gefolgt von "and shape[1] > 2" auf gleicher Zeile
    pattern = re.compile(r'\.shape\[0\]\s*<=\s*\.shape\[1\]')
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            # Erlaubt wenn "shape[1] > 2" oder "shape[0] <= 2 and" auf gleicher Zeile
            if 'shape[0] <= 2 and' in line or 'shape[1] > 2' in line:
                continue
            # Erlaubt wenn in Kommentar
            if line.strip().startswith('#'):
                continue
            issues.append(
                f"{filepath}:{i}: P1 shape[0]<=shape[1] ohne channels-first-Guard "
                f"(→ Broadcast-Crash bei channels-last mit N≤2). "
                f"FIX: `shape[0] <= 2 and shape[1] > 2`"
            )
    return issues


# ── P2: filtfilt ohne Längen-Guard ────────────────────────────────────────

def check_filtfilt_without_guard(filepath: str, source: str) -> list[str]:
    """Findet bare `filtfilt(` oder `signal.filtfilt(` (nicht safe_filtfilt)."""
    issues = []
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        # Bare filtfilt( calls (nicht safe_filtfilt, nicht sosfiltfilt)
        if re.search(r'(?<!safe_)(?<!sos)(?<!_)(?<!\.)\bfiltfilt\(', stripped):
            # Skip spec_constitution.py — filtfilt inside ForbiddenPattern strings
            if 'spec_constitution.py' in filepath:
                continue
            # Skip files that DEFINE safe_filtfilt (audio_utils.py)
            if 'def safe_filtfilt' in source:
                continue
            # Prüfe ob safe_filtfilt importiert oder im File definiert ist
            if 'from backend.core.audio_utils import safe_filtfilt' not in source and 'safe_filtfilt' not in source:
                issues.append(
                    f"{filepath}:{i}: P2 filtfilt() ohne Längen-Guard "
                    f"(→ padlen-Crash bei kurzem Audio). "
                    f"FIX: `from backend.core.audio_utils import safe_filtfilt` + Ersetzung"
                )
    return issues


# ── P3: stft ohne noverlap-Clamp ──────────────────────────────────────────

def check_stft_without_clamp(filepath: str, source: str) -> list[str]:
    """Findet `stft(` mit `noverlap=n_fft - hop` ohne min(n_fft-1)-Clamp."""
    issues = []
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if 'stft(' in stripped and 'noverlap=' in stripped:
            # Prüfe ob ein min(..., nperseg-1) Clamp existiert
            if 'max(0,' not in source.split('\n')[max(0,i-3):i+1].__str__():
                if 'min(' not in stripped:
                    # Dynamic noverlap ohne Clamp
                    if 'noverlap=n_fft - hop' in stripped or 'noverlap=nperseg - hop' in stripped:
                        issues.append(
                            f"{filepath}:{i}: P3 stft() noverlap ohne min(nperseg-1)-Clamp "
                            f"(→ noverlap-Crash bei kurzem Audio). "
                            f"FIX: `_noverlap = min(n_fft - hop, max(0, n_fft - 1))`"
                        )
    return issues


# ── P4: os.* ohne import os ───────────────────────────────────────────────

def check_os_without_import(filepath: str, source: str) -> list[str]:
    """Findet `os.`-Nutzung ohne `import os` auf Module-Ebene."""
    issues = []
    if 'os.' not in source:
        return issues
    # Prüfe ob import os existiert (nicht in Funktionen, sondern auf Module-Ebene)
    has_module_import = bool(re.search(r'^(import os|from os import)', source, re.MULTILINE))
    if not has_module_import:
        # Prüfe ob os.* in einer Funktion verwendet wird (wo import fehlen könnte)
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == 'os':
                        issues.append(
                            f"{filepath}:{node.lineno}: P4 os.{node.attr} ohne `import os` "
                            f"(→ UnboundLocalError in bestimmten Umgebungen). "
                            f"FIX: `import os` am Modul-Anfang"
                        )
        except SyntaxError:
            pass
    return issues


# ── P5: np.asarray(Tuple) in PhaseResult.__post_init__ ─────────────────────

def check_asarray_tuple(filepath: str, source: str) -> list[str]:
    """Findet `np.asarray(self.audio)` ohne Tuple-Check in __post_init__."""
    issues = []
    if 'def __post_init__' not in source:
        return issues
    if 'np.asarray(self.audio' not in source and 'np.asarray(self.audio' not in source:
        return issues
    # Prüfe ob Tuple-Check VOR asarray existiert
    if 'isinstance(self.audio, (tuple, list))' not in source:
        issues.append(
            f"{filepath}: P5 np.asarray(self.audio) ohne Tuple→ndarray-Guard "
            f"(→ inhomogeneous-Crash bei Tuple-Rückgaben). "
            f"FIX: isinstance-Check vor np.asarray()"
        )
    return issues


# ── P6: MaterialType-Enum als String in Dict-Lookup ─────────────────────────

def check_enum_as_dict_key(filepath: str, source: str) -> list[str]:
    """Findet Dict-Lookups mit material/mat wo Keys MaterialType-Enums sind."""
    issues = []
    # Nur in Dateien die MaterialType importieren
    if 'MaterialType' not in source:
        return issues
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        # Skip docstring/formula lines (inside triple-quoted strings)
        if '·' in stripped or 'log10' in stripped:
            continue
        # Pattern: DICT[material] oder DICT.get(material) wo MaterialType-Enum-Keys
        if re.search(r'\[material\]', stripped) or re.search(r'\.get\(material[,\)]', stripped):
            # Prüfe ob Normalisierung existiert
            context_start = max(0, i - 3)
            context = '\n'.join(lines[context_start:i])
            if 'isinstance(material, MaterialType)' not in context and \
               'hasattr(material, "value")' not in context and \
               '_mat_enum_' not in context and \
               '.get(_mat_' not in context:
                issues.append(
                    f"{filepath}:{i}: P6 Dict-Lookup [material] ohne Enum-Normalisierung "
                    f"(→ KeyError wenn material String statt Enum). "
                    f"FIX: isinstance(material, MaterialType) + .get()-Fallback"
                )
    return issues


# ── §v10.115 Continuous Analysis: Dynamische Pattern-Erkennung ────────────────

def _load_discovered_patterns() -> list[str]:
    """Lädt vom Pattern-Miner entdeckte Patterns aus logs/discovered_patterns.json."""
    import json
    if not _PATTERN_FEED_PATH.exists():
        return []
    try:
        with open(_PATTERN_FEED_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    issues = []
    for pattern in data.get("patterns", []):
        if pattern.get("status") != "active":
            continue
        regex = pattern.get("regex")
        if not regex:
            continue
        message = pattern.get("message", "P7 Dynamisch entdecktes Anti-Pattern")
        for root in data.get("scan_roots", ["backend/core"]):
            repo_root = _PATTERN_FEED_PATH.parents[1]
            scan_dir = repo_root / root
            if not scan_dir.exists():
                continue
            for dirpath, _dirnames, filenames in os.walk(scan_dir):
                for fn in filenames:
                    if not fn.endswith('.py'):
                        continue
                    fp = os.path.join(dirpath, fn)
                    try:
                        with open(fp, encoding='utf-8') as fh:
                            src = fh.read()
                    except (UnicodeDecodeError, IsADirectoryError):
                        continue
                    for i, line in enumerate(src.split('\n'), 1):
                        s = line.strip()
                        if s.startswith('#'):
                            continue
                        if re.search(regex, s):
                            issues.append(f"{fp}:{i}: {message}")
    return issues


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    """Scannt alle Python-Dateien auf bekannte Bug-Patterns.

    §v10.114: Scanner auf alle Layer ausgeweitet (backend/core, plugins, 
    Aurik10, denker, scripts).
    """
    root = Path(__file__).resolve().parents[3]  # .agents/skills/bug-prevention/ → repo root

    SCAN_ROOTS = [
        root / "backend" / "core",
        root / "plugins",
        root / "Aurik10",
        root / "denker",
        root / "scripts",
    ]

    all_issues: list[str] = []
    files_scanned = 0

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(scan_root):
            # Filtere Verzeichnisse
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for filename in filenames:
                if not filename.endswith('.py'):
                    continue
                if filename in EXCLUDE_FILES:
                    continue

                filepath = os.path.join(dirpath, filename)
                files_scanned += 1

                try:
                    with open(filepath, encoding='utf-8') as f:
                        source = f.read()
                except (UnicodeDecodeError, IsADirectoryError):
                    continue

                # Alle 6 Checks
                all_issues.extend(check_shape_anti_pattern(filepath, source))
                all_issues.extend(check_filtfilt_without_guard(filepath, source))
                all_issues.extend(check_stft_without_clamp(filepath, source))
                all_issues.extend(check_os_without_import(filepath, source))
                all_issues.extend(check_asarray_tuple(filepath, source))
                all_issues.extend(check_enum_as_dict_key(filepath, source))
    
    # §v10.115: Lade dynamisch entdeckte Patterns aus Exception-Forensik
    discovered = _load_discovered_patterns()
    all_issues.extend(discovered)

    # Ausgabe
    if all_issues:
        print(f"\n🔍 Aurik SOTA Bug-Scan: {len(all_issues)} potentielle Bugs gefunden "
              f"({files_scanned} Dateien gescannt)\n")
        for issue in sorted(all_issues):
            print(f"  {issue}")
        
        if MIN_SEVERITY == "error":
            print(f"\n❌ {len(all_issues)} Fehler — Commit blockiert.")
            print("   Behebe die oben genannten Anti-Patterns oder füge "
                  "   begründete Ausnahmen in EXCLUDE_FILES hinzu.")
            return 1
        else:
            print(f"\n⚠️  {len(all_issues)} Warnungen — bitte vor Commit prüfen.")
            return 0
    else:
        print(f"✅ Aurik SOTA Bug-Scan: Keine Anti-Patterns gefunden "
              f"({files_scanned} Dateien gescannt)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
