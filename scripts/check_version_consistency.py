#!/usr/bin/env python3
"""scripts/check_version_consistency.py — §v10.700 F1.

from typing import Any
Prüft, ob alle Dokumente die kanonische Versionsnummer aus
backend/core/version.py verwenden. Findet veraltete Referenzen.

Exit 0 = alle konsistent, Exit 1 = Inkonsistenzen gefunden.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


def get_canonical_version() -> str:
    """Liest die kanonische Version aus backend/core/version.py."""
    version_file = Path("backend/core/version.py")
    if not version_file.exists():
        print("❌ backend/core/version.py nicht gefunden")
        sys.exit(1)

    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        print("❌ __version__ nicht in backend/core/version.py gefunden")
        sys.exit(1)
    return match.group(1)


def find_md_files() -> list[Path]:
    """Findet alle .md-Dateien im Projekt."""
    root = Path(".")
    excluded = {
        ".venv_aurik",
        "__pycache__",
        ".git",
        "build",
        "dist",
        "models",
        "output_audio",
        "sessions",
        "logs",
        "node_modules",
    }
    md_files = []
    for path in root.rglob("*.md"):
        if any(ex in path.parts for ex in excluded):
            continue
        md_files.append(path)
    return sorted(md_files)


def check_file(path: Path, version: str) -> list[str]:
    """Prüft eine Datei auf veraltete Versionen. Gibt Warnungen zurück."""
    warnings: list[Any] = []  # type: ignore[name-defined]
    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError):
        return warnings

    # Alte Versionen, die nicht die kanonische sind
    old_patterns = [
        (r"\b10\.0\.[0-9]+\b", "10.0.x"),
        (r"\b10\.[1-9]\.[0-9]+\b", "10.x.y"),
        (r"\b9\.\d+\.\d+\b", "9.x.y"),
        (r"\b8\.\d+\.\d+\b", "8.x.y"),
    ]

    for pattern, label in old_patterns:
        matches = re.findall(pattern, content)
        for m in matches:
            if m != version:
                warnings.append(f"{path}: {label} → Version '{m}' (erwartet {version})")
                break  # Nur ein Match pro Pattern pro Datei

    return warnings


def main() -> int:
    version = get_canonical_version()
    print(f"Kanonische Version: {version}")
    print()

    md_files = find_md_files()
    print(f"Prüfe {len(md_files)} Markdown-Dateien...")
    print()

    all_warnings = []
    for path in md_files:
        warnings = check_file(path, version)
        all_warnings.extend(warnings)

    if all_warnings:
        print(f"⚠️  {len(all_warnings)} Inkonsistenzen gefunden:")
        for w in all_warnings:
            print(f"  {w}")
        return 1
    else:
        print(f"✅ Alle {len(md_files)} Dateien konsistent mit Version {version}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
