#!/usr/bin/env python3
"""Plugin-Validator — Prüft Plugin-Konformität mit Aurik SDK.

§15.6: Validiert Plugin-Verzeichnisse gegen das SDK-Schema.

Nutzung:
    python scripts/validate_plugin.py plugins/sdk/example_plugin
    python scripts/validate_plugin.py plugins/mein_plugin --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def validate_plugin_dir(plugin_dir: Path, strict: bool = False) -> tuple[bool, list[str]]:
    """Validiert ein Plugin-Verzeichnis.

    Returns:
        (ok, messages) — True wenn alle Checks bestanden, plus Meldungen.
    """
    messages: list[str] = []
    checks_passed = 0

    # 1. manifest.json existiert und ist valide
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.exists():
        messages.append("❌ manifest.json fehlt")
        if strict:
            return False, messages
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            required = ["name", "version", "description"]
            missing = [k for k in required if k not in manifest]
            if missing:
                messages.append(f"❌ manifest.json: Fehlende Felder: {missing}")
            else:
                messages.append(f"✅ manifest.json: {manifest['name']} v{manifest['version']}")
                checks_passed += 1
        except json.JSONDecodeError as e:
            messages.append(f"❌ manifest.json: JSON-Fehler: {e}")
            if strict:
                return False, messages

    # 2. Plugin-Python-Datei existiert
    py_files = list(plugin_dir.glob("*.py"))
    plugin_py = [f for f in py_files if f.stem not in ("__init__", "test_")]
    if not plugin_py:
        messages.append("❌ Keine Plugin-Implementierung (.py) gefunden")
    else:
        messages.append(f"✅ Plugin-Datei: {plugin_py[0].name}")
        checks_passed += 1

    # 3. __init__.py existiert
    init_path = plugin_dir / "__init__.py"
    if not init_path.exists():
        messages.append("⚠️  __init__.py fehlt (empfohlen)")
    else:
        messages.append("✅ __init__.py vorhanden")
        checks_passed += 1

    # 4. Test-Datei existiert
    test_files = list(plugin_dir.glob("test_*.py"))
    if not test_files:
        messages.append("⚠️  Keine Tests (test_*.py) gefunden")
    else:
        messages.append(f"✅ Tests: {len(test_files)} Datei(en)")
        checks_passed += 1

    # 5. README.md existiert
    readme_path = plugin_dir / "README.md"
    if not readme_path.exists():
        messages.append("⚠️  README.md fehlt")
    else:
        content = readme_path.read_text(encoding="utf-8")
        if len(content) < 50:
            messages.append("⚠️  README.md zu kurz (<50 Zeichen)")
        else:
            messages.append("✅ README.md vorhanden")
            checks_passed += 1

    ok = checks_passed >= 3 if not strict else checks_passed >= 5
    return ok, messages


def validate_all_plugins(plugins_dir: Path | None = None) -> tuple[int, int]:
    """Validiert alle Plugin-Verzeichnisse unter `plugins/` (Aurik10/ui/plugin_manager.py).

    Returns:
        (total, failed) — Anzahl geprüfter Plugins und Anzahl davon fehlgeschlagen.
    """
    base_dir = plugins_dir or (_PROJECT_ROOT / "plugins")
    if not base_dir.is_dir():
        return 0, 0

    total = 0
    failed = 0
    for item in sorted(base_dir.iterdir()):
        if not item.is_dir() or item.name.startswith("_") or item.name.startswith(".") or item.name == "sdk":
            continue
        if not (item / "manifest.json").exists():
            continue
        total += 1
        ok, _messages = validate_plugin_dir(item, strict=False)
        if not ok:
            failed += 1
    return total, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Aurik Plugin-Validator")
    parser.add_argument("plugin_dir", type=Path, help="Pfad zum Plugin-Verzeichnis")
    parser.add_argument("--strict", action="store_true", help="Alle Checks müssen bestehen")
    args = parser.parse_args()

    plugin_dir = args.plugin_dir
    if not plugin_dir.is_dir():
        print(f"❌ Kein Verzeichnis: {plugin_dir}")
        return 1

    ok, messages = validate_plugin_dir(plugin_dir, strict=args.strict)
    for msg in messages:
        print(msg)

    if ok:
        print(f"\n✅ Plugin validiert: {plugin_dir.name}")
        return 0
    else:
        print(f"\n❌ Plugin-Validierung fehlgeschlagen: {plugin_dir.name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
