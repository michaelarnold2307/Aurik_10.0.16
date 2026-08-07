#!/usr/bin/env python3
"""scripts/audit_bridge_coverage.py — §v10.700.2 Bridge-Coverage-Audit.

Prüft dass Frontend/CLI das Backend NUR via bridge.py importiert.
Direkte Core-Imports werden als Violations gemeldet.

Nutzung:
  python scripts/audit_bridge_coverage.py          # Audit
  python scripts/audit_bridge_coverage.py --ci      # CI-Mode: Exit 1 bei Violations
  python scripts/audit_bridge_coverage.py --json    # JSON-Ausgabe
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

FRONTEND_DIRS = ["Aurik10", "cli"]
ALLOWED_DIRECT_IMPORTS = {
    # Startup-pflicht: vor Bridge-Initialisierung
    "Aurik10/main.py": ["backend.core.ml_device_manager"],
    # UI-spezifische Business-Logik (kein DSP-Core, nur UI-Helfer)
    "Aurik10/ui/modern_window.py": ["backend.core.donation_reminder"],
    # CLI-spezifisch: kein GUI-Kontext
    "cli/aurik_debug.py": [
        "backend.core.unified_restorer_v3",
        "backend.core.pipeline_trace",
    ],
    "cli/aurik_cli.py": ["backend.core.cd_noise_profile"],
}


def find_violations() -> list[dict]:
    """Findet alle direkten Core-Imports in Frontend-Dateien."""
    violations = []
    pattern = re.compile(r"from (backend\.core\.\S+) import")

    for frontend_dir in FRONTEND_DIRS:
        for py_file in (REPO_ROOT / frontend_dir).rglob("*.py"):
            rel = str(py_file.relative_to(REPO_ROOT))
            try:
                content = py_file.read_text()
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue

            for match in pattern.finditer(content):
                import_path = match.group(1)
                # Skip bridge.py selbst
                if "bridge" in import_path:
                    continue

                # Check allowed
                allowed = ALLOWED_DIRECT_IMPORTS.get(rel, [])
                if import_path in allowed:
                    continue

                violations.append(
                    {
                        "file": rel,
                        "import": import_path,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

    return violations


def main() -> int:
    ci_mode = "--ci" in sys.argv
    json_mode = "--json" in sys.argv

    violations = find_violations()

    if json_mode:
        print(
            json.dumps(
                {
                    "violations": len(violations),
                    "details": violations,
                    "clean": len(violations) == 0,
                },
                indent=2,
            )
        )
        return 0

    print("🌉 Bridge-Coverage-Audit")
    print(f"   Frontend-Dirs: {', '.join(FRONTEND_DIRS)}")
    print()

    if violations:
        print(f"⚠️  {len(violations)} direkte Core-Imports gefunden:")
        for v in violations:
            print(f"  {v['file']}:{v['line']} → {v['import']}")
        print()
        print("Empfehlung: Route via backend.api.bridge oder füge zu ALLOWED_DIRECT_IMPORTS hinzu.")
    else:
        print("✅ Keine unerlaubten direkten Core-Imports — Bridge ist Clean.")

    return 1 if (ci_mode and violations) else 0


if __name__ == "__main__":
    sys.exit(main())
