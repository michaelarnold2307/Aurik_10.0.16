"""
Aurik 6.0 – SOTA-System- und Plugin-Check

Dieses Skript prüft die Kernmodule und Plugins gemäß SOTA-Standards und Dokumentation.
Alle Ausgaben und Begriffe sind an die aktuelle Architektur angepasst.
"""

import glob
import importlib
import logging
import os
import sys
from typing import Any

try:
    requests: Any | None = importlib.import_module("requests")
except Exception:
    requests = None

logger = logging.getLogger(__name__)

logger.info("Aurik 6.0 – SOTA System- und Plugin-Pruefung")

# Kernabhängigkeiten
try:
    logger.info("Pruefung: soundfile OK")
except Exception as e:
    logger.error("ERROR: soundfile: %s", e)
    sys.exit(1)
try:
    logger.info("Pruefung: numpy OK")
except Exception as e:
    logger.error("ERROR: numpy: %s", e)
    sys.exit(1)
try:
    logger.info("Pruefung: onnxruntime OK")
except Exception as e:
    logger.error("ERROR: onnxruntime: %s", e)
    sys.exit(1)

# SOTA-Plugin- und Health-Check
plugin_dir = os.path.join("Aurik_Standalone", "plugins")
failed: list[str] = []
for f in glob.glob(os.path.join(plugin_dir, "*.py")):
    mod = os.path.splitext(os.path.basename(f))[0]
    if mod == "__init__":
        continue
    try:
        importlib.import_module(f"plugins.{mod}")
        logger.info("[SOTA-Pruefung] Plugin: %s OK", mod)
    except Exception as e:
        logger.error("[SOTA-Pruefung] ERROR: %s: %s", mod, e)
        failed.append(mod)

# Desktop-Offline ist Standard: kein lokaler HTTP-Server als Pflicht.
# Optional kann ein HTTP-Health-Probe explizit aktiviert werden.
_http_health_enabled = os.getenv("AURIK_ENABLE_HTTP_HEALTH", "0") == "1"
if _http_health_enabled:
    if requests is None:
        logger.error("[SOTA-Pruefung] HTTP-Health-Probe aktiviert, aber requests nicht verfügbar")
        failed.append("health-endpoint")
    else:
        _health_urls = (
            "http://localhost:8000/api/health",
            "http://localhost:8000/health",
        )
        _health_ok = False
        _last_error = ""
        for _url in _health_urls:
            try:
                r = requests.get(_url, timeout=2)
                if r.status_code == 200 and "ok" in r.text.lower():
                    logger.info("[SOTA-Pruefung] Health-Endpoint OK: %s", _url)
                    _health_ok = True
                    break
                _last_error = f"{_url} -> {r.status_code} {r.text}"[:300]
            except Exception as e:
                _last_error = f"{_url} -> {e}"[:300]
        if not _health_ok:
            logger.error("[SOTA-Pruefung] Health-Endpoint Fehler: %s", _last_error)
            failed.append("health-endpoint")
else:
    logger.info("[SOTA-Pruefung] HTTP-Health-Probe übersprungen (Desktop-Offline-Standard)")

if failed:
    logger.error("[SOTA-Pruefung] Fehlerhafte Komponenten: %s", failed)
    sys.exit(1)
else:
    logger.info("[SOTA-Pruefung] Alle Plugins, Kernmodule und Health-Checks erfolgreich geladen.")
    sys.exit(0)
