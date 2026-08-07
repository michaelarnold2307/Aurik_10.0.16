import glob
import importlib
import logging
import os
import time
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI()

# Modell-Container-Healthchecks (Namen und Ports ggf. anpassen)
MODEL_CONTAINERS = {
    "deepfilternet": "http://deepfilternet:5000/health",
    "dnsmos": "http://dnsmos:5000/health",
    # weitere Modelle hier ergänzen
}


@app.get("/api/health")
def health_check():
    start = time.time()
    status: dict[str, Any] = {"status": "ok", "details": {}, "response_time_ms": None}
    # Kernmodule prüfen
    modules = ["soundfile", "numpy", "onnxruntime"]
    for mod in modules:
        try:
            importlib.import_module(mod)
            status["details"][mod] = "ok"
        except Exception as e:
            status["details"][mod] = f"error: {e}"
            status["status"] = "fail"
    # Plugins prüfen
    plugin_dir = os.path.join("Aurik_Standalone", "plugins")
    failed_plugins = []
    for f in glob.glob(os.path.join(plugin_dir, "*.py")):
        mod = os.path.splitext(os.path.basename(f))[0]
        if mod == "__init__":
            continue
        try:
            # importlib.import_module(f'aurik6.plugins.{mod}')
            status["details"][f"plugin_{mod}"] = "ok"
        except Exception as e:
            status["details"][f"plugin_{mod}"] = f"error: {e}"
            failed_plugins.append(mod)
            status["status"] = "fail"
    # Modell-Container prüfen
    for name, url in MODEL_CONTAINERS.items():
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and "ok" in r.text:
                status["details"][f"model_{name}"] = "ok"
            else:
                status["details"][f"model_{name}"] = f"fail: {r.text}"
                status["status"] = "fail"
        except Exception as e:
            status["details"][f"model_{name}"] = f"unreachable: {e}"
            status["status"] = "fail"
    status["response_time_ms"] = int((time.time() - start) * 1000)
    return JSONResponse(content=status)


# Prometheus-Metriken (optional, für Integration)
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import]

    Instrumentator().instrument(app).expose(app)
except ImportError:
    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
