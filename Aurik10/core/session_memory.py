"""Session-Gedächtnis — Frühere Ergebnisse & Fenster-Position. Spec v10.206 §6.

Speichert:
- Fenster-Position/Größe (restoreGeometry/saveGeometry)
- Frühere Restaurierungsergebnisse (Datei-Pfad → Qualität, Datum)
- Letzte Einstellungen (Modus, Material, Export-Format)
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

SESSION_PATH = Path.home() / ".aurik" / "session_memory.json"
MAX_HISTORY_ENTRIES: int = 50


class SessionMemory:
    """Singleton: Session-Gedächtnis."""

    _instance: SessionMemory | None = None

    def __new__(cls) -> SessionMemory:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        self._history: list[dict[str, Any]] = []
        self._window_geometry: bytes | None = None
        self._last_mode: str = "restoration"
        self._last_material: str = "unknown"
        self._last_export_format: str = "flac"
        if SESSION_PATH.is_file():
            try:
                with open(SESSION_PATH) as f:
                    data = json.load(f)
                self._history = data.get("history", [])
                geo_b64 = data.get("window_geometry")
                if geo_b64:
                    import base64
                    self._window_geometry = base64.b64decode(geo_b64)
                self._last_mode = data.get("last_mode", "restoration")
                self._last_material = data.get("last_material", "unknown")
                self._last_export_format = data.get("last_export_format", "flac")
            except Exception:
                pass

    def _save(self) -> None:
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        import base64
        data = {
            "history": self._history[-MAX_HISTORY_ENTRIES:],
            "window_geometry": base64.b64encode(self._window_geometry).decode() if self._window_geometry else None,
            "last_mode": self._last_mode,
            "last_material": self._last_material,
            "last_export_format": self._last_export_format,
        }
        with open(SESSION_PATH, "w") as f:
            json.dump(data, f, indent=2)

    # ── History ──────────────────────────────────────────────────────────

    def add_result(self, file_path: str, quality: float, mode: str, duration_s: float) -> None:
        self._history.append({
            "file": file_path,
            "quality": round(quality, 1),
            "mode": mode,
            "duration_s": round(duration_s, 1),
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        })
        self._save()

    def get_history(self, file_path: str | None = None) -> list[dict[str, Any]]:
        if file_path:
            return [h for h in self._history if h["file"] == file_path]
        return list(self._history)

    def get_last_result(self, file_path: str) -> dict[str, Any] | None:
        matches = self.get_history(file_path)
        return matches[-1] if matches else None

    def get_recommendation(self, file_path: str) -> str | None:
        """Gibt Empfehlung auf Basis eigener History."""
        last = self.get_last_result(file_path)
        if not last:
            return None
        q = last["quality"]
        if q >= 80:
            return f"Letztes Mal: {q}% Qualität. Gleicher Modus empfohlen."
        elif q >= 60:
            return f"Letztes Mal: {q}% Qualität. Studio-2026-Modus könnte besser sein."
        else:
            return f"Letztes Mal: {q}% Qualität. Manuelle Parameter-Anpassung empfohlen."

    # ── Fenster ──────────────────────────────────────────────────────────

    def save_window_geometry(self, geometry_bytes: bytes) -> None:
        self._window_geometry = geometry_bytes
        self._save()

    def get_window_geometry(self) -> bytes | None:
        return self._window_geometry

    # ── Einstellungen ────────────────────────────────────────────────────

    def save_last_settings(self, mode: str, material: str, export_format: str) -> None:
        self._last_mode = mode
        self._last_material = material
        self._last_export_format = export_format
        self._save()

    def get_last_settings(self) -> dict[str, str]:
        return {
            "mode": self._last_mode,
            "material": self._last_material,
            "export_format": self._last_export_format,
        }


def get_session_memory() -> SessionMemory:
    return SessionMemory()
