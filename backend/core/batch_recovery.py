"""
Batch-Recovery — Wiederherstellung nach Absturz bei Stapelverarbeitung.

Erweitert crash_recovery_guard.py um Batch-Koordination:
- Trackt welche Dateien in einem Batch bereits verarbeitet wurden
- Ermöglicht Resume nach Absturz
- Atomare Checkpoints pro Datei + Batch-Manifest

Integration: Wird vom BatchProcessor in der GUI/CLI genutzt.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BatchRecoveryManager:
    """Batch-weite Crash-Recovery."""

    def __init__(self, batch_id: str, checkpoint_dir: str = "~/.aurik/checkpoints"):
        self._batch_id = batch_id
        self._dir = Path(checkpoint_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._manifest_path = self._dir / f"{batch_id}_manifest.json"
        self._manifest: dict[str, Any] = self._load_manifest()

    # ── Manifest ──────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("batch_Wiederherstellung.py::_laden_manifest Ersatzpfad: %s", e)
        return {
            "batch_id": self._batch_id,
            "created": time.time(),
            "total_files": 0,
            "completed": [],
            "failed": [],
            "in_progress": None,
            "checkpoint_files": {},
        }

    def _save_manifest(self):
        with self._lock:
            self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    # ── File Tracking ─────────────────────────────────────────────────────

    def start_file(self, file_path: str):
        """Markiert eine Datei als 'in Bearbeitung'."""
        with self._lock:
            self._manifest["in_progress"] = file_path
            if file_path not in self._manifest["completed"]:
                self._manifest["total_files"] = max(
                    self._manifest["total_files"],
                    len(self._manifest["completed"]) + len(self._manifest["failed"]) + 1,
                )
            self._save_manifest()

    def complete_file(self, file_path: str):
        """Markiert eine Datei als erfolgreich abgeschlossen."""
        with self._lock:
            if file_path not in self._manifest["completed"]:
                self._manifest["completed"].append(file_path)
            self._manifest["in_progress"] = None
            self._save_manifest()

    def fail_file(self, file_path: str, error: str):
        """Markiert eine Datei als fehlgeschlagen."""
        with self._lock:
            self._manifest["failed"].append(
                {
                    "file": file_path,
                    "error": error,
                    "time": time.time(),
                }
            )
            self._manifest["in_progress"] = None
            self._save_manifest()

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save_checkpoint(self, file_path: str, audio: np.ndarray, metadata: dict):
        """Speichert Checkpoint für aktuelle Datei (atomar)."""
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(file_path).stem)
        cp_path = self._dir / f"{self._batch_id}_{safe_name}.npz"

        tmp = cp_path.with_suffix(".tmp")
        try:
            np.savez_compressed(tmp, audio=audio)
            tmp.replace(cp_path)
            meta_path = cp_path.with_suffix(".json")
            meta_path.write_text(json.dumps(metadata))
            with self._lock:
                self._manifest["checkpoint_files"][file_path] = str(cp_path)
            self._save_manifest()
        except Exception as e:
            logger.warning("Checkpoint fehlgeschlagen for %s: %s", file_path, e)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def has_checkpoint(self, file_path: str) -> bool:
        with self._lock:
            return file_path in self._manifest.get("checkpoint_files", {})

    def load_checkpoint(self, file_path: str) -> tuple[np.ndarray, dict] | None:
        with self._lock:
            cp = self._manifest.get("checkpoint_files", {}).get(file_path)
        if cp and Path(cp).exists():
            try:
                data = np.load(cp)
                audio = data["audio"]
                meta_path = Path(cp).with_suffix(".json")
                metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                return audio, metadata
            except Exception as e:
                logger.warning("laden checkpoint fehlgeschlagen: %s", e)
        return None

    # ── Recovery ──────────────────────────────────────────────────────────

    def get_pending_files(self, all_files: list[str]) -> list[str]:
        """Gibt Dateien zurück, die noch verarbeitet werden müssen."""
        with self._lock:
            completed = set(self._manifest.get("completed", []))
            failed = {f["file"] for f in self._manifest.get("failed", [])}
            done = completed | failed
            return [f for f in all_files if f not in done]

    def get_incomplete_file(self) -> str | None:
        """Gibt die Datei zurück, die beim letzten Abbruch in Bearbeitung war."""
        with self._lock:
            return self._manifest.get("in_progress")

    def cleanup(self):
        """Löscht Checkpoint-Dateien nach erfolgreichem Batch."""
        with self._lock:
            for cp_path in self._manifest.get("checkpoint_files", {}).values():
                try:
                    Path(cp_path).unlink(missing_ok=True)
                    Path(cp_path).with_suffix(".json").unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("batch_Wiederherstellung.py::cleanup Ersatzpfad: %s", e)
            self._manifest_path.unlink(missing_ok=True)

    def get_progress(self) -> dict:
        """Gibt Fortschrittsinfo zurück."""
        with self._lock:
            return {
                "batch_id": self._batch_id,
                "total": self._manifest["total_files"],
                "completed": len(self._manifest["completed"]),
                "failed": len(self._manifest["failed"]),
                "in_progress": self._manifest["in_progress"],
                "pending": max(
                    0, self._manifest["total_files"] - len(self._manifest["completed"]) - len(self._manifest["failed"])
                ),
            }
