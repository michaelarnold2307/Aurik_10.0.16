"""backend/core/crash_recovery_guard.py — Generic Crash Recovery Guard.

Provides `CrashRecoveryGuard`, a context-manager-based wrapper that
automatically checkpoints any processing function on failure (MemoryError,
SIGTERM, or any unhandled exception).  Complements the existing
`backend/core/recovery_checkpoint.py` OOM-Checkpoint-System (§2.39) by adding
a general-purpose guard that can wrap arbitrary pipeline stages, not just the
top-level unified restorer.

Differences from recovery_checkpoint.py:
  - recovery_checkpoint.py saves pipeline state inside unified_restorer_v3
    when OOM is detected mid-pipeline (phase list, analysis caches, audio).
  - crash_recovery_guard.py wraps *any* processing callable with automatic
    checkpoint-on-exception, SIGTERM-aware shutdown, and a startup recovery
    prompt.  It is designed to be usable from the GUI batch thread, CLI
    entry-points, and programmatic API.

Checkpoint lifecycle:
  1. guarded(context) wraps a processing block.
  2. On exception → _save_checkpoint() writes atomic JSON + compressed NPZ audio.
  3. On next startup → find_pending_checkpoints() discovers unfinished jobs.
  4. User confirms → load_checkpoint() returns saved state for resume.
  5. Successful completion → clear_checkpoint() removes temporary files.

Files on disk per checkpoint:
  ~/.aurik/checkpoints/<job_id>_crash_checkpoint.json  — metadata + traceback
  ~/.aurik/checkpoints/<job_id>_crash_audio.npz         — audio snapshot (np.savez_compressed)

Thread-safe: atomic writes (tmp + os.replace), re-entrant SIGTERM handling.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT_DIR: str = "~/.aurik/checkpoints"
_MAX_CHECKPOINT_AGE_S: float = 7 * 24 * 3600.0  # 7 days


# ---------------------------------------------------------------------------
# CrashRecoveryGuard
# ---------------------------------------------------------------------------


class CrashRecoveryGuard:
    """Wraps any processing function with automatic checkpoint + recovery.

    Usage (context manager)::

        guard = CrashRecoveryGuard()
        try:
            with guard.guarded(job_id, audio, metadata):
                result = process_audio(audio)
            guard.clear_checkpoint(job_id)
        except Exception as e:
            logger.warning("crash_recovery aborted: %s", e)
            raise

    Usage (recovery check at startup)::

        guard = CrashRecoveryGuard()
        pending = guard.find_pending_checkpoints()
        for cp in pending:
            print(f"Unfinished job: {cp['job_id']}")
            # Ask user whether to resume or discard
    """

    def __init__(self, checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR) -> None:
        self._dir = Path(checkpoint_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        # Prevent re-entrancy during SIGTERM
        self._shutdown_in_progress: bool = False
        # Track active guarded jobs for emergency checkpointing
        self._active_jobs: dict[str, dict[str, Any]] = {}
        # Install SIGTERM handler if not already installed by main.py
        self._install_sigterm_handler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def guarded(self, job_id: str, audio: np.ndarray, metadata: dict[str, Any]) -> Iterator[None]:
        """Context manager that checkpoints on exception.

        Args:
            job_id: Unique identifier for this processing job (e.g. filename stem).
            audio: The audio data being processed (for checkpoint snapshot).
            metadata: Arbitrary job metadata (input path, mode, phase list, etc.).

        Yields:
            None — the caller's block executes inside the context.

        Raises:
            Any exception after saving a checkpoint — the original exception
            is always re-raised so the caller can decide how to handle it.
        """
        self._active_jobs[job_id] = {"audio": audio, "metadata": dict(metadata)}
        try:
            yield
        except MemoryError:
            logger.critical("CrashRecoveryGuard: MemoryError in job %s — saving checkpoint", job_id)
            self._save_checkpoint(job_id, audio, metadata, "MemoryError", traceback.format_exc())
            raise
        except Exception as exc:
            logger.critical(
                "CrashRecoveryGuard: Exception in job %s (%s: %s) — saving checkpoint",
                job_id,
                type(exc).__name__,
                exc,
            )
            self._save_checkpoint(job_id, audio, metadata, f"{type(exc).__name__}: {exc}", traceback.format_exc())
            raise
        finally:
            self._active_jobs.pop(job_id, None)

    def has_checkpoint(self, job_id: str) -> bool:
        """Check if a checkpoint exists for a job.

        Returns True if both JSON metadata and audio NPZ exist and are readable.
        """
        json_path = self._checkpoint_json_path(job_id)
        audio_path = self._checkpoint_audio_path(job_id)
        return json_path.is_file() and audio_path.is_file()

    def load_checkpoint(self, job_id: str) -> dict[str, Any]:
        """Load checkpoint data (metadata + audio) for a job.

        Returns a dict with keys:
          - ``job_id`` (str)
          - ``audio`` (np.ndarray)
          - ``metadata`` (dict)
          - ``failure_reason`` (str)
          - ``traceback`` (str)
          - ``timestamp`` (float)

        Raises FileNotFoundError if the checkpoint does not exist.
        """
        json_path = self._checkpoint_json_path(job_id)
        audio_path = self._checkpoint_audio_path(job_id)

        if not json_path.is_file():
            raise FileNotFoundError(f"No checkpoint JSON for job {job_id}: {json_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"No checkpoint audio for job {job_id}: {audio_path}")

        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        audio = np.load(audio_path)["audio"]

        return {
            "job_id": job_id,
            "audio": audio,
            "metadata": meta.get("metadata", {}),
            "failure_reason": meta.get("failure_reason", "unknown"),
            "traceback": meta.get("traceback", ""),
            "timestamp": meta.get("timestamp", 0.0),
        }

    def clear_checkpoint(self, job_id: str) -> None:
        """Remove checkpoint files after successful completion."""
        json_path = self._checkpoint_json_path(job_id)
        audio_path = self._checkpoint_audio_path(job_id)
        for path in (json_path, audio_path):
            try:
                os.remove(path)
            except OSError as _crg1_exc:
                logger.debug(
                    "crash_Wiederherstellung_guard: checkpoint file removal fehlgeschlagen (unkritisch): %s", _crg1_exc
                )

    def find_pending_checkpoints(self) -> list[dict[str, Any]]:
        """Discover all valid crash checkpoints awaiting recovery.

        Filters out expired (> 7 days) and orphaned checkpoints.

        Returns a list of dicts with ``job_id``, ``metadata``, ``failure_reason``,
        ``timestamp`` keys suitable for displaying a recovery prompt.
        """
        results: list[dict[str, Any]] = []
        now = time.time()

        for json_file in self._dir.glob("*_crash_checkpoint.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                ts = data.get("timestamp", 0.0)
                if now - ts > _MAX_CHECKPOINT_AGE_S:
                    logger.debug("Checkpoint too old (%.0f days): %s", (now - ts) / 86400, json_file)
                    self._cleanup_checkpoint_files(json_file)
                    continue

                audio_path = data.get("audio_npz_path", "")
                if not os.path.isfile(audio_path):
                    logger.debug("Audio NPZ missing for checkpoint: %s", audio_path)
                    self._cleanup_checkpoint_files(json_file)
                    continue

                job_id = json_file.stem.replace("_crash_checkpoint", "")
                results.append(
                    {
                        "job_id": job_id,
                        "metadata": data.get("metadata", {}),
                        "failure_reason": data.get("failure_reason", "unknown"),
                        "timestamp": ts,
                        "checkpoint_path": str(json_file),
                    }
                )
            except Exception as exc:
                logger.debug("Invalid checkpoint %s: %s", json_file, exc)
                self._cleanup_checkpoint_files(json_file)

        return results

    def emergency_checkpoint_all(self) -> int:
        """Best-effort checkpoint all active jobs — called on SIGTERM.

        Returns the number of jobs successfully checkpointed.
        """
        saved = 0
        for job_id, state in list(self._active_jobs.items()):
            try:
                self._save_checkpoint(
                    job_id,
                    state["audio"],
                    state["metadata"],
                    "SIGTERM (emergency shutdown)",
                    "",
                )
                saved += 1
            except Exception as exc:
                logger.error("Emergency checkpoint fehlgeschlagen for job %s: %s", job_id, exc)
        return saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint_json_path(self, job_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)[:120]
        return self._dir / f"{safe}_crash_checkpoint.json"

    def _checkpoint_audio_path(self, job_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)[:120]
        return self._dir / f"{safe}_crash_audio.npz"

    def _save_checkpoint(
        self,
        job_id: str,
        audio: np.ndarray,
        metadata: dict[str, Any],
        failure_reason: str,
        tb: str,
    ) -> None:
        """Persist checkpoint atomically to disk.

        Uses tmp-file + os.replace for atomic writes, safe under concurrent
        processes and SIGTERM interruption.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)

            json_path = self._checkpoint_json_path(job_id)
            audio_path = self._checkpoint_audio_path(job_id)

            # 1. Write audio NPZ (atomic)
            audio_tmp = str(audio_path) + ".tmp"
            np.savez_compressed(audio_tmp, audio=audio)
            os.replace(audio_tmp, str(audio_path))

            # 2. Write JSON metadata (atomic)
            checkpoint_data: dict[str, Any] = {
                "job_id": job_id,
                "metadata": metadata,
                "failure_reason": failure_reason,
                "traceback": tb,
                "timestamp": time.time(),
                "audio_npz_path": str(audio_path),
            }
            json_tmp = str(json_path) + ".tmp"
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            os.replace(json_tmp, str(json_path))

            logger.info(
                "CrashRecoveryGuard: Checkpoint gespeichert for job %s (reason: %s)",
                job_id,
                failure_reason,
            )
        except Exception as exc:
            logger.error("CrashRecoveryGuard: konnte nicht speichern checkpoint for %s: %s", job_id, exc)

    def _cleanup_checkpoint_files(self, json_path: Path) -> None:
        """Remove checkpoint JSON and associated audio NPZ + tmp files."""
        stem = str(json_path).replace("_crash_checkpoint.json", "")
        for suffix in (
            "_crash_checkpoint.json",
            "_crash_audio.npz",
            "_crash_checkpoint.json.tmp",
            "_crash_audio.npz.tmp",
        ):
            try:
                os.remove(stem + suffix)
            except OSError as _crg2_exc:
                logger.debug(
                    "crash_Wiederherstellung_guard: suffix file removal fehlgeschlagen (unkritisch): %s", _crg2_exc
                )

    # ------------------------------------------------------------------
    # SIGTERM integration (§3.9.2 enhancement)
    # ------------------------------------------------------------------

    def _install_sigterm_handler(self) -> None:
        """Install a SIGTERM handler that emergency-checkpoints active jobs.

        This is designed to coexist with Aurik10/main.py's existing SIGTERM
        handler.  If main.py already installed one, this wraps it; otherwise
        it installs a standalone handler.

        The handler is re-entrant safe: _shutdown_in_progress prevents
        double-execution.
        """
        _existing = signal.getsignal(signal.SIGTERM)

        def _sigterm_wrapper(signum: int, frame: object) -> None:  # pylint: disable=unused-argument
            if self._shutdown_in_progress:
                return
            self._shutdown_in_progress = True
            logger.warning(
                "CrashRecoveryGuard: SIGTERM received — emergency checkpointing %d active jobs",
                len(self._active_jobs),
            )
            saved = self.emergency_checkpoint_all()
            logger.info(
                "CrashRecoveryGuard: %d/%d jobs checkpointed on SIGTERM", saved, len(self._active_jobs) if saved else 0
            )
            # Chain to the existing handler if it's callable and not the
            # default SIG_DFL/SIG_IGN
            if callable(_existing) and _existing not in (signal.SIG_DFL, signal.SIG_IGN):
                try:
                    _existing(signum, frame)  # type: ignore[arg-type]
                except Exception as e:
                    logger.warning("crash_Wiederherstellung_guard.py::_sigterm_wrapper Ersatzpfad: %s", e)
            else:
                # No existing handler — re-raise the original signal to let
                # the OS default behaviour (process termination) take over.
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        try:
            signal.signal(signal.SIGTERM, _sigterm_wrapper)
        except OSError:
            # May happen in non-main threads; that's fine — the
            # main.py handler is sufficient.
            logger.debug("CrashRecoveryGuard: SIGTERM handler install uebersprungen (non-main thread)")


# ---------------------------------------------------------------------------
# Module-level convenience: global singleton
# ---------------------------------------------------------------------------

_global_guard: CrashRecoveryGuard | None = None


def get_global_guard() -> CrashRecoveryGuard:
    """Return (or create) a process-wide CrashRecoveryGuard singleton.

    This is the preferred entry point for Aurik batch threads and CLI tools
    that want a single checkpoint directory shared across all jobs.
    """
    global _global_guard
    if _global_guard is None:
        _global_guard = CrashRecoveryGuard()
    return _global_guard


# ---------------------------------------------------------------------------
# Recovery prompt helper (GUI-aware)
# ---------------------------------------------------------------------------


def recovery_prompt_for_gui(guard: CrashRecoveryGuard | None = None) -> list[dict[str, Any]]:
    """Generate a list of pending recovery items suitable for a GUI dialog.

    Each item is a dict with keys the GUI can render directly:
      - ``job_id`` (str)
      - ``failure_reason`` (str)  — human-readable, e.g. "MemoryError" / "SIGTERM"
      - ``timestamp_iso`` (str)  — ISO 8601 timestamp
      - ``age_hours`` (float)    — how old the checkpoint is

    Call this at startup.  The GUI can then offer "Resume" / "Discard" per item.
    """
    g = guard if guard is not None else get_global_guard()
    pending = g.find_pending_checkpoints()
    now = time.time()
    result: list[dict[str, Any]] = []
    for cp in pending:
        age_hours = (now - cp["timestamp"]) / 3600.0
        result.append(
            {
                "job_id": cp["job_id"],
                "failure_reason": cp["failure_reason"],
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(cp["timestamp"])),
                "age_hours": round(age_hours, 1),
                "checkpoint_path": cp.get("checkpoint_path", ""),
            }
        )
    return result
