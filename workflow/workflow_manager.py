"""
Workflow Integration Module (GAP #26-28)

Provides integrated workflow management for AURIK:
- Batch Processing API (GAP #26)
- Undo/Redo System (GAP #27)
- Session Management Integration (GAP #28)

This module brings together existing components into a unified workflow API
that can be used by CLI, GUI, and API interfaces.

Author: AURIK Team
Version: 1.0.0
Date: 9. Februar 2026
"""

import logging
import shutil
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ===== GAP #26: Batch Processing API =====


@dataclass
class BatchJobConfig:
    """Configuration for a batch processing job."""

    input_files: list[Path]
    output_dir: Path
    processing_mode: str = "balanced"
    medium_type: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    parallel: bool = True
    max_workers: int | None = None
    on_progress: Callable[[int, int, str], None] | None = None
    on_file_complete: Callable[[Path, Path, bool, str], None] | None = None
    skip_existing: bool = False


@dataclass
class BatchJobResult:
    """Result from a batch processing job."""

    total_files: int
    success_count: int
    failed_count: int
    skipped_count: int
    total_time: float
    results: list[dict[str, Any]]

    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)."""
        if self.total_files == 0:
            return 0.0
        return self.success_count / self.total_files

    def summary(self) -> str:
        """Generiert human-readable summary."""
        return (
            f"Batch Complete: {self.success_count}/{self.total_files} succeeded, "
            f"{self.failed_count} failed, {self.skipped_count} skipped "
            f"({self.total_time:.1f}s total)"
        )


class BatchProcessor:
    """
    High-level batch processing API (GAP #26).

    Provides unified interface for processing multiple files in batch mode.
    Features:
    - Parallel or sequential processing
    - Progress tracking and callbacks
    - Error handling and recovery
    - Skip existing files option
    - Comprehensive result reporting
    """

    def __init__(self, restorer_factory: Callable | None = None):
        """
        Parameters:
        -----------
        restorer_factory : callable, optional
            Factory function to create restorer instances.
            If None, creates UnifiedRestorerV3 instances.
        """
        self.restorer_factory = restorer_factory or self._default_restorer_factory
        self.current_job: BatchJobConfig | None = None

    def _default_restorer_factory(self):
        """Default restorer factory — liefert AurikDenker-Singleton (§2.2 normativer Einstiegspunkt).

        Direkter UV3-Zugriff ist verboten (§RELEASE_MUST). Stets über get_aurik_denker().
        """
        try:
            from denker.aurik_denker import get_aurik_denker  # type: ignore[import]

            return get_aurik_denker()
        except Exception as e:
            logger.error("AurikDenker nicht verfügbar: %s", e)
            raise

    def process_batch(self, config: BatchJobConfig) -> BatchJobResult:
        """
        Verarbeitet a batch of audio files.

        Parameters:
        -----------
        config : BatchJobConfig
            Batch job configuration

        Returns:
        --------
        result : BatchJobResult
            Comprehensive results from batch processing
        """
        self.current_job = config
        start_time = datetime.now()

        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize counters
        success_count = 0
        failed_count = 0
        skipped_count = 0
        results = []

        total_files = len(config.input_files)

        if config.parallel and total_files > 1:
            # Parallel processing (joblib)
            results = self._process_parallel(config)
        else:
            # Sequential processing
            results = self._process_sequential(config)

        # Count results
        for result in results:
            if result["skipped"]:
                skipped_count += 1
            elif result["success"]:
                success_count += 1
            else:
                failed_count += 1

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        return BatchJobResult(
            total_files=total_files,
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            total_time=total_time,
            results=results,
        )

    def _process_sequential(self, config: BatchJobConfig) -> list[dict]:
        """Verarbeitet files sequentially."""
        results = []

        for idx, input_file in enumerate(config.input_files, 1):
            # Progress callback
            if config.on_progress:
                config.on_progress(idx, len(config.input_files), input_file.name)

            result = self._process_single_file(input_file, config)
            results.append(result)

            # File complete callback
            if config.on_file_complete:
                config.on_file_complete(
                    input_file,
                    Path(result["output_path"]) if result["output_path"] else None,  # type: ignore[arg-type]
                    result["success"],
                    result.get("error", ""),
                )

        return results

    def _process_parallel(self, config: BatchJobConfig) -> list[dict]:
        """Verarbeitet files in parallel."""
        try:
            from joblib import Parallel, delayed

            n_jobs = config.max_workers or -1

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._process_single_file)(input_file, config) for input_file in config.input_files
            )

            # Progress callback (after completion)
            if config.on_progress:
                config.on_progress(len(config.input_files), len(config.input_files), "Complete")

            return results  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Parallel processing fehlgeschlagen: {e}")
            # Fallback to sequential
            logger.info("Falling back to sequential processing")
            return self._process_sequential(config)

    def _process_single_file(self, input_file: Path, config: BatchJobConfig) -> dict:
        """Verarbeitet a single file."""
        result = {
            "input_path": str(input_file),
            "output_path": None,
            "success": False,
            "skipped": False,
            "error": None,
            "processing_time": 0.0,
        }

        try:
            # Generate output path
            output_filename = f"{input_file.stem}_restored{input_file.suffix}"
            output_path = config.output_dir / output_filename

            # Check if should skip
            if config.skip_existing and output_path.exists():
                result["skipped"] = True
                result["output_path"] = str(output_path)
                return result

            # Create restorer instance
            start = datetime.now()
            restorer = self.restorer_factory()

            # Load audio
            import soundfile as sf

            audio, sr = sf.read(input_file, always_2d=False)

            # §2.2: AurikDenker.denke() ist der normative Einstiegspunkt.
            # Custom factories mit .process()-API werden als Legacy-Pfad toleriert.
            if hasattr(restorer, "denke"):
                _res = restorer.denke(
                    audio,
                    sr,
                    mode=config.processing_mode or "restoration",
                )
                processed = _res.audio
            else:
                # Legacy-Pfad für custom restorer_factory (kein UV3-Default mehr)
                processed = restorer.process(
                    audio,
                    sr,
                    mode=config.processing_mode,
                    medium_type=config.medium_type,
                )

            # Save output
            sf.write(output_path, processed, sr)

            end = datetime.now()
            result["processing_time"] = (end - start).total_seconds()
            result["output_path"] = str(output_path)
            result["success"] = True

        except Exception as e:
            logger.error(f"konnte nicht verarbeiten {input_file}: {e}")
            result["error"] = str(e)
            result["success"] = False

        return result


# ===== GAP #27: Undo/Redo System =====


@dataclass
class ProcessingState:
    """Represents a processing state that can be undone/redone."""

    timestamp: datetime
    input_path: Path
    output_path: Path
    audio_backup: Path | None  # Temporary backup file
    settings: dict[str, Any]
    description: str

    def cleanup(self):
        """Clean up temporary backup file."""
        if self.audio_backup and self.audio_backup.exists():
            try:
                self.audio_backup.unlink()
            except Exception as e:
                logger.warning(f"konnte nicht cleanup backup: {e}")


class UndoRedoManager:
    """
    Undo/Redo system for audio processing (GAP #27).

    Provides undo/redo functionality by maintaining a history stack
    of processing states with backup files.

    Features:
    - Multi-level undo/redo (configurable depth)
    - Automatic backup file management
    - Memory-efficient (only keeps backups, not full audio in memory)
    - State description for UI display
    """

    def __init__(self, max_history: int = 10, backup_dir: Path | None = None):
        """
        Parameters:
        -----------
        max_history : int
            Maximum number of undo steps to keep
        backup_dir : Path, optional
            Directory for temporary backup files
        """
        self.max_history = max_history
        self.backup_dir = backup_dir or Path(".aurik_undo_cache")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.undo_stack: list[ProcessingState] = []
        self.redo_stack: list[ProcessingState] = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_state(self, input_path: Path, output_path: Path, settings: dict, description: str = "Processing"):
        """
        Speichert current state for undo.

        Creates a backup of the file before processing.
        """
        # Create backup
        backup_filename = f"{self.current_session_id}_{len(self.undo_stack)}_backup{output_path.suffix}"
        backup_path = self.backup_dir / backup_filename

        try:
            # Copy current output as backup (if exists)
            if output_path.exists():
                shutil.copy2(output_path, backup_path)
            else:
                backup_path = None  # type: ignore[assignment]
        except Exception as e:
            logger.warning(f"konnte nicht erstellen backup: {e}")
            backup_path = None  # type: ignore[assignment]

        # Create state
        state = ProcessingState(
            timestamp=datetime.now(),
            input_path=input_path,
            output_path=output_path,
            audio_backup=backup_path,
            settings=deepcopy(settings),
            description=description,
        )

        # Add to undo stack
        self.undo_stack.append(state)

        # Clear redo stack (new action invalidates redo history)
        self._clear_redo_stack()

        # Trim undo stack if needed
        if len(self.undo_stack) > self.max_history:
            old_state = self.undo_stack.pop(0)
            old_state.cleanup()

    def can_undo(self) -> bool:
        """Prüft if undo is available."""
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """Prüft if redo is available."""
        return len(self.redo_stack) > 0

    def undo(self) -> ProcessingState | None:
        """
        Undo last operation.

        Returns:
        --------
        state : ProcessingState or None
            The state that was undone, or None if nothing to undo
        """
        if not self.can_undo():
            return None

        # Pop from undo stack
        state = self.undo_stack.pop()

        # Save current state to redo stack
        self.redo_stack.append(state)

        # Restore backup if exists
        if state.audio_backup and state.audio_backup.exists():
            try:
                shutil.copy2(state.audio_backup, state.output_path)
                logger.info(f"wiederhergestellt backup: {state.output_path}")
            except Exception as e:
                logger.error(f"konnte nicht wiederherstellen backup: {e}")

        return state

    def redo(self) -> ProcessingState | None:
        """
        Redo previously undone operation.

        Returns:
        --------
        state : ProcessingState or None
            The state that was redone, or None if nothing to redo
        """
        if not self.can_redo():
            return None

        # Pop from redo stack
        state = self.redo_stack.pop()

        # Add back to undo stack
        self.undo_stack.append(state)

        # Re-apply processing (would need to re-run restorer)
        # For now, just mark state as current
        logger.info(f"Redo: {state.description}")

        return state

    def get_undo_history(self) -> list[str]:
        """Gibt zurück: list of undo descriptions."""
        return [state.description for state in reversed(self.undo_stack)]

    def get_redo_history(self) -> list[str]:
        """Gibt zurück: list of redo descriptions."""
        return [state.description for state in self.redo_stack]

    def _clear_redo_stack(self):
        """Löscht redo stack and cleanup backups."""
        for state in self.redo_stack:
            state.cleanup()
        self.redo_stack.clear()

    def cleanup_all(self):
        """Clean up all backup files."""
        for state in self.undo_stack + self.redo_stack:
            state.cleanup()

        # Clear stacks
        self.undo_stack.clear()
        self.redo_stack.clear()

        # Remove backup directory if empty
        try:
            if self.backup_dir.exists() and not any(self.backup_dir.iterdir()):
                self.backup_dir.rmdir()
        except Exception:
            logger.warning("workflow_manager.py::cleanup_all Ersatzpfad", exc_info=True)


# ===== GAP #28: Session Management Integration =====


class WorkflowSessionManager:
    """
    Session management integration for unified workflow (GAP #28).

    Wraps backend session manager and provides workflow-specific features:
    - Auto-save on file processing
    - Session recovery
    - Integration with undo/redo
    - Quick session switching
    """

    def __init__(self, sessions_dir: Path | None = None):
        """
        Parameters:
        -----------
        sessions_dir : Path, optional
            Directory for storing session files
        """
        # Import backend session manager
        try:
            from backend.core.session.session_manager import SessionManager

            self.backend_manager = SessionManager(sessions_dir=sessions_dir)
        except ImportError:
            logger.warning("Backend SessionManager not verfuegbar, using stub")
            self.backend_manager = None  # type: ignore[assignment]
        except Exception as e:
            logger.warning(f"konnte nicht initialisieren backend SessionManager: {e}, using stub")
            self.backend_manager = None  # type: ignore[assignment]

        self.current_session_id: str | None = None
        self.auto_save = True

    def create_session(self, name: str, description: str = "") -> str:
        """
        Erstellt new session.

        Returns:
        --------
        session_id : str
            ID of created session
        """
        if self.backend_manager:
            session = self.backend_manager.create_session(name, description)
            # backend may return a plain session_id string or an object with .session_id
            if isinstance(session, str):
                session_id = session
            else:
                session_id = session.session_id  # type: ignore[assignment]
            self.current_session_id = session_id
            return session_id
        return "stub_session"

    def load_session(self, session_id: str) -> bool:
        """
        Lädt existing session.

        Returns:
        --------
        success : bool
            True if session loaded successfully
        """
        if self.backend_manager:
            session = self.backend_manager.load_session(session_id)
            if session:
                self.current_session_id = session_id
                return True
        return False

    def add_processed_file(
        self, input_path: Path, output_path: Path, settings: dict, success: bool = True, error: str | None = None
    ):
        """Fügt hinzu: processed file to current session."""
        if self.backend_manager and self.current_session_id:
            try:
                from backend.core.session.session_manager import ProcessedFile

                processed_file = ProcessedFile(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    processing_mode=settings.get("mode", "unknown"),
                    processing_settings=settings,
                    success=success,
                    error_message=error,
                )

                self.backend_manager.add_to_session(processed_file)

                if self.auto_save:
                    self.backend_manager.save_session()

            except Exception as e:
                logger.error(f"konnte nicht add file to Sitzung: {e}")

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Gibt zurück: list of recent sessions."""
        if self.backend_manager:
            return self.backend_manager.get_recent_sessions(n=limit)
        return []

    def export_session(self, session_id: str, export_path: Path) -> bool:
        """Export session to file."""
        if self.backend_manager:
            return self.backend_manager.export_session(session_id, export_path)  # type: ignore[no-any-return]
        return False

    def import_session(self, import_path: Path) -> str | None:
        """
        Import session from file.

        Returns:
        --------
        session_id : str or None
            ID of imported session, or None if failed
        """
        if self.backend_manager:
            session = self.backend_manager.import_session(import_path)
            return session.session_id
        return None


# ===== Unified Workflow API =====


class WorkflowManager:
    """
    Einheitlicher Workflow-Manager mit allen Workflow-Funktionen.

    Provides single entry point for:
    - Batch processing (GAP #26)
    - Undo/Redo (GAP #27)
    - Session management (GAP #28)

    Example Usage:
    --------------
    ```python
    workflow = WorkflowManager()

    #Create session
    workflow.create_session("My Restoration Project")

    # Process batch with undo support
    config = BatchJobConfig(
        input_files=[Path("audio1.wav"), Path("audio2.wav")],
        output_dir=Path("output/"),
        processing_mode="balanced"
    )
    result = workflow.process_batch(config, enable_undo=True)

    # Undo if needed
    if not result.success_rate() > 0.8:
        workflow.undo()

    # Export session
    workflow.export_current_session(Path("my_project.aurik"))
    ```
    """

    def __init__(self, sessions_dir: Path | None = None, backup_dir: Path | None = None):
        """
        Parameters:
        -----------
        sessions_dir : Path, optional
            Directory for session files
        backup_dir : Path, optional
            Directory for undo backups
        """
        self.batch_processor = BatchProcessor()
        self.undo_manager = UndoRedoManager(backup_dir=backup_dir)
        self.session_manager = WorkflowSessionManager(sessions_dir=sessions_dir)

    # Session methods
    def create_session(self, name: str, description: str = "") -> str:
        """Erstellt new processing session."""
        return self.session_manager.create_session(name, description)

    def load_session(self, session_id: str) -> bool:
        """Lädt existing session."""
        return self.session_manager.load_session(session_id)

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Gibt zurück: recent sessions."""
        return self.session_manager.get_recent_sessions(limit)

    def export_current_session(self, export_path: Path) -> bool:
        """Export current session."""
        if self.session_manager.current_session_id:
            return self.session_manager.export_session(self.session_manager.current_session_id, export_path)
        return False

    # Batch processing methods
    def process_batch(self, config: BatchJobConfig, enable_undo: bool = True) -> BatchJobResult:
        """
        Verarbeitet batch of files with optional undo support.

        Parameters:
        -----------
        config : BatchJobConfig
            Batch processing configuration
        enable_undo : bool
            Whether to enable undo for this batch

        Returns:
        --------
        result : BatchJobResult
            Batch processing results
        """
        # Wrap callbacks to integrate with session/undo
        original_on_complete = config.on_file_complete

        def integrated_callback(input_path, output_path, success, error):
            # Add to session
            if output_path:
                self.session_manager.add_processed_file(input_path, output_path, config.settings, success, error)

            # Save undo state
            if enable_undo and success and output_path:
                self.undo_manager.save_state(input_path, output_path, config.settings, f"Processed {input_path.name}")

            # Call original callback
            if original_on_complete:
                original_on_complete(input_path, output_path, success, error)

        config.on_file_complete = integrated_callback

        # Process batch
        result = self.batch_processor.process_batch(config)

        return result

    # Undo/Redo methods
    def undo(self) -> bool:
        """Undo last operation."""
        state = self.undo_manager.undo()
        return state is not None

    def redo(self) -> bool:
        """Redo previously undone operation."""
        state = self.undo_manager.redo()
        return state is not None

    def can_undo(self) -> bool:
        """Prüft if undo is available."""
        return self.undo_manager.can_undo()

    def can_redo(self) -> bool:
        """Prüft if redo is available."""
        return self.undo_manager.can_redo()

    def get_undo_history(self) -> list[str]:
        """Gibt zurück: undo history."""
        return self.undo_manager.get_undo_history()

    def get_redo_history(self) -> list[str]:
        """Gibt zurück: redo history."""
        return self.undo_manager.get_redo_history()

    # Cleanup
    def cleanup(self):
        """Clean up temporary files."""
        self.undo_manager.cleanup_all()


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AURIK Workflow Manager")
    parser.add_argument("command", choices=["batch", "undo", "redo", "sessions"], help="Command to execute")
    parser.add_argument("--input", "-i", nargs="+", help="Input files")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--mode", default="balanced", help="Processing mode")
    parser.add_argument("--session", help="Session name")

    args = parser.parse_args()

    workflow = WorkflowManager()

    if args.command == "batch":
        if not args.input or not args.output:
            import logging

            logging.error("Error: --input and --output required for batch processing")
            sys.exit(1)

        config = BatchJobConfig(
            input_files=[Path(f) for f in args.input], output_dir=Path(args.output), processing_mode=args.mode
        )

        logging.info(f"Processing {len(config.input_files)} files...")
        result = workflow.process_batch(config)
        logging.info(result.summary())

    elif args.command == "undo":
        if workflow.undo():
            logging.info("✓ Undo successful")
        else:
            logging.warning("❌ Nothing to undo")

    elif args.command == "redo":
        if workflow.redo():
            logging.info("✓ Redo successful")
        else:
            logging.warning("❌ Nothing to redo")

    elif args.command == "sessions":
        sessions = workflow.get_recent_sessions()
        logging.info(f"\nRecent sessions ({len(sessions)}):")
        for s in sessions:
            logging.info(f"  - {s.get('session_name', 'Unnamed')}: {s.get('file_count', 0)} files")
