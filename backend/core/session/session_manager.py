"""
Session Management System

Component 5.1: Session Management
Impact: +1.0 Punkt - Laien können Arbeit wiederfinden und fortsetzen

Provides comprehensive session management for user workflow:
- Session save/load with all processing settings
- Recent sessions tracking (last 10)
- Session history (last 100 processed files)
- Favorites/bookmarks for quick access
- Processing settings persistence

Problem:
Laien verlieren ihre Arbeit zwischen Sessions:
- "Wo ist das File was ich gestern gemacht habe?"
- Keine History → Keine Wiederfindbarkeit
- Settings müssen jedes Mal neu eingegeben werden

Solution:
SessionManager verwaltet alle Sessions, History und Settings:
- Sessions als JSON gespeichert
- Automatic history tracking
- Recent sessions list
- Session export/import
- Settings templates

Author: AI Team
Date: 8. Februar 2026
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status types."""

    ACTIVE = "active"  # Currently being worked on
    COMPLETED = "completed"  # All files processed successfully
    IN_PROGRESS = "in_progress"  # Partially processed
    FAILED = "failed"  # Processing errors occurred


@dataclass
class ProcessedFile:
    """
    Record of a single processed file.

    Attributes:
        input_path: Original audio file path
        output_path: Restored audio file path
        processing_mode: Mode used (e.g., "restoration", "studio_2026")
        processing_settings: All processing parameters
        musical_goals_scores: Achieved Musical Goals scores
        timestamp: When file was processed
        processing_time: Time taken (seconds)
        success: Whether processing succeeded
        error_message: Error message if failed
    """

    input_path: str
    output_path: str
    processing_mode: str
    processing_settings: dict[str, Any] = field(default_factory=dict)
    musical_goals_scores: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time: float = 0.0
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Konvertiert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessedFile":
        """Erstellt from dictionary.

        Handles legacy JSON formats written by earlier Aurik versions where
        ``_recent.json`` / ``_history.json`` entries were stored as plain
        lists or with extra/missing keys. Unknown keys are silently dropped;
        missing optional keys get their dataclass defaults.
        """
        if not isinstance(data, dict):
            raise TypeError(f"ProcessedFile.from_dict: expected dict, got {type(data).__name__}")
        # Keep only known fields to avoid unexpected-keyword-argument errors
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class Session:
    """
    Verarbeitet session containing multiple processed files.

    Attributes:
        session_id: Unique session identifier
        session_name: User-friendly session name
        description: Optional session description
        created: Creation timestamp
        last_modified: Last modification timestamp
        status: Session status (active, completed, etc.)
        files: List of processed files in this session
        default_settings: Default processing settings for session
        favorites: Whether this session is favorited
        tags: User-defined tags for categorization
    """

    session_id: str
    session_name: str
    description: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    status: SessionStatus = SessionStatus.ACTIVE
    files: list[ProcessedFile] = field(default_factory=list)
    default_settings: dict[str, Any] = field(default_factory=dict)
    favorites: bool = False
    tags: list[str] = field(default_factory=list)

    def add_file(self, processed_file: ProcessedFile):
        """Fügt hinzu: a processed file to this session."""
        self.files.append(processed_file)
        self.last_modified = datetime.now().isoformat()

        # Update status based on files
        if all(f.success for f in self.files):
            self.status = SessionStatus.COMPLETED
        elif any(f.success for f in self.files):
            self.status = SessionStatus.IN_PROGRESS
        else:
            self.status = SessionStatus.FAILED

    def get_file_count(self) -> int:
        """Gibt zurück: total number of files in session."""
        return len(self.files)

    def get_success_count(self) -> int:
        """Gibt zurück: number of successfully processed files."""
        return sum(1 for f in self.files if f.success)

    def get_failed_count(self) -> int:
        """Gibt zurück: number of failed files."""
        return sum(1 for f in self.files if not f.success)

    def to_dict(self) -> dict:
        """Konvertiert to dictionary for JSON serialization."""
        data = {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "description": self.description,
            "created": self.created,
            "last_modified": self.last_modified,
            "status": self.status.value,
            "files": [f.to_dict() for f in self.files],
            "default_settings": self.default_settings,
            "favorites": self.favorites,
            "tags": self.tags,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Erstellt from dictionary."""
        # Convert files back to ProcessedFile objects
        files = [ProcessedFile.from_dict(f) for f in data.get("files", [])]

        # Convert status string back to enum
        status_str = data.get("status", "active")
        try:
            status = SessionStatus(status_str)
        except ValueError:
            status = SessionStatus.ACTIVE

        return cls(
            session_id=data["session_id"],
            session_name=data["session_name"],
            description=data.get("description", ""),
            created=data.get("created", datetime.now().isoformat()),
            last_modified=data.get("last_modified", datetime.now().isoformat()),
            status=status,
            files=files,
            default_settings=data.get("default_settings", {}),
            favorites=data.get("favorites", False),
            tags=data.get("tags", []),
        )


class SessionManager:
    """
    Manages user sessions, history, and processing workflow persistence.

    Features:
    - Session save/load (JSON-based)
    - Recent sessions tracking (last 10)
    - Session history (last 100 processed files across all sessions)
    - Favorites/bookmarks
    - Settings templates
    - Session export/import

    Example:
        >>> manager = SessionManager()
        >>>
        >>> # Create new session
        >>> session = manager.create_session("My Restoration Project")
        >>>
        >>> # Add processed file
        >>> processed = ProcessedFile(
        ...     input_path="/path/to/input.wav",
        ...     output_path="/path/to/output.wav",
        ...     processing_mode="restoration",
        ...     musical_goals_scores={'brillanz': 0.88, 'waerme': 0.85}
        ... )
        >>> manager.add_to_session(processed)
        >>>
        >>> # Save session
        >>> manager.save_session()
        >>>
        >>> # Load session later
        >>> manager.load_session("My Restoration Project")
    """

    def __init__(self, sessions_dir: Path | None = None, max_history: int = 100, max_recent: int = 10):
        """
        Initialisiert SessionManager.

        Args:
            sessions_dir: Directory for session storage (default: ./sessions)
            max_history: Maximum files to keep in history (default: 100)
            max_recent: Maximum recent sessions to track (default: 10)
        """
        self.sessions_dir = sessions_dir or Path("./sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.max_history = max_history
        self.max_recent = max_recent

        # Current active session
        self.current_session: Session | None = None

        # Recent sessions cache
        self._recent_sessions: list[dict[str, str]] = []
        self._load_recent_sessions()

        # Global history (last 100 files across all sessions)
        self._history: list[ProcessedFile] = []
        self._load_history()

        logger.info("SessionManager initialisiert with sessions_dir: %s", self.sessions_dir)

    def create_session(self, session_name: str, description: str = "", default_settings: dict | None = None) -> Session:
        """
        Erstellt a new session.

        Args:
            session_name: User-friendly session name
            description: Optional session description
            default_settings: Default processing settings for this session

        Returns:
            Created Session object
        """
        # Generate unique session ID
        session_id = self._generate_session_id(session_name)

        # Create session
        self.current_session = Session(
            session_id=session_id,
            session_name=session_name,
            description=description,
            default_settings=default_settings or {},
            status=SessionStatus.ACTIVE,
        )

        logger.info("erstellt new Sitzung: %s (ID: %s)", session_name, session_id)
        return self.current_session

    def add_to_session(self, processed_file: ProcessedFile):
        """
        Fügt hinzu: a processed file to the current session.

        Args:
            processed_file: ProcessedFile object to add
        """
        if not self.current_session:
            raise RuntimeError("No active session. Create a session first.")

        self.current_session.add_file(processed_file)

        # Add to global history
        self._history.append(processed_file)
        if len(self._history) > self.max_history:
            self._history.pop(0)  # Remove oldest

        logger.info("Added file to Sitzung: %s", processed_file.input_path)

    def save_session(self, session_name: str | None = None) -> Path:
        """
        Speichert current session to JSON file.

        Args:
            session_name: Optional custom session name (uses current if not provided)

        Returns:
            Path to saved session file
        """
        if not self.current_session:
            raise RuntimeError("No active session to save.")

        # Use custom name if provided
        if session_name:
            self.current_session.session_name = session_name
            self.current_session.session_id = self._generate_session_id(session_name)

        # Update last modified
        self.current_session.last_modified = datetime.now().isoformat()

        # Save to JSON
        session_path = self.sessions_dir / f"{self.current_session.session_id}.json"
        session_data = self.current_session.to_dict()

        session_path.write_text(json.dumps(session_data, indent=2))

        # Update recent sessions
        self._add_to_recent(session_name or self.current_session.session_name, session_path)

        # Save history
        self._save_history()

        logger.info("Sitzung gespeichert: %s", session_path)
        return session_path

    def load_session(self, session_name: str) -> Session:
        """
        Lädt a session by name.

        Args:
            session_name: Name of session to load

        Returns:
            Loaded Session object
        """
        # Find session file
        session_id = self._generate_session_id(session_name)
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_name}")

        # Load from JSON
        session_data = json.loads(session_path.read_text())
        self.current_session = Session.from_dict(session_data)

        # Update recent sessions
        self._add_to_recent(session_name, session_path)

        logger.info("Sitzung geladen: %s", session_name)
        return self.current_session

    def get_recent_sessions(self, n: int | None = None) -> list[dict[str, str]]:
        """
        Gibt zurück: list of recent sessions.

        Args:
            n: Number of sessions to return (default: max_recent)

        Returns:
            List of recent session info dicts with 'name' and 'path' keys
        """
        n = n or self.max_recent
        return self._recent_sessions[:n]

    def get_session_history(self, n: int | None = None) -> list[ProcessedFile]:
        """
        Gibt zurück: global session history (last N processed files across all sessions).

        Args:
            n: Number of files to return (default: all, max max_history)

        Returns:
            List of ProcessedFile objects
        """
        if n is None:
            return self._history.copy()
        return self._history[-n:]

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        Listet auf: all available sessions.

        Returns:
            List of session info dicts
        """
        sessions = []

        for session_path in self.sessions_dir.glob("*.json"):
            try:
                session_data = json.loads(session_path.read_text())
                sessions.append(
                    {
                        "name": session_data["session_name"],
                        "created": session_data["created"],
                        "last_modified": session_data["last_modified"],
                        "status": session_data["status"],
                        "file_count": len(session_data.get("files", [])),
                        "favorites": session_data.get("favorites", False),
                        "path": str(session_path),
                    }
                )
            except Exception as e:
                logger.warning("konnte nicht laden Sitzung %s: %s", session_path, e)
                continue

        # Sort by last modified (most recent first)
        sessions.sort(key=lambda s: s["last_modified"], reverse=True)

        return sessions

    def delete_session(self, session_name: str):
        """
        Delete a session.

        Args:
            session_name: Name of session to delete
        """
        session_id = self._generate_session_id(session_name)
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_name}")

        session_path.unlink()

        # Remove from recent sessions
        self._recent_sessions = [s for s in self._recent_sessions if s["name"] != session_name]
        self._save_recent_sessions()

        logger.info("Sitzung deleted: %s", session_name)

    def toggle_favorite(self, session_name: str):
        """
        Toggle favorite status for a session.

        Args:
            session_name: Name of session
        """
        session = self.load_session(session_name)
        session.favorites = not session.favorites
        current_session_backup = self.current_session
        self.current_session = session
        self.save_session()
        self.current_session = current_session_backup

    def export_session(self, session_name: str, export_path: Path):
        """
        Export session to external path.

        Args:
            session_name: Name of session to export
            export_path: Destination path for export
        """
        session_id = self._generate_session_id(session_name)
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_name}")

        shutil.copy(session_path, export_path)
        logger.info("Sitzung exported to: %s", export_path)

    def import_session(self, import_path: Path) -> Session:
        """
        Import session from external path.

        Args:
            import_path: Path to session JSON file

        Returns:
            Imported Session object
        """
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")

        # Load session data
        session_data = json.loads(import_path.read_text())
        session = Session.from_dict(session_data)

        # Save to sessions directory
        session_path = self.sessions_dir / f"{session.session_id}.json"
        session_path.write_text(json.dumps(session_data, indent=2))

        # Update recent sessions
        self._add_to_recent(session.session_name, session_path)

        logger.info("Sitzung imported: %s", session.session_name)
        return session

    def get_favorites(self) -> list[dict[str, Any]]:
        """
        Gibt zurück: all favorited sessions.

        Returns:
            List of favorite session info dicts
        """
        all_sessions = self.list_sessions()
        return [s for s in all_sessions if s["favorites"]]

    def search_sessions(self, query: str) -> list[dict[str, Any]]:
        """
        Search sessions by name or description.

        Args:
            query: Search query string

        Returns:
            List of matching session info dicts
        """
        all_sessions = self.list_sessions()
        query_lower = query.lower()

        matches = []
        for session_info in all_sessions:
            # Load full session to check description
            try:
                session_path = Path(session_info["path"])
                session_data = json.loads(session_path.read_text())

                name_match = query_lower in session_data["session_name"].lower()
                desc_match = query_lower in session_data.get("description", "").lower()

                if name_match or desc_match:
                    matches.append(session_info)
            except Exception as e:
                logger.warning("konnte nicht search Sitzung %s: %s", session_info["name"], e)
                continue

        return matches

    def _generate_session_id(self, session_name: str) -> str:
        """Generiert unique session ID from name."""
        # Simple ID: lowercase, replace spaces with underscores
        session_id = session_name.lower().replace(" ", "_")
        # Remove special characters
        session_id = "".join(c for c in session_id if c.isalnum() or c == "_")
        return session_id

    def _add_to_recent(self, session_name: str, session_path: Path):
        """Fügt hinzu: session to recent sessions list."""
        # Remove if already present
        self._recent_sessions = [s for s in self._recent_sessions if s["name"] != session_name]

        # Add to front
        self._recent_sessions.insert(
            0, {"name": session_name, "path": str(session_path), "accessed": datetime.now().isoformat()}
        )

        # Trim to max_recent
        if len(self._recent_sessions) > self.max_recent:
            self._recent_sessions = self._recent_sessions[: self.max_recent]

        # Save recent sessions
        self._save_recent_sessions()

    def _save_recent_sessions(self):
        """Speichert recent sessions to file."""
        recent_path = self.sessions_dir / "_recent.json"
        recent_path.write_text(json.dumps(self._recent_sessions, indent=2))

    def _load_recent_sessions(self):
        """Lädt recent sessions from file."""
        recent_path = self.sessions_dir / "_recent.json"

        if recent_path.exists():
            try:
                raw = json.loads(recent_path.read_text())
                # Legacy format was a list-of-lists; current format is list-of-dicts.
                if isinstance(raw, list) and raw and not isinstance(raw[0], dict):
                    logger.warning("_recent.json: legacy list-of-lists format erkannt — resetting.")
                    self._recent_sessions = []
                    recent_path.unlink(missing_ok=True)
                else:
                    self._recent_sessions = [e for e in raw if isinstance(e, dict)]
            except Exception as e:
                logger.warning("konnte nicht laden recent sessions: %s", e)
                self._recent_sessions = []
        else:
            self._recent_sessions = []

    def _save_history(self):
        """Speichert global history to file."""
        history_path = self.sessions_dir / "_history.json"
        history_data = [f.to_dict() for f in self._history]
        history_path.write_text(json.dumps(history_data, indent=2))

    def _load_history(self):
        """Lädt global history from file."""
        history_path = self.sessions_dir / "_history.json"

        if history_path.exists():
            try:
                history_data = json.loads(history_path.read_text())
                if not isinstance(history_data, list):
                    raise ValueError("_history.json: expected list at top level")
                loaded: list = []
                for entry in history_data:
                    try:
                        loaded.append(ProcessedFile.from_dict(entry))
                    except Exception as entry_exc:
                        logger.debug("_history.json: skipping malformed entry: %s", entry_exc)
                self._history = loaded
            except Exception as e:
                logger.warning("konnte nicht laden history: %s", e)
                self._history = []
        else:
            self._history = []


if __name__ == "__main__":
    # Example usage
    pass

    # Create session manager
    manager = SessionManager(sessions_dir=Path("./test_sessions"))

    # Create new session
    session = manager.create_session(
        "My Restoration Project",
        description="Restoring old vinyl recordings",
        default_settings={"mode": "restoration", "strength": 0.8},
    )

    # Add some processed files
    for i in range(3):
        processed = ProcessedFile(
            input_path=f"/path/to/input_{i}.wav",
            output_path=f"/path/to/output_{i}.wav",
            processing_mode="restoration",
            processing_settings={"mode": "restoration", "strength": 0.8},
            musical_goals_scores={"brillanz": 0.88, "waerme": 0.85, "natuerlichkeit": 0.90},
            processing_time=12.5,
            success=True,
        )
        manager.add_to_session(processed)

    # Save session
    session_path = manager.save_session()
    logger.debug("Sitzung gespeichert to: %s", session_path)

    # List all sessions
    sessions = manager.list_sessions()
    logger.debug("\nAll sessions (%s):", len(sessions))
    for s in sessions:
        logger.debug("  - %s: %s files, status: %s", s["name"], s["file_count"], s["status"])

    # Get recent sessions
    recent = manager.get_recent_sessions()
    logger.debug("\nRecent sessions (%s):", len(recent))
    for s in recent:
        logger.debug("  - %s", s["name"])

    # Get history
    history = manager.get_session_history(n=5)
    logger.debug("\nRecent history (%s files):", len(history))
    for f in history:
        logger.debug("  - %s at %s", f.input_path, f.timestamp)
