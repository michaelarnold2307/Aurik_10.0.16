"""
Real-Time Musical Goals Monitoring API.

WebSocket-basierter API für Live-Updates von Musical Goals während Processing.

Component 0.9.5: Real-Time Musical Goals Visualization
Impact: +0.5 Punkte - User Transparency
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:
    # FastAPI optional - keep module importable in test/offline environments.
    FastAPI = Any  # type: ignore[assignment,misc]
    HTTPException = Exception  # type: ignore[assignment,misc]
    WebSocket = Any  # type: ignore[assignment,misc]
    WebSocketDisconnect = Exception  # type: ignore[assignment,misc]
    HTMLResponse = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class GoalUpdate:
    """
    Single Musical Goal Update.

    Attributes:
        timestamp: Unix timestamp
        session_id: Session ID
        step_name: Processing step name
        goal_name: Musical goal name
        score: Current score (0-1)
        threshold: Threshold value
        violated: Whether threshold is violated
    """

    timestamp: float
    session_id: str
    step_name: str
    goal_name: str
    score: float
    threshold: float
    violated: bool

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class GoalsSnapshot:
    """
    Complete snapshot of all Musical Goals at specific time.

    Attributes:
        timestamp: Unix timestamp
        session_id: Session ID
        step_name: Current processing step
        goals: Dict of goal_name -> score
        thresholds: Dict of goal_name -> threshold
        violations: List of violated goal names
    """

    timestamp: float
    session_id: str
    step_name: str
    goals: dict[str, float]
    thresholds: dict[str, float]
    violations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dict for JSON serialization."""
        return asdict(self)


class ConnectionManager:
    """
    Manages WebSocket connections für Real-Time Updates.

    Features:
    - Multiple concurrent connections
    - Session-specific filtering
    - Broadcast to all or specific sessions
    """

    def __init__(self):
        """Initialisiert connection manager."""
        self.active_connections: list[WebSocket] = []
        self.session_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str | None = None):
        """
        Accept new WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Optional session ID for filtering
        """
        await websocket.accept()
        self.active_connections.append(websocket)

        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(websocket)

        logger.info("New WebSocket Verbindung (Sitzung=%s)", session_id)

    def disconnect(self, websocket: WebSocket, session_id: str | None = None):
        """
        Entfernt WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Optional session ID
        """
        self.active_connections.remove(websocket)

        if session_id and session_id in self.session_connections:
            self.session_connections[session_id].remove(websocket)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]

        logger.info("WebSocket disconnected (Sitzung=%s)", session_id)

    async def broadcast(self, message: dict[str, Any]):
        """
        Broadcast message to all connections.

        Args:
            message: Message dict to broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning("konnte nicht senden to Verbindung: %s", e)
                disconnected.append(connection)

        # Clean up disconnected
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

    async def send_to_session(self, session_id: str, message: dict[str, Any]):
        """
        Sendet message to specific session.

        Args:
            session_id: Session ID
            message: Message dict to send
        """
        if session_id not in self.session_connections:
            return

        disconnected = []
        for connection in self.session_connections[session_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning("konnte nicht senden to Sitzung %s: %s", session_id, e)
                disconnected.append(connection)

        # Clean up disconnected
        for connection in disconnected:
            if connection in self.session_connections[session_id]:
                self.session_connections[session_id].remove(connection)


class MusicalGoalsMonitorAPI:
    """
    Real-Time Musical Goals Monitoring API.

    Features:
    - WebSocket endpoint for live updates
    - REST endpoints for history
    - Session management
    - Custom threshold configuration

    Usage:
        api = MusicalGoalsMonitorAPI()
        app = api.create_app()

        # During processing:
        await api.update_goals(session_id, step_name, goals_dict, thresholds_dict)
    """

    def __init__(self, history_storage_path: Path | None = None):
        """
        Initialisiert API.

        Args:
            history_storage_path: Path to store goals history
        """
        self.connection_manager = ConnectionManager()
        self.history_storage_path = history_storage_path or Path("data/goals_history")
        self.history_storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory history (last 1000 updates per session)
        self.history: dict[str, list[GoalsSnapshot]] = {}
        self.max_history_size = 1000

        # Custom thresholds per session
        self.custom_thresholds: dict[str, dict[str, float]] = {}

    def create_app(self) -> Any:  # Returns FastAPI app
        """
        Erstellt FastAPI application with all endpoints.

        Returns:
            FastAPI app instance
        """
        try:
            app = FastAPI(title="Musical Goals Monitor API")

            # WebSocket endpoint
            @app.websocket("/ws/{session_id}")
            async def websocket_endpoint(websocket: WebSocket, session_id: str):
                await self.connection_manager.connect(websocket, session_id)
                try:
                    # Send current history
                    if session_id in self.history:
                        await websocket.send_json(
                            {"type": "history", "data": [snapshot.to_dict() for snapshot in self.history[session_id]]}
                        )

                    # Keep alive
                    while True:
                        data = await websocket.receive_text()
                        # Echo back (ping/pong)
                        await websocket.send_json({"type": "pong", "data": data})

                except WebSocketDisconnect:
                    self.connection_manager.disconnect(websocket, session_id)

            # REST endpoint: Get goals history
            @app.get("/api/goals/history/{session_id}")
            async def get_goals_history(session_id: str, limit: int = 100):
                """Gibt zurück: goals history for session."""
                if session_id not in self.history:
                    return {"session_id": session_id, "history": []}

                history = self.history[session_id][-limit:]
                return {"session_id": session_id, "history": [snapshot.to_dict() for snapshot in history]}

            # REST endpoint: Get current goals
            @app.get("/api/goals/current/{session_id}")
            async def get_current_goals(session_id: str):
                """Gibt zurück: current goals for session."""
                if session_id not in self.history or not self.history[session_id]:
                    raise HTTPException(status_code=404, detail="Session not found")

                current = self.history[session_id][-1]
                return current.to_dict()

            # REST endpoint: Update custom thresholds
            @app.post("/api/goals/thresholds/{session_id}")
            async def update_thresholds(session_id: str, thresholds: dict[str, float]):
                """Aktualisiert custom thresholds for session."""
                # Validate thresholds
                for goal, threshold in thresholds.items():
                    if not (0.0 <= threshold <= 1.0):
                        raise HTTPException(status_code=400, detail=f"Threshold {goal}={threshold} out of range [0, 1]")

                self.custom_thresholds[session_id] = thresholds

                # Broadcast update
                await self.connection_manager.send_to_session(
                    session_id, {"type": "thresholds_updated", "data": thresholds}
                )

                return {"status": "success", "thresholds": thresholds}

            # REST endpoint: Get active sessions
            @app.get("/api/sessions")
            async def get_active_sessions():
                """Gibt zurück: list of active sessions."""
                return {
                    "sessions": list(self.history.keys()),
                    "active_connections": len(self.connection_manager.active_connections),
                }

            # Serve dashboard
            @app.get("/", response_class=HTMLResponse)
            async def get_dashboard():
                """Serve dashboard HTML."""
                dashboard_path = Path(__file__).parent.parent.parent / "frontend" / "musical_goals_dashboard.html"
                if dashboard_path.exists():
                    return HTMLResponse(content=dashboard_path.read_text())
                else:
                    return HTMLResponse(
                        content="<h1>Dashboard not found</h1><p>Place dashboard at frontend/musical_goals_dashboard.html</p>"
                    )

            logger.info("Musical Goals Monitor API erstellt")
            return app

        except ImportError:
            logger.error("FastAPI not installed - API creation fehlgeschlagen")
            return None

    async def update_goals(
        self, session_id: str, step_name: str, goals: dict[str, float], thresholds: dict[str, float] | None = None
    ):
        """
        Aktualisiert goals for session and broadcast to connected clients.

        Args:
            session_id: Session ID
            step_name: Current processing step
            goals: Dict of goal_name -> score
            thresholds: Optional custom thresholds
        """
        # Use custom thresholds if set
        if session_id in self.custom_thresholds:
            thresholds = self.custom_thresholds[session_id]
        elif thresholds is None:
            # Default thresholds
            thresholds = {
                "bass-kraft": 0.85,
                "brillanz": 0.85,
                "waerme": 0.80,
                "natuerlichkeit": 0.90,
                "authentizitaet": 0.88,
                "emotionalitaet": 0.87,
                "transparenz": 0.89,
            }

        # Detect violations
        violations = [goal_name for goal_name, score in goals.items() if score < thresholds.get(goal_name, 0.0)]

        # Create snapshot
        snapshot = GoalsSnapshot(
            timestamp=datetime.now().timestamp(),
            session_id=session_id,
            step_name=step_name,
            goals=goals,
            thresholds=thresholds,
            violations=violations,
        )

        # Store in history
        if session_id not in self.history:
            self.history[session_id] = []

        self.history[session_id].append(snapshot)

        # Limit history size
        if len(self.history[session_id]) > self.max_history_size:
            self.history[session_id] = self.history[session_id][-self.max_history_size :]

        # Broadcast to connected clients
        await self.connection_manager.send_to_session(session_id, {"type": "update", "data": snapshot.to_dict()})

        # Log violations
        if violations:
            logger.warning("Sitzung %s step %s: Violations: %s", session_id, step_name, ", ".join(violations))

    def save_history(self, session_id: str):
        """
        Speichert session history to disk.

        Args:
            session_id: Session ID
        """
        if session_id not in self.history:
            return

        history_file = self.history_storage_path / f"{session_id}.json"

        with open(history_file, "w") as f:
            json.dump([snapshot.to_dict() for snapshot in self.history[session_id]], f, indent=2)

        logger.info("gespeichert history for Sitzung %s", session_id)

    def load_history(self, session_id: str) -> bool:
        """
        Lädt session history from disk.

        Args:
            session_id: Session ID

        Returns:
            True if loaded successfully
        """
        history_file = self.history_storage_path / f"{session_id}.json"

        if not history_file.exists():
            return False

        with open(history_file) as f:
            data = json.load(f)

        self.history[session_id] = [GoalsSnapshot(**snapshot) for snapshot in data]

        logger.info("geladen history for Sitzung %s", session_id)
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Gibt zurück: API statistics."""
        return {
            "active_sessions": len(self.history),
            "active_connections": len(self.connection_manager.active_connections),
            "total_updates": sum(len(h) for h in self.history.values()),
            "sessions_with_violations": sum(
                1 for history in self.history.values() if any(snapshot.violations for snapshot in history)
            ),
        }
