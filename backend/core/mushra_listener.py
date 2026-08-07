"""MUSHRA Listener — ITU-R BS.1534-3 konformes Web-Interface.

§15.3: Serviert MUSHRA-Test-Sessions mit Hidden Reference, 3.5-kHz-Anchor.
Integriert mit mushra_session.py für formale Session-Verwaltung.

Endpoint-Übersicht:
    POST /mushra/session/create  — Neue Session anlegen
    GET  /mushra/session/<id>/trial — Nächster Trial
    POST /mushra/session/<id>/rate  — Rating abgeben
    GET  /mushra/session/<id>/results — Ergebnisse abrufen
    GET  /mushra/health           — Health-Check

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import logging
import random
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Datenbank ───────────────────────────────────────────────────────────────
_DB_DIR = Path(__file__).parent.parent.parent / "data" / "mushra_sessions"
_DB_DIR.mkdir(parents=True, exist_ok=True)


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_DIR / "mushra_results.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        trial_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        score REAL NOT NULL,
        listener_id TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )""")
    conn.commit()
    return conn


@dataclass
class MUSHRATrial:
    trial_id: str
    scenario: str
    conditions: list[str]
    display_order: list[str]
    hidden_ref_key: str
    anchor_key: str
    stimulus_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class MUSHRASessionState:
    session_id: str
    listener_id: str
    trials: list[MUSHRATrial] = field(default_factory=list)
    current_trial_idx: int = 0
    created_at: str = ""
    completed: bool = False


class MUSHRASessionManager:
    """Verwaltet MUSHRA-Test-Sessions mit Randomisierung und Persistenz."""

    def __init__(self, seed: int = 42):
        self._sessions: dict[str, MUSHRASessionState] = {}
        self._lock = threading.Lock()
        self._rng = random.Random(seed)

    def create_session(
        self,
        listener_id: str,
        scenarios: list[str],
        conditions: list[str] | None = None,
        repetitions: int = 3,
    ) -> MUSHRASessionState:
        if conditions is None:
            conditions = ["reference", "aurik", "anchor"]
        session_id = str(uuid.uuid4())[:8]
        trials: list[MUSHRATrial] = []

        for si, scenario in enumerate(scenarios):
            for rep in range(repetitions):
                order = conditions.copy()
                self._rng.shuffle(order)
                trials.append(
                    MUSHRATrial(
                        trial_id=f"{session_id}_S{si:02d}_R{rep:02d}",
                        scenario=scenario,
                        conditions=conditions,
                        display_order=order,
                        hidden_ref_key="reference",
                        anchor_key="anchor",
                    )
                )

        state = MUSHRASessionState(
            session_id=session_id,
            listener_id=listener_id,
            trials=trials,
            created_at=__import__("datetime").datetime.now().isoformat(),
        )
        with self._lock:
            self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> MUSHRASessionState | None:
        return self._sessions.get(session_id)

    def get_current_trial(self, session_id: str) -> MUSHRATrial | None:
        state = self._sessions.get(session_id)
        if not state or state.current_trial_idx >= len(state.trials):
            return None
        return state.trials[state.current_trial_idx]

    def submit_rating(self, session_id: str, trial_id: str, ratings: dict[str, float]) -> bool:
        state = self._sessions.get(session_id)
        if not state:
            return False

        conn = _get_db()
        ts = __import__("datetime").datetime.now().isoformat()
        for condition, score in ratings.items():
            conn.execute(
                "INSERT INTO ratings (session_id, trial_id, condition, score, listener_id, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, trial_id, condition, float(score), state.listener_id, ts),
            )
        conn.commit()
        conn.close()

        state.current_trial_idx += 1
        if state.current_trial_idx >= len(state.trials):
            state.completed = True
        return True

    def get_results(self, session_id: str) -> dict | None:
        state = self._sessions.get(session_id)
        if not state:
            return None

        conn = _get_db()
        rows = conn.execute(
            "SELECT condition, score FROM ratings WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        conn.close()

        scores: dict[str, list[float]] = {}
        for cond, score in rows:
            scores.setdefault(cond, []).append(score)
        import numpy as np

        return {
            "session_id": session_id,
            "listener_id": state.listener_id,
            "total_trials": len(state.trials),
            "completed_trials": state.current_trial_idx,
            "completed": state.completed,
            "conditions": {
                cond: {
                    "mean": round(float(np.mean(vals)), 2),
                    "std": round(float(np.std(vals)), 2),
                    "n": len(vals),
                }
                for cond, vals in scores.items()
            },
        }


# ── Flask-App-Fabrik ────────────────────────────────────────────────────────


def create_mushra_app(manager: MUSHRASessionManager | None = None):
    """Erzeugt Flask-App mit MUSHRA-Endpoints."""
    try:
        from flask import Flask, jsonify, request
    except ImportError as err:
        raise RuntimeError("Flask nicht installiert. pip install flask") from err

    if manager is None:
        manager = MUSHRASessionManager()
    app = Flask(__name__)
    app.secret_key = str(uuid.uuid4())

    @app.route("/mushra/session/create", methods=["POST"])
    def create():
        data = request.get_json(force=True)
        lid = data.get("listener_id", str(uuid.uuid4())[:6])
        scenarios = data.get("scenarios", ["scenario_01"])
        conditions = data.get("conditions", ["reference", "aurik", "anchor"])
        reps = data.get("repetitions", 3)
        state = manager.create_session(lid, scenarios, conditions, reps)
        return jsonify({"session_id": state.session_id, "listener_id": lid, "total_trials": len(state.trials)})

    @app.route("/mushra/session/<sid>/trial", methods=["GET"])
    def get_trial(sid):
        trial = manager.get_current_trial(sid)
        if trial is None:
            st = manager.get_session(sid)
            if st and st.completed:
                return jsonify({"completed": True, "message": "Alle Trials abgeschlossen"})
            return jsonify({"error": "Session nicht gefunden"}), 404
        st = manager.get_session(sid)
        return jsonify(
            {
                "session_id": sid,
                "trial_id": trial.trial_id,
                "scenario": trial.scenario,
                "display_order": trial.display_order,
                "trial_index": st.current_trial_idx + 1,  # type: ignore[union-attr]
                "total_trials": len(st.trials),  # type: ignore[union-attr]
                "completed": False,
            }
        )

    @app.route("/mushra/session/<sid>/rate", methods=["POST"])
    def rate(sid):
        data = request.get_json(force=True)
        ok = manager.submit_rating(sid, data.get("trial_id"), data.get("ratings", {}))
        if not ok:
            return jsonify({"error": "Rating fehlgeschlagen"}), 400
        st = manager.get_session(sid)
        return jsonify(
            {
                "ok": True,
                "completed_trials": st.current_trial_idx,  # type: ignore[union-attr]
                "total_trials": len(st.trials),  # type: ignore[union-attr]
                "finished": st.completed,  # type: ignore[union-attr]
            }
        )

    @app.route("/mushra/session/<sid>/results", methods=["GET"])
    def results(sid):
        r = manager.get_results(sid)
        return jsonify(r) if r else (jsonify({"error": "Session nicht gefunden"}), 404)

    @app.route("/mushra/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "listener": "MUSHRA ITU-R BS.1534-3"})

    return app


_manager: MUSHRASessionManager | None = None


def get_mushra_manager() -> MUSHRASessionManager:
    global _manager
    if _manager is None:
        _manager = MUSHRASessionManager()
    return _manager


__all__ = ["MUSHRASessionManager", "MUSHRASessionState", "MUSHRATrial", "create_mushra_app", "get_mushra_manager"]
