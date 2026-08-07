"""ABX Listener — Web-basiertes ABX-Blindhör-Test-Interface.

§15.3: Serviert A/B/X-Triplets mit randomisierter Reihenfolge für
formale perzeptuelle Tests. Session-Management, Ergebnis-Persistenz in SQLite.

Die ABX-Methode: Der Hörer hört drei Samples — Referenz A, Referenz B,
und ein unbekanntes X (das entweder A oder B entspricht). Aufgabe:
Ist X = A oder X = B? Statistisch auswertbar via Binomialtest.

Endpoint-Übersicht:
    POST   /abx/session/create     — Neue ABX-Session erstellen
    GET    /abx/session/{id}       — Session-Status abrufen
    GET    /abx/session/{id}/trial — Aktuelles Trial (A, B, X)
    POST   /abx/session/{id}/answer — Antwort (is_a: true/false)
    GET    /abx/session/{id}/results — Endergebnisse

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Datenbank-Pfad ──────────────────────────────────────────────────────────
_DB_DIR = Path(__file__).parent.parent.parent / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "abx_sessions.db"


@dataclass
class ABXTrial:
    """Ein einzelner ABX-Trial."""

    trial_index: int
    stimulus_a: str  # Pfad zu Sample A
    stimulus_b: str  # Pfad zu Sample B
    stimulus_x: str  # Pfad zu Sample X (= A oder B, zufällig)
    x_is_a: bool  # True wenn X = A, False wenn X = B

    # Vom Teilnehmer auszufüllen:
    answer_is_a: bool | None = None
    answered_at: float | None = None
    correct: bool | None = None


@dataclass
class ABXSession:
    """Eine komplette ABX-Hör-Session."""

    session_id: str
    created_at: float
    metadata: dict = field(default_factory=dict)
    trials: list[ABXTrial] = field(default_factory=list)
    current_trial: int = 0
    completed: bool = False

    @property
    def total_trials(self) -> int:
        return len(self.trials)

    @property
    def completed_trials(self) -> int:
        return sum(1 for t in self.trials if t.answer_is_a is not None)

    @property
    def correct_count(self) -> int:
        return sum(1 for t in self.trials if t.correct is True)

    @property
    def accuracy(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.correct_count / self.total_trials

    @property
    def p_value(self) -> float:
        """Binomialtest: Wie wahrscheinlich ist dieses Ergebnis unter H0 (p=0.5)?"""
        if self.total_trials == 0:
            return 1.0
        from math import comb

        k = self.correct_count
        n = self.total_trials
        p = 0.5  # Nullhypothese: Raten
        # Summiere P(X >= k) unter H0
        prob = sum(comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))
        return min(1.0, max(0.0, prob))


class ABXListener:
    """ABX-Hör-Test-Manager (Singleton)."""

    _instance: ABXListener | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._sessions: dict[str, ABXSession] = {}
        self._init_db()

    @classmethod
    def get_instance(cls) -> ABXListener:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _init_db(self) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS abx_results (
                    session_id TEXT NOT NULL,
                    trial_index INTEGER NOT NULL,
                    stimulus_a TEXT,
                    stimulus_b TEXT,
                    x_is_a INTEGER,
                    answer_is_a INTEGER,
                    correct INTEGER,
                    answered_at REAL,
                    participant_id TEXT,
                    metadata TEXT,
                    PRIMARY KEY (session_id, trial_index)
                )
            """)
            conn.commit()

    def create_session(
        self,
        stimuli_a: list[str],
        stimuli_b: list[str],
        participant_id: str = "",
        metadata: dict | None = None,
        seed: int = 42,
    ) -> ABXSession:
        """Erstellt eine neue ABX-Session.

        Args:
            stimuli_a: Liste von Dateipfaden für Sample A (z.B. Aurik-Outputs).
            stimuli_b: Liste von Dateipfaden für Sample B (z.B. iZotope-Outputs).
            participant_id: Optionaler Teilnehmer-Identifier.
            metadata: Zusätzliche Metadaten.
            seed: Random-Seed für Reproduzierbarkeit.

        Returns:
            ABXSession mit randomisierten Trials.
        """
        rng = random.Random(seed)
        session_id = str(uuid.uuid4())[:12]

        n_trials = min(len(stimuli_a), len(stimuli_b))
        if n_trials == 0:
            raise ValueError("Mindestens 1 Stimulus-Paar erforderlich")

        trials: list[ABXTrial] = []
        for i in range(n_trials):
            x_is_a = rng.choice([True, False])
            stimulus_x = stimuli_a[i] if x_is_a else stimuli_b[i]

            trials.append(
                ABXTrial(
                    trial_index=i,
                    stimulus_a=stimuli_a[i],
                    stimulus_b=stimuli_b[i],
                    stimulus_x=stimulus_x,
                    x_is_a=x_is_a,
                )
            )

        # Trials randomisieren
        rng.shuffle(trials)
        # Trial-Indices neu setzen
        for idx, trial in enumerate(trials):
            trial.trial_index = idx

        session = ABXSession(
            session_id=session_id,
            created_at=time.time(),
            metadata=metadata or {},
            trials=trials,
        )

        with self._lock:
            self._sessions[session_id] = session

        logger.info("ABX-Sitzung erstellt: %s (%d Trials)", session_id, n_trials)
        return session

    def get_session(self, session_id: str) -> ABXSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_trial(self, session_id: str) -> dict[str, Any] | None:
        """Aktuelles Trial abrufen."""
        session = self.get_session(session_id)
        if session is None:
            return None
        if session.completed:
            return {"status": "completed", "session_id": session_id}
        if session.current_trial >= session.total_trials:
            session.completed = True
            return {"status": "completed", "session_id": session_id}

        trial = session.trials[session.current_trial]
        return {
            "status": "active",
            "session_id": session_id,
            "trial_index": trial.trial_index,
            "total_trials": session.total_trials,
            "completed_trials": session.completed_trials,
            "stimulus_a": trial.stimulus_a,
            "stimulus_b": trial.stimulus_b,
            "stimulus_x": trial.stimulus_x,
            # A/B-Labels NICHT preiszugeben, um Bias zu vermeiden
            "labels": {
                "a": "Referenz A",
                "b": "Referenz B",
                "x": "Unbekannt X",
            },
        }

    def submit_answer(self, session_id: str, is_a: bool) -> dict[str, Any]:
        """Antwort auf aktuelles Trial einreichen.

        Args:
            session_id: Session-ID.
            is_a: True wenn Hörer denkt X = A, False wenn X = B.

        Returns:
            Ergebnis mit correct-Flag und Fortschritt.
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": "Session nicht gefunden"}
        if session.completed:
            return {"error": "Session bereits abgeschlossen"}
        if session.current_trial >= session.total_trials:
            session.completed = True
            return {"error": "Keine Trials mehr"}

        trial = session.trials[session.current_trial]
        trial.answer_is_a = is_a
        trial.answered_at = time.time()
        trial.correct = is_a == trial.x_is_a

        session.current_trial += 1

        if session.current_trial >= session.total_trials:
            session.completed = True

        # In DB persistieren
        self._save_trial(session_id, trial)

        return {
            "status": "ok",
            "session_id": session_id,
            "trial_index": trial.trial_index,
            "correct": trial.correct,
            "x_was": "a" if trial.x_is_a else "b",
            "completed_trials": session.completed_trials,
            "total_trials": session.total_trials,
            "session_complete": session.completed,
        }

    def get_results(self, session_id: str) -> dict[str, Any] | None:
        """Endergebnisse abrufen."""
        session = self.get_session(session_id)
        if session is None:
            return None

        return {
            "session_id": session_id,
            "completed": session.completed,
            "total_trials": session.total_trials,
            "completed_trials": session.completed_trials,
            "correct_count": session.correct_count,
            "accuracy": session.accuracy,
            "p_value": session.p_value,
            "significant_05": session.p_value < 0.05,
            "significant_01": session.p_value < 0.01,
            "trials": [
                {
                    "trial_index": t.trial_index,
                    "x_is_a": t.x_is_a,
                    "answer_is_a": t.answer_is_a,
                    "correct": t.correct,
                }
                for t in session.trials
                if t.answer_is_a is not None
            ],
        }

    def _save_trial(self, session_id: str, trial: ABXTrial) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO abx_results
                   (session_id, trial_index, stimulus_a, stimulus_b, x_is_a,
                    answer_is_a, correct, answered_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    trial.trial_index,
                    trial.stimulus_a,
                    trial.stimulus_b,
                    int(trial.x_is_a),
                    int(trial.answer_is_a) if trial.answer_is_a is not None else None,
                    int(trial.correct) if trial.correct is not None else None,
                    trial.answered_at,
                    json.dumps({}),
                ),
            )
            conn.commit()


# ── Convenience-Funktionen ──────────────────────────────────────────────────


def create_abx_session(
    stimuli_a: list[str],
    stimuli_b: list[str],
    participant_id: str = "",
    **kwargs,
) -> ABXSession:
    return ABXListener.get_instance().create_session(
        stimuli_a=stimuli_a,
        stimuli_b=stimuli_b,
        participant_id=participant_id,
        **kwargs,
    )


def get_abx_results(session_id: str) -> dict[str, Any] | None:
    return ABXListener.get_instance().get_results(session_id)


__all__ = [
    "ABXListener",
    "ABXSession",
    "ABXTrial",
    "create_abx_session",
    "get_abx_results",
]
