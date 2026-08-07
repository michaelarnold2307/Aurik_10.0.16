"""
core/batch_session_learner.py — Aurik 10.0.0+ (§2.24)

BatchSessionLearner: Persistiert und überträgt GP-Learnings innerhalb einer
Batch-Session. Bei mehreren Dateien lernt Aurik progressiv: Die optimalen
Parameter der ersten Datei informieren die nächste als Warm-Start, da
Aufnahmen derselben Session ähnliche Charakteristika haben.

Persistenz: ~/.aurik/batch_sessions/<session_id>.json
Aktivierung: automatisch wenn ≥ 2 Dateien aus demselben Unterordner.

Invarianten:
    - Session-State ist kurzlebig (Session-Scope, nicht permanent)
    - Max. 50 Dateien pro Session
    - NaN/Inf in gespeicherten Scores: ignoriert, kein Update
    - Thread-sicher: Singleton mit threading.Lock()
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Zustand einer laufenden Batch-Session."""

    session_id: str
    material: str
    n_files: int = 0
    best_params: dict[str, float] = field(default_factory=dict)
    best_score: float = 0.0
    scores: list[float] = field(default_factory=list)
    all_params: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class BatchSessionLearner:
    """Sessionübergreifendes GP-Lernen für Batch-Verarbeitung (§2.24).

    Algorithmus:
        Session-Kennung: SHA256[:8] aus gemeinsamem Eltern-Ordnerpfad

        1. Erste Datei: GP-Proposal normal aus ~/.aurik/gp_memory/<material>.json
        2. Nach Restaurierung: Session-State gespeichert
        3. Folgedatei: GP warm-started aus Session-State (höhere Initialqualität)
        4. Am Session-Ende: Session-Learnings in permanentes gp_memory/ übernommen
           wenn Session-Score > bisheriger Best-Score

    Invarianten:
        - Max. 50 Dateien pro Session-Objekt (danach neuer Start)
        - Session-Files enthalten keine Audio-Daten, nur Parameter + Scores
        - NaN/Inf in Scores: ignoriert (math.isfinite guard)
        - Thread-sicher: threading.Lock() pro Instanz
    """

    SESSION_DIR: pathlib.Path = pathlib.Path.home() / ".aurik" / "batch_sessions"
    GP_MEMORY_DIR: pathlib.Path = pathlib.Path.home() / ".aurik" / "gp_memory"
    MAX_FILES_PER_SESSION: int = 50

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def start_session(self, file_paths: list[pathlib.Path]) -> str:
        """Erstellt oder lädt Session. Gibt session_id zurück.

        Args:
            file_paths: Liste der zu verarbeitenden Dateipfade.

        Returns:
            session_id (SHA256[:8] des gemeinsamen Eltern-Ordners)
        """
        session_id = self._detect_session_id(file_paths)
        with self._lock:
            if session_id not in self._sessions:
                # Versuche persistierten Zustand zu laden
                loaded = self._load_state(session_id)
                if loaded is not None:
                    self._sessions[session_id] = loaded
                    logger.info(
                        "📂 BatchSession %s: %d vorherige Dateien geladen (best=%.3f)",
                        session_id,
                        loaded.n_files,
                        loaded.best_score,
                    )
                else:
                    material = "unknown"
                    self._sessions[session_id] = SessionState(
                        session_id=session_id,
                        material=material,
                    )
                    logger.info("📂 BatchSession %s: Neue Sitzung gestartet", session_id)
        return session_id

    def get_warm_start(
        self,
        session_id: str,
        material: str,
    ) -> dict[str, float] | None:
        """GP-Warm-Start-Parameter für nächste Datei oder None.

        Gibt die bisher besten Parameter der Session zurück,
        wenn mindestens 1 Datei bereits verarbeitet wurde.

        Args:
            session_id: Session-Kennung aus start_session()
            material:   Material-Typ der aktuellen Datei

        Returns:
            Dict[str, float] mit GP-Parametern oder None beim ersten File.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.n_files == 0:
                return None
            if state.best_score <= 0.0 or not state.best_params:
                return None
            logger.debug(
                "BatchSession %s: Warm-Start aus %d Dateien (best=%.3f)",
                session_id,
                state.n_files,
                state.best_score,
            )
            return dict(state.best_params)

    def update(
        self,
        session_id: str,
        material: str,
        params: dict[str, float],
        score: float,
    ) -> None:
        """Aktualisiert Session-State nach Restaurierung einer Datei.

        Args:
            session_id: Session-Kennung
            material:   Material-Typ
            params:     GP-Parameter die verwendet wurden
            score:      Qualitäts-Score (z.B. PQS-MOS) dieser Datei
        """
        if not math.isfinite(score):
            logger.debug("BatchSession %s: Wert NaN/Inf, kein Aktualisierung", session_id)
            return

        # Params-Fingerabdruck: NaN entfernen
        clean_params = {k: float(v) for k, v in params.items() if math.isfinite(float(v))}

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(session_id=session_id, material=material)
                self._sessions[session_id] = state

            state.material = material
            state.n_files += 1
            state.scores.append(score)
            state.all_params.append(clean_params)

            if score > state.best_score:
                state.best_score = score
                state.best_params = clean_params
                logger.info(
                    "✨ BatchSession %s: Neuer Bestwert %.3f (Datei %d, Material=%s)",
                    session_id,
                    score,
                    state.n_files,
                    material,
                )

            # Max-Dateien-Check
            if state.n_files >= self.MAX_FILES_PER_SESSION:
                logger.info(
                    "BatchSession %s: Max-Dateien (%d) erreicht, Sitzung wird beendet",
                    session_id,
                    self.MAX_FILES_PER_SESSION,
                )
                self._finalize_unlocked(state)
                del self._sessions[session_id]
                return

            # Persistieren
            self._save_state(state)

    def finalize(self, session_id: str) -> None:
        """Überträgt Session-Learnings in permanentes gp_memory/ (wenn besser).

        Soll am Ende einer Batch-Session aufgerufen werden.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            self._finalize_unlocked(state)
            del self._sessions[session_id]

    # ----------------------------------------------------------------
    # Hilfsmethoden
    # ----------------------------------------------------------------

    def _finalize_unlocked(self, state: SessionState) -> None:
        """Überträgt Session-Learnings ohne Lock (muss unter Lock aufgerufen werden)."""
        if not state.best_params or state.best_score <= 0.0:
            return
        try:
            self.GP_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            memory_path = self.GP_MEMORY_DIR / f"{state.material}.json"

            # Bestehende permanente Learnings laden
            existing_best = 0.0
            if memory_path.exists():
                try:
                    with open(memory_path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    obs = data.get("observations", [])
                    if obs:
                        existing_best = max(o.get("score", 0.0) for o in obs if math.isfinite(o.get("score", 0.0)))
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

            # Nur übernehmen wenn Session besser
            if state.best_score > existing_best:
                from datetime import datetime

                entry = {
                    "params": state.best_params,
                    "score": state.best_score,
                    "ts": datetime.utcnow().isoformat(),
                    "source": "batch_session",
                    "n_files": state.n_files,
                }
                try:
                    if memory_path.exists():
                        with open(memory_path, encoding="utf-8") as fh:
                            data = json.load(fh)
                    else:
                        data = {"observations": [], "version": 1}
                    data["observations"].append(entry)
                    with open(memory_path, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                    logger.info(
                        "💾 BatchSession %s → gp_memory/%s.json (%.3f > %.3f)",
                        state.session_id,
                        state.material,
                        state.best_score,
                        existing_best,
                    )
                except Exception as exc:
                    logger.debug("gp_memory-Schreiben fehlgeschlagen: %s", exc)
        except Exception as exc:
            logger.debug("BatchSession abschliessen fehlgeschlagen: %s", exc)

    def _detect_session_id(self, file_paths: list[pathlib.Path]) -> str:
        """SHA256[:8] des gemeinsamen Eltern-Ordnerpfads."""
        if not file_paths:
            return "default"
        parent = str(file_paths[0].parent)
        return hashlib.sha256(parent.encode()).hexdigest()[:8]

    def _save_state(self, state: SessionState) -> None:
        """Persistiert Session-State als JSON."""
        try:
            self.SESSION_DIR.mkdir(parents=True, exist_ok=True)
            path = self.SESSION_DIR / f"{state.session_id}.json"
            data = {
                "session_id": state.session_id,
                "material": state.material,
                "n_files": state.n_files,
                "best_score": state.best_score,
                "best_params": state.best_params,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            logger.debug("BatchSession-Persistenz fehlgeschlagen: %s", exc)

    def _load_state(self, session_id: str) -> SessionState | None:
        """Lädt persistierten Session-State oder None."""
        path = self.SESSION_DIR / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return SessionState(
                session_id=data.get("session_id", session_id),
                material=data.get("material", "unknown"),
                n_files=data.get("n_files", 0),
                best_score=data.get("best_score", 0.0),
                best_params=data.get("best_params", {}),
            )
        except Exception as exc:
            logger.debug("BatchSession-Laden fehlgeschlagen: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: BatchSessionLearner | None = None
_lock = threading.Lock()


def get_batch_session_learner() -> BatchSessionLearner:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BatchSessionLearner()
    return _instance
