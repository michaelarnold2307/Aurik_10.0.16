"""
backend/core/artist_knowledge_base.py — ArtistKnowledgeBase (Aurik 10.0.0 §AKB-1)
==============================================================================
Persistent cross-session learning database for restoration intelligence.

A world-class human engineer accumulates wisdom over hundreds of restorations:
"German Schlager from EMI Electrola 1962 always needs +1.5 dB formant lift at
3.2 kHz after MIIPHER, and phase_03 strength should be 0.35 max."

This module gives Aurik the same compounding advantage:
    - Records per-restoration outcomes (era, material, label, phase strengths, VQI, OQS)
    - On next similar restoration: retrieves matched priors → feeds GPOptimizer
    - Artist-specific profiles accumulate over sessions (requires ≥ 3 records)

Storage: SQLite at ~/.aurik/artist_knowledge.db (local only, never transmitted).

Integration (UV3 §AKB-1):
    After LabelTransferDB, before GoalApplicabilityFilter:
        from backend.core.artist_knowledge_base import get_artist_knowledge_base
        akb = get_artist_knowledge_base()
        _akb_prior = akb.lookup_prior(era, material, label_hint, artist_hash)
        _restoration_context["artist_knowledge_prior"] = _akb_prior
    GPOptimizer reads "artist_knowledge_prior" and blends with its own suggestions.
    After successful export:
        akb.record_outcome(era, material, label, artist_hash, phase_strengths, vqi, oqs)

Privacy:
    - artist_hash = SHA-256[:16] of lowercase artist name (irreversible)
    - No audio data stored, only scalar strength/metric values
    - Database lives entirely on the user's machine
"""
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH: Path = Path.home() / ".aurik" / "artist_knowledge.db"
# Minimum similarity score (0..1) for a record to contribute to a prior
MIN_MATCH_SCORE: float = 0.40
# Number of similar records needed before prior is "strong" (confidence ≥ 0.7)
STRONG_PRIOR_N: int = 5
# Maximum records returned per lookup
MAX_LOOKUP_RECORDS: int = 20
# OQS/VQI minimum for a record to be stored (only successful restorations)
MIN_VQI_RECORD: float = 0.70
MIN_OQS_RECORD: float = 60.0

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS restoration_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    era             INTEGER NOT NULL,
    material        TEXT    NOT NULL,
    label_hint      TEXT    DEFAULT '',
    artist_hash     TEXT    DEFAULT '',
    genre           TEXT    DEFAULT '',
    mode            TEXT    DEFAULT 'restoration',
    phase_strengths TEXT    NOT NULL,
    vqi             REAL    NOT NULL,
    oqs             REAL    NOT NULL,
    timestamp       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_era_mat ON restoration_history (era, material);
CREATE INDEX IF NOT EXISTS idx_artist  ON restoration_history (artist_hash);
"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AKBPrior:
    """Phase-strength prior derived from matched historical records."""

    # dict[phase_id → recommended_strength]
    phase_strengths: dict[str, float] = field(default_factory=dict)
    # 0=no records, 1=very strong prior
    confidence: float = 0.0
    # Number of records that contributed
    n_records: int = 0
    # Mean VQI of contributing records
    mean_vqi: float = 0.0
    # Mean OQS of contributing records
    mean_oqs: float = 0.0

    def to_dict(self) -> dict:
        """Gibt a JSON-safe dictionary representation zurück."""
        return {
            "phase_strengths": self.phase_strengths,
            "confidence": float(self.confidence),
            "n_records": self.n_records,
            "mean_vqi": float(self.mean_vqi),
            "mean_oqs": float(self.mean_oqs),
        }


# ---------------------------------------------------------------------------
# ArtistKnowledgeBase
# ---------------------------------------------------------------------------


class ArtistKnowledgeBase:
    """Persistent cross-session restoration knowledge (§AKB-1).

    Thread-safe.  All DB operations are wrapped in a re-entrant lock.
    Errors are always non-blocking (logged at DEBUG level).
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            self._conn = conn
            logger.debug("§AKB-1 ArtistKnowledgeBase: DB ready at %s", self._db_path)
        except Exception as exc:
            logger.debug("§AKB-1 DB init fehlgeschlagen (nicht blockierend): %s", exc)
            self._conn = None

    def _get_conn(self) -> sqlite3.Connection | None:
        """Gibt active connection, attempting reconnect if needed zurück."""
        if self._conn is not None:
            return self._conn
        try:
            self._init_db()
        except Exception:
            logger.debug("_get_conn: silent except suppressed", exc_info=True)
        return self._conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_prior(
        self,
        era: int,
        material: str,
        label_hint: str = "",
        artist_hash: str = "",
        genre: str = "",
        mode: str = "restoration",
    ) -> AKBPrior:
        """Retrieve phase-strength priors for a new restoration.

        Matching strategy (descending priority):
            1. Same artist_hash (if non-empty) + same material
            2. Same era-decade + same material + same label_hint
            3. Same era-decade + same material
            4. Same era-decade only (broadest fallback)

        Records are weighted by match quality × VQI outcome.

        Args:
            era:         Year of recording (e.g. 1963)
            material:    Material string (e.g. "vinyl", "shellac")
            label_hint:  Optional record label (e.g. "EMI Electrola")
            artist_hash: SHA-256[:16] of artist name (empty = unknown)
            genre:       Genre string (optional, informational)
            mode:        "restoration" or "studio"

        Returns:
            AKBPrior with phase_strengths dict and confidence.
        """
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return AKBPrior()
            try:
                return self._lookup_impl(conn, era, material, label_hint, artist_hash, genre, mode)
            except Exception as exc:
                logger.debug("§AKB-1 lookup_prior fehlgeschlagen: %s", exc)
                return AKBPrior()

    def record_outcome(
        self,
        era: int,
        material: str,
        label_hint: str,
        artist_hash: str,
        phase_strengths: dict[str, float],
        vqi: float,
        oqs: float,
        genre: str = "",
        mode: str = "restoration",
    ) -> None:
        """Persist a successful restoration outcome for future learning.

        Called by UV3 only when:
            - HPI > 0
            - artifact_freedom >= 0.95
            - vqi >= MIN_VQI_RECORD (0.70)
            - oqs >= MIN_OQS_RECORD (60.0)

        Args:
            era:             Year of recording
            material:        Material string
            label_hint:      Record label (may be empty)
            artist_hash:     SHA-256[:16] of artist name (may be empty)
            phase_strengths: dict[phase_id → strength used]
            vqi:             Achieved VQI score
            oqs:             Achieved OQS score
            genre:           Genre string
            mode:            Processing mode
        """
        if vqi < MIN_VQI_RECORD or oqs < MIN_OQS_RECORD:
            logger.debug(
                "§AKB-1 aufzeichnen_outcome uebersprungen: vqi=%.3f oqs=%.1f below min thresholds",
                vqi,
                oqs,
            )
            return
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return
            try:
                self._insert_record(
                    conn, era, material, label_hint, artist_hash, phase_strengths, vqi, oqs, genre, mode
                )
            except Exception as exc:
                logger.debug("§AKB-1 aufzeichnen_outcome fehlgeschlagen: %s", exc)

    @staticmethod
    def make_artist_hash(artist_name: str) -> str:
        """Gibt irreversible SHA-256[:16] hash of the artist name zurück.

        Args:
            artist_name: Lowercase artist name string.

        Returns:
            16-character hex string.
        """
        return hashlib.sha256(artist_name.lower().strip().encode()).hexdigest()[:16]

    def get_record_count(self) -> int:
        """Gibt total number of stored restoration records zurück."""
        with self._lock:
            conn = self._get_conn()
            if conn is None:
                return 0
            try:
                cur = conn.execute("SELECT COUNT(*) FROM restoration_history")
                row = cur.fetchone()
                return int(row[0]) if row else 0
            except Exception as e:
                logger.warning("artist_knowledge_base.py::get_aufzeichnen_count Ersatzpfad: %s", e)
                return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_impl(  # pylint: disable=too-many-locals
        self,
        conn: sqlite3.Connection,
        era: int,
        material: str,
        label_hint: str,
        artist_hash: str,
        genre: str,
        mode: str,
    ) -> AKBPrior:
        """Führt aus: tiered lookup and aggregate phase-strength priors."""
        decade = (era // 10) * 10
        mat = material.lower().strip()
        lbl = label_hint.lower().strip()
        ah = artist_hash.strip()
        genre_key = genre.lower().strip()

        # Collect candidate rows with match scores
        rows: list[tuple[float, dict[str, float], float, float]] = []

        # Tier 1: artist match + material (strongest prior)
        if ah:
            cur = conn.execute(
                "SELECT phase_strengths, vqi, oqs FROM restoration_history "
                "WHERE artist_hash=? AND material=? AND mode=? "
                "ORDER BY vqi DESC LIMIT ?",
                (ah, mat, mode, MAX_LOOKUP_RECORDS),
            )
            for ps_json, vqi, oqs in cur.fetchall():
                try:
                    ps = json.loads(ps_json)
                    rows.append((0.90, ps, float(vqi), float(oqs)))
                except Exception:
                    logger.debug("_lookup_impl: silent except suppressed", exc_info=True)

        # Tier 2: era-decade + material + label (if label known)
        if lbl:
            cur = conn.execute(
                "SELECT phase_strengths, vqi, oqs FROM restoration_history "
                "WHERE era BETWEEN ? AND ? AND material=? AND label_hint=? AND mode=? "
                "ORDER BY vqi DESC LIMIT ?",
                (decade, decade + 9, mat, lbl, mode, MAX_LOOKUP_RECORDS),
            )
            for ps_json, vqi, oqs in cur.fetchall():
                try:
                    ps = json.loads(ps_json)
                    rows.append((0.75, ps, float(vqi), float(oqs)))
                except Exception:
                    logger.debug("_lookup_impl: silent except suppressed", exc_info=True)

        # Tier 2b: era-decade + material + genre (genre-informed but weaker than label)
        if genre_key:
            cur = conn.execute(
                "SELECT phase_strengths, vqi, oqs FROM restoration_history "
                "WHERE era BETWEEN ? AND ? AND material=? AND genre=? AND mode=? "
                "ORDER BY vqi DESC LIMIT ?",
                (decade, decade + 9, mat, genre_key, mode, MAX_LOOKUP_RECORDS),
            )
            for ps_json, vqi, oqs in cur.fetchall():
                try:
                    ps = json.loads(ps_json)
                    rows.append((0.65, ps, float(vqi), float(oqs)))
                except Exception:
                    logger.debug("_lookup_impl: silent except suppressed", exc_info=True)

        # Tier 3: era-decade + material
        cur = conn.execute(
            "SELECT phase_strengths, vqi, oqs FROM restoration_history "
            "WHERE era BETWEEN ? AND ? AND material=? AND mode=? "
            "ORDER BY vqi DESC LIMIT ?",
            (decade, decade + 9, mat, mode, MAX_LOOKUP_RECORDS),
        )
        for ps_json, vqi, oqs in cur.fetchall():
            try:
                ps = json.loads(ps_json)
                rows.append((0.55, ps, float(vqi), float(oqs)))
            except Exception:
                logger.debug("_lookup_impl: silent except suppressed", exc_info=True)

        # Tier 4: era-decade only (broadest fallback)
        if len(rows) < 3:
            cur = conn.execute(
                "SELECT phase_strengths, vqi, oqs FROM restoration_history "
                "WHERE era BETWEEN ? AND ? AND mode=? "
                "ORDER BY vqi DESC LIMIT ?",
                (decade, decade + 9, mode, MAX_LOOKUP_RECORDS),
            )
            for ps_json, vqi, oqs in cur.fetchall():
                try:
                    ps = json.loads(ps_json)
                    rows.append((0.35, ps, float(vqi), float(oqs)))
                except Exception:
                    logger.debug("_lookup_impl: silent except suppressed", exc_info=True)

        if not rows:
            return AKBPrior()

        # Aggregate: weighted mean of phase strengths (weight = match_score × vqi)
        phase_sums: dict[str, float] = {}
        phase_weights: dict[str, float] = {}
        total_vqi = 0.0
        total_oqs = 0.0
        total_weight = 0.0

        for match_score, ps, vqi, oqs in rows:
            if vqi < MIN_VQI_RECORD:
                continue
            w = match_score * float(np.clip(vqi, 0.0, 1.0))
            for phase_id, strength in ps.items():
                phase_sums[phase_id] = phase_sums.get(phase_id, 0.0) + w * float(strength)
                phase_weights[phase_id] = phase_weights.get(phase_id, 0.0) + w
            total_vqi += vqi * w
            total_oqs += oqs * w
            total_weight += w

        if total_weight < 1e-8:
            return AKBPrior()

        phase_strengths_out = {
            pid: float(np.clip(phase_sums[pid] / (phase_weights[pid] + 1e-12), 0.0, 1.0)) for pid in phase_sums
        }
        n_valid = sum(1 for _, _, v, _ in rows if v >= MIN_VQI_RECORD)
        confidence = float(np.clip(n_valid / STRONG_PRIOR_N, 0.0, 1.0))
        mean_vqi = float(total_vqi / total_weight)
        mean_oqs = float(total_oqs / total_weight)

        logger.debug(
            "§AKB-1 lookup_prior: era=%d mat=%s → n=%d conf=%.2f mean_vqi=%.3f",
            era,
            mat,
            n_valid,
            confidence,
            mean_vqi,
        )
        return AKBPrior(
            phase_strengths=phase_strengths_out,
            confidence=confidence,
            n_records=n_valid,
            mean_vqi=mean_vqi,
            mean_oqs=mean_oqs,
        )

    @staticmethod
    def _insert_record(
        conn: sqlite3.Connection,
        era: int,
        material: str,
        label_hint: str,
        artist_hash: str,
        phase_strengths: dict[str, float],
        vqi: float,
        oqs: float,
        genre: str,
        mode: str,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        ps_json = json.dumps({k: float(v) for k, v in phase_strengths.items()}, ensure_ascii=True)
        conn.execute(
            "INSERT INTO restoration_history "
            "(era, material, label_hint, artist_hash, genre, mode, phase_strengths, vqi, oqs, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(era),
                material.lower(),
                label_hint.lower(),
                artist_hash,
                genre,
                mode,
                ps_json,
                float(vqi),
                float(oqs),
                ts,
            ),
        )
        conn.commit()
        logger.debug(
            "§AKB-1 aufzeichnen_outcome: era=%d mat=%s vqi=%.3f oqs=%.1f stored",
            era,
            material,
            vqi,
            oqs,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: ArtistKnowledgeBase | None = None
_lock = threading.Lock()


def get_artist_knowledge_base() -> ArtistKnowledgeBase:
    """Thread-safe singleton (double-checked locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ArtistKnowledgeBase()
    return _instance


__all__ = [
    "AKBPrior",
    "ArtistKnowledgeBase",
    "get_artist_knowledge_base",
]
