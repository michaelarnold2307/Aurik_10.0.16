"""Song-specific strategy persistence for Aurik 10.0.0.

A human mastering engineer remembers their session notes:
"Last time Phase 42 before Phase 29 worked better — and strength 0.65
was better than 1.0 for this song."

This cache stores persistent phase-order and strength presets per
song (identified via file hash + mode) and provides UV3 with a warm-start
plan on the next run.

File: ~/.config/aurik/song_strategy_cache.json (max 500 entries, LRU eviction)

Spec: §SSC-1 Song-Strategy-Cache (v10.0.0)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.path.expanduser("~/.config/aurik"))
_CACHE_FILE = _CACHE_DIR / "song_strategy_cache.json"
_MAX_ENTRIES = 500  # LRU eviction threshold
_AUDIO_FINGERPRINT_SAMPLES = 4096  # samples used for audio fingerprint
_MIN_CONFIDENCE_TO_STORE = 0.5  # only store if HPI improvement ≥ 5 %


@dataclass
class PhaseStrategyEntry:
    """Persisted phase configuration for a song."""

    # Identification
    song_id: str  # SHA256[:16] of the audio fingerprint
    mode: str  # "restoration" | "studio_2026"
    last_used: float  # Unix timestamp
    use_count: int  # How often has this entry been used?

    # Strategy data
    phase_strength_overrides: dict[str, float]  # phase_id → strength
    hpi_achieved: float  # Last achieved HPI score
    vqi_achieved: float  # Last achieved VQI score (0.0 if no vocal content)
    oqs_achieved: float  # Last OQS score

    # Meta
    era: str = ""
    genre: str = ""
    material: str = ""
    confidence: float = 0.5  # 0.0–1.0: how reliable is this strategy?
    notes: str = ""


class SongStrategyCache:
    """Thread-safe persistent strategy cache.

    Implements LRU eviction: when > MAX_ENTRIES entries exist, the oldest
    (by last_used) entries are removed.
    """

    def __init__(self, cache_file: Path = _CACHE_FILE) -> None:
        self._cache_file = cache_file
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            if self._cache_file.exists():
                with open(self._cache_file, encoding="utf-8") as f:
                    raw = json.load(f)
                    if isinstance(raw, dict):
                        self._data = raw
        except Exception as exc:
            logger.debug("§SSC-1 Zwischenspeicher-Lade-Fehler (nicht blockierend): %s", exc)
            self._data = {}
        self._loaded = True

    def _save(self) -> None:
        """Speichert cache to disk — LRU eviction if entry limit exceeded."""
        try:
            if len(self._data) > _MAX_ENTRIES:
                # Remove oldest entries (by last_used)
                sorted_keys = sorted(
                    self._data.keys(),
                    key=lambda k: float(self._data[k].get("last_used", 0.0)),
                )
                to_remove = len(self._data) - _MAX_ENTRIES
                for k in sorted_keys[:to_remove]:
                    del self._data[k]
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.debug("§SSC-1 Zwischenspeicher-Speicher-Fehler (nicht blockierend): %s", exc)

    def _make_cache_key(self, song_id: str, mode: str) -> str:
        return f"{song_id}::{mode}"

    def get(self, song_id: str, mode: str) -> PhaseStrategyEntry | None:
        """Gibt stored strategy, or None if not present zurück."""
        with self._lock:
            self._ensure_loaded()
            key = self._make_cache_key(song_id, mode)
            raw = self._data.get(key)
            if raw is None:
                return None
            try:
                entry = PhaseStrategyEntry(**raw)
                # Update timestamp
                self._data[key]["last_used"] = time.time()
                self._data[key]["use_count"] = int(raw.get("use_count", 0)) + 1
                return entry
            except Exception as exc:
                logger.debug("§SSC-1 Zwischenspeicher read error for %s: %s", key, exc)
                return None

    def store(self, entry: PhaseStrategyEntry) -> None:
        """Store or update a strategy entry.

        Only stored if confidence ≥ _MIN_CONFIDENCE_TO_STORE.
        """
        if entry.confidence < _MIN_CONFIDENCE_TO_STORE:
            logger.debug(
                "§SSC-1 Zwischenspeicher store uebersprungen: confidence=%.2f < %.2f",
                entry.confidence,
                _MIN_CONFIDENCE_TO_STORE,
            )
            return
        with self._lock:
            self._ensure_loaded()
            key = self._make_cache_key(entry.song_id, entry.mode)
            # Merge with existing entry (higher HPI wins)
            existing = self._data.get(key)
            if existing is not None:
                existing_hpi = float(existing.get("hpi_achieved", 0.0))
                if entry.hpi_achieved <= existing_hpi * 0.98:
                    # Neue Strategie nicht besser — update nur Timestamps
                    self._data[key]["last_used"] = time.time()
                    self._data[key]["use_count"] = int(existing.get("use_count", 0)) + 1
                    return
            raw = asdict(entry)
            raw["last_used"] = time.time()
            self._data[key] = raw
            self._save()
            logger.info(
                "§SSC-1 strategy stored: song_id=%s Betriebsart=%s HPI=%.3f VQI=%.3f OQS=%.1f confidence=%.2f",
                entry.song_id[:8],
                entry.mode,
                entry.hpi_achieved,
                entry.vqi_achieved,
                entry.oqs_achieved,
                entry.confidence,
            )

    def size(self) -> int:
        """Gibt the number of cached strategy entries zurück."""
        with self._lock:
            self._ensure_loaded()
            return len(self._data)


# ---------------------------------------------------------------------------
# Audio-Fingerprint
# ---------------------------------------------------------------------------


def compute_audio_fingerprint(audio: np.ndarray, sr: int) -> str:
    """Berechnet a stable SHA256 fingerprint for an audio signal.

    Method: RMS-normalised downsampling to 4096 samples → SHA256.
    Robust against small level differences (LUFS normalisation).

    Returns:
        SHA256[:16] hex string
    """
    try:
        audio_f32 = np.asarray(audio, dtype=np.float32)
        # Mono
        if audio_f32.ndim == 2:
            audio_f32 = audio_f32.mean(axis=0) if audio_f32.shape[0] <= 2 else audio_f32.mean(axis=1)

        # RMS-Normalisierung
        rms = float(np.sqrt(np.mean(audio_f32**2)))
        if rms > 1e-9:
            audio_f32 = audio_f32 / rms

        n = len(audio_f32)
        if n < _AUDIO_FINGERPRINT_SAMPLES:
            # Zu kurz — zero-pad
            audio_f32 = np.pad(audio_f32, (0, _AUDIO_FINGERPRINT_SAMPLES - n))

        # Uniform downsampling to _AUDIO_FINGERPRINT_SAMPLES
        indices = np.linspace(0, n - 1, _AUDIO_FINGERPRINT_SAMPLES, dtype=int)
        fingerprint_vec = audio_f32[indices]

        # Quantise to int16 for hash stability
        fingerprint_int = (fingerprint_vec * 32767).astype(np.int16)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
        fingerprint_payload = f"sr={int(sr)}\n".encode("ascii") + fingerprint_int.tobytes()
        h = hashlib.sha256(fingerprint_payload).hexdigest()
        return h[:16]
    except Exception as exc:
        logger.debug("audio fingerprint fehlgeschlagen: %s", exc)
        # Fallback: time-based (no cache hit guaranteed)
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]


def build_strategy_entry_from_result(
    song_id: str,
    mode: str,
    phase_scores: dict[str, float],  # phase_id → achieved strength that led to best score
    hpi: float,
    vqi: float,
    oqs: float,
    era: str = "",
    genre: str = "",
    material: str = "",
) -> PhaseStrategyEntry:
    # pylint: disable=too-many-positional-arguments
    """Erstellt a PhaseStrategyEntry from pipeline result data.

    Confidence is derived from HPI + OQS:
    - HPI > 0.85 + OQS > 80 → confidence = 0.9
    - HPI > 0.70 + OQS > 70 → confidence = 0.7
    - otherwise → confidence = 0.5
    """
    if hpi > 0.85 and oqs > 80.0:
        confidence = 0.9
    elif hpi > 0.70 and oqs > 70.0:
        confidence = 0.7
    else:
        confidence = 0.5

    return PhaseStrategyEntry(
        song_id=song_id,
        mode=mode,
        last_used=time.time(),
        use_count=0,
        phase_strength_overrides=dict(phase_scores),
        hpi_achieved=float(hpi),
        vqi_achieved=float(vqi),
        oqs_achieved=float(oqs),
        era=era,
        genre=genre,
        material=material,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: SongStrategyCache | None = None
_lock = threading.Lock()


def get_song_strategy_cache() -> SongStrategyCache:
    """Gibt the singleton SongStrategyCache (thread-safe double-checked locking) zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SongStrategyCache()
    return _instance
