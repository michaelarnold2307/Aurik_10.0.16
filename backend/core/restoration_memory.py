"""RestorationMemory — Persistente GPOptimizer-Priors (§2.70, v10.0.0).

Speichert erfolgreiche Restaurierungs-Läufe (era × material × defect_cluster_hash)
als JSON unter ~/.aurik/restoration_memory.json und stellt sie als Prior für den
GPOptimizer zur Verfügung. Nur erfolgreiche Läufe werden gespeichert
(HPI > 0 AND artifact_freedom >= 0.95).

Design:
    - Singleton (thread-safe, double-checked locking).
    - Atomarer Schreibvorgang: .tmp → os.replace().
    - LRU-Eviction bei Dateigröße > 10 MB.
    - Kein Crash bei korrupter Datei (Fallback → leeres Dict).

Kanonische Nutzung:
    from backend.core.restoration_memory import get_restoration_memory
    mem = get_restoration_memory()

    # Vor GPOptimizer:
    prior = mem.get_prior((era, material, cluster_hash))  # None wenn kein Prior

    # Nach HolisticPerceptualGate (HPI > 0 AND artifact_freedom >= 0.95):
    mem.save_result(key=(era, material, cluster_hash), phase_params={...}, hpi_achieved=0.81)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximale Dateigröße in Bytes (10 MB). Bei Überschreitung → LRU-Eviction.
_MAX_FILE_BYTES: int = 10 * 1024 * 1024
# Pfad zur persistenten Datei (§Pfad-Mapping).
_DEFAULT_MEMORY_PATH: Path = Path.home() / ".aurik" / "restoration_memory.json"

_instance: RestorationMemory | None = None
_lock: threading.Lock = threading.Lock()


def _make_key_str(key: tuple[Any, ...]) -> str:
    """Wandelt einen Tupel-Schlüssel in einen stabilen String-Key um."""
    raw = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16] + "__" + raw[:64]


def _make_audio_fingerprint(audio: np.ndarray, sr: int) -> str:
    """Erstellt einen robusten Fingerabdruck für die ersten 5 Sekunden des Audios.

    Merkmale: ZCR, Spektral-Schwerpunkt, RMS-Energie-Profil (8 Frames).
    Zweck: 'Selbe Aufnahme'-Erkennung unabhängig vom Metadaten-Schlüssel (§2.70-Upgrade P5).

    Returns:
        16-stelliger Hex-String oder Fallback-Literal bei Fehler.
    """
    try:
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
        n_samples = min(len(mono), int(5.0 * sr))
        if n_samples < 512:
            return "short_audio"
        mono = np.nan_to_num(mono[:n_samples].astype(np.float32), nan=0.0)
        # ZCR — Nulldurchgangsrate als grobes Tonart-Merkmal
        zcr = float(np.abs(np.diff(np.sign(mono))).sum() / max(len(mono), 1))
        # Spektraler Schwerpunkt — dominanter Frequenzbereich
        n_fft = min(2048, len(mono))
        spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-10))
        # RMS-Energie-Profil: 8 gleichmäßige Frames → Energie-Kurve der Aufnahme
        frame_len = max(512, len(mono) // 8)
        rms_vals = [
            float(np.sqrt(np.mean(mono[i : i + frame_len] ** 2) + 1e-10))
            for i in range(0, min(len(mono), frame_len * 8), frame_len)
        ]
        rms_str = "_".join(f"{int(v * 1000):04d}" for v in rms_vals[:8])
        fp_str = f"{zcr:.4f}_{centroid:.1f}_{rms_str}"
        return hashlib.sha256(fp_str.encode()).hexdigest()[:16]
    except Exception as e:
        logger.warning("restoration_memory.py::_make_audio_fingerprint Ersatzpfad: %s", e)
        return "fingerprint_error"


class RestorationMemory:
    """Persistente Priors für GPOptimizer (§2.70).

    Nur über get_restoration_memory() instantiieren.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path: Path = path or _DEFAULT_MEMORY_PATH
        self._data: dict[str, Any] = {}
        self._fingerprint_index: dict[str, str] = {}  # fingerprint → key_str (§P5 Audio-Fingerprint)
        self._dirty: bool = False
        self._internal_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "prior_requests": 0,
            "prior_hits": 0,
            "prior_misses": 0,
            "save_attempts": 0,
            "save_success": 0,
            "save_rejected_hpi": 0,
            "save_rejected_not_better": 0,
        }
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prior(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        """Gibt gespeicherten Prior für einen (era, material, cluster_hash)-Schlüssel zurück.

        Returns:
            Dict mit 'phase_params' und 'hpi_achieved' oder None wenn kein Prior vorhanden.
        """
        key_str = _make_key_str(key)
        with self._internal_lock:
            self._stats["prior_requests"] += 1
            entry = self._data.get(key_str)
        if entry is None:
            with self._internal_lock:
                self._stats["prior_misses"] += 1
            return None
        with self._internal_lock:
            self._stats["prior_hits"] += 1
        # Zugriffszeit für LRU aktualisieren (in-memory only, kein sofortiger Schreibvorgang)
        entry["_last_access"] = time.time()
        return {
            "phase_params": entry.get("phase_params", {}),
            "hpi_achieved": float(entry.get("hpi_achieved", 0.0)),
        }

    def get_stats(self) -> dict[str, int]:
        """Liefert Laufzeit-Telemetrie zur Prior-Wirksamkeit (§2.70, non-blocking)."""
        with self._internal_lock:
            return dict(self._stats)

    def find_prior_by_audio(self, audio: np.ndarray, sr: int) -> dict[str, Any] | None:
        """Findet einen Prior anhand des Audio-Fingerprints (selbe Aufnahme, §P5).

        Ergänzt get_prior(): Wenn der Metadaten-Schlüssel variiert (z.B. anderes
        Defekt-Cluster-Hash bei identischer Aufnahme), kann dieser Pfad trotzdem
        den richtigen Prior liefern.

        Args:
            audio: Audiodaten (mono oder stereo).
            sr:    Abtastrate in Hz.

        Returns:
            Dict mit 'phase_params', 'hpi_achieved' und 'matched_by="audio_fingerprint"',
            oder None wenn kein passender Fingerabdruck gefunden.
        """
        fingerprint = _make_audio_fingerprint(audio, sr)
        if fingerprint in {"short_audio", "fingerprint_error"}:
            return None
        with self._internal_lock:
            key_str = self._fingerprint_index.get(fingerprint)
        if key_str is None:
            with self._internal_lock:
                self._stats["prior_misses"] += 1
            return None
        with self._internal_lock:
            entry = self._data.get(key_str)
        if entry is None:
            return None
        with self._internal_lock:
            self._stats["prior_hits"] += 1
        entry["_last_access"] = time.time()
        return {
            "phase_params": entry.get("phase_params", {}),
            "hpi_achieved": float(entry.get("hpi_achieved", 0.0)),
            "matched_by": "audio_fingerprint",
        }

    def save_result(
        self,
        key: tuple[Any, ...],
        phase_params: dict[str, Any],
        hpi_achieved: float,
    ) -> None:
        """Speichert ein erfolgreiches Ergebnis als Prior.

        Nur aufrufen wenn HPI > 0 AND artifact_freedom >= 0.95 (§2.70).

        Args:
            key:           (era, material, defect_cluster_hash)-Tupel.
            phase_params:  Phasen-Parameter-Dict aus GPOptimizer-Ergebnis.
            hpi_achieved:  HPI-Score dieses Laufs.
        """
        self._save_result_internal(key, phase_params, hpi_achieved, audio=None, sr=0)

    def save_result_with_audio(
        self,
        key: tuple[Any, ...],
        audio: np.ndarray,
        sr: int,
        phase_params: dict[str, Any],
        hpi_achieved: float,
    ) -> None:
        """Wie save_result(), aber speichert zusätzlich den Audio-Fingerprint (§P5).

        Ermöglicht spätere Prior-Suche via find_prior_by_audio() unabhängig vom
        Metadaten-Schlüssel — nützlich wenn derselbe Song mit anderem Defekt-Hash
        erneut verarbeitet wird.

        Args:
            key:          (era, material, defect_cluster_hash)-Tupel.
            audio:        Restauriertes Audio (zur Fingerabdruck-Berechnung).
            sr:           Abtastrate in Hz.
            phase_params: Phasen-Parameter-Dict aus GPOptimizer-Ergebnis.
            hpi_achieved: HPI-Score dieses Laufs.
        """
        self._save_result_internal(key, phase_params, hpi_achieved, audio=audio, sr=sr)

    def _save_result_internal(
        self,
        key: tuple[Any, ...],
        phase_params: dict[str, Any],
        hpi_achieved: float,
        audio: np.ndarray | None,
        sr: int,
    ) -> None:
        """Interne Implementierung für save_result und save_result_with_audio."""
        with self._internal_lock:
            self._stats["save_attempts"] += 1

        if hpi_achieved <= 0.0:
            with self._internal_lock:
                self._stats["save_rejected_hpi"] += 1
            logger.debug("RestorationMemory: HPI <= 0 → nicht gespeichert (key=%s)", key)
            return

        key_str = _make_key_str(key)
        entry: dict[str, Any] = {
            "phase_params": phase_params,
            "hpi_achieved": float(hpi_achieved),
            "_timestamp": time.time(),
            "_last_access": time.time(),
        }

        with self._internal_lock:
            existing = self._data.get(key_str)
            # Nur überschreiben wenn neuer Score besser als gespeicherter
            if existing is not None and float(existing.get("hpi_achieved", 0.0)) >= hpi_achieved:
                self._stats["save_rejected_not_better"] += 1
                logger.debug(
                    "RestorationMemory: Vorhandener Prior HPI=%.3f >= %.3f → nicht überschrieben",
                    existing.get("hpi_achieved", 0.0),
                    hpi_achieved,
                )
                return
            self._data[key_str] = entry
            self._dirty = True
            self._stats["save_success"] += 1

        self._persist()

        # §P5 Audio-Fingerprint-Index: wenn Audio übergeben → in Index eintragen für
        # spätere Suche via find_prior_by_audio() (selbe Aufnahme, anderer Defekt-Hash).
        if audio is not None and getattr(audio, "size", 0) > 0:
            try:
                _fp = _make_audio_fingerprint(audio, int(sr))
                if _fp not in {"short_audio", "fingerprint_error"}:
                    with self._internal_lock:
                        self._fingerprint_index[_fp] = key_str
            except Exception as e:
                logger.warning("restoration_memory.py::_speichern_Ergebnis_internal Ersatzpfad: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lädt die Memory-Datei; ignoriert Fehler (leeres Dict als Fallback)."""
        try:
            if self._path.exists():
                raw = self._path.read_bytes()
                loaded = json.loads(raw)
                if not isinstance(loaded, dict):
                    loaded = {}
                # §P5: Fingerabdruck-Index separat extrahieren
                self._fingerprint_index = loaded.pop("__fingerprint_index__", {})
                if not isinstance(self._fingerprint_index, dict):
                    self._fingerprint_index = {}
                self._data = loaded
                logger.debug("RestorationMemory: %d Einträge geladen aus %s", len(self._data), self._path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("RestorationMemory: Ladevorgang fehlgeschlagen (nicht blockierend): %s", exc)
            self._data = {}
            self._fingerprint_index = {}

    def _persist(self) -> None:
        """Schreibt _data atomar in die JSON-Datei (§2.70 Atomic-Write-Invariante)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._internal_lock:
                data_copy = dict(self._data)
                # §P5: Fingerabdruck-Index in JSON einbetten (Sonder-Schlüssel)
                if self._fingerprint_index:
                    data_copy["__fingerprint_index__"] = dict(self._fingerprint_index)

            payload = json.dumps(data_copy, indent=2, default=str)
            encoded = payload.encode("utf-8")

            # LRU-Eviction wenn > 10 MB
            if len(encoded) > _MAX_FILE_BYTES:
                data_copy = self._evict_lru(data_copy)
                payload = json.dumps(data_copy, indent=2, default=str)
                encoded = payload.encode("utf-8")
                with self._internal_lock:
                    self._data = data_copy

            # Atomarer Schreibvorgang: tmp-Datei → os.replace
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_bytes(encoded)
            os.replace(str(tmp_path), str(self._path))
            self._dirty = False
            logger.debug("RestorationMemory: %d Einträge geschrieben nach %s", len(data_copy), self._path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("RestorationMemory: Schreibvorgang fehlgeschlagen (nicht blockierend): %s", exc)

    @staticmethod
    def _evict_lru(data: dict[str, Any]) -> dict[str, Any]:
        """Entfernt die ältesten 20 % der Einträge (LRU nach _last_access)."""
        if not data:
            return data
        sorted_keys = sorted(data.keys(), key=lambda k: float(data[k].get("_last_access", 0.0)))
        evict_count = max(1, len(sorted_keys) // 5)
        for k in sorted_keys[:evict_count]:
            del data[k]
        logger.info("RestorationMemory LRU-Eviction: %d Einträge entfernt", evict_count)
        return data


def get_restoration_memory() -> RestorationMemory:
    """Thread-sicherer Singleton-Getter für RestorationMemory (§Singleton-Pattern)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RestorationMemory()
    return _instance
