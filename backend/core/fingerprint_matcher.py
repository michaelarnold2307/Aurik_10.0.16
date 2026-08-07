"""§AB: Audio-Fingerprint-Matching — Cross-Recording Parameter Transfer.

Wenn Aurik eine Aufnahme restauriert, speichert es einen kompakten
Audio-Fingerprint (Spektrum, Defekte, Dynamik, Material). Beim nächsten
Restaurierungsauftrag wird der Fingerprint mit allen gespeicherten
verglichen. Bei Übereinstimmung werden die erfolgreichen Parameter
der vorherigen Restaurierung übernommen.

Dadurch lernt Aurik aus jeder Restaurierung und wird mit jedem Lauf besser.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_FINGERPRINT_DB = Path(__file__).parent.parent.parent / "data" / "fingerprints.json"


@dataclass
class AudioFingerprint:
    """Kompakter akustischer Fingerabdruck einer Aufnahme."""

    spectral_hash: str = ""  # SHA256 der spektralen Kontur
    defect_signature: str = ""  # Kombination der Defekt-Typen
    material: str = ""
    genre: str = ""
    duration_s: float = 0.0
    rms_db: float = 0.0
    spectral_centroid_hz: float = 0.0
    dynamic_range_db: float = 0.0
    stereo_width: float = 0.0

    def similarity_key(self) -> str:
        """Kombinierter Key für Matching: spectral + defects + material."""
        return f"{self.spectral_hash[:16]}:{self.defect_signature}:{self.material}"


class FingerprintMatcher:
    """Speichert und matcht Audio-Fingerprints für Parameter-Transfer."""

    def __init__(self) -> None:
        self._db: dict[str, Any] = self._load_db()

    def _load_db(self) -> dict[str, Any]:
        try:
            _FINGERPRINT_DB.parent.mkdir(parents=True, exist_ok=True)
            if _FINGERPRINT_DB.exists():
                with open(_FINGERPRINT_DB) as f:
                    return json.load(f)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("fingerprint_matcher.py::_laden_db Ersatzpfad: %s", e)
        return {"fingerprints": {}, "parameters": {}}

    def _save_db(self) -> None:
        try:
            with open(_FINGERPRINT_DB, "w") as f:
                json.dump(self._db, f, indent=2, default=str)
        except Exception as e:
            logger.debug("§AB Fingerprint speichern fehlgeschlagen: %s", e)

    def compute_fingerprint(
        self, audio: np.ndarray, sr: int, material: str = "", genre: str = "", defect_types: list[str] | None = None
    ) -> AudioFingerprint:
        """Berechnet einen Fingerprint aus Audio + Metadaten."""
        try:
            mono = np.mean(audio, axis=0) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)

            # Spektrale Kontur (10-Band-Approximation)
            fft = np.abs(np.fft.rfft(mono, n=min(65536, len(mono))))
            freqs = np.fft.rfftfreq(min(65536, len(mono)), d=1.0 / sr)
            bands = [
                (20, 60),
                (60, 200),
                (200, 500),
                (500, 1000),
                (1000, 2000),
                (2000, 4000),
                (4000, 8000),
                (8000, 16000),
            ]
            contour = [float(np.sum(fft[(freqs >= lo) & (freqs <= hi)])) for lo, hi in bands]
            total = float(np.sum(fft)) + 1e-10
            contour_norm = [c / total for c in contour]
            spectral_hash = hashlib.sha256(",".join(f"{c:.4f}" for c in contour_norm).encode()).hexdigest()

            # Defekt-Signatur
            defects = sorted(defect_types or [])
            defect_signature = hashlib.md5(",".join(defects).encode(), usedforsecurity=False).hexdigest()[:16]

            # Akustische Metriken
            power = np.mean(mono * mono) + 1e-12
            rms_db = 10.0 * np.log10(float(power))
            centroid = float(np.average(freqs, weights=fft + 1e-10))
            peak = float(np.max(np.abs(mono)))
            dynamic_range = 20.0 * np.log10(peak / np.sqrt(power) + 1e-12)

            # Stereo-Breite
            stereo_width = 1.0
            if audio.ndim == 2 and audio.shape[0] >= 2:
                L, R = audio[0], audio[1]
                M, S = (L + R) / 2, (L - R) / 2
                stereo_width = float(np.clip(np.mean(S * S) / (np.mean(M * M) + 1e-12), 0.0, 1.0))

            return AudioFingerprint(
                spectral_hash=spectral_hash,
                defect_signature=defect_signature,
                material=material,
                genre=genre,
                duration_s=len(mono) / sr,
                rms_db=rms_db,
                spectral_centroid_hz=centroid,
                dynamic_range_db=dynamic_range,
                stereo_width=stereo_width,
            )
        except Exception as e:
            logger.debug("§AB Fingerprint computation fehlgeschlagen: %s", e)
            return AudioFingerprint()

    def store_result(self, fp: AudioFingerprint, parameters: dict[str, Any]) -> None:
        """Speichert erfolgreiche Parameter zu einem Fingerprint."""
        key = fp.similarity_key()
        self._db["fingerprints"][key] = {
            "spectral_hash": fp.spectral_hash,
            "defect_signature": fp.defect_signature,
            "material": fp.material,
            "genre": fp.genre,
            "timestamp": time.time(),
        }
        self._db["parameters"][key] = {
            "phase_strengths": parameters.get("phase_strengths", {}),
            "eq_profile": parameters.get("eq_profile", {}),
            "goal_weights": parameters.get("goal_weights", {}),
            "pmgg_scores": parameters.get("pmgg_scores", {}),
        }
        self._save_db()
        logger.info("§AB Fingerprint gespeichert: %s (total: %d)", key[:40], len(self._db["fingerprints"]))

    def find_match(self, fp: AudioFingerprint, min_similarity: float = 0.7) -> dict[str, Any] | None:
        """Sucht nach ähnlichen Fingerprints und gibt gespeicherte Parameter zurück."""
        best_score = 0.0
        best_params = None

        for key, stored in self._db["fingerprints"].items():
            score = 0.0
            # Gleiches Material = +40%
            if stored.get("material") == fp.material:
                score += 0.4
            # Gleiches Genre = +20%
            if stored.get("genre") == fp.genre:
                score += 0.2
            # Ähnliche Defekte = +25%
            if stored.get("defect_signature", "")[:8] == fp.defect_signature[:8]:
                score += 0.25
            # Ähnliches Spektrum = +15%
            if stored.get("spectral_hash", "")[:8] == fp.spectral_hash[:8]:
                score += 0.15

            if score > best_score:
                best_score = score
                best_params = self._db["parameters"].get(key)

        if best_score >= min_similarity and best_params:
            logger.info("§AB Match gefunden: similarity=%.0f%% → Parameter transferiert", best_score * 100)
            return dict(best_params)

        logger.debug("§AB Kein Match (best=%.0f%% < %.0f%%)", best_score * 100, min_similarity * 100)
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "total_fingerprints": len(self._db.get("fingerprints", {})),
            "total_parameters": len(self._db.get("parameters", {})),
        }
