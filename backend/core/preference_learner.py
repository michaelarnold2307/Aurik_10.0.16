"""§V+W: Reference-Track-Kalibrierung + Blind-A/B-Preference-Learning.

V: ReferenceTrackMatcher
   - Analysiert eine Referenz-Audiodatei („so soll es klingen")
   - Extrahiert Zielkurve: Spektrum, Dynamik, Stereo-Breite, LUFS
   - Passt Goal-Targets und EQ-Parameter an die Referenz an

W: PreferenceLearner
   - Speichert A/B-Entscheidungen des Nutzers
   - Lernt aus Präferenzen: welche Parameter-Kombinationen bevorzugt werden
   - Beeinflusst zukünftige Restaurierungen (Genre + Material + Präferenz-Matrix)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── §V Reference-Track-Kalibrierung ───────────────────────────────────────────


class ReferenceTrackMatcher:
    """Analysiert eine Referenz-Audiodatei und erstellt ein Ziel-Profil."""

    def __init__(self) -> None:
        self._profile: dict[str, Any] = {}

    def analyze(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Extrahiert das Klangprofil der Referenz."""
        try:
            mono = np.mean(audio, axis=0) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)

            # 1. Spektrale Balance (10-Band-Approximation)
            bands = [
                (20, 60),
                (60, 200),
                (200, 500),
                (500, 1000),
                (1000, 2000),
                (2000, 4000),
                (4000, 6000),
                (6000, 8000),
                (8000, 12000),
                (12000, 20000),
            ]
            spectrum: dict[str, float] = {}
            fft = np.abs(np.fft.rfft(mono, n=min(65536, len(mono))))
            freqs = np.fft.rfftfreq(min(65536, len(mono)), d=1.0 / sr)
            total = float(np.sum(fft)) + 1e-10
            for low, high in bands:
                mask = (freqs >= low) & (freqs <= high)
                spectrum[f"{low}-{high}"] = float(np.sum(fft[mask]) / total)

            # 2. Dynamik (PLR: Peak-to-Loudness Ratio)
            power = np.mean(mono * mono) + 1e-12
            peak = float(np.max(np.abs(mono)))
            plr = 20.0 * np.log10(peak / np.sqrt(power) + 1e-12)

            # 3. Stereo-Breite (wenn Stereo)
            stereo_width = 1.0
            if audio.ndim == 2 and audio.shape[0] >= 2:
                L, R = np.asarray(audio[0], dtype=np.float64), np.asarray(audio[1], dtype=np.float64)
                M = (L + R) / 2.0
                S = (L - R) / 2.0
                side_power = np.mean(S * S) + 1e-12
                mid_power = np.mean(M * M) + 1e-12
                stereo_width = float(np.clip(side_power / (mid_power + side_power + 1e-12), 0.0, 1.0))

            # 4. Integrated LUFS
            lufs = -0.691 + 10.0 * np.log10(power)

            self._profile = {
                "spectrum": spectrum,
                "plr_db": round(plr, 2),
                "stereo_width": round(stereo_width, 3),
                "integrated_lufs": round(lufs, 1),
                "peak_db": round(20.0 * np.log10(peak + 1e-12), 1),
                "duration_s": len(mono) / sr,
                "analyzed_at": time.time(),
            }
            return dict(self._profile)
        except Exception as e:
            logger.warning("§V ReferenceTrackMatcher.analyze fehlgeschlagen: %s", e)
            return {}

    def compute_goal_adjustments(self, source_profile: dict[str, Any]) -> dict[str, float]:
        """Berechnet Goal-Target-Anpassungen basierend auf Referenz-Matching."""
        if not self._profile or not source_profile:
            return {}

        adjustments: dict[str, float] = {}
        ref_spec = self._profile.get("spectrum", {})
        src_spec = source_profile.get("spectrum", {})

        if ref_spec and src_spec:
            # Wärme: 200-500 Hz Verhältnis
            warmth_ref = ref_spec.get("200-500", 0.1)
            warmth_src = src_spec.get("200-500", 0.1)
            if warmth_src > 0:
                ratio = warmth_ref / max(warmth_src, 0.01)
                adjustments["waerme"] = float(np.clip(1.0 + (ratio - 1.0) * 0.5, 0.7, 1.3))

            # Brillanz: 2k-8k Verhältnis
            brill_ref = (
                ref_spec.get("2000-4000", 0.05) + ref_spec.get("4000-6000", 0.05) + ref_spec.get("6000-8000", 0.05)
            )
            brill_src = (
                src_spec.get("2000-4000", 0.05) + src_spec.get("4000-6000", 0.05) + src_spec.get("6000-8000", 0.05)
            )
            if brill_src > 0:
                ratio_b = brill_ref / max(brill_src, 0.01)
                adjustments["brillanz"] = float(np.clip(1.0 + (ratio_b - 1.0) * 0.5, 0.7, 1.3))

            # Bass: 20-200 Hz
            bass_ref = ref_spec.get("20-60", 0.05) + ref_spec.get("60-200", 0.1)
            bass_src = src_spec.get("20-60", 0.05) + src_spec.get("60-200", 0.1)
            if bass_src > 0:
                ratio_c = bass_ref / max(bass_src, 0.01)
                adjustments["bass_praesenz"] = float(np.clip(1.0 + (ratio_c - 1.0) * 0.4, 0.6, 1.4))

        # Dynamik anpassen
        ref_plr = self._profile.get("plr_db", 12.0)
        src_plr = source_profile.get("plr_db", 12.0)
        if ref_plr > src_plr + 3:
            adjustments["makrodynamik"] = 1.2

        # Stereo-Breite
        ref_width = self._profile.get("stereo_width", 0.5)
        src_width = source_profile.get("stereo_width", 0.5)
        if ref_width > src_width + 0.1:
            adjustments["raeumlichkeit"] = 1.2

        return adjustments


# ── §W Blind-A/B-Preference-Learning ──────────────────────────────────────────

_PREFERENCE_FILE = Path(__file__).parent.parent.parent / "data" / "preferences.json"


class PreferenceLearner:
    """Lernt aus Nutzer-A/B-Entscheidungen und passt zukünftige Läufe an.

    Speichert: (genre, material, defect_types) → bevorzugter Parameter-Satz.
    """

    def __init__(self) -> None:
        self._prefs: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            _PREFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
            if _PREFERENCE_FILE.exists():
                with open(_PREFERENCE_FILE) as f:
                    return json.load(f)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("preference_learner.py::_laden Ersatzpfad: %s", e)
        return {"sessions": [], "genre_weights": {}, "material_weights": {}}

    def _save(self) -> None:
        try:
            _PREFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_PREFERENCE_FILE, "w") as f:
                json.dump(self._prefs, f, indent=2, default=str)
        except Exception as e:
            logger.debug("§W Preference speichern fehlgeschlagen: %s", e)

    def record_choice(
        self,
        choice: str,  # "A" = original processing, "B" = alternative
        genre: str = "",
        material: str = "",
        defect_types: list[str] | None = None,
        option_a_params: dict[str, Any] | None = None,
        option_b_params: dict[str, Any] | None = None,
    ) -> None:
        """Zeichnet eine A/B-Präferenz auf und aktualisiert die Gewichte."""
        session = {
            "timestamp": time.time(),
            "choice": choice,
            "genre": genre,
            "material": material,
            "defect_types": defect_types or [],
        }
        if choice == "B" and option_b_params:
            session["preferred_params"] = dict(option_b_params)
            session["rejected_params"] = dict(option_a_params or {})
        self._prefs["sessions"].append(session)

        # Genre-Gewicht aktualisieren
        if genre and choice == "B":
            gw = self._prefs.setdefault("genre_weights", {})
            gd = gw.setdefault(genre, {})
            gd["preference_count"] = gd.get("preference_count", 0) + 1

        # Material-Gewicht
        if material and choice == "B":
            mw = self._prefs.setdefault("material_weights", {})
            md = mw.setdefault(material, {})
            md["preference_count"] = md.get("preference_count", 0) + 1

        self._save()
        logger.info(
            "§W Preference recorded: choice=%s genre=%s material=%s (total sessions: %d)",
            choice,
            genre,
            material,
            len(self._prefs["sessions"]),
        )

    def get_recommendation(self, genre: str = "", material: str = "") -> dict[str, Any]:
        """Gibt Empfehlung basierend auf gelernten Präferenzen."""
        rec: dict[str, Any] = {"strength_bias": 0.0, "confidence": 0.0}
        gw = self._prefs.get("genre_weights", {}).get(genre, {})
        mw = self._prefs.get("material_weights", {}).get(material, {})

        gw.get("preference_count", 0)
        mw.get("preference_count", 0)
        total_sessions = len(self._prefs.get("sessions", []))

        if total_sessions > 0:
            # Confidence steigt mit mehr Daten
            rec["confidence"] = min(1.0, total_sessions / 50.0)
            # Strength Bias: wenn User häufig B wählt → leichte Anhebung
            b_choices = sum(1 for s in self._prefs["sessions"] if s.get("choice") == "B")
            rec["strength_bias"] = float(np.clip((b_choices / total_sessions - 0.5) * 0.2, -0.1, 0.1))

        return rec

    def stats(self) -> dict[str, Any]:
        return {
            "total_sessions": len(self._prefs.get("sessions", [])),
            "genres_learned": len(self._prefs.get("genre_weights", {})),
            "materials_learned": len(self._prefs.get("material_weights", {})),
        }


# ── §Z Batch-Intelligence ────────────────────────────────────────────────


class BatchIntelligence:
    """Batch-übergreifendes Lernen: Erkenntnisse aus Song 1 → Song 2–N."""

    def __init__(self) -> None:
        self._songs: list[dict] = []
        self._strengths: dict[str, list[float]] = {}
        self._eq: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def start_batch(self, bid: str = "") -> None:
        with self._lock:
            self._songs.clear()
            self._strengths.clear()
            self._eq.clear()
            self._bid = bid or str(time.time())
        logger.info("§Z Batch gestartet: %s", self._bid)

    def record(
        self,
        sid: str,
        genre: str,
        material: str,
        defects: list,
        strengths: dict,
        eq: dict | None = None,
        scores: dict | None = None,
    ) -> None:
        with self._lock:
            self._songs.append(
                {"id": sid, "genre": genre, "material": material, "defects": defects, "scores": dict(scores or {})}
            )
            for pid, s in (strengths or {}).items():
                self._strengths.setdefault(pid, []).append(s)
            for b, v in (eq or {}).items():
                self._eq.setdefault(b, []).append(v)

    def recommend(self) -> dict:
        with self._lock:
            r: dict = {}
            if not self._songs:
                return r
            from collections import Counter

            r["genre"] = (
                Counter(s["genre"] for s in self._songs if s["genre"]).most_common(1)[0][0]
                if any(s["genre"] for s in self._songs)
                else "?"
            )
            r["material"] = (
                Counter(s["material"] for s in self._songs if s["material"]).most_common(1)[0][0]
                if any(s["material"] for s in self._songs)
                else "?"
            )
            r["phase_strengths"] = {k: float(np.median(v)) for k, v in self._strengths.items() if len(v) >= 2}
            r["eq"] = {k: float(np.mean(v)) for k, v in self._eq.items() if len(v) >= 2}
            all_s: dict[Any, Any] = {}
            for s in self._songs:
                for g, v in s.get("scores", {}).items():
                    all_s.setdefault(g, []).append(v)
            r["best_scores"] = {g: float(np.max(v)) for g, v in all_s.items() if v}
            r["song_count"] = len(self._songs)
            logger.info(
                "§Z Batch-Empfehlung: %d Songs → %d Strengths, %d EQ",
                len(self._songs),
                len(r.get("phase_strengths", {})),
                len(r.get("eq", {})),
            )
            return r

    def finish(self) -> dict:
        r = self.recommend()
        r["bid"] = getattr(self, "_bid", "")
        return r


import logging
import threading

logger = logging.getLogger(__name__)
