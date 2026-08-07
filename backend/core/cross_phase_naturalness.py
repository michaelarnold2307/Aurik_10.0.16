"""
§v10.3 Cross-Phase Naturalness Consensus (§3.0 Roadmap).

Verhindert, dass mehrere Phasen denselben Frequenzbereich überbearbeiten.
Trackt kumulative Gain-Änderungen pro Frequenzband über alle Phasen hinweg.

Integration:
- Wird vom NaturalnessOptimizer vor jeder Stage aufgerufen
- Kann von UV3-Phasen via Bridge abgefragt werden
- Persistiert im restoration_context
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Frequenzbänder (Bark-Skala, vereinfacht)
BANDS = {
    "sub_bass": (20, 100, "Sub-Bass"),
    "bass": (100, 250, "Bass"),
    "low_mid": (250, 800, "Untere Mitten / Wärme"),
    "mid": (800, 2500, "Mitten / Präsenz"),
    "high_mid": (2500, 6000, "Obere Mitten / Sibilanz"),
    "presence": (6000, 10000, "Präsenz / Brillanz"),
    "air": (10000, 20000, "Luftband"),
}


class CrossPhaseTracker:
    """Trackt kumulative Verarbeitung pro Frequenzband."""

    def __init__(self):
        self._gain: dict[str, float] = dict.fromkeys(BANDS, 0.0)
        self._count: dict[str, int] = dict.fromkeys(BANDS, 0)
        self._history: list[dict] = []
        self._lock = threading.Lock()

    def record(self, phase: str, band_gains: dict[str, float]):
        """Zeichnet auf, was eine Phase/Stage in jedem Band verändert hat."""
        with self._lock:
            entry = {"phase": phase, "gains": dict(band_gains)}
            for band, gain_db in band_gains.items():
                if band in self._gain:
                    self._gain[band] += gain_db
                    self._count[band] += 1
            self._history.append(entry)

    def can_process(self, band: str, planned_gain_db: float = 2.0) -> bool:
        """Prüft, ob ein Band noch weiteren Processing zulässt.

        Regeln:
        - Max ±8 dB kumulativer Gain pro Band
        - Max 3 Phasen pro Band
        - Einzelne Phase: max ±4 dB
        """
        with self._lock:
            cumulative = abs(self._gain.get(band, 0.0))
            count = self._count.get(band, 0)
        if abs(planned_gain_db) > 4.0:
            return False
        if cumulative + abs(planned_gain_db) > 8.0:
            return False
        if count >= 3:
            return False
        return True

    def suggest_scale(self, bands: list[str]) -> float:
        """Schlägt Intensitätsskalierung (0.0-1.0) vor.

        1.0 = volle Intensität sicher
        0.5 = moderate Reduktion nötig
        0.0 = Band gesättigt, skip
        """
        with self._lock:
            scales = []
            for band in bands:
                cumulative = abs(self._gain.get(band, 0.0))
                count = self._count.get(band, 0)
                remaining = max(0.0, 8.0 - cumulative)
                if count >= 3:
                    remaining = 0.0
                scales.append(min(1.0, remaining / 4.0))
        return min(scales) if scales else 1.0

    def get_band_report(self) -> dict[str, dict]:
        """Gibt Status aller Bänder zurück (für Logging/GUI)."""
        with self._lock:
            return {
                band: {
                    "cumulative_gain_db": round(self._gain[band], 2),
                    "phase_count": self._count[band],
                    "saturated": self._count[band] >= 3 or abs(self._gain[band]) > 8.0,
                    "range": f"{low}-{high} Hz",
                    "name": name,
                }
                for band, (low, high, name) in BANDS.items()
            }

    def reset(self):
        """Setzt den Tracker zurück (neue Datei)."""
        with self._lock:
            self._gain = dict.fromkeys(BANDS, 0.0)
            self._count = dict.fromkeys(BANDS, 0)
            self._history.clear()


# Globaler Singleton
_tracker: CrossPhaseTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> CrossPhaseTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = CrossPhaseTracker()
    return _tracker


def reset_tracker():
    global _tracker
    with _tracker_lock:
        if _tracker is not None:
            _tracker.reset()


# ── NaturalnessOptimizer Integration ─────────────────────────────────────


def estimate_band_effects(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """Schätzt, welche Frequenzbänder eine Stage verändert hat.

    Vergleicht RMS-Energie pro Band vor/nach der Verarbeitung.
    """
    try:
        from scipy.signal import butter, sosfiltfilt

        mono_before = audio_before.mean(axis=1) if audio_before.ndim == 2 else audio_before
        mono_after = audio_after.mean(axis=1) if audio_after.ndim == 2 else audio_after
        nyq = sr / 2

        effects = {}
        for band, (low, high, _) in BANDS.items():
            if high > nyq * 0.95:
                high = nyq * 0.95  # type: ignore[assignment]
            if low >= high:
                continue
            sos = butter(2, [low / nyq, high / nyq], btype="band", output="sos")
            rms_before = float(np.sqrt(np.mean(sosfiltfilt(sos, mono_before) ** 2)) + 1e-12)
            rms_after = float(np.sqrt(np.mean(sosfiltfilt(sos, mono_after) ** 2)) + 1e-12)
            if rms_before > 1e-10:
                gain_db = 20.0 * np.log10(rms_after / rms_before)
            else:
                gain_db = 0.0
            effects[band] = float(np.clip(gain_db, -6.0, 6.0))
        return effects
    except Exception as e:
        logger.warning("cross_Verarbeitungsschritt_naturalness.py::estimate_band_effects Ersatzpfad: %s", e)
        return dict.fromkeys(BANDS, 0.0)


def guard_stage(
    stage_name: str,
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, float]:
    """Prüft und begrenzt eine Stage basierend auf Cross-Phase Consensus.

    Returns:
        (audio, applied_scale) — audio ggf. reduziert, scale 0.0-1.0
    """
    tracker = get_tracker()
    effects = estimate_band_effects(audio_before, audio_after, sr)
    active_bands = [b for b, g in effects.items() if abs(g) > 0.5]

    if not active_bands:
        tracker.record(stage_name, effects)
        return audio_after, 1.0

    scale = tracker.suggest_scale(active_bands)
    if scale < 0.3:
        logger.info("CrossPhase: %s übersprungen (Bänder gesättigt: %s)", stage_name, active_bands)
        return audio_before, 0.0

    if scale < 1.0:
        # Wet/Dry Mix: skaliere die Änderung
        audio_after = audio_before + (audio_after - audio_before) * scale
        # Skaliere Effekte entsprechend
        effects = {b: g * scale for b, g in effects.items()}

    tracker.record(stage_name, effects)
    return audio_after.astype(np.float32), scale
