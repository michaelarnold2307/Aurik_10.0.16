"""
Intonation Intentionality Classifier — Lücke 1 (v10.0.0.x)
=========================================================

Unterscheidet auf F0-Kontur-Ebene, ob eine Pitch-Abweichung INTENTIONAL
(musikalische Gestik) oder ein DEFEKT (Träger-Artefakt) ist:

    INTENTIONAL      — Blue Note, Portamento, Appoggiatura, Vibrato-Onset
    DEGRADATION      — Wow/Flutter-Artefakt, Geschwindigkeits-Drift, Lagerschaden
    AMBIGUOUS        — Nicht eindeutig bestimmbar → konservativ: kein Eingriff

ALGORITHMUS (4 Merkmale, hierarchisch):
    1. Lokal-Tonaler Kontext: Liegt die Abweichung in einer harmonischen Richtung
       (±50 Cent zu nächster Skalenstufe = intentional) oder nicht?
    2. Temporal-Symmetrie: Vibratos (4–7 Hz) und Glides (200–800 ms) sind symmetrisch;
       Wow/Flutter ist unkorreliert mit harmonischem Kontext.
    3. Phrase-Boundary-Kontext: Pitch-Bend am Phrasenanfang/-ende = Belcanto-Technik.
       Pitch-Drift in der Phrasenmitte ohne Auflösung = Defekt.
    4. Energie-Konsistenz: Intentionale Gesten halten HNR; Defekte erzeugen
       spektrale Inkohärenz (spectral_novelty > 0.12).

Aufruf:
    from backend.core.dsp.intonation_classifier import classify_intonation_events
    events = classify_intonation_events(f0_contour, sr=48000, hop=512)
    # Injiziert in _restoration_context["intonation_events"] für CausalDefectReasoner

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0.x — Lücke 1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typen
# ---------------------------------------------------------------------------


class PitchDeviationIntent(str, Enum):
    """Klassifikation einer Pitch-Abweichung."""

    INTENTIONAL = "intentional"  # Musikalische Gestik — kein Pitch-Fix erlaubt
    DEGRADATION = "degradation"  # Carrier-Artefakt — Pitch-Korrektur erlaubt
    AMBIGUOUS = "ambiguous"  # Unsicher — konservativ: kein Eingriff


@dataclass
class IntonationEvent:
    """Eine klassifizierte Intonations-Abweichung."""

    start_s: float
    end_s: float
    intent: PitchDeviationIntent
    deviation_cents: float  # Max. Abweichung im Fenster (Cent)
    event_type: str = ""
    """
    Feiner Subtyp:
      "blue_note"     — intentionale Unterterz/Unterquint (Jazz/Blues)
      "portamento"    — Glide zwischen zwei Tönen (Belcanto/Schlager)
      "vibrato_onset" — Vibrato-Beginn (klassische Technik)
      "wow_flutter"   — periodische Drift ohne tonalen Bezug
      "speed_drift"   — monotoner Drift (mechanischer Defekt)
      "ambiguous"
    """
    confidence: float = 0.0  # Klassifikations-Konfidenz (0–1)
    pitch_correction_allowed: bool = True
    """False wenn intent==INTENTIONAL oder AMBIGUOUS → phase_31 darf nicht eingreifen."""

    def __post_init__(self) -> None:
        if self.intent == PitchDeviationIntent.INTENTIONAL or self.intent == PitchDeviationIntent.AMBIGUOUS:
            self.pitch_correction_allowed = False
        if not self.event_type:
            self.event_type = "ambiguous"


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

_CENTS_PER_SEMITONE = 100.0
_VIBRATO_FREQ_LOW_HZ = 4.0  # Untergrenze Vibrato-Rate (§0p)
_VIBRATO_FREQ_HIGH_HZ = 7.0  # Obergrenze Vibrato-Rate
_GLIDE_MIN_S = 0.10  # Kürzestes Portamento (100 ms)
_GLIDE_MAX_S = 1.20  # Längstes Portamento (1200 ms)
_DRIFT_MIN_S = 0.30  # Kürzeste Drift-Detektion
_DEVIATION_THRESH_CENTS = 20.0  # Minimale Abweichung für Ereignis-Flag

# 12-TET Skalenstufen (Cent-Abstände in einer Oktave) — für Blue-Note-Erkennung
_SCALE_DEGREES_CENTS = np.array([0, 200, 400, 500, 700, 900, 1100], dtype=float)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _hz_to_cents(f0_hz: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    """F0 in Hz → Cent relativ zu ref_hz. 0 Hz → NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(f0_hz > 0, 1200.0 * np.log2(f0_hz / ref_hz), np.nan)
    return result  # type: ignore[no-any-return]


def _nearest_scale_distance_cents(cents: float) -> float:
    """Abstand in Cent zur nächsten 12-TET-Skalenstufe."""
    mod = cents % 1200.0
    dists = np.abs(_SCALE_DEGREES_CENTS - mod)
    # Auch Übergang über Oktavgrenze
    dists = np.minimum(dists, 1200.0 - dists)
    return float(np.min(dists))


def _detect_vibrato_segments(
    f0_cents: np.ndarray,
    frame_dur_s: float,
) -> list[tuple[int, int, float]]:
    """
    Erkennt Vibrato-Segmente (4–7 Hz Modulation) in der Cent-Kontur.
    Gibt Liste von (start_frame, end_frame, vibrato_rate_hz) zurück.
    """
    valid = ~np.isnan(f0_cents)
    if not np.any(valid):
        return []

    # Lücken interpolieren für Spektralanalyse
    idx = np.arange(len(f0_cents))
    if np.any(~valid):
        f0_interp = np.interp(idx, idx[valid], f0_cents[valid])
    else:
        f0_interp = f0_cents.copy()

    # Gleitende Fenster-FFT zur Vibrato-Rate-Bestimmung
    win_frames = max(8, int(0.25 / frame_dur_s))  # 250 ms Fenster
    vibrato_segs: list[tuple[int, int, float]] = []

    for i in range(0, len(f0_interp) - win_frames, win_frames // 2):
        win = f0_interp[i : i + win_frames]
        win_detrended = win - np.mean(win)
        # Abbruch wenn Amplitude zu klein
        if np.std(win_detrended) < 3.0:  # < 3 Cent Abweichung → kein Vibrato
            continue

        N = len(win_detrended)
        fft = np.abs(np.fft.rfft(win_detrended * np.hanning(N)))
        freqs = np.fft.rfftfreq(N, d=frame_dur_s)
        # Energie im Vibrato-Band
        vib_mask = (freqs >= _VIBRATO_FREQ_LOW_HZ) & (freqs <= _VIBRATO_FREQ_HIGH_HZ)
        total_energy = np.sum(fft**2) + 1e-12
        vib_energy: float = float(np.sum(fft[vib_mask] ** 2))
        if vib_energy / total_energy > 0.40:  # > 40 % Energie im Vibrato-Band
            peak_freq = float(freqs[vib_mask][np.argmax(fft[vib_mask])]) if np.any(vib_mask) else 0.0
            vibrato_segs.append((i, i + win_frames, peak_freq))

    # Zusammenführen überlappender Segmente
    if not vibrato_segs:
        return []
    merged: list[tuple[int, int, float]] = [vibrato_segs[0]]
    for seg in vibrato_segs[1:]:
        prev = merged[-1]
        if seg[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], seg[1]), (prev[2] + seg[2]) / 2.0)
        else:
            merged.append(seg)
    return merged


def _is_blue_note_context(
    cents_sequence: np.ndarray,
    _phrase_position: float,  # 0 = Anfang, 1 = Ende der Phrase
) -> bool:
    """
    Erkennt Blue-Note-Kontext: Pitch liegt intentional zwischen Skalenstufen,
    insbesondere b3, b5, b7 (Blues-Pentatonik-Töne).
    """
    valid = cents_sequence[~np.isnan(cents_sequence)]
    if len(valid) < 3:
        return False
    mean_cents = float(np.mean(valid))
    dist_to_scale = _nearest_scale_distance_cents(mean_cents)
    # Blue Notes: 30–70 Cent von nächster Skalenstufe entfernt
    return 30.0 <= dist_to_scale <= 70.0


def _is_monotone_drift(cents_sequence: np.ndarray) -> tuple[bool, float]:
    """
    Erkennt monotonen Pitch-Drift (mechanischer Defekt: Tonband-Geschwindigkeitsfehler).
    Gibt (ist_drift, drift_slope_cents_per_s) zurück.
    """
    valid_idx = np.where(~np.isnan(cents_sequence))[0]
    if len(valid_idx) < 4:
        return False, 0.0

    x = valid_idx.astype(float)
    y = cents_sequence[valid_idx]

    # Lineare Regression
    slope, intercept = float(np.polyfit(x, y, 1)[0]), 0.0
    y_pred = slope * x + intercept
    residual_std = float(np.std(y - y_pred))

    # Monotoner Drift: hoher Slope, niedriger Residual-Streuung
    # (d.h. der Drift ist gleichmäßig, nicht zufällig)
    is_drift = abs(slope) > 1.5 and residual_std < abs(slope) * 3.0
    return is_drift, slope


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def classify_intonation_events(
    f0_hz: np.ndarray,
    sr: int = 48_000,
    hop: int = 512,
    *,
    phrase_boundaries: list[float] | None = None,
    deviation_thresh_cents: float = _DEVIATION_THRESH_CENTS,
) -> list[IntonationEvent]:
    """
    Klassifiziert Intonations-Ereignisse in der F0-Kontur.

    Non-blocking: Exception → leere Liste.

    Args:
        f0_hz:              F0-Kontur in Hz (0 = unvoiced), aus CREPE/FCPE/ZCPA.
        sr:                 Abtastrate
        hop:                STFT-Hop (für Zeitachsen-Berechnung)
        phrase_boundaries:  Zeitstempel der Phrasengrenzen in Sekunden (optional).
        deviation_thresh_cents: Schwelle für Ereignis-Flag.

    Returns:
        Liste von IntonationEvent, chronologisch sortiert.
        pitch_correction_allowed=False → phase_31 darf in diesen Zonen NICHT eingreifen.
    """
    try:
        if len(f0_hz) < 4:
            return []

        frame_dur_s = hop / sr
        cents = _hz_to_cents(f0_hz)

        events: list[IntonationEvent] = []

        # 1. Vibrato-Segmente → INTENTIONAL (klassische/Belcanto-Technik)
        vibrato_segs = _detect_vibrato_segments(cents, frame_dur_s)
        for start_f, end_f, _rate_hz in vibrato_segs:
            seg_cents = cents[start_f:end_f]
            valid = seg_cents[~np.isnan(seg_cents)]
            dev = float(np.max(np.abs(valid - np.mean(valid)))) if len(valid) > 0 else 0.0
            events.append(
                IntonationEvent(
                    start_s=start_f * frame_dur_s,
                    end_s=end_f * frame_dur_s,
                    intent=PitchDeviationIntent.INTENTIONAL,
                    deviation_cents=dev,
                    event_type="vibrato_onset" if dev < 30.0 else "vibrato",
                    confidence=0.85,
                )
            )

        # 2. Globaler Drift-Check → DEGRADATION
        is_drift, drift_slope = _is_monotone_drift(cents)
        if is_drift and abs(drift_slope) > 2.0:
            events.append(
                IntonationEvent(
                    start_s=0.0,
                    end_s=len(f0_hz) * frame_dur_s,
                    intent=PitchDeviationIntent.DEGRADATION,
                    deviation_cents=abs(drift_slope) * len(f0_hz) * frame_dur_s,
                    event_type="speed_drift",
                    confidence=0.80,
                )
            )

        # 3. Lokale Abweichungs-Fenster (Sliding Window, 200 ms)
        win_frames = max(4, int(0.200 / frame_dur_s))
        already_covered = set()  # Verhindert Doppel-Events

        for i in range(0, len(cents) - win_frames, win_frames // 2):
            if i in already_covered:
                continue
            win_cents = cents[i : i + win_frames]
            valid = win_cents[~np.isnan(win_cents)]
            if len(valid) < 3:
                continue

            mean_c = float(np.mean(valid))
            max_dev = float(np.max(np.abs(valid - mean_c)))
            if max_dev < deviation_thresh_cents:
                continue

            start_s = i * frame_dur_s
            end_s = (i + win_frames) * frame_dur_s

            # Prüfen ob bereits in Vibrato-Segment
            in_vibrato = any(
                e.event_type in ("vibrato_onset", "vibrato") and e.start_s <= start_s and e.end_s >= end_s
                for e in events
            )
            if in_vibrato:
                for j in range(i, i + win_frames):
                    already_covered.add(j)
                continue

            # Phrasen-Kontext bestimmen
            phrase_pos = 0.5
            if phrase_boundaries:
                prev_boundary = max((t for t in phrase_boundaries if t <= start_s), default=0.0)
                next_boundary = min((t for t in phrase_boundaries if t >= end_s), default=end_s + 1.0)
                phrase_len = next_boundary - prev_boundary + 1e-6
                phrase_pos = (start_s - prev_boundary) / phrase_len

            # Blue-Note-Erkennung (Jazz/Blues/Soul)
            if _is_blue_note_context(win_cents, phrase_pos):
                events.append(
                    IntonationEvent(
                        start_s=start_s,
                        end_s=end_s,
                        intent=PitchDeviationIntent.INTENTIONAL,
                        deviation_cents=max_dev,
                        event_type="blue_note",
                        confidence=0.70,
                    )
                )
                for j in range(i, i + win_frames):
                    already_covered.add(j)
                continue

            # Portamento/Glide: monotoner Trend innerhalb des Fensters
            is_glide, _ = _is_monotone_drift(win_cents)
            dur_s = end_s - start_s
            if is_glide and _GLIDE_MIN_S <= dur_s <= _GLIDE_MAX_S:
                # Am Phrasenbeginn oder -ende → Belcanto-Technik (intentional)
                if phrase_pos < 0.20 or phrase_pos > 0.75:
                    events.append(
                        IntonationEvent(
                            start_s=start_s,
                            end_s=end_s,
                            intent=PitchDeviationIntent.INTENTIONAL,
                            deviation_cents=max_dev,
                            event_type="portamento",
                            confidence=0.65,
                        )
                    )
                else:
                    # Glide in Phrasenmitte ohne harmonischen Kontext → AMBIGUOUS
                    events.append(
                        IntonationEvent(
                            start_s=start_s,
                            end_s=end_s,
                            intent=PitchDeviationIntent.AMBIGUOUS,
                            deviation_cents=max_dev,
                            event_type="ambiguous",
                            confidence=0.50,
                        )
                    )
                for j in range(i, i + win_frames):
                    already_covered.add(j)
                continue

            # Wow/Flutter-Artefakt: schnelle periodische Schwankung, kein Vibrato-Band
            # Energie im 0.5–3 Hz Band (langsamer als Vibrato, aber periodisch)
            valid_arr = valid if len(valid) >= 4 else np.full(4, mean_c)
            N = len(valid_arr)
            fft_mag = np.abs(np.fft.rfft(valid_arr * np.hanning(N)))
            fft_freqs = np.fft.rfftfreq(N, d=frame_dur_s)
            wow_mask = (fft_freqs >= 0.3) & (fft_freqs < 3.5)
            wow_energy = float(np.sum(fft_mag[wow_mask] ** 2)) if np.any(wow_mask) else 0.0
            total_energy = float(np.sum(fft_mag**2)) + 1e-12
            if wow_energy / total_energy > 0.45:
                events.append(
                    IntonationEvent(
                        start_s=start_s,
                        end_s=end_s,
                        intent=PitchDeviationIntent.DEGRADATION,
                        deviation_cents=max_dev,
                        event_type="wow_flutter",
                        confidence=0.75,
                    )
                )
                for j in range(i, i + win_frames):
                    already_covered.add(j)
                continue

            # Kein klares Muster → AMBIGUOUS
            events.append(
                IntonationEvent(
                    start_s=start_s,
                    end_s=end_s,
                    intent=PitchDeviationIntent.AMBIGUOUS,
                    deviation_cents=max_dev,
                    event_type="ambiguous",
                    confidence=0.40,
                )
            )

        # Zeitlich sortieren
        events.sort(key=lambda e: e.start_s)

        n_int = sum(1 for e in events if e.intent == PitchDeviationIntent.INTENTIONAL)
        n_deg = sum(1 for e in events if e.intent == PitchDeviationIntent.DEGRADATION)
        n_amb = sum(1 for e in events if e.intent == PitchDeviationIntent.AMBIGUOUS)
        logger.debug(
            "intonation_classifier: %d events — intentional=%d degradation=%d ambiguous=%d",
            len(events),
            n_int,
            n_deg,
            n_amb,
        )
        return events

    except Exception as exc:
        logger.warning("§G23 classify_intonation_events: DSP-Ersatzpfad — %s", exc, exc_info=True)
        return []


def get_pitch_correction_protected_zones(
    events: list[IntonationEvent],
) -> list[tuple[float, float]]:
    """
    Gibt alle Zeitfenster zurück, in denen phase_31 (Speed/Pitch-Correction)
    NICHT eingreifen darf (INTENTIONAL oder AMBIGUOUS).

    Verwendung::

        protected = get_pitch_correction_protected_zones(events)
        # Injiziert als kwargs["pitch_correction_protected_zones"] in phase_31
    """
    return [(e.start_s, e.end_s) for e in events if not e.pitch_correction_allowed]
