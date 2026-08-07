"""
Atemgeräusch-Emotions-Klassifikation — §Lücke-F (v10.0.0.x)
===========================================================
Klassifiziert erkannte Atemgeräusche nach emotionalem Inhalt.
Erweitert den bestehenden §2.46f-Atemschutz um eine Qualitätsdimension.

Kategorien:
    EMOTIONAL_TENSION  — Zitterndes/anschwellendes Atemgeräusch vor
                         Klimax/High-Note → maximaler NR-Schutz (G_floor 0.85)
    CONTROLLED         — Professionell kontrollierter Atem → Standardschutz
    MECHANICAL_POP     — Nähe-Pop (RMS > −25 dBFS + tiefes Spektrum) → leichte
                         Korrektur erlaubt (G_floor 0.25)
    NATURAL            — Normaler Atem → §2.46f-Schutz (G_floor 0.50)

§0p: EMOTIONAL_TENSION-Atemgeräusche sind Frisson-Vorboten und sakrosankt.
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


class BreathCategory(str, Enum):
    """Emotionale Kategorie eines erkannten Atemgeräusches."""

    EMOTIONAL_TENSION = "emotional_tension"  # Zittern/Crescendo → max. Schutz
    CONTROLLED = "controlled"  # Professionell → Std.-Schutz
    MECHANICAL_POP = "mechanical_pop"  # Nähe-Pop → Korrektur erlaubt
    NATURAL = "natural"  # Normaler Atem → §2.46f-Schutz


@dataclass
class BreathSegment:
    """Ein erkanntes Atemgeräusch mit Kategorie."""

    start_s: float
    end_s: float
    category: BreathCategory
    rms_db: float
    spectral_flatness: float
    energy_slope: float  # positiv = anschwellend (Klimax-Vorbote)
    confidence: float  # 0.0–1.0

    # G_floor-Empfehlung für NR-Integration
    recommended_g_floor: float = 0.50

    def to_dict(self) -> dict:
        """Serialisiert das Segment als Dictionary für Protokoll und Metadaten."""
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "category": self.category.value,
            "rms_db": round(self.rms_db, 1),
            "spectral_flatness": round(self.spectral_flatness, 3),
            "energy_slope": round(self.energy_slope, 4),
            "confidence": round(self.confidence, 3),
            "recommended_g_floor": round(self.recommended_g_floor, 3),
        }


# ---------------------------------------------------------------------------
# Interne Helpers
# ---------------------------------------------------------------------------


def _compute_spectral_flatness(frame: np.ndarray) -> float:
    """Spektrale Flachheit via geometrischem/arithmetischem Mittel des Spektrums."""
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) + 1e-12
    geo_mean = float(np.exp(np.mean(np.log(spec))))
    arith_mean = float(np.mean(spec))
    return float(np.clip(geo_mean / max(arith_mean, 1e-12), 0.0, 1.0))


def _compute_spectral_centroid(frame: np.ndarray, sr: int) -> float:
    """Spektraler Schwerpunkt in Hz."""
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
    total = float(np.sum(spec))
    if total < 1e-10:
        return 500.0
    return float(np.sum(freqs * spec) / total)


def _db(rms: float) -> float:
    return float(20.0 * np.log10(max(rms, 1e-10)))


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def classify_breath_emotions(
    audio: np.ndarray,
    sr: int,
    *,
    lookahead_s: float = 0.20,
) -> list[BreathSegment]:
    """Erkennt und klassifiziert Atemgeräusche nach emotionalem Gehalt.

    Args:
        audio:        Input Audio (mono oder stereo channels-first), float32.
        sr:           Sample-Rate (Hz).
        lookahead_s:  Vorausschau-Fenster für Energie-Anstieg-Erkennung [s].

    Returns:
        Liste von BreathSegment — niemals None (Non-blocking, kann leer sein).
    """
    try:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio

        n_samples = len(mono)
        if n_samples < sr * 0.3:
            return []

        # Frame-Analyse
        frame_len = int(sr * 0.025)  # 25 ms
        hop = frame_len // 2

        rms_list: list[float] = []
        flatness_list: list[float] = []
        centroid_list: list[float] = []

        for i in range(0, n_samples - frame_len, hop):
            frame = mono[i : i + frame_len]
            rms_val = float(np.sqrt(np.mean(frame**2)))
            rms_list.append(rms_val)
            flatness_list.append(_compute_spectral_flatness(frame))
            centroid_list.append(_compute_spectral_centroid(frame, sr))

        rms_arr = np.array(rms_list, dtype=np.float64)
        flat_arr = np.array(flatness_list, dtype=np.float64)
        cent_arr = np.array(centroid_list, dtype=np.float64)

        # §2.46f: Atemgeräusch-Kandidaten — RMS −55 bis −25 dBFS + hohe Flatness
        _rms_db_arr = 20.0 * np.log10(np.maximum(rms_arr, 1e-10))
        _is_breath = (_rms_db_arr >= -55.0) & (_rms_db_arr <= -25.0) & (flat_arr >= 0.40)

        if not np.any(_is_breath):
            return []

        # Segmentierung: zusammenhängende Breath-Frames → Segmente
        lookahead_frames = max(1, int(lookahead_s / (hop / sr)))
        segments: list[BreathSegment] = []

        in_segment = False
        seg_start_frame = 0

        for idx in range(len(_is_breath)):
            if _is_breath[idx] and not in_segment:
                in_segment = True
                seg_start_frame = idx
            elif not _is_breath[idx] and in_segment:
                in_segment = False
                _classify_and_append(
                    segments,
                    rms_arr,
                    flat_arr,
                    cent_arr,
                    _rms_db_arr,
                    seg_start_frame,
                    idx,
                    hop,
                    sr,
                    lookahead_frames,
                    n_samples,
                )

        if in_segment:
            _classify_and_append(
                segments,
                rms_arr,
                flat_arr,
                cent_arr,
                _rms_db_arr,
                seg_start_frame,
                len(_is_breath),
                hop,
                sr,
                lookahead_frames,
                n_samples,
            )

        logger.info(
            "§Lücke-F BreathEmotionClassifier: %d Segmente (tension=%d controlled=%d pop=%d natural=%d)",
            len(segments),
            sum(1 for s in segments if s.category == BreathCategory.EMOTIONAL_TENSION),
            sum(1 for s in segments if s.category == BreathCategory.CONTROLLED),
            sum(1 for s in segments if s.category == BreathCategory.MECHANICAL_POP),
            sum(1 for s in segments if s.category == BreathCategory.NATURAL),
        )
        return segments

    except Exception as exc:
        logger.debug("BreathEmotionClassifier nicht blockierend Fehlschlag: %s", exc)
        return []


def _classify_and_append(
    segments: list[BreathSegment],
    rms_arr: np.ndarray,
    flat_arr: np.ndarray,
    cent_arr: np.ndarray,
    rms_db_arr: np.ndarray,
    start_f: int,
    end_f: int,
    hop: int,
    sr: int,
    lookahead_frames: int,
    _n_samples: int,
) -> None:
    """Klassifiziert ein Breath-Segment und hängt es an die Liste."""
    if end_f <= start_f:
        return

    seg_rms_db = float(np.mean(rms_db_arr[start_f:end_f]))
    seg_flat = float(np.mean(flat_arr[start_f:end_f]))
    seg_cent = float(np.mean(cent_arr[start_f:end_f]))

    start_s = float(start_f * hop / sr)
    end_s = float(end_f * hop / sr)

    # Energie-Anstieg danach (Klimax-Vorläufer?)
    look_end = min(end_f + lookahead_frames, len(rms_arr))
    after_rms = float(np.mean(rms_arr[end_f:look_end])) if look_end > end_f else 0.0
    seg_rms_mean = float(np.mean(rms_arr[start_f:end_f]))
    energy_slope = float(after_rms - seg_rms_mean)

    # Klassifikation
    if seg_rms_db > -25.0 and seg_cent < 600.0:
        # Nähe-Pop: laut + tiefes Spektrum
        category = BreathCategory.MECHANICAL_POP
        confidence = float(np.clip((seg_rms_db + 25.0) / 10.0 + (600.0 - seg_cent) / 600.0, 0.0, 1.0))
        g_floor = 0.25
    elif seg_flat > 0.55 and energy_slope > 5e-4:
        # Emotionaler Atem: hohe Flatness + Energie steigt danach an
        category = BreathCategory.EMOTIONAL_TENSION
        confidence = float(np.clip(seg_flat * 1.3 + energy_slope * 500, 0.0, 1.0))
        g_floor = 0.85
    elif seg_flat >= 0.40 and seg_flat <= 0.55 and abs(energy_slope) < 3e-4:
        # Kontrollierter Atem: mittlere Flatness + stabile Energie
        category = BreathCategory.CONTROLLED
        confidence = float(np.clip(1.0 - abs(seg_flat - 0.475) * 8.0, 0.4, 1.0))
        g_floor = 0.55
    else:
        category = BreathCategory.NATURAL
        confidence = 0.5
        g_floor = 0.50

    segments.append(
        BreathSegment(
            start_s=start_s,
            end_s=end_s,
            category=category,
            rms_db=seg_rms_db,
            spectral_flatness=seg_flat,
            energy_slope=energy_slope,
            confidence=confidence,
            recommended_g_floor=g_floor,
        )
    )


def get_breath_emotion_summary(segments: list[BreathSegment]) -> dict:
    """Kompaktes Summary für Protokoll und Integration."""
    n = len(segments)
    if n == 0:
        return {
            "n_total": 0,
            "n_emotional_tension": 0,
            "n_controlled": 0,
            "n_mechanical_pop": 0,
            "n_natural": 0,
            "emotional_tension_zones": [],
        }
    return {
        "n_total": n,
        "n_emotional_tension": sum(1 for s in segments if s.category == BreathCategory.EMOTIONAL_TENSION),
        "n_controlled": sum(1 for s in segments if s.category == BreathCategory.CONTROLLED),
        "n_mechanical_pop": sum(1 for s in segments if s.category == BreathCategory.MECHANICAL_POP),
        "n_natural": sum(1 for s in segments if s.category == BreathCategory.NATURAL),
        "emotional_tension_zones": [
            (s.start_s, s.end_s) for s in segments if s.category == BreathCategory.EMOTIONAL_TENSION
        ],
    }
