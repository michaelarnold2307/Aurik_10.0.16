"""§2.61 SectionGoalAdapter — verbindet MusicalStructureAnalyzer mit dem Fahrplan.

Der Adapter:
  1. Nimmt Audio + Samplerate
  2. Ruft MusicalStructureAnalyzer.analyze() für SSM-basierte Segmentierung
  3. Gibt Sektionen im Fahrplan-Format zurück: [(start_s, end_s, label), ...]

Minimal-Interface: eine Funktion `adapt(audio, sr) -> list[tuple[float, float, str]]`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_sections(
    audio: np.ndarray,
    sr: int,
    *,
    min_duration_s: float = 20.0,
) -> list[tuple[float, float, str]]:
    """Analysiert Audio und gibt Sektionen für den Fahrplan zurück.

    Args:
        audio: (samples,) oder (channels, samples) ndarray
        sr: Sample-Rate
        min_duration_s: Audio kürzer als das → Fallback auf eine "full"-Sektion

    Returns:
        Liste von (start_s, end_s, label) Tupeln, z.B.:
        [(0.0, 22.3, "intro"), (22.3, 67.8, "verse"), ...]
    """
    arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    n = arr.shape[0] if arr.ndim == 1 else max(arr.shape)
    duration_s = n / max(sr, 1)

    # Kurzes Audio → eine Sektion
    if duration_s < min_duration_s:
        return [(0.0, duration_s, "full")]

    try:
        from backend.core.musical_structure_analyzer import MusicalStructureAnalyzer

        analyzer = MusicalStructureAnalyzer()
        structure = analyzer.analyze(arr, sr)

        if not structure.segments:
            return [(0.0, duration_s, "full")]

        sections: list[tuple[float, float, str]] = []
        for seg in structure.segments:
            start = float(seg.start_s)
            end = float(seg.end_s)
            label = str(seg.label).lower().strip()
            # Normalisiere Label auf Fahrplan-kompatible Kategorien
            label = _normalize_label(label)
            sections.append((start, end, label))

        return _merge_adjacent(sections)

    except Exception as exc:
        logger.debug("SectionGoalAdapter: Analyse fehlgeschlagen → full: %s", exc)
        return [(0.0, duration_s, "full")]


def _normalize_label(label: str) -> str:
    """Vereinheitlicht Labels auf Fahrplan-kompatible Namen."""
    mapping: dict[str, str] = {
        "intro": "intro",
        "outro": "outro",
        "verse": "verse",
        "chorus": "chorus",
        "bridge": "bridge",
        "pre-chorus": "chorus",
        "pre_chorus": "chorus",
        "prechorus": "chorus",
        "post-chorus": "chorus",
        "post_chorus": "chorus",
        "solo": "bridge",  # Solo → konservativ wie Bridge
        "instrumental": "verse",
        "break": "bridge",
        "interlude": "bridge",
        "fade": "outro",
        "fade_out": "outro",
        "fadeout": "outro",
        "silence": "silence",
        "quiet": "outro",
        "unknown": "full",
        "full": "full",
    }
    return mapping.get(label, "full")


def _merge_adjacent(
    sections: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Merge benachbarte Sektionen mit gleichem Label."""
    if len(sections) <= 1:
        return sections

    merged: list[tuple[float, float, str]] = []
    for start, end, label in sections:
        if merged and merged[-1][2] == label:
            prev_start, _, prev_label = merged.pop()
            merged.append((prev_start, end, prev_label))
        else:
            merged.append((start, end, label))
    return merged


@dataclass
class SectionTarget:
    """§Gap7-wire v10.0.0: Per-Sektion Ziel-Parameter für SectionStrengthEnvelope (§INV-3).

    Konsumiert von `backend.core.dsp.section_strength_envelope.build_strength_envelope()`.
    """

    start_s: float
    end_s: float
    label: str = "full"
    nr_strength_scale: float = 1.0
    vq_weight: float = 1.0
    frisson_protection: bool = False


# §Gap7: Musikfunktionale Basis-Gewichtung pro Sektionstyp.
#
# nr_strength_scale — Rauschunterdrückungs-Intensität relativ zur Baseline (1.0).
#   Dichtere/lautere Sektionen maskieren Rauschen psychoakustisch stärker
#   (simultane Verdeckung, Fletcher-Munson) → geringere NR nötig, um Pumping/
#   Artefakte an Transienten zu vermeiden. Dünnere/leisere Sektionen (Intro,
#   Outro, Stille) legen den Rauschflor frei → mehr NR-Spielraum.
#
# vq_weight — Vokal-Qualitäts-Gewichtung. Der emotionale Fokus eines Songs
#   liegt auf dem Refrain (Hook) — dort muss die Stimme am saubersten/präsentesten
#   sein; Intro/Outro sind häufiger instrumental oder ausklingend.
_SECTION_PROFILE: dict[str, tuple[float, float]] = {
    # label:      (nr_strength_scale, vq_weight)
    "intro": (1.10, 0.80),
    "verse": (1.00, 1.00),
    "chorus": (0.90, 1.30),
    "bridge": (1.00, 1.10),
    "outro": (1.05, 0.90),
    "silence": (1.20, 0.50),
    "full": (1.00, 1.00),
}


def _normalize_frisson_zones(frisson_zones: list[Any] | None) -> list[tuple[float, float]]:
    """Normalisiert FrissonZone-Objekte (.start_s/.end_s) oder (start_s, end_s[, ...])-Tupel."""
    if not frisson_zones:
        return []
    out: list[tuple[float, float]] = []
    for z in frisson_zones:
        try:
            if hasattr(z, "start_s") and hasattr(z, "end_s"):
                out.append((float(z.start_s), float(z.end_s)))
            elif isinstance(z, (tuple, list)) and len(z) >= 2:
                out.append((float(z[0]), float(z[1])))
        except (TypeError, ValueError):
            continue
    return out


class SectionGoalAdapter:
    """§Gap7-wire v10.0.0: Sektionsweise Ziel-Anpassung (Intro/Vers/Chorus/Outro).

    Verbindet `get_sections()` (SSM-Boundary-Detektion) mit musikfunktionaler
    NR-Stärke-/Vokalqualitäts-Gewichtung sowie Frisson-Schutz aus VFA-Zonen.
    Zustandslos (keine per-Song-Daten werden im Objekt gehalten) — ein
    Singleton ist damit §V8-konform (kein Cross-Song-Contamination-Risiko).
    """

    def compute_section_targets(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        frisson_zones: list[Any] | None = None,
    ) -> list[SectionTarget]:
        """Berechnet per-Sektion Ziel-Parameter aus dem ORIGINAL-Audio.

        Args:
            audio: Original-Audio-Referenz (samples,) oder (channels, samples)
            sr: Sample-Rate
            frisson_zones: Optionale FrissonZone-Objekte/Tupel (§0p Klimax-Schutz)

        Returns:
            Liste von SectionTarget, sortiert nach start_s.
        """
        sections = get_sections(audio, sr)
        zones = _normalize_frisson_zones(frisson_zones)

        targets: list[SectionTarget] = []
        for start_s, end_s, label in sections:
            nr_scale, vq_weight = _SECTION_PROFILE.get(label, _SECTION_PROFILE["full"])
            protected = any(start_s < z_end and end_s > z_start for z_start, z_end in zones)
            targets.append(
                SectionTarget(
                    start_s=start_s,
                    end_s=end_s,
                    label=label,
                    nr_strength_scale=nr_scale,
                    vq_weight=vq_weight,
                    frisson_protection=protected,
                )
            )
        return targets


_adapter: SectionGoalAdapter | None = None


def get_section_goal_adapter() -> SectionGoalAdapter:
    """Gibt die globale SectionGoalAdapter-Instanz zurück (zustandslos, §V8-konform)."""
    global _adapter
    if _adapter is None:
        _adapter = SectionGoalAdapter()
    return _adapter
