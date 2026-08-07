"""
SongStructureAnalyzer — §2.52b [RELEASE_MUST]
=============================================

Segment-bewusste Pipeline: Erkennt Intro/Vers/Chorus/Bridge/Outro/Instrumental
und liefert segment-adaptive Strength-Skalare für jede Phase.

Spec: 02_pipeline_architecture.md §2.52b (v10.0.0)
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_instance: "SongStructureAnalyzer | None" = None
_lock = threading.Lock()


def get_song_structure_analyzer() -> "SongStructureAnalyzer":
    """Singleton-Getter (thread-safe, double-checked locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SongStructureAnalyzer()
    return _instance


# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------


@dataclass
class SongSegment:
    """Ein erkanntes Song-Segment (§2.52b)."""

    start_s: float
    end_s: float
    label: str  # "intro", "verse", "chorus", "bridge", "outro", "instrumental"
    energy_level: float  # [0, 1] normiert auf Ø RMS
    has_vocals: bool
    is_climax: bool


# Strength-Skalare pro Segment-Typ (§2.52b Tabelle) — bounded [0.70, 1.30]
_STRENGTH_SCALARS: dict[str, dict[str, float]] = {
    "verse": {
        "nr_strength": 1.15,
        "dereverb": 1.10,
        "default": 1.05,
    },
    "chorus": {
        "nr_strength": 0.85,
        "compression": 0.70,
        "default": 0.90,
    },
    "intro": {
        "default": 1.00,
    },
    "outro": {
        "default": 1.00,
    },
    "bridge": {
        "default": 0.95,
    },
    "instrumental": {
        "nr_strength": 1.00,
        "default": 1.00,
    },
    "silence": {
        "default": 0.70,  # nur passiv
    },
    "unknown": {
        "default": 1.00,
    },
}

# Zusätzliche Climax-Overrides (überschreiben Segment-Label)
_CLIMAX_SCALAR = 0.85


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class SongStructureAnalyzer:
    """Erkennt Song-Segmente und liefert segment-adaptive Strength-Skalare."""

    def analyze_structure(
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing_confidence: float = 0.0,
    ) -> list[SongSegment]:
        """Analysiert die Song-Struktur via librosa Boundary-Erkennung.

        Args:
            audio: Float32-Audio (mono oder stereo).
            sr:    Sample-Rate in Hz.
            panns_singing_confidence: Ø PANNs Singing-Score für das Stück (global).

        Returns:
            Liste von SongSegment-Objekten, sortiert nach start_s.
            Laufzeit: ≤ 2 s / Minute Audio (Pflicht §2.52b).
        """
        try:
            return self._analyze_librosa(audio, sr, panns_singing_confidence)
        except Exception as exc:
            logger.warning(
                "SongStructureAnalyzer.analyze_structure fehlgeschlagen: %s — Ersatzpfad: single segment", exc
            )
            duration_s = len(audio[0] if audio.ndim == 2 else audio) / sr
            return [
                SongSegment(
                    start_s=0.0,
                    end_s=float(duration_s),
                    label="unknown",
                    energy_level=0.5,
                    has_vocals=panns_singing_confidence >= 0.35,
                    is_climax=False,
                )
            ]

    def _analyze_librosa(
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing_confidence: float,
    ) -> list[SongSegment]:
        import librosa  # pylint: disable=import-outside-toplevel

        # Mono für Analyse
        if audio.ndim == 2:
            mono = audio.mean(axis=0) if audio.shape[0] == 2 else audio.mean(axis=1)
        else:
            mono = audio
        mono = np.nan_to_num(mono.astype(np.float32), nan=0.0)

        duration_s = len(mono) / sr

        # Boundary-Erkennung: MFCC + Chroma (≤ 2 s / min)
        hop = 512
        mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=12, hop_length=hop)
        chroma = librosa.feature.chroma_stft(y=mono, sr=sr, hop_length=hop)
        features = np.vstack([mfcc, chroma])  # (24, T)

        # Anzahl Boundaries heuristisch: 1 pro ~30 s, min 2, max 12
        k = max(2, min(12, int(duration_s / 30) + 1))
        try:
            boundaries = librosa.segment.agglomerative(features, k)  # type: ignore[attr-defined]
            boundary_times = librosa.frames_to_time(boundaries, sr=sr, hop_length=hop)
        except Exception:
            # Fallback: gleichmäßige Aufteilung
            n = max(2, int(duration_s / 30))
            boundary_times = np.linspace(0, duration_s, n + 1)[1:-1]

        # Segment-Grenzen aufbauen
        times = [0.0, *boundary_times.tolist(), duration_s]
        times = sorted(set(times))

        # RMS-Profil für Energie
        rms = librosa.feature.rms(y=mono, hop_length=hop)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

        global_rms_mean = float(np.mean(rms) + 1e-12)

        segments = []
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            if t1 - t0 < 0.5:
                continue

            # Segment-Energie
            seg_mask = (rms_times >= t0) & (rms_times < t1)
            seg_rms = rms[seg_mask]
            energy_norm = float(np.clip(np.mean(seg_rms) / global_rms_mean, 0.0, 2.0) / 2.0)

            # Vocal-Aktivität im Segment (Proxy: spectral flatness niedrig = tonales Material)
            i0 = max(0, int(t0 * sr))
            i1 = min(len(mono), int(t1 * sr))
            seg_audio = mono[i0:i1]
            has_vocals = self._estimate_vocal_activity(seg_audio, sr, panns_singing_confidence)

            # Label-Heuristik
            label = self._assign_label(i, len(times) - 1, energy_norm, has_vocals, duration_s, t0)

            # Is-Climax: Segment im Top-20% Energie UND hat Vocals
            is_climax = energy_norm > 0.75 and has_vocals

            segments.append(
                SongSegment(
                    start_s=float(t0),
                    end_s=float(t1),
                    label=label,
                    energy_level=float(energy_norm),
                    has_vocals=has_vocals,
                    is_climax=is_climax,
                )
            )

        return (
            segments
            if segments
            else [
                SongSegment(
                    start_s=0.0,
                    end_s=float(duration_s),
                    label="unknown",
                    energy_level=0.5,
                    has_vocals=panns_singing_confidence >= 0.35,
                    is_climax=False,
                )
            ]
        )

    def _estimate_vocal_activity(
        self,
        seg_audio: np.ndarray,
        sr: int,
        panns_confidence: float,
    ) -> bool:
        """Einfacher Vokal-Aktivitäts-Schätzer: spectral flatness + globale PANNs-Konfidenz."""
        if len(seg_audio) < 512:
            return panns_confidence >= 0.35
        try:
            from scipy.signal import welch  # pylint: disable=import-outside-toplevel

            _, psd = welch(seg_audio, fs=sr, nperseg=min(512, len(seg_audio)))
            psd = psd + 1e-12
            # Spectral flatness: hoch = rauschähnlich (kein Vokal); niedrig = tonal (Vokal)
            flatness = float(np.exp(np.mean(np.log(psd))) / (np.mean(psd) + 1e-12))
            # Vokal typisch: flatness < 0.15 UND globalem PANNs-Vertrauen
            return flatness < 0.20 and panns_confidence >= 0.20
        except Exception as e:
            logger.warning("song_structure_analyzer.py::_estimate_vocal_activity Ersatzpfad: %s", e)
            return panns_confidence >= 0.35

    def _assign_label(
        self,
        idx: int,
        n_segments: int,
        energy: float,
        has_vocals: bool,
        duration_s: float,
        t0: float,
    ) -> str:
        """Heuristische Label-Zuweisung basierend auf Position + Energie."""
        relative_pos = idx / max(1, n_segments - 1)  # [0, 1]

        if not has_vocals:
            return "instrumental"

        if relative_pos <= 0.10 or t0 < 15.0:
            return "intro"
        if relative_pos >= 0.90 or (duration_s - t0) < 15.0:
            return "outro"
        if energy > 0.65:
            return "chorus"
        if 0.40 <= relative_pos <= 0.65 and energy < 0.50:
            return "bridge"
        return "verse"

    def get_strength_scalar(
        self,
        segment: SongSegment | None,
        phase_type: str = "default",
    ) -> float:
        """Gibt den Strength-Skalar [0.70, 1.30] für ein Segment zurück.

        Args:
            segment:    Aktuelles SongSegment (None → 1.0).
            phase_type: "nr_strength", "dereverb", "compression", oder "default".

        Returns:
            Strength-Skalar; immer bounded [0.70, 1.30].
        """
        if segment is None:
            return 1.0

        # Climax überschreibt Label
        if segment.is_climax:
            scalar = _CLIMAX_SCALAR
        else:
            label_scalars = _STRENGTH_SCALARS.get(segment.label, _STRENGTH_SCALARS["unknown"])
            scalar = label_scalars.get(phase_type, label_scalars.get("default", 1.0))

        # Hard-Bound (§2.52b Invariante)
        return float(np.clip(scalar, 0.70, 1.30))

    def find_segment_at(
        self,
        segments: list[SongSegment],
        time_s: float,
    ) -> SongSegment | None:
        """Gibt das Segment zurück, in dem time_s liegt."""
        for seg in segments:
            if seg.start_s <= time_s < seg.end_s:
                return seg
        # Fallback: letztes Segment
        return segments[-1] if segments else None
