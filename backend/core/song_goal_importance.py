"""
backend/core/song_goal_importance.py
Aurik 10.0.0 — §2.56 Song-Specific Goal Importance (v10.0.0)

Computes per-song goal weights based on genre, era, material, and audio features.
These weights modulate PMGG regression thresholds, CIG drift tolerance,
FeedbackChain abort logic, and GPOptimizer scalarization.

The 15 Musical Goals form a Pareto front — physically, not all can be maximised
simultaneously (e.g. Brillanz↔Wärme, Transparenz↔Bass_Kraft).  For each song,
the optimal point on this front depends on the musical context.  This module
computes WHERE on the Pareto front the restoration should aim.

Design principles:
  - Weight 1.0 = neutral (same as before this feature existed)
  - Weight > 1.0 = more important → stricter regression threshold
  - Weight < 1.0 = less important → more lenient threshold
  - P1/P2 goals have a floor of 0.7 — they can be de-emphasised slightly
    but never below safety (§0 Primum non nocere)
  - P3–P5 goals: full range [0.3, 2.0]
  - 5-stage architecture: Label (genre/era/material/vocal/restorability)
    → Audio-derived (SNR, BW, dynamics, BPM, defects, tilt, carrier-chain)
    → Psychoacoustic (roughness, sharpness, flatness, tonality, freq-balance, masking, centroid)
    → Vocal/Harmonic (HNR pitch-period AC, STFT coherence, 99.9pctl crest, STFT transient density)
    → Cross-feature interactions (6 superadditive effects)
    → Soft-cap (rational compression k=3.0) + hard bounds
  - Weights are computed ONCE before the phase pipeline (no circular dependency with PMGG)
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MATERIAL_EXPECTED_BW = 20000.0
_RESTORATION_BRILLIANCE_FACTOR = 0.80 + 0.05

# All 15 canonical goal names
ALL_GOAL_NAMES: tuple[str, ...] = (
    "natuerlichkeit",
    "authentizitaet",
    "tonal_center",
    "timbre_authentizitaet",
    "artikulation",
    "transient_energie",
    "emotionalitaet",
    "micro_dynamics",
    "groove",
    "transparenz",
    "waerme",
    "bass_kraft",
    "separation_fidelity",
    "brillanz",
    "spatial_depth",
)

# P1/P2 goals — minimum weight floor (§0 safety)
_P1P2_GOALS: frozenset[str] = frozenset(
    {
        "natuerlichkeit",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
    }
)
_P1P2_WEIGHT_FLOOR: float = 0.70
_WEIGHT_MIN: float = 0.30
_WEIGHT_MAX: float = 2.00


@dataclass(frozen=True)
class SongGoalImportance:
    """Per-song goal importance weights.

    Attributes:
        weights: Dict mapping each of the 15 goal names to a float ∈ [0.3, 2.0].
                 1.0 = neutral.  Higher = more important for THIS song.
        reason: Human-readable description of why these weights were chosen.
        genre_profile: Genre label that contributed to the weights.
        era_profile: Era/decade that contributed.
        material_profile: Primary material type that contributed.
        vocal_detected: Whether vocals were detected (affects Artikulation, Emotionalität).
    """

    weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(ALL_GOAL_NAMES, 1.0))
    reason: str = "default (uniform)"
    genre_profile: str = ""
    era_profile: str = ""
    material_profile: str = ""
    vocal_detected: bool = False

    def weight_of(self, goal: str) -> float:
        """Gibt weight for a goal, 1.0 if unknown zurück."""
        return self.weights.get(goal, 1.0)

    def as_dict(self) -> dict[str, Any]:
        """Serialisation for metadata/logging."""
        return {
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "reason": self.reason,
            "genre_profile": self.genre_profile,
            "era_profile": self.era_profile,
            "material_profile": self.material_profile,
            "vocal_detected": self.vocal_detected,
        }


# ---------------------------------------------------------------------------
# Genre → Goal-Weight Profiles
# ---------------------------------------------------------------------------
# Each profile is a delta from neutral (1.0).  Missing goals stay at 1.0.
# Positive = more important, negative = less important.
# Derived from signal-theory Pareto analysis + musical genre conventions.

_GENRE_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "klassik": {
        "spatial_depth": 1.6,  # Concert hall acoustics are defining
        "micro_dynamics": 1.5,  # ppp→fff dynamics are genre-essential
        "artikulation": 1.4,  # Pizzicato, staccato, attack clarity
        "transparenz": 1.3,  # Separate orchestral voices
        "emotionalitaet": 1.3,  # Expressive dynamics (Adagio vs. Allegro)
        "waerme": 1.1,  # Warm string body
        "brillanz": 0.8,  # Not typically HF-bright
        "groove": 0.6,  # Rubato, not beat-locked
        "bass_kraft": 0.8,  # Rarely sub-bass
    },
    "oper": {
        "artikulation": 1.6,  # Consonant clarity in sung text
        "emotionalitaet": 1.5,  # Dramatic expression
        "spatial_depth": 1.4,  # Stage depth, chorus positioning
        "transparenz": 1.3,  # Voice vs. orchestra separation
        "micro_dynamics": 1.3,  # pp→ff within phrases
        "waerme": 1.1,  # Vocal warmth
        "groove": 0.5,  # Oper is not rhythmic
        "bass_kraft": 0.7,  # Limited sub-bass relevance
        "brillanz": 0.9,  # Natural resonance, not sparkle
    },
    "jazz": {
        "groove": 1.6,  # Swing, syncopation, polyrhythm
        "micro_dynamics": 1.4,  # Solo expression, brush patterns
        "spatial_depth": 1.3,  # Club atmosphere, trio positioning
        "waerme": 1.3,  # Warm horn/bass tone
        "transparenz": 1.2,  # Solo separation from rhythm
        "artikulation": 1.2,  # Brass articulation, piano touch
        "bass_kraft": 1.1,  # Walking bass foundation
        "brillanz": 0.8,  # Not HF-sparkle focused
    },
    "rock": {
        "bass_kraft": 1.4,  # Power chords, kick drum
        "groove": 1.3,  # Driving rhythm
        "brillanz": 1.2,  # Cymbal presence, guitar edge
        "emotionalitaet": 1.2,  # Energy, intensity
        "micro_dynamics": 0.8,  # Often compressed intentionally
        "spatial_depth": 0.8,  # Close-miked, in-your-face
        "transparenz": 1.1,  # Voice vs. band separation
    },
    "metal": {
        "bass_kraft": 1.5,  # Extreme low-end, double-bass
        "brillanz": 1.3,  # Cymbal wash, guitar harmonics
        "groove": 1.3,  # Blast beats, breakdowns
        "emotionalitaet": 1.2,  # Intensity
        "micro_dynamics": 0.6,  # Genre-definingly compressed
        "spatial_depth": 0.6,  # Dense wall of sound
        "waerme": 0.7,  # Cold, aggressive tone preferred
    },
    "electronic": {
        "bass_kraft": 1.6,  # Sub-bass is genre-defining
        "groove": 1.4,  # Beat precision
        "brillanz": 1.3,  # HF shimmer, synth sparkle
        "transparenz": 1.2,  # Mix clarity
        "spatial_depth": 1.1,  # Stereo panorama
        "micro_dynamics": 0.6,  # Heavily compressed genre
        "waerme": 0.7,  # Cold, precise production typical
        "artikulation": 0.7,  # No acoustic attack to preserve
    },
    "hip-hop": {
        "bass_kraft": 1.7,  # 808s, sub-bass
        "groove": 1.5,  # Beat, flow
        "artikulation": 1.3,  # Vocal clarity, rap intelligibility
        "transparenz": 1.2,  # Voice vs. beat separation
        "brillanz": 1.1,  # Hi-hat shimmer
        "micro_dynamics": 0.6,  # Heavily compressed
        "spatial_depth": 0.8,  # Centred vocal
        "waerme": 0.9,  # Neutral
    },
    "pop": {
        "transparenz": 1.3,  # Clean, polished mix
        "artikulation": 1.2,  # Vocal intelligibility
        "emotionalitaet": 1.2,  # Emotional connection
        "brillanz": 1.1,  # Modern bright mix
        "groove": 1.1,  # Danceability
        "bass_kraft": 1.0,  # Solid but not extreme
        "micro_dynamics": 0.8,  # Mastering compression
        "spatial_depth": 0.9,  # Centred vocal focus
    },
    "schlager": {
        "waerme": 1.4,  # Warm, inviting tone
        "emotionalitaet": 1.3,  # Emotional singalong
        "groove": 1.2,  # Dance rhythm
        "artikulation": 1.2,  # German vocal clarity
        "transparenz": 1.1,  # Vocal-band separation
        "bass_kraft": 0.9,  # Moderate
        "brillanz": 0.8,  # Not HF-bright by convention
        "micro_dynamics": 0.9,  # Fairly compressed production
        "spatial_depth": 0.8,  # Centred vocal
    },
    "soul/r&b": {
        "emotionalitaet": 1.5,  # Vocal soul expression
        "waerme": 1.4,  # Warm, intimate studio tone
        "groove": 1.3,  # Rhythmic pocket
        "artikulation": 1.2,  # Vocal nuance
        "micro_dynamics": 1.2,  # Dynamic swells
        "spatial_depth": 1.1,  # Studio ambience
        "bass_kraft": 1.1,  # Solid bass foundation
        "brillanz": 0.8,  # Not aggressive HF
    },
    "blues": {
        "waerme": 1.5,  # Tube amp warmth is defining
        "emotionalitaet": 1.4,  # Expression, bending, vibrato
        "groove": 1.3,  # Shuffle, swing
        "artikulation": 1.2,  # Slide, pick attack
        "micro_dynamics": 1.2,  # Quiet verse → explosive solo
        "spatial_depth": 1.1,  # Club atmosphere
        "brillanz": 0.8,  # Not sparkly, warm grit
        "bass_kraft": 1.0,  # Walking/boogie bass
    },
    "country": {
        "artikulation": 1.4,  # Banjo, mandolin, flat-pick attack
        "waerme": 1.2,  # Nashville warmth
        "groove": 1.2,  # Train beat, two-step
        "spatial_depth": 1.1,  # Nashville reverb
        "emotionalitaet": 1.1,  # Storytelling
        "transparenz": 1.1,  # Instrument separation
        "bass_kraft": 0.9,  # Moderate
        "brillanz": 1.0,  # Bright steel guitar
    },
    "folk": {
        "artikulation": 1.5,  # Finger-picking, string attack
        "waerme": 1.3,  # Natural acoustic warmth
        "spatial_depth": 1.3,  # Room atmosphere (small venue)
        "micro_dynamics": 1.3,  # Intimate dynamics
        "emotionalitaet": 1.2,  # Storytelling expression
        "transparenz": 1.1,  # Voice vs. guitar clarity
        "bass_kraft": 0.7,  # Minimal sub-bass
        "brillanz": 0.9,  # Not aggressive HF
        "groove": 0.8,  # Free timing
    },
    "reggae": {
        "groove": 1.7,  # One-drop, skank — genre-DEFINING
        "spatial_depth": 1.5,  # Dub echo/reverb is sacred
        "bass_kraft": 1.4,  # Massive bass lines
        "waerme": 1.3,  # Warm analogue tone
        "emotionalitaet": 1.1,  # Message/vibe
        "brillanz": 0.7,  # Not HF-focused
        "micro_dynamics": 0.8,  # Fairly compressed
        "artikulation": 0.9,  # Not precision-attack genre
    },
    "latin": {
        "groove": 1.6,  # Clave, rhythmic precision
        "artikulation": 1.4,  # Conga, bongo, timbales attack
        "bass_kraft": 1.2,  # Tumbao bass line
        "spatial_depth": 1.1,  # Live ensemble feel
        "waerme": 1.1,  # Latin brass warmth
        "emotionalitaet": 1.1,  # Musical fire
        "brillanz": 1.0,  # Brass shimmer
        "micro_dynamics": 1.0,  # Dynamic ensemble
    },
    "funk": {
        "groove": 1.7,  # THE groove genre
        "bass_kraft": 1.4,  # Slap bass, synth bass
        "artikulation": 1.3,  # Brass stabs, slap-attack
        "brillanz": 1.1,  # Wah-wah, hi-hat sizzle
        "transparenz": 1.1,  # Tight mix separation
        "micro_dynamics": 0.9,  # Tight but dynamic
        "spatial_depth": 0.8,  # Dense, tight mix
        "waerme": 1.0,  # Neutral-warm
    },
    "gospel": {
        "spatial_depth": 1.6,  # Church reverb is sacred
        "emotionalitaet": 1.5,  # Spiritual expression
        "artikulation": 1.3,  # Choir diction
        "waerme": 1.2,  # Warm vocal tone
        "micro_dynamics": 1.2,  # Quiet prayer → rejoicing
        "transparenz": 1.1,  # Solo vs. choir
        "groove": 0.9,  # Some groove, not primary
        "bass_kraft": 0.9,  # Moderate
        "brillanz": 0.8,  # Not aggressive HF
    },
}

# ---------------------------------------------------------------------------
# Era → Goal-Weight Modifiers (multiplicative on genre weights)
# ---------------------------------------------------------------------------
# Older recordings have physical bandwidth limits → adjust expectations.

_ERA_WEIGHT_MODIFIERS: dict[str, dict[str, float]] = {
    "1900er": {
        "brillanz": 0.3,  # No HF above ~4 kHz existed
        "spatial_depth": 0.3,  # Mono only
        "bass_kraft": 0.5,  # Limited LF response
        "waerme": 1.3,  # Emphasise what IS there
        "artikulation": 1.3,  # Voice clarity matters most
    },
    "1910er": {
        "brillanz": 0.3,
        "spatial_depth": 0.3,
        "bass_kraft": 0.5,
        "waerme": 1.3,
        "artikulation": 1.3,
    },
    "1920er": {
        "brillanz": 0.4,  # Electrical recording ~7 kHz
        "spatial_depth": 0.4,  # Early stereo experiments rare
        "bass_kraft": 0.6,
        "waerme": 1.2,
        "artikulation": 1.2,
    },
    "1930er": {
        "brillanz": 0.5,
        "spatial_depth": 0.4,
        "bass_kraft": 0.7,
        "waerme": 1.2,
    },
    "1940er": {
        "brillanz": 0.6,
        "spatial_depth": 0.5,
        "bass_kraft": 0.8,
        "waerme": 1.1,
    },
    "1950er": {
        "brillanz": 0.7,  # Early hi-fi, ~12 kHz
        "spatial_depth": 0.7,  # Early stereo
        "bass_kraft": 0.9,
    },
    "1960er": {
        "brillanz": 0.85,  # Improving HF
        "spatial_depth": 0.9,  # Stereo standard
    },
    "1970er": {
        "brillanz": 0.9,
        "waerme": 1.1,  # Analogue warmth era
    },
    "1980er": {
        "brillanz": 1.0,  # Full bandwidth available
        "spatial_depth": 1.05,  # Creative stereo use
    },
    # 1990er+ : no modification needed (full modern capabilities)
}

# ---------------------------------------------------------------------------
# Material → Goal-Weight Modifiers (multiplicative)
# ---------------------------------------------------------------------------
# Physical carrier limits affect achievability.

_MATERIAL_WEIGHT_MODIFIERS: dict[str, dict[str, float]] = {
    "wax_cylinder": {
        "brillanz": 0.3,
        "spatial_depth": 0.3,
        "bass_kraft": 0.4,
        "transparenz": 0.6,
        "artikulation": 1.3,  # Intelligibility is paramount
        "waerme": 1.2,
    },
    "shellac": {
        "brillanz": 0.5,
        "spatial_depth": 0.4,
        "bass_kraft": 0.6,
        "transparenz": 0.8,
        "waerme": 1.2,
        "artikulation": 1.2,
    },
    "wire_recording": {
        "brillanz": 0.4,
        "spatial_depth": 0.3,
        "bass_kraft": 0.5,
        "transparenz": 0.7,
        "artikulation": 1.3,
    },
    "vinyl": {
        "waerme": 1.15,  # Analogue warmth to preserve
        "brillanz": 0.9,  # RIAA limits HF slightly
        "bass_kraft": 1.05,  # RIAA bass boost
    },
    "reel_tape": {
        "waerme": 1.15,  # Tape saturation warmth
        "micro_dynamics": 1.1,  # Tape compression character
    },
    "tape": {
        "waerme": 1.1,
        "brillanz": 0.9,  # HF roll-off from tape age
    },
    "cassette": {
        "waerme": 1.05,
        "brillanz": 0.85,  # Limited HF
        "transparenz": 0.9,  # Inherently muffled
    },
    "mp3_low": {
        "transparenz": 1.2,  # Codec smearing needs fixing
        "brillanz": 1.1,  # Codec kills HF
        "separation_fidelity": 1.1,
        "waerme": 0.9,  # Digital, no analogue warmth
    },
    "mp3_high": {
        "transparenz": 1.05,
        "brillanz": 1.05,
    },
    "cd_digital": {
        # Neutral — full bandwidth, no carrier limitations
    },
    "dat": {
        # Neutral — full bandwidth
    },
    "minidisc": {
        "transparenz": 1.05,
        "brillanz": 1.05,
    },
    "radio_broadcast": {
        "transparenz": 1.1,
        "brillanz": 0.9,  # Limited bandwidth
        "bass_kraft": 0.9,  # Limited LF
        "artikulation": 1.15,  # Speech clarity important
    },
    "optical_film": {
        "spatial_depth": 1.1,  # Cinematic sound design
        "artikulation": 1.2,  # Dialogue clarity
        "waerme": 1.1,  # Analogue film warmth
        "brillanz": 0.85,  # Limited optical bandwidth
    },
}


# ---------------------------------------------------------------------------
# §C10 Active Listener Calibration — Bayesian EMA Feedback Store (v10.0.0)
# ---------------------------------------------------------------------------


@dataclass
class UserFeedbackEntry:
    """Single listener feedback record for a processed song."""

    genre: str
    material: str
    era: str
    rating_thumbs_up: bool  # True = thumbs-up; False = thumbs-down
    winning_goals: list[str]  # Goals that were particularly good
    failing_goals: list[str]  # Goals that disappointed the listener


class SongGoalFeedbackStore:
    """§C10 Persistent listener-calibrated goal weight adjustments.

    Stores up to MAX_ENTRIES feedback entries in sessions/goal_feedback.json.
    On each new entry, applies a Bayesian EMA update to per-goal weight nudges:
      thumbs_up  → w *= (1 + 0.05 × gradient_sign)
      thumbs_down → w *= (1 - 0.05 × gradient_sign)
    where gradient_sign = +1 for winning_goals, -1 for failing_goals.

    Calibrated nudges are blended into estimate_goal_importance() at
    FEEDBACK_BLEND_WEIGHT = 0.15 (advisory, non-overriding).
    """

    MAX_ENTRIES: int = 1000
    FEEDBACK_BLEND_WEIGHT: float = 0.15
    _EMA_LR: float = 0.05
    _PERSIST_PATH_REL: str = "sessions/goal_feedback.json"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[UserFeedbackEntry] = []
        self._nudges: dict[str, float] = dict.fromkeys(ALL_GOAL_NAMES, 1.0)
        self._loaded = False

    def _persist_path(self) -> str:
        base = os.path.join(os.path.dirname(__file__), "..", "..", self._PERSIST_PATH_REL)
        return os.path.normpath(base)

    def _load(self) -> None:
        """Lädt persisted nudges from disk (idempotent)."""
        if self._loaded:
            return
        try:
            p = self._persist_path()
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    nudges = data.get("nudges", {})
                    for g in ALL_GOAL_NAMES:
                        v = nudges.get(g)
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            self._nudges[g] = float(np.clip(v, 0.50, 2.00))
                    entries_raw = data.get("entries", [])
                    for e in entries_raw[-self.MAX_ENTRIES :]:
                        try:
                            self._entries.append(UserFeedbackEntry(**e))
                        except Exception as _entry_exc:
                            logger.debug("FeedbackStore: invalid entry uebersprungen: %s", _entry_exc)
        except Exception as _load_exc:
            logger.debug("§C10 FeedbackStore laden uebersprungen: %s", _load_exc)
        finally:
            self._loaded = True

    def _save(self) -> None:
        """Persist current nudges + entries to disk (non-blocking)."""
        try:
            p = self._persist_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            payload = {
                "nudges": {k: round(float(v), 6) for k, v in self._nudges.items()},
                "entries": [
                    {
                        "genre": e.genre,
                        "material": e.material,
                        "era": e.era,
                        "rating_thumbs_up": e.rating_thumbs_up,
                        "winning_goals": e.winning_goals,
                        "failing_goals": e.failing_goals,
                    }
                    for e in self._entries[-self.MAX_ENTRIES :]
                ],
            }
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as _save_exc:
            logger.debug("§C10 FeedbackStore speichern uebersprungen: %s", _save_exc)

    def record_feedback(self, entry: UserFeedbackEntry) -> None:
        """Record a new listener feedback entry and update nudges via Bayesian EMA."""
        with self._lock:
            self._load()
            self._entries.append(entry)
            if len(self._entries) > self.MAX_ENTRIES:
                self._entries = self._entries[-self.MAX_ENTRIES :]

            gradient_sign = +1.0 if entry.rating_thumbs_up else -1.0

            for g in entry.winning_goals:
                if g in self._nudges:
                    self._nudges[g] = float(np.clip(self._nudges[g] * (1.0 + self._EMA_LR * gradient_sign), 0.50, 2.00))
            for g in entry.failing_goals:
                if g in self._nudges:
                    self._nudges[g] = float(np.clip(self._nudges[g] * (1.0 - self._EMA_LR * gradient_sign), 0.50, 2.00))

            self._save()
            logger.info(
                "§C10 Listener feedback recorded: thumbs_%s winning=%s failing=%s",
                "up" if entry.rating_thumbs_up else "down",
                entry.winning_goals,
                entry.failing_goals,
            )

    def get_nudges(self) -> dict[str, float]:
        """Gibt a copy of the current per-goal nudges (thread-safe) zurück."""
        with self._lock:
            self._load()
            return dict(self._nudges)


# Module-level singleton
_feedback_store: SongGoalFeedbackStore | None = None
_feedback_store_lock = threading.Lock()


def get_feedback_store() -> SongGoalFeedbackStore:
    """Gibt the global SongGoalFeedbackStore singleton (thread-safe) zurück."""
    global _feedback_store  # pylint: disable=global-statement
    if _feedback_store is None:
        with _feedback_store_lock:
            if _feedback_store is None:
                _feedback_store = SongGoalFeedbackStore()
    return _feedback_store


def estimate_goal_importance(
    genre_label: str = "",
    era_decade: str = "",
    material_type: str = "unknown",
    vocal_detected: bool = False,
    vocal_confidence: float = 0.0,
    restorability_score: float = 50.0,
    is_studio_2026: bool = False,
    # ── Audio-derived features (§2.56 v10.0.0 — real per-song analysis) ──
    snr_db: float | None = None,
    effective_bandwidth_hz: float | None = None,
    dynamic_range_db: float | None = None,
    stereo_mono_compat: float | None = None,
    bpm: float | None = None,
    defect_severities: dict[str, float] | None = None,
    spectral_tilt_db_per_oct: float | None = None,
    # ── Carrier-chain features (§2.46/§2.46a — Tonträgerketten-Inversion) ──
    transfer_generation_count: int | None = None,
    cumulative_hf_loss_db: float | None = None,
    source_fidelity_confidence: float | None = None,
    # ── Psychoacoustic features (Zwicker/Aures/ISO 11172-3) ──
    roughness: float | None = None,
    sharpness: float | None = None,
    spectral_flatness: float | None = None,
    tonality: float | None = None,
    frequency_balance: dict[str, float] | None = None,
    masked_components_ratio: float | None = None,
    perceptual_centroid_bark: float | None = None,
    # ── Vocal/harmonic/transient features (music preservation) ──
    harmonic_to_noise_ratio_db: float | None = None,
    harmonic_coherence: float | None = None,
    crest_factor_db: float | None = None,
    transient_density: float | None = None,
) -> SongGoalImportance:
    """Berechnet per-song goal importance weights.

    Two-stage approach:
      1. **Label stage** (genre + era + material) — coarse Pareto positioning
      2. **Audio stage** (SNR, bandwidth, dynamics, stereo, defects) — fine-tuning
         based on the ACTUAL signal content of THIS specific song

    The weights are multiplicative modifiers on PMGG regression thresholds:
      - weight 1.5 for goal X → PMGG treats 0.02 regression in X as 0.03 (stricter)
      - weight 0.5 for goal X → PMGG treats 0.02 regression in X as 0.01 (more lenient)

    This resolves the Pareto trade-off: the system knows WHICH goals to prioritise
    for THIS specific song, enabling informed decisions when goals conflict.

    Args:
        genre_label: Genre classification result (e.g. "jazz", "schlager").
        era_decade: Era string (e.g. "1970er").
        material_type: Primary carrier material.
        vocal_detected: Whether PANNs detected singing.
        vocal_confidence: PANNs singing confidence [0, 1].
        restorability_score: 0–100.
        is_studio_2026: Studio 2026 mode (different balance).
        snr_db: Measured input SNR in dB (None = not available).
        effective_bandwidth_hz: Measured upper frequency limit where energy > -20dB.
        dynamic_range_db: Measured dynamic range in dB.
        stereo_mono_compat: Mono compatibility [0,1]; 1.0=perfect mono compat.
        bpm: Detected beats per minute.
        defect_severities: Dict of DefectType→severity from DefectScanner.
        spectral_tilt_db_per_oct: Measured spectral slope (positive=bright, negative=dark).
        transfer_generation_count: Number of carrier generations in the transfer chain
            (e.g. shellac→tape→CD→MP3 = 4).  More generations = more cumulative loss.
        cumulative_hf_loss_db: Estimated total HF loss across the entire carrier chain.
        source_fidelity_confidence: Confidence of the transfer-chain model [0, 1].
        roughness: Zwicker roughness [0, 1].  High = harsh modulation in critical bands.
        sharpness: Aures sharpness [0, 1].  High = strong HF emphasis.
        spectral_flatness: Spectral flatness [0, 1].  High = noise-like, low = tonal.
        tonality: Tonality score [0, 1].  High = strong tonal/harmonic content.
        frequency_balance: Dict with keys 'bass', 'mid', 'treble', 'air' → energy ratios.
        masked_components_ratio: Fraction of spectral components below masking threshold [0, 1].
        perceptual_centroid_bark: Spectral centroid in Bark scale (perceptual gravity center).
        harmonic_to_noise_ratio_db: HNR in dB.  High (>20) = clean harmonics; Low (<10) = noisy.
        harmonic_coherence: Autocorrelation pitch coherence [0, 1].  High = stable tonal content.
        crest_factor_db: Peak-to-RMS ratio in dB.  High (>15) = dynamic; Low (<8) = compressed.
        transient_density: Onset events per second.  High (>5) = percussive.

    Returns:
        SongGoalImportance with 15 weights.
    """
    weights: dict[str, float] = dict.fromkeys(ALL_GOAL_NAMES, 1.0)

    # Defensive normalisation: any argument may arrive as int/enum/None from
    # callers that ignore the type annotation.  str() always succeeds; the
    # ternary result is then always str, so .strip() is safe.
    def _to_str(v: object, default: str = "") -> str:
        if v is None:
            return default
        try:
            return str(v).strip().lower()
        except Exception as e:
            logger.warning("song_goal_importance.py::_to_str Ersatzpfad: %s", e)
            return default

    _genre = _to_str(genre_label)
    # era_decade may be int (1970) or str ("1970er") — normalise to "1970er" form
    _era_raw = era_decade
    if isinstance(_era_raw, (int, float)) and _era_raw > 1800:
        _era = f"{int(_era_raw) // 10 * 10}er"
    elif isinstance(_era_raw, str) and _era_raw:
        _era = _era_raw.strip().lower()
    elif _era_raw is not None:
        _era = _to_str(_era_raw)
    else:
        _era = ""
    _mat = _to_str(material_type, default="unknown") or "unknown"

    reasons: list[str] = []

    # --- Step 1: Genre profile (primary driver) ---
    # Robust alias resolution: GenreClassifier may return full names
    # (e.g. "Deutscher Schlager") while profiles use short keys ("schlager").
    _GENRE_ALIASES: dict[str, str] = {
        "soul": "soul/r&b",
        "r&b": "soul/r&b",
        "rnb": "soul/r&b",
        "deutscher schlager": "schlager",
        "dt. schlager": "schlager",
        "classic rock": "rock",
        "hard rock": "rock",
        "indie rock": "rock",
        "heavy metal": "metal",
        "death metal": "metal",
        "thrash metal": "metal",
        "hiphop": "hip-hop",
        "rap": "hip-hop",
        "edm": "electronic",
        "techno": "electronic",
        "electronica": "electronic",
        "elektronik": "electronic",
        "classical": "klassik",
        "orchestra": "klassik",
        "opera": "oper",
        "country & western": "country",
        "latin pop": "pop",
    }
    _genre_key = _GENRE_ALIASES.get(_genre, _genre)

    genre_profile = _GENRE_WEIGHT_PROFILES.get(_genre_key, {})
    if genre_profile:
        for goal, w in genre_profile.items():
            weights[goal] = w
        reasons.append(f"genre={_genre_key}")
    else:
        reasons.append(f"genre={_genre_key or 'unknown'} (no profile, neutral)")

    # --- Step 2: Era modifier (multiplicative on genre weights) ---
    era_mod = _ERA_WEIGHT_MODIFIERS.get(_era, {})
    if era_mod:
        for goal, m in era_mod.items():
            weights[goal] *= m
        reasons.append(f"era={_era}")

    # --- Step 3: Material modifier (multiplicative) ---
    mat_mod = _MATERIAL_WEIGHT_MODIFIERS.get(_mat, {})
    if mat_mod:
        for goal, m in mat_mod.items():
            weights[goal] *= m
        reasons.append(f"material={_mat}")

    # --- Step 4: Vocal detection boost ---
    if vocal_detected and vocal_confidence > 0.25:
        # Vocal content → boost goals whose psychoacoustic JND is reduced by
        # the presence of a singing voice (primary stream in vocal music).
        #
        # Sources driving the specific goal assignments:
        #   Artikulation:          London (2012) 2nd ed. + Repp & Su (2013)
        #                          Psychon Bull Rev 20:403 — timing JND ~5-10 ms
        #                          for vocal music; consonant clarity critical.
        #   Emotionalität:         Juslin (2019) "Musical Emotions Explained" OUP;
        #                          Zentner et al. (2008) Emotion 8:494 — vocal is
        #                          the primary carrier of musical emotion.
        #   Authentizität:         Kreiman & Sidtis (2011) "Foundations of Voice
        #                          Studies" — voice quality is the defining attribute
        #                          by which listeners identify singers; voice is the
        #                          most salient perceptual stream in vocal music.
        #   Tonal center:          Marjieh, Harrison, Lee, Deligiannaki & Jacoby
        #                          (2023) Music Percept. 40:183 — tonal hierarchy
        #                          remains highly salient even without explicit tonal
        #                          context; melody carried by voice anchors key.
        #   Transparenz:           Bregman (1990) + Toole (2018) — voice/accompaniment
        #                          clarity is the primary intelligibility axis.
        #   Timbre authentizität:  Caclin et al. (2005) JASA 118:2925 + McAdams
        #                          (2019) Curr Biol — timbral JND ≈1 % for musical
        #                          sounds; vocal timbre most discriminable by listeners.
        #   Separation fidelity:   Bregman (1990) "Auditory Scene Analysis" Ch.2;
        #                          McDermott (2009) Curr Biol 19:R1115 (cocktail-party
        #                          effect) — voice segregation from instrumentation is
        #                          the primary auditory scene analysis task in vocal music.
        _vocal_factor = 0.5 + 0.5 * float(np.clip(vocal_confidence, 0.0, 1.0))
        weights["artikulation"] *= 1.0 + 0.30 * _vocal_factor  # London (2012), Repp & Su (2013)
        weights["emotionalitaet"] *= 1.0 + 0.20 * _vocal_factor  # Juslin (2019)
        weights["authentizitaet"] *= 1.0 + 0.20 * _vocal_factor  # Kreiman & Sidtis (2011)
        weights["tonal_center"] *= 1.0 + 0.12 * _vocal_factor  # Marjieh et al. (2023)
        weights["transparenz"] *= 1.0 + 0.15 * _vocal_factor  # Bregman (1990), Toole (2018)
        weights["timbre_authentizitaet"] *= 1.0 + 0.10 * _vocal_factor  # Caclin et al. (2005)
        weights["separation_fidelity"] *= 1.0 + 0.10 * _vocal_factor  # Bregman (1990), McDermott (2009)
        reasons.append(f"vocal(conf={vocal_confidence:.2f})")

    # --- Step 5: Restorability adjustment ---
    # Very degraded material (low restorability): reduce expectations for
    # physically limited goals, boost core preservation goals.
    if restorability_score < 30:
        _degradation_factor = 1.0 - restorability_score / 60.0  # 0.5 at rest=0, 0.0 at rest=30
        weights["brillanz"] *= 1.0 - 0.3 * _degradation_factor
        weights["spatial_depth"] *= 1.0 - 0.3 * _degradation_factor
        weights["separation_fidelity"] *= 1.0 - 0.2 * _degradation_factor
        weights["natuerlichkeit"] *= 1.0 + 0.15 * _degradation_factor
        weights["authentizitaet"] *= 1.0 + 0.15 * _degradation_factor
        reasons.append(f"degraded(rest={restorability_score:.0f})")

    # --- Step 6: Mode-specific adjustment (§v10.15) ---
    # Studio 2026 aims for modern studio quality → boost enhancement goals.
    # Restoration Mode aims for naturalness/clarity/original character →
    # boost preservation goals, de-emphasize enhancement.
    if is_studio_2026:
        weights["brillanz"] *= 1.2
        weights["transparenz"] *= 1.15
        weights["bass_kraft"] *= 1.1
        weights["spatial_depth"] *= 1.15
        weights["separation_fidelity"] *= 1.1
        reasons.append("studio_2026")
    else:
        # §v10.15 Restoration Mode: „Natürlichkeit, Klarheit und der
        # ursprüngliche Charakter am Tag der Aufnahme im Studio".
        # Preservation-Ziele werden STRENGER gewichtet → PMGG
        # toleriert weniger Regression in diesen Dimensionen.
        # Enhancement-Ziele werden GELOCKERT → leichte Brillanz-
        # oder Bass-Änderungen werden toleriert.
        weights["natuerlichkeit"] *= 1.25
        weights["authentizitaet"] *= 1.20
        weights["tonal_center"] *= 1.15
        weights["timbre_authentizitaet"] *= 1.15
        weights["transparenz"] *= 1.10
        weights["artikulation"] *= 1.10
        weights["emotionalitaet"] *= 1.05
        # Enhancement-Goals sanft zurücknehmen
        weights["brillanz"] *= _RESTORATION_BRILLIANCE_FACTOR
        weights["bass_kraft"] *= 0.90
        weights["spatial_depth"] *= 0.90
        reasons.append("restoration_mode")

    # ===================================================================
    # STAGE 2: Audio-derived per-song fine-tuning
    # Labels (genre/era/material) are COARSE positioning on the Pareto front.
    # The actual audio tells us WHERE this specific song sits.
    # ===================================================================

    # --- Step 6b: SNR-based adjustment ---
    # Low SNR → transparenz/brillanz are hard to achieve, noise removal is key
    # High SNR → transparenz is already good, focus on musical goals
    if snr_db is not None:
        # §2.59: Kontinuierliche SNR-Messung statt hartem 15dB-Cutoff.
        # transparenz wichtiger bei Rauschen, brillanz limitiert bei niedrigem SNR.
        if snr_db is not None:
            _snr_ratio = min(max(float(snr_db) / 60.0, 0.0), 1.0)  # 0 dB→0.0, 60 dB→1.0
            if float(snr_db) < 30.0:
                weights["transparenz"] *= 1.0 + 0.30 * (1.0 - _snr_ratio)
                weights["brillanz"] *= 0.70 + 0.30 * _snr_ratio
                weights["spatial_depth"] *= 0.70 + 0.30 * _snr_ratio
                reasons.append(f"snr({snr_db:.0f}dB→transparenz×{1.0 + 0.30 * (1.0 - _snr_ratio):.2f})")
            if float(snr_db) > 35.0:
                weights["natuerlichkeit"] *= 0.95 + 0.25 * _snr_ratio
                weights["authentizitaet"] *= 0.95 + 0.25 * _snr_ratio
                weights["micro_dynamics"] *= 0.90 + 0.30 * _snr_ratio
                reasons.append(f"snr_clean({snr_db:.0f}dB)")

    # --- Step 6c: Bandwidth-based adjustment ---
    # §2.59: Kontinuierliche Messung statt Stufen. Brillanz skaliert
    # proportional zur verfuegbaren Bandbreite (0 Hz → 0.6, 20 kHz → 1.1).
    if effective_bandwidth_hz is not None and effective_bandwidth_hz > 0:
        _bw_ratio = min(float(effective_bandwidth_hz) / MATERIAL_EXPECTED_BW, 1.0)
        _brillanz_factor = 0.6 + 0.5 * _bw_ratio
        weights["brillanz"] *= _brillanz_factor
        if effective_bandwidth_hz < 8000.0:
            weights["waerme"] *= 1.0 + 0.2 * (1.0 - _bw_ratio)
        reasons.append(f"bw({effective_bandwidth_hz:.0f}Hz→brillanz×{_brillanz_factor:.2f})")

    # --- Step 6d: Dynamic range adjustment ---
    # Compressed signals → don't prioritise micro_dynamics restoration
    # Wide dynamics → micro_dynamics is genuinely important
    # §2.59: Kontinuierliche Dynamik-Messung.
    if dynamic_range_db is not None:
        _dr = float(dynamic_range_db)
        if _dr < 20.0:
            # Heavily compressed — micro_dynamics was intentional or irrecoverable
            weights["micro_dynamics"] *= 0.7
            weights["groove"] *= 1.1  # Focus on rhythmic impact instead
            reasons.append(f"dyn_compressed({dynamic_range_db:.0f}dB)")
        elif dynamic_range_db > 50.0:
            # Wide dynamics — this is genuinely dynamic music
            weights["micro_dynamics"] *= 1.2
            weights["emotionalitaet"] *= 1.1
            reasons.append(f"dyn_wide({dynamic_range_db:.0f}dB)")

    # --- Step 6e: Stereo field adjustment ---
    # Mono-incompatible input → spatial goals are critical to fix
    # Already perfect stereo → spatial goals less urgent
    if stereo_mono_compat is not None:
        if stereo_mono_compat < 0.4:
            # Critical stereo issues — spatial goals very important to fix
            weights["spatial_depth"] *= 1.3
            weights["separation_fidelity"] *= 1.2
            reasons.append(f"stereo_critical({stereo_mono_compat:.2f})")
        elif stereo_mono_compat > 0.9:
            # Excellent stereo field — don't waste pipeline budget
            weights["spatial_depth"] *= 0.9
            reasons.append(f"stereo_good({stereo_mono_compat:.2f})")

    # --- Step 6f: BPM-based groove importance ---
    # Fast/danceable tempos → groove is critical
    # Very slow → groove matters less, expression matters more
    if bpm is not None and bpm > 0:
        if bpm > 110.0:
            weights["groove"] *= 1.15
            reasons.append(f"bpm_fast({bpm:.0f})")
        elif bpm < 60.0:
            weights["groove"] *= 0.8
            weights["emotionalitaet"] *= 1.1
            weights["micro_dynamics"] *= 1.1
            reasons.append(f"bpm_slow({bpm:.0f})")

    # --- Step 6g: Defect-driven goal adjustment ---
    # Actual detected defects tell us which goals need protection
    # §2.59 Fix (2026-07-09): Defekt-Namen auf DefectType.values() abgestimmt.
    # Keys stammen aus DefectType.name.lower(), z.B. DefectType.CLICKS → "clicks".
    if defect_severities:
        _ds = defect_severities
        # Heavy noise → transparenz critical
        _noise_sev = max(
            _ds.get("high_freq_noise", 0.0),
            _ds.get("modulation_noise", 0.0),
            _ds.get("quantization_noise", 0.0),
            _ds.get("hum", 0.0),
        )
        if _noise_sev > 0.5:
            weights["transparenz"] *= 1.15
            reasons.append(f"defect_noise({_noise_sev:.2f})")

        # Heavy crackle/click → groove/timing preservation critical
        _crackle_sev = max(
            _ds.get("crackle", 0.0),
            _ds.get("clicks", 0.0),
        )
        if _crackle_sev > 0.5:
            weights["groove"] *= 1.1
            weights["artikulation"] *= 1.1
            reasons.append(f"defect_crackle({_crackle_sev:.2f})")

        # HF loss detected → brillanz physically limited
        _hf_loss_sev = max(
            _ds.get("bandwidth_loss", 0.0),
            _ds.get("hf_remanence_loss", 0.0),
        )
        if _hf_loss_sev > 0.5:
            weights["brillanz"] *= _brillanz_factor if effective_bandwidth_hz else 0.85
            reasons.append(f"defect_hf_loss({_hf_loss_sev:.2f})")

        # Wow/flutter → timing/groove at risk
        _wow_sev = max(_ds.get("wow", 0.0), _ds.get("flutter", 0.0))
        if _wow_sev > 0.3:
            weights["groove"] *= 1.15
            weights["tonal_center"] *= 1.1
            reasons.append(f"defect_wow({_wow_sev:.2f})")

    # --- Step 6h: Spectral tilt adjustment ---
    # Dark signal → don't over-emphasise brillanz
    # Bright signal → brillanz is genuinely present, protect it
    if spectral_tilt_db_per_oct is not None:
        if spectral_tilt_db_per_oct < -4.0:
            # Very dark → limited HF, don't demand brillanz
            weights["brillanz"] *= _brillanz_factor if effective_bandwidth_hz else 0.85
            weights["waerme"] *= 1.05
            reasons.append(f"tilt_dark({spectral_tilt_db_per_oct:.1f}dB/oct)")
        elif spectral_tilt_db_per_oct > -1.0:
            # Bright signal — protect that brightness
            weights["brillanz"] *= 1.1
            reasons.append(f"tilt_bright({spectral_tilt_db_per_oct:.1f}dB/oct)")

    # --- Step 6i: Carrier-chain degradation (§2.46/§2.46a) ---
    # Multi-generation transfers accumulate losses.  The deeper the chain,
    # the more the original studio sound has degraded.  Adapt goal weights
    # so that (a) physically-limited goals get relaxed, (b) core-fidelity
    # goals get boosted, and (c) carrier-repair phases get more room.
    if transfer_generation_count is not None and transfer_generation_count >= 2:
        # Confidence-weighted: trust the chain model more when confidence is high
        _chain_conf = float(np.clip(source_fidelity_confidence or 0.6, 0.3, 1.0))
        # Depth factor: 0.0 @ gen=1, 0.33 @ gen=2, 0.67 @ gen=3, 1.0 @ gen≥4
        _chain_depth = float(np.clip((transfer_generation_count - 1) / 3.0, 0.0, 1.0))
        _chain_factor = _chain_depth * _chain_conf

        # Physically limited goals: each generation removes HF, dynamics, stereo
        weights["brillanz"] *= 1.0 - 0.20 * _chain_factor
        weights["spatial_depth"] *= 1.0 - 0.20 * _chain_factor
        weights["separation_fidelity"] *= 1.0 - 0.15 * _chain_factor
        weights["micro_dynamics"] *= 1.0 - 0.10 * _chain_factor

        # Core fidelity: the more degraded, the more critical to preserve essence
        weights["natuerlichkeit"] *= 1.0 + 0.15 * _chain_factor
        weights["authentizitaet"] *= 1.0 + 0.15 * _chain_factor
        weights["timbre_authentizitaet"] *= 1.0 + 0.12 * _chain_factor
        weights["tonal_center"] *= 1.0 + 0.10 * _chain_factor

        reasons.append(f"chain_gen={transfer_generation_count}(conf={_chain_conf:.2f})")

    # Heavy cumulative HF loss from carrier chain → brillanz physically impossible
    if cumulative_hf_loss_db is not None and cumulative_hf_loss_db > 6.0:
        _hf_loss_factor = float(np.clip((cumulative_hf_loss_db - 6.0) / 18.0, 0.0, 1.0))
        weights["brillanz"] *= 1.0 - 0.25 * _hf_loss_factor
        weights["waerme"] *= 1.0 + 0.10 * _hf_loss_factor  # Focus on warmth instead
        if cumulative_hf_loss_db > 12.0:
            reasons.append(f"chain_hf_loss({cumulative_hf_loss_db:.1f}dB)")

    # ===================================================================
    # STAGE 3: Psychoacoustic per-song fine-tuning
    # Uses Aurik's psychoacoustic analysis modules (Zwicker roughness,
    # Aures sharpness, ISO 11172-3 masking, Bark-scale frequency balance)
    # to tune goal weights to the PERCEPTUAL reality of this specific song.
    # ===================================================================

    # --- Step 6j: Roughness (Zwicker) ---
    # High roughness = harsh amplitude modulation in critical bands.
    # The restoration must either fix it or protect against making it worse.
    if roughness is not None:
        if roughness > 0.4:
            # Harsh signal — artikulation and transparenz need strong protection
            weights["artikulation"] *= 1.15
            weights["transparenz"] *= 1.10
            # Natuerlichkeit is threatened by roughness
            weights["natuerlichkeit"] *= 1.10
            reasons.append(f"rough_high({roughness:.2f})")
        elif roughness < 0.1:
            # Very smooth — already natural, focus on other qualities
            weights["groove"] *= 1.05
            weights["emotionalitaet"] *= 1.05

    # --- Step 6k: Sharpness (Aures) ---
    # Perceptual HF emphasis — more reliable than spectral tilt because
    # it uses cochlea-weighted frequency binning, not linear Hz.
    if sharpness is not None:
        if sharpness > 0.6:
            # Already perceptually bright — protect brillanz (it's real)
            weights["brillanz"] *= 1.15
            # But high sharpness can indicate sibilance → artikulation at risk
            weights["artikulation"] *= 1.05
            reasons.append(f"sharp_high({sharpness:.2f})")
        elif sharpness < 0.2:
            # Perceptually dull — brillanz is genuinely absent, reduce expectations
            weights["brillanz"] *= _brillanz_factor if effective_bandwidth_hz else 0.85
            weights["waerme"] *= 1.08
            reasons.append(f"sharp_low({sharpness:.2f})")

    # --- Step 6l: Spectral Flatness ---
    # High flatness = noise-like spectrum (white/pink noise character).
    # Low flatness = tonal/harmonic (musical content).
    if spectral_flatness is not None:
        if spectral_flatness > 0.5:
            # Noise-dominated — transparenz is the #1 priority
            weights["transparenz"] *= 1.20
            weights["separation_fidelity"] *= 1.10
            # Brillanz unreliable in noise (HF energy ≠ musical brightness)
            weights["brillanz"] *= 0.90
            reasons.append(f"flat_noisy({spectral_flatness:.2f})")
        elif spectral_flatness < 0.1:
            # Highly tonal — tonal_center and timbre are defining
            weights["tonal_center"] *= 1.10
            weights["timbre_authentizitaet"] *= 1.08
            reasons.append(f"flat_tonal({spectral_flatness:.2f})")

    # --- Step 6m: Tonality ---
    # Strong tonal content → pitch/timbre goals critical.
    # Weak tonality → percussive/groove-driven → groove/dynamics more important.
    if tonality is not None:
        if tonality > 0.6:
            weights["tonal_center"] *= 1.12
            weights["timbre_authentizitaet"] *= 1.10
            weights["waerme"] *= 1.05
            reasons.append(f"tonal_strong({tonality:.2f})")
        elif tonality < 0.2:
            # Percussive/noisy → groove and dynamics define the experience
            weights["groove"] *= 1.12
            weights["micro_dynamics"] *= 1.10
            weights["bass_kraft"] *= 1.08
            reasons.append(f"tonal_weak({tonality:.2f})")

    # --- Step 6n: Frequency Balance (psychoacoustic) ---
    # Bass/mid/treble/air energy ratios from Bark-band analysis.
    # Maps directly to the spectral goal priorities.
    if frequency_balance and isinstance(frequency_balance, dict):
        _fb_bass = frequency_balance.get("bass", 0.25)
        _fb_mid = frequency_balance.get("mid", 0.25)
        _fb_treble = frequency_balance.get("treble", 0.25)
        _fb_air = frequency_balance.get("air", 0.25)

        # Bass-heavy signal → bass_kraft and waerme are defining
        if _fb_bass > 0.40:
            weights["bass_kraft"] *= 1.15
            weights["waerme"] *= 1.08
            reasons.append(f"fb_bass_heavy({_fb_bass:.2f})")
        elif _fb_bass < 0.10:
            weights["bass_kraft"] *= 0.75 + 0.25 * (_bw_ratio if effective_bandwidth_hz else 0.5)
            reasons.append(f"fb_bass_thin({_fb_bass:.2f})")

        # Treble+Air-heavy → brillanz genuinely important
        if (_fb_treble + _fb_air) > 0.40:
            weights["brillanz"] *= 1.12
            reasons.append(f"fb_bright({_fb_treble + _fb_air:.2f})")
        elif (_fb_treble + _fb_air) < 0.10:
            weights["brillanz"] *= _brillanz_factor if effective_bandwidth_hz else 0.85
            reasons.append(f"fb_dark({_fb_treble + _fb_air:.2f})")

        # Mid-dominated → articulation and transparency define intelligibility
        if _fb_mid > 0.50:
            weights["artikulation"] *= 1.08
            weights["transparenz"] *= 1.05

    # --- Step 6o: Masked Components Ratio ---
    # High masking = many spectral components are inaudible.
    # Spatial and separation goals are unrealistic if most is masked.
    # SANITY GUARD: Values ≥ 0.95 or ≤ 0.01 indicate measurement artifact
    # (e.g. spreading-function bug in MaskingAnalyzer).  Ignore unreliable values.
    if masked_components_ratio is not None and 0.01 < masked_components_ratio < 0.95:
        if masked_components_ratio > 0.5:
            # Half the spectrum is below masking threshold — limit expectations
            weights["spatial_depth"] *= 0.70 + 0.30 * (_bw_ratio if effective_bandwidth_hz else 0.5)
            weights["separation_fidelity"] *= 0.90
            # Transparenz becomes more critical (what's audible must be clear)
            weights["transparenz"] *= 1.10
            reasons.append(f"masked_high({masked_components_ratio:.2f})")
        elif masked_components_ratio < 0.1:
            # Almost everything is audible → full fidelity achievable
            weights["spatial_depth"] *= 1.08
            weights["separation_fidelity"] *= 1.05
            reasons.append(f"masked_low({masked_components_ratio:.2f})")

    # --- Step 6p: Perceptual Centroid (Bark) ---
    # Low centroid = warm/dark character; high centroid = bright/airy.
    # Typical music: 5-10 Bark.  < 4 = very dark, > 12 = very bright.
    if perceptual_centroid_bark is not None:
        if perceptual_centroid_bark < 4.0:
            # Dark character — waerme defines the experience
            weights["waerme"] *= 1.12
            weights["brillanz"] *= 0.90
            reasons.append(f"centroid_dark({perceptual_centroid_bark:.1f}Bark)")
        elif perceptual_centroid_bark > 12.0:
            # Bright character — brillanz defines the experience
            weights["brillanz"] *= 1.12
            weights["waerme"] *= 0.92
            reasons.append(f"centroid_bright({perceptual_centroid_bark:.1f}Bark)")

    # ===================================================================
    # STAGE 4: Vocal/Harmonic/Transient preservation features
    # These directly protect musical content from being damaged by
    # defect-removal phases.  A high HNR means the harmonics are CLEAN
    # and must be preserved; high transient density means percussive
    # attacks must survive denoising.
    # ===================================================================

    # --- Step 6q: Harmonic-to-Noise Ratio (HNR) ---
    # Pitch-period autocorrelation HNR for full music mixes.
    # Full-mix range: ~-10 to +20 dB (NOT isolated-vocal 15-35 dB).
    # CD pop mix: 3-10 dB, degraded vinyl: -5 to 3 dB.
    # Thresholds calibrated for multi-frame pitch-lag AC on full mixes.
    if harmonic_to_noise_ratio_db is not None:
        if harmonic_to_noise_ratio_db > 5.0:
            # Harmonics clearly above noise floor in full mix
            _hnr_f = min((harmonic_to_noise_ratio_db - 5.0) / 15.0, 1.0)
            weights["timbre_authentizitaet"] *= 1.0 + 0.12 * _hnr_f
            weights["tonal_center"] *= 1.0 + 0.10 * _hnr_f
            weights["artikulation"] *= 1.0 + 0.08 * _hnr_f
            weights["natuerlichkeit"] *= 1.0 + 0.05 * _hnr_f
            reasons.append(f"hnr_clean({harmonic_to_noise_ratio_db:.0f}dB)")
        elif harmonic_to_noise_ratio_db < 0.0:
            # Harmonics buried in noise — denoise urgent, transparenz critical
            _hnr_f = min(-harmonic_to_noise_ratio_db / 10.0, 1.0)
            weights["transparenz"] *= 1.0 + 0.12 * _hnr_f
            weights["timbre_authentizitaet"] *= 1.0 - 0.05 * _hnr_f
            reasons.append(f"hnr_noisy({harmonic_to_noise_ratio_db:.0f}dB)")

    # --- Step 6r: Harmonic Coherence ---
    # Autocorrelation-based pitch stability [0, 1].
    # High = stable tonal content (vocals, sustained instruments).
    # Low = inharmonic/noisy/percussive.
    if harmonic_coherence is not None:
        if harmonic_coherence > 0.7:
            # Strong pitch coherence — tonal goals define the experience
            weights["tonal_center"] *= 1.10
            weights["timbre_authentizitaet"] *= 1.08
            weights["emotionalitaet"] *= 1.05
            reasons.append(f"hcoh_strong({harmonic_coherence:.2f})")
        elif harmonic_coherence < 0.3:
            # Weak coherence — percussive or noise-dominated
            weights["groove"] *= 1.08
            weights["micro_dynamics"] *= 1.05
            reasons.append(f"hcoh_weak({harmonic_coherence:.2f})")

    # --- Step 6s: Crest Factor ---
    # 99.9th-percentile peak-to-RMS ratio in dB.
    # Full-mix 99.9pctl range: ~6-16 dB (lower than raw-max by ~2-3 dB).
    # Pop/Schlager: 9-13 dB, Classical: 14-18 dB, Brick-wall: 4-7 dB.
    if crest_factor_db is not None:
        if crest_factor_db > 12.0:
            # Highly dynamic material — protect peaks and micro-dynamics
            _crest_f = min((crest_factor_db - 12.0) / 6.0, 1.0)
            weights["micro_dynamics"] *= 1.0 + 0.15 * _crest_f
            weights["groove"] *= 1.0 + 0.08 * _crest_f
            weights["natuerlichkeit"] *= 1.0 + 0.05 * _crest_f
            reasons.append(f"crest_dynamic({crest_factor_db:.1f}dB)")
        elif crest_factor_db < 7.0:
            # Already compressed — micro_dynamics goals less achievable
            _crest_f = min((7.0 - crest_factor_db) / 4.0, 1.0)
            weights["micro_dynamics"] *= 1.0 - 0.10 * _crest_f
            weights["waerme"] *= 1.0 + 0.05 * _crest_f
            weights["transparenz"] *= 1.0 + 0.05 * _crest_f
            reasons.append(f"crest_compressed({crest_factor_db:.1f}dB)")

    # --- Step 6t: Transient Density ---
    # STFT spectral-flux onsets per second (50ms min-gap, adaptive threshold).
    # Pop/Schlager: 4-12/s, Classical: 1-4/s, Percussion-heavy: 10-18/s.
    # Noisy material inflates density by ~2-3/s due to spectral noise flux.
    if transient_density is not None:
        if transient_density > 8.0:
            # Percussive/rhythmic content — groove and artikulation critical
            _td_f = min((transient_density - 8.0) / 10.0, 1.0)
            weights["groove"] *= 1.0 + 0.12 * _td_f
            weights["artikulation"] *= 1.0 + 0.08 * _td_f
            weights["micro_dynamics"] *= 1.0 + 0.05 * _td_f
            reasons.append(f"transient_dense({transient_density:.1f}/s)")
        elif transient_density < 2.0:
            # Sustained content — timbre and warmth define the texture
            _td_f = min((2.0 - transient_density) / 2.0, 1.0)
            weights["timbre_authentizitaet"] *= 1.0 + 0.08 * _td_f
            weights["waerme"] *= 1.0 + 0.05 * _td_f
            weights["emotionalitaet"] *= 1.0 + 0.05 * _td_f
            reasons.append(f"transient_sustain({transient_density:.1f}/s)")

    # ===================================================================
    # STAGE 5: Cross-feature interactions
    # Individual features have synergistic effects that pure multiplication
    # cannot capture.  E.g. rough + noisy signal needs a MUCH stronger
    # transparenz boost than either condition alone — the perceptual damage
    # is super-additive.  These interactions represent the crucial difference
    # between heuristic stacking and musically aware importance estimation.
    # ===================================================================

    # --- Interaction 5a: Roughness × Low SNR → transparenz emergency ---
    # Rough + noisy = intelligibility crisis (vocal consonants buried)
    if roughness is not None and snr_db is not None:
        if roughness > 0.3 and snr_db < 20.0:
            _interact_strength = min(roughness, 0.8) * min((20.0 - snr_db) / 20.0, 1.0)
            weights["transparenz"] *= 1.0 + 0.15 * _interact_strength
            weights["artikulation"] *= 1.0 + 0.10 * _interact_strength
            if _interact_strength > 0.3:
                reasons.append(f"interact_rough×noisy({_interact_strength:.2f})")

    # --- Interaction 5b: High HNR × Vocal → vocal preservation critical ---
    # Clean harmonics + vocal = harmonics must survive denoising intact.
    # Threshold 5 dB = full-mix pitch-period HNR (not isolated-vocal 15+).
    # Siedenburg & McAdams (2017) J New Music Res 46:149: timbral distinction is
    # reliable even at moderate HNR in complex musical sounds.
    # McDermott (2009) + Bregman (1990): clean vocal harmonics increase the
    # auditory stream segregation demand — voice/accompaniment separation becomes
    # the primary perceptual task, so separation_fidelity is super-additively
    # elevated when both HNR and vocal content are high.
    if harmonic_to_noise_ratio_db is not None and vocal_detected:
        if harmonic_to_noise_ratio_db > 5.0:
            _vocal_hnr_f = min((harmonic_to_noise_ratio_db - 5.0) / 15.0, 1.0)
            _vocal_hnr_f *= float(np.clip(vocal_confidence, 0.3, 1.0))
            weights["timbre_authentizitaet"] *= 1.0 + 0.10 * _vocal_hnr_f  # Siedenburg & McAdams (2017)
            weights["natuerlichkeit"] *= 1.0 + 0.08 * _vocal_hnr_f
            weights["separation_fidelity"] *= 1.0 + 0.08 * _vocal_hnr_f  # McDermott (2009), Bregman (1990)
            if _vocal_hnr_f > 0.3:
                reasons.append(f"interact_vocal×hnr({_vocal_hnr_f:.2f})")

    # --- Interaction 5c: Low bandwidth × Dark centroid → brillanz impossible ---
    # When both bandwidth AND centroid are low, brillanz is physically absent
    if effective_bandwidth_hz is not None and perceptual_centroid_bark is not None:
        if effective_bandwidth_hz < 10000.0 and perceptual_centroid_bark < 5.0:
            _dark_f = (1.0 - effective_bandwidth_hz / 10000.0) * (1.0 - perceptual_centroid_bark / 5.0)
            weights["brillanz"] *= 1.0 - 0.20 * _dark_f
            weights["waerme"] *= 1.0 + 0.08 * _dark_f
            if _dark_f > 0.3:
                reasons.append(f"interact_bw×dark({_dark_f:.2f})")

    # --- Interaction 5d: High coherence × Tonal content → tonal_center dominates ---
    # Strong pitch + strong tonality = tonal center is THE defining feature
    if harmonic_coherence is not None and tonality is not None:
        if harmonic_coherence > 0.5 and tonality > 0.4:
            _tonal_f = min(harmonic_coherence, 0.9) * min(tonality, 0.9)
            weights["tonal_center"] *= 1.0 + 0.12 * _tonal_f
            if _tonal_f > 0.3:
                reasons.append(f"interact_coh×tonal({_tonal_f:.2f})")

    # --- Interaction 5e: Dynamic × Transient → groove/micro_dynamics synergy ---
    # High crest (99.9pctl) + high transient density = percussive AND dynamic.
    # Thresholds: crest > 10 dB (pctl-adjusted), density > 5/s (STFT onsets).
    if crest_factor_db is not None and transient_density is not None:
        if crest_factor_db > 10.0 and transient_density > 5.0:
            _perc_dyn = min((crest_factor_db - 10.0) / 8.0, 1.0) * min(transient_density / 12.0, 1.0)
            weights["groove"] *= 1.0 + 0.10 * _perc_dyn
            weights["micro_dynamics"] *= 1.0 + 0.10 * _perc_dyn
            if _perc_dyn > 0.2:
                reasons.append(f"interact_dyn×trans({_perc_dyn:.2f})")

    # --- Interaction 5f: Multi-generation carrier × Low SNR → core preservation ---
    # Deep chain + noise = original severely degraded; preserve what remains
    if transfer_generation_count is not None and snr_db is not None:
        if transfer_generation_count >= 3 and snr_db < 20.0:
            _deg_f = min((transfer_generation_count - 2) / 3.0, 1.0) * min((20.0 - snr_db) / 20.0, 1.0)
            weights["natuerlichkeit"] *= 1.0 + 0.10 * _deg_f
            weights["authentizitaet"] *= 1.0 + 0.10 * _deg_f
            if _deg_f > 0.2:
                reasons.append(f"interact_chain×noisy({_deg_f:.2f})")

    # --- Step 6u: NaN/Inf guard before soft-capping ---
    # Multiplicative compounding across 4 stages can produce NaN/Inf if any
    # input feature (roughness, HNR, coherence, carrier-chain conf) is NaN/Inf.
    # NaN weights make PMGG's weighted_reg = raw_reg * NaN → NaN → all retries
    # pass silently (NaN > threshold == False), effectively disabling PMGG.
    for goal in ALL_GOAL_NAMES:
        if not math.isfinite(weights[goal]):
            weights[goal] = 1.0  # reset to neutral; partial result > no result

    # --- Step 7: Diminishing returns for multiplicative stacking ---
    # With 4 stages (genre/era/material, audio, psychoacoustic, vocal/transient)
    # multiplicative compounding can push weights to extreme values.
    # Use rational compression: excess → excess/(1 + k·excess).
    # k=3.0 gives asymptote at 1.5 + 0.33 = 1.83 for boosted goals
    # and 0.5 - 0.33 = 0.17 for suppressed goals.
    # This preserves relative ranking while preventing extreme values
    # that would block restorative phases in PMGG/CIG.
    _SOFT_CAP_HIGH = 1.5
    _SOFT_CAP_LOW = 0.5
    _COMPRESSION = 3.0
    for goal in ALL_GOAL_NAMES:
        w = weights[goal]
        if w > _SOFT_CAP_HIGH:
            _excess = w - _SOFT_CAP_HIGH
            weights[goal] = _SOFT_CAP_HIGH + _excess / (1.0 + _COMPRESSION * _excess)
        elif w < _SOFT_CAP_LOW:
            _deficit = _SOFT_CAP_LOW - w
            weights[goal] = _SOFT_CAP_LOW - _deficit / (1.0 + _COMPRESSION * _deficit)

    # --- Step 7b: §C10 Active Listener Calibration blend ---
    # Blend persisted Bayesian EMA nudges at 15 % weight (advisory, non-overriding).
    try:
        _nudges = get_feedback_store().get_nudges()
        _blend = SongGoalFeedbackStore.FEEDBACK_BLEND_WEIGHT
        for goal in ALL_GOAL_NAMES:
            if goal in _nudges:
                _nudge = float(_nudges[goal])
                if abs(_nudge - 1.0) > 0.01:  # Only apply meaningful nudges
                    weights[goal] = weights[goal] * (1.0 - _blend) + weights[goal] * _nudge * _blend
    except Exception as _c10_exc:
        logger.debug("§C10 Listener Kalibrierung blend uebersprungen: %s", _c10_exc)

    # --- Step 8: Enforce hard bounds ---
    for goal in ALL_GOAL_NAMES:
        if goal in _P1P2_GOALS:
            weights[goal] = float(np.clip(weights[goal], _P1P2_WEIGHT_FLOOR, _WEIGHT_MAX))
        else:
            weights[goal] = float(np.clip(weights[goal], _WEIGHT_MIN, _WEIGHT_MAX))

    reason_str = " + ".join(reasons) if reasons else "neutral"

    importance = SongGoalImportance(
        weights=weights,
        reason=reason_str,
        genre_profile=_genre_key,
        era_profile=_era,
        material_profile=_mat,
        vocal_detected=vocal_detected,
    )

    logger.info(
        "§2.56 SongGoalImportance: %s — top3: %s",
        reason_str,
        ", ".join(f"{g}={weights[g]:.2f}" for g in sorted(weights, key=lambda k: weights[k], reverse=True)[:3]),
    )

    return importance


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------
_instance: SongGoalImportance | None = None
_lock = threading.Lock()


def get_default_importance() -> SongGoalImportance:
    """Gibt a neutral (uniform) importance — for use when no song context is available zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SongGoalImportance()
    return _instance
