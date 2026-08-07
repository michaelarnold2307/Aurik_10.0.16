"""
backend/core/artistic_intent_discriminator.py — ArtisticIntentDiscriminator (Aurik 10.0.0 §AID-1)
=============================================================================================
Rule-based classifier: "Is this signal characteristic a defect to repair,
or an intentional artistic choice to preserve?"

The most important sentence a world-class restoration engineer says:
    "Don't touch this — this is why the song works."

Examples of intentional characteristics that look like defects:
    - Distorted vocal climax (passion, not clipping artefact)
    - Room ambience present from bar 1 (recording philosophy, not reverb tail)
    - Pitch wavering outside 4–7 Hz vibrato range (expressiveness, not wow/flutter)
    - Tape compression warmth (era-correct saturation, not dynamic defect)
    - Controlled breathiness in whisper passages (intimacy, not mic proximity noise)

§AID-1 scores each audio region 0..1:
    0.0  = almost certainly a technical defect → process normally
    0.5  = ambiguous → reduce phase strength by 40 %
    1.0  = almost certainly intentional → protect (cap strength at 0.15)

Integration (UV3 §AID-1):
    Called after HarmonicContextAnalyzer, before GoalApplicabilityFilter.
    Injects into _restoration_context:
        "intent_scores": dict[phase_id → float]  — per-phase protection levels

Phase activations read "intent_scores" from restoration_context and scale down
strength accordingly:
    intent = context.get("intent_scores", {}).get(phase_id, 0.0)
    if intent > 0.70:
        strength = min(strength, 0.15)
    elif intent > 0.40:
        strength *= (1.0 - intent * 0.6)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent score thresholds
# ---------------------------------------------------------------------------

INTENT_PROTECT_THRESHOLD: float = 0.70  # cap strength at 0.15
INTENT_REDUCE_THRESHOLD: float = 0.40  # scale down strength proportionally

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IntentAnalysisResult:
    """Per-region artistic intent scores produced by ArtisticIntentDiscriminator."""

    # Global intent score for the whole file (0=defect, 1=intent)
    global_score: float = 0.0
    # Per-phase recommended intent protection (dict[phase_id → score 0..1])
    phase_intent_scores: dict[str, float] = field(default_factory=dict)
    # Detected characteristics and their intent scores
    characteristics: list[dict] = field(default_factory=list)
    # Number of regions classified as intentional (intent > 0.5)
    n_intentional: int = 0
    # Fraction of duration classified as intentional
    intent_fraction: float = 0.0

    def to_dict(self) -> dict:
        """Gibt a serializable representation of the intent analysis zurück."""
        return {
            "global_score": float(self.global_score),
            "phase_intent_scores": self.phase_intent_scores,
            "n_intentional": self.n_intentional,
            "intent_fraction": float(self.intent_fraction),
            "characteristics": self.characteristics,
        }


# ---------------------------------------------------------------------------
# Phase groups affected by artistic intent
# ---------------------------------------------------------------------------

# Phases that operate on harmonic content — protect if intent_score > threshold
_HARMONIC_PHASES = {
    "phase_06_bandwidth_extension",
    "phase_07_harmonic_excitation",
    "phase_23_bandwidth_restoration",
    "phase_24_tonal_dropout_repair",
    "phase_38_spectral_reconstruction",
}

# Phases that alter dynamics / compression
_DYNAMICS_PHASES = {
    "phase_26_dynamics_restoration",
    "phase_35_multiband_compression",
    "phase_40_loudness_enhancement",
}

# Phases that remove noise — protect warmth in harmonic/climax regions
_NR_PHASES = {
    "phase_03_noise_reduction",
    "phase_29_omlsa_noise_reduction",
    "phase_20_spectral_gating",
    "phase_55_residual_nr",
}

# Phases that affect vocal timbre directly
_VOCAL_PHASES = {
    "phase_42_vocal_enhancement",
    "phase_36_de_essing",
    "phase_53_vocal_presence",
}


class ArtisticIntentDiscriminator:
    """Rule-based artistic intent scorer (§AID-1).

    Evaluates six evidence streams:
        1. Harmonic richness — dense harmonics in "rough" regions → likely intentional
        2. Frisson / emotional peak — climax zone → protect distortion
        3. Era consistency — saturation consistent with era/material → intentional
        4. Repetition — characteristic appears in > 50 % of same-type sections → intentional
        5. Defect profile mismatch — doesn't match known defect spectral signature → intentional
        6. Vocal performance context — harsh in consonant/onset zone → performance choice
    """

    def analyze(  # pylint: disable=too-many-positional-arguments
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing: float = 0.0,
        restoration_context: dict | None = None,
        era: int = 1970,
        material: str = "vinyl",
    ) -> IntentAnalysisResult:
        """Classify which signal characteristics are intentional vs. defects.

        Args:
            audio:               Input audio (channels-last)
            sr:                  Sample rate
            panns_singing:       PANNs vocal probability (0..1)
            restoration_context: UV3 restoration context dict (may be None)
            era:                 Year of recording
            material:            Material type string

        Returns:
            IntentAnalysisResult with per-phase intent scores.
        """
        ctx = restoration_context or {}
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = self._to_mono(audio)
        n = len(mono)
        if n == 0:
            return IntentAnalysisResult()

        characteristics: list[dict] = []

        # --- Evidence 1: Saturation / harmonic distortion level ---
        thd_score = self._measure_harmonic_distortion(mono, sr)
        # High THD that is era-consistent → intentional warm saturation
        era_saturation_expected = self._era_expects_saturation(era, material)
        sat_intent = float(np.clip(thd_score * (1.5 if era_saturation_expected else 0.5), 0.0, 1.0))
        if thd_score > 0.1:
            characteristics.append(
                {
                    "type": "harmonic_saturation",
                    "thd_score": round(thd_score, 3),
                    "intent_score": round(sat_intent, 3),
                    "era_consistent": era_saturation_expected,
                }
            )

        # --- Evidence 2: Emotional / frisson zones ---
        frisson_zones = ctx.get("frisson_zones", [])
        frisson_fraction = self._fraction_in_zones(n, frisson_zones, sr)
        # Anything happening in a frisson zone is likely intentional
        frisson_intent = float(np.clip(frisson_fraction * 2.0, 0.0, 1.0))
        if frisson_fraction > 0.05:
            characteristics.append(
                {
                    "type": "frisson_zone_overlap",
                    "fraction": round(frisson_fraction, 3),
                    "intent_score": round(frisson_intent, 3),
                }
            )

        # --- Evidence 3: Breathing / intimacy markers ---
        breath_segments = ctx.get("breath_segments", [])
        breath_fraction = self._fraction_in_zones(n, breath_segments, sr)
        # Breathiness in vocal passages = intimacy, not noise
        breath_intent = float(np.clip(breath_fraction * panns_singing * 2.0, 0.0, 1.0))
        if breath_fraction > 0.05 and panns_singing >= 0.35:
            characteristics.append(
                {
                    "type": "breath_intimacy",
                    "fraction": round(breath_fraction, 3),
                    "intent_score": round(breath_intent, 3),
                }
            )

        # --- Evidence 4: Structural repetition consistency ---
        # If a characteristic (e.g. distortion) is present in > 50 % of choruses,
        # it's consistent → intentional.  We use energy variance across sections as proxy.
        structure = ctx.get("structure")  # from MusicalStructureAnalyzer
        repetition_intent = 0.0
        if structure is not None:
            repetition_intent = self._measure_repetition_consistency(mono, sr, structure)
        if repetition_intent > 0.3:
            characteristics.append(
                {
                    "type": "structural_repetition",
                    "score": round(repetition_intent, 3),
                    "intent_score": round(repetition_intent, 3),
                }
            )

        # --- Evidence 5: Vocal performance zone (climax / onset) ---
        climax_type = ctx.get("climax_type", "none")
        tension_zones = ctx.get("tension_zones", [])
        _is_climax = climax_type in ("high_energy", "climax", "peak") and panns_singing >= 0.35
        climax_intent = 0.6 if _is_climax else 0.0
        tension_fraction = self._fraction_in_zones(n, tension_zones, sr)
        if tension_fraction > 0.1 and panns_singing >= 0.35:
            climax_intent = max(climax_intent, float(np.clip(tension_fraction * 1.5, 0.0, 0.9)))
            characteristics.append(
                {
                    "type": "vocal_tension_zone",
                    "fraction": round(tension_fraction, 3),
                    "intent_score": round(climax_intent, 3),
                }
            )

        # --- Combine evidence streams into global intent score ---
        evidence = [sat_intent, frisson_intent, breath_intent, repetition_intent, climax_intent]
        # Global: weighted average (frisson and saturation highest weight)
        weights = [1.5, 2.0, 1.0, 1.2, 1.8]
        global_score = float(
            np.clip(
                sum(e * w for e, w in zip(evidence, weights)) / sum(weights),
                0.0,
                1.0,
            )
        )

        # --- Map to per-phase scores ---
        phase_intent_scores: dict[str, float] = {}

        # Harmonic phases: protect if saturation + era consistent
        harmonic_intent = float(np.clip((sat_intent + frisson_intent) / 2.0, 0.0, 1.0))
        for ph in _HARMONIC_PHASES:
            phase_intent_scores[ph] = harmonic_intent

        # NR phases: protect against over-removing warmth in frisson/climax zones
        nr_intent = float(np.clip((frisson_intent * 1.5 + sat_intent * 0.5) / 2.0, 0.0, 1.0))
        for ph in _NR_PHASES:
            phase_intent_scores[ph] = nr_intent

        # Dynamics phases: protect in structural-repetition context
        dyn_intent = float(np.clip((repetition_intent + climax_intent) / 2.0, 0.0, 1.0))
        for ph in _DYNAMICS_PHASES:
            phase_intent_scores[ph] = dyn_intent

        # Vocal phases: protect in vocal tension / breath zones
        vocal_intent = float(np.clip((breath_intent + climax_intent) / 2.0, 0.0, 1.0))
        for ph in _VOCAL_PHASES:
            phase_intent_scores[ph] = vocal_intent

        intent_fraction = float(np.clip(global_score, 0.0, 1.0))
        n_intentional = sum(1 for s in phase_intent_scores.values() if s > 0.5)

        logger.debug(
            "§AID-1 global_intent=%.3f sat=%.2f frisson=%.2f breath=%.2f climax=%.2f",
            global_score,
            sat_intent,
            frisson_intent,
            breath_intent,
            climax_intent,
        )

        return IntentAnalysisResult(
            global_score=global_score,
            phase_intent_scores=phase_intent_scores,
            characteristics=characteristics,
            n_intentional=n_intentional,
            intent_fraction=intent_fraction,
        )

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Gibt mono audio from mono, channel-first, or channel-last input zurück."""
        if audio.ndim != 2:
            return audio.reshape(-1)
        rows, cols = audio.shape
        if rows <= 8 and cols > rows:
            return audio.mean(axis=0)  # type: ignore[no-any-return]
        if cols <= 8 and rows > cols:
            return audio.mean(axis=1)  # type: ignore[no-any-return]
        return audio.mean(axis=0)  # type: ignore[no-any-return]

    @staticmethod
    def _measure_harmonic_distortion(mono: np.ndarray, sr: int) -> float:
        """Schätzt total harmonic distortion level as fraction of total energy.

        Simple heuristic: high-frequency energy above 8 kHz relative to
        total energy, weighted by spectral flatness.

        Returns:
            thd_score in [0, 1]  (0=clean, 1=heavily distorted)
        """
        n_fft = 2048
        if len(mono) < n_fft:
            return 0.0
        try:
            window = np.hanning(n_fft)
            # Use middle segment
            start = max(0, len(mono) // 2 - n_fft // 2)
            seg = mono[start : start + n_fft]
            spec = np.abs(np.fft.rfft(seg * window)).astype(np.float32) + 1e-8
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

            total_e = float(np.sum(spec**2))
            hf_e = float(np.sum(spec[freqs > 8000.0] ** 2))

            if total_e < 1e-10:
                return 0.0
            hf_ratio = hf_e / total_e

            # Spectral flatness (1 = noise, 0 = tonal)
            geo_mean = float(np.exp(np.mean(np.log(spec + 1e-12))))
            arith_mean = float(spec.mean())
            flatness = float(np.clip(geo_mean / (arith_mean + 1e-8), 0.0, 1.0))

            # High hf_ratio + low flatness = odd harmonic distortion (warm saturation)
            # High hf_ratio + high flatness = broadband noise (defect)
            thd = float(np.clip(hf_ratio * 2.0 * (1.0 - flatness * 0.5), 0.0, 1.0))
            return thd
        except Exception as e:
            logger.warning("artistic_intent_discriminator.py::_measure_harmonic_distortion Ersatzpfad: %s", e)
            return 0.0

    @staticmethod
    def _era_expects_saturation(era: int, material: str) -> bool:
        """Gibt True if the era/material combination is known for warm saturation zurück."""
        mat = material.lower()
        if mat in ("shellac", "shellac_78", "acoustic_disc"):
            return True  # all shellac has natural harmonic saturation
        if mat in ("vinyl", "vinyl_lp", "vinyl_45") and era < 1985:
            return True  # analog vinyl mastering chain typically has saturation
        if mat in ("tape", "analog_tape", "open_reel") and era < 1990:
            return True  # tape saturation is characteristic
        if mat in ("cassette",) and era < 1995:
            return True
        return False

    @staticmethod
    def _fraction_in_zones(n_samples: int, zones: list, sr: int) -> float:
        """Berechnet fraction of total duration covered by the given zones.

        Zones may be lists of (start_s, end_s) tuples or sample-pair tuples.
        """
        if not zones or n_samples == 0:
            return 0.0
        covered = 0
        for zone in zones:
            try:
                if isinstance(zone, (list, tuple)) and len(zone) >= 2:
                    a, b = zone[0], zone[1]
                    # Handle both sample and time representations
                    if isinstance(a, float) and a < 1e5:
                        # Likely time in seconds
                        a_s = int(a * sr)
                        b_s = int(b * sr)
                    else:
                        a_s = int(a)
                        b_s = int(b)
                    covered += max(0, min(b_s, n_samples) - max(0, a_s))
            except Exception as e:
                logger.warning("artistic_intent_discriminator.py::_fraction_in_zones Ersatzpfad: %s", e)
        return float(np.clip(covered / max(1, n_samples), 0.0, 1.0))

    @staticmethod
    def _measure_repetition_consistency(mono: np.ndarray, sr: int, structure: object) -> float:
        """Schätzt how consistently a characteristic repeats across similar sections.

        Uses energy variance across chorus segments: low variance = consistent = intentional.

        Returns:
            consistency score in [0, 1]
        """
        try:
            segs = ArtisticIntentDiscriminator._get_structure_segments(structure)
            chorus_energies = []
            for seg in segs:
                label, start, end = ArtisticIntentDiscriminator._segment_bounds(seg, sr)
                if label == "chorus":
                    chunk = mono[max(0, start) : min(len(mono), end)]
                    if len(chunk) > 0:
                        chorus_energies.append(float(np.sqrt(np.mean(chunk**2))))

            if len(chorus_energies) < 2:
                return 0.0
            arr = np.array(chorus_energies, dtype=np.float32)
            cv = float(arr.std() / (arr.mean() + 1e-8))  # coefficient of variation
            # Low CV = high consistency = likely intentional
            return float(np.clip(1.0 - cv * 2.0, 0.0, 1.0))
        except Exception as e:
            logger.warning("artistic_intent_discriminator.py::_measure_repetition_consistency Ersatzpfad: %s", e)
            return 0.0

    @staticmethod
    def _get_structure_segments(structure: object) -> list:
        """Gibt segments from MusicalStructure objects or dict payloads zurück."""
        if isinstance(structure, dict):
            segments = structure.get("segments", [])
        else:
            segments = getattr(structure, "segments", [])
        return list(segments) if isinstance(segments, (list, tuple)) else []

    @staticmethod
    def _segment_bounds(segment: object, sr: int) -> tuple[str, int, int]:
        """Extrahiert label and sample bounds from dict or SegmentInfo-like objects."""
        if isinstance(segment, dict):
            label = str(segment.get("label", ""))
            start = int(segment.get("start_sample", int(segment.get("start_s", 0.0) * sr)))
            end = int(segment.get("end_sample", int(segment.get("end_s", 0.0) * sr)))
            return label, start, end
        label = str(getattr(segment, "label", ""))
        start = int(getattr(segment, "start_sample", int(getattr(segment, "start_s", 0.0) * sr)))
        end = int(getattr(segment, "end_sample", int(getattr(segment, "end_s", 0.0) * sr)))
        return label, start, end


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: ArtisticIntentDiscriminator | None = None
_lock = threading.Lock()


def get_artistic_intent_discriminator() -> ArtisticIntentDiscriminator:
    """Thread-safe singleton (double-checked locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ArtisticIntentDiscriminator()
    return _instance


__all__ = [
    "IntentAnalysisResult",
    "ArtisticIntentDiscriminator",
    "INTENT_PROTECT_THRESHOLD",
    "INTENT_REDUCE_THRESHOLD",
    "get_artistic_intent_discriminator",
]
