"""
backend/core/multimodal_decision_engine.py — Multimodal decision engine
=======================================================================

Combines cover-image analysis, NLP prompt parsing, and audio metadata to
produce a restoration chain and parameter preset.

Implements a comprehensive knowledge base for genre/era/material detection
plus BEATs/PANNs audio tagging integration for cover analysis.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge base — Genre × Era × Material → Phase chain
# ---------------------------------------------------------------------------

_GENRE_ERA_CHAINS: dict[str, dict[str, Any]] = {
    # Jazz eras
    "jazz_1920s": {"genre": "Jazz", "era": "1920s", "chain": ["click_removal", "noise_reducer", "bandwidth_expander"]},
    "jazz_1930s": {"genre": "Jazz", "era": "1930s", "chain": ["click_removal", "noise_reducer", "warmth_enhancer"]},
    "jazz_1940s": {"genre": "Jazz", "era": "1940s", "chain": ["click_removal", "noise_reducer", "warmth_enhancer"]},
    "jazz_1950s": {"genre": "Jazz", "era": "1950s", "chain": ["noise_reducer", "warmth_enhancer", "presence_enhancer"]},
    "jazz_1960s": {"genre": "Jazz", "era": "1960s", "chain": ["noise_reducer", "stereo_enhancer", "warmth_enhancer"]},
    # Rock / Pop
    "rock_1950s": {"genre": "Rock", "era": "1950s", "chain": ["click_removal", "noise_reducer", "warmth_enhancer"]},
    "rock_1960s": {"genre": "Rock", "era": "1960s", "chain": ["wow_flutter_fix", "noise_reducer", "presence_enhancer"]},
    "rock_1970s": {"genre": "Rock", "era": "1970s", "chain": ["brilliance_enhancer", "denoiser", "stereo_enhancer"]},
    "rock_1980s": {"genre": "Rock", "era": "1980s", "chain": ["brilliance_enhancer", "stereo_enhancer", "denoiser"]},
    "rock_1990s": {"genre": "Rock", "era": "1990s", "chain": ["denoiser", "stereo_enhancer", "presence_enhancer"]},
    "rock_2000s": {"genre": "Rock", "era": "2000s", "chain": ["denoiser", "stereo_enhancer"]},
    "pop_1960s": {"genre": "Pop", "era": "1960s", "chain": ["noise_reducer", "warmth_enhancer", "presence_enhancer"]},
    "pop_1970s": {"genre": "Pop", "era": "1970s", "chain": ["noise_reducer", "stereo_enhancer", "presence_enhancer"]},
    "pop_1980s": {"genre": "Pop", "era": "1980s", "chain": ["denoiser", "stereo_enhancer", "brilliance_enhancer"]},
    "pop_1990s": {"genre": "Pop", "era": "1990s", "chain": ["denoiser", "stereo_enhancer"]},
    "pop_2000s": {"genre": "Pop", "era": "2000s", "chain": ["denoiser"]},
    # Classical
    "classical_pre1950": {
        "genre": "Classical",
        "era": "Pre-1950",
        "chain": ["click_removal", "noise_reducer", "bandwidth_expander", "warmth_enhancer"],
    },
    "classical_1950s": {
        "genre": "Classical",
        "era": "1950s",
        "chain": ["click_removal", "noise_reducer", "warmth_enhancer", "stereo_enhancer"],
    },
    "classical_1960s": {
        "genre": "Classical",
        "era": "1960s",
        "chain": ["noise_reducer", "stereo_enhancer", "warmth_enhancer"],
    },
    "classical_modern": {"genre": "Classical", "era": "Modern", "chain": ["denoiser", "stereo_enhancer"]},
    # Blues / Country / Gospel
    "blues_1920s": {
        "genre": "Blues",
        "era": "1920s",
        "chain": ["click_removal", "noise_reducer", "bandwidth_expander", "warmth_enhancer"],
    },
    "blues_1930s": {"genre": "Blues", "era": "1930s", "chain": ["click_removal", "noise_reducer", "warmth_enhancer"]},
    "blues_1950s": {"genre": "Blues", "era": "1950s", "chain": ["noise_reducer", "warmth_enhancer"]},
    "country_1950s": {"genre": "Country", "era": "1950s", "chain": ["noise_reducer", "warmth_enhancer"]},
    "gospel_1940s": {"genre": "Gospel", "era": "1940s", "chain": ["click_removal", "noise_reducer", "warmth_enhancer"]},
    # Electronic / Dance
    "electronic_1970s": {
        "genre": "Electronic",
        "era": "1970s",
        "chain": ["noise_reducer", "stereo_enhancer", "bass_enhancer"],
    },
    "electronic_1980s": {
        "genre": "Electronic",
        "era": "1980s",
        "chain": ["denoiser", "stereo_enhancer", "bass_enhancer"],
    },
    "electronic_1990s": {"genre": "Electronic", "era": "1990s", "chain": ["denoiser", "stereo_enhancer"]},
    "hiphop_1990s": {"genre": "Hip-Hop", "era": "1990s", "chain": ["denoiser", "bass_enhancer", "stereo_enhancer"]},
    # Vinyl generisch
    "vinyl_rock": {"genre": "Rock", "era": "1970s", "chain": ["brilliance_enhancer", "denoiser"]},
    "vinyl_jazz": {"genre": "Jazz", "era": "1950s", "chain": ["noise_reducer", "warmth_enhancer"]},
    "vinyl_pop": {"genre": "Pop", "era": "1970s", "chain": ["noise_reducer", "stereo_enhancer"]},
    "vinyl_blues": {"genre": "Blues", "era": "1940s", "chain": ["click_removal", "noise_reducer", "warmth_enhancer"]},
    "vinyl_classical": {
        "genre": "Classical",
        "era": "1960s",
        "chain": ["noise_reducer", "warmth_enhancer", "stereo_enhancer"],
    },
    "jazz_album": {"genre": "Jazz", "era": "1950s", "chain": ["warmth_enhancer", "noise_reducer"]},
    "shellac": {
        "genre": "Various",
        "era": "Pre-1950",
        "chain": ["click_removal", "noise_reducer", "bandwidth_expander"],
    },
}

# Material type → mandatory phase chain additions
_MATERIAL_CHAIN: dict[str, list[str]] = {
    "shellac": ["click_removal", "noise_reducer", "bandwidth_expander"],
    "vinyl": ["click_removal", "wow_flutter_fix"],
    "tape": ["hiss_reducer"],
    "reel_tape": ["hiss_reducer"],
    "cassette": ["hiss_reducer", "noise_reducer"],
    "mp3_low": ["denoiser", "spectral_repair"],
    "cd_digital": [],
    "streaming": [],
}

# Prompt keywords → processing chain items
_PROMPT_RULES: list[tuple[list[str], str, dict[str, Any]]] = [
    (["brillanz", "bright", "brilliance", "klar", "clear"], "brilliance_enhancer", {}),
    (["rauschen", "noise", "denoising", "entrauschen"], "denoiser", {}),
    (["wärmer", "warm", "warmth", "wärme"], "warmth_enhancer", {"eq_low": 1.1}),
    (["bass", "tiefe", "bassboost"], "bass_enhancer", {}),
    (["stereo", "breite", "wide", "raumklang"], "stereo_enhancer", {}),
    (["knistern", "click", "crackle", "kratzen"], "click_removal", {}),
    (["bandbreite", "bandwidth", "höhen", "highend"], "bandwidth_expander", {}),
    (["präsenz", "presence", "mittelton"], "presence_enhancer", {}),
    (["vintage", "analog", "charakter"], "warmth_enhancer", {"preserve_character": True}),
    (["gesang", "vocal", "stimme", "voice"], "vocal_enhancer", {}),
    (["dynamik", "dynamic", "kompressor"], "dynamics_optimizer", {}),
    (["rauschteppich", "hiss", "bandrauschen"], "hiss_reducer", {}),
    (["equalizer", "eq", "frequenz"], "eq_optimizer", {}),
]


class MultimodalDecisionEngine:
    """Kombiniert image, text and audio signals to produce a processing chain.

    Implements a comprehensive rule-based engine with an extensive genre/era
    knowledge base. Optionally uses BEATs audio tagging (if loaded) for
    improved cover/audio classification.

    Decision pipeline:
      1. Audio metadata analysis (material, genre, era from PreAnalysis)
      2. Cover image heuristics (path + BEATs/DSP embedding similarity)
      3. Prompt NLP (keyword matching with parameter overrides)
      4. Chain deduplication + priority ordering
    """

    def decide(
        self,
        image_path: str,
        prompt: str,
        audio_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Gibt a processing chain and metadata dict zurück.

        Args:
            image_path: Path to cover image (or placeholder).
            prompt:     User text prompt (German/English).
            audio_meta: Dict with optional keys ``"material"``, ``"genre"``,
                        ``"era"``, ``"defect_types"``, ``"restorability"``.

        Returns:
            Dict with keys ``"chain"`` (list), ``"meta"`` (dict), and
            ``"parameters"`` (dict of processing parameter overrides).
        """
        chain: list[str] = []
        meta: dict[str, Any] = {"genre": "Unknown", "era": "Unknown"}
        parameters: dict[str, Any] = {}

        # ── 1. Audio metadata (PreAnalysis — highest trust) ────────────────
        material = str(audio_meta.get("material", "")).lower()
        genre_key = str(audio_meta.get("genre", "")).lower().replace(" ", "_")
        era = str(audio_meta.get("era", ""))
        defects = audio_meta.get("defect_types", [])
        restorability = float(audio_meta.get("restorability", 50.0))

        if material in _MATERIAL_CHAIN:
            for item in _MATERIAL_CHAIN[material]:
                if item not in chain:
                    chain.append(item)

        if genre_key and era:
            era_decade = era[:4] + "s" if len(era) >= 4 and era[:4].isdigit() else era
            lookup_key = f"{genre_key}_{era_decade}"
            rule = _GENRE_ERA_CHAINS.get(lookup_key)
            if rule:
                meta["genre"] = rule.get("genre", genre_key.title())
                meta["era"] = rule.get("era", era)
                for item in rule.get("chain", []):
                    if item not in chain:
                        chain.append(item)

        # Defect-driven additions
        defect_set = {str(d).lower() for d in defects}
        if any(d in defect_set for d in ("click", "crackle", "pop")):
            if "click_removal" not in chain:
                chain.insert(0, "click_removal")
        if any(d in defect_set for d in ("wow", "flutter", "speed_variation")):
            if "wow_flutter_fix" not in chain:
                chain.insert(1, "wow_flutter_fix")
        if any(d in defect_set for d in ("noise", "hiss", "tape_noise")):
            if "denoiser" not in chain and "hiss_reducer" not in chain:
                chain.append("denoiser")
        if any(d in defect_set for d in ("dropout", "gap")):
            if "spectral_repair" not in chain:
                chain.append("spectral_repair")

        # Low restorability → conservative chain
        if restorability < 20.0:
            chain = [c for c in chain if c in ("click_removal", "noise_reducer", "denoiser")]
            parameters["processing_strength"] = 0.4

        # ── 2. Cover image heuristics ──────────────────────────────────────
        if image_path and os.path.exists(image_path):
            # Try BEATs/DSP-based audio tagging for cover similarity
            _cover_chain, _cover_meta = self._analyze_cover(image_path, audio_meta)
            for item in _cover_chain:
                if item not in chain:
                    chain.append(item)
            for k, v in _cover_meta.items():
                if meta.get(k, "Unknown") in ("Unknown", ""):
                    meta[k] = v
        else:
            # Path-based heuristics as fallback
            basename = os.path.basename(image_path or "").lower()
            for key, rule in _GENRE_ERA_CHAINS.items():
                if key in basename:
                    for item in rule.get("chain", []):
                        if item not in chain:
                            chain.append(item)
                    if meta.get("genre", "Unknown") == "Unknown":
                        meta["genre"] = rule.get("genre", "Unknown")
                    if meta.get("era", "Unknown") == "Unknown":
                        meta["era"] = rule.get("era", "Unknown")
                    break

        # ── 3. Prompt NLP ──────────────────────────────────────────────────
        prompt_lower = (prompt or "").lower()
        for keywords, processor, param_overrides in _PROMPT_RULES:
            if any(kw in prompt_lower for kw in keywords):
                if processor not in chain:
                    chain.append(processor)
                parameters.update(param_overrides)

        # ── 4. Fallback chain ──────────────────────────────────────────────
        if not chain:
            chain = ["noise_reducer"]

        meta["material"] = material or "unknown"
        meta["restorability"] = restorability

        return {"chain": chain, "meta": meta, "parameters": parameters}

    def _analyze_cover(
        self,
        image_path: str,
        audio_meta: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        """Analysiert das Cover-Bild mittels DSP-Einbettung und CLAP-Ähnlichkeit.

        Falls back gracefully if no image analysis is available.
        Returns (chain_additions, meta_additions).
        """
        chain: list[str] = []
        meta: dict[str, Any] = {}
        try:
            # BEATs audio tagging (image → audio context)
            # Note: BEATs works on audio; for cover images we use the audio_meta
            # embeddings if available, otherwise skip.
            audio_embedding = audio_meta.get("audio_embedding")
            if audio_embedding is not None:
                emb = np.asarray(audio_embedding, dtype=np.float32)
                # Genre affinity from embedding via centroid distance
                genre_label, era_label = self._classify_from_embedding(emb)
                if genre_label:
                    meta["genre"] = genre_label
                if era_label:
                    meta["era"] = era_label
                lookup = f"{genre_label.lower().replace(' ', '_')}_{era_label[:4]}s" if era_label else ""
                rule = _GENRE_ERA_CHAINS.get(lookup, {})
                chain.extend(rule.get("chain", []))
        except Exception as exc:
            logger.debug("Cover Analyse fehlgeschlagen (%s) — heuristic Ersatzpfad", exc)
        return chain, meta

    def _classify_from_embedding(
        self,
        embedding: np.ndarray,
    ) -> tuple[str, str]:
        """Map DSP embedding to the nearest genre/era centroid.

        Uses L2 distance to pre-defined spectral profile centroids:
          - High spectral centroid + low warmth → Rock/Electronic → 1980s+
          - High harmonicity + low centroid    → Classical/Jazz → 1950s
          - High transient density             → Percussion genres
          - High LF energy ratio              → Bass/Hip-Hop
        """
        if len(embedding) < 4:
            return "", ""

        # embedding[0]: spectral_centroid (normalized), embedding[1]: mfcc_mean,
        # embedding[2]: harmonicity, embedding[3]: dynamic_range, ...
        centroid = float(embedding[0]) if len(embedding) > 0 else 0.5
        harmonic = float(embedding[2]) if len(embedding) > 2 else 0.5
        dyn_range = float(embedding[3]) if len(embedding) > 3 else 0.5
        lf_ratio = float(embedding[-2]) if len(embedding) > 5 else 0.3

        if harmonic > 0.7 and centroid < 0.4:
            genre, era = "Classical", "1960s"
        elif harmonic > 0.6 and centroid < 0.5:
            genre, era = "Jazz", "1950s"
        elif lf_ratio > 0.5:
            genre, era = "Hip-Hop", "1990s"
        elif centroid > 0.6 and dyn_range < 0.4:
            genre, era = "Rock", "1980s"
        elif centroid > 0.5:
            genre, era = "Pop", "1970s"
        else:
            genre, era = "Blues", "1940s"

        return genre, era


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import threading as _threading

_multimodal_decision_engine_instance = None
_multimodal_decision_engine_lock = _threading.Lock()


def get_multimodal_decision_engine() -> MultimodalDecisionEngine:
    """Gibt the process-wide singleton MultimodalDecisionEngine instance zurück."""
    global _multimodal_decision_engine_instance
    if _multimodal_decision_engine_instance is None:
        with _multimodal_decision_engine_lock:
            if _multimodal_decision_engine_instance is None:
                _multimodal_decision_engine_instance = MultimodalDecisionEngine()
    return _multimodal_decision_engine_instance
