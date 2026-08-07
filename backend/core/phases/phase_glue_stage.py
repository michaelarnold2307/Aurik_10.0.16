"""
§v10 GlueStagePhase — Vorletzte Phase: subtile Bus-Kompression für kohärenten Mix.

Die Glue-Stage läuft in ALLEN Modi als vorletzte Phase (nach TruePeak, vor Dithering/
Output-Format). Sie liest das ArtisticIntent-Objekt aus dem Restoration-Kontext und
verwendet genre-adaptive Parameter.

Pipeline-Position:
    phase_47_truepeak_limiter → phase_glue_stage → phase_41_output_format_optimization

Wissenschaftliche Basis:
- Katz, B. (2015): "Mastering Audio" (3rd Ed.), Chapter 14
- SSL G-Bus Compressor (legendärer "Glue"-Klang)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class GlueStagePhase(PhaseInterface):
    """Subtile Stereo-Bus-Kompression als finale Glue-Stage."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_glue_stage",
            name="Glue Stage",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            version="1.0.0",
            estimated_time_factor=0.01,
            memory_requirement_mb=16,
            is_cpu_intensive=False,
            description="Subtile Bus-Kompression (1.1:1–1.5:1) für kohärenten Mix",
        )

    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        **kwargs: Any,
    ) -> PhaseResult:
        """Wendet die Glue-Stage auf das Audio an.

        Liest artistic_intent aus kwargs (vom Restoration-Kontext injiziert)
        und verwendet genre-adaptive Kompressor-Parameter.
        """
        from backend.core.dsp.glue_stage import apply_glue_stage

        # ArtisticIntent aus dem Kontext lesen
        intent = kwargs.get("artistic_intent")
        genre: str | None = None
        if intent is not None:
            try:
                genre = getattr(intent, "genre", None)
            except Exception:
                genre = None
            if genre is None:
                try:
                    genre = str(intent.genre) if hasattr(intent, "genre") else None
                except Exception:
                    genre = None

        # Fallback: genre_label aus restoration_context wenn kein intent.genre
        if genre is None:
            genre = kwargs.get("genre_label") or None

        result = apply_glue_stage(
            audio,
            sample_rate,
            genre=genre,
            enabled=True,
        )

        # NaN/Inf protection for audio output (safety-in-depth; PhaseResult.__post_init__ also sanitizes)
        if not np.isfinite(result.audio).all():
            logger.warning("Verarbeitungsschritt_glue_Stufe: NaN/Inf im Ausgabe — wird mit nan_to_num bereinigt")
        output_audio = np.nan_to_num(result.audio, nan=0.0, posinf=0.0, neginf=0.0)

        return PhaseResult(
            audio=output_audio,
            success=True,
            metadata={
                "gain_reduction_db": result.gain_reduction_db,
                "makeup_gain_db": result.makeup_gain_db,
                "applied": result.applied,
                "genre": genre,
            },
        )
