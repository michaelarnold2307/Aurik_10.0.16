"""
§v10 Glue Stage — Finale subtile Bus-Kompression für kohärenten Mix.

Der „Glue"-Effekt ist das, was ein menschlicher Toningenieur ganz am Ende
macht: eine extrem subtile Stereo-Bus-Kompression (1.1:1–1.5:1 Ratio, <2 dB
Gain-Reduction), die alle Frequenzbereiche und Spuren zu einem kohärenten
Ganzen „zusammenklebt".  Kein kreatives „Sound-Design" — nur der letzte
Schliff für ein professionell klingendes Master.

Wissenschaftliche Basis:
- Katz, B. (2015): „Mastering Audio" (3rd Ed.), Chapter 14: „The Final Touch"
- iZotope Ozone 11 Maximizer / Vintage Compressor (Glue-Preset)
- SSL G-Bus Compressor (legendärer „Glue"-Klang)

Parameter:
- Ratio: 1.2:1 (extrem sanft)
- Attack: 30 ms (lässt Transienten durch)
- Release: 100 ms (musikalisch, folgt dem Groove)
- Threshold: so dass max 1.5 dB GR entsteht
- Makeup-Gain: Auto (kompensiert exakt die GR)
- Mix: 100% (kein Parallel — das ist der finale Bus)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GlueStageResult:
    """Ergebnis der Glue-Stage."""

    audio: np.ndarray
    gain_reduction_db: float
    makeup_gain_db: float
    applied: bool


def apply_glue_stage(
    audio: np.ndarray,
    sr: int,
    *,
    ratio: float = 1.2,
    threshold_db: float = -18.0,
    attack_ms: float = 30.0,
    release_ms: float = 100.0,
    max_gr_db: float = 1.5,
    enabled: bool = True,
    genre: str | None = None,
) -> GlueStageResult:
    """Wendet eine subtile Bus-Kompression („Glue") auf das fertige Master an.

    Genre-adaptive Anpassungen:
    - „ballad"/„classical"/„ambient": ratio = 1.1:1, max_gr_db = 0.8 (noch sanfter)
    - „rock"/„metal"/„punk": ratio = 1.3:1, attack_ms = 20 (etwas mehr Biss)
    - „schlager"/„pop"/„electronic": Standard (1.2:1, 30ms Attack)

    Args:
        audio:        Stereo-Audio (2×N oder N×2), float32, nach Loudness/TruePeak
        sr:           Sample-Rate (muss 48000 sein)
        ratio:        Kompressions-Ratio (1.0 = keine Kompression)
        threshold_db: Schwellwert in dBFS
        attack_ms:    Attack-Zeit in ms
        release_ms:   Release-Zeit in ms
        max_gr_db:    Maximale Gain-Reduction in dB
        enabled:      False = Passthrough
        genre:        Optionales Genre für adaptive Parameter

    Returns:
        GlueStageResult mit komprimiertem Audio
    """
    if not enabled:
        return GlueStageResult(
            audio=np.asarray(audio, dtype=np.float32),
            gain_reduction_db=0.0,
            makeup_gain_db=0.0,
            applied=False,
        )

    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim < 2:
        # Mono: trotzdem Glue anwenden (leichte Sättigung)
        arr = np.column_stack([arr, arr])

    # Genre-adaptive Parameter
    if genre:
        g = genre.lower()
        if any(t in g for t in ("ballad", "classical", "ambient", "choral", "orchestral")):
            ratio = 1.1
            max_gr_db = 0.8
            attack_ms = 40.0
        elif any(t in g for t in ("rock", "metal", "punk", "hardcore")):
            ratio = 1.3
            attack_ms = 20.0
        elif any(t in g for t in ("jazz", "blues", "folk", "singer")):
            ratio = 1.15
            max_gr_db = 1.0

    # Zeitkonstanten in Samples
    attack_s = attack_ms / 1000.0
    release_s = release_ms / 1000.0
    attack_samp = max(1, int(attack_s * sr))
    release_samp = max(1, int(release_s * sr))

    # Attack/Release-Koeffizienten (exponentielle Glättung)
    alpha_attack = 1.0 - np.exp(-1.0 / attack_samp) if attack_samp > 0 else 1.0
    alpha_release = 1.0 - np.exp(-1.0 / release_samp) if release_samp > 0 else 1.0

    # Threshold linear
    threshold_lin = 10.0 ** (threshold_db / 20.0)

    # Envelope-Detektion (Peak, stereo-gekoppelt)
    if arr.shape[1] >= 2:
        envelope = np.maximum(np.abs(arr[:, 0]), np.abs(arr[:, 1]))
    else:
        envelope = np.abs(arr[:, 0])

    # Gain-Reduction Berechnung (Feed-Forward Kompressor)
    gain_reduction_lin = np.ones(len(envelope), dtype=np.float64)
    gr_state = 1.0  # Smoothing-Zustand

    for i in range(len(envelope)):
        # Berechne Ziel-GR für dieses Sample
        if envelope[i] > threshold_lin:
            over_db = 20.0 * np.log10(max(envelope[i] / threshold_lin, 1e-12))
            target_gr_db = over_db * (1.0 - 1.0 / ratio)
            target_gr_db = min(target_gr_db, max_gr_db)
            target_gr_lin = 10.0 ** (-target_gr_db / 20.0)
        else:
            target_gr_lin = 1.0  # Keine Reduktion unter Threshold

        # Smoothing (Attack/Release)
        if target_gr_lin < gr_state:
            gr_state = alpha_attack * target_gr_lin + (1.0 - alpha_attack) * gr_state
        else:
            gr_state = alpha_release * target_gr_lin + (1.0 - alpha_release) * gr_state

        gain_reduction_lin[i] = gr_state

    # Apply gain reduction
    if arr.shape[1] >= 2:
        arr[:, 0] *= gain_reduction_lin
        arr[:, 1] *= gain_reduction_lin
    else:
        arr *= gain_reduction_lin[:, np.newaxis]

    # Auto makeup gain (kompensiert exakt die durchschnittliche GR)
    avg_gr_lin = (
        float(np.mean(gain_reduction_lin[envelope > threshold_lin])) if np.any(envelope > threshold_lin) else 1.0
    )
    avg_gr_db = -20.0 * np.log10(max(avg_gr_lin, 1e-12))
    makeup_lin = 10.0 ** (min(avg_gr_db, max_gr_db) / 20.0)
    arr *= makeup_lin

    actual_gr_db = -20.0 * np.log10(max(float(np.min(gain_reduction_lin)), 1e-12))
    makeup_db = 20.0 * np.log10(makeup_lin)

    logger.debug(
        "Glue-Stufe: Verhaeltnis=%.1f:1, max_GR=%.1f dB, makeup=%.1f dB, genre=%s",
        ratio,
        actual_gr_db,
        makeup_db,
        genre or "auto",
    )

    if audio.ndim == 1:
        arr = arr[:, 0]

    return GlueStageResult(
        audio=np.clip(arr.astype(np.float32), -1.0, 1.0),
        gain_reduction_db=float(actual_gr_db),
        makeup_gain_db=float(makeup_db),
        applied=True,
    )
