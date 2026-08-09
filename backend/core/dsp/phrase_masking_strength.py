"""
Phrase Masking Strength Map — Lücke 3 (v10.0.0.x)
================================================

Berechnet eine zeitaufgelöste Stärken-Modulations-Map, die NR/Phase-Stärken
pro Zeitfenster an die psychoakustische Maskierungskapazität des Songs anpasst:

    Hohe Maskierungsenergie (laute Begleitung) → more_strength_allowed (bis +0.25)
    Exponierte Stellen (Solo-Stimme, leise Passagen) → strength_floor (−0.30)
    Frisson-Zonen (§0p) → protected (strength × 0.15)

ALGORITHMUS:
    1. RMS-Hüllkurve des Gesamtsignals (50 ms Frames)
    2. Psychoakustische Maskierungsschätzung: Energie im 500–4000 Hz Band
       (Begleitinstrumente) relativ zum Vokalenergie-Band (200–3500 Hz)
    3. Normierung: Masking-Score ∈ [0, 1] → Stärken-Modifier ∈ [−0.30, +0.25]
    4. Frisson-Zonen auf strength_scale = 0.15 erzwingen
    5. Glättung: Zeitliche Glättung (Median + Gaußscher Rolloff 250 ms)
       verhindert Frame-zu-Frame-Springen

Ausgabe: PhraseStrengthMap — aufrufbar mit Zeitstempel → gibt Stärken-Modifier.

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0.x — Lücke 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as sp_ndimage
import scipy.signal as sp_sig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_FRAME_MS = 50  # Analyse-Fenster in ms
_MASKING_BAND_LOW = 500.0  # Untere Grenze Maskierungsband (Hz)
_MASKING_BAND_HIGH = 4000.0  # Obere Grenze Maskierungsband (Hz)
_VOCAL_BAND_LOW = 200.0
_VOCAL_BAND_HIGH = 3500.0

# Stärken-Modifier-Grenzen
_STRENGTH_MOD_MAX = +0.25  # Zusätzliche Stärke bei hoher Maskierung
_STRENGTH_MOD_MIN = -0.30  # Stärken-Reduktion bei exponierten Stellen
_FRISSON_STRENGTH_SCALE = 0.15  # Frisson-Zonen: fast kein Eingriff


# ---------------------------------------------------------------------------
# Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class PhraseStrengthMap:
    """
    Zeitaufgelöste Stärken-Modifier-Map für alle Phasen.

    Verwendung::

        psm = compute_phrase_strength_map(audio, sr, frisson_zones)
        modifier = psm.get_modifier_at(t_seconds)  # ∈ [-0.30, +0.25]
        effective_strength = base_strength + modifier
    """

    frame_dur_s: float
    modifiers: np.ndarray  # Shape: (n_frames,), Werte ∈ [_STRENGTH_MOD_MIN, _STRENGTH_MOD_MAX]
    masking_scores: np.ndarray  # Shape: (n_frames,), Werte ∈ [0, 1]
    frisson_mask: np.ndarray  # Shape: (n_frames,), bool
    total_duration_s: float

    def get_modifier_at(self, t_s: float) -> float:
        """
        Stärken-Modifier zum Zeitpunkt t_s (Sekunden).

        Returns:
            float ∈ [_STRENGTH_MOD_MIN, +_STRENGTH_MOD_MAX]
            0.0 bei t_s außerhalb des Audio-Bereichs.
        """
        if t_s < 0.0 or len(self.modifiers) == 0:
            return 0.0
        idx = int(t_s / self.frame_dur_s)
        idx = min(idx, len(self.modifiers) - 1)
        return float(self.modifiers[idx])

    def get_strength_scale_at(self, t_s: float) -> float:
        """
        Multiplikativer Stärken-Skalierungsfaktor ∈ [0.15, 1.25].
        Direkter Multiplikator für kwargs["strength"] in Phasen.
        """
        if len(self.frisson_mask) > 0:
            idx = int(t_s / self.frame_dur_s)
            idx = min(idx, len(self.frisson_mask) - 1)
            if self.frisson_mask[idx]:
                return _FRISSON_STRENGTH_SCALE  # Frisson: minimaler Eingriff
        mod = self.get_modifier_at(t_s)
        return float(np.clip(1.0 + mod, _FRISSON_STRENGTH_SCALE, 1.0 + _STRENGTH_MOD_MAX))

    def get_segment_modifier(self, start_s: float, end_s: float) -> float:
        """Mittlerer Modifier über Zeitfenster [start_s, end_s]."""
        if len(self.modifiers) == 0:
            return 0.0
        i0 = max(0, int(start_s / self.frame_dur_s))
        i1 = min(len(self.modifiers), int(end_s / self.frame_dur_s) + 1)
        if i0 >= i1:
            return 0.0
        return float(np.mean(self.modifiers[i0:i1]))

    def to_dict(self) -> dict:
        """Kompakte Metadaten für Logging/Phase-Delta."""
        return {
            "n_frames": len(self.modifiers),
            "frame_dur_s": self.frame_dur_s,
            "modifier_mean": float(np.mean(self.modifiers)),
            "modifier_std": float(np.std(self.modifiers)),
            "exposed_fraction": float(np.mean(self.modifiers < -0.10)),
            "high_masking_fraction": float(np.mean(self.modifiers > 0.10)),
            "frisson_fraction": float(np.mean(self.frisson_mask)),
            "total_duration_s": self.total_duration_s,
        }


# ---------------------------------------------------------------------------
# Band-Energie-Hilfsfunktionen
# ---------------------------------------------------------------------------


def _band_rms_frames(
    mono: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    frame_len: int,
) -> np.ndarray:
    """RMS-Hüllkurve im Band [low_hz, high_hz], Frames à frame_len Samples."""
    nyq = sr / 2.0
    sos = sp_sig.butter(
        4,
        [low_hz / nyq, min(high_hz / nyq, 0.98)],
        btype="band",
        output="sos",
    )
    filtered = sp_sig.sosfiltfilt(sos, mono)
    n_frames = len(filtered) // frame_len
    rms = np.array(
        [float(np.sqrt(np.mean(filtered[i * frame_len : (i + 1) * frame_len] ** 2) + 1e-30)) for i in range(n_frames)]
    )
    return rms  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def compute_phrase_strength_map(
    audio: np.ndarray,
    sr: int,
    frisson_zones: list[tuple[float, float]] | None = None,
    *,
    frame_ms: float = _FRAME_MS,
    smooth_ms: float = 250.0,
) -> PhraseStrengthMap:
    """
    Berechnet PhraseStrengthMap für das Audio.

    Non-blocking: Bei Exception → neutrale Map (alle Modifier = 0.0).

    Args:
        audio:          Mono oder Stereo float32/64
        sr:             Abtastrate (48 000 Hz)
        frisson_zones:  Liste von (start_s, end_s) — Frisson-Schutzpassagen (§0p)
        frame_ms:       Analyse-Fensterlänge in ms
        smooth_ms:      Zeitliche Glättung des Modifier-Signals in ms

    Returns:
        PhraseStrengthMap — zeitaufgelöste Modifier.
    """
    _empty = PhraseStrengthMap(
        frame_dur_s=frame_ms / 1000.0,
        modifiers=np.zeros(0, dtype=np.float32),
        masking_scores=np.zeros(0, dtype=np.float32),
        frisson_mask=np.zeros(0, dtype=bool),
        total_duration_s=0.0,
    )

    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Mono
        if audio.ndim == 2:
            mono = audio.mean(axis=0).astype(np.float64)
        else:
            mono = audio.astype(np.float64)

        dur_s = len(mono) / sr
        if dur_s < 0.5:
            return _empty

        frame_len = max(1, int(sr * frame_ms / 1000.0))
        n_frames = len(mono) // frame_len

        if n_frames < 2:
            return _empty

        # 1. Gesamte RMS-Hüllkurve
        rms_total = np.array(
            [float(np.sqrt(np.mean(mono[i * frame_len : (i + 1) * frame_len] ** 2) + 1e-30)) for i in range(n_frames)]
        )

        # 2. Band-RMS für Maskierungsband und Vokalband
        rms_mask = _band_rms_frames(mono, sr, _MASKING_BAND_LOW, _MASKING_BAND_HIGH, frame_len)
        rms_vocal = _band_rms_frames(mono, sr, _VOCAL_BAND_LOW, _VOCAL_BAND_HIGH, frame_len)

        n_frames = min(n_frames, len(rms_mask), len(rms_vocal))
        rms_total = rms_total[:n_frames]
        rms_mask = rms_mask[:n_frames]
        rms_vocal = rms_vocal[:n_frames]

        # 3. Masking-Score: Verhältnis Begleitenergie / Vokalenergie
        # Hohe Begleitenergie relativ zum Vokal → viel Maskierung → mehr Spielraum
        mask_ratio = rms_mask / (rms_vocal + 1e-12)
        # Normiere auf [0, 1] mit Perzentilen (robust gegen Ausreißer)
        p10 = float(np.percentile(mask_ratio, 10))
        p90 = float(np.percentile(mask_ratio, 90))
        if p90 > p10:
            masking_scores = np.clip((mask_ratio - p10) / (p90 - p10 + 1e-12), 0.0, 1.0)
        else:
            masking_scores = np.full(n_frames, 0.5)

        # 4. Stärken-Modifier: linear von [0,1] → [MIN, MAX]
        modifiers = _STRENGTH_MOD_MIN + masking_scores * (_STRENGTH_MOD_MAX - _STRENGTH_MOD_MIN)

        # 5. Zusätzlicher Penalty für sehr leise Frames (exponierte Solo-Stellen)
        global_peak_rms = float(np.percentile(rms_total, 90)) + 1e-30
        relative_rms = rms_total / global_peak_rms
        silence_penalty = np.where(relative_rms < 0.15, -0.10, 0.0)
        modifiers += silence_penalty

        modifiers = np.clip(modifiers, _STRENGTH_MOD_MIN, _STRENGTH_MOD_MAX).astype(np.float32)

        # 6. Zeitliche Glättung
        smooth_frames = max(1, int(smooth_ms / frame_ms))
        modifiers = sp_ndimage.uniform_filter1d(modifiers, size=smooth_frames)

        # 7. Frisson-Maske
        frame_dur_s = frame_ms / 1000.0
        frisson_mask = np.zeros(n_frames, dtype=bool)
        if frisson_zones:
            for start_s, end_s in frisson_zones:
                i0 = max(0, int(start_s / frame_dur_s))
                i1 = min(n_frames, int(end_s / frame_dur_s) + 1)
                frisson_mask[i0:i1] = True
            # Frisson-Frames: minimalen Modifier erzwingen
            frisson_modifier = _FRISSON_STRENGTH_SCALE - 1.0  # z.B. -0.85 → scale 0.15
            modifiers[frisson_mask] = np.minimum(modifiers[frisson_mask], frisson_modifier)

        masking_scores_f32 = masking_scores[:n_frames].astype(np.float32)

        logger.debug(
            "phrase_strength_map: %d frames, modifier_mean=%.3f, frisson_frames=%d",
            n_frames,
            float(np.mean(modifiers)),
            int(np.sum(frisson_mask)),
        )

        return PhraseStrengthMap(
            frame_dur_s=frame_dur_s,
            modifiers=modifiers.astype(np.float32),
            masking_scores=masking_scores_f32,
            frisson_mask=frisson_mask,
            total_duration_s=n_frames * frame_dur_s,
        )

    except Exception as exc:
        logger.warning("§G23 berechnen_phrase_strength_map: DSP-Ersatzpfad — %s", exc, exc_info=True)
        return _empty
