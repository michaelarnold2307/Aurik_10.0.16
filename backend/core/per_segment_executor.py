"""§2.62 Per-Segment UV3-Ausführung — Audio splitten, pro Segment verarbeiten, crossfaden.

Hook: _profiled_phase_call prüft den Fahrplan auf per-Segment-Instruktionen.
Bei unterschiedlichen Stärken pro Segment wird die Phase segmentweise ausgeführt
und die Ergebnisse mit Hann-Crossfades an den Grenzen kombiniert.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Crossfade-Länge in Samples (bei 48 kHz)
XFADE_SAMPLES_48K = 576  # 12 ms — kurz genug um unhörbar zu sein


class _SegResult:
    """Phasen-Resultat mit .audio, .success, .resolved_defects für UV3-Kompatibilität.

    §v10.29a: .success und .resolved_defects wurden hinzugefügt, weil
    per_segment_executor._SegResult ohne diese Felder durch UV3's
    _normalize_phase_result() fiel und "invalid_phase_result_type:_SegResult"
    auslöste. Der Wrapper in _profiled_phase_call (isinstance(result, np.ndarray))
    konnte per_segment_executor._SegResult nicht wrappen, weil es kein ndarray ist.
    """

    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.success: bool = True
        self.resolved_defects: dict = {}
        self.warnings: list[str] = []


def run_phase_per_segment(
    audio: np.ndarray,
    sample_rate: int,
    phase_callable: Any,
    phase_kwargs: dict[str, Any],
    *,
    segment_bounds_s: list[float],
    segment_strengths: list[float],
    phase_id: str = "",
) -> _SegResult:
    """Führt eine Phase segmentweise mit unterschiedlichen Stärken aus.

    Args:
        audio: Eingabe-Audio (samples,) oder (channels, samples)
        sample_rate: Sample-Rate
        phase_callable: phase.process (aufrufbar mit audio, **kwargs)
        phase_kwargs: kwargs für phase.process (enthält "strength" das überschrieben wird)
        segment_bounds_s: [0.0, start_s1, start_s2, ..., end_s] in Sekunden, N+1 Werte
        segment_strengths: [strength_0, strength_1, ...] für jedes Segment, N Werte
        phase_id: Phase-ID für Logging

    Returns:
        _SegResult mit .audio (gleiche Shape wie Input)
    """
    is_1d = audio.ndim == 1
    audio_work = audio if is_1d else audio  # Bewahre originale Shape
    n_total = audio_work.shape[0] if is_1d else audio_work.shape[1]
    sr = max(1, sample_rate)

    # Crossfade-Länge an Sample-Rate anpassen
    xfade_samples = max(1, int(XFADE_SAMPLES_48K * sr / 48000))

    # Segment-Bounds in Samples umrechnen
    bounds_s = list(segment_bounds_s)
    if len(bounds_s) < 2:
        phase_kwargs_copy = dict(phase_kwargs)
        if segment_strengths:
            phase_kwargs_copy["strength"] = segment_strengths[0]
        result = phase_callable(audio, **phase_kwargs_copy)
        return _SegResult(np.asarray(getattr(result, "audio", result) if hasattr(result, "audio") else result))

    # Sample-Indizes der Grenzen
    bound_samples = [int(b * sr) for b in bounds_s]
    bound_samples[-1] = min(bound_samples[-1], n_total)

    # ── 1. Overlappende Segmente extrahieren ─────────────────────
    segments_audio: list[np.ndarray] = []
    segment_slices: list[tuple[int, int, int, int]] = []  # (seg_start, seg_end, left_pad, right_pad)

    for i in range(len(bound_samples) - 1):
        s_start = bound_samples[i]
        s_end = bound_samples[i + 1]

        left_pad = xfade_samples if i > 0 else 0
        right_pad = xfade_samples if i < len(bound_samples) - 2 else 0

        seg_start = max(0, s_start - left_pad)
        seg_end = min(n_total, s_end + right_pad)

        # §v10.117: Skip empty segments (causes broadcast (0,) (N,) errors)
        if seg_start >= seg_end:
            continue

        if is_1d:
            segments_audio.append(audio_work[seg_start:seg_end].copy())
        else:
            segments_audio.append(audio_work[:, seg_start:seg_end].copy())

        segment_slices.append((left_pad, right_pad, s_start, s_end))

    # ── 2. Pro Segment verarbeiten ───────────────────────────────
    segment_results: list[np.ndarray] = []
    for i, seg_audio in enumerate(segments_audio):
        kwargs_i = dict(phase_kwargs)
        if i < len(segment_strengths):
            kwargs_i["strength"] = float(segment_strengths[i])

        try:
            result = phase_callable(seg_audio, **kwargs_i)
            result_audio = getattr(result, "audio", result) if hasattr(result, "audio") else result
        except Exception as exc:
            logger.debug("§2.62 per-segment %s seg %d failed: %s", phase_id, i, exc)
            result_audio = seg_audio

        result_audio = np.asarray(result_audio)
        # §v10.31: Normalize channel layout to match input format (channels-first).
        # Some phases return channels-last (N, 2) while the pipeline uses
        # channels-first (2, N). Without normalization, the crossfade access
        # [ch, :xf_len] on (N, 2) gives shape (2,) instead of (xf_len,),
        # causing "operands could not be broadcast together with shapes (2,) (576,)".
        if not is_1d and result_audio.ndim == 2:
            _exp_channels = audio_work.shape[0]
            if result_audio.shape[0] != _exp_channels and result_audio.shape[1] == _exp_channels:
                result_audio = result_audio.T
        segment_results.append(result_audio)

    # ── 3. Crossfade und zusammensetzen ──────────────────────────
    if is_1d:
        output = np.zeros(n_total, dtype=np.float32)
    else:
        n_channels = audio_work.shape[0]
        output = np.zeros((n_channels, n_total), dtype=np.float32)

    fade_in = np.hanning(2 * xfade_samples)[:xfade_samples].astype(np.float32)
    fade_out = np.hanning(2 * xfade_samples)[xfade_samples:].astype(np.float32)

    for i, seg_result in enumerate(segment_results):
        left_pad, right_pad, s_start, s_end = segment_slices[i]

        core_in_start = left_pad
        core_in_end = len(seg_result) - right_pad if is_1d else seg_result.shape[1] - right_pad
        core_len = max(0, core_in_end - core_in_start)

        if is_1d:
            if core_len > 0 and s_start < n_total:
                write_end = min(s_start + core_len, n_total)
                output[s_start:write_end] = seg_result[core_in_start : core_in_start + (write_end - s_start)]

            # Left crossfade
            if left_pad > 0 and s_start > 0:
                xf_start = max(0, s_start - xfade_samples)
                xf_len = min(xfade_samples, s_start - xf_start)
                if xf_len > 0:
                    output[xf_start:s_start] = (
                        output[xf_start:s_start] * fade_out[-xf_len:] + seg_result[:xf_len] * fade_in[:xf_len]
                    )
        else:
            if core_len > 0 and s_start < n_total:
                write_end = min(s_start + core_len, n_total)
                output[:, s_start:write_end] = seg_result[:, core_in_start : core_in_start + (write_end - s_start)]

            # Left crossfade
            if left_pad > 0 and s_start > 0:
                xf_start = max(0, s_start - xfade_samples)
                xf_len = min(xfade_samples, s_start - xf_start)
                if xf_len > 0:
                    for ch in range(n_channels):
                        output[ch, xf_start:s_start] = (
                            output[ch, xf_start:s_start] * fade_out[-xf_len:]
                            + seg_result[ch, :xf_len] * fade_in[:xf_len]
                        )

    return _SegResult(output)


def get_segment_strengths_from_fahrplan(
    fahrplan: Any,
    phase_id: str,
    base_strength: float = 1.0,
) -> tuple[list[float], list[float]] | None:
    """Liest per-Segment-Stärken aus dem Fahrplan.

    Args:
        fahrplan: Fahrplan-Objekt mit .instructions und .sections
        phase_id: z.B. "phase_03_denoise"
        base_strength: Basis-Stärke aus kwargs["strength"]

    Returns:
        (segment_bounds_s, segment_strengths) wenn non-uniform, sonst None
    """
    if fahrplan is None:
        return None

    instructions = getattr(fahrplan, "instructions", {}) or {}
    sections = getattr(fahrplan, "sections", []) or []

    if not sections or not instructions:
        return None

    phase_instructions = instructions.get(phase_id)
    if not phase_instructions:
        return None

    bounds: list[float] = [0.0]
    strengths: list[float] = []

    for instr in phase_instructions:
        if hasattr(instr, "skip") and instr.skip:
            strengths.append(0.0)
        else:
            mod = instr.strength_mod if hasattr(instr, "strength_mod") else 1.0
            strengths.append(float(base_strength) * float(mod))

        idx = instr.section_idx if hasattr(instr, "section_idx") else 0
        if idx < len(sections):
            bounds.append(float(sections[idx][1]))

    if len(strengths) <= 1:
        return None

    if len(set(strengths)) <= 1:
        return None

    if len(bounds) != len(strengths) + 1:
        return None

    return bounds, strengths
