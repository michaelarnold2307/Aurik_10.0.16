"""
Per-Event Strength Oracle (§V38, §0l)

Problem: Phasen iterieren über Defekt-Ereignisse mit einheitlicher `strength`
für ALLE Events → leichte Events werden über-prozessiert, schwere Events in
VFA-Schutzzonen (Vibrato, Frisson, Flüster, Passaggio) werden nicht beschränkt.

Lösung: Pro Event eine lokale Stärke berechnen aus:
  1. Lokalem Energie-Anomalie-Proxy (250ms Kontext-RMS vor/nach dem Event)
  2. VFA-Schutzzonen-Cap (Vibrato 0.20, Frisson 0.30, Flüster 0.25, Passaggio 0.35)
  3. Basis-Stärke aus dem Defekt-Scanner

Integration:
    from backend.core.per_event_strength_oracle import compute_local_strength

    for event_start, event_end in defect_events:
        local_strength = compute_local_strength(
            mono_ref, event_start, event_end, sr,
            base_strength=base_strength,
            protected_zones=protected_zones  # from VFA detector
        )
        apply_repair(audio, start=event_start, end=event_end, strength=local_strength)

Architecture invariants:
  - base_strength < 1e-6 → returns 0.0 (Passthrough invariant)
  - Strength is clamped to [0.0, 1.0]
  - Protected zones ALWAYS cap at their max_cap, regardless of base_strength
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# VFA protection zone caps (§V38, §0l)
VFA_PROTECTION_CAPS = {
    "vibrato": 0.20,
    "frisson": 0.30,
    "fluester": 0.25,    # Flüster-Passagen
    "passaggio": 0.35,
}

# Default context window for local energy anomaly detection
DEFAULT_CONTEXT_MS = 250.0  # ms


def compute_local_strength(
    mono_ref: np.ndarray,
    event_start: int,
    event_end: int,
    sample_rate: int,
    base_strength: float = 0.5,
    protected_zones: Optional[list[tuple[int, int, str, float]]] = None,
    context_ms: float = DEFAULT_CONTEXT_MS,
) -> float:
    """Compute per-event local strength.

    Args:
        mono_ref:       Reference mono audio array (float32, [-1, 1]).
        event_start:    Start sample of the defect event.
        event_end:      End sample of the defect event.
        sample_rate:    Audio sample rate in Hz.
        base_strength:  Base strength from defect scanner (0.0–1.0).
        protected_zones: List of (start_s, end_s, zone_type, max_cap) tuples.
                         zone_type is one of: vibrato, frisson, fluester, passaggio.
                         max_cap overrides VFA_PROTECTION_CAPS defaults.
        context_ms:     Context window in ms for RMS comparison (default: 250ms).

    Returns:
        Local strength in [0.0, 1.0]. 0.0 means skip (passthrough).

    Invariants:
        - base_strength < 1e-6 → returns 0.0
        - Protected zone cap takes priority over base_strength
        - Result is clamped to [0.0, min(base_strength * 1.5, 1.0)]
    """
    # Passthrough invariant
    if base_strength < 1e-6:
        return 0.0

    # Clamp event bounds
    n = len(mono_ref)
    event_start = max(0, min(event_start, n - 1))
    event_end = max(event_start + 1, min(event_end, n))

    # ── 1. VFA Protection Zone Check ──────────────────────────────────
    if protected_zones:
        event_start_s = event_start / sample_rate
        event_end_s = event_end / sample_rate
        for zone_start_s, zone_end_s, zone_type, max_cap in protected_zones:
            # Check if event overlaps with protected zone
            if event_start_s < zone_end_s and event_end_s > zone_start_s:
                # Use zone-specific cap (default from VFA_PROTECTION_CAPS)
                cap = max_cap if max_cap > 0 else VFA_PROTECTION_CAPS.get(zone_type, 0.25)
                logger.debug(
                    "VFA protection: event [%.2fs-%.2fs] in %s zone → capped at %.2f",
                    event_start_s, event_end_s, zone_type, cap,
                )
                return min(base_strength, cap)

    # ── 2. Local Energy Anomaly ───────────────────────────────────────
    context_samples = int(context_ms / 1000.0 * sample_rate)

    # Pre-context: samples before the event
    pre_start = max(0, event_start - context_samples)
    pre_end = event_start
    pre_rms = _rms(mono_ref[pre_start:pre_end])

    # Post-context: samples after the event
    post_start = event_end
    post_end = min(n, event_end + context_samples)
    post_rms = _rms(mono_ref[post_start:post_end])

    # Event RMS
    event_rms = _rms(mono_ref[event_start:event_end])

    # Context average RMS
    context_rms = (pre_rms + post_rms) / 2.0 if (pre_rms + post_rms) > 1e-8 else 1e-8

    # Energy anomaly: how much louder/quieter is the event vs. context?
    anomaly_ratio = event_rms / context_rms

    # Scale: extreme anomalies (crackle, loud click) → higher strength
    #        subtle anomalies (minor dropout) → lower strength
    # Map ratio [0.1, 10.0] → scale [0.3, 1.5]
    if anomaly_ratio > 1.0:
        # Louder event: linear scale from 1.0 to 1.5
        energy_scale = 1.0 + 0.5 * min((anomaly_ratio - 1.0) / 9.0, 1.0)
    else:
        # Quieter event: linear scale from 0.3 to 1.0
        energy_scale = 0.3 + 0.7 * max(anomaly_ratio / 1.0, 0.0)

    # ── 3. Combined Strength ──────────────────────────────────────────
    local_strength = base_strength * energy_scale
    local_strength = np.clip(local_strength, 0.0, min(base_strength * 1.5, 1.0))

    return float(local_strength)


def _rms(audio: np.ndarray) -> float:
    """Compute RMS of audio segment."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))


def build_protected_zones_from_vfa(
    vfa_zones: list[dict],
    sample_rate: int,
) -> list[tuple[int, int, str, float]]:
    """Convert VFA detector output to protected zone format.

    Args:
        vfa_zones: List of VFA zone dicts with keys:
                   start_s, end_s, type, [max_cap]
        sample_rate: Audio sample rate.

    Returns:
        List of (start_sample, end_sample, zone_type, max_cap) tuples.
    """
    zones = []
    for zone in vfa_zones:
        start_s = float(zone.get("start_s", 0))
        end_s = float(zone.get("end_s", 0))
        zone_type = str(zone.get("type", "vibrato"))
        max_cap = float(zone.get("max_cap", -1.0))  # -1 = use default

        if max_cap < 0:
            max_cap = VFA_PROTECTION_CAPS.get(zone_type, 0.25)

        zones.append((
            int(start_s * sample_rate),
            int(end_s * sample_rate),
            zone_type,
            max_cap,
        ))
    return zones
