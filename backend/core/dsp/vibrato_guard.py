"""§2.72 Vibrato-Tiefe Guard.

Prüft nach NR/Dynamics-Phasen, ob die F0-Modulationstiefe (Vibrato) in
Vibrato-Zonen erhalten bleibt. Mehr als 10 % Reduktion → Strength × 0.5
in Vibrato-Segmenten.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation, VibratoDepthResult
    result = check_vibrato_depth_preservation(pre, post, sr)
    if not result.ok:
        # Strength-Reduktion um 50 % in Vibrato-Segmenten (s. §2.72)
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from backend.core.core_utils import fft_autocorr

logger = logging.getLogger(__name__)

# Vibrato-Fenster-Parameter
_WINDOW_MS = 150.0  # Analyse-Fensterlänge in ms
_HOP_MS = 50.0  # Fenster-Hop in ms
# Typischer Vibrato-Hub (max-min F0) in Hz
_MIN_VIBRATO_HZ = 0.3  # unter 0.3 Hz = kein Vibrato
_MAX_VIBRATO_HZ = 40.0  # über 40 Hz = kein Vibrato
# Erlaubte Tiefe-Reduktion in Prozent
VIBRATO_MAX_REDUCTION_PCT = 10.0


@dataclass
class VibratoDepthResult:
    """Ergebnis der Vibrato-Tiefe-Messung.

    Attributes:
        depth_pre_hz: F0-Modulationstiefe vor der Phase (max-min F0 in Hz).
        depth_post_hz: F0-Modulationstiefe nach der Phase.
        depth_reduction_pct: Relative Reduktion in Prozent (0–100).
        ok: True wenn depth_reduction_pct <= 10 %.
    """

    depth_pre_hz: float
    depth_post_hz: float
    depth_reduction_pct: float
    ok: bool


def _f0_from_autocorr(mono: np.ndarray, sr: int) -> np.ndarray:
    """Schätzt Frame-weise F0 via Autokorrelation (schnell, kein ML-Modell)."""
    hop = int(_HOP_MS / 1000.0 * sr)
    win = int(_WINDOW_MS / 1000.0 * sr)
    n = len(mono)
    n_frames = max(1, (n - win) // hop + 1)
    f0s = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = mono[start : start + win]
        if len(frame) < 64:
            continue
        # Energie-Check: stille Frames überspringen
        if float(np.abs(frame).max()) < 1e-4:
            continue
        # FFT-basierte Autokorrelation (O(N log N), V08-konform)
        corr = fft_autocorr(frame.astype(np.float64))
        # Suche Maximum im Periodenbereich 60–1000 Hz
        lag_min = max(1, int(sr / 1000.0))  # 1000 Hz
        lag_max = int(sr / 60.0)  # 60 Hz
        if lag_max > len(corr) - 1:
            lag_max = len(corr) - 1
        if lag_min >= lag_max:
            continue
        sub = corr[lag_min:lag_max]
        peak_lag = int(np.argmax(sub)) + lag_min
        if corr[0] > 1e-10:
            confidence = float(corr[peak_lag] / corr[0])
        else:
            confidence = 0.0
        if confidence > 0.3:
            f0s[i] = float(sr) / peak_lag
    return f0s  # type: ignore[no-any-return]


def _measure_vibrato_depth(f0: np.ndarray) -> float:
    """Berechnet die mediane Vibrato-Tiefe (max-min F0) über gleitende Fenster."""
    voiced = f0[f0 > 50.0]
    if len(voiced) < 4:
        return 0.0
    win_frames = max(3, int(_WINDOW_MS / _HOP_MS))
    depths = []
    for i in range(0, len(voiced) - win_frames, 1):
        seg = voiced[i : i + win_frames]
        depth = float(seg.max() - seg.min())
        if _MIN_VIBRATO_HZ < depth < _MAX_VIBRATO_HZ:
            depths.append(depth)
    return float(np.median(depths)) if depths else 0.0


def check_vibrato_depth_preservation(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
) -> VibratoDepthResult:
    """Prüft ob die Vibrato-Tiefe durch die Phase reduziert wurde.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase.
        sr: Sample-Rate (muss 48000 sein).

    Returns:
        VibratoDepthResult. ok=False wenn depth_reduction_pct > 10 %.
    """
    assert sr == 48000
    _fallback = VibratoDepthResult(depth_pre_hz=0.0, depth_post_hz=0.0, depth_reduction_pct=0.0, ok=True)

    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        pre_mono = pre.mean(axis=0) if pre.ndim == 2 else pre
        post_mono = post.mean(axis=0) if post.ndim == 2 else post

        f0_pre = _f0_from_autocorr(pre_mono, sr)
        f0_post = _f0_from_autocorr(post_mono, sr)

        depth_pre = _measure_vibrato_depth(f0_pre)
        depth_post = _measure_vibrato_depth(f0_post)

        if depth_pre < _MIN_VIBRATO_HZ:
            # Kein Vibrato im Material — kein Check nötig
            return _fallback

        reduction_pct = float(np.clip(100.0 * (depth_pre - depth_post) / (depth_pre + 1e-9), 0.0, 100.0))
        ok = reduction_pct <= VIBRATO_MAX_REDUCTION_PCT

        if not ok:
            logger.info(
                "§2.72 Vibrato-Tiefe: pre=%.2f Hz post=%.2f Hz reduction=%.1f%% > %.0f%%",
                depth_pre,
                depth_post,
                reduction_pct,
                VIBRATO_MAX_REDUCTION_PCT,
            )

        return VibratoDepthResult(
            depth_pre_hz=round(depth_pre, 3),
            depth_post_hz=round(depth_post, 3),
            depth_reduction_pct=round(reduction_pct, 2),
            ok=ok,
        )

    except Exception as exc:
        logger.debug("Pruefung_vibrato_depth_preservation nicht blockierend: %s", exc)
        return _fallback
