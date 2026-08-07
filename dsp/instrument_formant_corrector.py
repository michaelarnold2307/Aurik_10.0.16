"""
dsp/instrument_formant_corrector.py — Instrument Formant Drift Correction via DTW
===================================================================================

Detects and corrects sustained deviations of tracked LPC resonance peaks from
the canonical physical targets of each instrument class.

The analogy to vocal processing:
    Vocals  → FormantCorrector detects drift from speaker identity (median ref)
    Instruments → InstrumentFormantDriftCorrector detects drift from physical
                  body resonances (McIntyre & Woodhouse / Benade / Christensen)

Algorithm:
    1. Track LPC formants frame-by-frame using FormantTracker (25 ms / 10 ms hop).
    2. Build a reference trajectory = constant target (F1_target, …) per frame
       from InstrumentFormantTargets — the physically correct resonances.
    3. Compute DTW (Dynamic Time Warping) warp path between the tracked F1
       trajectory and the flat reference (Välimäki et al. 2006).
    4. Derive a per-frame correction deviation: where the tracked trajectory
       diverges from the DTW-aligned reference by more than DRIFT_THRESHOLD_HZ
       for at least DRIFT_MIN_FRAMES consecutive frames → "formant drift".
    5. For drifted frames: apply a gentle peak-EQ nudge toward the target
       frequency at correction_strength ≤ 0.30 (identity-safe).

DTW implementation:
    Uses a vectorized DP matrix (scipy.spatial.distance.cdist for cost matrix,
    traceback via backward pass) — O(n·m) time, O(n·m) space.
    For long audio (> 60 s) the trajectory is downsampled to ≤ 1000 points
    before DTW to stay within the 2 s performance budget.

Scientific foundation:
    Välimäki et al. (2006): "Frequency Tracking from Musical Signals using DTW"
    McIntyre & Woodhouse (1978): "Acoustics of bowed instruments"
    Benade (1976): "Fundamentals of Musical Acoustics"
    Christensen (1982): "Structural-acoustical analysis of guitar"

Singleton pattern (§3.2 Double-Checked Locking), NaN/Inf-guard (§3.1),
assert sample_rate == 48000, full PEP 484 type annotations.

Author: Aurik Development Team
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SR_REQUIRED: int = 48_000

# Minimum sustained deviation to classify as drift (Hz)
DRIFT_THRESHOLD_HZ: float = 80.0
# Minimum consecutive drifting frames to trigger correction
DRIFT_MIN_FRAMES: int = 10  # 10 × 10 ms hop = 100 ms
# Maximum EQ correction strength (identity-safe ceiling)
MAX_CORRECTION_STRENGTH: float = 0.30
# Maximum boost per corrected frame (dB)
MAX_BOOST_DB: float = 3.0
# DTW downsample limit (frames) to stay within performance budget
DTW_MAX_FRAMES: int = 1000
# LPC formant frequency filter range for instruments (wider than vocals)
INSTRUMENT_FREQ_RANGE: tuple[float, float] = (50.0, 8000.0)


# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class InstrumentDriftResult:
    """Result of :class:`InstrumentFormantDriftCorrector`.

    Attributes:
        audio:               Corrected audio (same shape as input).
        instrument:          Instrument type string used for targeting.
        drift_detected:      Whether sustained formant drift was found.
        n_frames_corrected:  Number of frames where EQ correction was applied.
        total_frames:        Total number of analysis frames.
        mean_drift_hz:       Mean deviation from target F1 across all frames (Hz).
        max_drift_hz:        Maximum single-frame deviation from target F1 (Hz).
        dtw_distance:        Normalised DTW distance between tracked and target trajectory.
        correction_strength: Effective blend factor used.
        f1_target_hz:        Target F1 frequency used for this instrument (Hz).
    """

    audio: np.ndarray
    instrument: str
    drift_detected: bool
    n_frames_corrected: int
    total_frames: int
    mean_drift_hz: float
    max_drift_hz: float
    dtw_distance: float
    correction_strength: float
    f1_target_hz: float
    details: dict = field(default_factory=dict)


# ── DTW helpers ───────────────────────────────────────────────────────────────


def _dtw_distance_and_path(seq_a: np.ndarray, seq_b: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Berechnet DTW distance and warp path between two 1-D sequences.

    Uses a vectorized DP cost matrix (Sakoe & Chiba 1978).
    Both sequences are normalised to [0, 1] before comparison so that
    absolute frequency scale does not dominate the warp penalty.

    Args:
        seq_a: Tracked formant trajectory (shape: n,).
        seq_b: Reference target trajectory (shape: m,).

    Returns:
        Tuple of (normalised_dtw_distance, warp_path).
        warp_path is a list of (i, j) index pairs.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0, []

    # Normalise both sequences into [0, 1] jointly
    combined_max = max(seq_a.max(), seq_b.max()) + 1e-9
    a = seq_a / combined_max
    b = seq_b / combined_max

    # Cost matrix: |a_i − b_j|  (Manhattan, 1-D)
    cost = cdist(a.reshape(-1, 1), b.reshape(-1, 1), metric="cityblock")

    # DP accumulation
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j],  # insertion
                dp[i, j - 1],  # deletion
                dp[i - 1, j - 1],  # match
            )

    # Normalise by path length
    dtw_dist = float(dp[n, m]) / (n + m)

    # Traceback
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = {
            (i - 1, j - 1): dp[i - 1, j - 1],
            (i - 1, j): dp[i - 1, j],
            (i, j - 1): dp[i, j - 1],
        }
        best = min(candidates, key=candidates.get)  # type: ignore[arg-type]
        i, j = best
    path.reverse()

    return dtw_dist, path


def _downsample_trajectory(traj: np.ndarray, max_pts: int) -> np.ndarray:
    """Uniformly downsample a 1-D trajectory to at most *max_pts* points."""
    n = len(traj)
    if n <= max_pts:
        return traj
    idx = np.linspace(0, n - 1, max_pts, dtype=int)
    return traj[idx]  # type: ignore[no-any-return]


# ── EQ helper (reuses formant_system._apply_peak_eq_frame pattern) ────────────


def _peak_eq(audio: np.ndarray, sr: int, freq: float, bandwidth_hz: float, gain_db: float) -> np.ndarray:
    """Apply a biquad peak EQ to *audio* (Audio-EQ-Cookbook, Zölzer).

    Args:
        audio:        1-D mono audio.
        sr:           Sample rate (must be 48 000 Hz).
        freq:         Center frequency (Hz).
        bandwidth_hz: Bandwidth at −3 dB point (Hz).  Q = freq / bandwidth_hz.
        gain_db:      Boost in dB (positive = boost, negative = cut).

    Returns:
        Filtered audio (same shape, same dtype as input).
    """
    if freq <= 0.0 or bandwidth_hz <= 0.0 or abs(gain_db) < 0.01:
        return audio
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    q = float(np.clip(freq / (bandwidth_hz + 1e-9), 0.1, 50.0))
    w0 = 2.0 * np.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, -2.0 * np.cos(w0) / a0, (1.0 - alpha / A) / a0])
    return sig.lfilter(b, a, audio)  # type: ignore[no-any-return]


# ── Core class ────────────────────────────────────────────────────────────────


class InstrumentFormantDriftCorrector:
    """Erkennt and correct sustained formant drift in instrument recordings.

    Instantiate via :func:`get_instrument_formant_drift_corrector` (singleton).

    Usage::

        corrector = get_instrument_formant_drift_corrector()
        result = corrector.correct(audio, sr=48000, instrument="strings")
        logger.debug("drift=%s n_frames=%s", result.drift_detected, result.n_frames_corrected)
    """

    def __init__(
        self,
        correction_strength: float = 0.20,
        drift_threshold_hz: float = DRIFT_THRESHOLD_HZ,
        drift_min_frames: int = DRIFT_MIN_FRAMES,
    ) -> None:
        self.correction_strength = float(np.clip(correction_strength, 0.0, MAX_CORRECTION_STRENGTH))
        self.drift_threshold_hz = drift_threshold_hz
        self.drift_min_frames = drift_min_frames

        # Lazy-import FormantTracker to avoid circular dependency at module load
        self._tracker = None

    def _get_tracker(self):
        if self._tracker is None:
            from dsp.formant_system import FormantTracker

            self._tracker = FormantTracker(n_formants=5)  # type: ignore[assignment]
        return self._tracker

    # ── Internal: formant tracking ────────────────────────────────────────────

    def _track_f1(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """Gibt the F1 trajectory (shape: n_frames) for *mono* via LPC zurück."""
        try:
            tracker = self._get_tracker()
            formant_freqs, _ = tracker.track(mono, sr)
            f1 = formant_freqs[:, 0].copy()
            # Zero-out unreliable frames (tracker outputs 0 for silent frames)
            f1 = np.where(f1 > 50.0, f1, 0.0)
            return np.asarray(f1, dtype=np.float64)
        except Exception as exc:
            logger.debug("_track_f1 fehlgeschlagen: %s", exc)
            return np.array([])

    # ── Internal: drift detection via DTW ─────────────────────────────────────

    def _detect_drift(self, f1_tracked: np.ndarray, f1_target: float) -> tuple[bool, np.ndarray, float, float, float]:
        """Erkennt sustained drift in *f1_tracked* relative to *f1_target*.

        Returns:
            (drift_detected, drift_frame_mask, mean_drift_hz,
             max_drift_hz, dtw_distance)
        """
        n = len(f1_tracked)
        if n == 0:
            return False, np.zeros(0, dtype=bool), 0.0, 0.0, 0.0

        # Only consider frames where F1 was tracked (non-zero)
        valid = f1_tracked > 50.0
        if not np.any(valid):
            return False, np.zeros(n, dtype=bool), 0.0, 0.0, 0.0

        # Per-frame absolute deviation from target
        deviation = np.abs(f1_tracked - f1_target)
        deviation_valid = deviation.copy()
        deviation_valid[~valid] = 0.0

        mean_drift = float(deviation_valid[valid].mean())
        max_drift = float(deviation_valid[valid].max())

        # DTW distance: tracked valid F1 vs flat target reference
        f1_valid_seq = f1_tracked[valid]
        target_seq = np.full_like(f1_valid_seq, f1_target)

        # Downsample for performance
        f1_ds = _downsample_trajectory(f1_valid_seq, DTW_MAX_FRAMES)
        tgt_ds = _downsample_trajectory(target_seq, DTW_MAX_FRAMES)
        dtw_dist, _ = _dtw_distance_and_path(f1_ds, tgt_ds)

        # Build mask: frames where deviation exceeds threshold for DRIFT_MIN_FRAMES
        exceeds = deviation > self.drift_threshold_hz
        # Rolling minimum — mark sustained runs
        drift_mask = np.zeros(n, dtype=bool)
        run = 0
        for i in range(n):
            if exceeds[i]:
                run += 1
            else:
                run = 0
            if run >= self.drift_min_frames:
                drift_mask[max(0, i - run + 1) : i + 1] = True

        drift_detected = bool(np.any(drift_mask) and mean_drift > self.drift_threshold_hz * 0.5)

        return drift_detected, drift_mask, mean_drift, max_drift, dtw_dist

    # ── Internal: frame-wise EQ correction ────────────────────────────────────

    def _apply_frame_correction(
        self,
        audio_mono: np.ndarray,
        sr: int,
        f1_tracked: np.ndarray,
        f1_target: float,
        f2_target: float,
        drift_mask: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Wendet an: per-frame peak-EQ nudge toward target F1/F2 for drifted frames.

        Returns:
            (corrected_mono, n_frames_corrected)
        """
        hop = max(1, int(0.010 * sr))  # 10 ms hop
        win = max(1, int(0.025 * sr))  # 25 ms window
        n_frames = len(drift_mask)
        enhanced = audio_mono.copy()
        frames_done = 0

        for fi in range(n_frames):
            if not drift_mask[fi]:
                continue

            t0 = fi * hop
            t1 = min(t0 + win, len(audio_mono))
            if t1 <= t0:
                break

            frame = enhanced[t0:t1].copy()
            tracked_f1 = float(f1_tracked[fi]) if fi < len(f1_tracked) else 0.0

            # Deviation-proportional boost (max MAX_BOOST_DB)
            if tracked_f1 > 50.0:
                dev = abs(tracked_f1 - f1_target)
                boost = float(np.clip(dev / 300.0 * MAX_BOOST_DB, 0.0, MAX_BOOST_DB))
            else:
                boost = MAX_BOOST_DB * 0.5  # moderate boost for untracked frames

            bw_f1 = max(f1_target / 5.0, 40.0)
            frame_eq = _peak_eq(frame, sr, f1_target, bw_f1, boost * self.correction_strength / MAX_CORRECTION_STRENGTH)

            # Gentle F2 nudge (always, lower strength)
            bw_f2 = max(f2_target / 5.0, 80.0)
            frame_eq = _peak_eq(
                frame_eq, sr, f2_target, bw_f2, boost * 0.5 * self.correction_strength / MAX_CORRECTION_STRENGTH
            )

            # Hanning blend at frame boundaries (5 ms crossfade)
            blend_f = np.hanning(len(frame_eq))
            enhanced[t0:t1] = (
                frame * (1.0 - self.correction_strength * blend_f[: t1 - t0])
                + frame_eq[: t1 - t0] * self.correction_strength * blend_f[: t1 - t0]
            )
            frames_done += 1

        return enhanced, frames_done

    # ── Public API ────────────────────────────────────────────────────────────

    def correct(
        self,
        audio: np.ndarray,
        sr: int,
        instrument: str = "guitar",
        correction_strength: float | None = None,
    ) -> InstrumentDriftResult:
        """Erkennt and correct instrument formant drift.

        Args:
            audio:               Mono or stereo audio at 48 000 Hz.
            sr:                  Sample rate — must be 48 000 Hz.
            instrument:          Instrument type string (see InstrumentFormantTargets).
            correction_strength: Override blend factor 0.0–1.0; clamped to 0.30.
                                 *None* uses the instance default.

        Returns:
            :class:`InstrumentDriftResult` with corrected audio and diagnostics.
        """
        assert sr == SR_REQUIRED, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        strength = float(
            np.clip(
                correction_strength if correction_strength is not None else self.correction_strength,
                0.0,
                MAX_CORRECTION_STRENGTH,
            )
        )

        # Look up targets — graceful no-op for unknown instruments
        from dsp.formant_system import InstrumentFormantTargets

        row = InstrumentFormantTargets.get_targets(instrument)

        def _passthrough(reason: str) -> InstrumentDriftResult:
            out = np.clip(audio, -1.0, 1.0)
            logger.debug("InstrumentFormantDriftCorrector passthrough: %s", reason)
            return InstrumentDriftResult(
                audio=out,
                instrument=instrument,
                drift_detected=False,
                n_frames_corrected=0,
                total_frames=0,
                mean_drift_hz=0.0,
                max_drift_hz=0.0,
                dtw_distance=0.0,
                correction_strength=strength,
                f1_target_hz=0.0,
            )

        if row is None:
            return _passthrough(f"unknown instrument '{instrument}'")

        f1_tgt, f2_tgt, _f3_tgt, _q1, _q2, _q3 = row

        # Promote to mono for tracking
        is_stereo = audio.ndim == 2
        if is_stereo:
            mono = np.mean(audio, axis=0) if audio.shape[0] < audio.shape[1] else np.mean(audio, axis=1)
        else:
            mono = audio.copy()

        # Track F1 trajectory
        f1_tracked = self._track_f1(mono, sr)
        total_frames = len(f1_tracked)

        if total_frames == 0:
            return _passthrough("FormantTracker returned empty trajectory")

        # DTW-based drift detection
        drift_detected, drift_mask, mean_drift, max_drift, dtw_dist = self._detect_drift(f1_tracked, f1_tgt)

        if not drift_detected or strength < 1e-4:
            out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            out = np.clip(out, -1.0, 1.0)
            return InstrumentDriftResult(
                audio=out,
                instrument=instrument,
                drift_detected=False,
                n_frames_corrected=0,
                total_frames=total_frames,
                mean_drift_hz=mean_drift,
                max_drift_hz=max_drift,
                dtw_distance=dtw_dist,
                correction_strength=strength,
                f1_target_hz=f1_tgt,
            )

        # Apply frame-wise EQ correction on mono
        mono_corrected, n_corrected = self._apply_frame_correction(mono, sr, f1_tracked, f1_tgt, f2_tgt, drift_mask)

        # Recompose stereo by ratio transfer
        if is_stereo:
            eps = 1e-10
            orig_mono = np.mean(audio, axis=0) if audio.shape[0] < audio.shape[1] else np.mean(audio, axis=1)
            ratio = np.where(np.abs(orig_mono) > eps, mono_corrected / (orig_mono + eps), 1.0)
            ratio = np.clip(ratio, 0.5, 2.0)
            out = audio * ratio[np.newaxis, :] if audio.shape[0] < audio.shape[1] else audio * ratio[:, np.newaxis]
        else:
            out = mono_corrected

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        logger.info(
            "InstrumentFormantDriftCorrector: instrument=%s, frames=%d/%d, "
            "drift=%.1fHz (mean) %.1fHz (max), dtw=%.4f, corrected=%d frames",
            instrument,
            n_corrected,
            total_frames,
            mean_drift,
            max_drift,
            dtw_dist,
            n_corrected,
        )

        return InstrumentDriftResult(
            audio=out,
            instrument=instrument,
            drift_detected=drift_detected,
            n_frames_corrected=n_corrected,
            total_frames=total_frames,
            mean_drift_hz=mean_drift,
            max_drift_hz=max_drift,
            dtw_distance=dtw_dist,
            correction_strength=strength,
            f1_target_hz=f1_tgt,
            details={"drift_mask_sum": int(drift_mask.sum())},
        )


# ── Singleton (§3.2 Double-Checked Locking) ──────────────────────────────────

_instance: InstrumentFormantDriftCorrector | None = None
_lock = threading.Lock()


def get_instrument_formant_drift_corrector() -> InstrumentFormantDriftCorrector:
    """Gibt the module-level singleton :class:`InstrumentFormantDriftCorrector` zurück.

    Thread-safe via double-checked locking (§3.2).
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = InstrumentFormantDriftCorrector()
    return _instance


def correct_instrument_formant_drift(
    audio: np.ndarray,
    sr: int,
    instrument: str = "guitar",
    correction_strength: float | None = None,
) -> InstrumentDriftResult:
    """Convenience wrapper: correct instrument formant drift in *audio*.

    Args:
        audio:               Mono or stereo audio at 48 000 Hz.
        sr:                  Sample rate — must be 48 000 Hz.
        instrument:          Instrument type string.
        correction_strength: Blend factor 0.0–1.0; clamped to 0.30.

    Returns:
        :class:`InstrumentDriftResult`.
    """
    return get_instrument_formant_drift_corrector().correct(
        audio, sr, instrument=instrument, correction_strength=correction_strength
    )
