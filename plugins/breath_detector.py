"""
BreathDetector – ZCR + Energy-based breath segment detection (§2.8).

Algorithm (per Aurik 10.0.0 spec §2.8):
    Kriterium: ZCR > ZCR_THRESHOLD AND energy < ENERGY_THRESHOLD_DBFS
    1. Short-time RMS energy in 25 ms frames (hop 10 ms)
    2. Zero-crossing rate in the same frames
    3. Frames that satisfy both criteria → breath candidates
    4. Adjacent breath frames with < GAP_MS gap werden zusammengeführt
    5. 5 ms Hanning crossfade at segment boundaries

Math:
    RMS_k   = sqrt(mean(frame_k²))
    E_k     = 20·log10(max(RMS_k, 1e-9))   [dBFS]
    ZCR_k   = (1/(N-1)) · Σ |sign(x[n]) - sign(x[n-1])| / 2
    breath_k = (ZCR_k > 0.30) AND (E_k < -38.0)

Invariants:
    - NaN/Inf-safe: np.nan_to_num on entry
    - Output breath_positions in samples, confidence ∈ [0, 1]
    - Pure DSP — no external ML model required
    - Thread-safe singleton via get_breath_detector()

References:
    §2.8 VocalAIEnhancement spec: BreathDetector criterion ZCR > 0.3, energy < -38 dBFS
    §3.1 Numerische Robustheit
    §3.2 Singleton-Pattern
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten (§2.8 Spec)
# ---------------------------------------------------------------------------
ZCR_THRESHOLD: float = 0.30  # Normalized ZCR per spec
ENERGY_THRESHOLD_DBFS: float = -38.0  # dBFS energy ceiling for breath
FRAME_SIZE_MS: float = 25.0  # Analysis frame length [ms]
HOP_SIZE_MS: float = 10.0  # Frame hop [ms]
MIN_BREATH_MS: float = 60.0  # Min breath segment length [ms]
MAX_BREATH_MS: float = 2000.0  # Max breath segment length [ms]
GAP_MERGE_MS: float = 30.0  # Max gap to merge adjacent breaths [ms]
CROSSFADE_MS: float = 5.0  # Hanning crossfade at boundaries [ms]
TARGET_SR: int = 48_000  # Internal processing sample rate


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class BreathDetectionResult:
    """Ergebnis der Atem-Erkennung.

    Attributes:
        breath_positions: Start-Sample-Indizes der Atemsegmente (48 kHz)
        breath_end_positions: End-Sample-Indizes (gepaart mit breath_positions)
        breath_durations_ms: Länge jedes Segments in Millisekunden
        confidence: Gesamt-Konfidenz ∈ [0, 1]
        n_frames_analyzed: Anzahl analysierter Kurzzeit-Frames
        energy_profile_db: RMS-Energie pro Frame in dBFS (für Visualisierung)
        zcr_profile: Normalisierte ZCR pro Frame (für Visualisierung)
    """

    breath_positions: list[int] = field(default_factory=list)
    breath_end_positions: list[int] = field(default_factory=list)
    breath_durations_ms: list[float] = field(default_factory=list)
    confidence: float = 0.0
    n_frames_analyzed: int = 0
    energy_profile_db: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    zcr_profile: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------
class BreathDetector:
    """ZCR + Energy-based breath segment detector for vocal audio (§2.8).

    Algorithm:
        1. Framing: 25 ms frames, 10 ms hop at 48 kHz
        2. RMS energy per frame → dBFS
        3. ZCR per frame (normalized to [0, 1])
        4. Breath criterion: zcr[k] > 0.30 AND energy_db[k] < -38 dBFS
        5. Merge short-gap detections (< 30 ms gap)
        6. Filter by min/max duration (60 ms – 2000 ms)
        7. 5 ms Hanning crossfade windows at boundaries

    Args:
        zcr_threshold: ZCR threshold (default 0.30 per §2.8)
        energy_threshold_dbfs: Energy ceiling in dBFS (default −38 dBFS)
        crossfade_ms: Crossfade window at segment edges (default 5 ms)

    Thread-safety: stateless per call; use get_breath_detector() singleton.
    """

    def __init__(
        self,
        zcr_threshold: float = ZCR_THRESHOLD,
        energy_threshold_dbfs: float = ENERGY_THRESHOLD_DBFS,
        crossfade_ms: float = CROSSFADE_MS,
    ) -> None:
        self._zcr_threshold = float(zcr_threshold)
        self._energy_threshold_dbfs = float(energy_threshold_dbfs)
        self._crossfade_ms = float(crossfade_ms)
        logger.debug(
            "BreathDetector initialisiert: zcr_Schwelle=%.2f, energy_dbfs=%.1f",
            self._zcr_threshold,
            self._energy_threshold_dbfs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> BreathDetectionResult:
        """Erkennt breath segments in vocal audio.

        Args:
            audio: float32/64 ndarray, mono or stereo (mixed to mono)
            sample_rate: Sample rate in Hz

        Returns:
            BreathDetectionResult with breath_positions, confidence, profiles.

        Raises:
            ValueError: If audio is empty or sample_rate < 8000.
        """
        if audio.size == 0:
            raise ValueError("audio must not be empty")
        if sample_rate < 8_000:
            raise ValueError(f"sample_rate must be >= 8000, got {sample_rate}")

        # NaN/Inf guard
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Mix to mono
        mono: npt.NDArray[np.float32] = audio if audio.ndim == 1 else audio.mean(axis=-1)

        # Resample to TARGET_SR if needed
        mono = self._resample_if_needed(mono, sample_rate, TARGET_SR)
        sr = TARGET_SR

        frame_size = int(FRAME_SIZE_MS * sr / 1000)
        hop_size = int(HOP_SIZE_MS * sr / 1000)

        energy_db, zcr = self._compute_features(mono, frame_size, hop_size)
        n_frames = len(energy_db)

        # Binary breath mask
        breath_mask: npt.NDArray[np.bool_] = (zcr > self._zcr_threshold) & (energy_db < self._energy_threshold_dbfs)

        segments = self._frames_to_sample_segments(breath_mask, hop_size, frame_size)

        gap_samples = int(GAP_MERGE_MS * sr / 1000)
        segments = self._merge_segments(segments, gap_samples)

        min_samp = int(MIN_BREATH_MS * sr / 1000)
        max_samp = int(MAX_BREATH_MS * sr / 1000)
        segments = [(s, e) for s, e in segments if min_samp <= (e - s) <= max_samp]

        # Confidence: breath frame fraction × ZCR margin
        confidence = 0.0
        if n_frames > 0 and breath_mask.any():
            breath_frac = float(breath_mask.sum()) / n_frames
            margin = float(np.mean(zcr[breath_mask] - self._zcr_threshold))
            confidence = float(np.clip(breath_frac * 3.0 + margin, 0.0, 1.0))

        result = BreathDetectionResult(
            breath_positions=[int(s) for s, _ in segments],
            breath_end_positions=[int(e) for _, e in segments],
            breath_durations_ms=[float((e - s) * 1000 / sr) for s, e in segments],
            confidence=confidence,
            n_frames_analyzed=n_frames,
            energy_profile_db=energy_db.astype(np.float32),
            zcr_profile=zcr.astype(np.float32),
        )

        logger.debug(
            "BreathDetector: %d segments, confidence=%.2f",
            len(segments),
            confidence,
        )
        return result

    def apply_crossfade_gate(
        self,
        audio: npt.NDArray[np.float32],
        result: BreathDetectionResult,
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """Wendet 5-ms-Hanning-Überblend-Gate an Atemsegmentgrenzen an (§2.8).

        Softens voiced ↔ breath transitions to prevent click artifacts.
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out = audio.copy()
        xfade = int(self._crossfade_ms * sr / 1000)
        if xfade < 2:
            return out
        window = np.hanning(2 * xfade).astype(np.float32)
        fade_in = window[:xfade]
        fade_out = window[xfade:]

        for start, end in zip(result.breath_positions, result.breath_end_positions):
            s_start = max(0, start - xfade)
            s_end = min(len(out), start)
            rlen = s_end - s_start
            if rlen > 0:
                out[s_start:s_end] *= fade_out[-rlen:]

            e_start = min(len(out), end)
            e_end = min(len(out), end + xfade)
            rlen = e_end - e_start
            if rlen > 0:
                out[e_start:e_end] *= fade_in[:rlen]

        return np.clip(out, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_features(
        mono: npt.NDArray[np.float32],
        frame_size: int,
        hop_size: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Berechnet RMS energy (dBFS) and ZCR per short-time frame."""
        n = len(mono)
        n_frames = max(1, 1 + (n - frame_size) // hop_size)
        energy_db = np.full(n_frames, -120.0, dtype=np.float32)
        zcr = np.zeros(n_frames, dtype=np.float32)

        for k in range(n_frames):
            start = k * hop_size
            frame = mono[start : start + frame_size]
            if len(frame) == 0:
                continue
            rms = float(np.sqrt(np.mean(frame**2)))
            energy_db[k] = 20.0 * math.log10(max(rms, 1e-9))
            signs = np.sign(frame)
            signs[signs == 0] = 1
            zcr[k] = float(np.sum(np.abs(np.diff(signs))) / 2) / max(len(frame) - 1, 1)

        return energy_db, zcr

    @staticmethod
    def _frames_to_sample_segments(
        mask: npt.NDArray[np.bool_],
        hop_size: int,
        frame_size: int,
    ) -> list[tuple[int, int]]:
        """Konvertiert boolean frame mask to (start_sample, end_sample) pairs."""
        segments: list[tuple[int, int]] = []
        in_seg = False
        seg_start = 0
        for k, val in enumerate(mask):
            if val and not in_seg:
                seg_start = k * hop_size
                in_seg = True
            elif not val and in_seg:
                segments.append((seg_start, k * hop_size + frame_size))
                in_seg = False
        if in_seg:
            segments.append((seg_start, len(mask) * hop_size + frame_size))
        return segments

    @staticmethod
    def _merge_segments(
        segments: list[tuple[int, int]],
        gap_samples: int,
    ) -> list[tuple[int, int]]:
        """Führt zusammen: adjacent segments separated by fewer than gap_samples."""
        if not segments:
            return []
        merged: list[tuple[int, int]] = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= gap_samples:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _resample_if_needed(
        audio: npt.NDArray[np.float32],
        src_sr: int,
        dst_sr: int,
    ) -> npt.NDArray[np.float32]:
        """Resample audio from src_sr to dst_sr (Lanczos-4 via scipy/resampy)."""
        if src_sr == dst_sr:
            return audio
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(dst_sr, src_sr)
            return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)  # type: ignore[no-any-return]
        except ImportError:
            # Minimal linear interpolation fallback
            orig_len = len(audio)
            new_len = int(orig_len * dst_sr / src_sr)
            return np.interp(
                np.linspace(0, orig_len - 1, new_len),
                np.arange(orig_len),
                audio,
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------
_instance: BreathDetector | None = None
_lock = threading.Lock()


def get_breath_detector() -> BreathDetector:
    """Thread-safe singleton (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BreathDetector()
    return _instance


def detect_breaths(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> BreathDetectionResult:
    """Convenience wrapper — Atemerkennung ohne Klassen-Instantiierung.

    Args:
        audio: float32 audio (mono or stereo)
        sample_rate: Sample rate in Hz

    Returns:
        BreathDetectionResult with breath_positions, durations, confidence.
    """
    return get_breath_detector().detect(audio, sample_rate)
