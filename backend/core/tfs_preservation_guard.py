"""Temporal Fine Structure (TFS) Preservation Guard.

Measures and guards the temporal fine structure fidelity of audio signals
across restoration phases.  TFS encodes pitch, binaural localisation cues
and consonant texture information below ~1.5 kHz — a perceptual dimension
that is invisible to envelope-based metrics like MDEM (400 ms) and
ArticulationMetric (attack-time only).

Scientific basis:
- Moore, B.C.J. (2008). "The Role of TFS Processing in Pitch Perception,
  Masking, and Speech Perception for Normal-Hearing and Hearing-Impaired
  People". *JARO* 9(4), 399-416.
- Lorenzi, C., Gilbert, G., Carn, H., Garnier, S. & Moore, B.C.J. (2006).
  "Speech perception problems of the hearing-impaired reflect inability to use
  TFS information". *PNAS* 103(49), 18866-18869.
- Hilbert analytic signal: Marple, S.L. (1999). "Computing the Discrete-Time
  Analytic Signal via FFT". *IEEE Trans. Signal Process.* 47(9), 2600-2603.

The guard operates per ERB (Equivalent Rectangular Bandwidth) band in the
range 100 Hz – 1.5 kHz, where TFS is perceptually critical.  Above ~1.5 kHz
the auditory system increasingly relies on the envelope rather than TFS
(Moore 2008, Fig. 2).

Module invariants (§3.x compliant):
- Thread-safe singleton via double-checked locking
- NaN/Inf guard on all numeric outputs
- No audio modification — measurement/annotation only
- No sample-rate assertion (analysis utility; works at any SR)
- English docstrings and log messages
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from backend.core.audio_utils import safe_filtfilt  # §v10.101

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ERB centre-frequency grid (Moore & Glasberg 1990)
# ---------------------------------------------------------------------------


def _erb_hz(f_hz: float) -> float:
    """Equivalent Rectangular Bandwidth at frequency *f_hz* (Glasberg & Moore 1990).

    Formula: ERB(f) = 24.7 * (4.37 * f/1000 + 1)
    """
    return 24.7 * (4.37 * f_hz / 1000.0 + 1.0)


def _erb_centre_frequencies(
    f_low: float = 100.0,
    f_high: float = 1500.0,
    n_bands: int = 12,
) -> np.ndarray:
    """Gibt *n_bands* centre frequencies uniformly spaced on the ERB-rate scale zurück.

    ERB-rate N(f) = 21.4 * log10(0.00437*f + 1)  (Glasberg & Moore 1990)
    """
    n_low = 21.4 * np.log10(0.00437 * f_low + 1.0)
    n_high = 21.4 * np.log10(0.00437 * f_high + 1.0)
    n_vals = np.linspace(n_low, n_high, n_bands)
    # Inverse: f = (10^(N/21.4) - 1) / 0.00437
    return (10.0 ** (n_vals / 21.4) - 1.0) / 0.00437  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TFSBandResult:
    """TFS coherence result for a single ERB band."""

    centre_freq_hz: float
    erb_width_hz: float
    tfs_coherence: float  # 0.0–1.0, higher = better preservation
    n_voiced_frames: int  # frames where carrier had sufficient energy


@dataclass
class TFSResult:
    """Aggregate TFS fidelity result across all measured ERB bands."""

    band_results: list[TFSBandResult] = field(default_factory=list)
    mean_coherence: float = 0.0  # mean across all bands
    min_coherence: float = 0.0  # worst band
    n_bands: int = 0
    passes_threshold: bool = True  # True if mean_coherence >= threshold


# ---------------------------------------------------------------------------
# Core guard
# ---------------------------------------------------------------------------


class TFSPreservationGuard:
    """Misst temporal fine structure fidelity between original and processed audio.

    Algorithm:
    1.  For each ERB-spaced centre frequency f_c in [100 Hz, 1500 Hz]:
        a.  Bandpass filter both signals ±0.5 ERB around f_c
            (4th-order Butterworth; zero-phase via filtfilt)
        b.  Extract instantaneous phase via Hilbert analytic signal
        c.  Compute inter-signal phase coherence (circular mean of
            e^{j(φ_orig − φ_rest)}) only on voiced frames (energy
            above -40 dBFS in the band)
    2.  Report per-band and aggregate coherence.

    Complexity: ~O(n_bands × N log N) — typically < 0.5 s for 60 s mono @ 48 kHz.
    """

    DEFAULT_THRESHOLD = 0.85
    _N_BANDS = 12
    _F_LOW = 100.0
    _F_HIGH = 1500.0
    _ENERGY_FLOOR_DB = -40.0  # only measure TFS on frames louder than this
    _FRAME_SAMPLES = 2048  # ~43 ms @ 48 kHz — captures TFS detail

    def measure(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> TFSResult:
        """Misst TFS coherence between *original* and *restored* audio.

        Parameters
        ----------
        original, restored : np.ndarray
            Audio signals (mono or stereo, same length and sample rate).
        sr : int
            Sample rate in Hz.
        threshold : float
            Pass/fail threshold for mean_coherence (default 0.85).

        Returns
        -------
        TFSResult with per-band coherence values.
        """
        from scipy.signal import butter, filtfilt, hilbert

        orig_m = self._to_mono_f64(original)
        rest_m = self._to_mono_f64(restored)

        # Align lengths
        min_len = min(len(orig_m), len(rest_m))
        if min_len < self._FRAME_SAMPLES * 2:
            logger.debug("TFS: audio too short (%d samples), returning perfect coherence", min_len)
            return TFSResult(
                band_results=[],
                mean_coherence=1.0,
                min_coherence=1.0,
                n_bands=0,
                passes_threshold=True,
            )
        orig_m = orig_m[:min_len]
        rest_m = rest_m[:min_len]

        centres = _erb_centre_frequencies(self._F_LOW, self._F_HIGH, self._N_BANDS)
        nyquist = sr / 2.0

        band_results: list[TFSBandResult] = []

        for fc in centres:
            erb_w = _erb_hz(fc)
            f_lo = max(20.0, fc - 0.5 * erb_w)
            f_hi = min(nyquist - 1.0, fc + 0.5 * erb_w)

            if f_hi <= f_lo or f_hi >= nyquist:
                continue

            # 4th-order Butterworth bandpass (zero-phase)
            try:
                b, a = butter(4, [f_lo / nyquist, f_hi / nyquist], btype="band")
                orig_band = safe_filtfilt(b, a, orig_m)
            except Exception:
                logger.debug("TFS: filter failed for fc=%.0f Hz, skipping", fc)
                continue

            # Frame-wise energy and coherence (only voiced frames)
            n_frames = max(1, min_len // self._FRAME_SAMPLES)
            trim_len = n_frames * self._FRAME_SAMPLES
            energy_floor_lin = 10.0 ** (self._ENERGY_FLOOR_DB / 20.0)

            band_frames = orig_band[:trim_len].reshape(n_frames, self._FRAME_SAMPLES)
            frame_peaks = np.max(np.abs(band_frames), axis=1)
            valid_peak_mask = np.isfinite(frame_peaks) & (frame_peaks > 0.0)
            frame_rms = np.zeros(n_frames, dtype=np.float64)
            if np.any(valid_peak_mask):
                norm_frames = band_frames[valid_peak_mask] / frame_peaks[valid_peak_mask, None]
                frame_rms[valid_peak_mask] = frame_peaks[valid_peak_mask] * np.sqrt(
                    np.mean(norm_frames * norm_frames, axis=1) + 1e-15
                )

            voiced_mask = frame_rms >= energy_floor_lin
            n_voiced = int(np.count_nonzero(voiced_mask))

            if n_voiced < 3:
                # Not enough voiced frames to measure TFS — skip band
                # (do NOT assume perfect coherence; exclude from aggregate)
                continue

            try:
                rest_band = safe_filtfilt(b, a, rest_m)
            except Exception:
                logger.debug("TFS: restored-band filter failed for fc=%.0f Hz, skipping", fc)
                continue

            # Instantaneous phase via Hilbert analytic signal.
            # Run only for bands that passed the voiced-energy gate.
            orig_phase = np.angle(hilbert(orig_band))
            rest_phase = np.angle(hilbert(rest_band))

            # Circular coherence: |mean(e^{j·Δφ})|
            all_diffs = (orig_phase[:trim_len] - rest_phase[:trim_len]).reshape(n_frames, self._FRAME_SAMPLES)[
                voiced_mask
            ]
            coherence = float(np.abs(np.mean(np.exp(1j * all_diffs))))
            coherence = float(np.nan_to_num(np.clip(coherence, 0.0, 1.0), nan=1.0))

            band_results.append(
                TFSBandResult(
                    centre_freq_hz=float(fc),
                    erb_width_hz=float(erb_w),
                    tfs_coherence=coherence,
                    n_voiced_frames=n_voiced,
                )
            )

        if not band_results:
            return TFSResult(
                band_results=[],
                mean_coherence=1.0,
                min_coherence=1.0,
                n_bands=0,
                passes_threshold=True,
            )

        coherences = [br.tfs_coherence for br in band_results]
        mean_c = float(np.mean(coherences))
        min_c = float(np.min(coherences))

        result = TFSResult(
            band_results=band_results,
            mean_coherence=float(np.nan_to_num(mean_c, nan=1.0)),
            min_coherence=float(np.nan_to_num(min_c, nan=1.0)),
            n_bands=len(band_results),
            passes_threshold=mean_c >= threshold,
        )

        logger.info(
            "TFS Guard: %d bands, mean_coherence=%.4f, min=%.4f, passes=%s",
            result.n_bands,
            result.mean_coherence,
            result.min_coherence,
            result.passes_threshold,
        )
        for br in band_results:
            logger.debug(
                "  TFS band fc=%.0f Hz (ERB %.1f Hz): coherence=%.4f (%d voiced frames)",
                br.centre_freq_hz,
                br.erb_width_hz,
                br.tfs_coherence,
                br.n_voiced_frames,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono_f64(audio: np.ndarray) -> np.ndarray:
        """Konvertiert to mono float64 with NaN/Inf guard."""
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] > arr.shape[1] else arr.mean(axis=1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Thread-safe singleton (double-checked locking — §3.2)
# ---------------------------------------------------------------------------

_instance: TFSPreservationGuard | None = None
_lock = threading.Lock()


def get_tfs_preservation_guard() -> TFSPreservationGuard:
    """Gibt thread-safe singleton TFSPreservationGuard zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TFSPreservationGuard()
    return _instance
