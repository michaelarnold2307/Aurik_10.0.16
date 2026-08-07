"""
§2.60 Stereo Temporal Coherence Guard (STCG)
============================================

Prevents temporal misalignment between:
  1. L and R stereo channels (inter-channel delay, e.g. from independent-channel DSP)
  2. Vocal stem and instrumental stem during phase_42 recombination (processing-chain latency)

Two public entry points:
  - correct_interchannel_delay(audio, sr, phase_id) → aligned audio
  - align_stem_to_reference(processed_stem, original_stem, sr, stem_label) → latency-compensated stem

Algorithm (SOTA §v10.13):
  - Normalized time-domain cross-correlation via scipy.signal.correlate (Pearson r)
    on a 10-second mid-song window (FFT-accelerated, <10ms for 48kHz 10s audio)
  - Parabolic interpolation of correlation peak for sub-sample precision (Smith 2011)
  - scipy.ndimage.shift with cubic spline interpolation for the actual sub-sample correction
    (numerically stable, avoids Lagrange FIR boundary artefacts for |frac| ≤ 0.5)

Performance budget: ≤ 15ms per call for songs up to 10 min @ 48kHz.

§0 Minimal-Intervention: no correction when |delay| < 0.5 samples.
§2.51: All corrections are applied to both channels symmetrically (linked-stereo),
       not independently per channel.

Singleton: get_stereo_temporal_coherence_guard()

References:
  - Smith, J.O. (2011). Spectral Audio Signal Processing. CCRMA.
  - Laakso et al. (1996). Splitting the Unit Delay. IEEE Signal Processing Magazine 13(1), 30–60.
  - Knapp & Carter (1976). The Generalized Correlation Method. IEEE TASLP 24(4), 320–327.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from scipy.ndimage import shift as _ndimage_shift
from scipy.signal import correlate as _signal_correlate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Analysis window: 10 s from the mid-song (480 000 samples @ 48 kHz)
_ANALYSIS_WINDOW_S: float = 10.0

# Maximum inter-channel delay to search for: ±200 ms (9 600 samples @ 48 kHz)
# Covers tape-head azimuth errors, ML-plugin output latency, analogue playback chains,
# and pipeline-introduced phase-vocoder/PSOLA latency (observed: 8 777 samples = 182.9 ms).
_MAX_DELAY_SAMPLES: int = 9_600

# Correction threshold: delays ≤ this are left untouched (§0 Minimal-Intervention).
_CORRECTION_THRESHOLD_SAMPLES: float = 0.5

# scipy.ndimage.shift interpolation order (3 = cubic spline, good quality/speed balance)
_INTERP_ORDER: int = 3

# §v10.14: Global maximum plausible inter-channel delay.
# Any measured delay > 20 ms is physically impossible for a commercial stereo
# recording. MP3 joint-stereo encoding creates phase relationships that cross-
# correlation misinterprets as time delay — these false positives are often
# consistent across measurement positions (tight spread) but the magnitude
# (50–200 ms) is physically implausible. This guard applies to ALL callers
# (pre_pipeline, post_pipeline, intra-phase) and replaces the earlier
# per-caller limits (20 ms pre, 200 ms post).
#
# Only hardware errors (tape-head azimuth < 1 ms, ADC clock drift < 0.1 ms)
# produce genuine inter-channel delays on properly mastered stereo audio.
# Pipeline-introduced delays > 20 ms indicate a phase bug, not something
# STCG should silently "fix" by shifting a channel a quarter-second.
_GLOBAL_MAX_MS: float = 20.0


# ---------------------------------------------------------------------------
# Core DSP helpers
# ---------------------------------------------------------------------------


def _to_mono_analysis(audio: np.ndarray) -> np.ndarray:
    """Collapse audio to a 1-D float32 mono array for correlation analysis.

    Accepts (N,), (2, N) channels-first, or (N, 2) channels-last.
    Returns a contiguous float32 array of shape (N,).
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr  # type: ignore[no-any-return]
    if arr.ndim == 2:
        # Detect orientation: channels-first (2, N) vs channels-last (N, 2)
        if arr.shape[0] == 2 and arr.shape[1] > 2:
            return ((arr[0] + arr[1]) * 0.5).astype(np.float32)  # type: ignore  # (2, N) → mono
        if arr.shape[1] == 2 and arr.shape[0] > 2:
            return ((arr[:, 0] + arr[:, 1]) * 0.5).astype(np.float32)  # type: ignore  # (N, 2) → mono
        # Fallback: sum along shorter axis
        return arr.mean(axis=0 if arr.shape[0] <= arr.shape[1] else 1)  # type: ignore
    # Unexpected rank — return first row/channel
    return arr.reshape(-1)[: arr.size // arr.shape[0]]  # type: ignore[no-any-return]


def _mid_window(signal: np.ndarray, sr: int) -> np.ndarray:
    """Extrahiert a 10-second window from the middle of *signal*."""
    n_window = int(_ANALYSIS_WINDOW_S * sr)
    n = min(len(signal), n_window)
    start = max(0, (len(signal) - n) // 2)
    return signal[start : start + n]


def _estimate_delay_subsample(ref: np.ndarray, target: np.ndarray, sr: int) -> float:
    """Schätzt the fractional-sample delay of *target* relative to *ref*.

    Convention:
      positive  → target is AHEAD of ref (target occurred earlier in time)
      negative  → target is BEHIND ref (target occurred later in time)

    Returns 0.0 when:
      - Signal too short (< 250 ms) for reliable estimation
      - SNR < 5.0 (uncorrelated or noisy signals, §G13 threshold)

    Algorithm (SOTA §v10.13):
      Normalized time-domain cross-correlation via scipy.signal.correlate
      (FFT-accelerated).  Unlike GCC-PHAT the normalisation uses signal
      standard deviation, not per-bin whitening, so the correlation peak
      has a physically meaningful [0,1] scale and does NOT amplify noise
      in low-energy frequency bins.  Parabolic sub-sample interpolation
      on the peak.
    """
    r_full = _to_mono_analysis(ref)
    t_full = _to_mono_analysis(target)

    r = _mid_window(r_full, sr)
    t = _mid_window(t_full, sr)

    n = min(len(r), len(t))
    if n < sr // 4:  # < 250 ms — not enough context
        return 0.0

    r = r[:n].astype(np.float64)
    t = t[:n].astype(np.float64)

    # Energy check — skip silence channels
    r_rms = float(np.sqrt(np.mean(r**2)))
    t_rms = float(np.sqrt(np.mean(t**2)))
    if r_rms < 1e-8 or t_rms < 1e-8:
        return 0.0

    # ── SOTA: Normalized time-domain cross-correlation ──
    # Mean-centre and normalise by std so the correlation peak is the
    # Pearson r coefficient [0,1].  PHAT whitening is NOT used because
    # it amplifies noise in low-energy bins and produces false positives
    # on short or band-limited windows.
    _l_ms = r - float(np.mean(r))
    _r_ms = t - float(np.mean(t))
    _l_std = float(np.std(_l_ms)) + 1e-12
    _r_std = float(np.std(_r_ms)) + 1e-12
    _corr = _signal_correlate(
        _l_ms / (_l_std * float(n)),
        _r_ms / _r_std,
        method="fft",
    )
    _center = len(r) - 1
    _max_lag = min(_MAX_DELAY_SAMPLES, _center)
    _lo = max(0, _center - _max_lag)
    _hi = min(len(_corr), _center + _max_lag + 1)
    _search = _corr[_lo:_hi]

    # Confidence gate (§G13 v10.13): Normalized XCorr peak is Pearson's r
    # coefficient [0,1].  Correlated L/R signals (music, speech, noise) give
    # peak ≥ 0.9; uncorrelated signals peak at ~0.005.  Threshold of 0.1
    # leaves a >10× safety margin against false positives while accepting
    # narrow-band signals (pure tones) where SNR-based gates fail.
    _peak = float(np.max(np.abs(_search)))
    if _peak < 0.1:
        logger.debug(
            "STCG: normalisiert-XCorr peak=%.4f < 0.1 — no correction",
            _peak,
        )
        return 0.0

    # Integer peak + parabolic sub-sample interpolation
    _peak_local_idx = int(np.argmax(np.abs(_search)))
    _global_peak = _lo + _peak_local_idx

    if 1 <= _global_peak < len(_corr) - 1:
        _y0 = _corr[_global_peak - 1]
        _y1 = _corr[_global_peak]
        _y2 = _corr[_global_peak + 1]
        _denom = 2.0 * (2.0 * _y1 - _y0 - _y2)
        _frac_offset = float((_y0 - _y2) / _denom) if abs(_denom) > 1e-12 else 0.0
        _frac_offset = float(np.clip(_frac_offset, -0.5, 0.5))
    else:
        _frac_offset = 0.0

    delay = float(_global_peak - _center) + _frac_offset
    return delay


def _apply_correction_shift(signal: np.ndarray, shift_samples: float) -> np.ndarray:
    """Wendet eine vorzeichenbehaftete fraktionale Sampleverzögerung auf ein 1-D-float32-Array an.

    Positive shift_samples → signal content moves to later indices (rightward, delay).
    Negative shift_samples → signal content moves to earlier indices (advance).

    Uses scipy.ndimage.shift with cubic spline interpolation (order=3).
    Boundary condition: mode='constant', cval=0.0 (silence at edges).

    Args:
        signal: 1-D float32 array.
        shift_samples: Fractional shift in samples (signed).

    Returns:
        Shifted signal, same length as input, clipped to [-1, 1].
    """
    if abs(shift_samples) < 1e-4:
        return signal.astype(np.float32)  # type: ignore[no-any-return]

    shifted = _ndimage_shift(
        signal.astype(np.float64),
        shift=shift_samples,
        mode="constant",
        cval=0.0,
        order=_INTERP_ORDER,
    )
    return np.asarray(  # type: ignore[no-any-return]
        np.clip(
            np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        ),
        dtype=np.float32,
    )


def _apply_shift_to_audio(audio: np.ndarray, shift_samples: float) -> np.ndarray:
    """Wendet an: fractional-sample shift to audio (mono or stereo, any orientation).

    For stereo, the SAME shift is applied to ALL channels (linked-stereo, §2.51):
    the inter-channel relationship is preserved; the whole signal is moved in time.

    Args:
        audio: 1-D, (2, N), or (N, 2) float32 array.
        shift_samples: Fractional shift in samples (positive = delay).

    Returns:
        Shifted audio, same shape and dtype as input.
    """
    if abs(shift_samples) < 1e-4:
        return audio

    orig_dtype = audio.dtype
    arr = np.asarray(audio, dtype=np.float32)

    if arr.ndim == 1:
        return _apply_correction_shift(arr, shift_samples).astype(orig_dtype)  # type: ignore[no-any-return]

    if arr.ndim == 2:
        channels_first = arr.shape[0] == 2 and arr.shape[1] > 2
        if channels_first:
            shifted_ch = np.vstack(
                [_apply_correction_shift(arr[i], shift_samples)[np.newaxis, :] for i in range(arr.shape[0])]
            )
        else:
            shifted_ch = np.column_stack(
                [_apply_correction_shift(arr[:, i], shift_samples) for i in range(arr.shape[1])]
            )
        return shifted_ch.astype(orig_dtype)  # type: ignore[no-any-return]

    # Unexpected rank — return unchanged
    logger.debug("STCG: unexpected audio rank %d — skipping shift", arr.ndim)
    return audio


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------


class StereoTemporalCoherenceGuard:
    """Stellt sicher: temporal coherence across stereo channels and between separated stems.

    Thread-safe. Instantiate via get_stereo_temporal_coherence_guard().

    Two responsibilities:
      1. correct_interchannel_delay  — detects and corrects L-R sample offset
                                       (call pre-pipeline and/or post-pipeline).
      2. align_stem_to_reference     — re-aligns a processed stem to the timing of
                                       its un-processed original (phase_42 stem remix).
    """

    _cumulative_correction_samples: float = 0.0
    _last_audio_id: int = 0

    # ------------------------------------------------------------------
    # 1. Inter-channel delay correction (L vs R)
    # ------------------------------------------------------------------

    def correct_interchannel_delay(
        self,
        audio: np.ndarray,
        sr: int,
        phase_id: str = "unknown",
    ) -> np.ndarray:
        """Erkennt and correct any L-R inter-channel temporal offset.

        The RIGHT channel is corrected to align with the LEFT channel as reference.
        LEFT channel is never modified (preserves mono-down-mix identity).

        §0 Minimal-Intervention: returns audio unchanged if |delay| < 0.5 samples.
        §2.51: Correction is applied only to R (linked-stereo, not independent).

        Args:
            audio: Audio array. Shape: mono (N,), channels-first (2, N),
                   or channels-last (N, 2). Mono → returned unchanged.
            sr:    Sample rate — must be 48000.
            phase_id: Label for logging (e.g. "pre_pipeline", "post_pipeline").

        Returns:
            Audio with corrected L-R alignment. Same shape and dtype as input.
        """
        assert sr == 48000, f"STCG requires sr=48000, got {sr}"

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 2:
            return audio  # Mono — no inter-channel alignment needed

        # Detect stereo orientation
        channels_first = arr.shape[0] == 2 and arr.shape[1] > 2
        if channels_first:
            ch_l = arr[0]
            ch_r = arr[1]
        elif arr.shape[1] == 2 and arr.shape[0] > 2:
            ch_l = arr[:, 0]
            ch_r = arr[:, 1]
        else:
            return audio  # Ambiguous shape — skip

        if len(ch_l) < sr // 4:
            return audio  # < 250 ms — too short to measure

        # §G13/F2 + §v10.14: Multi-point measurement is the primary lag source.
        # Single mid-window can miss temporally varying lags — multi-point median
        # captures the dominant characteristic. All corrections are gated by
        # _GLOBAL_MAX_MS (20 ms) regardless of caller; any measured delay
        # exceeding this is a cross-correlation false positive (MP3 joint-stereo
        # encoding phase artifacts), not a real hardware/alignment error.
        _mp_verified = self._verify_lag_multi_point(ch_l, ch_r, sr)
        if _mp_verified.get("num_points", 0) >= 2:
            delay = float(_mp_verified["median_lag"])
            _mp_spread = _mp_verified.get("max_spread", 0)
        else:
            # §v10.18 Single-point fallback: multi-point verification can fail on
            # narrow-bandwidth or MP3-encoded material where GCC-PHAT struggles
            # in sub-windows. Before giving up, try a single full-length measurement
            # with tighter guards: require correlation > 0.60 (not 0.40) and
            # lag < 10 ms (not 20 ms). This catches genuine hardware lags while
            # protecting against false positives from stereo-panning artifacts.
            _num_pts = _mp_verified.get("num_points", 0)
            try:
                _single_lag = _estimate_delay_subsample(
                    ch_l[: min(len(ch_l), sr * 30)].astype(np.float32),
                    ch_r[: min(len(ch_r), sr * 30)].astype(np.float32),
                    sr,
                )
                _single_corr = (
                    abs(
                        float(
                            np.corrcoef(
                                ch_l[: min(len(ch_l), sr * 10)],
                                ch_r[: min(len(ch_r), sr * 10)],
                            ).flat[1]
                        )
                    )
                    if len(ch_l) > 10
                    else 0.0
                )
                _single_corr = _single_corr if np.isfinite(_single_corr) else 0.0
                if _single_corr >= 0.60 and abs(_single_lag) < sr // 100:  # < 10 ms
                    delay = float(_single_lag)
                    _mp_spread = 0
                    logger.info(
                        "STCG [%s]: multi-point got %d point(s) — single-full-length "
                        "Ersatzpfad: lag=%d samples (%.1f ms), corr=%.3f",
                        phase_id,
                        _num_pts,
                        int(delay),
                        delay / sr * 1000,
                        _single_corr,
                    )
                else:
                    # SOTA: Keine Korrektur ohne verifizierbare Messung.
                    # Multi-point hatte 0 valide Punkte, Single-Fallback
                    # unterschreitet Korrelations-Schwelle oder überschreitet
                    # Lag-Limit. Audio wird unverändert zurückgegeben —
                    # "do no harm" vor blinder Korrektur.
                    logger.info(
                        "STCG [%s]: multi-point got %d point(s), single-Ersatzpfad "
                        "rejected (corr=%.3f < 0.60 or lag=%.1f ms > 10 ms) — skipping",
                        phase_id,
                        _num_pts,
                        _single_corr,
                        _single_lag / sr * 1000,
                    )
                    return audio
            except Exception:
                delay = 0.0
                _mp_spread = -1
                logger.warning(
                    "STCG [%s]: only %d measurement point(s) and single-Ersatzpfad fehlgeschlagen — skipping correction",
                    phase_id,
                    _num_pts,
                )
                return audio

        delay_ms = delay / sr * 1000.0

        if abs(delay) < _CORRECTION_THRESHOLD_SAMPLES:
            logger.debug(
                "STCG [%s]: delay=%.4f samples (%.3f ms) — within Schwelle, no correction",
                phase_id,
                delay,
                delay_ms,
            )
            return audio

        # §v10.14 UNIVERSAL plausibility guard: any delay > 20 ms is physically
        # impossible for a commercial stereo recording. MP3 joint-stereo encoding
        # creates phase artefacts that cross-correlation misinterprets as time
        # delay — these are often consistent (spread ≤ 20 samples) but the
        # magnitude is physically implausible. Skip correction unconditionally.
        # §v10.16 Stereo-Coherence-Guard: Inter-channel correlation must be
        # sufficient (> 0.4) for the delay measurement to be meaningful.
        # Uncorrelated channels (mean_corr < 0.3) contain different program
        # material (e.g. stereo-panned vocals, wide instrument placement).
        # Applying a time-shift to uncorrelated channels creates COMB FILTERING
        # that destroys bass frequencies and introduces distortion — this is
        # the root cause of "verschluckte Bässe" and "verzerrter Klang" in
        # previous runs.
        _mean_corr = (
            float(np.corrcoef(ch_l[: min(len(ch_l), sr * 10)], ch_r[: min(len(ch_r), sr * 10)]).flat[1])
            if len(ch_l) > 10
            else 1.0
        )
        _mean_corr = abs(_mean_corr) if np.isfinite(_mean_corr) else 1.0
        if _mean_corr < 0.40:
            logger.info(
                "STCG [%s]: inter-channel correlation=%.3f < 0.40 — "
                "channels contain different material (stereo panning), NOT a timing error — skipping",
                phase_id,
                _mean_corr,
            )
            return audio

        if abs(delay_ms) > _GLOBAL_MAX_MS:
            logger.info(
                "STCG [%s]: delay=%.1f ms > global limit %.0f ms (spread=%d) — "
                "physically implausible; likely MP3 joint-stereo artifact, skipping",
                phase_id,
                delay_ms,
                _GLOBAL_MAX_MS,
                int(_mp_spread),
            )
            return audio

        logger.info(
            "STCG [%s]: delay=%.4f samples (%.3f ms, spread=%d) — correcting R channel",
            phase_id,
            delay,
            delay_ms,
            int(_mp_spread),
        )

        # §v10.16 Cumulative correction: track total applied shift.
        # If we've already shifted > 5 ms cumulatively for this audio, skip.
        self._cumulative_correction_samples += abs(delay)
        _audio_id = hash(ch_l[:1024].tobytes()) if len(ch_l) >= 1024 else 0
        if _audio_id != self._last_audio_id:
            self._cumulative_correction_samples = abs(delay)
            self._last_audio_id = _audio_id
        if self._cumulative_correction_samples > 240:  # > 5 ms total
            logger.info(
                "STCG [%s]: cumulative correction %.1f ms exceeds 5 ms limit — "
                "skipping to prevent comb filtering from accumulated shifts",
                phase_id,
                self._cumulative_correction_samples / sr * 1000,
            )
            self._cumulative_correction_samples = 0.0
            return audio

        # Positive delay means R is AHEAD of L → shift R to the right (delay R) to align
        ch_r_corrected = _apply_correction_shift(ch_r, shift_samples=delay)

        orig_dtype = audio.dtype
        if channels_first:
            result = np.vstack([ch_l[np.newaxis, :], ch_r_corrected[np.newaxis, :]])
        else:
            result = np.column_stack([ch_l, ch_r_corrected])

        return result.astype(orig_dtype)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # 1a. Multi-point lag verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_lag_multi_point(
        ch_l: np.ndarray, ch_r: np.ndarray, sr: int, num_points: int = 3, window_s: float = 5.0
    ) -> dict:
        """Verify lag consistency across multiple song positions.

        A single mid-window GCC-PHAT measurement can produce false positives
        when stereo panning or decorrelation mimics a channel delay.
        Multi-point verification across the song ensures the lag is a genuine
        hardware/alignment artifact (consistent across positions), not a
        stereo-imaging artefact (varies by position).

        Returns dict with keys:
            verified:    True if lag is consistent (spread ≤ 20 samples) and
                         at least 2 of 3 points agree within 20 samples.
            median_lag:  Median lag across all valid measurement points.
            max_spread:  Max absolute difference between any two points.
            num_points:  Number of valid measurement points.
        """
        import numpy as np

        n = min(len(ch_l), len(ch_r))
        window_n = int(sr * window_s)
        if n < window_n * 2:
            return {"verified": False, "median_lag": 0, "max_spread": 9999, "num_points": 0}

        lags: list[int] = []
        for i in range(num_points):
            frac = (i + 1) / (num_points + 1)  # positions at 25%, 50%, 75%
            center = int(frac * n)
            start = max(0, center - window_n // 2)
            end = min(n, start + window_n)
            if end - start < sr // 4:
                continue
            try:
                lag = _estimate_delay_subsample(
                    ch_l[start:end].astype(np.float32),
                    ch_r[start:end].astype(np.float32),
                    sr,
                )
                lags.append(int(round(lag)))
            except Exception as _stcg_lag_exc:
                logger.debug(
                    "stereo_temporal_coherence_guard: multi-point lag measurement fehlgeschlagen (unkritisch): %s",
                    _stcg_lag_exc,
                )

        if len(lags) < 2:
            return {"verified": False, "median_lag": 0, "max_spread": 9999, "num_points": len(lags)}

        median_lag = int(np.median(lags))
        max_spread = max(abs(a - b) for a in lags for b in lags) if len(lags) > 1 else 9999

        # Verified: at least 2 points agree AND max spread ≤ 20 samples (~0.4ms @ 48kHz)
        points_near_median = sum(1 for lag in lags if abs(lag - median_lag) <= 20)
        verified = points_near_median >= 2 and max_spread <= 20

        return {
            "verified": verified,
            "median_lag": median_lag,
            "max_spread": max_spread,
            "num_points": len(lags),
            "lags": lags,
        }

    # ------------------------------------------------------------------
    # 2. Stem latency compensation (processed vs original)
    # ------------------------------------------------------------------

    def align_stem_to_reference(
        self,
        processed_stem: np.ndarray,
        original_stem: np.ndarray,
        sr: int,
        stem_label: str = "stem",
    ) -> np.ndarray:
        """Re-align *processed_stem* to the timing of *original_stem*.

        Used in phase_42 to compensate for any latency introduced by the vocal
        enhancement chain (STFT, ML inference, compression, formant EQ) before
        the enhanced vocals are mixed back with the instrumental stem.

        The original_stem is the un-processed stem (timing reference).
        The processed_stem is the same signal after enhancement/processing.

        §0 Minimal-Intervention: returns processed_stem unchanged if
        |delay| < 0.5 samples.

        Args:
            processed_stem: Enhanced/processed stem, any orientation.
            original_stem:  Original un-processed stem (timing reference).
            sr:             Sample rate — must be 48000.
            stem_label:     Label for logging.

        Returns:
            processed_stem aligned to original_stem's timing. Same shape as input.
        """
        assert sr == 48000, f"STCG requires sr=48000, got {sr}"

        mono_orig = _to_mono_analysis(original_stem)
        mono_proc = _to_mono_analysis(processed_stem)

        if len(mono_orig) < sr // 4 or len(mono_proc) < sr // 4:
            return processed_stem  # Too short

        # How many samples is processed AHEAD of original?
        # Positive = processed is ahead → shift processed to the right to align
        delay = _estimate_delay_subsample(mono_orig, mono_proc, sr)
        delay_ms = delay / sr * 1000.0

        if abs(delay) < _CORRECTION_THRESHOLD_SAMPLES:
            logger.debug(
                "STCG stem [%s]: delay=%.4f samples (%.3f ms) — within Schwelle, no correction",
                stem_label,
                delay,
                delay_ms,
            )
            return processed_stem

        logger.info(
            "STCG stem [%s]: processing latency=%.4f samples (%.3f ms) — compensating",
            stem_label,
            delay,
            delay_ms,
        )

        return _apply_shift_to_audio(processed_stem, shift_samples=delay)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_instance: StereoTemporalCoherenceGuard | None = None
_lock = threading.Lock()


def get_stereo_temporal_coherence_guard() -> StereoTemporalCoherenceGuard:
    """Gibt the process-wide StereoTemporalCoherenceGuard singleton zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StereoTemporalCoherenceGuard()
    return _instance
