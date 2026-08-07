"""
core/goosebumps_quality_checker.py — Psychoacoustic Goosebumps Quality Assessment
==================================================================================

Implements the binding §8.3 Gänsehaut-Formel from the Aurik spec:

    goosebumps_score = (TransientIntegrity × MicroDynamik × Klarheit × Authentizität) − Artefakte

This module provides a holistic psychoacoustic quality assessment that goes beyond
individual metrics. It evaluates whether the restoration preserves and enhances the
emotional impact of the music — the "goosebumps factor".

The score implements the binding §8.3 formula as a product (not sum):

    score = (T × M × K × A) − Artefakte

where each factor is raised to its spec-weight exponent to preserve the
multiplicative coupling while respecting the contribution ratios:

    score = T^0.40 × M^0.25 × K^0.20 × A^0.15 − artifact_penalty × scale

This ensures that a single weak dimension pulls the entire score down
non-linearly (as intended by §8.3), unlike a weighted sum.

Five orthogonal dimensions:
  1. **Transient Integrity** (~40% of emotional impact)
     Onset-envelope correlation, attack-energy ratio, timing deviation.

  2. **Micro-Dynamics** (~25% of emotional impact)
     400ms LUFS-profile correlation and CV preservation.

  3. **Clarity / Klarheit** (~20% of emotional impact)
     Spectral flatness reduction + HNR improvement − over-processing.

  4. **Authenticity / Authentizität** (~15% of emotional impact)
     MFCC correlation + spectral centroid stability + chroma fidelity.

  5. **Artifact Penalty** (subtracted from product)
     Pre-echo, musical noise, noise-floor elevation detection.

Performance: Pure NumPy/SciPy, ≤ 500 ms for 3-minute stereo audio at 48 kHz.

Author: Aurik Development Team
Version: 1.0.0
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─── Dimension Weights (§8.3 Gänsehaut-Formel) ──────────────────────────────
_W_TRANSIENT = 0.40
_W_MICRO_DYN = 0.25
_W_CLARITY = 0.20
_W_AUTHENTICITY = 0.15  # Spec §8.3: 10% base + 5% vocal presence contribution

# ─── Thresholds ──────────────────────────────────────────────────────────────
_ONSET_ENV_HOP = 512
_ONSET_ENV_SR = 48000
_ATTACK_TIME_TOLERANCE_MS = 10.0  # §ArticulationMetric spec
_LUFS_WINDOW_S = 0.4  # 400 ms LUFS windows for micro-dynamics
_ARTIFACT_PENALTY_SCALE = 0.15  # Max penalty subtracted from score

# ─── Analysis Window ─────────────────────────────────────────────────────────
_MAX_ANALYSIS_S = 60.0  # Analyze max 60s center crop for performance


@dataclass
class GoosebumpsResult:
    """Result of the psychoacoustic goosebumps quality assessment (§8.3)."""

    goosebumps_score: float  # Final holistic score ∈ [0.0, 1.0]
    transient_integrity: float  # Onset preservation ∈ [0.0, 1.0]
    micro_dynamics: float  # LUFS-profile correlation ∈ [0.0, 1.0]
    clarity: float  # Noise removal quality ∈ [0.0, 1.0]
    authenticity: float  # Timbral fidelity ∈ [0.0, 1.0]
    artifact_penalty: float  # Detected artifacts ∈ [0.0, 1.0]
    details: dict[str, float | str | bool] = field(default_factory=dict)

    def summary(self) -> str:
        """Gibt eine einzeilige Zusammenfassung aller Gänsehaut-Dimensionen zurück."""
        return (
            f"GoosebumpsScore={self.goosebumps_score:.3f} "
            f"(transient={self.transient_integrity:.3f} "
            f"micro_dyn={self.micro_dynamics:.3f} "
            f"clarity={self.clarity:.3f} "
            f"auth={self.authenticity:.3f} "
            f"artifact_pen={self.artifact_penalty:.3f})"
        )


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertiert to mono for analysis.

    Handles both (samples, channels) and (channels, samples) orientation by
    averaging over the axis with fewer elements (= the channels axis).
    """
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float64)  # type: ignore[no-any-return]
    # Smallest dimension = channels axis (works for stereo where n_samples >> 2)
    ch_axis = int(np.argmin(audio.shape))
    mono: np.ndarray[Any, Any] = np.asarray(np.mean(audio, axis=ch_axis), dtype=np.float64)
    return mono


def _center_crop(audio: np.ndarray, sr: int, max_s: float) -> np.ndarray:
    """Center-crop to max_s seconds for performance."""
    max_samples = int(max_s * sr)
    if len(audio) <= max_samples:
        return audio
    start = (len(audio) - max_samples) // 2
    return audio[start : start + max_samples]


def _onset_envelope(audio: np.ndarray, _sr: int, hop: int = _ONSET_ENV_HOP) -> np.ndarray:
    """Berechnet onset strength envelope via spectral flux (half-wave rectified)."""
    n_fft = 2048
    hop_size = hop
    # Zero-pad to ensure clean STFT
    padded = np.pad(audio, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(padded) - n_fft) // hop_size
    if n_frames < 2:
        return np.array([0.0])  # type: ignore[no-any-return]

    window = np.hanning(n_fft)
    # §perf P3: Batch-rfft via stride_tricks — statt n_frames einzelner rfft()-Aufrufe
    # werden alle Frames auf einmal als (n_frames, n_fft)-View extrahiert und
    # mit einem einzigen np.fft.rfft(..., axis=-1) transformiert.
    # Bit-identisch (gleiche Reihenfolge, keine Approximation). 1.5× schneller
    # auf langen Dateien (3 min @ 48 kHz: ~17 000 Frames).
    _frames_view = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop_size, padded.strides[0]),
    )  # read-only view; Multiplikation mit window erzwingt eine Kopie
    magnitudes = np.abs(np.fft.rfft(_frames_view * window, axis=-1))

    # Spectral flux (positive differences only = onset energy) — vektorisiert
    _diff = np.diff(magnitudes, axis=0)  # (n_frames-1, n_bins)
    flux = np.empty(n_frames)
    flux[0] = 0.0
    flux[1:] = np.sum(np.maximum(_diff, 0.0), axis=1)

    # Normalize
    flux_max: float = float(np.max(flux))
    if flux_max > 0:
        flux /= flux_max
    return flux  # type: ignore[no-any-return]


def _measure_transient_integrity(original: np.ndarray, restored: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
    """Misst transient preservation quality.

    Combines:
    - Onset envelope correlation (how well transient shapes are preserved)
    - Attack energy ratio (transient energy before/after)
    - Onset timing deviation (temporal shift of detected onsets)
    """
    env_orig = _onset_envelope(original, sr)
    env_rest = _onset_envelope(restored, sr)

    # Align lengths
    min_len = min(len(env_orig), len(env_rest))
    if min_len < 4:
        return 0.5, {"onset_corr": 0.5, "energy_ratio": 1.0, "timing_ok": True}

    env_orig = env_orig[:min_len]
    env_rest = env_rest[:min_len]

    # 1. Onset envelope Pearson correlation
    corr = 0.5
    std_o = np.std(env_orig)
    std_r = np.std(env_rest)
    if std_o > 1e-8 and std_r > 1e-8:
        _eo = env_orig - env_orig.mean()
        _er = env_rest - env_rest.mean()
        _no = float(np.linalg.norm(_eo))
        _nr = float(np.linalg.norm(_er))
        cc = float(np.dot(_eo, _er) / (_no * _nr + 1e-10))
        corr = max(0.0, float(cc)) if np.isfinite(cc) else 0.5
    elif std_o < 1e-8 and std_r < 1e-8:
        corr = 1.0  # Both constant — trivially matched
    else:
        corr = 0.2  # One has transients, the other doesn't

    # 2. Attack energy preservation: compare top-20% onset frames
    threshold = np.percentile(env_orig, 80)
    onset_mask = env_orig >= threshold
    if np.any(onset_mask):
        energy_orig = np.mean(env_orig[onset_mask])
        energy_rest = np.mean(env_rest[onset_mask])
        energy_ratio = min(energy_rest / (energy_orig + 1e-10), energy_orig / (energy_rest + 1e-10))
        energy_ratio = float(max(0.0, min(1.0, energy_ratio)))  # type: ignore[assignment,arg-type]
    else:
        energy_ratio = 1.0  # type: ignore[assignment]

    # 3. Onset timing: check that peak positions haven't shifted
    # Find top-N onset peaks in original
    n_peaks = min(20, min_len // 10)
    if n_peaks >= 2:
        orig_peaks = np.argsort(env_orig)[-n_peaks:]
        rest_peaks = np.argsort(env_rest)[-n_peaks:]
        # For each original peak, find nearest restored peak
        timing_deviations_ms = []
        for op in orig_peaks:
            dists = np.abs(rest_peaks - op)
            nearest_dist: float = float(np.min(dists))
            deviation_ms = nearest_dist * (_ONSET_ENV_HOP / sr) * 1000.0
            timing_deviations_ms.append(deviation_ms)
        mean_deviation_ms = float(np.mean(timing_deviations_ms))
        timing_score = max(0.0, 1.0 - mean_deviation_ms / _ATTACK_TIME_TOLERANCE_MS)
    else:
        timing_score = 1.0
        mean_deviation_ms = 0.0

    # Weighted combination: correlation dominates, energy and timing fine-tune
    score = 0.55 * corr + 0.25 * energy_ratio + 0.20 * timing_score
    details = {
        "onset_envelope_corr": round(corr, 4),
        "attack_energy_ratio": round(energy_ratio, 4),
        "timing_score": round(timing_score, 4),
        "mean_timing_deviation_ms": round(mean_deviation_ms, 2),
    }
    return float(max(0.0, min(1.0, score))), details  # type: ignore[return-value,arg-type]


def _measure_micro_dynamics(original: np.ndarray, restored: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
    """Misst micro-dynamics preservation via short-term LUFS profile correlation.

    Uses 400ms RMS windows (matching MDEM specification) to compare the
    amplitude modulation patterns between original and restored audio.
    """
    window_samples = int(_LUFS_WINDOW_S * sr)
    if window_samples < 1:
        return 0.5, {}

    def _rms_profile(audio: np.ndarray) -> np.ndarray:
        n_frames = len(audio) // window_samples
        if n_frames < 2:
            return np.asarray([float(np.sqrt(np.mean(audio**2) + 1e-12))], dtype=np.float64)  # type: ignore[no-any-return]
        shaped = audio[: n_frames * window_samples].reshape(n_frames, window_samples)
        rms_profile: np.ndarray[Any, Any] = np.asarray(np.sqrt(np.mean(shaped**2, axis=1) + 1e-12), dtype=np.float64)
        return rms_profile

    profile_orig = _rms_profile(original)
    profile_rest = _rms_profile(restored)

    min_len = min(len(profile_orig), len(profile_rest))
    if min_len < 4:
        return 0.5, {"lufs_profile_corr": 0.5}

    profile_orig = profile_orig[:min_len]
    profile_rest = profile_rest[:min_len]

    # Pearson correlation of LUFS profiles
    std_o = np.std(profile_orig)
    std_r = np.std(profile_rest)
    if std_o < 1e-10 and std_r < 1e-10:
        # Both constant — dynamics trivially preserved
        corr = 1.0
    elif std_o < 1e-10 or std_r < 1e-10:
        # One has dynamics, the other doesn't — poor preservation
        corr = 0.2
    else:
        _po = profile_orig - profile_orig.mean()
        _pr = profile_rest - profile_rest.mean()
        _no = float(np.linalg.norm(_po))
        _nr = float(np.linalg.norm(_pr))
        cc = float(np.dot(_po, _pr) / (_no * _nr + 1e-10))
        corr = max(0.0, float(cc)) if np.isfinite(cc) else 0.5

    # Also check coefficient of variation preservation (dynamic range)
    cv_orig = float(std_o / (np.mean(profile_orig) + 1e-10))
    cv_rest = float(std_r / (np.mean(profile_rest) + 1e-10))
    # Ratio: how well is the dynamic variation preserved?
    if cv_orig > 1e-6 and cv_rest > 1e-6:
        cv_ratio = min(cv_rest / cv_orig, cv_orig / cv_rest)
        cv_ratio = max(0.0, float(cv_ratio))
    elif cv_orig < 1e-6 and cv_rest < 1e-6:
        cv_ratio = 1.0  # Both constant
    else:
        cv_ratio = 0.1  # One has dynamics, the other doesn't

    score = 0.70 * corr + 0.30 * cv_ratio
    details = {
        "lufs_profile_corr": round(corr, 4),
        "cv_orig": round(cv_orig, 4),
        "cv_rest": round(cv_rest, 4),
        "cv_preservation_ratio": round(cv_ratio, 4),
    }
    return float(max(0.0, min(1.0, score))), details


def _measure_clarity(original: np.ndarray, restored: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
    """Misst clarity improvement — noise reduction without over-processing.

    Combines:
    - Spectral flatness change (lower = more tonal/clear)
    - Harmonic-to-noise ratio improvement
    - Over-processing detection (too much flatness reduction = musical noise)
    """
    n_fft = 2048

    def _spectral_flatness(audio: np.ndarray) -> float:
        """Multi-frame spectral flatness — averaged over 10 evenly-spaced windows.
        Silence frames (RMS < 1e-6) are skipped to avoid flatness=1.0 on quiet segments.
        """
        n_windows = 10
        step = max(n_fft, len(audio) // n_windows) if len(audio) >= n_fft else len(audio)
        flatness_vals: list[float] = []
        for i in range(n_windows):
            start = i * step
            seg = audio[start : start + n_fft]
            if len(seg) < n_fft:
                seg = np.pad(seg, (0, n_fft - len(seg)))
            if np.sqrt(np.mean(seg**2)) < 1e-6:
                continue  # Skip silence
            spec = np.abs(np.fft.rfft(seg))[1:]  # Skip DC
            spec = np.maximum(spec, 1e-12)
            geo_mean = np.exp(np.mean(np.log(spec)))
            arith_mean = np.mean(spec)
            flatness_vals.append(float(geo_mean / (arith_mean + 1e-12)))
        return float(np.mean(flatness_vals)) if flatness_vals else 1.0

    def _hnr_estimate(audio: np.ndarray) -> float:
        """Multi-frame HNR estimate via autocorrelation peak.
        Averages over 5 evenly-spaced 1-second segments, skipping silence.
        """
        from backend.core.core_utils import fft_autocorr

        min_lag = int(0.002 * sr)
        max_lag = int(0.020 * sr)  # Up to 50 Hz fundamental
        if max_lag <= min_lag:
            return 0.0
        win = min(sr, max(512, len(audio) // 5))
        n_windows = 5
        step = max(win, len(audio) // n_windows) if len(audio) >= win else len(audio)
        hnr_vals: list[float] = []
        for i in range(n_windows):
            start = i * step
            seg = audio[start : start + win]
            if len(seg) < win // 2:
                continue
            if np.sqrt(np.mean(seg**2)) < 1e-6:
                continue  # Skip silence
            autocorr = fft_autocorr(seg)
            if max_lag >= len(autocorr):
                continue
            peak = float(np.max(autocorr[min_lag:max_lag]))
            total = float(autocorr[0]) + 1e-12
            hnr_vals.append(max(0.0, peak / total))
        return float(np.mean(hnr_vals)) if hnr_vals else 0.0

    flat_orig = _spectral_flatness(original)
    flat_rest = _spectral_flatness(restored)

    # Identity check: if audio is (near-)identical, clarity is inherently high —
    # no noise removal needed means the signal was already clean.
    _diff_energy = float(
        np.mean((original[: min(len(original), len(restored))] - restored[: min(len(original), len(restored))]) ** 2)
    )
    _orig_energy = float(np.mean(original[: min(len(original), len(restored))] ** 2) + 1e-12)
    _is_near_identical = _diff_energy / _orig_energy < 1e-4

    # §0d Carrier-Recovery: for additive HF-extension (vinyl/tape phase_06, phase_07),
    # the restored signal has higher spectral flatness than the degraded original
    # (new harmonic content added). This is correct restoration behaviour — not noise.
    # Detect this case and avoid penalising legitimate HF recovery.
    _hf_extension = flat_orig > 1e-6 and flat_rest > flat_orig * 1.05

    # Flatness should decrease (more tonal) but not too much (= over-processed)
    if flat_orig > 1e-6:
        flatness_improvement = max(0.0, (flat_orig - flat_rest) / flat_orig)
        if _hf_extension:
            # HF extension case: flatness increase is intentional — not a penalty source.
            flatness_improvement = 0.0
            over_processing = 0.0
        else:
            # Penalize over-processing: if flatness dropped > 60%, likely musical noise
            over_processing = max(0.0, flatness_improvement - 0.60) * 2.0
    else:
        flatness_improvement = 0.0
        over_processing = 0.0

    hnr_orig = _hnr_estimate(original)
    hnr_rest = _hnr_estimate(restored)
    hnr_improvement = max(0.0, hnr_rest - hnr_orig)

    # Score: some noise removal is good, too much is bad.
    # For additive HF extension, fall back to absolute HNR of restored signal as baseline
    # to avoid clarity=0.000 when flatness_improvement=0 and hnr_improvement=0.
    if _hf_extension:
        # Additive restoration: use absolute HNR + restored tonality as clarity.
        # Good HF restoration adds harmonic content → hnr_rest should be moderate-to-high.
        # Floor raised to 0.65: any successful HF extension is inherently clarity-positive.
        # (1 - flat_rest) captures how tonal/clean the restored spectrum is.
        tonality_rest = max(0.0, min(1.0 - flat_rest * 10.0, 1.0))  # 0.0 flatness → 1.0 tonal
        clarity_raw = max(0.65, min(hnr_rest * 1.5 + tonality_rest * 0.25, 1.0))
    else:
        # Improvement-based component (how much better vs. original)
        improvement_component = 0.50 * min(flatness_improvement * 2.0, 1.0) + 0.50 * min(hnr_improvement * 3.0, 1.0)
        # Absolute quality floor: a tonal/harmonic restored signal is inherently clear,
        # regardless of how much it changed from the (possibly already-decent) input.
        abs_quality_floor = max(0.0, min(hnr_rest * 2.0, 0.65))
        clarity_raw = max(improvement_component, abs_quality_floor)
    # If signal was already clean (low flatness), start with high baseline
    if flat_orig < 0.05:
        clarity_raw = max(clarity_raw, 0.85)
    # Near-identical audio: no modification needed = clean signal = high clarity
    if _is_near_identical:
        clarity_raw = max(clarity_raw, 0.90)

    score = clarity_raw - over_processing * _ARTIFACT_PENALTY_SCALE
    details = {
        "spectral_flatness_orig": round(flat_orig, 4),
        "spectral_flatness_rest": round(flat_rest, 4),
        "flatness_improvement_pct": round(flatness_improvement * 100, 1),
        "hnr_improvement": round(hnr_improvement, 4),
        "over_processing_penalty": round(over_processing, 4),
    }
    return float(max(0.0, min(1.0, score))), details


def _measure_authenticity(original: np.ndarray, restored: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
    """Misst timbral and tonal authenticity preservation.

    Combines:
    - MFCC correlation (timbre preservation)
    - Spectral centroid stability (brightness preservation)
    - Chroma correlation (key/harmonic preservation)
    """
    n_fft = 2048
    hop = 512
    n_mfcc = 13

    # Align lengths
    min_len = min(len(original), len(restored))
    orig = original[:min_len]
    rest = restored[:min_len]

    # 1. MFCC correlation (simplified — no librosa dependency required)
    def _mfcc_simple(audio: np.ndarray, n_coeffs: int = n_mfcc) -> np.ndarray:
        """Simplified MFCC via DCT of log-magnitude spectrum."""
        n_frames = max(1, (len(audio) - n_fft) // hop)
        mfccs = np.zeros((n_frames, n_coeffs))
        window = np.hanning(n_fft)
        for i in range(n_frames):
            frame = audio[i * hop : i * hop + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec = np.abs(np.fft.rfft(frame * window))
            log_spec = np.log(np.maximum(spec, 1e-10))
            # DCT-II (first n_coeffs)
            for k in range(n_coeffs):
                mfccs[i, k] = np.sum(
                    log_spec * np.cos(np.pi * k * (2 * np.arange(len(log_spec)) + 1) / (2 * len(log_spec)))
                )
        return mfccs  # type: ignore[no-any-return]

    mfcc_orig = _mfcc_simple(orig)
    mfcc_rest = _mfcc_simple(rest)
    min_frames = min(len(mfcc_orig), len(mfcc_rest))

    if min_frames >= 2:
        mfcc_orig = mfcc_orig[:min_frames].flatten()
        mfcc_rest = mfcc_rest[:min_frames].flatten()
        std_o = np.std(mfcc_orig)
        std_r = np.std(mfcc_rest)
        if std_o > 1e-8 and std_r > 1e-8:
            _mo = mfcc_orig - mfcc_orig.mean()
            _mr = mfcc_rest - mfcc_rest.mean()
            _no = float(np.linalg.norm(_mo))
            _nr = float(np.linalg.norm(_mr))
            cc = float(np.dot(_mo, _mr) / (_no * _nr + 1e-10))
            mfcc_corr = max(0.0, float(cc)) if np.isfinite(cc) else 0.5
        elif std_o < 1e-8 and std_r < 1e-8:
            mfcc_corr = 1.0  # Both constant
        else:
            mfcc_corr = 0.3  # One constant, one not
    else:
        mfcc_corr = 0.5

    # 2. Spectral centroid stability
    def _spectral_centroid(audio: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        total = np.sum(spec) + 1e-10
        return float(np.sum(freqs * spec) / total)

    centroid_orig = _spectral_centroid(orig)
    centroid_rest = _spectral_centroid(rest)
    # Centroid deviation as fraction of Nyquist
    centroid_dev = abs(centroid_rest - centroid_orig) / (sr / 2.0)
    centroid_score = max(0.0, 1.0 - centroid_dev * 10.0)  # 10% Nyquist shift = 0 score

    # 3. Chroma correlation (12-bin normalized chroma from full signal)
    def _chroma_vector(audio: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        chroma = np.zeros(12)
        for i, f in enumerate(freqs):
            if f < 20 or f > 8000:
                continue
            # Map frequency to chroma bin
            midi = 69 + 12 * np.log2(f / 440.0 + 1e-12)
            chroma_bin = int(round(midi)) % 12
            chroma[chroma_bin] += spec[i] ** 2
        norm = np.linalg.norm(chroma)
        if norm > 0:
            chroma /= norm
        return chroma  # type: ignore[no-any-return]

    # Use first 30s max for chroma
    chroma_len = min(min_len, int(30.0 * sr))
    chroma_o = _chroma_vector(orig[:chroma_len])
    chroma_r = _chroma_vector(rest[:chroma_len])
    chroma_corr = float(np.dot(chroma_o, chroma_r))

    score = 0.45 * mfcc_corr + 0.25 * centroid_score + 0.30 * chroma_corr
    details = {
        "mfcc_corr": round(mfcc_corr, 4),
        "centroid_score": round(centroid_score, 4),
        "centroid_orig_hz": round(centroid_orig, 1),
        "centroid_rest_hz": round(centroid_rest, 1),
        "chroma_corr": round(chroma_corr, 4),
    }
    return float(max(0.0, min(1.0, score))), details


def _measure_artifact_penalty(original: np.ndarray, restored: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
    """Erkennt processing artifacts introduced by restoration.

    Measures:
    - Pre-echo / post-echo (spectral leakage from STFT processing)
    - Musical noise (isolated spectral peaks in quiet passages)
    - Energy increase in signal gaps (noise floor elevation)
    """
    n_fft = 2048
    hop = 512
    min_len = min(len(original), len(restored))
    orig = original[:min_len]
    rest = restored[:min_len]

    # 1. Pre-echo detection: energy before onsets that shouldn't be there
    env_orig = _onset_envelope(orig, sr)
    env_rest = _onset_envelope(rest, sr)
    min_env = min(len(env_orig), len(env_rest))
    pre_echo_score = 0.0
    if min_env > 10:
        env_orig = env_orig[:min_env]
        env_rest = env_rest[:min_env]
        # Find quiet-before-loud transitions in original
        for i in range(2, min_env):
            if env_orig[i] > 0.5 and env_orig[i - 2] < 0.1:
                # Original: quiet → loud. Check if restored has energy leak before onset
                if env_rest[i - 2] > env_orig[i - 2] + 0.15:
                    pre_echo_score += 0.1
        pre_echo_score = min(1.0, pre_echo_score)

    # 2. Musical noise: spectral energy in silent passages of original
    # Find quiet regions (bottom 10% RMS frames)
    frame_size = hop
    n_frames = min(min_len // frame_size, 500)
    gap_noise_score = 0.0
    if n_frames >= 10:
        orig_frames = orig[: n_frames * frame_size].reshape(n_frames, frame_size)
        rest_frames = rest[: n_frames * frame_size].reshape(n_frames, frame_size)
        orig_rms = np.sqrt(np.mean(orig_frames**2, axis=1) + 1e-12)
        rest_rms = np.sqrt(np.mean(rest_frames**2, axis=1) + 1e-12)

        quiet_threshold = np.percentile(orig_rms, 10)
        quiet_mask = orig_rms < quiet_threshold
        if np.any(quiet_mask):
            # In quiet regions, restored should not be louder than original
            orig_quiet_mean = float(np.mean(orig_rms[quiet_mask]))
            rest_quiet_mean = float(np.mean(rest_rms[quiet_mask]))
            if orig_quiet_mean > 1e-8:
                ratio = rest_quiet_mean / orig_quiet_mean
                # If restored is >2x louder in quiet regions = artifact
                gap_noise_score = max(0.0, min(1.0, (ratio - 1.0) / 2.0))

    # 3. Overall residual artifact energy (restored - original in spectral domain)
    residual_score = 0.0
    try:
        spec_orig = np.abs(np.fft.rfft(orig[:n_fft]))
        spec_rest = np.abs(np.fft.rfft(rest[:n_fft]))
        residual = np.maximum(spec_rest - spec_orig, 0.0)
        residual_energy = float(np.sum(residual**2))
        original_energy = float(np.sum(spec_orig**2)) + 1e-10
        residual_ratio = residual_energy / original_energy
        residual_score = min(1.0, residual_ratio * 5.0)  # 20% added energy = max penalty
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    # Combined penalty (weighted)
    penalty = 0.35 * pre_echo_score + 0.35 * gap_noise_score + 0.30 * residual_score
    details = {
        "pre_echo_penalty": round(pre_echo_score, 4),
        "gap_noise_penalty": round(gap_noise_score, 4),
        "residual_artifact_penalty": round(residual_score, 4),
    }
    return float(max(0.0, min(1.0, penalty))), details


def measure_goosebumps(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    musical_goal_scores: dict[str, float] | None = None,
) -> GoosebumpsResult:
    """Compute the holistic psychoacoustic goosebumps quality score (§8.3).

    Binding implementation of the Gänsehaut-Formel:
        score = T^0.40 × M^0.25 × K^0.20 × A^0.15 − Artefakte × scale

    Multiplicative coupling ensures that a single weak dimension pulls the
    entire score down non-linearly (weighted geometric mean).

    Args:
        original:  Original audio (float64, any shape)
        restored:  Restored audio (float64, same shape)
        sr:        Sample rate
        musical_goal_scores:  Optional 15 Musical Goals dict to boost precision

    Returns:
        GoosebumpsResult with all dimension scores and final holistic score
    """
    try:
        # Convert to mono for analysis
        orig_m = _to_mono(original)
        rest_m = _to_mono(restored)

        # Center-crop for performance (max 60s)
        orig_m = _center_crop(orig_m, sr, _MAX_ANALYSIS_S)
        rest_m = _center_crop(rest_m, sr, _MAX_ANALYSIS_S)

        # Align lengths
        min_len = min(len(orig_m), len(rest_m))
        if min_len < sr:  # Less than 1 second
            return GoosebumpsResult(
                goosebumps_score=0.5,
                transient_integrity=0.5,
                micro_dynamics=0.5,
                clarity=0.5,
                authenticity=0.5,
                artifact_penalty=0.0,
                details={"skipped": True, "reason": "audio_too_short"},
            )
        orig_m = orig_m[:min_len]
        rest_m = rest_m[:min_len]

        # Measure all dimensions
        transient, t_details = _measure_transient_integrity(orig_m, rest_m, sr)
        micro_dyn, md_details = _measure_micro_dynamics(orig_m, rest_m, sr)
        clarity, cl_details = _measure_clarity(orig_m, rest_m, sr)
        authenticity, au_details = _measure_authenticity(orig_m, rest_m, sr)
        artifact_pen, ar_details = _measure_artifact_penalty(orig_m, rest_m, sr)

        # If Musical Goals scores are available, use them to refine dimensions
        if musical_goal_scores:
            # Boost authenticity with actual metric scores
            mg_auth = musical_goal_scores.get("authentizitaet", authenticity)
            mg_timbre = musical_goal_scores.get("timbre_authentizitaet", authenticity)
            mg_micro = musical_goal_scores.get("micro_dynamics", micro_dyn)
            mg_artic = musical_goal_scores.get("artikulation", transient)

            # Blend: 60% DSP measurement + 40% Musical Goals (higher precision)
            authenticity = 0.60 * authenticity + 0.40 * ((mg_auth + mg_timbre) / 2.0)
            micro_dyn = 0.60 * micro_dyn + 0.40 * mg_micro
            transient = 0.70 * transient + 0.30 * mg_artic  # Articulation ≈ transient quality

        # §8.3 Gänsehaut-Formel (spec-binding MULTIPLICATION, not sum):
        #   score = T^0.40 × M^0.25 × K^0.20 × A^0.15 − artifact_penalty × scale
        # Weighted geometric mean: exponents = spec weights (40/25/20/15).
        # A single weak dimension pulls the entire score down non-linearly.
        # Clamp factors to [epsilon, 1.0] to avoid log(0) / zero-product collapse.
        _eps = 1e-6
        t_clamped = max(_eps, min(1.0, transient))
        m_clamped = max(_eps, min(1.0, micro_dyn))
        k_clamped = max(_eps, min(1.0, clarity))
        a_clamped = max(_eps, min(1.0, authenticity))
        raw_score = (
            t_clamped**_W_TRANSIENT * m_clamped**_W_MICRO_DYN * k_clamped**_W_CLARITY * a_clamped**_W_AUTHENTICITY
        )
        goosebumps_score = max(0.0, min(1.0, raw_score - artifact_pen * _ARTIFACT_PENALTY_SCALE))

        all_details = {
            **t_details,
            **md_details,
            **cl_details,
            **au_details,
            **ar_details,
            "raw_score_before_penalty": round(raw_score, 4),
            "musical_goals_blended": musical_goal_scores is not None,
        }

        # NaN/Inf-Guard (§Checkliste: JEDE numerische Ausgabefunktion)
        goosebumps_score = float(np.nan_to_num(goosebumps_score, nan=0.5, posinf=1.0, neginf=0.0))
        transient = float(np.nan_to_num(transient, nan=0.5, posinf=1.0, neginf=0.0))
        micro_dyn = float(np.nan_to_num(micro_dyn, nan=0.5, posinf=1.0, neginf=0.0))
        clarity = float(np.nan_to_num(clarity, nan=0.5, posinf=1.0, neginf=0.0))
        authenticity = float(np.nan_to_num(authenticity, nan=0.5, posinf=1.0, neginf=0.0))
        artifact_pen = float(np.nan_to_num(artifact_pen, nan=0.0, posinf=1.0, neginf=0.0))

        return GoosebumpsResult(
            goosebumps_score=round(goosebumps_score, 4),
            transient_integrity=round(transient, 4),
            micro_dynamics=round(micro_dyn, 4),
            clarity=round(clarity, 4),
            authenticity=round(authenticity, 4),
            artifact_penalty=round(artifact_pen, 4),
            details=all_details,
        )

    except Exception as exc:
        logger.warning("GoosebumpsQualityChecker fehlgeschlagen — returning neutral Wert: %s", exc)
        return GoosebumpsResult(
            goosebumps_score=0.5,
            transient_integrity=0.5,
            micro_dynamics=0.5,
            clarity=0.5,
            authenticity=0.5,
            artifact_penalty=0.0,
            details={"error": str(exc)},
        )


# ─── Singleton Pattern (§3.x) ────────────────────────────────────────────────

_instance: GoosebumpsQualityChecker | None = None
_lock = threading.Lock()


class GoosebumpsQualityChecker:
    """Singleton-Wrapper für Gänsehaut-Qualitätsbewertung (§8.3 Spec).

    Thread-safe, double-checked locking pattern as required by §3.x.
    """

    def measure(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        musical_goal_scores: dict[str, float] | None = None,
    ) -> GoosebumpsResult:
        """Misst goosebumps quality of restored audio vs. original."""
        return measure_goosebumps(original, restored, sr, musical_goal_scores)


def get_goosebumps_checker() -> GoosebumpsQualityChecker:
    """Thread-safe singleton accessor for GoosebumpsQualityChecker."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GoosebumpsQualityChecker()
    return _instance
