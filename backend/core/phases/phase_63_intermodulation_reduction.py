"""
Phase 63 — Intermodulation Distortion Reduction.

IMD creates sum/difference frequency products (f1±f2) from nonlinear
signal paths that are NOT harmonically related to either input frequency.
This is distinct from THD (handled by phase_60) and clipping (phase_23).

Algorithm (Bispectrum-informed, §2.51 M/S compliant):
1. Identify tonal peaks via bispectrum analysis (detects f1+f2 correlations
   without guessing) — more robust than peak-pair enumeration alone.
2. Confirm IMD products: energy at predicted f1+f2 / |f1−f2| is significantly
   above noise floor AND coherent with the fundamental-pair bispectrum.
3. Build frequency-domain notch mask on the Mid channel (M/S domain).
4. Apply notch mask to Mid; apply at reduced strength (≤30 %) on Side.
5. Reconstruct L/R from processed M/S.

§2.51 compliance: no independent L/R processing — notch derived from Mid
spectrum and applied symmetrically (linked stereo semantics in M/S domain).

Scientific basis: Volterra series models; SMPTE RP120-1994;
Kim & Powers (1979) "Digital Bispectral Analysis and its Applications";
Farina (2000) "Simultaneous Measurement of Impulse Response and Distortion".
"""

from __future__ import annotations

import logging
import time as _time
from typing import cast

import numpy as np
import scipy.signal as sps

logger = logging.getLogger(__name__)


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    if sig.ndim == 2:
        _mono = sig.mean(axis=0 if sig.shape[0] <= 2 else 1).astype(np.float32)
    else:
        _mono = np.asarray(sig, dtype=np.float32).ravel()
    _frame = 480  # 10 ms @ 48 kHz
    _n_frames = len(_mono) // _frame
    if _n_frames == 0:
        return -96.0
    _frames = _mono[: _n_frames * _frame].reshape(_n_frames, _frame)
    _frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(_frames**2, axis=1)) + 1e-10)
    _mask = _frame_rms_db > -50.0
    if not np.any(_mask):
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(_frames[_mask] ** 2)) + 1e-10))


_MIN_IMD_SCORE: float = 0.10
_NOTCH_WIDTH_HZ: float = 50.0
_BISPECTRUM_NFFT: int = 2048
_PROC_NFFT: int = 8192


def _build_imd_notch_mask(
    x_mono: np.ndarray,
    sample_rate: int,
    strength: float,
    notch_width_hz: float,
) -> np.ndarray:
    """Berechnet frequency-domain gain mask for IMD notch filtering.

    Uses bispectrum-coherence to confirm whether energy at predicted IMD
    product locations is causally related to the identified fundamentals.
    Returns gain mask of length PROC_NFFT//2+1, values in [0, 1].
    """
    n = len(x_mono)
    n_fft = _BISPECTRUM_NFFT
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # Long-term power spectrum to find stable tonal peaks
    n_frames = max(1, (n - n_fft) // (n_fft // 4))
    sum_spec = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        s = i * (n_fft // 4)
        if s + n_fft > n:
            break
        frame = x_mono[s : s + n_fft] * np.hanning(n_fft)
        sum_spec += np.abs(np.fft.rfft(frame)) ** 2

    if n_frames > 0:
        sum_spec /= n_frames

    global_db = 10.0 * np.log10(sum_spec + 1e-20)
    noise_floor = float(np.percentile(global_db, 20))

    # Detect peaks ≥ 20 dB above noise floor
    peak_mask = global_db > (noise_floor + 20.0)
    peak_indices = [int(i) for i in np.where(peak_mask)[0]]
    if len(peak_indices) < 2:
        proc_freqs = np.fft.rfftfreq(_PROC_NFFT, 1.0 / sample_rate)
        return np.ones(len(proc_freqs), dtype=np.float32)  # type: ignore[no-any-return]

    # Cluster nearby peaks → fundamental frequencies
    clusters: list[int] = []
    if peak_indices:
        grp = [peak_indices[0]]
        for idx in peak_indices[1:]:
            if (idx - grp[-1]) * freq_res < 10.0:
                grp.append(idx)
            else:
                clusters.append(int(np.mean(grp)))
                grp = [idx]
        clusters.append(int(np.mean(grp)))
    clusters = clusters[:12]

    # Predict IMD product bins (2nd-order: f1+f2 and |f1-f2|)
    imd_targets: list[int] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            f1 = freqs[clusters[i]]
            f2 = freqs[clusters[j]]
            for target_f in [abs(f1 - f2), f1 + f2]:
                if target_f < 50 or target_f > sample_rate / 2 - 100:
                    continue
                # Exclude harmonics of either fundamental
                is_harmonic = any(abs(target_f - h * base) < freq_res * 3 for base in [f1, f2] for h in range(1, 8))
                if is_harmonic:
                    continue
                target_idx = int(target_f / freq_res)
                if 0 < target_idx < len(global_db):
                    # Confirm: at least 5 dB above noise floor (avoid false positives)
                    if global_db[target_idx] > noise_floor + 5.0:
                        # ── Bispectrum coherence check ────────────────────────
                        # B(f1, f2) = E[X(f1) · X(f2) · conj(X(f1+f2))]
                        # If |B| / (P(f1)·P(f2)·P(f1+f2))^0.5 is high, the product
                        # energy is causally related to the fundamentals → true IMD.
                        # Approximate: use single frame for speed.
                        frame_len = min(n, n_fft)
                        mid_start = max(0, n // 2 - frame_len // 2)
                        frame = x_mono[mid_start : mid_start + frame_len] * np.hanning(frame_len)
                        X = np.fft.rfft(frame, n=n_fft)
                        b_coherence = 0.0
                        if clusters[i] < len(X) and clusters[j] < len(X) and target_idx < len(X):
                            bispec = abs(X[clusters[i]] * X[clusters[j]] * np.conj(X[target_idx]))
                            denom_b = (
                                (abs(X[clusters[i]]) ** 2 + 1e-20)
                                * (abs(X[clusters[j]]) ** 2 + 1e-20)
                                * (abs(X[target_idx]) ** 2 + 1e-20)
                            ) ** (1.0 / 3.0)
                            b_coherence = float(bispec / denom_b)
                        if b_coherence > 0.15:  # Confirmed IMD via bispectrum
                            imd_targets.append(target_idx)

    if not imd_targets:
        proc_freqs = np.fft.rfftfreq(_PROC_NFFT, 1.0 / sample_rate)
        return np.ones(len(proc_freqs), dtype=np.float32)  # type: ignore[no-any-return]

    # Build notch mask at PROC_NFFT resolution
    proc_freqs = np.fft.rfftfreq(_PROC_NFFT, 1.0 / sample_rate)
    proc_freq_res = float(proc_freqs[1] - proc_freqs[0]) if len(proc_freqs) > 1 else 1.0
    notch_width_bins = max(1, int(notch_width_hz / proc_freq_res))
    gain_mask = np.ones(len(proc_freqs), dtype=np.float32)

    for analysis_idx in imd_targets:
        # Map from analysis (BISPECTRUM_NFFT) to proc (PROC_NFFT) resolution
        target_freq_hz = freqs[analysis_idx]
        proc_idx = int(target_freq_hz / proc_freq_res)
        lo = max(0, proc_idx - notch_width_bins // 2)
        hi = min(len(gain_mask), proc_idx + notch_width_bins // 2 + 1)
        for k in range(lo, hi):
            dist = abs(k - proc_idx) / max(1, notch_width_bins // 2)
            notch_depth = float(np.clip(1.0 - strength * (1.0 - dist**2), 0.15, 1.0))
            gain_mask[k] = min(gain_mask[k], notch_depth)

    return gain_mask  # type: ignore[no-any-return]


def _apply_stft_mask(
    x: np.ndarray,
    gain_mask: np.ndarray,
    sample_rate: int,  # pylint: disable=unused-argument
) -> np.ndarray:
    """Wendet eine Frequenzbereichs-Gain-Maske via STFT-Overlap-Add an (vektorisiert)."""
    n = len(x)
    n_fft = _PROC_NFFT
    hop = n_fft // 4
    window = sps.windows.hann(n_fft, sym=False).astype(np.float32)
    n_frames = max(1, (n - n_fft) // hop + 1)
    # Trim n_frames so every frame fits inside x
    while n_frames > 0 and (n_frames - 1) * hop + n_fft > n:
        n_frames -= 1
    if n_frames == 0:
        return x.astype(np.float32)  # type: ignore[no-any-return]

    # Batch STFT: stride_tricks framing + vectorised rfft
    x_f32 = np.asarray(x, dtype=np.float32)
    _sw = np.lib.stride_tricks.sliding_window_view(x_f32, n_fft)[::hop][:n_frames]  # (T, n_fft)
    _spectra = np.fft.rfft(_sw * window, axis=1)  # (T, F)
    _spectra *= gain_mask  # broadcast over T
    _frames_out = np.fft.irfft(_spectra, n=n_fft, axis=1) * window  # (T, n_fft)

    out = np.zeros(n, dtype=np.float32)
    win_sum = np.zeros(n, dtype=np.float32)
    win_sq = window**2
    for i in range(n_frames):
        s = i * hop
        out[s : s + n_fft] += _frames_out[i]
        win_sum[s : s + n_fft] += win_sq

    win_sum = np.maximum(win_sum, 1e-8)
    out /= win_sum
    return out  # type: ignore[no-any-return]


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.55,
    defect_scores: dict | None = None,
    min_imd_score: float = _MIN_IMD_SCORE,
    notch_width_hz: float = _NOTCH_WIDTH_HZ,
) -> np.ndarray:
    """Haupt-entry point for Phase 63 — §2.51 M/S-compliant."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        imd_score = float(defect_scores.get("intermodulation_distortion", 0.0))
        if imd_score < min_imd_score:
            logger.debug("Verarbeitungsschritt 63: IMD Wert %.3f < %.3f — uebersprungen", imd_score, min_imd_score)
            return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 M/S domain: derive notch mask from Mid; apply to Mid and Side at
        # reduced strength. This keeps L/R coherent (no independent per-channel mask).
        if audio.shape[0] == 2 and audio.shape[1] != 2:
            left = audio[0].astype(np.float32)
            right = audio[1].astype(np.float32)
            channels_first = True
        else:
            left = audio[:, 0].astype(np.float32)
            right = audio[:, 1].astype(np.float32)
            channels_first = False

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        # Notch mask derived from Mid signal only
        gain_mask = _build_imd_notch_mask(mid, sample_rate, strength, notch_width_hz)

        mid_clean = _apply_stft_mask(mid, gain_mask, sample_rate)
        # Side: apply same mask at 30 % strength (IMD arises from Mid nonlinearity)
        side_gain = gain_mask * 0.3 + 0.7  # blend: 30 % of the notch
        side_clean = _apply_stft_mask(side, side_gain, sample_rate)

        left_out = np.clip(mid_clean + side_clean, -1.0, 1.0)
        right_out = np.clip(mid_clean - side_clean, -1.0, 1.0)

        left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
        right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)

        if channels_first:
            return np.stack([left_out, right_out], axis=0).astype(np.float32)  # type: ignore[no-any-return]
        else:
            return np.stack([left_out, right_out], axis=1).astype(np.float32)  # type: ignore[no-any-return]

    # Mono path
    x = audio.astype(np.float32)
    gain_mask = _build_imd_notch_mask(x, sample_rate, strength, notch_width_hz)
    out = _apply_stft_mask(x, gain_mask, sample_rate)
    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import (  # pylint: disable=wrong-import-position
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)


class IntermodulationReductionPhase(PhaseInterface):
    """Phase 63: Volterra-based intermodulation distortion reduction."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_63_intermodulation_reduction",
            name="Intermodulation Distortion Reduction",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_04"],
            estimated_time_factor=0.05,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.55,
            description=(
                "Targeted spectral notch filtering of intermodulation distortion "
                "products (sum/difference frequencies). Identifies IMD products "
                "from tonal peak analysis and removes non-harmonic artifacts."
            ),
        )

    @staticmethod
    def _compute_imd_profile(material_key: str, quality_mode: str, restorability_score: float) -> dict[str, float]:
        material = str(material_key or "unknown").strip().lower()
        mode = str(quality_mode or "balanced").strip().lower()
        mode = {"restoration": "balanced", "studio_2026": "maximum"}.get(mode, mode)

        if any(token in material for token in ("shellac", "wax_cylinder", "optical_film")):
            min_imd_score = 0.24
            notch_width_hz = 70.0
        elif "vinyl" in material:
            min_imd_score = 0.18
            notch_width_hz = 55.0
        elif any(token in material for token in ("cassette", "tape", "reel_tape")):
            min_imd_score = 0.12
            notch_width_hz = 50.0
        elif any(token in material for token in ("cd_digital", "dat", "mp3", "flac", "streaming")):
            min_imd_score = 0.08
            notch_width_hz = 40.0
        else:
            min_imd_score = 0.12
            notch_width_hz = 50.0

        rest_norm = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0)) / 100.0
        min_imd_score += (rest_norm - 0.5) * 0.14
        notch_width_hz += (rest_norm - 0.5) * 20.0

        score_off, width_off = {
            "fast": (0.04, 12.0),
            "balanced": (0.00, 0.0),
            "quality": (-0.03, -10.0),
            "maximum": (-0.05, -15.0),
        }.get(mode, (0.0, 0.0))
        min_imd_score += score_off
        notch_width_hz += width_off

        return {
            "min_imd_score": float(np.clip(min_imd_score, 0.05, 0.30)),
            "notch_width_hz": float(np.clip(notch_width_hz, 20.0, 120.0)),
        }

    @staticmethod
    def _local_event_strength(key: str, loc: tuple[float, float], event_metadata: dict[str, dict] | None) -> float:
        duration_s = max(0.0, float(loc[1]) - float(loc[0]))
        duration_factor = float(np.clip(duration_s / 0.65, 0.28, 1.0))
        key_factor = {
            "intermodulation_distortion": 1.0,
            "imd": 1.0,
            "digital_artifacts": 0.62,
            "overload_distortion": 0.46,
        }.get(key, 0.64)
        severity = 0.55
        confidence = 0.78
        meta_obj = (event_metadata or {}).get(key)
        if isinstance(meta_obj, dict):
            severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
            confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
        return float(np.clip(key_factor * (0.34 + 0.46 * severity + 0.20 * confidence) * duration_factor, 0.12, 1.0))

    @staticmethod
    def _collect_protected_zones(kwargs: dict) -> list[tuple[float, float, float]]:
        zones: list[tuple[float, float, float]] = []
        for key, cap in (
            ("vibrato_zones", 0.20),
            ("frisson_zones", 0.30),
            ("whisper_zones", 0.25),
            ("passaggio_zones", 0.35),
        ):
            for zone in kwargs.get(key) or []:
                try:
                    start_s = float(getattr(zone, "start_s", None) or zone[0])
                    end_s = float(getattr(zone, "end_s", None) or zone[1])
                    if end_s > start_s:
                        zones.append((start_s, end_s, cap))
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
        return zones

    @staticmethod
    def _build_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
        event_metadata: dict[str, dict] | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, float]:
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0

        keys = ("intermodulation_distortion", "imd", "digital_artifacts", "overload_distortion")
        mask = np.zeros(n_samples, dtype=np.float32)
        for key in keys:
            pad = int((0.040 if key in {"intermodulation_distortion", "imd"} else 0.025) * sample_rate)
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start = int(max(0.0, float(loc[0])) * sample_rate)
                    end = int(max(0.0, float(loc[1])) * sample_rate)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
                if end <= start:
                    continue
                start = max(0, start - pad)
                end = min(n_samples, end + pad)
                if end > start:
                    strength = IntermodulationReductionPhase._local_event_strength(key, loc, event_metadata)
                    mask[start:end] = np.maximum(mask[start:end], strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0

        smooth = max(16, int(0.018 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if protected_zones:
            for start_s, end_s, cap in protected_zones:
                start = int(max(0.0, float(start_s)) * sample_rate)
                end = int(max(0.0, float(end_s)) * sample_rate)
                if end > start:
                    mask[start : min(n_samples, end)] = np.minimum(mask[start : min(n_samples, end)], float(cap))
        return mask, float(np.mean(mask))

    @staticmethod
    def _blend_with_locality(reference: np.ndarray, candidate: np.ndarray, profile: np.ndarray) -> np.ndarray:
        if reference.shape != candidate.shape or profile.size == 0:
            return candidate
        wet: np.ndarray
        if reference.ndim == 1:
            wet = np.asarray(profile, dtype=np.float32)
        elif reference.ndim == 2 and reference.shape[0] == profile.size and reference.shape[1] <= 8:
            wet = np.asarray(profile[:, np.newaxis], dtype=np.float32)
        elif reference.ndim == 2 and reference.shape[1] == profile.size:
            wet = np.asarray(profile[np.newaxis, :], dtype=np.float32)
        else:
            return candidate
        blended = cast(np.ndarray, reference + wet * (candidate - reference))
        clipped = cast(np.ndarray, np.clip(np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0))
        return cast(np.ndarray, clipped.astype(np.float32))

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        _defect_scores = kwargs.get("defect_scores") or kwargs.get("defect_analysis", {})
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 0.55))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        _profile_63 = self._compute_imd_profile(
            str(material_type or kwargs.get("material") or "unknown"),
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "imd_score": float((_defect_scores or {}).get("intermodulation_distortion", 0.0)),
                    "strength": _pmgg_strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Intermodulation reduction skipped due to zero effective strength"],
            )
        _imd_score_63 = float((_defect_scores or {}).get("intermodulation_distortion", 0.0))
        _imd_threshold_63 = float(_profile_63["min_imd_score"])
        _mat_63 = material_type or kwargs.get("material")
        _mat_str_63 = str(_mat_63).lower() if _mat_63 is not None else ""
        if _imd_score_63 < _imd_threshold_63:
            _passthrough_63 = np.clip(np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            logger.debug(
                "Verarbeitungsschritt 63: IMD %.3f < material-adaptive Schwelle %.3f (%s) → ueberspringen",
                _imd_score_63,
                _imd_threshold_63,
                _mat_str_63 or "unknown",
            )
            return PhaseResult(
                audio=_passthrough_63,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={"imd_score": _imd_score_63, "strength": _pmgg_strength, "effective_strength": 0.0},
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "imd_threshold": _imd_threshold_63,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["IMD below material-adaptive threshold"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_imd_score=_profile_63["min_imd_score"],
            notch_width_hz=_profile_63["notch_width_hz"],
        )
        _n_samples63 = (
            audio.shape[1] if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2 else audio.shape[0]
        )
        _local_profile63, _local_coverage63 = self._build_locality_profile(
            int(_n_samples63),
            sample_rate,
            kwargs.get("defect_locations"),
            kwargs.get("defect_event_metadata"),
            self._collect_protected_zones(kwargs),
        )
        if _local_coverage63 > 0.0:
            result_audio = self._blend_with_locality(audio, result_audio, _local_profile63)
        elapsed = _time.perf_counter() - t0

        # §4.5 Psychoacoustic Masking Clamp — only reduce audible intermodulation
        try:
            from backend.core.dsp.psychoacoustics import (
                apply_psychoacoustic_masking_clamp,  # pylint: disable=import-outside-toplevel
            )

            result_audio = apply_psychoacoustic_masking_clamp(
                audio,
                result_audio,
                sample_rate,
                strength=_effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            logger.debug("Verarbeitungsschritt63 masking clamp nicht blockierend: %s", _pm_exc)

        # V20 Mikrodynamik-Korrelation (§2.75): IMD-Notch-Filterung darf voiced Frames
        # nicht in ihrer Frame-Energie degradieren (panns_singing ≥ 0.25).
        _panns63 = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if _panns63 >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (
                    frame_energy_correlation,  # pylint: disable=import-outside-toplevel
                    recommend_mikrodynamik_wet,
                )

                _corr63 = frame_energy_correlation(audio, result_audio, sample_rate, frame_ms=10.0)
                if _corr63 < 0.97:
                    _need63 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet63 = recommend_mikrodynamik_wet(_corr63, _panns63, global_need=_need63)
                    result_audio = (_wet63 * result_audio + (1.0 - _wet63) * audio).astype(np.float32)
                    logger.warning(
                        "Verarbeitungsschritt63 V20 Mikrodynamik-Korr=%.3f < 0.97 → wet=%.3f Blend",
                        _corr63,
                        _wet63,
                    )
            except Exception as _dyn63_exc:
                logger.debug("Verarbeitungsschritt63 V20 Mikrodynamik-Guard (nicht blockierend): %s", _dyn63_exc)

        # V26 Onset-Guard (§2.77): IMD-Notch-Filterung darf Transient-Energie in Onset-
        # Fenstern (0–20 ms nach Transient) nicht um mehr als 1.5 dB verschieben.
        try:
            from backend.core.dsp.onset_guard import (
                apply_onset_protection_mask,  # pylint: disable=import-outside-toplevel
            )

            result_audio = apply_onset_protection_mask(audio, result_audio, None, max_delta_db=1.5)
        except Exception as _on63_exc:
            logger.debug("Verarbeitungsschritt63 V26 Onset-Guard (nicht blockierend): %s", _on63_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — IMD-Notch-Filtering darf
        # Atemgeräusche (Inter-Phrasen-Energie) und Vibrato-Segmente nicht tilgen.
        try:
            from backend.core.natural_performance_detector import (
                get_natural_performance_detector,  # pylint: disable=import-outside-toplevel
            )

            _npa_a63 = audio
            if _npa_a63.ndim == 2 and _npa_a63.shape[0] == 2 and _npa_a63.shape[1] > _npa_a63.shape[0]:
                _npa_a63 = _npa_a63.T
            _npa_r63 = get_natural_performance_detector().detect(_npa_a63, sample_rate)
            _npa_n63 = (
                result_audio.shape[1]
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else result_audio.shape[0]
            )
            _npa_m63 = _npa_r63.get_protected_mask(_npa_n63, sample_rate)
            if np.any(_npa_m63):
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _npa_m63] = audio[:, _npa_m63]
                    elif result_audio.shape == audio.shape:
                        result_audio[_npa_m63, :] = audio[_npa_m63, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_npa_m63] = audio[_npa_m63]
        except Exception as _npa63_exc:
            logger.debug("§2.46f Verarbeitungsschritt_63 NPA-Guard (nicht blockierend): %s", _npa63_exc)

        # §V19 Noise-Textur-Invariante (VERBOTEN-V19): Residual bewahrt Materialcharakter
        _mat63_str = str(material_type or "unknown").lower()
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt63_fn,
            )

            if result_audio.shape == audio.shape:
                _nt63_d = _nt63_fn(
                    audio.astype(np.float32) - result_audio.astype(np.float32), _mat63_str, sr=sample_rate
                )
                if _nt63_d > 0.25:
                    result_audio = (0.5 * result_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_63 noise_texture dist=%.3f > 0.25 → 50%%-Blend",
                        _nt63_d,
                    )
        except Exception as _nt63_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_63 noise_texture_guard (nicht blockierend): %s",
                _nt63_exc,
            )

        # §V24 Spektralfarbe-Prüfung (VERBOTEN-V24): 1/3-Oktav-Profil darf nicht verfärbt werden
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg63,
            )

            if result_audio.shape == audio.shape:
                _sc63 = _scg63(audio.astype(np.float32), result_audio.astype(np.float32), sample_rate)
                if not _sc63.ok:
                    result_audio = (0.70 * result_audio + 0.30 * audio).astype(np.float32)
        except Exception as _sc63_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_63 spectral_color_guard (nicht blockierend): %s",
                _sc63_exc,
            )

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "imd_score": float((_defect_scores or {}).get("intermodulation_distortion", 0.0)),
                "strength": _pmgg_strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "imd_profile": dict(_profile_63),
                "imd_threshold": float(_profile_63["min_imd_score"]),
                "notch_width_hz": float(_profile_63["notch_width_hz"]),
                "repair_locality_coverage": round(float(_local_coverage63), 6),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )
