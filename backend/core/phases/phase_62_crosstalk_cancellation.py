"""
Phase 62 — Crosstalk Cancellation.

Channel crosstalk in early stereo recordings where channel separation was
limited (< 20 dB).  Implements the analytically exact inverse of the Vinyl
groove constant-α crosstalk mixing model (Blauert 1997; IEC 60098):

    L_played = L_cut + α(f) · R_cut
    R_played = R_cut + α(f) · L_cut

Inverse (assuming α(f) uniform per frame):
    det = 1 − α²
    L_clean = (L_played − α · R_played) / det
    R_clean = (R_played − α · L_played) / det

α(f) is frequency-dependent: typically −25 to −30 dB at low frequencies,
increasing toward HF (worse channel separation above 5 kHz).

Algorithm:
1. Compute long-term average auto/cross-spectra from full signal
2. Estimate α(f): |S_LR(f)| / sqrt(S_LL(f) · S_RR(f)), capped at 0.70
3. Apply per-frame analytical inverse matrix in STFT domain
4. Wet/dry blend controlled by strength

Scientific basis: Blauert (1997) "Spatial Hearing", §3.1 Channel Separation;
IEC 60098:1987 Groove dimensions; Avendano & Jot (2002) "Frequency-Domain
Crosstalk Separation".
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


_MIN_CROSSTALK_SCORE: float = 0.10
# Maximum α to prevent over-cancellation (α > 0.70 → |det| < 0.51 → unstable)
_ALPHA_MAX: float = 0.70
# Estimation window: 4096 for frequency resolution at 48 kHz (≈11.7 Hz/bin)
_ESTIM_NFFT: int = 4096
# Processing STFT
_PROC_NFFT: int = 4096
_PROC_HOP: int = _PROC_NFFT // 4


def _estimate_alpha_f(
    left: np.ndarray,
    right: np.ndarray,
    n_fft: int,
    hop: int,
    alpha_max: float,
) -> np.ndarray:
    """Schätzt frequency-dependent crosstalk coefficient α(f) via long-term spectra.

    Uses the coherence between L and R as the magnitude estimate for α, which is
    the correct estimator for the model  L = S + αR, R = S' + αL (Avendano & Jot 2002).

    Returns α(f) array of length n_fft//2 + 1, dtype float64, clipped to _ALPHA_MAX.
    """
    window = sps.windows.hann(n_fft, sym=False)
    n = len(left)
    n_frames = max(1, (n - n_fft) // hop + 1)

    sum_ll = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    sum_rr = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    sum_lr_r = np.zeros(n_fft // 2 + 1, dtype=np.float64)  # Real part of S_LR
    sum_lr_i = np.zeros(n_fft // 2 + 1, dtype=np.float64)  # Imag part of S_LR

    for i in range(n_frames):
        s = i * hop
        e = s + n_fft
        if e > n:
            break
        fl = np.fft.rfft(left[s:e] * window)
        fr = np.fft.rfft(right[s:e] * window)
        sum_ll += np.abs(fl) ** 2
        sum_rr += np.abs(fr) ** 2
        cross = fl * np.conj(fr)
        sum_lr_r += cross.real
        sum_lr_i += cross.imag

    # α(f) = |S_LR(f)| / sqrt(S_LL(f) · S_RR(f))  — coherence magnitude
    denom = np.sqrt(np.maximum(sum_ll * sum_rr, 1e-30))
    alpha_f = np.sqrt(sum_lr_r**2 + sum_lr_i**2) / denom
    # Hard cap to guarantee invertibility and prevent over-correction
    return np.clip(alpha_f, 0.0, alpha_max)  # type: ignore[no-any-return]


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.5,
    defect_scores: dict | None = None,
    min_crosstalk_score: float = _MIN_CROSSTALK_SCORE,
    alpha_max: float = _ALPHA_MAX,
) -> np.ndarray:
    """Haupt-entry point for Phase 62."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        xt_score = float(defect_scores.get("crosstalk", 0.0))
        if xt_score < min_crosstalk_score:
            logger.debug(
                "Verarbeitungsschritt 62: crosstalk Wert %.3f < %.3f — uebersprungen", xt_score, min_crosstalk_score
            )
            return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

    # Crosstalk cancellation only applies to stereo
    if audio.ndim != 2:
        logger.debug("Verarbeitungsschritt 62: mono Eingabe — uebersprungen (no crosstalk possible)")
        return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

    # Normalise to [channels, samples] = (2, N)
    if audio.shape[0] == 2 and audio.shape[1] != 2:
        left = audio[0].astype(np.float32)
        right = audio[1].astype(np.float32)
        channels_first = True
    else:
        # samples-first (N, 2)
        left = audio[:, 0].astype(np.float32)
        right = audio[:, 1].astype(np.float32)
        channels_first = False

    n = len(left)
    window = sps.windows.hann(_PROC_NFFT, sym=False).astype(np.float32)

    # ── Step 1: estimate α(f) from full signal ─────────────────────────────
    alpha_f = _estimate_alpha_f(left, right, _ESTIM_NFFT, _ESTIM_NFFT // 4, alpha_max)
    # Interpolate to processing NFFT bins (ESTIM and PROC have same n_fft here)
    # Scale alpha by strength so we only invert the fraction the user requests
    alpha_applied = alpha_f * float(np.clip(strength, 0.0, 1.0))

    # Precompute per-bin denominator 1/(1 - α²)
    det_inv = 1.0 / np.maximum(1.0 - alpha_applied**2, 1e-6)

    # ── Step 2: apply per-frame analytical inverse — vectorised ────────────
    n_frames = max(1, (n - _PROC_NFFT) // _PROC_HOP + 1)
    # Trim so every frame fits inside the signal
    while n_frames > 0 and (n_frames - 1) * _PROC_HOP + _PROC_NFFT > n:
        n_frames -= 1

    left_out = np.zeros(n, dtype=np.float32)
    right_out = np.zeros(n, dtype=np.float32)
    win_sum = np.zeros(n, dtype=np.float32)

    if n_frames > 0:
        l_f32 = np.asarray(left, dtype=np.float32)
        r_f32 = np.asarray(right, dtype=np.float32)
        # Batch STFT via stride_tricks
        _fl_sw = np.lib.stride_tricks.sliding_window_view(l_f32, _PROC_NFFT)[::_PROC_HOP][:n_frames]
        _fr_sw = np.lib.stride_tricks.sliding_window_view(r_f32, _PROC_NFFT)[::_PROC_HOP][:n_frames]
        fl = np.fft.rfft(_fl_sw * window, axis=1)  # (T, F)
        fr = np.fft.rfft(_fr_sw * window, axis=1)
        # Analytical inverse — broadcast alpha_applied / det_inv over T
        fl_clean = (fl - alpha_applied * fr) * det_inv
        fr_clean = (fr - alpha_applied * fl) * det_inv
        frames_l = (np.fft.irfft(fl_clean, n=_PROC_NFFT, axis=1) * window).astype(np.float32)
        frames_r = (np.fft.irfft(fr_clean, n=_PROC_NFFT, axis=1) * window).astype(np.float32)
        win_sq = window**2
        for i in range(n_frames):
            s = i * _PROC_HOP
            left_out[s : s + _PROC_NFFT] += frames_l[i]
            right_out[s : s + _PROC_NFFT] += frames_r[i]
            win_sum[s : s + _PROC_NFFT] += win_sq

    win_sum = np.maximum(win_sum, 1e-8)
    left_out /= win_sum
    right_out /= win_sum

    left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
    right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)

    if channels_first:
        result = np.stack([left_out, right_out], axis=0)
    else:
        result = np.stack([left_out, right_out], axis=1)

    return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import (  # pylint: disable=wrong-import-position
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)


class CrosstalkCancellationPhase(PhaseInterface):
    """Phase 62: Analytical crosstalk cancellation for early stereo recordings.

    Implements the exact inverse of the vinyl groove α-mixing model rather than
    coherence-thresholded BSS, yielding predictable separation without coherence
    artefacts.
    """

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_62_crosstalk_cancellation",
            name="Crosstalk Cancellation",
            category=PhaseCategory.RESTORATION,
            priority=6,
            dependencies=["phase_14"],
            estimated_time_factor=0.05,
            version="2.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.65,
            description=(
                "Analytical inverse of the vinyl-groove constant-α crosstalk "
                "mixing model. Estimates α(f) from long-term L/R coherence, "
                "then applies the exact 2×2 matrix inverse per frequency bin."
            ),
        )

    @staticmethod
    def _compute_crosstalk_profile(
        material_key: str, quality_mode: str, restorability_score: float
    ) -> dict[str, float]:
        material = str(material_key or "unknown").strip().lower()
        mode = str(quality_mode or "balanced").strip().lower()
        mode = {"restoration": "balanced", "studio_2026": "maximum"}.get(mode, mode)

        if "vinyl" in material:
            min_score = 0.10
            alpha_max = 0.62
        elif "shellac" in material:
            min_score = 0.12
            alpha_max = 0.60
        elif any(token in material for token in ("cd_digital", "dat", "flac", "streaming")):
            min_score = 0.18
            alpha_max = 0.68
        else:
            min_score = 0.15
            alpha_max = 0.65

        rest_norm = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0)) / 100.0
        min_score += (rest_norm - 0.5) * 0.10
        alpha_max += (rest_norm - 0.5) * 0.08

        score_off, alpha_off = {
            "fast": (0.03, 0.03),
            "balanced": (0.00, 0.00),
            "quality": (-0.02, -0.03),
            "maximum": (-0.04, -0.05),
        }.get(mode, (0.0, 0.0))
        min_score += score_off
        alpha_max += alpha_off

        return {
            "min_crosstalk_score": float(np.clip(min_score, 0.05, 0.25)),
            "alpha_max": float(np.clip(alpha_max, 0.50, 0.70)),
        }

    @staticmethod
    def _local_event_strength(key: str, loc: tuple[float, float], event_metadata: dict[str, dict] | None) -> float:
        duration_s = max(0.0, float(loc[1]) - float(loc[0]))
        duration_factor = float(np.clip(duration_s / 0.90, 0.30, 1.0))
        key_factor = {
            "crosstalk": 1.0,
            "stereo_imbalance": 0.72,
            "phase_issues": 0.66,
            "azimuth_error": 0.58,
        }.get(key, 0.62)
        severity = 0.55
        confidence = 0.80
        meta_obj = (event_metadata or {}).get(key)
        if isinstance(meta_obj, dict):
            severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
            confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
        return float(np.clip(key_factor * (0.36 + 0.44 * severity + 0.20 * confidence) * duration_factor, 0.14, 1.0))

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

        keys = ("crosstalk", "stereo_imbalance", "phase_issues", "azimuth_error")
        mask = np.zeros(n_samples, dtype=np.float32)
        for key in keys:
            pad = int((0.080 if key == "crosstalk" else 0.050) * sample_rate)
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
                    strength = CrosstalkCancellationPhase._local_event_strength(key, loc, event_metadata)
                    mask[start:end] = np.maximum(mask[start:end], strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0

        smooth = max(16, int(0.030 * sample_rate))
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
        _pmgg_strength = float(kwargs.get("strength", 0.5))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        _profile_62 = self._compute_crosstalk_profile(
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
                    "crosstalk_score": float((_defect_scores or {}).get("crosstalk", 0.0)),
                    "strength": _pmgg_strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Crosstalk cancellation skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_crosstalk_score=_profile_62["min_crosstalk_score"],
            alpha_max=_profile_62["alpha_max"],
        )
        _n_samples62 = (
            audio.shape[1] if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2 else audio.shape[0]
        )
        _local_profile62, _local_coverage62 = self._build_locality_profile(
            int(_n_samples62),
            sample_rate,
            kwargs.get("defect_locations"),
            kwargs.get("defect_event_metadata"),
            self._collect_protected_zones(kwargs),
        )
        if _local_coverage62 > 0.0:
            result_audio = self._blend_with_locality(audio, result_audio, _local_profile62)
        elapsed = _time.perf_counter() - t0

        # V20 Mikrodynamik-Korrelation (§2.75): Crosstalk-Inversion darf voiced Frames
        # nicht in ihrer Frame-Energie degradieren (panns_singing ≥ 0.25).
        _panns62 = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if _panns62 >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (
                    frame_energy_correlation,  # pylint: disable=import-outside-toplevel
                    recommend_mikrodynamik_wet,
                )

                _corr62 = frame_energy_correlation(audio, result_audio, sample_rate, frame_ms=10.0)
                if _corr62 < 0.97:
                    _need62 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet62 = recommend_mikrodynamik_wet(_corr62, _panns62, global_need=_need62)
                    result_audio = (_wet62 * result_audio + (1.0 - _wet62) * audio).astype(np.float32)
                    logger.warning(
                        "Verarbeitungsschritt62 V20 Mikrodynamik-Korr=%.3f < 0.97 → wet=%.3f Blend",
                        _corr62,
                        _wet62,
                    )
            except Exception as _dyn62_exc:
                logger.debug("Verarbeitungsschritt62 V20 Mikrodynamik-Guard (nicht blockierend): %s", _dyn62_exc)

        # V26 Onset-Guard (§2.77): Matrix-Inversion darf Transient-Energie in Onset-
        # Fenstern (0–20 ms nach Transient) nicht um mehr als 1.5 dB verschieben.
        try:
            from backend.core.dsp.onset_guard import (
                apply_onset_protection_mask,  # pylint: disable=import-outside-toplevel
            )

            result_audio = apply_onset_protection_mask(audio, result_audio, None, max_delta_db=1.5)
        except Exception as _on62_exc:
            logger.debug("Verarbeitungsschritt62 V26 Onset-Guard (nicht blockierend): %s", _on62_exc)

        # §2.46f NPA-Guard: Atemgeräusche und Early-Reflections vor Crosstalk-Subtraktion schützen.
        try:
            from backend.core.natural_performance_detector import (
                get_natural_performance_detector,  # pylint: disable=import-outside-toplevel
            )

            _mono62 = audio.mean(axis=0) if audio.ndim == 2 else audio
            _npa_mask62 = (
                get_natural_performance_detector()
                .detect(_mono62, sample_rate)
                .get_protected_mask(len(_mono62), sample_rate)
            )
            if _npa_mask62 is not None and _npa_mask62.any():
                if result_audio.ndim == 2:
                    result_audio[:, _npa_mask62] = audio[:, _npa_mask62]
                else:
                    result_audio[_npa_mask62] = audio[_npa_mask62]
        except Exception as _npa62_exc:
            logger.debug("§2.46f Verarbeitungsschritt62 NPA-Guard (nicht blockierend): %s", _npa62_exc)

        # §2.62 Psychoakustischer Masking-Guard: Crosstalk-Subtraktion entfernt
        # keine vom Musiksignal maskierten Komponenten (G_floor ≥ 0.10).
        try:
            from backend.core.dsp.psychoacoustics import (
                apply_psychoacoustic_masking_clamp,  # pylint: disable=import-outside-toplevel
            )

            result_audio = apply_psychoacoustic_masking_clamp(audio, result_audio, sample_rate, mode="restoration")
        except Exception as _pmask62_exc:
            logger.debug("§2.62 Verarbeitungsschritt62 Masking-Guard (nicht blockierend): %s", _pmask62_exc)

        # §V19 Noise-Textur-Invariante (VERBOTEN-V19): Residual bewahrt Materialcharakter
        _mat62_str = str(material_type or "unknown").lower()
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt62_fn,
            )

            if result_audio.shape == audio.shape:
                _nt62_d = _nt62_fn(
                    audio.astype(np.float32) - result_audio.astype(np.float32), _mat62_str, sr=sample_rate
                )
                if _nt62_d > 0.25:
                    result_audio = (0.5 * result_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_62 noise_texture dist=%.3f > 0.25 → 50%%-Blend",
                        _nt62_d,
                    )
        except Exception as _nt62_exc:
            logger.debug(
                "§V19 (Spec-Vintage-Guard) Verarbeitungsschritt_62 noise_texture_guard (nicht blockierend): %s",
                _nt62_exc,
            )

        # §V24 Spektralfarbe-Prüfung (VERBOTEN-V24): 1/3-Oktav-Profil darf nicht verfärbt werden
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg62,
            )

            if result_audio.shape == audio.shape:
                _sc62 = _scg62(audio.astype(np.float32), result_audio.astype(np.float32), sample_rate)
                if not _sc62.ok:
                    result_audio = (0.70 * result_audio + 0.30 * audio).astype(np.float32)
        except Exception as _sc62_exc:
            logger.debug(
                "§V24 (Spec-Vintage-Guard) Verarbeitungsschritt_62 spectral_color_guard (nicht blockierend): %s",
                _sc62_exc,
            )

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "crosstalk_score": float((_defect_scores or {}).get("crosstalk", 0.0)),
                "strength": _pmgg_strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "crosstalk_profile": dict(_profile_62),
                "min_crosstalk_score": float(_profile_62["min_crosstalk_score"]),
                "alpha_max": float(_profile_62["alpha_max"]),
                "repair_locality_coverage": round(float(_local_coverage62), 6),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )


# EOF
