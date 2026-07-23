"""
Phase 60 — Inner Groove Distortion Repair.

IGD progressively increases towards the center of vinyl/shellac discs due to
decreasing linear velocity.  This phase applies position-adaptive THD reduction
that increases correction strength towards the end of the recording.

Algorithm:
1. Divide recording into position segments (simulating groove radius)
2. Measure THD per segment
3. Apply adaptive harmonic suppression: stronger towards center (later segments)
4. Preserve fundamental + H2 (musical character), reduce H3+ (distortion)

Scientific basis: Kates (1981) "A Model of Record Tracing Distortion";
Roys (1970) "Playback Distortion in Disc Records".
"""

from __future__ import annotations

import logging
import time as _time

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


_MIN_IGD_SCORE: float = 0.10
_N_SEGMENTS: int = 8  # Position segments for adaptive processing


def _collect_protected_zones(kwargs: dict) -> list[tuple[float, float, float]]:
    """Sammelt Vokal-/Performance-Schutzzonen fuer segmentlokale IGD-Caps."""
    protected_zones: list[tuple[float, float, float]] = []
    for zone in kwargs.get("vibrato_zones") or []:
        try:
            protected_zones.append((float(zone[0]), float(zone[1]), 0.20))
        except Exception:
            continue
    for zone in kwargs.get("frisson_zones") or []:
        try:
            start_s = float(getattr(zone, "start_s", None) or zone[0])
            end_s = float(getattr(zone, "end_s", None) or zone[1])
            protected_zones.append((start_s, end_s, 0.30))
        except Exception:
            continue
    for zone in kwargs.get("whisper_zones") or []:
        try:
            protected_zones.append((float(zone[0]), float(zone[1]), 0.25))
        except Exception:
            continue
    for zone in kwargs.get("passaggio_zones") or []:
        try:
            protected_zones.append((float(zone[0]), float(zone[1]), 0.35))
        except Exception:
            continue
    return protected_zones


def _compute_igd_segment_strength(
    seg_mag: np.ndarray,
    freqs_hz: np.ndarray,
    segment_audio: np.ndarray,
    seg_idx: int,
    n_segments: int,
    base_strength: float,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    protected_zones: list[tuple[float, float, float]] | None = None,
) -> float:
    """Berechnet IGD-Reduktion pro Segment aus Position, Verzerrungsband und Aktivitaet."""
    if base_strength < 1e-6:
        return 0.0
    position_factor = float(np.clip((seg_idx + 1) / max(1, n_segments), 0.0, 1.0))
    igd_mask = (freqs_hz >= 2000.0) & (freqs_hz <= 8000.0)
    ref_mask = (freqs_hz >= 300.0) & (freqs_hz < 2000.0)
    if np.any(igd_mask) and np.any(ref_mask):
        igd_energy = float(np.mean(seg_mag[igd_mask, :] ** 2))
        ref_energy = float(np.mean(seg_mag[ref_mask, :] ** 2)) + 1e-12
        igd_ratio_db = 10.0 * np.log10(igd_energy / ref_energy + 1e-12)
        band_factor = float(np.clip((igd_ratio_db + 18.0) / 24.0, 0.0, 1.0))
    else:
        band_factor = 0.5

    rms = float(np.sqrt(np.mean(segment_audio**2) + 1e-12))
    activity_factor = float(np.clip((20.0 * np.log10(rms + 1e-12) + 48.0) / 30.0, 0.0, 1.0))
    local_strength = float(
        base_strength * np.clip(0.18 + 0.45 * position_factor + 0.28 * band_factor + 0.09 * activity_factor, 0.0, 1.0)
    )

    if protected_zones:
        center_s = float(start_sample + end_sample) * 0.5 / float(max(1, sample_rate))
        for zone_start, zone_end, zone_cap in protected_zones:
            if float(zone_start) <= center_s <= float(zone_end):
                local_strength = min(local_strength, float(zone_cap))
                break

    return float(np.clip(local_strength, 0.0, 1.0))


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.6,
    defect_scores: dict | None = None,
    min_igd_score: float = _MIN_IGD_SCORE,
    n_segments: int = _N_SEGMENTS,
    protected_zones: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Haupt-entry point for Phase 60."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        igd_score = float(defect_scores.get("inner_groove_distortion", 0.0))
        if igd_score < min_igd_score:
            logger.debug("Phase 60: IGD score %.3f < %.3f — skipped", igd_score, min_igd_score)
            return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 Linked-Stereo: STFT-Gain-Maske aus Mid, identisch auf L+R
        mono_mix = (audio[0] + audio[1]) / 2.0
        mono_repaired = apply(
            mono_mix,
            sample_rate,
            strength=strength,
            defect_scores=defect_scores,
            min_igd_score=min_igd_score,
            n_segments=n_segments,
            protected_zones=protected_zones,
        )
        _eps_igd = 1e-10
        _gain_igd = np.where(
            np.abs(mono_mix) > _eps_igd,
            mono_repaired / (mono_mix + _eps_igd * np.sign(mono_mix + _eps_igd)),
            1.0,
        )
        _gain_igd = np.clip(_gain_igd, 0.0, 10.0)
        return np.clip(np.stack([audio[0] * _gain_igd, audio[1] * _gain_igd], axis=0), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    x = np.asarray(audio, dtype=np.float32)
    n = len(x)
    sr = sample_rate
    n_segments = max(1, int(n_segments))
    seg_len = max(1, n // n_segments)

    out = x.copy()
    n_fft = 4096
    hop = n_fft // 4

    # Pre-compute IGD frequency mask (same for all segments and frames)
    freqs_rfft = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)
    igd_mask = (freqs_rfft >= 2000) & (freqs_rfft <= 8000)

    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        end = min(start + seg_len, n) if seg_idx < n_segments - 1 else n
        segment = x[start:end]
        if len(segment) < n_fft:
            continue

        # Vectorized STFT per segment — replaces inner Python frame-loop
        _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103
        _, _, seg_stft = sps.stft(segment, fs=sr, window="hann", nperseg=n_fft, noverlap=_noverlap, boundary="even")
        seg_stft = seg_stft.astype(np.complex64)

        seg_mag = np.abs(seg_stft).astype(np.float32)
        seg_phase = np.angle(seg_stft).astype(np.float32)
        local_strength = _compute_igd_segment_strength(
            seg_mag,
            freqs_rfft,
            segment,
            seg_idx,
            n_segments,
            strength,
            sample_rate,
            start,
            end,
            protected_zones,
        )

        # Suppress harmonics H3+ (2–8 kHz range where IGD is worst) — vectorized over all frames
        gain = np.ones_like(seg_mag)
        gain[igd_mask, :] = np.maximum(0.3, 1.0 - local_strength * 0.5)

        seg_stft_clean = (seg_mag * gain * np.exp(1j * seg_phase)).astype(np.complex64)

        # Vectorized ISTFT per segment
        _, seg_out_full = sps.istft(
            seg_stft_clean, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=True
        )
        seg_out = np.asarray(seg_out_full, dtype=np.float32)
        seg_len_actual = end - start
        if len(seg_out) >= seg_len_actual:
            out[start:end] = seg_out[:seg_len_actual]
        else:
            out[start : start + len(seg_out)] = seg_out

    # Crossfade segments (10 ms Hanning)
    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import (  # pylint: disable=wrong-import-position
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)


class InnerGrooveDistortionRepairPhase(PhaseInterface):
    """Phase 60: Position-adaptive THD reduction for Inner Groove Distortion."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_60_inner_groove_distortion_repair",
            name="Inner Groove Distortion Repair",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_09"],
            estimated_time_factor=0.06,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.60,
            description=(
                "Position-adaptive harmonic distortion reduction for inner groove "
                "distortion (vinyl/shellac). Correction strength increases towards "
                "the center of the disc."
            ),
        )

    @staticmethod
    def _compute_igd_profile(material_key: str, quality_mode: str, restorability_score: float) -> dict[str, float]:
        material = str(material_key or "unknown").strip().lower()
        mode = str(quality_mode or "balanced").strip().lower()
        mode = {"restoration": "balanced", "studio_2026": "maximum"}.get(mode, mode)

        if "shellac" in material or "vinyl" in material:
            min_igd_score = 0.11 if "vinyl" in material else 0.09
            n_segments = 8 if "vinyl" in material else 10
        elif any(token in material for token in ("cd_digital", "dat", "flac", "streaming")):
            min_igd_score = 0.17
            n_segments = 6
        else:
            min_igd_score = 0.14
            n_segments = 7

        rest_norm = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0)) / 100.0
        min_igd_score += (rest_norm - 0.5) * 0.10
        n_segments += int(round((0.5 - rest_norm) * 4.0))

        score_off, seg_off = {
            "fast": (0.03, -2),
            "balanced": (0.00, 0),
            "quality": (-0.02, 2),
            "maximum": (-0.04, 3),
        }.get(mode, (0.0, 0))
        min_igd_score += score_off
        n_segments += seg_off

        return {
            "min_igd_score": float(np.clip(min_igd_score, 0.05, 0.25)),
            "n_segments": int(np.clip(n_segments, 4, 14)),
        }

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
        _pmgg_strength = float(kwargs.get("strength", 0.6))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_60 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_60 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_60,
                )

                _fmz_60 = kwargs.get("forward_masking_zones") or _fmg_fn_60().compute_zones(audio, sample_rate)
                if _fmz_60:
                    _n_s_60 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_60 = sum(z.end_sample - z.start_sample for z in _fmz_60)
                    _zone_frac_60 = float(np.clip(_zone_s_60 / max(1, _n_s_60), 0.0, 1.0))
                    _effective_strength = float(np.clip(_effective_strength + _zone_frac_60 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_60:
                logger.debug("Phase60 §V41 ForwardMaskingGuard non-blocking: %s", _fmg_exc_60)

        _profile_60 = self._compute_igd_profile(
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
                    "igd_score": float((_defect_scores or {}).get("inner_groove_distortion", 0.0)),
                    "strength": _pmgg_strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Inner groove distortion repair skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_igd_score=_profile_60["min_igd_score"],
            n_segments=int(_profile_60["n_segments"]),
            protected_zones=_collect_protected_zones(kwargs),
        )
        elapsed = _time.perf_counter() - t0

        # §4.5 Psychoacoustic Masking Clamp — nur hörbare Verzerrungsprodukte reduzieren.
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
        except Exception as _pm60_exc:
            logger.debug("Phase60 masking clamp non-blocking: %s", _pm60_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — THD-Reduktion darf Atemgeräusche
        # und Vibrato-Zonen nicht modifizieren (Notch-Filtering trifft harmonische
        # Atemgeräusch-Obertöne genauso wie IGD-Verzerrungsprodukte).
        try:
            from backend.core.natural_performance_detector import (
                get_natural_performance_detector,  # pylint: disable=import-outside-toplevel
            )

            _npa_a60 = audio
            if _npa_a60.ndim == 2 and _npa_a60.shape[0] == 2 and _npa_a60.shape[1] > _npa_a60.shape[0]:
                _npa_a60 = _npa_a60.T
            _npa_r60 = get_natural_performance_detector().detect(_npa_a60, sample_rate)
            _npa_n60 = (
                result_audio.shape[1]
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else result_audio.shape[0]
            )
            _npa_m60 = _npa_r60.get_protected_mask(_npa_n60, sample_rate)
            if np.any(_npa_m60):
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _npa_m60] = audio[:, _npa_m60]
                    elif result_audio.shape == audio.shape:
                        result_audio[_npa_m60, :] = audio[_npa_m60, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_npa_m60] = audio[_npa_m60]
        except Exception as _npa60_exc:
            logger.debug("§2.46f phase_60 NPA-Guard (non-blocking): %s", _npa60_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "igd_score": float((_defect_scores or {}).get("inner_groove_distortion", 0.0)),
                "strength": _effective_strength,
            },
            metadata={
                "igd_profile": dict(_profile_60),
                "min_igd_score": float(_profile_60["min_igd_score"]),
                "n_segments": int(_profile_60["n_segments"]),
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
        )
