"""
Phase 59 — Modulation Noise Reduction.

Signal-dependent modulation noise (unique to analog tape) where noise level
tracks the signal envelope.  Unlike stationary tape hiss (phase_29), modulation
noise requires signal-adaptive noise gating that varies its threshold with the
signal level.

Algorithm (Esquef & Biscainho 2006):
1. Estimate signal envelope (10 ms RMS)
2. Estimate noise floor as a function of signal level
3. Apply signal-dependent spectral gating: G(f) = max(G_floor, 1 - alpha * N(f)/S(f))
   where N(f) is estimated noise at frequency f given current signal level
4. Wet/dry blend with strength parameter

Scientific basis: Esquef & Biscainho (2006), Czyzewski et al. (2020).
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import scipy.signal as sps

from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

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
    # Vectorized: reshape → (n_frames, 480), per-frame RMS in one pass — no Python loop
    _frames = _mono[: _n_frames * _frame].reshape(_n_frames, _frame)
    _frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(_frames**2, axis=1)) + 1e-10)
    _mask = _frame_rms_db > -50.0
    if not np.any(_mask):
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(_frames[_mask] ** 2)) + 1e-10))


_MIN_MODULATION_NOISE_SCORE: float = 0.10
_G_FLOOR: float = 0.10  # Minimum spectral gain — §2.62: VERBOTEN < 0.10


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.7,
    defect_scores: dict | None = None,
    min_modulation_noise_score: float = _MIN_MODULATION_NOISE_SCORE,
    g_floor: float = _G_FLOOR,
) -> np.ndarray:
    """Haupt-entry point for Phase 59.

    Args:
        audio:        Input audio float32, mono or stereo
        sample_rate:  Must be 48000 Hz
        strength:     Processing strength 0–1
        defect_scores: Defect scan scores (optional, for gating)

    Returns:
        Processed audio, same shape as input
    """
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Gate: only process if modulation noise is detected
    if defect_scores is not None:
        mn_score = float(defect_scores.get("modulation_noise", 0.0))
        if mn_score < min_modulation_noise_score:
            logger.debug(
                "Verarbeitungsschritt 59: modulation_noise Wert %.3f < %.3f — uebersprungen",
                mn_score,
                min_modulation_noise_score,
            )
            return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 Linked-Stereo: Noise-Modell aus Mid, identischer STFT-Gain auf L+R
        mono_mix = (audio[0] + audio[1]) / 2.0
        mono_denoised = apply(mono_mix, sample_rate, strength=strength, defect_scores=defect_scores)
        _eps_mn = 1e-10
        _gain_mn = np.where(
            np.abs(mono_mix) > _eps_mn,
            mono_denoised / (mono_mix + _eps_mn * np.sign(mono_mix + _eps_mn)),
            1.0,
        )
        _gain_mn = np.clip(_gain_mn, 0.0, 10.0)
        return np.clip(np.stack([audio[0] * _gain_mn, audio[1] * _gain_mn], axis=0), -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    x = np.asarray(audio, dtype=np.float32)
    n = len(x)

    # STFT parameters
    n_fft = 2048
    hop = n_fft // 4
    _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103

    # Vectorized STFT via scipy — replaces Python frame-loop (~21 000 iterations for 225 s)
    _, _, stft = sps.stft(x, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=_noverlap, boundary="even")
    stft = stft.astype(np.complex64)

    mag = np.abs(stft).astype(np.float32)
    phase = np.angle(stft).astype(np.float32)

    # Signal envelope (per-frame RMS)
    frame_rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)

    # Estimate noise model: noise_level(f) = alpha * signal_level
    # Use low-energy frames to calibrate the noise/signal ratio
    noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)

    # Signal-dependent noise estimate per frame
    alpha = float(np.clip(strength * 0.8, 0.1, 0.9))
    noise_estimate = noise_floor * (frame_rms / (np.median(frame_rms) + 1e-12))[np.newaxis, :]

    # Spectral gating: reduce noise proportional to signal level
    gain = np.maximum(g_floor, 1.0 - alpha * noise_estimate / (mag + 1e-12))
    gain = np.clip(gain, g_floor, 1.0).astype(np.float32)

    # §2.62 Psychoakustischer Masking-Guard (ISO 11172-3) — per-Band Floor (non-blocking)
    try:
        from backend.core.dsp.psychoacoustics import (
            compute_masking_threshold_iso11172 as _cmask_p59,  # pylint: disable=import-outside-toplevel
        )

        _mask_ratio_p59 = _cmask_p59(x, sample_rate, n_fft=n_fft, hop_length=hop)
        _mfloor_p59 = np.mean(_mask_ratio_p59, axis=1).astype(np.float32)  # (n_freq,)
        _mfreqs_p59 = np.linspace(0.0, sample_rate / 2.0, _mask_ratio_p59.shape[0], dtype=np.float32)
        _mfloor_interp59 = np.interp(
            np.linspace(0.0, sample_rate / 2.0, gain.shape[0], dtype=np.float32), _mfreqs_p59, _mfloor_p59
        ).astype(np.float32)
        gain = np.maximum(gain, _mfloor_interp59[:, np.newaxis])
    except Exception as e:
        logger.warning("Verarbeitungsschritt_59_modulation_noise_reduction.py::unbekannter Ersatzpfad: %s", e)
        pass  # nie pipeline-blockierend

    # Apply gain and reconstruct
    stft_clean = (mag * gain * np.exp(1j * phase)).astype(np.complex64)

    # Vectorized ISTFT via scipy — replaces Python OLA-loop (~21 000 iterations for 225 s)
    _, out_full = sps.istft(
        stft_clean, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=True
    )
    out_full = np.asarray(out_full, dtype=np.float32)
    out = out_full[:n] if len(out_full) >= n else np.pad(out_full, (0, n - len(out_full)))

    # Wet/dry blend
    result = x * (1.0 - strength) + out * strength
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


class ModulationNoiseReductionPhase(PhaseInterface):
    """Phase 59: Signal-dependent modulation noise reduction for analog tape."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_59_modulation_noise_reduction",
            name="Modulation Noise Reduction",
            category=PhaseCategory.RESTORATION,
            priority=6,
            dependencies=["phase_03"],
            estimated_time_factor=0.05,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.65,
            description=(
                "Signal-dependent modulation noise reduction using adaptive spectral "
                "gating that tracks signal envelope. Targets noise that varies with "
                "signal level (unique to analog tape recordings)."
            ),
        )

    @staticmethod
    def _compute_modulation_noise_profile(
        material_key: str,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive modulation-noise profile."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("tape", "reel_tape", "cassette")):
            min_modulation_noise_score = 0.10
            g_floor = 0.10  # §2.62 hard min
        elif any(token in _material for token in ("cd_digital", "dat", "streaming", "flac")):
            min_modulation_noise_score = 0.18
            g_floor = 0.12
        else:
            min_modulation_noise_score = 0.14
            g_floor = 0.10

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        min_modulation_noise_score += (_rest_norm - 0.5) * 0.10
        g_floor += (0.5 - _rest_norm) * 0.08

        _mode_offsets = {
            "fast": (0.04, 0.03),
            "balanced": (0.0, 0.0),
            "quality": (-0.03, -0.02),
            "maximum": (-0.05, -0.03),
        }
        _score_off, _g_off = _mode_offsets.get(_mode, (0.0, 0.0))
        min_modulation_noise_score += _score_off
        g_floor += _g_off

        return {
            "min_modulation_noise_score": float(np.clip(min_modulation_noise_score, 0.05, 0.25)),
            "g_floor": float(np.clip(g_floor, 0.05, 0.30)),  # §2.62: quality_mode darf < 0.10 (bis 0.05)
        }

    @staticmethod
    def _local_event_strength(key: str, loc: tuple[float, float], event_metadata: dict[str, dict] | None) -> float:
        duration_s = max(0.0, float(loc[1]) - float(loc[0]))
        duration_factor = float(np.clip(duration_s / 1.20, 0.30, 1.0))
        key_factor = {
            "modulation_noise": 1.0,
            "signal_dependent_noise": 0.90,
            "nr_breathing_artifact": 0.72,
            "tape_noise_modulation": 0.95,
        }.get(str(key).strip().lower(), 0.80)
        severity = 0.55
        confidence = 0.75
        meta_obj = (event_metadata or {}).get(key) or (event_metadata or {}).get(str(key).strip().lower())
        if isinstance(meta_obj, dict):
            severity = float(np.clip(float(meta_obj.get("severity", severity)), 0.0, 1.0))
            confidence = float(np.clip(float(meta_obj.get("confidence", confidence)), 0.0, 1.0))
        return float(np.clip(key_factor * (0.35 + 0.45 * severity + 0.20 * confidence) * duration_factor, 0.12, 1.0))

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
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 1.0

        keys = ("modulation_noise", "signal_dependent_noise", "nr_breathing_artifact", "tape_noise_modulation")
        mask = np.zeros(n_samples, dtype=np.float32)
        pad = int(0.10 * sample_rate)
        for key in keys:
            for loc in defect_locations.get(key, []) or []:
                try:
                    start_s, end_s = float(loc[0]), float(loc[1])
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue
                s = int(max(0.0, start_s) * sample_rate)
                e = int(max(0.0, end_s) * sample_rate)
                s = max(0, s - pad)
                e = min(n_samples, e + pad)
                if e > s:
                    strength = ModulationNoiseReductionPhase._local_event_strength(key, loc, event_metadata)
                    mask[s:e] = np.maximum(mask[s:e], strength)
        if not np.any(mask):
            return np.ones(n_samples, dtype=np.float32), 1.0

        smooth = max(16, int(0.04 * sample_rate))
        mask = np.convolve(mask, np.ones(smooth, dtype=np.float32) / float(smooth), mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if protected_zones:
            for start_s, end_s, cap in protected_zones:
                s = int(max(0.0, float(start_s)) * sample_rate)
                e = int(max(0.0, float(end_s)) * sample_rate)
                if e > s:
                    mask[s : min(n_samples, e)] = np.minimum(mask[s : min(n_samples, e)], float(cap))
        return mask, float(np.mean(mask))

    @staticmethod
    def _blend_with_locality(audio: np.ndarray, candidate: np.ndarray, profile: np.ndarray) -> np.ndarray:
        if profile.size == 0 or profile.size == 1:
            return candidate
        if audio.ndim == 2 and audio.shape[0] == profile.size:
            profile_2d = profile[:, np.newaxis]
        elif audio.ndim == 2 and audio.shape[-1] == profile.size:
            profile_2d = profile[np.newaxis, :]
        else:
            profile_2d = profile
        blended: np.ndarray = np.asarray(np.clip(audio + profile_2d * (candidate - audio), -1.0, 1.0), dtype=np.float32)
        return blended

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="59")
        check_ml_model_ready("Whisper", phase_name="59")
        sample_rate = kwargs.get("sample_rate", sample_rate)
        strength = float(kwargs.get("strength", 0.7))
        defect_scores: dict | None = kwargs.get("defect_scores")
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        _profile_59 = self._compute_modulation_noise_profile(
            str(kwargs.get("material_type", kwargs.get("material", "unknown"))).lower(),
            kwargs.get("quality_mode"),
            float(kwargs.get("restorability_score", 50.0)),
        )

        # §v10.200 Depth-adaptive: tiefere Ketten brauchen höheren g_floor
        # und sensitiveren Detection-Threshold (mehr Modulationsrauschen)
        _td_p59 = max(1, int(kwargs.get("transfer_chain_depth", 1)))
        if _td_p59 >= 4:
            _profile_59["g_floor"] = float(np.clip(_profile_59["g_floor"] + 0.08, 0.10, 0.35))
            _profile_59["min_modulation_noise_score"] = float(
                np.clip(_profile_59["min_modulation_noise_score"] - 0.05, 0.05, 0.25)
            )
        elif _td_p59 >= 3:
            _profile_59["g_floor"] = float(np.clip(_profile_59["g_floor"] + 0.04, 0.10, 0.30))
            _profile_59["min_modulation_noise_score"] = float(
                np.clip(_profile_59["min_modulation_noise_score"] - 0.03, 0.05, 0.25)
            )

        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", strength))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V40 NMR-Feedback: NR-Stärke adaptiv anpassen (FeedbackChain-aware).
        try:
            from backend.core.dsp.nmr_feedback import (
                compute_nmr_score as _nmr_fn_59,  # pylint: disable=import-outside-toplevel
            )

            _nmr_result_59 = _nmr_fn_59(audio, sample_rate)
            if not _nmr_result_59.ok:
                logger.warning(
                    "Verarbeitungsschritt59 §V40 NMR: nmr_above_masking → §2.45 Minimal-Intervention prüfen",
                )
            _effective_strength = float(
                np.clip(
                    _effective_strength + _nmr_result_59.recommended_nr_strength_delta,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "Verarbeitungsschritt59 §V40 NMR: delta=%.3f → eff_str=%.3f",
                _nmr_result_59.recommended_nr_strength_delta,
                _effective_strength,
            )
        except Exception as _nmr_exc_59:  # pylint: disable=broad-except
            logger.debug("Verarbeitungsschritt59 §V40 NMR nicht blockierend: %s", _nmr_exc_59)

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "modulation_noise_score": float((_defect_scores or {}).get("modulation_noise", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Modulation noise reduction skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_modulation_noise_score=_profile_59["min_modulation_noise_score"],
            g_floor=_profile_59["g_floor"],
        )
        _n_profile_samples = audio.shape[-1] if audio.ndim == 2 and audio.shape[0] == 2 else audio.shape[0]
        _locality_profile, _locality_coverage = self._build_locality_profile(
            n_samples=int(_n_profile_samples),
            sample_rate=sample_rate,
            defect_locations=kwargs.get("defect_locations"),
            event_metadata=kwargs.get("defect_event_metadata"),
            protected_zones=self._collect_protected_zones(kwargs),
        )
        result_audio = self._blend_with_locality(audio, result_audio, _locality_profile)
        elapsed = _time.perf_counter() - t0

        # §4.5 Psychoacoustic Masking Clamp — only reduce audible modulation noise
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
            logger.debug("Verarbeitungsschritt59 masking clamp nicht blockierend: %s", _pm_exc)

        # §2.36 Phonem-Schutz: Plosiv-/Frikativ-Frames via get_phoneme_mask() schützen.
        # Das signal-adaptive Spektral-Gating dämpft Konsonanten-Bursts wie Modulationsrauschen.
        try:
            from backend.core.lyrics_guided_enhancement import (
                get_phoneme_mask as _get_pmask_59,  # pylint: disable=import-outside-toplevel
            )

            _hop_59 = 512
            _mono_59: np.ndarray
            if audio.ndim == 2:
                _mono_59 = np.mean(audio, axis=0) if audio.shape[0] == 2 else np.mean(audio, axis=1)
            else:
                _mono_59 = audio
            _pmask_59 = _get_pmask_59(_mono_59.astype(np.float32), sample_rate, hop_length=_hop_59)
            if np.any(_pmask_59):
                _n59 = len(_mono_59)
                _smask_59 = np.zeros(_n59, dtype=bool)
                for _fi59, _fp59 in enumerate(_pmask_59):
                    if _fp59:
                        _fs59 = _fi59 * _hop_59
                        _fe59 = min(_n59, _fs59 + _hop_59)
                        _smask_59[_fs59:_fe59] = True
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _smask_59] = audio[:, _smask_59]
                    else:
                        result_audio[_smask_59, :] = audio[_smask_59, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_smask_59] = audio[_smask_59]
                logger.debug(
                    "§2.36 Verarbeitungsschritt_59 Phonem-Schutz: %d/%d Frames restauriert",
                    int(np.sum(_pmask_59)),
                    len(_pmask_59),
                )
        except Exception as _pmask59_exc:
            logger.debug("§2.36 Verarbeitungsschritt_59 Phonem-Mask (nicht blockierend): %s", _pmask59_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — Atemgeräusche und Vibrato schützen.
        # Das Gating dämpft Atemgeräusche zwischen Phrasen als "niedrig-pegel Rauschen".
        try:
            from backend.core.natural_performance_detector import (
                get_natural_performance_detector,  # pylint: disable=import-outside-toplevel
            )

            _npa_a59 = audio
            if _npa_a59.ndim == 2 and _npa_a59.shape[0] == 2 and _npa_a59.shape[1] > _npa_a59.shape[0]:
                _npa_a59 = _npa_a59.T
            _npa_r59 = get_natural_performance_detector().detect(_npa_a59, sample_rate)
            _npa_n59 = (
                result_audio.shape[1]
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else result_audio.shape[0]
            )
            _npa_m59 = _npa_r59.get_protected_mask(_npa_n59, sample_rate)
            if np.any(_npa_m59):
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _npa_m59] = audio[:, _npa_m59]
                    elif result_audio.shape == audio.shape:
                        result_audio[_npa_m59, :] = audio[_npa_m59, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_npa_m59] = audio[_npa_m59]
                logger.debug(
                    "§2.46f Verarbeitungsschritt_59 NPA: %d protected samples restauriert", int(np.sum(_npa_m59))
                )
        except Exception as _npa59_exc:
            logger.debug("§2.46f Verarbeitungsschritt_59 NPA-Guard (nicht blockierend): %s", _npa59_exc)

        # §0p/V19/V20/V21/V26/§2.72 Vokal- + Textur-Guards nach Modulations-NR (RELEASE_MUST §0p V19-V26)
        _p59_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        _mat59_guards = str(kwargs.get("material_type", kwargs.get("material", "unknown")) or "unknown").lower()
        if _p59_panns >= 0.25:
            try:
                from backend.core.dsp.hnr_guard import apply_hnr_blend as _apply_hnr_59  # pylint: disable=import-outside-toplevel  # noqa: I001

                _hnr_blended_59, _hnr_diag_59 = _apply_hnr_59(
                    audio.astype(np.float32), result_audio.astype(np.float32), sample_rate
                )
                if _hnr_diag_59.get("over_cleaned"):
                    result_audio = _hnr_blended_59
            except Exception as _hnr_59_exc:
                logger.debug("§0p HNR-Blend Verarbeitungsschritt_59 (nicht blockierend): %s", _hnr_59_exc)

        _nt59_residual = audio - result_audio
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt59_dist_fn,
            )

            if _nt59_residual.shape == audio.shape:
                _nt59_d = _nt59_dist_fn(_nt59_residual, _mat59_guards, sr=sample_rate)
                if _nt59_d > 0.25:
                    result_audio = (0.5 * result_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§V19 Verarbeitungsschritt_59: noise_texture_dist=%.3f > 0.25 → 50%% dry-blend", _nt59_d
                    )
        except Exception as _nt59_exc:
            logger.debug("§V19 Verarbeitungsschritt_59 noise_texture nicht blockierend: %s", _nt59_exc)

        if _p59_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (  # pylint: disable=import-outside-toplevel
                    frame_energy_correlation as _fec59,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr59 = _fec59(audio, result_audio, sample_rate, frame_ms=10.0)
                if _corr59 < 0.97:
                    _need59 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet59 = _recommend_mkk_wet(_corr59, _p59_panns, global_need=_need59)
                    result_audio = (_wet59 * result_audio + (1.0 - _wet59) * audio).astype(np.float32)
                    logger.warning(
                        "§V20 Verarbeitungsschritt_59: mikrodynamik_corr=%.4f < 0.97 → wet=%.3f", _corr59, _wet59
                    )
            except Exception as _v20_59_exc:
                logger.debug("§V20 Verarbeitungsschritt_59 mikrodynamik nicht blockierend: %s", _v20_59_exc)

        if any(x in _mat59_guards for x in ("shellac", "vinyl", "tape", "analog")):
            try:
                from backend.core.dsp.noise_floor_guard import (  # pylint: disable=import-outside-toplevel
                    apply_noise_floor_minimum as _nfmin59,
                )

                result_audio = _nfmin59(result_audio, sample_rate, _mat59_guards, original_audio=audio)
            except Exception as _v21_59_exc:
                logger.debug("§V21 Verarbeitungsschritt_59 noise_floor nicht blockierend: %s", _v21_59_exc)

        # §V24 Spektralfarbe-Prüfung nach NR (§2.74, non-blocking WARNING)
        try:
            from backend.core.dsp.spectral_color_guard import (  # pylint: disable=import-outside-toplevel
                check_spectral_color_preservation as _scg_59,
            )

            _sc_result_59 = _scg_59(audio, result_audio, sample_rate)
            if not _sc_result_59.ok:
                _sc_wet_59 = 0.70  # Phase-Strength −30 % (§V24)
                result_audio = (_sc_wet_59 * result_audio + (1.0 - _sc_wet_59) * audio).astype(np.float32)
        except Exception as _sc_exc_59:  # pylint: disable=broad-except
            logger.debug("§V24 Verarbeitungsschritt_59 spectral_color nicht blockierend: %s", _sc_exc_59)

        try:
            from backend.core.dsp.onset_guard import (  # pylint: disable=import-outside-toplevel
                apply_onset_protection_mask as _opm59,
            )

            result_audio = _opm59(audio, result_audio, None, max_delta_db=1.5)
        except Exception as _v26_59_exc:
            logger.debug("§V26 Verarbeitungsschritt_59 onset_guard nicht blockierend: %s", _v26_59_exc)

        if _p59_panns >= 0.25:
            try:
                from backend.core.dsp.vibrato_guard import (  # pylint: disable=import-outside-toplevel
                    check_vibrato_depth_preservation as _vib59_fn,
                )

                _vibr59 = _vib59_fn(audio, result_audio, sample_rate)
                if not _vibr59.ok:
                    result_audio = (0.5 * result_audio + 0.5 * audio).astype(np.float32)
                    logger.warning(
                        "§2.72 Verarbeitungsschritt_59: vibrato_reduction=%.1f%% → 50%% dry-blend",
                        _vibr59.depth_reduction_pct,
                    )
            except Exception as _vib59_exc:
                logger.debug("§2.72 Verarbeitungsschritt_59 vibrato nicht blockierend: %s", _vib59_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "modulation_noise_score": float((_defect_scores or {}).get("modulation_noise", 0.0)),
                "strength": strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "modulation_noise_profile": dict(_profile_59),
                "min_modulation_noise_score": float(_profile_59["min_modulation_noise_score"]),
                "g_floor": float(_profile_59["g_floor"]),
                "phase_locality_factor": phase_locality_factor,
                "repair_locality_coverage": float(_locality_coverage),
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )
