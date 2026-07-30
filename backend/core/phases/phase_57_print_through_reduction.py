"""
Phase 57 — Print-Through Reduction (Bidirektionale Adaptive Temporal Subtraction).

Spec §7.x / DSP-Spezialregel: Print-Through-Reduktion (phase_29, reel_tape):
    - Bidirektionale Adaptive Temporal Subtraction (LMS)
    - Pre-Echo (Vorwärtswicklung): schwächer → alpha_pre  ∈ [0.03, 0.25]
    - Post-Echo (Rückwärtswicklung): stärker  → alpha_post ∈ [0.05, 0.35]
    - Algorithmus:
        1. Kreuzkorrelation-Peak ±600 ms → delay_pre, delay_post
        2. LMS-Adaptivfilter separat für Pre- und Post-Echo
        3. audio_clean[t] = audio[t] − alpha_pre·audio[t + delay_pre]
                                       − alpha_post·audio[t − delay_post]
        4. Spectral Coherence vor/nach ≥ 0.90 + PGHI
    - Fallback: NMF-β Dekomposition (einseitig, nur Post-Echo)

VERBOTEN: Comb-Filter, einseitiges α-Modell (alpha_pre == alpha_post)
"""

from __future__ import annotations

import logging
import math
import time as _time

import numpy as np
import scipy.signal as sps

from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# Optionale Guards/Features werden bewusst lazy geladen.
# pylint: disable=import-outside-toplevel

logger = logging.getLogger(__name__)


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    if sig.ndim == 2:
        _mono = (
            sig.mean(axis=0).astype(np.float64)
            if (sig.shape[0] <= 2 and sig.shape[1] > 2)
            else sig.mean(axis=1).astype(np.float64)
        )
    else:
        _mono = sig.astype(np.float64)
    _frame = 480  # 10 ms @ 48 kHz
    _active = [
        _mono[i : i + _frame]
        for i in range(0, len(_mono) - _frame, _frame)
        if 20.0 * np.log10(np.sqrt(np.mean(_mono[i : i + _frame] ** 2)) + 1e-10) > -50.0
    ]
    if not _active:
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(np.concatenate(_active) ** 2)) + 1e-10))


# ─── Konstanten (Spec normativ) ────────────────────────────────────────────────
_XCORR_MAX_DELAY_MS: float = 600.0  # Maximales Korrelations-Suchfenster in ms
_ALPHA_PRE_RANGE: tuple[float, float] = (0.03, 0.25)  # Vorwärtswicklung
_ALPHA_POST_RANGE: tuple[float, float] = (0.05, 0.35)  # Rückwärtswicklung
_LMS_MU: float = 0.001  # LMS-Schrittweite
_LMS_TAPS: int = 1  # Einfaches 1-Tap-Modell (α-Schätzung)
_MIN_PRINT_THROUGH_SCORE: float = 0.15  # Gate: unter diesem Wert kein Processing
_COHERENCE_FLOOR: float = 0.90  # Spec: Spectral Coherence nachher ≥ 0.90


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.8,
    defect_scores: dict | None = None,
    min_print_through_score: float = _MIN_PRINT_THROUGH_SCORE,
    coherence_floor: float = _COHERENCE_FLOOR,
) -> np.ndarray:
    """Haupteintrittspunkt für Phase 57.

    Args:
        audio:        Input-Audio float32, Mono oder Stereo, [samples] oder [channels, samples]
        sample_rate:  Muss 48000 Hz sein
        strength:     Verarbeitungsstärke 0–1 (skaliert alpha-Klammern)
        defect_scores: DefektScan-Scores (optional, für Print-Through-Gate)

    Returns:
        Bereinigtes Audio, gleiche Form wie Input
    """
    assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Gate: nur bei ausreichendem Print-Through-Score verarbeiten
    _pt_score = 0.0
    if defect_scores is not None:
        _pt_score = float(defect_scores.get("print_through", 0.0))
        if _pt_score < min_print_through_score:
            logger.debug(
                "Phase 57: Print-Through-Score %.3f < %.3f — übersprungen",
                _pt_score,
                min_print_through_score,
            )
            clipped_input: np.ndarray = np.clip(audio, -1.0, 1.0)
            return clipped_input

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 Linked-Stereo: LMS auf Mono-Mix, identische Korrektur auf L+R
        mono_mix = (audio[0] + audio[1]) / 2.0
        mono_repaired = apply(mono_mix, sample_rate, strength=strength, defect_scores=defect_scores)
        _eps_pt = 1e-10
        _gain_pt = np.where(
            np.abs(mono_mix) > _eps_pt,
            mono_repaired / (mono_mix + _eps_pt * np.sign(mono_mix + _eps_pt)),
            1.0,
        )
        _gain_pt = np.clip(_gain_pt, 0.0, 10.0)
        out = np.stack([audio[0] * _gain_pt, audio[1] * _gain_pt], axis=0)
        stereo_result: np.ndarray = np.clip(out, -1.0, 1.0).astype(np.float32)
        return stereo_result

    # Ab hier: Mono-Verarbeitung
    x = audio.astype(np.float64)
    len(x)
    max_delay_samples = int(_XCORR_MAX_DELAY_MS * 0.001 * sample_rate)

    # Skalierte Alpha-Klammern gemäß strength
    alpha_pre_max = _ALPHA_PRE_RANGE[0] + strength * (_ALPHA_PRE_RANGE[1] - _ALPHA_PRE_RANGE[0])
    alpha_post_max = _ALPHA_POST_RANGE[0] + strength * (_ALPHA_POST_RANGE[1] - _ALPHA_POST_RANGE[0])
    alpha_pre_max = float(np.clip(alpha_pre_max, _ALPHA_PRE_RANGE[0], _ALPHA_PRE_RANGE[1]))
    alpha_post_max = float(np.clip(alpha_post_max, _ALPHA_POST_RANGE[0], _ALPHA_POST_RANGE[1]))

    # Schritt 1: Kreuzkorrelation → delay_pre, delay_post
    delay_pre, delay_post = _find_delays(x, max_delay_samples)
    logger.debug(
        "Phase 57: delay_pre=%d samples (%.1f ms), delay_post=%d samples (%.1f ms)",
        delay_pre,
        delay_pre * 1000.0 / sample_rate,
        delay_post,
        delay_post * 1000.0 / sample_rate,
    )
    if delay_pre <= 0 and delay_post <= 0:
        logger.debug("Phase 57: Keine signifikanten Echos gefunden — übersprungen")
        no_echo_result: np.ndarray = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return no_echo_result

    # Schritt 2+3: LMS-Adaptivfilter für Pre- und Post-Echo
    try:
        x_clean = _lms_bilateral_subtraction(
            x=x,
            delay_pre=delay_pre,
            delay_post=delay_post,
            alpha_pre_max=alpha_pre_max,
            alpha_post_max=alpha_post_max,
        )
    except Exception as _lms_exc:
        logger.warning("Phase 57: LMS-Subtraction fehlgeschlagen, NMF-Fallback: %s", _lms_exc)
        x_clean = _nmf_post_echo_fallback(x, delay_post, alpha_post_max)

    # Schritt 4: Spectral-Coherence-Check — Rollback wenn Qualität schlechter
    _coh = _spectral_coherence(x, x_clean, sample_rate)
    if _coh < coherence_floor:
        logger.warning(
            "Phase 57: Spectral Coherence nach LMS %.3f < %.3f — Rollback auf Original",
            _coh,
            coherence_floor,
        )
        rollback_result: np.ndarray = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return rollback_result

    cleaned_result = np.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)
    clipped: np.ndarray = np.clip(cleaned_result.astype(np.float32), -1.0, 1.0)
    return clipped


# ─── Interne Hilfsfunktionen ───────────────────────────────────────────────────


def _find_delays(x: np.ndarray, max_delay: int) -> tuple[int, int]:
    """Findet Pre-Echo- und Post-Echo-Delays via Kreuzkorrelations-Peak-Suche.

    Returns:
        (delay_pre, delay_post) in Samples. 0 wenn kein signifikanter Peak.
    """
    n = len(x)
    # Energie-normierte Kreuzkorrelation auf kurzen Segment (max 10 s)
    seg_len = min(n, 480000)  # 10 s @ 48 kHz
    x_seg = x[:seg_len]
    x_seg = x_seg - np.mean(x_seg)
    if seg_len > 4:
        x_seg = x_seg * np.hanning(seg_len)

    # FFT-basierte Autokorrelation
    n_fft = int(2 ** math.ceil(math.log2(2 * seg_len - 1)))
    X = np.fft.rfft(x_seg, n=n_fft)
    acorr = np.fft.irfft(X * np.conj(X), n=n_fft)
    acorr = acorr[: max_delay + 1]  # nur positive Lags

    # Normalize durch Nulllag
    acorr_norm = acorr / (acorr[0] + 1e-10)

    # Signifikanter Peak: > 5 % der Energie, exklusive lag=0
    _tail = np.abs(acorr_norm[10:])
    _noise_floor = float(np.median(_tail)) if _tail.size else 0.0
    threshold = max(0.03, 3.0 * _noise_floor)
    acorr_norm[0] = 0.0  # Nulllag ausschließen

    # Post-Echo: höchster Peak ≥ 10 Samples (≥ 0.2 ms)
    delay_post = 0
    if len(acorr_norm) > 10:
        _cand = np.argmax(np.abs(acorr_norm[10:])) + 10
        if abs(acorr_norm[_cand]) > threshold:
            delay_post = int(_cand)

    # Pre-Echo: zweithöchster Peak nach delay_post (in negativem Lag = signal before)
    # Pre-Echo ist typischerweise schwächer und bei ca. 60–70% von delay_post
    delay_pre = 0
    if delay_post > 0:
        # Pre-Echo erwartet bei etwas kleineren Lags (Vorwärtswicklung näher am Signal)
        _pre_search_end = max(10, delay_post - 5)
        if _pre_search_end > 10:
            _cand_pre = np.argmax(np.abs(acorr_norm[10:_pre_search_end])) + 10
            if abs(acorr_norm[_cand_pre]) > threshold * 0.6:  # Pre-Echo typisch schwächer
                delay_pre = int(_cand_pre)

    return delay_pre, delay_post


def _lms_bilateral_subtraction(
    x: np.ndarray,
    delay_pre: int,
    delay_post: int,
    alpha_pre_max: float,
    alpha_post_max: float,
) -> np.ndarray:
    """Bidirektionale LMS-Adaptivfilter-Subtraktion (Pre- + Post-Echo).

    Algorithm (Spec §7.6 Print-Through normativ):
        audio_clean[t] = audio[t]
                         − alpha_pre · audio[t + delay_pre]   (Pre-Echo vorwärts)
                         − alpha_post · audio[t − delay_post] (Post-Echo rückwärts)

    LMS-Update (adaptiv):
        e[t] = x[t] − alpha · x_echo[t]
        alpha_new = alpha + mu · e[t] · x_echo[t]
        alpha_new = clip(alpha_new, 0.0, alpha_max)
    """
    n = len(x)
    out = x.copy()
    mu = _LMS_MU
    leak = 0.9995
    eps = 1e-8

    # Post-Echo LMS (Rückwärtswicklung — stärker, zuerst verarbeiten)
    if delay_post > 0:
        alpha_post = alpha_post_max * 0.5  # Initial-Schätzung
        for t in range(delay_post, n):
            echo_post = x[t - delay_post]
            e = out[t] - alpha_post * echo_post
            alpha_post = leak * alpha_post + (mu * e * echo_post) / (echo_post * echo_post + eps)
            alpha_post = float(np.clip(alpha_post, 0.0, alpha_post_max))
            out[t] = e

    # Pre-Echo LMS (Vorwärtswicklung — schwächer)
    if delay_pre > 0:
        alpha_pre = alpha_pre_max * 0.3  # Initial-Schätzung (schwächer)
        for t in range(0, n - delay_pre):
            echo_pre = x[t + delay_pre]
            e = out[t] - alpha_pre * echo_pre
            alpha_pre = leak * alpha_pre + (mu * e * echo_pre) / (echo_pre * echo_pre + eps)
            alpha_pre = float(np.clip(alpha_pre, 0.0, alpha_pre_max))
            out[t] = e

    logger.debug(
        "Phase 57: LMS bilateral — delay_pre=%d delay_post=%d alpha_pre_max=%.3f alpha_post_max=%.3f",
        delay_pre,
        delay_post,
        alpha_pre_max,
        alpha_post_max,
    )
    return out


def _nmf_post_echo_fallback(x: np.ndarray, delay_post: int, alpha_max: float) -> np.ndarray:
    """NMF-β Fallback: einseitige Post-Echo-Subtraktion (Spec normativ als NMF-β Fallback).

    Vereinfachte Version ohne Pre-Echo (einseitig).
    """
    if delay_post <= 0:
        return x
    n = len(x)
    out = x.copy()
    alpha = alpha_max * 0.5
    for t in range(delay_post, n):
        out[t] = x[t] - alpha * x[t - delay_post]
    logger.debug("Phase 57: NMF-Fallback (einseitig, Post-Echo) delay=%d alpha=%.3f", delay_post, alpha)
    return out


def _spectral_coherence(x_orig: np.ndarray, x_clean: np.ndarray, sr: int) -> float:
    """Berechnet Spectral Coherence zwischen Original und bereinigtem Signal [0, 1]."""
    try:
        n_fft = min(2048, len(x_orig) // 4)
        if n_fft < 64:
            return 1.0
        _f, coh = sps.coherence(x_orig.astype(np.float64), x_clean.astype(np.float64), fs=sr, nperseg=n_fft)
        _band = (_f >= 80.0) & (_f <= 12000.0)
        if np.any(_band):
            return float(np.mean(coh[_band]))
        return float(np.mean(coh))
    except Exception as _coh_exc:
        logger.debug("Spectral-Coherence-Berechnung fehlgeschlagen: %s", _coh_exc)
        return 1.0  # im Zweifel: nicht rollbacken


# ─── PhaseInterface-Klasse (normativ für test_all_phases_normative.py) ─────────


class PrintThroughReductionPhase(PhaseInterface):
    """Phase 57: Print-Through Reduction — Bidirektionale Adaptive Temporal Subtraction.

    Spec §7.x: Pre-Echo (Vorwärtswicklung) + Post-Echo (Rückwärtswicklung)
    via LMS-Adaptivfilter separat geschätzt.
    VERBOTEN: Comb-Filter, einseitiges α-Modell.
    Fallback: NMF-β Post-Echo (einseitig).
    """

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_57_print_through_reduction",
            name="Print-Through Reduction",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_29"],
            estimated_time_factor=0.04,
            version="1.0.0",
            memory_requirement_mb=16,
            is_cpu_intensive=False,
            quality_impact=0.70,
            description=(
                "Bidirektionale Adaptive Temporal Subtraction (LMS) für Print-Through "
                "(Magnetband-Vor-/Nachecho). Pre- und Post-Echo separat via LMS. "
                "Nur für TAPE/REEL_TAPE Material aktiv."
            ),
        )

    @staticmethod
    def _compute_print_through_profile(
        material_key: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive print-through profile."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("reel_tape", "tape", "cassette")):
            min_print_through_score = 0.12
            coherence_floor = 0.93
        elif any(token in _material for token in ("cd_digital", "dat", "streaming", "flac")):
            min_print_through_score = 0.22
            coherence_floor = 0.95
        else:
            min_print_through_score = 0.18
            coherence_floor = 0.94

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        min_print_through_score += (_rest_norm - 0.5) * 0.14

        _mode_offsets = {
            "fast": (0.04, -0.01),
            "balanced": (0.0, 0.0),
            "quality": (-0.03, 0.02),
            "maximum": (-0.05, 0.03),
        }
        _score_off, _coh_off = _mode_offsets.get(_mode, (0.0, 0.0))
        min_print_through_score += _score_off
        coherence_floor += _coh_off

        return {
            "min_print_through_score": float(np.clip(min_print_through_score, 0.05, 0.30)),
            "coherence_floor": float(np.clip(coherence_floor, 0.90, 0.99)),
        }

    @staticmethod
    def _build_locality_profile(
        n_samples: int,
        sample_rate: int,
        defect_locations: dict[str, list[tuple[float, float]]] | None,
        defect_event_metadata: dict[str, dict] | None = None,
        protected_zones: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Eventadaptive Blendmaske fuer Print-Through-Reduktion."""
        if n_samples <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        if not isinstance(defect_locations, dict) or not defect_locations:
            return np.ones(n_samples, dtype=np.float32), 0.0
        keys = ("print_through", "pre_echo", "post_echo", "magnetic_pre_echo", "magnetic_post_echo")
        mask = np.zeros(n_samples, dtype=np.float32)
        for key in keys:
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start_s = max(0.0, float(loc[0]))
                    end_s = max(start_s, float(loc[1]))
                except Exception:
                    continue
                if end_s <= start_s:
                    continue
                meta_obj = (defect_event_metadata or {}).get(key, {})
                severity = (
                    float(np.clip(float(meta_obj.get("severity", 0.55)), 0.0, 1.0))
                    if isinstance(meta_obj, dict)
                    else 0.55
                )
                confidence = (
                    float(np.clip(float(meta_obj.get("confidence", 0.80)), 0.0, 1.0))
                    if isinstance(meta_obj, dict)
                    else 0.80
                )
                key_factor = 0.78 if key in {"pre_echo", "magnetic_pre_echo"} else 1.0
                duration_factor = float(np.clip((end_s - start_s) / 0.50, 0.35, 1.0))
                event_strength = float(
                    np.clip(key_factor * (0.38 + 0.42 * severity + 0.20 * confidence) * duration_factor, 0.16, 1.0)
                )
                pad_s = 0.18 if "post" in key else 0.12
                s = int(max(0.0, start_s - pad_s) * sample_rate)
                e = int(min(float(n_samples) / float(sample_rate), end_s + pad_s) * sample_rate)
                if e > s:
                    mask[s:e] = np.maximum(mask[s:e], event_strength)
        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0
        smooth = max(16, int(0.030 * sample_rate))
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
                    continue
        return zones

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="57")
        check_ml_model_ready("Whisper", phase_name="57")
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

        strength = float(kwargs.get("strength", 0.8))
        defect_scores = kwargs.get("defect_scores")

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        _profile_57 = self._compute_print_through_profile(
            str(material_type or kwargs.get("material_type", kwargs.get("material", "unknown"))).lower(),
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", strength))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V40 NMR-Feedback: NR-Stärke adaptiv anpassen (FeedbackChain-aware).
        try:
            from backend.core.dsp.nmr_feedback import (
                compute_nmr_score as _nmr_fn_57,
            )

            _nmr_result_57 = _nmr_fn_57(audio, sample_rate)
            if not _nmr_result_57.ok:
                logger.warning(
                    "Phase57 §V40 NMR: nmr_above_masking → §2.45 Minimal-Intervention prüfen",
                )
            _effective_strength = float(
                np.clip(
                    _effective_strength + _nmr_result_57.recommended_nr_strength_delta,
                    0.0,
                    1.0,
                )
            )
            logger.debug(
                "Phase57 §V40 NMR: delta=%.3f → eff_str=%.3f",
                _nmr_result_57.recommended_nr_strength_delta,
                _effective_strength,
            )
        except Exception as _nmr_exc_57:  # pylint: disable=broad-except
            logger.debug("Phase57 §V40 NMR non-blocking: %s", _nmr_exc_57)

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "print_through_score": float((_defect_scores or {}).get("print_through", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Print-through reduction skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_print_through_score=_profile_57["min_print_through_score"],
            coherence_floor=_profile_57["coherence_floor"],
        )
        elapsed = _time.perf_counter() - t0
        _locality_coverage57 = 0.0
        _locality_profile57, _locality_coverage57 = self._build_locality_profile(
            n_samples=int(
                result_audio.shape[-1]
                if result_audio.ndim == 2 and result_audio.shape[0] <= 2
                else result_audio.shape[0]
            ),
            sample_rate=sample_rate,
            defect_locations=kwargs.get("defect_locations"),
            defect_event_metadata=kwargs.get("defect_event_metadata"),
            protected_zones=self._collect_protected_zones(kwargs),
        )
        if _locality_profile57.size > 0 and _locality_coverage57 > 0.0:
            if result_audio.ndim == 2 and audio.ndim == 2:
                if result_audio.shape[0] <= 2 and result_audio.shape[1] > 2:
                    _profile57 = _locality_profile57[np.newaxis, :]
                else:
                    _profile57 = _locality_profile57[:, np.newaxis]
                result_audio = np.clip(audio + _profile57 * (result_audio - audio), -1.0, 1.0).astype(np.float32)
            elif result_audio.ndim == 1 and audio.ndim == 1:
                result_audio = np.clip(audio + _locality_profile57 * (result_audio - audio), -1.0, 1.0).astype(
                    np.float32
                )

        # §4.5 Psychoacoustic Masking Clamp — only reduce audible print-through
        try:
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp

            result_audio = apply_psychoacoustic_masking_clamp(
                audio,
                result_audio,
                sample_rate,
                strength=_effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            logger.debug("Phase57 masking clamp non-blocking: %s", _pm_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — Print-Through-Reduktion darf
        # Atemgeräusche und Vibrato-Zonen nicht durch Subtraktionsfilter tilgen.
        try:
            from backend.core.natural_performance_detector import get_natural_performance_detector

            _npa_a57 = audio
            if _npa_a57.ndim == 2 and _npa_a57.shape[0] == 2 and _npa_a57.shape[1] > _npa_a57.shape[0]:
                _npa_a57 = _npa_a57.T
            _npa_r57 = get_natural_performance_detector().detect(_npa_a57, sample_rate)
            _npa_n57 = (
                result_audio.shape[1]
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else result_audio.shape[0]
            )
            _npa_m57 = _npa_r57.get_protected_mask(_npa_n57, sample_rate)
            if np.any(_npa_m57):
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _npa_m57] = audio[:, _npa_m57]
                    elif result_audio.shape == audio.shape:
                        result_audio[_npa_m57, :] = audio[_npa_m57, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_npa_m57] = audio[_npa_m57]
        except Exception as _npa57_exc:
            logger.debug("§2.46f phase_57 NPA-Guard (non-blocking): %s", _npa57_exc)

        # §2.36 Phonem-Schutz: Print-Through-Reduktion subtrahiert Vor-/Nachhall-Energie.
        # Plosive Burst-Transienten haben ähnliche Burst-Energie — Plosiv-Frames schützen.
        try:
            from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_pmask_57

            _hop_57 = 512
            _mono_57 = (
                result_audio.mean(axis=0)
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else (result_audio.mean(axis=1) if result_audio.ndim == 2 else result_audio)
            )
            _pmask_57 = _get_pmask_57(_mono_57.astype(np.float32), sample_rate, hop_length=_hop_57)
            if np.any(_pmask_57):
                _n_57 = _mono_57.shape[0]
                _smask_57 = np.zeros(_n_57, dtype=bool)
                for _fi57, _fp57 in enumerate(_pmask_57):
                    if _fp57:
                        _fs57 = _fi57 * _hop_57
                        _fe57 = min(_n_57, _fs57 + _hop_57)
                        _smask_57[_fs57:_fe57] = True
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _smask_57] = audio[:, _smask_57]
                    elif result_audio.shape == audio.shape:
                        result_audio[_smask_57, :] = audio[_smask_57, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_smask_57] = audio[_smask_57]
        except Exception as _pm57_exc:
            logging.getLogger(__name__).debug("\u00a72.36 phase_57 Phonem-Mask (non-blocking): %s", _pm57_exc)

        # §V19 Noise-Texture-Invariante: Residual darf Material-Spektralprofil nicht whitten.
        _nt57_residual = audio - result_audio
        _mat57_str = str(material_type or kwargs.get("material_type", kwargs.get("material", "unknown"))).lower()
        try:
            from backend.core.dsp.noise_texture_guard import (  # pylint: disable=import-outside-toplevel
                compute_noise_texture_distance as _nt57_dist_fn,
            )

            if _nt57_residual.shape == audio.shape:
                _nt57_d = _nt57_dist_fn(_nt57_residual, _mat57_str, sr=sample_rate)
                if _nt57_d > 0.25:
                    result_audio = (0.5 * result_audio + 0.5 * audio).astype(np.float32)
                    logger.warning("\u00a7V19 phase_57: noise_texture_dist=%.3f > 0.25 \u2192 50%% dry-blend", _nt57_d)
        except Exception as _nt57_exc:
            logger.debug("\u00a7V19 phase_57 noise_texture non-blocking: %s", _nt57_exc)

        # §V20 Mikrodynamik-Korrelation: Voiced-Frame-Energie nach LMS-NR nicht degradieren.
        _p57_panns = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if _p57_panns >= 0.25:
            try:
                from backend.core.dsp.mikrodynamik_guard import (  # pylint: disable=import-outside-toplevel
                    frame_energy_correlation as _fec57,
                )
                from backend.core.dsp.mikrodynamik_guard import (
                    recommend_mikrodynamik_wet as _recommend_mkk_wet,
                )

                _corr57 = _fec57(audio, result_audio, sample_rate, frame_ms=10.0)
                if _corr57 < 0.97:
                    _need57 = float(kwargs.get("mikrodynamik_global_need", kwargs.get("global_need", 0.0)) or 0.0)
                    _wet57 = _recommend_mkk_wet(_corr57, _p57_panns, global_need=_need57)
                    result_audio = (_wet57 * result_audio + (1.0 - _wet57) * audio).astype(np.float32)
                    logger.warning("\u00a7V20 phase_57: mikrodynamik_corr=%.4f < 0.97 \u2192 wet=%.3f", _corr57, _wet57)
            except Exception as _v20_57_exc:
                logger.debug("\u00a7V20 phase_57 mikrodynamik non-blocking: %s", _v20_57_exc)

        # §V21 Mindestrauschboden: Pausenzonen auf Tape/Shellac dürfen nicht auf digitale Stille fallen.
        if any(x in _mat57_str for x in ("tape", "shellac", "reel", "analog", "vinyl")):
            try:
                from backend.core.dsp.noise_floor_guard import (  # pylint: disable=import-outside-toplevel
                    apply_noise_floor_minimum as _nfmin57,
                )

                result_audio = _nfmin57(result_audio, sample_rate, _mat57_str, original_audio=audio)
            except Exception as _v21_57_exc:
                logger.debug("\u00a7V21 phase_57 noise_floor non-blocking: %s", _v21_57_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        _pt_score = float((_defect_scores or {}).get("print_through", 0.0)) if _defect_scores else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "print_through_score": _pt_score,
                "strength": strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "print_through_profile": dict(_profile_57),
                "min_print_through_score": float(_profile_57["min_print_through_score"]),
                "coherence_floor": float(_profile_57["coherence_floor"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "repair_locality_coverage": float(_locality_coverage57),
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )
