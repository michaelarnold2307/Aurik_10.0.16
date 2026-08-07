"""
Phase 46: Spatial Enhancement v2.1 — Psychoacoustic Room Diffusion
===================================================================

Vollständige DSP-Implementierung ohne aurik_ml.
Ersetzt den kaputten ML-Stub.

ALGORITHMUS (v2.1 — upgraded):
  1. Mono-Check: Mono-Signale passieren unverändert.

  2. M/S Enkodierung für selektive Side-Verarbeitung.

  3. Psychoakustisch kalibrierte Early Reflections (Haas 1951):
     - 4 diskrete Reflexionen (6–22 ms): klingen als Raumbreite wahrgenommen,
       nicht als Echo (Haas-Grenze: < 35 ms)
     - Delays aus Primzahl-Verhältnissen (Schroeder 1962): minimale Kammfilter
     - Amplitude: −8 bis −16 dB unter Direktsignal (Blauert 1997)

  4. Schroeder-Allpass-Diffusionsnetz (Side-Kanal):
     - 3 kaskadierende Allpässe mit Primzahl-ähnlichen Delays
     - Glättet den Side-Kanal ohne Freq-Färbung

  5. IACC-Guard (Spec §8.2 — Mono-Ären):
     - Inter-Aural Cross-Correlation ≥ 0.97 für Mono-Quellen
     - Überschreitung → Side-Reduktion

  6. Normalisierungs-Pass: Pegel-Erhalt

Author: Aurik Development Team
Version: 2.1.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import to_channels_last
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# Frühe Reflexionen: Delays in ms + Amplituden-Dämpfung in dB
_EARLY_REFLECTIONS = [
    (6.1, -8.5),  # Erste Seitenreflexion (Blauert perceptual plane front-left)
    (11.3, -11.0),  # Zweite Reflexion (Primzahl-Ratio)
    (17.7, -13.5),  # Dritte Reflexion
    (22.4, -16.0),  # Vierte Reflexion (kurz genug um Echo zu vermeiden)
]

# Allpass-Diffusion: Delays + Gains
_ALLPASS_PARAMS = [
    (5.0, 0.45),  # schnell — kleine Raumgröße
    (13.3, 0.40),  # mittel
    (23.7, 0.35),  # langsam — Diffusion
]

# IACC-Guard: Ziel-Korrelation für Mono-kompatible Quellen
_IACC_MIN = 0.97


def _early_reflection_mix(
    L: np.ndarray, R: np.ndarray, sr: int, dry_wet: float = 0.18
) -> tuple[np.ndarray, np.ndarray]:
    """Fügt hinzu: psychoacoustically distributed early reflections to extend perceived width."""
    L_out = L.copy()
    R_out = R.copy()
    for delay_ms, atten_db in _EARLY_REFLECTIONS:
        delay_s = max(1, int(delay_ms / 1000.0 * sr))
        gain = 10.0 ** (atten_db / 20.0)
        if delay_s >= len(L):
            continue
        # Alternate: odd reflections to L, even to R (lateral distribution)
        idx = _EARLY_REFLECTIONS.index((delay_ms, atten_db))
        delayed_L = np.concatenate([np.zeros(delay_s, dtype=L.dtype), L[:-delay_s]])
        delayed_R = np.concatenate([np.zeros(delay_s, dtype=R.dtype), R[:-delay_s]])
        if idx % 2 == 0:
            L_out += gain * dry_wet * delayed_R  # Cross-feed: R delayed to L
        else:
            R_out += gain * dry_wet * delayed_L  # Cross-feed: L delayed to R
    return L_out, R_out


def _allpass_diffuse(signal_in: np.ndarray, sr: int) -> np.ndarray:
    """Cascade of Schroeder Allpass filters for Side-channel diffusion."""
    out = signal_in.copy()
    for delay_ms, g in _ALLPASS_PARAMS:
        D = max(1, int(delay_ms / 1000.0 * sr))
        if len(out) <= D:
            continue
        b = np.zeros(D + 1)
        b[0] = -g
        b[-1] = 1.0
        a = np.zeros(D + 1)
        a[0] = 1.0
        a[-1] = -g
        out = sig.lfilter(b, a, out)
    return out


def _compute_iacc(L: np.ndarray, R: np.ndarray, max_lag_ms: float = 1.0, sr: int = 48000) -> float:
    """Inter-Aural Cross-Correlation (IACC): peak of normalized XCF within ±1ms."""
    max_lag = max(1, int(max_lag_ms / 1000.0 * sr))
    n = min(len(L), len(R), 65536)  # limit for performance
    L_n = L[:n] / (np.std(L[:n]) + 1e-10)
    R_n = R[:n] / (np.std(R[:n]) + 1e-10)
    from backend.core.core_utils import fft_crosscorr  # pylint: disable=import-outside-toplevel

    xcf = fft_crosscorr(L_n, R_n)
    center = len(xcf) // 2
    window = xcf[center - max_lag : center + max_lag + 1]
    iacc = float(np.max(np.abs(window))) / n if len(window) > 0 else 1.0
    return float(np.clip(iacc, 0.0, 1.0))


class SpatialEnhancementPhase(PhaseInterface):
    """Psychoacoustic early reflections + M/S Allpass-Diffusion für erweitertes Stereobild."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Spatial Enhancement (Early Reflections + M/S Allpass)"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_46_spatial_enhancement",
            name="Spatial Enhancement (Early Reflections + M/S Allpass)",
            category=PhaseCategory.STEREO,
            priority=3,
            version="2.1.0",
            dependencies=[],
            estimated_time_factor=0.05,
            memory_requirement_mb=60,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.87,
            description=(
                "Stereobild-Erweiterung via 4 frühe Reflexionen (6–22 ms, −8 bis −16 dB) "
                "nach Blauert (1997) und M/S-Allpass-Diffusion (3-stufig). "
                "IACC-Guard für Mono-Kompatibilität. Kein aurik_ml."
            ),
        )

    def process(
        self, audio: np.ndarray, sample_rate: int = 48000, material_type: str = "unknown", **kwargs
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="46")
        """
        Räumliche Erweiterung via Early Reflections + M/S Diffusion.

        Args:
            audio:        Mono oder Stereo (float32/float64)
            sample_rate:  Hz
            **kwargs:     dry_wet     (float, default 0.18)  — Reflexions-Mix
                          diffuse     (bool, default True)   — Allpass-Diffusion
                          iacc_guard  (bool, default True)   — Mono-Kompatibilitätsprüfung
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p46_transposed = to_channels_last(audio)
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_46 = float(
            kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", kwargs.get("vocal_confidence", 0.0)))
        )
        if _panns_s_46 >= 0.25 and effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_46,
                )

                _fmz_46 = kwargs.get("forward_masking_zones") or _fmg_fn_46().compute_zones(audio, sample_rate)
                if _fmz_46:
                    _n_s_46 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_46 = sum(z.end_sample - z.start_sample for z in _fmz_46)
                    _zone_frac_46 = float(np.clip(_zone_s_46 / max(1, _n_s_46), 0.0, 1.0))
                    effective_strength = float(np.clip(effective_strength + _zone_frac_46 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_46:
                logger.debug("Verarbeitungsschritt46 §V41 ForwardMaskingGuard nicht blockierend: %s", _fmg_exc_46)

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        # §0 Primum non nocere: Frühe Reflexionen (cross-feed) rücken Gesang nach hinten.
        # Restoration-Mode darf nur konservative Raumtiefe hinzufügen.
        _quality_mode = str(kwargs.get("quality_mode", kwargs.get("mode", "restoration"))).strip().lower()
        _is_studio_mode = bool(kwargs.get("is_studio_mode", False)) or ("studio" in _quality_mode)
        _p46_panns = float(
            kwargs.get(
                "panns_singing",
                kwargs.get("panns_singing_confidence", kwargs.get("vocal_confidence", 0.0)),
            )
        )
        _panns_tags = kwargs.get("panns_tags", {})
        if isinstance(_panns_tags, dict):
            try:
                _tag_vocal = float(_panns_tags.get("vocals", 0.0))
                _tag_singing = float(_panns_tags.get("singing", 0.0))
                _p46_panns = max(_p46_panns, _tag_vocal, _tag_singing)
            except Exception as e:
                logger.warning("Verarbeitungsschritt_46_spatial_enhancement.py::verarbeiten Ersatzpfad: %s", e)

        _vocal_echo_guard = (not _is_studio_mode) and _p46_panns >= 0.25
        _side_width_gain = 1.0
        if _vocal_echo_guard:
            # §0p/§2.46e: keine delay-basierten Raumanteile auf Gesang.
            # Phase bleibt dennoch sinnvoll: minimale, IACC-gesicherte Stereo-Balance.
            _side_width_gain = 1.03
            logger.warning(
                "Verarbeitungsschritt 46 VocalEcho-Guard: restoration-safe Betriebsart (panns_singing=%.2f)",
                _p46_panns,
            )

        if not _is_studio_mode:
            effective_strength = float(np.clip(effective_strength, 0.0, 0.20))
            logger.debug(
                "Verarbeitungsschritt_46: Restoration-Betriebsart — Strength auf %.2f gedeckelt (kein künstlicher Hall)",
                effective_strength,
            )
        dry_wet: float = float(kwargs.get("dry_wet", 0.18 if _is_studio_mode else 0.05))
        diffuse: bool = bool(kwargs.get("diffuse", True))
        iacc_guard: bool = bool(kwargs.get("iacc_guard", True))
        if _vocal_echo_guard:
            dry_wet = 0.0
            diffuse = False
        dry_wet = float(np.clip(dry_wet, 0.0, 1.0)) * effective_strength

        if audio.ndim == 1:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "skipped": "mono_input",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"effective_strength": effective_strength},
            )

        L = audio[:, 0]
        R = audio[:, 1]
        peak_in = float(np.percentile(np.abs(audio), 99.9))  # §2.49 Peak-Guard

        # 1. Early reflections (psychoacoustic lateral energy)
        if dry_wet > 0.0:
            L_ref, R_ref = _early_reflection_mix(L, R, sample_rate, dry_wet)
        else:
            L_ref, R_ref = L.copy(), R.copy()

        # 2. M/S encoding of the reflected signal
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        M = (L_ref + R_ref) * inv_sqrt2
        S = (L_ref - R_ref) * inv_sqrt2
        if _side_width_gain != 1.0:
            S *= _side_width_gain

        # 3. Allpass diffusion on Side channel
        if diffuse:
            S = _allpass_diffuse(S, sample_rate)

        # 4. M/S decode
        L_out = (M + S) * inv_sqrt2
        R_out = (M - S) * inv_sqrt2

        # 5. IACC-Guard: ensure mono compatibility
        iacc_val = 1.0
        side_reduction = 1.0
        if iacc_guard:
            iacc_val = _compute_iacc(L_out, R_out, sr=sample_rate)
            if iacc_val < _IACC_MIN:
                # Reduce Side component to restore mono compatibility
                excess = (_IACC_MIN - iacc_val) / _IACC_MIN
                side_reduction = max(0.3, 1.0 - excess * 2.0)
                S_red = S * side_reduction
                L_out = (M + S_red) * inv_sqrt2
                R_out = (M - S_red) * inv_sqrt2
                logger.debug(
                    "Verarbeitungsschritt 46 IACC-Guard: iacc=%.3f < %.2f → side_reduction=%.2f",
                    iacc_val,
                    _IACC_MIN,
                    side_reduction,
                )

        processed = np.column_stack([L_out, R_out])

        if 0.0 < effective_strength < 1.0:
            processed = audio + effective_strength * (processed - audio)

        # 6. Level preservation — §2.49 Peak-Guard: percentile(99.9)
        peak_out = float(np.percentile(np.abs(processed), 99.9))
        if peak_out > 1e-8 and peak_in > 1e-8:
            processed = processed * (peak_in / peak_out)

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        logger.info(
            "Verarbeitungsschritt 46 SpatialEnhancement: dry_wet=%.2f, diffuse=%s, iacc=%.3f, side_red=%.2f",
            dry_wet,
            diffuse,
            iacc_val,
            side_reduction,
        )

        # §2.46e Hallucination-Guard: Räumliche Cues dürfen keine nicht-originären
        # Spektralanteile einführen (VERBOTEN-§2.46e).
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _hg_46,
            )

            _mode_46 = str(kwargs.get("mode", "restoration"))
            _hg_result_46 = _hg_46(audio, processed, sr=sample_rate, mode=_mode_46)
            if getattr(_hg_result_46, "requires_rollback", False):
                processed = audio.copy()
                logger.warning("Verarbeitungsschritt46 §2.46e Hallucination-Guard Rollback (spectral_novelty > 0.15)")
        except Exception as _hg_exc_46:
            logger.debug("Verarbeitungsschritt46 §2.46e Hallucination-Guard (nicht blockierend): %s", _hg_exc_46)

        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "algorithm": "vocal_echo_guard_safe" if _vocal_echo_guard else "phase_46_default",
                "panns_singing": _p46_panns,
                "quality_mode": _quality_mode,
                "dry_wet": dry_wet,
                "diffuse": diffuse,
                "iacc": iacc_val,
                "side_reduction": side_reduction,
                "side_width_gain": _side_width_gain,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )
