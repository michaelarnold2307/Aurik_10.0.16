"""
Phase 48: Stereo Width Enhancer v2.1 — Frequenzabhängige M/S-Breite + IACC-Guard
==================================================================================

ALGORITHMUS (v2.1 — upgraded):
  Frequenzabhängige Breite (EBU R128 Best Practice + Moulton 2000):
    - LF < 200 Hz:   width × 0.6  (LF-Mono — Bassdrum bleibt in Mitte)
    - MF 200–8 kHz:  width × 1.0  (Standardbreite)
    - HF > 8 kHz:    width × 1.15 (Luftigkeit/Raumgefühl)

  IACC-Guard (Spec §8.2):
    - Messe IACC (Inter-Aural Cross Correlation) nach Widening
    - Falls IACC < 0.97 (Mono-Ären): Side-Faktor progressiv reduzieren bis IACC ≥ 0.97
      (Blauert 1997, Tab. 2.1)

  PSYCHOAKUSTISCHE BASIS:
    - Tiefbass mono → vermeidet Auslöschungen in Mono-Wiedergabe
    - Differenz-Gruppenverschiebung bei LF führt zu uneindeutiger Lokalisation (Rayleigh 1907)
    - Hohe Frequenzen weiser Breite: ILD dominiert HRTF > 1.5 kHz

Author: Aurik Development Team
Version: 2.1.0
"""

from __future__ import annotations

import os
import logging
import time

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import to_channels_last
from backend.core.core_utils import fft_crosscorr
from backend.core.dsp.hallucination_guard import check_hallucination
from backend.core.phase_strength_contract import resolve_phase_strength_contract

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

_DEFAULT_WIDTH = 1.25
_ALLPASS_DELAYS_MS = [17.1, 19.7, 23.3]
_ALLPASS_GAIN = 0.60

# Frequenzabhängige Breiten-Korrekturfaktoren
_LF_CUTOFF_HZ = 200.0  # LF schmaler als Basisbreite
_HF_CUTOFF_HZ = 8000.0  # HF etwas breiter als Basisbreite
_LF_WIDTH_FACTOR = 0.60  # Tiefbass deutlich schmaler (Mono-Kompatibilität)
_HF_WIDTH_FACTOR = 1.15  # Hochton leicht breiter (Luftigkeit)

# IACC-Guard
_IACC_MIN = 0.97
_IACC_MAX_LAG_MS = 1.0  # Über ±1ms mitteln (Blauert 1997)

# STFT-Fenstergröße für Frequenzband-Processing
_N_FFT = 2048
_HOP = 512


def _compute_iacc(L: np.ndarray, R: np.ndarray, sr: int) -> float:
    """Berechnet Inter-Aural Cross-Correlation (peak innerhalb ±1ms)."""
    max_lag = max(1, int(_IACC_MAX_LAG_MS / 1000.0 * sr))
    n = min(len(L), len(R), 65536)
    L_n = L[:n] / (np.std(L[:n]) + 1e-10)
    R_n = R[:n] / (np.std(R[:n]) + 1e-10)
    xcf = fft_crosscorr(L_n, R_n)
    center = len(xcf) // 2
    window = xcf[center - max_lag : center + max_lag + 1]
    if len(window) == 0:
        return 1.0
    return float(np.clip(np.max(np.abs(window)) / n, 0.0, 1.0))


def _freq_dependent_ms_width(
    L: np.ndarray,
    R: np.ndarray,
    sr: int,
    width: float,
    diffuse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    M/S Breitensteuerer mit frequenzabhängiger Skalierung (STFT-basiert).

    LF-Bereich wird auf width × _LF_WIDTH_FACTOR reduziert, um
    Mono-Basswiedergabe zu erhalten. HF wird auf width × _HF_WIDTH_FACTOR
    leicht angehoben für Luftigkeit.
    """
    n_fft = _N_FFT
    hop = _HOP
    # M/S encode
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    M = (L + R) * inv_sqrt2
    S = (L - R) * inv_sqrt2

    # STFT auf Side-Kanal
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    f_lo = _LF_CUTOFF_HZ
    f_hi = _HF_CUTOFF_HZ

    # Width-Profile über Frequenzarrays
    w_profile = np.ones(len(freqs)) * width
    w_profile[freqs < f_lo] = width * _LF_WIDTH_FACTOR
    w_profile[freqs > f_hi] = width * _HF_WIDTH_FACTOR
    # §AO Guard: STFT params must be valid
    if n_fft <= hop:
        n_fft = hop * 2  # Ensure nperseg > hop for valid noverlap
    if len(S) < n_fft:
        return S  # Audio too short for STFT
    _, _, Z = sig.stft(
        S,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=max(0, n_fft - hop),
        boundary="zeros",
        padded=True,
    )
    Z_scaled = Z * w_profile[:, None]
    _, S_out = sig.istft(
        Z_scaled,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=max(0, n_fft - hop),
        input_onesided=True,
        boundary=True,
    )
    if len(S_out) < len(S):
        S_out = np.pad(S_out, (0, len(S) - len(S_out)))
    else:
        S_out = S_out[: len(S)]

    if diffuse and width > 1.05:
        S_out = _allpass_chain(S_out, sr)

    L_out = (M + S_out) * inv_sqrt2
    R_out = (M - S_out) * inv_sqrt2
    return L_out, R_out


def _allpass_chain(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Kaskadierende Schroeder-Allpass-Filter (Schroeder 1962)."""
    out = signal.copy()
    for delay_ms in _ALLPASS_DELAYS_MS:
        D = max(1, int(delay_ms / 1000.0 * sample_rate))
        g = _ALLPASS_GAIN
        b = np.zeros(D + 1)
        b[0] = -g
        b[-1] = 1.0
        a = np.zeros(D + 1)
        a[0] = 1.0
        a[-1] = -g
        out = sig.lfilter(b, a, out)
    return out


class StereoWidthEnhancerPhase(PhaseInterface):
    """M/S-Stereobreiten-Enhancer mit frequenzabhängiger Breite + IACC-Guard."""

    _PHASE_ID = "phase_48_stereo_width_enhancer"
    _NAME = "Stereo Width Enhancer (Freq-Dep M/S + IACC-Guard)"
    description = (
        "Frequenzabhängige M/S-Breitensteuererung: LF < 200 Hz schmaler (Mono-Basis), "
        "HF > 8 kHz leicht breiter (Luftigkeit). IACC-Guard für Mono-Kompatibilität (≥ 0.97). "
        "Allpass-Diffusion bei width > 1.05."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
            category=PhaseCategory.STEREO,
            priority=3,
            version="2.1.0",
            dependencies=[],
            estimated_time_factor=0.04,
            memory_requirement_mb=40,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.85,
            description=self.description,
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        """
        Frequenzabhängige Stereobreite + IACC-Guard.

        Args:
            audio:       Mono oder Stereo
            sample_rate: Abtastrate Hz
            material_type: Unbenutzt, nur fuer kanonische PhaseInterface-Signatur.
            **kwargs:    width   (float, default 1.25)
                         diffuse (bool, default True)
                         iacc_guard (bool, default True)
        """

        # §v10.16 Normalize stereo orientation to (N,2) for consistent processing
        if isinstance(audio, np.ndarray) and audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = np.ascontiguousarray(audio.T)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p48_transposed = to_channels_last(audio)
        self.validate_input(audio)
        t0 = time.time()

        _strength_ctx = resolve_phase_strength_contract(kwargs)
        phase_locality_factor = float(_strength_ctx["phase_locality_factor"])
        effective_strength = float(_strength_ctx["effective_strength"])

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

        # §0 Primum non nocere: STFT-basiertes HF-Side-Widening (×1.15) komprimiert
        # das Zentrum relativ zur Side → Gesang klingt weiter entfernt.
        # Nur Studio 2026 darf volle Stereobreite hinzufügen.
        _p48_studio = bool(kwargs.get("is_studio_mode", False))
        if not _p48_studio:
            effective_strength = float(np.clip(effective_strength, 0.0, 0.25))
            logger.debug(
                "phase_48: Restoration-Mode — Strength auf %.2f, Width-Cap 1.08 (Gesang-Präsenz-Schutz)",
                effective_strength,
            )
        _p48_width_default = _DEFAULT_WIDTH if _p48_studio else 1.08
        width: float = float(kwargs.get("width", _p48_width_default))
        if not _p48_studio:
            width = float(np.clip(width, 1.0, 1.08))
        diffuse: bool = bool(kwargs.get("diffuse", True))
        iacc_guard: bool = bool(kwargs.get("iacc_guard", True))
        width = max(0.0, width) * effective_strength

        if audio.ndim == 1:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "skipped": "mono_input",
                    "width": width,
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

        # Frequenzabhängige M/S-Breite (STFT basiert)
        L_out, R_out = _freq_dependent_ms_width(L, R, sample_rate, width, diffuse)

        # IACC-Guard
        iacc_val = 1.0
        side_reduction = 1.0
        if iacc_guard:
            iacc_val = _compute_iacc(L_out, R_out, sample_rate)
            if iacc_val < _IACC_MIN:
                excess = (_IACC_MIN - iacc_val) / _IACC_MIN
                reduced_width = max(1.0, width - excess * width * 0.8)
                L_out, R_out = _freq_dependent_ms_width(L, R, sample_rate, reduced_width, diffuse)
                side_reduction = reduced_width / width if width > 0 else 1.0
                logger.debug(
                    "Phase 48 IACC-Guard: iacc=%.3f < %.2f → width %.2f → %.2f",
                    iacc_val,
                    _IACC_MIN,
                    width,
                    reduced_width,
                )

        processed = np.column_stack([L_out, R_out])

        if 0.0 < effective_strength < 1.0:
            processed = audio + effective_strength * (processed - audio)

        # Pegel-Erhalt — §2.49 Peak-Guard: percentile(99.9)
        peak_out = float(np.percentile(np.abs(processed), 99.9))
        if peak_out > 1e-8 and peak_in > 1e-8:
            processed = processed * (peak_in / peak_out)

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        # §2.46e Hallucination-Guard (Pflicht für additive Phasen)
        try:
            _mode_48 = "studio_2026" if _p48_studio else "restoration"
            _hg_48 = check_hallucination(audio, processed, sr=sample_rate, mode=_mode_48)
            if _hg_48.requires_rollback:
                logger.warning(
                    "phase_48: hallucination_guard rollback (spectral_novelty=%.3f)", _hg_48.spectral_novelty
                )
                processed = np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            elif _hg_48.score_penalty > 0.0:
                logger.info(
                    "phase_48: hallucination_guard penalty=%.1f (spectral_novelty=%.3f)",
                    _hg_48.score_penalty,
                    _hg_48.spectral_novelty,
                )
        except Exception as _hg48_exc:
            logger.debug("phase_48: hallucination_guard failed (non-blocking): %s", _hg48_exc)

        logger.info(
            "Phase 48 StereoWidth: width=%.2f, diffuse=%s, iacc=%.3f, side_red=%.2f",
            width,
            diffuse,
            iacc_val,
            side_reduction,
        )

        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "width": width,
                "diffuse": diffuse,
                "iacc": iacc_val,
                "side_reduction": side_reduction,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            metrics={"width": width, "iacc": iacc_val, "effective_strength": effective_strength},
        )

    def _allpass_chain(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Kaskadierende Schroeder-Allpass-Filter."""
        return _allpass_chain(signal, sample_rate)
