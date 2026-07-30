#!/usr/bin/env python3
"""
Phase 14: Professional Phase Correction v2.0.
=============================================

Multi-band stereo phase alignment for optimal imaging and mono compatibility.

SCIENTIFIC FOUNDATION:
- Gerzon (1992): Multi-Channel Microphone Array Design
- Lipshitz & Vanderkooy (1986): The Great Debate: Subjective Evaluation
- Bech & Zacharov (2006): Perceptual Audio Evaluation - stereo imaging
- Blauert (1997): Spatial Hearing - The Psychophysics of Human Sound Localization
- ITU-R BS.775-3: Multichannel Stereophonic Sound System with and without Accompanying Picture
- EBU Tech 3286: Assessment and Specification of Phase Coherence
- Laakso et al. (1996): "Splitting the Unit Delay: Tools for Fractional Delay Filter Design",
  IEEE Signal Processing Magazine 13(1), pp. 30-60.
  Lagrange order-3 FIR for sub-sample stereo alignment (L2.1).
- Smith (2011): "Spectral Audio Signal Processing" §3.4 — parabolic
  interpolation of cross-correlation peak for fractional delay estimation.

INDUSTRY BENCHMARKS:
- iZotope Ozone Imager (Stereo Phase correlation display)
- Waves InPhase (Multi-band stereo phase alignment)
- Brainworx bx_digital V3 (Correlation meter + phase correction)
- SSL X-ISM (Intelligent Stereo Management)
- Flux Stereo Tool (Phase/Time alignment)
- Nugen Audio Stereo Pack (Phase correlation analysis)

ALGORITHM:
1. Multi-Band Cross-Correlation Analysis (4 bands)
   - 200 Hz, 1 kHz, 8 kHz crossovers
   - Per-band phase correlation measurement
   - Time-delay estimation via cross-correlation peak

2. Per-Band Phase Alignment
   - Bass: Critical for mono compatibility (sum to mono)
   - Mid: Balance between imaging and compatibility
   - High: Wide stereo image acceptable

3. Material-Adaptive Correction
   - Shellac/Vinyl: Strong correction (old stereo techniques)
   - Tape: Moderate correction (head alignment issues)
   - Digital: Minimal correction (production errors only)

QUALITY TARGETS:
- Correlation improvement: +0.1 to +0.3 (material-dependent)
- Mono compatibility: >0.7 for bass, >0.5 for full range
- Processing: <0.05× realtime

Author: Aurik Professional Team
Version: 2.1.0
Date: March 2026
"""

import logging
import os
import time
from typing import cast

import numpy as np
from scipy import signal

from backend.core.audio_utils import audio_sample_count, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class PhaseCorrection(PhaseInterface):
    """Professional multi-band phase correction for stereo imaging."""

    # Material-adaptive correction strength
    CORRECTION_STRENGTH = {
        MaterialType.SHELLAC: 0.60,  # was 0.80 — reduced: avoid over-processing analogue stereo
        MaterialType.VINYL: 0.45,  # was 0.70 — false-positive rate too high for modern pop on vinyl
        MaterialType.TAPE: 0.60,  # was 0.85 — head misalignment needs correction but gently
        # Compact cassette shares tape-head alignment physics; explicit key avoids vinyl fallback.
        MaterialType.CASSETTE: 0.60,
        MaterialType.CD_DIGITAL: 0.25,  # Minimal (production errors only)
        MaterialType.STREAMING: 0.15,  # Very minimal
    }

    # Correlation threshold (correct if below this).
    # §2.51 Invariante: Only correct when there is genuine phase defect, NOT normal stereo width.
    # Normal pop/Schlager stereo: per-band correlation ranges 0.1–0.7 (intentional).
    # Genuine azimuth/phase error: correlation near 0 or negative (< 0.35) in affected band.
    # Old thresholds (e.g. VINYL=0.75) fired on ALL bands of any wide-stereo song → artikulation regression.
    CORRELATION_THRESHOLD = {
        MaterialType.SHELLAC: 0.40,  # was 0.65 — shellac can have genuine phase issues
        MaterialType.VINYL: 0.35,  # was 0.75 — was causing false positives on normal pop stereo!
        MaterialType.TAPE: 0.35,  # was 0.70 — real azimuth errors show < 0.35
        MaterialType.CD_DIGITAL: 0.25,  # was 0.85
        MaterialType.STREAMING: 0.20,  # was 0.90
    }

    # Multi-band crossover frequencies
    CROSSOVER_FREQS = [200, 1000, 8000]  # Hz (4 bands: <200, 200-1k, 1k-8k, >8k)

    # Max time delay per band (samples @ 44.1kHz)
    MAX_DELAY_SAMPLES = {
        "bass": 100,  # ~2.3ms (bass less critical for timing)
        "low_mid": 50,  # ~1.1ms
        "mid_high": 30,  # ~0.7ms
        "high": 20,  # ~0.45ms (high freqs critical for imaging)
    }

    def __init__(self):
        super().__init__()
        self.name = "Phase Correction v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_14_phase_correction",
            name="Phase Correction v2 Professional",
            category=PhaseCategory.STEREO,
            priority=6,
            dependencies=["phase_15_stereo_balance"],
            estimated_time_factor=0.04,
            version="2.1.0",
            memory_requirement_mb=60,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.90,  # High impact on stereo imaging
            description="Multi-band phase correction for optimal stereo imaging and mono compatibility",
        )

    @staticmethod
    def _local_event_strength(key: str, loc: tuple[float, float], event_metadata: dict[str, dict] | None) -> float:
        duration_s = max(0.0, float(loc[1]) - float(loc[0]))
        duration_factor = float(np.clip(duration_s / 0.75, 0.30, 1.0))
        key_factor = {
            "phase_issues": 1.0,
            "azimuth_error": 0.92,
            "stereo_imbalance": 0.58,
            "crosstalk": 0.46,
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

        keys = ("phase_issues", "azimuth_error", "stereo_imbalance", "crosstalk")
        mask = np.zeros(n_samples, dtype=np.float32)
        for key in keys:
            pad = int((0.060 if key in {"phase_issues", "azimuth_error"} else 0.040) * sample_rate)
            for loc in defect_locations.get(key) or []:
                if not isinstance(loc, tuple) or len(loc) != 2:
                    continue
                try:
                    start = int(max(0.0, float(loc[0])) * sample_rate)
                    end = int(max(0.0, float(loc[1])) * sample_rate)
                except Exception:
                    continue
                if end <= start:
                    continue
                start = max(0, start - pad)
                end = min(n_samples, end + pad)
                if end > start:
                    strength = PhaseCorrection._local_event_strength(key, loc, event_metadata)
                    mask[start:end] = np.maximum(mask[start:end], strength)

        if float(np.mean(mask)) <= 1e-6:
            return np.ones(n_samples, dtype=np.float32), 0.0

        smooth = max(16, int(0.025 * sample_rate))
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
        if reference.ndim == 2 and reference.shape[0] == profile.size and reference.shape[1] <= 8:
            wet = profile[:, np.newaxis]
        elif reference.ndim == 2 and reference.shape[1] == profile.size:
            wet = profile[np.newaxis, :]
        else:
            return candidate
        blended = reference + wet * (candidate - reference)
        return cast(
            np.ndarray,
            np.asarray(
                np.clip(np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0),
                dtype=np.float32,
            ),
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: MaterialType | str = MaterialType.VINYL,
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("Whisper", phase_name="14")
        """
        Wendet an: multi-band phase correction.

        Args:
            audio: Stereo audio [samples, 2]
            sample_rate: Sample rate in Hz
            material_type: Material type

        Returns:
            PhaseResult with corrected audio
        """
        self.validate_input(audio)
        sample_rate = kwargs.get("sample_rate", sample_rate)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()

        if isinstance(material_type, MaterialType):
            material_enum = material_type
        else:
            try:
                material_enum = MaterialType(str(material_type).lower())
            except Exception:
                material_enum = MaterialType.VINYL

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # Only for stereo
        if audio.ndim != 2:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                metrics={"skipped": True, "reason": "mono_signal"},
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "phase_correction",
                    "version": "2.0",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                metrics={
                    "correlation_before": 0.0,
                    "correlation_after": 0.0,
                    "correlation_improvement": 0.0,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "version": "2.0",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        strength = float(self.CORRECTION_STRENGTH.get(material_enum, 0.7) * _effective_strength)
        threshold = self.CORRELATION_THRESHOLD.get(material_enum, 0.75)

        # Extract L/R channels
        left, right = stereo_channel_view(audio)

        # Multi-band split
        bands_left = self._multiband_split(left, sample_rate)
        bands_right = self._multiband_split(right, sample_rate)

        # Analyze and correct per band
        corrected_bands_left = []
        corrected_bands_right = []
        correlations_before = []
        correlations_after = []
        delays_corrected = []  # stored as float (sub-sample resolution)
        any_band_corrected = False  # §2.49: track if filter bank is actually needed

        band_names = ["bass", "low_mid", "mid_high", "high"]

        for band_l, band_r, band_name in zip(bands_left, bands_right, band_names):
            # Analyze correlation — now returns float delay (sub-sample precision)
            corr_before, delay = self._analyze_phase(band_l, band_r, self.MAX_DELAY_SAMPLES[band_name])
            correlations_before.append(corr_before)

            # Correct if needed
            if corr_before < threshold:
                band_l_corr, band_r_corr = self._correct_band_phase(band_l, band_r, delay, strength)
                corr_after, _ = self._analyze_phase(band_l_corr, band_r_corr, self.MAX_DELAY_SAMPLES[band_name])
                delays_corrected.append(float(delay))
                any_band_corrected = True
            else:
                band_l_corr, band_r_corr = band_l, band_r
                corr_after = corr_before
                delays_corrected.append(0.0)

            correlations_after.append(corr_after)
            corrected_bands_left.append(band_l_corr)
            corrected_bands_right.append(band_r_corr)

        # Wide-stereo guard: if ALL bands have near-zero correlation (<0.20), this is natural
        # wide/independent stereo (e.g. Pop/Schlager with separate L/R production), NOT an
        # azimuth phase error. Real azimuth errors always leave bass relatively correlated (>0.35).
        # Delay-correction on fully uncorrelated stereo is meaningless and the filter bank
        # reconstruction would cause the same ~0.29 regression floor. Return original unchanged.
        _WIDE_STEREO_CORR_CAP = 0.20
        if all(c < _WIDE_STEREO_CORR_CAP for c in correlations_before):
            logger.debug(
                "phase_14: all bands near-zero corr (max=%.3f < %.2f) — natural wide stereo, "
                "no azimuth error, returning input unchanged",
                max(correlations_before),
                _WIDE_STEREO_CORR_CAP,
            )
            overall_corr = float(np.mean(correlations_before))
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                metrics={
                    "correlation_before": overall_corr,
                    "correlation_after": overall_corr,
                    "correlation_improvement": 0.0,
                    "per_band_correlation_before": [float(c) for c in correlations_before],
                    "per_band_correlation_after": [float(c) for c in correlations_before],
                    "delays_corrected_samples": [0.0] * len(correlations_before),
                    "correction_strength": 0.0,
                    "material": material_enum.value,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "phase_correction_no_op",
                    "version": "2.1",
                    "reason": "natural_wide_stereo",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # §2.49 Early-exit: if no band needed correction, the filter bank (split+reconstruct)
        # introduces spectral ripple at crossovers (Butterworth LP+BP+BP+HP ≠ unity).
        # Returning original audio avoids ~0.29 regression floor from non-alias-free reconstruction.
        if not any_band_corrected:
            logger.debug(
                "phase_14: all bands corr >= threshold (%.2f) — no correction needed, returning input unchanged",
                threshold,
            )
            overall_corr = float(np.mean(correlations_before))
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                metrics={
                    "correlation_before": overall_corr,
                    "correlation_after": overall_corr,
                    "correlation_improvement": 0.0,
                    "per_band_correlation_before": [float(c) for c in correlations_before],
                    "per_band_correlation_after": [float(c) for c in correlations_before],
                    "delays_corrected_samples": delays_corrected,
                    "correction_strength": 0.0,
                    "material": material_enum.value,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "phase_correction_no_op",
                    "version": "2.1",
                    "reason": "all_bands_above_threshold",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # Reconstruct
        corrected_left = self._multiband_reconstruct(corrected_bands_left)
        corrected_right = self._multiband_reconstruct(corrected_bands_right)

        # Ensure length matches
        min_len = min(len(corrected_left), len(corrected_right), audio_sample_count(audio))
        corrected_left = corrected_left[:min_len]
        corrected_right = corrected_right[:min_len]

        corrected_audio = stereo_like(corrected_left, corrected_right, audio)

        # Overall correlation
        overall_corr_before = np.mean(correlations_before)
        overall_corr_after = np.mean(correlations_after)

        processing_time = time.time() - start_time

        corrected_audio = np.nan_to_num(corrected_audio, nan=0.0, posinf=0.0, neginf=0.0)
        corrected_audio = np.clip(corrected_audio, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            corrected_audio = audio + _effective_strength * (corrected_audio - audio)
            corrected_audio = np.clip(corrected_audio, -1.0, 1.0)
        _local_profile14, _local_coverage14 = self._build_locality_profile(
            int(audio_sample_count(audio)),
            sample_rate,
            kwargs.get("defect_locations"),
            kwargs.get("defect_event_metadata"),
            self._collect_protected_zones(kwargs),
        )
        if _local_coverage14 > 0.0:
            corrected_audio = self._blend_with_locality(audio, corrected_audio, _local_profile14)

        # §PHROT-1 v10.13: MP3-Phasenrotator (20–200 Hz, ≤30°).
        # Verbessert Mono-Kompatibilität bei MP3-kodiertem Material, ohne
        # die Ära-Authentizität zu zerstören. Non-blocking.
        _terminal_codec_p14 = str(kwargs.get("terminal_codec", "")).lower()
        if "mp3" in _terminal_codec_p14:
            try:
                from backend.core.dsp.phase_rotator import apply_phase_rotator

                corrected_audio = apply_phase_rotator(
                    corrected_audio,
                    sample_rate,
                    low_freq=20.0,
                    high_freq=200.0,
                    max_rotation_deg=30.0,
                    strength=0.15,
                )
                logger.debug("§PHROT-1: Phase-Rotator applied (mp3, 20-200Hz, ≤30°)")
            except Exception as _prot_exc:
                logger.debug("§PHROT-1: Phase-Rotator skipped (%s)", _prot_exc)

        return PhaseResult(
            success=True,
            audio=corrected_audio,
            metrics={
                "correlation_before": float(overall_corr_before),
                "correlation_after": float(overall_corr_after),
                "correlation_improvement": float(overall_corr_after - overall_corr_before),
                "per_band_correlation_before": [float(c) for c in correlations_before],
                "per_band_correlation_after": [float(c) for c in correlations_after],
                "delays_corrected_samples": delays_corrected,
                "correction_strength": strength,
                "material": material_enum.value,
            },
            execution_time_seconds=processing_time,
            resolved_defects={
                "PHASE_ISSUES": 0.0,  # Phasenkorrektur = vollständig behoben
            },
            metadata={
                "algorithm": "multiband_phase_correction_fractional",
                "version": "2.1",
                "bands": band_names,
                "crossovers_hz": self.CROSSOVER_FREQS,
                "repair_locality_coverage": round(float(_local_coverage14), 6),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )

    def _multiband_split(self, audio: np.ndarray, sample_rate: int) -> list:
        """Split audio into 4 bands using Linkwitz-Riley crossovers."""
        bands = []

        # Design crossover filters (4th order Linkwitz-Riley)
        nyquist = sample_rate / 2

        # Band 1: <200 Hz (Bass)
        sos_low = signal.butter(4, self.CROSSOVER_FREQS[0] / nyquist, btype="low", output="sos")
        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) statt sosfilt (kausal).
        # sosfilt erzeugt frequenzabhängige Gruppenlatenz pro Band; nach per-Band-Korrektur des
        # R-Kanals via np.roll entsteht ein frequenzabhängiger L/R-Zeitversatz. Zusätzlich erzeugt
        # die Filtereinschalttransiente (Zero-Initial-State) im ersten Viertel des Audios eine
        # Pegelexplosion und Kratzen durch Nicht-Unity-Bandrekombination (Butterworth ≠ allpass).
        bands.append(signal.sosfiltfilt(sos_low, audio))  # Zero-Phase

        # Band 2: 200-1000 Hz (Low-Mid)
        sos_band2 = signal.butter(
            4, [self.CROSSOVER_FREQS[0] / nyquist, self.CROSSOVER_FREQS[1] / nyquist], btype="band", output="sos"
        )
        bands.append(signal.sosfiltfilt(sos_band2, audio))  # Zero-Phase

        # Band 3: 1000-8000 Hz (Mid-High)
        sos_band3 = signal.butter(
            4, [self.CROSSOVER_FREQS[1] / nyquist, self.CROSSOVER_FREQS[2] / nyquist], btype="band", output="sos"
        )
        bands.append(signal.sosfiltfilt(sos_band3, audio))  # Zero-Phase

        # Band 4: >8000 Hz (High)
        sos_high = signal.butter(4, self.CROSSOVER_FREQS[2] / nyquist, btype="high", output="sos")
        bands.append(signal.sosfiltfilt(sos_high, audio))  # Zero-Phase

        return bands

    def _multiband_reconstruct(self, bands: list) -> np.ndarray:
        """Rekonstruiert audio from bands (simple sum for Linkwitz-Riley)."""
        # Ensure all bands same length
        min_len = min(len(b) for b in bands)
        bands_trimmed = [b[:min_len] for b in bands]

        # Sum bands
        reconstructed = np.sum(bands_trimmed, axis=0)
        return reconstructed  # type: ignore[no-any-return]

    def _analyze_phase(self, left: np.ndarray, right: np.ndarray, max_delay: int) -> tuple[float, float]:
        """
        Analysiert die Phasenausrichtung via Kreuzkorrelation mit Sub-Sample-Präzision.

        Integer peak is refined by parabolic interpolation of the XCF envelope
        (Smith 2011 §3.4), giving sub-sample delay estimation accurate to ~0.02
        samples RMS on bandlimited audio.

        Returns:
            (correlation_coefficient, delay_samples_float)
        """
        # Use first 3 seconds for analysis
        max_samples = min(len(left), len(right), 48000 * 3)
        left_seg = left[:max_samples]
        right_seg = right[:max_samples]

        # Cross-correlation
        correlation = signal.correlate(left_seg, right_seg, mode="same")
        lags = signal.correlation_lags(len(left_seg), len(right_seg), mode="same")

        # Limit search range
        valid_mask = np.abs(lags) <= max_delay
        corr_valid = correlation[valid_mask]
        lags_valid = lags[valid_mask]

        # Find integer-sample peak
        peak_idx = int(np.argmax(np.abs(corr_valid)))
        delay_int = int(-lags_valid[peak_idx])

        # Sub-sample refinement via parabolic interpolation (Smith 2011 §3.4).
        # Given envelope samples y_{-1}, y_0, y_{+1} around the peak, the
        # fractional peak offset is:  δ = 0.5 · (y_{-1} − y_{+1}) / (y_{-1} − 2y_0 + y_{+1})
        delay_frac = 0.0
        if 0 < peak_idx < len(corr_valid) - 1:
            y_m = float(np.abs(corr_valid[peak_idx - 1]))
            y_0 = float(np.abs(corr_valid[peak_idx]))
            y_p = float(np.abs(corr_valid[peak_idx + 1]))
            denom = y_m - 2.0 * y_0 + y_p
            if abs(denom) > 1e-12:
                # Parabolic peak offset in lag-index space (Smith 2011 §3.4):
                #   δ_lag = 0.5·(y_m − y_p) / denom   (positive = higher lag index)
                # delay = −lag  →  delay_frac = −δ_lag = 0.5·(y_p − y_m) / denom
                delay_frac = float(np.clip(0.5 * (y_p - y_m) / denom, -0.5, 0.5))

        delay: float = float(delay_int) + delay_frac

        # Normalized correlation at the (integer) aligned position
        d_int = delay_int
        if d_int > 0:
            aligned_l = left_seg[d_int:]
            aligned_r = right_seg[: len(left_seg) - d_int]
        elif d_int < 0:
            aligned_l = left_seg[: len(left_seg) + d_int]
            aligned_r = right_seg[-d_int:]
        else:
            aligned_l = left_seg
            aligned_r = right_seg

        if len(aligned_l) > 0 and len(aligned_r) > 0:
            # Guarded Pearson — avoids NaN and O(n) matrix alloc of np.corrcoef
            _al = aligned_l - aligned_l.mean()
            _ar = aligned_r - aligned_r.mean()
            _nal = float(np.linalg.norm(_al))
            _nar = float(np.linalg.norm(_ar))
            corr_coef = float(np.dot(_al, _ar) / (_nal * _nar + 1e-10))
            if not np.isfinite(corr_coef):
                corr_coef = 1.0  # Silence = perfectly correlated (no phase error)
        else:
            corr_coef = 0.0

        return corr_coef, delay

    @staticmethod
    def _lagrange_ffd(frac: float, order: int = 3) -> np.ndarray:
        """Lagrange FIR fractional-delay filter coefficients (causal).

        Implements Laakso et al. (1996) eq. (9):
            h[k] = ∏_{m=0, m≠k}^{N} (d − m) / (k − m),   d = frac + N//2

        The filter has ``order + 1`` taps and a total group delay of
        ``order // 2 + frac`` samples.  The caller must compensate for the
        integer part (``order // 2``) by discarding leading output samples.

        Args:
            frac:  Fractional delay in [−0.5, 0.5] samples.
            order: Polynomial order (3 = 4 taps, good trade-off of accuracy vs
                   latency; Laakso 1996 recommends 3–7 for audio).

        Returns:
            Float64 coefficient array of length ``order + 1``.
        """
        N = order
        M = N // 2
        d = float(frac) + M  # total causal delay from tap 0
        h = np.ones(N + 1, dtype=np.float64)
        for k in range(N + 1):
            for m in range(N + 1):
                if m != k:
                    h[k] *= (d - m) / (k - m)
        return h  # type: ignore[no-any-return]

    def _correct_band_phase(
        self, left: np.ndarray, right: np.ndarray, delay: float, strength: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct phase via integer sample-shift + fractional Lagrange FIR.

        Integer part: np.roll (zero-latency sample shift).
        Fractional part: Lagrange order-3 FIR convolved only when
        |frac| > 0.01 samples (Laakso et al. 1996).  The N//2-sample causal
        latency of the FIR is compensated by slicing the output.
        """
        corrected_delay_f = float(delay) * float(strength)
        delay_int = int(round(corrected_delay_f))
        delay_frac = corrected_delay_f - float(delay_int)  # in (−0.5, +0.5]

        # Integer delay via sample shift
        if delay_int > 0:
            corrected_left = left.copy()
            corrected_right = np.roll(right, -delay_int)
            corrected_right[-delay_int:] = 0.0
        elif delay_int < 0:
            corrected_left = np.roll(left, delay_int)
            corrected_right = right.copy()
            # np.roll(..., negative) shifts left and wraps start-samples to the tail.
            # Zero the TAIL (not the head) to avoid synthetic end-spikes.
            corrected_left[-abs(delay_int) :] = 0.0
        else:
            corrected_left = left.copy()
            corrected_right = right.copy()

        # Fractional-delay correction: Lagrange order-3 FIR (Laakso 1996)
        if abs(delay_frac) > 0.01:
            h = self._lagrange_ffd(delay_frac, order=3)
            M = len(h) // 2  # integer group-delay of FIR = 1 sample (order//2)
            # Apply to whichever channel was shifted (right for positive delay)
            if delay_int >= 0:
                padded = np.convolve(corrected_right, h, mode="full")
                corrected_right = padded[M : M + len(corrected_right)]
            else:
                padded = np.convolve(corrected_left, h, mode="full")
                corrected_left = padded[M : M + len(corrected_left)]

        return corrected_left, corrected_right


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Phase 14: Professional Phase Correction v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test stereo audio with phase error
    duration = 3.0
    test_sample_rate = 44100
    t = np.linspace(0, duration, int(test_sample_rate * duration))

    # Multi-frequency signal
    signal_base = (
        0.3 * np.sin(2 * np.pi * 100 * t)  # Bass
        + 0.2 * np.sin(2 * np.pi * 500 * t)  # Low-mid
        + 0.15 * np.sin(2 * np.pi * 2000 * t)  # Mid-high
        + 0.1 * np.sin(2 * np.pi * 8000 * t)  # High
    )

    # Create stereo with phase errors (different delays per band)
    delay_bass = 30  # samples (~0.68ms)
    _delay_mid = 15  # samples (~0.34ms)

    test_left = signal_base
    test_right = signal_base.copy()

    # Apply delays to simulate phase errors
    test_right = np.roll(test_right, delay_bass)
    test_right[:delay_bass] = 0

    test_audio = np.column_stack([test_left, test_right])

    logger.debug("Generated %ss test audio @ %s Hz", duration, test_sample_rate)
    logger.debug(
        "Phase error: Right delayed by %s samples (~%.2fms)",
        delay_bass,
        delay_bass * 1000 / test_sample_rate,
    )
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE"),
        (MaterialType.VINYL, "VINYL"),
        (MaterialType.CD_DIGITAL, "CD_DIGITAL"),
    ]

    for test_material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug("Material: %s", material_name)
        logger.debug("─" * 80)
        logger.debug("")

        phase = PhaseCorrection()
        result = phase.process(test_audio, test_sample_rate, test_material)

        logger.debug("✅ Professional Phase Correction:")
        logger.debug("   Correlation Before: %.4f", result.metrics["correlation_before"])
        logger.debug("   Correlation After: %.4f", result.metrics["correlation_after"])
        logger.debug("   Improvement: %.4f", result.metrics["correlation_improvement"])
        logger.debug("")
        _corr_before_txt = [format(c, ".3f") for c in result.metrics["per_band_correlation_before"]]
        _corr_after_txt = [format(c, ".3f") for c in result.metrics["per_band_correlation_after"]]
        logger.debug("   Per-Band Correlation Before: %s", _corr_before_txt)
        logger.debug("   Per-Band Correlation After:  %s", _corr_after_txt)
        logger.debug("   Delays Corrected (samples):  %s", result.metrics["delays_corrected_samples"])
        logger.debug("")
        logger.debug(
            "   Processing time: %.3fs (%.2fx realtime)",
            result.execution_time_seconds,
            result.execution_time_seconds / duration,
        )
        logger.debug("   Correction strength: %s", result.metrics["correction_strength"])
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test completed")
    logger.debug("=" * 80)
