#!/usr/bin/env python3
"""
Phase 51: Drums/Percussion Enhancement v1.0 - Tier 1 ML-Hybrid.
Professional drums and percussion processing for instrumental restoration.

Algorithm Overview:
1. Kick Drum Enhancement (20-80 Hz sub-bass punch)
2. Snare Crack Enhancement (200-400 Hz + 1-3 kHz articulation)
3. Hi-Hat Clarification (8-12 kHz clarity and presence)
4. Cymbal Shimmer Enhancement (12-20 kHz air and shimmer)
5. Transient shaping (attack/sustain optimization)

Components:
- KickDrumEnhancer: Sub-bass reinforcement
- SnareCrackEnhancer: Articulation and definition
- HiHatClarifier: High-frequency clarity
- CymbalShimmerEnhancer: Air band processing

Scientific Foundation:
- Masri (1996): Computer Modeling of Sound for Transformation and Synthesis of Musical Signals
- Duxbury et al. (2003): Complex Domain Onset Detection for Musical Signals
- Bello et al. (2005): A Tutorial on Onset Detection in Music Signals
- Amatriain et al. (2003): CLAM: A Framework for Efficient and Rapid Development of Cross-platform Audio Applications

Industry Benchmarks:
- iZotope Nectar/Trash (Transient shaping)
- Waves Dbx 160 (Drum compression)
- SPL Transient Designer (Transient control)
- SSL Drumstrip (Professional drum processing)
- Slate Digital Trigger (Drum enhancement)

Tier 1 Priority: PRIORITY 2 (after Bass, critical for groove and punch)
Quality Target: Transparenz 87% → 93% (+6% for percussive content)
Performance Target: <0.25× realtime

Author: Aurik Development Team  Phase 2.3 - Tier 1 ML-Hybrid
Version: 1.0.0
Date: 16. Februar 2026
"""

import logging
import time

import numpy as np

from backend.core.audio_utils import to_channels_last
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .output_guard import evaluate_output_guard
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# Import Drums Enhancement DSP module
try:
    from dsp.drums_enhancement import DrumsEnhancementSystem

    DRUMS_ENHANCEMENT_AVAILABLE = True
except ImportError:
    DRUMS_ENHANCEMENT_AVAILABLE = False
    logging.warning("DrumsEnhancementSystem not available")

try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls

    _FORMANT_SYSTEM_DRUMS_STATE: dict = {"instance": None}
except Exception:
    _FormantSystemCls = None  # type: ignore[assignment,misc]
    _FORMANT_SYSTEM_DRUMS_STATE = {"instance": None}

try:
    from dsp.instrument_formant_corrector import (
        correct_instrument_formant_drift as _correct_instrument_formant_drift_fn,
    )
except Exception:
    _correct_instrument_formant_drift_fn = None  # type: ignore[assignment]

try:
    from backend.core.sub_stem_processor import process_sub_stems as _process_sub_stems_fn
except Exception:
    _process_sub_stems_fn = None  # type: ignore[assignment]

try:
    from backend.core.physics_resonance_enhancer import (
        enhance_physics_resonance as _enhance_physics_resonance_fn,
    )
except Exception:
    _enhance_physics_resonance_fn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DrumsEnhancementV1(PhaseInterface):
    """
    Professional Drums/Percussion Enhancement Engine (Tier 1).

    Key Features:
    - Kick drum sub-bass punch (20-80 Hz)
    - Snare articulation and crack (200-400 Hz, 1-3 kHz)
    - Hi-hat clarity (8-12 kHz)
    - Cymball shimmer (12-20 kHz)
    - Transient shaping
    - Material-adaptive intensity

    Use Cases:
    - Restore lost drum impact from over-processing
    - Enhance percussive elements in mixes
    - Improve groove and punch
    - Broadcast and mastering drum optimization

    Performance: <0.25× realtime on modern CPU
    """

    # Material-specific enhancement parameters
    ENHANCEMENT_CONFIG = {
        MaterialType.SHELLAC: {
            "kick_gain_db": 4.0,  # Strong (restore missing low-end)
            "snare_articulation": 0.9,  # High (restore attack)
            "hihat_clarity_db": 3.0,  # Strong (restore HF)
            "cymbal_shimmer_db": 2.5,
            "transient_enhancement": 0.8,
            "mix": 0.60,  # 60% enhancement
        },
        MaterialType.VINYL: {
            "kick_gain_db": 3.0,  # Moderate
            "snare_articulation": 0.75,
            "hihat_clarity_db": 2.5,
            "cymbal_shimmer_db": 2.0,
            "transient_enhancement": 0.7,
            "mix": 0.50,  # 50% enhancement
        },
        MaterialType.TAPE: {
            "kick_gain_db": 2.5,  # Gentle (tape has good bass)
            "snare_articulation": 0.6,
            "hihat_clarity_db": 2.0,  # Restore HF loss
            "cymbal_shimmer_db": 1.5,
            "transient_enhancement": 0.5,
            "mix": 0.40,  # 40% enhancement
        },
        MaterialType.CASSETTE: {  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
            "kick_gain_db": 2.5,
            "snare_articulation": 0.6,
            "hihat_clarity_db": 2.0,
            "cymbal_shimmer_db": 1.5,
            "transient_enhancement": 0.5,
            "mix": 0.40,
        },
        MaterialType.CD_DIGITAL: {
            "kick_gain_db": 2.0,  # Subtle
            "snare_articulation": 0.5,
            "hihat_clarity_db": 1.5,
            "cymbal_shimmer_db": 1.0,
            "transient_enhancement": 0.4,
            "mix": 0.30,  # 30% enhancement
        },
        MaterialType.STREAMING: {
            "kick_gain_db": 1.5,  # Minimal (already processed)
            "snare_articulation": 0.4,
            "hihat_clarity_db": 1.0,
            "cymbal_shimmer_db": 0.8,
            "transient_enhancement": 0.3,
            "mix": 0.25,  # 25% enhancement
        },
    }

    DEFAULT_CONFIG = {
        "kick_gain_db": 2.5,
        "snare_articulation": 0.6,
        "hihat_clarity_db": 2.0,
        "cymbal_shimmer_db": 1.5,
        "transient_enhancement": 0.5,
        "mix": 0.40,
    }

    def __init__(self, sample_rate: int = 48000, **kwargs):
        """
        Initialisiert Drums Enhancement Phase.

        Args:
            sample_rate: Audio sample rate (Hz)
            **kwargs: Override parameters
        """
        super().__init__(sample_rate, **kwargs)
        self.enhancer = None  # Lazy init

    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: MaterialType = MaterialType.CD_DIGITAL,
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="51")
        """
        Enhance drums and percussion.

        Args:
            audio: Input audio (mono or stereo)
            material_type: Source material type
            **kwargs: Additional parameters

        Returns:
            PhaseResult with enhanced audio
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p51_transposed = to_channels_last(audio)
        start_time = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # §V41 ForwardMaskingGuard — Enhancement-Stärke in post-transienten Masking-Zonen erhöhen
        _panns_s_51 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_51 >= 0.25 and effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_51,
                )

                _fmz_51 = kwargs.get("forward_masking_zones") or _fmg_fn_51().compute_zones(audio, sample_rate)
                if _fmz_51:
                    _n_s_51 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_s_51 = sum(z.end_sample - z.start_sample for z in _fmz_51)
                    _zone_frac_51 = float(np.clip(_zone_s_51 / max(1, _n_s_51), 0.0, 1.0))
                    effective_strength = float(np.clip(effective_strength + _zone_frac_51 * 0.15, 0.0, 1.0))
            except Exception as _fmg_exc_51:
                logger.debug("Phase51 §V41 ForwardMaskingGuard non-blocking: %s", _fmg_exc_51)

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                metrics={"effective_strength": 0.0},
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[],
                modifications={},
            )

        # §2.46g soft_saturation-Guard: Kick-Boost auf gesättigtem Material macht
        # Sättigungsverzerrung auf Transienten hörbarer. Hard-Cap: 45 %.
        _p51_soft_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        _p51_soft_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        if _p51_soft_sat_preserve or _p51_soft_sat_sev > 0.35:
            _p51_sat_scale = 1.0
            if _p51_soft_sat_sev > 0.35:
                _p51_sat_scale = float(np.clip(1.0 - (_p51_soft_sat_sev - 0.35) * 0.7, 0.30, 1.0))
            if _p51_soft_sat_preserve and _p51_sat_scale > 0.45:
                _p51_sat_scale = 0.45
            effective_strength = float(effective_strength * _p51_sat_scale)
            logger.debug(
                "Phase 51 soft_saturation guard: severity=%.2f preserve=%s → scale=%.2f (strength=%.3f)",
                _p51_soft_sat_sev,
                _p51_soft_sat_preserve,
                _p51_sat_scale,
                effective_strength,
            )

        # ── PANNs Drums-Confidence-Check (Spec §2.9: Schwellwert ≥ 0.5) ────
        panns_tags = kwargs.get("panns_tags", {})
        drums_confidence = 0.0
        for tag_name, conf in panns_tags.items():
            tag_lower = tag_name.lower()
            if any(k in tag_lower for k in ("drum", "percussion", "kick", "snare", "hihat", "cymbal")):
                drums_confidence = max(drums_confidence, float(conf))
        # Wenn PANNs-Tags vorhanden aber Drums nicht erkannt → Phase überspringen
        if panns_tags and drums_confidence < 0.50:
            logger.info(
                "Phase 51: Drums-Confidence %.2f < 0.50 — Phase übersprungen (Spec §2.9)",
                drums_confidence,
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                metrics={
                    "skipped": True,
                    "drums_confidence": drums_confidence,
                    "effective_strength": effective_strength,
                },
                execution_time_seconds=0.0,
                metadata={
                    "algorithm": "skip_panns_confidence",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[],
                modifications={},
            )
        if drums_confidence > 0:
            logger.info("Phase 51: PANNs Drums-Confidence=%.2f ≥ 0.50 — Verarbeitung aktiv", drums_confidence)

        if not DRUMS_ENHANCEMENT_AVAILABLE:
            logger.warning("DrumsEnhancementSystem not available, bypassing...")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=False,
                audio=audio,
                execution_time_seconds=time.time() - start_time,
                warnings=["DrumsEnhancementSystem module not available"],
            )

        try:
            # Get material-specific config
            config = self.ENHANCEMENT_CONFIG.get(material_type, self.DEFAULT_CONFIG).copy()
            config.update(kwargs)  # Allow override

            quality_mode = str(kwargs.get("quality_mode", "balanced")).lower()
            if quality_mode in ("quality", "maximum", "studio2026"):
                hq_scale = 1.12 if quality_mode in ("maximum", "studio2026") else 1.06
                config["mix"] = float(np.clip(config["mix"] * hq_scale, 0.0, 0.75))
                config["kick_gain_db"] = float(np.clip(config["kick_gain_db"] * hq_scale, 0.0, 5.5))
                config["hihat_clarity_db"] = float(np.clip(config["hihat_clarity_db"] * hq_scale, 0.0, 4.0))
                config["cymbal_shimmer_db"] = float(np.clip(config["cymbal_shimmer_db"] * hq_scale, 0.0, 3.5))
            else:
                hq_scale = 1.0

            # Lazy init enhancer
            if self.enhancer is None:
                self.enhancer = DrumsEnhancementSystem(  # type: ignore[assignment]
                    kick_gain_db=config["kick_gain_db"],
                    snare_articulation=config["snare_articulation"],
                    hihat_clarity_db=config["hihat_clarity_db"],
                    cymbal_shimmer_db=config["cymbal_shimmer_db"],
                )

            # Process audio
            processed_audio, report = self.enhancer.process(audio, self.sample_rate)  # type: ignore[union-attr,attr-defined]

            # Mix with original (parallel processing)
            mix = float(np.clip(config["mix"], 0.0, 1.0)) * effective_strength
            enhanced = audio * (1.0 - mix) + processed_audio * mix

            execution_time = time.time() - start_time

            # Build metrics
            metrics = {
                "kick_energy_change_db": report.get("kick_energy_change_db", 0.0),
                "snare_energy_change_db": report.get("snare_energy_change_db", 0.0),
                "hihat_energy_change_db": report.get("hihat_energy_change_db", 0.0),
                "cymbal_energy_change_db": report.get("cymbal_energy_change_db", 0.0),
                "transient_enhancement": report.get("transient_enhancement", 0.0),
                "mix_ratio": mix,
                "effective_strength": effective_strength,
                "material_type": material_type.value,
                "config_applied": config,
            }

            enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced = np.clip(enhanced, -1.0, 1.0)
            enhanced_pre_refine = enhanced.copy()

            # Instrument-guided formant enhancement (drums resonance targets: Rossing 1992)
            igt_frames = 0
            try:
                if _FormantSystemCls is not None:
                    if _FORMANT_SYSTEM_DRUMS_STATE["instance"] is None:
                        _FORMANT_SYSTEM_DRUMS_STATE["instance"] = _FormantSystemCls(enhance_singers_formant=False)
                    formant_strength = float(np.clip(0.15 * hq_scale, 0.10, 0.22))
                    enhanced, igt_report = _FORMANT_SYSTEM_DRUMS_STATE["instance"].instrument_guided_enhance(
                        enhanced,
                        self.sample_rate,
                        instrument="drums",
                        correction_strength=formant_strength,
                    )
                    igt_frames = igt_report.get("frames_processed", 0)
                    logger.debug("Phase 51 InstrumentFormant: drums frames=%d", igt_frames)
            except Exception as _igt_exc:
                logger.debug("Phase 51 instrument_guided_enhance skipped: %s", _igt_exc)

            # Formant-Drift-Korrektur via DTW (Schritt 3)
            try:
                drift_result = _correct_instrument_formant_drift_fn(  # type: ignore[misc]
                    enhanced, sample_rate, instrument="drums"
                )
                enhanced = drift_result.audio
                logger.debug(
                    "Phase 51 drift correction: detected=%s frames=%d/%d drift=%.1fHz",
                    drift_result.drift_detected,
                    drift_result.n_frames_corrected,
                    drift_result.total_frames,
                    drift_result.mean_drift_hz,
                )
            except Exception as _drift_exc:
                logger.debug("Phase 51 drift correction skipped: %s", _drift_exc)

            # Sub-Stem-Verarbeitung (Schritt 4)
            try:
                sub_stem_strength = float(np.clip(0.30 * hq_scale, 0.25, 0.40))
                ss_result = _process_sub_stems_fn(  # type: ignore[misc]
                    enhanced, sample_rate, instrument="drums", processing_strength=sub_stem_strength
                )
                enhanced = ss_result.audio
                logger.debug(
                    "Phase 51 sub-stem: bands=%d strength=%.2f", ss_result.n_bands, ss_result.processing_strength
                )
            except Exception as _ss_exc:
                logger.debug("Phase 51 sub-stem skipped: %s", _ss_exc)

            # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
            try:
                physics_strength = float(np.clip(0.35 * hq_scale, 0.30, 0.48))
                pr_result = _enhance_physics_resonance_fn(  # type: ignore[misc]
                    enhanced, sample_rate, instrument="drums", enhancement_strength=physics_strength
                )
                enhanced = pr_result.audio
                logger.debug(
                    "Phase 51 physics resonance: peaks=%d strength=%.2f",
                    pr_result.n_peaks,
                    pr_result.enhancement_strength,
                )
            except Exception as _pr_exc:
                logger.debug("Phase 51 physics resonance skipped: %s", _pr_exc)

            if 0.0 < effective_strength < 1.0:
                enhanced = audio + effective_strength * (enhanced - audio)

            enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced = np.clip(enhanced, -1.0, 1.0)

            # Conservative output guard for high quality modes only.
            output_guard_enabled = quality_mode in ("quality", "maximum", "studio2026")
            guard = evaluate_output_guard(
                original=audio,
                candidate=enhanced,
                enabled=output_guard_enabled,
                max_abs_rms_delta_db=1.2,
                stereo_side_ratio_min=0.55,
                stereo_side_ratio_max=1.45,
            )

            if guard.fallback:
                enhanced = enhanced_pre_refine.copy()
                if 0.0 < effective_strength < 1.0:
                    enhanced = audio + effective_strength * (enhanced - audio)
                enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
                enhanced = np.clip(enhanced, -1.0, 1.0)

            # §2.46e HallucinationGuard: Additive Phase darf kein halluziniertes Material einführen (VERBOTEN)
            try:
                from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                    check_hallucination as _chk_hg51,
                )

                _hg_mode_51 = str(kwargs.get("mode", "restoration"))
                _pre51_mono = (
                    audio.mean(axis=0)
                    if (audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2)
                    else (audio.mean(axis=1) if audio.ndim == 2 else audio)
                )
                _post51_mono = (
                    enhanced.mean(axis=0)
                    if (enhanced.ndim == 2 and enhanced.shape[0] == 2 and enhanced.shape[1] > 2)
                    else (enhanced.mean(axis=1) if enhanced.ndim == 2 else enhanced)
                )
                _hg51 = _chk_hg51(
                    _pre51_mono.astype(np.float32),
                    _post51_mono.astype(np.float32),
                    sr=sample_rate,
                    mode=_hg_mode_51,
                    bw_extension_context=True,
                )
                if _hg51.requires_rollback:
                    enhanced = audio.copy()
                    logger.warning("§2.46e phase_51 HallucinationGuard: rollback (spectral_novelty > 0.15)")
            except Exception as _hg51_exc:
                logger.debug("§2.46e phase_51 HallucinationGuard (non-blocking): %s", _hg51_exc)

            return PhaseResult(
                success=True,
                audio=enhanced,
                execution_time_seconds=execution_time,
                metadata={
                    **metrics,
                    "instrument_formant_frames": igt_frames,
                    "quality_mode": quality_mode,
                    "hq_scale": hq_scale,
                    "output_guard_enabled": output_guard_enabled,
                    "output_guard_fallback": guard.fallback,
                    "output_guard_reason": guard.reason,
                    "rms_delta_db": guard.rms_delta_db,
                    "stereo_side_ratio": guard.stereo_side_ratio,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                modifications={
                    "drums_enhanced": True,
                    "kick_enhanced": report.get("kick_energy_change_db", 0) > 0.5,
                    "snare_enhanced": report.get("snare_energy_change_db", 0) > 0.5,
                    "hihat_enhanced": report.get("hihat_energy_change_db", 0) > 0.3,
                },
            )

        except Exception as e:
            logger.error("Drums enhancement failed: %s", e, exc_info=True)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=False,
                audio=audio,  # Return original audio
                execution_time_seconds=time.time() - start_time,
                warnings=[f"Drums enhancement failed: {e!s}"],
            )

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_51_drums_enhancement",
            name="Drums/Percussion Enhancement (Tier 1)",
            category=PhaseCategory.ENHANCEMENT,
            priority=8,  # High priority (Tier 1)
            dependencies=[],  # Independent
            estimated_time_factor=0.15,  # 15% of audio duration
            version="1.0.0",
            memory_requirement_mb=100,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.85,  # High impact on percussive content
            description=(
                "Professional drums and percussion enhancement: kick punch, snare crack, hi-hat clarity, cymbal shimmer"
            ),
        )

    def supports_material(self, material_type: MaterialType) -> bool:
        """Prüft if material type is supported."""
        return material_type in self.ENHANCEMENT_CONFIG or material_type in [
            MaterialType.CD_DIGITAL,
            MaterialType.DAT,
            MaterialType.MP3_LOW,
            MaterialType.MP3_HIGH,
            MaterialType.AAC,
            MaterialType.STREAMING,
        ]

    def estimate_time(self, audio_duration_seconds: float) -> float:
        """Schätzt processing time."""
        return audio_duration_seconds * 0.15  # 15% of audio duration

    def validate_input(self, audio: np.ndarray) -> tuple[bool, str | None]:
        """Validiert input audio."""
        if audio.size == 0:
            return False, "Empty audio input"
        if not np.isfinite(audio).all():
            return False, "Audio contains NaN or Inf values"
        if audio.ndim > 2:
            return False, "Audio must be mono or stereo"
        return True, None


# Export the class
__all__ = ["DrumsEnhancementV1"]
