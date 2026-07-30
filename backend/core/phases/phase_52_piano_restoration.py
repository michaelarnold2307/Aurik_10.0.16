#!/usr/bin/env python3
"""
Phase 52: Piano Restoration System v1.0 - Tier 1 ML-Hybrid
Professional piano restoration for classical, jazz, and modern recordings.

Algorithm Overview:
1. Hammer Transient Enhancement (Attack Clarity)
   - Transient detection (onset detection)
   - Attack shaping (5-20ms window)
   - Frequency focus: 2-8 kHz (hammer/string impact)
   - Material-adaptive intensity

2. String Resonance Enhancement (Sympathetic Vibrations)
   - Harmonic series analysis (F0 + overtones)
   - Sympathetic string modeling (decay enhancement)
   - String coupling simulation (adjacent notes)
   - Sustain enhancement (natural decay curve)

3. Pedal Noise Reduction
   - Pedal event detection (mechanical noise @ 100-500 Hz)
   - Context-aware suppression (preserve musical intent)
   - Thump reduction (damper pedal lift/press)
   - Intelligibility: Keep subtle pedal noise for realism

4. Dynamic Range Restoration
   - Velocity curve optimization (soft/medium/hard attacks)
   - Micro-dynamics enhancement (note-to-note variation)
   - Compression artifact removal (over-compressed recordings)
   - Material-adaptive expansion ratios

Components:
- HammerTransientEnhancer: Attack clarity and definition
- StringResonanceModeler: Sympathetic vibrations and sustain
- PedalNoiseReducer: Mechanical noise suppression
- PianoDynamicsRestorer: Dynamic range optimization

Scientific Foundation:
- Fletcher & Rossing (1998): The Physics of Musical Instruments (Piano chapter)
- Askenfelt & Jansson (1990): From Touch to String Vibrations (Piano mechanics)
- Bank & Sujbert (2005): Generation of Piano Tones Using Physical Models
- Stulov (2005): Hysteretic Model of the Grand Piano Hammer Felt
- Giordano & Jiang (2004): Physical Modeling of the Piano
- Välimäki et al. (2006): Physics-Based Sound Synthesis of the Piano

Industry Benchmarks:
- iZotope RX 10 De-Click (Transient processing)
- Waves Abbey Road Saturator (Harmonic enhancement)
- Sonnox Oxford Dynamics (Piano dynamics)
- FabFilter Pro-Q 3 (Surgical EQ for piano)
- Acustica Audio Diamond (Vintage piano character)

Tier 1 Priority: PRIORITY 3 (after Bass & Drums, critical for classical/jazz)
Quality Target: Naturalness 85% → 95% (+12% for piano content)
Performance Target: <0.20× realtime

Author: Aurik Development Team - Phase 2.3 Tier 1
Version: 1.0.0
Date: 16. Februar 2026
"""

import logging
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.audio_utils import to_channels_last
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .output_guard import evaluate_output_guard
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls

    _FORMANT_SYSTEM_PIANO: Any = None
except Exception:
    _FormantSystemCls = None  # type: ignore[assignment,misc]
    _FORMANT_SYSTEM_PIANO = None

logger = logging.getLogger(__name__)


class PianoRestorationV1(PhaseInterface):
    """
    Professional Piano Restoration Engine (Tier 1, Priority 3).

    Features:
    - Hammer transient enhancement (attack clarity 2-8 kHz)
    - String resonance modeling (sympathetic vibrations)
    - Pedal noise reduction (mechanical noise 100-500 Hz)
    - Dynamic range restoration (velocity curve optimization)
    - Material-adaptive processing (Shellac/Vinyl/Tape/Digital)

    Use Cases:
    - Classical piano restoration (Steinway, Bösendorfer)
    - Jazz piano enhancement (Rhodes, Wurlitzer, acoustic)
    - Modern piano production (pop, rock, electronic)
    - Historical recordings (pre-1960s piano)

    Performance: <0.20× realtime on modern CPU
    """

    # Piano frequency ranges
    PIANO_RANGES = {
        "bass": (27.5, 82.4),  # A0-E2 (lowest notes)
        "tenor": (82.4, 261.6),  # E2-C4 (middle register)
        "treble": (261.6, 1046.5),  # C4-C6 (upper register)
        "brilliance": (1046.5, 4186),  # C6-C8 (highest notes)
        "hammer_impact": (2000, 8000),  # Hammer-string impact
        "string_body": (100, 2000),  # String fundamental + low harmonics
        "pedal_noise": (100, 500),  # Mechanical pedal sounds
    }

    # Material-adaptive restoration configs
    RESTORATION_CONFIG = {
        MaterialType.SHELLAC: {
            "hammer_enhancement": 0.75,  # Strong (restore lost transients)
            "string_resonance": 0.70,  # Strong (restore harmonics)
            "pedal_reduction": 0.80,  # Aggressive (reduce noise)
            "dynamics_expansion": 1.30,  # Moderate (restore some dynamics)
            "attack_gain_db": 4.0,
            "resonance_decay_factor": 1.5,
            "pedal_threshold_db": -35,
            "mix": 0.65,  # 65% processed
        },
        MaterialType.VINYL: {
            "hammer_enhancement": 0.60,
            "string_resonance": 0.60,
            "pedal_reduction": 0.60,
            "dynamics_expansion": 1.20,
            "attack_gain_db": 3.0,
            "resonance_decay_factor": 1.3,
            "pedal_threshold_db": -40,
            "mix": 0.55,  # 55% processed
        },
        MaterialType.TAPE: {
            "hammer_enhancement": 0.50,  # Gentle (tape has good transients)
            "string_resonance": 0.50,
            "pedal_reduction": 0.40,  # Preserve natural sound
            "dynamics_expansion": 1.15,
            "attack_gain_db": 2.5,
            "resonance_decay_factor": 1.2,
            "pedal_threshold_db": -45,
            "mix": 0.45,  # 45% processed
        },
        MaterialType.CASSETTE: {  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
            "hammer_enhancement": 0.50,
            "string_resonance": 0.50,
            "pedal_reduction": 0.40,
            "dynamics_expansion": 1.15,
            "attack_gain_db": 2.5,
            "resonance_decay_factor": 1.2,
            "pedal_threshold_db": -45,
            "mix": 0.45,
        },
        MaterialType.CD_DIGITAL: {
            "hammer_enhancement": 0.35,  # Subtle (digital is clean)
            "string_resonance": 0.40,
            "pedal_reduction": 0.30,
            "dynamics_expansion": 1.10,  # Slight expansion (over-compression)
            "attack_gain_db": 2.0,
            "resonance_decay_factor": 1.1,
            "pedal_threshold_db": -50,
            "mix": 0.35,  # 35% processed
        },
        MaterialType.STREAMING: {
            "hammer_enhancement": 0.30,  # Minimal (often already processed)
            "string_resonance": 0.35,
            "pedal_reduction": 0.25,
            "dynamics_expansion": 1.08,
            "attack_gain_db": 1.5,
            "resonance_decay_factor": 1.05,
            "pedal_threshold_db": -55,
            "mix": 0.30,  # 30% processed
        },
    }

    DEFAULT_CONFIG = {
        "hammer_enhancement": 0.50,
        "string_resonance": 0.50,
        "pedal_reduction": 0.50,
        "dynamics_expansion": 1.15,
        "attack_gain_db": 2.5,
        "resonance_decay_factor": 1.2,
        "pedal_threshold_db": -45,
        "mix": 0.50,
    }

    def __init__(self, sample_rate: int = 48000, **kwargs):
        """
        Initialisiert Piano Restoration Phase.

        Args:
            sample_rate: Audio sample rate (Hz)
            **kwargs: Override parameters
        """
        super().__init__(sample_rate, **kwargs)
        self._formant_system_piano: Any = None  # lazy Singleton-Initialisierung (W0201)

    def _fletcher_partial_correction(
        self,
        freq_hz: float,
        n: int,
        f0_hz: float,
        B: float = 0.0020,
    ) -> float:
        """Inharmonizitäts-korrigierte Partial-Frequenz (Fletcher 1964).

        Formel: fₙ = n · f₀ · √(1 + B · n²)
        B-Koeffizient nach Spec §2.11 INHARMONICITY_PRIORS:
            piano_bass:   0.0080
            piano_mid:    0.0020
            piano_treble: 0.0001

        Args:
            freq_hz: Gemessene Partial-Frequenz (Hz)
            n:       Partial-Index (1=Grundton, 2=2.Oberton, ...)
            f0_hz:   Grundton-Frequenz (Hz)
            B:       Inharmonizitäts-Koeffizient (registerabhängig)

        Returns:
            Korrigierte Soll-Frequenz in Hz.

        Reference:
            Fletcher (1964): "Normal Vibration Frequencies of a Stiff Piano String"
        """
        if f0_hz <= 0 or n <= 0:
            return freq_hz
        return n * f0_hz * np.sqrt(1.0 + B * float(n) ** 2)  # type: ignore[no-any-return]

    def _get_piano_B_coefficient(self, f0_hz: float) -> float:
        """Inharmonicity coefficient B per spec §4.5b (Piano-Inharmonizität-B table).

        Physical reality: short, stiff treble strings have HIGH B; long, wound
        bass strings have LOW B.  Mapped to MIDI octave ranges (Fletcher model):

            MIDI 21–35  (f0 <  65 Hz)  → B ≈ 0.00015  (long wound contrabass)
            MIDI 36–47  (f0 < 247 Hz)  → B ≈ 0.00065  (bass / tenor)
            MIDI 48–71  (f0 < 987 Hz)  → B ≈ 0.002    (mid / alto)
            MIDI 72–108 (f0 ≥ 987 Hz)  → B ≈ 0.005    (short stiff treble)

        Reference: §4.5b; Fletcher (1964) "Normal Vibration Frequencies of a
        Stiff Piano String"; Conklin (1999) string inharmonicity tables.
        """
        if f0_hz < 65.0:
            return 0.00015  # MIDI 21–35: long wound contrabass strings
        if f0_hz < 247.0:
            return 0.00065  # MIDI 36–47: bass / tenor register
        if f0_hz < 987.0:
            return 0.002  # MIDI 48–71: mid / alto register
        return 0.005  # MIDI 72–108: short stiff treble strings

    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: MaterialType = MaterialType.CD_DIGITAL,
        **kwargs,
    ) -> PhaseResult:
        check_ml_model_ready("MERT", phase_name="52")
        check_ml_model_ready("PANNs", phase_name="52")
        """
        Restauriert piano recordings with material-adaptive processing.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz (must be 48000)
            material_type: Source material type
            **kwargs: Additional parameters

        Returns:
            PhaseResult with restored piano audio
        """
        start_time = time.time()
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p52_transposed = to_channels_last(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # PANNs piano-confidence gate (Spec §2.9: threshold >= 0.50).
        panns_tags = kwargs.get("panns_tags", {})
        piano_confidence = 0.0
        for tag_name, conf in panns_tags.items():
            tag_lower = str(tag_name).lower()
            if any(k in tag_lower for k in ("piano", "keyboard", "keys")):
                piano_confidence = max(piano_confidence, float(conf))

        if panns_tags and piano_confidence < 0.50:
            logger.info(
                "Phase 52: Piano-Confidence %.2f < 0.50 — Phase skipped (Spec §2.9)",
                piano_confidence,
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skip_panns_confidence",
                    "piano_confidence": piano_confidence,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        if piano_confidence > 0.0:
            logger.info("Phase 52: PANNs Piano-Confidence=%.2f >= 0.50 — processing active", piano_confidence)

        # Get material-specific config
        config = self.RESTORATION_CONFIG.get(material_type, self.DEFAULT_CONFIG).copy()

        quality_mode = str(kwargs.get("quality_mode", "balanced")).lower()
        if quality_mode in ("quality", "maximum", "studio2026"):
            hq_scale = 1.12 if quality_mode in ("maximum", "studio2026") else 1.06
            config["mix"] = float(np.clip(config["mix"] * hq_scale, 0.0, 0.75))
            config["hammer_enhancement"] = float(np.clip(config["hammer_enhancement"] * hq_scale, 0.0, 1.0))
            config["string_resonance"] = float(np.clip(config["string_resonance"] * hq_scale, 0.0, 1.0))
            config["dynamics_expansion"] = float(
                np.clip(config["dynamics_expansion"] * (1.0 + 0.5 * (hq_scale - 1.0)), 1.0, 1.35)
            )
        else:
            hq_scale = 1.0

        # Preserve stereo image: process Mid, keep original Side, then reconstruct L/R.
        is_stereo = audio.ndim == 2
        if is_stereo:
            left = audio[:, 0].astype(np.float64)
            right = audio[:, 1].astype(np.float64)
            original_mid = 0.5 * (left + right)
            original_side = 0.5 * (left - right)
            audio_mono = original_mid.copy()
        else:
            original_mid = audio.astype(np.float64)
            original_side = None
            audio_mono = original_mid.copy()

        logger.info("Piano Restoration: %s, config=%s", material_type.value, config)

        # Stage 1: Hammer Transient Enhancement
        if config["hammer_enhancement"] > 0:
            audio_mono = self._enhance_hammer_transients(
                audio_mono, intensity=config["hammer_enhancement"], gain_db=config["attack_gain_db"]
            )

        # Stage 2: String Resonance Enhancement
        if config["string_resonance"] > 0:
            audio_mono = self._enhance_string_resonance(
                audio_mono, intensity=config["string_resonance"], decay_factor=config["resonance_decay_factor"]
            )

        # Stage 3: Pedal Noise Reduction
        if config["pedal_reduction"] > 0:
            audio_mono = self._reduce_pedal_noise(
                audio_mono, intensity=config["pedal_reduction"], threshold_db=config["pedal_threshold_db"]
            )

        # Stage 4: Dynamic Range Restoration
        if config["dynamics_expansion"] > 1.0:
            audio_mono = self._restore_dynamics(audio_mono, expansion_ratio=config["dynamics_expansion"])

        # Dry/wet mix on Mid channel
        original_mono = original_mid

        mix = float(np.clip(config["mix"], 0.0, 1.0)) * effective_strength
        audio_mono = mix * audio_mono + (1.0 - mix) * original_mono

        # Prevent clipping (soft limiter at 0.95) — §2.49 Peak-Guard: percentile(99.9)
        peak = float(np.percentile(np.abs(audio_mono), 99.9))
        if peak > 0.95:
            audio_mono = audio_mono * (0.95 / peak)

        # Reconstruct stereo from processed Mid + original Side.
        if is_stereo and original_side is not None:
            out_left = audio_mono + original_side
            out_right = audio_mono - original_side
            audio_out = np.column_stack([out_left, out_right])
        else:
            audio_out = audio_mono

        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)
        audio_out_pre_refine = audio_out.copy()

        # Instrument-guided formant enhancement (Young 1952 / Weinreich 1977 piano resonances)
        igt_frames = 0
        try:
            if _FormantSystemCls is not None:
                if self._formant_system_piano is None:
                    self._formant_system_piano = _FormantSystemCls(enhance_singers_formant=False)
                formant_strength = float(np.clip(0.20 * hq_scale, 0.15, 0.28))
                audio_out, igt_report = self._formant_system_piano.instrument_guided_enhance(
                    audio_out,
                    self.sample_rate,
                    instrument="keys",
                    correction_strength=formant_strength,
                )
                igt_frames = igt_report.get("frames_processed", 0)
                logger.debug("Phase 52 InstrumentFormant: piano frames=%d", igt_frames)
        except Exception as _igt_exc:
            logger.debug("Phase 52 instrument_guided_enhance skipped: %s", _igt_exc)

        # Formant-Drift-Korrektur via DTW (Schritt 3)
        try:
            from dsp.instrument_formant_corrector import (
                correct_instrument_formant_drift,  # pylint: disable=import-outside-toplevel
            )

            drift_result = correct_instrument_formant_drift(audio_out, self.sample_rate, instrument="keys")
            audio_out = drift_result.audio
            logger.debug(
                "Phase 52 drift correction: detected=%s frames=%d/%d drift=%.1fHz",
                drift_result.drift_detected,
                drift_result.n_frames_corrected,
                drift_result.total_frames,
                drift_result.mean_drift_hz,
            )
        except Exception as _drift_exc:
            logger.debug("Phase 52 drift correction skipped: %s", _drift_exc)

        # Sub-Stem-Verarbeitung (Schritt 4)
        try:
            from backend.core.sub_stem_processor import process_sub_stems  # pylint: disable=import-outside-toplevel

            sub_stem_strength = float(np.clip(0.30 * hq_scale, 0.25, 0.42))
            ss_result = process_sub_stems(
                audio_out, self.sample_rate, instrument="keys", processing_strength=sub_stem_strength
            )
            audio_out = ss_result.audio
            logger.debug("Phase 52 sub-stem: bands=%d strength=%.2f", ss_result.n_bands, ss_result.processing_strength)
        except Exception as _ss_exc:
            logger.debug("Phase 52 sub-stem skipped: %s", _ss_exc)

        # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
        try:
            from backend.core.physics_resonance_enhancer import (
                enhance_physics_resonance,  # pylint: disable=import-outside-toplevel
            )

            physics_strength = float(np.clip(0.40 * hq_scale, 0.35, 0.55))
            pr_result = enhance_physics_resonance(
                audio_out, self.sample_rate, instrument="keys", enhancement_strength=physics_strength
            )
            audio_out = pr_result.audio
            logger.debug(
                "Phase 52 physics resonance: peaks=%d strength=%.2f", pr_result.n_peaks, pr_result.enhancement_strength
            )
        except Exception as _pr_exc:
            logger.debug("Phase 52 physics resonance skipped: %s", _pr_exc)

        if 0.0 < effective_strength < 1.0:
            audio_out = audio + effective_strength * (audio_out - audio)

        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        # Conservative output guard for high quality modes only.
        output_guard_enabled = quality_mode in ("quality", "maximum", "studio2026")
        guard = evaluate_output_guard(
            original=audio,
            candidate=audio_out,
            enabled=output_guard_enabled,
            max_abs_rms_delta_db=1.0,
            stereo_side_ratio_min=0.60,
            stereo_side_ratio_max=1.40,
        )

        if guard.fallback:
            audio_out = audio_out_pre_refine.copy()
            if 0.0 < effective_strength < 1.0:
                audio_out = audio + effective_strength * (audio_out - audio)
            audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)

        # §2.46e HallucinationGuard: Additive Phase darf kein halluziniertes Material einführen (VERBOTEN)
        try:
            from backend.core.dsp.hallucination_guard import (  # pylint: disable=import-outside-toplevel
                check_hallucination as _chk_hg52,
            )

            _hg_mode_52 = str(kwargs.get("mode", "restoration"))
            _pre52_mono = (
                audio.mean(axis=0)
                if (audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2)
                else (audio.mean(axis=1) if audio.ndim == 2 else audio)
            )
            _post52_mono = (
                audio_out.mean(axis=0)
                if (audio_out.ndim == 2 and audio_out.shape[0] == 2 and audio_out.shape[1] > 2)
                else (audio_out.mean(axis=1) if audio_out.ndim == 2 else audio_out)
            )
            _hg52 = _chk_hg52(
                _pre52_mono.astype(np.float32),
                _post52_mono.astype(np.float32),
                sr=sample_rate,
                mode=_hg_mode_52,
                bw_extension_context=True,
            )
            if _hg52.requires_rollback:
                audio_out = audio.copy()
                logger.warning("§2.46e phase_52 HallucinationGuard: rollback (spectral_novelty > 0.15)")
        except Exception as _hg52_exc:
            logger.debug("§2.46e phase_52 HallucinationGuard (non-blocking): %s", _hg52_exc)

        return PhaseResult(
            success=True,
            audio=audio_out,
            execution_time_seconds=time.time() - start_time,
            metadata={
                "material_type": material_type.value,
                "hammer_enhancement": config["hammer_enhancement"],
                "string_resonance": config["string_resonance"],
                "pedal_reduction": config["pedal_reduction"],
                "dynamics_expansion": config["dynamics_expansion"],
                "mix": mix,
                "stereo_image_preserved": bool(is_stereo),
                "quality_mode": quality_mode,
                "hq_scale": hq_scale,
                "output_guard_enabled": output_guard_enabled,
                "output_guard_fallback": guard.fallback,
                "output_guard_reason": guard.reason,
                "rms_delta_db": guard.rms_delta_db,
                "stereo_side_ratio": guard.stereo_side_ratio,
                "fletcher_B_available": True,
                "instrument_formant_frames": igt_frames,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )

    def _enhance_hammer_transients(self, audio: np.ndarray, intensity: float, gain_db: float) -> np.ndarray:
        """
        Enhance hammer-string impact transients (attack clarity).

        Algorithm:
        1. Bandpass filter 2-8 kHz (hammer impact region)
        2. Envelope follower (fast attack, slow release)
        3. Transient detection (onset peaks)
        4. Gain boost at transient locations (5-20ms window)
        """
        # Bandpass filter: hammer impact region (2-8 kHz)
        nyquist = self.sample_rate / 2
        low_freq = self.PIANO_RANGES["hammer_impact"][0] / nyquist
        high_freq = min(self.PIANO_RANGES["hammer_impact"][1] / nyquist, 0.99)

        sos = signal.butter(4, [low_freq, high_freq], btype="band", output="sos")
        hammer_band = signal.sosfilt(sos, audio)

        # Envelope detection (fast attack, slow release)
        envelope = np.abs(hammer_band)
        envelope = signal.sosfilt(signal.butter(2, 50 / nyquist, output="sos"), envelope)  # 50 Hz lowpass

        # Transient detection (peaks in envelope)
        peak_indices, _ = signal.find_peaks(
            envelope,
            distance=int(0.05 * self.sample_rate),  # Min 50ms between transients
            height=np.max(envelope) * 0.1,  # 10% of max
        )

        # Create transient mask (20ms windows around peaks)
        transient_mask = np.zeros_like(audio)
        window_samples = int(0.020 * self.sample_rate)  # 20ms

        for peak_idx in peak_indices:
            start_idx = max(0, peak_idx - window_samples // 4)
            end_idx = min(len(audio), peak_idx + window_samples)

            # Gaussian window for smooth transition
            window_len = end_idx - start_idx
            window = signal.windows.gaussian(window_len, std=window_len / 6)
            transient_mask[start_idx:end_idx] += window

        # Normalize mask
        transient_mask = np.clip(transient_mask, 0, 1)

        # Apply gain boost at transients
        gain_linear = 10 ** (gain_db / 20)
        enhancement = audio.copy()
        enhancement = enhancement * (1.0 + (gain_linear - 1.0) * transient_mask * intensity)

        return enhancement  # type: ignore[no-any-return]

    def _enhance_string_resonance(self, audio: np.ndarray, intensity: float, decay_factor: float) -> np.ndarray:
        """
        Enhance string resonance and sympathetic vibrations.

        Algorithm:
        1. Harmonic analysis (FFT-based)
        2. Identify fundamental + overtones
        3. Extend decay envelope (sustain enhancement)
        4. Add subtle harmonic content
        """
        # Short-time FFT for harmonic analysis
        nperseg = 2048
        noverlap = nperseg // 2

        # STFT
        f, _t, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap, boundary="even")

        # Enhance harmonic content (boost overtones)
        # Focus on piano string body (100-2000 Hz)
        string_mask = (f >= self.PIANO_RANGES["string_body"][0]) & (f <= self.PIANO_RANGES["string_body"][1])

        # Boost harmonics with decay_factor
        Zxx[string_mask, :] *= 1.0 + (decay_factor - 1.0) * intensity

        # §4.5b Piano inharmonicity integration:
        # f_n = n * f0 * sqrt(1 + B * n^2)
        # Apply a subtle frequency-selective lift around inharmonic partial targets
        # to preserve realistic piano string stiffness behavior.
        try:
            mag_mean = np.mean(np.abs(Zxx), axis=1)
            f0_search = (f >= 55.0) & (f <= 1500.0)
            if np.any(f0_search):
                search_idx = np.where(f0_search)[0]
                peak_rel = int(np.argmax(mag_mean[f0_search]))
                f0_idx = int(search_idx[peak_rel])
                f0_hz = float(f[f0_idx])
                if f0_hz > 0.0:
                    b_coeff = float(self._get_piano_B_coefficient(f0_hz))
                    partial_mask = np.zeros_like(f, dtype=np.float64)
                    nyq = self.sample_rate * 0.5
                    max_n = int(min(16, nyq / max(f0_hz, 1e-9)))
                    for n in range(2, max_n + 1):
                        target_hz = float(self._fletcher_partial_correction(n * f0_hz, n, f0_hz, b_coeff))
                        if target_hz <= 0.0 or target_hz > nyq:
                            continue
                        sigma_hz = 18.0 + 2.5 * n
                        partial_mask += np.exp(-0.5 * ((f - target_hz) / sigma_hz) ** 2)
                    if np.any(partial_mask > 0.0):
                        partial_mask /= float(np.max(partial_mask))
                        lift = 1.0 + 0.12 * float(intensity) * partial_mask[:, None]
                        Zxx *= lift
        except Exception as _inh_exc:
            logger.debug("Phase 52 inharmonic partial shaping skipped: %s", _inh_exc)

        # Inverse STFT
        _, audio_enhanced = signal.istft(Zxx, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap, boundary=True)

        # Trim to original length
        audio_enhanced = audio_enhanced[: len(audio)]

        return audio_enhanced  # type: ignore[no-any-return]

    def _reduce_pedal_noise(self, audio: np.ndarray, intensity: float, threshold_db: float) -> np.ndarray:
        """
        Reduce mechanical pedal noise (damper pedal thump).

        Algorithm:
        1. Bandpass filter 100-500 Hz (pedal noise region)
        2. Detect pedal events (short bursts of energy)
        3. Apply spectral gating at pedal times
        4. Preserve musical bass content
        """
        # Bandpass filter: pedal noise region (100-500 Hz)
        nyquist = self.sample_rate / 2
        low_freq = self.PIANO_RANGES["pedal_noise"][0] / nyquist
        high_freq = self.PIANO_RANGES["pedal_noise"][1] / nyquist

        sos = signal.butter(4, [low_freq, high_freq], btype="band", output="sos")
        # sosfiltfilt (zero-phase) required: pedal_band is subtracted from audio in recombination;
        # causal sosfilt would introduce group delay → timing skew → Pegelexplosion (§2.51, V11)
        pedal_band = signal.sosfiltfilt(sos, audio)

        # Envelope detection
        envelope = np.abs(pedal_band)
        envelope = signal.sosfilt(
            signal.butter(2, 20 / nyquist, output="sos"), envelope
        )  # 20 Hz lowpass — analysis only

        # Convert threshold to linear
        threshold_linear = 10 ** (threshold_db / 20) * np.max(envelope)

        # Create reduction mask (reduce when envelope > threshold)
        reduction_mask = np.where(
            envelope > threshold_linear,
            intensity,
            0.0,  # Reduce by intensity amount  # No reduction
        )

        # Apply reduction to pedal band only
        pedal_band_reduced = pedal_band * (1.0 - reduction_mask)

        # Reconstruct: original - pedal_band + pedal_band_reduced
        audio_cleaned = audio - pedal_band + pedal_band_reduced

        return audio_cleaned  # type: ignore[no-any-return]

    def _restore_dynamics(self, audio: np.ndarray, expansion_ratio: float) -> np.ndarray:
        """
        Restauriert dynamic range (upward expansion for over-compressed recordings).

        Algorithm:
        1. RMS envelope detection
        2. Upward expansion (boost quiet passages)
        3. Preserve loud passages
        4. Smooth transitions
        """
        # RMS envelope (50ms window)
        window_size = int(0.050 * self.sample_rate)
        audio_squared = audio**2

        # Convolve with rectangular window for RMS
        window = np.ones(window_size) / window_size
        rms_envelope = np.sqrt(np.convolve(audio_squared, window, mode="same"))

        # Upward expansion: boost quiet passages
        # Threshold at -20 dB (relative to max)
        # Guard: bei Stille (max==0) wäre threshold=0 => division by zero
        threshold = max(0.1 * float(np.max(rms_envelope)), 1e-12)  # -20 dB

        # §2.45a Silence gate: frames below -46 dBFS are true silence (noise floor /
        # fadeout) and must NOT be boosted — expansion is only for musical content.
        # Without this gate, silence frames receive gain = expansion_ratio (1.08–1.35),
        # which raises the noise floor by 3–11 dB (Pegelexplosion in quiet sections).
        _SILENCE_GATE_LINEAR = 10.0 ** (-46.0 / 20.0)  # ≈ 0.005 linear

        # Calculate gain (boost below threshold, but never below silence gate)
        gain = np.where(
            rms_envelope < _SILENCE_GATE_LINEAR,
            1.0,  # True silence — no expansion, preserve noise floor
            np.where(
                rms_envelope < threshold,
                expansion_ratio ** np.clip((threshold - rms_envelope) / threshold, -40.0, 40.0),
                1.0,  # No change above threshold
            ),
        )

        # Smooth gain (avoid artifacts)
        nyquist = self.sample_rate / 2
        gain = signal.sosfilt(signal.butter(2, 10 / nyquist, output="sos"), gain)  # 10 Hz lowpass

        # Apply gain
        audio_expanded = audio * gain

        # Normalize to prevent clipping — §2.49 Peak-Guard: percentile(99.9)
        max_val = float(np.percentile(np.abs(audio), 99.9))
        if float(np.percentile(np.abs(audio_expanded), 99.9)) > max_val:
            audio_expanded = audio_expanded / (float(np.percentile(np.abs(audio_expanded), 99.9)) + 1e-10) * max_val

        return audio_expanded  # type: ignore[no-any-return]

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_52_piano_restoration",
            name="Piano Restoration System v1.0",
            category=PhaseCategory.ENHANCEMENT,
            priority=8,  # High priority (Tier 1, PRIORITY 3 after Bass/Drums)
            dependencies=[],
            estimated_time_factor=0.20,  # 20% of audio duration
            version="1.0.0",
            memory_requirement_mb=120,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,  # High impact on piano content
            description=(
                "Professional piano restoration: hammer transients, string resonance, "
                "pedal noise reduction, dynamic range restoration"
            ),
        )

    def supports_material(self, material_type: MaterialType) -> bool:
        """Prüft if material type is supported."""
        return material_type in self.RESTORATION_CONFIG or material_type in [
            MaterialType.REEL_TAPE,
            MaterialType.DAT,
            MaterialType.AAC,
            MaterialType.MP3_LOW,
            MaterialType.MP3_HIGH,
        ]

    def estimate_time(self, audio_duration_seconds: float) -> float:
        """Schätzt processing time."""
        return audio_duration_seconds * 0.20  # 0.20× realtime


# Test harness
if __name__ == "__main__":
    logger.debug("=" * 70)
    logger.debug("AURIK 9.0 - PIANO RESTORATION SYSTEM v1.0 TEST")
    logger.debug("=" * 70)
    logger.debug("Tier 1 Priority 3: Classical, Jazz, Modern Piano")
    logger.debug("=" * 70)

    # Create test instance
    phase = PianoRestorationV1(sample_rate=48000)
    metadata = phase.get_metadata()

    logger.debug("\n📋 Phase Metadata:")
    logger.debug("   ID: %s", metadata.phase_id)
    logger.debug("   Name: %s", metadata.name)
    logger.debug("   Category: %s", metadata.category.value)
    logger.debug("   Priority: %s", metadata.priority)
    logger.debug("   Estimated time: %s× RT", metadata.estimated_time_factor)
    logger.debug("   Quality impact: %.0f%%", metadata.quality_impact * 100)

    # Generate test audio (simulated piano with noise)
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Piano fundamental (C4 = 261.6 Hz) with harmonics
    piano_signal = 0.5 * np.sin(2 * np.pi * 261.6 * t)  # Fundamental
    piano_signal += 0.3 * np.sin(2 * np.pi * 523.2 * t)  # 2nd harmonic
    piano_signal += 0.2 * np.sin(2 * np.pi * 784.8 * t)  # 3rd harmonic
    piano_signal += 0.1 * np.sin(2 * np.pi * 1046.4 * t)  # 4th harmonic

    # Add decay envelope
    decay = np.exp(-2 * t)
    piano_signal *= decay

    # Add pedal noise (low-frequency thump)
    pedal_times = [0.5, 1.0, 1.5]
    for pt in pedal_times:
        pedal_idx = int(pt * sr)
        pedal_length = int(0.05 * sr)
        if pedal_idx < len(piano_signal):
            pedal_envelope = np.exp(-50 * (t - pt))[pedal_idx : pedal_idx + pedal_length]
            piano_signal[pedal_idx : pedal_idx + len(pedal_envelope)] += (
                0.2 * pedal_envelope * np.sin(2 * np.pi * 150 * (t - pt)[pedal_idx : pedal_idx + len(pedal_envelope)])
            )

    # Normalize
    piano_signal = piano_signal / np.percentile(np.abs(piano_signal), 99.9) * 0.7

    # Test on different materials
    materials = [MaterialType.SHELLAC, MaterialType.VINYL, MaterialType.CD_DIGITAL]

    logger.debug("\n🎹 Testing Piano Restoration on %s material types:", len(materials))
    logger.debug("   Audio: %ss, %s Hz", duration, sr)

    for material in materials:
        result = phase.process(piano_signal, material_type=material)

        logger.debug("\n   %s:", material.value)
        logger.debug("      Time: %.3fs", result.execution_time_seconds)
        logger.debug("      RT Factor: %.3f×", result.execution_time_seconds / duration)
        logger.debug("      Shape: %s", result.audio.shape)
        logger.debug("      Max: %.3f", np.max(np.abs(result.audio)))
        logger.debug(
            "      Config: hammer=%.2f, resonance=%.2f, pedal_reduction=%.2f",
            result.metadata["hammer_enhancement"],
            result.metadata["string_resonance"],
            result.metadata["pedal_reduction"],
        )

    logger.debug("\n%s", "=" * 70)
    logger.debug("✅ PIANO RESTORATION TEST COMPLETE")
    logger.debug("=" * 70)
    logger.debug("\n🎯 Next Steps:")
    logger.debug("   1. Add to __init__.py exports")
    logger.debug("   2. Integrate into UnifiedRestorerV3 _select_phases()")
    logger.debug("   3. Create integration tests")
    logger.debug("   4. Test with real piano recordings")
