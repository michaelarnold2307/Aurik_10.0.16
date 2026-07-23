#!/usr/bin/env python3
"""
Phase 39: Air Band Enhancement v2.0 - Professional
High-frequency shimmer and "air" enhancement (12-20 kHz).

Algorithm Overview:
1. Frequency Focus: 12-20 kHz (air band)
2. Harmonic Excitation: Generate missing HF content
3. Shelving EQ: Smooth HF lift
4. Saturation: Subtle harmonic distortion for warmth
5. Material Adaptation:
   - Shellac: Strong (restore bandwidth-limited treble)
   - Vinyl: Moderate (add air and sparkle)
   - Tape: Light (tape often has natural HF roll-off)
   - Digital: Moderate (add analog-style air)

Scientific Foundation:
- Fastl & Zwicker (2007): Psychoacoustics - Facts and Models
- Gabrielsson & Sjögren (1979): Perceived Sound Quality of Sound-Reproducing Systems
- Toole (1986): Loudspeaker Measurements and Their Relationship to Listener Preferences
- Moore (2012): An Introduction to the Psychology of Hearing

Industry Benchmarks:
- Maag EQ4 (Famous "Air Band" @ 40 kHz downsampled)
- Aphex Aural Exciter (HF harmonic generator)
- BBE Sonic Maximizer (Phase compensation + HF boost)
- Pultec HLF-3C (High-frequency boost)
- Dangerous Music BAX EQ (Shelf filter mastering)

Quality Target: 0.82 → 0.93 (+13% improvement)
Performance Target: <0.08× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import os
import logging
import threading
import time

import numpy as np
from scipy import signal

from backend.core.audio_utils import safe_filtfilt,  safe_to_mono, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .output_guard import evaluate_output_guard
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

try:
    from backend.core.dsp.hallucination_guard import check_hallucination as _check_hallucination39
except Exception:
    _check_hallucination39 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# §2.46b Spectral-Tilt-Preservation: material-adaptive tolerance in dB/octave
_TILT_TOLERANCE_P39: dict[str, float] = {
    "digital": 1.5,
    "cd_digital": 1.5,
    "streaming": 1.5,
    "tape": 1.875,
    "reel_tape": 1.875,
    "vinyl": 2.25,
    "minidisc": 2.25,
    "shellac": 3.0,
    "wax_cylinder": 3.0,
    "wire_recording": 3.0,
}


def _est_tilt_p39(audio: np.ndarray, sr: int) -> float:
    """Quick spectral tilt estimate in dB/octave (§2.46b)."""
    mono = safe_to_mono(audio) if audio.ndim == 2 else audio
    n = min(len(mono), 8192)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(mono[:n] * np.hanning(n))) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    valid = (freqs >= 100.0) & (freqs <= sr * 0.45)
    if np.sum(valid) < 8:
        return 0.0
    log_f = np.log2(freqs[valid] + 1e-12)
    log_m = 20.0 * np.log10(spec[valid])
    log_f_c = log_f - log_f.mean()
    log_m_c = log_m - log_m.mean()
    denom = float(np.dot(log_f_c, log_f_c))
    return float(np.dot(log_f_c, log_m_c) / denom) if denom > 1e-10 else 0.0


class AirBandEnhancement(PhaseInterface):
    """
    Professional Air Band Enhancement Engine.

    Key Features:
    - High-frequency shelving (12-20 kHz)
    - Harmonic excitation (add sparkle)
    - Material-adaptive intensity
    - Psychoacoustic optimization
    - Analog-style warmth

    Use Cases:
    - Restore missing treble from bandwidth-limited sources
    - Add "air" and "shimmer" to vocals and instruments
    - Enhance perceived detail and clarity
    - Modernize vintage recordings

    Performance: <0.08× realtime on modern CPU
    """

    # Air band frequency range
    AIR_BAND_HZ = (12000, 20000)

    # Enhancement parameters (material-adaptive)
    AIR_CONFIG = {
        MaterialType.SHELLAC: {
            "shelf_gain_db": 6.0,  # Strong (restore missing HF)
            "shelf_freq_hz": 10000,
            "exciter_mix": 0.40,
            "saturation_drive": 0.30,
        },
        MaterialType.VINYL: {
            "shelf_gain_db": 4.0,
            "shelf_freq_hz": 12000,
            "exciter_mix": 0.30,
            "saturation_drive": 0.20,
        },
        MaterialType.TAPE: {
            "shelf_gain_db": 5.0,  # v10.0.0: ↑3.0→5.0 — Tape-HF nach Rauschreduktion sicher erweiterbar
            "shelf_freq_hz": 13000,
            "exciter_mix": 0.28,  # v10.0.0: ↑0.20→0.28
            "saturation_drive": 0.20,  # v10.0.0: ↑0.15→0.20
        },
        MaterialType.CASSETTE: {  # §6.2c BW-Ceiling 14 kHz (central definition) — konservativ
            "shelf_gain_db": 3.0,  # ≤ 0.35 Stärke — kein HF über 14 kHz synthetisieren
            "shelf_freq_hz": 10000,  # Unter 14-kHz-Ceiling bleiben
            "exciter_mix": 0.18,
            "saturation_drive": 0.15,
        },
        MaterialType.CD_DIGITAL: {
            "shelf_gain_db": 5.0,  # v10.0.0: ↑3.5→5.0 — CD hat klare HF-Basis
            "shelf_freq_hz": 12000,
            "exciter_mix": 0.30,  # v10.0.0: ↑0.25→0.30
            "saturation_drive": 0.25,
        },
        MaterialType.STREAMING: {
            "shelf_gain_db": 4.5,  # v10.0.0: ↑4.0→4.5
            "shelf_freq_hz": 11000,
            "exciter_mix": 0.32,  # v10.0.0: ↑0.30→0.32
            "saturation_drive": 0.22,  # v10.0.0: ↑0.20→0.22
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "Air Band Enhancement v2 Professional"
        self._sos_air_cache: dict[int, np.ndarray] = {}
        self._shelf_coeffs: dict[tuple, tuple] = {}
        self._cache_lock = threading.Lock()

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_39_air_band_enhancement",
            name="Air Band Enhancement v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            dependencies=[],
            estimated_time_factor=0.08,
            version="2.0.0",
            memory_requirement_mb=35,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.93,
            description="High-frequency shimmer and air enhancement (12-20 kHz)",
        )

    # pylint: disable-next=arguments-renamed
    def process(  # type: ignore[override]
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="39")
        """
        Wendet Air-Band-Verbesserung auf Audio an.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with enhanced audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 [RELEASE_MUST] ForwardMaskingGuard: Additive Air-Band-Phase bei Vokal
        # stärker (NR-Stärke in post-transienten Fenstern erhöhen, psychoakustisch sicher).
        _panns_s_39 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_39 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_39,  # pylint: disable=import-outside-toplevel
                )

                _fmz_39 = kwargs.get("forward_masking_zones") or _fmg_fn_39().compute_zones(audio, sample_rate)
                if _fmz_39:
                    _n_s_39 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_samples_39 = sum(z.end_sample - z.start_sample for z in _fmz_39)
                    _zone_frac_39 = float(np.clip(_zone_samples_39 / max(1, _n_s_39), 0.0, 1.0))
                    _boost_39 = _zone_frac_39 * 0.15
                    _effective_strength = float(np.clip(_effective_strength + _boost_39, 0.0, 1.0))
                    logger.debug(
                        "Phase39 §V41 ForwardMasking: zone_frac=%.2f boost=%.3f → eff_str=%.3f",
                        _zone_frac_39,
                        _boost_39,
                        _effective_strength,
                    )
            except Exception as _fmg_exc_39:  # pylint: disable=broad-except
                logger.debug("Phase39 §V41 ForwardMaskingGuard non-blocking: %s", _fmg_exc_39)

        if _effective_strength <= 0.0:
            logger.info(
                "Phase 39: skipped — effective_strength=%.3f (no air band enhancement applied)", _effective_strength
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            # §5/5: Echte Peak-Messung auch bei Skip
            _p_peak = float(20.0 * np.log10(np.percentile(np.abs(audio), 99.9) + 1e-10))
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[],
            )

        # §0a §2.46e BUG-FIX v10.0.0 (Bug 6): Air-Band-Enhancement in Restoration-Mode
        # ist VERBOTEN fuer analoge Materialien (vinyl, shellac, tape) —
        # Air-Band-Enhancement ist ein Harmonic Exciter (§0a) und eine additive
        # Halluzination (§2.46e): es fuegt Energie ueber das physikalische BW-Ceiling hinzu,
        # die im Original nicht vorhanden war.
        _proc_mode_39 = str(kwargs.get("mode", kwargs.get("processing_mode", "restoration"))).lower()
        _ANALOG_RESTORATION_SKIP = {
            "vinyl",
            "shellac",
            "wax_cylinder",
            "wire_recording",
            "tape",
            "reel_tape",
            "cassette",
            "lacquer_disc",
        }
        # §CD-Quality: bandwidth_loss > 0.5 → Air-Band rekonstruieren (Original hatte bereits HF-Verlust)
        _bw_loss_39 = float(kwargs.get("bandwidth_loss", 0.0) or 0.0)
        if _bw_loss_39 <= 0.0:
            _defect_scores_39 = kwargs.get("defect_scores", {}) or {}
            _bw_loss_39 = float(_defect_scores_39.get("bandwidth_loss", 0.0) or 0.0)
        _mat_name_39 = str(getattr(material, "name", str(material))).lower().replace(" ", "_").replace("-", "_")
        if _proc_mode_39 == "restoration" and _mat_name_39 in _ANALOG_RESTORATION_SKIP and _bw_loss_39 <= 0.5:
            _skip_audio = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            _skip_audio = np.clip(_skip_audio, -1.0, 1.0)
            logger.info(
                "Phase 39 §0a skip: Restoration-Mode + analog material '%s' — "
                "Air-Band-Enhancement (Harmonic Exciter) verboten",
                _mat_name_39,
            )
            return PhaseResult(
                success=True,
                audio=_skip_audio,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_restoration_analog_material",
                    "material": _mat_name_39,
                    "mode": _proc_mode_39,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[],
            )

        is_stereo = audio.ndim == 2
        config = dict(self.AIR_CONFIG.get(material, self.AIR_CONFIG[MaterialType.CD_DIGITAL]))

        quality_mode = str(kwargs.get("quality_mode", "balanced")).lower()
        if quality_mode in ("quality", "maximum", "studio2026"):
            hq_scale = 1.12 if quality_mode in ("maximum", "studio2026") else 1.06
        else:
            hq_scale = 1.0

        config["shelf_gain_db"] = float(config["shelf_gain_db"] * _effective_strength)
        config["exciter_mix"] = float(config["exciter_mix"] * _effective_strength)
        config["saturation_drive"] = float(config["saturation_drive"] * _effective_strength)
        config["shelf_gain_db"] = float(np.clip(config["shelf_gain_db"] * hq_scale, 0.0, 7.0))
        config["exciter_mix"] = float(np.clip(config["exciter_mix"] * hq_scale, 0.0, 0.55))
        config["saturation_drive"] = float(np.clip(config["saturation_drive"] * hq_scale, 0.0, 0.45))

        # ── Era/Genre-adaptive air band scaling (context injection §2.x) ──
        _brillanz = kwargs.get("brillanz_target")
        _decade = kwargs.get("decade")
        _genre = kwargs.get("genre_label", "")
        if _brillanz is not None:
            _b_scale = 0.30 + 0.90 * float(_brillanz)
            _b_scale = max(0.65, min(1.20, _b_scale))
            config["shelf_gain_db"] *= _b_scale
            config["exciter_mix"] *= _b_scale
            logger.debug("Phase 39: brillanz_target=%.2f → air scale=%.2f", float(_brillanz), _b_scale)
        if _decade is not None:
            _dec = int(_decade)
            if _dec <= 1950:
                # Vintage: cap air enhancement (Spec §5: Rolloff ≤ 7 kHz not expand)
                config["shelf_gain_db"] = min(config["shelf_gain_db"], 2.5)
                config["exciter_mix"] = min(config["exciter_mix"], 0.15)
                config["shelf_freq_hz"] = max(config["shelf_freq_hz"], 14000)
            elif _dec <= 1970:
                # Early stereo era: moderate air
                config["shelf_gain_db"] *= 0.80
                config["exciter_mix"] *= 0.80
            elif _dec >= 2000:
                # Modern: less exciter (already has HF content)
                config["exciter_mix"] *= 0.75
        _genre_lower_39 = str(_genre).lower()
        if _genre_lower_39 in ("klassik", "oper"):
            # Classical: minimal air exciter to preserve natural overtones
            config["exciter_mix"] *= 0.50
            config["saturation_drive"] *= 0.50
        elif _genre_lower_39 == "reggae":
            # Reggae: warm/dark production style \u2014 HF air excitation would strip character.
            config["exciter_mix"] *= 0.40
            config["saturation_drive"] *= 0.50
        elif _genre_lower_39 in ("blues", "folk"):
            # Blues/Folk: natural warmth and harmonic texture, minimal air processing.
            config["exciter_mix"] *= 0.65
            config["saturation_drive"] *= 0.70
        elif _genre_lower_39 in ("electronic", "hip-hop"):
            # Electronic/Hip-Hop: typically already has engineered HF presence.
            config["exciter_mix"] *= 0.70
            config["saturation_drive"] *= 0.75

        # §2.41 (v10.0.0) SOTA: Ära-bewusste Air-Band-Deckelung aus SourceFidelityTarget.
        # Physikalische Invariante (Klangtreue §2.41): Luft-Anhebung nie ÜBER der ären-
        # typischen Original-Bandbreite — was das Original nicht enthalten konnte,
        # darf nicht synthetisiert werden (falsche Helligkeit).
        # Aber: akkumulierter HF-Verlust (Generationsverlust) soll KOMPENSIERT werden.
        _sfr_cal_39 = kwargs.get("song_calibration_profile", {})
        _sfr_bw_39 = float(_sfr_cal_39.get("source_fidelity_bandwidth_target_hz", 0.0))
        _sfr_hf_loss_39 = float(_sfr_cal_39.get("source_fidelity_hf_loss_db", 0.0))
        _sfr_conf_39 = float(_sfr_cal_39.get("source_fidelity_confidence", 0.5))
        if _sfr_bw_39 > 0.0 and _sfr_conf_39 >= 0.30:
            # Shelf-Frequenz-Deckelung: nie höher als 85% der Original-Bandbreite
            _air_bw_ceil = float(np.clip(_sfr_bw_39 * 0.85, 6000.0, self.AIR_BAND_HZ[1]))
            if _air_bw_ceil < config["shelf_freq_hz"]:
                config["shelf_freq_hz"] = _air_bw_ceil
                logger.debug(
                    "Phase 39: era bw_target=%.0f → shelf_freq capped at %.0f Hz (conf=%.2f)",
                    _sfr_bw_39,
                    _air_bw_ceil,
                    _sfr_conf_39,
                )
            # HF-Verlust-Kompensation: mehr Generationen verloren → exciter-Boost
            # (kompensiert nur was real verloren ging, nicht over-synthesis)
            if _sfr_hf_loss_39 > 0.5:
                _loss_scale = float(np.clip(1.0 + _sfr_hf_loss_39 / 18.0 * _sfr_conf_39, 1.0, 1.35))
                config["exciter_mix"] = float(np.clip(config["exciter_mix"] * _loss_scale, 0.0, 0.55))
                logger.debug(
                    "Phase 39: hf_loss=%.1f dB → exciter_mix×%.2f (scale=%.2f)",
                    _sfr_hf_loss_39,
                    config["exciter_mix"],
                    _loss_scale,
                )
        # §soft_saturation-Guard: Air-Band-Enhancement bei saturiertem Material begrenzen.
        # Soft_saturation erzeugt HF-Obertöne im Presence/Air-Band (4–16 kHz).
        # Zusätzlicher Shelf-Boost auf diesem Oberton-Profil → harsch/kratzig.
        # soft_saturation_preserve=True → max. 40 % Stärke; drive maximal 50 %.
        _p39_soft_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        _p39_soft_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        if _p39_soft_sat_preserve or _p39_soft_sat_sev > 0.3:
            _p39_sat_scale = 1.0
            if _p39_soft_sat_sev > 0.3:
                _p39_sat_scale = float(np.clip(1.0 - (_p39_soft_sat_sev - 0.3) * 1.1, 0.18, 1.0))
            if _p39_soft_sat_preserve and _p39_sat_scale > 0.40:
                _p39_sat_scale = 0.40
            config["shelf_gain_db"] = float(config["shelf_gain_db"] * _p39_sat_scale)
            config["exciter_mix"] = float(config["exciter_mix"] * _p39_sat_scale)
            config["saturation_drive"] = float(config["saturation_drive"] * max(_p39_sat_scale, 0.5))
            logger.debug(
                "Phase 39 soft_saturation guard: severity=%.2f preserve=%s → scale=%.2f (shelf=%.2f dB exciter=%.3f)",
                _p39_soft_sat_sev,
                _p39_soft_sat_preserve,
                _p39_sat_scale,
                config["shelf_gain_db"],
                config["exciter_mix"],
            )
        # Measure initial HF energy
        hf_energy_before = self._measure_hf_energy(audio, sample_rate)

        # §2.51 M/S-Domain: Air Band auf Mid; Side unver\u00e4ndert
        if is_stereo:
            _ch0, _ch1 = stereo_channel_view(audio)
            mid = (_ch0 + _ch1) / np.sqrt(2.0)
            side = (_ch0 - _ch1) / np.sqrt(2.0)
            mid_enhanced = self._enhance_channel(mid, sample_rate, config)
            out_l = (mid_enhanced + side) / np.sqrt(2.0)
            out_r = (mid_enhanced - side) / np.sqrt(2.0)
            enhanced_audio = stereo_like(out_l, out_r, audio)
        else:
            enhanced_audio = self._enhance_channel(audio, sample_rate, config)

        if 0.0 < _effective_strength < 1.0:
            enhanced_audio = audio + _effective_strength * (enhanced_audio - audio)

        # §2.46b Spectral-Tilt-Guard: cap Air-Band synthesis if tilt deviates beyond tolerance
        _tilt_capped_p39 = False
        try:
            _mat_k39 = str(getattr(material, "value", str(material))).lower().replace(" ", "_").replace("-", "_")
            _tol39 = _TILT_TOLERANCE_P39.get(_mat_k39, 2.0)
            _tb39 = _est_tilt_p39(audio, sample_rate)
            _ta39 = _est_tilt_p39(enhanced_audio, sample_rate)
            _dev39 = abs(_ta39 - _tb39)
            if _dev39 > _tol39:
                _cap39 = float(np.clip(1.0 - (_dev39 - _tol39) / (_tol39 * 2.0), 0.5, 1.0))
                enhanced_audio = _cap39 * enhanced_audio + (1.0 - _cap39) * audio
                enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
                _tilt_capped_p39 = True
                logger.info(
                    "phase_39 §2.46b tilt-cap: before=%.2f after=%.2f dev=%.2f tol=%.2f cap=%.2f",
                    _tb39,
                    _ta39,
                    _dev39,
                    _tol39,
                    _cap39,
                )
        except Exception as _tc39:
            logger.debug("phase_39 §2.46b tilt-cap skipped (graceful): %s", _tc39)

        enhanced_audio_pre_guard = enhanced_audio.copy()

        # Measure final HF energy
        hf_energy_after = self._measure_hf_energy(enhanced_audio, sample_rate)
        hf_boost_db = 20 * np.log10((hf_energy_after + 1e-10) / (hf_energy_before + 1e-10))

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        # ── HF-Kumulativ-Limit (Spec §8.2: Presence + Air kumulativ ≤ +4 dB) ──
        # Listening-Fatigue-Schutz: Gesamtanhebung 2–20 kHz limitieren
        hf_cumul_db = float(kwargs.get("hf_cumulative_gain_db", 0.0))
        MAX_HF_CUMUL_DB = 4.0
        if hf_cumul_db > MAX_HF_CUMUL_DB:
            logger.warning(
                "Phase 39: HF-Kumulativ-Limit erreicht (%.1f dB > %.1f dB) — "
                "Air-Band-Gain reduziert (Listening-Fatigue-Schutz, Spec §8.2)",
                hf_cumul_db,
                MAX_HF_CUMUL_DB,
            )
            # Gain-Korrektur: Überschuss rückgängig machen
            excess_db = hf_cumul_db - MAX_HF_CUMUL_DB
            gain_correction = 10 ** (-excess_db / 20.0)
            if enhanced_audio.ndim == 1:
                enhanced_audio = enhanced_audio * gain_correction
            else:
                enhanced_audio = enhanced_audio * gain_correction
            enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        output_guard_enabled = quality_mode in ("quality", "maximum", "studio2026")
        guard = evaluate_output_guard(
            original=audio,
            candidate=enhanced_audio,
            enabled=output_guard_enabled,
            max_abs_rms_delta_db=1.2,
            stereo_side_ratio_min=0.60,
            stereo_side_ratio_max=1.40,
        )
        if guard.fallback:
            enhanced_audio = enhanced_audio_pre_guard
            enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        # §0a / §6.2c BW-Ceiling Hard-Cap: Air-Band-Enhancement (Shelving-EQ) darf
        # das physikalische Trägerlimit nicht überschreiten. Shellac shelf_freq=10kHz
        # würde über das 8-kHz-Ceiling boostten — LPF kappt die Energie.
        _BW_CEILING_39: dict[str, float] = {
            "shellac": 8000.0,
            "wax_cylinder": 5000.0,
            "vinyl": 16000.0,
            "reel_tape": 18000.0,
            "cassette": 15000.0,
        }
        _mat_key_39 = str(getattr(material, "name", str(material))).lower().replace(" ", "_").replace("-", "_")
        _bw_cap_39 = _BW_CEILING_39.get(_mat_key_39)
        if _bw_cap_39 is not None and sample_rate > 0:
            try:
                _nyq39 = float(sample_rate) / 2.0
                _ratio39 = float(np.clip(_bw_cap_39 / _nyq39, 0.01, 0.99))
                _sos_lp39 = signal.butter(8, _ratio39, btype="low", output="sos")
                if enhanced_audio.ndim == 2:
                    if enhanced_audio.shape[0] == 2 and enhanced_audio.shape[1] > 2:
                        enhanced_audio = np.stack(
                            [signal.sosfiltfilt(_sos_lp39, enhanced_audio[c]) for c in range(2)], axis=0
                        ).astype(np.float32)
                    else:
                        _nc39 = enhanced_audio.shape[1]
                        enhanced_audio = np.stack(
                            [signal.sosfiltfilt(_sos_lp39, enhanced_audio[:, c]) for c in range(_nc39)], axis=1
                        ).astype(np.float32)
                else:
                    enhanced_audio = signal.sosfiltfilt(_sos_lp39, enhanced_audio).astype(np.float32)
                enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
            except Exception as _bw39_exc:
                logger.debug("§6.2c phase_39 BW-Ceiling (non-blocking): %s", _bw39_exc)

        # §2.46e Hallucination-Guard: check_hallucination aus backend.core.dsp.hallucination_guard
        # Pflicht nach letzter additiver Op — unconditional, fuer alle Materialien (§2.46e VERBOTEN).
        # `.requires_rollback` → enhanced_audio = audio.copy(); `.score_penalty > 0` → Metadata-0.3.
        _hg_mode_39 = str(kwargs.get("mode", kwargs.get("processing_mode", "restoration"))).lower()
        _hg_score_penalty_39 = 0.0
        _hg_rollback_39 = False
        if _check_hallucination39 is not None:
            try:
                _hg_result39 = _check_hallucination39(
                    audio,
                    enhanced_audio,
                    sr=sample_rate,
                    mode=_hg_mode_39,
                    material_bw_ceiling_hz=_bw_cap_39,
                )
                if _hg_result39.requires_rollback:
                    logger.warning(
                        "§2.46e Phase-39 Hallucination-Rollback: spectral_novelty=%.3f (Threshold 0.15)",
                        _hg_result39.spectral_novelty,
                    )
                    enhanced_audio = audio.copy()
                    _hg_rollback_39 = True
                if _hg_result39.score_penalty > 0:
                    _hg_score_penalty_39 = float(_hg_result39.score_penalty)
            except Exception as _hg39_exc:
                logger.debug("Phase 39 HallucinationGuard (non-blocking): %s", _hg39_exc)

        # §V22 Pre-Echo-Prevention — Additive Air-Band auf Transient-Shifts prüfen (§2.73, non-blocking)
        try:
            from backend.core.dsp.transient_guard import (
                detect_transient_shifts as _dts_39,  # pylint: disable=import-outside-toplevel
            )

            _ax_v22_39 = -1 if audio.ndim == 2 and audio.shape[-1] <= 8 else 0
            _pre_v22_39 = (
                audio.mean(axis=_ax_v22_39).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
            )
            _ax_post_39 = -1 if enhanced_audio.ndim == 2 and enhanced_audio.shape[-1] <= 8 else 0
            _post_v22_39 = (
                enhanced_audio.mean(axis=_ax_post_39).astype(np.float32)
                if enhanced_audio.ndim == 2
                else enhanced_audio.astype(np.float32)
            )
            _ts_39 = _dts_39(_pre_v22_39, _post_v22_39, sample_rate)
            if not _ts_39.ok:
                _wet_ts_39 = max(0.0, 1.0 - _ts_39.blend_reduction)
                enhanced_audio = (_wet_ts_39 * enhanced_audio + (1.0 - _wet_ts_39) * audio).astype(np.float32)
                logger.warning(
                    "§V22 phase_39: onset_shift=%.2f ms → blend_reduction=%.2f",
                    _ts_39.max_shift_ms,
                    _ts_39.blend_reduction,
                )
        except Exception as _v22_39_exc:
            logger.debug("§V22 phase_39 transient_guard non-blocking: %s", _v22_39_exc)

        return PhaseResult(
            success=True,
            audio=enhanced_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "hf_boost_db": float(hf_boost_db),
                "shelf_gain_db": float(config["shelf_gain_db"]),
                "shelf_freq_hz": float(config["shelf_freq_hz"]),
                "rt_factor": float(rt_factor),
                "hf_cumulative_db": float(hf_cumul_db),
                "quality_mode": quality_mode,
                "hq_scale": hq_scale,
                "output_guard_enabled": output_guard_enabled,
                "output_guard_fallback": guard.fallback,
                "output_guard_reason": guard.reason,
                "rms_delta_db": guard.rms_delta_db,
                "stereo_side_ratio": guard.stereo_side_ratio,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "spectral_tilt_capped": _tilt_capped_p39,
                "hg_score_penalty": _hg_score_penalty_39,
                "hg_rollback": _hg_rollback_39,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[],
        )

    def _enhance_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """
        Enhance air band in a single audio channel.

        v2.1 fix: Correct combination formula.
        - Old (WRONG): shelved * 0.7 + excited * 0.3
          → Scales down the dry signal, loses energy at non-HF bands
        - New (CORRECT): shelved + wet_hf * exciter_mix
          → Keeps the shelf as the primary output; adds only the exciter delta on top
          → exciter_mix controls how much of the HF exciter blend is added

        Also: Bandwidth guard — only add exciter energy if source has content above 8 kHz
        (avoids adding artificial HF air above source's actual bandwidth limit).
        """
        # 1. Bandwidth guard: measure HF energy [10k–18k Hz] vs broadband
        if sample_rate >= 22050 and len(audio) > 8:
            n_fft = 1024
            segment = audio[: min(len(audio), n_fft * 8)]
            spec = np.abs(np.fft.rfft(segment)) ** 2
            freqs = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)
            broadband = float(np.sum(spec) + 1e-12)
            hf_mask = freqs >= 8000.0
            hf_content = float(np.sum(spec[hf_mask]) + 1e-12)
            hf_fraction = hf_content / broadband
            # If source has almost no HF content (bandwidth-limited), reduce exciter
            if hf_fraction < 0.005:
                # Very limited bandwidth — don't add synthetic air
                bw_scale = min(1.0, hf_fraction / 0.005)
                config = dict(config)  # don't mutate the outer config
                config["exciter_mix"] = config["exciter_mix"] * bw_scale
                logger.debug("Phase 39 BW-guard: hf_frac=%.4f → exciter scale=%.2f", hf_fraction, bw_scale)

        # 2. Apply the shelving EQ (primary output)
        shelved = self._apply_high_shelf(audio, sample_rate, config["shelf_freq_hz"], config["shelf_gain_db"])

        # 3. Generate exciter delta (only the HF harmonic component added to dry signal)
        excited = self._apply_exciter(audio, sample_rate, config["exciter_mix"], config["saturation_drive"])

        # 4. Correct combination: shelved is the base; add additive exciter delta
        #    exciter_mix is already applied inside _apply_exciter (returns audio + hf_delta*mix)
        #    So: combine shelf + (excited - audio) * additional blend factor
        hf_delta = excited - audio  # just the harmonic addition
        enhanced = shelved + hf_delta  # proper additive combination

        return enhanced  # type: ignore[no-any-return]

    def _apply_high_shelf(self, audio: np.ndarray, sample_rate: int, freq_hz: float, gain_db: float) -> np.ndarray:
        """Wendet an: high-frequency shelving filter (biquad coefficients cached per key)."""
        cache_key = (sample_rate, freq_hz, gain_db)
        with self._cache_lock:
            if cache_key not in self._shelf_coeffs:
                w0 = 2 * np.pi * freq_hz / sample_rate
                A = 10 ** (gain_db / 40)
                alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2)
                b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
                b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
                b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
                a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
                a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
                a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1, a1 / a0, a2 / a0])
                self._shelf_coeffs[cache_key] = (b, a)
            b, a = self._shelf_coeffs[cache_key]
        # §2.51 Anti-Zeitversatz: filtfilt (Zero-Phase) statt lfilter — Shelf-EQ darf keine
        # Gruppenlatenz erzeugen (hörbar als Zeitversatz auf HF-Transienten/Vokaleinsätzen).
        if len(audio) >= 9:
            return safe_filtfilt(b, a, audio)  # type: ignore[no-any-return]
        return signal.lfilter(b, a, audio)  # type: ignore[no-any-return]

    def _apply_exciter(self, audio: np.ndarray, sample_rate: int, mix: float, drive: float) -> np.ndarray:
        """Wendet an: harmonic exciter to HF region (SOS filter cached per sample_rate)."""
        with self._cache_lock:
            if sample_rate not in self._sos_air_cache:
                self._sos_air_cache[sample_rate] = signal.butter(
                    4, self.AIR_BAND_HZ, btype="band", fs=sample_rate, output="sos"
                )
            sos = self._sos_air_cache[sample_rate]
        # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) statt sosfilt — hf + audio werden gemischt.
        hf = signal.sosfiltfilt(sos, audio)
        excited_hf = np.tanh(hf * drive * 2) / (drive + 0.5)
        return audio + excited_hf * mix  # type: ignore[no-any-return]

    def _measure_hf_energy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Misst high-frequency energy (12-20 kHz RMS, cached SOS filter)."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel
        with self._cache_lock:
            if sample_rate not in self._sos_air_cache:
                self._sos_air_cache[sample_rate] = signal.butter(
                    4, self.AIR_BAND_HZ, btype="band", fs=sample_rate, output="sos"
                )
            sos = self._sos_air_cache[sample_rate]
        hf = signal.sosfilt(sos, audio)
        return float(np.sqrt(np.mean(hf**2)))
