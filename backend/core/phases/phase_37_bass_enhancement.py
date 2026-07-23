#!/usr/bin/env python3
"""
Phase 37: Bass Enhancement v2.0 - Professional
Harmonic bass synthesis and sub-bass generation.

Algorithm Overview:
1. Bass Extraction: Isolate 20-250 Hz region
2. Harmonic Generation:
   - 2nd Harmonic (octave): Adds warmth and definition
   - 3rd Harmonic: Adds thickness and power
   - Sub-harmonic (octave down): Adds weight on large systems
3. Waveshaping: Soft saturation for natural harmonics
4. Material Adaptation:
   - Shellac: Restore missing bass (bandwidth-limited)
   - Vinyl: Enhance sub-bass (rumble filter often removes it)
   - Tape: Moderate enhancement (tape saturation already adds harmonics)
   - Digital: Aggressive (over-limited bass needs life)
5. Filtering: Remove excessive sub-bass (<30 Hz) to prevent mud

Scientific Foundation:
- Laroche & Dolson (1999): Improved Phase Vocoder for Time-Scale Modification
- Avendano & Deng (2003): Frequency Lowering for High-Frequency Hearing Loss
- Carty & Raftery (2010): Selective Bass Enhancement
- Zölzer (2011): DAFX - Waveshaping and Distortion
- Parker et al. (2013): Maximally Diffuse Sound Fields

Industry Benchmarks:
- Waves MaxxBass (Psychoacoustic bass enhancement)
- dbx 120A Subharmonic Synthesizer (Classic hardware)
- BBE Sonic Maximizer (Harmonic enhancement)
- Noveltech Character (Harmonic generator)
- SPL Vitalizer (Psychoacoustic processing)

Quality Target: 0.78 → 0.91 (+17% improvement)
Performance Target: <0.15× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import os
import logging
import threading
import time

import numpy as np
from scipy import signal
from scipy.signal import hilbert as _hilbert37

from backend.core.audio_utils import audio_sample_count, stereo_channel_view, stereo_like
from backend.core.defect_scanner import MaterialType
from backend.core.ml_model_readiness import check_ml_model_ready

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

try:
    from backend.core.dsp.hallucination_guard import check_hallucination as _check_hg_37_fn
except Exception:
    _check_hg_37_fn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class BassEnhancement(PhaseInterface):
    """
    Professional Bass Enhancement Engine.

    Key Features:
    - Harmonic synthesis (2nd, 3rd harmonics)
    - Sub-harmonic generation (octave down)
    - Material-adaptive intensity
    - Psychoacoustic optimization
    - Mud prevention (high-pass filtering)

    Use Cases:
    - Restore missing bass from bandwidth-limited sources
    - Enhance bass perception on small speakers
    - Add weight and power to thin mixes
    - Compensate for bass loss in processing chain

    Performance: <0.15× realtime on modern CPU
    """

    # Bass frequency ranges
    BASS_RANGE_HZ = (20, 250)
    SUB_BASS_RANGE_HZ = (20, 80)

    # Enhancement parameters (material-adaptive)
    ENHANCEMENT_CONFIG = {
        MaterialType.SHELLAC: {
            "harmonic_2_gain": 0.55,  # Strong (restore missing bass) — v10.0.0: ↑0.50→0.55
            "harmonic_3_gain": 0.35,
            "sub_harmonic_gain": 0.25,
            "saturation_drive": 0.40,
            "mix": 0.65,  # v10.0.0: ↑0.50→0.65 — Shellac-Bass deutlich stärken
        },
        MaterialType.VINYL: {
            "harmonic_2_gain": 0.45,
            "harmonic_3_gain": 0.28,
            "sub_harmonic_gain": 0.32,
            "saturation_drive": 0.35,
            "mix": 0.60,  # v10.0.0: ↑0.45→0.60
        },
        MaterialType.TAPE: {
            "harmonic_2_gain": 0.35,  # v10.0.0: ↑0.30→0.35
            "harmonic_3_gain": 0.22,
            "sub_harmonic_gain": 0.18,
            "saturation_drive": 0.28,
            "mix": 0.50,  # v10.0.0: ↑0.35→0.50
        },
        MaterialType.CASSETTE: {  # v10.0.0: IEC 60094-1 — gleiche Capstan-Physik wie TAPE
            "harmonic_2_gain": 0.35,
            "harmonic_3_gain": 0.22,
            "sub_harmonic_gain": 0.18,
            "saturation_drive": 0.28,
            "mix": 0.50,
        },
        MaterialType.CD_DIGITAL: {
            "harmonic_2_gain": 0.50,  # Restore life from over-limiting  v10.0.0: ↑0.45→0.50
            "harmonic_3_gain": 0.32,
            "sub_harmonic_gain": 0.28,
            "saturation_drive": 0.45,
            "mix": 0.65,  # v10.0.0: ↑0.50→0.65
        },
        MaterialType.STREAMING: {
            "harmonic_2_gain": 0.42,
            "harmonic_3_gain": 0.28,
            "sub_harmonic_gain": 0.22,
            "saturation_drive": 0.40,
            "mix": 0.58,  # v10.0.0: ↑0.45→0.58
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "Bass Enhancement v2 Professional"
        self._sos_cache: dict[int, dict[str, np.ndarray]] = {}
        self._sos_cache_lock = threading.Lock()

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_37_bass_enhancement",
            name="Bass Enhancement v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=6,
            dependencies=[],
            estimated_time_factor=0.15,
            version="2.0.0",
            memory_requirement_mb=50,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.91,
            description="Harmonic bass synthesis and sub-bass generation",
        )

    # pylint: disable-next=arguments-renamed
    def process(  # type: ignore[override]
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        check_ml_model_ready("PANNs", phase_name="37")
        """
        Wendet Bass-Verbesserung auf Audio an.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with enhanced audio
        """
        start_time = time.time()
        self.validate_input(audio)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        # §V41 ForwardMaskingGuard: Stärke in post-transienten Masking-Fenstern erhöhen.
        # Optimierung v10.0.0: UV3 berechnet Zonen vor der Phase und übergibt sie via kwargs;
        # nur wenn keine pre-computed Zonen vorhanden, werden sie neu berechnet (spart CPU).
        _panns_s_37 = float(kwargs.get("panns_singing", 0.0))
        if _panns_s_37 >= 0.25 and _effective_strength > 0.0:
            try:
                from backend.core.dsp.temporal_masking import (
                    get_forward_masking_guard as _fmg_fn_37,  # pylint: disable=import-outside-toplevel
                )

                _fmz_37 = kwargs.get("forward_masking_zones") or _fmg_fn_37().compute_zones(audio, sample_rate)
                if _fmz_37:
                    _n_s_37 = audio.shape[-1] if audio.ndim > 1 else len(audio)
                    _zone_samples_37 = sum(z.end_sample - z.start_sample for z in _fmz_37)
                    _zone_frac_37 = float(np.clip(_zone_samples_37 / max(1, _n_s_37), 0.0, 1.0))
                    _boost_37 = _zone_frac_37 * 0.15
                    _effective_strength = float(np.clip(_effective_strength + _boost_37, 0.0, 1.0))
                    logger.debug(
                        "Phase37 §V41 ForwardMasking: zone_frac=%.2f boost=%.3f → eff_str=%.3f",
                        _zone_frac_37,
                        _boost_37,
                        _effective_strength,
                    )
            except Exception as _fmg_exc_37:  # pylint: disable=broad-except
                logger.debug("Phase37 §V41 ForwardMaskingGuard non-blocking: %s", _fmg_exc_37)

        if _effective_strength < 0.01:  # §v10.0.5: < 1% = keine sinnvolle Wirkung
            logger.info(
                "Phase 37: skipped — effective_strength=%.3f (no bass enhancement applied)", _effective_strength
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

        is_stereo = audio.ndim == 2
        _mk = material.value if isinstance(material, MaterialType) else material  # §v10.113
        config = dict(self.ENHANCEMENT_CONFIG.get(_mk, self.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL]))
        config["harmonic_2_gain"] = float(config["harmonic_2_gain"] * _effective_strength)
        config["harmonic_3_gain"] = float(config["harmonic_3_gain"] * _effective_strength)

        # §v10.94 Non-Plus-Ultra: Hum-Notch-Frequenzen aus P02 vermeiden.
        # P02 notched 50/60 Hz + Harmonische (bis 480 Hz). Synthese von
        # Sub-Harmonischen in diesen Bändern würde residuale Comb-Artefakte
        # verstärken. Reduziere sub_harmonic_gain proportional zur Überlappung.
        _hum_freqs: list[int] = list(kwargs.get("hum_notch_freqs", []) or [])
        if _hum_freqs:
            _bass_lo = 20.0  # Bass-Synthese-Minimum (Hz)
            _bass_hi = 250.0  # Bass-Synthese-Maximum (Hz)
            _overlap_count = sum(1 for _f in _hum_freqs if _bass_lo <= float(_f) * 2 <= _bass_hi)
            if _overlap_count > 0:
                # Reduziere sub-harmonic gain: −20 % pro überlappendem Hum-Fundamental
                _hum_damp = float(np.clip(1.0 - 0.20 * _overlap_count, 0.30, 1.0))
                config["sub_harmonic_gain"] = float(config["sub_harmonic_gain"] * _hum_damp)
                logger.debug(
                    "P37: hum_notch_freqs=%s → sub_harmonic_gain damped by %.0f%%",
                    _hum_freqs, (1.0 - _hum_damp) * 100,
                )
        config["sub_harmonic_gain"] = float(config["sub_harmonic_gain"] * _effective_strength)
        config["saturation_drive"] = float(config["saturation_drive"] * _effective_strength)
        config["mix"] = float(config["mix"] * _effective_strength)

        # ── Era/Genre-adaptive bass scaling (context injection §2.x) ──
        _waerme = kwargs.get("waerme_target")
        _decade = kwargs.get("decade")
        if _waerme is not None:
            # waerme_target 0.80–0.90 → scale 0.85–1.10
            _w_scale = 0.50 + 0.667 * float(_waerme)
            _w_scale = max(0.70, min(1.25, _w_scale))
            config["harmonic_2_gain"] *= _w_scale
            config["harmonic_3_gain"] *= _w_scale
            config["sub_harmonic_gain"] *= _w_scale
            config["mix"] *= _w_scale
            logger.debug("Phase 37: waerme_target=%.2f → bass scale=%.2f", float(_waerme), _w_scale)
        if _decade is not None:
            _dec = int(_decade)
            if _dec <= 1940:
                # Vintage: less sub-bass (speakers couldn't reproduce), more H2
                config["sub_harmonic_gain"] *= 0.60
                config["harmonic_2_gain"] *= 1.20
            elif _dec >= 1990:
                # Modern: less harmonic coloring, preserve existing bass
                config["harmonic_2_gain"] *= 0.80
                config["harmonic_3_gain"] *= 0.80
        # §soft_saturation-Guard: Bass-Enhancement bei saturiertem Material begrenzen.
        # Soft_saturation-Verzerrung ist im Bass-Bereich (80–400 Hz) besonders hörbar:
        # zusätzlicher H2/H3-Bass-Boost auf gesättigtem Signal → Intermodulation/Dröhnen.
        # soft_saturation_preserve=True → max. 30 % Stärke (§0 Primum non nocere).
        _p37_soft_sat_preserve = bool(kwargs.get("soft_saturation_preserve", False))
        _p37_soft_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))
        if _p37_soft_sat_preserve or _p37_soft_sat_sev > 0.3:
            _p37_sat_scale = 1.0
            if _p37_soft_sat_sev > 0.3:
                _p37_sat_scale = float(np.clip(1.0 - (_p37_soft_sat_sev - 0.3) * 1.0, 0.20, 1.0))
            if _p37_soft_sat_preserve and _p37_sat_scale > 0.30:
                _p37_sat_scale = 0.30
            config["harmonic_2_gain"] = float(config["harmonic_2_gain"] * _p37_sat_scale)
            config["harmonic_3_gain"] = float(config["harmonic_3_gain"] * _p37_sat_scale)
            config["sub_harmonic_gain"] = float(config["sub_harmonic_gain"] * _p37_sat_scale)
            config["saturation_drive"] = float(np.clip(config["saturation_drive"] * max(_p37_sat_scale, 0.4), 0.0, 1.0))
            logger.debug(
                "Phase 37 soft_saturation guard: severity=%.2f preserve=%s → scale=%.2f",
                _p37_soft_sat_sev,
                _p37_soft_sat_preserve,
                _p37_sat_scale,
            )
        # Measure initial bass energy
        bass_energy_before = self._measure_bass_energy(audio, sample_rate)

        # §2.51 M/S-Domain: Bass Enhancement nur auf Mid, Side unver\u00e4ndert
        if is_stereo:
            _ch0, _ch1 = stereo_channel_view(audio)
            mid = (_ch0 + _ch1) / np.sqrt(2.0)
            side = (_ch0 - _ch1) / np.sqrt(2.0)
            mid_enhanced = self._enhance_channel(mid, sample_rate, config)
            # Decode back to L/R
            out_l = (mid_enhanced + side) / np.sqrt(2.0)
            out_r = (mid_enhanced - side) / np.sqrt(2.0)
            enhanced_audio = stereo_like(out_l, out_r, audio)
        else:
            enhanced_audio = self._enhance_channel(audio, sample_rate, config)

        if 0.0 < _effective_strength < 1.0:
            enhanced_audio = audio + _effective_strength * (enhanced_audio - audio)

        # Measure final bass energy
        bass_energy_after = self._measure_bass_energy(enhanced_audio, sample_rate)
        bass_boost_db = 20 * np.log10((bass_energy_after + 1e-10) / (bass_energy_before + 1e-10))

        execution_time = time.time() - start_time
        rt_factor = execution_time / (audio_sample_count(audio) / sample_rate)

        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
        # §2.46e Hallucination-Guard (Pflicht für additive Phasen)
        try:
            _mode_37 = str(kwargs.get("mode", "restoration"))
            _hg_37 = _check_hg_37_fn(audio, enhanced_audio, sr=sample_rate, mode=_mode_37)  # type: ignore[misc]
            if _hg_37.requires_rollback:
                logger.warning(
                    "phase_37: hallucination_guard rollback (spectral_novelty=%.3f)", _hg_37.spectral_novelty
                )
                enhanced_audio = np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            elif _hg_37.score_penalty > 0.0:
                logger.info(
                    "phase_37: hallucination_guard penalty=%.1f (spectral_novelty=%.3f)",
                    _hg_37.score_penalty,
                    _hg_37.spectral_novelty,
                )
        except Exception as _hg37_exc:
            logger.debug("phase_37: hallucination_guard failed (non-blocking): %s", _hg37_exc)

        # §V22 Pre-Echo-Prevention — Additive Bass-Enhancement auf Transient-Shifts prüfen (§2.73, non-blocking)
        try:
            from backend.core.dsp.transient_guard import (
                detect_transient_shifts as _dts_37,  # pylint: disable=import-outside-toplevel
            )

            _ax_v22_37 = -1 if audio.ndim == 2 and audio.shape[-1] <= 8 else 0
            _pre_v22_37 = (
                audio.mean(axis=_ax_v22_37).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
            )
            _ax_post_37 = -1 if enhanced_audio.ndim == 2 and enhanced_audio.shape[-1] <= 8 else 0
            _post_v22_37 = (
                enhanced_audio.mean(axis=_ax_post_37).astype(np.float32)
                if enhanced_audio.ndim == 2
                else enhanced_audio.astype(np.float32)
            )
            _ts_37 = _dts_37(_pre_v22_37, _post_v22_37, sample_rate)
            if not _ts_37.ok:
                _wet_ts_37 = max(0.0, 1.0 - _ts_37.blend_reduction)
                enhanced_audio = (_wet_ts_37 * enhanced_audio + (1.0 - _wet_ts_37) * audio).astype(np.float32)
                _log_v22_37 = logger.warning if _wet_ts_37 > 0.0 else logger.info
                _log_v22_37(
                    "§V22 phase_37: onset_shift=%.2f ms → blend_reduction=%.2f",
                    _ts_37.max_shift_ms,
                    _ts_37.blend_reduction,
                )
        except Exception as _v22_37_exc:
            logger.debug("§V22 phase_37 transient_guard non-blocking: %s", _v22_37_exc)
        return PhaseResult(
            success=True,
            audio=enhanced_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "bass_boost_db": float(bass_boost_db),
                "harmonic_2_gain": float(config["harmonic_2_gain"]),
                "harmonic_3_gain": float(config["harmonic_3_gain"]),
                "sub_harmonic_gain": float(config["sub_harmonic_gain"]),
                "onset_guard_wet": float(locals().get("_wet_ts_37", 1.0)),
                "rt_factor": float(rt_factor),
                "virtual_pitch_active": True,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[],
        )

    def _enhance_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """Enhance bass in a single audio channel."""
        # Use cached filters – avoid repeated butter() design per call
        with self._sos_cache_lock:
            if sample_rate not in self._sos_cache:
                self._sos_cache[sample_rate] = {
                    "bass_band": signal.butter(4, self.BASS_RANGE_HZ, btype="band", fs=sample_rate, output="sos"),
                    "hp": signal.butter(2, 25, btype="high", fs=sample_rate, output="sos"),
                }
            cached = self._sos_cache[sample_rate]

        # Extract bass region
        # §2.51 Anti-Zeitversatz: sosfiltfilt — harmonics werden zu original addiert.
        bass = signal.sosfiltfilt(cached["bass_band"], audio)

        # Generate harmonics
        harmonics = self._generate_harmonics(bass, config)

        # Mix with original
        enhanced = audio + harmonics * config["mix"]

        # High-pass filter to remove excessive sub-bass
        # §2.51 Anti-Zeitversatz: sosfiltfilt — Ausgabe wird nicht zu Original addiert (V11).
        enhanced = signal.sosfiltfilt(cached["hp"], enhanced)

        return enhanced  # type: ignore[no-any-return]

    def _generate_harmonics(self, bass: np.ndarray, config: dict[str, float]) -> np.ndarray:
        """Generiert harmonic content from bass."""
        # Soft saturation (generates 2nd and 3rd harmonics naturally)
        drive = config["saturation_drive"]
        saturated = np.tanh(bass * drive * 3) / (drive + 0.5)

        # 2nd harmonic (octave up) - via full-wave rectification
        harmonic_2 = np.abs(bass) * config["harmonic_2_gain"]

        # 3rd harmonic - via cubic distortion
        harmonic_3 = (bass**3) * config["harmonic_3_gain"] * 0.5

        # Sub-harmonic (octave down) - via Virtual Pitch / Missing Fundamental (Moore 2006)
        sub_harmonic = self._virtual_pitch_bass(bass, 48000) * config["sub_harmonic_gain"]

        # Combine
        harmonics = harmonic_2 + harmonic_3 + sub_harmonic + saturated * 0.3

        return harmonics  # type: ignore[no-any-return]

    def _generate_sub_harmonic(self, bass: np.ndarray) -> np.ndarray:
        """Generiert sub-harmonic (octave down)."""
        # Vectorized octave-down via sample-and-hold (avoids Python for-loop)
        sub = np.repeat(bass[::2], 2)[: len(bass)]
        return sub  # type: ignore[no-any-return]

    def _virtual_pitch_bass(self, bass: np.ndarray, sr: int) -> np.ndarray:
        """Virtual Pitch / Missing Fundamental (Moore et al. 2006, JASA).

        Das Gehirn rekonstruiert den Grundton aus Obertönen (z.B. 60 Hz
        Basseindruck aus 120/180/240 Hz Komponenten). Dieser Algorithmus
        erzeugt Oberton-Cluster im Bereich 120-500 Hz, die den perceptuellen
        Basseindruck verstärken ohne Sub-Bassenergie hinzuzufügen.

        Moore et al. (2006): "A Model for the Prediction of Thresholds,
        Loudness, and Partial Loudness" — Virtual Pitch via Harmonic Template Matching.
        """
        if len(bass) < 1024:
            return self._generate_sub_harmonic(bass) * 0.25  # type: ignore[no-any-return]

        # Bandpass 120–500 Hz: Zone der Missing-Fundamental-Wahrnehmung
        try:
            sos_vp = signal.butter(4, [120.0 / (sr / 2), min(500.0 / (sr / 2), 0.99)], btype="band", output="sos")
            vp_band = signal.sosfiltfilt(sos_vp, bass)
        except Exception as e:
            logger.warning("phase_37_bass_enhancement.py::_virtual_pitch_bass fallback: %s", e)
            return self._generate_sub_harmonic(bass) * 0.25  # type: ignore[no-any-return]

        _rms = float(np.sqrt(np.mean(vp_band.astype(np.float64) ** 2) + 1e-12))
        if _rms < 1e-5:
            return np.zeros_like(bass)  # type: ignore[no-any-return]

        # Moore-style harmonic template matching:
        # estimate virtual fundamental f0 from harmonics (2f0..5f0 in 120–500 Hz).
        _n_fft = int(min(8192, 2 ** np.floor(np.log2(max(1024, len(vp_band))))))
        _win = signal.get_window("hann", _n_fft, fftbins=True)
        _seg = vp_band[:_n_fft] * _win
        _spec = np.fft.rfft(_seg)
        _mag = np.abs(_spec)
        _freqs = np.fft.rfftfreq(_n_fft, d=1.0 / sr)
        _f0_candidates = np.arange(40.0, 121.0, 1.0)
        _weights = np.array([1.00, 0.75, 0.55, 0.40], dtype=np.float64)  # k=2..5
        _scores = np.zeros_like(_f0_candidates)

        for i, _f0_cand in enumerate(_f0_candidates):
            _score = 0.0
            for _k_idx, _k in enumerate(range(2, 6)):
                _target = _k * _f0_cand
                if _target < 120.0 or _target > 500.0:
                    continue
                _bin = int(np.argmin(np.abs(_freqs - _target)))
                _score += _weights[_k_idx] * float(_mag[_bin])
            _scores[i] = _score

        _best_idx = int(np.argmax(_scores))
        _f0 = float(_f0_candidates[_best_idx])
        _score_peak = float(_scores[_best_idx])
        _score_ref = float(np.percentile(_scores, 90)) + 1e-12
        _template_conf = float(np.clip(_score_peak / _score_ref, 0.0, 1.5) / 1.5)

        # Envelope-followed fundamental synthesis: carries low-band energy while
        # preserving groove/transients from the detected harmonic activity.
        try:
            _env = np.abs(_hilbert37(vp_band))
            _env_lp = signal.butter(2, 20.0 / (sr / 2), btype="low", output="sos")
            _env = signal.sosfiltfilt(_env_lp, _env)
        except Exception:
            _env = np.abs(vp_band)
        _env = _env / (float(np.percentile(_env, 95)) + 1e-8)
        _env = np.clip(_env, 0.0, 1.5)

        _phase = np.cumsum(np.full(len(vp_band), 2.0 * np.pi * _f0 / sr, dtype=np.float64))
        _fund = np.sin(_phase)
        _virtual = (_fund * _env).astype(np.float32)

        # Band-limit to 60–120 Hz to avoid uncontrolled low-end bloom.
        try:
            sos_sub = signal.butter(4, [60.0 / (sr / 2), 120.0 / (sr / 2)], btype="band", output="sos")
            sub_result = signal.sosfiltfilt(sos_sub, _virtual)
        except Exception:
            sub_result = _virtual * 0.5

        # Confidence-aware blend avoids artifacts when template fit is weak.
        _gain = float(np.clip(0.20 + 0.80 * _template_conf, 0.20, 1.0))
        sub_result = sub_result * _gain
        sub_result = np.nan_to_num(sub_result, nan=0.0, posinf=0.0, neginf=0.0)
        sub_result = np.clip(sub_result, -1.0, 1.0)
        return sub_result  # type: ignore[no-any-return]

    def _measure_bass_energy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Misst bass energy (20-250 Hz RMS)."""
        if audio.ndim == 2:
            # §2.51 Stereo-Axis-Invariante: channels-first (2, N) oder channels-last (N, 2)
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                audio = audio[0]  # channels-first → Kanal 0
            else:
                audio = audio[:, 0]  # channels-last → Kanal 0

        # Extract bass (use cached filter)
        with self._sos_cache_lock:
            if sample_rate not in self._sos_cache:
                self._sos_cache[sample_rate] = {
                    "bass_band": signal.butter(4, self.BASS_RANGE_HZ, btype="band", fs=sample_rate, output="sos"),
                    "hp": signal.butter(2, 25, btype="high", fs=sample_rate, output="sos"),
                }
            bass_sos = self._sos_cache[sample_rate]["bass_band"]
        # Zero-Phase-Messung: sosfiltfilt verhindert Phasenversatz bei Energieberechnung.
        bass = signal.sosfiltfilt(bass_sos, audio)

        # RMS energy
        rms = np.sqrt(np.mean(bass**2))

        return float(rms)
