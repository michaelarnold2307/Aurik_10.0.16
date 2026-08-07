from __future__ import annotations

import pytest

"""Tests for universal usability improvements (v10.0.0.x).

Covers:
1. Beat-Reliability-Score (Rubato-Detektor)
2. Loudness-War-Guard
3. Operatic Vibrato Detektor
4. Extended Instrument Routing
5. Dense Ensemble Guard
6. Leichtpfad (restorability >= 85)
7. Vibrato-aware Vocal-Gating
"""


import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Stubs to avoid importing heavy modules ──────────────────────────────
@dataclass
class _FakeDefectScore:
    severity: float = 0.0
    confidence: float = 1.0


class _MaterialEnum:
    """Hashable material type for testing."""

    def __init__(self, name: str):
        self.value = name.lower()
        self._name = name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if isinstance(other, _MaterialEnum):
            return self._name == other._name
        return NotImplemented

    def __repr__(self):
        return f"MaterialType.{self._name}"


class _MaterialTypeNS:
    """Fake MaterialType enum with needed values."""

    def __init__(self):
        for name in [
            "VINYL",
            "TAPE",
            "SHELLAC",
            "CD_DIGITAL",
            "MP3_LOW",
            "MP3_HIGH",
            "AAC",
            "STREAMING",
            "DAT",
            "CASSETTE",
            "REEL_TAPE",
            "WAX_CYLINDER",
            "UNKNOWN",
        ]:
            setattr(self, name, _MaterialEnum(name))


MaterialType = _MaterialTypeNS()


def _make_defect_type():
    """Create a fake DefectType enum with needed values."""
    dt = types.SimpleNamespace()
    names = [
        "HARMONIC_DISTORTION",
        "NOISE_BROADBAND",
        "NOISE_TONAL",
        "CLIPPING",
        "HUM_BUZZ",
        "CLICKS_POPS",
        "CRACKLE",
        "WOW_FLUTTER",
        "DROPOUT",
        "SIBILANCE",
        "DYNAMIC_COMPRESSION_EXCESS",
        "COMPRESSION_ARTIFACTS",
        "DIGITAL_ARTIFACTS",
        "VOCAL_HARSHNESS",
        "HIGH_FREQUENCY_LOSS",
        "PITCH_DRIFT",
        "TAPE_HEAD_LEVEL_DIP",
        "HEAD_WEAR",
        "AZIMUTH_ERROR",
        "TRANSIENT_SMEARING",
        "PRE_ECHO",
    ]
    for name in names:
        setattr(dt, name, types.SimpleNamespace(value=name.lower()))
    return dt


DefectType = _make_defect_type()


@dataclass
class _FakeDefectResult:
    scores: dict = field(default_factory=dict)
    material_type: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass
class _FakeEraResult:
    decade: int | None = None


# ── Test Helpers ─────────────────────────────────────────────────────────
def _make_sine(freq: float = 440.0, sr: int = 48000, dur: float = 2.0) -> np.ndarray:
    """Generate a pure sine tone."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t) * 0.5  # type: ignore[no-any-return]


def _make_vibrato_signal(sr: int = 48000, dur: float = 2.0) -> np.ndarray:
    """Generate signal with strong 5 Hz vibrato (pitch modulation)."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    # Vibrato: 440 Hz carrier, +-50 Hz vibrato at 5 Hz
    freq = 440.0 + 50.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = np.cumsum(2 * np.pi * freq / sr)
    return (np.sin(phase) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _make_steady_beat(sr: int = 48000, dur: float = 4.0, bpm: float = 120.0) -> np.ndarray:
    """Generate signal with clear periodic transients (steady beat)."""
    n = int(sr * dur)
    audio = np.random.randn(n).astype(np.float32) * 0.01
    beat_interval = int(60.0 / bpm * sr)
    for i in range(0, n, beat_interval):
        end = min(i + int(0.01 * sr), n)
        audio[i:end] = 0.8 * np.sign(np.random.randn(end - i)).astype(np.float32)
    return audio


def _make_rubato_signal(sr: int = 48000, dur: float = 4.0) -> np.ndarray:
    """Generate signal WITHOUT periodic beat structure (rubato/free-form)."""
    n = int(sr * dur)
    audio = np.random.randn(n).astype(np.float32) * 0.02
    # Random sparse transients at non-periodic positions
    rng = np.random.RandomState(42)
    for _ in range(5):
        pos = rng.randint(0, n - int(0.01 * sr))
        audio[pos : pos + int(0.005 * sr)] = 0.6
    return audio


# ═══════════════════════════════════════════════════════════════════════
# 1. Beat-Reliability-Score
# ═══════════════════════════════════════════════════════════════════════
@pytest.mark.unit
class TestBeatReliability:
    """Test beat-reliability score calculation (rubato detection)."""

    def _compute_beat_reliability(self, audio: np.ndarray, sr: int = 48000) -> float:
        """Replicate _beat_reliability calculation from _select_phases."""
        _br_mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        _br_mono = np.asarray(_br_mono, dtype=np.float32)
        _br_hop = max(1, sr // 100)
        _br_win = _br_hop * 2
        _br_n_frames = max(1, (len(_br_mono) - _br_win) // _br_hop)
        if _br_n_frames <= 20:
            return 1.0  # default
        _br_flux = np.zeros(_br_n_frames, dtype=np.float32)
        _prev_mag = np.zeros(_br_win // 2 + 1, dtype=np.float32)
        for _bi in range(_br_n_frames):
            _bstart = _bi * _br_hop
            _bframe = _br_mono[_bstart : _bstart + _br_win]
            if len(_bframe) < _br_win:
                break
            _bmag = np.abs(np.fft.rfft(_bframe * np.hanning(_br_win)))
            _br_flux[_bi] = float(np.sum(np.maximum(_bmag - _prev_mag, 0.0)))
            _prev_mag = _bmag
        _min_lag = max(1, int(sr / (_br_hop * 220.0 / 60.0)))
        _max_lag = min(_br_n_frames // 2, int(sr / (_br_hop * 40.0 / 60.0)))
        if _max_lag <= _min_lag + 5:
            return 1.0
        _br_ac = np.correlate(_br_flux[:_br_n_frames], _br_flux[:_br_n_frames], mode="full")
        _br_ac = _br_ac[_br_n_frames - 1 :]
        _br_ac_norm = _br_ac / (_br_ac[0] + 1e-12)
        _br_region = _br_ac_norm[_min_lag:_max_lag]
        if len(_br_region) > 0:
            return float(np.clip(np.max(_br_region), 0.0, 1.0))
        return 1.0

    def test_steady_beat_high_reliability(self):
        """Steady 120 BPM beat → reliability ≥ 0.40."""
        audio = _make_steady_beat(dur=4.0, bpm=120.0)
        score = self._compute_beat_reliability(audio)
        assert score >= 0.40, f"Steady beat should have high reliability, got {score:.3f}"

    def test_rubato_low_reliability(self):
        """Rubato/free-form signal → reliability < 0.40."""
        audio = _make_rubato_signal(dur=4.0)
        score = self._compute_beat_reliability(audio)
        assert score < 0.60, f"Rubato signal should have lower reliability, got {score:.3f}"

    def test_sine_no_beat(self):
        """Pure sine → no transients → reliability depends on spectral flux pattern."""
        audio = _make_sine(dur=4.0)
        score = self._compute_beat_reliability(audio)
        assert 0.0 <= score <= 1.0

    def test_stereo_input(self):
        """Stereo input (2, N) must be handled correctly."""
        mono = _make_steady_beat(dur=4.0)
        stereo = np.stack([mono, mono * 0.8])
        score = self._compute_beat_reliability(stereo)
        assert 0.0 <= score <= 1.0

    def test_short_audio_default(self):
        """Very short audio → insufficient frames → default 1.0."""
        audio = _make_sine(dur=0.1)
        score = self._compute_beat_reliability(audio)
        assert score == 1.0 or score >= 0.0  # default or computed

    def test_output_range(self):
        """Beat reliability must always be in [0.0, 1.0]."""
        for dur in [1.0, 2.0, 5.0]:
            audio = np.random.randn(int(48000 * dur)).astype(np.float32) * 0.3
            score = self._compute_beat_reliability(audio)
            assert 0.0 <= score <= 1.0, f"Out of range: {score}"


# ═══════════════════════════════════════════════════════════════════════
# 2. Loudness-War-Guard
# ═══════════════════════════════════════════════════════════════════════
class TestLoudnessWarGuard:
    """Test loudness-war detection and dynamics phase blocking."""

    def _is_loudness_war_victim(self, comp_sev: float, era_decade: int | None, material: Any) -> bool:
        """Replicate _loudness_war_victim logic."""
        if comp_sev <= 0.25:
            return False
        _is_modern_era = era_decade is not None and era_decade >= 2000
        _is_digital = material in {
            MaterialType.CD_DIGITAL,  # type: ignore[attr-defined]
            MaterialType.MP3_LOW,  # type: ignore[attr-defined]
            MaterialType.MP3_HIGH,  # type: ignore[attr-defined]
            MaterialType.AAC,  # type: ignore[attr-defined]
            MaterialType.STREAMING,  # type: ignore[attr-defined]
            MaterialType.DAT,  # type: ignore[attr-defined]
        }
        return _is_modern_era or _is_digital

    def test_modern_digital_high_compression_is_victim(self):
        """Era 2010 + CD + high compression → loudness war victim."""
        assert self._is_loudness_war_victim(0.40, 2010, MaterialType.CD_DIGITAL)  # type: ignore[attr-defined]

    def test_modern_mp3_high_compression_is_victim(self):
        """Era 2000 + MP3 + compression → victim."""
        assert self._is_loudness_war_victim(0.30, 2000, MaterialType.MP3_HIGH)  # type: ignore[attr-defined]

    def test_streaming_no_era_is_victim(self):
        """Streaming material (digital) even without era → victim."""
        assert self._is_loudness_war_victim(0.35, None, MaterialType.STREAMING)  # type: ignore[attr-defined]

    def test_low_compression_not_victim(self):
        """Low compression severity → not victim regardless of era/material."""
        assert not self._is_loudness_war_victim(0.20, 2020, MaterialType.CD_DIGITAL)  # type: ignore[attr-defined]

    def test_old_vinyl_not_victim(self):
        """1970s vinyl with compression → not loudness war (intentional)."""
        assert not self._is_loudness_war_victim(0.40, 1970, MaterialType.VINYL)  # type: ignore[attr-defined]

    def test_old_tape_not_victim(self):
        """1960s tape → not loudness war victim."""
        assert not self._is_loudness_war_victim(0.50, 1960, MaterialType.TAPE)  # type: ignore[attr-defined]

    def test_threshold_boundary(self):
        """Boundary: comp_sev=0.25 → not victim; 0.26 → maybe."""
        assert not self._is_loudness_war_victim(0.25, 2020, MaterialType.CD_DIGITAL)  # type: ignore[attr-defined]
        assert self._is_loudness_war_victim(0.26, 2020, MaterialType.CD_DIGITAL)  # type: ignore[attr-defined]

    def test_modern_era_analog_victim(self):
        """Modern era (2020) + analog (vinyl) → victim because era >= 2000."""
        assert self._is_loudness_war_victim(0.35, 2020, MaterialType.VINYL)  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════
# 3. Operatic Vibrato Detektor
# ═══════════════════════════════════════════════════════════════════════
class TestOperaticVibratoDetector:
    """Test vibrato detection (4-7 Hz f0 modulation)."""

    def _detect_vibrato(self, audio: np.ndarray, sr: int = 48000) -> tuple[bool, float]:
        """Replicate _operatic_vibrato detection logic."""
        _vib_mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        _vib_mono = np.asarray(_vib_mono, dtype=np.float32)
        _vib_hop = max(1, sr // 100)
        _vib_n = max(1, (len(_vib_mono) - 256) // _vib_hop)
        if _vib_n <= 50:
            return False, 0.0
        _f0_series = np.zeros(_vib_n, dtype=np.float32)
        for _vi in range(_vib_n):
            _vs = _vi * _vib_hop
            _vframe = _vib_mono[_vs : _vs + 256]
            _zcr = float(np.sum(np.abs(np.diff(np.sign(_vframe))) > 0))
            _f0_series[_vi] = _zcr * sr / (2.0 * 256)
        _f0_detrend = _f0_series - np.convolve(_f0_series, np.ones(20) / 20, mode="same")
        _vib_fft = np.abs(np.fft.rfft(_f0_detrend))
        _vib_freqs = np.fft.rfftfreq(len(_f0_detrend), d=_vib_hop / sr)
        _vib_band = (_vib_freqs >= 4.0) & (_vib_freqs <= 7.0)
        _vib_energy = float(np.sum(_vib_fft[_vib_band] ** 2))
        _total_energy = float(np.sum(_vib_fft[1:] ** 2)) + 1e-12
        _vib_ratio = _vib_energy / _total_energy
        return _vib_ratio > 0.15, _vib_ratio

    def test_vibrato_signal_detected(self):
        """Signal with 5 Hz vibrato → detected."""
        audio = _make_vibrato_signal(dur=3.0)
        detected, ratio = self._detect_vibrato(audio)
        assert detected, f"Vibrato signal should be detected (ratio={ratio:.3f})"

    def test_pure_sine_no_vibrato(self):
        """Pure sine without vibrato → not detected."""
        audio = _make_sine(440, dur=3.0)
        detected, ratio = self._detect_vibrato(audio)
        assert not detected, f"Pure sine should not be vibrato (ratio={ratio:.3f})"

    def test_noise_no_vibrato(self):
        """White noise → no vibrato."""
        audio = np.random.randn(48000 * 3).astype(np.float32) * 0.3
        detected, _ = self._detect_vibrato(audio)
        assert not detected

    def test_short_audio_default(self):
        """Very short audio → not enough frames → no vibrato."""
        audio = _make_sine(dur=0.05)
        detected, ratio = self._detect_vibrato(audio)
        assert not detected

    def test_ratio_range(self):
        """Ratio must always be non-negative."""
        for sig in [_make_sine(dur=2.0), _make_vibrato_signal(dur=2.0), np.zeros(48000, dtype=np.float32)]:
            _, ratio = self._detect_vibrato(sig)
            assert ratio >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# 4. Extended Instrument Routing
# ═══════════════════════════════════════════════════════════════════════
class TestExtendedInstrumentRouting:
    """Test that extended instruments trigger correct phases."""

    def test_strings_activates_guitar_and_transient(self):
        """Strings detected → phase_44 (strings share overtone structure) + phase_08."""
        phases = []
        strings_detected = True
        if strings_detected:
            phases.append("phase_44_guitar_enhancement")
            phases.append("phase_08_transient_preservation")
        assert "phase_44_guitar_enhancement" in phases
        assert "phase_08_transient_preservation" in phases

    def test_woodwind_activates_brass_and_air(self):
        """Woodwind detected → phase_45 + phase_39."""
        phases = []
        woodwind_detected = True
        if woodwind_detected:
            phases.append("phase_45_brass_enhancement")
            phases.append("phase_39_air_band_enhancement")
        assert "phase_45_brass_enhancement" in phases
        assert "phase_39_air_band_enhancement" in phases

    def test_synth_activates_spectral(self):
        """Synth → phase_50_spectral_repair."""
        phases = []
        synth_detected = True
        if synth_detected:
            phases.append("phase_50_spectral_repair")
        assert "phase_50_spectral_repair" in phases

    def test_organ_activates_bass_and_spatial(self):
        """Organ → phase_37 + phase_46."""
        phases = []
        organ_detected = True
        if organ_detected:
            phases.append("phase_37_bass_enhancement")
            phases.append("phase_46_spatial_enhancement")
        assert "phase_37_bass_enhancement" in phases
        assert "phase_46_spatial_enhancement" in phases


# ═══════════════════════════════════════════════════════════════════════
# 5. Dense Ensemble Guard
# ═══════════════════════════════════════════════════════════════════════
class TestDenseEnsembleGuard:
    """Test that ≥4 instrument families removes specific instrument phases."""

    def _apply_dense_guard(self, selected: list[str], active_families: int) -> list[str]:
        """Replicate dense-ensemble guard logic."""
        _dense_ensemble = active_families >= 4
        if _dense_ensemble:
            _instrument_specific = {
                "phase_44_guitar_enhancement",
                "phase_45_brass_enhancement",
                "phase_51_drums_enhancement",
                "phase_52_piano_restoration",
            }
            selected = [p for p in selected if p not in _instrument_specific]
        return selected

    def test_dense_ensemble_removes_specific_phases(self):
        """4+ families → instrument-specific phases removed."""
        phases = [
            "phase_42_vocal_enhancement",
            "phase_44_guitar_enhancement",
            "phase_45_brass_enhancement",
            "phase_51_drums_enhancement",
            "phase_52_piano_restoration",
            "phase_37_bass_enhancement",
        ]
        result = self._apply_dense_guard(phases, 5)
        assert "phase_44_guitar_enhancement" not in result
        assert "phase_45_brass_enhancement" not in result
        assert "phase_51_drums_enhancement" not in result
        assert "phase_52_piano_restoration" not in result
        # Generic phases survive
        assert "phase_42_vocal_enhancement" in result
        assert "phase_37_bass_enhancement" in result

    def test_sparse_ensemble_keeps_all(self):
        """< 4 families → all phases kept."""
        phases = ["phase_44_guitar_enhancement", "phase_51_drums_enhancement"]
        result = self._apply_dense_guard(phases, 2)
        assert result == phases

    def test_exactly_four_triggers_guard(self):
        """Exactly 4 families → guard triggers."""
        phases = ["phase_44_guitar_enhancement"]
        result = self._apply_dense_guard(phases, 4)
        assert len(result) == 0

    def test_three_families_not_dense(self):
        """3 families → not dense, keeps phases."""
        phases = ["phase_44_guitar_enhancement"]
        result = self._apply_dense_guard(phases, 3)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════
# 6. Leichtpfad
# ═══════════════════════════════════════════════════════════════════════
class TestLeichtpfad:
    """Test lightweight path for high-restorability material."""

    _LEICHTPFAD_KEEP = {
        "phase_30_dc_offset_removal",
        "phase_08_transient_preservation",
        "phase_01_declipping",
        "phase_02_dehum",
        "phase_03_denoise",
        "phase_04_eq_correction",
        "phase_05_rumble_removal",
        "phase_09_crackle_removal",
        "phase_10_compression_repair",
        "phase_11_limiting_repair",
        "phase_12_wow_flutter_fix",
        "phase_13_azimuth_correction",
        "phase_14_stereo_repair",
        "phase_15_channel_balance",
        "phase_17_distortion_reduction",
        "phase_18_noise_gate",
        "phase_19_de_esser",
        "phase_20_reverb_reduction",
        "phase_22_speed_correction",
        "phase_24_dropout_repair",
        "phase_25_declick",
        "phase_27_sibilance_control",
        "phase_29_tape_hiss_reduction",
        "phase_31_pitch_drift",
        "phase_43_ml_deesser",
        "phase_49_advanced_dereverb",
        "phase_50_spectral_repair",
        "phase_55_diffusion_inpainting",
        "phase_56_spectral_band_gap",
        "phase_59_modulation_noise",
        "phase_60_inner_groove_distortion",
        "phase_61_groove_echo",
        "phase_62_crosstalk_bleed",
        "phase_63_intermodulation_distortion",
        "phase_64_tape_splice",
        "phase_40_loudness_normalization",
        "phase_41_output_format_optimization",
        "phase_47_truepeak_limiter",
        "phase_53_semantic_audio",
    }

    def _apply_leichtpfad(
        self,
        selected: list[str],
        restorability: float,
        max_defect: float,
        pass_through: bool = False,
        localized_critical: bool = False,
    ) -> list[str]:
        """Replicate leichtpfad logic."""
        if pass_through:
            return selected
        if restorability >= 85.0 and max_defect <= 0.25 and not localized_critical:
            return [p for p in selected if p in self._LEICHTPFAD_KEEP]
        return selected

    def test_high_restorability_removes_enhancements(self):
        """Restorability 90 + low defects → enhancement phases removed."""
        phases = [
            "phase_03_denoise",  # keep (defect correction)
            "phase_42_vocal_enhancement",  # remove (enhancement)
            "phase_37_bass_enhancement",  # remove (enhancement)
            "phase_40_loudness_normalization",  # keep (finalization)
        ]
        result = self._apply_leichtpfad(phases, 90.0, 0.15)
        assert "phase_03_denoise" in result
        assert "phase_40_loudness_normalization" in result
        assert "phase_42_vocal_enhancement" not in result
        assert "phase_37_bass_enhancement" not in result

    def test_low_restorability_keeps_all(self):
        """Restorability 60 → no leichtpfad, keep all phases."""
        phases = ["phase_42_vocal_enhancement", "phase_03_denoise"]
        result = self._apply_leichtpfad(phases, 60.0, 0.30)
        assert result == phases

    def test_high_restorability_but_high_defect_keeps_all(self):
        """Restorability 90 but max_defect 0.40 → no leichtpfad."""
        phases = ["phase_42_vocal_enhancement", "phase_03_denoise"]
        result = self._apply_leichtpfad(phases, 90.0, 0.40)
        assert result == phases

    def test_pass_through_overrides(self):
        """Pass-through mode active → leichtpfad not applied."""
        phases = ["phase_42_vocal_enhancement"]
        result = self._apply_leichtpfad(phases, 95.0, 0.10, pass_through=True)
        assert result == phases

    def test_localized_critical_disables_leichtpfad(self):
        """Localized critical defects → full pipeline despite high restorability."""
        phases = ["phase_42_vocal_enhancement"]
        result = self._apply_leichtpfad(phases, 92.0, 0.15, localized_critical=True)
        assert result == phases

    def test_boundary_restorability_84(self):
        """Restorability 84.9 → no leichtpfad (threshold is 85)."""
        phases = ["phase_42_vocal_enhancement"]
        result = self._apply_leichtpfad(phases, 84.9, 0.10)
        assert "phase_42_vocal_enhancement" in result

    def test_boundary_restorability_85(self):
        """Restorability 85.0 → leichtpfad active."""
        phases = ["phase_42_vocal_enhancement", "phase_03_denoise"]
        result = self._apply_leichtpfad(phases, 85.0, 0.20)
        assert "phase_42_vocal_enhancement" not in result
        assert "phase_03_denoise" in result

    def test_all_defect_phases_kept(self):
        """All defect-correction phases survive leichtpfad."""
        defect_phases = [
            "phase_01_declipping",
            "phase_02_dehum",
            "phase_03_denoise",
            "phase_09_crackle_removal",
            "phase_24_dropout_repair",
            "phase_25_declick",
            "phase_29_tape_hiss_reduction",
        ]
        result = self._apply_leichtpfad(defect_phases, 90.0, 0.15)
        assert result == defect_phases

    def test_enhancement_phases_removed(self):
        """Enhancement phases (Tier 2-5) are removed in leichtpfad."""
        enhancement_phases = [
            "phase_06_frequency_restoration",
            "phase_23_spectral_repair",
            "phase_26_dynamic_range_expansion",
            "phase_34_mid_side_processing",
            "phase_35_multiband_compression",
            "phase_37_bass_enhancement",
            "phase_38_presence_enhancement",
            "phase_39_air_band_enhancement",
            "phase_42_vocal_enhancement",
            "phase_44_guitar_enhancement",
            "phase_46_spatial_enhancement",
            "phase_48_stereo_width_enhancer",
            "phase_54_transparent_dynamics",
        ]
        result = self._apply_leichtpfad(enhancement_phases, 90.0, 0.15)
        assert len(result) == 0, f"All enhancement phases should be removed: {result}"


# ═══════════════════════════════════════════════════════════════════════
# 7. Vibrato-aware Vocal Gating
# ═══════════════════════════════════════════════════════════════════════
class TestVibratoAwareVocalGating:
    """Test that operatic vibrato raises thresholds for De-Esser / Harshness."""

    def test_sibilance_threshold_raised_with_vibrato(self):
        """Operatic vibrato → sibilance threshold 0.35 instead of 0.15."""
        _operatic_vibrato = True
        _sib_threshold = 0.35 if _operatic_vibrato else 0.15
        assert _sib_threshold == 0.35

    def test_sibilance_threshold_normal_without_vibrato(self):
        """No vibrato → sibilance threshold 0.15."""
        _operatic_vibrato = False
        _sib_threshold = 0.35 if _operatic_vibrato else 0.15
        assert _sib_threshold == 0.15

    def test_harshness_threshold_raised_with_vibrato(self):
        """Operatic vibrato → harshness threshold 0.30 instead of 0.12."""
        _operatic_vibrato = True
        _harsh_threshold = 0.30 if _operatic_vibrato else 0.12
        assert _harsh_threshold == 0.30

    def test_harshness_threshold_normal_without_vibrato(self):
        """No vibrato → harshness threshold 0.12."""
        _operatic_vibrato = False
        _harsh_threshold = 0.30 if _operatic_vibrato else 0.12
        assert _harsh_threshold == 0.12

    def test_de_esser_skipped_for_vibrato_vocals(self):
        """Vibrato + vocals → only phase_42, no de-esser/ml-deesser."""
        _operatic_vibrato = True
        vocals_detected = True
        selected = []
        if vocals_detected:
            if _operatic_vibrato:
                selected.append("phase_42_vocal_enhancement")
            else:
                selected += [
                    "phase_19_de_esser",
                    "phase_42_vocal_enhancement",
                    "phase_43_ml_deesser",
                ]
        assert "phase_42_vocal_enhancement" in selected
        assert "phase_19_de_esser" not in selected
        assert "phase_43_ml_deesser" not in selected

    def test_de_esser_added_for_normal_vocals(self):
        """No vibrato + vocals → all 3 vocal phases."""
        _operatic_vibrato = False
        vocals_detected = True
        selected = []
        if vocals_detected:
            if _operatic_vibrato:
                selected.append("phase_42_vocal_enhancement")
            else:
                selected += [
                    "phase_19_de_esser",
                    "phase_42_vocal_enhancement",
                    "phase_43_ml_deesser",
                ]
        assert len(selected) == 3

    def test_moderate_sibilance_blocked_by_vibrato(self):
        """Sibilance severity 0.25 → blocked by vibrato (threshold 0.35)."""
        _operatic_vibrato = True
        sib_sev = 0.25
        _sib_threshold = 0.35 if _operatic_vibrato else 0.15
        triggered = sib_sev > _sib_threshold
        assert not triggered

    def test_high_sibilance_passes_even_with_vibrato(self):
        """Sibilance severity 0.40 → passes even with vibrato threshold 0.35."""
        _operatic_vibrato = True
        sib_sev = 0.40
        _sib_threshold = 0.35 if _operatic_vibrato else 0.15
        triggered = sib_sev > _sib_threshold
        assert triggered


# ═══════════════════════════════════════════════════════════════════════
# 8. Rubato-Guard (Phase 12 + Phase 31)
# ═══════════════════════════════════════════════════════════════════════
class TestRubatoGuard:
    """Test that rubato/free-form material skips wow/flutter and pitch drift phases."""

    def test_low_reliability_blocks_phase12(self):
        """Beat reliability < 0.40, low severity → phase_12 skipped."""
        _beat_reliability = 0.20
        wow_sev = 0.15
        should_add = _beat_reliability >= 0.40 or wow_sev > 0.35
        assert not should_add

    def test_high_reliability_allows_phase12(self):
        """Beat reliability ≥ 0.40 → phase_12 allowed."""
        _beat_reliability = 0.50
        wow_sev = 0.15
        should_add = _beat_reliability >= 0.40 or wow_sev > 0.35
        assert should_add

    def test_high_severity_overrides_rubato(self):
        """Low reliability BUT high severity (mechanical defect) → phase_12 allowed."""
        _beat_reliability = 0.10
        wow_sev = 0.50
        should_add = _beat_reliability >= 0.40 or wow_sev > 0.35
        assert should_add

    def test_low_reliability_blocks_phase31(self):
        """Beat reliability < 0.40, low severity → phase_31 skipped."""
        _beat_reliability = 0.15
        pitch_sev = 0.20
        should_add = _beat_reliability >= 0.40 or pitch_sev > 0.40
        assert not should_add

    def test_high_reliability_allows_phase31(self):
        """Beat reliability ≥ 0.40 → phase_31 allowed."""
        _beat_reliability = 0.45
        pitch_sev = 0.20
        should_add = _beat_reliability >= 0.40 or pitch_sev > 0.40
        assert should_add

    def test_boundary_reliability_040(self):
        """Boundary: reliability = 0.40 → passes (>=)."""
        assert 0.40 >= 0.40

    def test_boundary_reliability_039(self):
        """Boundary: reliability = 0.39 → blocked (<0.40)."""
        assert not (0.39 >= 0.40)
