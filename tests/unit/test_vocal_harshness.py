from __future__ import annotations

import pytest

"""Tests for VOCAL_HARSHNESS DefectType — detection, routing, and remediation (v10.0.0).

Validates:
1. DefectType.VOCAL_HARSHNESS exists and is properly integrated
2. _detect_vocal_harshness() detects harsh/distorted vocals correctly
3. Clean vocals do NOT trigger false positives
4. CausalDefectReasoner maps vocal_harshness to correct phases
5. Phase 42 applies harshness reduction when severity > 0
6. Phase 42 attenuates presence boost when harshness is high
7. Material-adaptive thresholds present for all MaterialTypes
"""


import numpy as np

SR = 48_000


def _scanner(sr: int = SR):
    from backend.core.defect_scanner import DefectScanner

    return DefectScanner(sample_rate=sr)


def _sine(freq: float = 440.0, amp: float = 0.5, duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _clean_vocal_sim(duration: float = 3.0) -> np.ndarray:
    """Simulated clean vocal: fundamental 220 Hz + formants at 800, 1200, 2500 Hz."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    sig = np.zeros_like(t, dtype=np.float32)
    # Fundamental
    sig += 0.30 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    # Formants
    sig += 0.15 * np.sin(2 * np.pi * 800 * t).astype(np.float32)
    sig += 0.10 * np.sin(2 * np.pi * 1200 * t).astype(np.float32)
    sig += 0.05 * np.sin(2 * np.pi * 2500 * t).astype(np.float32)
    # Some gentle breath noise above 8 kHz
    sig += 0.005 * np.random.randn(len(t)).astype(np.float32)
    return sig


def _harsh_vocal_sim(duration: float = 3.0, harshness: float = 0.8) -> np.ndarray:
    """Simulated harsh vocal: excessive 2–6 kHz energy, low crest factor, odd harmonics.

    Creates a signal with:
    - Strong presence band energy (3–5 kHz) dominating the spectrum
    - Low crest factor via hard clipping (simulating mic preamp saturation)
    - Odd harmonic distortion signature
    """
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    sig = np.zeros_like(t, dtype=np.float32)

    # Fundamental
    sig += 0.25 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    # Excessive presence band energy (2–6 kHz) — harsh zone
    for freq in [2500, 3000, 3500, 4000, 4500, 5000, 5500]:
        sig += (0.08 * harshness) * np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Odd harmonics (distortion signature)
    for k in [3, 5, 7, 9]:
        sig += (0.06 * harshness) * np.sin(2 * np.pi * 220 * k * t).astype(np.float32)

    # Simulate clipping/saturation (reduces crest factor)
    clip_level = 1.0 - 0.5 * harshness  # 0.6 for harshness=0.8
    sig = np.clip(sig, -clip_level, clip_level)
    # Normalize back
    sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.9

    return sig  # type: ignore[no-any-return]


# ============================================================
# DefectType existence and integration
# ============================================================


@pytest.mark.unit
class TestVocalHarshnessDefectType:
    """VOCAL_HARSHNESS must exist and be integrated into the system."""

    def test_defect_type_exists(self):
        from backend.core.defect_scanner import DefectType

        assert hasattr(DefectType, "VOCAL_HARSHNESS")
        assert DefectType.VOCAL_HARSHNESS.value == "vocal_harshness"

    def test_material_sensitivity_present_for_all(self):
        """All MaterialType entries must include VOCAL_HARSHNESS threshold."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        for mat in MaterialType:
            if mat == MaterialType.UNKNOWN:
                continue  # UNKNOWN uses dict.fromkeys
            thresholds = DefectScanner.MATERIAL_SENSITIVITY.get(mat, {})
            assert DefectType.VOCAL_HARSHNESS in thresholds, (
                f"MaterialType.{mat.name} missing VOCAL_HARSHNESS threshold"
            )
            val = thresholds[DefectType.VOCAL_HARSHNESS]
            assert 0.0 < val <= 1.0, f"MaterialType.{mat.name}: threshold {val} out of range"

    def test_cd_digital_threshold_low(self):
        """CD/Digital should have LOW threshold (0.25) — harsh mastering is common."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        val = DefectScanner.MATERIAL_SENSITIVITY[MaterialType.CD_DIGITAL][DefectType.VOCAL_HARSHNESS]
        assert val <= 0.30, f"CD_DIGITAL threshold {val} too high — harshness is common in digital"


# ============================================================
# Detection — Anti-False-Positive
# ============================================================


class TestVocalHarshnessAntiFP:
    """Clean signals must NOT trigger vocal harshness detection."""

    def test_pure_sine_no_harshness(self):
        sc = _scanner()
        score = sc._detect_vocal_harshness(_sine(440, 0.5))
        assert score.severity < 0.10

    def test_clean_vocal_no_harshness(self):
        sc = _scanner()
        score = sc._detect_vocal_harshness(_clean_vocal_sim())
        assert score.severity < 0.15, f"Clean vocal triggered harshness: {score.severity:.3f}"

    def test_silence_no_harshness(self):
        sc = _scanner()
        silence = np.zeros(SR * 2, dtype=np.float32)
        score = sc._detect_vocal_harshness(silence)
        assert score.severity == 0.0

    def test_short_audio_no_harshness(self):
        sc = _scanner()
        short = np.zeros(100, dtype=np.float32)
        score = sc._detect_vocal_harshness(short)
        assert score.severity == 0.0

    def test_low_level_signal_no_harshness(self):
        """Very quiet signal should not trigger."""
        sc = _scanner()
        score = sc._detect_vocal_harshness(_sine(440, 0.01))
        assert score.severity < 0.05


# ============================================================
# Detection — True Positives
# ============================================================


class TestVocalHarshnessDetection:
    """Harsh/distorted signals must trigger vocal harshness detection."""

    def test_harsh_vocal_detected(self):
        sc = _scanner()
        harsh = _harsh_vocal_sim(harshness=0.8)
        score = sc._detect_vocal_harshness(harsh)
        assert score.severity > 0.15, f"Harsh vocal not detected: severity={score.severity:.3f}"

    def test_very_harsh_vocal_high_severity(self):
        sc = _scanner()
        harsh = _harsh_vocal_sim(harshness=1.0)
        score = sc._detect_vocal_harshness(harsh)
        assert score.severity > 0.20, f"Very harsh vocal low severity: {score.severity:.3f}"

    def test_harsh_vs_clean_severity_ordering(self):
        """Harsh signal must have higher severity than clean signal."""
        sc = _scanner()
        clean_score = sc._detect_vocal_harshness(_clean_vocal_sim())
        harsh_score = sc._detect_vocal_harshness(_harsh_vocal_sim(harshness=0.9))
        assert harsh_score.severity > clean_score.severity

    def test_hard_clipped_signal_detected(self):
        """Hard-clipped signal (low crest factor) should detect harshness."""
        sc = _scanner()
        # Use 440 Hz: clipping harmonics land at 1320, 2200, 3080, 3960, 4840 Hz
        # — ample coverage in the 2–6 kHz presence band used by the detector.
        sig = _sine(440, 0.9, 3.0)
        # Hard clip at 0.4 (severe distortion → many odd harmonics)
        sig = np.clip(sig, -0.4, 0.4)
        sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.9
        score = sc._detect_vocal_harshness(sig)
        assert score.severity > 0.10, f"Hard-clipped signal not detected: {score.severity:.3f}"

    def test_metadata_contains_indicators(self):
        sc = _scanner()
        score = sc._detect_vocal_harshness(_harsh_vocal_sim())
        assert "crest_factor_db" in score.metadata
        assert "flux_score" in score.metadata
        assert "presence_ratio_score" in score.metadata
        assert "presence_concentration_score" in score.metadata

    def test_locations_present_when_detected(self):
        sc = _scanner()
        score = sc._detect_vocal_harshness(_harsh_vocal_sim(duration=5.0))
        if score.severity > 0.1:
            # Should have at least some location events
            assert isinstance(score.locations, list)

    def test_defect_type_correct(self):
        from backend.core.defect_scanner import DefectType

        sc = _scanner()
        score = sc._detect_vocal_harshness(_harsh_vocal_sim())
        assert score.defect_type == DefectType.VOCAL_HARSHNESS


# ============================================================
# CausalDefectReasoner mapping
# ============================================================


class TestVocalHarshnessReasoning:
    """CausalDefectReasoner must map vocal_harshness to repair phases."""

    def test_cause_to_phases_mapping_exists(self):
        from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES

        assert "vocal_harshness" in CAUSE_TO_PHASES

    def test_mapping_excludes_phase_42(self):
        """§0a BUG-FIX v10.0.0: phase_42 darf nie in CAUSE_TO_PHASES stehen."""
        from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES

        phases = CAUSE_TO_PHASES["vocal_harshness"]
        assert "phase_42_vocal_enhancement" not in phases

    def test_mapping_includes_de_esser(self):
        from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES

        phases = CAUSE_TO_PHASES["vocal_harshness"]
        assert "phase_19_de_esser" in phases

    def test_mapping_includes_spectral_repair(self):
        from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES

        phases = CAUSE_TO_PHASES["vocal_harshness"]
        assert "phase_23_spectral_repair" in phases

    def test_reasoner_does_not_recommend_phase42_for_harshness(self):
        """§0a BUG-FIX v10.0.0: CausalDefectReasoner darf phase_42 nie vorschlagen."""
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        reasoner = CausalDefectReasoner()
        plan = reasoner.reason({"vocal_harshness": 1.0}, material="cd_digital")
        assert "phase_42_vocal_enhancement" not in plan.recommended_phases


# ============================================================
# Phase 42 — Harshness reduction integration
# ============================================================


class TestPhase42HarshnessReduction:
    """Phase 42 must reduce harshness when VOCAL_HARSHNESS is detected."""

    def _make_phase42(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        return VocalEnhancement()

    def test_reduce_harshness_method_exists(self):
        p42 = self._make_phase42()
        assert hasattr(p42, "_reduce_harshness")

    def test_harshness_reduction_attenuates_presence(self):
        """With high harshness severity, 2–6 kHz energy should decrease."""
        from scipy import signal as scipy_signal

        p42 = self._make_phase42()
        audio = _harsh_vocal_sim(harshness=0.8)

        # Measure 2–6 kHz energy before
        sos = scipy_signal.butter(4, [2000, 6000], btype="band", fs=SR, output="sos")
        pres_before = scipy_signal.sosfilt(sos, audio)
        energy_before = float(np.mean(pres_before**2))

        # Apply reduction
        reduced = p42._reduce_harshness(audio, SR, severity=0.7)

        pres_after = scipy_signal.sosfilt(sos, reduced)
        energy_after = float(np.mean(pres_after**2))

        assert energy_after < energy_before, (
            f"Harshness reduction failed: energy {energy_before:.6f} → {energy_after:.6f}"
        )

    def test_harshness_reduction_no_nan_inf(self):
        p42 = self._make_phase42()
        audio = _harsh_vocal_sim()
        reduced = p42._reduce_harshness(audio, SR, severity=0.5)
        assert np.isfinite(reduced).all()

    def test_harshness_reduction_in_range(self):
        p42 = self._make_phase42()
        audio = _harsh_vocal_sim()
        reduced = p42._reduce_harshness(audio, SR, severity=0.8)
        assert np.max(np.abs(reduced)) <= 1.0 + 1e-6

    def test_harshness_reduction_preserves_length(self):
        p42 = self._make_phase42()
        audio = _harsh_vocal_sim()
        reduced = p42._reduce_harshness(audio, SR, severity=0.5)
        assert len(reduced) == len(audio)

    def test_clean_audio_minimal_change(self):
        """Clean audio should barely be affected by harshness reduction."""
        p42 = self._make_phase42()
        audio = _clean_vocal_sim()
        reduced = p42._reduce_harshness(audio, SR, severity=0.3)
        # Difference should be small for clean audio (median presence not exceeding threshold)
        diff = float(np.sqrt(np.mean((audio - reduced) ** 2)))
        assert diff < 0.1, f"Clean audio changed too much: rmsd={diff:.4f}"

    def test_enhance_channel_uses_harshness(self):
        """_enhance_channel with harshness > 0 should invoke reduction."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        p42 = VocalEnhancement()
        config = p42.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL]

        audio = _harsh_vocal_sim(harshness=0.8)

        # Without harshness
        result_no_harsh = p42._enhance_channel(audio.copy(), SR, config, harshness_severity=0.0)
        # With harshness
        result_harsh = p42._enhance_channel(audio.copy(), SR, config, harshness_severity=0.7)

        # Results should differ
        diff = float(np.sqrt(np.mean((result_no_harsh - result_harsh) ** 2)))
        assert diff > 0.001, f"Harshness severity had no effect: diff={diff:.6f}"

    def test_presence_boost_attenuated_with_high_harshness(self):
        """With harshness > 0.3, presence_gain_db should be reduced."""
        from scipy import signal as scipy_signal

        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        p42 = VocalEnhancement()
        config = p42.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL]
        audio = _clean_vocal_sim()

        # Enhance with harshness=0 (full presence boost)
        result_full = p42._enhance_channel(audio.copy(), SR, config, harshness_severity=0.0)
        # Enhance with harshness=0.8 (presence should be attenuated)
        result_att = p42._enhance_channel(audio.copy(), SR, config, harshness_severity=0.8)

        # Measure 3–6 kHz energy (presence boost zone)
        sos = scipy_signal.butter(4, [3000, 6000], btype="band", fs=SR, output="sos")
        pres_full = float(np.mean(scipy_signal.sosfilt(sos, result_full) ** 2))
        pres_att = float(np.mean(scipy_signal.sosfilt(sos, result_att) ** 2))

        # Attenuated version should have less presence energy
        assert pres_att < pres_full, f"Presence not attenuated: full={pres_full:.6f} att={pres_att:.6f}"


# ============================================================
# Full scan integration
# ============================================================


class TestVocalHarshnessScanIntegration:
    """VOCAL_HARSHNESS in full DefectScanner.scan() results."""

    def test_scan_includes_vocal_harshness(self):
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        audio = _harsh_vocal_sim(duration=5.0)
        result = sc.scan(audio, material_type=MaterialType.CD_DIGITAL)
        assert DefectType.VOCAL_HARSHNESS in result.scores

    def test_scan_clean_audio_low_severity(self):
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        audio = _clean_vocal_sim(duration=5.0)
        result = sc.scan(audio, material_type=MaterialType.CD_DIGITAL)
        assert DefectType.VOCAL_HARSHNESS in result.scores
        assert result.scores[DefectType.VOCAL_HARSHNESS].severity < 0.20

    def test_scan_stereo_includes_vocal_harshness(self):
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        mono = _harsh_vocal_sim(duration=5.0)
        stereo = np.column_stack([mono, mono])
        result = sc.scan(stereo, material_type=MaterialType.CD_DIGITAL)
        assert DefectType.VOCAL_HARSHNESS in result.scores
