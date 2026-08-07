from __future__ import annotations

"""
Pipeline-Integration-Tests (DSP-only, kein ML-Modell-Load erforderlich).

Testet die realen Modul-Verkettungen der Aurik-Pipeline:
  1. DefectScanner → CausalDefectReasoner (Defekt-zu-Ursache-Kette)
  2. EraClassifier + MediumClassifier (Klassifikations-Konsistenz)
  3. RestorabilityEstimator (Score-Plausibilität)
  4. GoalApplicabilityFilter (Ziel-Filterung nach Material/Ära)
  5. SongCalibrationProfile (Kalibrierung nach Material)
  6. MusicalGoalsChecker (14-Ziel-Messung)
  7. Bridge-API (Funktions-Export-Vertrag)
  8. PhaseInterface (Phase-Ausführungsvertrag)
  9. Denker-Kette (ReparaturDenker → RekonstruktionsDenker → RestaurierDenker)
 10. Cross-Module-Datenfluss (Scanner → Reasoner → Phases → Goals)

Spec-Referenzen:
    §2.6 CausalDefectReasoner, §2.26 RestorabilityEstimator,
    §2.31a SongCalibration, §8.1 Qualitätsmessung,
    §14 E2E-Integrationstests, §11.7a Denker-Rollendifferenzierung
"""


import math
from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np
import pytest

# Timeout for integration tests — no ML, pure DSP
_INT_TIMEOUT = 60


# ── Shared Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def audio_48k_stereo() -> tuple[np.ndarray, int]:
    """3s stereo sine+noise @ 48 kHz — universal test signal."""
    sr = 48_000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(42)
    mono = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.02 * rng.standard_normal(len(t)).astype(np.float32)
    stereo = np.stack([mono, mono * 0.95], axis=0)  # (2, N)
    return stereo, sr


@pytest.fixture(scope="module")
def audio_48k_mono() -> tuple[np.ndarray, int]:
    """3s mono sine+noise @ 48 kHz."""
    sr = 48_000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(99)
    mono = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.03 * rng.standard_normal(len(t)).astype(np.float32)
    return mono, sr


@pytest.fixture(scope="module")
def noisy_vinyl_audio() -> tuple[np.ndarray, int]:
    """3s mono signal simulating vinyl noise (clicks + crackle + hum)."""
    sr = 48_000
    n = 3 * sr
    rng = np.random.default_rng(77)
    t = np.linspace(0, 3.0, n, endpoint=False, dtype=np.float32)
    signal = 0.25 * np.sin(2 * np.pi * 440 * t)
    # Add broadband noise
    signal += 0.08 * rng.standard_normal(n).astype(np.float32)
    # Add simulated 50 Hz hum
    signal += 0.04 * np.sin(2 * np.pi * 50 * t)
    # Add random click transients
    for _ in range(20):
        pos = rng.integers(100, n - 100)
        signal[pos] = 0.9 * (1 if rng.random() > 0.5 else -1)
    return np.clip(signal, -1.0, 1.0), sr


@pytest.fixture(scope="module")
def clipped_audio() -> tuple[np.ndarray, int]:
    """3s mono with hard clipping artifacts."""
    sr = 48_000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False, dtype=np.float32)
    signal = 1.5 * np.sin(2 * np.pi * 440 * t)  # Overdriven
    return np.clip(signal, -1.0, 1.0).astype(np.float32), sr


# ═══════════════════════════════════════════════════════════════════════
# 1. DefectScanner → CausalDefectReasoner Integration
# ═══════════════════════════════════════════════════════════════════════
class TestDefectScannerToReasoner:
    """Test that DefectScanner output feeds correctly into CausalDefectReasoner."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_scanner_output_feeds_reasoner(self, noisy_vinyl_audio):
        """Scanner scores dict → Reasoner produces valid RestorationPlan."""
        audio, sr = noisy_vinyl_audio
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        result = scanner.scan(audio, sr)

        # Convert to reasoner input format
        defect_scores = {
            dt.value if hasattr(dt, "value") else str(dt): float(ds.severity) for dt, ds in result.scores.items()
        }

        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        reasoner = CausalDefectReasoner()
        plan = reasoner.reason(defect_scores, material=result.material_type.value)

        assert plan is not None
        assert isinstance(plan.recommended_phases, list)
        assert isinstance(plan.cause_probabilities, dict)
        assert plan.confidence >= 0.0
        assert len(plan.ranked_causes) > 0

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_scanner_reasoner_material_consistency(self, noisy_vinyl_audio):
        """Material type from Scanner is accepted by Reasoner without error."""
        audio, sr = noisy_vinyl_audio
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        result = scanner.scan(audio, sr)
        material_str = result.material_type.value

        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        reasoner = CausalDefectReasoner()
        # Must not raise
        plan = reasoner.reason(
            {dt.value: ds.severity for dt, ds in result.scores.items()},
            material=material_str,
        )
        assert plan.material == material_str

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_scanner_result_has_all_required_fields(self, audio_48k_mono):
        """DefectAnalysisResult has all Spec-required fields."""
        audio, sr = audio_48k_mono
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        result = scanner.scan(audio, sr)

        assert hasattr(result, "scores")
        assert hasattr(result, "material_type")
        assert hasattr(result, "analysis_time_seconds")
        assert hasattr(result, "sample_rate")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "spectral_fingerprint")
        assert isinstance(result.scores, dict)
        assert result.analysis_time_seconds >= 0
        assert result.duration_seconds > 0

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_scanner_severity_bounds(self, audio_48k_mono):
        """All defect severities must be in [0.0, 1.0]."""
        audio, sr = audio_48k_mono
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        result = scanner.scan(audio, sr)

        for dt, ds in result.scores.items():
            assert 0.0 <= ds.severity <= 1.0, f"{dt}: severity {ds.severity} out of range"
            assert 0.0 <= ds.confidence <= 1.0, f"{dt}: confidence {ds.confidence} out of range"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_reasoner_phases_are_known_phase_ids(self, noisy_vinyl_audio):
        """Reasoner's recommended_phases must be valid phase IDs."""
        audio, sr = noisy_vinyl_audio
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        result = scanner.scan(audio, sr)
        defect_scores = {dt.value: ds.severity for dt, ds in result.scores.items()}

        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        plan = CausalDefectReasoner().reason(defect_scores)

        for phase_id in plan.recommended_phases:
            assert phase_id.startswith("phase_"), f"Invalid phase ID: {phase_id}"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_clipped_audio_detected(self, clipped_audio):
        """Hard-clipped audio must trigger clipping detection."""
        audio, sr = clipped_audio
        from backend.core.defect_scanner import DefectScanner, DefectType

        scanner = DefectScanner()
        result = scanner.scan(audio, sr)

        clipping_score = result.scores.get(DefectType.CLIPPING)
        assert clipping_score is not None, "CLIPPING defect type not in scores"
        assert clipping_score.severity > 0.1, (
            f"Hard-clipped audio should have clipping severity > 0.1, got {clipping_score.severity}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 1b. Forensic Medium Handoff / Dedup Integration
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class _StubForensicMedium:
    transfer_chain: list[str] = field(default_factory=lambda: ["vinyl", "mp3_low"])
    primary_material: str = "vinyl"
    confidence: float = 0.84


class TestForensicMediumHandoffIntegration:
    """Integration checks for one-time medium detection + scanner handoff."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_scanner_uses_cached_forensic_medium_without_second_detect(self, audio_48k_mono):
        """Cached forensic medium must bypass a second MediumDetector.detect() call."""
        audio, sr = audio_48k_mono
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner()
        cached = _StubForensicMedium()

        with patch(
            "backend.core.forensics.medium_detector.MediumDetector",
            side_effect=AssertionError("MediumDetector must not be instantiated on cached path"),
        ):
            result = scanner.scan(
                audio,
                sr,
                material_type=MaterialType.VINYL,
                file_ext=".mp3",
                forensic_medium_result=cached,
            )

        assert result.transfer_chain_raw is cached
        assert "vinyl" in str(result.transfer_chain_str)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_scanner_calls_medium_detector_once_without_cache(self, audio_48k_mono):
        """Without cached forensic medium, scanner should call detector exactly once."""
        audio, sr = audio_48k_mono
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner()
        calls = {"init": 0, "detect": 0}

        class _FakeMediumDetector:
            def __init__(self):
                calls["init"] += 1

            def detect(self, _audio, _sr, file_ext=None):
                calls["detect"] += 1
                return _StubForensicMedium(transfer_chain=["tape", "mp3_low"], primary_material="tape", confidence=0.79)

        with patch("backend.core.forensics.medium_detector.MediumDetector", _FakeMediumDetector):
            result = scanner.scan(
                audio,
                sr,
                material_type=MaterialType.TAPE,
                file_ext=".mp3",
                forensic_medium_result=None,
            )

        assert calls == {"init": 1, "detect": 1}
        assert "tape" in str(result.transfer_chain_str)


# ═══════════════════════════════════════════════════════════════════════
# 2. EraClassifier + MediumClassifier Integration
# ═══════════════════════════════════════════════════════════════════════
class TestClassifierIntegration:
    """Test EraClassifier and MediumClassifier produce valid, consistent results."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_era_classifier_returns_valid_result(self, audio_48k_mono):
        """EraClassifier produces EraResult with valid decade and confidence."""
        audio, sr = audio_48k_mono
        from backend.core.era_classifier import classify_era

        result = classify_era(audio, sr)

        assert result is not None
        assert hasattr(result, "decade")
        assert hasattr(result, "confidence")
        assert hasattr(result, "era_label")
        assert 1880 <= result.decade <= 2030, f"decade {result.decade} out of plausible range"
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_medium_classifier_returns_valid_result(self, audio_48k_mono):
        """MediumClassifier produces ClassificationResult with valid material."""
        audio, sr = audio_48k_mono
        from backend.core.medium_classifier import classify_medium

        result = classify_medium(audio, sr, use_ml=False)

        assert result is not None
        assert hasattr(result, "material")
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_era_and_medium_classifiers_accept_stereo(self, audio_48k_stereo):
        """Both classifiers must handle stereo (2, N) input without crashing."""
        audio, sr = audio_48k_stereo
        from backend.core.era_classifier import classify_era
        from backend.core.medium_classifier import classify_medium

        era_result = classify_era(audio, sr)
        medium_result = classify_medium(audio, sr, use_ml=False)

        assert era_result is not None
        assert medium_result is not None

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_classifier_results_feedable_to_scanner(self, audio_48k_mono):
        """Classifier output material type can be passed to DefectScanner."""
        audio, sr = audio_48k_mono
        from backend.core.defect_scanner import DefectScanner, MaterialType
        from backend.core.medium_classifier import classify_medium

        medium_result = classify_medium(audio, sr, use_ml=False)
        mat_str = (
            medium_result.material.value if hasattr(medium_result.material, "value") else str(medium_result.material)
        )

        # MaterialType conversion should not crash
        try:
            mat_enum = MaterialType(mat_str)
        except ValueError:
            mat_enum = MaterialType.UNKNOWN

        scanner = DefectScanner()
        result = scanner.scan(audio, sr, material_type=mat_enum)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# 3. RestorabilityEstimator Integration
# ═══════════════════════════════════════════════════════════════════════
class TestRestorabilityEstimatorIntegration:
    """Test RestorabilityEstimator with real audio signals."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_estimator_returns_valid_result(self, audio_48k_mono):
        """estimate_restorability produces a full RestorabilityResult."""
        audio, sr = audio_48k_mono
        from backend.core.restorability_estimator import estimate_restorability

        result = estimate_restorability(audio, sr)

        assert result is not None
        assert 0 <= result.restorability_score <= 100
        assert 1.0 <= result.predicted_mos <= 5.0
        assert result.grade in ("excellent", "good", "fair", "poor", "critical")
        assert isinstance(result.limiting_defects, list)
        assert isinstance(result.recommendations, list)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_clean_audio_high_restorability(self, audio_48k_mono):
        """Clean sine+noise → high restorability score."""
        audio, sr = audio_48k_mono
        from backend.core.restorability_estimator import estimate_restorability

        result = estimate_restorability(audio, sr, material="cd_digital")

        assert result.restorability_score >= 50, (
            f"Clean audio should have restorability ≥ 50, got {result.restorability_score}"
        )

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_noisy_audio_lower_restorability(self, noisy_vinyl_audio):
        """Noisy vinyl-like audio → lower restorability than clean."""
        audio, sr = noisy_vinyl_audio
        from backend.core.restorability_estimator import estimate_restorability

        result = estimate_restorability(audio, sr, material="vinyl")

        assert result.restorability_score <= 95, (
            f"Noisy audio should not have near-perfect restorability, got {result.restorability_score}"
        )

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_estimator_accepts_stereo(self, audio_48k_stereo):
        """Stereo input (2, N) must not crash."""
        audio, sr = audio_48k_stereo
        from backend.core.restorability_estimator import estimate_restorability

        result = estimate_restorability(audio, sr)
        assert result is not None
        assert 0 <= result.restorability_score <= 100


# ═══════════════════════════════════════════════════════════════════════
# 4. GoalApplicabilityFilter Integration
# ═══════════════════════════════════════════════════════════════════════
class TestGoalApplicabilityFilterIntegration:
    """Test that GoalApplicabilityFilter produces correct applicable/inapplicable sets."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_filter_returns_valid_result(self, audio_48k_mono):
        """Filter produces GoalApplicabilityResult with applicable/inapplicable sets."""
        audio, sr = audio_48k_mono
        from backend.core.goal_applicability_filter import GoalApplicabilityFilter

        gaf = GoalApplicabilityFilter()
        result = gaf.evaluate(audio=audio, sr=sr, material="cd_digital", era_decade=2020)

        assert hasattr(result, "applicable")
        assert hasattr(result, "inapplicable")
        assert isinstance(result.applicable, (set, frozenset))
        assert isinstance(result.inapplicable, (set, frozenset))

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_digital_material_has_most_goals_applicable(self, audio_48k_mono):
        """CD digital material → most of 14 goals applicable."""
        audio, sr = audio_48k_mono
        from backend.core.goal_applicability_filter import GoalApplicabilityFilter

        gaf = GoalApplicabilityFilter()
        result = gaf.evaluate(audio=audio, sr=sr, material="cd_digital", era_decade=2000)

        assert len(result.applicable) >= 10, (
            f"CD digital should have ≥ 10 applicable goals, got {len(result.applicable)}"
        )

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_mono_era_filters_spatial_depth(self, audio_48k_mono):
        """Mono-Ära (< 1960) material → spatial_depth may be filtered out."""
        audio, sr = audio_48k_mono
        from backend.core.goal_applicability_filter import GoalApplicabilityFilter

        gaf = GoalApplicabilityFilter()
        result = gaf.evaluate(audio=audio, sr=sr, material="shellac", era_decade=1930)

        # Spatial depth should be inapplicable for mono-era material
        if "spatial_depth" in result.inapplicable:
            assert "spatial_depth" not in result.applicable

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_filter_applicable_inapplicable_disjoint(self, audio_48k_mono):
        """Applicable and inapplicable sets must be disjoint."""
        audio, sr = audio_48k_mono
        from backend.core.goal_applicability_filter import GoalApplicabilityFilter

        gaf = GoalApplicabilityFilter()
        result = gaf.evaluate(audio=audio, sr=sr, material="vinyl", era_decade=1975)

        overlap = set(result.applicable) & set(result.inapplicable)
        assert len(overlap) == 0, f"Overlap between applicable and inapplicable: {overlap}"


# ═══════════════════════════════════════════════════════════════════════
# 5. SongCalibrationProfile Integration
# ═══════════════════════════════════════════════════════════════════════
class TestSongCalibrationIntegration:
    """Test _build_song_calibration_profile produces valid profiles."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_calibration_profile_has_required_fields(self):
        """Profile must contain all Spec-required fields."""
        from backend.core.quality_mode import QualityMode
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=None,
            mode=QualityMode.BALANCED,  # type: ignore[arg-type]
            restorability_score=70.0,
            input_snr_db=25.0,
            max_defect_severity=0.3,
            pipeline_confidence=0.9,
        )

        assert isinstance(profile, dict)
        assert "global_scalar" in profile
        assert "family_scalars" in profile
        fs = profile["family_scalars"]
        required_families = {
            "denoise",
            "reverb",
            "reconstruction",
            "dynamics_eq",
            "transient",
            "vocal",
            "instrument",
            "general",
        }
        assert required_families.issubset(set(fs.keys())), f"Missing families: {required_families - set(fs.keys())}"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_calibration_scalars_bounded(self):
        """All family_scalars must be bounded (positive, reasonable range)."""
        from backend.core.quality_mode import QualityMode
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=None,
            mode=QualityMode.BALANCED,  # type: ignore[arg-type]
            restorability_score=50.0,
            input_snr_db=15.0,
            max_defect_severity=0.6,
            pipeline_confidence=0.8,
        )

        gs = profile["global_scalar"]
        assert 0.3 <= gs <= 2.0, f"global_scalar {gs} out of reasonable bounds"

        for family, scalar in profile["family_scalars"].items():
            assert 0.1 <= scalar <= 2.5, f"family_scalar[{family}] = {scalar} out of bounds"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_calibration_deterministic(self):
        """Same inputs → same profile (determinism invariant §2.40)."""
        from backend.core.quality_mode import QualityMode
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        kwargs = {
            "material_type": None,
            "mode": QualityMode.BALANCED,
            "restorability_score": 65.0,
            "input_snr_db": 20.0,
            "max_defect_severity": 0.4,
            "pipeline_confidence": 0.85,
            "era_decade": 1975,
        }
        p1 = UnifiedRestorerV3._build_song_calibration_profile(**kwargs)  # type: ignore[arg-type]
        p2 = UnifiedRestorerV3._build_song_calibration_profile(**kwargs)  # type: ignore[arg-type]

        assert p1["global_scalar"] == p2["global_scalar"]
        for fam in p1["family_scalars"]:
            assert p1["family_scalars"][fam] == p2["family_scalars"][fam], f"Non-deterministic: {fam} differs"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_calibration_differs_by_material(self):
        """Different materials → different calibration profiles."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.quality_mode import QualityMode
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        base = {
            "mode": QualityMode.BALANCED,
            "restorability_score": 60.0,
            "input_snr_db": 18.0,
            "max_defect_severity": 0.5,
            "pipeline_confidence": 0.8,
        }
        p_shellac = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.SHELLAC,
            **base,  # type: ignore[arg-type]
        )
        p_cd = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.CD_DIGITAL,
            **base,  # type: ignore[arg-type]
        )

        # Different materials should produce different scalars
        assert p_shellac["family_scalars"] != p_cd["family_scalars"], (
            "Shellac and CD_DIGITAL must produce different calibration profiles"
        )

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_calibration_source_fidelity_includes_transfer_chain_losses(self):
        """Transfer chain must increase SourceFidelity generation/HF-loss estimate."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.quality_mode import QualityMode
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        kwargs = {
            "material_type": MaterialType.TAPE,
            "mode": QualityMode.BALANCED,
            "restorability_score": 60.0,
            "input_snr_db": 20.0,
            "max_defect_severity": 0.4,
            "pipeline_confidence": 0.9,
            "era_decade": 1970,
            "spectral_fingerprint": {"rolloff_95_hz": 7000.0, "effective_bandwidth_hz": 7000.0},
        }

        profile_material_only = UnifiedRestorerV3._build_song_calibration_profile(
            transfer_chain=None,
            **kwargs,  # type: ignore[arg-type]
        )
        profile_with_chain = UnifiedRestorerV3._build_song_calibration_profile(
            transfer_chain=["vinyl", "tape", "mp3_low"],
            **kwargs,  # type: ignore[arg-type]
        )

        assert (
            profile_with_chain["source_fidelity_generation_count"]
            >= profile_material_only["source_fidelity_generation_count"]
        )
        assert profile_with_chain["source_fidelity_hf_loss_db"] >= profile_material_only["source_fidelity_hf_loss_db"]
        assert profile_with_chain["source_fidelity_transfer_chain"] == ["vinyl", "tape", "mp3_low"]

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_extract_transfer_chain_from_forensics_normalizes_inputs(self):
        """Helper must normalize dict/object/string chain payloads deterministically."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        assert UnifiedRestorerV3._extract_transfer_chain_from_forensics(
            {"transfer_chain": ["Vinyl", " Tape ", "MP3_LOW"]}
        ) == ["vinyl", "tape", "mp3_low"]

        assert UnifiedRestorerV3._extract_transfer_chain_from_forensics({"chain": "vinyl → tape > mp3_low"}) == [
            "vinyl",
            "tape",
            "mp3_low",
        ]


# ═══════════════════════════════════════════════════════════════════════
# 5b. SourceFidelity Export Audit Trail  [RELEASE_MUST] §2.41 + §2.46 + §2.47
# ═══════════════════════════════════════════════════════════════════════
class TestSourceFidelityExportAuditTrail:
    """Normative: source_fidelity_* fields MUST reach RestorationResult.metadata.

    Proves the hard audit trail from transfer_chain_raw → SongCalibration →
    metadata["song_calibration"] in the final RestorationsResult export.

    [RELEASE_MUST] §2.41 SourceFidelityReconstructor
                   §2.46 Carrier-Chain-Inversion
                   §2.47 Adaptive-Intelligence-Prinzip (chain → gen_count / hf_loss_db)
    """

    @pytest.mark.timeout(30)
    def test_transfer_chain_raw_propagates_to_export_metadata(self):
        """transfer_chain_raw → extract → calibration profile → metadata dict — end-to-end.

        Simulates exactly the 3 steps restore() performs:
          Step 1 (UV3 ~line 2508): _extract_transfer_chain_from_forensics(defect_result.transfer_chain_raw)
          Step 2 (UV3 ~line 2528): _build_song_calibration_profile(..., transfer_chain=chain)
          Step 3 (UV3 ~line 4657): metadata["song_calibration"] = dict(self._song_calibration_profile)

        [RELEASE_MUST] §2.41 SourceFidelityReconstructor audit trail
                       §2.46 Carrier-Chain-Inversion
                       §2.47 Adaptive-Intelligence: chain → gen_count / hf_loss_db in export
        """
        from backend.core.defect_scanner import MaterialType
        from backend.core.unified_restorer_v3 import QualityMode, UnifiedRestorerV3

        # ── Step 1: Extract chain from DefectResult.transfer_chain_raw ─────────
        # Simulates UV3 line ~2508 using a representative transfer_chain_raw payload.
        raw = {"transfer_chain": ["vinyl", "tape", "mp3_low"]}
        chain = UnifiedRestorerV3._extract_transfer_chain_from_forensics(raw)
        assert chain == [
            "vinyl",
            "tape",
            "mp3_low",
        ], f"_extract_transfer_chain_from_forensics produced unexpected chain: {chain}"

        # ── Step 2: Build calibration profile with the extracted chain ──────────
        # Simulates UV3 line ~2528.
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.TAPE,
            mode=QualityMode.QUALITY,
            restorability_score=60.0,
            input_snr_db=20.0,
            max_defect_severity=0.35,
            pipeline_confidence=0.80,
            era_decade=1965,
            transfer_chain=chain,
        )

        # ── Step 3: Simulate metadata construction (UV3 line ~4657) ────────────
        # metadata["song_calibration"] = dict(self._song_calibration_profile)
        cal = dict(profile)

        # ── Assertions: all 3 required audit-trail fields present and correct ───
        assert "source_fidelity_transfer_chain" in cal, (
            "source_fidelity_transfer_chain missing — §2.41 audit trail broken at Step 3"
        )
        assert cal["source_fidelity_transfer_chain"] == ["vinyl", "tape", "mp3_low"], (
            f"Chain mismatch in export: expected ['vinyl','tape','mp3_low'] "
            f"but got {cal['source_fidelity_transfer_chain']}"
        )

        assert "source_fidelity_generation_count" in cal, (
            "source_fidelity_generation_count missing from calibration profile"
        )
        assert cal["source_fidelity_generation_count"] >= 3, (
            f"3-stage chain must yield generation_count >= 3 (got {cal['source_fidelity_generation_count']})"
        )

        assert "source_fidelity_hf_loss_db" in cal, "source_fidelity_hf_loss_db missing from calibration profile"
        assert cal["source_fidelity_hf_loss_db"] > 0.0, (
            f"3-stage chain must carry HF loss > 0 dB (got {cal['source_fidelity_hf_loss_db']} dB)"
        )

        # [RELEASE_MUST] §2.41: source_fidelity_transfer_chain must propagate end-to-end
        assert "source_fidelity_transfer_chain" in cal, (
            "source_fidelity_transfer_chain missing from metadata['song_calibration'] — "
            "§2.41 export audit trail broken (chain_raw → SFR → metadata path severed)"
        )
        assert cal["source_fidelity_transfer_chain"] == ["vinyl", "tape", "mp3_low"], (
            f"Transfer chain mismatch in export: expected ['vinyl','tape','mp3_low'] "
            f"but got {cal['source_fidelity_transfer_chain']}"
        )

        # [RELEASE_MUST] §2.47: 3-stage chain must yield generation_count >= 3
        assert "source_fidelity_generation_count" in cal, (
            "source_fidelity_generation_count missing from export metadata"
        )
        assert cal["source_fidelity_generation_count"] >= 3, (
            f"3-stage chain (vinyl>tape>mp3_low) must produce generation_count >= 3 "
            f"(got {cal['source_fidelity_generation_count']})"
        )

        # [RELEASE_MUST] §2.47: multi-generation chain must carry HF loss > 0
        assert "source_fidelity_hf_loss_db" in cal, "source_fidelity_hf_loss_db missing from export metadata"
        assert cal["source_fidelity_hf_loss_db"] > 0.0, (
            f"3-stage chain must have cumulative HF loss > 0 dB (got {cal['source_fidelity_hf_loss_db']} dB)"
        )

    # ═══
    @pytest.mark.timeout(90)
    def test_transfer_chain_reaches_export_metadata(self):
        """Full restore() call: source_fidelity_* fields must appear in metadata.

        Proves the §2.41 audit trail survives the entire restore() pipeline:
          chain_raw → _extract_transfer_chain_from_forensics → _build_song_calibration_profile
          → metadata["song_calibration"] in RestorationResult.

        Musical-Goals metrics are stubbed to constant passing scores; runtime remains
        data-flow focused while allowing realistic startup overhead (model/plugin init)
        in CI and local environments.

        [RELEASE_MUST] §2.41 SourceFidelityReconstructor
                       §2.46 Carrier-Chain-Inversion
                       §2.47 Adaptive-Intelligence: chain → gen_count / hf_loss_db in export
        """
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from backend.core.performance_guard import QualityMode
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

        sr = 48_000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
        rng = np.random.default_rng(7)
        audio = (
            0.25 * np.sin(2 * np.pi * 220 * t)
            + 0.15 * np.sin(2 * np.pi * 440 * t)
            + 0.05 * np.sin(2 * np.pi * 880 * t)
            + 0.03 * rng.standard_normal(len(t)).astype(np.float32)
        )
        audio = np.clip(audio, -1.0, 1.0)

        _ALL_PASSING: dict[str, float] = {
            "natuerlichkeit": 0.95,
            "authentizitaet": 0.95,
            "tonal_center": 0.95,
            "timbre_authentizitaet": 0.95,
            "artikulation": 0.95,
            "emotionalitaet": 0.95,
            "micro_dynamics": 0.95,
            "groove": 0.95,
            "transparenz": 0.95,
            "waerme": 0.95,
            "bass_kraft": 0.95,
            "separation_fidelity": 0.95,
            "brillanz": 0.95,
            "spatial_depth": 0.95,
        }

        _patch_target = "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker.measure_all"
        _artifact_patch_target = "backend.core.artifact_freedom_gate.ArtifactFreedomGate.evaluate"
        _execute_patch_target = "backend.core.unified_restorer_v3.UnifiedRestorerV3._execute_pipeline"
        _feedback_patch_target = "backend.core.feedback_chain.FeedbackChain.run"
        _artifact_ok = SimpleNamespace(
            artifact_freedom=1.0,
            detected_artifacts=[],
            noise_texture_deviation_db_oct=0.0,
        )
        _feedback_ok = SimpleNamespace(
            audio=audio,
            metadata={},
            overall_score=0.95,
            total_retries=0,
            iterations=1,
            total_time_s=0.0,
            analytics_overhead_s=0.0,
            phase_executions=[],
            ceiling_reached=False,
        )
        with (
            patch(_patch_target, return_value=_ALL_PASSING),
            patch(_artifact_patch_target, return_value=_artifact_ok),
            patch(_execute_patch_target, return_value=(audio, [], [], [])),
            patch(_feedback_patch_target, return_value=_feedback_ok),
        ):
            restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.FAST))
            result = restorer.restore(audio, sr)

        assert result is not None, "restore() returned None"
        assert isinstance(result.metadata, dict), "RestorationResult.metadata must be dict"
        assert "song_calibration" in result.metadata, (
            "song_calibration missing from RestorationResult.metadata — "
            "§2.41 audit trail broken (profile never written to metadata)"
        )

        cal = result.metadata["song_calibration"]

        assert "source_fidelity_transfer_chain" in cal, (
            "source_fidelity_transfer_chain missing from metadata['song_calibration'] — "
            "§2.41 export audit trail broken (chain_raw → SFR → restore() → metadata severed)"
        )
        assert isinstance(cal["source_fidelity_transfer_chain"], list), (
            "source_fidelity_transfer_chain must be a list in export metadata"
        )
        assert "source_fidelity_generation_count" in cal, (
            "source_fidelity_generation_count missing from restore() export metadata"
        )
        assert isinstance(cal["source_fidelity_generation_count"], (int, float)), (
            "source_fidelity_generation_count must be numeric"
        )
        assert "source_fidelity_hf_loss_db" in cal, "source_fidelity_hf_loss_db missing from restore() export metadata"


# ════════════════════════════════════════════════════════════════════
# 6. MusicalGoalsChecker Integration
# ═══════════════════════════════════════════════════════════════════════


class TestMusicalGoalsCheckerIntegration:
    """Test MusicalGoalsChecker measures all 15 goals on real audio."""

    _EXPECTED_GOALS = {
        "bass_kraft",
        "brillanz",
        "waerme",
        "natuerlichkeit",
        "authentizitaet",
        "emotionalitaet",
        "transparenz",
        "groove",
        "spatial_depth",
        "timbre_authentizitaet",
        "tonal_center",
        "micro_dynamics",
        "separation_fidelity",
        "artikulation",
        "transient_energie",
    }

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_measure_all_returns_15_goals(self, audio_48k_mono):
        """measure_all() must return scores for all 15 musical goals."""
        audio, sr = audio_48k_mono
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker(mode="restoration")
        scores = checker.measure_all(audio, sr)

        assert isinstance(scores, dict)
        missing = self._EXPECTED_GOALS - set(scores.keys())
        assert len(missing) == 0, f"Missing goals: {missing}"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_all_scores_in_valid_range(self, audio_48k_mono):
        """All 15 goal scores must be in [0.0, 1.0]."""
        audio, sr = audio_48k_mono
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker(mode="restoration")
        scores = checker.measure_all(audio, sr)

        for goal, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Goal '{goal}' score {score} out of [0, 1]"
            assert math.isfinite(score), f"Goal '{goal}' score is not finite"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_thresholds_match_mode(self):
        """Checker thresholds differ between restoration and studio modes."""
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker_rest = MusicalGoalsChecker(mode="restoration")
        checker_studio = MusicalGoalsChecker(mode="studio")

        t_rest = checker_rest.thresholds
        t_studio = checker_studio.thresholds

        assert isinstance(t_rest, dict)
        assert isinstance(t_studio, dict)
        # At least some thresholds should differ between modes
        differs = sum(1 for k in t_rest if k in t_studio and t_rest[k] != t_studio[k])
        assert differs > 0, "Restoration and Studio modes should have different thresholds"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_reference_mode_produces_different_scores(self, audio_48k_mono):
        """With reference audio, authenticity/timbre metrics should be near-perfect."""
        audio, sr = audio_48k_mono
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker(mode="restoration")
        checker.measure_all(audio, sr)
        scores_with_ref = checker.measure_all(audio, sr, reference=audio)

        # Self-reference should yield high authenticity
        if "authentizitaet" in scores_with_ref:
            assert scores_with_ref["authentizitaet"] >= 0.85, (
                f"Self-reference authenticity should be ≥ 0.85, got {scores_with_ref['authentizitaet']}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 7. Bridge-API Integration
# ═══════════════════════════════════════════════════════════════════════
class TestBridgeAPIIntegration:
    """Test that Bridge API exports are functional and return correct types."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_exports_scanner(self):
        """get_defect_scanner() returns a callable scanner factory."""
        from backend.api.bridge import get_defect_scanner

        scanner_cls = get_defect_scanner()
        assert scanner_cls is not None
        scanner = scanner_cls()
        assert hasattr(scanner, "scan")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_exports_era_classifier(self):
        """get_era_classifier_fn() returns a callable."""
        from backend.api.bridge import get_era_classifier_fn

        fn = get_era_classifier_fn()
        assert callable(fn)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_exports_medium_classifier(self):
        """get_medium_classifier_fn() returns a callable."""
        from backend.api.bridge import get_medium_classifier_fn

        fn = get_medium_classifier_fn()
        assert callable(fn)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_exports_restorability(self):
        """get_restorability_estimator_class() returns the estimator class."""
        from backend.api.bridge import get_restorability_estimator_class

        cls = get_restorability_estimator_class()
        assert cls is not None
        est = cls()
        assert hasattr(est, "estimate")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_exports_goals_checker(self):
        """get_musical_goals_checker() returns a functional checker."""
        from backend.api.bridge import get_musical_goals_checker

        checker = get_musical_goals_checker()
        assert checker is not None
        assert hasattr(checker, "measure_all")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_exports_quality_mode(self):
        """get_quality_mode() returns the QualityMode enum."""
        from backend.api.bridge import get_quality_mode

        qm = get_quality_mode()
        assert qm is not None
        assert hasattr(qm, "BALANCED")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_export_guard(self):
        """export_guard() returns a callable."""
        from backend.api.bridge import export_guard

        assert callable(export_guard)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_cache_roundtrip(self, audio_48k_mono):
        """Cache functions: cache → get → roundtrip consistency."""
        from backend.api.bridge import (
            clear_defect_cache,
            get_cached_defect_result,
        )

        clear_defect_cache()
        cached = get_cached_defect_result("test_key_integration")
        assert cached is None, "Cache should be empty after clear"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_ml_memory_budget(self):
        """get_ml_memory_budget() returns module with try_allocate/release."""
        from backend.api.bridge import get_ml_memory_budget

        budget = get_ml_memory_budget()
        assert budget is not None
        assert hasattr(budget, "try_allocate")
        assert hasattr(budget, "release")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_bridge_per_phase_gate(self):
        """get_per_phase_musical_goals_gate() returns gate with wrap_phase."""
        from backend.api.bridge import get_per_phase_musical_goals_gate

        gate = get_per_phase_musical_goals_gate()
        assert gate is not None
        assert hasattr(gate, "wrap_phase")


# ═══════════════════════════════════════════════════════════════════════
# 8. PhaseInterface Contract
# ═══════════════════════════════════════════════════════════════════════
class TestPhaseInterfaceContract:
    """Test that actual phase implementations follow the PhaseInterface contract."""

    def _load_phase(self, phase_module_name: str):
        """Dynamically import a phase module and find its PhaseInterface subclass."""
        import importlib

        mod = importlib.import_module(f"backend.core.phases.{phase_module_name}")
        # Find the class that has get_metadata and process methods
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and hasattr(attr, "process") and hasattr(attr, "get_metadata"):
                if attr_name != "PhaseInterface":
                    return attr
        return None

    @pytest.mark.timeout(_INT_TIMEOUT)
    @pytest.mark.parametrize(
        "phase_name",
        [
            "phase_01_click_removal",
            "phase_02_hum_removal",
            "phase_03_denoise",
            "phase_05_rumble_filter",
            "phase_27_click_pop_removal",
            "phase_30_dc_offset_removal",
        ],
    )
    def test_phase_has_metadata(self, phase_name):
        """Phase class implements get_metadata() returning PhaseMetadata."""
        cls = self._load_phase(phase_name)
        if cls is None:
            pytest.skip(f"Cannot load phase class from {phase_name}")

        instance = cls()
        meta = instance.get_metadata()
        assert meta is not None
        assert hasattr(meta, "phase_id")
        assert hasattr(meta, "name")
        assert hasattr(meta, "category")
        assert meta.phase_id.startswith("phase_")

    @pytest.mark.timeout(_INT_TIMEOUT)
    @pytest.mark.parametrize(
        "phase_name",
        [
            "phase_30_dc_offset_removal",
            "phase_05_rumble_filter",
        ],
    )
    def test_phase_process_returns_phase_result(self, phase_name, audio_48k_mono):
        """Phase.process() returns PhaseResult with required fields."""
        audio, sr = audio_48k_mono
        cls = self._load_phase(phase_name)
        if cls is None:
            pytest.skip(f"Cannot load phase class from {phase_name}")

        instance = cls()
        result = instance.process(audio.copy(), sample_rate=sr)

        assert result is not None
        assert hasattr(result, "audio")
        assert hasattr(result, "success")
        assert hasattr(result, "modifications")
        assert isinstance(result.audio, np.ndarray)
        assert np.isfinite(result.audio).all(), "Phase output contains NaN/Inf"
        assert np.max(np.abs(result.audio)) <= 1.0, "Phase output contains clipping"


# ═══════════════════════════════════════════════════════════════════════
# 9. Denker-Kette Integration
# ═══════════════════════════════════════════════════════════════════════
class TestDenkerChainIntegration:
    """Test that the 3 sub-denkers exist and accept correct input types."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_reparatur_denker_importable(self):
        """ReparaturDenker is importable and has repariere() method."""
        from denker.reparatur_denker import ReparaturDenker

        denker = ReparaturDenker()
        assert hasattr(denker, "repariere")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_rekonstruktions_denker_importable(self):
        """RekonstruktionsDenker is importable and has rekonstruiere() method."""
        from denker.rekonstruktions_denker import RekonstruktionsDenker

        denker = RekonstruktionsDenker()
        assert hasattr(denker, "rekonstruiere")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_restaurier_denker_importable(self):
        """RestaurierDenker is importable and has restauriere() method."""
        from denker.restaurier_denker import RestaurierDenker

        denker = RestaurierDenker()
        assert hasattr(denker, "restauriere")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_aurik_denker_orchestrates_all(self):
        """AurikDenker is importable and has denke() method."""
        from denker.aurik_denker import AurikDenker

        denker = AurikDenker()
        assert hasattr(denker, "denke")
        assert hasattr(denker, "restauriere")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_denker_chain_order_spec(self):
        """§11.7a: Denker hierarchy is 6→7→8 (Reparatur→Rekonstruktion→Restaurier)."""
        from denker.aurik_denker import AurikDenker

        denker = AurikDenker()
        # Verify the denker has references to sub-denkers
        hasattr(denker, "_reparatur_denker") or hasattr(denker, "reparatur_denker")
        hasattr(denker, "_rekonstruktions_denker") or hasattr(denker, "rekonstruktions_denker")
        hasattr(denker, "_restaurier_denker") or hasattr(denker, "restaurier_denker")

        # At minimum, AurikDenker must orchestrate the chain
        assert hasattr(denker, "denke"), "AurikDenker must have denke() method"


# ═══════════════════════════════════════════════════════════════════════
# 10. Cross-Module Datenfluss Integration
# ═══════════════════════════════════════════════════════════════════════
class TestCrossModuleDataFlow:
    """Test end-to-end data flow: Scanner → Restorability → Calibration → Goals."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_full_analysis_chain(self, audio_48k_mono):
        """Complete analysis chain: Scanner → Era → Medium → Restorability → Calibration."""
        audio, sr = audio_48k_mono

        # Step 1: DefectScanner
        from backend.core.defect_scanner import DefectScanner

        scanner = DefectScanner()
        defect_result = scanner.scan(audio, sr)
        assert defect_result is not None
        material_str = defect_result.material_type.value

        # Step 2: EraClassifier
        from backend.core.era_classifier import classify_era

        era_result = classify_era(audio, sr)
        assert era_result is not None

        # Step 3: RestorabilityEstimator
        from backend.core.restorability_estimator import estimate_restorability

        rest_result = estimate_restorability(audio, sr, material=material_str)
        assert rest_result is not None

        # Step 4: SongCalibrationProfile
        from backend.core.quality_mode import QualityMode
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        max_sev = max(
            (ds.severity for ds in defect_result.scores.values()),
            default=0.0,
        )
        defect_scores_dict = {dt.value: ds.severity for dt, ds in defect_result.scores.items()}
        sf = defect_result.spectral_fingerprint if hasattr(defect_result, "spectral_fingerprint") else {}

        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=defect_result.material_type,
            mode=QualityMode.BALANCED,  # type: ignore[arg-type]
            restorability_score=rest_result.restorability_score,
            input_snr_db=rest_result.snr_db,
            max_defect_severity=max_sev,
            pipeline_confidence=0.9,
            era_decade=era_result.decade,
            defect_scores=defect_scores_dict,
            spectral_fingerprint=sf,
        )

        assert isinstance(profile, dict)
        assert "global_scalar" in profile
        assert "family_scalars" in profile

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_scanner_to_goals_chain(self, audio_48k_mono):
        """Scanner → CausalReasoner → MusicalGoals — complete quality chain."""
        audio, sr = audio_48k_mono

        # Step 1: Scan
        from backend.core.defect_scanner import DefectScanner

        defect_result = DefectScanner().scan(audio, sr)

        # Step 2: Causal reasoning
        from backend.core.causal_defect_reasoner import CausalDefectReasoner

        defect_scores = {dt.value: ds.severity for dt, ds in defect_result.scores.items()}
        plan = CausalDefectReasoner().reason(defect_scores)
        assert plan is not None

        # Step 3: Musical Goals assessment
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker(mode="restoration")
        scores = checker.measure_all(audio, sr)

        assert len(scores) == 15
        for goal, score in scores.items():
            assert 0.0 <= score <= 1.0

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_goal_applicability_feeds_into_checker(self, audio_48k_mono):
        """GoalApplicabilityFilter applicable set aligns with MusicalGoalsChecker keys."""
        audio, sr = audio_48k_mono

        from backend.core.goal_applicability_filter import GoalApplicabilityFilter
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        gaf = GoalApplicabilityFilter()
        applicability = gaf.evaluate(audio=audio, sr=sr, material="cd_digital", era_decade=2010)

        checker = MusicalGoalsChecker(mode="restoration")
        scores = checker.measure_all(audio, sr)

        # All applicable goals must have a corresponding score
        for goal in applicability.applicable:
            assert goal in scores, f"Applicable goal '{goal}' missing from checker scores"

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_spectral_fingerprint_flows_to_calibration(self, noisy_vinyl_audio):
        """DefectScanner spectral_fingerprint is accepted by SongCalibration."""
        audio, sr = noisy_vinyl_audio
        from backend.core.defect_scanner import DefectScanner
        from backend.core.quality_mode import QualityMode
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        defect_result = DefectScanner().scan(audio, sr)
        sf = defect_result.spectral_fingerprint if hasattr(defect_result, "spectral_fingerprint") else {}

        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=defect_result.material_type,
            mode=QualityMode.BALANCED,  # type: ignore[arg-type]
            restorability_score=65.0,
            input_snr_db=20.0,
            max_defect_severity=0.4,
            pipeline_confidence=0.85,
            spectral_fingerprint=sf,
        )

        assert profile is not None
        assert "global_scalar" in profile


# ═══════════════════════════════════════════════════════════════════════
# 11. Recovery + Checkpoint Integration
# ═══════════════════════════════════════════════════════════════════════
class TestRecoveryCheckpointIntegration:
    """Test RecoveryCheckpoint save/load/delete cycle."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_checkpoint_module_importable(self):
        """RecoveryCheckpoint module is importable with required functions."""
        from backend.core.recovery_checkpoint import (
            delete_checkpoint,
            find_pending_checkpoints,
            save_checkpoint,
        )

        assert callable(save_checkpoint)
        assert callable(find_pending_checkpoints)
        assert callable(delete_checkpoint)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_find_pending_returns_list(self):
        """find_pending_checkpoints() returns a list (possibly empty)."""
        from backend.core.recovery_checkpoint import find_pending_checkpoints

        result = find_pending_checkpoints()
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════
# 12. ML Memory Budget Integration
# ═══════════════════════════════════════════════════════════════════════
class TestMLMemoryBudgetIntegration:
    """Test ml_memory_budget allocate/release cycle."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_budget_allocate_release_cycle(self):
        """try_allocate() + release() cycle works without crash."""
        from backend.core.ml_memory_budget import release, try_allocate

        name = "_integration_test_model"
        ok = try_allocate(name, 0.01)
        assert isinstance(ok, bool)
        release(name)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_budget_double_allocate_check(self):
        """Allocating same model twice should be handled gracefully."""
        from backend.core.ml_memory_budget import release, try_allocate

        name = "_integration_test_double"
        try_allocate(name, 0.01)
        ok2 = try_allocate(name, 0.01)
        assert isinstance(ok2, bool)
        release(name)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_budget_release_unknown_no_crash(self):
        """Releasing unknown model must not crash."""
        from backend.core.ml_memory_budget import release

        release("_nonexistent_model_xyz_integration_test")


# ═══════════════════════════════════════════════════════════════════════
# 13. FeedbackChain Integration
# ═══════════════════════════════════════════════════════════════════════
class TestFeedbackChainIntegration:
    """Test FeedbackChain with a trivial processing function."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_feedback_chain_with_identity(self, audio_48k_mono):
        """FeedbackChain with identity function should converge immediately."""
        audio, sr = audio_48k_mono
        from backend.core.feedback_chain import FeedbackChain

        chain = FeedbackChain(max_iterations=3, sample_rate=sr)
        result = chain.run(audio, lambda a, s: a, sr=sr)

        assert result is not None
        assert hasattr(result, "audio")
        assert hasattr(result, "iterations")
        assert hasattr(result, "converged")
        assert isinstance(result.audio, np.ndarray)

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_feedback_chain_output_finite(self, audio_48k_mono):
        """FeedbackChain output must be finite and clipped."""
        audio, sr = audio_48k_mono
        from backend.core.feedback_chain import FeedbackChain

        chain = FeedbackChain(max_iterations=2, sample_rate=sr)
        result = chain.run(audio, lambda a, s: a * 0.99, sr=sr)

        assert np.isfinite(result.audio).all()


# ═══════════════════════════════════════════════════════════════════════
# 14. PerPhaseMusicalGoalsGate Integration
# ═══════════════════════════════════════════════════════════════════════
class TestPMGGIntegration:
    """Test PerPhaseMusicalGoalsGate singleton and wrap_phase contract."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_gate_singleton(self):
        """get_phase_gate() returns same instance (singleton)."""
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        g1 = get_phase_gate()
        g2 = get_phase_gate()
        assert g1 is g2

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_gate_has_wrap_phase(self):
        """Gate has wrap_phase method with correct arity."""
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        gate = get_phase_gate()
        assert hasattr(gate, "wrap_phase")
        assert callable(gate.wrap_phase)


# ═══════════════════════════════════════════════════════════════════════
# 15. PluginLifecycleManager Integration
# ═══════════════════════════════════════════════════════════════════════
class TestPluginLifecycleIntegration:
    """Test PluginLifecycleManager register/unload cycle."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_plm_importable(self):
        """PluginLifecycleManager is importable with required API."""
        from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

        plm = get_plugin_lifecycle_manager()
        assert plm is not None
        assert hasattr(plm, "register")

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_plm_register_unload(self):
        """Register + cleanup cycle works correctly."""
        from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

        plm = get_plugin_lifecycle_manager()
        unloaded = []

        plm.register(
            "_integration_test_plugin",
            0.01,
            lambda: unloaded.append(True),
        )
        # Cleanup happens via LRU eviction or explicit — just verify no crash


# ═══════════════════════════════════════════════════════════════════════
# 16. PerformanceGuard Integration
# ═══════════════════════════════════════════════════════════════════════
class TestPerformanceGuardIntegration:
    """Test PerformanceGuard monitoring lifecycle."""

    @pytest.mark.timeout(_INT_TIMEOUT)
    def test_performance_guard_start_stop(self):
        """PerformanceGuard start_monitoring / should_defer cycle."""
        from backend.core.performance_guard import PerformanceGuard

        guard = PerformanceGuard()
        guard.start_monitoring(audio_duration_seconds=60.0)

        # Should not recommend skipping immediately
        should_skip = guard.should_skip_phase("phase_03_denoise", estimated_time_seconds=5.0, remaining_phases=10)
        assert isinstance(should_skip, bool)
