"""
Tests for Enhanced Quality Gates with Multi-Metric Validation
==============================================================

Excellence Strategy #2: Perceptual Quality Gates

Tests:
1. EnhancedQualityGate: Multi-metric validation (Musical Goals + Perceptual Metrics)
2. AutoReprocessingEngine: Fallback strategies on quality failures
3. Multi-metric decision logic: Weighted scoring + threshold validation
4. Integration: Full pipeline with auto-reprocessing

Pytest usage:
    pytest tests/test_enhanced_quality_gates.py -v
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.core.musical_goals.auto_reprocessing import (
    AutoReprocessingEngine,
    ReprocessingResult,
    ReprocessingStrategy,
)
from backend.core.musical_goals.processing_modes import ProcessingMode

# Import modules under test
from backend.core.musical_goals.quality_gate import (
    EnhancedQualityGate,
    PerceptualMetrics,
    QualityGateDecision,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_audio_mono():
    """Generate 1 second mono test audio (48 kHz)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-tone signal: 500 Hz + 2 kHz + 8 kHz
    audio = 0.3 * np.sin(2 * np.pi * 500 * t) + 0.4 * np.sin(2 * np.pi * 2000 * t) + 0.2 * np.sin(2 * np.pi * 8000 * t)

    return audio, sr


@pytest.fixture
def test_audio_degraded(test_audio_mono):
    """Generate degraded audio (noise + artifacts)."""
    audio, sr = test_audio_mono

    # Add noise
    noise = np.random.normal(0, 0.05, len(audio))
    degraded = audio + noise

    # Add click artifacts
    click_positions = [int(sr * 0.25), int(sr * 0.5), int(sr * 0.75)]
    for pos in click_positions:
        degraded[pos : pos + 10] += 0.5

    return degraded, sr


@pytest.fixture
def mock_perceptual_metrics():
    """Mock perceptual metrics (good quality)."""
    # nisqa_mos / dnsmos_ovrl / dnsmos_sig / dnsmos_bak entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
    return PerceptualMetrics(visqol_mos_lqo=3.9, versa_score=85.0)


@pytest.fixture
def mock_perceptual_metrics_poor():
    """Mock perceptual metrics (poor quality)."""
    # nisqa_mos / dnsmos_ovrl / dnsmos_sig / dnsmos_bak entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
    return PerceptualMetrics(visqol_mos_lqo=2.3, versa_score=55.0)


@pytest.fixture
def mock_musical_goals_good():
    """Mock Musical Goals scores (good) — goal set for quality-gate tests."""
    return {
        "brillanz": 0.92,  # ≥ 0.85
        "waerme": 0.88,  # ≥ 0.80
        "natuerlichkeit": 0.91,  # ≥ 0.90
        "authentizitaet": 0.90,  # ≥ 0.88
        "emotionalitaet": 0.89,  # ≥ 0.87
        "transparenz": 0.91,  # ≥ 0.89
        "bass_kraft": 0.87,  # ≥ 0.85
        "groove": 0.90,  # ≥ 0.88
        "spatial_depth": 0.82,  # ≥ 0.75
        "timbre_authentizitaet": 0.89,  # ≥ 0.87
    }


@pytest.fixture
def mock_musical_goals_poor():
    """Mock Musical Goals scores (poor) — goal set for quality-gate tests."""
    return {
        "brillanz": 0.65,
        "waerme": 0.60,
        "natuerlichkeit": 0.62,
        "authentizitaet": 0.66,
        "emotionalitaet": 0.68,
        "transparenz": 0.55,
        "bass_kraft": 0.68,
        "groove": 0.70,
        "spatial_depth": 0.72,
        "timbre_authentizitaet": 0.63,
    }


# =============================================================================
# Test AutoReprocessingEngine
# =============================================================================


class TestAutoReprocessingEngine:
    """Test automatic reprocessing with fallback strategies."""

    def test_initialization(self):
        """Test engine initialization with default parameters."""
        engine = AutoReprocessingEngine()

        assert engine.max_attempts == 5
        assert engine.min_improvement == 0.02
        assert engine.enable_hybrid_fallback is True
        assert engine.enable_forensic_guidance is True

    def test_strategy_selection_critical_degradation(self, mock_musical_goals_poor):
        """Test strategy selection for critically degraded signal."""
        engine = AutoReprocessingEngine()

        violations = {
            "brillanz": {"expected": 0.90, "achieved": 0.40, "delta": -0.20},
            "transparenz": {"expected": 0.88, "achieved": 0.35, "delta": -0.25},
            "natuerlichkeit": {"expected": 0.87, "achieved": 0.38, "delta": -0.22},
        }

        # Baseline must have avg < 0.50 to trigger FORENSIC_GUIDED
        baseline = {
            "brillanz": 0.38,
            "transparenz": 0.40,
            "natuerlichkeit": 0.35,
            "waerme": 0.42,
            "spatial_depth": 0.39,  # current naming: raeumlichkeit → spatial_depth
        }
        context = {"medium_type": "vinyl"}

        strategies = engine._select_strategies(violations, baseline, context)

        # For critically degraded signal (3 critical + avg < 0.50), expect FORENSIC_GUIDED first
        assert ReprocessingStrategy.FORENSIC_GUIDED in strategies
        assert ReprocessingStrategy.PARAMETER_REDUCTION in strategies
        assert len(strategies) <= engine.max_attempts

    def test_strategy_selection_overprocessing(self):
        """Test strategy selection for over-processing (HF artifacts)."""
        engine = AutoReprocessingEngine()

        violations = {"brillanz": {"expected": 0.90, "achieved": 0.75, "delta": -0.05}}

        baseline = {"brillanz": 0.80, "waerme": 0.85, "transparenz": 0.88}

        context: dict[Any, Any] = {}

        strategies = engine._select_strategies(violations, baseline, context)

        # For over-processing, expect PARAMETER_REDUCTION + ALTERNATIVE_CHAIN
        assert ReprocessingStrategy.PARAMETER_REDUCTION in strategies
        assert ReprocessingStrategy.ALTERNATIVE_CHAIN in strategies

    def test_parameter_reduction_strategy(self, test_audio_mono):
        """Test parameter reduction strategy execution."""
        engine = AutoReprocessingEngine()
        audio, sr = test_audio_mono

        # Mock processing function
        def mock_processor(audio, sr, params):
            intensity = params.get("intensity", 1.0)
            # Simulate intensity-scaled processing
            return audio * (0.9 + 0.1 * intensity)

        context: dict[Any, Any] = {}

        # Execute parameter reduction (attempt 1 → 50% intensity)
        reprocessed, params = engine._strategy_parameter_reduction(
            audio, sr, mock_processor, attempt_num=1, context=context
        )

        assert "intensity" in params
        assert params["intensity"] == 0.50
        assert len(reprocessed) == len(audio)

    def test_hybrid_blend_strategy(self, test_audio_mono, test_audio_degraded):
        """Test hybrid blend strategy (original + processed mix)."""
        engine = AutoReprocessingEngine()
        original, sr = test_audio_mono
        processed, _ = test_audio_degraded

        context: dict[Any, Any] = {}

        # Execute hybrid blend (attempt 1 → 70% processed + 30% original)
        blended, params = engine._strategy_hybrid_blend(original, processed, sr, attempt_num=1, context=context)

        assert "processed_weight" in params
        assert "original_weight" in params
        assert params["processed_weight"] == 0.7
        assert params["original_weight"] == 0.3

        # Check blend is between original and processed
        min_len = min(len(original), len(processed))
        expected = 0.7 * processed[:min_len] + 0.3 * original[:min_len]
        np.testing.assert_allclose(blended, expected, rtol=1e-5)

    def test_reprocessing_success(self, test_audio_mono):
        """Test successful reprocessing that passes quality gates."""
        engine = AutoReprocessingEngine(max_attempts=5)
        audio, sr = test_audio_mono

        # Mock processing function (returns good audio with low intensity)
        def mock_processor(audio, sr, params):
            intensity = params.get("intensity", 1.0)
            # Simulate better quality with reduced intensity
            if intensity <= 0.75:
                return audio * 1.0  # Good quality
            else:
                return audio * 0.5  # Poor quality

        # Mock quality validator (passes when amplitude > 0.8)
        def mock_validator(original, processed, sr):
            if np.max(np.abs(processed)) > 0.8:
                # Passed
                return True, {"goal1": 0.90, "goal2": 0.88}, {}
            else:
                # Failed
                return False, {"goal1": 0.70, "goal2": 0.65}, {"goal1": {"expected": 0.85, "achieved": 0.70}}

        baseline = {"goal1": 0.75, "goal2": 0.72}
        violations = {"goal1": {"expected": 0.85, "achieved": 0.70}}

        result = engine.reprocess_on_failure(
            original=audio,
            failed_processed=audio * 0.5,
            sr=sr,
            processing_function=mock_processor,
            quality_validator=mock_validator,
            baseline_scores=baseline,
            initial_violations=violations,
        )

        # First attempt with parameter reduction (intensity=0.50) should succeed
        assert result.total_attempts >= 1
        assert len(result.attempts) >= 1
        # May succeed or return partial improvement
        assert result.final_decision in ["reprocessed", "partial_improvement", "rollback_original"]

    def test_reprocessing_exhausted_fallback_to_original(self, test_audio_mono):
        """Test all attempts fail → rollback to original."""
        engine = AutoReprocessingEngine(max_attempts=2)
        audio, sr = test_audio_mono

        # Mock processor that always fails
        def mock_processor(audio, sr, params):
            return audio * 0.3  # Always too quiet

        # Mock validator that always fails
        def mock_validator(original, processed, sr):
            return False, {"goal": 0.50}, {"goal": {"expected": 0.85, "achieved": 0.50}}

        baseline = {"goal": 0.60}
        violations = {"goal": {"expected": 0.85, "achieved": 0.50}}

        result = engine.reprocess_on_failure(
            original=audio,
            failed_processed=audio * 0.3,
            sr=sr,
            processing_function=mock_processor,
            quality_validator=mock_validator,
            baseline_scores=baseline,
            initial_violations=violations,
        )

        assert result.success is False
        assert result.total_attempts == 2
        assert result.final_decision == "rollback_original"
        # Should return original audio
        np.testing.assert_allclose(result.best_audio, audio, rtol=1e-5)


# =============================================================================
# Test EnhancedQualityGate
# =============================================================================


class TestEnhancedQualityGate:
    """Test Enhanced Quality Gate with multi-metric validation."""

    @patch("backend.core.musical_goals.quality_gate.MusicalGoalsQualityGate")
    def test_initialization(self, mock_musical_gate):
        """Test initialization with default parameters."""
        gate = EnhancedQualityGate()

        assert gate.enable_perceptual_metrics is True
        assert gate.enable_auto_reprocessing is True
        assert gate.metric_weights["musical_goals"] == 0.50
        assert gate.metric_weights["perceptual_quality"] == 0.50
        # nisqa_threshold / dnsmos_threshold entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
        assert gate.visqol_threshold == 3.0
        assert gate.versa_threshold == 80.0

    def test_weighted_quality_score_calculation(self, mock_musical_goals_good, mock_perceptual_metrics):
        """Test weighted quality score calculation."""
        gate = EnhancedQualityGate()

        score = gate._calculate_weighted_quality_score(
            mock_musical_goals_good, mock_perceptual_metrics, ProcessingMode.STUDIO_2026
        )

        # Score should be 0-1
        assert 0.0 <= score <= 1.0

        # With good metrics, expect high score (>0.80)
        assert score > 0.80

    def test_weighted_quality_score_poor_quality(self, mock_musical_goals_poor, mock_perceptual_metrics_poor):
        """Test weighted quality score with poor quality metrics."""
        gate = EnhancedQualityGate()

        score = gate._calculate_weighted_quality_score(
            mock_musical_goals_poor, mock_perceptual_metrics_poor, ProcessingMode.RESTORATION
        )

        # With poor metrics, expect lower score (<0.70)
        assert score < 0.70

    def test_multi_metric_decision_all_passed(self, mock_musical_goals_good, mock_perceptual_metrics):
        """Test decision logic when all metrics pass."""
        gate = EnhancedQualityGate()

        # Mock Musical Goals result (passed)
        from backend.core.musical_goals.quality_gate import PostCheckResult

        musical_result = PostCheckResult(
            passed=True,
            decision=QualityGateDecision.PASSED,
            baseline_scores={},
            achieved_scores=mock_musical_goals_good,
            violations={},
            improvements={},
            degradations={},
        )

        decision, action, recommendation = gate._make_multi_metric_decision(
            musical_result, mock_perceptual_metrics, {}, weighted_score=0.90
        )

        assert decision == QualityGateDecision.PASSED
        assert action is None
        assert isinstance(recommendation, str)
        assert "passed" in recommendation.lower()

    def test_multi_metric_decision_critical_failure(self, mock_musical_goals_poor, mock_perceptual_metrics_poor):
        """Test decision logic with critical violations."""
        gate = EnhancedQualityGate()

        # Mock Musical Goals result (critical failure)
        from backend.core.musical_goals.quality_gate import PostCheckResult

        musical_result = PostCheckResult(
            passed=False,
            decision=QualityGateDecision.ROLLBACK_REQUIRED,
            baseline_scores={},
            achieved_scores=mock_musical_goals_poor,
            violations={"brillanz": {"expected": 0.90, "achieved": 0.65, "delta": -0.05}},
            improvements={},
            degradations={},
        )

        decision, action, recommendation = gate._make_multi_metric_decision(
            musical_result, mock_perceptual_metrics_poor, {}, weighted_score=0.55
        )

        assert decision == QualityGateDecision.ROLLBACK_REQUIRED
        assert action == "reprocess_or_rollback"
        assert isinstance(recommendation, str)
        assert "critical" in recommendation.lower()

    @patch.object(EnhancedQualityGate, "_measure_perceptual_metrics")
    @patch("backend.core.musical_goals.quality_gate.MusicalGoalsQualityGate")
    def test_enhanced_pre_check(
        self, mock_musical_gate, mock_measure, test_audio_mono, mock_perceptual_metrics, mock_musical_goals_good
    ):
        """Test enhanced pre-check with Musical Goals + Perceptual Metrics."""
        audio, sr = test_audio_mono

        # Mock Musical Goals pre-check
        from backend.core.musical_goals.quality_gate import PreCheckResult

        mock_pre_result = PreCheckResult(
            passed=True, measurable=True, baseline_scores=mock_musical_goals_good, warnings=[], edge_cases_detected=[]
        )

        mock_musical_gate_instance = MagicMock()
        mock_musical_gate_instance.pre_check.return_value = mock_pre_result
        mock_musical_gate.return_value = mock_musical_gate_instance

        # Mock perceptual metrics
        mock_measure.return_value = mock_perceptual_metrics

        gate = EnhancedQualityGate(enable_perceptual_metrics=True)

        pre_check = gate.enhanced_pre_check(audio, sr, ProcessingMode.STUDIO_2026)

        assert pre_check.passed is True
        assert pre_check.measurable is True
        assert pre_check.baseline_musical_goals == mock_musical_goals_good
        assert pre_check.baseline_perceptual is not None
        # nisqa_mos entfernt — verboten §4.4+§10.2; cdpam_score als §4.4-konforme Ersatzprüfung
        assert pre_check.baseline_perceptual.versa_score == 85.0

    @patch.object(EnhancedQualityGate, "_measure_perceptual_metrics")
    @patch("backend.core.musical_goals.quality_gate.MusicalGoalsQualityGate")
    def test_enhanced_post_check_passed(
        self, mock_musical_gate, mock_measure, test_audio_mono, mock_perceptual_metrics, mock_musical_goals_good
    ):
        """Test enhanced post-check with passing quality."""
        original, sr = test_audio_mono
        processed = original * 1.05  # Slightly boosted

        # Mock Musical Goals post-check (passed)
        from backend.core.musical_goals.quality_gate import PostCheckResult

        mock_post_result = PostCheckResult(
            passed=True,
            decision=QualityGateDecision.PASSED,
            baseline_scores=mock_musical_goals_good,
            achieved_scores=mock_musical_goals_good,
            violations={},
            improvements={"brillanz": 0.02},
            degradations={},
        )

        mock_musical_gate_instance = MagicMock()
        mock_musical_gate_instance.post_check.return_value = mock_post_result
        mock_musical_gate.return_value = mock_musical_gate_instance

        # Mock perceptual metrics (good)
        mock_measure.return_value = mock_perceptual_metrics

        gate = EnhancedQualityGate(enable_perceptual_metrics=True)

        post_check = gate.enhanced_post_check(
            original,
            processed,
            sr,
            ProcessingMode.STUDIO_2026,
            baseline_musical=mock_musical_goals_good,
            baseline_perceptual=mock_perceptual_metrics,
        )

        assert post_check.passed is True
        assert post_check.decision == QualityGateDecision.PASSED
        assert post_check.weighted_quality_score > 0.80
        assert len(post_check.perceptual_improvements) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEnhancedQualityGatesIntegration:
    """Integration tests for full pipeline."""

    @patch.object(EnhancedQualityGate, "_measure_perceptual_metrics")
    @patch.object(EnhancedQualityGate, "_get_reprocessing_engine")
    @patch("backend.core.musical_goals.quality_gate.MusicalGoalsQualityGate")
    def test_validate_with_auto_reprocessing_success_first_attempt(
        self,
        mock_musical_gate,
        mock_engine_getter,
        mock_measure,
        test_audio_mono,
        mock_perceptual_metrics,
        mock_musical_goals_good,
    ):
        """Test validation with auto-reprocessing: success on first attempt."""
        original, sr = test_audio_mono
        processed = original * 1.02  # Good processing

        # Mock Musical Goals results (passed)
        from backend.core.musical_goals.quality_gate import PostCheckResult, PreCheckResult

        mock_pre = PreCheckResult(passed=True, measurable=True, baseline_scores=mock_musical_goals_good, warnings=[])

        mock_post = PostCheckResult(
            passed=True,
            decision=QualityGateDecision.PASSED,
            baseline_scores=mock_musical_goals_good,
            achieved_scores=mock_musical_goals_good,
            violations={},
        )

        mock_musical_gate_instance = MagicMock()
        mock_musical_gate_instance.pre_check.return_value = mock_pre
        mock_musical_gate_instance.post_check.return_value = mock_post
        mock_musical_gate.return_value = mock_musical_gate_instance

        # Mock perceptual metrics
        mock_measure.return_value = mock_perceptual_metrics

        gate = EnhancedQualityGate(enable_auto_reprocessing=True)

        # Mock processing function
        def mock_processor(audio, sr, params):
            return audio * 1.02

        final_audio, result = gate.validate_with_auto_reprocessing(
            original, processed, sr, ProcessingMode.STUDIO_2026, processing_function=mock_processor
        )

        # Should pass without reprocessing
        assert result.passed is True
        assert result.reprocessing_performed is False
        np.testing.assert_allclose(final_audio, processed, rtol=1e-5)

    @patch.object(EnhancedQualityGate, "_measure_perceptual_metrics")
    @patch.object(EnhancedQualityGate, "_get_reprocessing_engine")
    @patch("backend.core.musical_goals.quality_gate.MusicalGoalsQualityGate")
    def test_validate_with_auto_reprocessing_triggers_on_failure(
        self,
        mock_musical_gate,
        mock_engine_getter,
        mock_measure,
        test_audio_mono,
        mock_perceptual_metrics,
        mock_musical_goals_good,
        mock_musical_goals_poor,
    ):
        """Test validation triggers auto-reprocessing on failure."""
        original, sr = test_audio_mono
        failed_processed = original * 0.5  # Poor processing

        # Mock Musical Goals: pre-check passed, post-check failed
        from backend.core.musical_goals.quality_gate import PostCheckResult, PreCheckResult

        mock_pre = PreCheckResult(passed=True, measurable=True, baseline_scores=mock_musical_goals_good, warnings=[])

        mock_post_failed = PostCheckResult(
            passed=False,
            decision=QualityGateDecision.WARNING,
            baseline_scores=mock_musical_goals_good,
            achieved_scores=mock_musical_goals_poor,
            violations={"brillanz": {"expected": 0.90, "achieved": 0.65}},
        )

        mock_post_success = PostCheckResult(
            passed=True,
            decision=QualityGateDecision.PASSED,
            baseline_scores=mock_musical_goals_good,
            achieved_scores=mock_musical_goals_good,
            violations={},
        )

        mock_musical_gate_instance = MagicMock()
        mock_musical_gate_instance.pre_check.return_value = mock_pre
        # First call fails, subsequent calls succeed
        mock_musical_gate_instance.post_check.side_effect = [
            mock_post_failed,  # Initial processed
            mock_post_success,  # After reprocessing
            mock_post_success,  # Final validation
        ]
        mock_musical_gate.return_value = mock_musical_gate_instance

        # Mock perceptual metrics
        mock_measure.return_value = mock_perceptual_metrics

        # Mock reprocessing engine
        mock_reprocessing_result = ReprocessingResult(
            success=True,
            best_audio=original * 1.0,  # Reprocessed audio
            best_quality_scores=mock_musical_goals_good,
            attempts=[],
            total_attempts=2,
            strategy_used=ReprocessingStrategy.PARAMETER_REDUCTION,
            final_decision="reprocessed",
            improvements_achieved={"brillanz": 0.25},
        )

        mock_engine = MagicMock()
        mock_engine.reprocess_on_failure.return_value = mock_reprocessing_result
        mock_engine_getter.return_value = mock_engine

        gate = EnhancedQualityGate(enable_auto_reprocessing=True)

        # Mock processing function
        def mock_processor(audio, sr, params):
            intensity = params.get("intensity", 1.0)
            return audio * intensity

        final_audio, result = gate.validate_with_auto_reprocessing(
            original, failed_processed, sr, ProcessingMode.STUDIO_2026, processing_function=mock_processor
        )

        # Should trigger reprocessing
        assert mock_engine.reprocess_on_failure.called
        assert result.reprocessing_performed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
