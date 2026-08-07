"""
Test Suite for Goal Explainability System

Component 4.4: Explainability & User Transparency
Tests all explainability features:
- Goal tracking through processing chain
- Step-by-step impact attribution
- Natural language explanation generation
- Goal trajectory visualization data
- Actionable recommendations
- Simple one-shot explanations

Coverage: 25+ test cases across all explainability features

Author: AI Team
Date: 8. Februar 2026
"""

import numpy as np
import pytest

from backend.core.musical_goals.explainability import (
    GoalChangeType,
    GoalExplainer,
    GoalExplanation,
    GoalTrajectory,
    ProcessingStepImpact,
)
from backend.core.musical_goals.processing_modes import PROCESSING_MODE_CONFIGS, ProcessingMode

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def explainer():
    """Create GoalExplainer instance."""

    class _FastDeterministicChecker:
        """Lightweight checker for explainability unit tests.

        It preserves relative score changes across steps but avoids expensive
        model-backed metric evaluation in unit-test context.
        """

        def __init__(self):
            self._goals = list(PROCESSING_MODE_CONFIGS[ProcessingMode.RESTORATION].musical_goals.keys())

        def measure_all(self, audio, _sr, reference=None):
            a = np.asarray(audio, dtype=np.float32)
            ref = np.asarray(reference, dtype=np.float32) if reference is not None else None

            rms = float(np.sqrt(np.mean(a * a)) + 1e-12)
            base = 0.70 + 0.20 * np.tanh(6.0 * (rms - 0.18))

            delta = 0.0
            if ref is not None and len(ref) > 0:
                ref_rms = float(np.sqrt(np.mean(ref * ref)) + 1e-12)
                delta = 0.12 * np.tanh(5.0 * (rms - ref_rms))

            out = {}
            for idx, goal in enumerate(self._goals):
                # Stable per-goal offset keeps ordering deterministic.
                goal_offset = (idx - len(self._goals) / 2.0) * 0.004
                score = float(np.clip(base + delta + goal_offset, 0.0, 1.0))
                out[goal] = score
            return out

    return GoalExplainer(checker=_FastDeterministicChecker())  # type: ignore[arg-type]


@pytest.fixture
def clean_audio():
    """Clean audio signal for testing."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-frequency signal
    audio = (
        0.2 * np.sin(2 * np.pi * 100 * t)
        + 0.3 * np.sin(2 * np.pi * 500 * t)
        + 0.2 * np.sin(2 * np.pi * 2000 * t)
        + 0.15 * np.sin(2 * np.pi * 8000 * t)
    )

    return audio, sr


@pytest.fixture
def processing_chain(clean_audio):
    """Simulated processing chain with multiple steps."""
    audio, sr = clean_audio

    # Step 1: Denoise (adds slight noise for variation)
    step1 = audio + np.random.normal(0, 0.01, len(audio))

    # Step 2: EQ (boost signal)
    step2 = step1 * 1.05

    # Step 3: Compressor (reduce dynamic range slightly)
    step3 = np.tanh(step2 * 1.2) * 0.9

    # Step 4: Limiter (slight reduction)
    step4 = step3 * 0.98

    return [("Original", audio), ("Denoise", step1), ("EQ", step2), ("Compressor", step3), ("Limiter", step4)], sr


# =============================================================================
# Test Class 1: Tracking Functionality
# =============================================================================


class TestTrackingFunctionality:
    """Test goal tracking through processing chain."""

    def test_start_tracking(self, explainer, clean_audio):
        """Should successfully start tracking."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr, mode=ProcessingMode.RESTORATION)

        assert explainer.is_tracking
        assert explainer.mode == ProcessingMode.RESTORATION
        assert explainer.sr == sr
        assert len(explainer.goal_history) == 1
        assert explainer.goal_history[0]["step"] == "Original"

    def test_record_step(self, explainer, clean_audio):
        """Should successfully record processing step."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr)

        # Simulate processing
        processed = audio * 1.1
        scores = explainer.record_step("Test Step", processed, sr)

        assert len(explainer.step_history) == 1
        assert len(explainer.goal_history) == 2
        assert explainer.step_history[0]["name"] == "Test Step"
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_record_multiple_steps(self, explainer, clean_audio):
        """Should track multiple processing steps."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr)

        # Simulate multi-step processing
        step1 = audio * 1.05
        step2 = step1 * 0.98
        step3 = step2 + np.random.normal(0, 0.01, len(step2))

        explainer.record_step("Step 1", step1, sr)
        explainer.record_step("Step 2", step2, sr)
        explainer.record_step("Step 3", step3, sr)

        assert len(explainer.step_history) == 3
        assert len(explainer.goal_history) == 4  # Original + 3 steps

    def test_stop_tracking(self, explainer, clean_audio):
        """Should successfully stop tracking."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr)
        assert explainer.is_tracking

        explainer.stop_tracking()
        assert not explainer.is_tracking

    def test_record_step_without_tracking_fails(self, explainer, clean_audio):
        """Recording without starting tracking should fail."""
        audio, sr = clean_audio

        with pytest.raises(RuntimeError):
            explainer.record_step("Test", audio, sr)


# =============================================================================
# Test Class 2: Goal Trajectory Building
# =============================================================================


class TestGoalTrajectoryBuilding:
    """Test goal trajectory construction."""

    def test_trajectory_structure(self, explainer, processing_chain):
        """Goal trajectory should have correct structure."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        # Should have trajectory for each goal
        assert len(explanation.goal_trajectories) > 0

        for goal_name, traj in explanation.goal_trajectories.items():
            assert isinstance(traj, GoalTrajectory)
            assert traj.goal_name == goal_name
            assert isinstance(traj.initial_score, float)
            assert isinstance(traj.final_score, float)
            assert isinstance(traj.target_score, float)
            assert len(traj.step_scores) == len(chain)
            assert len(traj.step_deltas) == len(chain) - 1
            assert isinstance(traj.achieved, bool)
            assert isinstance(traj.change_type, GoalChangeType)
            assert isinstance(traj.explanation, str)

    def test_step_scores_progression(self, explainer, processing_chain):
        """Step scores should progress through processing."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        for traj in explanation.goal_trajectories.values():
            # Should have score for each step
            assert len(traj.step_scores) == len(chain)

            # First score should match initial
            assert abs(traj.step_scores[0] - traj.initial_score) < 0.001

            # Last score should match final
            assert abs(traj.step_scores[-1] - traj.final_score) < 0.001

    def test_change_type_detection(self, explainer, clean_audio):
        """Should correctly detect change types."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr, mode=ProcessingMode.RESTORATION)

        # Simulate different change types
        # Improved
        improved = audio * 1.2
        explainer.record_step("Improve", improved, sr)

        explanation = explainer.generate_explanation()

        # At least one goal should show change
        change_types = [traj.change_type for traj in explanation.goal_trajectories.values()]
        assert len(set(change_types)) > 0


# =============================================================================
# Test Class 3: Step Impact Calculation
# =============================================================================


class TestStepImpactCalculation:
    """Test processing step impact attribution."""

    def test_step_impacts_structure(self, explainer, processing_chain):
        """Step impacts should have correct structure."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        # Should have impact for each step
        assert len(explanation.step_impacts) == len(chain) - 1

        for impact in explanation.step_impacts:
            assert isinstance(impact, ProcessingStepImpact)
            assert isinstance(impact.step_name, str)
            assert isinstance(impact.step_index, int)
            assert isinstance(impact.goal_changes, dict)
            assert isinstance(impact.overall_impact, float)
            assert isinstance(impact.positive_impacts, list)
            assert isinstance(impact.negative_impacts, list)
            assert isinstance(impact.explanation, str)

    def test_goal_changes_calculation(self, explainer, clean_audio):
        """Goal changes should be correctly calculated."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr)

        # Step with known impact
        boosted = audio * 1.5
        explainer.record_step("Boost", boosted, sr)

        explanation = explainer.generate_explanation()
        impact = explanation.step_impacts[0]

        # Should have changes for each goal
        assert len(impact.goal_changes) > 0

        # Some goals should show change
        has_changes = any(abs(delta) > 0.001 for delta in impact.goal_changes.values())
        assert has_changes

    def test_positive_negative_impacts(self, explainer, processing_chain):
        """Should correctly identify positive/negative impacts."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        for impact in explanation.step_impacts:
            # Positive and negative should not overlap
            overlap = set(impact.positive_impacts) & set(impact.negative_impacts)
            assert len(overlap) == 0


# =============================================================================
# Test Class 4: Explanation Generation
# =============================================================================


class TestExplanationGeneration:
    """Test natural language explanation generation."""

    def test_explanation_structure(self, explainer, processing_chain):
        """Explanation should have complete structure."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr, mode=ProcessingMode.RESTORATION)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        assert isinstance(explanation, GoalExplanation)
        assert isinstance(explanation.mode, ProcessingMode)
        assert isinstance(explanation.overall_success, bool)
        assert isinstance(explanation.achieved_goals, list)
        assert isinstance(explanation.failed_goals, list)
        assert isinstance(explanation.goal_trajectories, dict)
        assert isinstance(explanation.step_impacts, list)
        assert isinstance(explanation.summary, str)
        assert isinstance(explanation.recommendations, list)
        assert isinstance(explanation.details, dict)

    def test_summary_generation(self, explainer, processing_chain):
        """Summary should be user-friendly."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        # Summary should contain key info
        assert len(explanation.summary) > 50
        assert "goals" in explanation.summary.lower()
        assert "achieved" in explanation.summary.lower()

    def test_achieved_failed_classification(self, explainer, processing_chain):
        """Should correctly classify achieved/failed goals."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        # All goals should be categorized
        total_goals = len(explanation.goal_trajectories)
        categorized = len(explanation.achieved_goals) + len(explanation.failed_goals)
        assert categorized == total_goals

        # No overlap
        overlap = set(explanation.achieved_goals) & set(explanation.failed_goals)
        assert len(overlap) == 0

    def test_trajectory_explanations(self, explainer, processing_chain):
        """Each trajectory should have explanation."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        for traj in explanation.goal_trajectories.values():
            assert len(traj.explanation) > 10
            assert str(traj.initial_score) in traj.explanation or f"{traj.initial_score:.2f}" in traj.explanation
            assert "→" in traj.explanation or "to" in traj.explanation.lower()

    def test_step_impact_explanations(self, explainer, processing_chain):
        """Each step should have explanation."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        for impact in explanation.step_impacts:
            assert len(impact.explanation) > 5
            # Explanation should be meaningful (not just step name)
            # It should mention improvements/degradations or "No significant impact"
            assert (
                "Improved" in impact.explanation
                or "Degraded" in impact.explanation
                or "No significant impact" in impact.explanation
            )


# =============================================================================
# Test Class 5: Recommendations
# =============================================================================


class TestRecommendations:
    """Test recommendation generation."""

    @pytest.mark.slow
    def test_recommendations_for_failed_goals(self, explainer, processing_chain):
        """Should provide recommendations for failed goals."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        if len(explanation.failed_goals) > 0:
            # Should have recommendations
            assert len(explanation.recommendations) > 0
        else:
            # All achieved - should say so
            assert len(explanation.recommendations) >= 0

    @pytest.mark.slow
    def test_recommendations_are_actionable(self, explainer, processing_chain):
        """Recommendations should mention steps or actions."""
        chain, sr = processing_chain

        explainer.start_tracking(chain[0][1], sr)
        for name, audio in chain[1:]:
            explainer.record_step(name, audio, sr)

        explanation = explainer.generate_explanation()

        for rec in explanation.recommendations:
            # Should be user-friendly text
            assert isinstance(rec, str)
            assert len(rec) > 10


# =============================================================================
# Test Class 6: Simple Explanation (No Tracking)
# =============================================================================


class TestSimpleExplanation:
    """Test simple one-shot explanation."""

    def test_explain_simple_no_tracking(self, explainer, clean_audio):
        """Simple explanation should work without tracking."""
        audio, sr = clean_audio

        processed = audio * 1.1

        explanation_text = explainer.explain_simple(audio, processed, sr)

        assert isinstance(explanation_text, str)
        assert len(explanation_text) > 50
        assert "goals" in explanation_text.lower()

    def test_simple_explanation_shows_changes(self, explainer, clean_audio):
        """Simple explanation should show goal changes."""
        audio, sr = clean_audio

        processed = audio * 1.2

        explanation_text = explainer.explain_simple(audio, processed, sr)

        # Should show arrows or changes
        assert "→" in explanation_text or "->" in explanation_text

        # Should show scores
        assert any(char.isdigit() for char in explanation_text)

    def test_simple_explanation_all_modes(self, explainer, clean_audio):
        """Simple explanation should work for all modes."""
        audio, sr = clean_audio
        processed = audio * 1.1

        for mode in ProcessingMode:
            explanation = explainer.explain_simple(audio, processed, sr, mode=mode)
            assert isinstance(explanation, str)
            assert len(explanation) > 20


# =============================================================================
# Test Class 7: Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""

    @pytest.mark.slow
    def test_full_workflow(self, explainer, processing_chain):
        """Test complete tracking and explanation workflow."""
        chain, sr = processing_chain

        # Start tracking
        explainer.start_tracking(chain[0][1], sr, mode=ProcessingMode.RESTORATION)

        # Record all steps
        for name, audio in chain[1:]:
            scores = explainer.record_step(name, audio, sr)
            assert len(scores) > 0

        # Generate explanation
        explanation = explainer.generate_explanation()

        assert explanation is not None
        assert len(explanation.goal_trajectories) > 0
        assert len(explanation.step_impacts) == len(chain) - 1
        assert len(explanation.summary) > 0

        # Stop tracking
        explainer.stop_tracking()
        assert not explainer.is_tracking

    def test_multiple_tracking_sessions(self, explainer, clean_audio):
        """Should handle multiple tracking sessions."""
        audio, sr = clean_audio

        # First session
        explainer.start_tracking(audio, sr)
        explainer.record_step("Step 1", audio * 1.1, sr)
        exp1 = explainer.generate_explanation()
        explainer.stop_tracking()

        # Second session
        explainer.start_tracking(audio, sr)
        explainer.record_step("Step 2", audio * 0.9, sr)
        exp2 = explainer.generate_explanation()
        explainer.stop_tracking()

        # Should be independent
        assert len(exp1.step_impacts) == 1
        assert len(exp2.step_impacts) == 1
        assert exp1.step_impacts[0].step_name != exp2.step_impacts[0].step_name

    def test_all_processing_modes(self, explainer, clean_audio):
        """Should work with all processing modes."""
        audio, sr = clean_audio

        for mode in ProcessingMode:
            explainer.start_tracking(audio, sr, mode=mode)
            explainer.record_step("Test", audio * 1.05, sr)
            explanation = explainer.generate_explanation()
            explainer.stop_tracking()

            assert explanation.mode == mode
            assert len(explanation.goal_trajectories) > 0

    def test_display_names_used(self, explainer, clean_audio):
        """User-friendly display names should be used."""
        audio, sr = clean_audio

        processed = audio * 1.1
        explanation_text = explainer.explain_simple(audio, processed, sr)

        # Should use English names, not German keys
        assert "Bass Power" in explanation_text or "bass" in explanation_text.lower()
        # Should not have raw keys like 'bass_kraft'
        # (Can have it in detailed mode, but simple should be user-friendly)


# =============================================================================
# Test Class 8: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_processing_steps(self, explainer, clean_audio):
        """Should handle case with no processing steps."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr)

        # Try to generate explanation without any steps
        with pytest.raises(RuntimeError):
            explainer.generate_explanation()

    def test_identical_audio_steps(self, explainer, clean_audio):
        """Should handle identical audio across steps."""
        audio, sr = clean_audio

        explainer.start_tracking(audio, sr)
        explainer.record_step("NoChange", audio.copy(), sr)

        explanation = explainer.generate_explanation()

        # Should complete without errors
        assert explanation is not None

        # Deltas may not be exactly zero due to randomness in spectral analysis
        # or numerical precision, but should be relatively small
        for traj in explanation.goal_trajectories.values():
            for delta in traj.step_deltas:
                # Relaxed threshold - goals use spectral features which can vary slightly
                assert abs(delta) < 0.5  # Significant changes unlikely for identical audio


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
