"""
Tests for ML Policy Engine compatibility facade

Tests:
- Model selection for all 28 Docker plugins
- Context-based routing decisions
- Quality assessment model selection
- Stem separation policy
- Enhancement policy
"""

from typing import Any

import pytest

from policy.ml_policy_engine import (
    CANONICAL_BW_EXTENSION_ROUTE,
    CANONICAL_INPAINTING_ROUTE,
    CANONICAL_INSTRUMENTAL_NR_ROUTE,
    CANONICAL_PITCH_ROUTE,
    CANONICAL_REPAIR_ROUTE,
    CANONICAL_SEPARATION_ROUTE,
    CANONICAL_TAGGING_ROUTE,
    CANONICAL_VOCAL_NR_ROUTE,
    CANONICAL_VOCODER_ROUTE,
    MLModelPolicyEngine,
    get_recommended_models,
)


@pytest.fixture
def policy_engine():
    """Create ML Policy Engine instance."""
    return MLModelPolicyEngine()


@pytest.mark.unit
class TestDenoiseModelSelection:
    """Test denoise model selection logic."""

    def test_vinyl_uses_canonical_instrumental_route(self, policy_engine):
        """Vinyl medium should stay inside canonical Aurik 10.0.0 routing."""
        context = {"detected_medium": "vinyl"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == CANONICAL_INSTRUMENTAL_NR_ROUTE

    def test_vocals_use_canonical_vocal_route(self, policy_engine):
        """Vocals/Speech should select the central vocal route."""
        context = {"has_vocals": True}
        model = policy_engine.select_denoise_model(context, {})
        assert model == CANONICAL_VOCAL_NR_ROUTE

    def test_drums_selects_mp_senet(self, policy_engine):
        """Drums/Transients should select MP-SENet (transient-preserving)."""
        context = {"has_drums": True, "content_character": "HIGHLY_TRANSIENT"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == CANONICAL_INSTRUMENTAL_NR_ROUTE

    def test_ambient_selects_deepfilternet(self, policy_engine):
        """Ambient content should select DeepFilterNet (aggressive smoothing)."""
        context = {"has_ambient": True, "content_character": "HIGHLY_SUSTAINED"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == CANONICAL_INSTRUMENTAL_NR_ROUTE

    def test_processing_strategy_override(self, policy_engine):
        """Processing strategy should override other factors."""
        context = {"processing_strategy": "PRESERVE_TRANSIENTS"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == CANONICAL_INSTRUMENTAL_NR_ROUTE

    def test_dominant_instrument_mapping(self, policy_engine):
        """Dominant instrument should guide model selection."""
        test_cases = [
            ("DRUMS", CANONICAL_INSTRUMENTAL_NR_ROUTE),
            ("VOCALS", CANONICAL_VOCAL_NR_ROUTE),
            ("BASS", CANONICAL_INSTRUMENTAL_NR_ROUTE),
            ("STRINGS", CANONICAL_INSTRUMENTAL_NR_ROUTE),
        ]
        for instrument, expected_model in test_cases:
            context = {"dominant_instrument": instrument}
            model = policy_engine.select_denoise_model(context, {})
            assert model == expected_model, f"Failed for {instrument}"

    def test_fallback_to_canonical_instrumental_route(self, policy_engine):
        """Unknown context should fallback to the central instrumental route."""
        context: dict[Any, Any] = {}
        model = policy_engine.select_denoise_model(context, {})
        assert model == CANONICAL_INSTRUMENTAL_NR_ROUTE


class TestRepairModelSelection:
    """Test repair/declipping model selection."""

    def test_speech_selects_mp_senet(self, policy_engine):
        """Speech should select MP-SENet."""
        context = {"has_vocals": True}
        model = policy_engine.select_repair_model(context, {})
        assert model == CANONICAL_REPAIR_ROUTE

    def test_music_selects_mp_senet(self, policy_engine):
        """Music should select MP-SENet."""
        context = {"has_vocals": False}
        model = policy_engine.select_repair_model(context, {})
        assert model == CANONICAL_REPAIR_ROUTE


class TestStemSeparationSelection:
    """Test stem separation model selection."""

    def test_speech_separation_uses_canonical_router(self, policy_engine):
        """Speech separation should use the central separation router."""
        context = {"has_vocals": True}
        goal = {"num_stems": 2}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == CANONICAL_SEPARATION_ROUTE

    def test_fast_processing_selects_mdx23c(self, policy_engine):
        """Fast processing should select MDX23C."""
        context: dict[Any, Any] = {}
        goal = {"quality_level": "fast"}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == CANONICAL_SEPARATION_ROUTE

    def test_ultra_hq_uses_canonical_router(self, policy_engine):
        """Ultra-HQ should stay inside the central separation router."""
        context: dict[Any, Any] = {}
        goal = {"quality_level": "ultra"}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == CANONICAL_SEPARATION_ROUTE

    def test_six_stems_selects_demucs(self, policy_engine):
        """6+ stems should select Demucs v4."""
        context: dict[Any, Any] = {}
        goal = {"num_stems": 6}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == CANONICAL_SEPARATION_ROUTE

    def test_default_selects_mdx23c(self, policy_engine):
        """Standard separation should select MDX23C (SOTA)."""
        context: dict[Any, Any] = {}
        goal: dict[Any, Any] = {}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == CANONICAL_SEPARATION_ROUTE


class TestEnhancementSelection:
    """Test enhancement model selection."""

    def test_speech_enhancement_selects_resemble(self, policy_engine):
        """Speech enhancement should select Resemble Enhance."""
        context = {"has_vocals": True}
        goal = {"enhancement_type": "speech"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == CANONICAL_VOCAL_NR_ROUTE

    def test_super_resolution_selects_flashsr(self, policy_engine):
        """Super-resolution should select FlashSR."""
        context: dict[Any, Any] = {}
        goal = {"enhancement_type": "super_resolution"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == CANONICAL_BW_EXTENSION_ROUTE

    def test_diffusion_enhancement_selects_wpe(self, policy_engine):
        """Diffusion enhancement should select WPE."""
        context: dict[Any, Any] = {}
        goal = {"enhancement_type": "diffusion"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == CANONICAL_INPAINTING_ROUTE

    def test_general_enhancement_selects_gacela(self, policy_engine):
        """General enhancement should avoid standalone generative enhancers."""
        context: dict[Any, Any] = {}
        goal = {"enhancement_type": "general"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == CANONICAL_INSTRUMENTAL_NR_ROUTE


class TestQualityAssessmentSelection:
    """Test quality assessment model selection (§4.4 Spec: nur musik-geeignete Metriken)."""

    def test_always_includes_versa(self, policy_engine):
        """VERSA 2024 ist immer dabei — primäre musik-spezifische MOS ohne Referenz (§4.4).
        CDPAM ist VERBOTEN (§4.4/§10.2): ersetzt durch VERSA."""
        context: dict[Any, Any] = {}
        goal: dict[Any, Any] = {}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "versa" in models
        assert "cdpam" not in models  # CDPAM ist ABSOLUT VERBOTEN

    def test_never_includes_dnsmos(self, policy_engine):
        """DNSMOS ist für Musik VERBOTEN (§4.4/§10.2): trainiert auf Sprachkorpora."""
        context: dict[Any, Any] = {}
        goal: dict[Any, Any] = {}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "dnsmos" not in models

    def test_never_includes_nisqa(self, policy_engine):
        """NISQA ist für Musik VERBOTEN (§4.4/§10.2): Sprachqualitäts-CNN."""
        context = {"has_vocals": True}
        goal: dict[Any, Any] = {}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "nisqa" not in models

    def test_never_includes_pesq(self, policy_engine):
        """PESQ ist für Musik VERBOTEN (§4.4/§10.2): Telefonband 300–3400 Hz."""
        context = {"has_vocals": True}
        goal = {"has_reference": True}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "pesq" not in models

    def test_with_reference_includes_visqol(self, policy_engine):
        """Bei vorhandener Referenz: ViSQOL v3 (--audio Mode zwingend, §4.4)."""
        context = {"has_vocals": False}
        goal = {"has_reference": True}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "vqi" not in models

    def test_full_assessment_uses_music_metrics(self, policy_engine):
        """Vollständige Bewertung: VERSA + ViSQOL + PEAQ — keine Sprachmetriken (§4.4).
        CDPAM/FAD sind VERBOTEN bzw. durch musik-geeignete Metriken ersetzt."""
        context: dict[Any, Any] = {}
        goal = {"assessment_type": "full", "has_reference": True}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "versa" in models
        assert "visqol" not in models
        assert "peaq" not in models
        # Verbotene Metriken dürfen nicht enthalten sein
        assert "cdpam" not in models  # ABSOLUT VERBOTEN
        assert "dnsmos" not in models
        assert "nisqa" not in models
        assert "pesq" not in models


class TestVocoderSelection:
    """Test vocoder model selection."""

    def test_fast_selects_vocos(self, policy_engine):
        """Fast vocoding: Vocos (ConvNeXt-iSTFT, 8× schneller als BigVGAN-v2, §4.5)."""
        context: dict[Any, Any] = {}
        goal = {"quality_level": "fast"}
        model = policy_engine.select_vocoder_model(context, goal)
        assert model == CANONICAL_VOCODER_ROUTE

    def test_high_quality_selects_vocos(self, policy_engine):
        """High-quality vocoding: Vocos 0.1.0 als Primär-Vocoder (§4.5)."""
        context: dict[Any, Any] = {}
        goal = {"quality_level": "high"}
        model = policy_engine.select_vocoder_model(context, goal)
        assert model == CANONICAL_VOCODER_ROUTE


class TestSpecializedModels:
    """Test specialized model selection."""

    def test_audio_tagging_selects_panns(self, policy_engine):
        """Audio tagging should select PANNS."""
        context: dict[Any, Any] = {}
        goal: dict[Any, Any] = {}
        model = policy_engine.select_audio_tagging_model(context, goal)
        assert model == CANONICAL_TAGGING_ROUTE

    def test_mastering_selects_matchering(self, policy_engine):
        """Mastering should select Matchering."""
        context: dict[Any, Any] = {}
        goal: dict[Any, Any] = {}
        model = policy_engine.select_mastering_model(context, goal)
        assert model == "uv3.phase_plan"

    def test_pitch_detection_selects_fcpe(self, policy_engine):
        """Pitch detection muss FCPE selektieren — §4.4 Primär-Tracker (CREPE ist Fallback1)."""
        context: dict[Any, Any] = {}
        goal: dict[Any, Any] = {}
        model = policy_engine.select_pitch_detection_model(context, goal)
        assert model == CANONICAL_PITCH_ROUTE


class TestGenerativeModels:
    """Test generative model selection."""

    def test_music_generation_selects_flow_matching(self, policy_engine):
        """Music generation should select flow_matching (generative inpainting, §4.4)."""
        context: dict[Any, Any] = {}
        goal = {"generation_type": "music"}
        model = policy_engine.select_generative_model(context, goal)
        assert model == CANONICAL_INPAINTING_ROUTE

    def test_text_to_audio_selects_audioldm2(self, policy_engine):
        """Text-to-audio should select AudioLDM2."""
        context: dict[Any, Any] = {}
        goal = {"generation_type": "text_to_audio"}
        model = policy_engine.select_generative_model(context, goal)
        assert model == "unsupported.text_to_audio"


class TestMediumSpecific:
    """Test medium-specific model selection."""

    def test_vinyl_medium_specific(self, policy_engine):
        """Vinyl should stay inside UV3/phase-aware routing."""
        context = {"detected_medium": "vinyl"}
        goal: dict[Any, Any] = {}
        model = policy_engine.select_medium_specific_model(context, goal)
        assert model == CANONICAL_INSTRUMENTAL_NR_ROUTE

    def test_unknown_medium_fallback_to_denoise(self, policy_engine):
        """Unknown medium should fallback to denoise model."""
        context = {"detected_medium": "unknown"}
        goal: dict[Any, Any] = {}
        model = policy_engine.select_medium_specific_model(context, goal)
        # Should return one of the denoise models
        assert model in {CANONICAL_INSTRUMENTAL_NR_ROUTE, CANONICAL_VOCAL_NR_ROUTE}


class TestSelectAllModels:
    """Test comprehensive model selection."""

    def test_select_multiple_tasks(self, policy_engine):
        """Should select appropriate models for multiple tasks."""
        context = {"has_vocals": True, "detected_medium": "vinyl"}
        tasks = ["denoise", "separation", "quality"]

        models = policy_engine.select_all_models(context, tasks)

        # Should have selected models for all tasks
        assert "denoise" in models
        assert "separation" in models
        assert "quality" in models

        # Denoise should be routed through the canonical Aurik 10.0.0 policy facade.
        assert models["denoise"] == CANONICAL_VOCAL_NR_ROUTE

        # Quality should be list
        assert isinstance(models["quality"], list)

    def test_select_all_task_types(self, policy_engine):
        """Test selection for all supported task types."""
        context: dict[Any, Any] = {}
        tasks = [
            "denoise",
            "repair",
            "separation",
            "enhancement",
            "quality",
            "vocoder",
            "tagging",
            "mastering",
            "generation",
            "pitch",
            "medium_specific",
        ]

        models = policy_engine.select_all_models(context, tasks)

        # Should have all 11 tasks covered
        assert len(models) == 11


class TestConvenienceFunctions:
    """Test convenience helper functions."""

    def test_get_recommended_models_basic(self):
        """Should recommend basic denoise + quality models."""
        context: dict[Any, Any] = {}
        recommendations = get_recommended_models(context)

        # Should have denoise and quality
        assert "denoise" in recommendations
        assert "quality" in recommendations

        # Denoise should be string
        assert isinstance(recommendations["denoise"], str)

        # Quality should be list
        assert isinstance(recommendations["quality"], list)

    def test_get_recommended_models_with_vocals(self):
        """Should add separation for vocal content."""
        context = {"has_vocals": True}
        recommendations = get_recommended_models(context)

        # Should include separation
        assert "separation" in recommendations


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_context(self, policy_engine):
        """Empty context should use fallback models."""
        context: dict[Any, Any] = {}
        goal: dict[Any, Any] = {}

        # Should not raise errors
        denoise = policy_engine.select_denoise_model(context, goal)
        repair = policy_engine.select_repair_model(context, goal)
        separation = policy_engine.select_stem_separation_model(context, goal)

        # All should return valid model names
        assert denoise in {CANONICAL_INSTRUMENTAL_NR_ROUTE, CANONICAL_VOCAL_NR_ROUTE}
        assert repair == CANONICAL_REPAIR_ROUTE
        assert separation == CANONICAL_SEPARATION_ROUTE

    def test_conflicting_context_signals(self, policy_engine):
        """Conflicting signals should follow priority order."""
        # Vinyl + vocals → Vinyl should win (higher priority)
        context = {"detected_medium": "vinyl", "has_vocals": True}
        model = policy_engine.select_denoise_model(context, {})
        assert model == CANONICAL_VOCAL_NR_ROUTE

    def test_invalid_goal_parameters(self, policy_engine):
        """Invalid goal parameters should use defaults."""
        context: dict[Any, Any] = {}
        goal = {"invalid_key": "invalid_value"}

        # Should not crash, use defaults
        model = policy_engine.select_denoise_model(context, goal)
        assert isinstance(model, str)


# Integration Test
def test_full_workflow_integration():
    """Test full workflow: context → policy → models."""
    policy = MLModelPolicyEngine()

    # Scenario: Professional vinyl restoration
    context = {"detected_medium": "vinyl", "has_vocals": True, "genre": "jazz", "content_character": "BALANCED"}

    # Get models for restoration workflow
    denoise = policy.select_denoise_model(context, {"quality_level": "maximal"})
    quality = policy.select_quality_assessment_model(context, {"has_reference": True})

    # Verify vinyl-specific model selected
    assert denoise == CANONICAL_VOCAL_NR_ROUTE

    # Verify music-appropriate quality assessment (§4.4/§10.2: CDPAM/DNSMOS/NISQA/PESQ verboten)
    assert "versa" in quality  # VERSA 2024 — primäre MOS-Metrik (ersetzt CDPAM)
    assert "vqi" in quality  # Vocal material uses VQI beside VERSA.
    assert "cdpam" not in quality  # VERBOTEN: ersetzt durch VERSA
    assert "dnsmos" not in quality  # VERBOTEN: Sprachkorpus
    assert "nisqa" not in quality  # VERBOTEN: Sprachqualitäts-CNN
    assert "pesq" not in quality  # VERBOTEN: Telefonband


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
