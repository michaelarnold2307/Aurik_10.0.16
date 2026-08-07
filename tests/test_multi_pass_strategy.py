"""
Tests für Multi-Pass Processing Strategy & Adaptive Selection

Test Coverage:
1. ProcessingVariant creation und variants generation
2. ObjectiveScorer mit verschiedenen Metriken
3. ConfidenceCalculator für top-2 variance
4. MultiPassEngine End-to-End
5. Integration mit verschiedenen ProcessingModes

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import numpy as np
import pytest

from backend.core.multi_pass_strategy import (
    ConfidenceCalculator,
    MultiPassEngine,
    ObjectiveScore,
    ObjectiveScorer,
    ProcessingVariant,
    VariantStrategy,
    create_default_variants,
)
from backend.core.processing_modes import ProcessingMode


@pytest.mark.unit
class TestProcessingVariant:
    """Test ProcessingVariant creation und Konfiguration."""

    def test_create_conservative(self):
        """Test conservative variant creation."""
        variant = ProcessingVariant.create_conservative(ProcessingMode.RESTORATION)

        assert variant.name == "conservative"
        assert variant.strategy == VariantStrategy.CONSERVATIVE
        assert variant.config.denoise_strength < 0.5  # Ultra-conservative
        assert variant.config.preserve_breaths is True
        assert variant.config.preserve_room_tone is True
        assert variant.weight == 1.0

    def test_create_balanced(self):
        """Test balanced variant creation."""
        variant = ProcessingVariant.create_balanced(ProcessingMode.RESTORATION)

        assert variant.name == "balanced"
        assert variant.strategy == VariantStrategy.BALANCED
        assert variant.config is not None
        assert 0.0 <= variant.config.denoise_strength <= 1.0

    def test_create_aggressive(self):
        """Test aggressive variant creation."""
        variant = ProcessingVariant.create_aggressive(ProcessingMode.STUDIO_2026)

        assert variant.name == "aggressive"
        assert variant.strategy == VariantStrategy.AGGRESSIVE
        # Aggressive sollte höhere werte haben
        assert variant.config.enhancement_strength > 0.5

    def test_create_gentle_denoise(self):
        """Test gentle denoise variant."""
        variant = ProcessingVariant.create_gentle_denoise()

        assert variant.name == "gentle_denoise"
        assert variant.strategy == VariantStrategy.GENTLE_DENOISE
        assert variant.config.denoise_strength < 0.2  # Ultra-gentle
        assert variant.config.preserve_analog_character is True

    def test_create_strong_dynamics(self):
        """Test strong dynamics variant."""
        variant = ProcessingVariant.create_strong_dynamics()

        assert variant.name == "strong_dynamics"
        assert variant.strategy == VariantStrategy.STRONG_DYNAMICS
        assert variant.config.compression_ratio > 4.0  # Strong compression
        assert variant.config.target_lufs <= -14.0  # type: ignore[operator]  # Streaming standard

    def test_variant_to_dict(self):
        """Test variant serialization."""
        variant = ProcessingVariant.create_balanced()
        variant_dict = variant.to_dict()

        assert "name" in variant_dict
        assert "strategy" in variant_dict
        assert "config" in variant_dict
        assert variant_dict["strategy"] == "balanced"

    def test_variant_different_parameters(self):
        """Test dass verschiedene Varianten verschiedene Parameter haben."""
        conservative = ProcessingVariant.create_conservative()
        aggressive = ProcessingVariant.create_aggressive()

        # Conservative sollte schwächere Parameter haben
        assert conservative.config.denoise_strength < aggressive.config.denoise_strength
        assert conservative.config.compression_ratio < aggressive.config.compression_ratio


class TestObjectiveScore:
    """Test ObjectiveScore dataclass."""

    def test_objective_score_creation(self):
        """Test ObjectiveScore creation."""
        score = ObjectiveScore(
            cdpam_score=0.25,
            dnsmos_score=3.8,
            musical_goals_avg=0.75,
            snr_db=25.0,
            thd_percent=0.5,
            composite_score=0.82,
            confidence=0.9,
            variant_name="test_variant",
        )

        assert score.variant_name == "test_variant"
        assert score.composite_score == 0.82
        assert score.confidence == 0.9

    def test_objective_score_to_dict(self):
        """Test ObjectiveScore serialization."""
        score = ObjectiveScore(variant_name="test", composite_score=0.75)

        score_dict = score.to_dict()
        assert "variant_name" in score_dict
        assert "composite_score" in score_dict
        assert score_dict["variant_name"] == "test"

    def test_objective_score_str(self):
        """Test ObjectiveScore string representation."""
        score = ObjectiveScore(
            variant_name="balanced", composite_score=0.80, confidence=0.85, cdpam_score=0.28, dnsmos_score=3.9
        )

        score_str = str(score)
        assert "balanced" in score_str
        assert "0.800" in score_str  # composite
        assert "0.85" in score_str  # confidence


class TestObjectiveScorer:
    """Test ObjectiveScorer mit mock audio."""

    @pytest.fixture
    def mock_audio(self):
        """Generate mock audio signal."""
        duration = 2.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Simple sine wave + noise
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        audio += 0.05 * np.random.randn(len(t))  # Small noise

        return audio, sample_rate

    def test_scorer_initialization(self):
        """Test ObjectiveScorer initialization.

        §10.2/§4.4: DNSMOS ist für Musik VERBOTEN — per Spec immer enable_dnsmos=False.
        VERSA (non-reference MOS) und Musical Goals sind die primären Musik-Qualitätsmetriken.
        """
        # Standard-Konfiguration: VERSA + Musical Goals (§4.4: VERSA ersetzt CDPAM)
        scorer = ObjectiveScorer(enable_cdpam=True, enable_dnsmos=False, enable_musical_goals=True)

        assert scorer.enable_cdpam is True
        assert scorer.enable_dnsmos is False  # §10.2: DNSMOS VERBOTEN für Musik
        assert scorer.enable_musical_goals is True

    def test_scorer_without_plugins(self):
        """Test ObjectiveScorer ohne externe plugins."""
        # Disable external plugins (für CI ohne VERSA/DNSMOS)
        scorer = ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=True)

        # §4.4: versa_plugin ersetzt cdpam_plugin
        assert scorer.versa_plugin is None
        assert scorer.dnsmos_plugin is None

    def test_score_audio_basic(self, mock_audio):
        """Test basic audio scoring."""
        audio, sr = mock_audio

        # Scorer ohne externe plugins (für robuste tests)
        scorer = ObjectiveScorer(enable_versa=False, enable_dnsmos=False, enable_musical_goals=False)

        score = scorer.score(audio=audio, sample_rate=sr, variant_name="test_variant")

        assert score.variant_name == "test_variant"
        assert 0.0 <= score.composite_score <= 1.0

    def test_score_with_reference(self, mock_audio):
        """Test scoring mit reference audio (für CDPAM)."""
        audio, sr = mock_audio
        reference = audio.copy()

        scorer = ObjectiveScorer(
            enable_cdpam=False,
            enable_dnsmos=False,
            enable_musical_goals=False,  # Mock, da CDPAM evtl nicht installiert
        )

        score = scorer.score(audio=audio, sample_rate=sr, variant_name="test", reference_audio=reference)

        assert score is not None

    def test_composite_calculation(self):
        """Test composite score calculation."""
        scorer = ObjectiveScorer()

        # Mock score mit bekannten Werten (§10.2: DNSMOS VERBOTEN, dnsmos_score=0.0)
        test_score = ObjectiveScore(
            cdpam_score=0.2,  # Good (niedrig) — CDPAM als Musik-Wahrnehmungsmetrik (§4.4)
            dnsmos_score=0.0,  # §10.2: DNSMOS VERBOTEN für Musik — kein Beitrag zum Composite
            musical_goals_avg=0.8,  # Good (hoch) — 15 Musical Goals (§1.2)
            snr_db=30.0,  # Good (hoch)
            thd_percent=0.5,  # Good (niedrig)
        )

        composite = scorer._calculate_composite(test_score)

        # Sollte relativ hoch sein da alle Metriken gut sind
        assert composite > 0.7
        assert 0.0 <= composite <= 1.0

    def test_versa_receives_proper_mono_from_channel_first(self):
        """Channel-first audio [C, N] must be converted to usable mono before VERSA."""

        class _FakeVersa:
            def score(self, audio, sample_rate):
                assert sample_rate == 48000
                assert isinstance(audio, np.ndarray)
                assert audio.ndim == 1
                assert audio.size > 1000

                class _R:
                    mos = 3.2

                return _R()

        scorer = ObjectiveScorer(enable_versa=False, enable_dnsmos=False, enable_musical_goals=False)
        scorer.versa_plugin = _FakeVersa()  # type: ignore[assignment]
        scorer.enable_versa = True

        n = 48000
        ch1 = 0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, n, endpoint=False))
        ch2 = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, n, endpoint=False))
        audio_cf = np.stack([ch1, ch2], axis=0).astype(np.float32)  # [C, N]

        score = scorer.score(audio=audio_cf, sample_rate=48000, variant_name="cf")
        assert score.versa_active is True
        assert 0.0 <= score.versa_score <= 1.0


class TestConfidenceCalculator:
    """Test ConfidenceCalculator für variance-based confidence."""

    def test_confidence_high_gap(self):
        """Test confidence mit großem Gap zwischen top-2."""
        score1 = ObjectiveScore(composite_score=0.90, variant_name="best")
        score2 = ObjectiveScore(composite_score=0.70, variant_name="second")
        score3 = ObjectiveScore(composite_score=0.65, variant_name="third")

        scores = [score1, score2, score3]

        calc = ConfidenceCalculator()
        confidence = calc.calculate_confidence(scores, score1)

        # Large gap → high confidence
        assert confidence > 0.7
        assert 0.0 <= confidence <= 1.0

    def test_confidence_small_gap(self):
        """Test confidence mit kleinem Gap."""
        score1 = ObjectiveScore(composite_score=0.75, variant_name="best")
        score2 = ObjectiveScore(composite_score=0.73, variant_name="second")
        score3 = ObjectiveScore(composite_score=0.72, variant_name="third")

        scores = [score1, score2, score3]

        calc = ConfidenceCalculator()
        confidence = calc.calculate_confidence(scores, score1)

        # Small gap → lower confidence
        assert confidence < 0.8

    def test_confidence_single_variant(self):
        """Test confidence mit nur 1 Variante."""
        score1 = ObjectiveScore(composite_score=0.85, variant_name="only")

        calc = ConfidenceCalculator()
        confidence = calc.calculate_confidence([score1], score1)

        # Not enough data → medium confidence
        assert confidence == 0.5


class TestMultiPassEngine:
    """Test MultiPassEngine End-to-End."""

    @pytest.fixture
    def mock_audio(self):
        """Generate mock audio."""
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio += 0.05 * np.random.randn(len(t))
        return audio, sample_rate

    @pytest.fixture
    def mock_process_func(self):
        """Mock processing function für tests."""

        def process(audio, sample_rate, config):
            # Simplified mock: apply small gain based on config strength
            gain = 1.0 + (config.enhancement_strength * 0.1)
            return audio * gain

        return process

    def test_engine_initialization(self):
        """Test MultiPassEngine initialization."""
        engine = MultiPassEngine()

        assert engine.scorer is not None
        assert engine.confidence_calc is not None

    def test_engine_with_custom_scorer(self):
        """Test engine mit custom scorer."""
        scorer = ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False)
        engine = MultiPassEngine(scorer=scorer)

        assert engine.scorer is scorer

    def test_process_with_3_variants(self, mock_audio, mock_process_func):
        """Test processing mit 3 Varianten."""
        audio, sr = mock_audio

        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        # Validate result structure
        assert "audio" in result
        assert "variant_name" in result
        assert "confidence" in result
        assert "composite_score" in result
        assert "all_scores" in result

        # Validate result values
        assert result["audio"] is not None
        assert len(result["audio"]) > 0
        assert result["variant_name"] in ["conservative", "balanced", "aggressive"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["composite_score"] <= 1.0
        assert len(result["all_scores"]) == 3

    def test_process_with_5_variants(self, mock_audio, mock_process_func):
        """Test processing mit 5 Varianten."""
        audio, sr = mock_audio

        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=5)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        assert len(result["all_scores"]) == 5
        assert len(result["processing_times"]) == 5

    def test_process_no_variants_error(self, mock_audio):
        """Test dass leere Varianten-Liste Error wirft."""
        audio, sr = mock_audio

        engine = MultiPassEngine()

        with pytest.raises(ValueError, match="Mindestens 1 ProcessingVariant"):
            engine.process_with_variants(audio=audio, sample_rate=sr, variants=[])

    def test_process_with_reference(self, mock_audio, mock_process_func):
        """Test processing mit reference audio."""
        audio, sr = mock_audio
        reference = audio.copy()

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine(scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False))  # Mock

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, reference_audio=reference, process_func=mock_process_func
        )

        assert result is not None

    def test_scores_sorted_by_composite(self, mock_audio, mock_process_func):
        """Test dass Scores sortiert sind (best first)."""
        audio, sr = mock_audio

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        all_scores = result["all_scores"]

        # Sollte descending sortiert sein
        for i in range(len(all_scores) - 1):
            assert all_scores[i].composite_score >= all_scores[i + 1].composite_score

    def test_processing_times_recorded(self, mock_audio, mock_process_func):
        """Test dass processing times recorded werden."""
        audio, sr = mock_audio

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine()

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        times = result["processing_times"]

        assert len(times) == 3
        assert "conservative" in times
        assert "balanced" in times
        assert "aggressive" in times

        # All times should be > 0
        for variant_name, time_sec in times.items():
            assert time_sec > 0.0

    def test_default_process_func_maps_variant_to_restore_mode(self, mock_audio):
        """Default path must not run all variants with the same fixed restore mode.

        Mapping expectations per _derive_restore_mode thresholds:
          naturalness_first (denoise=0.08) → "fast"
          gentle_denoise    (denoise=0.15) → "balanced"
          balanced_variant  (denoise=0.30) → "restoration"
          strong_dynamics   (comp=6.0)     → "maximum"
        """

        class _FakeResult:
            def __init__(self, audio: np.ndarray) -> None:
                self.audio = audio

        class _FakeRestorer:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def restore(self, audio, sample_rate, mode="restoration", **kwargs):
                self.calls.append(mode)
                return _FakeResult(audio)

        audio, sr = mock_audio
        engine = MultiPassEngine()
        engine._restorer = _FakeRestorer()

        natural = ProcessingVariant.create_naturalness_first().config  # denoise=0.08 → fast
        gentle = ProcessingVariant.create_gentle_denoise().config  # denoise=0.15 → balanced
        balanced_cfg = ProcessingVariant.create_balanced().config  # denoise=0.30 → restoration
        strong = ProcessingVariant.create_strong_dynamics().config  # comp=6.0    → maximum

        _ = engine._default_process_func(audio, sr, natural)
        _ = engine._default_process_func(audio, sr, gentle)
        _ = engine._default_process_func(audio, sr, balanced_cfg)
        _ = engine._default_process_func(audio, sr, strong)

        calls = engine._restorer.calls  # type: ignore[union-attr]
        assert calls[0] == "fast", f"naturalness_first expected fast, got {calls[0]}"
        assert calls[1] == "balanced", f"gentle_denoise expected balanced, got {calls[1]}"
        assert calls[2] == "restoration", f"balanced expected restoration, got {calls[2]}"
        assert calls[3] == "maximum", f"strong_dynamics expected maximum, got {calls[3]}"

    def test_default_process_func_uses_full_program_length(self, mock_audio):
        """MultiPass default processing must evaluate full program, not only first 10s."""

        class _FakeResult:
            def __init__(self, audio: np.ndarray) -> None:
                self.audio = audio

        class _FakeRestorer:
            def __init__(self) -> None:
                self.lengths: list[int] = []

            def restore(self, audio, sample_rate, mode="restoration", **kwargs):
                self.lengths.append(len(audio))
                return _FakeResult(audio)

        audio, sr = mock_audio
        long_audio = np.tile(audio, 8)  # deutlich länger als 10 s

        engine = MultiPassEngine()
        engine._restorer = _FakeRestorer()

        _ = engine._default_process_func(long_audio, sr, ProcessingVariant.create_balanced().config)
        assert engine._restorer.lengths[0] == len(long_audio)  # type: ignore[union-attr]

    def test_default_process_func_raises_on_missing_audio_payload(self, mock_audio):
        """A variant without audio payload must fail hard (no silent original fallback)."""

        class _FakeResult:
            def __init__(self) -> None:
                self.audio = None

        class _FakeRestorer:
            def restore(self, audio, sample_rate, mode="restoration", **kwargs):
                return _FakeResult()

        audio, sr = mock_audio
        engine = MultiPassEngine()
        engine._restorer = _FakeRestorer()

        with pytest.raises(RuntimeError, match="MultiPass default processing failed"):
            _ = engine._default_process_func(audio, sr, ProcessingVariant.create_balanced().config)


class TestCreateDefaultVariants:
    """Test create_default_variants helper function."""

    def test_create_3_variants(self):
        """Test creating 3 variants."""
        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

        assert len(variants) == 3
        assert variants[0].name == "conservative"
        assert variants[1].name == "balanced"
        assert variants[2].name == "aggressive"

    def test_create_5_variants(self):
        """Test creating 5 variants."""
        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=5)

        assert len(variants) == 5
        names = [v.name for v in variants]
        assert "conservative" in names
        assert "gentle_denoise" in names
        assert "balanced" in names
        assert "aggressive" in names
        assert "strong_dynamics" in names

    def test_create_invalid_num_variants(self):
        """Test dass invalid num_variants Error wirft."""
        with pytest.raises(ValueError):
            create_default_variants(num_variants=4)  # Only 3 or 5 supported


class TestIntegrationScenarios:
    """Integration tests für realistic scenarios."""

    @pytest.fixture
    def realistic_audio(self):
        """Generate realistic test audio mit some "defects"."""
        duration = 3.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Music-like signal: fundamental + harmonics
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3
        audio += 0.2 * np.sin(2 * np.pi * 440 * t)  # A4
        audio += 0.1 * np.sin(2 * np.pi * 880 * t)  # A5

        # Add "defects"
        audio += 0.08 * np.random.randn(len(t))  # Noise

        # Add some clicks (simplified)
        click_positions = np.random.choice(len(audio), size=10, replace=False)
        audio[click_positions] *= 3.0

        return audio, sample_rate

    def test_realistic_restoration_scenario(self, realistic_audio):
        """Test realistic restoration scenario."""
        audio, sr = realistic_audio

        def simple_denoise_process(audio, sample_rate, config):
            # Simplified denoising: High-pass filter
            from scipy import signal

            strength = config.denoise_strength
            cutoff = 80 * (1 + strength)  # Higher strength → higher cutoff

            sos = signal.butter(3, cutoff, "high", fs=sample_rate, output="sos")

            return signal.sosfilt(sos, audio)

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=simple_denoise_process
        )

        # Should select one variant
        assert result["variant_name"] is not None
        assert result["confidence"] > 0.0

        # Output should be similar length
        assert len(result["audio"]) == len(audio)

    def test_different_processing_modes(self, realistic_audio):
        """Test variants mit verschiedenen ProcessingModes."""
        audio, sr = realistic_audio

        def mock_process(audio, sr, config):
            return audio * 0.99  # Simplified

        # RESTORATION mode
        variants_restore = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

        # STUDIO_2026 mode
        variants_studio = create_default_variants(base_mode=ProcessingMode.STUDIO_2026, num_variants=3)

        # Configs should differ
        assert variants_restore[0].config.target_lufs != variants_studio[0].config.target_lufs


# === Performance & Edge Cases ===


class TestEdgeCases:
    """Test edge cases und error handling."""

    def test_silent_audio(self):
        """Test mit silent audio."""
        audio = np.zeros(16000 * 2)  # 2 sec silence
        sr = 16000

        variants = create_default_variants(num_variants=3)

        def mock_process(audio, sr, config):
            return audio

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(audio=audio, sample_rate=sr, variants=variants, process_func=mock_process)

        # Should still work (wählt irgendeine Variante)
        assert result["variant_name"] is not None

    def test_very_short_audio(self):
        """Test mit sehr kurzer audio."""
        audio = np.random.randn(1000)  # ~62ms @ 16kHz
        sr = 16000

        variants = [ProcessingVariant.create_balanced()]

        def mock_process(audio, sr, config):
            return audio

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(audio=audio, sample_rate=sr, variants=variants, process_func=mock_process)

        assert len(result["audio"]) == len(audio)

    def test_clipped_audio(self):
        """Test mit clipped audio."""
        audio = np.clip(np.random.randn(16000 * 2) * 2.0, -1.0, 1.0)
        sr = 16000

        variants = create_default_variants(num_variants=3)

        def mock_process(audio, sr, config):
            return audio * 0.9  # Simple gain reduction

        engine = MultiPassEngine()

        result = engine.process_with_variants(audio=audio, sample_rate=sr, variants=variants, process_func=mock_process)

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ── IAQS (Intrinsic Audio Quality Scorer) Tests ──


class TestIAQS:
    def test_iaqs_import_and_instantiate(self):
        from backend.core.intrinsic_audio_quality_scorer import IntrinsicQualityScore

        iqs = IntrinsicQualityScore()
        assert iqs is not None

    def test_iaqs_has_required_fields(self):
        from backend.core.intrinsic_audio_quality_scorer import IntrinsicQualityScore

        iqs = IntrinsicQualityScore()
        for field in ["overall", "snr_score", "harmonicity", "transient_clarity"]:
            assert hasattr(iqs, field), f"Missing: {field}"

    def test_iaqs_overall_valid_range(self):
        from backend.core.intrinsic_audio_quality_scorer import IntrinsicQualityScore

        iqs = IntrinsicQualityScore()
        score = getattr(iqs, "overall", 0.0) or 0.0
        assert 0.0 <= score <= 1.0

    def test_iaqs_fields_are_finite(self):
        import numpy as np

        from backend.core.intrinsic_audio_quality_scorer import IntrinsicQualityScore

        iqs = IntrinsicQualityScore()
        for field in ["overall", "snr_score", "harmonicity"]:
            val = getattr(iqs, field, 0.0)
            if isinstance(val, (int, float)):
                assert np.isfinite(val)
