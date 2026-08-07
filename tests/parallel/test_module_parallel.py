"""
Tests for Module Parallel Processing.

Tests the ModuleParallelProcessor and PipelineBuilder classes,
validating dependency resolution, parallel execution, and pipeline building.

Author: AURIK Team
Date: 8. Februar 2026
"""

import time
from typing import Any

import numpy as np
import pytest

from backend.core.parallel.module_parallel import (
    ModuleDependency,
    ModuleInfo,
    ModuleParallelProcessor,
    ModuleResult,
    PipelineBuilder,
)


@pytest.fixture
def sample_audio():
    """Create sample audio."""
    sr = 44100
    duration = 0.1  # 100ms for fast tests
    samples = int(sr * duration)
    audio = np.sin(2 * np.pi * 440 * np.arange(samples) / sr).astype(np.float32)
    return audio, sr


@pytest.fixture
def processor():
    """Create module parallel processor."""
    return ModuleParallelProcessor(max_workers=4, enable_parallel=True)


@pytest.mark.unit
class TestBasicProcessing:
    """Test basic module parallel processing."""

    def test_single_independent_module(self, processor, sample_audio):
        """Test processing with single module."""
        audio, sr = sample_audio

        def gain(audio, sr):
            return audio * 2.0

        modules = [ModuleInfo("gain", gain)]
        result = processor.process(audio, sr, modules)

        np.testing.assert_allclose(result, audio * 2.0, rtol=1e-5)

    def test_two_independent_modules(self, processor, sample_audio):
        """Test parallel processing of independent modules."""
        audio, sr = sample_audio

        def module1(audio, sr):
            return audio * 1.5

        def module2(audio, sr):
            return audio * 2.0

        modules = [ModuleInfo("module1", module1), ModuleInfo("module2", module2)]

        result = processor.process(audio, sr, modules)

        # Both modules should run in parallel on same input
        # Last module's output is returned
        assert result.shape == audio.shape

    def test_sequential_dependent_modules(self, processor, sample_audio):
        """Test sequential processing of dependent modules."""
        audio, sr = sample_audio

        def step1(audio, sr):
            return audio * 2.0

        def step2(audio, sr):
            return audio + 0.1

        modules = [ModuleInfo("step1", step1), ModuleInfo("step2", step2, depends_on=["step1"])]

        result = processor.process(audio, sr, modules)

        # step2 depends on step1, so should use step1's output
        assert result.shape == audio.shape


class TestDependencyResolution:
    """Test dependency resolution and phase building."""

    def test_build_simple_phases(self, processor):
        """Test building simple execution phases."""

        def dummy(audio, sr):
            return audio

        modules = [ModuleInfo("a", dummy), ModuleInfo("b", dummy), ModuleInfo("c", dummy, depends_on=["a", "b"])]

        phases = processor._build_execution_phases(modules)

        # Should have 2 phases: [a, b] then [c]
        assert len(phases) == 2
        assert len(phases[0]) == 2  # a and b parallel
        assert len(phases[1]) == 1  # c sequential

    def test_build_complex_phases(self, processor):
        """Test building complex dependency graph."""

        def dummy(audio, sr):
            return audio

        modules = [
            ModuleInfo("a", dummy),
            ModuleInfo("b", dummy),
            ModuleInfo("c", dummy, depends_on=["a"]),
            ModuleInfo("d", dummy, depends_on=["b"]),
            ModuleInfo("e", dummy, depends_on=["c", "d"]),
        ]

        phases = processor._build_execution_phases(modules)

        # Phase 0: [a, b]
        # Phase 1: [c, d]
        # Phase 2: [e]
        assert len(phases) == 3
        assert len(phases[0]) == 2
        assert len(phases[1]) == 2
        assert len(phases[2]) == 1

    def test_circular_dependency_detection(self, processor):
        """Test detection of circular dependencies."""

        def dummy(audio, sr):
            return audio

        modules = [ModuleInfo("a", dummy, depends_on=["b"]), ModuleInfo("b", dummy, depends_on=["a"])]

        with pytest.raises(ValueError, match="Circular dependency"):
            processor._build_execution_phases(modules)

    def test_invalid_dependency(self, processor, sample_audio):
        """Test handling of invalid dependencies."""
        audio, sr = sample_audio

        def dummy(audio, sr):
            return audio

        modules = [ModuleInfo("a", dummy, depends_on=["nonexistent"])]

        with pytest.raises(ValueError, match="Circular dependency|unsatisfied"):
            processor.process(audio, sr, modules)

    def test_priority_based_ordering(self, processor):
        """Test priority-based execution ordering."""

        def dummy(audio, sr):
            return audio

        modules = [
            ModuleInfo("low", dummy, priority=0),
            ModuleInfo("high", dummy, priority=10),
            ModuleInfo("medium", dummy, priority=5),
        ]

        phases = processor._build_execution_phases(modules)

        # Should have 3 phases, ordered by priority
        assert len(phases) == 3
        # Higher priority first
        assert phases[0][0].name == "high"
        assert phases[1][0].name == "medium"
        assert phases[2][0].name == "low"


class TestErrorHandling:
    """Test error handling in module processing."""

    def test_module_raises_exception(self, processor, sample_audio):
        """Test handling of module exceptions."""
        audio, sr = sample_audio

        def failing_module(audio, sr):
            raise RuntimeError("Module failed")

        modules = [ModuleInfo("fail", failing_module)]

        with pytest.raises(RuntimeError, match="Phase 0 failed"):
            processor.process(audio, sr, modules)

    def test_module_returns_none(self, processor, sample_audio):
        """Test handling of module returning None."""
        audio, sr = sample_audio

        def none_module(audio, sr):
            return None

        modules = [ModuleInfo("none", none_module)]

        with pytest.raises(RuntimeError, match="Phase 0 failed"):
            processor.process(audio, sr, modules)

    def test_module_changes_length(self, processor, sample_audio):
        """Test handling of module changing audio length."""
        audio, sr = sample_audio

        def length_change(audio, sr):
            return audio[: len(audio) // 2]

        modules = [ModuleInfo("length", length_change)]

        with pytest.raises(RuntimeError, match="Phase 0 failed"):
            processor.process(audio, sr, modules)

    def test_partial_phase_failure(self, processor, sample_audio):
        """Test handling when some modules in a phase fail."""
        audio, sr = sample_audio

        def good_module(audio, sr):
            return audio * 2.0

        def bad_module(audio, sr):
            raise ValueError("Bad module")

        modules = [ModuleInfo("good", good_module), ModuleInfo("bad", bad_module)]

        with pytest.raises(RuntimeError, match="Phase 0 failed"):
            processor.process(audio, sr, modules)


class TestPerformance:
    """Test performance and parallel execution."""

    def test_parallel_faster_than_sequential(self, sample_audio):
        """Test that parallel execution is faster."""
        audio, sr = sample_audio
        # Use longer audio
        audio = np.tile(audio, 10)

        def slow_module(audio, sr):
            time.sleep(0.01)  # 10ms delay
            return audio * 1.1

        modules = [ModuleInfo("m1", slow_module), ModuleInfo("m2", slow_module), ModuleInfo("m3", slow_module)]

        # Parallel processing
        parallel_proc = ModuleParallelProcessor(enable_parallel=True)
        start = time.time()
        parallel_proc.process(audio, sr, modules)
        parallel_time = time.time() - start

        # Sequential processing
        sequential_proc = ModuleParallelProcessor(enable_parallel=False)
        start = time.time()
        sequential_proc.process(audio, sr, modules)
        sequential_time = time.time() - start

        # Parallel should be faster
        assert parallel_time < sequential_time * 0.9

    def test_stats_tracking(self, processor, sample_audio):
        """Test statistics tracking."""
        audio, sr = sample_audio

        def module(audio, sr):
            return audio

        modules = [ModuleInfo("m1", module), ModuleInfo("m2", module), ModuleInfo("m3", module, depends_on=["m1"])]

        processor.process(audio, sr, modules)

        stats = processor.get_stats()
        assert stats["total_processed"] == 1
        assert stats["parallel_phases"] >= 1
        assert stats["sequential_phases"] >= 0

    def test_reset_stats(self, processor, sample_audio):
        """Test stats reset."""
        audio, sr = sample_audio

        def module(audio, sr):
            return audio

        modules = [ModuleInfo("m", module)]

        processor.process(audio, sr, modules)
        processor.reset_stats()

        stats = processor.get_stats()
        assert stats["total_processed"] == 0
        assert stats["parallel_phases"] == 0


class TestModuleResult:
    """Test ModuleResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        audio = np.ones(1000, dtype=np.float32)
        result = ModuleResult(module_name="test", audio=audio, success=True, processing_time=0.1)

        assert result.module_name == "test"
        assert result.success
        assert result.error is None
        assert len(result.audio) == 1000

    def test_error_result(self):
        """Test error result."""
        audio = np.ones(1000, dtype=np.float32)
        result = ModuleResult(module_name="test", audio=audio, success=False, error="Processing failed")

        assert result.module_name == "test"
        assert not result.success
        assert result.error == "Processing failed"


class TestPipelineBuilder:
    """Test PipelineBuilder helper class."""

    def test_add_single_module(self):
        """Test adding single module."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        builder.add_module("test", process)
        modules = builder.build()

        assert len(modules) == 1
        assert modules[0].name == "test"

    def test_add_multiple_modules(self):
        """Test adding multiple modules."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        builder.add_module("m1", process)
        builder.add_module("m2", process)
        builder.add_module("m3", process)
        modules = builder.build()

        assert len(modules) == 3

    def test_module_with_dependencies(self):
        """Test adding module with dependencies."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        builder.add_module("m1", process)
        builder.add_module("m2", process, depends_on=["m1"])
        modules = builder.build()

        assert modules[1].depends_on == ["m1"]
        assert modules[1].dependency_type == ModuleDependency.DEPENDS_ON_MODULES

    def test_module_with_priority(self):
        """Test adding module with priority."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        builder.add_module("low", process, priority=0)
        builder.add_module("high", process, priority=10)
        modules = builder.build()

        assert modules[0].priority == 0
        assert modules[1].priority == 10

    def test_duplicate_module_name(self):
        """Test error on duplicate module name."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        builder.add_module("test", process)

        with pytest.raises(ValueError, match="already exists"):
            builder.add_module("test", process)

    def test_invalid_dependency(self):
        """Test error on invalid dependency."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        builder.add_module("m1", process, depends_on=["nonexistent"])

        with pytest.raises(ValueError, match="unknown module"):
            builder.build()

    def test_method_chaining(self):
        """Test method chaining."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        modules = (
            builder.add_module("m1", process)
            .add_module("m2", process)
            .add_module("m3", process, depends_on=["m1"])
            .build()
        )

        assert len(modules) == 3

    def test_clear_builder(self):
        """Test clearing builder."""

        def process(audio, sr):
            return audio

        builder = PipelineBuilder()
        builder.add_module("m1", process)
        builder.clear()
        modules = builder.build()

        assert len(modules) == 0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_realistic_restoration_pipeline(self, sample_audio):
        """Test realistic audio restoration pipeline."""
        audio, sr = sample_audio

        # Simulate restoration modules
        def remove_clicks(audio, sr):
            return audio * 0.98  # Slight attenuation

        def remove_crackle(audio, sr):
            return audio * 0.97

        def denoise(audio, sr):
            return audio * 0.95  # More attenuation

        def eq(audio, sr):
            return audio * 1.05

        def compress(audio, sr):
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
            return audio

        # Build pipeline with dependencies
        builder = PipelineBuilder()
        (
            builder.add_module("click_removal", remove_clicks, priority=10)
            .add_module("crackle_removal", remove_crackle, priority=10)
            .add_module("denoise", denoise, depends_on=["click_removal", "crackle_removal"], priority=5)
            .add_module("eq", eq, depends_on=["denoise"], priority=0)
            .add_module("compress", compress, depends_on=["eq"], priority=0)
        )

        modules = builder.build()

        processor = ModuleParallelProcessor()
        result = processor.process(audio, sr, modules)

        # Verify output
        assert result.shape == audio.shape
        assert np.max(np.abs(result)) <= 1.0

        # Verify stats
        stats = processor.get_stats()
        assert stats["total_processed"] == 1
        assert stats["parallel_phases"] >= 1  # click and crackle in parallel

    def test_pipeline_with_metadata(self, sample_audio):
        """Test pipeline with module metadata."""
        audio, sr = sample_audio

        def process(audio, sr):
            return audio * 1.1

        builder = PipelineBuilder()
        (
            builder.add_module("m1", process, metadata={"version": "1.0", "algo": "basic"}).add_module(
                "m2", process, metadata={"version": "2.0", "algo": "advanced"}
            )
        )

        modules = builder.build()

        assert modules[0].metadata["version"] == "1.0"
        assert modules[1].metadata["algo"] == "advanced"

        processor = ModuleParallelProcessor()
        result = processor.process(audio, sr, modules)
        assert result.shape == audio.shape

    def test_empty_pipeline(self, processor, sample_audio):
        """Test processing with no modules."""
        audio, sr = sample_audio

        modules: list[Any] = []
        result = processor.process(audio, sr, modules)

        # Should return copy of original audio
        np.testing.assert_allclose(result, audio, rtol=1e-5, atol=1e-8)
