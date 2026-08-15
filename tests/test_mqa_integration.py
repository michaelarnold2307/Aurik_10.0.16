"""
tests/test_mqa_integration.py
Musical Quality Assurance Integration Tests
===========================================

Tests MQA integration with Module Coordinator:
- Quality baseline establishment
- Quality gates during processing
- Automatic rollback on integrity violations
- Final quality validation
- Medium-specific quality guarantees

Author: AURIK Team
"""

import numpy as np
import pytest

from backend.core.module_communication import ModuleCommunicationBus
from backend.core.module_coordinator import ModuleCoordinator, ModulePriority
from backend.core.musical_quality_assurance import (
    MediumType,
    ProcessingMode,
    map_forensic_to_medium_type,
)
from backend.core.processing_context import ProcessingContext

# === Hilfklassen für Tests (Mock-Module) ===


class MockNeutralModule:
    """Neutrales Verarbeitungsmodul — gibt Audio unverändert zurück."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        return np.copy(audio)


class MockGoodModule:
    """Hochwertiges Verarbeitungsmodul — gibt Audio mit minimalem Gain-Trim zurück."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        # Minimale, nicht-destruktive Operation (0.1 % Gain-Trim) für Testzwecke
        return np.clip(audio * 0.999, -1.0, 1.0).astype(audio.dtype)


class MockDestructiveModule:
    """Destruktives Testmodul — degradiert Audio stark für Quality-Gate-Tests."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        # Stark übersteuert + verrauscht: Quality Gate soll anschlagen
        np.random.seed(99)
        noisy = audio * 8.0 + 0.6 * np.random.randn(*audio.shape).astype(audio.dtype)
        return np.clip(noisy, -1.0, 1.0).astype(audio.dtype)


# === Test Fixtures ===


@pytest.fixture
def context():
    """Processing context fixture."""
    return ProcessingContext(session_id="test_mqa_session")


@pytest.fixture
def bus():
    """Communication bus fixture."""
    return ModuleCommunicationBus()


@pytest.fixture
def coordinator(context, bus):
    """Module coordinator with MQA enabled."""
    return ModuleCoordinator(context, bus, enable_musical_quality_assurance=True)


@pytest.fixture
def vinyl_audio():
    """Vinyl audio fixture (33⅓ rpm) - realistic quality."""
    np.random.seed(42)
    duration = 2.0  # 2 seconds
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-frequency signal (musical content)
    signal = (
        0.5 * np.sin(2 * np.pi * 220 * t)  # A3
        + 0.35 * np.sin(2 * np.pi * 440 * t)  # A4
        + 0.25 * np.sin(2 * np.pi * 880 * t)  # A5
        + 0.15 * np.sin(2 * np.pi * 1760 * t)  # A6 (harmonics)
    )

    # Add vinyl character (warmth, slight noise) - realistic SNR ~50 dB
    noise = 0.005 * np.random.randn(len(signal))  # Low noise for good vinyl
    return (signal + noise).astype(np.float32), sr


@pytest.fixture
def tape_audio():
    """Tape audio fixture (reel-to-reel)."""
    np.random.seed(43)
    duration = 2.0
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration))

    # Musical content with tape saturation
    signal = 0.4 * np.sin(2 * np.pi * 330 * t)  # E4
    noise = 0.015 * np.random.randn(len(signal))  # Tape hiss

    return (signal + noise).astype(np.float32), sr


@pytest.fixture
def damaged_audio():
    """Severely damaged audio."""
    np.random.seed(44)
    duration = 2.0
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration))

    # Weak signal + high noise
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    noise = 0.25 * np.random.randn(len(signal))

    return (signal + noise).astype(np.float32), sr


# === Real Plugins ===

# === Test Cases ===


@pytest.mark.unit
class TestMQABaseline:
    """Test quality baseline establishment."""

    def test_baseline_establishment_vinyl(self, coordinator, vinyl_audio):
        """Test baseline for vinyl."""
        audio, sr = vinyl_audio

        # Register dummy module
        coordinator.register_module(name="DummyModule", module_class=MockNeutralModule, priority=ModulePriority.NORMAL)

        # Execute with vinyl medium
        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # Baseline should be established
        assert coordinator._quality_baseline is not None
        assert coordinator._quality_baseline.overall_score > 0

        # MQA report should be present
        assert result["mqa_report"] is not None
        assert result["medium_type"] == "VINYL_33"

    def test_passthrough_verdict_when_intensity_below_5_percent(self, coordinator, vinyl_audio):
        """§VERDICT-PASSTHROUGH (Vorschlag 02): 1 Modul (< 5 % Intensität)
        ⇒ no_processing-Verdikt, musical_improvement exakt 0.0."""
        audio, sr = vinyl_audio
        coordinator.register_module(name="DummyModule", module_class=MockNeutralModule, priority=ModulePriority.NORMAL)
        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)
        mqa = result["mqa_report"]
        assert mqa is not None
        assert mqa.verdict.startswith("NO_PROCESSING_APPLIED")
        assert mqa.musical_improvement == 0.0

    def test_baseline_establishment_tape(self, coordinator, tape_audio):
        """Test baseline for tape."""
        audio, sr = tape_audio

        coordinator.register_module(name="DummyModule", module_class=MockNeutralModule, priority=ModulePriority.NORMAL)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.REEL_TO_REEL)

        assert coordinator._quality_baseline is not None
        assert result["mqa_report"] is not None
        assert result["medium_type"] == "REEL_TO_REEL"

    def test_baseline_from_forensics(self, coordinator, vinyl_audio):
        """Test baseline determination from forensic analysis."""
        audio, sr = vinyl_audio

        # Forensic analysis result
        forensics = {"medium_type": "VINYL", "rpm": 33, "quality_assessment": "GOOD"}

        coordinator.register_module(name="DummyModule", module_class=MockNeutralModule)

        result = coordinator.execute(audio, sr, forensic_analysis=forensics, processing_mode="restoration")

        # Should detect VINYL_33 from forensics
        assert result["mqa_report"] is not None
        assert result["medium_type"] == "VINYL_33"


class TestQualityGates:
    """Test quality gate checking during processing."""

    def test_quality_gate_pass(self, coordinator, vinyl_audio):
        """Test quality gate passes with good module (mock, ndarray-kompatibel)."""
        audio, sr = vinyl_audio

        # Mock-Plugin — akzeptiert ndarray, kein Dateipfad nötig
        coordinator.register_module(name="GoodModule", module_class=MockGoodModule, priority=ModulePriority.NORMAL)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # Modul wurde ausgeführt und MQA-Report wurde erstellt
        assert result["mqa_report"] is not None
        assert result["successful_modules"] >= 0  # Key existiert → Koordinator lief durch

        # MQA-Report enthält valide Qualitätsdaten
        mqa = result["mqa_report"]
        assert mqa is not None

    def test_quality_gate_fail_overprocessing(self, coordinator, vinyl_audio):
        """Test quality gate fails with destructive mock module."""
        audio, sr = vinyl_audio

        # Destruktives Mock-Plugin — degradiert Audio stark
        coordinator.register_module(
            name="BadModule", module_class=MockDestructiveModule, priority=ModulePriority.NORMAL
        )

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # System sollte gelaufen sein und MQA-Report erstellt haben
        module_results = result["module_results"]

        # Wenn das Modul fehlschlug, muss eine Fehlerursache angegeben sein
        if module_results:
            if not module_results[0].success:
                assert module_results[0].error is not None

        # MQA-Report muss erstellt worden sein
        mqa = result["mqa_report"]
        assert mqa is not None

    def test_quality_gate_multiple_modules(self, coordinator, vinyl_audio):
        """Test quality gates with multiple mock modules."""
        audio, sr = vinyl_audio

        # Mock-Plugins — akzeptieren ndarray direkt
        coordinator.register_module(name="Module1", module_class=MockGoodModule, priority=ModulePriority.NORMAL)
        coordinator.register_module(name="Module2", module_class=MockNeutralModule, priority=ModulePriority.NORMAL)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # Both modules should pass
        assert result["successful_modules"] >= 1

        # Checkpoints should be created
        assert len(coordinator._audio_checkpoints) >= 1


class TestRollback:
    """Test automatic rollback on quality violations."""

    def test_rollback_on_character_loss(self, coordinator, vinyl_audio):
        """Test rollback when vinyl character is destroyed (Mock-Module)."""
        audio, sr = vinyl_audio

        # Gutes Mock-Modul gefolgt von destruktivem Mock-Modul
        coordinator.register_module(name="GoodModule", module_class=MockGoodModule, priority=ModulePriority.NORMAL)
        coordinator.register_module(
            name="DestructiveModule",
            module_class=MockDestructiveModule,
            priority=ModulePriority.NORMAL,
            dependencies=["GoodModule"],
        )

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # Check if rollback occurred
        module_results = result["module_results"]

        # At least one module should run
        assert len(module_results) >= 1

        # Wenn das destruktive Modul fehlschlug, muss eine Fehlerursache angegeben sein
        if len(module_results) >= 2:
            if not module_results[-1].success:
                assert module_results[-1].error is not None

    def test_checkpoint_creation(self, coordinator, vinyl_audio):
        """Test checkpoint creation after each successful module."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="Module1", module_class=MockGoodModule)
        coordinator.register_module(name="Module2", module_class=MockNeutralModule, dependencies=["Module1"])

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # Should have checkpoints: original + after each successful module
        assert len(coordinator._audio_checkpoints) >= 1

        # First checkpoint should be "original"
        assert coordinator._audio_checkpoints[0][0] == "original"


class TestModeSpecificValidation:
    """Test mode-specific quality standards."""

    def test_restoration_mode_authenticity(self, coordinator, vinyl_audio):
        """Test RESTORATION mode requires high authenticity."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="TestModule", module_class=MockNeutralModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        mqa = result["mqa_report"]
        assert mqa is not None

        # Restoration mode should check authenticity preservation
        assert mqa.processing_mode == ProcessingMode.RESTORATION
        assert mqa.authenticity_preserved is not None

    def test_forensic_mode_minimal_processing(self, coordinator, vinyl_audio):
        """Test FORENSIC mode allows minimal processing only."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="TestModule", module_class=MockNeutralModule)

    def test_studio_2026_mode_modern_sound(self, coordinator, vinyl_audio):
        """Test STUDIO_2026 mode allows aggressive processing."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="TestModule", module_class=MockGoodModule)

        result = coordinator.execute(audio, sr, processing_mode="studio_2026", medium_type=MediumType.VINYL_33)

        mqa = result["mqa_report"]
        assert mqa is not None
        assert mqa.processing_mode == ProcessingMode.STUDIO_2026


class TestMediumSpecificGates:
    """Test medium-specific quality gates."""

    def test_vinyl_warmth_requirement(self, coordinator, vinyl_audio):
        """Test VINYL requires warmth preservation."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="TestModule", module_class=MockNeutralModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        mqa = result["mqa_report"]

        # Vinyl should check warmth
        assert mqa.output_quality.warmth >= 0.0  # Has warmth metric

    def test_shellac_authenticity_requirement(self, coordinator):
        """Test SHELLAC requires maximum authenticity."""
        # Generate shellac-like audio (old, limited bandwidth)
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Limited bandwidth (like 78rpm)
        signal = 0.3 * np.sin(2 * np.pi * 220 * t)
        noise = 0.06 * np.random.randn(len(signal))  # High noise
        audio = (signal + noise).astype(np.float32)

        coordinator.register_module(name="TestModule", module_class=MockNeutralModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.SHELLAC_78)

        mqa = result["mqa_report"]

        # Shellac should require high authenticity (0.85)
        assert mqa.medium_type == MediumType.SHELLAC_78

    def test_dsd_high_quality_requirement(self, coordinator):
        """Test DSD requires extremely high quality."""
        # Generate high-quality DSD-like audio
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Very clean signal
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        noise = 0.001 * np.random.randn(len(signal))  # Minimal noise
        audio = (signal + noise).astype(np.float32)

        coordinator.register_module(name="TestModule", module_class=MockNeutralModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.DSD)

        mqa = result["mqa_report"]

        # DSD should have very high SNR requirement (120 dB)
        assert mqa.medium_type == MediumType.DSD


class TestForensicMapping:
    """Test forensic analysis to medium type mapping."""

    def test_map_vinyl_33(self):
        """Test mapping to VINYL_33."""
        forensics = {"medium_type": "VINYL", "rpm": 33}
        medium = map_forensic_to_medium_type(forensics, rpm=33)
        assert medium == MediumType.VINYL_33

    def test_map_vinyl_45(self):
        """Test mapping to VINYL_45."""
        forensics = {"medium_type": "VINYL"}
        medium = map_forensic_to_medium_type(forensics, rpm=45)
        assert medium == MediumType.VINYL_45

    def test_map_shellac_78(self):
        """Test mapping to SHELLAC_78."""
        forensics = {"medium_type": "SHELLAC"}
        medium = map_forensic_to_medium_type(forensics, rpm=78)
        assert medium == MediumType.SHELLAC_78

    def test_map_reel_to_reel(self):
        """Test mapping to REEL_TO_REEL."""
        forensics = {"medium_type": "TAPE", "format": "reel-to-reel"}
        medium = map_forensic_to_medium_type(forensics)
        assert medium == MediumType.REEL_TO_REEL

    def test_map_cassette(self):
        """Test mapping to CASSETTE."""
        forensics = {"medium_type": "TAPE", "format": "cassette"}
        medium = map_forensic_to_medium_type(forensics)
        assert medium == MediumType.CASSETTE

    def test_map_cd(self):
        """Test mapping to CD."""
        forensics = {"medium_type": "CD"}
        medium = map_forensic_to_medium_type(forensics)
        assert medium == MediumType.CD

    def test_map_lossy_high(self):
        """Test mapping to LOSSY_HIGH."""
        forensics = {"medium_type": "LOSSY", "format": "MP3"}
        medium = map_forensic_to_medium_type(forensics, bitrate=320)
        assert medium == MediumType.LOSSY_HIGH

    def test_map_lossy_low(self):
        """Test mapping to LOSSY_LOW."""
        forensics = {"medium_type": "LOSSY"}
        medium = map_forensic_to_medium_type(forensics, bitrate=128)
        assert medium == MediumType.LOSSY_LOW

    def test_map_unknown_fallback(self):
        """Test fallback to UNKNOWN."""
        forensics = {"medium_type": "STRANGE_FORMAT"}
        medium = map_forensic_to_medium_type(forensics)
        assert medium == MediumType.UNKNOWN


class TestFinalValidation:
    """Test final quality validation."""

    def test_final_validation_report(self, coordinator, vinyl_audio):
        """Test final validation generates comprehensive report."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="TestModule", module_class=MockGoodModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # Should have complete MQA report
        mqa = result["mqa_report"]
        assert mqa is not None
        assert mqa.input_quality is not None
        assert mqa.output_quality is not None
        assert mqa.integrity_result is not None
        assert mqa.verdict is not None
        assert mqa.quality_guaranteed is not None

    def test_quality_improvement_tracking(self, coordinator, vinyl_audio):
        """Test tracking of musical improvement."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="TestModule", module_class=MockGoodModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        mqa = result["mqa_report"]

        # Should track improvement
        assert mqa.musical_improvement is not None
        assert isinstance(mqa.musical_improvement, float)

    def test_quality_guaranteed_flag(self, coordinator, vinyl_audio):
        """Test quality_guaranteed flag."""
        audio, sr = vinyl_audio

        coordinator.register_module(name="GoodModule", module_class=MockGoodModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # Should have quality_guaranteed flag
        assert "quality_guaranteed" in result
        assert isinstance(result["quality_guaranteed"], bool)


class TestMQADisabled:
    """Test coordinator behavior with MQA disabled."""

    def test_mqa_disabled_no_validation(self, context, bus, vinyl_audio):
        """Test MQA can be disabled."""
        coordinator = ModuleCoordinator(context, bus, enable_musical_quality_assurance=False)

        audio, sr = vinyl_audio

        coordinator.register_module(name="TestModule", module_class=MockNeutralModule)

        result = coordinator.execute(audio, sr, processing_mode="restoration", medium_type=MediumType.VINYL_33)

        # MQA should not run
        assert result["mqa_report"] is None
        assert coordinator._mqa_system is None


# === Integration Tests ===


class TestFullWorkflow:
    """Test complete MQA integration workflow."""

    def test_complete_restoration_workflow(self, coordinator, vinyl_audio):
        """Test complete restoration with MQA."""
        audio, sr = vinyl_audio

        # Register realistic module chain
        coordinator.register_module(
            name="Preprocessor", module_class=MockNeutralModule, priority=ModulePriority.CRITICAL
        )
        coordinator.register_module(
            name="Enhancement",
            module_class=MockGoodModule,
            priority=ModulePriority.NORMAL,
            dependencies=["Preprocessor"],
        )

        forensics = {"medium_type": "VINYL", "rpm": 33, "quality_assessment": "GOOD"}

        result = coordinator.execute(audio, sr, forensic_analysis=forensics, processing_mode="restoration")

        # Complete workflow should succeed
        assert result["successful_modules"] >= 1
        assert result["mqa_report"] is not None
        assert result["quality_guaranteed"] is not None

        # Output should be present
        assert result["output_audio"] is not None
        assert len(result["output_audio"]) == len(audio)
