"""
Tests für Processing Report Generator

Test Coverage:
1. ReportSection creation
2. ProcessingReport structure
3. ReportExporter (Markdown, JSON)
4. ProcessingReportGenerator End-to-End
5. Report content validation

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import json
from typing import Any

import numpy as np
import pytest

from audit.processing_report_generator import (
    ProcessingReport,
    ProcessingReportGenerator,
    ReportExporter,
    ReportSection,
    ReportSectionType,
)


@pytest.mark.unit
class TestReportSection:
    """Test ReportSection dataclass."""

    def test_report_section_creation(self):
        """Test creating a ReportSection."""
        section = ReportSection(
            section_type=ReportSectionType.INPUT_ANALYSIS,
            title="Test Section",
            content={"duration": "10.5s", "channels": 2},
            summary="Test summary",
        )

        assert section.section_type == ReportSectionType.INPUT_ANALYSIS
        assert section.title == "Test Section"
        assert section.content["duration"] == "10.5s"
        assert section.summary == "Test summary"

    def test_section_to_dict(self):
        """Test ReportSection serialization."""
        section = ReportSection(
            section_type=ReportSectionType.DETECTED_ISSUES, title="Issues", content={"issues": []}, summary="No issues"
        )

        section_dict = section.to_dict()

        assert "section_type" in section_dict
        assert "title" in section_dict
        assert "content" in section_dict
        assert section_dict["section_type"] == "detected_issues"


class TestProcessingReport:
    """Test ProcessingReport dataclass."""

    def test_report_creation(self):
        """Test creating a ProcessingReport."""
        report = ProcessingReport(
            report_id="TEST_001", timestamp="2026-02-10T14:30:00", input_file="input.wav", output_file="output.wav"
        )

        assert report.report_id == "TEST_001"
        assert report.input_file == "input.wav"
        assert len(report.sections) == 0

    def test_add_section(self):
        """Test adding sections to report."""
        report = ProcessingReport(report_id="TEST_002", timestamp="2026-02-10T14:30:00")

        section1 = ReportSection(section_type=ReportSectionType.INPUT_ANALYSIS, title="Input", content={})

        section2 = ReportSection(section_type=ReportSectionType.DETECTED_ISSUES, title="Issues", content={})

        report.add_section(section1)
        report.add_section(section2)

        assert len(report.sections) == 2

    def test_get_section(self):
        """Test retrieving section by type."""
        report = ProcessingReport(report_id="TEST_003", timestamp="2026-02-10T14:30:00")

        input_section = ReportSection(
            section_type=ReportSectionType.INPUT_ANALYSIS, title="Input", content={"test": "data"}
        )

        report.add_section(input_section)

        retrieved = report.get_section(ReportSectionType.INPUT_ANALYSIS)

        assert retrieved is not None
        assert retrieved.content["test"] == "data"

    def test_get_nonexistent_section(self):
        """Test getting section that doesn't exist."""
        report = ProcessingReport(report_id="TEST_004", timestamp="2026-02-10T14:30:00")

        retrieved = report.get_section(ReportSectionType.MUSICAL_GOALS)

        assert retrieved is None

    def test_report_to_dict(self):
        """Test ProcessingReport serialization."""
        report = ProcessingReport(
            report_id="TEST_005", timestamp="2026-02-10T14:30:00", overall_confidence=0.85, processing_time_sec=10.5
        )

        section = ReportSection(section_type=ReportSectionType.INPUT_ANALYSIS, title="Input", content={})
        report.add_section(section)

        report_dict = report.to_dict()

        assert "report_id" in report_dict
        assert "sections" in report_dict
        assert report_dict["overall_confidence"] == 0.85
        assert len(report_dict["sections"]) == 1


class TestReportExporter:
    """Test ReportExporter für Markdown und JSON."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        report = ProcessingReport(
            report_id="TEST_EXPORT_001",
            timestamp="2026-02-10T14:30:00",
            input_file="test_input.wav",
            output_file="test_output.wav",
            overall_summary="Test processing completed successfully",
            overall_confidence=0.90,
            processing_time_sec=15.5,
        )

        # Input Analysis
        input_section = ReportSection(
            section_type=ReportSectionType.INPUT_ANALYSIS,
            title="Input Audio Analysis",
            content={
                "Duration": "10.0 seconds",
                "Sample Rate": "44100 Hz",
                "Channels": "Stereo",
                "Peak Level": "-3.2 dBFS",
            },
            summary="10.0s stereo audio at 44100Hz",
        )
        report.add_section(input_section)

        # Detected Issues
        issues_section = ReportSection(
            section_type=ReportSectionType.DETECTED_ISSUES,
            title="Detected Issues",
            content={
                "issues": [
                    {
                        "type": "Background Noise",
                        "severity": "MEDIUM",
                        "description": "Moderate noise detected",
                        "confidence": 0.85,
                    }
                ]
            },
            summary="Detected 1 issue(s)",
        )
        report.add_section(issues_section)

        # Applied Modules
        modules_section = ReportSection(
            section_type=ReportSectionType.APPLIED_MODULES,
            title="Applied Processing Modules",
            content={"modules": [{"name": "Spectral Denoiser", "reason": "Reduce background noise", "strength": 0.30}]},
            summary="Applied 1 module(s)",
        )
        report.add_section(modules_section)

        # Before/After Metrics
        metrics_section = ReportSection(
            section_type=ReportSectionType.BEFORE_AFTER_METRICS,
            title="Before/After Metrics",
            content={
                "before": {"SNR_dB": 18.5, "THD_percent": 1.2},
                "after": {"SNR_dB": 28.3, "THD_percent": 0.8},
                "improvement": {"SNR_dB": 9.8, "THD_percent": -0.4},
            },
            summary="Metrics comparison",
        )
        report.add_section(metrics_section)

        return report

    def test_export_markdown(self, sample_report, tmp_path):
        """Test exporting report as Markdown."""
        output_file = tmp_path / "test_report.md"

        result = ReportExporter.export_markdown(sample_report, output_file)

        assert result is True
        assert output_file.exists()

        # Read and validate content
        content = output_file.read_text()

        assert "# AURIK Audio Restoration Report" in content
        assert "TEST_EXPORT_001" in content
        assert "Input Audio Analysis" in content
        assert "Detected Issues" in content
        assert "Applied Processing Modules" in content

    def test_markdown_contains_all_sections(self, sample_report, tmp_path):
        """Test that Markdown contains all sections."""
        output_file = tmp_path / "test_report_full.md"

        ReportExporter.export_markdown(sample_report, output_file)
        content = output_file.read_text()

        # Check section titles
        for section in sample_report.sections:
            assert section.title in content

    def test_export_json(self, sample_report, tmp_path):
        """Test exporting report as JSON."""
        output_file = tmp_path / "test_report.json"

        result = ReportExporter.export_json(sample_report, output_file)

        assert result is True
        assert output_file.exists()

        # Read and validate JSON structure
        with open(output_file) as f:
            data = json.load(f)

        assert "report_id" in data
        assert "sections" in data
        assert data["report_id"] == "TEST_EXPORT_001"
        assert len(data["sections"]) == 4

    def test_json_structure_valid(self, sample_report, tmp_path):
        """Test that JSON structure is valid and complete."""
        output_file = tmp_path / "test_report_structure.json"

        ReportExporter.export_json(sample_report, output_file)

        with open(output_file) as f:
            data = json.load(f)

        # Validate top-level structure
        assert "timestamp" in data
        assert "input_file" in data
        assert "output_file" in data
        assert "overall_confidence" in data
        assert "processing_time_sec" in data

        # Validate section structure
        for section in data["sections"]:
            assert "section_type" in section
            assert "title" in section
            assert "content" in section


class TestProcessingReportGenerator:
    """Test ProcessingReportGenerator End-to-End."""

    @pytest.fixture
    def test_audio(self):
        """Generate test audio signals."""
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Input: noisy signal
        input_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        input_audio += 0.1 * np.random.randn(len(t))  # Heavy noise

        # Output: cleaned signal
        output_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        output_audio += 0.02 * np.random.randn(len(t))  # Light noise

        return input_audio, output_audio, sample_rate

    def test_generator_initialization(self):
        """Test ProcessingReportGenerator initialization."""
        generator = ProcessingReportGenerator()

        assert generator is not None

    def test_create_basic_report(self, test_audio):
        """Test creating a basic report."""
        input_audio, output_audio, sr = test_audio

        generator = ProcessingReportGenerator()

        report = generator.create_report(
            input_audio=input_audio,
            output_audio=output_audio,
            sample_rate=sr,
            input_file="test_input.wav",
            output_file="test_output.wav",
            processing_time_sec=5.0,
        )

        # Validate report structure
        assert report.report_id.startswith("AURIK_")
        assert report.input_file == "test_input.wav"
        assert report.output_file == "test_output.wav"
        assert report.processing_time_sec == 5.0
        assert len(report.sections) > 0

    def test_report_has_input_analysis(self, test_audio):
        """Test that report contains Input Analysis section."""
        input_audio, output_audio, sr = test_audio

        generator = ProcessingReportGenerator()
        report = generator.create_report(input_audio=input_audio, output_audio=output_audio, sample_rate=sr)

        input_section = report.get_section(ReportSectionType.INPUT_ANALYSIS)

        assert input_section is not None
        assert "Duration" in input_section.content
        assert "Sample Rate" in input_section.content

    def test_report_has_metrics(self, test_audio):
        """Test that report contains Before/After Metrics."""
        input_audio, output_audio, sr = test_audio

        generator = ProcessingReportGenerator()
        report = generator.create_report(input_audio=input_audio, output_audio=output_audio, sample_rate=sr)

        metrics_section = report.get_section(ReportSectionType.BEFORE_AFTER_METRICS)

        assert metrics_section is not None
        assert "before" in metrics_section.content
        assert "after" in metrics_section.content
        assert "improvement" in metrics_section.content

    def test_report_with_processing_history(self, test_audio):
        """Test report with processing history."""
        input_audio, output_audio, sr = test_audio

        history = {
            "detected_issues": [
                {
                    "type": "Background Noise",
                    "severity": "HIGH",
                    "description": "Significant noise detected",
                    "confidence": 0.90,
                }
            ],
            "applied_modules": [{"name": "Spectral Denoiser", "reason": "Remove background noise", "strength": 0.50}],
            "parameters": {"denoise_strength": 0.50, "processing_mode": "RESTORATION"},
            "confidence_scores": {"Noise Detection": 0.90, "Overall Processing": 0.85},
        }

        generator = ProcessingReportGenerator()
        report = generator.create_report(
            input_audio=input_audio, output_audio=output_audio, sample_rate=sr, processing_history=history
        )

        # Check that all history sections are present
        assert report.get_section(ReportSectionType.DETECTED_ISSUES) is not None
        assert report.get_section(ReportSectionType.APPLIED_MODULES) is not None
        assert report.get_section(ReportSectionType.PROCESSING_PARAMETERS) is not None
        assert report.get_section(ReportSectionType.CONFIDENCE_SCORES) is not None

    def test_report_has_recommendations(self, test_audio):
        """Test that report contains Recommendations."""
        input_audio, output_audio, sr = test_audio

        generator = ProcessingReportGenerator()
        report = generator.create_report(input_audio=input_audio, output_audio=output_audio, sample_rate=sr)

        recommendations_section = report.get_section(ReportSectionType.RECOMMENDATIONS)

        assert recommendations_section is not None
        assert "recommendations" in recommendations_section.content

    def test_overall_confidence_calculated(self, test_audio):
        """Test that overall confidence is calculated."""
        input_audio, output_audio, sr = test_audio

        history = {"confidence_scores": {"Detection": 0.90, "Processing": 0.85, "Verification": 0.88}}

        generator = ProcessingReportGenerator()
        report = generator.create_report(
            input_audio=input_audio, output_audio=output_audio, sample_rate=sr, processing_history=history
        )

        # Should be average of confidence scores
        expected_confidence = (0.90 + 0.85 + 0.88) / 3

        assert abs(report.overall_confidence - expected_confidence) < 0.01

    def test_export_complete_workflow(self, test_audio, tmp_path):
        """Test complete workflow: create + export."""
        input_audio, output_audio, sr = test_audio

        generator = ProcessingReportGenerator()

        report = generator.create_report(
            input_audio=input_audio,
            output_audio=output_audio,
            sample_rate=sr,
            input_file="workflow_input.wav",
            output_file="workflow_output.wav",
        )

        # Export Markdown
        md_file = tmp_path / "workflow_report.md"
        result_md = generator.export_report(report, md_file, format="markdown")
        assert result_md is True
        assert md_file.exists()

        # Export JSON
        json_file = tmp_path / "workflow_report.json"
        result_json = generator.export_report(report, json_file, format="json")
        assert result_json is True
        assert json_file.exists()


class TestIntegrationScenarios:
    """Integration tests für realistic scenarios."""

    @pytest.fixture
    def realistic_processing_history(self):
        """Create realistic processing history."""
        return {
            "detected_issues": [
                {
                    "type": "Background Noise",
                    "severity": "MEDIUM",
                    "description": "Moderate background noise (SNR: 20 dB)",
                    "confidence": 0.85,
                },
                {
                    "type": "Click Artifacts",
                    "severity": "LOW",
                    "description": "5 click artifacts detected",
                    "confidence": 0.75,
                },
                {
                    "type": "Clipping",
                    "severity": "HIGH",
                    "description": "Clipping detected at 12 locations",
                    "confidence": 0.95,
                },
            ],
            "applied_modules": [
                {"name": "Spectral Denoiser", "reason": "Reduce background noise", "strength": 0.30},
                {"name": "Click Removal", "reason": "Remove click artifacts", "strength": 0.50},
                {"name": "Declipping", "reason": "Restore clipped samples", "strength": 0.60},
            ],
            "parameters": {
                "processing_mode": "RESTORATION",
                "denoise_strength": 0.30,
                "click_removal_sensitivity": 0.50,
                "declip_strength": 0.60,
                "preserve_breaths": True,
                "preserve_room_tone": True,
            },
            "confidence_scores": {
                "Noise Detection": 0.85,
                "Click Detection": 0.75,
                "Clipping Detection": 0.95,
                "Overall Processing": 0.88,
            },
        }

    def test_full_restoration_report(self, realistic_processing_history, tmp_path):
        """Test generating a full restoration report."""
        # Generate audio
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))

        input_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        input_audio += 0.15 * np.random.randn(len(t))  # Noisy

        output_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        output_audio += 0.03 * np.random.randn(len(t))  # Clean

        # Generate report
        generator = ProcessingReportGenerator()
        report = generator.create_report(
            input_audio=input_audio,
            output_audio=output_audio,
            sample_rate=sr,
            processing_history=realistic_processing_history,
            input_file="archive_recording.wav",
            output_file="restored_recording.wav",
            processing_time_sec=45.3,
        )

        # Validate report completeness
        assert len(report.sections) >= 6  # Multiple sections

        # Export and validate Markdown
        md_file = tmp_path / "full_restoration_report.md"
        result = generator.export_report(report, md_file, format="markdown")
        assert result is True

        content = md_file.read_text()

        # Validate key content
        assert "Background Noise" in content
        assert "Click Artifacts" in content
        assert "Clipping" in content
        assert "Spectral Denoiser" in content
        assert "45.30 seconds" in content or "45.3 seconds" in content

    def test_minimal_processing_report(self, tmp_path):
        """Test report for minimal processing (no issues detected)."""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Clean audio (no processing needed)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        history: Any = {"detected_issues": [], "applied_modules": [], "parameters": {}, "confidence_scores": {}}

        generator = ProcessingReportGenerator()
        report = generator.create_report(
            input_audio=audio,
            output_audio=audio,
            sample_rate=sr,
            processing_history=history,  # Unchanged
        )

        # Should still have basic sections
        assert report.get_section(ReportSectionType.INPUT_ANALYSIS) is not None
        assert report.get_section(ReportSectionType.RECOMMENDATIONS) is not None

        # Export should work
        json_file = tmp_path / "minimal_report.json"
        result = generator.export_report(report, json_file, format="json")
        assert result is True


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_processing_history(self):
        """Test report with no processing history."""
        sr = 44100
        audio = np.random.randn(sr * 2)

        generator = ProcessingReportGenerator()
        report = generator.create_report(
            input_audio=audio,
            output_audio=audio,
            sample_rate=sr,
            processing_history=None,  # No history
        )

        # Should still create basic report
        assert report is not None
        assert len(report.sections) > 0

    def test_very_short_audio(self):
        """Test report for very short audio."""
        sr = 44100
        audio = np.random.randn(4410)  # 0.1 seconds

        generator = ProcessingReportGenerator()
        report = generator.create_report(input_audio=audio, output_audio=audio, sample_rate=sr)

        assert report is not None
        input_section = report.get_section(ReportSectionType.INPUT_ANALYSIS)
        assert input_section is not None

    def test_silent_audio(self):
        """Test report for silent audio."""
        sr = 44100
        audio = np.zeros(sr * 2)

        generator = ProcessingReportGenerator()
        report = generator.create_report(input_audio=audio, output_audio=audio, sample_rate=sr)

        # Should handle gracefully
        assert report is not None

    def test_export_unknown_format(self, tmp_path):
        """Test export with unknown format."""
        report = ProcessingReport(report_id="TEST", timestamp="2026-02-10T14:30:00")

        generator = ProcessingReportGenerator()
        output_file = tmp_path / "test.xyz"

        result = generator.export_report(report, output_file, format="unknown")

        # Should return False for unknown format
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
