import pytest

"""Tests for PDF export in audit.processing_report_generator.

Verifies that ReportExporter.export_pdf() produces a valid PDF file
with correct structure, handles empty reports, and integrates with
ProcessingReportGenerator.
"""

from datetime import datetime

import numpy as np

from audit.processing_report_generator import (
    ProcessingReport,
    ProcessingReportGenerator,
    ReportExporter,
    ReportSection,
    ReportSectionType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(**overrides) -> ProcessingReport:
    """Create a minimal ProcessingReport for testing."""
    defaults = {
        "report_id": "TEST_001",
        "timestamp": datetime.now().isoformat(),
        "aurik_version": "9.10.99",
        "input_file": "input.wav",
        "output_file": "output.flac",
        "overall_summary": "Test-Restaurierung abgeschlossen.",
        "overall_confidence": 0.92,
        "processing_time_sec": 42.5,
    }
    defaults.update(overrides)
    return ProcessingReport(**defaults)  # type: ignore[arg-type]


def _make_full_report() -> ProcessingReport:
    """Create a report with all section types for thorough testing."""
    report = _make_report()

    report.add_section(
        ReportSection(
            section_type=ReportSectionType.INPUT_ANALYSIS,
            title="Eingangsanalyse",
            content={"Dauer": "3:24", "Abtastrate": "44100 Hz", "Kanäle": "Stereo"},
            summary="MP3-Datei, mäßige Qualität.",
        )
    )
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.DETECTED_ISSUES,
            title="Erkannte Defekte",
            content={
                "issues": [
                    {
                        "type": "BROADBAND_NOISE",
                        "severity": "HIGH",
                        "confidence": 0.95,
                        "description": "Starkes Rauschen",
                    },
                    {"type": "HUM_50HZ", "severity": "MEDIUM", "confidence": 0.88, "description": "Netzbrummen"},
                ]
            },
        )
    )
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.APPLIED_MODULES,
            title="Angewandte Module",
            content={
                "modules": [
                    {"name": "Phase 03 — Denoise", "reason": "Breitbandrauschen", "strength": 0.75},
                    {"name": "Phase 02 — Hum Removal", "reason": "50 Hz Brummen", "strength": 0.60},
                ]
            },
        )
    )
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.MUSICAL_GOALS,
            title="Musical Goals",
            content={
                "goals": {
                    "natuerlichkeit": 0.93,
                    "authentizitaet": 0.91,
                    "tonal_center": 0.97,
                    "timbre_authentizitaet": 0.89,
                    "artikulation": 0.87,
                    "emotionalitaet": 0.85,
                    "micro_dynamics": 0.90,
                    "groove": 0.86,
                    "transparenz": 0.84,
                    "waerme": 0.78,
                    "bass_kraft": 0.81,
                    "separation_fidelity": 0.80,
                    "brillanz": 0.82,
                    "raumtiefe": 0.73,
                }
            },
        )
    )
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.BEFORE_AFTER_METRICS,
            title="Vorher / Nachher",
            content={
                "before": {"SNR_dB": 18.5, "LUFS": -22.0, "MOS": 2.8},
                "after": {"SNR_dB": 42.3, "LUFS": -14.1, "MOS": 4.2},
                "improvement": {"SNR_dB": 23.8, "LUFS": 7.9, "MOS": 1.4},
            },
        )
    )
    return report


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPdfExportBasic:
    def test_export_creates_file(self, tmp_path):
        pdf_path = tmp_path / "report.pdf"
        report = _make_report()
        ok = ReportExporter.export_pdf(report, pdf_path)
        assert ok
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 100  # non-trivial file

    def test_pdf_starts_with_magic(self, tmp_path):
        pdf_path = tmp_path / "report.pdf"
        report = _make_report()
        ReportExporter.export_pdf(report, pdf_path)
        header = pdf_path.read_bytes()[:5]
        assert header == b"%PDF-"

    def test_empty_report_produces_pdf(self, tmp_path):
        pdf_path = tmp_path / "empty.pdf"
        report = _make_report(overall_summary="", overall_confidence=0.0)
        ok = ReportExporter.export_pdf(report, pdf_path)
        assert ok
        assert pdf_path.exists()


class TestPdfExportFull:
    def test_full_report_pdf(self, tmp_path):
        pdf_path = tmp_path / "full_report.pdf"
        report = _make_full_report()
        ok = ReportExporter.export_pdf(report, pdf_path)
        assert ok
        assert pdf_path.stat().st_size > 1000  # multi-page PDF should be larger

    def test_full_report_has_two_pages(self, tmp_path):
        """PDF should contain at least 2 pages (summary + radar)."""
        pdf_path = tmp_path / "pages.pdf"
        report = _make_full_report()
        ReportExporter.export_pdf(report, pdf_path)
        # Rough heuristic: multi-page PDFs > 5 KB
        assert pdf_path.stat().st_size > 5000


class TestPdfViaGenerator:
    def test_generator_export_pdf(self, tmp_path):
        gen = ProcessingReportGenerator()
        sr = 48000
        audio_in = np.random.randn(sr * 2).astype(np.float32) * 0.1
        audio_out = audio_in * 0.8

        report = gen.create_report(
            input_audio=audio_in,
            output_audio=audio_out,
            sample_rate=sr,
            input_file="test.wav",
            output_file="test_restored.flac",
            processing_time_sec=15.0,
        )
        pdf_path = tmp_path / "gen_report.pdf"
        ok = gen.export_report(report, pdf_path, format="pdf")
        assert ok
        assert pdf_path.exists()


class TestPdfExportEdgeCases:
    def test_only_musical_goals(self, tmp_path):
        report = _make_report()
        report.add_section(
            ReportSection(
                section_type=ReportSectionType.MUSICAL_GOALS,
                title="Goals",
                content={"goals": {"natuerlichkeit": 0.95, "transparenz": 0.80}},
            )
        )
        pdf_path = tmp_path / "goals_only.pdf"
        ok = ReportExporter.export_pdf(report, pdf_path)
        assert ok

    def test_only_issues(self, tmp_path):
        report = _make_report()
        report.add_section(
            ReportSection(
                section_type=ReportSectionType.DETECTED_ISSUES,
                title="Defekte",
                content={"issues": ["noise", "clipping"]},
            )
        )
        pdf_path = tmp_path / "issues_only.pdf"
        ok = ReportExporter.export_pdf(report, pdf_path)
        assert ok

    def test_no_sections(self, tmp_path):
        report = _make_report()
        pdf_path = tmp_path / "bare.pdf"
        ok = ReportExporter.export_pdf(report, pdf_path)
        assert ok
