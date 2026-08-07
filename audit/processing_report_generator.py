"""
Processing Report Generator für Explainability & Transparency

Implements GAP #4: Comprehensive Processing Reports.
Ein production-ready System braucht Explainability: "Warum wurde X gemacht?"

Architecture:
1. ProcessingReport - Strukturierte Report-Daten
2. ReportSection - Einzelne Sections (Issues, Modules, Metrics, etc.)
3. ReportExporter - Export zu Markdown, JSON, PDF
4. ProcessingReportGenerator - Main API

Report Sections:
- Input Analysis: Original audio characteristics
- Detected Issues: Clicks, clipping, noise, etc.
- Applied Modules: Welche Module wurden verwendet und warum
- Processing Parameters: Alle verwendeten Parameter
- Before/After Metrics: SNR, THD, LUFS, Musical Goals, etc.
- Confidence Scores: Confidence in decisions
- Recommendations: Suggestions für weitere improvements

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ReportSectionType(Enum):
    """Types of report sections."""

    INPUT_ANALYSIS = "input_analysis"
    """Initial audio analysis: Duration, SR, channels, format."""

    DETECTED_ISSUES = "detected_issues"
    """Detected defects: Clicks, noise, clipping, etc."""

    APPLIED_MODULES = "applied_modules"
    """Processing modules that were applied."""

    PROCESSING_PARAMETERS = "processing_parameters"
    """Detailed processing parameters."""

    BEFORE_AFTER_METRICS = "before_after_metrics"
    """Objective metrics before/after processing."""

    MUSICAL_GOALS = "musical_goals"
    """Musical goals evaluation (Klarheit, Wärme, etc.)."""

    CONFIDENCE_SCORES = "confidence_scores"
    """Confidence in processing decisions."""

    RECOMMENDATIONS = "recommendations"
    """Recommendations for further improvement."""

    WARNINGS = "warnings"
    """Warnings about potential issues."""


@dataclass
class ReportSection:
    """
    A section of a processing report.
    """

    section_type: ReportSectionType
    """Type of section."""

    title: str
    """Section title."""

    content: dict[str, Any]
    """Section content (structured data)."""

    summary: str = ""
    """Brief summary (1-2 sentences)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_type": self.section_type.value,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
        }


@dataclass
class ProcessingReport:
    """
    Comprehensive processing report.

    Contains all information about the processing:
    - What was detected?
    - What was done?
    - Why was it done?
    - What were the results?
    - How confident are we?
    """

    # === Metadata ===
    report_id: str
    """Unique report ID."""

    timestamp: str
    """ISO 8601 timestamp."""

    aurik_version: str = "1.0.0"
    """AURIK version."""

    # === Input Info ===
    input_file: str = ""
    """Input filename."""

    output_file: str = ""
    """Output filename."""

    # === Sections ===
    sections: list[ReportSection] = field(default_factory=list)
    """Report sections."""

    # === Summary ===
    overall_summary: str = ""
    """Overall processing summary."""

    overall_confidence: float = 0.0
    """Overall confidence (0.0-1.0)."""

    processing_time_sec: float = 0.0
    """Total processing time in seconds."""

    def add_section(self, section: ReportSection):
        """Add a section to the report."""
        self.sections.append(section)

    def get_section(self, section_type: ReportSectionType) -> ReportSection | None:
        """Get a section by type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "aurik_version": self.aurik_version,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "overall_summary": self.overall_summary,
            "overall_confidence": self.overall_confidence,
            "processing_time_sec": self.processing_time_sec,
            "sections": [s.to_dict() for s in self.sections],
        }


def _processing_report_from_dict(payload: dict[str, Any]) -> ProcessingReport:
    """Reconstruct ProcessingReport from serialized payload."""
    report = ProcessingReport(
        report_id=str(payload.get("report_id", "")),
        timestamp=str(payload.get("timestamp", "")),
        aurik_version=str(payload.get("aurik_version", "1.0.0")),
        input_file=str(payload.get("input_file", "")),
        output_file=str(payload.get("output_file", "")),
        overall_summary=str(payload.get("overall_summary", "")),
        overall_confidence=float(payload.get("overall_confidence", 0.0) or 0.0),
        processing_time_sec=float(payload.get("processing_time_sec", 0.0) or 0.0),
    )
    for section in payload.get("sections", []) or []:
        try:
            report.add_section(
                ReportSection(
                    section_type=ReportSectionType(str(section.get("section_type", "warnings"))),
                    title=str(section.get("title", "")),
                    content=dict(section.get("content", {}) or {}),
                    summary=str(section.get("summary", "")),
                )
            )
        except Exception as exc:
            logger.debug("Skipping malformed report section during reconstruction: %s", exc)
    return report


class ReportExporter:
    """
    Export ProcessingReport to various formats.

    Supported formats:
    - Markdown (.md) - Human-readable
    - JSON (.json) - Machine-readable
    - (PDF - future)
    """

    @staticmethod
    def _json_default(obj: Any):
        """JSON-Serializer fuer NumPy-Typen im Reporting-Export."""
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    @staticmethod
    def export_markdown(report: ProcessingReport, output_path: Path) -> bool:
        """
        Export report als Markdown.

        Args:
            report: ProcessingReport
            output_path: Output .md file

        Returns:
            True if successful
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # Header
                f.write("# AURIK Audio Restoration Report\n\n")
                f.write(f"**Report ID:** `{report.report_id}`  \n")
                f.write(f"**Timestamp:** {report.timestamp}  \n")
                f.write(f"**AURIK Version:** {report.aurik_version}  \n")
                f.write(f"**Processing Time:** {report.processing_time_sec:.2f} seconds  \n")
                f.write("\n---\n\n")

                # Files
                if report.input_file:
                    f.write(f"**Input File:** `{report.input_file}`  \n")
                if report.output_file:
                    f.write(f"**Output File:** `{report.output_file}`  \n")
                f.write("\n")

                # Overall Summary
                f.write("## Overall Summary\n\n")
                f.write(f"{report.overall_summary}\n\n")
                f.write(f"**Confidence:** {report.overall_confidence:.1%}  \n")
                f.write("\n---\n\n")

                # Sections
                for section in report.sections:
                    f.write(f"## {section.title}\n\n")

                    if section.summary:
                        f.write(f"_{section.summary}_\n\n")

                    # Format content based on section type
                    if section.section_type == ReportSectionType.INPUT_ANALYSIS:
                        ReportExporter._write_input_analysis_md(f, section.content)

                    elif section.section_type == ReportSectionType.DETECTED_ISSUES:
                        ReportExporter._write_detected_issues_md(f, section.content)

                    elif section.section_type == ReportSectionType.APPLIED_MODULES:
                        ReportExporter._write_applied_modules_md(f, section.content)

                    elif section.section_type == ReportSectionType.PROCESSING_PARAMETERS:
                        ReportExporter._write_processing_params_md(f, section.content)

                    elif section.section_type == ReportSectionType.BEFORE_AFTER_METRICS:
                        ReportExporter._write_before_after_metrics_md(f, section.content)

                    elif section.section_type == ReportSectionType.MUSICAL_GOALS:
                        ReportExporter._write_musical_goals_md(f, section.content)

                    elif section.section_type == ReportSectionType.CONFIDENCE_SCORES:
                        ReportExporter._write_confidence_scores_md(f, section.content)

                    elif section.section_type == ReportSectionType.RECOMMENDATIONS:
                        ReportExporter._write_recommendations_md(f, section.content)

                    elif section.section_type == ReportSectionType.WARNINGS:
                        ReportExporter._write_warnings_md(f, section.content)

                    else:
                        # Generic content
                        f.write("```json\n")
                        f.write(json.dumps(section.content, indent=2))
                        f.write("\n```\n\n")

                    f.write("\n---\n\n")

                # Footer
                f.write(f"\n*Report generated by AURIK v{report.aurik_version}*\n")

            logger.info(f"✅ Markdown report exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Markdown Ausgabe fehlgeschlagen: {e}")
            return False

    @staticmethod
    def _write_input_analysis_md(f, content: dict):
        """Write Input Analysis section."""
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        for key, value in content.items():
            f.write(f"| {key} | {value} |\n")
        f.write("\n")

    @staticmethod
    def _write_detected_issues_md(f, content: dict):
        """Write Detected Issues section."""
        issues = content.get("issues", [])

        if not issues:
            f.write("✅ No significant issues detected.\n\n")
            return

        f.write(f"Detected {len(issues)} issue(s):\n\n")
        for issue in issues:
            # Handle plain string issues (e.g. ["noise", "clipping"])
            if isinstance(issue, str):
                f.write(f"- ⚠️ {issue}\n\n")
                continue
            severity = issue.get("severity", "MEDIUM")
            issue_type = issue.get("type", "Unknown")
            description = issue.get("description", "")
            confidence = issue.get("confidence", 0.0)

            emoji = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"

            f.write(f"- {emoji} **{issue_type}** ({severity}, {confidence:.0%} confidence)\n")
            f.write(f"  _{description}_\n\n")

    @staticmethod
    def _write_applied_modules_md(f, content: dict):
        """Write Applied Modules section."""
        modules = content.get("modules", [])

        if not modules:
            f.write("No processing modules applied.\n\n")
            return

        f.write(f"Applied {len(modules)} processing module(s):\n\n")
        for i, module in enumerate(modules, 1):
            name = module.get("name", "Unknown")
            reason = module.get("reason", "")
            strength = module.get("strength", 0.0)

            f.write(f"{i}. **{name}** (strength: {strength:.0%})\n")
            f.write(f"   - Reason: {reason}\n\n")

    @staticmethod
    def _write_processing_params_md(f, content: dict):
        """Write Processing Parameters section."""
        f.write("```yaml\n")
        for key, value in content.items():
            f.write(f"{key}: {value}\n")
        f.write("```\n\n")

    @staticmethod
    def _write_before_after_metrics_md(f, content: dict):
        """Write Before/After Metrics section."""
        before = content.get("before", {})
        after = content.get("after", {})
        improvement = content.get("improvement", {})

        f.write("| Metric | Before | After | Improvement |\n")
        f.write("|--------|--------|-------|-------------|\n")

        for metric in sorted(before.keys()):
            b_val = before.get(metric, 0.0)
            a_val = after.get(metric, 0.0)
            imp = improvement.get(metric, 0.0)

            imp_str = f"+{imp:.1f}" if imp > 0 else f"{imp:.1f}"
            f.write(f"| {metric} | {b_val:.2f} | {a_val:.2f} | {imp_str} |\n")
        f.write("\n")

    @staticmethod
    def _write_musical_goals_md(f, content: dict):
        """Write Musical Goals section."""
        goals = content.get("goals", {})

        f.write("| Goal | Score | Status |\n")
        f.write("|------|-------|--------|\n")

        for goal_name, score in goals.items():
            status = "✅ Excellent" if score > 0.8 else "✓ Good" if score > 0.6 else "⚠ Fair"
            f.write(f"| {goal_name} | {score:.2f} | {status} |\n")
        f.write("\n")

    @staticmethod
    def _write_confidence_scores_md(f, content: dict):
        """Write Confidence Scores section."""
        scores = content.get("scores", {})

        for decision, confidence in scores.items():
            conf_str = f"{confidence:.0%}"
            emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            f.write(f"- {emoji} **{decision}**: {conf_str}\n")
        f.write("\n")

    @staticmethod
    def _write_recommendations_md(f, content: dict):
        """Write Recommendations section."""
        recommendations = content.get("recommendations", [])

        if not recommendations:
            f.write("No additional recommendations.\n\n")
            return

        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        f.write("\n")

    @staticmethod
    def _write_warnings_md(f, content: dict):
        """Write Warnings section."""
        warnings = content.get("warnings", [])

        if not warnings:
            f.write("No warnings.\n\n")
            return

        for warning in warnings:
            f.write(f"⚠️ {warning}\n\n")

    @staticmethod
    def export_json(report: ProcessingReport, output_path: Path) -> bool:
        """
        Export report als JSON.

        Args:
            report: ProcessingReport
            output_path: Output .json file

        Returns:
            True if successful
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"✅ JSON report exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"JSON Ausgabe fehlgeschlagen: {e}")
            return False

    @staticmethod
    def _export_pdf_via_subprocess(report: ProcessingReport, output_path: Path) -> bool:
        """Fallback: render PDF in a fresh Python process to avoid poisoned matplotlib state."""
        import os
        import subprocess
        import sys
        import tempfile

        if os.environ.get("AURIK_PDF_SUBPROCESS") == "1":
            return False

        payload = report.to_dict()
        with tempfile.TemporaryDirectory(prefix="aurik_pdf_") as tmpdir:
            payload_path = Path(tmpdir) / "report_payload.json"
            payload_path.write_text(
                json.dumps(payload, ensure_ascii=False, default=ReportExporter._json_default),
                encoding="utf-8",
            )
            script = (
                "import json, os, sys; "
                "from pathlib import Path; "
                "from audit.processing_report_generator import _processing_report_from_dict, ReportExporter; "
                "payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); "
                "report=_processing_report_from_dict(payload); "
                "os.environ['AURIK_PDF_SUBPROCESS']='1'; "
                "raise SystemExit(0 if ReportExporter.export_pdf(report, Path(sys.argv[2])) else 1)"
            )
            env = dict(os.environ)
            env["AURIK_PDF_SUBPROCESS"] = "1"
            result = subprocess.run(
                [sys.executable, "-c", script, str(payload_path), str(output_path)],
                cwd=str(Path(__file__).resolve().parent.parent),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error(
                    "PDF subprocess Ersatzpfad fehlgeschlagen (code=%s): %s",
                    result.returncode,
                    (result.stderr or result.stdout).strip(),
                )
                return False
            return True

    @staticmethod
    def _export_pdf_plain_fallback(report: ProcessingReport, output_path: Path) -> bool:
        """Fallback ohne externe PDF-Backends.

        Schreibt ein valides, einfaches Multi-Page-PDF (Text-only), falls
        matplotlibs backend_pdf in der aktuellen Umgebung defekt ist.
        """

        def _sanitize_pdf_text(text: str) -> str:
            s = str(text)
            s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            # Helvetica/WinAnsi-safe; unbekannte Zeichen werden ersetzt.
            return s.encode("latin-1", "replace").decode("latin-1")

        def _build_stream(lines: list[str]) -> bytes:
            y = 800
            parts = ["BT", "/F1 10 Tf", "50 820 Td", "14 TL"]
            for line in lines:
                if y < 40:
                    break
                parts.append(f"({_sanitize_pdf_text(line)}) Tj")
                parts.append("T*")
                y -= 14
            parts.append("ET")
            return "\n".join(parts).encode("latin-1", "replace")

        try:
            meta_lines = [
                "AURIK - Audio-Restaurierungsbericht",
                f"Report-ID: {report.report_id}",
                f"Zeit: {report.timestamp}",
                f"Version: {report.aurik_version}",
                f"Input: {report.input_file}",
                f"Output: {report.output_file}",
                f"Konfidenz: {report.overall_confidence:.3f}",
                f"Processing-Zeit (s): {report.processing_time_sec:.3f}",
                "",
                "Zusammenfassung:",
                str(report.overall_summary or "-")[:400],
                "",
                "Sections:",
            ]
            for sec in report.sections:
                meta_lines.append(f"- {sec.section_type.value}: {sec.title}")

            json_lines = json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                default=ReportExporter._json_default,
            ).splitlines()
            all_lines = [*meta_lines, "", "Raw-Report:", *json_lines]

            lines_per_page = 54
            pages: list[list[str]] = []
            for i in range(0, len(all_lines), lines_per_page):
                pages.append(all_lines[i : i + lines_per_page])
            if not pages:
                pages = [["AURIK Report (leer)"]]

            # Bei umfangreichen Reports mindestens 2 Seiten fuer stabile Lesbarkeit.
            if len(report.sections) >= 3 and len(pages) < 2:
                pages.append(["Weitere Details siehe JSON-Abschnitt."])

            n_pages = len(pages)
            font_obj = 3
            first_page_obj = 4
            first_content_obj = first_page_obj + n_pages
            total_objects = 3 + (2 * n_pages)

            objects: list[bytes] = [b""] * total_objects

            # 1: Catalog
            objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"

            # 2: Pages
            kids = " ".join(f"{first_page_obj + i} 0 R" for i in range(n_pages))
            objects[1] = f"<< /Type /Pages /Count {n_pages} /Kids [{kids}] >>".encode("ascii")

            # 3: Font
            objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

            # Pages + content streams
            for i, page_lines in enumerate(pages):
                page_no = first_page_obj + i
                content_no = first_content_obj + i
                stream = _build_stream(page_lines)
                objects[page_no - 1] = (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                    f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
                    f"/Contents {content_no} 0 R >>"
                ).encode("ascii")
                objects[content_no - 1] = (
                    f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
                )

            pdf = bytearray()
            pdf.extend(b"%PDF-1.4\n")
            pdf.extend(b"%\xe2\xe3\xcf\xd3\n")

            offsets = [0]
            for idx, obj in enumerate(objects, start=1):
                offsets.append(len(pdf))
                pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
                pdf.extend(obj)
                pdf.extend(b"\nendobj\n")

            xref_start = len(pdf)
            pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
            pdf.extend(b"0000000000 65535 f \n")
            for off in offsets[1:]:
                pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

            pdf.extend(
                (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n").encode(
                    "ascii"
                )
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(bytes(pdf))
            logger.info("PDF report exported via plain Ersatzpfad: %s", output_path)
            return True
        except Exception as e:
            logger.error("Plain PDF Ersatzpfad fehlgeschlagen: %s", e)
            return False

    @staticmethod
    def export_pdf(report: ProcessingReport, output_path: Path) -> bool:
        """Export report as a multi-page PDF with tables and Musical Goals radar chart.

        Uses matplotlib (already in requirements) — no extra dependencies.

        Parameters
        ----------
        report : ProcessingReport
            The report to render.
        output_path : Path
            Destination ``.pdf`` file.

        Returns
        -------
        bool
            True if successful.
        """
        try:
            with warnings.catch_warnings():
                # pyparsing 3.3+ can emit parseString deprecation warnings while
                # matplotlib parses font patterns during import. In strict test mode
                # (-W error::Warning), suppress only this known third-party warning.
                warnings.filterwarnings(
                    "ignore",
                    message=r".*parseString.*deprecated.*",
                    category=Warning,
                )
                try:
                    from pyparsing.warnings import PyparsingDeprecationWarning as _PypDW

                    warnings.filterwarnings("ignore", category=_PypDW)
                except ImportError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                import sys

                import matplotlib

                # A previous failed matplotlib import can leave a half-initialized
                # module in sys.modules (missing rcParams). In strict warning/error
                # test runs this happens after third-party deprecation warnings.
                if not hasattr(matplotlib, "rcParams"):
                    for _mod_name in list(sys.modules):
                        if _mod_name == "matplotlib" or _mod_name.startswith("matplotlib."):
                            sys.modules.pop(_mod_name, None)
                    import matplotlib  # re-import clean module

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            logger.error("matplotlib not installed — PDF Ausgabe nicht verfuegbar")
            return False
        except Exception as e:
            logger.error(f"matplotlib import fehlgeschlagen for PDF Ausgabe: {e}")
            if ReportExporter._export_pdf_via_subprocess(report, output_path):
                logger.info("PDF report exported via subprocess Ersatzpfad: %s", output_path)
                return True
            return False

        try:
            with PdfPages(str(output_path)) as pdf:
                # ── Page 1: Header + Summary + Detected Issues + Modules ──
                fig, axes = plt.subplots(
                    3,
                    1,
                    figsize=(8.27, 11.69),
                    gridspec_kw={"height_ratios": [1.2, 1.8, 2.0]},
                )
                fig.patch.set_facecolor("#0d1117")
                plt.subplots_adjust(top=0.92, bottom=0.04, left=0.06, right=0.94, hspace=0.35)

                fig.text(
                    0.5,
                    0.96,
                    "AURIK — Audio-Restaurierungsbericht",
                    ha="center",
                    va="top",
                    fontsize=16,
                    fontweight="bold",
                    color="#c9d1d9",
                )
                fig.text(
                    0.5,
                    0.935,
                    f"Version {report.aurik_version}  ·  {report.timestamp[:19]}  ·  "
                    f"Verarbeitung: {report.processing_time_sec:.1f} s",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#8b949e",
                )

                # Summary table
                ax0 = axes[0]
                ax0.axis("off")
                summary_data = [
                    ["Eingabedatei", report.input_file or "—"],
                    ["Ausgabedatei", report.output_file or "—"],
                    ["Konfidenz", f"{report.overall_confidence:.0%}"],
                ]
                if report.overall_summary:
                    summary_data.append(["Zusammenfassung", report.overall_summary[:80]])
                tbl0 = ax0.table(
                    cellText=summary_data,
                    colLabels=["Eigenschaft", "Wert"],
                    loc="center",
                    cellLoc="left",
                )
                _style_pdf_table(tbl0)

                # Detected Issues
                ax1 = axes[1]
                ax1.axis("off")
                issues_sec = report.get_section(ReportSectionType.DETECTED_ISSUES)
                if issues_sec:
                    issues = issues_sec.content.get("issues", [])
                    if issues:
                        issue_rows = []
                        for iss in issues[:12]:
                            if isinstance(iss, str):
                                issue_rows.append([iss, "—", "—"])
                            else:
                                issue_rows.append(
                                    [
                                        str(iss.get("type", "?")),
                                        str(iss.get("severity", "?")),
                                        (
                                            f"{iss.get('confidence', 0):.0%}"
                                            if isinstance(iss.get("confidence"), (int, float))
                                            else "—"
                                        ),
                                    ]
                                )
                        tbl1 = ax1.table(
                            cellText=issue_rows,
                            colLabels=["Defekt", "Schwere", "Konfidenz"],
                            loc="center",
                            cellLoc="left",
                        )
                        _style_pdf_table(tbl1)
                    else:
                        ax1.text(
                            0.5,
                            0.5,
                            "Keine signifikanten Defekte erkannt ✓",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="#82B89A",
                        )
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "Defektanalyse nicht verfügbar",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#8b949e",
                    )

                # Applied Modules
                ax2 = axes[2]
                ax2.axis("off")
                modules_sec = report.get_section(ReportSectionType.APPLIED_MODULES)
                if modules_sec:
                    mods = modules_sec.content.get("modules", [])
                    if mods:
                        mod_rows = []
                        for m in mods[:15]:
                            if isinstance(m, dict):
                                mod_rows.append(
                                    [
                                        str(m.get("name", "?")),
                                        str(m.get("reason", ""))[:50],
                                        (
                                            f"{m.get('strength', 0):.0%}"
                                            if isinstance(m.get("strength"), (int, float))
                                            else "—"
                                        ),
                                    ]
                                )
                            else:
                                mod_rows.append([str(m), "", "—"])
                        tbl2 = ax2.table(
                            cellText=mod_rows,
                            colLabels=["Modul", "Grund", "Stärke"],
                            loc="center",
                            cellLoc="left",
                        )
                        _style_pdf_table(tbl2)

                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

                # ── Page 2: Musical Goals Radar + Before/After Metrics ────
                goals_sec = report.get_section(ReportSectionType.MUSICAL_GOALS)
                metrics_sec = report.get_section(ReportSectionType.BEFORE_AFTER_METRICS)

                fig2 = plt.figure(figsize=(11.69, 8.27))
                fig2.patch.set_facecolor("#0d1117")
                fig2.text(
                    0.5,
                    0.97,
                    "Musical Goals & Metriken",
                    ha="center",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color="#c9d1d9",
                )

                # Radar chart for Musical Goals
                ax_radar = fig2.add_subplot(121, projection="polar")
                ax_radar.set_facecolor("#161b22")

                if goals_sec:
                    goals = goals_sec.content.get("goals", {})
                    if goals:
                        labels = list(goals.keys())
                        values = [float(v) for v in goals.values()]
                        n = len(labels)
                        import math as _math

                        angles = [i / n * 2 * _math.pi for i in range(n)]
                        angles.append(angles[0])
                        values.append(values[0])

                        ax_radar.plot(angles, values, "o-", linewidth=1.5, color="#667eea", markersize=4)
                        ax_radar.fill(angles, values, alpha=0.15, color="#667eea")
                        ax_radar.set_xticks(angles[:-1])
                        ax_radar.set_xticklabels(
                            [la.replace("_", "\n") for la in labels],
                            fontsize=6,
                            color="#c9d1d9",
                        )
                        ax_radar.set_ylim(0, 1.0)
                        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                        ax_radar.set_yticklabels(
                            ["0.2", "0.4", "0.6", "0.8", "1.0"],
                            fontsize=6,
                            color="#8b949e",
                        )
                        ax_radar.spines["polar"].set_color("#30363d")
                        ax_radar.grid(color="#30363d", linewidth=0.5)
                        ax_radar.set_title("Musical Goals", fontsize=10, color="#c9d1d9", pad=12)

                # Before/After metrics table
                ax_metrics = fig2.add_subplot(122)
                ax_metrics.axis("off")
                ax_metrics.set_facecolor("#0d1117")

                if metrics_sec:
                    before = metrics_sec.content.get("before", {})
                    after = metrics_sec.content.get("after", {})
                    improvement = metrics_sec.content.get("improvement", {})
                    if before:
                        met_rows = []
                        for metric in sorted(before.keys()):
                            b = before.get(metric, 0.0)
                            a = after.get(metric, 0.0)
                            imp = improvement.get(metric, 0.0)
                            imp_str = f"+{imp:.2f}" if imp > 0 else f"{imp:.2f}"
                            met_rows.append([metric, f"{b:.2f}", f"{a:.2f}", imp_str])
                        tbl_m = ax_metrics.table(
                            cellText=met_rows,
                            colLabels=["Metrik", "Vorher", "Nachher", "Δ"],
                            loc="center",
                            cellLoc="left",
                        )
                        _style_pdf_table(tbl_m)
                        ax_metrics.set_title(
                            "Vorher / Nachher",
                            fontsize=10,
                            color="#c9d1d9",
                            pad=10,
                        )
                else:
                    ax_metrics.text(
                        0.5,
                        0.5,
                        "Keine Metriken verfügbar",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#8b949e",
                    )

                plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])  # type: ignore[arg-type]
                pdf.savefig(fig2, facecolor=fig2.get_facecolor())
                plt.close(fig2)

            logger.info("PDF report exported: %s", output_path)
            return True

        except Exception as e:
            logger.error("PDF Ausgabe fehlgeschlagen: %s", e)
            if ReportExporter._export_pdf_plain_fallback(report, output_path):
                return True
            return False


def _style_pdf_table(tbl) -> None:
    """Apply dark-theme styling to a matplotlib table for PDF rendering."""
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#c9d1d9", fontweight="bold")
        else:
            cell.set_facecolor("#0d1117")
            cell.set_text_props(color="#c9d1d9")


class ProcessingReportGenerator:
    """
    Main API für Processing Report Generation.

    Usage:
        generator = ProcessingReportGenerator()

        report = generator.create_report(
            input_audio=audio,
            output_audio=result_audio,
            processing_history=history,
            input_file="input.wav",
            output_file="output.wav"
        )

        generator.export_report(report, "report.md", format="markdown")
    """

    def create_report(
        self,
        input_audio: np.ndarray,
        output_audio: np.ndarray,
        sample_rate: int,
        processing_history: dict[str, Any] | None = None,
        input_file: str = "",
        output_file: str = "",
        processing_time_sec: float = 0.0,
    ) -> ProcessingReport:
        """
        Create a comprehensive processing report.

        Args:
            input_audio: Original audio
            output_audio: Processed audio
            sample_rate: Sample rate
            processing_history: Dict mit applied modules, parameters, etc.
            input_file: Input filename
            output_file: Output filename
            processing_time_sec: Processing time

        Returns:
            ProcessingReport
        """
        # Generate report ID
        report_id = f"AURIK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()

        report = ProcessingReport(
            report_id=report_id,
            timestamp=timestamp,
            input_file=input_file,
            output_file=output_file,
            processing_time_sec=processing_time_sec,
        )

        # === 1. Input Analysis ===
        input_section = self._create_input_analysis_section(input_audio, sample_rate)
        report.add_section(input_section)

        # === 2. Detected Issues ===
        if processing_history and "detected_issues" in processing_history:
            issues_section = self._create_detected_issues_section(processing_history["detected_issues"])
            report.add_section(issues_section)

        # === 3. Applied Modules ===
        _modules_raw = None
        if processing_history:
            _modules_raw = processing_history.get("applied_modules") or processing_history.get("modules_applied")
        if _modules_raw is not None:
            # Normalize: convert plain strings to dicts so _write_applied_modules_md works
            _modules_dicts = [
                m if isinstance(m, dict) else {"name": str(m), "reason": "", "strength": 1.0}
                for m in (_modules_raw if isinstance(_modules_raw, list) else [])
            ]
            modules_section = self._create_applied_modules_section(_modules_dicts)
            report.add_section(modules_section)

        # === 4. Processing Parameters ===
        _params_raw = None
        if processing_history:
            _params_raw = processing_history.get("parameters") or processing_history.get("processing_parameters")
        if _params_raw is not None:
            params_section = self._create_processing_params_section(
                _params_raw if isinstance(_params_raw, dict) else {"info": str(_params_raw)}
            )
            report.add_section(params_section)

        # === 5. Before/After Metrics ===
        metrics_section = self._create_before_after_metrics_section(input_audio, output_audio, sample_rate)
        report.add_section(metrics_section)

        # === 6. Musical Goals ===
        try:
            musical_goals_section = self._create_musical_goals_section(output_audio, sample_rate)
            report.add_section(musical_goals_section)
        except Exception as e:
            logger.warning(f"Musical goals evaluation fehlgeschlagen: {e}")

        # === 7. Confidence Scores ===
        if processing_history and "confidence_scores" in processing_history:
            confidence_section = self._create_confidence_section(processing_history["confidence_scores"])
            report.add_section(confidence_section)

        # === 8. Recommendations ===
        recommendations_section = self._create_recommendations_section(input_audio, output_audio, sample_rate)
        report.add_section(recommendations_section)

        # === Overall Summary ===
        report.overall_summary = self._generate_overall_summary(report)
        report.overall_confidence = self._calculate_overall_confidence(report)

        return report

    def _create_input_analysis_section(self, audio: np.ndarray, sample_rate: int) -> ReportSection:
        """Create Input Analysis section."""
        duration_sec = len(audio) / sample_rate
        channels = 2 if audio.ndim == 2 else 1
        peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)

        content = {
            "Duration": f"{duration_sec:.2f} seconds",
            "Sample Rate": f"{sample_rate} Hz",
            "Channels": "Stereo" if channels == 2 else "Mono",
            "Peak Level": f"{peak_db:.1f} dBFS",
            "RMS Level": f"{rms_db:.1f} dBFS",
            "Dynamic Range": f"{peak_db - rms_db:.1f} dB",
        }

        return ReportSection(
            section_type=ReportSectionType.INPUT_ANALYSIS,
            title="Input Audio Analysis",
            content=content,
            summary=f"{duration_sec:.1f}s {channels}ch audio at {sample_rate}Hz",
        )

    def _create_detected_issues_section(self, issues: list[dict[str, Any]]) -> ReportSection:
        """Create Detected Issues section."""
        return ReportSection(
            section_type=ReportSectionType.DETECTED_ISSUES,
            title="Detected Issues",
            content={"issues": issues},
            summary=f"Detected {len(issues)} issue(s) requiring processing",
        )

    def _create_applied_modules_section(self, modules: list[dict[str, Any]]) -> ReportSection:
        """Create Applied Modules section."""
        return ReportSection(
            section_type=ReportSectionType.APPLIED_MODULES,
            title="Applied Processing Modules",
            content={"modules": modules},
            summary=f"Applied {len(modules)} processing module(s)",
        )

    def _create_processing_params_section(self, parameters: dict[str, Any]) -> ReportSection:
        """Create Processing Parameters section."""
        return ReportSection(
            section_type=ReportSectionType.PROCESSING_PARAMETERS,
            title="Processing Parameters",
            content=parameters,
            summary="Detailed processing configuration",
        )

    def _create_before_after_metrics_section(
        self, input_audio: np.ndarray, output_audio: np.ndarray, sample_rate: int
    ) -> ReportSection:
        """Create Before/After Metrics section."""
        # Simplified metrics (real implementation würde enhanced_metrics nutzen)

        # Peak
        before_peak = 20 * np.log10(np.max(np.abs(input_audio)) + 1e-10)
        after_peak = 20 * np.log10(np.max(np.abs(output_audio)) + 1e-10)

        # RMS
        before_rms = 20 * np.log10(np.sqrt(np.mean(input_audio**2)) + 1e-10)
        after_rms = 20 * np.log10(np.sqrt(np.mean(output_audio**2)) + 1e-10)

        before = {"Peak_dBFS": before_peak, "RMS_dBFS": before_rms}

        after = {"Peak_dBFS": after_peak, "RMS_dBFS": after_rms}

        improvement = {"Peak_dBFS": after_peak - before_peak, "RMS_dBFS": after_rms - before_rms}

        return ReportSection(
            section_type=ReportSectionType.BEFORE_AFTER_METRICS,
            title="Before/After Metrics",
            content={"before": before, "after": after, "improvement": improvement},
            summary="Objective metric comparison",
        )

    def _create_musical_goals_section(self, audio: np.ndarray, sample_rate: int) -> ReportSection:
        """Create Musical Goals section."""
        try:
            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

            checker = MusicalGoalsChecker()
            goals = checker.measure_all(audio, sample_rate)

            avg_score = np.mean(list(goals.values()))

            return ReportSection(
                section_type=ReportSectionType.MUSICAL_GOALS,
                title="Musical Goals Evaluation",
                content={"goals": goals},
                summary=f"Average musical goals score: {avg_score:.2f}",
            )
        except Exception:
            logger.warning("processing_report_generator.py::_erstellen_musical_goals_section Ersatzpfad", exc_info=True)
            return ReportSection(
                section_type=ReportSectionType.MUSICAL_GOALS,
                title="Musical Goals Evaluation",
                content={"goals": {}},
                summary="Musical goals evaluation not available",
            )

    def _create_confidence_section(self, confidence_scores: dict[str, float]) -> ReportSection:
        """Create Confidence Scores section."""
        vals = list(confidence_scores.values())
        avg_confidence = float(np.mean(vals)) if vals else 0.0

        return ReportSection(
            section_type=ReportSectionType.CONFIDENCE_SCORES,
            title="Confidence Scores",
            content={"scores": confidence_scores},
            summary=f"Average confidence: {avg_confidence:.0%}",
        )

    def _create_recommendations_section(
        self, input_audio: np.ndarray, output_audio: np.ndarray, sample_rate: int
    ) -> ReportSection:
        """Create Recommendations section."""
        recommendations = []

        # Check peak levels
        peak_db = 20 * np.log10(np.max(np.abs(output_audio)) + 1e-10)
        if peak_db > -1.0:
            recommendations.append(
                f"Consider True Peak limiting: Current peak is {peak_db:.1f} dBFS (target: < -1.0 dBTP)"
            )

        # Check RMS levels
        rms_db = 20 * np.log10(np.sqrt(np.mean(output_audio**2)) + 1e-10)
        if rms_db < -30.0:
            recommendations.append(f"Audio is very quiet (RMS: {rms_db:.1f} dBFS). Consider loudness normalization.")

        if not recommendations:
            recommendations.append("Audio meets professional standards. No additional processing recommended.")

        return ReportSection(
            section_type=ReportSectionType.RECOMMENDATIONS,
            title="Recommendations",
            content={"recommendations": recommendations},
            summary=f"{len(recommendations)} recommendation(s)",
        )

    def _generate_overall_summary(self, report: ProcessingReport) -> str:
        """Generate overall summary from report sections."""
        summaries = []

        for section in report.sections:
            if section.summary:
                summaries.append(section.summary)

        return " • ".join(summaries[:3])  # First 3 summaries

    def _calculate_overall_confidence(self, report: ProcessingReport) -> float:
        """Calculate overall confidence from confidence scores."""
        confidence_section = report.get_section(ReportSectionType.CONFIDENCE_SCORES)

        if confidence_section and "scores" in confidence_section.content:
            scores = confidence_section.content["scores"]
            if scores:
                return float(np.mean(list(scores.values())))

        return 0.8  # Default medium-high confidence

    def export_report(self, report: ProcessingReport, output_path: Path, format: str = "markdown") -> bool:
        """
        Export report to file.

        Args:
            report: ProcessingReport
            output_path: Output file path
            format: "markdown", "json", or "pdf"

        Returns:
            True if successful
        """
        output_path = Path(output_path)

        if format.lower() == "markdown":
            return ReportExporter.export_markdown(report, output_path)
        elif format.lower() == "json":
            return ReportExporter.export_json(report, output_path)
        elif format.lower() == "pdf":
            return ReportExporter.export_pdf(report, output_path)
        else:
            logger.error(f"Unknown Ausgabe format: {format}")
            return False


# === Example Usage ===
if __name__ == "__main__":
    import soundfile as sf

    # Load audio
    input_audio, sr = sf.read("test_audio/test_input.wav")
    output_audio, _ = sf.read("test_output/processed.wav")

    # Mock processing history
    history = {
        "detected_issues": [
            {
                "type": "Background Noise",
                "severity": "MEDIUM",
                "description": "Moderate background noise detected (SNR: 18 dB)",
                "confidence": 0.85,
            },
            {
                "type": "Click Artifacts",
                "severity": "LOW",
                "description": "3 click artifacts detected",
                "confidence": 0.75,
            },
        ],
        "applied_modules": [
            {"name": "Spectral Denoiser", "reason": "Reduce background noise", "strength": 0.30},
            {"name": "Click Removal", "reason": "Remove detected click artifacts", "strength": 0.50},
        ],
        "parameters": {
            "processing_mode": "RESTORATION",
            "denoise_strength": 0.30,
            "click_removal_sensitivity": 0.50,
            "preserve_breaths": True,
        },
        "confidence_scores": {"Noise Detection": 0.85, "Click Detection": 0.75, "Overall Processing": 0.90},
    }

    # Generate report
    generator = ProcessingReportGenerator()

    report = generator.create_report(
        input_audio=input_audio,
        output_audio=output_audio,
        sample_rate=sr,
        processing_history=history,
        input_file="test_input.wav",
        output_file="processed.wav",
        processing_time_sec=12.5,
    )

    # Export Markdown
    generator.export_report(report, Path("test_output/processing_report.md"), format="markdown")

    # Export JSON
    generator.export_report(report, Path("test_output/processing_report.json"), format="json")

    print("\n✅ Processing Report Generated")
    print(f"   Report ID: {report.report_id}")
    print(f"   Sections: {len(report.sections)}")
    print(f"   Overall Confidence: {report.overall_confidence:.0%}")
