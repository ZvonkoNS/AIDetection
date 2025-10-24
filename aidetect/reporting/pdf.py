from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..core.types import AnalysisResult

try:
    from fpdf import FPDF  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    FPDF = None
    _PDF_IMPORT_ERROR = exc
else:
    _PDF_IMPORT_ERROR = None


def _require_pdf_support() -> None:
    if FPDF is None:
        raise NotImplementedError(
            "PDF reporting requires the 'fpdf2' package. Install it to enable PDF exports."
        ) from _PDF_IMPORT_ERROR


def _add_section_title(pdf: "FPDF", title: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(4)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font("Helvetica", "", 11)


def _mechanism_status_map(result: AnalysisResult) -> Dict[str, Dict[str, Optional[str]]]:
    mechanism_debug = {}
    for entry in result.debug.get("mechanisms", []):
        if isinstance(entry, dict) and entry.get("mechanism"):
            mechanism_debug[entry["mechanism"]] = {
                "status": entry.get("status"),
                "elapsed_ms": entry.get("elapsed_ms"),
            }
    return mechanism_debug


def _write_mechanism_table(pdf: "FPDF", result: AnalysisResult) -> None:
    mechanism_debug = _mechanism_status_map(result)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(70, 8, "Mechanism", border=1)
    pdf.cell(30, 8, "AI Prob.", border=1)
    pdf.cell(30, 8, "Status", border=1)
    pdf.cell(30, 8, "Time (ms)", border=1, ln=True)
    pdf.set_font("Helvetica", "", 11)

    for mechanism in result.mechanisms:
        debug = mechanism_debug.get(mechanism.mechanism.value, {})
        status = debug.get("status") or "n/a"
        elapsed = debug.get("elapsed_ms")
        elapsed_str = str(elapsed) if elapsed not in (None, "") else "-"

        pdf.cell(70, 8, mechanism.mechanism.value.replace("_", " ").title(), border=1)
        pdf.cell(30, 8, f"{mechanism.probability_ai * 100:5.1f}%", border=1)
        pdf.cell(30, 8, status, border=1)
        pdf.cell(30, 8, elapsed_str, border=1, ln=True)


def write_pdf_report(result: AnalysisResult, output_path: str | Path) -> None:
    """
    Generate a concise PDF report summarizing the analysis result.
    """
    _require_pdf_support()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AI Image Detection Report", ln=True)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"File: {Path(result.file.path).name}", ln=True)
    pdf.cell(0, 8, f"SHA-256: {result.file.sha256}", ln=True)
    pdf.cell(0, 8, f"Size: {result.file.size_bytes:,} bytes", ln=True)

    _add_section_title(pdf, "Verdict")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"{result.verdict.value} ({result.confidence:.1%} / {result.confidence_level})", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Confidence explanation: "
        f"{result.detection_rationale[0] if result.detection_rationale else 'See mechanism breakdown.'}",
    )

    if result.detection_rationale:
        _add_section_title(pdf, "Key Findings")
        pdf.set_font("Helvetica", "", 11)
        for idx, finding in enumerate(result.detection_rationale[:5], start=1):
            pdf.multi_cell(0, 6, f"{idx}. {finding}")

    _add_section_title(pdf, "Mechanism Breakdown")
    _write_mechanism_table(pdf, result)

    _add_section_title(pdf, "Processing Details")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Processing Time: {result.processing_ms} ms", ln=True)
    pdf.cell(0, 6, f"Schema Version: {result.schema_version}", ln=True)
    
    # Add Next Sight branding footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, "NEXT SIGHT cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com", ln=True, align="C")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
