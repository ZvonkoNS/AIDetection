import pytest

from aidetect.core.types import AnalysisResult, FileInfo, MechanismResult, MechanismType, Verdict
from aidetect.reporting import pdf


def _minimal_result() -> AnalysisResult:
    file_info = FileInfo(path="example.jpg", sha256="b" * 64, size_bytes=2048)
    mechanisms = [MechanismResult(mechanism=MechanismType.METADATA, probability_ai=0.5)]
    result = AnalysisResult(
        file=file_info,
        verdict=Verdict.LIKELY_HUMAN,
        confidence=0.4,
        mechanisms=mechanisms,
    )
    result.processing_ms = 50
    result.debug["mechanisms"] = [
        {
            "mechanism": MechanismType.METADATA.value,
            "status": "ran",
            "elapsed_ms": 5,
            "missing_dependencies": [],
        }
    ]
    result.debug["mechanism_summary"] = {
        "status_breakdown": {"ran": 1},
        "total_mechanisms": 1,
    }
    return result


def test_pdf_report_requires_fpdf(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf, "FPDF", None)
    monkeypatch.setattr(pdf, "_PDF_IMPORT_ERROR", ImportError("missing fpdf2"))
    result = _minimal_result()

    with pytest.raises(NotImplementedError):
        pdf.write_pdf_report(result, tmp_path / "report.pdf")
