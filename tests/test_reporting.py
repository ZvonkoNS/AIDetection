import csv
import json

from aidetect.core.types import AnalysisResult, FileInfo, MechanismResult, MechanismType, Verdict
from aidetect.reporting.csv import write_summary_csv
from aidetect.reporting.json import result_to_json_str


def _build_result() -> AnalysisResult:
    file_info = FileInfo(path="example.jpg", sha256="a" * 64, size_bytes=1024)
    mechanisms = [
        MechanismResult(mechanism=MechanismType.METADATA, probability_ai=0.7),
        MechanismResult(mechanism=MechanismType.CLASSIFIER_XCEPTION, probability_ai=0.4),
    ]
    result = AnalysisResult(
        file=file_info,
        verdict=Verdict.AI_GENERATED,
        confidence=0.65,
        mechanisms=mechanisms,
        confidence_level="HIGH",
        detection_rationale=["Metadata contained AI generator tags."],
    )
    result.processing_ms = 120
    result.debug["mechanisms"] = [
        {
            "mechanism": MechanismType.METADATA.value,
            "status": "ran",
            "elapsed_ms": 25,
            "missing_dependencies": [],
        },
        {
            "mechanism": MechanismType.CLASSIFIER_XCEPTION.value,
            "status": "skipped",
            "elapsed_ms": 0,
            "missing_dependencies": ["onnxruntime"],
        },
    ]
    result.debug["mechanism_summary"] = {
        "status_breakdown": {"ran": 1, "skipped": 1},
        "total_mechanisms": 2,
    }
    return result


def test_json_output_includes_run_diagnostics():
    result = _build_result()
    payload = json.loads(result_to_json_str(result, indent=None))
    diagnostics = payload["run_diagnostics"]
    assert diagnostics["status_breakdown"]["ran"] == 1
    assert diagnostics["status_breakdown"]["skipped"] == 1
    assert diagnostics["total_mechanisms"] == 2


def test_csv_output_contains_status_columns(tmp_path):
    result = _build_result()
    destination = tmp_path / "summary.csv"
    write_summary_csv([result], destination)

    with destination.open("r", encoding="utf-8") as fh:
        rows = [row for row in csv.reader(fh) if row]

    # Skip comment lines and find the header row
    header = rows[4]  # Now at row 4 due to additional brand comment line
    assert "metadata_status" in header
    assert "xception_status" in header
    data_row = rows[5]  # Data row is now at row 5
    assert "ran" in data_row
    assert "skipped" in data_row
