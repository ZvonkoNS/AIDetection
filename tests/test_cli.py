from aidetect.cli import _derive_output_path
from aidetect.core.types import ReportFormat


def test_derive_output_path_respects_output_dir(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    input_file = input_dir / "sample.jpg"
    input_file.touch()

    output_dir = tmp_path / "reports"
    result_path = _derive_output_path(input_file, ReportFormat.JSON, output_dir, input_dir)

    assert result_path.parent == output_dir
    assert result_path.name == "sample_report.json"


def test_derive_output_path_preserves_relative_structure(tmp_path):
    base = tmp_path / "root"
    nested = base / "subdir"
    nested.mkdir(parents=True)
    input_file = nested / "image.png"
    input_file.touch()

    output_dir = tmp_path / "reports"
    result_path = _derive_output_path(input_file, ReportFormat.JSON, output_dir, base)
    assert (output_dir / "subdir").exists()
    assert result_path.name == "image_report.json"
