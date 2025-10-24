import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

from aidetect.core.branding import BRAND_ASCII, BRAND_LINE, PROJECT_TAGLINE, about_text, quick_usage_block

from aidetect.core.config import AppConfig, parse_weights_override
from aidetect.core.determinism import set_seed
from aidetect.core.logging import setup_logging
from aidetect.core.types import AnalysisResult, ReportFormat
from aidetect.reporting.csv import write_summary_csv
from aidetect.reporting.json import result_to_json_str
from aidetect.reporting.pdf import write_pdf_report
from aidetect.reporting.text import format_text_report
from aidetect.runner import run_batch_analysis, run_single_analysis

logger = logging.getLogger(__name__)


def _derive_output_path(
    source_path: Path,
    report_format: ReportFormat,
    output_dir: Optional[Path],
    base_input: Optional[Path] = None,
) -> Path:
    """
    Determine where to place a generated report. If an output directory is provided,
    reports mirror the relative structure beneath it.
    """
    if output_dir is not None:
        if base_input is not None:
            try:
                relative = source_path.relative_to(base_input)
            except ValueError:
                relative_path = Path(source_path.name)
            else:
                relative_path = relative.with_suffix("")
        else:
            relative_path = Path(source_path.name).with_suffix("")
        dest = output_dir / relative_path.parent / f"{relative_path.stem}_report.{report_format.value}"
    else:
        dest = source_path.with_name(
            f"{source_path.stem}_report.{report_format.value}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _log_mechanism_summary(result: AnalysisResult, level: int = logging.INFO) -> None:
    summary = result.debug.get("mechanism_summary", {})
    breakdown = summary.get("status_breakdown", {})
    message = (
        "Mechanism execution summary - ran: {ran}, skipped: {skipped}, errors: {errors}"
    ).format(
        ran=breakdown.get("ran", 0),
        skipped=breakdown.get("skipped", 0),
        errors=breakdown.get("error", 0),
    )
    logger.log(level, message)


def _write_json_report(result: AnalysisResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result_to_json_str(result), encoding="utf-8")
    logger.info("JSON report saved to %s", path)


def _write_pdf_report(result: AnalysisResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pdf_report(result, path)
    logger.info("PDF report saved to %s", path)


def handle_single_file(
    input_path: Path,
    report_format: ReportFormat,
    output_dir: Optional[Path],
    config: AppConfig,
) -> None:
    """Process a single file based on CLI arguments."""
    result = run_single_analysis(str(input_path), config)
    _log_mechanism_summary(result)
    if report_format is ReportFormat.TEXT:
        print(format_text_report(result))
    elif report_format is ReportFormat.JSON:
        output_path = _derive_output_path(input_path, report_format, output_dir)
        _write_json_report(result, output_path)
    elif report_format is ReportFormat.PDF:
        output_path = _derive_output_path(input_path, report_format, output_dir)
        _write_pdf_report(result, output_path)
    else:  # pragma: no cover - future formats
        raise NotImplementedError(f"Unsupported format: {report_format}")


def _write_directory_reports(
    results: Iterable[AnalysisResult],
    report_format: ReportFormat,
    output_dir: Optional[Path],
    input_root: Path,
) -> None:
    for result in results:
        source_path = Path(result.file.path)
        dest = _derive_output_path(source_path, report_format, output_dir, input_root)
        if report_format is ReportFormat.JSON:
            _write_json_report(result, dest)
        elif report_format is ReportFormat.PDF:
            _write_pdf_report(result, dest)


def handle_directory(
    input_path: Path,
    report_format: ReportFormat,
    output_dir: Optional[Path],
    recursive: bool,
    workers: int,
    config: AppConfig,
) -> None:
    """Process a directory based on CLI arguments."""
    results: List[AnalysisResult] = list(
        run_batch_analysis(
            str(input_path),
            config,
            recursive=recursive,
            workers=workers,
        )
    )
    if not results:
        return

    aggregate_status = Counter()
    for result in results:
        summary = result.debug.get("mechanism_summary", {})
        breakdown = summary.get("status_breakdown", {})
        aggregate_status.update(breakdown)

    logger.info(
        "Batch mechanism summary - ran: %d, skipped: %d, errors: %d",
        aggregate_status.get("ran", 0),
        aggregate_status.get("skipped", 0),
        aggregate_status.get("error", 0),
    )

    if report_format is ReportFormat.TEXT:
        for result in results:
            _log_mechanism_summary(result, level=logging.DEBUG)
            print(format_text_report(result))
            print()
    elif report_format in {ReportFormat.JSON, ReportFormat.PDF}:
        _write_directory_reports(results, report_format, output_dir, input_path)

    summary_dir = output_dir if output_dir is not None else input_path
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "summary_report.csv"
    write_summary_csv(results, csv_path)
    logger.info("Summary CSV saved to %s", csv_path)


def _print_splash() -> None:
    print(BRAND_ASCII)
    print()
    print(PROJECT_TAGLINE)
    print(BRAND_LINE)
    print()


def _interactive_menu() -> None:
    while True:
        _print_splash()
        print("Interactive mode:\n")
        print("  1) Analyze a single image")
        print("  2) Analyze a directory")
        print("  3) About")
        print("  4) Help")
        print("  5) Exit")
        try:
            choice = input("Select an option [1-5]: ").strip()
        except EOFError:
            return
        if choice == "1":
            target = input("Enter image path: ").strip()
            if target:
                sys.argv = [sys.argv[0], "analyze", "--input", target]
                main()
        elif choice == "2":
            target = input("Enter directory path: ").strip()
            fmt = input("Format [text/json/pdf] (default text): ").strip() or "text"
            recursive = input("Recursive? [y/N]: ").strip().lower() in {"y", "yes"}
            workers = input("Workers (blank = default): ").strip()
            args = [sys.argv[0], "analyze", "--input", target, "--format", fmt]
            if recursive:
                args.append("--recursive")
            if workers:
                args.extend(["--workers", workers])
            sys.argv = args
            main()
        elif choice == "3":
            _print_splash()
            print(about_text())
            print()
            print(quick_usage_block())
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            _print_splash()
            parser = argparse.ArgumentParser(description="Forensic AI Detection Tool (AIDT)")
            parser.print_help()
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            return
        else:
            print("Invalid selection. Please choose 1-5.")
            input("\nPress Enter to try again...")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Forensic AI Detection Tool (AIDT)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Enable conservative mode (reduce false positives, require stronger AI evidence)."
    )
    parser.add_argument(
        "--config",
        help="Path to a TOML/YAML configuration file (overrides pyproject defaults).",
    )
    parser.add_argument(
        "--weights",
        help="Override ensemble weights (e.g. 'METADATA=0.2,FREQUENCY=0.3').",
    )
    parser.add_argument(
        "--about",
        action="store_true",
        help="Show product banner, company info, and a short description.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch a simple interactive menu.",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a single image or a directory of images."
    )
    analyze_parser.add_argument(
        "--input", "-i", required=True, help="Path to an image file or a directory."
    )
    analyze_parser.add_argument(
        "--format",
        "-f",
        default=ReportFormat.TEXT.value,
        choices=[f.value for f in ReportFormat],
        help="Output report format.",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to place generated reports (defaults to alongside inputs).",
    )
    analyze_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recursively scan directories for supported images.",
    )
    analyze_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum parallel workers for batch analysis (default from config).",
    )

    args = parser.parse_args()

    if args.about and not args.command:
        _print_splash()
        print(about_text())
        print()
        print(quick_usage_block())
        sys.exit(0)

    if args.interactive and not args.command:
        _interactive_menu()
        sys.exit(0)

    # --- Setup ---
    setup_logging(args.log_level)
    config_path = None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            logger.critical("Configuration file not found: %s", config_path)
            sys.exit(1)
    config = AppConfig.load(config_path=config_path)
    if config.config_source:
        logger.debug("Loaded configuration from %s", config.config_source)

    if args.seed is not None:
        config.seed = args.seed

    if args.conservative or config.conservative_mode:
        config.conservative_mode = True
        logger.info(
            "Conservative mode enabled - higher threshold, requires positive AI evidence"
        )

    if args.weights:
        try:
            overrides = parse_weights_override(args.weights)
        except ValueError as exc:
            logger.critical("Invalid weights override: %s", exc)
            sys.exit(1)
        config.apply_weight_overrides(overrides)
        logger.debug("Applied weight overrides: %s", overrides)

    set_seed(config.seed)

    if not args.command:
        # No subcommand provided: launch interactive menu by default
        _interactive_menu()
        sys.exit(0)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        logger.critical("Input path does not exist: %s", input_path)
        sys.exit(1)

    report_format = ReportFormat(args.format)

    output_dir: Optional[Path] = args.output_dir
    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    recursive = (
        args.recursive
        if args.recursive is not None
        else config.default_recursive
    )
    workers = args.workers if args.workers is not None else config.default_workers
    workers = max(1, workers or 1)

    try:
        if input_path.is_file():
            handle_single_file(input_path, report_format, output_dir, config)
        elif input_path.is_dir():
            handle_directory(
                input_path,
                report_format,
                output_dir,
                recursive=bool(recursive),
                workers=workers,
                config=config,
            )
    except NotImplementedError as e:
        logger.critical(f"Feature not implemented: {e}")
        sys.exit(1)
    except Exception:
        logger.critical("An unexpected error occurred", exc_info=True)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
