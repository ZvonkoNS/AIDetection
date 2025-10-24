## Relevant Files

- `aidetect/cli.py` - CLI entrypoint and argument parsing.
- `aidetect/runner/single.py` - Single-image analysis runner.
- `aidetect/runner/batch.py` - Directory traversal and progress bar.
- `aidetect/core/types.py` - Core data models, enums, and result schemas.
- `aidetect/core/config.py` - Configuration and defaults.
- `aidetect/core/logging.py` - Structured logging setup.
- `aidetect/core/determinism.py` - Seeding and deterministic execution utilities.
- `aidetect/io/image_loader.py` - Image I/O and preprocessing.
- `aidetect/analysis/metadata.py` - EXIF/metadata extraction and heuristics.
- `aidetect/analysis/frequency.py` - FFT/frequency-domain analysis utilities.
- `aidetect/analysis/artifacts.py` - Known artifact detectors for generative models.
- `aidetect/models/xception.py` - XceptionNet classifier wrapper (CPU inference).
- `aidetect/models/vit.py` - ViT classifier wrapper (CPU inference).
- `aidetect/analysis/ensemble.py` - Ensemble weighting and overall scoring.
- `aidetect/analysis/calibration.py` - Score calibration and thresholds.
- `aidetect/reporting/text.py` - Console/text renderer.
- `aidetect/reporting/json.py` - JSON writer with schema versioning.
- `aidetect/reporting/pdf.py` - PDF report generator (offline-capable).
- `aidetect/reporting/csv.py` - Summary CSV writer for batch runs.
- `assets/` - PDF templates, fonts, and embedded resources (offline).
- `models/` - Bundled model weights and metadata.
- `pyproject.toml` - Build configuration.
- `requirements.txt` - Pinned runtime dependencies.

### Notes

- Tests should be colocated with modules they cover and run via `pytest`.
- JSON schema and PDF templates should be versioned.
- Avoid network calls; all assets and models must be local.

## Tasks

- [x] 1.0 Build multi-mechanism analysis engine (ensemble, metadata, frequency, artifacts)
  - [x] 1.1 Define core result models and enums in `aidetect/core/types.py`
  - [x] 1.2 Implement image I/O and preprocessing in `aidetect/io/image_loader.py`
  - [x] 1.3 Implement EXIF/metadata extraction and heuristics in `aidetect/analysis/metadata.py`
  - [x] 1.4 Implement frequency-domain analysis utilities in `aidetect/analysis/frequency.py`
  - [x] 1.5 Implement known artifact detectors in `aidetect/analysis/artifacts.py`
  - [x] 1.6 Add classifier wrappers for XceptionNet and ViT in `aidetect/models/`
  - [x] 1.7 Implement ensemble weighting and overall scoring in `aidetect/analysis/ensemble.py`
  - [x] 1.8 Implement score calibration and thresholds in `aidetect/analysis/calibration.py`
  - [x] 1.9 Unit tests for analysis components (types, io, metadata, frequency, artifacts, models, ensemble)

- [x] 2.0 Implement CLI and batch processing with progress bar
  - [x] 2.1 Create CLI entrypoint `aidetect/cli.py` with `analyze` command and flags
  - [x] 2.2 Implement single-file analysis runner `aidetect/runner/single.py`
  - [x] 2.3 Implement directory batch runner with progress bar `aidetect/runner/batch.py`
  - [x] 2.4 Integrate engine components; map errors to explicit exit codes
  - [x] 2.5 CLI and runner tests, including error scenarios

- [x] 3.0 Implement reporting outputs: text, JSON, PDF, and summary CSV
  - [x] 3.1 Implement console/text renderer `aidetect/reporting/text.py`
  - [x] 3.2 Define JSON schema and implement writer `aidetect/reporting/json.py`
  - [x] 3.3 Implement PDF report generator `aidetect/reporting/pdf.py`
  - [x] 3.4 Implement summary CSV writer `aidetect/reporting/csv.py`
  - [x] 3.5 Golden/snapshot tests for report content

- [x] 4.0 Implement determinism, logging, and configuration flags
  - [x] 4.1 Add deterministic seeding/utilities `aidetect/core/determinism.py`
  - [x] 4.2 Implement structured logging and `--log-level` `aidetect/core/logging.py`
  - [x] 4.3 Implement config parsing and defaults `aidetect/core/config.py`; wire to CLI
  - [x] 4.4 Input validation and user-friendly error messages
  - [x] 4.5 Tests for determinism, logging, and configuration

- [x] 5.0 Package offline cross-platform executables with bundled models
  - [x] 5.1 Create `pyproject.toml` and `requirements.txt` with pinned versions
  - [x] 5.2 Organize `models/` and `assets/`; add secure local loading
  - [x] 5.3 Build single-file executables for Win/Linux/macOS (PyInstaller/Nuitka)
  - [x] 5.4 Verify fully offline operation (no network calls) in e2e tests
  - [x] 5.5 Smoke test packaged binaries on target OSes

- [ ] 6.0 Testing, benchmarking, and accuracy/performance validation
  - [x] 6.1 Set up `pytest` and CI workflow; code coverage thresholds
  - [x] 6.2 Create fixtures and sample datasets for unit/integration tests
  - [x] 6.3 Build accuracy benchmark and target ≥90% on curated datasets
  - [x] 6.4 Measure performance (≤15s per ≤10MB image) on reference hardware
  - [x] 6.5 Document evaluation methodology and results in `docs/`


