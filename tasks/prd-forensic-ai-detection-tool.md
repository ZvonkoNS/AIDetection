## PRD: Forensic AI Detection Tool (AIDT)

- **Status**: Draft
- **Author**: Project Team
- **Last Updated**: October 2, 2025

### 1. Introduction / Overview

Law enforcement and digital forensics teams increasingly encounter AI‑generated images during investigations. Determining the authenticity of imagery is critical to maintain evidentiary integrity and avoid miscarriages of justice. The Forensic AI Detection Tool (AIDT) is a command‑line application that operates fully offline to analyze images and produce a clear, defensible verdict on whether an image is likely human‑captured or AI‑generated, accompanied by a detailed, auditable report.

### 2. Goals

- **Reliable authenticity verdicts**: Provide a top‑line verdict of AI‑Generated vs Likely Human with a calibrated confidence score.
- **Offline operation**: Run fully air‑gapped with all models and assets bundled into a single distributable executable.
- **Forensic reporting**: Produce succinct summaries and detailed, step‑by‑step evidence suitable for reports and court exhibits.
- **Performance**: Analyze a single ≤10MB image on a modern CPU in under 15 seconds.
- **Batch efficiency**: Support folder analysis with progress visibility and per‑image detailed outputs plus a CSV summary.
- **Cross‑platform portability**: Distribute builds for Windows (x64), Linux (x64), and macOS (ARM64/x64).

### 3. User Stories

- As an investigator, I want to analyze a single image and get a clear verdict with a confidence score so I can decide evidentiary next steps.
- As a lab analyst, I want to analyze all images in a folder with a progress bar so I can process large evidence sets efficiently.
- As a report writer, I want a human‑readable PDF report and machine‑readable JSON so I can include findings in case files and automate downstream workflows.
- As an auditor, I want a SHA256 file hash included in reports so I can verify integrity and chain of custody.
- As an IT/security admin, I want the tool to run fully offline so it can be used on air‑gapped machines without network access.

### 4. Functional Requirements

FR‑01. Multi‑Mechanism Analysis Engine

- Combine multiple detection vectors into an ensemble score:
  - Ensemble of at least two state‑of‑the‑art classifiers (e.g., CNN such as XceptionNet and Transformer such as ViT) operating on image pixels.
  - Metadata (EXIF) analysis to flag anomalies (e.g., missing camera data, inconsistent timestamps, or software tags like "Stable Diffusion", "Midjourney", "InvokeAI").
  - Frequency domain analysis via Fourier transform to detect unnatural spectral patterns (e.g., periodic artifacts, atypical noise profiles).
  - Known artifact detection for fingerprints typical of generative models (e.g., GAN upsampling traces, diffusion noise artifacts).
- Produce per‑mechanism scores and an overall weighted confidence with rationale.

FR‑02. Command‑Line Interface (CLI)

- Primary command for single image or directory:
  - `aidetect analyze --input <path/to/image.jpg>`
  - `aidetect analyze --input <path/to/folder/>`
- Output format selection:
  - `--format text` (default, prints to console)
  - `--format json` (writes JSON report)
  - `--format pdf` (writes PDF report)
- Display a progress bar for directory (batch) analysis.

FR‑03. Detailed Reporting

- Top‑line summary per image:
  - File SHA256 hash
  - Overall verdict: `AI‑GENERATED` or `LIKELY HUMAN`
  - Confidence score as a percentage
- Detailed breakdown per mechanism including key indicators, flags, and per‑model probabilities.
- Report artifacts:
  - Console text for interactive runs
  - JSON report per image when `--format json` is specified
  - PDF report per image when `--format pdf` is specified

FR‑04. Batch Processing

- When `--input` is a directory, analyze all supported image files within it.
- Emit a `summary_report.csv` in the target directory containing filename, verdict, and confidence.
- Produce individual JSON/PDF reports per image when requested.

FR‑05. Supported Inputs and Limits

- Support common forensic image types: JPEG/JPG, PNG, TIFF. (Assumption; see Open Questions.)
- Maximum single file size target: ≤10MB for performance objective; larger files are allowed but may exceed performance targets.

FR‑06. Determinism and Logging

- Ensure run‑to‑run determinism for the same input and version (e.g., fixed seeds, consistent preprocessing).
- Provide a `--log-level` flag and structured logs suitable for audit trails.

### 5. Non‑Goals (Out of Scope)

- Graphical User Interface (GUI)
- Video analysis
- Third‑party forensic suite integrations via API
- On‑device model re‑training or updates by end users

### 6. Design Considerations (Optional)

- CLI UX: clear errors, helpful `--help`, and explicit exit codes for automation.
- Report clarity: concise top‑line summary with expandable detail sections in PDF; JSON schemas versioned.
- Evidence safety: avoid modifying input files; write outputs to a configurable directory.

### 7. Technical Considerations (Optional)

- Language/runtime: Python 3.10+ with type hints and modular architecture.
- Packaging: Produce single‑file executables (e.g., PyInstaller/Nuitka) with all models and assets bundled; strictly offline, no network calls.
- Performance: CPU‑only inference; FFT/frequency analysis via optimized libraries; careful image I/O and preprocessing.
- PDF generation: Use an offline‑friendly library; embed fonts/images as needed to avoid external dependencies.
- Metadata parsing: Robust EXIF handling; tolerate malformed metadata without crashing.
- Cross‑platform builds: Windows (x64), Linux (x64), macOS (ARM64/x64).
- Testing: Unit tests for analysis components, integration tests for CLI, and golden tests for report outputs.

### 8. Success Metrics

- ≥90% accuracy on an internal, curated benchmark distinguishing AI‑generated vs real photographs across multiple generators and camera datasets.
- ≤15s median analysis time per ≤10MB image on a modern CPU.
- 0 network calls observed during runtime (validated by e2e tests and static checks).
- Successful execution and consistent outputs across Windows, Linux, and macOS target architectures.

### 9. Open Questions

1. Which exact classifier architectures and pretrained checkpoints should be used and how will their licenses impact distribution?
2. What are the weighting/thresholding rules for the ensemble, and will they be configurable?
3. Which image formats beyond JPEG/PNG/TIFF must be supported (e.g., HEIC, RAW/CR2/NEF)?
4. What is the desired JSON schema and PDF layout template (branding, redactions, localization)?
5. Are there lab‑specific constraints on output directories, naming conventions, or evidence tagging?
6. Is GPU acceleration explicitly out of scope for v1, or should we optionally detect and use it when present?
7. What is the maximum acceptable memory footprint during batch processing on typical lab machines?
8. What evaluation datasets will be used to validate the ≥90% accuracy target, and who will curate them?
9. Should reports include sanitized excerpts of metadata and frequency plots, and if so, at what detail level?

### 10. Assumptions (Until Clarified)

- Initial supported formats are JPEG/JPG, PNG, and TIFF.
- PDF generation will use an offline‑capable library and embed necessary assets.
- CPU‑only inference is sufficient for v1 performance goals on modern lab machines.



