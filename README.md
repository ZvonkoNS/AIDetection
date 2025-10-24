# AIDetection

**AIDetection** is a forensic tool designed to detect AI-generated images through a multi-mechanism analysis approach. It examines various aspects of an image, including metadata, frequency domains, texture, quality, and compression artifacts to provide a robust assessment of its origin.

This project is licensed under the terms of the MIT license.

## Features

*   **Multi-Mechanism Analysis**: Combines several detection methods for higher accuracy.
*   **Metadata Forensics**: Analyzes EXIF, GPS, ICC, and MakerNote data.
*   **Frequency-Domain Analysis**: Uses DCT/FFT and spectral residuals.
*   **Texture Pattern Analysis**: Employs LBP, GLCM, and Gabor filters.
*   **Quality Metrics**: Measures sharpness, noise, lens distortion, and banding.
*   **Compression & Pixel Statistics**: Detects quantization, double-compression, and bit-plane anomalies.
*   **Multiple Report Formats**: Generates reports in JSON, PDF, and plain text.
*   **Command-Line and Interactive Modes**: Can be used for single-file analysis, batch processing, or through an interactive menu.

## Installation

To get started with AIDetection, clone the repository and install the required dependencies.

```bash
git clone https://github.com/ZvonkoNS/AIDetection.git
cd AIDetection
pip install -r requirements.txt
```

## Usage

AIDetection can be run from the command line or in an interactive mode.

### Interactive Mode

To launch the interactive menu, simply run the script without any arguments:

```bash
python run_aidetect.py
```

From the menu, you can choose to analyze a single image, a directory of images, or view information about the tool.

### Command-Line Interface (CLI)

#### Analyzing a Single Image

```bash
python run_aidetect.py analyze --input /path/to/your/image.jpg
```

#### Analyzing a Directory

```bash
python run_aidetect.py analyze --input /path/to/your/directory --format json --recursive
```

You can specify the output format (`--format`), and choose to scan recursively (`--recursive`). For batch analysis, you can also set the number of parallel workers (`--workers`).

### Available Arguments

*   `--input`: Path to the image or directory.
*   `--format`: Report format (text, json, pdf).
*   `--output-dir`: Directory to save reports.
*   `--recursive`: Recursively scan directories.
*   `--workers`: Number of parallel workers for batch processing.
*   `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   `--conservative`: Use a higher threshold for AI detection to reduce false positives.
*   `--about`: Show information about the tool.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
