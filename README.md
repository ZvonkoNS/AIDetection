# AIDetection

**AIDetection** is a forensic tool designed to detect AI-generated images through a multi-mechanism analysis approach. It examines various aspects of an image, including metadata, frequency domains, texture patterns, and compression artifacts to provide a robust assessment of its origin.

This project is open-source and licensed under the terms of the MIT license.

![Next Sight Banner](https://raw.githubusercontent.com/ZvonkoNS/AIDetection/main/assets/brand_banner.txt)

## Key Features

*   **Comprehensive Analysis**: Employs a suite of detection mechanisms for high accuracy:
    *   **Metadata Forensics**: Analyzes EXIF, GPS, and other embedded metadata for signs of manipulation.
    *   **Frequency Analysis**: Detects unnatural patterns in the frequency domain using DCT and FFT.
    *   **Texture Analysis**: Identifies synthetic textures using Local Binary Patterns (LBP).
    *   **Compression Analysis**: Finds artifacts indicative of digital alteration or generation.
    *   **Pixel Statistics**: Examines statistical properties of pixels for anomalies.
*   **Multiple Report Formats**: Generates user-friendly reports in `TEXT`, `JSON`, `PDF`, and a summary `CSV` for batch operations.
*   **Flexible Interface**: Can be operated via a straightforward command-line interface (CLI) or a simple interactive menu.
*   **High Performance**: Optimized for fast analysis, with support for parallel processing in batch mode.
*   **Cross-Platform**: Fully compatible with Windows, macOS, and Linux.

## Installation

To get started with AIDetection, it is recommended to use a virtual environment.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ZvonkoNS/AIDetection.git
    cd AIDetection
    ```

2.  **Create and Activate a Virtual Environment:**
    *   On **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   On **macOS/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

AIDetection is launched via the `run_aidetect.py` script.

### Interactive Mode

For ease of use, you can run the tool without any arguments to launch an interactive menu.

```bash
python run_aidetect.py
```

The interactive mode will guide you through analyzing a single image or an entire directory.

### Command-Line Interface (CLI)

For automation and advanced use, the CLI provides full control over the tool's features.

#### **Analyze a Single Image**

The default output is a text report printed to the console.

```bash
python run_aidetect.py analyze --input /path/to/your/image.jpg
```

To save a report in a different format, such as JSON:

```bash
python run_aidetect.py analyze --input /path/to/image.jpg --format json --output-dir /path/to/reports
```

#### **Analyze a Directory of Images**

When analyzing a directory, a summary report (`summary_report.csv`) will be generated in the output directory.

```bash
python run_aidetect.py analyze --input /path/to/your/directory
```

For a deep scan of a directory and its subdirectories, use the `--recursive` flag. You can also increase performance by using multiple processor cores with the `--workers` flag.

```bash
python run_aidetect.py analyze --input /path/to/your/directory --format pdf --recursive --workers 4
```

### **Command-Line Arguments**

| Argument               | Alias | Description                                                                    |
| ---------------------- | ----- | ------------------------------------------------------------------------------ |
| `--input`              | `-i`  | Path to the input image file or directory. **(Required)**                        |
| `--format`             | `-f`  | Output report format (`text`, `json`, `pdf`). Default is `text`.                 |
| `--output-dir`         |       | Directory to save generated reports. Defaults to the input directory.          |
| `--recursive`          |       | Recursively scan for images in subdirectories.                                 |
| `--workers`            |       | Number of parallel workers for batch analysis.                                 |
| `--log-level`          |       | Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).         |
| `--conservative`       |       | Use a higher threshold for AI detection to reduce false positives.               |
| `--config`             |       | Path to a custom TOML/YAML configuration file.                                 |
| `--weights`            |       | Override ensemble weights (e.g., `'METADATA=0.2,FREQUENCY=0.3'`).              |
| `--about`              |       | Show product banner and information.                                           |
| `--interactive`        |       | Launch the simple interactive menu.                                            |

## Building a Standalone Executable

You can create a standalone executable for your platform by using the provided build scripts.

*   On **Windows**:
    ```bash
    .\build.bat
    ```
*   On **macOS/Linux**:
    ```bash
    chmod +x build.sh
    ./build.sh
    ```

The executable will be located in the `dist/` directory.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository, create a new branch for your feature, and submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
