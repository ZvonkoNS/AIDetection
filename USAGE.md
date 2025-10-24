# How to Use AIDETECT

## The Executable is a Command-Line Tool

The `aidetect.exe` file is **not** a graphical application that you can double-click and use. It's a **command-line interface (CLI)** tool, which means you need to run it from a terminal/command prompt with specific arguments.

## Why Does it Close Immediately?

When you double-click `aidetect.exe`, Windows opens it, but since you haven't provided any arguments (like which image to analyze), it either shows a brief help message or exits immediately. This is **normal behavior** for CLI tools.

## How to Use It Correctly

### Option 1: Use the Test Script (Easiest)

Double-click `test-aidetect.bat` to see the help information and learn how to use the tool.

### Option 2: Use PowerShell or Command Prompt

1. **Open PowerShell** (or Command Prompt) in the project directory
2. Run one of these commands:

#### View Help
```powershell
.\dist\aidetect.exe --help
```

#### Analyze a Single Image
```powershell
.\dist\aidetect.exe analyze --input "C:\path\to\your\image.jpg"
```

#### Analyze a Directory of Images
```powershell
.\dist\aidetect.exe analyze --input "C:\path\to\your\images\folder"
```

#### Generate a JSON Report
```powershell
.\dist\aidetect.exe analyze --input "C:\path\to\image.jpg" --format json
```

#### Set Logging Level
```powershell
.\dist\aidetect.exe --log-level DEBUG analyze --input "C:\path\to\image.jpg"
```

#### Set Random Seed (for reproducibility)
```powershell
.\dist\aidetect.exe --seed 42 analyze --input "C:\path\to\image.jpg"
```

#### Show banner/about or open interactive menu
```powershell
.\dist\aidetect.exe --about
.\dist\aidetect.exe --interactive
```

## Example Output

When you run the tool on an image, you'll see output like:

```
Global random seed set to 42
--- Analysis Report for myimage.jpg ---
Verdict: LIKELY HUMAN (Confidence: 85.2%)

Mechanisms:
- METADATA: 10.0% AI Probability
- FREQUENCY: 50.0% AI Probability
- CLASSIFIER_XCEPTION: 50.0% AI Probability
- CLASSIFIER_VIT: 50.0% AI Probability
- ARTIFACTS: 50.0% AI Probability

Processed in: 234 ms
```

## Batch Processing

When analyzing a directory, the tool will:
1. Process all supported images (.jpg, .png, .tiff)
2. Show a progress bar (if `tqdm` is installed)
3. Print a report for each image
4. Generate a `summary_report.csv` in the same directory

## Distribution

To share the tool with others:
1. Copy the entire `dist` folder to the target machine
2. The executable is **portable** and **works offline** - no Python installation required!
3. Provide them with this USAGE guide

## Troubleshooting

**Q: The window flashes and closes immediately**  
A: This is normal. Use PowerShell or the test batch script instead.

**Q: It says "command not found"**  
A: Make sure you're in the correct directory and use `.\` prefix (PowerShell) or just the name (Command Prompt).

**Q: Where are my reports saved?**  
A: 
- Text reports are printed to the console
- JSON reports are saved next to the input file with `_report.json` suffix
- CSV summaries for batch runs are saved as `summary_report.csv` in the input directory

---
NEXT SIGHT cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com

