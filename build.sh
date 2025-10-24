#!/bin/bash
set -e

echo "--- Installing Dependencies ---"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo "\n--- Building Executable with PyInstaller ---"
pyinstaller --name aidetect --onefile --console run_aidetect.py

echo "\n--- Build Complete ---"
echo "Executable is in the 'dist' directory."
