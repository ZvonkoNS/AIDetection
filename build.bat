@echo off
setlocal enabledelayedexpansion

:: Set a short, clean temporary directory to avoid long path issues
set "PIP_TEMP_DIR=C:\pip-temp"
echo "Setting temporary directory to %PIP_TEMP_DIR%"
mkdir "%PIP_TEMP_DIR%" >nul 2>nul
set "TEMP=%PIP_TEMP_DIR%"
set "TMP=%PIP_TEMP_DIR%"

echo "--- Upgrading core build tools ---"
python -m pip install --upgrade pip setuptools wheel
if !errorlevel! neq 0 (
    echo "ERROR: Failed to upgrade build tools."
    goto:failure
)

echo "--- Installing dependencies (requiring binary wheels) ---"
rem This is a stricter approach. It will fail fast if a pre-compiled wheel
rem is not available for your Python version and architecture.
pip install --only-binary=:all: -r requirements.txt
if !errorlevel! neq 0 (
    echo "ERROR: Failed to install dependencies from requirements.txt."
    echo "This may mean a pre-compiled package is not available for your Python version."
    goto:failure
)

echo "--- Installing development dependencies ---"
pip install --only-binary=:all: -r requirements-dev.txt
if !errorlevel! neq 0 (
    echo "ERROR: Failed to install development dependencies."
    goto:failure
)

echo "--- Building executable with PyInstaller ---"
pyinstaller --name aidetect --onefile --console run_aidetect.py
if !errorlevel! neq 0 (
    echo "ERROR: Failed to build the executable."
    goto:failure
)

echo "--- Build Complete ---"
echo "Executable is in the 'dist' directory."
goto:cleanup

:failure
echo.
echo "--- Build script failed. ---"
goto:cleanup

:cleanup
echo "Cleaning up temporary directory..."
rmdir /s /q "%PIP_TEMP_DIR%" >nul 2>nul
endlocal
