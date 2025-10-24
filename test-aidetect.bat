@echo off
REM Helper script to test the aidetect.exe executable
REM This prevents the window from closing immediately so you can see the output

echo.
echo ========================================
echo  AIDETECT - Forensic AI Detection Tool
echo ========================================
echo.

REM Show help
echo Displaying help information:
echo.
dist\aidetect.exe --help

echo.
echo.
echo ========================================
echo To analyze an image, use:
echo   dist\aidetect.exe analyze --input path\to\image.jpg
echo.
echo To analyze a directory:
echo   dist\aidetect.exe analyze --input path\to\directory
echo.
echo For JSON output:
echo   dist\aidetect.exe analyze --input path\to\image.jpg --format json
echo ========================================
echo.

pause

