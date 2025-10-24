#!/usr/bin/env python
"""
Entrypoint script for the aidetect CLI.
This allows PyInstaller to properly package the application.
"""

from aidetect.cli import main

if __name__ == "__main__":
    main()

