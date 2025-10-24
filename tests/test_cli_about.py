from __future__ import annotations

import subprocess
import sys


def test_cli_about_runs_and_prints_banner():
    # Run module with --about and capture output
    proc = subprocess.run(
        [sys.executable, "-m", "aidetect.cli", "--about"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert "NEXT SIGHT" in out
    assert "contact us at info@next-sight.com" in out

