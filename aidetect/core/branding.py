from __future__ import annotations

from textwrap import dedent


# Centralized branding and messaging for consistent reuse across CLI and docs

BRAND_ASCII = dedent(
    r"""
 _   _           _     ____  _       _     _   
| \ | | _____  _| |_  / ___|(_) __ _| |__ | |_ 
|  \| |/ _ \ \/ / __| \___ \| |/ _ | '_ \| __|
| |\  |  __/>  <| |_   ___) | | (_| | | | | |_ 
|_| \_|\___/_/\_\\__|  |____/|_|\__, |_| |_|\__|
                                |___/           
    """
).rstrip()


BRAND_LINE = (
    "Next Sight cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com"
)


PROJECT_TAGLINE = (
    "Forensic AI image detection via multi-mechanism analysis (metadata, frequency, texture, quality, compression, pixel stats)."
)


def about_text() -> str:
    """Return a concise About message suitable for CLI display."""
    return dedent(
        f"""
        {BRAND_LINE}

        AIDetection analyzes images offline and produces a clear verdict with rich diagnostics:
        - Metadata & camera forensics (EXIF, GPS, ICC, MakerNote)
        - Frequency-domain analysis (DCT/FFT, spectral residuals)
        - Texture patterns (LBP, GLCM, Gabor)
        - Quality metrics (sharpness, noise, lens distortion, banding)
        - Compression & pixel statistics (quantization, double-compression, bit-planes)
        """
    ).strip()


def quick_usage_block() -> str:
    return dedent(
        """
        Quick start:
          aidetect analyze --input /path/to/image.jpg
          aidetect analyze --input /path/to/folder --format json --recursive --workers 4

        More help:
          aidetect --help
          aidetect analyze --help
        """
    ).rstrip()

