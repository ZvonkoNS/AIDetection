from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - allow import without hard dependency at import time
    Image = None  # type: ignore
    ImageOps = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ..core.types import FileInfo


def compute_sha256(file_path: str | Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute SHA256 hash for a file using a buffered read to support large files.
    """
    sha256 = hashlib.sha256()
    path = Path(file_path)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def sniff_mime_type(file_path: str | Path) -> Optional[str]:
    """
    Guess MIME type based on filename. For forensic robustness, primary detection
    is by extension only; deeper content sniffing can be added later if needed.
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type


def load_image_pil(file_path: str | Path, apply_exif_orientation: bool = True) -> "Image.Image":
    """
    Load an image with PIL and optionally apply EXIF orientation normalization.
    Returns a PIL Image in RGB mode for downstream consistency.
    
    Security: Validates file exists and is readable before attempting to load.
    """
    if Image is None:
        raise RuntimeError(
            "Pillow (PIL) is not available. Please install 'Pillow' to load images."
        )

    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"Path is not a valid file: {path}")

    try:
        img = Image.open(path)
        # Some formats are lazy; force load to catch errors early
        img.load()
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to load image: {e}") from e
    
    if apply_exif_orientation and ImageOps is not None:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            # If EXIF parsing fails, continue without transpose
            pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def image_to_numpy(img: "Image.Image") -> "np.ndarray":
    """
    Convert a PIL Image (assumed RGB) to a NumPy array with dtype uint8 and
    shape (H, W, 3). Raises if NumPy is unavailable.
    """
    if np is None:
        raise RuntimeError(
            "NumPy is not available. Please install 'numpy' to convert to arrays."
        )
    return np.asarray(img, dtype=np.uint8)


def load_image_and_fileinfo(file_path: str | Path) -> Tuple["Image.Image", FileInfo]:
    """
    Convenience loader that returns both the normalized PIL Image and FileInfo
    including SHA256, size, and MIME type.
    
    Security: Validates and normalizes path before processing.
    """
    try:
        abs_path = Path(file_path).expanduser().resolve()
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid file path: {e}") from e

    if not abs_path.is_file():
        raise FileNotFoundError(f"File not found: {abs_path}")

    try:
        size_bytes = abs_path.stat().st_size
        sha256 = compute_sha256(abs_path)
        mime_type = sniff_mime_type(abs_path)
        img = load_image_pil(abs_path, apply_exif_orientation=True)
    except Exception as e:
        raise RuntimeError(f"Failed to process file {abs_path}: {e}") from e

    return img, FileInfo(path=str(abs_path), sha256=sha256, size_bytes=size_bytes, mime_type=mime_type)


