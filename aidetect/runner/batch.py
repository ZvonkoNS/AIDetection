from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional

from ..core.config import AppConfig
from ..core.types import AnalysisResult
from .single import run_single_analysis

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.tiff",
    "*.tif",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
    "*.TIFF",
    "*.TIF",
)


def find_images(directory: str, recursive: bool = False) -> List[str]:
    """
    Find all supported image files in a directory.
    """
    base_path = Path(directory).expanduser().resolve()
    if not base_path.is_dir():
        logger.warning("Path is not a directory: %s", base_path)
        return []

    image_paths: List[str] = []
    for pattern in SUPPORTED_EXTENSIONS:
        try:
            iterator = base_path.rglob(pattern) if recursive else base_path.glob(pattern)
            image_paths.extend(str(path.resolve()) for path in iterator if path.is_file())
        except (OSError, ValueError) as exc:
            logger.warning("Error scanning for %s: %s", pattern, exc)

    deduplicated = sorted(set(image_paths))
    return deduplicated


def run_batch_analysis(
    directory: str,
    config: AppConfig,
    *,
    recursive: bool = False,
    workers: int = 1,
) -> Iterator[AnalysisResult]:
    """
    Finds supported images and runs analysis, yielding results.
    """
    image_paths = find_images(directory, recursive=recursive)
    logger.info(
        "Found %d supported images in '%s'%s.",
        len(image_paths),
        directory,
        " (recursive)" if recursive else "",
    )

    if not image_paths:
        logger.warning("No supported images found in '%s'", directory)
        return iter([])

    if workers <= 1:
        iterator = tqdm(image_paths) if tqdm else image_paths
        for path in iterator:
            try:
                yield run_single_analysis(path, config)
            except Exception:
                logger.error("Failed to process %s", Path(path).name, exc_info=True)
    else:
        total = len(image_paths)
        progress = tqdm(total=total) if tqdm else None

        def _process(path: str) -> Optional[AnalysisResult]:
            try:
                return run_single_analysis(path, config)
            except Exception:
                logger.error("Failed to process %s", Path(path).name, exc_info=True)
                return None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_process, image_paths):
                if progress:
                    progress.update(1)
                if result is not None:
                    yield result

        if progress:
            progress.close()
