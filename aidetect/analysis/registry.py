from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Tuple

from ..core.config import AppConfig
from ..core.dependencies import check_dependencies
from ..core.types import MechanismResult, MechanismType
from ..io.image_loader import image_to_numpy, load_image_and_fileinfo
from ..models.registry import get_classifier
from .artifacts import analyze_artifacts
from .compression import analyze_compression
from .frequency import analyze_frequency
from .metadata import analyze_metadata
from .pixel_stats import analyze_pixel_statistics
from .quality import analyze_quality_metrics
from .texture import analyze_texture_lbp

logger = logging.getLogger(__name__)

MechanismRunner = Callable[["AnalysisContext"], MechanismResult]


@dataclass
class AnalysisContext:
    """
    Shared context for mechanism execution to prevent duplicated work such as
    repeatedly converting images to numpy arrays.
    """

    config: AppConfig
    image_path: str
    image: "Image.Image"
    file_info: "FileInfo"
    _image_array: Optional["np.ndarray"] = field(default=None, init=False, repr=False)

    def get_image_array(self) -> "np.ndarray":
        """
        Lazily convert the PIL image into a NumPy array, caching the result.
        """
        if self._image_array is None:
            self._image_array = image_to_numpy(self.image)
        return self._image_array


@dataclass(frozen=True)
class MechanismDefinition:
    mechanism: MechanismType
    name: str
    category: str
    runner: MechanismRunner
    required_modules: Tuple[str, ...] = ()


@dataclass
class MechanismExecution:
    result: MechanismResult
    status: str
    elapsed_ms: int
    missing_dependencies: List[str]
    error: Optional[str] = None


def _run_metadata(ctx: AnalysisContext) -> MechanismResult:
    return analyze_metadata(ctx.image, file_path=ctx.image_path)


def _run_frequency(ctx: AnalysisContext) -> MechanismResult:
    return analyze_frequency(ctx.image, file_path=ctx.image_path)


def _run_artifacts(ctx: AnalysisContext) -> MechanismResult:
    return analyze_artifacts(ctx.image)


def _run_compression(ctx: AnalysisContext) -> MechanismResult:
    return analyze_compression(ctx.image, file_path=ctx.image_path)


def _run_pixel_stats(ctx: AnalysisContext) -> MechanismResult:
    return analyze_pixel_statistics(ctx.image)


def _run_texture(ctx: AnalysisContext) -> MechanismResult:
    return analyze_texture_lbp(ctx.image, ctx.get_image_array())


def _run_quality(ctx: AnalysisContext) -> MechanismResult:
    return analyze_quality_metrics(ctx.image, ctx.get_image_array())


def _run_xception(ctx: AnalysisContext) -> MechanismResult:
    classifier = get_classifier("xception", ctx.config)
    return classifier.run(ctx.get_image_array())


def _run_vit(ctx: AnalysisContext) -> MechanismResult:
    classifier = get_classifier("vit", ctx.config)
    return classifier.run(ctx.get_image_array())


def _neutral_result(mechanism: MechanismType, reason: str) -> MechanismResult:
    return MechanismResult(
        mechanism=mechanism,
        probability_ai=0.5,
        notes=reason,
        warnings=[reason],
    )


def get_mechanism_registry() -> Tuple[MechanismDefinition, ...]:
    """
    Return the ordered tuple of mechanism definitions used during analysis.
    """
    return (
        MechanismDefinition(
            mechanism=MechanismType.METADATA,
            name="EXIF Metadata",
            category="forensic",
            runner=_run_metadata,
            required_modules=("PIL",),
        ),
        MechanismDefinition(
            mechanism=MechanismType.FREQUENCY,
            name="Frequency Analysis",
            category="forensic",
            runner=_run_frequency,
            required_modules=("PIL", "numpy", "scipy"),
        ),
        MechanismDefinition(
            mechanism=MechanismType.ARTIFACTS,
            name="Artifact Analysis",
            category="forensic",
            runner=_run_artifacts,
            required_modules=("PIL",),
        ),
        MechanismDefinition(
            mechanism=MechanismType.COMPRESSION,
            name="Compression Analysis",
            category="forensic",
            runner=_run_compression,
            required_modules=("PIL", "numpy"),
        ),
        MechanismDefinition(
            mechanism=MechanismType.PIXEL_STATISTICS,
            name="Pixel Statistics Analysis",
            category="forensic",
            runner=_run_pixel_stats,
            required_modules=("PIL", "numpy"),
        ),
        MechanismDefinition(
            mechanism=MechanismType.TEXTURE_LBP,
            name="Texture Analysis (LBP)",
            category="forensic",
            runner=_run_texture,
            required_modules=("PIL", "numpy", "skimage"),
        ),
        MechanismDefinition(
            mechanism=MechanismType.QUALITY_METRICS,
            name="Image Quality Metrics",
            category="forensic",
            runner=_run_quality,
            required_modules=("PIL", "numpy", "scipy"),
        ),
        MechanismDefinition(
            mechanism=MechanismType.CLASSIFIER_XCEPTION,
            name="Xception Classifier",
            category="ml",
            runner=_run_xception,
            required_modules=("numpy",),
        ),
        MechanismDefinition(
            mechanism=MechanismType.CLASSIFIER_VIT,
            name="Vision Transformer Classifier",
            category="ml",
            runner=_run_vit,
            required_modules=("numpy",),
        ),
    )


def execute_mechanism(definition: MechanismDefinition, ctx: AnalysisContext) -> MechanismExecution:
    """
    Execute a single mechanism with dependency checks, timing, and error capture.
    """
    available, missing = check_dependencies(definition.required_modules)

    if not available:
        note = f"{definition.name} skipped - missing dependencies: {', '.join(missing)}"
        return MechanismExecution(
            result=_neutral_result(definition.mechanism, note),
            status="skipped",
            elapsed_ms=0,
            missing_dependencies=list(missing),
        )

    start = time.perf_counter()
    try:
        result = definition.runner(ctx)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return MechanismExecution(
            result=result,
            status="ran",
            elapsed_ms=elapsed_ms,
            missing_dependencies=[],
        )
    except Exception as exc:
        logger.error(
            "Mechanism %s failed: %s", definition.mechanism.value, exc, exc_info=True
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        fallback = _neutral_result(
            definition.mechanism,
            f"{definition.name} failed: {str(exc)[:120]}",
        )
        return MechanismExecution(
            result=fallback,
            status="error",
            elapsed_ms=elapsed_ms,
            missing_dependencies=[],
            error=str(exc),
        )


def prepare_context(file_path: str, config: AppConfig) -> AnalysisContext:
    """
    Load the image and metadata required for downstream mechanisms.
    """
    image, file_info = load_image_and_fileinfo(file_path)
    return AnalysisContext(
        config=config,
        image_path=file_info.path,
        image=image,
        file_info=file_info,
    )
