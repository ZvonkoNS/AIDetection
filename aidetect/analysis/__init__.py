"""Core analysis mechanisms for AIDT."""

from .artifacts import analyze_artifacts
from .calibration import calibrate_and_decide
from .compression import analyze_compression
from .ensemble import compute_ensemble_score
from .evidence import (
    EvidenceType,
    build_detection_rationale,
    classify_mechanism_evidence,
    count_agreeing_mechanisms,
    has_positive_ai_evidence,
)
from .frequency import analyze_frequency
from .metadata import analyze_metadata
from .pixel_stats import analyze_pixel_statistics
from .quality import analyze_quality_metrics
from .texture import analyze_texture_lbp

__all__ = [
    "analyze_artifacts",
    "analyze_compression",
    "analyze_pixel_statistics",
    "calibrate_and_decide",
    "compute_ensemble_score",
    "analyze_frequency",
    "analyze_metadata",
    "analyze_quality_metrics",
    "analyze_texture_lbp",
    "EvidenceType",
    "build_detection_rationale",
    "classify_mechanism_evidence",
    "count_agreeing_mechanisms",
    "has_positive_ai_evidence",
]
