from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(Enum):
    """
    Overall top-line outcome for an analyzed image.
    """

    AI_GENERATED = "AI-GENERATED"
    LIKELY_HUMAN = "LIKELY HUMAN"


class MechanismType(Enum):
    """
    Distinct analysis mechanisms contributing to the ensemble decision.
    """

    METADATA = "METADATA"
    FREQUENCY = "FREQUENCY"
    CLASSIFIER_XCEPTION = "CLASSIFIER_XCEPTION"
    CLASSIFIER_VIT = "CLASSIFIER_VIT"
    ARTIFACTS = "ARTIFACTS"
    TEXTURE_LBP = "TEXTURE_LBP"
    QUALITY_METRICS = "QUALITY_METRICS"
    COMPRESSION = "COMPRESSION"
    PIXEL_STATISTICS = "PIXEL_STATISTICS"


class ReportFormat(Enum):
    """Output report formats supported by the CLI."""

    TEXT = "text"
    JSON = "json"
    PDF = "pdf"


@dataclass
class FileInfo:
    """
    File integrity and identity details used in reports and logs.
    """

    path: str
    sha256: str
    size_bytes: int
    mime_type: Optional[str] = None


@dataclass
class MechanismResult:
    """
    Per-mechanism score and auxiliary diagnostics.

    probability_ai is the mechanism's probability that the image is AI-generated
    in range [0, 1]. probability_human is derived as (1 - probability_ai).
    """

    mechanism: MechanismType
    probability_ai: float
    notes: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp to [0, 1] to keep invariants even if upstream slightly violates bounds
        if self.probability_ai < 0.0:
            self.probability_ai = 0.0
        elif self.probability_ai > 1.0:
            self.probability_ai = 1.0

    @property
    def probability_human(self) -> float:
        return 1.0 - self.probability_ai


@dataclass
class AnalysisResult:
    """
    Top-level result combining all mechanisms into a clear verdict with a
    calibrated confidence score and supporting breakdown.
    """

    file: FileInfo
    verdict: Verdict
    confidence: float  # overall confidence in [0, 1]
    mechanisms: List[MechanismResult]
    schema_version: str = "1.1.0"  # Updated for Phase 1
    overall_reason: Optional[str] = None
    processing_ms: Optional[int] = None
    confidence_level: str = "MEDIUM"  # NEW: HIGH/MEDIUM/LOW
    detection_rationale: List[str] = field(default_factory=list)  # NEW: Why this verdict
    debug: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0



