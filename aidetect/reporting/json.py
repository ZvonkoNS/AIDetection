import dataclasses
import enum
import json
from collections import Counter
from typing import Any, Dict

from ..core.types import AnalysisResult, MechanismType


# Mechanism descriptions for JSON output
MECHANISM_INFO = {
    MechanismType.METADATA: {
        "name": "Camera & EXIF Metadata Analysis",
        "category": "forensic",
        "description": "Examines EXIF tags, camera models, and software metadata to detect AI generation signatures"
    },
    MechanismType.FREQUENCY: {
        "name": "Frequency Domain Analysis",
        "category": "forensic",
        "description": "Analyzes DCT coefficient patterns typical of AI-generated vs. real camera images"
    },
    MechanismType.ARTIFACTS: {
        "name": "Compression Artifact Analysis",
        "category": "forensic",
        "description": "Detects anomalous JPEG compression artifacts characteristic of AI generation"
    },
    MechanismType.TEXTURE_LBP: {
        "name": "Texture Pattern Analysis",
        "category": "forensic",
        "description": "Uses Local Binary Patterns to identify unnatural texture distributions"
    },
    MechanismType.QUALITY_METRICS: {
        "name": "Image Quality Metrics",
        "category": "forensic",
        "description": "Analyzes sharpness, noise, and quality characteristics that differ between AI and real images"
    },
    MechanismType.CLASSIFIER_XCEPTION: {
        "name": "Xception Deep Learning Classifier",
        "category": "ml_classification",
        "description": "CNN-based deep learning model trained to distinguish AI from real images"
    },
    MechanismType.CLASSIFIER_VIT: {
        "name": "Vision Transformer Classifier",
        "category": "ml_classification",
        "description": "Transformer-based model trained to detect AI-generated image patterns"
    },
    MechanismType.COMPRESSION: {
        "name": "JPEG Compression Analysis",
        "category": "forensic",
        "description": "Analyzes JPEG structure, quantization tables, and compression artifacts"
    },
    MechanismType.PIXEL_STATISTICS: {
        "name": "Pixel Statistics Analysis",
        "category": "forensic",
        "description": "Analyzes bit planes, histograms, channel correlations, and gradient patterns"
    },
}


class CustomJsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle dataclasses and enums.
    """

    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


def _build_structured_output(result: AnalysisResult) -> Dict[str, Any]:
    """
    Build a structured, human-friendly JSON output with explanations.
    """
    # Categorize mechanisms
    forensic_results = []
    ml_results = []
    
    for mech in result.mechanisms:
        mech_info = MECHANISM_INFO.get(mech.mechanism, {})
        mech_data = {
            "mechanism": mech.mechanism.value,
            "name": mech_info.get("name", mech.mechanism.value),
            "description": mech_info.get("description", ""),
            "ai_probability": round(mech.probability_ai, 4),
            "ai_probability_percent": f"{mech.probability_ai * 100:.1f}%",
            "notes": mech.notes,
            "warnings": mech.warnings if mech.warnings else None,
        }
        
        category = mech_info.get("category", "unknown")
        if category == "forensic":
            forensic_results.append(mech_data)
        elif category == "ml_classification":
            ml_results.append(mech_data)
    
    # Build structured output
    output = {
        "report_metadata": {
            "schema_version": result.schema_version,
            "report_type": "AI Image Detection Analysis",
            "analysis_framework": "Multi-mechanism ensemble detection",
            "provider": "NEXT SIGHT cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com"
        },
        "file_info": {
            "filename": result.file.path,
            "sha256": result.file.sha256,
            "size_bytes": result.file.size_bytes,
            "mime_type": result.file.mime_type,
        },
        "verdict": {
            "classification": result.verdict.value,
            "confidence_score": round(result.confidence, 4),
            "confidence_percent": f"{result.confidence * 100:.1f}%",
            "confidence_level": result.confidence_level,
            "explanation": _get_confidence_explanation(result.confidence_level),
        },
        "analysis_summary": {
            "methodology": "This analysis uses multiple independent detection mechanisms combining forensic analysis and machine learning classification. Each mechanism provides an AI likelihood score which are combined using weighted ensemble scoring.",
            "key_findings": result.detection_rationale,
            "processing_time_ms": result.processing_ms,
        },
        "detailed_results": {
            "forensic_analysis": forensic_results,
            "ml_classification": ml_results,
        },
        "run_diagnostics": _build_run_diagnostics(result),
        "interpretation_guide": {
            "ai_probability_meaning": "Each mechanism outputs a probability (0.0-1.0) indicating likelihood of AI generation. Values >0.7 are strong AI indicators, <0.3 are strong human indicators.",
            "confidence_levels": {
                "HIGH": "Multiple mechanisms strongly agree on the verdict",
                "MEDIUM": "Several indicators support the verdict with some disagreement",
                "LOW": "Results are borderline or mechanisms show significant disagreement"
            },
            "verdict_types": {
                "AI-GENERATED": "The image shows strong indicators of being AI-generated",
                "LIKELY HUMAN": "The image shows characteristics consistent with real camera capture"
            }
        }
    }
    
    return output


def _get_confidence_explanation(confidence_level: str) -> str:
    """Get explanation for confidence level."""
    explanations = {
        "HIGH": "Multiple detection mechanisms strongly agree on this verdict",
        "MEDIUM": "Several indicators support this verdict with some disagreement",
        "LOW": "Results are borderline or mechanisms show significant disagreement"
    }
    return explanations.get(confidence_level, "Unknown confidence level")


def _build_run_diagnostics(result: AnalysisResult) -> Dict[str, Any]:
    mechanisms_debug = result.debug.get("mechanisms", [])
    status_counts = Counter()
    total_elapsed = 0

    for entry in mechanisms_debug:
        status = entry.get("status", "unknown")
        status_counts[status] += 1
        try:
            total_elapsed += int(entry.get("elapsed_ms", 0))
        except (TypeError, ValueError):
            continue

    return {
        "total_mechanisms": len(mechanisms_debug),
        "status_breakdown": dict(status_counts),
        "total_mechanism_time_ms": total_elapsed,
        "mechanisms": mechanisms_debug,
    }


def result_to_json_str(result: AnalysisResult, indent: int | None = 2) -> str:
    """
    Serializes an AnalysisResult to a structured, user-friendly JSON string.
    
    The output includes:
    - Report metadata and framework information
    - File information with integrity hashes
    - Verdict with confidence scoring and explanation
    - Analysis summary with methodology
    - Detailed results categorized by mechanism type
    - Interpretation guide for understanding results
    """
    structured_output = _build_structured_output(result)
    return json.dumps(structured_output, indent=indent, ensure_ascii=False)
