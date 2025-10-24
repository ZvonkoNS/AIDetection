import os

from ..core.types import AnalysisResult, MechanismType


# Human-readable descriptions of each analysis mechanism
MECHANISM_DESCRIPTIONS = {
    MechanismType.METADATA: "Camera & EXIF metadata examination",
    MechanismType.FREQUENCY: "Frequency domain analysis (DCT patterns)",
    MechanismType.ARTIFACTS: "Compression artifact analysis",
    MechanismType.TEXTURE_LBP: "Texture pattern analysis (Local Binary Patterns)",
    MechanismType.QUALITY_METRICS: "Image quality metrics (sharpness, noise)",
    MechanismType.CLASSIFIER_XCEPTION: "Deep learning classifier (Xception CNN)",
    MechanismType.CLASSIFIER_VIT: "Deep learning classifier (Vision Transformer)",
}


def _get_confidence_explanation(confidence: float, confidence_level: str, verdict: str) -> str:
    """Generate explanation for the confidence score."""
    if confidence_level == "HIGH":
        return f"High confidence - Multiple detection mechanisms strongly agree"
    elif confidence_level == "MEDIUM":
        return f"Moderate confidence - Several indicators support this verdict"
    else:
        return f"Low confidence - Results are borderline or mechanisms disagree"


def _format_mechanism_section(mechanisms) -> list:
    """Format the mechanism results into categorized sections."""
    lines = []
    
    # Separate mechanisms by category
    forensic = []
    ml_classifiers = []
    
    for mech in mechanisms:
        if mech.mechanism in [MechanismType.CLASSIFIER_XCEPTION, MechanismType.CLASSIFIER_VIT]:
            ml_classifiers.append(mech)
        else:
            forensic.append(mech)
    
    # Forensic Analysis section
    if forensic:
        lines.append("\n+-- FORENSIC ANALYSIS RESULTS")
        for mech in forensic:
            desc = MECHANISM_DESCRIPTIONS.get(mech.mechanism, mech.mechanism.value)
            prob_percent = mech.probability_ai * 100
            bar = _create_visual_bar(mech.probability_ai)
            lines.append(f"|  {desc}")
            lines.append(f"|    {bar} {prob_percent:.1f}% AI likelihood")
            if mech.notes:
                # Show first note only for brevity
                note = mech.notes.split(';')[0].strip()
                lines.append(f"|    > {note}")
        lines.append("+--")
    
    # ML Classification section
    if ml_classifiers:
        lines.append("\n+-- MACHINE LEARNING CLASSIFICATION")
        for mech in ml_classifiers:
            desc = MECHANISM_DESCRIPTIONS.get(mech.mechanism, mech.mechanism.value)
            prob_percent = mech.probability_ai * 100
            bar = _create_visual_bar(mech.probability_ai)
            lines.append(f"|  {desc}")
            lines.append(f"|    {bar} {prob_percent:.1f}% AI likelihood")
        lines.append("+--")
    
    return lines


def _create_visual_bar(probability: float, width: int = 20) -> str:
    """Create a visual bar chart for probability."""
    filled = int(probability * width)
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}]"


def _format_mechanism_statuses(result: AnalysisResult) -> list:
    entries = [
        entry
        for entry in result.debug.get("mechanisms", [])
        if isinstance(entry, dict)
    ]
    if not entries:
        return []

    lines = []
    lines.append("\n" + "-" * 70)
    lines.append("MECHANISM EXECUTION STATUS")
    lines.append("-" * 70)
    for entry in entries:
        mechanism = entry.get("mechanism", "UNKNOWN")
        status = entry.get("status", "unknown")
        elapsed = entry.get("elapsed_ms")
        missing = entry.get("missing_dependencies") or []
        error = entry.get("error")
        parts = [status]
        if elapsed not in (None, ""):
            parts.append(f"{elapsed} ms")
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if error:
            parts.append(f"error: {str(error)[:60]}")
        lines.append(f"  {mechanism}: {', '.join(parts)}")
    return lines


def format_text_report(result: AnalysisResult) -> str:
    """
    Formats an analysis result into a comprehensive, user-friendly report.
    
    The report is structured into clear sections:
    - Header with verdict and confidence
    - Methodology explanation
    - Detailed mechanism results (categorized)
    - Key findings and rationale
    - Technical details
    """
    lines = []
    
    # === HEADER ===
    lines.append("=" * 70)
    lines.append(f"  AI IMAGE DETECTION REPORT")
    lines.append(f"  File: {os.path.basename(result.file.path)}")
    lines.append("=" * 70)
    
    # === VERDICT SECTION ===
    verdict_symbol = "[AI]" if result.verdict.value == "AI-GENERATED" else "[REAL]"
    lines.append(f"\n{verdict_symbol} VERDICT: {result.verdict.value}")
    lines.append(f"   Confidence: {result.confidence:.1%} ({result.confidence_level})")
    lines.append(f"   {_get_confidence_explanation(result.confidence, result.confidence_level, result.verdict.value)}")
    
    # === ANALYSIS METHODOLOGY ===
    lines.append("\n" + "-" * 70)
    lines.append("ANALYSIS METHODOLOGY")
    lines.append("-" * 70)
    lines.append("This image was analyzed using multiple independent detection mechanisms:")
    lines.append("  * Forensic Analysis: Examines metadata, frequency patterns, compression,")
    lines.append("                       texture, and quality metrics")
    lines.append("  * ML Classification: Deep learning models trained on AI/real images")
    lines.append("\nEach mechanism provides an AI likelihood score (0-100%), which are")
    lines.append("combined using weighted ensemble scoring to produce the final verdict.")
    
    # === KEY FINDINGS ===
    if result.detection_rationale:
        lines.append("\n" + "-" * 70)
        lines.append("KEY FINDINGS")
        lines.append("-" * 70)
        for i, reason in enumerate(result.detection_rationale[:3], 1):
            lines.append(f"  {i}. {reason}")
    
    # === DETAILED RESULTS ===
    lines.append("\n" + "-" * 70)
    lines.append("DETAILED ANALYSIS RESULTS")
    lines.append("-" * 70)
    lines.extend(_format_mechanism_section(result.mechanisms))
    lines.extend(_format_mechanism_statuses(result))
    
    # === TECHNICAL DETAILS ===
    lines.append("\n" + "-" * 70)
    lines.append("TECHNICAL DETAILS")
    lines.append("-" * 70)
    lines.append(f"  Schema Version: {result.schema_version}")
    lines.append(f"  File SHA-256:   {result.file.sha256[:32]}...")
    lines.append(f"  File Size:      {result.file.size_bytes:,} bytes")
    if result.file.mime_type:
        lines.append(f"  MIME Type:      {result.file.mime_type}")
    lines.append(f"  Processing Time: {result.processing_ms} ms")
    
    lines.append("\n" + "=" * 70)
    lines.append("NEXT SIGHT cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com")
    lines.append("=" * 70 + "\n")
    
    return "\n".join(lines)
