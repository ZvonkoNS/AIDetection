import csv
import os
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from ..core.types import AnalysisResult, MechanismType


def write_summary_csv(results: List[AnalysisResult], output_path: os.PathLike | str) -> None:
    """
    Writes a comprehensive CSV summary report for a list of analysis results.
    
    The CSV includes:
    - File identification and verdict
    - Confidence scoring with levels
    - Individual mechanism scores (categorized)
    - Key findings
    - Processing metrics
    """
    if not results:
        return

    # Build comprehensive header
    header = [
        "filename",
        "verdict",
        "confidence_score",
        "confidence_percent",
        "confidence_level",
        "top_finding",
        "mechanisms_ran",
        "mechanisms_skipped",
        "mechanisms_failed",
        # Forensic mechanisms
        "metadata_score",
        "metadata_status",
        "metadata_ms",
        "frequency_score",
        "frequency_status",
        "frequency_ms",
        "texture_score",
        "texture_status",
        "texture_ms",
        "quality_score",
        "quality_status",
        "quality_ms",
        "artifacts_score",
        "artifacts_status",
        "artifacts_ms",
        # ML classifiers
        "xception_score",
        "xception_status",
        "xception_ms",
        "vit_score",
        "vit_status",
        "vit_ms",
        # Metadata
        "file_size_bytes",
        "processing_ms",
        "sha256_hash"
    ]

    rows = []
    for result in results:
        # Get mechanism scores by type
        mech_scores = {m.mechanism: m.probability_ai for m in result.mechanisms}
        mechanism_debug = {
            entry.get("mechanism"): entry
            for entry in result.debug.get("mechanisms", [])
            if isinstance(entry, dict) and entry.get("mechanism")
        }
        status_counts = Counter(
            entry.get("status", "unknown") for entry in mechanism_debug.values()
        )

        def mechanism_runtime(mechanism: MechanismType) -> Tuple[str, str]:
            entry = mechanism_debug.get(mechanism.value)
            if not entry:
                return "", ""
            status = entry.get("status", "")
            elapsed = entry.get("elapsed_ms", "")
            return str(status or ""), str(elapsed or "")
        
        # Get top finding
        top_finding = result.detection_rationale[0] if result.detection_rationale else ""
        
        row = [
            Path(result.file.path).name,
            result.verdict.value,
            f"{result.confidence:.4f}",
            f"{result.confidence * 100:.1f}%",
            result.confidence_level,
            top_finding,
            status_counts.get("ran", 0),
            status_counts.get("skipped", 0),
            status_counts.get("error", 0),
            # Forensic mechanisms
            f"{mech_scores.get(MechanismType.METADATA, 0):.4f}",
            *mechanism_runtime(MechanismType.METADATA),
            f"{mech_scores.get(MechanismType.FREQUENCY, 0):.4f}",
            *mechanism_runtime(MechanismType.FREQUENCY),
            f"{mech_scores.get(MechanismType.TEXTURE_LBP, 0):.4f}",
            *mechanism_runtime(MechanismType.TEXTURE_LBP),
            f"{mech_scores.get(MechanismType.QUALITY_METRICS, 0):.4f}",
            *mechanism_runtime(MechanismType.QUALITY_METRICS),
            f"{mech_scores.get(MechanismType.ARTIFACTS, 0):.4f}",
            *mechanism_runtime(MechanismType.ARTIFACTS),
            # ML classifiers
            f"{mech_scores.get(MechanismType.CLASSIFIER_XCEPTION, 0):.4f}",
            *mechanism_runtime(MechanismType.CLASSIFIER_XCEPTION),
            f"{mech_scores.get(MechanismType.CLASSIFIER_VIT, 0):.4f}",
            *mechanism_runtime(MechanismType.CLASSIFIER_VIT),
            # Metadata
            result.file.size_bytes,
            result.processing_ms or 0,
            result.file.sha256[:16] + "...",  # Truncate for CSV readability
        ]
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header comment explaining the CSV
        writer.writerow(["# AI Image Detection Tool - Batch Analysis Summary"])
        writer.writerow(["# NEXT SIGHT cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com"])
        writer.writerow(["# Confidence Score: 0.0-1.0 range, higher = more confident"])
        writer.writerow(["# Mechanism Scores: AI probability (0.0-1.0), >0.7=strong AI, <0.3=strong human"])
        writer.writerow([])  # Empty row for separation
        writer.writerow(header)
        writer.writerows(rows)
