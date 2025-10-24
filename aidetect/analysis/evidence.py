from __future__ import annotations

from enum import Enum
from typing import List, Tuple

from ..core.types import MechanismResult, MechanismType


class EvidenceType(Enum):
    """Classification of evidence from analysis mechanisms."""
    
    POSITIVE_AI = "POSITIVE_AI"        # Strong evidence of AI generation
    NEGATIVE_AI = "NEGATIVE_AI"        # Strong evidence of real camera
    NEUTRAL = "NEUTRAL"                # Inconclusive or missing data


def classify_mechanism_evidence(result: MechanismResult) -> EvidenceType:
    """
    Classify a mechanism result as positive AI evidence, negative AI evidence, or neutral.
    
    Positive AI evidence: High probability (>0.7) of AI generation
    Negative AI evidence: Low probability (<0.3) of AI generation  
    Neutral: Moderate probability or missing data
    """
    prob = result.probability_ai
    
    if prob >= 0.7:
        return EvidenceType.POSITIVE_AI
    elif prob <= 0.3:
        return EvidenceType.NEGATIVE_AI
    else:
        return EvidenceType.NEUTRAL


def has_positive_ai_evidence(results: List[MechanismResult]) -> bool:
    """
    Check if there is at least one strong positive AI signal.
    This prevents flagging images as AI just because they lack camera metadata.
    """
    for result in results:
        if classify_mechanism_evidence(result) == EvidenceType.POSITIVE_AI:
            return True
    return False


def count_agreeing_mechanisms(results: List[MechanismResult], threshold: float = 0.6) -> int:
    """
    Count how many mechanisms agree that the image is likely AI-generated.
    Used for conservative mode consensus requirement.
    """
    count = 0
    for result in results:
        if result.probability_ai >= threshold:
            count += 1
    return count


def build_detection_rationale(results: List[MechanismResult], verdict_ai: bool) -> List[str]:
    """
    Build a list of human-readable reasons explaining the verdict.
    Returns top 3 most significant findings.
    """
    rationale = []
    
    # Sort mechanisms by how strongly they support the verdict
    if verdict_ai:
        # Sort by highest AI probability
        sorted_results = sorted(results, key=lambda r: r.probability_ai, reverse=True)
    else:
        # Sort by lowest AI probability (most human-like)
        sorted_results = sorted(results, key=lambda r: r.probability_ai)
    
    # Collect top findings
    for i, result in enumerate(sorted_results[:3]):
        prob = result.probability_ai
        mechanism = result.mechanism.value
        
        if verdict_ai and prob >= 0.6:
            if result.notes:
                # Extract first meaningful note
                note = result.notes.split(';')[0].strip()
                rationale.append(f"{mechanism}: {prob:.1%} AI - {note}")
            else:
                rationale.append(f"{mechanism}: {prob:.1%} AI probability")
        elif not verdict_ai and prob <= 0.4:
            if result.notes:
                note = result.notes.split(';')[0].strip()
                rationale.append(f"{mechanism}: {prob:.1%} AI - {note}")
            else:
                rationale.append(f"{mechanism}: {prob:.1%} AI probability (human-like)")
    
    return rationale if rationale else ["Borderline case - multiple factors balanced"]
