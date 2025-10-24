from __future__ import annotations

from typing import List

from ..core.config import AppConfig
from ..core.types import AnalysisResult, MechanismResult, Verdict
from .evidence import build_detection_rationale, count_agreeing_mechanisms, has_positive_ai_evidence


def calibrate_and_decide(
    file_info, mechanism_results: List[MechanismResult], ensemble_prob_ai: float, config: AppConfig
) -> AnalysisResult:
    """
    Calibrates the ensemble probability to a final confidence score and makes
    a final verdict based on threshold and conservative mode settings.
    
    In conservative mode:
    - Threshold raised to 0.65 (from 0.5)
    - Requires at least one positive AI evidence signal
    - Requires 2+ mechanisms to agree (>0.6)
    """
    # Determine threshold based on mode
    if config.conservative_mode:
        threshold = 0.65
        # Conservative mode: require positive AI evidence AND consensus
        has_evidence = has_positive_ai_evidence(mechanism_results)
        agreeing_count = count_agreeing_mechanisms(mechanism_results, threshold=0.6)
        
        # Conservative mode logic:
        # 1. If strong evidence (3+ mechanisms >60%) AND ensemble >0.6, flag as AI
        # 2. Otherwise, require higher threshold (0.65)
        if has_evidence and agreeing_count >= 3 and ensemble_prob_ai >= 0.60:
            # Strong consensus with evidence - flag as AI even if below 0.65 threshold
            verdict = Verdict.AI_GENERATED
            confidence = 0.5 + 0.5 * (ensemble_prob_ai - 0.5) / 0.5
            if ensemble_prob_ai >= 0.75:
                confidence_level = "HIGH"
            else:
                confidence_level = "MEDIUM"
        elif not has_evidence or agreeing_count < 2:
            # No evidence or weak consensus - default to HUMAN
            verdict = Verdict.LIKELY_HUMAN
            confidence = 0.5 + 0.5 * (threshold - ensemble_prob_ai) / threshold
            confidence_level = "MEDIUM"
        elif ensemble_prob_ai >= threshold:
            # Above conservative threshold
            verdict = Verdict.AI_GENERATED
            confidence = 0.5 + 0.5 * (ensemble_prob_ai - threshold) / (1.0 - threshold)
            if ensemble_prob_ai >= 0.8 and agreeing_count >= 3:
                confidence_level = "HIGH"
            else:
                confidence_level = "MEDIUM"
        else:
            # Below threshold
            verdict = Verdict.LIKELY_HUMAN
            confidence = 0.5 + 0.5 * (threshold - ensemble_prob_ai) / threshold
            confidence_level = "MEDIUM"
    else:
        # Normal mode
        threshold = config.ai_verdict_threshold
        
        if ensemble_prob_ai >= threshold:
            verdict = Verdict.AI_GENERATED
            confidence = 0.5 + 0.5 * (ensemble_prob_ai - threshold) / (1.0 - threshold)
            # Determine confidence level
            agreeing_count = count_agreeing_mechanisms(mechanism_results, threshold=0.7)
            if ensemble_prob_ai >= 0.75 and agreeing_count >= 3:
                confidence_level = "HIGH"
            elif ensemble_prob_ai >= 0.6:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
        else:
            verdict = Verdict.LIKELY_HUMAN
            confidence = 0.5 + 0.5 * (threshold - ensemble_prob_ai) / threshold
            # Confidence in HUMAN verdict
            if ensemble_prob_ai <= 0.3:
                confidence_level = "HIGH"
            elif ensemble_prob_ai <= 0.45:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"

    # Build detection rationale
    is_ai_verdict = (verdict == Verdict.AI_GENERATED)
    rationale = build_detection_rationale(mechanism_results, is_ai_verdict)

    return AnalysisResult(
        file=file_info,
        verdict=verdict,
        confidence=confidence,
        mechanisms=mechanism_results,
        confidence_level=confidence_level,
        detection_rationale=rationale,
    )
