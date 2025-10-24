from __future__ import annotations

import logging
import time
from collections import Counter
from typing import List

from ..analysis import calibrate_and_decide, compute_ensemble_score
from ..analysis.registry import (
    execute_mechanism,
    get_mechanism_registry,
    prepare_context,
)
from ..core.config import AppConfig
from ..core.types import AnalysisResult

logger = logging.getLogger(__name__)


def run_single_analysis(file_path: str, config: AppConfig) -> AnalysisResult:
    """
    Orchestrates the full analysis pipeline for a single image file.
    """
    logger.info(f"Starting analysis for: {file_path}")
    start_time = time.monotonic()

    context = prepare_context(file_path, config)
    file_info = context.file_info
    logger.debug("Loaded image %s (%d bytes)", file_info.sha256, file_info.size_bytes)

    mechanism_results = []
    mechanism_debug: List[dict] = []

    for definition in get_mechanism_registry():
        execution = execute_mechanism(definition, context)
        mechanism_results.append(execution.result)
        mechanism_debug.append(
            {
                "mechanism": definition.mechanism.value,
                "name": definition.name,
                "status": execution.status,
                "elapsed_ms": execution.elapsed_ms,
                "missing_dependencies": execution.missing_dependencies,
                "error": execution.error,
            }
        )

    logger.debug("Completed %d analysis mechanisms.", len(mechanism_results))

    ensemble_prob_ai = compute_ensemble_score(mechanism_results, config)
    logger.debug(f"Computed ensemble AI probability: {ensemble_prob_ai:.4f}")

    final_result = calibrate_and_decide(
        file_info, mechanism_results, ensemble_prob_ai, config
    )

    end_time = time.monotonic()
    final_result.processing_ms = int((end_time - start_time) * 1000)
    final_result.debug.setdefault("mechanisms", mechanism_debug)
    status_counts = Counter(entry.get("status", "unknown") for entry in mechanism_debug)
    final_result.debug["mechanism_summary"] = {
        "status_breakdown": dict(status_counts),
        "total_mechanisms": len(mechanism_debug),
    }
    logger.info(
        f"Finished analysis for {file_path} in {final_result.processing_ms} ms. "
        f"Verdict: {final_result.verdict.value}"
    )

    return final_result
