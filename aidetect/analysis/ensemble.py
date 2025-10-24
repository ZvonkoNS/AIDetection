from __future__ import annotations

from typing import Dict, List

from ..core.config import AppConfig
from ..core.types import MechanismResult, MechanismType


def compute_ensemble_score(
    results: List[MechanismResult], config: AppConfig
) -> float:
    """
    Computes a weighted average of probabilities from multiple mechanism results.
    """
    weights = config.ensemble_weights
    total_weight = 0.0
    weighted_sum = 0.0

    results_map = {r.mechanism: r for r in results}

    for mechanism, weight in weights.items():
        if weight > 0 and mechanism in results_map:
            total_weight += weight
            weighted_sum += weight * results_map[mechanism].probability_ai

    if total_weight == 0:
        return 0.5

    return weighted_sum / total_weight
