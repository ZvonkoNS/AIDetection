from unittest.mock import MagicMock

import pytest

from aidetect.analysis.ensemble import compute_ensemble_score
from aidetect.analysis.registry import MechanismDefinition, execute_mechanism
from aidetect.core.types import MechanismResult, MechanismType


def test_compute_ensemble_score_handles_missing_weights():
    mechanisms = [
        MechanismResult(mechanism=MechanismType.METADATA, probability_ai=0.8),
        MechanismResult(mechanism=MechanismType.FREQUENCY, probability_ai=0.2),
    ]
    config = MagicMock()
    config.ensemble_weights = {
        MechanismType.METADATA: 0.0,
        MechanismType.FREQUENCY: 0.0,
    }

    score = compute_ensemble_score(mechanisms, config)
    assert score == pytest.approx(0.5), "Zero weights should fall back to neutral score"


def test_execute_mechanism_skips_when_dependencies_missing():
    definition = MechanismDefinition(
        mechanism=MechanismType.METADATA,
        name="Metadata",
        category="forensic",
        runner=lambda ctx: pytest.fail("runner should not execute when deps missing"),
        required_modules=("module_that_should_not_exist",),
    )

    execution = execute_mechanism(definition, MagicMock())
    assert execution.status == "skipped"
    assert execution.result.mechanism is MechanismType.METADATA
    assert execution.result.probability_ai == pytest.approx(0.5)
    assert execution.missing_dependencies == ["module_that_should_not_exist"]


def test_execute_mechanism_returns_neutral_on_error():
    def failing_runner(ctx):
        raise RuntimeError("boom")

    definition = MechanismDefinition(
        mechanism=MechanismType.FREQUENCY,
        name="Frequency",
        category="forensic",
        runner=failing_runner,
        required_modules=(),
    )

    execution = execute_mechanism(definition, MagicMock())
    assert execution.status == "error"
    assert execution.result.probability_ai == pytest.approx(0.5)
    assert "failed" in execution.result.notes
