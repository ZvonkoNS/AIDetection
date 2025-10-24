import textwrap

import pytest

from aidetect.core.config import AppConfig, parse_weights_override
from aidetect.core.types import MechanismType


def test_parse_weights_override_parses_values():
    overrides = parse_weights_override("METADATA=0.3,frequency=0.2")
    assert overrides[MechanismType.METADATA] == pytest.approx(0.3)
    assert overrides[MechanismType.FREQUENCY] == pytest.approx(0.2)


def test_parse_weights_override_rejects_invalid():
    with pytest.raises(ValueError):
        parse_weights_override("not-a-valid-entry")


def test_app_config_loads_pyproject(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [tool.aidetect]
            seed = 123
            ai_verdict_threshold = 0.6

            [tool.aidetect.ensemble_weights]
            metadata = 0.3
            """
        ).strip()
    )

    config = AppConfig.load(root=tmp_path)
    assert config.seed == 123
    assert config.ai_verdict_threshold == pytest.approx(0.6)
    assert config.ensemble_weights[MechanismType.METADATA] == pytest.approx(0.3)
