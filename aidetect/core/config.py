from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .types import MechanismType

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Default weights for each mechanism. These can be overridden by a config file.
DEFAULT_WEIGHTS: Dict[MechanismType, float] = {
    MechanismType.METADATA: 0.18,
    MechanismType.FREQUENCY: 0.18,
    MechanismType.TEXTURE_LBP: 0.12,
    MechanismType.QUALITY_METRICS: 0.10,
    MechanismType.COMPRESSION: 0.10,
    MechanismType.PIXEL_STATISTICS: 0.08,
    MechanismType.ARTIFACTS: 0.08,
    MechanismType.CLASSIFIER_XCEPTION: 0.10,
    MechanismType.CLASSIFIER_VIT: 0.06,
}

DEFAULT_CONFIG_FILES = ("aidetect.toml", "aidetect.yaml", "aidetect.yml")


def _mechanism_from_key(key: str) -> MechanismType:
    normalized = key.strip().upper().replace("-", "_")
    try:
        return MechanismType[normalized]
    except KeyError:
        for mechanism in MechanismType:
            if mechanism.value.upper() == normalized:
                return mechanism
    raise KeyError(f"Unknown mechanism key: {key}")


def parse_weights_mapping(mapping: Mapping[str, Any]) -> Dict[MechanismType, float]:
    weights: Dict[MechanismType, float] = {}
    for raw_key, raw_value in mapping.items():
        try:
            mech = _mechanism_from_key(str(raw_key))
        except KeyError:
            continue
        try:
            weights[mech] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return weights


def parse_weights_override(weights_str: str) -> Dict[MechanismType, float]:
    """
    Parse a CLI-style weights override string like 'METADATA=0.2,FREQUENCY=0.3'.
    """
    pieces = [segment.strip() for segment in weights_str.split(",") if segment.strip()]
    parsed: Dict[MechanismType, float] = {}
    for piece in pieces:
        if "=" not in piece:
            raise ValueError(f"Invalid weight override: '{piece}' (expected KEY=VALUE)")
        key, value = piece.split("=", 1)
        mech = _mechanism_from_key(key)
        try:
            parsed[mech] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for '{key}': {value}") from exc
    return parsed


def _load_toml(path: Path) -> Mapping[str, Any]:
    if tomllib is None:
        raise RuntimeError("tomllib is required to load TOML configuration files.")
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML configuration files.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _extract_tool_config(data: Mapping[str, Any]) -> Mapping[str, Any]:
    # Look for pyproject [tool.aidetect]
    tool_section = data.get("tool")
    if isinstance(tool_section, Mapping):
        aidetect_section = tool_section.get("aidetect")
        if isinstance(aidetect_section, Mapping):
            return aidetect_section
    return {}


def _load_config_mapping(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".toml":
        mapping = _load_toml(path)
        # pyproject special-case
        if path.name == "pyproject.toml":
            return _extract_tool_config(mapping)
        return mapping
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    raise ValueError(f"Unsupported config format: {path}")


@dataclass
class AppConfig:
    """
    Global application configuration loaded from optional TOML/YAML sources
    plus CLI overrides.
    """

    seed: int = 42
    ensemble_weights: Dict[MechanismType, float] = field(
        default_factory=lambda: DEFAULT_WEIGHTS.copy()
    )
    conservative_mode: bool = False
    ai_verdict_threshold: float = 0.5
    model_path_xception: Optional[str] = None
    model_path_vit: Optional[str] = None
    default_workers: int = 1
    default_recursive: bool = False
    config_source: Optional[str] = None

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        for key, value in overrides.items():
            key_normalized = key.lower()
            if key_normalized == "ensemble_weights" and isinstance(value, Mapping):
                self.ensemble_weights.update(parse_weights_mapping(value))
            elif key_normalized == "weights" and isinstance(value, Mapping):
                self.ensemble_weights.update(parse_weights_mapping(value))
            elif key_normalized == "seed":
                self.seed = int(value)
            elif key_normalized == "conservative_mode":
                self.conservative_mode = bool(value)
            elif key_normalized in {"ai_verdict_threshold", "threshold"}:
                self.ai_verdict_threshold = float(value)
            elif key_normalized == "model_path_xception":
                self.model_path_xception = str(value)
            elif key_normalized == "model_path_vit":
                self.model_path_vit = str(value)
            elif key_normalized == "default_workers":
                self.default_workers = max(1, int(value))
            elif key_normalized == "default_recursive":
                self.default_recursive = bool(value)

    @classmethod
    def load(
        cls,
        *,
        root: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> "AppConfig":
        root_path = Path(root) if root is not None else Path.cwd()
        config = cls()

        candidate_paths = []
        if config_path is not None:
            candidate_paths.append(Path(config_path))
        else:
            pyproject = root_path / "pyproject.toml"
            if pyproject.exists():
                candidate_paths.append(pyproject)
            for name in DEFAULT_CONFIG_FILES:
                path = root_path / name
                if path.exists():
                    candidate_paths.append(path)
                    break

        for path in candidate_paths:
            try:
                mapping = _load_config_mapping(path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load config from {path}: {exc}") from exc
            if mapping:
                config.apply_overrides(mapping)
                config.config_source = str(path)
                break

        return config

    def apply_weight_overrides(self, weights: Mapping[MechanismType, float]) -> None:
        if not weights:
            return
        self.ensemble_weights.update(weights)
