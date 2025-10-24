from __future__ import annotations

from typing import Callable, Dict

from ..core.config import AppConfig
from .base import Classifier
from .vit import ViTClassifier
from .xception import XceptionClassifier

_CLASSIFIER_FACTORIES: Dict[str, Callable[[AppConfig], Classifier]] = {
    "xception": lambda cfg: XceptionClassifier(model_path=cfg.model_path_xception),
    "vit": lambda cfg: ViTClassifier(model_path=cfg.model_path_vit),
}

_CLASSIFIER_CACHE: Dict[str, Classifier] = {}


def get_classifier(name: str, config: AppConfig) -> Classifier:
    """
    Return a cached classifier instance keyed by the classifier name and the
    configured model path so that overrides reload cleanly.
    """
    key = name.lower()
    if key not in _CLASSIFIER_FACTORIES:
        raise KeyError(f"Unknown classifier: {name}")

    model_path = getattr(config, f"model_path_{key}", None)
    cache_key = f"{key}:{model_path or 'default'}"

    if cache_key not in _CLASSIFIER_CACHE:
        factory = _CLASSIFIER_FACTORIES[key]
        _CLASSIFIER_CACHE[cache_key] = factory(config)

    return _CLASSIFIER_CACHE[cache_key]
