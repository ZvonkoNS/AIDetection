from __future__ import annotations

import abc
from typing import Any

try:
    import numpy as np
except Exception:
    np = None

from ..core.types import MechanismResult


class Classifier(abc.ABC):
    """Abstract base class for image classifiers."""

    @abc.abstractmethod
    def preprocess(self, img_array: "np.ndarray") -> Any:
        """Preprocess a NumPy array for the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, preprocessed_img: Any) -> MechanismResult:
        """Run inference and return a MechanismResult."""
        raise NotImplementedError

    def run(self, img_array: "np.ndarray") -> MechanismResult:
        """Convenience method to run the full pipeline."""
        preprocessed = self.preprocess(img_array)
        return self.predict(preprocessed)
