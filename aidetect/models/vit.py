from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

from ..core.types import MechanismResult, MechanismType
from .base import Classifier


class ViTClassifier(Classifier):
    """
    Vision Transformer classifier with optional ONNX runtime support. Falls back
    to a deterministic heuristic when a model is not configured.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self._session: Any = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

    def _ensure_session(self) -> Optional["ort.InferenceSession"]:
        if self.model_path and ort is not None and self._session is None:
            self._session = ort.InferenceSession(self.model_path)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
        return self._session

    def _resize_image(self, img_array: "np.ndarray", size: int = 224) -> "np.ndarray":
        if Image is not None:
            pil_img = Image.fromarray(img_array.astype("uint8"))
            resized = pil_img.resize((size, size))
            return np.asarray(resized, dtype=np.float32)
        return np.resize(img_array, (size, size, img_array.shape[2]))

    def preprocess(self, img_array: "np.ndarray") -> Any:
        if np is None:
            raise RuntimeError("NumPy is not installed.")
        session = self._ensure_session()
        if session:
            resized = self._resize_image(img_array, size=224)
            normalized = resized.astype(np.float32) / 255.0
            tensor = np.transpose(normalized, (2, 0, 1))
            return np.expand_dims(tensor, axis=0)
        return img_array

    def predict(self, preprocessed_img: Any) -> MechanismResult:
        session = self._ensure_session()
        if session and np is not None and self._input_name and self._output_name:
            outputs = session.run(
                [self._output_name],
                {self._input_name: preprocessed_img},
            )
            score = float(np.clip(outputs[0].squeeze(), 0.0, 1.0))
            note = "ViT ONNX model loaded."
        else:
            if np is None:
                raise RuntimeError("NumPy is not installed.")
            variance = float(np.var(preprocessed_img) / (255.0 ** 2))
            score = float(np.clip(variance, 0.0, 1.0))
            note = "ViTClassifier fallback heuristic (no ONNX model configured)."

        return MechanismResult(
            mechanism=MechanismType.CLASSIFIER_VIT,
            probability_ai=score,
            notes=note,
        )
