import logging
import random

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)

# In a real app, you would also need to set seeds for PyTorch, TensorFlow, etc.
# For example:
# try:
#     import torch
# except Exception:
#     torch = None


def set_seed(seed: int) -> None:
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    if np:
        np.random.seed(seed)
    # if torch:
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    logger.info(f"Global random seed set to {seed}")
