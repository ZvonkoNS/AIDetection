from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Iterable, List, Tuple


@lru_cache(maxsize=None)
def _spec_available(module_name: str) -> bool:
    """
    Return True if the given module can be imported.
    """
    return importlib.util.find_spec(module_name) is not None


def check_dependencies(modules: Iterable[str]) -> Tuple[bool, List[str]]:
    """
    Check that all modules in the iterable can be imported.
    Returns (all_available, missing_modules).
    """
    missing = [name for name in modules if not _spec_available(name)]
    return len(missing) == 0, missing
