"""Stable cut-score normalization for public classification adapters."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Sequence

import numpy as np

_ERROR = "cutscores entries must be finite real scalars"


def _normalize(module: ModuleType, cutscores: Sequence[float]) -> list[float]:
    """Normalize trusted cut scores while containing platform float overflow."""

    try:
        iterator = iter(cutscores)
    except TypeError as error:
        raise ValueError(_ERROR) from error

    normalized: list[float] = []
    for value in iterator:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(_ERROR)
        value_type = type(value)
        if not (
            value_type is int
            or value_type is float
            or module._trusted_numpy_integer(value)
            or module._trusted_numpy_float(value)
        ):
            raise ValueError(_ERROR)
        try:
            parsed = float(value)
        except OverflowError as error:
            raise ValueError(_ERROR) from error
        if not math.isfinite(parsed):
            raise ValueError(_ERROR)
        normalized.append(parsed)
    return normalized


def install(module: ModuleType) -> None:
    """Install overflow-stable cut-score admission on the classification module."""

    module._normalize_cutscores = lambda cutscores: _normalize(module, cutscores)
