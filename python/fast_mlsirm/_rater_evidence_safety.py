"""Callback-free admission for remaining Rust-backed rater reliability APIs.

This module changes validation and marshalling only. All agreement/reliability
statistics and result-affecting arithmetic remain in the existing Rust-backed
implementations in :mod:`fast_mlsirm.reliability`.
"""

from __future__ import annotations

import builtins
import sys
from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

from ._icc_control_safety import (
    _choice,
    _integer,
    _reject_masked_array,
    _trusted_sequence_tree,
)


def _real_numeric_source_array(
    value: Any,
    name: str,
    *,
    dimension_error: str,
) -> np.ndarray:
    """Return inert 2-D real-numeric source evidence without narrowing it."""
    value_type = builtins.type(value)
    if value_type is np.ndarray:
        arr = value
    elif value_type is list or value_type is tuple:
        trusted, excess_rank = _trusted_sequence_tree(
            value,
            allow_bool=False,
            max_depth=2,
        )
        if excess_rank:
            raise ValueError(dimension_error)
        if not trusted:
            raise ValueError(f"{name} must be real numeric evidence")
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be real numeric evidence") from exc
    else:
        raise ValueError(f"{name} must be real numeric evidence")

    if np.iscomplexobj(arr) or arr.dtype.kind not in "fiub":
        raise ValueError(f"{name} must be real numeric evidence")
    if arr.dtype.kind == "b":
        raise ValueError(f"{name} must be real numeric evidence")
    if arr.ndim != 2:
        raise ValueError(dimension_error)
    return arr


def install(reliability_module: ModuleType) -> None:
    """Install callback-free adapters for remaining rater reliability APIs."""
    current_kripp: Callable[..., Any] = reliability_module.kripp_alpha
    if getattr(current_kripp, "__fast_mlsirm_rater_evidence_hardened__", False):
        return

    original_kripp: Callable[..., Any] = current_kripp
    original_finn: Callable[..., Any] = reliability_module.finn_coefficient
    original_maxwell: Callable[..., Any] = reliability_module.maxwell_re
    original_robinson: Callable[..., Any] = reliability_module.robinson_a

    @wraps(original_kripp)
    def safe_kripp_alpha(ratings: Any, method: str = "nominal") -> Any:
        """Validate Krippendorff method and ratings before native discovery."""
        method_value = _choice(
            method,
            "method",
            ("nominal", "ordinal", "interval", "ratio"),
        )
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D raters x subjects array",
        )
        return original_kripp(ratings_value, method=method_value)

    @wraps(original_finn)
    def safe_finn_coefficient(
        ratings: Any,
        s_levels: int,
        model: str = "oneway",
    ) -> Any:
        """Validate Finn controls and ratings before native discovery."""
        levels_value = _integer(s_levels, "s_levels")
        if levels_value < 2:
            raise ValueError("s_levels must be at least 2")
        model_value = _choice(model, "model", ("oneway", "twoway"))
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D subjects x raters array",
        )
        return original_finn(
            ratings_value,
            levels_value,
            model=model_value,
        )

    @wraps(original_maxwell)
    def safe_maxwell_re(ratings: Any) -> Any:
        """Validate Maxwell ratings before native discovery."""
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D subjects x 2 array",
        )
        return original_maxwell(ratings_value)

    @wraps(original_robinson)
    def safe_robinson_a(ratings: Any) -> Any:
        """Validate Robinson ratings before native discovery."""
        _reject_masked_array(ratings, "ratings")
        ratings_value = _real_numeric_source_array(
            ratings,
            "ratings",
            dimension_error="ratings must be a 2-D subjects x raters array",
        )
        return original_robinson(ratings_value)

    safe_kripp_alpha.__fast_mlsirm_rater_evidence_hardened__ = True
    reliability_module.kripp_alpha = safe_kripp_alpha
    reliability_module.finn_coefficient = safe_finn_coefficient
    reliability_module.maxwell_re = safe_maxwell_re
    reliability_module.robinson_a = safe_robinson_a

    legacy_name = f"{reliability_module.__package__}._legacy_init"
    legacy_module = sys.modules.get(legacy_name)
    if legacy_module is not None:
        legacy_module.kripp_alpha = safe_kripp_alpha
        legacy_module.finn_coefficient = safe_finn_coefficient
        legacy_module.maxwell_re = safe_maxwell_re
        legacy_module.robinson_a = safe_robinson_a
