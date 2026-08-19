"""Callback-free admission for public automated-scoring validation controls.

This module changes only Python semantic-control validation. Validation metrics,
threshold decisions, agreement arithmetic, and pass/fail computation remain
owned by the existing Rust agreement kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_THRESHOLD_NAMES = (
    "qwk_min",
    "pearson_r_min",
    "degradation_max",
    "overall_smd_max",
    "subgroup_smd_max",
)
_TRUSTED_THRESHOLD_TYPES = (int, float, np.float64)


def _has_exact_type(value: object, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value`` has one exact package-trusted scalar type."""
    value_type = type(value)
    return any(value_type is trusted_type for trusted_type in trusted_types)


def _safe_validation_policy_post_init(self: Any) -> None:
    """Normalize policy controls without executing caller-defined callbacks."""
    if type(self.policy_id) is not str or not self.policy_id.strip():
        raise ValueError("policy_id must be a non-empty string")
    if type(self.policy_version) is not str or not self.policy_version.strip():
        raise ValueError("policy_version must be a non-empty string")

    for name in _THRESHOLD_NAMES:
        value = getattr(self, name)
        if not _has_exact_type(value, _TRUSTED_THRESHOLD_TYPES):
            raise ValueError(f"{name} must be a real number in 0..1")
        normalized = float(value)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError(f"{name} must be in 0..1")
        object.__setattr__(self, name, normalized)

    subgroup_n = self.min_subgroup_n
    if type(subgroup_n) is not int or subgroup_n < 2:
        raise ValueError("min_subgroup_n must be an integer >= 2")


def install(validation_module: Any) -> None:
    """Install callback-safe policy validation while preserving class identity."""
    validation_module.ValidationPolicy.__post_init__ = _safe_validation_policy_post_init
