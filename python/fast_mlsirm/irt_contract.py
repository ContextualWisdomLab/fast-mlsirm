"""Explicit response-shape checks for cross-component IRT inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from .config import MAX_POLYTOMOUS_CATEGORIES

IRTItemType = Literal["dichotomous", "polytomous"]
MIN_IRT_ITEMS = 2


def validate_irt_response_matrix(
    responses: Iterable[Iterable[float]] | np.ndarray,
    item_type: IRTItemType,
    *,
    n_categories: int | None = None,
) -> np.ndarray:
    """Validate and return a persons-by-items matrix for an IRT experiment.

    This is the integration boundary for response data produced by another
    component, such as an LLM judge. It deliberately requires at least two
    item columns. Low-level one-item numerical primitives remain available for
    diagnostics and unit tests, but a one-item result is not an IRT experiment
    contract.

    NaN denotes a missing response. Dichotomous observations are 0/1;
    polytomous observations are integer category indices in
    0..n_categories-1.
    """
    if item_type not in {"dichotomous", "polytomous"}:
        raise ValueError("item_type must be 'dichotomous' or 'polytomous'")
    if item_type == "dichotomous" and n_categories is not None:
        raise ValueError("n_categories is only valid for polytomous responses")
    if item_type == "polytomous" and (
        not isinstance(n_categories, int)
        or isinstance(n_categories, bool)
        or not 2 <= n_categories <= MAX_POLYTOMOUS_CATEGORIES
    ):
        raise ValueError(
            "polytomous responses require n_categories in "
            f"2..{MAX_POLYTOMOUS_CATEGORIES}"
        )

    try:
        matrix = np.asarray(responses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("responses must be numeric") from exc
    if matrix.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items matrix")
    n_persons, n_items = matrix.shape
    if n_persons < 1:
        raise ValueError("responses must contain at least one person")
    if n_items < MIN_IRT_ITEMS:
        raise ValueError(
            "IRT responses must contain at least two item columns; "
            "a scalar or one-item result is not an IRT experiment"
        )

    missing = np.isnan(matrix)
    if np.any(~missing & ~np.isfinite(matrix)):
        raise ValueError("observed responses must be finite or NaN")
    observed = matrix[~missing]
    if observed.size and np.any(observed != np.floor(observed)):
        raise ValueError("observed responses must be integer category values")
    if item_type == "dichotomous":
        if observed.size and np.any((observed < 0) | (observed > 1)):
            raise ValueError("dichotomous responses must be 0, 1, or NaN")
    elif observed.size and np.any((observed < 0) | (observed >= int(n_categories))):
        raise ValueError(
            f"polytomous responses must be in 0..{int(n_categories) - 1} or NaN"
        )
    return matrix


__all__ = ["MIN_IRT_ITEMS", "IRTItemType", "validate_irt_response_matrix"]
