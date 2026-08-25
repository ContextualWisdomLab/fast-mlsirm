"""Rust-backed residual interaction maps for downstream measurement products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core


@dataclass(frozen=True)
class ResidualInteractionMap:
    """Complete-case Gabriel coordinates and auditable cell decomposition."""

    person_indices: np.ndarray
    item_indices: np.ndarray
    person_coordinates: np.ndarray
    item_coordinates: np.ndarray
    singular_values: np.ndarray
    axis_shares: np.ndarray
    reconstruction: np.ndarray
    unexplained: np.ndarray
    cross_share: np.ndarray


def residual_interaction_map(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    axis_count: int,
) -> ResidualInteractionMap:
    """Factor ``observed - expected`` using Gabriel symmetric scaling.

    Missing observed cells are represented by ``NaN`` and excluded through a
    complete-case rectangle; they are never filled with zero. ``axis_count`` is
    required because the consuming measurement contract, not this library,
    determines how many reader-visible axes are retained.

    References:
        Gabriel, K. R. (1971). The biplot graphic display of matrices with
            application to principal component analysis. *Biometrika, 58*(3),
            453–467. https://doi.org/10.1093/biomet/58.3.453
        Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping
            unobserved item-respondent interactions: A latent space item
            response model with interaction map. *Psychometrika, 86*(2),
            378–403. https://doi.org/10.1007/s11336-021-09762-5
    """
    observed_array = np.asarray(observed, dtype=np.float64, order="C")
    expected_array = np.asarray(expected, dtype=np.float64, order="C")
    if observed_array.ndim != 2 or expected_array.ndim != 2:
        raise ValueError("observed and expected must be two-dimensional")
    if isinstance(axis_count, (bool, np.bool_)) or not isinstance(
        axis_count, (int, np.integer)
    ):
        raise TypeError("axis_count must be a positive integer")
    axis_count_value = int(axis_count)
    if axis_count_value <= 0:
        raise ValueError("axis_count must be a positive integer")
    raw = dict(
        _core.residual_interaction_map(observed_array, expected_array, axis_count_value)
    )
    person_indices = np.asarray(raw["person_indices"], dtype=np.int64)
    item_indices = np.asarray(raw["item_indices"], dtype=np.int64)
    rows = person_indices.size
    columns = item_indices.size
    return ResidualInteractionMap(
        person_indices=person_indices,
        item_indices=item_indices,
        person_coordinates=np.asarray(
            raw["person_coordinates"], dtype=np.float64
        ).reshape(rows, axis_count_value),
        item_coordinates=np.asarray(raw["item_coordinates"], dtype=np.float64).reshape(
            columns, axis_count_value
        ),
        singular_values=np.asarray(raw["singular_values"], dtype=np.float64),
        axis_shares=np.asarray(raw["axis_shares"], dtype=np.float64),
        reconstruction=np.asarray(raw["reconstruction"], dtype=np.float64).reshape(
            rows, columns
        ),
        unexplained=np.asarray(raw["unexplained"], dtype=np.float64).reshape(
            rows, columns
        ),
        cross_share=np.asarray(
            [np.nan if value is None else value for value in raw["cross_share"]],
            dtype=np.float64,
        ).reshape(rows, columns),
    )
