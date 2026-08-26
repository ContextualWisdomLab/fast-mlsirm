"""Public Python marshalling for the versioned Rust residual interaction-map envelope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._interaction_map_core_loader import interaction_map_core
from .interaction_map import _axis_count, _trusted_matrix


RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION = "fast-mlsirm.residual-interaction-map.v1"


@dataclass(frozen=True)
class ResidualInteractionMapEnvelope:
    """Versioned product-neutral interaction-map result calculated by Rust.

    Python validates and marshals caller evidence only. Rank, coverage, retained
    cells, coordinates, extrema, distances, residuals, and reconstruction are
    returned by the Rust numerical owner and are not recomputed here.
    """

    schema_version: str
    algorithm_id: str
    implementation_version: str
    calculation_provenance: str
    requested_axis_count: int
    cell_extrema_tie_policy: str
    finite_value_status: bool
    retained_person_ids: tuple[str, ...]
    retained_item_ids: tuple[str, ...]
    closest_cell_ids: tuple[str, str] | None
    farthest_cell_ids: tuple[str, str] | None
    person_indices: np.ndarray
    item_indices: np.ndarray
    scored_person_count: int
    scored_item_count: int
    effective_rank: int
    map_person_count: int
    map_item_count: int
    incomplete_person_count: int
    incomplete_item_count: int
    closest_cell: tuple[int, int] | None
    farthest_cell: tuple[int, int] | None
    person_coordinates: np.ndarray
    item_coordinates: np.ndarray
    singular_values: np.ndarray
    axis_shares: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    residual: np.ndarray
    distance: np.ndarray
    reconstruction: np.ndarray
    unexplained: np.ndarray
    cross_share: np.ndarray


def _schema_version(value: object) -> str:
    """Admit only the exact supported public schema before caller data work."""
    if type(value) is not str or value != RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION:
        raise ValueError("unsupported residual interaction map schema version")
    return value


def _opaque_ids(name: str, value: object) -> list[str]:
    """Copy exact inert opaque identifiers without caller iteration/coercion hooks."""
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an exact built-in list or tuple of strings")
    normalized: list[str] = []
    for identifier in value:
        if type(identifier) is not str:
            raise ValueError(f"{name} must contain only exact strings")
        normalized.append(identifier)
    return normalized


def _optional_string_pair(value: object) -> tuple[str, str] | None:
    """Normalize one package-owned optional pair returned by the Rust binding."""
    if value is None:
        return None
    pair = tuple(value)  # Rust-owned tuple/list; no caller object reaches this boundary.
    if len(pair) != 2:
        raise RuntimeError("Rust interaction-map envelope returned an invalid identifier pair")
    return str(pair[0]), str(pair[1])


def _optional_index_pair(value: object) -> tuple[int, int] | None:
    """Normalize one package-owned optional index pair returned by the Rust binding."""
    if value is None:
        return None
    pair = tuple(value)
    if len(pair) != 2:
        raise RuntimeError("Rust interaction-map envelope returned an invalid index pair")
    return int(pair[0]), int(pair[1])


def residual_interaction_map_envelope(
    observed: object,
    expected: object,
    *,
    person_ids: list[str] | tuple[str, ...],
    item_ids: list[str] | tuple[str, ...],
    axis_count: int,
    schema_version: str = RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
) -> ResidualInteractionMapEnvelope:
    """Return one versioned Rust-owned residual interaction-map envelope.

    ``NaN`` is the only observed-response missing value. Expected values must be
    finite. Opaque identifier carriers and the schema/axis controls are sealed
    before caller matrix protocols; numerical map quantities are calculated only
    by the Rust core.
    """
    schema = _schema_version(schema_version)
    axis = _axis_count(axis_count)
    persons = _opaque_ids("person_ids", person_ids)
    items = _opaque_ids("item_ids", item_ids)

    observed_array = _trusted_matrix("observed", observed, allow_nan=True)
    expected_array = _trusted_matrix("expected", expected, allow_nan=False)
    if observed_array.shape != expected_array.shape:
        raise ValueError("observed and expected must have the same two-dimensional shape")

    raw = dict(
        interaction_map_core().residual_interaction_map_envelope(
            schema,
            persons,
            items,
            observed_array,
            expected_array,
            axis,
        )
    )
    map_person_count = int(raw["map_person_count"])
    map_item_count = int(raw["map_item_count"])

    return ResidualInteractionMapEnvelope(
        schema_version=str(raw["schema_version"]),
        algorithm_id=str(raw["algorithm_id"]),
        implementation_version=str(raw["implementation_version"]),
        calculation_provenance=str(raw["calculation_provenance"]),
        requested_axis_count=int(raw["requested_axis_count"]),
        cell_extrema_tie_policy=str(raw["cell_extrema_tie_policy"]),
        finite_value_status=bool(raw["finite_value_status"]),
        retained_person_ids=tuple(str(value) for value in raw["retained_person_ids"]),
        retained_item_ids=tuple(str(value) for value in raw["retained_item_ids"]),
        closest_cell_ids=_optional_string_pair(raw["closest_cell_ids"]),
        farthest_cell_ids=_optional_string_pair(raw["farthest_cell_ids"]),
        person_indices=np.asarray(raw["person_indices"], dtype=np.int64),
        item_indices=np.asarray(raw["item_indices"], dtype=np.int64),
        scored_person_count=int(raw["scored_person_count"]),
        scored_item_count=int(raw["scored_item_count"]),
        effective_rank=int(raw["effective_rank"]),
        map_person_count=map_person_count,
        map_item_count=map_item_count,
        incomplete_person_count=int(raw["incomplete_person_count"]),
        incomplete_item_count=int(raw["incomplete_item_count"]),
        closest_cell=_optional_index_pair(raw["closest_cell"]),
        farthest_cell=_optional_index_pair(raw["farthest_cell"]),
        person_coordinates=np.asarray(raw["person_coordinates"], dtype=np.float64).reshape(
            map_person_count, axis
        ),
        item_coordinates=np.asarray(raw["item_coordinates"], dtype=np.float64).reshape(
            map_item_count, axis
        ),
        singular_values=np.asarray(raw["singular_values"], dtype=np.float64),
        axis_shares=np.asarray(raw["axis_shares"], dtype=np.float64),
        observed=np.asarray(raw["observed"], dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        expected=np.asarray(raw["expected"], dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        residual=np.asarray(raw["residual"], dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        distance=np.asarray(raw["distance"], dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        reconstruction=np.asarray(raw["reconstruction"], dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        unexplained=np.asarray(raw["unexplained"], dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        cross_share=np.asarray(
            [np.nan if value is None else value for value in raw["cross_share"]],
            dtype=np.float64,
        ).reshape(map_person_count, map_item_count),
    )
