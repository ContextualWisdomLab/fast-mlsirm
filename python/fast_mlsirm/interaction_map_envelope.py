"""Public Python marshalling for the versioned Rust residual interaction-map envelope."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version

import numpy as np

from ._interaction_map_core_loader import interaction_map_core
from .interaction_map import _axis_count, _trusted_matrix


RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION = "fast-mlsirm.residual-interaction-map.v1"
_RESIDUAL_INTERACTION_MAP_ALGORITHM_ID = "gabriel-complete-case-symmetric-residual-map.v1"
_RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE = (
    "mlsirm-core::interaction_map::residual_interaction_map"
)
_RESIDUAL_INTERACTION_MAP_TIE_POLICY = "lexicographic-first-original-index"
_MAX_INTERACTION_MAP_IDENTIFIER_COUNT = 20_000_000


@dataclass(frozen=True)
class ResidualInteractionMapEnvelope:
    """Versioned product-neutral interaction-map result calculated by Rust.

    Python validates and marshals caller evidence only. Rank, coverage, retained
    cells, coordinates, extrema, distances, residuals, reconstruction, and
    decomposition shares are returned by the Rust numerical owner and are not
    recomputed here.
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
    explained_share: np.ndarray
    unexplained: np.ndarray
    cross_share: np.ndarray


def _schema_version(value: object) -> str:
    """Admit only the exact supported public schema before caller data work."""
    if type(value) is not str or value != RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION:
        raise ValueError("unsupported residual interaction map schema version")
    return value


def _opaque_ids(name: str, value: object) -> list[str]:
    """Copy bounded exact opaque identifiers without caller coercion hooks."""
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an exact built-in list or tuple of strings")
    identifier_count = len(value)
    if identifier_count > _MAX_INTERACTION_MAP_IDENTIFIER_COUNT:
        raise ValueError(
            f"{name} identifier count exceeds {_MAX_INTERACTION_MAP_IDENTIFIER_COUNT}"
        )
    normalized: list[str] = []
    for identifier in value:
        if type(identifier) is not str:
            raise ValueError(f"{name} must contain only exact strings")
        normalized.append(identifier)
    return normalized


def _required_rust_metadata(raw: dict[str, object], key: str) -> object:
    """Read one required Rust-owned metadata field without coercing foreign values."""
    try:
        return raw[key]
    except KeyError as exc:
        raise RuntimeError(f"Rust interaction-map envelope is missing {key}") from exc


def _installed_package_version() -> str:
    """Return the installed Python distribution identity used to load the Rust extension."""
    try:
        return distribution_version("fast-mlsirm")
    except PackageNotFoundError as exc:
        raise RuntimeError("fast-mlsirm distribution version is unavailable") from exc


def _validate_rust_metadata(raw: dict[str, object], requested_axis_count: int) -> str:
    """Replay the public v1 metadata contract before numerical payload marshalling."""
    schema_version = _required_rust_metadata(raw, "schema_version")
    if type(schema_version) is not str or schema_version != RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION:
        raise RuntimeError("Rust interaction-map envelope schema version mismatch")

    algorithm_id = _required_rust_metadata(raw, "algorithm_id")
    if type(algorithm_id) is not str or algorithm_id != _RESIDUAL_INTERACTION_MAP_ALGORITHM_ID:
        raise RuntimeError("Rust interaction-map envelope algorithm mismatch")

    implementation_version = _required_rust_metadata(raw, "implementation_version")
    expected_implementation_version = _installed_package_version()
    if (
        type(implementation_version) is not str
        or implementation_version != expected_implementation_version
    ):
        raise RuntimeError("Rust interaction-map envelope implementation version mismatch")

    calculation_provenance = _required_rust_metadata(raw, "calculation_provenance")
    if (
        type(calculation_provenance) is not str
        or calculation_provenance != _RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE
    ):
        raise RuntimeError("Rust interaction-map envelope calculation provenance mismatch")

    returned_axis_count = _required_rust_metadata(raw, "requested_axis_count")
    if type(returned_axis_count) is not int or returned_axis_count != requested_axis_count:
        raise RuntimeError("Rust interaction-map envelope requested axis count mismatch")

    tie_policy = _required_rust_metadata(raw, "cell_extrema_tie_policy")
    if type(tie_policy) is not str or tie_policy != _RESIDUAL_INTERACTION_MAP_TIE_POLICY:
        raise RuntimeError("Rust interaction-map envelope tie policy mismatch")

    finite_value_status = _required_rust_metadata(raw, "finite_value_status")
    if type(finite_value_status) is not bool or finite_value_status is not True:
        raise RuntimeError("Rust interaction-map envelope finite-value status mismatch")

    return implementation_version


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
    implementation_version = _validate_rust_metadata(raw, axis)
    map_person_count = int(raw["map_person_count"])
    map_item_count = int(raw["map_item_count"])

    return ResidualInteractionMapEnvelope(
        schema_version=RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        algorithm_id=_RESIDUAL_INTERACTION_MAP_ALGORITHM_ID,
        implementation_version=implementation_version,
        calculation_provenance=_RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE,
        requested_axis_count=axis,
        cell_extrema_tie_policy=_RESIDUAL_INTERACTION_MAP_TIE_POLICY,
        finite_value_status=True,
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
        explained_share=np.asarray(
            [np.nan if value is None else value for value in raw["explained_share"]],
            dtype=np.float64,
        ).reshape(map_person_count, map_item_count),
        unexplained=np.asarray(raw["unexplained"], dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        cross_share=np.asarray(
            [np.nan if value is None else value for value in raw["cross_share"]],
            dtype=np.float64,
        ).reshape(map_person_count, map_item_count),
    )
