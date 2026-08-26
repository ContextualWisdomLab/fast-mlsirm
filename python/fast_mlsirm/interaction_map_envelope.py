"""Public Python marshalling for the versioned Rust residual interaction-map envelope."""

from __future__ import annotations

import hashlib
import math
import struct
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
    cells, coordinates, extrema, distances, residuals, and reconstruction are
    returned by the Rust numerical owner and are not recomputed here.
    """

    schema_version: str
    algorithm_id: str
    implementation_version: str
    calculation_provenance: str
    input_digest: str
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


def _digest_field(
    digest: "hashlib._Hash", tag: str, payload: bytes | memoryview
) -> None:
    """Append one unambiguous tagged byte field to the request digest."""
    tag_bytes = tag.encode("ascii")
    digest.update(struct.pack(">H", len(tag_bytes)))
    digest.update(tag_bytes)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _canonical_float64_bytes(array: np.ndarray) -> memoryview:
    """Expose validated matrix evidence as C-order little-endian float64 bytes."""
    canonical = array.astype("<f8", copy=False)
    return memoryview(canonical).cast("B")


def _input_digest(
    schema: str,
    axis: int,
    persons: list[str],
    items: list[str],
    observed: np.ndarray,
    expected: np.ndarray,
) -> str:
    """Return SHA-256 over the exact validated interaction-map request evidence."""
    digest = hashlib.sha256()
    _digest_field(digest, "schema", schema.encode("utf-8"))
    _digest_field(digest, "axis_count", struct.pack(">Q", axis))
    _digest_field(digest, "person_count", struct.pack(">Q", len(persons)))
    for identifier in persons:
        _digest_field(digest, "person_id", identifier.encode("utf-8"))
    _digest_field(digest, "item_count", struct.pack(">Q", len(items)))
    for identifier in items:
        _digest_field(digest, "item_id", identifier.encode("utf-8"))
    rows, columns = observed.shape
    _digest_field(digest, "matrix_shape", struct.pack(">QQ", rows, columns))
    _digest_field(digest, "observed_f64le", _canonical_float64_bytes(observed))
    _digest_field(digest, "expected_f64le", _canonical_float64_bytes(expected))
    return digest.hexdigest()


def _required_rust_value(raw: dict[str, object], key: str) -> object:
    """Read one required Rust-owned result field without coercing foreign values."""
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


def _validate_rust_metadata(
    raw: dict[str, object], requested_axis_count: int, expected_input_digest: str
) -> str:
    """Replay the public v1 metadata contract before numerical payload marshalling."""
    schema_version = _required_rust_value(raw, "schema_version")
    if type(schema_version) is not str or schema_version != RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION:
        raise RuntimeError("Rust interaction-map envelope schema version mismatch")

    algorithm_id = _required_rust_value(raw, "algorithm_id")
    if type(algorithm_id) is not str or algorithm_id != _RESIDUAL_INTERACTION_MAP_ALGORITHM_ID:
        raise RuntimeError("Rust interaction-map envelope algorithm mismatch")

    implementation_version = _required_rust_value(raw, "implementation_version")
    expected_implementation_version = _installed_package_version()
    if (
        type(implementation_version) is not str
        or implementation_version != expected_implementation_version
    ):
        raise RuntimeError("Rust interaction-map envelope implementation version mismatch")

    calculation_provenance = _required_rust_value(raw, "calculation_provenance")
    if (
        type(calculation_provenance) is not str
        or calculation_provenance != _RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE
    ):
        raise RuntimeError("Rust interaction-map envelope calculation provenance mismatch")

    input_digest = _required_rust_value(raw, "input_digest")
    if type(input_digest) is not str or input_digest != expected_input_digest:
        raise RuntimeError("Rust interaction-map envelope input digest mismatch")

    returned_axis_count = _required_rust_value(raw, "requested_axis_count")
    if type(returned_axis_count) is not int or returned_axis_count != requested_axis_count:
        raise RuntimeError("Rust interaction-map envelope requested axis count mismatch")

    tie_policy = _required_rust_value(raw, "cell_extrema_tie_policy")
    if type(tie_policy) is not str or tie_policy != _RESIDUAL_INTERACTION_MAP_TIE_POLICY:
        raise RuntimeError("Rust interaction-map envelope tie policy mismatch")

    finite_value_status = _required_rust_value(raw, "finite_value_status")
    if type(finite_value_status) is not bool or finite_value_status is not True:
        raise RuntimeError("Rust interaction-map envelope finite-value status mismatch")

    return implementation_version


def _rust_nonnegative_int(raw: dict[str, object], key: str) -> int:
    """Return one exact non-negative Rust integer without invoking coercion hooks."""
    value = _required_rust_value(raw, key)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"Rust interaction-map envelope {key} must be a non-negative integer")
    return value


def _rust_sequence(raw: dict[str, object], key: str) -> list[object] | tuple[object, ...]:
    """Return one exact Rust list/tuple result without accepting protocol-bearing carriers."""
    value = _required_rust_value(raw, key)
    if type(value) not in (list, tuple):
        raise RuntimeError(f"Rust interaction-map envelope {key} must be an exact list or tuple")
    return value


def _rust_string_vector(raw: dict[str, object], key: str, expected_length: int) -> tuple[str, ...]:
    """Validate one exact Rust string vector and its expected cardinality."""
    value = _rust_sequence(raw, key)
    if len(value) != expected_length:
        raise RuntimeError(f"Rust interaction-map envelope {key} length mismatch")
    if any(type(item) is not str for item in value):
        raise RuntimeError(f"Rust interaction-map envelope {key} must contain exact strings")
    return tuple(value)


def _rust_index_vector(
    raw: dict[str, object],
    key: str,
    expected_length: int,
    upper_bound: int,
) -> tuple[int, ...]:
    """Validate one exact Rust original-index vector before NumPy materialization."""
    value = _rust_sequence(raw, key)
    if len(value) != expected_length:
        raise RuntimeError(f"Rust interaction-map envelope {key} length mismatch")
    normalized: list[int] = []
    previous = -1
    for item in value:
        if type(item) is not int or item < 0 or item >= upper_bound:
            raise RuntimeError(f"Rust interaction-map envelope {key} contains an invalid index")
        if item <= previous:
            raise RuntimeError(f"Rust interaction-map envelope {key} must be strictly increasing")
        normalized.append(item)
        previous = item
    return tuple(normalized)


def _rust_float_vector(
    raw: dict[str, object],
    key: str,
    expected_length: int,
    *,
    allow_none: bool = False,
) -> tuple[float | None, ...]:
    """Validate one Rust f64/Option<f64> vector before NumPy coercion."""
    value = _rust_sequence(raw, key)
    if len(value) != expected_length:
        raise RuntimeError(f"Rust interaction-map envelope {key} length mismatch")
    normalized: list[float | None] = []
    for item in value:
        if allow_none and item is None:
            normalized.append(None)
            continue
        if type(item) is not float:
            raise RuntimeError(f"Rust interaction-map envelope {key} must contain exact floats")
        if not math.isfinite(item):
            raise RuntimeError(f"Rust interaction-map envelope {key} contains a non-finite value")
        normalized.append(item)
    return tuple(normalized)


def _rust_string_pair(raw: dict[str, object], key: str) -> tuple[str, str] | None:
    """Validate one package-owned optional string pair without caller coercion."""
    value = _required_rust_value(raw, key)
    if value is None:
        return None
    if type(value) not in (list, tuple) or len(value) != 2:
        raise RuntimeError(f"Rust interaction-map envelope {key} returned an invalid pair")
    if type(value[0]) is not str or type(value[1]) is not str:
        raise RuntimeError(f"Rust interaction-map envelope {key} must contain exact strings")
    return value[0], value[1]


def _rust_index_pair(raw: dict[str, object], key: str) -> tuple[int, int] | None:
    """Validate one package-owned optional original-index pair without coercion."""
    value = _required_rust_value(raw, key)
    if value is None:
        return None
    if type(value) not in (list, tuple) or len(value) != 2:
        raise RuntimeError(f"Rust interaction-map envelope {key} returned an invalid pair")
    if type(value[0]) is not int or type(value[1]) is not int or value[0] < 0 or value[1] < 0:
        raise RuntimeError(f"Rust interaction-map envelope {key} must contain non-negative integers")
    return value[0], value[1]


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

    input_digest = _input_digest(schema, axis, persons, items, observed_array, expected_array)
    raw_result = interaction_map_core().residual_interaction_map_envelope(
        schema,
        input_digest,
        persons,
        items,
        observed_array,
        expected_array,
        axis,
    )
    if type(raw_result) is not dict:
        raise RuntimeError("Rust interaction-map envelope must return an exact dict")
    raw = raw_result
    implementation_version = _validate_rust_metadata(raw, axis, input_digest)

    map_person_count = _rust_nonnegative_int(raw, "map_person_count")
    map_item_count = _rust_nonnegative_int(raw, "map_item_count")
    scored_person_count = _rust_nonnegative_int(raw, "scored_person_count")
    scored_item_count = _rust_nonnegative_int(raw, "scored_item_count")
    effective_rank = _rust_nonnegative_int(raw, "effective_rank")
    incomplete_person_count = _rust_nonnegative_int(raw, "incomplete_person_count")
    incomplete_item_count = _rust_nonnegative_int(raw, "incomplete_item_count")

    input_person_count, input_item_count = observed_array.shape
    if scored_person_count > input_person_count or scored_item_count > input_item_count:
        raise RuntimeError("Rust interaction-map envelope scored counts exceed input shape")
    if map_person_count > scored_person_count or map_item_count > scored_item_count:
        raise RuntimeError("Rust interaction-map envelope map counts exceed scored counts")
    if incomplete_person_count != scored_person_count - map_person_count:
        raise RuntimeError("Rust interaction-map envelope incomplete_person_count mismatch")
    if incomplete_item_count != scored_item_count - map_item_count:
        raise RuntimeError("Rust interaction-map envelope incomplete_item_count mismatch")
    if effective_rank > min(map_person_count, map_item_count):
        raise RuntimeError("Rust interaction-map envelope effective_rank exceeds map dimensions")

    person_indices = _rust_index_vector(
        raw, "person_indices", map_person_count, input_person_count
    )
    item_indices = _rust_index_vector(raw, "item_indices", map_item_count, input_item_count)
    retained_person_ids = _rust_string_vector(raw, "retained_person_ids", map_person_count)
    retained_item_ids = _rust_string_vector(raw, "retained_item_ids", map_item_count)
    if retained_person_ids != tuple(persons[index] for index in person_indices):
        raise RuntimeError("Rust interaction-map envelope retained_person_ids mismatch")
    if retained_item_ids != tuple(items[index] for index in item_indices):
        raise RuntimeError("Rust interaction-map envelope retained_item_ids mismatch")

    closest_cell = _rust_index_pair(raw, "closest_cell")
    farthest_cell = _rust_index_pair(raw, "farthest_cell")
    closest_cell_ids = _rust_string_pair(raw, "closest_cell_ids")
    farthest_cell_ids = _rust_string_pair(raw, "farthest_cell_ids")
    retained_person_index_set = set(person_indices)
    retained_item_index_set = set(item_indices)
    for name, cell, cell_ids in (
        ("closest", closest_cell, closest_cell_ids),
        ("farthest", farthest_cell, farthest_cell_ids),
    ):
        if cell is None:
            if cell_ids is not None:
                raise RuntimeError(f"Rust interaction-map envelope {name} cell identity mismatch")
            continue
        person_index, item_index = cell
        if person_index not in retained_person_index_set or item_index not in retained_item_index_set:
            raise RuntimeError(f"Rust interaction-map envelope {name} cell index mismatch")
        expected_ids = (persons[person_index], items[item_index])
        if cell_ids != expected_ids:
            raise RuntimeError(f"Rust interaction-map envelope {name} cell identifier mismatch")

    cell_count = map_person_count * map_item_count
    person_coordinates = _rust_float_vector(
        raw, "person_coordinates", map_person_count * axis
    )
    item_coordinates = _rust_float_vector(raw, "item_coordinates", map_item_count * axis)
    singular_values = _rust_float_vector(raw, "singular_values", effective_rank)
    axis_shares = _rust_float_vector(raw, "axis_shares", axis)
    observed_values = _rust_float_vector(raw, "observed", cell_count)
    expected_values = _rust_float_vector(raw, "expected", cell_count)
    residual_values = _rust_float_vector(raw, "residual", cell_count)
    distance_values = _rust_float_vector(raw, "distance", cell_count)
    reconstruction_values = _rust_float_vector(raw, "reconstruction", cell_count)
    unexplained_values = _rust_float_vector(raw, "unexplained", cell_count)
    cross_share_values = _rust_float_vector(raw, "cross_share", cell_count, allow_none=True)

    return ResidualInteractionMapEnvelope(
        schema_version=RESIDUAL_INTERACTION_MAP_SCHEMA_VERSION,
        algorithm_id=_RESIDUAL_INTERACTION_MAP_ALGORITHM_ID,
        implementation_version=implementation_version,
        calculation_provenance=_RESIDUAL_INTERACTION_MAP_CALCULATION_PROVENANCE,
        input_digest=input_digest,
        requested_axis_count=axis,
        cell_extrema_tie_policy=_RESIDUAL_INTERACTION_MAP_TIE_POLICY,
        finite_value_status=True,
        retained_person_ids=retained_person_ids,
        retained_item_ids=retained_item_ids,
        closest_cell_ids=closest_cell_ids,
        farthest_cell_ids=farthest_cell_ids,
        person_indices=np.asarray(person_indices, dtype=np.int64),
        item_indices=np.asarray(item_indices, dtype=np.int64),
        scored_person_count=scored_person_count,
        scored_item_count=scored_item_count,
        effective_rank=effective_rank,
        map_person_count=map_person_count,
        map_item_count=map_item_count,
        incomplete_person_count=incomplete_person_count,
        incomplete_item_count=incomplete_item_count,
        closest_cell=closest_cell,
        farthest_cell=farthest_cell,
        person_coordinates=np.asarray(person_coordinates, dtype=np.float64).reshape(
            map_person_count, axis
        ),
        item_coordinates=np.asarray(item_coordinates, dtype=np.float64).reshape(
            map_item_count, axis
        ),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        axis_shares=np.asarray(axis_shares, dtype=np.float64),
        observed=np.asarray(observed_values, dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        expected=np.asarray(expected_values, dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        residual=np.asarray(residual_values, dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        distance=np.asarray(distance_values, dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        reconstruction=np.asarray(reconstruction_values, dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        unexplained=np.asarray(unexplained_values, dtype=np.float64).reshape(
            map_person_count, map_item_count
        ),
        cross_share=np.asarray(
            [np.nan if value is None else value for value in cross_share_values],
            dtype=np.float64,
        ).reshape(map_person_count, map_item_count),
    )