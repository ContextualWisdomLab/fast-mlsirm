"""Immutable contextual-membership and longitudinal measurement contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from enum import Enum
import math
from typing import Any

from ._validation import (
    MULTILEVEL_SCHEMA_VERSION,
    MultilevelContractError,
    artifact_digest,
    autoregressive_coefficient,
    bounded_values,
    contract_error,
    descriptive_identifier,
    exact_integer,
    fingerprint,
    membership_weight,
    schema_version,
    strict_boolean,
)

MAX_CONTEXT_MEMBERSHIPS = 100_000
MAX_TEMPORAL_OCCASIONS = 100_000
MEMBERSHIP_WEIGHT_TOLERANCE = 1e-12

_MEMBERSHIP_TOKEN = object()
_MEMBERSHIP_DESIGN_TOKEN = object()
_OCCASION_TOKEN = object()
_STATE_SPEC_TOKEN = object()
_LONGITUDINAL_DESIGN_TOKEN = object()


class LongitudinalStateKind(str, Enum):
    """Supported initial latent-state structures for repeated measurement."""

    RANDOM_INTERCEPT_SLOPE = "random_intercept_slope"
    STATIONARY_AUTOREGRESSIVE = "stationary_autoregressive"


def _state_kind(value: Any) -> LongitudinalStateKind:
    """Return one supported longitudinal state kind without value reflection."""
    if isinstance(value, LongitudinalStateKind):
        return value
    try:
        return LongitudinalStateKind(value)
    except (TypeError, ValueError, OverflowError):
        raise contract_error(
            "invalid_state_kind",
            "$.state_kind",
            "state_kind must be one of the supported longitudinal structures",
        ) from None


@dataclass(frozen=True)
class ContextMembership:
    """One weighted observation-to-context edge with exact revision provenance."""

    observation_id: str
    context_id: str
    membership_weight: float
    membership_revision_fingerprint: str
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _membership_token: InitVar[object | None] = None

    def __post_init__(self, _membership_token: object | None) -> None:
        """Reject direct construction and normalize one membership edge."""
        if _membership_token is not _MEMBERSHIP_TOKEN:
            raise contract_error(
                "unverified_context_membership",
                "$",
                "use build_context_membership",
            )
        object.__setattr__(
            self,
            "observation_id",
            descriptive_identifier(self.observation_id, "observation_id"),
        )
        object.__setattr__(
            self,
            "context_id",
            descriptive_identifier(self.context_id, "context_id"),
        )
        object.__setattr__(
            self,
            "membership_weight",
            membership_weight(self.membership_weight),
        )
        object.__setattr__(
            self,
            "membership_revision_fingerprint",
            fingerprint(
                self.membership_revision_fingerprint,
                "membership_revision_fingerprint",
            ),
        )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical edge content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "context_id": self.context_id,
            "membership_weight": self.membership_weight,
            "membership_revision_fingerprint": (
                self.membership_revision_fingerprint
            ),
        }

    @property
    def membership_fingerprint(self) -> str:
        """Return SHA-256 over the normalized membership edge."""
        return artifact_digest(self._content_dict())

    @property
    def membership_handle(self) -> str:
        """Return a descriptive 128-bit public membership handle."""
        return f"context_membership_{self.membership_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical edge content and deterministic identities."""
        return {
            **self._content_dict(),
            "membership_handle": self.membership_handle,
            "membership_fingerprint": self.membership_fingerprint,
        }


@dataclass(frozen=True)
class ContextMembershipDesign:
    """Factory-sealed sparse multiple-membership design."""

    memberships: tuple[ContextMembership, ...]
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated design factory."""
        if _design_token is not _MEMBERSHIP_DESIGN_TOKEN:
            raise contract_error(
                "unverified_context_membership_design",
                "$",
                "use build_context_membership_design",
            )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))

    @property
    def observation_ids(self) -> tuple[str, ...]:
        """Return observation identifiers in deterministic design order."""
        return tuple(sorted({value.observation_id for value in self.memberships}))

    @property
    def context_ids(self) -> tuple[str, ...]:
        """Return context identifiers in deterministic design order."""
        return tuple(sorted({value.context_id for value in self.memberships}))

    def _grouped_memberships(self) -> dict[str, tuple[ContextMembership, ...]]:
        """Return deterministic observation-level membership groups."""
        grouped: dict[str, list[ContextMembership]] = {}
        for value in self.memberships:
            grouped.setdefault(value.observation_id, []).append(value)
        return {
            observation_id: tuple(grouped[observation_id])
            for observation_id in self.observation_ids
        }

    @property
    def membership_counts(self) -> tuple[int, ...]:
        """Return membership counts aligned with ``observation_ids``."""
        grouped = self._grouped_memberships()
        return tuple(len(grouped[value]) for value in self.observation_ids)

    @property
    def membership_weights(self) -> tuple[tuple[float, ...], ...]:
        """Return exact weights aligned with observations and context order."""
        grouped = self._grouped_memberships()
        return tuple(
            tuple(edge.membership_weight for edge in grouped[observation_id])
            for observation_id in self.observation_ids
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical sparse design content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "memberships": [value._content_dict() for value in self.memberships],
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete ordered membership design."""
        return artifact_digest(self._content_dict())

    @property
    def design_handle(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"context_membership_design_{self.design_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return sparse design content and deterministic identities."""
        return {
            "schema_version": self.schema_version,
            "memberships": [value.to_dict() for value in self.memberships],
            "observation_ids": list(self.observation_ids),
            "context_ids": list(self.context_ids),
            "membership_counts": list(self.membership_counts),
            "design_handle": self.design_handle,
            "design_fingerprint": self.design_fingerprint,
        }


@dataclass(frozen=True)
class TemporalOccasion:
    """One exact repeated-measurement occasion for one respondent or system."""

    respondent_id: str
    occasion_id: str
    sequence_index: int
    time_offset_milliseconds: int
    occasion_revision_fingerprint: str
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _occasion_token: InitVar[object | None] = None

    def __post_init__(self, _occasion_token: object | None) -> None:
        """Reject direct construction and normalize one temporal occasion."""
        if _occasion_token is not _OCCASION_TOKEN:
            raise contract_error(
                "unverified_temporal_occasion",
                "$",
                "use build_temporal_occasion",
            )
        object.__setattr__(
            self,
            "respondent_id",
            descriptive_identifier(self.respondent_id, "respondent_id"),
        )
        object.__setattr__(
            self,
            "occasion_id",
            descriptive_identifier(self.occasion_id, "occasion_id"),
        )
        object.__setattr__(
            self,
            "sequence_index",
            exact_integer(
                self.sequence_index,
                "sequence_index",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "time_offset_milliseconds",
            exact_integer(
                self.time_offset_milliseconds,
                "time_offset_milliseconds",
            ),
        )
        object.__setattr__(
            self,
            "occasion_revision_fingerprint",
            fingerprint(
                self.occasion_revision_fingerprint,
                "occasion_revision_fingerprint",
            ),
        )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical occasion content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "respondent_id": self.respondent_id,
            "occasion_id": self.occasion_id,
            "sequence_index": self.sequence_index,
            "time_offset_milliseconds": self.time_offset_milliseconds,
            "occasion_revision_fingerprint": self.occasion_revision_fingerprint,
        }

    @property
    def occasion_fingerprint(self) -> str:
        """Return SHA-256 over the normalized temporal occasion."""
        return artifact_digest(self._content_dict())

    @property
    def occasion_handle(self) -> str:
        """Return a descriptive 128-bit public occasion handle."""
        return f"temporal_occasion_{self.occasion_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical occasion content and deterministic identities."""
        return {
            **self._content_dict(),
            "occasion_handle": self.occasion_handle,
            "occasion_fingerprint": self.occasion_fingerprint,
        }


@dataclass(frozen=True)
class LongitudinalStateSpec:
    """One immutable latent-state and residual-dependence specification."""

    state_kind: LongitudinalStateKind
    autoregressive_coefficient: float | None
    include_lagged_response_dependence: bool
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _state_spec_token: InitVar[object | None] = None

    def __post_init__(self, _state_spec_token: object | None) -> None:
        """Reject direct construction and normalize the state specification."""
        if _state_spec_token is not _STATE_SPEC_TOKEN:
            raise contract_error(
                "unverified_longitudinal_state_spec",
                "$",
                "use build_longitudinal_state_spec",
            )
        normalized_kind = _state_kind(self.state_kind)
        object.__setattr__(self, "state_kind", normalized_kind)
        if normalized_kind is LongitudinalStateKind.STATIONARY_AUTOREGRESSIVE:
            object.__setattr__(
                self,
                "autoregressive_coefficient",
                autoregressive_coefficient(self.autoregressive_coefficient),
            )
        elif self.autoregressive_coefficient is not None:
            raise contract_error(
                "unexpected_autoregressive_coefficient",
                "$.autoregressive_coefficient",
                "random-intercept/slope states do not accept an AR coefficient",
            )
        object.__setattr__(
            self,
            "include_lagged_response_dependence",
            strict_boolean(
                self.include_lagged_response_dependence,
                "include_lagged_response_dependence",
            ),
        )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical state content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "state_kind": self.state_kind.value,
            "autoregressive_coefficient": self.autoregressive_coefficient,
            "include_lagged_response_dependence": (
                self.include_lagged_response_dependence
            ),
        }

    @property
    def state_spec_fingerprint(self) -> str:
        """Return SHA-256 over the exact longitudinal state specification."""
        return artifact_digest(self._content_dict())

    @property
    def state_spec_handle(self) -> str:
        """Return a descriptive 128-bit public state-specification handle."""
        return f"longitudinal_state_spec_{self.state_spec_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical state content and deterministic identities."""
        return {
            **self._content_dict(),
            "state_spec_handle": self.state_spec_handle,
            "state_spec_fingerprint": self.state_spec_fingerprint,
        }


@dataclass(frozen=True)
class LongitudinalDesign:
    """Factory-sealed repeated-measurement design with ordered provenance."""

    occasions: tuple[TemporalOccasion, ...]
    state_spec: LongitudinalStateSpec
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated design factory."""
        if _design_token is not _LONGITUDINAL_DESIGN_TOKEN:
            raise contract_error(
                "unverified_longitudinal_design",
                "$",
                "use build_longitudinal_design",
            )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))

    @property
    def respondent_ids(self) -> tuple[str, ...]:
        """Return respondent identifiers in deterministic design order."""
        return tuple(sorted({value.respondent_id for value in self.occasions}))

    def _grouped_occasions(self) -> dict[str, tuple[TemporalOccasion, ...]]:
        """Return deterministic respondent-level occasion sequences."""
        grouped: dict[str, list[TemporalOccasion]] = {}
        for value in self.occasions:
            grouped.setdefault(value.respondent_id, []).append(value)
        return {
            respondent_id: tuple(grouped[respondent_id])
            for respondent_id in self.respondent_ids
        }

    @property
    def occasion_counts(self) -> tuple[int, ...]:
        """Return occasion counts aligned with ``respondent_ids``."""
        grouped = self._grouped_occasions()
        return tuple(len(grouped[value]) for value in self.respondent_ids)

    @property
    def time_offsets_milliseconds(self) -> tuple[tuple[int, ...], ...]:
        """Return exact irregular time offsets aligned with respondent order."""
        grouped = self._grouped_occasions()
        return tuple(
            tuple(
                occasion.time_offset_milliseconds
                for occasion in grouped[respondent_id]
            )
            for respondent_id in self.respondent_ids
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical longitudinal content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "occasions": [value._content_dict() for value in self.occasions],
            "state_spec": self.state_spec._content_dict(),
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete ordered longitudinal design."""
        return artifact_digest(self._content_dict())

    @property
    def design_handle(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"longitudinal_design_{self.design_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return longitudinal design content and deterministic identities."""
        return {
            "schema_version": self.schema_version,
            "occasions": [value.to_dict() for value in self.occasions],
            "state_spec": self.state_spec.to_dict(),
            "respondent_ids": list(self.respondent_ids),
            "occasion_counts": list(self.occasion_counts),
            "design_handle": self.design_handle,
            "design_fingerprint": self.design_fingerprint,
        }


def build_context_membership(
    *,
    observation_id: str,
    context_id: str,
    membership_weight: float,
    membership_revision_fingerprint: str,
) -> ContextMembership:
    """Build one validated observation-to-context membership edge."""
    return ContextMembership(
        observation_id=observation_id,
        context_id=context_id,
        membership_weight=membership_weight,
        membership_revision_fingerprint=membership_revision_fingerprint,
        _membership_token=_MEMBERSHIP_TOKEN,
    )


def build_context_membership_design(
    memberships: Iterable[ContextMembership],
) -> ContextMembershipDesign:
    """Build one bounded canonical multiple-membership design."""
    raw = bounded_values(
        memberships,
        "memberships",
        minimum=1,
        maximum=MAX_CONTEXT_MEMBERSHIPS,
    )
    for index, value in enumerate(raw):
        if not isinstance(value, ContextMembership):
            raise contract_error(
                "invalid_context_membership",
                f"$.memberships[{index}]",
                "memberships must contain ContextMembership values",
            )
    ordered = tuple(
        sorted(
            raw,
            key=lambda value: (
                value.observation_id,
                value.context_id,
                value.membership_revision_fingerprint,
            ),
        )
    )
    cells: set[tuple[str, str]] = set()
    revisions: dict[str, tuple[str, str]] = {}
    grouped_weights: dict[str, list[float]] = {}
    for index, value in enumerate(ordered):
        cell = (value.observation_id, value.context_id)
        if cell in cells:
            raise contract_error(
                "duplicate_context_membership",
                f"$.memberships[{index}]",
                "each observation-context cell may occur only once",
            )
        cells.add(cell)
        previous = revisions.get(value.membership_revision_fingerprint)
        if previous is not None and previous != cell:
            raise contract_error(
                "membership_revision_conflict",
                f"$.memberships[{index}].membership_revision_fingerprint",
                "one membership revision cannot identify different assignments",
            )
        revisions[value.membership_revision_fingerprint] = cell
        grouped_weights.setdefault(value.observation_id, []).append(
            value.membership_weight
        )
    for weights in grouped_weights.values():
        if not math.isclose(
            math.fsum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=MEMBERSHIP_WEIGHT_TOLERANCE,
        ):
            raise contract_error(
                "membership_weight_total_mismatch",
                "$.memberships",
                "membership weights must sum to one for every observation",
            )
    return ContextMembershipDesign(
        memberships=ordered,
        _design_token=_MEMBERSHIP_DESIGN_TOKEN,
    )


def build_temporal_occasion(
    *,
    respondent_id: str,
    occasion_id: str,
    sequence_index: int,
    time_offset_milliseconds: int,
    occasion_revision_fingerprint: str,
) -> TemporalOccasion:
    """Build one validated repeated-measurement occasion."""
    return TemporalOccasion(
        respondent_id=respondent_id,
        occasion_id=occasion_id,
        sequence_index=sequence_index,
        time_offset_milliseconds=time_offset_milliseconds,
        occasion_revision_fingerprint=occasion_revision_fingerprint,
        _occasion_token=_OCCASION_TOKEN,
    )


def build_longitudinal_state_spec(
    *,
    state_kind: LongitudinalStateKind | str,
    autoregressive_coefficient: float | None = None,
    include_lagged_response_dependence: bool = False,
) -> LongitudinalStateSpec:
    """Build one validated latent-state and lag-dependence specification."""
    return LongitudinalStateSpec(
        state_kind=state_kind,  # type: ignore[arg-type]
        autoregressive_coefficient=autoregressive_coefficient,
        include_lagged_response_dependence=(
            include_lagged_response_dependence
        ),
        _state_spec_token=_STATE_SPEC_TOKEN,
    )


def build_longitudinal_design(
    *,
    occasions: Iterable[TemporalOccasion],
    state_spec: LongitudinalStateSpec,
) -> LongitudinalDesign:
    """Build one bounded canonical longitudinal measurement design."""
    raw = bounded_values(
        occasions,
        "occasions",
        minimum=1,
        maximum=MAX_TEMPORAL_OCCASIONS,
    )
    if not isinstance(state_spec, LongitudinalStateSpec):
        raise contract_error(
            "invalid_longitudinal_state_spec",
            "$.state_spec",
            "state_spec must be a LongitudinalStateSpec",
        )
    for index, value in enumerate(raw):
        if not isinstance(value, TemporalOccasion):
            raise contract_error(
                "invalid_temporal_occasion",
                f"$.occasions[{index}]",
                "occasions must contain TemporalOccasion values",
            )
    ordered = tuple(
        sorted(
            raw,
            key=lambda value: (
                value.respondent_id,
                value.sequence_index,
                value.occasion_id,
                value.occasion_revision_fingerprint,
            ),
        )
    )
    revisions: dict[str, tuple[str, str]] = {}
    grouped: dict[str, list[tuple[int, TemporalOccasion]]] = {}
    for index, value in enumerate(ordered):
        identity = (value.respondent_id, value.occasion_id)
        previous = revisions.get(value.occasion_revision_fingerprint)
        if previous is not None and previous != identity:
            raise contract_error(
                "occasion_revision_conflict",
                f"$.occasions[{index}].occasion_revision_fingerprint",
                "one occasion revision cannot identify different occasions",
            )
        revisions[value.occasion_revision_fingerprint] = identity
        grouped.setdefault(value.respondent_id, []).append((index, value))
    for values in grouped.values():
        occasion_ids: set[str] = set()
        sequence_indices: set[int] = set()
        time_offsets: set[int] = set()
        previous_time: int | None = None
        for index, value in values:
            if value.occasion_id in occasion_ids:
                raise contract_error(
                    "duplicate_temporal_occasion",
                    f"$.occasions[{index}].occasion_id",
                    "occasion identifiers must be unique within a respondent",
                )
            occasion_ids.add(value.occasion_id)
            if value.sequence_index in sequence_indices:
                raise contract_error(
                    "duplicate_temporal_sequence",
                    f"$.occasions[{index}].sequence_index",
                    "sequence indices must be unique within a respondent",
                )
            sequence_indices.add(value.sequence_index)
            if value.time_offset_milliseconds in time_offsets:
                raise contract_error(
                    "duplicate_temporal_offset",
                    f"$.occasions[{index}].time_offset_milliseconds",
                    "time offsets must be unique within a respondent",
                )
            time_offsets.add(value.time_offset_milliseconds)
            if (
                previous_time is not None
                and value.time_offset_milliseconds <= previous_time
            ):
                raise contract_error(
                    "nonincreasing_temporal_order",
                    "$.occasions",
                    "time offsets must increase with sequence order",
                )
            previous_time = value.time_offset_milliseconds
    return LongitudinalDesign(
        occasions=ordered,
        state_spec=state_spec,
        _design_token=_LONGITUDINAL_DESIGN_TOKEN,
    )


__all__ = [
    "ContextMembership",
    "ContextMembershipDesign",
    "LongitudinalDesign",
    "LongitudinalStateKind",
    "LongitudinalStateSpec",
    "MAX_CONTEXT_MEMBERSHIPS",
    "MAX_TEMPORAL_OCCASIONS",
    "MEMBERSHIP_WEIGHT_TOLERANCE",
    "MultilevelContractError",
    "TemporalOccasion",
    "build_context_membership",
    "build_context_membership_design",
    "build_longitudinal_design",
    "build_longitudinal_state_spec",
    "build_temporal_occasion",
]
