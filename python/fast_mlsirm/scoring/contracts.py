"""Content-addressed assessment and policy contracts for automated scoring.

The contracts bind an assessment to exact :mod:`fast_mlsirm.rubric` revisions.
They validate, serialize, and preserve provenance only; all psychometric
estimation remains in the existing Rust-backed public APIs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any
import json

from ..rubric.models import (
    SCHEMA_VERSION,
    ResponseFormat,
    RubricSpecification,
    _FINGERPRINT_PATTERN,
    _identifier,
    _schema_version,
    _semantic_version,
    _sha256_hex,
    _text,
)
from ._json import canonical_object_json, decode_object_json
from .errors import ScoringContractError, contract_error

MAX_CONSTRUCTS = 32
MAX_RUBRIC_BINDINGS = 64
_POLICY_TOKEN = object()
_ASSESSMENT_TOKEN = object()
_RUBRIC_BINDING_TOKEN = object()


class PolicyKind(str, Enum):
    """Operational policy families required by every assessment contract."""

    ENGINE = "engine_policy"
    CALIBRATION = "calibration_policy"
    VALIDATION = "validation_policy"
    ADJUDICATION = "adjudication_policy"
    MONITORING = "monitoring_policy"


def _identifier_field(value: Any, field: str, path: str) -> str:
    """Normalize an identifier and translate failure into a stable error."""
    try:
        return _identifier(value, field)
    except ValueError as exc:
        raise contract_error(f"invalid_{field}", path, str(exc)) from None


def _text_field(
    value: Any,
    field: str,
    path: str,
    maximum: int,
) -> str:
    """Normalize bounded text and translate failure into a stable error."""
    try:
        return _text(value, field, maximum=maximum)
    except ValueError as exc:
        raise contract_error(f"invalid_{field}", path, str(exc)) from None


def _version_field(value: Any, field: str, path: str) -> str:
    """Normalize a canonical semantic version."""
    try:
        return _semantic_version(value, field)
    except ValueError as exc:
        raise contract_error(f"invalid_{field}", path, str(exc)) from None


def _schema_field(value: Any, path: str) -> str:
    """Require the schema version implemented by this package slice."""
    try:
        return _schema_version(value)
    except ValueError as exc:
        raise contract_error("invalid_schema_version", path, str(exc)) from None


def _fingerprint_field(value: Any, field: str, path: str) -> str:
    """Normalize one complete lower-hexadecimal SHA-256 fingerprint."""
    normalized = _text_field(value, field, path, 64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise contract_error(
            f"invalid_{field}",
            path,
            f"{field} must be 64 lower hexadecimal characters",
        )
    return normalized


def _bounded_iterable(
    values: Any,
    field: str,
    minimum: int,
    maximum: int,
) -> tuple[Any, ...]:
    """Materialize a bounded iterable without accepting text as a collection."""
    path = f"$.{field}"
    if isinstance(values, (str, bytes)):
        raise contract_error(
            f"invalid_{field}",
            path,
            f"{field} must be a collection",
        )
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise contract_error(
            f"invalid_{field}",
            path,
            f"{field} must be a collection",
        ) from exc
    output: list[Any] = []
    for index, value in enumerate(iterator):
        if index >= maximum:
            raise contract_error(
                f"invalid_{field}",
                path,
                f"{field} must contain at most {maximum} values",
            )
        output.append(value)
    if len(output) < minimum:
        raise contract_error(
            f"invalid_{field}",
            path,
            f"{field} must contain at least {minimum} value",
        )
    return tuple(output)


def _policy_kind(
    value: PolicyKind | str,
    path: str = "$.policy_kind",
) -> PolicyKind:
    """Normalize a policy-family enum value."""
    if isinstance(value, PolicyKind):
        return value
    try:
        return PolicyKind(value)
    except (TypeError, ValueError) as exc:
        raise contract_error(
            "invalid_policy_kind",
            path,
            f"policy_kind must be one of {[kind.value for kind in PolicyKind]}",
        ) from exc


@dataclass(frozen=True)
class ConstructSpec:
    """One declared latent construct used by exact rubric revisions."""

    construct_id: str
    label: str
    definition: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the construct and enforce descriptive public fields."""
        object.__setattr__(
            self,
            "construct_id",
            _identifier_field(
                self.construct_id,
                "construct_id",
                "$.construct_id",
            ),
        )
        object.__setattr__(
            self,
            "label",
            _text_field(self.label, "label", "$.label", 256),
        )
        object.__setattr__(
            self,
            "definition",
            _text_field(
                self.definition,
                "definition",
                "$.definition",
                8_192,
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            _schema_field(self.schema_version, "$.schema_version"),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return construct content without derived identity fields."""
        return {
            "schema_version": self.schema_version,
            "construct_id": self.construct_id,
            "label": self.label,
            "definition": self.definition,
        }

    @property
    def construct_fingerprint(self) -> str:
        """Return SHA-256 over the complete normalized construct content."""
        return _sha256_hex(self._content_dict())

    @property
    def construct_handle(self) -> str:
        """Return a descriptive 128-bit public construct handle."""
        return f"construct_spec_{self.construct_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical construct content and deterministic identities."""
        return {
            **self._content_dict(),
            "construct_handle": self.construct_handle,
            "construct_fingerprint": self.construct_fingerprint,
        }


@dataclass(frozen=True)
class PolicyDocument:
    """One factory-sealed content-addressed operational scoring policy."""

    policy_id: str
    policy_version: str
    policy_kind: PolicyKind
    settings_json: str
    schema_version: str = SCHEMA_VERSION
    _policy_token: InitVar[object | None] = None

    def __post_init__(self, _policy_token: object | None) -> None:
        """Reject direct construction and normalize governed policy fields."""
        if _policy_token is not _POLICY_TOKEN:
            raise contract_error(
                "unverified_policy_document",
                "$",
                "use build_policy_document",
            )
        object.__setattr__(
            self,
            "policy_id",
            _identifier_field(self.policy_id, "policy_id", "$.policy_id"),
        )
        object.__setattr__(
            self,
            "policy_version",
            _version_field(
                self.policy_version,
                "policy_version",
                "$.policy_version",
            ),
        )
        object.__setattr__(
            self,
            "policy_kind",
            _policy_kind(self.policy_kind),
        )
        settings = decode_object_json(self.settings_json, "settings_json")
        if canonical_object_json(settings, "settings") != self.settings_json:
            raise contract_error(
                "noncanonical_policy_settings",
                "$.settings",
                "settings must use canonical JSON",
            )
        object.__setattr__(
            self,
            "schema_version",
            _schema_field(self.schema_version, "$.schema_version"),
        )

    @property
    def settings(self) -> dict[str, Any]:
        """Return a fresh decoded copy of immutable policy settings."""
        return decode_object_json(self.settings_json, "settings_json")

    def _content_dict(self) -> dict[str, Any]:
        """Return policy content without derived identity fields."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_kind": self.policy_kind.value,
            "settings": self.settings,
        }

    @property
    def policy_fingerprint(self) -> str:
        """Return SHA-256 over the complete normalized policy content."""
        return _sha256_hex(self._content_dict())

    @property
    def policy_handle(self) -> str:
        """Return a descriptive 128-bit public policy handle."""
        return f"policy_document_{self.policy_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical policy content and deterministic identities."""
        return {
            **self._content_dict(),
            "policy_handle": self.policy_handle,
            "policy_fingerprint": self.policy_fingerprint,
        }


@dataclass(frozen=True)
class RubricBinding:
    """Factory-issued immutable binding to one exact rubric specification."""

    rubric_id: str
    rubric_version: str
    rubric_fingerprint: str
    construct_id: str
    response_format: ResponseFormat
    schema_version: str = SCHEMA_VERSION
    _binding_token: InitVar[object | None] = None

    def __post_init__(self, _binding_token: object | None) -> None:
        """Reject caller-forged bindings and normalize copied rubric metadata."""
        if _binding_token is not _RUBRIC_BINDING_TOKEN:
            raise contract_error(
                "unverified_rubric_binding",
                "$",
                "bindings are issued by build_assessment_spec",
            )
        object.__setattr__(
            self,
            "rubric_id",
            _identifier_field(self.rubric_id, "rubric_id", "$.rubric_id"),
        )
        object.__setattr__(
            self,
            "rubric_version",
            _version_field(
                self.rubric_version,
                "rubric_version",
                "$.rubric_version",
            ),
        )
        object.__setattr__(
            self,
            "rubric_fingerprint",
            _fingerprint_field(
                self.rubric_fingerprint,
                "rubric_fingerprint",
                "$.rubric_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "construct_id",
            _identifier_field(
                self.construct_id,
                "construct_id",
                "$.construct_id",
            ),
        )
        if not isinstance(self.response_format, ResponseFormat):
            raise contract_error(
                "invalid_response_format",
                "$.response_format",
                (
                    "response_format must be one of "
                    f"{[member.value for member in ResponseFormat]}"
                ),
            )
        object.__setattr__(
            self,
            "schema_version",
            _schema_field(self.schema_version, "$.schema_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact rubric identity bound into an assessment."""
        return {
            "schema_version": self.schema_version,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "response_format": self.response_format.value,
        }


@dataclass(frozen=True)
class AssessmentSpec:
    """Factory-sealed assessment bound to exact rubrics and policy revisions."""

    assessment_id: str
    assessment_version: str
    constructs: tuple[ConstructSpec, ...]
    rubric_bindings: tuple[RubricBinding, ...]
    policy_documents: tuple[PolicyDocument, ...]
    metadata_json: str
    schema_version: str = SCHEMA_VERSION
    _assessment_token: InitVar[object | None] = None

    def __post_init__(self, _assessment_token: object | None) -> None:
        """Reject direct construction and recheck cross-contract invariants."""
        if _assessment_token is not _ASSESSMENT_TOKEN:
            raise contract_error(
                "unverified_assessment_spec",
                "$",
                "use build_assessment_spec",
            )
        object.__setattr__(
            self,
            "assessment_id",
            _identifier_field(
                self.assessment_id,
                "assessment_id",
                "$.assessment_id",
            ),
        )
        object.__setattr__(
            self,
            "assessment_version",
            _version_field(
                self.assessment_version,
                "assessment_version",
                "$.assessment_version",
            ),
        )
        self._validate_constructs()
        self._validate_rubrics()
        self._validate_policies()
        metadata = decode_object_json(self.metadata_json, "metadata_json")
        if canonical_object_json(metadata, "metadata") != self.metadata_json:
            raise contract_error(
                "noncanonical_assessment_metadata",
                "$.metadata",
                "metadata must use canonical JSON",
            )
        object.__setattr__(
            self,
            "schema_version",
            _schema_field(self.schema_version, "$.schema_version"),
        )

    def _validate_constructs(self) -> None:
        """Validate canonical construct membership and order."""
        if not self.constructs or any(
            not isinstance(value, ConstructSpec)
            for value in self.constructs
        ):
            raise contract_error(
                "invalid_constructs",
                "$.constructs",
                "constructs must contain ConstructSpec values",
            )
        identifiers = tuple(value.construct_id for value in self.constructs)
        if len(set(identifiers)) != len(identifiers):
            raise contract_error(
                "duplicate_construct_id",
                "$.constructs",
                "construct identifiers must be unique",
            )
        if identifiers != tuple(sorted(identifiers)):
            raise contract_error(
                "noncanonical_construct_order",
                "$.constructs",
                "constructs must be ordered by construct_id",
            )

    def _validate_rubrics(self) -> None:
        """Validate canonical rubric identities and construct references."""
        if not self.rubric_bindings or any(
            not isinstance(value, RubricBinding)
            for value in self.rubric_bindings
        ):
            raise contract_error(
                "invalid_rubric_bindings",
                "$.rubric_bindings",
                "rubric bindings must be validated",
            )
        rubric_ids = tuple(value.rubric_id for value in self.rubric_bindings)
        if len(set(rubric_ids)) != len(rubric_ids):
            raise contract_error(
                "duplicate_rubric_id",
                "$.rubric_bindings",
                "rubric identifiers must be unique",
            )
        if rubric_ids != tuple(sorted(rubric_ids)):
            raise contract_error(
                "noncanonical_rubric_order",
                "$.rubric_bindings",
                "rubric bindings must be ordered by rubric_id",
            )
        declared = {value.construct_id for value in self.constructs}
        if any(
            value.construct_id not in declared
            for value in self.rubric_bindings
        ):
            raise contract_error(
                "undeclared_rubric_construct",
                "$.rubric_bindings",
                "every rubric construct must be declared",
            )

    def _validate_policies(self) -> None:
        """Require one canonical policy document for every policy kind."""
        if len(self.policy_documents) != len(PolicyKind) or any(
            not isinstance(value, PolicyDocument)
            for value in self.policy_documents
        ):
            raise contract_error(
                "invalid_policy_documents",
                "$.policy_documents",
                "exactly one validated policy is required for every policy kind",
            )
        kinds = tuple(value.policy_kind for value in self.policy_documents)
        if len(set(kinds)) != len(kinds):
            raise contract_error(
                "duplicate_policy_kind",
                "$.policy_documents",
                "policy kinds must be unique",
            )
        if kinds != tuple(PolicyKind):
            raise contract_error(
                "noncanonical_policy_order",
                "$.policy_documents",
                "policies must be ordered by PolicyKind",
            )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a fresh decoded copy of immutable assessment metadata."""
        return decode_object_json(self.metadata_json, "metadata_json")

    @property
    def rubric_ids(self) -> tuple[str, ...]:
        """Return rubric identifiers in canonical binding order."""
        return tuple(value.rubric_id for value in self.rubric_bindings)

    @property
    def rubric_fingerprints(self) -> tuple[str, ...]:
        """Return exact rubric fingerprints in canonical binding order."""
        return tuple(
            value.rubric_fingerprint for value in self.rubric_bindings
        )

    @property
    def policy_fingerprints(self) -> tuple[str, ...]:
        """Return policy fingerprints in canonical policy-kind order."""
        return tuple(
            value.policy_fingerprint for value in self.policy_documents
        )

    def policy(self, policy_kind: PolicyKind | str) -> PolicyDocument:
        """Return the policy document for one declared policy family."""
        resolved = _policy_kind(policy_kind)
        return self.policy_documents[tuple(PolicyKind).index(resolved)]

    def _content_dict(self) -> dict[str, Any]:
        """Return assessment content without derived identity fields."""
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "assessment_version": self.assessment_version,
            "constructs": [value.to_dict() for value in self.constructs],
            "rubric_bindings": [
                value.to_dict() for value in self.rubric_bindings
            ],
            "policy_documents": [
                value.to_dict() for value in self.policy_documents
            ],
            "metadata": self.metadata,
        }

    @property
    def assessment_fingerprint(self) -> str:
        """Return SHA-256 over the complete assessment contract content."""
        return _sha256_hex(self._content_dict())

    @property
    def assessment_handle(self) -> str:
        """Return a descriptive 128-bit public assessment handle."""
        return f"assessment_spec_{self.assessment_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical assessment content and deterministic identities."""
        return {
            **self._content_dict(),
            "assessment_handle": self.assessment_handle,
            "assessment_fingerprint": self.assessment_fingerprint,
        }

    def to_canonical_json(self) -> str:
        """Return byte-stable canonical JSON for the complete assessment."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def build_policy_document(
    *,
    policy_id: str,
    policy_version: str,
    policy_kind: PolicyKind | str,
    settings: Mapping[str, Any] | None = None,
) -> PolicyDocument:
    """Build one immutable content-addressed operational policy document."""
    return PolicyDocument(
        policy_id=policy_id,
        policy_version=policy_version,
        policy_kind=_policy_kind(policy_kind),
        settings_json=canonical_object_json(settings, "settings"),
        schema_version=SCHEMA_VERSION,
        _policy_token=_POLICY_TOKEN,
    )


def _rubric_binding(rubric: RubricSpecification) -> RubricBinding:
    """Issue an immutable binding from one validated rubric specification."""
    return RubricBinding(
        rubric_id=rubric.rubric_id,
        rubric_version=rubric.rubric_version,
        rubric_fingerprint=rubric.fingerprint,
        construct_id=rubric.construct_id,
        response_format=rubric.response_format,
        schema_version=rubric.schema_version,
        _binding_token=_RUBRIC_BINDING_TOKEN,
    )


def build_assessment_spec(
    *,
    assessment_id: str,
    assessment_version: str,
    constructs: Iterable[ConstructSpec],
    rubrics: Iterable[RubricSpecification],
    policy_documents: Iterable[PolicyDocument],
    metadata: Mapping[str, Any] | None = None,
) -> AssessmentSpec:
    """Build a deterministic assessment bound to exact rubrics and policies."""
    construct_values = _bounded_iterable(
        constructs,
        "constructs",
        1,
        MAX_CONSTRUCTS,
    )
    for index, value in enumerate(construct_values):
        if not isinstance(value, ConstructSpec):
            raise contract_error(
                "invalid_construct",
                f"$.constructs[{index}]",
                "expected ConstructSpec",
            )
    normalized_constructs = tuple(
        sorted(construct_values, key=lambda value: value.construct_id)
    )
    if len(
        {value.construct_id for value in normalized_constructs}
    ) != len(normalized_constructs):
        raise contract_error(
            "duplicate_construct_id",
            "$.constructs",
            "construct identifiers must be unique",
        )

    rubric_values = _bounded_iterable(
        rubrics,
        "rubrics",
        1,
        MAX_RUBRIC_BINDINGS,
    )
    for index, value in enumerate(rubric_values):
        if not isinstance(value, RubricSpecification):
            raise contract_error(
                "invalid_rubric",
                f"$.rubrics[{index}]",
                "expected RubricSpecification",
            )
    normalized_rubrics = tuple(
        sorted(rubric_values, key=lambda value: value.rubric_id)
    )
    rubric_ids = tuple(value.rubric_id for value in normalized_rubrics)
    if len(set(rubric_ids)) != len(rubric_ids):
        raise contract_error(
            "duplicate_rubric_id",
            "$.rubrics",
            "rubric identifiers must be unique",
        )
    declared = {value.construct_id for value in normalized_constructs}
    for index, value in enumerate(normalized_rubrics):
        if value.construct_id not in declared:
            raise contract_error(
                "undeclared_rubric_construct",
                f"$.rubrics[{index}].construct_id",
                "rubric construct is not declared by the assessment",
            )
    rubric_bindings = tuple(
        _rubric_binding(value) for value in normalized_rubrics
    )

    policies = _bounded_iterable(
        policy_documents,
        "policy_documents",
        1,
        len(PolicyKind),
    )
    for index, value in enumerate(policies):
        if not isinstance(value, PolicyDocument):
            raise contract_error(
                "invalid_policy_document",
                f"$.policy_documents[{index}]",
                "expected PolicyDocument",
            )
    by_kind: dict[PolicyKind, PolicyDocument] = {}
    for index, value in enumerate(policies):
        if value.policy_kind in by_kind:
            raise contract_error(
                "duplicate_policy_kind",
                f"$.policy_documents[{index}].policy_kind",
                "one policy is allowed per kind",
            )
        by_kind[value.policy_kind] = value
    missing = tuple(kind for kind in PolicyKind if kind not in by_kind)
    if missing:
        raise contract_error(
            "missing_policy_kind",
            "$.policy_documents",
            "one policy is required for every kind",
        )
    normalized_policies = tuple(by_kind[kind] for kind in PolicyKind)

    return AssessmentSpec(
        assessment_id=assessment_id,
        assessment_version=assessment_version,
        constructs=normalized_constructs,
        rubric_bindings=rubric_bindings,
        policy_documents=normalized_policies,
        metadata_json=canonical_object_json(metadata, "metadata"),
        schema_version=SCHEMA_VERSION,
        _assessment_token=_ASSESSMENT_TOKEN,
    )


__all__ = [
    "AssessmentSpec",
    "ConstructSpec",
    "MAX_CONSTRUCTS",
    "MAX_RUBRIC_BINDINGS",
    "PolicyDocument",
    "PolicyKind",
    "RubricBinding",
    "ScoringContractError",
    "build_assessment_spec",
    "build_policy_document",
]
