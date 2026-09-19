# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Transport-agnostic remote execution contracts for legal numerical split units.

Issue #2001 L4 slice: job envelopes and an in-process loopback executor for
independent split units that may later be dispatched across a heterogenous
worker pool. Valkey/Redis transport is intentionally out of scope here.

Legal remote split families (inventory comment-5742475833, table C):

- ``fit_restart`` — parallel EM/JMLE restarts within one fit
- ``scoring_person`` — embarrassingly parallel person scoring shards
- ``mc_replicate`` — bootstrap / Monte Carlo replicate units

Sequential or unsplittable families — standard errors and derivatives,
regression/contrast OLS, EM M-step iterations, FIPC, and two-tier GRM —
are rejected at envelope admission and must not be queued remotely.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import platform
import re
import socket
import time
import unicodedata
from typing import Protocol

REMOTE_EXEC_SCHEMA_VERSION = "1.0"
SEED_DERIVATION_RULE = "index_golden_ratio_v1"

# 64-bit golden-ratio step; matches ``bifactor_bootstrap.run_bifactor_bootstrap``.
INDEX_SEED_STEP = 0x9E37_79B9_7F4A_7C15

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)

_MAX_TEXT = 256
_MAX_RUN_ID = 128


class RemoteJobFamily(str, Enum):
    """Numerical families that admit index-derived-seed remote split units."""

    FIT_RESTART = "fit_restart"
    SCORING_PERSON = "scoring_person"
    MC_REPLICATE = "mc_replicate"


FORBIDDEN_REMOTE_JOB_FAMILIES: frozenset[str] = frozenset(
    {
        "se_derivatives",
        "regression_contrasts",
        "em_m_step",
        "fipc",
        "two_tier",
    }
)


class ExecutionFloatPath(str, Enum):
    """Declared floating-point path for one remote execution cohort."""

    F64 = "f64"
    F32 = "f32"


class RemoteJobDeliveryState(str, Enum):
    """Delivery outcome for one remote job unit."""

    COMPLETED = "completed"
    FAILED = "failed"


def _text(value: object, name: str, *, maximum: int = _MAX_TEXT) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return normalized


def _fingerprint(value: object, name: str) -> str:
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _semantic_version(value: object, name: str) -> str:
    normalized = _text(value, name, maximum=64)
    if _SEMANTIC_VERSION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a canonical semantic version")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _enum_value(value: object, enum_type: type[Enum], name: str) -> Enum:
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise ValueError(f"{name} must be a supported {enum_type.__name__} value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = [member.value for member in enum_type]
        raise ValueError(f"{name} must be one of {choices}") from exc


def derive_index_seed(base_seed: int, unit_index: int) -> int:
    """Return the deterministic unit seed from a master seed and split index.

    The rule matches ``run_bifactor_bootstrap`` replicate seeding so remote
    MC/bootstrap workers reproduce in-process reference draws bit-for-bit on
    the same float path.
    """
    base = _non_negative_int(base_seed, "base_seed")
    index = _non_negative_int(unit_index, "unit_index")
    return int((base + index * INDEX_SEED_STEP) & 0xFFFF_FFFF_FFFF_FFFF)


def admit_remote_job_family(value: object) -> RemoteJobFamily:
    """Normalize one job family and reject unsplittable families fail-closed."""
    if type(value) is RemoteJobFamily:
        return value
    if type(value) is not str:
        raise ValueError("job family must be a RemoteJobFamily or string")
    normalized = _text(value, "job family", maximum=64).lower()
    if normalized in FORBIDDEN_REMOTE_JOB_FAMILIES:
        raise ValueError(
            f"job family {normalized!r} is not a legal remote split unit "
            f"(forbidden: {sorted(FORBIDDEN_REMOTE_JOB_FAMILIES)})"
        )
    try:
        return RemoteJobFamily(normalized)
    except ValueError as exc:
        legal = [member.value for member in RemoteJobFamily]
        raise ValueError(
            f"job family {normalized!r} must be one of {legal}"
        ) from exc


@dataclass(frozen=True, slots=True)
class RemoteRunManifest:
    """Fail-closed cohort identity shared by driver and worker."""

    schema_version: str
    library_version: str
    source_sha256: str
    seed_derivation_rule: str
    float_path: ExecutionFloatPath
    payload_sha256: str
    integration_nodes_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self) is not RemoteRunManifest:
            raise ValueError("RemoteRunManifest must be an exact package record")
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version", maximum=16),
        )
        if self.schema_version != REMOTE_EXEC_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be '{REMOTE_EXEC_SCHEMA_VERSION}'"
            )
        object.__setattr__(
            self,
            "library_version",
            _semantic_version(self.library_version, "library_version"),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _fingerprint(self.source_sha256, "source_sha256"),
        )
        object.__setattr__(
            self,
            "seed_derivation_rule",
            _text(self.seed_derivation_rule, "seed_derivation_rule", maximum=64),
        )
        if self.seed_derivation_rule != SEED_DERIVATION_RULE:
            raise ValueError(
                f"seed_derivation_rule must be '{SEED_DERIVATION_RULE}'"
            )
        object.__setattr__(
            self,
            "float_path",
            _enum_value(self.float_path, ExecutionFloatPath, "float_path"),
        )
        object.__setattr__(
            self,
            "payload_sha256",
            _fingerprint(self.payload_sha256, "payload_sha256"),
        )
        if self.integration_nodes_sha256 is not None:
            object.__setattr__(
                self,
                "integration_nodes_sha256",
                _fingerprint(
                    self.integration_nodes_sha256,
                    "integration_nodes_sha256",
                ),
            )

    def cohort_key(self) -> tuple[str, ...]:
        """Return the immutable fields that must match across one remote run."""
        return (
            self.schema_version,
            self.library_version,
            self.source_sha256,
            self.seed_derivation_rule,
            self.float_path.value,
            self.payload_sha256,
            self.integration_nodes_sha256 or "",
        )

    def compatible_with(self, other: RemoteRunManifest) -> bool:
        """Return whether ``other`` may join the same remote cohort."""
        if type(other) is not RemoteRunManifest:
            return False
        return self.cohort_key() == other.cohort_key()

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-compatible manifest mapping."""
        return {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "source_sha256": self.source_sha256,
            "seed_derivation_rule": self.seed_derivation_rule,
            "float_path": self.float_path.value,
            "payload_sha256": self.payload_sha256,
            "integration_nodes_sha256": self.integration_nodes_sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> RemoteRunManifest:
        """Build one manifest from a JSON-compatible mapping."""
        if type(payload) is not dict:
            raise ValueError("manifest payload must be a mapping")
        return cls(
            schema_version=payload["schema_version"],
            library_version=payload["library_version"],
            source_sha256=payload["source_sha256"],
            seed_derivation_rule=payload["seed_derivation_rule"],
            float_path=payload["float_path"],
            payload_sha256=payload["payload_sha256"],
            integration_nodes_sha256=payload.get("integration_nodes_sha256"),
        )


@dataclass(frozen=True, slots=True)
class RemoteJobEnvelope:
    """One independently schedulable remote numerical unit."""

    run_id: str
    family: RemoteJobFamily
    unit_index: int
    base_seed: int
    payload_ref: str
    manifest: RemoteRunManifest

    def __post_init__(self) -> None:
        if type(self) is not RemoteJobEnvelope:
            raise ValueError("RemoteJobEnvelope must be an exact package record")
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", maximum=_MAX_RUN_ID))
        object.__setattr__(
            self,
            "family",
            admit_remote_job_family(self.family),
        )
        object.__setattr__(self, "unit_index", _non_negative_int(self.unit_index, "unit_index"))
        object.__setattr__(self, "base_seed", _non_negative_int(self.base_seed, "base_seed"))
        object.__setattr__(
            self,
            "payload_ref",
            _fingerprint(self.payload_ref, "payload_ref"),
        )
        if type(self.manifest) is not RemoteRunManifest:
            raise ValueError("manifest must be a RemoteRunManifest")
        if self.payload_ref != self.manifest.payload_sha256:
            raise ValueError(
                "payload_ref must equal manifest.payload_sha256 for one cohort"
            )

    def unit_seed(self) -> int:
        """Return the index-derived seed for this envelope."""
        return derive_index_seed(self.base_seed, self.unit_index)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible envelope mapping."""
        return {
            "run_id": self.run_id,
            "family": self.family.value,
            "unit_index": self.unit_index,
            "base_seed": self.base_seed,
            "payload_ref": self.payload_ref,
            "manifest": self.manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> RemoteJobEnvelope:
        """Build one envelope from a JSON-compatible mapping."""
        if type(payload) is not dict:
            raise ValueError("envelope payload must be a mapping")
        return cls(
            run_id=payload["run_id"],
            family=payload["family"],
            unit_index=payload["unit_index"],
            base_seed=payload["base_seed"],
            payload_ref=payload["payload_ref"],
            manifest=RemoteRunManifest.from_dict(payload["manifest"]),
        )


@dataclass(frozen=True, slots=True)
class RemoteWorkerProvenance:
    """Per-unit worker environment recorded with each remote outcome."""

    hostname: str
    architecture: str
    operating_system: str
    library_version: str
    source_sha256: str
    requested_device: str
    effective_device: str
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        if type(self) is not RemoteWorkerProvenance:
            raise ValueError("RemoteWorkerProvenance must be an exact package record")
        object.__setattr__(self, "hostname", _text(self.hostname, "hostname", maximum=128))
        object.__setattr__(
            self,
            "architecture",
            _text(self.architecture, "architecture", maximum=128),
        )
        object.__setattr__(
            self,
            "operating_system",
            _text(self.operating_system, "operating_system", maximum=128),
        )
        object.__setattr__(
            self,
            "library_version",
            _semantic_version(self.library_version, "library_version"),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _fingerprint(self.source_sha256, "source_sha256"),
        )
        object.__setattr__(
            self,
            "requested_device",
            _text(self.requested_device, "requested_device", maximum=32),
        )
        object.__setattr__(
            self,
            "effective_device",
            _text(self.effective_device, "effective_device", maximum=32),
        )
        if (
            type(self.wall_clock_seconds) is not float
            or self.wall_clock_seconds < 0.0
            or not (self.wall_clock_seconds < float("inf"))
        ):
            raise ValueError("wall_clock_seconds must be a finite non-negative float")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible provenance mapping."""
        return {
            "hostname": self.hostname,
            "architecture": self.architecture,
            "operating_system": self.operating_system,
            "library_version": self.library_version,
            "source_sha256": self.source_sha256,
            "requested_device": self.requested_device,
            "effective_device": self.effective_device,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


@dataclass(frozen=True, slots=True)
class RemoteJobOutcome:
    """One completed or failed remote unit with worker provenance."""

    run_id: str
    unit_index: int
    unit_seed: int
    family: RemoteJobFamily
    delivery_state: RemoteJobDeliveryState
    result: object | None
    error_message: str | None
    provenance: RemoteWorkerProvenance

    def __post_init__(self) -> None:
        if type(self) is not RemoteJobOutcome:
            raise ValueError("RemoteJobOutcome must be an exact package record")
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", maximum=_MAX_RUN_ID))
        object.__setattr__(self, "unit_index", _non_negative_int(self.unit_index, "unit_index"))
        object.__setattr__(self, "unit_seed", _non_negative_int(self.unit_seed, "unit_seed"))
        object.__setattr__(
            self,
            "family",
            admit_remote_job_family(self.family),
        )
        object.__setattr__(
            self,
            "delivery_state",
            _enum_value(
                self.delivery_state,
                RemoteJobDeliveryState,
                "delivery_state",
            ),
        )
        if type(self.provenance) is not RemoteWorkerProvenance:
            raise ValueError("provenance must be a RemoteWorkerProvenance")
        if self.delivery_state is RemoteJobDeliveryState.COMPLETED:
            if self.error_message is not None:
                raise ValueError("completed outcomes must not carry error_message")
        elif self.delivery_state is RemoteJobDeliveryState.FAILED:
            if type(self.error_message) is not str or not self.error_message.strip():
                raise ValueError("failed outcomes require a non-empty error_message")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible outcome mapping."""
        return {
            "run_id": self.run_id,
            "unit_index": self.unit_index,
            "unit_seed": self.unit_seed,
            "family": self.family.value,
            "delivery_state": self.delivery_state.value,
            "result": self.result,
            "error_message": self.error_message,
            "provenance": self.provenance.to_dict(),
        }


class RemoteExecutionBackend(Protocol):
    """Transport-agnostic batch executor for remote numerical units."""

    def run_batch(
        self,
        envelopes: Sequence[RemoteJobEnvelope],
        handler: Callable[[RemoteJobEnvelope, int], object],
        *,
        worker_manifest: RemoteRunManifest,
        requested_device: str = "cpu",
        effective_device: str = "cpu",
    ) -> tuple[RemoteJobOutcome, ...]:
        """Execute ``envelopes`` and return one outcome per unit in index order."""


class CohortMismatchError(ValueError):
    """Raised when a worker manifest fails the fail-closed cohort gate."""


def local_worker_provenance(
    manifest: RemoteRunManifest,
    *,
    requested_device: str,
    effective_device: str,
    wall_clock_seconds: float,
) -> RemoteWorkerProvenance:
    """Build provenance for the current interpreter process."""
    return RemoteWorkerProvenance(
        hostname=socket.gethostname(),
        architecture=platform.machine(),
        operating_system=platform.system(),
        library_version=manifest.library_version,
        source_sha256=manifest.source_sha256,
        requested_device=requested_device,
        effective_device=effective_device,
        wall_clock_seconds=wall_clock_seconds,
    )


def envelope_fingerprint(envelope: RemoteJobEnvelope) -> str:
    """Return a stable SHA-256 over the envelope JSON (idempotency key material)."""
    return hashlib.sha256(
        json.dumps(
            envelope.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class LoopbackExecutor:
    """In-process L4 stand-in that validates cohort identity and runs handlers."""

    def run_batch(
        self,
        envelopes: Sequence[RemoteJobEnvelope],
        handler: Callable[[RemoteJobEnvelope, int], object],
        *,
        worker_manifest: RemoteRunManifest,
        requested_device: str = "cpu",
        effective_device: str = "cpu",
    ) -> tuple[RemoteJobOutcome, ...]:
        if not isinstance(envelopes, Sequence):
            raise TypeError("envelopes must be a sequence")
        outcomes: list[RemoteJobOutcome] = []
        for envelope in envelopes:
            if type(envelope) is not RemoteJobEnvelope:
                raise TypeError("each envelope must be a RemoteJobEnvelope")
            if not worker_manifest.compatible_with(envelope.manifest):
                raise CohortMismatchError(
                    "worker manifest is incompatible with envelope cohort "
                    f"(run_id={envelope.run_id!r}, unit_index={envelope.unit_index})"
                )
            unit_seed = envelope.unit_seed()
            started = time.perf_counter()
            try:
                result = handler(envelope, unit_seed)
            except Exception as exc:  # handler failures are recorded, not raised
                elapsed = time.perf_counter() - started
                outcomes.append(
                    RemoteJobOutcome(
                        run_id=envelope.run_id,
                        unit_index=envelope.unit_index,
                        unit_seed=unit_seed,
                        family=envelope.family,
                        delivery_state=RemoteJobDeliveryState.FAILED,
                        result=None,
                        error_message=str(exc),
                        provenance=local_worker_provenance(
                            worker_manifest,
                            requested_device=requested_device,
                            effective_device=effective_device,
                            wall_clock_seconds=elapsed,
                        ),
                    )
                )
                continue
            elapsed = time.perf_counter() - started
            outcomes.append(
                RemoteJobOutcome(
                    run_id=envelope.run_id,
                    unit_index=envelope.unit_index,
                    unit_seed=unit_seed,
                    family=envelope.family,
                    delivery_state=RemoteJobDeliveryState.COMPLETED,
                    result=result,
                    error_message=None,
                    provenance=local_worker_provenance(
                        worker_manifest,
                        requested_device=requested_device,
                        effective_device=effective_device,
                        wall_clock_seconds=elapsed,
                    ),
                )
            )
        return tuple(outcomes)
