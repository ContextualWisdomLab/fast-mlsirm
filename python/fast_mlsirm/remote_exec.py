# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Transport-agnostic remote execution contracts for legal numerical split units.

Issue #2001 L4 slice: job envelopes and an in-process loopback executor for
independent split units that may later be dispatched across a heterogenous
worker pool. The included ``ValkeyStreamsOutcomeStore`` and
``ValkeyStreamsBackend`` provide injected-client transport contracts. Source
and recording-client validation does not certify real-server operation,
installed artifacts, cross-host execution, measured devices or scientific fit.

Legal remote split families (inventory comment-5742475833, table C):

- ``fit_restart`` — parallel EM/JMLE restarts within one fit
- ``scoring_person`` — embarrassingly parallel person scoring shards
- ``mc_replicate`` — bootstrap / Monte Carlo replicate units

Sequential families — standard errors and derivatives, regression/contrast
OLS, EM M-step iterations, FIPC, and two-tier GRM — may run remotely as one
complete call but must not be partitioned across index-derived split units.
``admit_remote_job_internal_shard`` call sites:

- ``LoopbackExecutor.run_batch`` batch preflight
- ``SubprocessExecutor.run_batch`` batch preflight
- ``ValkeyStreamsBackend.run_batch`` through ``_admit_payload_batch`` preflight
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
import platform
import math
import re
import shlex
import socket
import sqlite3
import subprocess
import sys
import tempfile
import signal
import time
import unicodedata
from pathlib import Path
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
    """Numerical families that admit whole-call remote placement."""

    FIT_RESTART = "fit_restart"
    SCORING_PERSON = "scoring_person"
    MC_REPLICATE = "mc_replicate"
    SE_DERIVATIVES = "se_derivatives"
    REGRESSION_CONTRASTS = "regression_contrasts"
    EM_M_STEP = "em_m_step"
    FIPC = "fipc"
    TWO_TIER = "two_tier"
    BIFACTOR_BOOTSTRAP_REPLICATE = "bifactor_bootstrap_replicate"


# These families may run remotely as complete calls. Only partitioning their
# sequential internals across workers is disallowed.
INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES: frozenset[str] = frozenset(
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
    """Normalize a family for placement of one complete call on a remote worker."""
    if type(value) is RemoteJobFamily:
        return value
    if type(value) is not str:
        raise ValueError("job family must be a RemoteJobFamily or string")
    normalized = _text(value, "job family", maximum=64).lower()
    try:
        return RemoteJobFamily(normalized)
    except ValueError as exc:
        legal = [member.value for member in RemoteJobFamily]
        raise ValueError(
            f"job family {normalized!r} must be one of {legal}"
        ) from exc


def admit_remote_job_internal_shard(value: object) -> RemoteJobFamily:
    """Reject families whose sequential internals must remain inside one call."""
    family = admit_remote_job_family(value)
    if family.value in INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES:
        raise ValueError(
            f"job family {family.value!r} cannot be sharded inside one call"
        )
    return family


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
    """Per-unit environment and declared, unverified source/device identity.

    ``source_sha256`` and device fields are caller declarations, not measured
    installed-artifact or execution-path attestation. Legacy records retain
    that limitation when restored without ``identity_verification``.
    """

    hostname: str
    architecture: str
    operating_system: str
    library_version: str
    source_sha256: str
    requested_device: str
    effective_device: str
    wall_clock_seconds: float
    worker_host: str
    worker_pid: int
    cross_host_execution: bool
    identity_verification: str = "declared_unverified"

    def __post_init__(self) -> None:
        if type(self.identity_verification) is not str or self.identity_verification != "declared_unverified":
            raise ValueError("identity_verification must be declared_unverified")
        if type(self) is not RemoteWorkerProvenance:
            raise ValueError("RemoteWorkerProvenance must be an exact package record")
        object.__setattr__(self, "hostname", _text(self.hostname, "hostname", maximum=128))
        object.__setattr__(
            self,
            "worker_host",
            _text(self.worker_host, "worker_host", maximum=128),
        )
        if type(self.worker_pid) is not int or isinstance(self.worker_pid, bool) or self.worker_pid <= 0:
            raise ValueError("worker_pid must be a positive integer")
        if type(self.cross_host_execution) is not bool:
            raise ValueError("cross_host_execution must be a boolean")
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
            "worker_host": self.worker_host,
            "worker_pid": self.worker_pid,
            "cross_host_execution": self.cross_host_execution,
            "identity_verification": self.identity_verification,
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
    input_identity_sha256: str
    output_identity_sha256: str | None
    envelope_fingerprint: str
    driver_host: str
    driver_pid: int

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
        object.__setattr__(
            self,
            "input_identity_sha256",
            _fingerprint(self.input_identity_sha256, "input_identity_sha256"),
        )
        object.__setattr__(
            self,
            "envelope_fingerprint",
            _fingerprint(self.envelope_fingerprint, "envelope_fingerprint"),
        )
        if self.output_identity_sha256 is not None:
            object.__setattr__(
                self,
                "output_identity_sha256",
                _fingerprint(self.output_identity_sha256, "output_identity_sha256"),
            )
        object.__setattr__(self, "driver_host", _text(self.driver_host, "driver_host", maximum=128))
        if type(self.driver_pid) is not int or isinstance(self.driver_pid, bool) or self.driver_pid <= 0:
            raise ValueError("driver_pid must be a positive integer")
        if type(self.provenance) is not RemoteWorkerProvenance:
            raise ValueError("provenance must be a RemoteWorkerProvenance")
        if self.delivery_state is RemoteJobDeliveryState.COMPLETED:
            if self.error_message is not None:
                raise ValueError("completed outcomes must not carry error_message")
            if self.output_identity_sha256 is None:
                raise ValueError("completed outcomes require output_identity_sha256")
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
            "input_identity_sha256": self.input_identity_sha256,
            "output_identity_sha256": self.output_identity_sha256,
            "envelope_fingerprint": self.envelope_fingerprint,
            "driver_host": self.driver_host,
            "driver_pid": self.driver_pid,
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


def _admit_remote_device_declarations(requested_device: object, effective_device: object) -> None:
    """Limit transport labels until a measured device-readback contract exists."""
    if (
        type(requested_device) is not str or requested_device != "cpu"
        or type(effective_device) is not str or effective_device != "cpu"
    ):
        raise ValueError("remote transport supports only cpu device declarations; actual device is unverified")


def local_worker_provenance(
    manifest: RemoteRunManifest,
    *,
    requested_device: str,
    effective_device: str,
    wall_clock_seconds: float,
    worker_host: str | None = None,
    worker_pid: int | None = None,
    cross_host_execution: bool = False,
) -> RemoteWorkerProvenance:
    """Record local process details with unverified source/device declarations."""
    host = worker_host or socket.gethostname()
    pid = worker_pid if worker_pid is not None else os.getpid()
    return RemoteWorkerProvenance(
        hostname=socket.gethostname(),
        architecture=platform.machine(),
        operating_system=platform.system(),
        library_version=manifest.library_version,
        source_sha256=manifest.source_sha256,
        requested_device=requested_device,
        effective_device=effective_device,
        wall_clock_seconds=wall_clock_seconds,
        worker_host=host,
        worker_pid=pid,
        cross_host_execution=cross_host_execution,
    )


def result_identity_sha256(result: object) -> str:
    """Return a stable SHA-256 digest for one finite JSON remote result.

    Nonfinite number literals are rejected rather than hashed as successful
    output (Bray, 2017, Section 6, p. 7; Section 10, p. 10). Finite output
    retains the existing canonical encoding. Python's permissive default is
    overridden (Python Software Foundation, n.d., "Infinite and NaN Number
    Values"). This validates transport, not scientific convergence.

    References:
        Bray, T. (Ed.). (2017). The JavaScript Object Notation (JSON) data
            interchange format (RFC 8259). Internet Engineering Task Force.
        Python Software Foundation. (n.d.). json—JSON encoder and decoder.
            Python 3.14 documentation.
    """
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class OutcomeCommitLedger:
    """At-most-once successful outcome commit keyed by envelope fingerprint."""

    def __init__(self) -> None:
        self._successful: dict[str, RemoteJobOutcome] = {}

    def successful_count(self, fingerprint: str) -> int:
        """Return how many successful outcomes were committed for ``fingerprint``."""
        return 1 if fingerprint in self._successful else 0

    def committed_success(self, fingerprint: str) -> RemoteJobOutcome | None:
        """Return a winner only after revalidating its mutable result.

        Finite JSON is required (Bray, 2017, Section 6, p. 7; Section 10,
        p. 10), using the existing strict decoder rather than a new encoding
        (Python Software Foundation, n.d., "JSONEncoder", ``allow_nan``).

        References:
            Bray, T. (Ed.). (2017). The JavaScript Object Notation (JSON) data
                interchange format (RFC 8259). Internet Engineering Task Force.
            Python Software Foundation. (n.d.). json—JSON encoder and decoder.
                Python 3.14 documentation.
        """
        existing = self._successful.get(fingerprint)
        if existing is not None:
            _successful_outcome_from_dict(existing.to_dict(), fingerprint=fingerprint)
        return existing

    def commit_success(self, fingerprint: str, outcome: RemoteJobOutcome) -> RemoteJobOutcome:
        """Validate the candidate and any prior winner before commit or reuse.

        Finite JSON is required (Bray, 2017, Section 6, p. 7; Section 10,
        p. 10), using the existing strict decoder rather than a new encoding
        (Python Software Foundation, n.d., "JSONEncoder", ``allow_nan``).

        References:
            Bray, T. (Ed.). (2017). The JavaScript Object Notation (JSON) data
                interchange format (RFC 8259). Internet Engineering Task Force.
            Python Software Foundation. (n.d.). json—JSON encoder and decoder.
                Python 3.14 documentation.
        """
        if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
            raise ValueError("commit_success requires a completed outcome")
        if outcome.envelope_fingerprint != fingerprint:
            raise ValueError("outcome fingerprint does not match commit key")
        _successful_outcome_from_dict(outcome.to_dict(), fingerprint=fingerprint)
        existing = self.committed_success(fingerprint)
        if existing is not None:
            return existing
        self._successful[fingerprint] = outcome
        return outcome


class OutcomeCommitStore(Protocol):
    """Successful-outcome store shared by remote execution backends."""

    def successful_count(self, fingerprint: str) -> int: ...

    def committed_success(self, fingerprint: str) -> RemoteJobOutcome | None: ...

    def commit_success(self, fingerprint: str, outcome: RemoteJobOutcome) -> RemoteJobOutcome: ...


class SQLiteOutcomeCommitLedger:
    """Durable, atomic successful-outcome ledger backed by SQLite.

    This deduplicates successful outcome commits; workers may still compute the
    same envelope more than once. A shared database file coordinates processes
    on one host, not workers on multiple hosts.

    :class:`ValkeyStreamsOutcomeStore` implements the same store contract for
    the Valkey Streams backend. SQLite remains the local durable adapter.
    """

    def __init__(self, database: str | Path) -> None:
        self._database = Path(database)
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS successful_outcomes "
                "(fingerprint TEXT PRIMARY KEY, outcome_json TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database, timeout=30.0)

    def successful_count(self, fingerprint: str) -> int:
        """Return whether one successful outcome is durably committed."""
        key = _fingerprint(fingerprint, "fingerprint")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM successful_outcomes WHERE fingerprint = ?",
                (key,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def committed_success(self, fingerprint: str) -> RemoteJobOutcome | None:
        """Return the durable successful outcome for ``fingerprint``, if any."""
        key = _fingerprint(fingerprint, "fingerprint")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT outcome_json FROM successful_outcomes WHERE fingerprint = ?",
                (key,),
            ).fetchone()
        return None if row is None else _successful_outcome_from_dict(json.loads(row[0]), fingerprint=key)

    def commit_success(self, fingerprint: str, outcome: RemoteJobOutcome) -> RemoteJobOutcome:
        """Atomically commit the first success and return the durable winner."""
        key = _fingerprint(fingerprint, "fingerprint")
        if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
            raise ValueError("commit_success requires a completed outcome")
        if outcome.envelope_fingerprint != key:
            raise ValueError("outcome fingerprint does not match commit key")
        # Validate the candidate before INSERT OR IGNORE: otherwise invalid new
        # results can be stored, or hidden by an existing valid winner.
        _outcome_from_dict(outcome.to_dict(), fingerprint=key)
        payload = json.dumps(
            outcome.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO successful_outcomes (fingerprint, outcome_json) "
                "VALUES (?, ?)",
                (key, payload),
            )
            row = connection.execute(
                "SELECT outcome_json FROM successful_outcomes WHERE fingerprint = ?",
                (key,),
            ).fetchone()
        if row is None:  # pragma: no cover - SQLite transaction invariant
            raise RuntimeError("successful outcome commit was not persisted")
        return _successful_outcome_from_dict(json.loads(row[0]), fingerprint=key)


def _successful_outcome_from_dict(payload: object, *, fingerprint: str) -> RemoteJobOutcome:
    """Restore a completed ledger winner without narrowing generic outcome decoding."""
    outcome = _outcome_from_dict(payload, fingerprint=fingerprint)
    if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
        raise ValueError("stored successful outcome must be completed")
    return outcome


def _outcome_from_dict(payload: object, *, fingerprint: str) -> RemoteJobOutcome:
    """Restore durable JSON only after checking its key and result identity.

    Completed results retain the package's canonical digest encoding, with
    nonfinite numbers rejected (Bray, 2017, Section 6, p. 7; Section 10,
    p. 10). Python's permissive encoding is explicitly disabled (Python
    Software Foundation, n.d., "JSONEncoder", ``allow_nan`` parameter).
    Digest consistency does not establish worker attestation or convergence.

    References:
        Bray, T. (Ed.). (2017). The JavaScript Object Notation (JSON) data
            interchange format (RFC 8259). Internet Engineering Task Force.
        Python Software Foundation. (n.d.). json—JSON encoder and decoder.
            Python 3.14 documentation.
    """
    if type(payload) is not dict or type(payload.get("provenance")) is not dict:
        raise ValueError("stored outcome must be a package outcome mapping")
    if payload.get("envelope_fingerprint") != fingerprint:
        raise ValueError("stored outcome fingerprint does not match commit key")
    if payload.get("delivery_state") == RemoteJobDeliveryState.COMPLETED.value:
        try:
            encoded_result = json.dumps(
                payload.get("result"), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("stored completed result is not finite JSON") from exc
        if payload.get("output_identity_sha256") != hashlib.sha256(encoded_result).hexdigest():
            raise ValueError("stored completed result does not match output identity")
    provenance = payload["provenance"]
    return RemoteJobOutcome(
        run_id=payload["run_id"],
        unit_index=payload["unit_index"],
        unit_seed=payload["unit_seed"],
        family=payload["family"],
        delivery_state=payload["delivery_state"],
        result=payload["result"],
        error_message=payload["error_message"],
        provenance=RemoteWorkerProvenance(**provenance),
        input_identity_sha256=payload["input_identity_sha256"],
        output_identity_sha256=payload["output_identity_sha256"],
        envelope_fingerprint=payload["envelope_fingerprint"],
        driver_host=payload["driver_host"],
        driver_pid=payload["driver_pid"],
    )


def _admit_payload_batch(
    envelopes: Sequence[RemoteJobEnvelope],
    worker_manifest: RemoteRunManifest,
    payload: Mapping[str, object] | None,
) -> tuple[RemoteJobEnvelope, ...]:
    """Fail closed on envelope type, shard, cohort, and payload identity before dispatch."""
    envelope_batch = tuple(envelopes)
    for envelope in envelope_batch:
        if type(envelope) is not RemoteJobEnvelope:
            raise TypeError("each envelope must be a RemoteJobEnvelope")
    _preflight_internal_shard_batch(envelope_batch)
    payload_sha256 = None if payload is None else payload_identity_sha256(payload)
    for envelope in envelope_batch:
        if not worker_manifest.compatible_with(envelope.manifest):
            raise CohortMismatchError(
                "worker manifest is incompatible with envelope cohort "
                f"(run_id={envelope.run_id!r}, unit_index={envelope.unit_index})"
            )
        if payload is None:
            raise ValueError(f"payload is required for {envelope.family.value}")
        if payload_sha256 != envelope.manifest.payload_sha256:
            raise CohortMismatchError(
                "payload identity is incompatible with envelope cohort "
                f"(run_id={envelope.run_id!r}, unit_index={envelope.unit_index})"
            )
    return envelope_batch


class RemoteDispatchBackend(Protocol):
    """Transport executor that ships envelopes and the verified payload to workers.

    Unlike :class:`RemoteExecutionBackend`, the handler is not a caller
    argument: the worker selects it from the envelope family, so the payload
    whose identity matches ``manifest.payload_sha256`` travels instead.
    """

    def run_batch(
        self,
        envelopes: Sequence[RemoteJobEnvelope],
        *,
        worker_manifest: RemoteRunManifest,
        requested_device: str = "cpu",
        effective_device: str = "cpu",
        payload: Mapping[str, object] | None = None,
    ) -> tuple[RemoteJobOutcome, ...]:
        """Dispatch ``envelopes`` and return one outcome per unit in index order."""


def _valkey_text(value: object) -> str:
    """Normalize redis-py text responses without accepting other coercions."""
    if type(value) is bytes:
        return value.decode("utf-8")
    if type(value) is str:
        return value
    raise ValueError("Valkey stream fields must be UTF-8 text")


_MAX_VALKEY_OUTCOME_JSON_BYTES = 1_048_576


_MAX_VALKEY_OUTCOME_JSON_DEPTH = 64


def _json_nesting_exceeds(text: str, limit: int) -> bool:
    """Return whether JSON ``text`` nests arrays/objects deeper than ``limit``.

    Decoder recursion limits depend on the platform stack, so depth is checked
    explicitly before ``json.loads``. Brackets inside strings are ignored.
    """
    if text.count("[") + text.count("{") <= limit:
        return False
    depth = 0
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > limit:
                return True
        elif char in "]}":
            depth -= 1
    return False


def _valkey_outcome(raw: object, *, fingerprint: str) -> RemoteJobOutcome:
    """Decode one Valkey outcome under the worker stdout size and depth bounds."""
    text = _valkey_text(raw)
    if len(text.encode("utf-8")) > _MAX_VALKEY_OUTCOME_JSON_BYTES:
        raise ValueError(
            f"Valkey outcome JSON exceeds {_MAX_VALKEY_OUTCOME_JSON_BYTES} bytes"
        )
    if _json_nesting_exceeds(text, _MAX_VALKEY_OUTCOME_JSON_DEPTH):
        raise ValueError(
            f"Valkey outcome JSON nests deeper than {_MAX_VALKEY_OUTCOME_JSON_DEPTH}"
        )
    return _outcome_from_dict(json.loads(text), fingerprint=_fingerprint(fingerprint, "fingerprint"))


def _valkey_successful_outcome(raw: object, *, fingerprint: str) -> RemoteJobOutcome:
    """Apply bounded Valkey decoding and the committed-success state invariant."""
    outcome = _valkey_outcome(raw, fingerprint=fingerprint)
    if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
        raise ValueError("stored success requires a completed outcome")
    return outcome


class ValkeyStreamsOutcomeStore:
    """Outcome store using a Valkey consumer group and pending-entry reclaim.

    The injected client follows redis-py's synchronous Streams API. This keeps
    the core package dependency-free while allowing either ``redis`` or
    ``valkey`` clients at the deployment boundary. Stream delivery is separate
    from the durable ``{stream}:committed`` hash, which mirrors the SQLite
    first-success contract across process restarts and consumer-group members.
    Failed outcomes are terminal too: the first one per fingerprint lands in
    ``{stream}:failed`` so callers receive it instead of waiting for a timeout,
    while a later success still wins.
    """

    def __init__(
        self,
        client: object,
        *,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int = 60_000,
        batch_size: int = 100,
        block_ms: int = 1_000,
    ) -> None:
        self._client = client
        self._stream = _text(stream, "stream")
        self._committed_key = f"{self._stream}:committed"
        self._failed_key = f"{self._stream}:failed"
        self._group = _text(group, "group")
        self._consumer = _text(consumer, "consumer")
        self._min_idle_ms = _non_negative_int(min_idle_ms, "min_idle_ms")
        self._batch_size = _non_negative_int(batch_size, "batch_size")
        self._block_ms = _non_negative_int(block_ms, "block_ms")
        if self._batch_size == 0:
            raise ValueError("batch_size must be > 0")
        try:
            client.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    @staticmethod
    def _serialize_outcome(outcome: RemoteJobOutcome) -> str:
        return json.dumps(
            outcome.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _committed_outcome(self, fingerprint: str) -> RemoteJobOutcome | None:
        key = _fingerprint(fingerprint, "fingerprint")
        stored = self._client.hget(self._committed_key, key)
        if stored is None:
            return None
        return _valkey_successful_outcome(stored, fingerprint=key)

    def _terminal_outcome(self, fingerprint: str) -> RemoteJobOutcome | None:
        """Return the committed success, else the first recorded failure."""
        committed = self._committed_outcome(fingerprint)
        if committed is not None:
            return committed
        stored = self._client.hget(self._failed_key, _fingerprint(fingerprint, "fingerprint"))
        return None if stored is None else _valkey_outcome(stored, fingerprint=_fingerprint(fingerprint, "fingerprint"))

    def _persist_committed(
        self, fingerprint: str, outcome: RemoteJobOutcome
    ) -> RemoteJobOutcome:
        key = _fingerprint(fingerprint, "fingerprint")
        _successful_outcome_from_dict(outcome.to_dict(), fingerprint=key)
        payload = self._serialize_outcome(outcome)
        _valkey_successful_outcome(payload, fingerprint=key)
        self._client.hsetnx(self._committed_key, key, payload)
        stored = self._client.hget(self._committed_key, key)
        if stored is None:  # pragma: no cover - hash write invariant
            raise RuntimeError("successful outcome commit was not persisted")
        return _valkey_successful_outcome(stored, fingerprint=key)

    def _block_ms_for_read(self, deadline: float | None) -> int | None:
        """Return BLOCK milliseconds, or ``None`` to omit BLOCK (non-blocking).

        Redis/Valkey treat ``BLOCK 0`` as wait-forever. Non-blocking reads must
        omit the BLOCK option entirely; blocking reads must be capped to any
        remaining wait deadline.
        """
        if self._block_ms == 0:
            return None
        if deadline is None:
            return self._block_ms
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            return None
        return min(self._block_ms, remaining_ms)

    def _xreadgroup(self, *, count: int, block_ms: int | None):
        """Issue XREADGROUP; omit BLOCK when ``block_ms`` is ``None``."""
        streams = {self._stream: ">"}
        if block_ms is None:
            return self._client.xreadgroup(
                self._group,
                self._consumer,
                streams,
                count,
            )
        return self._client.xreadgroup(
            self._group,
            self._consumer,
            streams,
            count,
            block_ms,
        )

    @staticmethod
    def _deadline_exceeded(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _accept_records(
        self, records: Sequence[tuple[object, dict[object, object]]]
    ) -> None:
        """Persist and ACK one batch in stream-id order."""
        ordered = sorted(
            records,
            key=lambda record: tuple(
                int(part) for part in _valkey_text(record[0]).split("-")
            ),
        )
        for record_id, raw_fields in ordered:
            fields = {
                _valkey_text(key): _valkey_text(value)
                for key, value in raw_fields.items()
            }
            fingerprint = _fingerprint(fields.get("fingerprint"), "fingerprint")
            outcome = _valkey_outcome(fields["outcome"], fingerprint=fingerprint)
            if outcome.envelope_fingerprint != fingerprint:
                raise ValueError("Valkey outcome fingerprint does not match its payload")
            if outcome.delivery_state is RemoteJobDeliveryState.COMPLETED:
                self._persist_committed(fingerprint, outcome)
            else:
                self._client.hsetnx(
                    self._failed_key, fingerprint, self._serialize_outcome(outcome)
                )
            self._client.xack(self._stream, self._group, record_id)

    def _drain(self, *, deadline: float | None = None) -> None:
        """Claim then read new records in bounded batches with mid persist/ACK.

        Each XAUTOCLAIM / XREADGROUP batch is persisted and acknowledged before
        the next fetch so a deadline exit cannot leave a large in-memory backlog
        uncommitted. Claim and fresh loops re-check the remaining deadline so a
        stream that keeps receiving messages cannot hang past ``deadline``.
        Fresh responses may be RESP2 pairs, unified mappings, or native RESP3
        mappings with one enclosing entry list (Redis contributors, 2026,
        ``parse_xread``, ``parse_xread_unified``, and ``parse_xread_resp3``).
        Normalize only that container; retain outcome validation before ACK.

        References:
            Redis contributors. (2026). helpers.py (Version 8.1.0)
                [Source code]. redis-py.
        """
        start_id = "0-0"
        while True:
            if self._deadline_exceeded(deadline):
                return
            claimed = self._client.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                self._min_idle_ms,
                start_id,
                count=self._batch_size,
            )
            next_id = _valkey_text(claimed[0]) if claimed else "0-0"
            batch = list(claimed[1]) if claimed else []
            if batch:
                self._accept_records(batch)
            # XAUTOCLAIM may return a short/empty batch while the PEL cursor
            # still has later entries (e.g. ineligible idle time). Advance
            # until the server reports cursor 0-0.
            if next_id == "0-0":
                break
            start_id = next_id

        block_ms = self._block_ms_for_read(deadline)
        while True:
            if self._deadline_exceeded(deadline):
                return
            fresh = self._xreadgroup(count=self._batch_size, block_ms=block_ms)
            # Never pass BLOCK 0: that is infinite wait on Redis/Valkey.
            # Follow-up reads omit BLOCK; only the first read in this drain may block.
            block_ms = None
            if not fresh:
                break
            batch: list[tuple[object, dict[object, object]]] = []
            streams = fresh.items() if isinstance(fresh, Mapping) else fresh
            for _stream, messages in streams:
                if (
                    isinstance(fresh, Mapping)
                    and len(messages) == 1
                    and isinstance(messages[0], (list, tuple))
                    and (
                        not messages[0]
                        or isinstance(messages[0][0], (list, tuple))
                    )
                ):
                    messages = messages[0]
                batch.extend(messages)
            if batch:
                self._accept_records(batch)

    def successful_count(self, fingerprint: str) -> int:
        """Return whether a completed record exists for ``fingerprint``."""
        key = _fingerprint(fingerprint, "fingerprint")
        if self._committed_outcome(key) is not None:
            return 1
        self._drain()
        return int(self._committed_outcome(key) is not None)

    def consume_available(self) -> Mapping[str, RemoteJobOutcome]:
        """Consume reclaimed/new stream records and return durable winners."""
        self._drain()
        raw = self._client.hgetall(self._committed_key)
        return {
            _valkey_text(fingerprint): _valkey_successful_outcome(payload, fingerprint=_fingerprint(_valkey_text(fingerprint), "fingerprint"))
            for fingerprint, payload in raw.items()
        }

    def wait_for_terminal(
        self,
        fingerprints: Sequence[str],
        *,
        deadline: float,
        prior_failed_fingerprints: Sequence[str] = (),
    ) -> Mapping[str, RemoteJobOutcome]:
        """Drain/claim until ``deadline`` or every fingerprint has a terminal outcome.

        A committed success is preferred over a recorded failure. The caller
        may supply fingerprints whose failure was already present before new
        publication. For those fingerprints only, keep waiting for a success
        and omit ambiguous failures even at the deadline. This local policy
        does not identify attempts: HSETNX retains the first hash value
        (Valkey contributors, n.d.-a, command description), while XAUTOCLAIM
        can redeliver prior pending records (Valkey contributors, n.d.-b,
        command description). No failed hash is deleted, and record admission
        and ACK behavior remain unchanged (Valkey contributors, n.d.-c,
        command description).

        Without an exclusion, cached failures remain terminal. After the wait
        loop exits, perform one final lookup so an admitted success persisted
        at/after the deadline is returned rather than reported as a timeout.

        References:
            Valkey contributors. (n.d.-a). HSETNX. Valkey command documentation.
            Valkey contributors. (n.d.-b). XAUTOCLAIM. Valkey command documentation.
            Valkey contributors. (n.d.-c). XACK. Valkey command documentation.
        """
        needed = {_fingerprint(fingerprint, "fingerprint") for fingerprint in fingerprints}
        prior_failures = {
            _fingerprint(fingerprint, "prior_failed_fingerprint")
            for fingerprint in prior_failed_fingerprints
        }
        found: dict[str, RemoteJobOutcome] = {}
        while needed - found.keys() and time.monotonic() < deadline:
            for fingerprint in list(needed - found.keys()):
                outcome = self._terminal_outcome(fingerprint)
                if outcome is not None and (
                    outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
                    or fingerprint not in prior_failures
                ):
                    found[fingerprint] = outcome
            if len(found) == len(needed):
                break
            self._drain(deadline=deadline)
        for fingerprint in list(needed - found.keys()):
            outcome = self._terminal_outcome(fingerprint)
            if outcome is not None and (
                outcome.delivery_state is RemoteJobDeliveryState.COMPLETED
                or fingerprint not in prior_failures
            ):
                found[fingerprint] = outcome
        return found

    def committed_success(self, fingerprint: str) -> RemoteJobOutcome | None:
        """Return the durable successful outcome for ``fingerprint``, if any."""
        key = _fingerprint(fingerprint, "fingerprint")
        existing = self._committed_outcome(key)
        if existing is not None:
            return existing
        self._drain()
        return self._committed_outcome(key)

    def commit_success(
        self, fingerprint: str, outcome: RemoteJobOutcome
    ) -> RemoteJobOutcome:
        """Append one completed outcome and atomically commit the first success."""
        key = _fingerprint(fingerprint, "fingerprint")
        if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
            raise ValueError("commit_success requires a completed outcome")
        if outcome.envelope_fingerprint != key:
            raise ValueError("outcome fingerprint does not match commit key")
        _successful_outcome_from_dict(outcome.to_dict(), fingerprint=key)
        payload = self._serialize_outcome(outcome)
        _valkey_successful_outcome(payload, fingerprint=key)
        self._client.xadd(
            self._stream,
            {
                "fingerprint": key,
                "outcome": payload,
            },
        )
        return self._persist_committed(key, outcome)


class ValkeyStreamsBackend:
    """Remote backend that publishes envelopes and consumes Valkey outcomes."""

    def __init__(
        self,
        client: object,
        *,
        jobs_stream: str,
        outcomes_stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int = 60_000,
        block_ms: int = 1_000,
        wait_timeout_s: float = 60.0,
    ) -> None:
        self._client = client
        self._jobs_stream = _text(jobs_stream, "jobs_stream")
        if (
            type(wait_timeout_s) not in (int, float)
            or not math.isfinite(wait_timeout_s)
            or wait_timeout_s <= 0
        ):
            raise ValueError("wait_timeout_s must be a finite positive number")
        self._wait_timeout_s = float(wait_timeout_s)
        self._outcomes = ValkeyStreamsOutcomeStore(
            client,
            stream=outcomes_stream,
            group=group,
            consumer=consumer,
            min_idle_ms=min_idle_ms,
            block_ms=block_ms,
        )

    def run_batch(
        self,
        envelopes: Sequence[RemoteJobEnvelope],
        *,
        worker_manifest: RemoteRunManifest,
        requested_device: str = "cpu",
        effective_device: str = "cpu",
        payload: Mapping[str, object] | None = None,
    ) -> tuple[RemoteJobOutcome, ...]:
        """Publish admitted jobs without reusing a known prior failure as a reply.

        Snapshot already-recorded failures before any job publication. For
        those fingerprints, a later success is admissible but a failure remains
        ambiguous, so the existing timeout is used if no success arrives.
        HSETNX preserves the first field value (Valkey contributors, n.d.-a,
        command description); pending-entry reclaim is not attempt identity
        (Valkey contributors, n.d.-b, command description). This bounded policy
        does not identify failures absent from the prepublication snapshot,
        concurrent attempts, or a later failure of the new dispatch. Success
        first-winner storage and malformed-record rejection remain unchanged.

        References:
            Valkey contributors. (n.d.-a). HSETNX. Valkey command documentation.
            Valkey contributors. (n.d.-b). XAUTOCLAIM. Valkey command documentation.
        """
        _admit_remote_device_declarations(requested_device, effective_device)
        requested = _text(requested_device, "requested_device", maximum=32)
        effective = _text(effective_device, "effective_device", maximum=32)
        envelope_batch = _admit_payload_batch(envelopes, worker_manifest, payload)
        payload_json = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        fingerprints = tuple(envelope_fingerprint(envelope) for envelope in envelope_batch)
        prior_failures = tuple(
            fingerprint
            for fingerprint in fingerprints
            if (outcome := self._outcomes._terminal_outcome(fingerprint)) is not None
            and outcome.delivery_state is RemoteJobDeliveryState.FAILED
        )
        for envelope in envelope_batch:
            fingerprint = envelope_fingerprint(envelope)
            self._client.xadd(
                self._jobs_stream,
                {
                    "fingerprint": fingerprint,
                    "envelope": json.dumps(
                        envelope.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "requested_device": requested,
                    "effective_device": effective,
                    "payload": payload_json,
                },
            )
        deadline = time.monotonic() + self._wait_timeout_s
        committed = self._outcomes.wait_for_terminal(
            fingerprints,
            deadline=deadline,
            prior_failed_fingerprints=prior_failures,
        )
        outcomes = []
        for envelope in envelope_batch:
            fingerprint = envelope_fingerprint(envelope)
            outcome = committed.get(fingerprint)
            if outcome is None:
                raise TimeoutError(f"no Valkey outcome available for {fingerprint}")
            outcomes.append(outcome)
        return tuple(
            sorted(outcomes, key=lambda outcome: (outcome.unit_index, outcome.run_id))
        )



def _is_ssh_worker_host(worker_host: str) -> bool:
    """Return whether ``worker_host`` names an SSH destination (``user@host``)."""
    return "@" in worker_host


def _worker_subprocess_env() -> Mapping[str, str]:
    """Return environment for worker subprocesses with package import path preserved."""
    env = os.environ.copy()
    import fast_mlsirm

    package_root = str(Path(fast_mlsirm.__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    if existing:
        if package_root not in existing.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([package_root, existing])
    else:
        env["PYTHONPATH"] = package_root
    return env


def _worker_module_command(interpreter: str) -> list[str]:
    """Return argv to run ``fast_mlsirm.remote_worker`` with ``interpreter``."""
    normalized = _text(interpreter, "remote_interpreter", maximum=512)
    return [normalized, "-m", "fast_mlsirm.remote_worker"]


def _invoke_worker_process(
    payload: str,
    *,
    worker_host: str,
    remote_interpreter: str,
    stdout_limit: int,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    """Run a worker with deadline-bounded local input, output and wait.

    Nonblocking raw pipes retain at most ``stdout_limit + 1`` bytes. On POSIX,
    cancellation signals only the new session's process group; elsewhere it
    kills only the direct child. SSH cancellation stops the local client, not
    necessarily the remote worker. Escaped descendants are not guaranteed to
    terminate. Process creation is not interruptible and reaping has a separate
    one-second allowance (Python Software Foundation, n.d.-a, ``Popen`` and
    ``Popen.wait``; n.d.-b, ``os.set_blocking`` and ``os.killpg``).

    References:
        Python Software Foundation. (n.d.-a). subprocess—Subprocess management.
            Python 3.14 documentation.
        Python Software Foundation. (n.d.-b). os—Miscellaneous operating system
            interfaces. Python 3.14 documentation.
    """
    worker_command = _worker_module_command(remote_interpreter)
    if _is_ssh_worker_host(worker_host):
        argv = [
            "ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "--", worker_host, shlex.join(worker_command),
        ]
        env = None
    else:
        argv = worker_command
        env = _worker_subprocess_env()
    request = memoryview(payload.encode("utf-8"))
    deadline = time.monotonic() + timeout_seconds
    stdout = bytearray()
    timed_out = False
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr_file, env=env, bufsize=0,
            start_new_session=os.name == "posix",
        )

        def _cancel_owned_process() -> None:
            """Cancel only the process/session created by this invocation."""
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # Group cancellation may be denied; keep the local deadline
                    # by terminating only this invocation's direct child.
                    process.kill()
            else:
                process.kill()

        try:
            if process.stdin is None or process.stdout is None:
                raise OSError("worker process did not provide input/output pipes")
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            offset = 0
            output_eof = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _cancel_owned_process()
                    break
                progressed = False
                if not output_eof:
                    chunk = process.stdout.read(min(65536, stdout_limit + 1 - len(stdout)))
                    if chunk == b"":
                        output_eof = True
                    elif chunk is not None:
                        stdout.extend(chunk)
                        progressed = True
                        if len(stdout) > stdout_limit:
                            _cancel_owned_process()
                            break
                if not process.stdin.closed:
                    if offset == len(request):
                        process.stdin.close()
                    else:
                        try:
                            written = process.stdin.write(request[offset:offset + 65536])
                        except BrokenPipeError:
                            process.stdin.close()
                        else:
                            if written:
                                offset += written
                                progressed = True
                if output_eof and process.poll() is not None:
                    break
                if not progressed:
                    time.sleep(min(0.005, max(0, deadline - time.monotonic())))
            returncode = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            # Cancellation has already been attempted before this wait. Do not
            # silently grant a second reap allowance; the caller records FAILED.
            raise
        except BaseException:
            _cancel_owned_process()
            process.wait(timeout=1.0)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stdin is not None:
                process.stdin.close()
        stderr_file.seek(0, os.SEEK_END)
        stderr_file.seek(max(0, stderr_file.tell() - 4096))
        stderr = stderr_file.read().decode("utf-8", errors="replace")
    if timed_out:
        returncode = returncode or -9
        stderr = f"remote worker timed out after {timeout_seconds:g}s"
    return subprocess.CompletedProcess(
        argv, returncode, stdout.decode("utf-8", errors="replace"), stderr
    )


def _parse_worker_wall_clock(raw: object, *, elapsed: float) -> float:
    """Return a finite non-negative wall-clock duration from a worker payload field."""
    if raw is None:
        wall_clock = float(elapsed)
    elif type(raw) is float:
        wall_clock = raw
    elif type(raw) is int and not isinstance(raw, bool):
        wall_clock = float(raw)
    else:
        raise ValueError("remote worker wall_clock_seconds must be a finite non-negative float")
    if wall_clock < 0.0 or not (wall_clock < float("inf")) or wall_clock != wall_clock:
        raise ValueError("remote worker wall_clock_seconds must be a finite non-negative float")
    return wall_clock


def _remote_worker_provenance_or_error(
    worker_payload: Mapping[str, object],
    *,
    worker_manifest: RemoteRunManifest,
    requested_device: str,
    effective_device: str,
    elapsed: float,
    worker_host: str,
    worker_pid: int,
    worker_hostname: str,
    driver_host: str,
) -> RemoteWorkerProvenance | str:
    """Validate version/timing readback; retain declared source/device identity.

    Source/device declarations are not measured worker attestation. The
    serialized identity_verification label keeps this distinction explicit.
    """
    library_version = worker_payload.get("library_version")
    if type(library_version) is not str or not library_version.strip():
        return "remote worker payload missing library_version"
    if library_version != worker_manifest.library_version:
        return (
            "remote worker library_version "
            f"{library_version!r} incompatible with cohort {worker_manifest.library_version!r}"
        )
    try:
        wall_clock_seconds = _parse_worker_wall_clock(
            worker_payload.get("wall_clock_seconds"),
            elapsed=elapsed,
        )
        return RemoteWorkerProvenance(
            hostname=worker_hostname,
            architecture=str(worker_payload.get("architecture", platform.machine())),
            operating_system=str(worker_payload.get("operating_system", platform.system())),
            library_version=library_version,
            source_sha256=worker_manifest.source_sha256,
            requested_device=requested_device,
            effective_device=effective_device,
            wall_clock_seconds=wall_clock_seconds,
            worker_host=worker_host,
            worker_pid=worker_pid,
            cross_host_execution=worker_hostname != driver_host,
        )
    except ValueError as exc:
        return str(exc)


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


def payload_identity_sha256(payload: Mapping[str, object]) -> str:
    """Return the canonical JSON identity for one remote numerical payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _preflight_internal_shard_batch(envelopes: Sequence[RemoteJobEnvelope]) -> None:
    """Reject batches that partition one run/family across multiple split units."""
    unit_counts: dict[tuple[str, str], int] = {}
    for envelope in envelopes:
        key = (envelope.run_id, envelope.family.value)
        unit_counts[key] = unit_counts.get(key, 0) + 1
        if unit_counts[key] > 1:
            admit_remote_job_internal_shard(envelope.family)


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
        envelope_batch = tuple(envelopes)
        for envelope in envelope_batch:
            if type(envelope) is not RemoteJobEnvelope:
                raise TypeError("each envelope must be a RemoteJobEnvelope")
            if not worker_manifest.compatible_with(envelope.manifest):
                raise CohortMismatchError(
                    "worker manifest is incompatible with envelope cohort "
                    f"(run_id={envelope.run_id!r}, unit_index={envelope.unit_index})"
                )
        _preflight_internal_shard_batch(envelope_batch)

        outcomes: list[RemoteJobOutcome] = []
        for envelope in envelope_batch:
            unit_seed = envelope.unit_seed()
            fingerprint = envelope_fingerprint(envelope)
            started = time.perf_counter()
            try:
                result = handler(envelope, unit_seed)
                output_identity = result_identity_sha256(result)
            except Exception as exc:  # handler/output failures are recorded, not raised
                elapsed = time.perf_counter() - started
                failure_message = str(exc)
                outcomes.append(
                    RemoteJobOutcome(
                        run_id=envelope.run_id,
                        unit_index=envelope.unit_index,
                        unit_seed=unit_seed,
                        family=envelope.family,
                        delivery_state=RemoteJobDeliveryState.FAILED,
                        result=None,
                        error_message=(
                            failure_message
                            if failure_message.strip()
                            else type(exc).__name__
                        ),
                        provenance=local_worker_provenance(
                            worker_manifest,
                            requested_device=requested_device,
                            effective_device=effective_device,
                            wall_clock_seconds=elapsed,
                        ),
                        input_identity_sha256=fingerprint,
                        output_identity_sha256=None,
                        envelope_fingerprint=fingerprint,
                        driver_host=socket.gethostname(),
                        driver_pid=os.getpid(),
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
                    input_identity_sha256=fingerprint,
                    output_identity_sha256=output_identity,
                    envelope_fingerprint=fingerprint,
                    driver_host=socket.gethostname(),
                    driver_pid=os.getpid(),
                )
            )
        return tuple(
            sorted(outcomes, key=lambda outcome: (outcome.unit_index, outcome.run_id))
        )


class SubprocessExecutor:
    """Subprocess L4 backend that dispatches legal families to real library calls."""

    _MAX_WORKER_STDOUT_BYTES = 1_048_576

    def __init__(
        self,
        worker_host: str,
        *,
        remote_interpreter: str | None = None,
        ledger: OutcomeCommitStore | None = None,
        driver_host: str | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.worker_host = _text(worker_host, "worker_host", maximum=128)
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = float(timeout_seconds)
        if remote_interpreter is None:
            if _is_ssh_worker_host(self.worker_host):
                raise ValueError(
                    "remote_interpreter is required when worker_host is an SSH destination"
                )
            remote_interpreter = sys.executable
        self.remote_interpreter = _text(remote_interpreter, "remote_interpreter", maximum=512)
        self._ledger = ledger or OutcomeCommitLedger()
        self._driver_host = driver_host or socket.gethostname()
        self._driver_pid = os.getpid()

    def run_batch(
        self,
        envelopes: Sequence[RemoteJobEnvelope],
        *,
        worker_manifest: RemoteRunManifest,
        requested_device: str = "cpu",
        effective_device: str = "cpu",
        payload: Mapping[str, object] | None = None,
    ) -> tuple[RemoteJobOutcome, ...]:
        if not isinstance(envelopes, Sequence):
            raise TypeError("envelopes must be a sequence")
        envelope_batch = tuple(envelopes)
        for envelope in envelope_batch:
            if type(envelope) is not RemoteJobEnvelope:
                raise TypeError("each envelope must be a RemoteJobEnvelope")
        _preflight_internal_shard_batch(envelope_batch)
        _admit_remote_device_declarations(requested_device, effective_device)
        if payload is not None:
            payload_sha256 = payload_identity_sha256(payload)
        for envelope in envelope_batch:
            if not worker_manifest.compatible_with(envelope.manifest):
                raise CohortMismatchError(
                    "worker manifest is incompatible with envelope cohort "
                    f"(run_id={envelope.run_id!r}, unit_index={envelope.unit_index})"
                )
            if payload is None:
                raise ValueError(f"payload is required for {envelope.family.value}")
            if payload_sha256 != envelope.manifest.payload_sha256:
                raise CohortMismatchError(
                    "payload identity is incompatible with envelope cohort "
                    f"(run_id={envelope.run_id!r}, unit_index={envelope.unit_index})"
                )

        outcomes: list[RemoteJobOutcome] = []
        for envelope in envelope_batch:
            fingerprint = envelope_fingerprint(envelope)
            cached = self._ledger.committed_success(fingerprint)
            if cached is not None:
                outcomes.append(cached)
                continue
            outcome = self._execute_one(
                envelope,
                worker_manifest=worker_manifest,
                requested_device=requested_device,
                effective_device=effective_device,
                fingerprint=fingerprint,
                payload=payload,
            )
            if outcome.delivery_state is RemoteJobDeliveryState.COMPLETED:
                outcome = self._ledger.commit_success(fingerprint, outcome)
            outcomes.append(outcome)
        return tuple(
            sorted(outcomes, key=lambda outcome: (outcome.unit_index, outcome.run_id))
        )

    def _execute_one(
        self,
        envelope: RemoteJobEnvelope,
        *,
        worker_manifest: RemoteRunManifest,
        requested_device: str,
        effective_device: str,
        fingerprint: str,
        payload: Mapping[str, object] | None,
    ) -> RemoteJobOutcome:
        unit_seed = envelope.unit_seed()
        started = time.perf_counter()
        payload = json.dumps(
            {
                "envelope": envelope.to_dict(),
                "worker_host": self.worker_host,
                "requested_device": requested_device,
                "effective_device": effective_device,
                "payload": None if payload is None else dict(payload),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            completed = _invoke_worker_process(
                payload,
                worker_host=self.worker_host,
                remote_interpreter=self.remote_interpreter,
                stdout_limit=self._MAX_WORKER_STDOUT_BYTES,
                timeout_seconds=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            elapsed = time.perf_counter() - started
            return RemoteJobOutcome(
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
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        elapsed = time.perf_counter() - started
        stdout = completed.stdout or ""
        if len(stdout.encode("utf-8")) > self._MAX_WORKER_STDOUT_BYTES:
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message="remote worker stdout exceeded bound",
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            message = stderr or f"remote worker exited with code {completed.returncode}"
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message=message,
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        try:
            worker_payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message=f"remote worker returned invalid JSON: {exc}",
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        if type(worker_payload) is not dict:
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message="remote worker payload must be a mapping",
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        worker_pid = worker_payload.get("worker_pid")
        worker_hostname = worker_payload.get("hostname")
        if type(worker_pid) is not int or isinstance(worker_pid, bool) or worker_pid <= 0:
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message="remote worker payload missing worker_pid",
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )
        if type(worker_hostname) is not str or not worker_hostname.strip():
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message="remote worker payload missing hostname",
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        raw_delivery_state = worker_payload.get("delivery_state")
        try:
            delivery_state = RemoteJobDeliveryState(raw_delivery_state)
        except (TypeError, ValueError):
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message=(
                    "remote worker payload has missing or unknown delivery_state "
                    f"{raw_delivery_state!r}"
                ),
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        provenance_or_error = _remote_worker_provenance_or_error(
            worker_payload,
            worker_manifest=worker_manifest,
            requested_device=requested_device,
            effective_device=effective_device,
            elapsed=elapsed,
            worker_host=self.worker_host,
            worker_pid=worker_pid,
            worker_hostname=worker_hostname,
            driver_host=self._driver_host,
        )
        if type(provenance_or_error) is str:
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message=provenance_or_error,
                provenance=local_worker_provenance(
                    worker_manifest,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=elapsed,
                    worker_host=self.worker_host,
                    worker_pid=os.getpid(),
                    cross_host_execution=False,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        if delivery_state is RemoteJobDeliveryState.FAILED:
            error_message = worker_payload.get("error_message")
            if type(error_message) is not str or not error_message.strip():
                error_message = "remote worker failed without error_message"
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message=error_message,
                provenance=provenance_or_error,
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        result = worker_payload.get("result")
        try:
            output_identity = result_identity_sha256(result)
        except (TypeError, ValueError) as exc:
            output_identity = None
            result_error = f"remote worker result is not finite JSON: {exc}"
        else:
            result_error = "remote worker output_identity_sha256 does not hash its result"
        if output_identity is None or worker_payload.get("output_identity_sha256") != output_identity:
            return RemoteJobOutcome(
                run_id=envelope.run_id,
                unit_index=envelope.unit_index,
                unit_seed=unit_seed,
                family=envelope.family,
                delivery_state=RemoteJobDeliveryState.FAILED,
                result=None,
                error_message=result_error,
                provenance=provenance_or_error,
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )
        return RemoteJobOutcome(
            run_id=envelope.run_id,
            unit_index=envelope.unit_index,
            unit_seed=unit_seed,
            family=envelope.family,
            delivery_state=RemoteJobDeliveryState.COMPLETED,
            result=result,
            error_message=None,
            provenance=provenance_or_error,
            input_identity_sha256=fingerprint,
            output_identity_sha256=output_identity,
            envelope_fingerprint=fingerprint,
            driver_host=self._driver_host,
            driver_pid=self._driver_pid,
        )
