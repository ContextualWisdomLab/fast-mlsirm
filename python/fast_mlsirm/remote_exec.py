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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
import platform
import re
import socket
import sqlite3
import subprocess
import sys
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
    worker_host: str
    worker_pid: int
    cross_host_execution: bool

    def __post_init__(self) -> None:
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
    """Build provenance for the current interpreter process."""
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
    """Return a stable SHA-256 digest for one JSON-serializable remote result."""
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
        """Return the committed successful outcome for ``fingerprint``, if any."""
        return self._successful.get(fingerprint)

    def commit_success(self, fingerprint: str, outcome: RemoteJobOutcome) -> RemoteJobOutcome:
        """Commit one successful outcome or return the prior commit without re-recording."""
        if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
            raise ValueError("commit_success requires a completed outcome")
        existing = self._successful.get(fingerprint)
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

    A future Valkey Streams transport can implement :class:`OutcomeCommitStore`
    without changing executors. SQLite is the local durable adapter.
    """

    def __init__(self, database: str | Path) -> None:
        self._database = Path(database)
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS successful_outcomes "
                "(fingerprint TEXT PRIMARY KEY, outcome_json TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database, timeout=30.0)

    def successful_count(self, fingerprint: str) -> int:
        """Return whether one successful outcome is durably committed."""
        key = _fingerprint(fingerprint, "fingerprint")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM successful_outcomes WHERE fingerprint = ?",
                (key,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def committed_success(self, fingerprint: str) -> RemoteJobOutcome | None:
        """Return the durable successful outcome for ``fingerprint``, if any."""
        key = _fingerprint(fingerprint, "fingerprint")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT outcome_json FROM successful_outcomes WHERE fingerprint = ?",
                (key,),
            ).fetchone()
        return None if row is None else _outcome_from_dict(json.loads(row[0]))

    def commit_success(self, fingerprint: str, outcome: RemoteJobOutcome) -> RemoteJobOutcome:
        """Atomically commit the first success and return the durable winner."""
        key = _fingerprint(fingerprint, "fingerprint")
        if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
            raise ValueError("commit_success requires a completed outcome")
        payload = json.dumps(
            outcome.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
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
        return _outcome_from_dict(json.loads(row[0]))


def _outcome_from_dict(payload: object) -> RemoteJobOutcome:
    """Restore one package-produced outcome from durable JSON."""
    if type(payload) is not dict or type(payload.get("provenance")) is not dict:
        raise ValueError("stored outcome must be a package outcome mapping")
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


def _valkey_text(value: object) -> str:
    """Normalize redis-py text responses without accepting other coercions."""
    if type(value) is bytes:
        return value.decode("utf-8")
    if type(value) is str:
        return value
    raise ValueError("Valkey stream fields must be UTF-8 text")


class ValkeyStreamsOutcomeStore:
    """Outcome store using a Valkey consumer group and pending-entry reclaim.

    The injected client follows redis-py's synchronous Streams API. This keeps
    the core package dependency-free while allowing either ``redis`` or
    ``valkey`` clients at the deployment boundary. Stream delivery is separate
    from the durable ``{stream}:committed`` hash, which mirrors the SQLite
    first-success contract across process restarts and consumer-group members.
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
        return _outcome_from_dict(json.loads(_valkey_text(stored)))

    def _persist_committed(
        self, fingerprint: str, outcome: RemoteJobOutcome
    ) -> RemoteJobOutcome:
        key = _fingerprint(fingerprint, "fingerprint")
        payload = self._serialize_outcome(outcome)
        self._client.hsetnx(self._committed_key, key, payload)
        stored = self._client.hget(self._committed_key, key)
        if stored is None:  # pragma: no cover - hash write invariant
            raise RuntimeError("successful outcome commit was not persisted")
        return _outcome_from_dict(json.loads(_valkey_text(stored)))

    def _drain(self) -> None:
        records: list[tuple[object, dict[object, object]]] = []
        start_id = "0-0"
        while True:
            claimed = self._client.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                self._min_idle_ms,
                start_id,
                count=self._batch_size,
            )
            batch = list(claimed[1]) if claimed else []
            records.extend(batch)
            if not batch or len(batch) < self._batch_size:
                break
            start_id = _valkey_text(claimed[0])

        fresh = self._client.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: ">"},
            count=self._batch_size,
            block=self._block_ms,
        )
        while fresh:
            for _stream, messages in fresh:
                records.extend(messages)
            fresh = self._client.xreadgroup(
                self._group,
                self._consumer,
                {self._stream: ">"},
                count=self._batch_size,
                block=0,
            )

        records.sort(
            key=lambda record: tuple(
                int(part) for part in _valkey_text(record[0]).split("-")
            )
        )
        for record_id, raw_fields in records:
            fields = {
                _valkey_text(key): _valkey_text(value)
                for key, value in raw_fields.items()
            }
            fingerprint = _fingerprint(fields.get("fingerprint"), "fingerprint")
            outcome = _outcome_from_dict(json.loads(fields["outcome"]))
            if outcome.delivery_state is not RemoteJobDeliveryState.COMPLETED:
                raise ValueError("Valkey outcome stream accepts completed outcomes only")
            if outcome.envelope_fingerprint != fingerprint:
                raise ValueError("Valkey outcome fingerprint does not match its payload")
            self._persist_committed(fingerprint, outcome)
            self._client.xack(self._stream, self._group, record_id)

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
            _valkey_text(fingerprint): _outcome_from_dict(json.loads(_valkey_text(payload)))
            for fingerprint, payload in raw.items()
        }

    def wait_for_committed(
        self,
        fingerprints: Sequence[str],
        *,
        deadline: float,
    ) -> Mapping[str, RemoteJobOutcome]:
        """Drain/claim until ``deadline`` or every fingerprint is durably committed."""
        needed = {_fingerprint(fingerprint, "fingerprint") for fingerprint in fingerprints}
        found: dict[str, RemoteJobOutcome] = {}
        while needed - found.keys() and time.monotonic() < deadline:
            for fingerprint in list(needed - found.keys()):
                outcome = self._committed_outcome(fingerprint)
                if outcome is not None:
                    found[fingerprint] = outcome
            if len(found) == len(needed):
                break
            self._drain()
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
        self._client.xadd(
            self._stream,
            {
                "fingerprint": key,
                "outcome": self._serialize_outcome(outcome),
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
        if wait_timeout_s <= 0:
            raise ValueError("wait_timeout_s must be > 0")
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
    ) -> tuple[RemoteJobOutcome, ...]:
        requested = _text(requested_device, "requested_device", maximum=32)
        effective = _text(effective_device, "effective_device", maximum=32)
        envelope_batch = tuple(envelopes)
        for envelope in envelope_batch:
            if type(envelope) is not RemoteJobEnvelope:
                raise TypeError("each envelope must be a RemoteJobEnvelope")
            if not worker_manifest.compatible_with(envelope.manifest):
                raise CohortMismatchError(
                    "worker manifest is incompatible with envelope cohort"
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
                },
            )
        deadline = time.monotonic() + self._wait_timeout_s
        fingerprints = tuple(envelope_fingerprint(envelope) for envelope in envelope_batch)
        committed = self._outcomes.wait_for_committed(fingerprints, deadline=deadline)
        outcomes = []
        for envelope in envelope_batch:
            fingerprint = envelope_fingerprint(envelope)
            outcome = committed.get(fingerprint)
            if outcome is None:
                raise TimeoutError(f"no Valkey outcome available for {fingerprint}")
            outcomes.append(outcome)
        return tuple(sorted(outcomes, key=lambda outcome: outcome.unit_index))


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
) -> subprocess.CompletedProcess[str]:
    """Run ``fast_mlsirm.remote_worker`` locally or over SSH for ``worker_host``."""
    worker_command = _worker_module_command(remote_interpreter)
    if _is_ssh_worker_host(worker_host):
        return subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "BatchMode=yes",
                worker_host,
                *worker_command,
            ],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
        )
    return subprocess.run(
        worker_command,
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        env=_worker_subprocess_env(),
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
        envelope_batch = tuple(envelopes)
        for envelope in envelope_batch:
            if type(envelope) is not RemoteJobEnvelope:
                raise TypeError("each envelope must be a RemoteJobEnvelope")
            if not worker_manifest.compatible_with(envelope.manifest):
                raise CohortMismatchError(
                    "worker manifest is incompatible with envelope cohort "
                    f"(run_id={envelope.run_id!r}, unit_index={envelope.unit_index})"
                )

        outcomes: list[RemoteJobOutcome] = []
        for envelope in envelope_batch:
            unit_seed = envelope.unit_seed()
            fingerprint = envelope_fingerprint(envelope)
            started = time.perf_counter()
            try:
                result = handler(envelope, unit_seed)
            except Exception as exc:  # handler failures are recorded, not raised
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
                    output_identity_sha256=result_identity_sha256(result),
                    envelope_fingerprint=fingerprint,
                    driver_host=socket.gethostname(),
                    driver_pid=os.getpid(),
                )
            )
        return tuple(sorted(outcomes, key=lambda outcome: outcome.unit_index))


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
    ) -> None:
        self.worker_host = _text(worker_host, "worker_host", maximum=128)
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
            )
            if outcome.delivery_state is RemoteJobDeliveryState.COMPLETED:
                outcome = self._ledger.commit_success(fingerprint, outcome)
            outcomes.append(outcome)
        return tuple(sorted(outcomes, key=lambda outcome: outcome.unit_index))

    def _execute_one(
        self,
        envelope: RemoteJobEnvelope,
        *,
        worker_manifest: RemoteRunManifest,
        requested_device: str,
        effective_device: str,
        fingerprint: str,
    ) -> RemoteJobOutcome:
        unit_seed = envelope.unit_seed()
        started = time.perf_counter()
        payload = json.dumps(
            {
                "envelope": envelope.to_dict(),
                "worker_host": self.worker_host,
                "requested_device": requested_device,
                "effective_device": effective_device,
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
            )
        except OSError as exc:
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

        if worker_payload.get("delivery_state") == RemoteJobDeliveryState.FAILED.value:
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
                provenance=RemoteWorkerProvenance(
                    hostname=worker_hostname,
                    architecture=str(worker_payload.get("architecture", platform.machine())),
                    operating_system=str(worker_payload.get("operating_system", platform.system())),
                    library_version=str(
                        worker_payload.get("library_version", worker_manifest.library_version)
                    ),
                    source_sha256=worker_manifest.source_sha256,
                    requested_device=requested_device,
                    effective_device=effective_device,
                    wall_clock_seconds=float(worker_payload.get("wall_clock_seconds", elapsed)),
                    worker_host=self.worker_host,
                    worker_pid=worker_pid,
                    cross_host_execution=worker_hostname != self._driver_host,
                ),
                input_identity_sha256=fingerprint,
                output_identity_sha256=None,
                envelope_fingerprint=fingerprint,
                driver_host=self._driver_host,
                driver_pid=self._driver_pid,
            )

        result = worker_payload.get("result")
        output_identity = worker_payload.get("output_identity_sha256")
        if type(output_identity) is not str:
            output_identity = result_identity_sha256(result)
        return RemoteJobOutcome(
            run_id=envelope.run_id,
            unit_index=envelope.unit_index,
            unit_seed=unit_seed,
            family=envelope.family,
            delivery_state=RemoteJobDeliveryState.COMPLETED,
            result=result,
            error_message=None,
            provenance=RemoteWorkerProvenance(
                hostname=worker_hostname,
                architecture=str(worker_payload.get("architecture", platform.machine())),
                operating_system=str(worker_payload.get("operating_system", platform.system())),
                library_version=str(
                    worker_payload.get("library_version", worker_manifest.library_version)
                ),
                source_sha256=worker_manifest.source_sha256,
                requested_device=requested_device,
                effective_device=effective_device,
                wall_clock_seconds=float(worker_payload.get("wall_clock_seconds", elapsed)),
                worker_host=self.worker_host,
                worker_pid=worker_pid,
                cross_host_execution=worker_hostname != self._driver_host,
            ),
            input_identity_sha256=fingerprint,
            output_identity_sha256=output_identity,
            envelope_fingerprint=fingerprint,
            driver_host=self._driver_host,
            driver_pid=self._driver_pid,
        )
