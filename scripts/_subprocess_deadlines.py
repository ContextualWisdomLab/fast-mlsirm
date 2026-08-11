"""Fail-first operation-specific subprocess deadline boundary.

This module intentionally provides only the public configuration/error surface
needed for the regression tests to reach subprocess execution. Process-group
isolation, timeout cleanup, and ignored-Rust runner integration are implemented
only after the exact RED boundary is observed.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

PROCESS_GROUP_GRACE_SECONDS = 5.0


class SubprocessOperation(str, Enum):
    """Supported subprocess operation classes."""

    CARGO_METADATA = "cargo_metadata"
    CARGO_TEST_LIST = "cargo_test_list"
    STATISTICAL_TEST = "statistical_test"


@dataclass(frozen=True)
class _DeadlinePolicy:
    """Deadline configuration for one operation class."""

    env_key: str
    default_seconds: int
    minimum_seconds: int
    maximum_seconds: int


_POLICIES = {
    SubprocessOperation.CARGO_METADATA: _DeadlinePolicy(
        "FAST_MLSIRM_CARGO_METADATA_TIMEOUT_SECONDS", 30, 5, 120
    ),
    SubprocessOperation.CARGO_TEST_LIST: _DeadlinePolicy(
        "FAST_MLSIRM_CARGO_TEST_LIST_TIMEOUT_SECONDS", 120, 30, 600
    ),
    SubprocessOperation.STATISTICAL_TEST: _DeadlinePolicy(
        "FAST_MLSIRM_STATISTICAL_TEST_TIMEOUT_SECONDS", 1800, 60, 7200
    ),
}


class BoundedSubprocessTimeout(RuntimeError):
    """Redacted timeout evidence for one operation class."""

    def __init__(self, *, operation: SubprocessOperation, timeout_seconds: float) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"{operation.value} exceeded bounded timeout ({timeout_seconds:g} seconds)"
        )

    def as_dict(self) -> dict[str, object]:
        """Return machine-readable package-owned timeout evidence."""
        return {
            "status": "timeout",
            "operation": self.operation.value,
            "timeout_seconds": self.timeout_seconds,
        }


def resolve_timeout_seconds(
    operation: SubprocessOperation,
    environ: Mapping[str, str] | None = None,
) -> float:
    """Resolve one bounded operation-specific deadline."""
    if not isinstance(operation, SubprocessOperation):
        raise ValueError("operation must be a SubprocessOperation")
    policy = _POLICIES[operation]
    source = os.environ if environ is None else environ
    raw = source.get(policy.env_key)
    if raw is None:
        return float(policy.default_seconds)
    if not raw.isdecimal():
        raise ValueError(
            f"{operation.name} timeout must be an integer number of seconds between "
            f"{policy.minimum_seconds} and {policy.maximum_seconds}"
        )
    seconds = int(raw)
    if not policy.minimum_seconds <= seconds <= policy.maximum_seconds:
        raise ValueError(
            f"{operation.name} timeout must be between {policy.minimum_seconds} and "
            f"{policy.maximum_seconds} seconds"
        )
    return float(seconds)


def _validate_command(command: Sequence[str]) -> list[str]:
    """Materialize and validate a subprocess argument vector."""
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a non-empty sequence of strings")
    argv = list(command)
    if not all(isinstance(part, str) and part for part in argv):
        raise ValueError("command must be a non-empty sequence of non-empty strings")
    return argv


def _posix_process_group_exists(process_group_id: int) -> bool:
    """Fail-first liveness probe used by the cleanup contract."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_bounded(
    command: Sequence[str],
    *,
    operation: SubprocessOperation,
    check: bool = False,
    capture_output: bool = False,
    text: bool = False,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run one command; fail-first version lacks required process-group cleanup."""
    argv = _validate_command(command)
    timeout_seconds = resolve_timeout_seconds(operation)
    kwargs: dict[str, object] = {"text": text, "env": env}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = subprocess.Popen(argv, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        raise BoundedSubprocessTimeout(
            operation=operation,
            timeout_seconds=timeout_seconds,
        ) from None
    completed = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if check:
        completed.check_returncode()
    return completed
