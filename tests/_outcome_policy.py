"""Fail-closed accounting for pytest outcomes that do not execute assertions."""

from __future__ import annotations

from collections.abc import Mapping, Sized

import pytest


_NON_EXECUTION_BUCKETS = ("skipped", "xfailed", "xpassed")


def _non_execution_counts(stats: Mapping[str, Sized]) -> dict[str, int]:
    """Return non-zero skip/xfail/xpass counts from pytest terminal statistics."""
    return {
        bucket: len(stats.get(bucket, ()))
        for bucket in _NON_EXECUTION_BUCKETS
        if len(stats.get(bucket, ())) > 0
    }


def _format_non_execution_counts(counts: Mapping[str, int]) -> str:
    """Render deterministic terminal evidence for prohibited non-execution states."""
    detail = ", ".join(f"{bucket}={counts[bucket]}" for bucket in _NON_EXECUTION_BUCKETS if bucket in counts)
    return f"non-execution outcomes are non-passing: {detail}"


def _enforce_no_hidden_outcomes(session: object, terminalreporter: object | None) -> None:
    """Fail a successful invocation on non-execution without erasing stronger failures."""
    if session.exitstatus != pytest.ExitCode.OK:
        return
    if terminalreporter is None:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        return
    counts = _non_execution_counts(terminalreporter.stats)
    if counts:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
