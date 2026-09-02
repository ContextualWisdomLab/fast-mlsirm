"""Tests for repository-wide fail-closed pytest outcome accounting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from _outcome_policy import _enforce_no_hidden_outcomes, _non_execution_counts


def test_clean_terminal_statistics_preserve_success() -> None:
    """An all-executed test run keeps its successful exit status."""
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    reporter = SimpleNamespace(stats={"passed": [object()]})

    _enforce_no_hidden_outcomes(session, reporter)

    assert session.exitstatus == pytest.ExitCode.OK


@pytest.mark.parametrize("bucket", ["skipped", "xfailed", "xpassed"])
def test_each_non_execution_bucket_forces_suite_failure(bucket: str) -> None:
    """Every pytest non-execution outcome makes the complete invocation fail closed."""
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    reporter = SimpleNamespace(stats={bucket: [object()]})

    _enforce_no_hidden_outcomes(session, reporter)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_missing_terminal_reporter_fails_closed() -> None:
    """A run that cannot account for terminal outcomes cannot claim GREEN evidence."""
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)

    _enforce_no_hidden_outcomes(session, None)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_non_execution_counts_omit_zero_buckets() -> None:
    """Diagnostics remain deterministic and contain only observed non-execution states."""
    stats = {"skipped": [object(), object()], "xfailed": [], "passed": [object()]}

    assert _non_execution_counts(stats) == {"skipped": 2}
