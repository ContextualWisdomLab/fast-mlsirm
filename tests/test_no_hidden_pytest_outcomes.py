"""Tests for repository-wide fail-closed pytest outcome accounting."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from _outcome_policy import (
    _enforce_no_hidden_outcomes,
    _format_non_execution_counts,
    _non_execution_counts,
)


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


@pytest.mark.parametrize(
    "exitstatus",
    [
        pytest.ExitCode.TESTS_FAILED,
        pytest.ExitCode.INTERRUPTED,
        pytest.ExitCode.INTERNAL_ERROR,
        pytest.ExitCode.USAGE_ERROR,
        pytest.ExitCode.NO_TESTS_COLLECTED,
    ],
)
def test_existing_non_success_exit_status_is_preserved(exitstatus: pytest.ExitCode) -> None:
    """Non-execution evidence must not erase a more specific pre-existing pytest failure."""
    session = SimpleNamespace(exitstatus=exitstatus)
    reporter = SimpleNamespace(stats={"skipped": [object()]})

    _enforce_no_hidden_outcomes(session, reporter)

    assert session.exitstatus == exitstatus


def test_missing_terminal_reporter_fails_closed() -> None:
    """A successful run that cannot account for terminal outcomes cannot claim GREEN evidence."""
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)

    _enforce_no_hidden_outcomes(session, None)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_non_execution_counts_omit_zero_buckets() -> None:
    """Diagnostics remain deterministic and contain only observed non-execution states."""
    stats = {"skipped": [object(), object()], "xfailed": [], "passed": [object()]}

    assert _non_execution_counts(stats) == {"skipped": 2}


def test_non_execution_diagnostic_uses_stable_bucket_order() -> None:
    """Failure logs name every observed bucket without depending on mapping insertion order."""
    counts = {"xpassed": 3, "skipped": 2, "xfailed": 1}

    assert _format_non_execution_counts(counts) == (
        "non-execution outcomes are non-passing: skipped=2, xfailed=1, xpassed=3"
    )


def test_real_pytest_hook_preserves_interrupt_after_skip(tmp_path: Path) -> None:
    """Actual hook discovery must preserve interruption even when a prior test skipped."""
    probe = tmp_path / "test_outcome_policy_probe.py"
    probe.write_text(
        "import pytest\n"
        "def test_a_non_execution(): pytest.skip('integration probe')\n"
        "def test_b_interrupt(): raise KeyboardInterrupt()\n",
        encoding="utf-8",
    )
    test_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(test_root), env.get("PYTHONPATH", "")) if part
    )

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "conftest", str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == int(pytest.ExitCode.INTERRUPTED)
    assert "non-execution outcomes are non-passing" not in completed.stdout
