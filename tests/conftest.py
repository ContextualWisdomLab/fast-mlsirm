"""Repository-wide pytest governance for fail-closed test execution."""

from __future__ import annotations

import pytest

from _outcome_policy import (
    _enforce_no_hidden_outcomes,
    _format_non_execution_counts,
    _non_execution_counts,
)


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Fail successful invocations on non-execution while preserving primary failures."""
    del exitstatus
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if session.exitstatus == pytest.ExitCode.OK and terminalreporter is not None:
        counts = _non_execution_counts(terminalreporter.stats)
        if counts:
            terminalreporter.write_sep("=", _format_non_execution_counts(counts))
    _enforce_no_hidden_outcomes(session, terminalreporter)
