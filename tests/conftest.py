"""Repository-wide pytest governance for fail-closed test execution."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from _outcome_policy import (
    _enforce_no_hidden_outcomes,
    _format_non_execution_counts,
    _non_execution_counts,
)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(
    session: object,
    exitstatus: int,
) -> Generator[None, None, None]:
    """Enforce non-execution policy after every ordinary session-finish hook."""
    original_exitstatus = exitstatus
    yield

    if original_exitstatus != pytest.ExitCode.OK and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = original_exitstatus

    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if session.exitstatus == pytest.ExitCode.OK and terminalreporter is not None:
        counts = _non_execution_counts(terminalreporter.stats)
        if counts:
            terminalreporter.write_sep("=", _format_non_execution_counts(counts))
    _enforce_no_hidden_outcomes(session, terminalreporter)
