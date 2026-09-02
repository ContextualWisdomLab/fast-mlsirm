"""Repository-wide pytest governance for fail-closed test execution."""

from __future__ import annotations

from _outcome_policy import (
    _enforce_no_hidden_outcomes,
    _format_non_execution_counts,
    _non_execution_counts,
)


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Fail the invocation when pytest reports any skip, xfail, or xpass outcome."""
    del exitstatus
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is not None:
        counts = _non_execution_counts(terminalreporter.stats)
        if counts:
            terminalreporter.write_sep("=", _format_non_execution_counts(counts))
    _enforce_no_hidden_outcomes(session, terminalreporter)
