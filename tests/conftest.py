"""Repository-wide pytest governance for fail-closed test execution."""

from __future__ import annotations

from _outcome_policy import _enforce_no_hidden_outcomes


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Fail the invocation when pytest reports any skip, xfail, or xpass outcome."""
    del exitstatus
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    _enforce_no_hidden_outcomes(session, terminalreporter)
