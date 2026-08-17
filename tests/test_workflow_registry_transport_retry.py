"""Transport-level retry contract for the read-only workflow registry audit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.audit_workflow_registry as audit


@pytest.mark.parametrize("status", [403, 404, 429, 500, 502, 503, 504])
def test_run_gh_api_retries_transient_http_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=f"HTTP {status}: transient synthetic failure",
            )
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    assert audit._run_gh_api(
        "repos/ContextualWisdomLab/fast-mlsirm/actions/workflows",
        max_attempts=2,
        retry_sleep_seconds=0,
    ) == {"ok": True}
    assert calls == 2


def test_run_gh_api_exhausts_persistent_retryable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="HTTP 403: persistent synthetic permission/rate-limit failure",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    with pytest.raises(audit.GitHubApiError, match="HTTP 403"):
        audit._run_gh_api(
            "repos/ContextualWisdomLab/fast-mlsirm/actions/workflows",
            max_attempts=3,
            retry_sleep_seconds=0,
        )

    assert calls == 3


def test_run_gh_api_does_not_retry_nontransient_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="HTTP 401: bad credentials",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    with pytest.raises(audit.GitHubApiError, match="HTTP 401"):
        audit._run_gh_api(
            "repos/ContextualWisdomLab/fast-mlsirm/actions/workflows",
            max_attempts=3,
            retry_sleep_seconds=0,
        )

    assert calls == 1
