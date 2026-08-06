"""Contract tests for the hourly PR-queue governance workflow."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "hourly-pr-governance.yml"


def _workflow_text() -> str:
    """Return the scheduled workflow as UTF-8 text."""
    return _WORKFLOW.read_text(encoding="utf-8")


def _build_step_script() -> str:
    """Return the live governance shell block with YAML indentation removed."""
    text = _workflow_text()
    step = text.index("      - name: Build live PR queue governance evidence")
    run = text.index("        run: |\n", step) + len("        run: |\n")
    end = text.index("      - name:", run)
    return textwrap.dedent(text[run:end])


def test_hourly_governance_workflow_exists_and_runs_every_hour():
    """The repository records queue-governance evidence on an hourly cadence."""
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert 'cron: "0 * * * *"' in text


def test_hourly_governance_workflow_is_read_only_and_bounded():
    """The scheduled loop cannot mutate pull requests or repository contents."""
    text = _workflow_text()
    assert "contents: read" in text
    assert "pull-requests: read" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "cancel-in-progress: true" in text
    assert "timeout-minutes: 10" in text


def test_hourly_governance_workflow_builds_and_publishes_evidence():
    """Each run invokes the governed builder and uploads both audit artifacts."""
    text = _workflow_text()
    assert "python scripts/build_pr_queue_governance.py" in text
    assert "hourly-pr-queue-governance" in text
    assert "pr_queue_governance_manifest.json" in text
    assert "pr_queue_governance_report.html" in text
    assert "actions/upload-artifact" in text
    assert "if: always()" in text


def test_hourly_governance_workflow_tracks_implementation_and_contract_tests():
    """Changes to duplicate-claim governance trigger the evidence workflow."""
    text = _workflow_text()
    assert "scripts/build_pr_queue_governance.py" in text
    assert "tests/test_pr_queue_governance.py" in text
    assert "tests/test_hourly_pr_governance_workflow.py" in text


def test_governance_workflow_fails_closed_when_no_tests_are_discovered():
    """A renamed or removed test suite cannot make self-verification silently pass."""
    text = _workflow_text()
    assert "test_names = [" in text
    assert "if not test_names:" in text
    guard = 'raise RuntimeError("no hourly governance contract tests were discovered")'
    assert guard in text
    assert text.index("if not test_names:") < text.index(guard)
    assert text.index(guard) < text.index("for name in test_names:")


def test_hourly_governance_workflow_retries_transient_github_api_failures():
    """Transient GitHub 5xx failures receive a small bounded backoff retry."""
    text = _workflow_text()
    assert "max_attempts=3" in text
    assert '"HTTP 502"' in text
    assert '"HTTP 503"' in text
    assert 'delay=$((attempt * 10))' in text
    assert 'attempt=$((attempt + 1))' in text


def test_hourly_governance_workflow_does_not_retry_governance_failures():
    """Only transient GitHub errors are retried; real gate failures stay failed."""
    text = _workflow_text()
    assert 'manifest.get("github", {})' in text
    assert 'bool(errors) and all(' in text
    guard = 'if [[ "$retryable" != "true" || "$attempt" -ge "$max_attempts" ]]; then'
    assert guard in text
    assert text.index(guard) < text.index("exit 1")
    assert "continue-on-error: true" not in text


def test_hourly_governance_retry_shell_is_syntactically_valid():
    """The bounded retry block remains valid Bash after YAML de-indentation."""
    completed = subprocess.run(
        ["bash", "-n"],
        input=_build_step_script(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
