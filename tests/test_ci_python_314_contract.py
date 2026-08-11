"""Contracts for the supported Python versions exercised by pull-request CI."""

from pathlib import Path


_CI_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _python_job_source() -> str:
    """Return the Python test job without unrelated CI jobs."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  python:\n")
    end = workflow.index("\n  rust:\n", start)
    return workflow[start:end]


def test_python_job_runs_full_suite_on_python_314() -> None:
    """Pull-request CI must execute the full fail-slow suite on CPython 3.14."""
    python_job = _python_job_source()

    assert 'python-version: ["3.12", "3.14"]' in python_job
    assert "python-version: ${{ matrix.python-version }}" in python_job
    assert "fail-fast: false" in python_job
    assert "- run: pytest" in python_job
