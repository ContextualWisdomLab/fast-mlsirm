"""Fail-first contract for the advertised minimum Python runtime."""

from __future__ import annotations

from pathlib import Path


_CI_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _python_matrix_job_source() -> str:
    """Return the full-suite Python matrix job without unrelated CI jobs."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  python-matrix:\n")
    end = workflow.index("\n  python:\n", start)
    return workflow[start:end]


def test_declared_minimum_python_310_runs_the_complete_suite() -> None:
    """CPython 3.10 must run the same Rust-primary pytest suite as newer runtimes."""
    python_job = _python_matrix_job_source()

    assert 'python-version: ["3.10", "3.12", "3.14"]' in python_job
    assert "python-version: ${{ matrix.python-version }}" in python_job
    assert "fail-fast: false" in python_job
    assert "python -m pip install --require-hashes -r requirements/ci.txt" in python_job
    assert "Verify Rust core is the resolved default backend" in python_job
    assert "- run: pytest" in python_job
