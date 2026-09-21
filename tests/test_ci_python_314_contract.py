"""Contracts for the supported Python versions exercised by pull-request CI."""

from pathlib import Path


_CI_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
_PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _python_matrix_job_source() -> str:
    """Return the matrix Python suite job without unrelated CI jobs."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  python-matrix:\n")
    end = workflow.index("\n  python:\n", start)
    return workflow[start:end]


def _python_gate_job_source() -> str:
    """Return the required-status aggregate job named exactly ``python``."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  python:\n")
    end = workflow.index("\n  rust:\n", start)
    return workflow[start:end]


def _gpu_executor_tests() -> list[str]:
    """Return the workflow-level GPU-executor node IDs owned by gpu-smoke."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    block = workflow.split("  GPU_EXECUTOR_TESTS: >-\n", 1)[1].split("\n\n", 1)[0]
    return [line.strip() for line in block.splitlines()]


def test_python_job_runs_full_suite_on_supported_cpythons() -> None:
    """Pull-request CI must execute the full fail-slow suite on supported CPythons.

    The matrix deselects only the GPU-executor nodes, which gpu-smoke runs on
    the same CPythons; nothing else may be excluded.
    """
    python_job = _python_matrix_job_source()

    assert 'python-version: ["3.12", "3.14"]' in python_job
    assert "python-version: ${{ matrix.python-version }}" in python_job
    assert "fail-fast: false" in python_job
    assert 'for node in $GPU_EXECUTOR_TESTS; do args+=(--deselect "$node"); done' in python_job
    assert 'pytest "${args[@]}"' in python_job


def test_gpu_executor_nodes_are_conserved_in_gpu_smoke() -> None:
    """Every deselected node exists and runs in gpu-smoke under a zero-skip gate."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    nodes = _gpu_executor_tests()
    assert len(nodes) == len(set(nodes)) == 7
    root = _CI_WORKFLOW.parents[2]
    for node in nodes:
        path, name = node.split("::")
        assert f"def {name}(" in (root / path).read_text(encoding="utf-8"), node
    gpu_job = workflow.split("\n  gpu-smoke:\n", 1)[1].split("\n  fuzz:\n", 1)[0]
    assert 'python-version: ["3.12", "3.14"]' in gpu_job
    assert 'STAGE5_HIGH_Q: "1"' in gpu_job
    assert "pytest $GPU_EXECUTOR_TESTS" in gpu_job
    assert "if executed != expected:" in gpu_job
    assert '"no usable GPU adapter was found" in output' in gpu_job


def test_advertised_python_floor_matches_ci_and_dependency_lock() -> None:
    """pyproject must not advertise a floor below the CI matrix / installable lock."""
    pyproject = _PYPROJECT.read_text(encoding="utf-8")
    python_job = _python_matrix_job_source()

    assert 'requires-python = ">=3.12"' in pyproject
    assert 'python-version: ["3.12", "3.14"]' in python_job
    # Hashed CI deps currently pin NumPy 2.5.x which needs CPython >=3.12.
    assert "3.10" not in python_job


def test_required_python_check_context_aggregates_matrix() -> None:
    """Branch protection requires the exact check name ``python``.

    Matrix legs alone report as versioned Python checks and do not satisfy that
    context. A non-matrix aggregate job must re-export matrix success under the
    protected name.
    """
    gate = _python_gate_job_source()
    assert "needs: [python-matrix, gpu-smoke]" in gate
    assert "name: python" in gate
    assert 'test "${{ needs.python-matrix.result }}" = "success"' in gate
    assert 'test "${{ needs.gpu-smoke.result }}" = "success"' in gate
