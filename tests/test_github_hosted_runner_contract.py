"""Repository-owned Linux workflows must use explicit supported Ubuntu images.

A floating ``ubuntu-latest`` selector has repeatedly remained pre-checkout with
``runner_id=0`` while explicit ``ubuntu-24.04`` jobs in sibling repositories
have acquired GitHub-hosted runners. Required PR gates, scheduled scientific
evidence, governance automation, and release controls therefore pin Linux
runner identity instead of silently drifting with GitHub's floating alias.
"""

from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
PR_WORKFLOWS = (
    WORKFLOW_DIRECTORY / "ci.yml",
    WORKFLOW_DIRECTORY / "codeql.yml",
)
QUEUE_SENSITIVE_PR_WORKFLOWS = (WORKFLOW_DIRECTORY / "codeql.yml",)
FLOATING_RUNNER_ASSIGNMENT = re.compile(
    r"(?m)^\s*runs-on:\s*ubuntu-latest\s*(?:#.*)?$"
)


def _workflow_paths(directory: Path) -> tuple[Path, ...]:
    """Return every GitHub workflow regardless of the accepted YAML extension."""
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix in {".yml", ".yaml"}
        )
    )


def test_required_pr_workflows_use_explicit_ubuntu_2404() -> None:
    """Require every Linux runner in the repository-owned PR gates to be pinned."""
    for workflow in PR_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert FLOATING_RUNNER_ASSIGNMENT.search(source) is None, workflow
        assert "runs-on: ubuntu-24.04" in source, workflow


def test_workflow_inventory_includes_yml_and_yaml(tmp_path: Path) -> None:
    """A workflow cannot evade runner policy by choosing the other YAML suffix."""
    yml = tmp_path / "first.yml"
    yaml = tmp_path / "second.yaml"
    ignored = tmp_path / "README.md"
    for path in (yml, yaml, ignored):
        path.write_text("name: fixture\n", encoding="utf-8")

    assert _workflow_paths(tmp_path) == (yml, yaml)


def test_repository_workflows_do_not_float_ubuntu_runner_identity() -> None:
    """Reject only active floating runner assignments, not harmless prose/data."""
    workflow_paths = _workflow_paths(WORKFLOW_DIRECTORY)
    assert workflow_paths, "repository workflow inventory must not be empty"

    offenders = [
        workflow.relative_to(REPOSITORY_ROOT).as_posix()
        for workflow in workflow_paths
        if FLOATING_RUNNER_ASSIGNMENT.search(workflow.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"floating Ubuntu runner selectors remain: {offenders}"


def test_queue_sensitive_pr_workflows_cancel_predecessor_heads() -> None:
    """A superseded PR head must not retain scarce hosted-runner queue capacity."""
    expected_group = (
        "group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}"
    )
    for workflow in QUEUE_SENSITIVE_PR_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert "\nconcurrency:\n" in source, workflow
        concurrency = source.split("\nconcurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        assert expected_group in concurrency, workflow
        assert "cancel-in-progress: true" in concurrency, workflow


def test_required_python_context_cannot_pass_without_gpu_parity() -> None:
    """The protected ``python`` context must fail when explicit GPU parity fails."""
    source = (WORKFLOW_DIRECTORY / "ci.yml").read_text(encoding="utf-8")
    python_job = source.split("\n  python:\n", 1)[1].split("\n\n  rust:\n", 1)[0]

    assert "needs: [python-matrix, gpu-smoke]" in python_job
    assert 'test "${{ needs.python-matrix.result }}" = "success"' in python_job
    assert 'test "${{ needs.gpu-smoke.result }}" = "success"' in python_job
