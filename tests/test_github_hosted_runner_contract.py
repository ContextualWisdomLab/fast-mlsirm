"""Repository-owned Linux workflows must use explicit supported Ubuntu images.

A floating ``ubuntu-latest`` selector has repeatedly remained pre-checkout with
``runner_id=0`` while explicit ``ubuntu-24.04`` jobs in sibling repositories
have acquired GitHub-hosted runners. Required PR gates, scheduled scientific
evidence, governance automation, and release controls therefore pin Linux
runner identity instead of silently drifting with GitHub's floating alias.
"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
PR_WORKFLOWS = (
    WORKFLOW_DIRECTORY / "ci.yml",
    WORKFLOW_DIRECTORY / "codeql.yml",
)


def test_required_pr_workflows_use_explicit_ubuntu_2404() -> None:
    """Require every Linux runner in the repository-owned PR gates to be pinned."""
    for workflow in PR_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert "runs-on: ubuntu-latest" not in source, workflow
        assert "runs-on: ubuntu-24.04" in source, workflow


def test_repository_workflows_do_not_float_ubuntu_runner_identity() -> None:
    """Reject floating Ubuntu selectors across all repository-owned workflows."""
    workflow_paths = tuple(sorted(WORKFLOW_DIRECTORY.glob("*.yml")))
    assert workflow_paths, "repository workflow inventory must not be empty"

    offenders = [
        workflow.relative_to(REPOSITORY_ROOT).as_posix()
        for workflow in workflow_paths
        if "ubuntu-latest" in workflow.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"floating Ubuntu runner selectors remain: {offenders}"
