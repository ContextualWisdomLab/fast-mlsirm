"""Repository-owned PR workflows must use an explicit supported Ubuntu image.

A floating ``ubuntu-latest`` selector has repeatedly remained pre-checkout with
``runner_id=0`` while explicit ``ubuntu-24.04`` jobs in sibling repositories
have acquired GitHub-hosted runners. This contract keeps the repository-owned
CI and CodeQL gates on the reviewed runner image rather than silently drifting
with GitHub's floating alias.
"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PR_WORKFLOWS = (
    REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml",
    REPOSITORY_ROOT / ".github" / "workflows" / "codeql.yml",
)


def test_required_pr_workflows_use_explicit_ubuntu_2404() -> None:
    """Require every Linux runner in the repository-owned PR gates to be pinned."""
    for workflow in PR_WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert "runs-on: ubuntu-latest" not in source, workflow
        assert "runs-on: ubuntu-24.04" in source, workflow
