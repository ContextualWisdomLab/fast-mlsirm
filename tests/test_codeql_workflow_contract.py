"""Regression contract for repository-managed CodeQL workflow ownership."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "codeql.yml"
GOVERNANCE_INDEX_PATH = REPO_ROOT / "docs" / "GOVERNANCE_INDEX.md"
THREAT_MODEL_PATH = REPO_ROOT / "docs" / "security" / "threat-model.md"


def test_default_setup_owns_required_actions_codeql_check() -> None:
    """Keep the required check supplied by one CodeQL execution path."""
    assert not WORKFLOW_PATH.exists()

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (GOVERNANCE_INDEX_PATH, THREAT_MODEL_PATH)
    )
    assert "CodeQL Default setup" in documentation
    assert "`Analyze (actions)`" in documentation
    assert "disabling default setup would block every pr" in documentation.casefold()
