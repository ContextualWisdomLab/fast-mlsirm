"""Contract tests for the fail-closed manual release-tag workflow."""

from __future__ import annotations

from pathlib import Path


_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "release-tag.yml"


def _workflow_text() -> str:
    """Return the release-tag workflow as UTF-8 text."""
    return _WORKFLOW.read_text(encoding="utf-8")


def test_release_tag_workflow_is_manual_least_privilege_and_serialized():
    """Publishing is explicit, globally serialized, and write-scoped to one job."""
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "push:" not in text
    assert "permissions:\n  contents: read" in text
    assert text.count("contents: write") == 1
    assert "group: release-tag\n" in text
    assert "cancel-in-progress: false" in text
    assert "timeout-minutes: 10" in text


def test_release_dispatch_must_target_the_repository_default_branch():
    """A branch containing release metadata cannot publish itself as a release."""
    text = _workflow_text()
    guard = "expected_ref=\"refs/heads/$DEFAULT_BRANCH\""
    checkout = "actions/checkout@"
    assert "github.event.repository.default_branch" in text
    assert "github.ref" in text
    assert guard in text
    assert "release dispatch must target" in text
    assert text.index(guard) < text.index(checkout)


def test_release_version_and_changelog_section_fail_closed():
    """The requested version matches source and exactly one nonempty release note section."""
    text = _workflow_text()
    assert "^[0-9]+\\.[0-9]+\\.[0-9]+$" in text
    assert "pyproject.toml version" in text
    assert "expected exactly one CHANGELOG section" in text
    assert "if len(matches) != 1:" in text
    assert "grep -q '[^[:space:]]' release_notes.md" in text


def test_existing_tag_or_release_and_api_uncertainty_block_publication():
    """Neither an existing Git ref nor an API failure can be treated as absence."""
    text = _workflow_text()
    assert 'check_absent "git/ref/tags/v$RELEASE_VERSION"' in text
    assert 'check_absent "releases/tags/v$RELEASE_VERSION"' in text
    assert "404)" in text
    assert "200)" in text
    assert "GitHub API returned HTTP $status" in text
    assert "refusing to overwrite or reuse it" in text


def test_release_creation_targets_the_verified_dispatch_commit():
    """The immutable workflow dispatch SHA is the target of the new release tag."""
    text = _workflow_text()
    assert 'gh release create "v$RELEASE_VERSION"' in text
    assert '--target "$GITHUB_SHA"' in text
    assert '--notes-file release_notes.md' in text
