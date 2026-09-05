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


def test_release_dispatch_requires_an_explicit_source_commit():
    """A release is bound to its reviewed cut even if default-branch HEAD moves."""
    text = _workflow_text()
    validation = "Validate release source identity"
    checkout = "actions/checkout@"
    assert "release_commit:" in text
    assert "Release source commit" in text
    assert "required: true" in text
    assert "RELEASE_COMMIT: ${{ inputs.release_commit }}" in text
    assert "release_commit must be a canonical" in text
    assert "merge-base --is-ancestor" in text
    assert "release commit must be an ancestor of the default branch" in text
    assert "ref: ${{ inputs.release_commit }}" in text
    assert text.index(validation) < text.index(checkout)


def test_release_version_and_changelog_section_fail_closed():
    """The requested canonical version matches exactly one nonempty release section."""
    text = _workflow_text()
    assert 'r"(?:0|[1-9][0-9]*)\\."' in text
    assert text.count('r"(?:0|[1-9][0-9]*)') == 3
    assert "without leading zeros" in text
    assert "pyproject.toml version" in text
    assert "expected exactly one CHANGELOG section" in text
    assert "if len(matches) != 1:" in text
    assert "grep -q '[^[:space:]]' release_notes.md" in text


def test_release_source_is_the_version_cut_transition_not_a_later_descendant():
    """A same-version post-cut descendant cannot be selected as the release source."""
    text = _workflow_text()
    version_step = "Verify the requested version is the released source version"
    transition_step = "Verify the release source is the version-cut transition"
    tag_state_step = "Verify release state and classify tag recovery"
    assert transition_step in text
    assert 'git rev-parse "$RELEASE_COMMIT^"' in text
    assert "release commit must have a verifiable first parent" in text
    assert "parent project version already equals requested release version" in text
    assert "parent CHANGELOG already contains requested release section" in text
    assert text.index(version_step) < text.index(transition_step) < text.index(tag_state_step)


def test_published_release_and_api_uncertainty_block_publication():
    """Published release state and API uncertainty must fail closed."""
    text = _workflow_text()
    assert 'gh api --paginate --slurp "repos/$GITHUB_REPOSITORY/releases?per_page=100"' in text
    assert '"releases/tags/v$RELEASE_VERSION"' not in text
    assert '"git/ref/tags/v$RELEASE_VERSION"' in text
    assert "GitHub API returned HTTP $status" in text
    assert "is already published; refusing to overwrite or reuse it" in text


def test_existing_matching_draft_can_resume_before_publication():
    """An interrupted draft is reusable only with the same release identity and tag."""
    text = _workflow_text()
    assert "resume_existing_release=false" in text
    assert "resume_existing_release=true" in text
    assert 'release.get("draft") is not True' in text
    assert 'release.get("tag_name") != expected_name' in text
    assert 'release.get("name") != expected_name' in text
    assert "existing draft release identity does not match the requested release" in text
    assert "existing draft release is missing its required release tag" in text
    assert "resuming exact-tag publication" in text
    guard = "if: steps.release_tag_state.outputs.resume_existing_release != 'true'"
    create_step = "Create the draft GitHub release from the verified tag"
    assert guard in text
    assert text.index(create_step) < text.index(guard) + len(guard)
    assert abs(text.index(guard) - text.index(create_step)) < 200


def test_tag_without_release_resumes_only_at_the_requested_source_commit():
    """An interrupted run resumes only when the immutable tag targets the source."""
    text = _workflow_text()
    assert "resume_existing_tag=true" in text
    assert "resume_existing_tag=false" in text
    assert "tag v$RELEASE_VERSION exists and targets the requested immutable source" in text
    assert "existing tag does not target the requested release commit" in text
    assert 'actual_tag_sha != os.environ["RELEASE_COMMIT"]' in text
    guard = "if: steps.release_tag_state.outputs.resume_existing_tag != 'true'"
    create_step = "Atomically create the immutable release tag"
    assert guard in text
    assert text.index(create_step) < text.index(guard) + len(guard)
    assert abs(text.index(guard) - text.index(create_step)) < 200


def test_oversized_release_notes_are_capped_with_authoritative_pointer():
    """A CHANGELOG section beyond the API body limit becomes a linked summary."""
    text = _workflow_text()
    assert "body_limit = 120_000" in text
    assert "release-body limit" in text
    assert "Full authoritative notes" in text
    assert "CHANGELOG.md" in text
    assert 'line.startswith("#### ")' in text
    assert "(contents list truncated)" in text


def test_release_tag_is_created_atomically_at_the_explicit_source_commit():
    """The tag creation API binds the immutable ref to the reviewed release cut."""
    text = _workflow_text()
    create_step = "Atomically create the immutable release tag"
    post_ref = '"$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/git/refs"'
    assert create_step in text
    assert '"ref": f"refs/tags/v{os.environ[\'RELEASE_VERSION\']}"' in text
    assert '"sha": os.environ["RELEASE_COMMIT"]' in text
    assert '"sha": os.environ["GITHUB_SHA"]' not in text
    assert "--request POST" in text
    assert '--data-binary "@$request_file"' in text
    assert post_ref in text
    assert 'if [ "$status" != "201" ]' in text
    assert 'actual_sha != os.environ["RELEASE_COMMIT"]' in text


def test_release_creation_requires_the_verified_existing_tag_and_stays_draft():
    """Assets remain attachable until the downstream publication workflow finalizes."""
    text = _workflow_text()
    preflight = "Verify release state and classify tag recovery"
    atomic_create = "Atomically create the immutable release tag"
    create_release = 'gh release create "v$RELEASE_VERSION"'
    assert create_release in text
    assert "--verify-tag" in text
    assert "--draft" in text
    assert '--target "$GITHUB_SHA"' not in text
    assert "--notes-file release_notes.md" in text
    assert text.index(preflight) < text.index(atomic_create) < text.index(create_release)
