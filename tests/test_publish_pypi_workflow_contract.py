from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
RELEASE_TAG_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-tag.yml"
PYPI_PUBLISH_SHA = "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _release_tag_workflow_text() -> str:
    return RELEASE_TAG_WORKFLOW.read_text(encoding="utf-8")


def _job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
    )
    assert match is not None, f"missing {name!r} job"
    return match.group(0)


def test_release_builds_are_bound_to_the_reviewed_source_commit() -> None:
    text = _workflow_text()
    verify = _job_block(text, "verify-release")
    sdist = _job_block(text, "sdist")
    wheels = _job_block(text, "wheels")

    assert "      release_commit:\n" in text
    assert re.search(
        r"(?m)^      release_commit:\n(?:        .*\n)*?        required: true$",
        text,
    )

    for job in (verify, sdist, wheels):
        assert "ref: ${{ inputs.release_commit }}" in job
        assert "persist-credentials: false" in job
        assert "ref: ${{ inputs.release_tag }}" not in job

    assert "fetch-depth: 0" in verify
    assert 'RELEASE_TAG: ${{ inputs.release_tag }}' in verify
    assert 'RELEASE_COMMIT: ${{ inputs.release_commit }}' in verify
    assert "release_commit must be a canonical 40-character lowercase SHA-1" in verify
    assert 'git rev-parse HEAD' in verify
    assert 'git rev-parse "$RELEASE_TAG^{commit}"' in verify
    assert "checked-out release source does not match release_commit" in verify
    assert "release tag does not target release_commit" in verify
    assert 'tomllib.load' in verify or 'tomllib.loads' in verify
    assert 'f"v{project[\'version\']}"' in verify
    assert text.count("maturin-version: v1.14.1") == 2


def test_publication_requires_exact_unpublished_draft_before_destructive_retry() -> None:
    """Direct dispatch must not let draft-recovery semantics mutate a published release."""
    verify = _job_block(_workflow_text(), "verify-release")

    assert "- name: Require exact draft release state" in verify
    assert "GH_TOKEN: ${{ github.token }}" in verify
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in verify
    assert "gh api --paginate --slurp" in verify
    assert 'releases?per_page=100' in verify
    assert 'release.get("tag_name") == expected_tag' in verify
    assert "len(matches) != 1" in verify
    assert 'release.get("draft") is not True' in verify
    assert "publication requires exactly one matching unpublished draft release" in verify


def test_wheels_cover_supported_cpython_versions_on_every_platform() -> None:
    wheels = _job_block(_workflow_text(), "wheels")

    # The extension is not built with PyO3 abi3, so a CPython 3.12 wheel cannot
    # satisfy 3.13/3.14 callers. Each release platform must build all currently
    # evidenced supported CPython versions instead of forcing newer callers to
    # compile the sdist with a local Rust toolchain.
    for version in ("3.12", "3.13", "3.14"):
        assert wheels.count(f'python-version: "{version}"') == 4
        assert wheels.count(f"interpreter: python{version}") == 2

    assert wheels.count("interpreter: python\n") == 6
    assert "python-version: ${{ matrix.python-version }}" in wheels
    assert "args: --release --out dist -i ${{ matrix.interpreter }}" in wheels
    assert "name: dist-wheel-${{ matrix.target }}-py${{ matrix.python-version }}" in wheels


def test_release_tag_workflow_explicitly_dispatches_package_publish() -> None:
    publish_text = _workflow_text()
    release_text = _release_tag_workflow_text()
    release_job = _job_block(release_text, "publish-release-tag")
    verify = _job_block(publish_text, "verify-release")

    # GITHUB_TOKEN-created release events do not recursively start ordinary
    # event-triggered workflows. Package publication therefore uses the one
    # supported recursive trigger. The control-plane workflow comes from the
    # protected default branch while artifact source identity is an immutable
    # explicit commit, so an older release tag cannot select an outdated
    # publication workflow definition. The dispatch also carries the exact
    # release-tag run commit so a moving default branch cannot silently select a
    # different publication control plane between verification and dispatch.
    assert "  workflow_dispatch:\n" in publish_text
    assert "      release_tag:\n" in publish_text
    assert "      release_commit:\n" in publish_text
    assert "      control_plane_commit:\n" in publish_text
    assert publish_text.count("        required: true\n") >= 3
    assert "types: [published]" not in publish_text

    assert 'DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}' in verify
    assert 'PUBLISH_REF: ${{ github.ref }}' in verify
    assert 'CONTROL_PLANE_COMMIT: ${{ inputs.control_plane_commit }}' in verify
    assert 'CONTROL_PLANE_SHA: ${{ github.sha }}' in verify
    assert 'expected_ref="refs/heads/$DEFAULT_BRANCH"' in verify
    assert 'if [ "$PUBLISH_REF" != "$expected_ref" ]' in verify
    assert 'if [ "$CONTROL_PLANE_SHA" != "$CONTROL_PLANE_COMMIT" ]' in verify
    assert "publication control plane moved after release verification" in verify

    assert "permissions:\n      contents: write\n      actions: write" in release_job
    assert "gh workflow run publish-pypi.yml" in release_job
    assert 'DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}' in release_job
    assert 'CONTROL_PLANE_COMMIT: ${{ github.sha }}' in release_job
    assert '--ref "$DEFAULT_BRANCH"' in release_job
    assert '--ref "v$RELEASE_VERSION"' not in release_job
    assert '-f release_tag="v$RELEASE_VERSION"' in release_job
    assert '-f release_commit="$RELEASE_COMMIT"' in release_job
    assert '-f control_plane_commit="$CONTROL_PLANE_COMMIT"' in release_job
    assert 'RELEASE_COMMIT: ${{ inputs.release_commit }}' in release_job
    assert release_job.index('gh release create "v$RELEASE_VERSION"') < release_job.index(
        "gh workflow run publish-pypi.yml"
    )


def test_release_asset_write_is_isolated_and_recoverable_while_draft() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    assert "permissions:\n      contents: write" in assets
    assert "GH_TOKEN: ${{ github.token }}" in assets
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in assets
    assert 'gh release upload "$RELEASE_TAG"' in assets
    # The release is deliberately still a draft here. A retry after a prior
    # successful asset job must replace same-name draft assets instead of
    # deadlocking publication before the independently retryable PyPI sink.
    assert "--clobber" in assets
    assert "secrets.PIPY_TOKEN" not in assets

    assert "environment: pypi" in publish
    assert "permissions:\n      contents: read" in publish
    assert "gh release upload" not in publish
    assert "contents: write" not in publish


def test_release_mutations_replay_draft_state_at_the_mutation_seam() -> None:
    """A stale early draft check must not authorize later irreversible publication."""
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")
    finalize = _job_block(text, "finalize-release")

    for block, mutation in (
        (assets, 'gh release upload "$RELEASE_TAG"'),
        (publish, f"uses: pypa/gh-action-pypi-publish@{PYPI_PUBLISH_SHA}"),
        (finalize, 'gh release edit "$RELEASE_TAG"'),
    ):
        assert "- name: Revalidate exact draft release state" in block
        assert "gh api --paginate --slurp" in block
        assert 'release.get("tag_name") == expected_tag' in block
        assert "len(matches) != 1" in block
        assert 'release.get("draft") is not True' in block
        assert block.index("- name: Revalidate exact draft release state") < block.index(
            mutation
        )


def test_pypi_publish_uses_a_pinned_package_owned_uploader() -> None:
    text = _workflow_text()
    publish = _job_block(text, "publish-pypi")

    assert "python -m pip install" not in text
    assert "python -m twine upload" not in text
    assert "TWINE_USERNAME" not in text
    assert f"uses: pypa/gh-action-pypi-publish@{PYPI_PUBLISH_SHA}" in publish
    assert "password: ${{ secrets.PIPY_TOKEN }}" in publish
    assert "attestations: false" in publish
    assert "skip-existing" not in publish


def test_pypi_publish_can_recover_independently_of_draft_asset_upload() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    # Both publication sinks require the same verified builds and successful
    # SBOM/provenance evidence. They remain independent after that prerequisite;
    # release-assets is retry-safe while the GitHub release is still a draft.
    assert "needs: [sdist, wheels, sbom]" in assets
    assert "needs: [sdist, wheels, sbom]" in publish
    assert "release-assets" not in publish.split("needs:", 1)[1].split("\n", 1)[0]
    assert "--clobber" in assets


def test_new_release_sbom_job_uses_the_explicit_linux_runner_contract() -> None:
    """New release-evidence jobs must not reintroduce a moving Ubuntu alias."""
    sbom = _job_block(_workflow_text(), "sbom")

    assert "runs-on: ubuntu-24.04" in sbom
    assert "runs-on: ubuntu-latest" not in sbom


def test_immutable_release_is_published_only_after_assets_and_pypi_succeed() -> None:
    """Assets must be attached while the release is still mutable, then sealed once."""
    publish_text = _workflow_text()
    release_job = _job_block(_release_tag_workflow_text(), "publish-release-tag")
    finalize = _job_block(publish_text, "finalize-release")

    assert 'gh release create "v$RELEASE_VERSION"' in release_job
    assert "--draft" in release_job
    assert release_job.index('gh release create "v$RELEASE_VERSION"') < release_job.index(
        "gh workflow run publish-pypi.yml"
    )

    assert "needs: [release-assets, publish-pypi]" in finalize
    assert "permissions:\n      contents: write" in finalize
    assert "GH_TOKEN: ${{ github.token }}" in finalize
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in finalize
    assert 'gh release edit "$RELEASE_TAG" --repo "${{ github.repository }}" --draft=false' in finalize
