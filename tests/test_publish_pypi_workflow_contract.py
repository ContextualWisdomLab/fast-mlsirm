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


def test_release_builds_are_bound_to_the_published_version() -> None:
    text = _workflow_text()
    verify = _job_block(text, "verify-release")
    sdist = _job_block(text, "sdist")
    wheels = _job_block(text, "wheels")

    for job in (verify, sdist, wheels):
        assert "ref: ${{ inputs.release_tag }}" in job
        assert "persist-credentials: false" in job

    assert 'RELEASE_TAG: ${{ inputs.release_tag }}' in verify
    assert 'tomllib.load' in verify or 'tomllib.loads' in verify
    assert 'f"v{project[\'version\']}"' in verify
    assert text.count("maturin-version: v1.14.1") == 2


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

    # GITHUB_TOKEN-created release events do not recursively start ordinary
    # event-triggered workflows. Package publication therefore uses the one
    # supported recursive trigger: workflow_dispatch, bound to the immutable tag.
    assert "  workflow_dispatch:\n" in publish_text
    assert "      release_tag:\n" in publish_text
    assert "        required: true\n" in publish_text
    assert "types: [published]" not in publish_text

    assert "permissions:\n      contents: write\n      actions: write" in release_job
    assert "gh workflow run publish-pypi.yml" in release_job
    assert '--ref "v$RELEASE_VERSION"' in release_job
    assert '-f release_tag="v$RELEASE_VERSION"' in release_job
    assert release_job.index('gh release create "v$RELEASE_VERSION"') < release_job.index(
        "gh workflow run publish-pypi.yml"
    )


def test_release_asset_write_is_isolated_from_pypi_credentials() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    assert "permissions:\n      contents: write" in assets
    assert "GH_TOKEN: ${{ github.token }}" in assets
    assert "RELEASE_TAG: ${{ inputs.release_tag }}" in assets
    assert 'gh release upload "$RELEASE_TAG"' in assets
    assert "--clobber" not in assets
    assert "secrets.PIPY_TOKEN" not in assets

    assert "environment: pypi" in publish
    assert "permissions:\n      contents: read" in publish
    assert "gh release upload" not in publish
    assert "contents: write" not in publish


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


def test_pypi_publish_can_recover_independently_of_immutable_asset_upload() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    # Release assets and PyPI are two independent publication sinks fed by the
    # same verified build artifacts. A rerun after GitHub assets already exist
    # must still be able to retry a previously failed PyPI publication rather
    # than being skipped because immutable asset upload correctly fails closed.
    assert "needs: [sdist, wheels]" in assets
    assert "needs: [sdist, wheels]" in publish
    assert "release-assets" not in publish.split("needs:", 1)[1].split("\n", 1)[0]
