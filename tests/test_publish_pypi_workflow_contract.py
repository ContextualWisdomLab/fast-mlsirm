from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
PYPI_PUBLISH_SHA = "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


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
        assert "ref: ${{ github.event.release.tag_name }}" in job
        assert "persist-credentials: false" in job

    assert 'RELEASE_TAG: ${{ github.event.release.tag_name }}' in verify
    assert 'tomllib.load' in verify or 'tomllib.loads' in verify
    assert 'f"v{project[\'version\']}"' in verify
    assert text.count("maturin-version: v1.14.1") == 2


def test_release_asset_write_is_isolated_from_pypi_credentials() -> None:
    text = _workflow_text()
    assets = _job_block(text, "release-assets")
    publish = _job_block(text, "publish-pypi")

    assert "permissions:\n      contents: write" in assets
    assert "GH_TOKEN: ${{ github.token }}" in assets
    assert "RELEASE_TAG: ${{ github.event.release.tag_name }}" in assets
    assert 'gh release upload "$RELEASE_TAG"' in assets
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
