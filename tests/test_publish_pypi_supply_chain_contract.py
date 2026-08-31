"""Contract tests for release SBOM and build-provenance evidence."""

from __future__ import annotations

from pathlib import Path


_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "publish-pypi.yml"
_ATTEST_BUILD_PROVENANCE_SHA = "977bb373ede98d70efdf65b84cb5f73e068dcc2a"
_SBOM_ACTION_SHA = "e22c389904149dbc22b58101806040fa8d37a610"
_SYFT_VERSION = "v1.51.1"


def _workflow_text() -> str:
    """Return the package-publication workflow as UTF-8 text."""
    return _WORKFLOW.read_text(encoding="utf-8")


def _job_block(text: str, name: str, next_name: str | None = None) -> str:
    """Return one top-level workflow job without requiring a YAML dependency."""
    marker = f"\n  {name}:\n"
    start = text.index(marker)
    if next_name is None:
        return text[start:]
    end = text.index(f"\n  {next_name}:\n", start + len(marker))
    return text[start:end]


def _assert_builder_local_attestation(block: str, build_step: str) -> None:
    """Require build provenance to be emitted before the artifact leaves its builder."""
    attest = f"uses: actions/attest-build-provenance@{_ATTEST_BUILD_PROVENANCE_SHA}"
    upload = "uses: actions/upload-artifact@"
    assert "contents: read" in block
    assert "attestations: write" in block
    assert "id-token: write" in block
    assert attest in block
    assert "subject-path: dist/*" in block
    assert block.index(build_step) < block.index(attest) < block.index(upload)


def test_sdist_and_wheels_emit_builder_local_provenance():
    """Each distribution is attested by the job that actually built it."""
    text = _workflow_text()
    _assert_builder_local_attestation(
        _job_block(text, "sdist", "wheels"), "- name: Build sdist"
    )
    _assert_builder_local_attestation(
        _job_block(text, "wheels", "sbom"), "- name: Build wheel"
    )


def test_sbom_is_exact_release_source_bound_spdx_and_attested():
    """The released SBOM is generated and attested from the reviewed source commit."""
    text = _workflow_text()
    block = _job_block(text, "sbom", "release-assets")
    attest = f"uses: actions/attest-build-provenance@{_ATTEST_BUILD_PROVENANCE_SHA}"
    sbom = f"uses: anchore/sbom-action@{_SBOM_ACTION_SHA}"

    assert "needs: verify-release" in block
    assert "contents: read" in block
    assert "attestations: write" in block
    assert "id-token: write" in block
    assert "ref: ${{ inputs.release_commit }}" in block
    assert "persist-credentials: false" in block
    assert sbom in block
    assert "path: ." in block
    assert "format: spdx-json" in block
    assert f"syft-version: {_SYFT_VERSION}" in block
    assert "output-file: dist/fast-mlsirm.spdx.json" in block
    assert 'upload-artifact: "false"' in block
    assert 'upload-release-assets: "false"' in block
    assert attest in block
    assert "subject-path: dist/fast-mlsirm.spdx.json" in block
    assert block.index(sbom) < block.index(attest)
    assert "name: release-sbom" in block
    assert "path: dist/fast-mlsirm.spdx.json" in block


def test_release_assets_attach_sbom_without_leaking_it_into_pypi_upload():
    """SBOM success gates PyPI, but its JSON never enters the upload directory."""
    text = _workflow_text()
    release_assets = _job_block(text, "release-assets", "publish-pypi")
    publish = _job_block(text, "publish-pypi")

    assert "needs: [sdist, wheels, sbom]" in release_assets
    assert "pattern: dist-*" in release_assets
    assert "name: release-sbom" in release_assets
    assert "path: dist" in release_assets
    assert 'gh release upload "$RELEASE_TAG" dist/*' in release_assets

    assert "needs: [sdist, wheels, sbom]" in publish
    assert "pattern: dist-*" in publish
    assert "release-sbom" not in publish
    assert "attestations: false" in publish
