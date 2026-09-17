"""Contracts for fail-closed recovery after a partial PyPI upload."""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
PYPI_PUBLISH_SHA = "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"


def _publish_job() -> str:
    """Return the publication job without adding a YAML parser dependency."""
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^  publish-pypi:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
    )
    assert match is not None, "missing publish-pypi job"
    return match.group(0)


def test_partial_pypi_retry_is_digest_bound_before_duplicate_suppression() -> None:
    """A retry may skip only artifacts PyPI already stores with the exact local digest."""
    publish = _publish_job()
    prepare = "- name: Prepare digest-bound PyPI retry set"
    upload = f"uses: pypa/gh-action-pypi-publish@{PYPI_PUBLISH_SHA}"
    verify = "- name: Verify complete PyPI artifact set"

    assert prepare in publish
    assert "id: pypi-recovery" in publish
    assert 'https://pypi.org/pypi/fast-mlsirm/${release_version}/json' in publish
    assert "hashlib.sha256" in publish
    assert "unexpected PyPI files for this release" in publish
    assert "published PyPI digest does not match the reviewed artifact" in publish
    assert 'needs_publish={str(bool(missing)).lower()}' in publish

    # The official uploader still fails loudly on ordinary duplicates. Matching
    # files are removed from this job's upload directory only after the digest
    # preflight proves they are the exact artifacts already stored by PyPI.
    assert "skip-existing" not in publish
    assert "already-published" in publish
    assert "local_path.replace" in publish
    assert "if: steps.pypi-recovery.outputs.needs_publish == 'true'" in publish

    # Final GitHub release publication must not be authorized merely because the
    # uploader returned success: PyPI must expose the complete exact filename /
    # SHA-256 set after the upload or an all-already-published recovery.
    assert verify in publish
    assert "PyPI artifact set is incomplete or contains unexpected files" in publish
    assert "PyPI artifact digest mismatch after publication" in publish
    assert publish.index(prepare) < publish.index(upload) < publish.index(verify)
