from pathlib import Path

import pytest

from scripts import build_acquisition_release as subject


def test_acquisition_output_cannot_cover_repository_source(tmp_path: Path) -> None:
    """Reject output roots whose cleanliness exception contains source files."""
    repo_root = (tmp_path / "repo").resolve()
    repo_root.mkdir()

    for unsafe_output in (repo_root, tmp_path.resolve()):
        with pytest.raises(
            RuntimeError,
            match="output directory must not be the repository root or an ancestor",
        ):
            subject._admit_output_root(repo_root, unsafe_output)


def test_acquisition_output_accepts_nested_or_external_directories(tmp_path: Path) -> None:
    """Permit bounded repository-local output and disjoint external output."""
    repo_root = (tmp_path / "repo").resolve()
    repo_root.mkdir()
    nested_output = repo_root / "acquisition-release"
    external_output = tmp_path / "external-evidence"

    assert subject._admit_output_root(repo_root, nested_output) == nested_output.resolve()
    assert subject._admit_output_root(repo_root, external_output) == external_output.resolve()
