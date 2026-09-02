"""Regression tests for release-acceptance source sealing around built distributions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import release_acceptance as subject


def _status(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["git", "status"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def test_clean_source_allows_only_inert_artifacts_under_admitted_distribution_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A custom non-ignored dist is an explicit narrow exception, not a global suffix rule."""
    repo_root = tmp_path / "repo"
    dist_dir = repo_root / "custom-dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "fast_mlsirm-0.9.2-py3-none-any.whl").write_bytes(b"wheel")
    (dist_dir / "fast_mlsirm-0.9.2.tar.gz").write_bytes(b"sdist")
    status = "?? custom-dist/fast_mlsirm-0.9.2-py3-none-any.whl\0?? custom-dist/fast_mlsirm-0.9.2.tar.gz\0"
    monkeypatch.setattr(subject.subprocess, "run", lambda *args, **kwargs: _status(status))

    subject._require_clean_source(repo_root, allowed_distribution_root=dist_dir)


def test_distribution_artifact_suffix_outside_admitted_root_remains_source_violation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unrelated archive cannot evade source sealing merely by looking like a distribution."""
    repo_root = tmp_path / "repo"
    dist_dir = repo_root / "custom-dist"
    fixture_dir = repo_root / "fixtures"
    dist_dir.mkdir(parents=True)
    fixture_dir.mkdir(parents=True)
    (dist_dir / "fast_mlsirm-0.9.2.whl").write_bytes(b"wheel")
    (fixture_dir / "adversarial-source.tar.gz").write_bytes(b"source fixture")
    status = "?? custom-dist/fast_mlsirm-0.9.2.whl\0?? fixtures/adversarial-source.tar.gz\0"
    monkeypatch.setattr(subject.subprocess, "run", lambda *args, **kwargs: _status(status))

    with pytest.raises(
        RuntimeError,
        match=r"source working tree is not clean: .*fixtures/adversarial-source\.tar\.gz",
    ):
        subject._require_clean_source(repo_root, allowed_distribution_root=dist_dir)


def test_distribution_artifact_exception_does_not_hide_source_like_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A generated dist directory must not turn into a generic untracked subtree bypass."""
    repo_root = tmp_path / "repo"
    dist_dir = repo_root / "custom-dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "fast_mlsirm-0.9.2.whl").write_bytes(b"wheel")
    (dist_dir / "rogue.py").write_text("raise SystemExit", encoding="utf-8")
    status = "?? custom-dist/fast_mlsirm-0.9.2.whl\0?? custom-dist/rogue.py\0"
    monkeypatch.setattr(subject.subprocess, "run", lambda *args, **kwargs: _status(status))

    with pytest.raises(RuntimeError, match=r"source working tree is not clean: .*rogue\.py"):
        subject._require_clean_source(repo_root, allowed_distribution_root=dist_dir)


def test_distribution_artifact_exception_never_hides_tracked_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Tracked/staged mutations remain fatal even inside the admitted distribution root."""
    repo_root = tmp_path / "repo"
    dist_dir = repo_root / "custom-dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "fast_mlsirm-0.9.2.whl").write_bytes(b"wheel")
    status = " M custom-dist/fast_mlsirm-0.9.2.whl\0"
    monkeypatch.setattr(subject.subprocess, "run", lambda *args, **kwargs: _status(status))

    with pytest.raises(RuntimeError, match="source working tree is not clean"):
        subject._require_clean_source(repo_root, allowed_distribution_root=dist_dir)
