"""Repository-hygiene regression tests for merge-conflict artifacts."""

from __future__ import annotations

from pathlib import Path


def test_merge_conflict_backup_artifacts_are_absent_and_ignored():
    """Tracked source trees exclude ``.orig``/``.rej`` files and ignore recurrence."""
    repository_root = Path(__file__).resolve().parents[1]
    artifacts = sorted(
        path.relative_to(repository_root).as_posix()
        for suffix in ("*.orig", "*.rej")
        for path in repository_root.rglob(suffix)
        if ".git" not in path.parts
    )
    assert artifacts == []

    ignore_lines = {
        line.strip()
        for line in (repository_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "*.orig" in ignore_lines
    assert "*.rej" in ignore_lines
