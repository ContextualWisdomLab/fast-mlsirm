"""Repository-hygiene regression tests for transient patch artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess


def _tracked_paths(repository_root: Path) -> tuple[str, ...]:
    """Return repository-tracked paths without scanning generated build trees."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        timeout=10,
    )
    return tuple(path.decode("utf-8") for path in completed.stdout.split(b"\0") if path)


def test_merge_conflict_backup_artifacts_are_absent_and_ignored():
    """Tracked source excludes ``.orig``/``.rej`` files and ignores recurrence."""
    repository_root = Path(__file__).resolve().parents[1]
    artifacts = sorted(
        path
        for path in _tracked_paths(repository_root)
        if Path(path).suffix in {".orig", ".rej"}
    )
    assert artifacts == []

    ignore_lines = {
        line.strip()
        for line in (repository_root / ".gitignore")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert "*.orig" in ignore_lines
    assert "*.rej" in ignore_lines


def test_root_patch_tool_scratch_artifacts_are_not_tracked():
    """Reject ad-hoc patch-tool payloads accidentally committed at repository root."""
    repository_root = Path(__file__).resolve().parents[1]
    artifacts: list[str] = []
    for tracked in _tracked_paths(repository_root):
        path = Path(tracked)
        if len(path.parts) != 1:
            continue
        is_patch_payload = path.suffix in {".diff", ".patch"}
        is_patch_driver = path.suffix == ".py" and (
            path.stem.startswith("fix_") or path.stem.startswith("patch")
        )
        if is_patch_payload or is_patch_driver:
            artifacts.append(tracked)

    assert sorted(artifacts) == []
