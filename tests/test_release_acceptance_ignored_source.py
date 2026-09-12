import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_release_acceptance():
    script = Path(__file__).resolve().parents[1] / "scripts" / "release_acceptance.py"
    spec = importlib.util.spec_from_file_location("release_acceptance_ignored_source", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _init_repo(root: Path, ignore: str) -> None:
    root.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "acceptance@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Acceptance Fixture"], cwd=root, check=True)
    (root / ".gitignore").write_text(ignore, encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=root, check=True)


def test_require_clean_source_rejects_ignored_source(tmp_path):
    module = _load_release_acceptance()
    repo = tmp_path / "repo"
    _init_repo(repo, "*.py\n")
    ignored_source = repo / "python" / "fast_mlsirm" / "shadow.py"
    ignored_source.parent.mkdir(parents=True)
    ignored_source.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source working tree is not clean"):
        module._require_clean_source(repo)


def test_require_clean_source_allows_only_ignored_distribution_artifacts(tmp_path):
    module = _load_release_acceptance()
    repo = tmp_path / "repo"
    _init_repo(repo, "dist/\n")
    dist = repo / "dist"
    dist.mkdir()
    (dist / "candidate.whl").write_bytes(b"wheel")
    (dist / "candidate.tar.gz").write_bytes(b"sdist")

    module._require_clean_source(repo, allowed_distribution_root=dist)

    (dist / "shadow.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source working tree is not clean"):
        module._require_clean_source(repo, allowed_distribution_root=dist)
