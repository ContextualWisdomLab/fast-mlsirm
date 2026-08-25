"""Contracts for the dependency lock's supported Python floor."""

from pathlib import Path
import tomllib


_ROOT = Path(__file__).parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_UV_LOCK = _ROOT / "uv.lock"


def test_uv_lock_python_floor_matches_project_metadata() -> None:
    """The dependency lock must not advertise unsupported Python runtimes."""
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    lock = tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.12"
    assert lock["requires-python"] == project["project"]["requires-python"]
