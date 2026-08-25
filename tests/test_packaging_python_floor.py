"""Packaging-contract tests for the advertised Python floor.

The project's supported interpreter floor lives in exactly one authoritative
place, ``pyproject.toml`` (``project.requires-python``), and the committed
lockfile ``uv.lock`` must advertise the identical floor. A drift between the
two means the lock resolves — and therefore reproduces builds for —
interpreter ranges the package no longer supports, which silently breaks the
"rebuild from immutable provenance" guarantee documented in
``docs/PRD.md`` (PRD-PRN-006).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_UV_LOCK = _REPO_ROOT / "uv.lock"

_REQUIRES_PYTHON = re.compile(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", re.M)
_STALE_PRE_312_MARKERS = (
    "python_full_version == '3.10.*'",
    "python_full_version == '3.11.*'",
    "python_full_version < '3.11'",
    "python_full_version < '3.12'",
    "python_full_version <= '3.10'",
    "python_full_version <= '3.11'",
)


def _read_floor(path: Path) -> str:
    """Return the ``requires-python`` value declared at the top of *path*.

    Raises
    ------
    AssertionError
        If *path* does not exist or declares no ``requires-python`` entry,
        because a missing declaration is itself floor drift that must fail
        this contract instead of being silently ignored.
    """
    assert path.exists(), f"missing packaging file: {path}"
    match = _REQUIRES_PYTHON.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"{path.name} declares no requires-python floor"
    return match.group(1)


def test_uv_lock_floor_matches_pyproject_floor() -> None:
    """The lockfile must advertise the exact pyproject interpreter floor."""
    assert _read_floor(_UV_LOCK) == _read_floor(_PYPROJECT)


@pytest.mark.parametrize("stale_marker", _STALE_PRE_312_MARKERS)
def test_uv_lock_has_no_stale_pre_3_12_resolution_markers(stale_marker: str) -> None:
    """The lock must not retain partitions that admit dropped interpreters.

    uv can express a dropped 3.10/3.11 partition either with an equality
    marker such as ``== '3.11.*'`` or through an upper-bound partition such as
    ``< '3.12'``. Guard every form the lock generator can use so a regenerated
    lock cannot silently reintroduce unsupported interpreter partitions.
    """
    lock_text = _UV_LOCK.read_text(encoding="utf-8")
    assert stale_marker not in lock_text, (
        f"uv.lock still contains stale pre-3.12 marker {stale_marker!r}; "
        "regenerate with `uv lock` after confirming the pyproject floor"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
