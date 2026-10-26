"""Fail-closed contracts for the repository pytest runtime configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_async_fixture_loop_scope_is_explicit() -> None:
    """pytest-asyncio must not choose a changing default fixture-loop scope."""
    repository_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads(
        (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert (
        pyproject["tool"]["pytest"]["ini_options"][
            "asyncio_default_fixture_loop_scope"
        ]
        == "function"
    )
