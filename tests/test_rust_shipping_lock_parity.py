"""Contracts for dependency identity across shipping Rust lock roots."""

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_LOCK = REPO_ROOT / "Cargo.lock"
PY_BINDING_LOCK = REPO_ROOT / "crates" / "fast-mlsirm-py" / "Cargo.lock"


def _locked_package(path: Path, name: str) -> tuple[str, str]:
    """Return the unique registry version and checksum for ``name``."""
    packages = tomllib.loads(path.read_text(encoding="utf-8"))["package"]
    matches = [
        package
        for package in packages
        if package["name"] == name
        and package.get("source", "").startswith("registry+")
    ]
    assert len(matches) == 1, f"expected one registry package {name!r} in {path}"
    package = matches[0]
    return package["version"], package["checksum"]


def test_uuid_identity_matches_all_shipping_lock_roots() -> None:
    """The Rust core's UUID dependency must not diverge by package surface."""
    root_identity = _locked_package(ROOT_LOCK, "uuid")
    binding_identity = _locked_package(PY_BINDING_LOCK, "uuid")

    assert root_identity == binding_identity
