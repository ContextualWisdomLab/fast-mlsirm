from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCKFILES = (
    Path("Cargo.lock"),
    Path("crates/fast-mlsirm-py/Cargo.lock"),
    Path("fuzz/Cargo.lock"),
)


def _read_toml(path: Path) -> dict:
    with (ROOT / path).open("rb") as handle:
        return tomllib.load(handle)


def _sha2_versions(lockfile: Path) -> set[str]:
    packages = _read_toml(lockfile)["package"]
    return {package["version"] for package in packages if package["name"] == "sha2"}


def test_all_committed_cargo_locks_resolve_the_core_sha2_requirement() -> None:
    core_manifest = _read_toml(Path("crates/mlsirm-core/Cargo.toml"))
    expected = core_manifest["dependencies"]["sha2"]

    for lockfile in LOCKFILES:
        assert _sha2_versions(lockfile) == {expected}, (
            f"{lockfile} must resolve the mlsirm-core sha2 requirement {expected}; "
            "refresh the independent lockfile before merging dependency updates"
        )
