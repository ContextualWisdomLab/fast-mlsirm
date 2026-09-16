#!/usr/bin/env python3
"""Apply the reviewed Graphify engine patch with fail-closed checks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

REQUIRED_KEYS = {"source_sha256", "patch_sha256", "final_sha256", "tool_version", "wheel_sha256"}
EXPECTED_SOURCE_SHA256 = "921e37f525e1de9e406af163e7825f807d2770d677e3270bf6e5120b36428b12"


def file_digest(file_path: Path) -> str:
    file_checksum = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for data_chunk in iter(lambda: file_handle.read(1 << 20), b""):
            file_checksum.update(data_chunk)
    return file_checksum.hexdigest()


def is_path_within(child_path: Path, parent_path: Path) -> bool:
    try:
        child_path.relative_to(parent_path)
    except ValueError:
        return False
    return True


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    helper_directory = Path(__file__).resolve().parent
    virtual_link = helper_directory / ".venv"
    if virtual_link.is_symlink():
        raise SystemExit("refusing symlinked local venv")
    virtual_environment = virtual_link.resolve()
    if not is_path_within(virtual_environment, helper_directory):
        raise SystemExit("refusing venv outside helper directory")
    if Path(sys.prefix).resolve() != virtual_environment:
        raise SystemExit(f"refusing non-local interpreter: {sys.prefix}")
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    if not is_path_within(site_packages, virtual_environment):
        raise SystemExit("refusing site-packages outside the local venv")
    engine_link = site_packages / "graphify" / "extractors" / "engine.py"
    engine_path = engine_link.resolve(strict=False)
    if engine_link.is_symlink() or not is_path_within(engine_path, site_packages):
        raise SystemExit("refusing engine path outside local site-packages")
    return engine_path, helper_directory / "engine.patch", helper_directory / "manifest.json", site_packages


def load_manifest(manifest_path: Path) -> dict[str, object]:
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid manifest: {error}") from error
    if not isinstance(manifest_value, dict) or not REQUIRED_KEYS <= set(manifest_value):
        raise SystemExit("manifest missing required keys")
    for key_name in ("source_sha256", "patch_sha256", "final_sha256", "wheel_sha256"):
        digest_value = manifest_value[key_name]
        if not isinstance(digest_value, str) or len(digest_value) != 64 or any(character not in "0123456789abcdef" for character in digest_value):
            raise SystemExit(f"manifest {key_name} must be lowercase SHA-256 hex")
    if not isinstance(manifest_value["tool_version"], str) or not manifest_value["tool_version"]:
        raise SystemExit("manifest tool_version must be a non-empty string")
    return manifest_value


def main() -> int:
    engine_path, patch_path, manifest_path, site_packages = resolve_paths()
    manifest_value = load_manifest(manifest_path)
    if manifest_value["source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise SystemExit("refusing unpinned engine source")
    if not patch_path.is_file() or file_digest(patch_path) != manifest_value["patch_sha256"]:
        raise SystemExit("refusing patch hash mismatch")
    if not engine_path.is_file() or engine_path.is_symlink():
        raise SystemExit("refusing missing or symlinked engine")
    current_digest = file_digest(engine_path)
    if current_digest == manifest_value["final_sha256"]:
        return 0
    if current_digest != manifest_value["source_sha256"]:
        raise SystemExit("refusing unknown engine source")
    temporary_handle = tempfile.NamedTemporaryFile(dir=engine_path.parent, prefix=".extract.py.", delete=False)
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        shutil.copyfile(engine_path, temporary_path)
        subprocess.run(["patch", "--forward", "--batch", str(temporary_path), str(patch_path)], cwd=site_packages, check=True)
        if file_digest(temporary_path) != manifest_value["final_sha256"]:
            raise SystemExit("final engine digest mismatch")
        os.replace(temporary_path, engine_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_path.with_name(temporary_path.name + ".orig").unlink(missing_ok=True)
        temporary_path.with_name(temporary_path.name + ".rej").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
