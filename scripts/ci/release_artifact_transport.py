"""Download only preselected immutable artifact IDs; verify ZIP bytes first."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import zipfile

CHUNK_BYTES = 64 * 1024  # I/O buffer, not an artifact acceptance limit.


def copy_and_hash(source, destination=None) -> str:
    """Hash incrementally; optionally copy the same bytes to a file."""
    digest = hashlib.sha256()
    while chunk := source.read(CHUNK_BYTES):
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    """Hash a regular file without allocating its entire contents."""
    with path.open("rb") as source:
        return copy_and_hash(source)


def materialize(selection: list[dict], repository: str, root: Path, fetch) -> list[dict]:
    """Treat archives as inert flat files, never as code or install inputs."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid artifact repository")
    ids, names, receipt = set(), set(), []
    for artifact in selection:
        artifact_id, name = artifact["id"], artifact["name"]
        if type(artifact_id) is not int or artifact_id <= 0 or artifact_id in ids:
            raise ValueError("missing or duplicate immutable artifact ID")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError("invalid or duplicate artifact name")
        ids.add(artifact_id)
        names.add(name)
        expected = artifact["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
            raise ValueError("missing artifact digest")
        with tempfile.TemporaryFile() as archive_file:
            fetch(repository, artifact_id, archive_file)
            archive_file.seek(0)
            actual = "sha256:" + copy_and_hash(archive_file)
            if actual != expected:
                raise ValueError("download ZIP digest mismatch")
            root.mkdir(parents=True, exist_ok=True)
            destination = root / name
            if destination.exists() or destination.is_symlink():
                raise ValueError("artifact destination already exists")
            members = {}
            # Only this private staging directory is cleaned on failure.
            with tempfile.TemporaryDirectory(prefix=".artifact-", dir=root) as staging:
                with zipfile.ZipFile(archive_file) as archive:
                    for entry in archive.infolist():
                        mode = entry.external_attr >> 16
                        if (not re.fullmatch(r"[A-Za-z0-9_.+-]+", entry.filename)
                                or entry.filename in (".", "..") or entry.filename in members
                                or entry.is_dir() or stat.S_ISLNK(mode)
                                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                            raise ValueError("unsafe, non-flat or duplicate ZIP member")
                        with archive.open(entry) as source, (Path(staging) / entry.filename).open("xb") as output:
                            members[entry.filename] = copy_and_hash(source, output)
                if not members:
                    raise ValueError("empty artifact archive")
                # Publish only complete, hash-verified members.
                Path(staging).rename(destination)
        if not members:
            raise ValueError("empty artifact archive")
        receipt.append({"id": artifact_id, "name": name, "digest": actual, "members": members})
    return receipt


def fetch(repository: str, artifact_id: int, destination) -> None:
    """Stream gh output to scratch; reap the child on failure/interruption."""
    process = subprocess.Popen(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"],
        stdout=subprocess.PIPE,
    )
    try:
        copy_and_hash(process.stdout, destination)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, process.args)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()


def verify_materialized(selection: list[dict], receipt: list[dict], root: Path) -> None:
    """Bind downstream reads to the selected IDs and verified ZIP members."""
    if [{k: r[k] for k in ("id", "name", "digest")} for r in receipt] != selection:
        raise ValueError("transport receipt does not match selected immutable IDs")
    for artifact in receipt:
        directory = root / artifact["name"]
        actual = {}
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("unexpected downloaded artifact member")
            actual[path.name] = hash_file(path)
        if actual != artifact["members"]:
            raise ValueError("downloaded artifact members changed after digest verification")


if __name__ == "__main__":
    selection = json.loads(Path("selected-artifacts.json").read_text())
    receipt = materialize(selection, sys.argv[1], Path("downloaded"), fetch)
    Path("transport-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
