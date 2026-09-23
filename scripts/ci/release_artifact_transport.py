"""Download only preselected immutable artifact IDs; verify ZIP bytes first."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile


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
        archive_bytes = fetch(repository, artifact_id)
        actual = "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
        if actual != expected:
            raise ValueError("download ZIP digest mismatch")
        # Keep each artifact isolated: merged extraction must not overwrite an
        # earlier leg's same-name member before cardinality is checked.
        destination = root / name
        destination.mkdir(parents=True, exist_ok=False)
        members = {}
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for entry in archive.infolist():
                mode = entry.external_attr >> 16
                if (not re.fullmatch(r"[A-Za-z0-9_.+-]+", entry.filename)
                        or entry.filename in (".", "..") or entry.filename in members
                        or entry.is_dir() or stat.S_ISLNK(mode)
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                    raise ValueError("unsafe, non-flat or duplicate ZIP member")
                data = archive.read(entry)
                members[entry.filename] = hashlib.sha256(data).hexdigest()
                with (destination / entry.filename).open("xb") as output:
                    output.write(data)
        if not members:
            raise ValueError("empty artifact archive")
        receipt.append({"id": artifact_id, "name": name, "digest": actual, "members": members})
    return receipt


def fetch(repository: str, artifact_id: int) -> bytes:
    return subprocess.run(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"],
        check=True, stdout=subprocess.PIPE,
    ).stdout


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
            actual[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != artifact["members"]:
            raise ValueError("downloaded artifact members changed after digest verification")


if __name__ == "__main__":
    selection = json.loads(Path("selected-artifacts.json").read_text())
    receipt = materialize(selection, sys.argv[1], Path("downloaded"), fetch)
    Path("transport-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
