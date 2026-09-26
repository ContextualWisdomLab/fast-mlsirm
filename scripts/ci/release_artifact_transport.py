"""Download only preselected immutable artifact IDs; verify ZIP bytes first."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import tarfile
import tomllib
import zipfile

CHUNK_BYTES = 64 * 1024  # I/O buffer, not an artifact acceptance limit.
MAX_BUNDLE_BYTES = 1024 * 1024 * 1024


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


def bundle_inventory(artifact: Path, leg: str, source_sha: str, build_env: str) -> dict:
    """Hash every regular member in the finished distribution; never extract it."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not leg or not build_env:
        raise ValueError("bundle inventory lacks build identity")
    if (leg == "sdist") != artifact.name.endswith(".tar.gz") or (leg != "sdist" and not artifact.name.endswith(".whl")):
        raise ValueError("bundle inventory distribution kind mismatch")
    members, seen, total = [], set(), 0

    def record(name: str, size: int, stream) -> None:
        nonlocal total
        if (not name or name.startswith("/") or "\\" in name
                or str(PurePosixPath(name)) != name or ".." in PurePosixPath(name).parts
                or name in seen or size < 0):
            raise ValueError("unsafe or duplicate bundle member")
        seen.add(name)
        total += size
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("bundle members exceed size limit")
        members.append({"path": name, "size": size, "sha256": copy_and_hash(stream)})

    if leg == "sdist":
        with tarfile.open(artifact, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("non-regular sdist member")
                with archive.extractfile(member) as stream:
                    record(member.name, member.size, stream)
    else:
        with zipfile.ZipFile(artifact) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                mode = member.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError("non-regular wheel member")
                with archive.open(member) as stream:
                    record(member.filename, member.file_size, stream)
    if not members:
        raise ValueError("empty distribution bundle")
    return {"schema_version": 1, "source_sha": source_sha, "leg": leg,
            "file": artifact.name, "sha256": hash_file(artifact), "build_env": build_env,
            "members": sorted(members, key=lambda item: item["path"])}


def verify_runtime_inventory(record: dict, requirements: Path, row: dict,
                             source: Path, source_sha: str) -> None:
    """Bind a target install receipt to the selected wheel and exact source lock."""
    keys = {"schema_version", "source_sha", "leg", "file", "sha256", "build_env",
            "uv_version", "python_version", "implementation", "sys_platform", "machine",
            "requirements_sha256", "uv_lock_sha256", "locked_dependencies", "installed"}
    if type(record) is not dict or set(record) != keys or type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("runtime inventory schema differs")
    leg = row["target"]
    if (record["source_sha"] != source_sha or record["leg"] != leg
            or record["file"] != row["file"] or record["sha256"] != row["sha256"]
            or record["build_env"] != row["build_env"]
            or record["requirements_sha256"] != hash_file(requirements)
            or record["uv_lock_sha256"] != hash_file(source / "uv.lock")
            or not isinstance(record["uv_version"], str)
            or record["uv_version"].split()[:2] != ["uv", "0.12.5"]):
        raise ValueError(f"{leg}: runtime inventory differs from selected source or wheel")
    target, python = leg.rsplit("-py", 1)
    platforms = {
        "x86_64-unknown-linux-gnu": ("linux", {"x86_64"}),
        "aarch64-unknown-linux-gnu": ("linux", {"aarch64"}),
        "universal2-apple-darwin": ("darwin", {"arm64", "x86_64"}),
        "x86_64-pc-windows-msvc": ("win32", {"AMD64", "x86_64"}),
    }
    if (target not in platforms or record["python_version"] != python
            or record["implementation"] != "cpython"
            or record["sys_platform"] != platforms[target][0]
            or record["machine"] not in platforms[target][1]):
        raise ValueError(f"{leg}: runtime interpreter differs from wheel target")
    before, installed = record["locked_dependencies"], record["installed"]
    def valid_packages(packages):
        return (type(packages) is list and all(
            type(item) is dict and set(item) == {"name", "version"}
            and type(item["name"]) is str and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item["name"])
            and type(item["version"]) is str and bool(item["version"])
            for item in packages)
            and packages == sorted(packages, key=lambda item: item["name"])
            and len({item["name"] for item in packages}) == len(packages))
    version = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    project = {"name": "fast-mlsirm", "version": version}
    if (not valid_packages(before) or not valid_packages(installed) or not before
            or project in before or installed != sorted([*before, project], key=lambda item: item["name"])):
        raise ValueError(f"{leg}: runtime closure differs from locked install plus wheel")


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


def scope_identity(artifact: Path, leg: str, source: Path, source_sha: str, build_env: str) -> dict:
    """Bind declarations to bytes; this does not resolve or approve any closure."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not build_env:
        raise ValueError("scope identity requires exact source and build environment")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != source_sha:
        raise ValueError("scope source checkout mismatch")
    tracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", source_sha], text=True,
    ).splitlines()
    paths = sorted(p for p in tracked if p.endswith(".lock")
                   or Path(p).name in ("pyproject.toml", "Cargo.toml")
                   or (Path(p).name.startswith("requirements") and p.endswith(".txt")))
    if "pyproject.toml" not in paths or not any(p.endswith(".lock") for p in paths):
        raise ValueError("scope source declarations or locks missing")
    declarations = {}
    for path in paths:
        # Hash immutable blobs, not potentially changed working-tree files.
        blob = subprocess.check_output(["git", "-C", str(source), "show", f"{source_sha}:{path}"])
        declarations[path] = hashlib.sha256(blob).hexdigest()
    members, tags = {}, []
    if leg == "sdist":
        if not artifact.name.endswith(".tar.gz"):
            raise ValueError("sdist scope requires a source archive")
        with tarfile.open(artifact, "r:gz") as archive:
            for member in archive:
                parts = member.name.split("/")
                if len(parts) == 2 and parts[1] in ("PKG-INFO", "pyproject.toml"):
                    if not member.isfile() or member.name in members:
                        raise ValueError("invalid or duplicate sdist declaration")
                    with archive.extractfile(member) as stream:
                        members[member.name] = copy_and_hash(stream)
        if sorted(p.split("/")[1] for p in members) != ["PKG-INFO", "pyproject.toml"]:
            raise ValueError("sdist declarations missing")
        if len({p.split("/")[0] for p in members}) != 1:
            raise ValueError("sdist declaration roots differ")
        if any(h != declarations["pyproject.toml"] for p, h in members.items() if p.endswith("/pyproject.toml")):
            raise ValueError("sdist pyproject differs from release source")
        target, python, abi, platforms = "source", None, None, []
    else:
        match = re.fullmatch(r"(.+)-py(3\.[0-9]+)", leg)
        if not match or not artifact.name.endswith(".whl"):
            raise ValueError("invalid wheel scope leg")
        target, python = match.groups()
        parts = artifact.name[:-4].rsplit("-", 3)
        abi = "cp" + python.replace(".", "")
        if len(parts) != 4 or parts[1:3] != [abi, abi]:
            raise ValueError("wheel scope ABI mismatch")
        platforms = parts[3].split(".")
        platform_pattern = {
            "x86_64-unknown-linux-gnu": r"manylinux(?:2014|_2_[0-9]+)_x86_64",
            "aarch64-unknown-linux-gnu": r"manylinux(?:2014|_2_[0-9]+)_aarch64",
            "universal2-apple-darwin": r"macosx_[0-9]+_[0-9]+_(?:x86_64|arm64|universal2)",
            "x86_64-pc-windows-msvc": r"win_amd64",
        }.get(target)
        if platform_pattern is None or not all(re.fullmatch(platform_pattern, p) for p in platforms):
            raise ValueError("wheel scope target platform mismatch")
        if target == "universal2-apple-darwin" and not any(p.endswith("_universal2") for p in platforms):
            raise ValueError("universal2 wheel has no universal2 platform tag")
        with zipfile.ZipFile(artifact) as archive:
            for entry in archive.infolist():
                parts = entry.filename.split("/")
                if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] in ("METADATA", "WHEEL"):
                    if entry.filename in members or stat.S_ISLNK(entry.external_attr >> 16):
                        raise ValueError("invalid or duplicate wheel declaration")
                    with archive.open(entry) as stream:
                        members[entry.filename] = copy_and_hash(stream)
                    if parts[1] == "WHEEL":
                        with archive.open(entry) as stream:
                            tags.extend(line.decode("utf-8").strip()[5:] for line in stream if line.startswith(b"Tag: "))
        if sorted(p.split("/")[1] for p in members) != ["METADATA", "WHEEL"] or len({p.split("/")[0] for p in members}) != 1:
            raise ValueError("wheel declarations missing or ambiguous")
        if len(tags) != len(set(tags)) or set(tags) != {f"{abi}-{abi}-{p}" for p in platforms}:
            raise ValueError("wheel scope WHEEL tags mismatch")
    return {
        "schema_version": 1, "source_sha": source_sha, "leg": leg,
        "kind": "sdist" if leg == "sdist" else "wheel", "file": artifact.name,
        "sha256": hash_file(artifact), "build_env": build_env,
        "target": target, "python": python, "abi": abi, "platform_tags": platforms,
        "wheel_tags": sorted(tags), "metadata_members": members, "source_declarations": declarations,
        # No collector is connected yet. A manifest, empty array or asserted
        # boolean cannot upgrade these declarations to verified closure evidence.
        "scopes": {scope: {"status": "UNKNOWN", "evidence": None} for scope in
                   ("runtime", "build", "dev", "optional", "native", "bundled")},
    }


def verify_scope_identities(records: list[dict], rows: dict, distributions: dict,
                            source: Path, source_sha: str) -> None:
    """Rebind every record, then refuse unresolved scope before final admission."""
    if not isinstance(records, list) or len(records) != len(rows):
        raise ValueError("scope identity set missing or duplicated")
    seen = set()
    for record in records:
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record.get("leg") not in rows or record["leg"] in seen):
            raise ValueError("scope identity leg missing or duplicated")
        leg = record["leg"]
        seen.add(leg)
        row = rows[leg]
        expected = scope_identity(distributions[row["file"]], leg, source, source_sha, row["build_env"])
        if record != expected:
            raise ValueError(f"{leg}: scope identity or unresolved evidence changed")
        if record["sha256"] != row["sha256"]:
            raise ValueError(f"{leg}: scope artifact differs from reproducibility record")
    # Deliberately no positive sentinel until a trusted collector and its full
    # expected-set contract exist. All identities are checked before this HOLD.
    raise ValueError("release scope HOLD: runtime/build/dev/optional/native/bundled evidence UNKNOWN")


if __name__ == "__main__":
    selection = json.loads(Path("selected-artifacts.json").read_text())
    receipt = materialize(selection, sys.argv[1], Path("downloaded"), fetch)
    Path("transport-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
