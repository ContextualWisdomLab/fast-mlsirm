"""Build each wheel target again from the exact release sdist."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

from release_artifact_transport import bundle_inventory, hash_file, scope_identity


def _row(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or len(lines[0].split("\t")) != 7:
        raise ValueError("consumer build requires one reproducibility row")
    keys = ("target", "byte_verified", "verification", "sha256", "rebuild_sha256", "file", "build_env")
    row = dict(zip(keys, lines[0].split("\t")))
    if (row["byte_verified"] != "true" or row["verification"] != "clean-target-repeat-same-env"
            or row["sha256"] != row["rebuild_sha256"]):
        raise ValueError("consumer build row is not byte-verified")
    return row


def prepare(source: Path, source_sha: str, row_path: Path, dist: Path, output: Path) -> dict:
    row = _row(row_path)
    if row["target"] != "sdist" or row_path.name != "sdist.tsv":
        raise ValueError("consumer input is not the release sdist row")
    archive = dist / row["file"]
    if (set(dist.iterdir()) != {archive} or archive.is_symlink() or not archive.is_file()
            or hash_file(archive) != row["sha256"]):
        raise ValueError("consumer sdist bytes differ from the build row")
    scope_identity(archive, "sdist", source, source_sha, row["build_env"])
    inventory = bundle_inventory(archive, "sdist", source_sha, row["build_env"])
    roots = {member["path"].split("/", 1)[0] for member in inventory["members"]}
    if len(roots) != 1 or output.exists() or output.is_symlink():
        raise ValueError("consumer sdist has no unique fresh extraction root")
    root = next(iter(roots))
    destination = output / "source"
    expected = {member["path"] for member in inventory["members"]}
    with tarfile.open(archive, "r:gz") as source_archive:
        for member in source_archive:
            if not member.isfile():
                continue
            if member.name not in expected or not member.name.startswith(root + "/"):
                raise ValueError("consumer sdist changed after validation")
            relative = member.name.removeprefix(root + "/")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_archive.extractfile(member) as stream, target.open("wb") as written:
                shutil.copyfileobj(stream, written)
    if hash_file(archive) != row["sha256"]:
        raise ValueError("consumer sdist changed during extraction")
    record = {"schema_version": 1, "source_sha": source_sha,
              "sdist_file": row["file"], "sdist_sha256": row["sha256"]}
    (output / "input.json").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return record


def capture(source: Path, source_sha: str, row_path: Path, dist: Path,
            prepared: Path, output: Path) -> dict:
    row = _row(row_path)
    leg = row["target"]
    if leg == "sdist" or row_path.name != f"{leg}.tsv":
        raise ValueError("consumer output is not a wheel leg")
    input_record = json.loads((prepared / "input.json").read_text(encoding="utf-8"))
    if (set(input_record) != {"schema_version", "source_sha", "sdist_file", "sdist_sha256"}
            or input_record["schema_version"] != 1 or input_record["source_sha"] != source_sha):
        raise ValueError("consumer input differs from release source")
    direct = dist / row["file"]
    built = list((prepared / "source/dist-consumer").glob("*.whl"))
    if (not direct.is_file() or hash_file(direct) != row["sha256"]
            or len(built) != 1 or built[0].name != row["file"]):
        raise ValueError("consumer wheel differs from published leg")
    direct_scope = scope_identity(direct, leg, source, source_sha, row["build_env"])
    consumer_scope = scope_identity(built[0], leg, source, source_sha, row["build_env"])
    if (direct_scope["metadata_members"] != consumer_scope["metadata_members"]
            or direct_scope["wheel_tags"] != consumer_scope["wheel_tags"]):
        raise ValueError("consumer wheel metadata differs from published wheel")
    inventory = bundle_inventory(built[0], leg, source_sha, row["build_env"])
    extension = [item for item in inventory["members"]
                 if item["path"].startswith("fast_mlsirm/_core.")
                 and item["path"].endswith((".so", ".pyd"))]
    if len(extension) != 1:
        raise ValueError("consumer wheel has no unique native extension")
    receipt = {"schema_version": 1, "source_sha": source_sha, "leg": leg,
               "build_env": row["build_env"], "sdist_file": input_record["sdist_file"],
               "sdist_sha256": input_record["sdist_sha256"], "file": row["file"],
               "published_sha256": row["sha256"], "consumer_sha256": inventory["sha256"],
               "metadata_members": consumer_scope["metadata_members"],
               "native_extension": {"member": extension[0]["path"],
                                    "sha256": extension[0]["sha256"]}}
    copied = output / f"{leg}.consumer.whl"
    if copied.exists() or copied.is_symlink():
        raise ValueError("consumer wheel receipt already exists")
    shutil.copyfile(built[0], copied)
    (output / f"{leg}.consumer.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    root = Path.cwd()
    sha = os.environ["RELEASE_COMMIT"]
    if mode == "prepare" and len(args) == 3:
        prepare(root, sha, *(Path(arg) for arg in args))
    elif mode == "capture" and len(args) == 4:
        capture(root, sha, *(Path(arg) for arg in args))
    else:
        raise SystemExit("usage: release_sdist_consumer.py prepare ROW DIST OUTPUT | capture ROW DIST PREPARED OUTPUT")
