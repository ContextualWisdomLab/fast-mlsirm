"""Verify the central full-set verdict against this run's release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")


def _utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("missing UTC artifact timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid UTC artifact timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("artifact timestamp is not UTC")
    return parsed


def _artifact(item: Any, name: str, artifact_id: int, digest: str,
              run_id: int, control_sha: str, started: datetime) -> None:
    if (not isinstance(item, Mapping) or item.get("name") != name
            or type(item.get("id")) is not int or item["id"] != artifact_id
            or item.get("digest") != digest or item.get("expired") is not False
            or not isinstance(item.get("workflow_run"), Mapping)
            or item["workflow_run"].get("id") != run_id
            or item["workflow_run"].get("head_sha") != control_sha
            or _utc(item.get("created_at")) <= started):
        raise ValueError(f"{name}: foreign, stale, or changed artifact")


def verify_full_set_verdict(
    manifest: Any, verdict: Any, artifacts: list[Any], attempt: Any, *,
    repository: str, source_sha: str, control_sha: str, run_id: int,
    run_attempt: int, record_id: int, record_digest: str,
    verdict_id: int, verdict_digest: str, verdict_name: str,
) -> None:
    """Require the same successful run, exact distribution rows, and binding IDs."""
    if (not SHA.fullmatch(source_sha) or not SHA.fullmatch(control_sha)
            or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
            or type(run_id) is not int or run_id <= 0
            or type(run_attempt) is not int or run_attempt <= 0
            or type(record_id) is not int or record_id <= 0
            or type(verdict_id) is not int or verdict_id <= 0
            or not DIGEST.fullmatch(record_digest) or not DIGEST.fullmatch(verdict_digest)):
        raise ValueError("invalid expected release identity")
    if (not isinstance(attempt, Mapping) or type(attempt.get("id")) is not int
            or attempt["id"] != run_id or type(attempt.get("run_attempt")) is not int
            or attempt["run_attempt"] != run_attempt
            or attempt.get("head_sha") != control_sha):
        raise ValueError("workflow attempt differs from the release")
    started = _utc(attempt.get("run_started_at"))
    listed: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ValueError("artifact listing contains an invalid item")
        if item["name"] in listed:
            raise ValueError("duplicate artifact name")
        listed[item["name"]] = item
    _artifact(listed.get("reproducibility-record"), "reproducibility-record",
              record_id, record_digest, run_id, control_sha, started)
    _artifact(listed.get(verdict_name), verdict_name, verdict_id, verdict_digest,
              run_id, control_sha, started)
    if (not isinstance(manifest, Mapping) or not isinstance(verdict, Mapping)
            or manifest.get("schema_version") != 1
            or verdict.get("schema") != "cwl.release-full-set-verdict/1"
            or verdict.get("result") != "PASS"):
        raise ValueError("full-set verdict or distribution manifest is malformed")
    expected = {"source_repository": repository, "source_sha": source_sha,
                "control_sha": control_sha, "run_id": run_id,
                "run_attempt": run_attempt}
    if any(manifest.get(key) != value or verdict.get(key) != value
           for key, value in expected.items()):
        raise ValueError("verdict or manifest differs from release identity")
    if any(type(document.get(key)) is not int for document in (manifest, verdict)
           for key in ("run_id", "run_attempt")):
        raise ValueError("verdict or manifest execution identity is malformed")
    if (verdict.get("record_artifact_id") != record_id
            or verdict.get("record_artifact_digest") != record_digest
            or verdict.get("distributions") != manifest.get("distributions")):
        raise ValueError("verdict does not cover the verified distribution set")
    rows = manifest.get("distributions")
    if (not isinstance(rows, list) or len(rows) != 13
            or len({row.get("leg") for row in rows if isinstance(row, Mapping)}) != 13
            or sum(row.get("leg") == "sdist" for row in rows if isinstance(row, Mapping)) != 1):
        raise ValueError("verdict does not cover twelve wheels and one sdist")
    seen_ids = {record_id, verdict_id}
    distribution_names: set[str] = set()
    for row in rows:
        name, artifact_id, digest = (row.get(field) for field in
                                     ("artifact_name", "artifact_id", "artifact_digest"))
        if (not isinstance(name, str) or name != (
                "dist-sdist" if row["leg"] == "sdist" else f"dist-wheel-{row['leg']}")
                or name in distribution_names or type(artifact_id) is not int
                or artifact_id <= 0 or artifact_id in seen_ids or not isinstance(digest, str)
                or not DIGEST.fullmatch(digest)):
            raise ValueError("duplicate or malformed distribution artifact identity")
        _artifact(listed.get(name), name, artifact_id, digest, run_id, control_sha, started)
        distribution_names.add(name)
        seen_ids.add(artifact_id)
    if {name for name in listed if name.startswith("dist-")} != distribution_names:
        raise ValueError("distribution artifact set differs from the verdict")
    bindings = verdict.get("binding_artifacts")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("verdict has no Strix binding set")
    seen_names: set[str] = set()
    seen_keys: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("malformed Strix binding identity")
        name, key, artifact_id, digest = (binding.get(field) for field in ("name", "key", "id", "digest"))
        expected_name = (f"release-strix-binding-a{run_attempt}-"
                         + hashlib.sha256(key.encode()).hexdigest()) if isinstance(key, str) else ""
        if (not isinstance(name, str) or name != expected_name
                or not isinstance(key, str) or not key or name in seen_names or key in seen_keys
                or type(artifact_id) is not int or artifact_id <= 0 or artifact_id in seen_ids
                or not isinstance(digest, str) or not DIGEST.fullmatch(digest)):
            raise ValueError("duplicate or malformed Strix binding identity")
        _artifact(listed.get(name), name, artifact_id, digest, run_id, control_sha, started)
        seen_ids.add(artifact_id)
        seen_names.add(name)
        seen_keys.add(key)
    if {name for name in listed if name.startswith(f"release-strix-binding-a{run_attempt}-")} != seen_names:
        raise ValueError("same-attempt Strix artifact set differs from the verdict")


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "verdict", "artifacts", "attempt", "repository", "source-sha",
                 "control-sha", "run-id", "run-attempt", "record-id", "record-digest",
                 "verdict-id", "verdict-digest", "verdict-name"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    verify_full_set_verdict(
        json.loads(Path(args.manifest).read_text()), json.loads(Path(args.verdict).read_text()),
        [json.loads(line) for line in Path(args.artifacts).read_text().splitlines()],
        json.loads(Path(args.attempt).read_text()), repository=args.repository,
        source_sha=args.source_sha, control_sha=args.control_sha,
        run_id=int(args.run_id), run_attempt=int(args.run_attempt),
        record_id=int(args.record_id), record_digest=args.record_digest,
        verdict_id=int(args.verdict_id), verdict_digest=args.verdict_digest,
        verdict_name=args.verdict_name,
    )


if __name__ == "__main__":
    main()
