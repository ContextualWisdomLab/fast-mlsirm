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
    manifest: Any, scope_set: Any, verdict: Any, artifacts: list[Any], attempt: Any, *,
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
    scope_rows = scope_set.get("evidence") if isinstance(scope_set, Mapping) else None
    if (not isinstance(scope_set, Mapping) or not isinstance(scope_rows, list)
            or set(scope_set) != {"schema_version", *expected, "evidence"}
            or type(scope_set.get("schema_version")) is not int
            or scope_set["schema_version"] != 1
            or len(scope_rows) != 13 or not isinstance(verdict.get("scope_evidence"), list)
            or sorted(scope_rows, key=lambda row: row.get("leg", "") if isinstance(row, Mapping) else "")
            != verdict["scope_evidence"]
            or any(scope_set.get(key) != value for key, value in expected.items())):
        raise ValueError("central verdict does not seal the exact scope artifact set")
    scope_names: set[str] = set()
    for row in scope_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "leg", "artifact_id", "artifact_name", "artifact_digest"
        }:
            raise ValueError("scope artifact identity is malformed")
        leg, name, artifact_id, digest = (row[field] for field in
                                         ("leg", "artifact_name", "artifact_id", "artifact_digest"))
        if (not isinstance(leg, str) or leg not in {item["leg"] for item in rows}
                or name != f"repro-digest-{leg}" or name in scope_names
                or type(artifact_id) is not int or artifact_id <= 0 or artifact_id in seen_ids
                or not isinstance(digest, str) or not DIGEST.fullmatch(digest)):
            raise ValueError("scope artifact identity is missing or duplicated")
        _artifact(listed.get(name), name, artifact_id, digest, run_id, control_sha, started)
        scope_names.add(name)
        seen_ids.add(artifact_id)
    if {name for name in listed if name.startswith("repro-digest-")} != scope_names:
        raise ValueError("scope artifact set differs from the central verdict")
    bindings = verdict.get("binding_artifacts")
    archive_bindings = verdict.get("runtime_archive_binding_artifacts")
    build_bindings = verdict.get("build_package_binding_artifacts")
    if (not isinstance(bindings, list) or not bindings
            or not isinstance(archive_bindings, list) or not archive_bindings
            or not isinstance(build_bindings, list) or not build_bindings
            or not isinstance(verdict.get("runtime_archive_license_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", verdict["runtime_archive_license_sha256"])):
        raise ValueError("verdict has no Strix binding set")
    seen_names: set[str] = set()
    seen_keys: set[str] = set()
    for binding in [*bindings, *archive_bindings, *build_bindings]:
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
    if any(not isinstance(binding.get("key"), str)
           or re.fullmatch(r"pypi/[a-z0-9-]+@[^/]+/sha256/[0-9a-f]{64}", binding["key"]) is None
           for binding in [*archive_bindings, *build_bindings]):
        raise ValueError("scoped dependency Strix binding key is malformed")
    if {name for name in listed if name.startswith(f"release-strix-binding-a{run_attempt}-")} != seen_names:
        raise ValueError("same-attempt Strix artifact set differs from the verdict")


def verify_runtime_dependency_coverage(verdict: Any, report: Any, report_bytes: bytes,
                                       archive_report: Any, archive_report_bytes: bytes,
                                       runtime_records: list[dict], build_records: list[dict], *,
                                       repository: str, source_sha: str) -> None:
    """Require every installed wheel dependency in the licensed, Strix-bound set."""
    if (not isinstance(verdict, Mapping) or not isinstance(report, Mapping)
            or verdict.get("gate_report_sha256") != hashlib.sha256(report_bytes).hexdigest()
            or report.get("schema") != "cwl.release-dependency-gate/1"
            or report.get("result") != "PASS" or report.get("stage") != "full"
            or report.get("source_repository") != repository
            or report.get("source_sha") != source_sha
            or report.get("failures") != []):
        raise ValueError("full dependency report differs from the sealed verdict")
    dependencies = report.get("dependencies")
    bindings = verdict.get("binding_artifacts")
    archive_bindings = verdict.get("runtime_archive_binding_artifacts")
    build_bindings = verdict.get("build_package_binding_artifacts")
    reviews = report.get("runtime_archive_reviews")
    build_reviews = report.get("build_package_reviews")
    licensed_archives = archive_report.get("archives") if isinstance(archive_report, Mapping) else None
    licensed_build = archive_report.get("build_packages") if isinstance(archive_report, Mapping) else None
    if (not isinstance(dependencies, list) or not dependencies
            or type(report.get("dependency_count")) is not int
            or report["dependency_count"] != len(dependencies)
            or not isinstance(bindings, list) or not bindings
            or not isinstance(archive_bindings, list) or not archive_bindings
            or not isinstance(build_bindings, list) or not build_bindings
            or not isinstance(reviews, list) or not reviews
            or not isinstance(build_reviews, list) or not build_reviews
            or not isinstance(archive_report, Mapping)
            or archive_report.get("schema") != "cwl.release-runtime-archive-licenses/2"
            or not isinstance(licensed_archives, list) or not licensed_archives
            or not isinstance(licensed_build, list) or not licensed_build
            or verdict.get("runtime_archive_license_sha256")
            != hashlib.sha256(archive_report_bytes).hexdigest()):
        raise ValueError("full dependency report has no complete dependency set")
    sources = {}
    for row in dependencies:
        if (not isinstance(row, Mapping)
                or row.get("ecosystem") not in ("pypi", "cargo")
                or not all(isinstance(row.get(field), str) and row[field]
                           for field in ("key", "name", "version", "license", "source_sha256", "fixture_sha256"))
                or row["key"] != f"{row['ecosystem']}/{row['name']}@{row['version']}"
                or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])
                or row["key"] in sources):
            raise ValueError("full dependency report contains an invalid dependency")
        sources[row["key"]] = row["source_sha256"]
    keys = set(sources)
    if (keys != {binding.get("key") for binding in bindings if isinstance(binding, Mapping)}
            or len(bindings) != len(keys) or len(runtime_records) != 12):
        raise ValueError("Strix bindings or wheel runtime receipts do not cover the dependency set")
    expected_archives: dict[str, set[str]] = {}
    for runtime in runtime_records:
        locked = {f"pypi/{package['name']}@{package['version']}"
                  for package in runtime["locked_dependencies"]}
        archived = {f"pypi/{archive['name']}@{archive['version']}"
                    for archive in runtime["archives"]}
        if locked != archived or len(locked) != len(runtime["archives"]):
            raise ValueError(f"{runtime['leg']}: installed dependencies differ from reviewed archives")
        for archive in runtime["archives"]:
            package_key = f"pypi/{archive['name']}@{archive['version']}"
            if not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]):
                raise ValueError(f"{runtime['leg']}: dependency archive lacks a licensed package identity")
            key = f"{package_key}/sha256/{archive['sha256']}"
            expected_archives.setdefault(key, set()).add(runtime["leg"])
    if (len(licensed_archives) != len(expected_archives)
            or len(reviews) != len(expected_archives)
            or len(archive_bindings) != len(expected_archives)
            or {binding.get("key") for binding in archive_bindings if isinstance(binding, Mapping)}
            != set(expected_archives)):
        raise ValueError("runtime archive licence and Strix sets differ from installed bytes")
    approved = {}
    for row in licensed_archives:
        if not isinstance(row, Mapping):
            raise ValueError("runtime archive licence row is malformed")
        key, package_key, sha = row.get("key"), row.get("package_key"), row.get("source_sha256")
        fixture, fixture_sha = row.get("fixture"), row.get("fixture_sha256")
        if (not isinstance(key, str) or key not in expected_archives or key in approved
                or not isinstance(package_key, str)
                or not isinstance(sha, str) or key != f"{package_key}/sha256/{sha}"
                or not isinstance(fixture, Mapping) or fixture.get("id") != key
                or not isinstance(fixture_sha, str)
                or fixture_sha != hashlib.sha256(json.dumps(fixture, sort_keys=True,
                                                             separators=(",", ":")).encode()).hexdigest()
                or not isinstance(row.get("license"), str) or not row["license"]
                or not isinstance(row.get("legs"), list)
                or set(row["legs"]) != expected_archives[key]):
            raise ValueError("runtime archive licence row differs from installed bytes")
        approved[key] = row
    seen_reviews = set()
    for review in reviews:
        if (not isinstance(review, Mapping) or review.get("key") not in approved
                or review["key"] in seen_reviews):
            raise ValueError("runtime archive Strix review is missing or duplicated")
        row = approved[review["key"]]
        if any(review.get(field) != row.get(field) for field in
               ("package_key", "source_sha256", "license", "fixture_sha256", "legs")):
            raise ValueError("runtime archive Strix review differs from licence verdict")
        seen_reviews.add(review["key"])
    if len(build_records) != 13 or {row.get("leg") for row in build_records} != {
            "sdist", *{runtime["leg"] for runtime in runtime_records}}:
        raise ValueError("build interpreter evidence does not cover all release legs")
    expected_build: dict[str, dict] = {}
    for record in build_records:
        leg = record["leg"]
        snapshot_sha = record["python_snapshot_sha256"]
        if (not re.fullmatch(r"[0-9a-f]{64}", snapshot_sha)
                or not isinstance(record.get("python_packages"), list)
                or not record["python_packages"]):
            raise ValueError("build interpreter snapshot identity is malformed")
        for package in record["python_packages"]:
            name, version = package["name"], package["version"]
            sha = hashlib.sha256(json.dumps(
                {"name": name, "version": version, "files": package["files"]},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            key = f"pypi/{name}@{version}/sha256/{sha}"
            row = expected_build.setdefault(key, {"legs": set(), "snapshots": {},
                                                  "name": name, "version": version,
                                                  "source_sha256": sha})
            if leg in row["legs"]:
                raise ValueError("build interpreter package is duplicated")
            row["legs"].add(leg)
            row["snapshots"][leg] = snapshot_sha
    if (len(licensed_build) != len(expected_build)
            or len(build_reviews) != len(expected_build)
            or len(build_bindings) != len(expected_build)
            or {binding.get("key") for binding in build_bindings if isinstance(binding, Mapping)}
            != set(expected_build)):
        raise ValueError("build package licence and Strix sets differ from installed files")
    approved_build = {}
    for row in licensed_build:
        if not isinstance(row, Mapping):
            raise ValueError("build package licence row is malformed")
        key, package_key, sha = row.get("key"), row.get("package_key"), row.get("source_sha256")
        fixture, fixture_sha = row.get("fixture"), row.get("fixture_sha256")
        if (not isinstance(key, str) or key not in expected_build or key in approved_build
                or not isinstance(package_key, str) or not isinstance(sha, str)
                or key != f"{package_key}/sha256/{sha}"
                or package_key != f"pypi/{expected_build[key]['name']}@{expected_build[key]['version']}"
                or sha != expected_build[key]["source_sha256"]
                or not isinstance(fixture, Mapping) or fixture.get("id") != key
                or fixture.get("dependency") != {
                    "ecosystem": "pypi", "name": expected_build[key]["name"],
                    "version": expected_build[key]["version"], "source_sha256": sha}
                or not isinstance(fixture_sha, str)
                or fixture_sha != hashlib.sha256(json.dumps(
                    fixture, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                or not isinstance(row.get("license"), str) or not row["license"]
                or set(row.get("legs", [])) != expected_build[key]["legs"]
                or row.get("snapshots") != expected_build[key]["snapshots"]):
            raise ValueError("build package licence differs from installed files")
        approved_build[key] = row
    seen_build_reviews = set()
    for review in build_reviews:
        if (not isinstance(review, Mapping) or review.get("key") not in approved_build
                or review["key"] in seen_build_reviews):
            raise ValueError("build package Strix review is missing or duplicated")
        row = approved_build[review["key"]]
        if any(review.get(field) != row.get(field) for field in
               ("package_key", "source_sha256", "license", "fixture_sha256", "legs")):
            raise ValueError("build package Strix review differs from licence verdict")
        seen_build_reviews.add(review["key"])


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "scope-set", "verdict", "artifacts", "attempt", "repository", "source-sha",
                 "control-sha", "run-id", "run-attempt", "record-id", "record-digest",
                 "verdict-id", "verdict-digest", "verdict-name"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    verify_full_set_verdict(
        json.loads(Path(args.manifest).read_text()), json.loads(Path(args.scope_set).read_text()),
        json.loads(Path(args.verdict).read_text()),
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
