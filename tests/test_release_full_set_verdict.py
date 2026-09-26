"""The complete central verdict must name this run's exact thirteen files."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.ci.verify_release_full_set_verdict import (
    verify_full_set_verdict, verify_runtime_dependency_coverage,
)


SOURCE = "a" * 40
CONTROL = "b" * 40
DIGEST = "sha256:" + "c" * 64


def _case() -> dict:
    identity = {"source_repository": "ContextualWisdomLab/fast-mlsirm",
                "source_sha": SOURCE, "control_sha": CONTROL,
                "run_id": 42, "run_attempt": 2}
    rows = [{"leg": f"wheel-{number}", "file": f"wheel-{number}.whl",
             "sha256": f"{number:064x}", "artifact_id": number + 10,
             "artifact_name": f"dist-wheel-wheel-{number}", "artifact_digest": DIGEST}
            for number in range(12)]
    rows.append({"leg": "sdist", "file": "source.tar.gz", "sha256": "d" * 64,
                 "artifact_id": 99, "artifact_name": "dist-sdist", "artifact_digest": DIGEST})
    manifest = {"schema_version": 1, **identity, "distributions": rows}
    scope_rows = [{"leg": row["leg"], "artifact_id": index + 200,
                   "artifact_name": f"repro-digest-{row['leg']}", "artifact_digest": DIGEST}
                  for index, row in enumerate(rows)]
    scope_set = {"schema_version": 1, **identity, "evidence": scope_rows}
    name = "release-dependency-sealed-evidence--full-set-verdict"
    binding = {"key": "pypi/example@1", "name": "release-strix-binding-a2-"
               + hashlib.sha256(b"pypi/example@1").hexdigest(),
               "id": 102, "digest": DIGEST}
    archive_key = "pypi/example@1/sha256/" + "f" * 64
    archive_binding = {"key": archive_key, "name": "release-strix-binding-a2-"
                       + hashlib.sha256(archive_key.encode()).hexdigest(),
                       "id": 103, "digest": DIGEST}
    verdict = {"schema": "cwl.release-full-set-verdict/1", "result": "PASS", **identity,
               "record_artifact_id": 100, "record_artifact_digest": DIGEST,
               "distributions": copy.deepcopy(rows), "binding_artifacts": [binding],
               "runtime_archive_binding_artifacts": [archive_binding],
               "runtime_archive_license_sha256": "f" * 64,
               "scope_evidence": sorted(copy.deepcopy(scope_rows), key=lambda row: row["leg"])}

    def artifact(name: str, artifact_id: int) -> dict:
        return {"name": name, "id": artifact_id, "digest": DIGEST,
                "created_at": "2026-09-26T12:01:00Z", "expired": False,
                "workflow_run": {"id": 42, "head_sha": CONTROL}}

    return {"manifest": manifest, "scope_set": scope_set, "verdict": verdict,
            "artifacts": [artifact("reproducibility-record", 100), artifact(name, 101),
                          artifact(binding["name"], 102), artifact(archive_binding["name"], 103)]
                         + [artifact(row["artifact_name"], row["artifact_id"]) for row in rows]
                         + [artifact(row["artifact_name"], row["artifact_id"]) for row in scope_rows],
            "attempt": {"id": 42, "run_attempt": 2, "head_sha": CONTROL,
                        "run_started_at": "2026-09-26T12:00:00Z"},
            "name": name}


def _verify(case: dict) -> None:
    verify_full_set_verdict(
        case["manifest"], case["scope_set"], case["verdict"], case["artifacts"], case["attempt"],
        repository="ContextualWisdomLab/fast-mlsirm", source_sha=SOURCE,
        control_sha=CONTROL, run_id=42, run_attempt=2, record_id=100,
        record_digest=DIGEST, verdict_id=101, verdict_digest=DIGEST,
        verdict_name=case["name"],
    )


def test_accepts_exact_current_run_full_set() -> None:
    _verify(_case())


def test_rejects_forged_missing_stale_or_changed_verdict() -> None:
    def other_run(case):
        case["artifacts"][1]["workflow_run"]["id"] = 41

    def stale(case):
        case["artifacts"][2]["created_at"] = "2026-09-26T11:59:00Z"

    def wrong_source(case):
        case["verdict"]["source_sha"] = "f" * 40

    def changed_file(case):
        case["verdict"]["distributions"][0]["sha256"] = "f" * 64

    def missing_file(case):
        case["verdict"]["distributions"].pop()

    def missing_binding(case):
        case["artifacts"].pop(2)

    def extra_binding(case):
        case["artifacts"].append({**case["artifacts"][2], "id": 104,
                                  "name": "release-strix-binding-a2-" + "f" * 64})

    def wrong_attempt(case):
        case["attempt"]["run_attempt"] = 1

    def changed_verdict_digest(case):
        case["artifacts"][1]["digest"] = "sha256:" + "f" * 64

    def changed_binding_key(case):
        case["verdict"]["binding_artifacts"][0]["key"] = "pypi/other@1"

    def changed_scope_digest(case):
        case["verdict"]["scope_evidence"][0]["artifact_digest"] = "sha256:" + "f" * 64

    def foreign_scope_artifact(case):
        next(item for item in case["artifacts"] if item["name"].startswith("repro-digest-"))[
            "workflow_run"]["id"] = 1

    def missing_scope_artifact(case):
        case["artifacts"] = [item for item in case["artifacts"]
                             if item["name"] != case["scope_set"]["evidence"][0]["artifact_name"]]

    for mutate in (other_run, stale, wrong_source, changed_file, missing_file,
                   missing_binding, extra_binding, wrong_attempt, changed_verdict_digest,
                   changed_binding_key, changed_scope_digest, foreign_scope_artifact,
                   missing_scope_artifact):
        case = _case()
        mutate(case)
        with pytest.raises(ValueError):
            _verify(case)


def test_runtime_dependencies_require_matching_licensed_and_strix_bound_report() -> None:
    key = "pypi/numpy@2.5.1"
    archive_key = key + "/sha256/" + "d" * 64
    fixture = {"id": archive_key,
               "dependency": {"ecosystem": "pypi", "name": "numpy",
                              "version": "2.5.1", "source_sha256": "d" * 64}}
    fixture_sha = hashlib.sha256(json.dumps(fixture, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    legs = [f"wheel-{index}" for index in range(12)]
    archive_report = {"schema": "cwl.release-runtime-archive-licenses/1", "archives": [
        {"key": archive_key, "package_key": key, "name": "numpy", "version": "2.5.1",
         "source_sha256": "d" * 64, "license": "BSD-3-Clause",
         "fixture": fixture, "fixture_sha256": fixture_sha, "legs": legs}]}
    archive_raw = (json.dumps(archive_report, sort_keys=True) + "\n").encode()
    report = {"schema": "cwl.release-dependency-gate/1", "result": "PASS",
              "stage": "full", "source_repository": "ContextualWisdomLab/fast-mlsirm",
              "source_sha": SOURCE, "failures": [], "dependency_count": 1,
              "dependencies": [{"key": key, "ecosystem": "pypi", "name": "numpy",
                                "version": "2.5.1", "license": "BSD-3-Clause",
                                "source_sha256": "d" * 64, "fixture_sha256": "e" * 64}],
              "runtime_archive_reviews": [{"key": archive_key, "package_key": key,
                                           "source_sha256": "d" * 64,
                                           "license": "BSD-3-Clause", "fixture_sha256": fixture_sha,
                                           "legs": legs}]}
    raw = (json.dumps(report, sort_keys=True) + "\n").encode()
    verdict = {"gate_report_sha256": hashlib.sha256(raw).hexdigest(),
               "binding_artifacts": [{"key": key}],
               "runtime_archive_binding_artifacts": [{"key": archive_key}],
               "runtime_archive_license_sha256": hashlib.sha256(archive_raw).hexdigest()}
    receipts = [{"leg": f"wheel-{index}",
                 "locked_dependencies": [{"name": "numpy", "version": "2.5.1"}],
                 "archives": [{"name": "numpy", "version": "2.5.1", "sha256": "d" * 64}]}
                for index in range(12)]

    def check() -> None:
        verify_runtime_dependency_coverage(
            verdict, report, raw, archive_report, archive_raw, receipts,
            repository="ContextualWisdomLab/fast-mlsirm", source_sha=SOURCE,
        )

    check()
    report["dependencies"][0].update(key="pypi/numpy@2.5.2", version="2.5.2")
    verdict["binding_artifacts"][0]["key"] = "pypi/numpy@2.5.2"
    variant_report = (json.dumps(report, sort_keys=True) + "\n").encode()
    verdict["gate_report_sha256"] = hashlib.sha256(variant_report).hexdigest()
    verify_runtime_dependency_coverage(
        verdict, report, variant_report, archive_report, archive_raw, receipts,
        repository="ContextualWisdomLab/fast-mlsirm", source_sha=SOURCE,
    )
    report["dependencies"][0].update(key=key, version="2.5.1")
    verdict["binding_artifacts"][0]["key"] = key
    verdict["gate_report_sha256"] = hashlib.sha256(raw).hexdigest()
    for mutate in (
        lambda: verdict.update(gate_report_sha256="0" * 64),
        lambda: receipts[0]["locked_dependencies"][0].update(version="2.5.2"),
        lambda: receipts[0]["archives"][0].update(sha256="0" * 64),
        lambda: verdict["binding_artifacts"][0].update(key="pypi/other@1"),
        lambda: verdict["runtime_archive_binding_artifacts"][0].update(key="pypi/other@1/sha256/" + "d" * 64),
        lambda: verdict.update(runtime_archive_license_sha256="0" * 64),
    ):
        original = (copy.deepcopy(verdict), copy.deepcopy(receipts))
        mutate()
        with pytest.raises(ValueError):
            check()
        verdict.clear()
        verdict.update(original[0])
        receipts[:] = original[1]

    for mutate in (
        lambda: archive_report["archives"][0].update(license="GPL-3.0-only"),
        lambda: archive_report["archives"][0].update(source_sha256="0" * 64),
        lambda: report["runtime_archive_reviews"].clear(),
        lambda: report["runtime_archive_reviews"][0].update(fixture_sha256="0" * 64),
    ):
        original = (copy.deepcopy(archive_report), copy.deepcopy(report))
        mutate()
        with pytest.raises(ValueError):
            check()
        archive_report.clear()
        archive_report.update(original[0])
        report.clear()
        report.update(original[1])
