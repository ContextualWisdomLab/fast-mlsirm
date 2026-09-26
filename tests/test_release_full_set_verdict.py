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
    name = "release-dependency-sealed-evidence--full-set-verdict"
    binding = {"key": "pypi/example@1", "name": "release-strix-binding-a2-"
               + hashlib.sha256(b"pypi/example@1").hexdigest(),
               "id": 102, "digest": DIGEST}
    verdict = {"schema": "cwl.release-full-set-verdict/1", "result": "PASS", **identity,
               "record_artifact_id": 100, "record_artifact_digest": DIGEST,
               "distributions": copy.deepcopy(rows), "binding_artifacts": [binding]}

    def artifact(name: str, artifact_id: int) -> dict:
        return {"name": name, "id": artifact_id, "digest": DIGEST,
                "created_at": "2026-09-26T12:01:00Z", "expired": False,
                "workflow_run": {"id": 42, "head_sha": CONTROL}}

    return {"manifest": manifest, "verdict": verdict,
            "artifacts": [artifact("reproducibility-record", 100), artifact(name, 101),
                          artifact(binding["name"], 102)]
                         + [artifact(row["artifact_name"], row["artifact_id"]) for row in rows],
            "attempt": {"id": 42, "run_attempt": 2, "head_sha": CONTROL,
                        "run_started_at": "2026-09-26T12:00:00Z"},
            "name": name}


def _verify(case: dict) -> None:
    verify_full_set_verdict(
        case["manifest"], case["verdict"], case["artifacts"], case["attempt"],
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
        case["artifacts"].append({**case["artifacts"][2], "id": 103,
                                  "name": "release-strix-binding-a2-" + "f" * 64})

    def wrong_attempt(case):
        case["attempt"]["run_attempt"] = 1

    def changed_verdict_digest(case):
        case["artifacts"][1]["digest"] = "sha256:" + "f" * 64

    def changed_binding_key(case):
        case["verdict"]["binding_artifacts"][0]["key"] = "pypi/other@1"

    for mutate in (other_run, stale, wrong_source, changed_file, missing_file,
                   missing_binding, extra_binding, wrong_attempt, changed_verdict_digest,
                   changed_binding_key):
        case = _case()
        mutate(case)
        with pytest.raises(ValueError):
            _verify(case)


def test_runtime_dependencies_require_matching_licensed_and_strix_bound_report() -> None:
    key = "pypi/numpy@2.5.1"
    report = {"schema": "cwl.release-dependency-gate/1", "result": "PASS",
              "stage": "full", "source_repository": "ContextualWisdomLab/fast-mlsirm",
              "source_sha": SOURCE, "failures": [], "dependency_count": 1,
              "dependencies": [{"key": key, "ecosystem": "pypi", "name": "numpy",
                                "version": "2.5.1", "license": "BSD-3-Clause",
                                "source_sha256": "d" * 64, "fixture_sha256": "e" * 64}]}
    raw = (json.dumps(report, sort_keys=True) + "\n").encode()
    verdict = {"gate_report_sha256": hashlib.sha256(raw).hexdigest(),
               "binding_artifacts": [{"key": key}]}
    receipts = [{"leg": f"wheel-{index}",
                 "locked_dependencies": [{"name": "numpy", "version": "2.5.1"}]}
                for index in range(12)]

    def check() -> None:
        verify_runtime_dependency_coverage(
            verdict, report, raw, receipts,
            repository="ContextualWisdomLab/fast-mlsirm", source_sha=SOURCE,
        )

    check()
    for mutate in (
        lambda: verdict.update(gate_report_sha256="0" * 64),
        lambda: receipts[0]["locked_dependencies"][0].update(version="2.5.2"),
        lambda: verdict["binding_artifacts"][0].update(key="pypi/other@1"),
    ):
        original = (copy.deepcopy(verdict), copy.deepcopy(receipts))
        mutate()
        with pytest.raises(ValueError):
            check()
        verdict.clear()
        verdict.update(original[0])
        receipts[:] = original[1]
