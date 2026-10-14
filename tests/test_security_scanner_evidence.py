"""Offline synthetic REST-shaped inputs, not scanner or Actions execution."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ci/security_scanner_evidence.py"
SUPPLIER = "31a9aefd8d075295c2ad09ac2a9a71949810c9e8"


def consumer():
    """Load only the owned pure evidence contract, never a numeric backend."""
    assert SOURCE.is_file(), "missing per-scanner consumer evidence contract"
    spec = importlib.util.spec_from_file_location("security_scanner_evidence", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case() -> list[Any]:
    """Synthetic API field shapes with actual pinned producer step names."""
    expected = {
        "repository": "ContextualWisdomLab/fast-mlsirm", "pr": 2302,
        "base_sha": "a" * 40, "head_sha": "b" * 40,
        "caller_sha": "c" * 40, "caller_path": ".github/workflows/security.yml",
        "event": "pull_request_target", "run_id": 123, "run_attempt": 2,
        "supplier_sha": SUPPLIER,
        "job_names": {
            "scope": "security / Detect changed scope",
            "osv-scan": "security / osv-scan",
            "dependency-review": "security / dependency-review",
            "trivy-fs": "security / trivy-fs",
            "scorecard": "security / scorecard",
        },
    }
    event = {"number": 2302, "action": "synchronize", "pull_request": {
        "number": 2302,
        "base": {"sha": expected["base_sha"], "repo": {"full_name": expected["repository"]}},
        "head": {"sha": expected["head_sha"], "repo": {"full_name": "fork-owner/fast-mlsirm"}},
    }}
    run = {"id": 123, "run_attempt": 2, "event": expected["event"],
           "repository": {"full_name": expected["repository"]},
           "path": expected["caller_path"], "head_sha": expected["caller_sha"],
           "status": "completed", "conclusion": "success",
           "referenced_workflows": [{"path": "ContextualWisdomLab/.github/.github/workflows/security-scan.yml@" + SUPPLIER,
                                     "sha": SUPPLIER}]}
    steps = {
        "scope": ["Classify changed paths"],
        "osv-scan": ["Verify OSV base checkout", "Verify OSV head checkout", "Require OSV scan output", "Report PR-introduced OSV findings"],
        "dependency-review": ["Verify Dependency Review head checkout", "Check dependency review support", "Dependency review"],
        "trivy-fs": ["Verify Trivy head checkout", "Trivy filesystem scan", "Require Trivy SARIF output", "Print Trivy findings that failed the gate"],
        "scorecard": ["Verify Scorecard head checkout", "Run Scorecard"],
    }
    jobs = []
    for index, (key, name) in enumerate(expected["job_names"].items(), 1):
        jobs.append({"id": index, "run_id": 123, "run_attempt": 2,
                     "head_sha": expected["caller_sha"], "name": name,
                     "status": "completed", "conclusion": "success",
                     "steps": [{"number": n, "name": step, "status": "completed", "conclusion": "success"}
                               for n, step in enumerate(steps[key], 1)]})
    listing = {"total_count": len(jobs), "jobs": jobs}
    scope = {"run_id": 123, "run_attempt": 2, "job_id": 1,
             "stdout": "changed-scope code=true deps=true\n"}
    report = {"run_id": 123, "run_attempt": 2, "job_id": 5,
              "sarif": {"runs": [{"results": [{"ruleId": "Branch-Protection"}]}]}}
    return [expected, event, run, listing, scope, report]


def test_aggregate_success_is_not_scanner_execution():
    """Missing scanner jobs stay missing even with aggregate workflow success."""
    module = consumer()
    inputs = case()
    inputs[3] = {"total_count": 1, "jobs": inputs[3]["jobs"][:1]}
    inputs[5] = None  # No orphan Scorecard artifact when that job is absent.
    result = module.classify_security_evidence(*inputs)
    assert set(result["scanners"]) == {"osv-scan", "dependency-review", "trivy-fs", "scorecard"}
    assert all(row["outcome"] == "NONEXECUTION" for row in result["scanners"].values())
    assert all(row["reason"] == "missing_job" for row in result["scanners"].values())


def test_identity_drift_rejects_entire_evidence_packet():
    """No scanner row may be classified from another run, ref, PR, or attempt."""
    module = consumer()
    mutations = (
        lambda values: values[1]["pull_request"]["head"].__setitem__("sha", "d" * 40),
        lambda values: values[2].__setitem__("run_attempt", 3),
        lambda values: values[2].__setitem__("path", ".github/workflows/other.yml"),
        lambda values: values[2]["referenced_workflows"][0].__setitem__("sha", "e" * 40),
        lambda values: values[3]["jobs"][1].__setitem__("run_attempt", 1),
        lambda values: values[3]["jobs"].append(copy.deepcopy(values[3]["jobs"][1])),
        lambda values: values[3].__setitem__("total_count", 99),
    )
    for mutate in mutations:
        values = case()
        mutate(values)
        try:
            module.classify_security_evidence(*values)
        except ValueError as error:
            assert "security evidence identity" in str(error)
        else:
            raise AssertionError("identity drift was accepted")


def test_completed_scanners_are_distinct_from_soft_scorecard_findings():
    """Actual hard-gate step names and valid SARIF determine separate outcomes."""
    result = consumer().classify_security_evidence(*case())
    assert set(result["scanners"]) == {"osv-scan", "dependency-review", "trivy-fs", "scorecard"}
    assert {name: row["outcome"] for name, row in result["scanners"].items()} == {
        "osv-scan": "PASS", "dependency-review": "PASS", "trivy-fs": "PASS",
        "scorecard": "SOFT_FINDINGS",
    }
    assert result["scanners"]["scorecard"]["findings"] == 1
    assert result["binding"]["run_attempt"] == 2
    assert result["binding"]["head_sha"] == "b" * 40


def test_closed_or_cancelled_run_has_only_nonexecution_rows():
    """Retained successful jobs cannot turn a closed/cancelled attempt into PASS."""
    for which in ("closed", "cancelled", "queued", "unknown"):
        values = case()
        if which == "closed":
            values[1]["action"] = "closed"
        elif which == "queued":
            values[2]["status"] = "queued"
        else:
            values[2]["conclusion"] = which
        result = consumer().classify_security_evidence(*values)
        assert all(row["outcome"] == "NONEXECUTION" for row in result["scanners"].values())


def test_unknown_or_conflicting_scope_is_nonexecution():
    """Missing/ambiguous scope stdout must not authorize the successful jobs."""
    for output in ("", "changed-scope code=maybe deps=true\n",
                   "changed-scope code=true deps=true\nchanged-scope code=false deps=false\n"):
        values = case()
        values[4]["stdout"] = output
        result = consumer().classify_security_evidence(*values)
        assert all(row["outcome"] == "NONEXECUTION" for row in result["scanners"].values())


def test_scope_skip_and_missing_hard_steps_do_not_pass():
    """Applicability false and incomplete native gates remain nonexecution."""
    values = case()
    values[4]["stdout"] = "changed-scope code=false deps=false\n"
    result = consumer().classify_security_evidence(*values)
    assert all(row["outcome"] == "NONEXECUTION" for row in result["scanners"].values())
    values = case()
    values[3]["jobs"][1]["steps"].pop()
    assert consumer().classify_security_evidence(*values)["scanners"]["osv-scan"]["outcome"] == "NONEXECUTION"


def test_scorecard_unavailable_is_not_soft_findings_or_hard_pass():
    """continue-on-error job success cannot conceal the real tool-step failure."""
    values = case()
    values[3]["jobs"][-1]["steps"][-1]["conclusion"] = "failure"
    row = consumer().classify_security_evidence(*values)["scanners"]["scorecard"]
    assert row["outcome"] == "TOOL_UNAVAILABLE"
    values[3]["jobs"][-1]["conclusion"] = "failure"
    assert consumer().classify_security_evidence(*values)["scanners"]["scorecard"]["outcome"] == "TOOL_UNAVAILABLE"
    values = case()
    values[5]["sarif"]["runs"][0]["results"] = []
    assert consumer().classify_security_evidence(*values)["scanners"]["scorecard"]["outcome"] == "SOFT_COMPLETE"


def test_scope_and_report_are_bound_to_their_unique_jobs():
    """A same-run envelope from another job cannot supply tool or scope evidence."""
    for index in (4, 5):
        values = case()
        values[index]["job_id"] = 999
        try:
            consumer().classify_security_evidence(*values)
        except ValueError as error:
            assert "security evidence identity" in str(error)
        else:
            raise AssertionError("foreign evidence job accepted")


def test_individual_hard_failure_is_not_hidden_by_aggregate_failure():
    """A terminal failed run still exposes its actual per-job hard failure."""
    values = case()
    values[2]["conclusion"] = "failure"
    values[3]["jobs"][1]["conclusion"] = "failure"
    result = consumer().classify_security_evidence(*values)
    assert result["scanners"]["osv-scan"]["outcome"] == "FAIL"
    assert result["scanners"]["trivy-fs"]["outcome"] == "PASS"


def test_rows_expose_applicability_and_raw_producer_outcomes():
    """Scope and observed job outcome remain explicit beside each classification."""
    values = case()
    values[4]["stdout"] = "changed-scope code=true deps=false\n"
    for job in values[3]["jobs"][1:3]:
        job["conclusion"] = "skipped"
        job["steps"] = []
    rows = consumer().classify_security_evidence(*values)["scanners"]
    assert rows["osv-scan"]["applicable"] is False
    assert rows["osv-scan"]["job_conclusion"] == "skipped"
    assert rows["trivy-fs"]["applicable"] is True
    assert rows["scorecard"]["tool_conclusion"] == "success"


def test_each_scanner_skip_cancel_missing_outcome_is_nonexecution():
    """Nonexecuted jobs include Scorecard; soft policy never converts them."""
    for index in range(1, 5):
        for conclusion in ("skipped", "cancelled", None, "unknown"):
            values = case()
            values[3]["jobs"][index]["conclusion"] = conclusion
            name = tuple(values[0]["job_names"])[index]
            assert consumer().classify_security_evidence(*values)["scanners"][name]["outcome"] == "NONEXECUTION"


def test_contract_is_pure_and_preserves_untrusted_inputs(monkeypatch):
    """Run the new positive case with network/subprocess operations denied."""
    import socket
    import subprocess

    def denied(*args, **kwargs):
        raise AssertionError("evidence contract attempted an external operation")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(subprocess, "run", denied)
    inputs = case()
    before = copy.deepcopy(inputs)
    result = consumer().classify_security_evidence(*inputs)
    assert result["scanners"]["scorecard"]["outcome"] == "SOFT_FINDINGS"
    assert inputs == before


def test_order1_rejects_reversed_duplicate_or_missing_ordinals():
    """Each hard scanner must prove the ordered sequence, not just a name set."""
    for index in (1, 2, 3):
        for mutation in ("reverse", "reverse_numbered", "duplicate", "missing", "zero", "boolean"):
            values = case()
            steps = values[3]["jobs"][index]["steps"]
            if mutation.startswith("reverse"):
                steps.reverse()
                if mutation == "reverse_numbered":
                    for ordinal, step in enumerate(steps, 1):
                        step["number"] = ordinal
            elif mutation == "duplicate":
                steps[-1]["number"] = steps[0]["number"]
            elif mutation == "missing":
                steps[-1].pop("number")
            else:
                steps[-1]["number"] = 0 if mutation == "zero" else True
            scanner = tuple(values[0]["job_names"])[index]
            try:
                rows = consumer().classify_security_evidence(*values)["scanners"]
            except ValueError as error:
                assert "security evidence identity" in str(error)
            else:
                assert rows[scanner]["outcome"] != "PASS", (scanner, mutation)


def test_order1_preserves_optional_producer_steps_and_other_outcomes():
    """Optional skipped supplier steps do not invalidate the required subsequence."""
    values = case()
    steps = values[3]["jobs"][1]["steps"]
    steps.insert(0, {"name": "Explain OSV scan mode and timeout budget",
                     "status": "completed", "conclusion": "success"})
    steps.insert(2, {"name": "Retry base OSV without transitive resolution",
                     "status": "completed", "conclusion": "skipped"})
    for ordinal, step in enumerate(steps, 1):
        step["number"] = ordinal
    rows = consumer().classify_security_evidence(*values)["scanners"]
    assert all(rows[name]["outcome"] == "PASS" for name in ("osv-scan", "dependency-review", "trivy-fs"))
    assert rows["scorecard"]["outcome"] == "SOFT_FINDINGS"


def test_sarif2_refuses_explicit_execution_errors_with_empty_results():
    """Report-level execution errors are unavailable tools, never clean posture."""
    invocations = [
        {"executionSuccessful": False, "toolExecutionNotifications": [{"level": "error"}]},
        {"executionSuccessful": False},
        {"executionSuccessful": True, "toolExecutionNotifications": [{"level": "error"}]},
        {"toolConfigurationNotifications": [{"level": "error"}]},
        {"exitCode": 1},
    ]
    for invocation in invocations:
        values = case()
        sarif_run = values[5]["sarif"]["runs"][0]
        sarif_run["results"] = []
        sarif_run["invocations"] = [invocation]
        row = consumer().classify_security_evidence(*values)["scanners"]["scorecard"]
        assert row["outcome"] == "TOOL_UNAVAILABLE", invocation


def test_sarif2_accepts_clean_optional_invocations_and_soft_findings():
    """Absent optional metadata and warnings do not create invented tool failures."""
    for invocation in (None, {}, {"executionSuccessful": True, "exitCode": 0},
                       {"executionSuccessful": True, "toolExecutionNotifications": [{"level": "warning"}]}):
        for findings in ([], [{"ruleId": "Branch-Protection", "level": "error"}]):
            values = case()
            sarif_run = values[5]["sarif"]["runs"][0]
            sarif_run["results"] = findings
            if invocation is not None:
                sarif_run["invocations"] = [invocation]
            row = consumer().classify_security_evidence(*values)["scanners"]["scorecard"]
            assert row["outcome"] == ("SOFT_FINDINGS" if findings else "SOFT_COMPLETE")
