"""Classify per-scanner security evidence without manufacturing scanner passes.

This pure contract consumes already-retrieved, REST-shaped producer records. It
does not call GitHub, run scanners, publish statuses, or establish approval.
"""

from __future__ import annotations

from typing import Any, Mapping
import re

SCANNERS = ("osv-scan", "dependency-review", "trivy-fs", "scorecard")
SUPPLIER_PATH = "ContextualWisdomLab/.github/.github/workflows/security-scan.yml@"
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


def _identity(expected: Mapping[str, Any], event: Mapping[str, Any],
              run: Mapping[str, Any], listing: Mapping[str, Any],
              scope: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Reject incomplete, duplicate or cross-run producer metadata."""
    try:
        repository = expected["repository"]
        if (not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository)
                or any(part in (".", "..") for part in repository.split("/"))
                or any(not isinstance(expected[key], str) or not _SHA.fullmatch(expected[key])
                       for key in ("base_sha", "head_sha", "caller_sha", "supplier_sha"))
                or type(expected["pr"]) is not int or expected["pr"] <= 0
                or type(expected["run_id"]) is not int or expected["run_id"] <= 0
                or type(expected["run_attempt"]) is not int or expected["run_attempt"] <= 0
                or expected["event"] not in ("pull_request", "pull_request_target")
                or not isinstance(expected["caller_path"], str)
                or not re.fullmatch(r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml", expected["caller_path"])):
            raise ValueError
        names = expected["job_names"]
        if (not isinstance(names, Mapping) or set(names) != {"scope", *SCANNERS}
                or any(not isinstance(value, str) or not value for value in names.values())
                or len(set(names.values())) != 5):
            raise ValueError
        pr = event["pull_request"]
        if (type(event["number"]) is not int
                or not isinstance(event.get("action"), str) or not event["action"]
                or event["number"] != expected["pr"] or type(pr["number"]) is not int
                or pr["number"] != expected["pr"]
                or pr["base"]["sha"] != expected["base_sha"]
                or pr["base"]["repo"]["full_name"] != repository
                or pr["head"]["sha"] != expected["head_sha"]
                or not isinstance(pr["head"]["repo"]["full_name"], str)
                or not _REPOSITORY.fullmatch(pr["head"]["repo"]["full_name"])):
            raise ValueError
        if (run["id"] != expected["run_id"] or type(run["id"]) is not int
                or run["run_attempt"] != expected["run_attempt"]
                or type(run["run_attempt"]) is not int
                or run["event"] != expected["event"]
                or run["repository"]["full_name"] != repository
                or run["path"] != expected["caller_path"]
                or run["head_sha"] != expected["caller_sha"]):
            raise ValueError
        refs = run["referenced_workflows"]
        if (not isinstance(refs, list) or len(refs) != 1
                or refs[0].get("path") != SUPPLIER_PATH + expected["supplier_sha"]
                or refs[0].get("sha") != expected["supplier_sha"]):
            raise ValueError
        jobs = listing["jobs"]
        if (not isinstance(jobs, list) or type(listing["total_count"]) is not int
                or listing["total_count"] != len(jobs) or len(jobs) > 100
                or any(not isinstance(job, Mapping) or type(job.get("id")) is not int
                       or job["id"] <= 0 or job.get("run_id") != expected["run_id"]
                       or job.get("run_attempt") != expected["run_attempt"]
                       or type(job.get("run_attempt")) is not int
                       or job.get("head_sha") != expected["caller_sha"]
                       or not isinstance(job.get("name"), str) for job in jobs)
                or len({job["id"] for job in jobs}) != len(jobs)
                or any(sum(job["name"] == name for job in jobs) > 1 for name in names.values())):
            raise ValueError
        job_ids = {name: next((job["id"] for job in jobs if job["name"] == value), None)
                   for name, value in names.items()}
        for evidence, owner in ((scope, "scope"), (report, "scorecard")):
            if evidence is None:
                continue
            if (not isinstance(evidence, Mapping)
                    or evidence.get("run_id") != expected["run_id"]
                    or type(evidence.get("run_id")) is not int
                    or evidence.get("run_attempt") != expected["run_attempt"]
                    or type(evidence.get("run_attempt")) is not int
                    or type(evidence.get("job_id")) is not int
                    or evidence["job_id"] != job_ids[owner]):
                raise ValueError
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        raise ValueError("security evidence identity mismatch") from exc


def classify_security_evidence(
    expected: Mapping[str, Any],
    event: Mapping[str, Any],
    run: Mapping[str, Any],
    listing: Mapping[str, Any],
    scope: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one explicit outcome per required scanner."""
    _identity(expected, event, run, listing, scope, report)
    names = expected["job_names"]
    jobs = {job["name"]: job for job in listing["jobs"]}
    scanners = {}
    scope_job = jobs.get(names["scope"])
    flags = None
    if (scope_job is not None and _completed(scope_job, ("Classify changed paths",)) and scope is not None
            and scope["job_id"] == scope_job["id"]
            and isinstance(scope.get("stdout"), str)):
        matches = re.findall(r"(?m)^changed-scope code=(true|false) deps=(true|false)$",
                             scope["stdout"])
        if len(matches) == 1:
            flags = {"code": matches[0][0] == "true", "deps": matches[0][1] == "true"}
    run_complete = (event["action"] != "closed" and run.get("status") == "completed"
                    and run.get("conclusion") in ("success", "failure"))
    for scanner in SCANNERS:
        job = jobs.get(names[scanner])
        if job is None:
            scanners[scanner] = {"outcome": "NONEXECUTION", "reason": "missing_job"}
            continue
        if not run_complete:
            scanners[scanner] = {"outcome": "NONEXECUTION", "reason": "run_not_successfully_completed", "job_id": job["id"]}
            continue
        if flags is None:
            scanners[scanner] = {"outcome": "NONEXECUTION", "reason": "scope_unavailable", "job_id": job["id"]}
            continue
        applicable = flags["code" if scanner in {"trivy-fs", "scorecard"} else "deps"]
        if not applicable:
            scanners[scanner] = {"outcome": "NONEXECUTION", "reason": "not_applicable", "job_id": job["id"]}
            continue
        if scanner == "scorecard":
            scanners[scanner] = _scorecard(job, report)
        elif _completed(job, REQUIRED_STEPS[scanner]):
            scanners[scanner] = {"outcome": "PASS", "job_id": job["id"]}
        else:
            scanners[scanner] = _non_pass(job)
    for scanner, row in scanners.items():
        job = jobs.get(names[scanner])
        row["applicable"] = (False if event["action"] == "closed" else None if flags is None
                             else flags["code" if scanner in {"trivy-fs", "scorecard"} else "deps"])
        row["job_name"] = names[scanner]
        row["job_status"] = None if job is None else job.get("status")
        row["job_conclusion"] = None if job is None else job.get("conclusion")
        if scanner == "scorecard":
            row["tool_conclusion"] = None if job is None else _steps(job).get("Run Scorecard", {}).get("conclusion")
    return {"binding": {key: expected[key] for key in (
        "repository", "pr", "base_sha", "head_sha", "event", "caller_path",
        "caller_sha", "supplier_sha", "run_id", "run_attempt")},
            "scanners": scanners}


REQUIRED_STEPS = {
    "osv-scan": ("Verify OSV base checkout", "Verify OSV head checkout", "Require OSV scan output", "Report PR-introduced OSV findings"),
    "dependency-review": ("Verify Dependency Review head checkout", "Check dependency review support", "Dependency review"),
    "trivy-fs": ("Verify Trivy head checkout", "Trivy filesystem scan", "Require Trivy SARIF output", "Print Trivy findings that failed the gate"),
}


def _steps(job: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return unique step records or reject ambiguous step identity."""
    steps = job.get("steps")
    if not isinstance(steps, list) or any(not isinstance(step, Mapping) or not isinstance(step.get("name"), str) for step in steps):
        raise ValueError("security evidence identity mismatch")
    names = [step["name"] for step in steps]
    ordinals = [step.get("number") for step in steps]
    if (len(names) != len(set(names))
            or any(type(number) is not int or number <= 0 for number in ordinals)
            or len(ordinals) != len(set(ordinals)) or ordinals != sorted(ordinals)):
        raise ValueError("security evidence identity mismatch")
    return {step["name"]: step for step in steps}


def _completed(job: Mapping[str, Any] | None, required: tuple[str, ...]) -> bool:
    """Require the successful ordered named subsequence, allowing optional steps."""
    if job is None or job.get("status") != "completed" or job.get("conclusion") != "success":
        return False
    steps = _steps(job)
    ordinals = [steps[name]["number"] for name in required if name in steps]
    return (len(ordinals) == len(required) and ordinals == sorted(ordinals)
            and all(steps.get(name, {}).get("status") == "completed"
                    and steps.get(name, {}).get("conclusion") == "success" for name in required))


def _non_pass(job: Mapping[str, Any]) -> dict[str, Any]:
    """Classify an applicable incomplete or failed scanner without passing it."""
    if job.get("status") == "completed" and job.get("conclusion") == "failure":
        return {"outcome": "FAIL", "job_id": job["id"]}
    if job.get("status") == "completed" and job.get("conclusion") in {"cancelled", "skipped"}:
        return {"outcome": "NONEXECUTION", "reason": job["conclusion"], "job_id": job["id"]}
    return {"outcome": "NONEXECUTION", "reason": "incomplete_or_unrecognized", "job_id": job["id"]}


def _scorecard_invocations(runs: list[Mapping[str, Any]]) -> None:
    """Reject explicit SARIF invocation errors without requiring optional fields."""
    for run in runs:
        invocations = run.get("invocations", [])
        if not isinstance(invocations, list):
            raise ValueError("malformed SARIF invocation inventory")
        for invocation in invocations:
            if not isinstance(invocation, Mapping):
                raise ValueError("malformed SARIF invocation")
            if "executionSuccessful" in invocation and invocation["executionSuccessful"] is not True:
                raise ValueError("SARIF tool execution failed")
            if "exitCode" in invocation and (type(invocation["exitCode"]) is not int
                                             or invocation["exitCode"] != 0):
                raise ValueError("SARIF tool exited unsuccessfully")
            for name in ("toolExecutionNotifications", "toolConfigurationNotifications"):
                notices = invocation.get(name, [])
                if not isinstance(notices, list):
                    raise ValueError("malformed SARIF notification inventory")
                for notice in notices:
                    if not isinstance(notice, Mapping) or notice.get("level") == "error":
                        raise ValueError("SARIF tool execution notification error")


def _scorecard(job: Mapping[str, Any], report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Separate posture findings from unavailable Scorecard tool execution."""
    if job.get("status") != "completed" or job.get("conclusion") not in {"success", "failure"}:
        return _non_pass(job)
    try:
        if (not _completed(job, ("Verify Scorecard head checkout", "Run Scorecard"))
                or report is None or report.get("job_id") != job["id"]):
            raise ValueError
        runs = report["sarif"]["runs"]
        if not isinstance(runs, list) or not runs or any(not isinstance(run, Mapping) for run in runs):
            raise ValueError
        _scorecard_invocations(runs)
        findings = 0
        for run in runs:
            results = run["results"]
            if not isinstance(results, list) or any(not isinstance(result, Mapping) for result in results):
                raise ValueError
            findings += len(results)
    except (KeyError, TypeError, ValueError):
        return {"outcome": "TOOL_UNAVAILABLE", "job_id": job["id"]}
    return {"outcome": "SOFT_FINDINGS" if findings else "SOFT_COMPLETE", "job_id": job["id"], "findings": findings}
