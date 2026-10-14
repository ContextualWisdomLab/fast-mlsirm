"""Offline contracts for the bounded authenticated security evidence reader."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ci/collect_security_scanner_evidence.py"
SUPPLIER = "31a9aefd8d075295c2ad09ac2a9a71949810c9e8"
API = "https://api.github.com"


def module():
    """Load only the owned adapter source."""
    assert SOURCE.is_file(), "missing authenticated security evidence adapter"
    spec = importlib.util.spec_from_file_location("collect_security_scanner_evidence", SOURCE)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def identities():
    """Trusted expected identity and original PR event are separate inputs."""
    expected = {
        "repository": "ContextualWisdomLab/fast-mlsirm", "pr": 2302,
        "base_sha": "a" * 40, "head_sha": "b" * 40,
        "caller_sha": "c" * 40, "caller_path": ".github/workflows/security.yml",
        "event": "pull_request_target", "run_id": 123, "run_attempt": 2,
        "supplier_sha": SUPPLIER,
        "job_names": {"scope": "security / Detect changed scope",
                      "osv-scan": "security / osv-scan",
                      "dependency-review": "security / dependency-review",
                      "trivy-fs": "security / trivy-fs", "scorecard": "security / scorecard"},
    }
    event = {"number": 2302, "action": "synchronize", "pull_request": {
        "number": 2302,
        "base": {"sha": expected["base_sha"], "repo": {"full_name": expected["repository"]}},
        "head": {"sha": expected["head_sha"], "repo": {"full_name": "fork/fast-mlsirm"}},
    }}
    return expected, event


def producer_records():
    """Synthetic admitted fields; their actual upstream availability is unverified."""
    expected, _ = identities()
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
    for job_id, (key, name) in enumerate(expected["job_names"].items(), 11):
        jobs.append({"id": job_id, "name": name, "head_sha": expected["caller_sha"],
                     "run_id": expected["run_id"], "run_attempt": expected["run_attempt"],
                     "status": "completed", "conclusion": "success",
                     "steps": [{"number": number, "name": step, "status": "completed", "conclusion": "success"}
                               for number, step in enumerate(steps[key], 1)]})
    return run, jobs


def response(url, payload, *, status=200, body=None, headers=None):
    """Construct one complete synthetic transport response."""
    raw = json.dumps(payload, separators=(",", ":")).encode() if body is None else body
    return {"status": status, "url": url,
            "headers": {"content-type": "application/json", "content-length": str(len(raw)), **(headers or {})},
            "body": raw}


class RecordingTransport:
    """Record fixed requests and return predeclared responses without I/O."""

    def __init__(self, responses):
        self.responses = {key: list(value) for key, value in responses.items()}
        self.requests = []

    def get(self, request):
        self.requests.append(request)
        return self.responses[request.url].pop(0)


def test_collects_documented_run_and_jobs_without_fabricating_scope_or_sarif():
    """A complete REST read stays nonexecution until provenance seams exist."""
    adapter = module()
    expected, event = identities()
    run, jobs = producer_records()
    run_url = API + "/repos/ContextualWisdomLab/fast-mlsirm/actions/runs/123/attempts/2"
    jobs_url = run_url + "/jobs?per_page=100&page=1"
    transport = RecordingTransport({run_url: [response(run_url, run)],
                                    jobs_url: [response(jobs_url, {"total_count": 5, "jobs": jobs})]})
    before = copy.deepcopy((expected, event, run, jobs))
    result = adapter.collect_security_scanner_evidence(expected, event, transport)
    assert result["run"] == {**run}
    assert result["listing"]["total_count"] == 5
    assert all(job["run_id"] == 123 and job["run_attempt"] == 2 for job in result["listing"]["jobs"])
    assert result["scope"] is None and result["report"] is None
    assert all(row["outcome"] == "NONEXECUTION" for row in result["classification"]["scanners"].values())
    assert all(request.origin == API and request.auth_scope == "actions:read" for request in transport.requests)
    assert (expected, event, run, jobs) == before


def _transport(run=None, pages=None, *, mutate=None):
    """Build exact URLs and optionally corrupt one response."""
    expected, event = identities()
    default_run, jobs = producer_records()
    run = default_run if run is None else run
    run_url = API + "/repos/ContextualWisdomLab/fast-mlsirm/actions/runs/123/attempts/2"
    pages = pages or [{"total_count": len(jobs), "jobs": jobs}]
    responses = {run_url: [response(run_url, run)]}
    for number, payload in enumerate(pages, 1):
        url = f"{run_url}/jobs?per_page=100&page={number}"
        headers = {}
        if number < len(pages):
            next_url = f"{run_url}/jobs?per_page=100&page={number + 1}"
            headers["link"] = f'<{next_url}>; rel="next"'
        responses[url] = [response(url, payload, headers=headers)]
    if mutate:
        mutate(responses, run_url)
    return expected, event, RecordingTransport(responses)


def _rejects(expected, event, transport):
    try:
        module().collect_security_scanner_evidence(expected, event, transport)
    except ValueError:
        return
    raise AssertionError("invalid security evidence read was accepted")


def test_transport_failures_are_rejected_before_classification():
    """Characterize origin, status, redirect, type and length guards."""
    def status(responses, run_url):
        responses[run_url][0]["status"] = 500

    def redirect(responses, run_url):
        responses[run_url][0]["url"] = "https://example.invalid/run"

    def truncated(responses, run_url):
        responses[run_url][0]["headers"]["content-length"] = "999999"

    def text(responses, run_url):
        responses[run_url][0]["headers"]["content-type"] = "text/plain"

    for mutation in (status, redirect, truncated, text):
        _rejects(*_transport(mutate=mutation))
    expected, event, transport = _transport()
    expected["repository"] = "../repository"
    _rejects(expected, event, transport)


def test_pagination_duplicate_jobs_and_identity_drift_are_rejected():
    """Characterize complete-page and trusted-run identity requirements."""
    _, jobs = producer_records()
    _rejects(*_transport(pages=[{"total_count": 6, "jobs": jobs}]))
    _rejects(*_transport(pages=[{"total_count": 5, "jobs": jobs[:2]}, {"total_count": 4, "jobs": jobs[2:]}]))
    duplicate = copy.deepcopy(jobs)
    duplicate[-1]["id"] = duplicate[0]["id"]
    _rejects(*_transport(pages=[{"total_count": 5, "jobs": duplicate}]))
    for field, value in (("run_attempt", 1), ("path", ".github/workflows/other.yml"), ("head_sha", "d" * 40)):
        run, _ = producer_records()
        run[field] = value
        _rejects(*_transport(run=run))
    conflicting = copy.deepcopy(jobs)
    conflicting[1]["run_attempt"] = 1
    _rejects(*_transport(pages=[{"total_count": 5, "jobs": conflicting}]))


def test_classifier_identity_drift_still_fails_closed():
    """Event/supplier drift reaches the adopted classifier and is rejected."""
    expected, event, transport = _transport()
    event["pull_request"]["head"]["sha"] = "d" * 40
    _rejects(expected, event, transport)
    run, _ = producer_records()
    run["referenced_workflows"][0]["sha"] = "e" * 40
    _rejects(*_transport(run=run))


def test_missing_unconfirmed_job_identity_is_refused_not_filled():
    """Unknown upstream identity fields are not invented from a request URL."""
    _, jobs = producer_records()
    for job in jobs:
        job.pop("run_attempt", None)
    _rejects(*_transport(pages=[{"total_count": 5, "jobs": jobs}]))


def test_exact_multi_page_listing_has_explicit_source_identity_and_receipts():
    """Pagination uses declared next links and retains complete response digests."""
    _, jobs = producer_records()
    expected, event, transport = _transport(pages=[{"total_count": 5, "jobs": jobs[:2]}, {"total_count": 5, "jobs": jobs[2:]}])
    result = module().collect_security_scanner_evidence(expected, event, transport)
    assert [job["id"] for job in result["listing"]["jobs"]] == [11, 12, 13, 14, 15]
    assert len(result["reads"]) == 3
    assert all(len(item["sha256"]) == 64 for item in result["reads"])
    assert result["authentication"] == "NOT_ESTABLISHED_BY_THIS_MODULE"


def test_foreign_or_early_termination_pagination_cannot_be_accepted():
    """Redirect-like next links and contradictory extra pages fail closed."""
    _, jobs = producer_records()
    def foreign(responses, root):
        responses[root + "/jobs?per_page=100&page=1"][0]["headers"]["link"] = '<https://evil.invalid/jobs>; rel="next"'
    _rejects(*_transport(pages=[{"total_count": 5, "jobs": jobs[:2]}, {"total_count": 5, "jobs": jobs[2:]}], mutate=foreign))
    def extra(responses, root):
        next_url = root + "/jobs?per_page=100&page=2"
        responses[root + "/jobs?per_page=100&page=1"][0]["headers"]["link"] = f'<{next_url}>; rel="next"'
    _rejects(*_transport(mutate=extra))


def test_transport_exception_diagnostics_do_not_expose_injected_secrets():
    """Synthetic credential-bearing transport errors must not enter tracebacks."""
    import traceback
    expected, event, transport = _transport()
    def broken(request):
        raise RuntimeError("synthetic_auth_secret_42")
    transport.get = broken
    try:
        module().collect_security_scanner_evidence(expected, event, transport)
    except ValueError as error:
        assert "synthetic_auth_secret_42" not in "".join(traceback.format_exception(error))
    else:
        raise AssertionError("failed transport was accepted")


def test_no_live_io_or_scope_report_fabrication(monkeypatch):
    """Read recording frames while network/process APIs are denied."""
    import socket
    import subprocess
    def denied(*args, **kwargs):
        raise AssertionError("unexpected live operation")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(subprocess, "run", denied)
    run, _ = producer_records()
    run.update({"authenticated": True, "scope": {"code": True, "deps": True},
                "scorecard_sarif": {"runs": [{"results": []}]}})
    expected, event, transport = _transport(run=run)
    result = module().collect_security_scanner_evidence(expected, event, transport)
    assert result["scope"] is None and result["report"] is None
    assert result["authentication"] == "NOT_ESTABLISHED_BY_THIS_MODULE"
    assert all(row["outcome"] == "NONEXECUTION" for row in result["classification"]["scanners"].values())


def test_invalid_response_frames_and_duplicate_json_are_refused():
    """No contradictory byte frame or ambiguous JSON can be adopted."""
    bad = [b'{"id":123,"id":124}', b'{"number":NaN}', b'not-json', b'\xff']
    for body in bad:
        def mutation(responses, root):
            responses[root][0]["body"] = body
            responses[root][0]["headers"]["content-length"] = str(len(body))
        _rejects(*_transport(mutate=mutation))
    for code in (302, 401, 403):
        def status(responses, root):
            responses[root][0]["status"] = code
        _rejects(*_transport(mutate=status))


def _numeric_transport(literal, location):
    """Inject a raw JSON numeric token without changing frame/count identities."""
    marker = "__finite_json_numeric_token__"
    def numeric(responses, root):
        url = root + "/jobs?per_page=100&page=1" if location == "nested_job" else root
        frame = responses[url][0]
        payload = json.loads(frame["body"])
        if location == "top_run":
            payload["opaque_numeric_metadata"] = marker
        elif location == "nested_object":
            payload["opaque_numeric_metadata"] = {"nested": {"value": marker}}
        elif location == "nested_array":
            payload["opaque_numeric_metadata"] = [{"values": [1.25, [marker]]}]
        else:
            payload["jobs"][0]["opaque_numeric_metadata"] = {"values": [[marker]]}
        text = json.dumps(payload, separators=(",", ":"))
        assert text.count(json.dumps(marker)) == 1
        raw = text.replace(json.dumps(marker), literal, 1).encode()
        frame["body"] = raw
        frame["headers"]["content-length"] = str(len(raw))
    return _transport(mutate=numeric)


@pytest.mark.parametrize("literal", ["1e999", "-1e999"])
@pytest.mark.parametrize("location", ["top_run", "nested_object", "nested_array", "nested_job"])
def test_unit1_json_finite1_rejects_overflow_everywhere(literal, location):
    """Overflow tokens must fail before any nonfinite value enters output."""
    expected, event, transport = _numeric_transport(literal, location)
    with pytest.raises(ValueError, match="non-finite"):
        module().collect_security_scanner_evidence(expected, event, transport)
    assert len(transport.requests) == (2 if location == "nested_job" else 1)


@pytest.mark.parametrize("literal", ["1.25", "-1.25", "1e308", "-1e308"])
def test_unit1_json_finite1_preserves_finite_numbers(literal):
    """Finite nested metadata remains strict-serializable and nonexecuted."""
    expected, event, transport = _numeric_transport(literal, "nested_array")
    result = module().collect_security_scanner_evidence(expected, event, transport)
    number = result["run"]["opaque_numeric_metadata"][0]["values"][1][0]
    assert number == float(literal)
    json.dumps(result, allow_nan=False)
    assert all(row["outcome"] == "NONEXECUTION" for row in result["classification"]["scanners"].values())
