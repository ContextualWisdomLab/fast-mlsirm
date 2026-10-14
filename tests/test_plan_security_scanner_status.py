"""New offline unit3 controls; SYNTHETIC data is never publishing authority."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

@pytest.fixture(autouse=True)
def deny_live_operations(monkeypatch):
    """All unit3 cases run with network and process creation denied."""
    import socket
    import subprocess
    def denied(*args, **kwargs):
        raise AssertionError("unexpected live operation")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ci/plan_security_scanner_status.py"
CONTEXTS = ("osv-scan", "dependency-review", "trivy-fs", "scorecard")
DATA_KEYS = ("expected", "event", "run", "listing", "scope", "report")


def load(path, name):
    """Import inspected pure definitions, not the predecessor test cohort."""
    assert path.is_file(), "missing pure dry status planner"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def planner():
    return load(SOURCE, "unit3_status_planner")


def classify(packet):
    from scripts.ci.security_scanner_evidence import classify_security_evidence
    packet["classification"] = classify_security_evidence(
        packet["expected"], packet["event"], packet["run"], packet["listing"],
        packet["scope"], packet["report"])


def packet_case():
    """Actual classifier over existing SYNTHETIC producer fixture definitions."""
    fixture = load(ROOT / "tests/test_security_scanner_evidence.py", "unit3_fixture")
    expected, event, run, listing, scope, report = fixture.case()
    event["pull_request"]["state"] = "open"
    packet: dict[str, Any] = dict(zip(DATA_KEYS, (copy.deepcopy(expected), event, run, listing, scope, report)))
    packet["authentication"] = "NOT_ESTABLISHED_BY_THIS_MODULE"
    classify(packet)
    return copy.deepcopy(expected), packet


def seal_synthetic(current, packet):
    """Integrity fingerprint only, not authentication or a real producer receipt."""
    raw = json.dumps({key: packet[key] for key in DATA_KEYS}, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()
    jobs = packet["listing"]["jobs"]
    packet["provenance"] = {
        "kind": "SYNTHETIC", "binding": copy.deepcopy(packet["classification"]["binding"]),
        "job_ids": {key: next(job["id"] for job in jobs if job["name"] == name)
                    for key, name in current["job_names"].items()
                    if any(job["name"] == name for job in jobs)},
        "packet_sha256": hashlib.sha256(raw).hexdigest(),
    }
    packet["authentication"] = "SYNTHETIC"


def synthetic_case():
    current, packet = packet_case()
    seal_synthetic(current, packet)
    return current, packet


def assert_held(plan):
    assert tuple(row.context for row in plan.contexts) == CONTEXTS
    assert all(row.intent == "NO_PUBLISH" for row in plan.contexts)
    assert plan.publication_authorized is False


def test_unknown_authentication_retains_all_four_no_publish_contexts():
    current, packet = packet_case()
    plan = planner().plan_security_scanner_status(current, packet)
    assert_held(plan)
    assert plan.evidence_class == "UNVERIFIED"


def test_fully_bound_synthetic_positive_and_soft_hold():
    current, packet = synthetic_case()
    before = copy.deepcopy((current, packet))
    plan = planner().plan_security_scanner_status(current, packet)
    assert tuple(row.context for row in plan.contexts) == CONTEXTS
    assert [row.intent for row in plan.contexts] == ["SUCCESS_INTENT"] * 3 + ["NO_PUBLISH"]
    assert plan.contexts[-1].outcome == "SOFT_FINDINGS"
    assert plan.contexts[-1].reason == "soft_semantics_pending_review"
    assert [row.job_id for row in plan.contexts] == [2, 3, 4, 5]
    assert dict(plan.binding) == packet["classification"]["binding"]
    assert plan.evidence_class == "SYNTHETIC" and plan.provenance_bound is True
    assert plan.publication_authorized is False
    assert (current, packet) == before


def test_boolean_bound_claim_is_not_provenance():
    current, packet = packet_case()
    packet["authentication"] = "SYNTHETIC"
    packet["provenance"] = {"bound": True}
    assert_held(planner().plan_security_scanner_status(current, packet))


@pytest.mark.parametrize("key,value", [("head_sha", "d" * 40), ("base_sha", "d" * 40),
    ("caller_sha", "d" * 40), ("supplier_sha", "d" * 40),
    ("caller_path", ".github/workflows/other.yml"), ("run_id", 124),
    ("run_attempt", 3), ("pr", 2303), ("repository", "other/repo")])
def test_independent_current_identity_drift_is_no_publish(key, value):
    current, packet = synthetic_case()
    current[key] = value
    assert_held(planner().plan_security_scanner_status(current, packet))


@pytest.mark.parametrize("tamper", ["digest", "binding", "job_id", "duplicate_id",
    "boolean_id", "extra_context", "missing_context", "forged_pass", "missing_source"])
def test_source_or_row_provenance_drift_is_no_publish(tamper):
    current, packet = synthetic_case()
    if tamper == "digest":
        packet["provenance"]["packet_sha256"] = "0" * 64
    elif tamper == "binding":
        packet["provenance"]["binding"]["run_attempt"] = 1
    elif tamper in {"job_id", "duplicate_id", "boolean_id"}:
        packet["provenance"]["job_ids"]["osv-scan"] = {
            "job_id": 999, "duplicate_id": 1, "boolean_id": True}[tamper]
    elif tamper == "extra_context":
        packet["classification"]["scanners"]["other"] = {"outcome": "PASS"}
    elif tamper == "missing_context":
        del packet["classification"]["scanners"]["osv-scan"]
    elif tamper == "forged_pass":
        packet["classification"]["scanners"]["scorecard"]["outcome"] = "PASS"
    else:
        del packet["listing"]
    assert_held(planner().plan_security_scanner_status(current, packet))


@pytest.mark.parametrize("mode", ["closed", "unknown_action", "skipped", "cancelled",
    "missing", "unknown_job", "scope_missing", "not_applicable"])
def test_actual_classifier_nonexecution_seam_stays_no_publish(mode):
    current, packet = packet_case()
    if mode == "closed":
        packet["event"]["action"] = "closed"
        packet["event"]["pull_request"]["state"] = "closed"
    elif mode == "unknown_action":
        packet["event"]["action"] = "unrecognized"
    elif mode == "missing":
        packet["listing"] = {"total_count": 1, "jobs": packet["listing"]["jobs"][:1]}
        packet["report"] = None
    elif mode == "scope_missing":
        packet["scope"] = None
    elif mode == "not_applicable":
        packet["scope"]["stdout"] = "changed-scope code=false deps=false\n"
    else:
        for job in packet["listing"]["jobs"][1:]:
            job["conclusion"] = {"skipped": "skipped", "cancelled": "cancelled",
                                 "unknown_job": "unrecognized"}[mode]
    classify(packet)
    seal_synthetic(current, packet)
    assert_held(planner().plan_security_scanner_status(current, packet))


@pytest.mark.parametrize("soft", ["SOFT_COMPLETE", "SOFT_FINDINGS", "TOOL_UNAVAILABLE"])
def test_scorecard_states_remain_distinct_and_no_publish(soft):
    current, packet = packet_case()
    if soft == "SOFT_COMPLETE":
        packet["report"]["sarif"]["runs"][0]["results"] = []
    elif soft == "TOOL_UNAVAILABLE":
        packet["report"] = None
    classify(packet)
    seal_synthetic(current, packet)
    row = planner().plan_security_scanner_status(current, packet).contexts[-1]
    assert row.outcome == soft and row.intent == "NO_PUBLISH"


def test_hard_failure_is_nonpassing_without_invented_vulnerability_cause():
    current, packet = packet_case()
    packet["listing"]["jobs"][1]["conclusion"] = "failure"
    packet["run"]["conclusion"] = "failure"
    classify(packet)
    seal_synthetic(current, packet)
    plan = planner().plan_security_scanner_status(current, packet)
    assert plan.contexts[0].intent == "FAILURE_INTENT"
    assert plan.contexts[0].reason == "hard_gate_failure"
    assert plan.contexts[1].intent == "SUCCESS_INTENT"
    assert plan.publication_authorized is False


@pytest.mark.parametrize("authentication", [None, True, "authenticated", "UNVERIFIED",
    "NOT_ESTABLISHED_BY_THIS_MODULE"])
def test_authentication_claims_do_not_authorize_intents(authentication):
    current, packet = synthetic_case()
    packet["authentication"] = authentication
    plan = planner().plan_security_scanner_status(current, packet)
    assert_held(plan)
    assert plan.evidence_class == "UNVERIFIED"


def test_actual_collector_data_seam_stays_held_without_scope_sarif_or_auth():
    fixture = load(ROOT / "tests/test_collect_security_scanner_evidence.py", "unit3_adapter_fixture")
    expected, event = fixture.identities()
    run, jobs = fixture.producer_records()
    root = fixture.API + "/repos/ContextualWisdomLab/fast-mlsirm/actions/runs/123/attempts/2"
    jobs_url = root + "/jobs?per_page=100&page=1"
    transport = fixture.RecordingTransport({root: [fixture.response(root, run)],
        jobs_url: [fixture.response(jobs_url, {"total_count": 5, "jobs": jobs})]})
    packet = fixture.module().collect_security_scanner_evidence(expected, event, transport)
    assert_held(planner().plan_security_scanner_status(expected, packet))
    assert len(transport.requests) == 2


def test_runtime_network_process_denial_and_frozen_result(monkeypatch):
    import dataclasses
    import socket
    import subprocess
    import urllib.request
    def denied(*args, **kwargs):
        raise AssertionError("unexpected live operation")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)
    current, packet = synthetic_case()
    plan = planner().plan_security_scanner_status(current, packet)
    assert plan.contexts[0].intent == "SUCCESS_INTENT"
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.publication_authorized = True
    with pytest.raises(ValueError):
        dataclasses.replace(plan, publication_authorized=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.contexts[0].intent = "PUBLISH"


@pytest.mark.parametrize("state,conclusion", [("queued", None), ("in_progress", None),
    ("completed", "cancelled"), ("completed", "skipped"), ("completed", "unrecognized")])
def test_incomplete_or_cancelled_run_seam_is_no_publish(state, conclusion):
    current, packet = packet_case()
    packet["run"].update(status=state, conclusion=conclusion)
    classify(packet)
    seal_synthetic(current, packet)
    assert_held(planner().plan_security_scanner_status(current, packet))


@pytest.mark.parametrize("field", ["job_name", "job_id", "job_status", "applicable"])
def test_success_row_cannot_be_relabelled_away_from_actual_job(field):
    current, packet = synthetic_case()
    packet["classification"]["scanners"]["osv-scan"][field] = {
        "job_name": "other / osv-scan", "job_id": 2.0,
        "job_status": "queued", "applicable": 1}[field]
    assert_held(planner().plan_security_scanner_status(current, packet))


def test_raw_boolean_run_id_is_not_an_integer_identity():
    current, packet = packet_case()
    current["run_id"] = packet["expected"]["run_id"] = packet["run"]["id"] = 1
    for job in packet["listing"]["jobs"]:
        job["run_id"] = 1
    packet["listing"]["jobs"][1]["run_id"] = True
    packet["scope"]["run_id"] = packet["report"]["run_id"] = 1
    classify(packet)
    seal_synthetic(current, packet)
    assert_held(planner().plan_security_scanner_status(current, packet))


@pytest.mark.parametrize("bad", [None, [], {}, {"classification": None}])
def test_malformed_packet_is_inert_not_a_partial_plan(bad):
    current, _ = packet_case()
    assert_held(planner().plan_security_scanner_status(current, bad))
