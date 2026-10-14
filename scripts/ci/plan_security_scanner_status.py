"""Pure unit3 dry plans, not a publisher or an authentication mechanism.

The only positive-intent mode is explicitly SYNTHETIC. Its canonical packet
fingerprint is an integrity check, not proof of origin. Real authentication,
caller wiring, Scorecard rendering and every publication remain separate gates.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from scripts.ci.security_scanner_evidence import SCANNERS, classify_security_evidence

DATA_KEYS = ("expected", "event", "run", "listing", "scope", "report")
OPEN_ACTIONS = {"opened", "reopened", "synchronize", "edited", "ready_for_review", "converted_to_draft"}


@dataclass(frozen=True)
class StatusIntent:
    """One context's inert intent and its individually classified job identity."""

    context: str
    intent: Literal["NO_PUBLISH", "SUCCESS_INTENT", "FAILURE_INTENT"]
    outcome: str
    reason: str
    job_id: int | None
    job_name: str | None


@dataclass(frozen=True)
class StatusPlan:
    """An immutable dry result, never a capability or executable status request."""

    binding: tuple[tuple[str, str | int], ...]
    evidence_class: Literal["SYNTHETIC", "UNVERIFIED"]
    provenance_bound: bool
    contexts: tuple[StatusIntent, ...]
    publication_authorized: Literal[False] = field(default=False, init=False)
    plan_kind: Literal["DRY_ONLY"] = field(default="DRY_ONLY", init=False)


def _canonical(value: Any) -> bytes:
    """Encode pure JSON without admitting non-finite values or numeric aliases."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _plan(classification: Mapping[str, Any], bound: bool, reason: str) -> StatusPlan:
    """Keep four exact contexts; unknown and soft outcomes never become success."""
    rows = []
    scanners = classification.get("scanners", {})
    for scanner in SCANNERS:
        row = scanners.get(scanner, {})
        outcome = row.get("outcome", "UNKNOWN")
        intent: Literal["NO_PUBLISH", "SUCCESS_INTENT", "FAILURE_INTENT"] = "NO_PUBLISH"
        why = reason
        if bound:
            why = "nonexecution_or_unavailable"
            if scanner == "scorecard":
                if outcome in {"SOFT_COMPLETE", "SOFT_FINDINGS"}:
                    why = "soft_semantics_pending_review"
            elif row.get("applicable") is True and row.get("job_status") == "completed":
                if outcome == "PASS" and row.get("job_conclusion") == "success":
                    intent, why = "SUCCESS_INTENT", "synthetic_dry_only"
                elif outcome == "FAIL" and row.get("job_conclusion") == "failure":
                    intent, why = "FAILURE_INTENT", "hard_gate_failure"
        rows.append(StatusIntent(scanner, intent, outcome, why,
                                 row.get("job_id"), row.get("job_name")))
    return StatusPlan(tuple(classification.get("binding", {}).items()),
                      "SYNTHETIC" if bound else "UNVERIFIED", bound, tuple(rows))


def plan_security_scanner_status(current: Mapping[str, Any], packet: Mapping[str, Any]) -> StatusPlan:
    """Verify the collector-shaped pure-data seam against independent current inputs.

    ``current`` has the existing classifier's expected signature: repository,
    PR/base/head, caller path/commit, supplier commit, event, run/attempt and five
    unique job names. Do not derive it from the packet being checked. The input
    packet uses unit1's expected/event/run/listing/scope/report/classification
    fields. Positive offline fixtures additionally supply authentication=
    SYNTHETIC and provenance={kind, binding, job_ids, packet_sha256}. These fields
    are not a live trust route. No authenticated-production mode is implemented.
    """
    classification: Mapping[str, Any] = {}
    try:
        if not isinstance(current, Mapping) or not isinstance(packet, Mapping):
            raise ValueError
        if _canonical(packet["expected"]) != _canonical(current):
            raise ValueError
        classification = classify_security_evidence(
            current, packet["event"], packet["run"], packet["listing"],
            packet["scope"], packet["report"])
        if _canonical(packet["classification"]) != _canonical(classification):
            raise ValueError
        if (packet["event"].get("action") not in OPEN_ACTIONS
                or packet["event"]["pull_request"].get("state") != "open"):
            return _plan(classification, False, "open_current_event_unavailable")
        if packet.get("authentication") != "SYNTHETIC":
            return _plan(classification, False, "authentication_unverified")
        provenance = packet.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("kind") != "SYNTHETIC":
            raise ValueError
        if _canonical(provenance.get("binding")) != _canonical(classification["binding"]):
            raise ValueError
        jobs = packet["listing"]["jobs"]
        if any(type(job.get("run_id")) is not int for job in jobs):
            raise ValueError
        job_ids = {key: next(job["id"] for job in jobs if job["name"] == name)
                   for key, name in current["job_names"].items()}
        if _canonical(provenance.get("job_ids")) != _canonical(job_ids):
            raise ValueError
        raw = _canonical({key: packet[key] for key in DATA_KEYS})
        if provenance.get("packet_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError, StopIteration, RecursionError):
        return _plan({}, False, "invalid_or_stale_source_evidence")
    return _plan(classification, True, "synthetic_dry_only")
