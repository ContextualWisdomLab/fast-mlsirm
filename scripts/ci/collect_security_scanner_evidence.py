"""Bounded run/job read-adapter contract; live authentication remains external.

GitHub's documented run-attempt and attempt-job GET routes are used. No live
transport, credentials, scanner, caller, status publisher or redirect follower
is provided. Unconfirmed fields are refused, not filled from requested IDs.
Scope stdout and same-job SARIF have no admitted retrieval seam in this unit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import quote, urlsplit

from scripts.ci.security_scanner_evidence import classify_security_evidence

API_ORIGIN = "https://api.github.com"
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_JOB_PAGES = 20
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


@dataclass(frozen=True)
class ReadRequest:
    """Declare one fixed-origin GET; the transport must bound bytes before reading."""

    url: str
    origin: str = API_ORIGIN
    auth_scope: str = "actions:read"
    accept: str = "application/vnd.github+json"
    api_version: str = "2026-03-10"
    max_body_bytes: int = MAX_BODY_BYTES


class ReadTransport(Protocol):
    """Trusted injected GET port, not an authentication assertion or live client.

    Return status, final URL, lowercase HTTP headers and complete received bytes.
    The accepted future transport must separately establish credential/TLS/source
    custody. A response's caller-supplied authenticated flag is never proof.
    """

    def get(self, request: ReadRequest) -> Mapping[str, Any]: ...


class ReadEvidenceError(ValueError):
    """Preserve sanitized completed-read observations on ordinary refusal."""

    def __init__(self, reason: str, reads: list[dict[str, Any]]) -> None:
        super().__init__(reason)
        self.reads = copy.deepcopy(reads)


def _trusted_repository(expected: Mapping[str, Any]) -> tuple[str, str]:
    """Validate trusted repository path components before any transport call."""
    repository = expected.get("repository")
    if (not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository)
            or any(part in (".", "..") for part in repository.split("/"))):
        raise ValueError("invalid trusted security repository")
    owner, name = repository.split("/", 1)
    return quote(owner, safe=""), quote(name, safe="")


def _json_get(transport: ReadTransport, url: str, reads: list[dict[str, Any]]) -> tuple[Any, Mapping[str, Any]]:
    """Validate complete JSON frames, retaining no raw headers or credentials."""
    parsed = urlsplit(url)
    if f"{parsed.scheme}://{parsed.netloc}" != API_ORIGIN or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("security read URL left the fixed API origin")
    try:
        response = transport.get(ReadRequest(url=url))
    except Exception:
        raise ValueError("security read transport did not return a response") from None
    if not isinstance(response, Mapping):
        raise ValueError("security read response is incomplete")
    status, final_url, headers, body = (response.get(key) for key in ("status", "url", "headers", "body"))
    if (type(status) is not int or status != 200 or final_url != url
            or not isinstance(headers, Mapping) or type(body) is not bytes
            or len(body) > MAX_BODY_BYTES):
        raise ValueError("security read failed, redirected, or exceeded its bound")
    content_type, content_length = headers.get("content-type"), headers.get("content-length")
    if (not isinstance(content_type, str)
            or content_type.split(";", 1)[0].strip().lower() != "application/json"
            or not isinstance(content_length, str) or not content_length.isascii()
            or not content_length.isdecimal() or int(content_length) != len(body)):
        raise ValueError("security read body is truncated or not JSON")
    reads.append({"url": url, "status": status, "size": len(body),
                  "sha256": hashlib.sha256(body).hexdigest()})

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject duplicate JSON object fields, including nested records."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("security read JSON contains duplicate keys")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        """Reject non-JSON floating constants rather than accepting NaN/Infinity."""
        raise ValueError("security read JSON contains non-JSON constants")

    def finite_float(value: str) -> float:
        """Refuse finite JSON syntax that overflows to a non-finite native float."""
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("security read JSON contains non-finite numeric overflow")
        return number

    try:
        payload = json.loads(body.decode("utf-8"), object_pairs_hook=unique,
                             parse_constant=invalid_constant, parse_float=finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("security read JSON is malformed") from exc
    return payload, headers


def _run(expected: Mapping[str, Any], transport: ReadTransport, root: str,
         reads: list[dict[str, Any]]) -> dict[str, Any]:
    """Require exact supplied run identity; do not fabricate absent metadata."""
    record, _ = _json_get(transport, root, reads)
    if not isinstance(record, dict):
        raise ValueError("workflow run attempt is not an object")
    required = {"id": expected.get("run_id"), "run_attempt": expected.get("run_attempt"),
                "event": expected.get("event"), "path": expected.get("caller_path"),
                "head_sha": expected.get("caller_sha")}
    if (any(record.get(key) != value for key, value in required.items())
            or type(record.get("id")) is not int or type(record.get("run_attempt")) is not int
            or not isinstance(record.get("repository"), Mapping)
            or record["repository"].get("full_name") != expected.get("repository")):
        raise ValueError("workflow run attempt differs from trusted identity")
    return copy.deepcopy(record)


def _next_page(headers: Mapping[str, Any], root: str, page: int) -> bool:
    """Admit only canonical pagination URLs in the same repository/run/attempt."""
    link = headers.get("link", "")
    if not isinstance(link, str):
        raise ValueError("workflow pagination link is malformed")
    if not link:
        return False
    seen: set[str] = set()
    has_next = False
    for part in link.split(","):
        match = re.fullmatch(r'<([^>]+)>;\s*rel="(next|prev|first|last)"', part.strip())
        if match is None:
            raise ValueError("workflow pagination relation is unsupported")
        target, relation = match.groups()
        number_match = re.fullmatch(re.escape(root + "/jobs?per_page=100&page=") + r"([1-9][0-9]*)", target)
        if number_match is None or relation in seen:
            raise ValueError("workflow pagination left its exact attempt or duplicated a relation")
        number = int(number_match.group(1))
        if number > MAX_JOB_PAGES:
            raise ValueError("workflow pagination exceeds its bound")
        seen.add(relation)
        if relation == "next":
            if number != page + 1:
                raise ValueError("workflow pagination skips a page")
            has_next = True
    return has_next


def _jobs(expected: Mapping[str, Any], transport: ReadTransport, root: str,
          reads: list[dict[str, Any]]) -> dict[str, Any]:
    """Read the complete bounded listing, preserving raw admitted job identity."""
    collected: list[dict[str, Any]] = []
    total = -1
    for page in range(1, MAX_JOB_PAGES + 1):
        payload, headers = _json_get(transport, f"{root}/jobs?per_page=100&page={page}", reads)
        if not isinstance(payload, dict) or type(payload.get("total_count")) is not int:
            raise ValueError("workflow jobs page is malformed")
        page_total = payload["total_count"]
        page_jobs = payload.get("jobs")
        if not isinstance(page_jobs, list) or len(page_jobs) > 100:
            raise ValueError("workflow jobs page exceeds its contract")
        if total < 0:
            total = page_total
            if total < 0 or total > 100:
                raise ValueError("workflow jobs exceed the adopted classifier's 100-job contract")
        elif page_total != total:
            raise ValueError("workflow jobs total changed during pagination")
        for item in page_jobs:
            if (not isinstance(item, dict) or type(item.get("run_id")) is not int
                    or item["run_id"] != expected.get("run_id")
                    or type(item.get("run_attempt")) is not int
                    or item["run_attempt"] != expected.get("run_attempt")):
                raise ValueError("workflow job lacks admitted raw run identity")
            collected.append(copy.deepcopy(item))
        next_exists = _next_page(headers, root, page)
        if len(collected) == total:
            if next_exists:
                raise ValueError("workflow jobs total contradicts its next-page link")
            break
        if not page_jobs or len(collected) > total or not next_exists:
            raise ValueError("workflow jobs pagination is incomplete or oversized")
    if total < 0 or len(collected) != total:
        raise ValueError("workflow jobs pagination did not settle")
    ids = [job.get("id") for job in collected]
    if any(type(value) is not int or value <= 0 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("workflow jobs contain missing or duplicate identities")
    return {"total_count": total, "jobs": collected}


def collect_security_scanner_evidence(
    expected: Mapping[str, Any], event: Mapping[str, Any], transport: ReadTransport
) -> dict[str, Any]:
    """Collect admitted metadata only; absent scope/SARIF always stay unverified."""
    trusted = copy.deepcopy(dict(expected))
    original_event = copy.deepcopy(dict(event))
    reads: list[dict[str, Any]] = []
    try:
        owner, repository = _trusted_repository(trusted)
        run_id, attempt = trusted.get("run_id"), trusted.get("run_attempt")
        if type(run_id) is not int or run_id <= 0 or type(attempt) is not int or attempt <= 0:
            raise ValueError("invalid trusted run attempt")
        root = f"{API_ORIGIN}/repos/{owner}/{repository}/actions/runs/{run_id}/attempts/{attempt}"
        run = _run(trusted, transport, root, reads)
        listing = _jobs(trusted, transport, root, reads)
        # Frozen classifier permits None at runtime; its annotations are retained.
        classification = classify_security_evidence(trusted, original_event, run, listing, None, None)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ReadEvidenceError(str(exc), reads) from exc
    return {"expected": trusted, "event": original_event, "run": run, "listing": listing,
            "scope": None, "report": None, "classification": classification,
            "reads": reads, "collector_schema": "cwl.security-actions-read/1",
            "authentication": "NOT_ESTABLISHED_BY_THIS_MODULE"}
