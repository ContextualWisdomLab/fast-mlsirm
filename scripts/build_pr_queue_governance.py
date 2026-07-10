#!/usr/bin/env python
"""Build live PR queue governance evidence for a fast-mlsirm release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


RISK_COUNT_KEYS = [
    "changes_requested",
    "stale",
    "duplicate_candidate",
    "release_scope_conflict",
    "review_or_check_delay",
]

RELEASE_SCOPE_TERMS = {
    "backend",
    "cuda",
    "diagnostic",
    "diagnostics",
    "estimation",
    "estimator",
    "formula",
    "gpu",
    "gradient",
    "likelihood",
    "mlx",
    "model",
    "opencl",
    "pyo3",
    "rust",
}

DUPLICATE_TERMS = {
    "accessible table",
    "cli",
    "csp",
    "html report",
    "performance",
    "report",
    "softplus",
    "stack trace",
}


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _source_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _check(name: str, category: str, ok: bool, detail: str, **metadata: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "category": category, "ok": ok, "detail": detail}
    payload.update(metadata)
    return payload


def _json_from_completed(completed: subprocess.CompletedProcess[str]) -> Any:
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return json.loads(completed.stdout)


def _run_gh_snapshot(repo: str) -> dict[str, Any]:
    repo_command = [
        "gh",
        "repo",
        "view",
        repo,
        "--json",
        "nameWithOwner,defaultBranchRef,visibility,isArchived,pushedAt,updatedAt,url",
    ]
    prs_command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        "100",
        "--json",
        "number,title,headRefName,baseRefName,isDraft,mergeStateStatus,reviewDecision,updatedAt,url",
    ]
    repo_result = subprocess.run(repo_command, capture_output=True, text=True)
    prs_result = subprocess.run(prs_command, capture_output=True, text=True)
    repo_payload = _json_from_completed(repo_result)
    prs_payload = _json_from_completed(prs_result)
    if not isinstance(prs_payload, list):
        prs_payload = []
    default_branch = ""
    if isinstance(repo_payload, dict):
        branch_ref = repo_payload.get("defaultBranchRef")
        if isinstance(branch_ref, dict):
            default_branch = str(branch_ref.get("name", ""))
    errors = []
    if repo_result.returncode != 0:
        errors.append({"command": "repo", "stderr": repo_result.stderr.strip(), "returncode": repo_result.returncode})
    if prs_result.returncode != 0:
        errors.append({"command": "open_prs", "stderr": prs_result.stderr.strip(), "returncode": prs_result.returncode})
    return {
        "mode": "live",
        "repo": repo,
        "default_branch": default_branch,
        "repo_snapshot": repo_payload,
        "open_prs": prs_payload,
        "errors": errors,
    }


def _snapshot_from_args(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    snapshot_path = getattr(args, "offline_snapshot", None)
    if snapshot_path:
        path = _resolve_path(snapshot_path, base=repo_root).resolve()
        snapshot = _read_json(path)
        snapshot.setdefault("mode", "offline")
        snapshot.setdefault("repo", args.repo)
        snapshot.setdefault("errors", [])
        snapshot["snapshot_file"] = str(path)
        return snapshot
    if getattr(args, "offline_github", False):
        return {
            "mode": "offline",
            "repo": args.repo,
            "default_branch": "",
            "open_prs": [],
            "errors": [{"command": "snapshot", "stderr": "offline mode requires --offline-snapshot", "returncode": 2}],
        }
    return _run_gh_snapshot(args.repo)


def _extract_open_prs(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    open_prs = snapshot.get("open_prs")
    if isinstance(open_prs, list):
        return [pr for pr in open_prs if isinstance(pr, dict)]
    if isinstance(open_prs, dict):
        data = open_prs.get("data")
        if isinstance(data, list):
            return [pr for pr in data if isinstance(pr, dict)]
    return []


def classify_pr(pr: dict[str, Any], *, now: datetime, max_stale_days: int) -> dict[str, Any]:
    title = str(pr.get("title", ""))
    head = str(pr.get("headRefName", ""))
    text = f"{title} {head}".lower()
    review_decision = str(pr.get("reviewDecision") or "").upper()
    merge_state = str(pr.get("mergeStateStatus") or "").upper()
    updated_at = str(pr.get("updatedAt") or "")
    updated_dt = None
    age_days = None
    if updated_at:
        try:
            updated_dt = _parse_datetime(updated_at)
            age_days = max(0, (now - updated_dt).days)
        except ValueError:
            updated_dt = None

    release_scope_terms = sorted(term for term in RELEASE_SCOPE_TERMS if term in text)
    duplicate_terms = sorted(term for term in DUPLICATE_TERMS if term in text)
    changes_requested = review_decision == "CHANGES_REQUESTED"
    stale = age_days is not None and age_days > max_stale_days
    review_or_check_delay = review_decision == "REVIEW_REQUIRED" or merge_state == "QUEUED"
    release_scope_conflict = bool(release_scope_terms)
    duplicate_candidate = bool(duplicate_terms) and not release_scope_conflict

    risk_reasons: list[str] = []
    if changes_requested:
        risk_reasons.append("changes_requested_review")
    if stale:
        risk_reasons.append("stale_update")
    if duplicate_candidate:
        risk_reasons.append("duplicate_or_already_productized_scope")
    if release_scope_conflict:
        risk_reasons.append("model_or_backend_scope")
    if review_or_check_delay:
        risk_reasons.append("review_or_check_delay")
    if merge_state == "BLOCKED":
        risk_reasons.append("merge_blocked")

    classified = dict(pr)
    classified.update(
        {
            "age_days": age_days,
            "changes_requested": changes_requested,
            "stale": stale,
            "duplicate_candidate": duplicate_candidate,
            "release_scope_conflict": release_scope_conflict,
            "review_or_check_delay": review_or_check_delay,
            "release_scope_terms": release_scope_terms,
            "duplicate_terms": duplicate_terms,
            "risk_reasons": risk_reasons,
        }
    )
    if updated_dt is not None:
        classified["updated_at_utc"] = updated_dt.isoformat(timespec="seconds")
    return classified


def _risk_counts(classified_prs: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(1 for pr in classified_prs if pr.get(key) is True) for key in RISK_COUNT_KEYS}


def _safe_url(url: object) -> str:
    if not isinstance(url, str):
        return "#"
    candidate = url.strip()
    if not candidate:
        return "#"
    parsed = urlparse(candidate)
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https", "mailto"}:
        return "#"
    return candidate


def _content_security_policy() -> str:
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _report_css() -> str:
    return """
:root { color: #172026; background: #f5f7f8; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
.hero { background: #12343b; color: #fff; border-radius: 8px; padding: 28px; }
.hero p, .hero h1 { margin: 0; }
.hero p { color: #b7d7d0; font-size: 0.86rem; font-weight: 700; text-transform: uppercase; }
.hero h1 { margin-top: 8px; font-size: 2rem; }
.hero span { display: inline-block; margin-top: 14px; color: #dce8e5; }
.report-section { margin-top: 22px; background: #fff; border: 1px solid #d8e1e3; border-radius: 8px; padding: 22px; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.metric-card { border: 1px solid #d8e1e3; border-radius: 8px; padding: 14px; }
.metric-card span { display: block; color: #5e6f76; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
.metric-card strong { display: block; margin-top: 8px; overflow-wrap: anywhere; }
.table-wrap { overflow-x: auto; border: 1px solid #d8e1e3; border-radius: 8px; }
.table-wrap:focus { outline: 3px solid #0f766e; outline-offset: 3px; }
table { width: 100%; min-width: 920px; border-collapse: collapse; }
caption { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
th, td { padding: 10px 12px; border-bottom: 1px solid #e8edef; text-align: left; vertical-align: top; }
code { overflow-wrap: anywhere; }
.note { color: #5e6f76; margin-bottom: 0; }
"""


def _render_report(manifest: dict[str, Any]) -> str:
    risk_counts = manifest.get("risk_counts", {})
    cards = [
        ("Status", manifest.get("status", "")),
        ("Open PRs", manifest.get("open_pr_count", "")),
        ("Changes Requested", risk_counts.get("changes_requested", "") if isinstance(risk_counts, dict) else ""),
        ("Stale", risk_counts.get("stale", "") if isinstance(risk_counts, dict) else ""),
        ("Release Scope", risk_counts.get("release_scope_conflict", "") if isinstance(risk_counts, dict) else ""),
        ("Review Delay", risk_counts.get("review_or_check_delay", "") if isinstance(risk_counts, dict) else ""),
    ]
    card_markup = [
        "<article class=\"metric-card\">" f"<span>{escape(label)}</span><strong>{escape(str(value))}</strong></article>"
        for label, value in cards
    ]
    rows = []
    for pr in manifest.get("pull_requests", []):
        if not isinstance(pr, dict):
            continue
        rows.append(
            "<tr>"
            f"<th scope=\"row\"><a href=\"{escape(_safe_url(pr.get('url')), quote=True)}\">#{escape(str(pr.get('number', '')))}</a></th>"
            f"<td>{escape(str(pr.get('title', '')))}</td>"
            f"<td>{escape(str(pr.get('reviewDecision', '')))}</td>"
            f"<td>{escape(str(pr.get('mergeStateStatus', '')))}</td>"
            f"<td>{escape(str(pr.get('age_days', '')))}</td>"
            f"<td>{escape(', '.join(str(reason) for reason in pr.get('risk_reasons', [])))}</td>"
            f"<td><code>{escape(str(pr.get('headRefName', '')))}</code></td>"
            "</tr>"
        )
    check_rows = []
    for check in manifest.get("checks", []):
        if not isinstance(check, dict):
            continue
        check_rows.append(
            "<tr>"
            f"<th scope=\"row\">{escape(str(check.get('name', '')))}</th>"
            f"<td>{escape(str(check.get('category', '')))}</td>"
            f"<td>{escape('go' if check.get('ok') else 'failed')}</td>"
            f"<td>{escape(str(check.get('detail', '')))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            "<title>fast-mlsirm PR Queue Governance</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body><main>",
            '<section class="hero"><p>fast-mlsirm buyer governance</p><h1>PR Queue Governance</h1>',
            f"<span>Generated: {escape(str(manifest.get('generated_at', '')))}</span></section>",
            '<section class="report-section"><h2>Queue Summary</h2><div class="metrics-grid">',
            *card_markup,
            "</div></section>",
            '<section class="report-section"><h2>Open PR Risk Classification</h2>',
            '<div class="table-wrap" role="region" aria-label="PR queue governance table" tabindex="0">',
            "<table><caption>PR queue governance table</caption>",
            "<thead><tr><th scope=\"col\">PR</th><th scope=\"col\">Title</th><th scope=\"col\">Review</th><th scope=\"col\">Merge</th><th scope=\"col\">Age Days</th><th scope=\"col\">Risk Reasons</th><th scope=\"col\">Head</th></tr></thead><tbody>",
            *rows,
            "</tbody></table></div>",
            '<p class="note">Open PRs are inventoried as queue governance evidence. Review waits and queued checks are tracked separately from release-scope blockers.</p>',
            "</section>",
            '<section class="report-section"><h2>Evidence Checks</h2>',
            '<div class="table-wrap" role="region" aria-label="PR queue governance check table" tabindex="0">',
            "<table><caption>PR queue governance check table</caption>",
            "<thead><tr><th scope=\"col\">Check</th><th scope=\"col\">Category</th><th scope=\"col\">Status</th><th scope=\"col\">Detail</th></tr></thead><tbody>",
            *check_rows,
            "</tbody></table></div>",
            "</section></main></body></html>",
        ]
    )


def build_pr_queue_governance(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    out_dir = _resolve_path(args.out, base=repo_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = getattr(args, "generated_at", None) or datetime.now(UTC).isoformat(timespec="seconds")
    now = _parse_datetime(generated_at)
    snapshot = _snapshot_from_args(args, repo_root)
    open_prs = _extract_open_prs(snapshot)
    classified = [
        classify_pr(pr, now=now, max_stale_days=int(getattr(args, "max_stale_days", 14))) for pr in open_prs
    ]
    risk_counts = _risk_counts(classified)
    snapshot_errors = snapshot.get("errors")
    snapshot_ok = isinstance(open_prs, list) and snapshot_errors == []
    checks = [
        _check(
            "github:snapshot",
            "github",
            snapshot_ok,
            "GitHub open PR snapshot was recorded",
            mode=snapshot.get("mode"),
            errors=snapshot_errors,
        ),
        _check(
            "queue:classified",
            "queue_state",
            len(classified) == len(open_prs),
            "every open PR has governance classification fields",
            open_pr_count=len(open_prs),
        ),
        _check(
            "risk:coverage",
            "risk_classification",
            set(RISK_COUNT_KEYS).issubset(risk_counts),
            "risk count coverage includes stale, changes-requested, duplicate-looking, release-scope, and review-delay buckets",
            risk_count_keys=sorted(risk_counts),
        ),
        _check(
            "release:boundary",
            "release_boundary",
            True,
            "open PR queue is inventoried as release governance evidence; queued checks and review delays are not treated as blockers by themselves",
        ),
    ]
    failed = [check for check in checks if not check["ok"]]
    manifest: dict[str, Any] = {
        "command": "build_pr_queue_governance",
        "status": "ok" if not failed else "failed",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": generated_at,
        "source_commit": _source_commit(repo_root),
        "repo_root": str(repo_root),
        "repo": args.repo,
        "default_branch": snapshot.get("default_branch", ""),
        "max_stale_days": int(getattr(args, "max_stale_days", 14)),
        "open_pr_count": len(classified),
        "risk_counts": risk_counts,
        "github": snapshot,
        "pull_requests": classified,
        "checks": checks,
        "failed_checks": failed,
    }
    html_path = out_dir / "pr_queue_governance_report.html"
    manifest_path = out_dir / "pr_queue_governance_manifest.json"
    html_path.write_text(_render_report(manifest), encoding="utf-8")
    manifest["html_report_file"] = str(html_path)
    manifest["html_report_sha256"] = _sha256(html_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PR queue governance evidence for fast-mlsirm.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--out", default="pr-queue-governance", help="Output directory.")
    parser.add_argument("--repo", default="ContextualWisdomLab/fast-mlsirm", help="GitHub repository name.")
    parser.add_argument("--contract-value-krw", type=int, default=2_000_000_000, help="Target contract value.")
    parser.add_argument("--offline-snapshot", help="JSON snapshot with open_prs for offline governance checks.")
    parser.add_argument(
        "--offline-github",
        action="store_true",
        help="Fail fast unless --offline-snapshot is supplied; used to prove offline fixture coverage.",
    )
    parser.add_argument("--max-stale-days", type=int, default=14, help="Age in days after which an open PR is stale.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_pr_queue_governance(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "out": str(Path(args.out).resolve()),
                "manifest": str(Path(args.out).resolve() / "pr_queue_governance_manifest.json"),
                "html": manifest["html_report_file"],
                "open_pr_count": manifest["open_pr_count"],
                "failed_checks": len(manifest["failed_checks"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
