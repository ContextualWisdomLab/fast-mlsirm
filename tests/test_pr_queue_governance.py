import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def _load_governance():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pr_queue_governance.py"
    spec = importlib.util.spec_from_file_location("build_pr_queue_governance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_snapshot(path: Path) -> Path:
    snapshot = {
        "mode": "offline",
        "repo": "ContextualWisdomLab/fast-mlsirm",
        "default_branch": "main",
        "open_prs": [
            {
                "number": 60,
                "title": "Palette: CLI simulation command error handling",
                "headRefName": "feat/cli-dx-improvement",
                "baseRefName": "main",
                "isDraft": False,
                "mergeStateStatus": "UNKNOWN",
                "reviewDecision": "CHANGES_REQUESTED",
                "updatedAt": "2026-06-20T00:00:00Z",
                "url": "https://github.com/ContextualWisdomLab/fast-mlsirm/pull/60",
            },
            {
                "number": 59,
                "title": "Sentinel: add CSP to HTML reports",
                "headRefName": "sentinel-add-csp-to-html-reports",
                "baseRefName": "main",
                "isDraft": False,
                "mergeStateStatus": "UNKNOWN",
                "reviewDecision": "CHANGES_REQUESTED",
                "updatedAt": "2026-06-19T00:00:00Z",
                "url": "https://github.com/ContextualWisdomLab/fast-mlsirm/pull/59",
            },
            {
                "number": 51,
                "title": "Add GPU support for estimation models",
                "headRefName": "copilot/add-gpu-support-for-estimation-models",
                "baseRefName": "main",
                "isDraft": False,
                "mergeStateStatus": "BLOCKED",
                "reviewDecision": "",
                "updatedAt": "2026-07-02T00:00:00Z",
                "url": "https://github.com/ContextualWisdomLab/fast-mlsirm/pull/51",
            },
            {
                "number": 73,
                "title": "Add procurement due diligence evidence gate",
                "headRefName": "codex/20b-procurement-due-diligence",
                "baseRefName": "main",
                "isDraft": False,
                "mergeStateStatus": "QUEUED",
                "reviewDecision": "REVIEW_REQUIRED",
                "updatedAt": "2026-07-02T22:09:54Z",
                "url": "https://github.com/ContextualWisdomLab/fast-mlsirm/pull/73",
            },
        ],
    }
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def _args(root: Path, snapshot: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(root),
        out=str(out),
        repo="ContextualWisdomLab/fast-mlsirm",
        contract_value_krw=2_000_000_000,
        offline_snapshot=str(snapshot),
        offline_github=False,
        max_stale_days=7,
        generated_at="2026-07-03T00:00:00+00:00",
    )


def test_pr_queue_governance_creates_manifest_and_report(tmp_path):
    module = _load_governance()
    snapshot = _write_snapshot(tmp_path / "snapshot.json")

    manifest = module.build_pr_queue_governance(
        _args(tmp_path, snapshot, tmp_path / "pr-queue-governance")
    )

    assert manifest["status"] == "ok"
    assert manifest["open_pr_count"] == 4
    assert manifest["risk_counts"]["changes_requested"] == 2
    assert manifest["risk_counts"]["stale"] == 2
    assert manifest["risk_counts"]["release_scope_conflict"] == 1
    assert manifest["risk_counts"]["review_or_check_delay"] == 1
    assert manifest["failed_checks"] == []
    assert manifest["checks"][0]["category"] == "github"
    out = tmp_path / "pr-queue-governance"
    assert (out / "pr_queue_governance_manifest.json").exists()
    html = (out / "pr_queue_governance_report.html").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "PR queue governance table" in html


def test_classify_pr_marks_duplicate_and_release_scope_risks():
    module = _load_governance()

    classified = module.classify_pr(
        {
            "number": 51,
            "title": "Add GPU support for estimation models",
            "headRefName": "copilot/add-gpu-support-for-estimation-models",
            "baseRefName": "main",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "reviewDecision": "",
            "updatedAt": "2026-07-02T00:00:00Z",
            "url": "https://github.com/ContextualWisdomLab/fast-mlsirm/pull/51",
        },
        now=module._parse_datetime("2026-07-03T00:00:00+00:00"),
        max_stale_days=7,
    )

    assert classified["release_scope_conflict"] is True
    assert "model_or_backend_scope" in classified["risk_reasons"]


def test_pr_queue_governance_fails_without_snapshot_when_offline(tmp_path):
    module = _load_governance()
    args = argparse.Namespace(
        repo_root=str(tmp_path),
        out=str(tmp_path / "pr-queue-governance"),
        repo="ContextualWisdomLab/fast-mlsirm",
        contract_value_krw=2_000_000_000,
        offline_snapshot=None,
        offline_github=True,
        max_stale_days=7,
        generated_at="2026-07-03T00:00:00+00:00",
    )

    manifest = module.build_pr_queue_governance(args)

    assert manifest["status"] == "failed"
    failed = {check["name"] for check in manifest["failed_checks"]}
    assert "github:snapshot" in failed


def test_safe_url_allows_known_safe_schemes_and_blocks_unsafe_schemes():
    module = _load_governance()

    assert module._safe_url("https://github.com/org/repo/pull/1") == "https://github.com/org/repo/pull/1"
    assert module._safe_url("http://example.test/report") == "http://example.test/report"
    assert module._safe_url("mailto:security@example.test") == "mailto:security@example.test"
    assert module._safe_url("/relative/report.html") == "/relative/report.html"
    assert module._safe_url("javascript:alert(1)") == "#"
    assert module._safe_url("data:text/html,<script>alert(1)</script>") == "#"
    assert module._safe_url("vbscript:msgbox(1)") == "#"
    assert module._safe_url("ftp://example.test/report") == "#"
    assert module._safe_url("") == "#"
    assert module._safe_url(None) == "#"
