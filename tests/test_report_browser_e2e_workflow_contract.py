"""Contract tests for the real-browser report evidence workflow."""

from pathlib import Path


_REPO_ROOT = Path(__file__).parents[1]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "report-browser-e2e.yml"
_RENDERER_SOURCES = (
    "scripts/build_benchmark_report.py",
    "scripts/build_buyer_packet.py",
    "scripts/build_commercial_release.py",
    "scripts/build_figma_evidence_sync.py",
    "scripts/build_pr_queue_governance.py",
    "scripts/build_procurement_due_diligence.py",
    "scripts/build_release_evidence_index.py",
)


def test_report_browser_workflow_watches_every_verified_renderer_source() -> None:
    """A renderer-only PR must still trigger its real-browser evidence lane."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    for source_path in _RENDERER_SOURCES:
        assert f'- "{source_path}"' in workflow
