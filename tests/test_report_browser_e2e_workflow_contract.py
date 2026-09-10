"""Contract tests for the real-browser report evidence workflow."""

from pathlib import Path


_REPO_ROOT = Path(__file__).parents[1]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "report-browser-e2e.yml"
_VERIFIED_SOURCES = (
    "python/fast_mlsirm/rubric/item_bank_report.py",
    "scripts/build_benchmark_report.py",
    "scripts/build_buyer_packet.py",
    "scripts/build_commercial_release.py",
    "scripts/build_figma_evidence_sync.py",
    "scripts/build_pr_queue_governance.py",
    "scripts/build_procurement_due_diligence.py",
    "scripts/build_release_evidence_index.py",
    "tests/test_item_bank_report_browser_e2e.py",
    "tests/test_rubric_item_bank_report.py",
)


def test_report_browser_workflow_watches_every_verified_source() -> None:
    """A verified report or browser-test change must trigger real-browser evidence."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    for source_path in _VERIFIED_SOURCES:
        assert f'- "{source_path}"' in workflow
