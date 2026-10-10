"""Workflow integration contract for bounded PR snapshot capture."""
from tests.workflow_contract_source import workflow_source

from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "hourly-pr-governance.yml"


def test_hourly_workflow_captures_split_snapshot_before_building_governance():
    """The hourly job must use bounded split capture before the existing builder."""
    text = workflow_source(WORKFLOW)
    capture = "python scripts/capture_pr_queue_snapshot.py"
    builder = "python scripts/build_pr_queue_governance.py"
    assert capture in text
    assert "--offline-snapshot" in text
    assert "github_snapshot.json" in text
    assert text.index(capture) < text.index(builder)

    wrapper = WORKFLOW.read_text(encoding="utf-8")
    push_start = wrapper.index("  push:\n")
    push_end = wrapper.index("\n\npermissions:", push_start)
    push_trigger = wrapper[push_start:push_end]
    assert "scripts/capture_pr_queue_snapshot.py" in push_trigger
    assert "tests/test_capture_pr_queue_snapshot.py" in push_trigger
    assert "tests/test_hourly_snapshot_split_workflow.py" in push_trigger


def test_hourly_workflow_publishes_raw_snapshot_with_derived_evidence():
    """Auditors receive the source snapshot alongside JSON and HTML derivations."""
    text = workflow_source(WORKFLOW)
    artifact_step = text.index("      - name: Publish hourly governance evidence")
    artifact_text = text[artifact_step:]
    assert "hourly-pr-queue-governance/github_snapshot.json" in artifact_text
    assert "pr_queue_governance_manifest.json" in artifact_text
    assert "pr_queue_governance_report.html" in artifact_text
