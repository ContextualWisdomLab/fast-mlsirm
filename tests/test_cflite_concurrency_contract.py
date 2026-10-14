"""Contract test for pull-request fuzzing concurrency."""
from tests.workflow_contract_source import workflow_source

from pathlib import Path


def test_cflite_cancels_only_the_superseded_head_for_one_repository_pr() -> None:
    caller = (Path(__file__).parents[1] / ".github" / "workflows" / "cflite_pr.yml").read_text()
    workflow = workflow_source(Path(__file__).parents[1] / ".github" / "workflows" / "cflite_pr.yml")

    assert (
        "group: ${{ github.workflow }}-${{ github.repository }}-"
        "${{ github.event.pull_request.number }}"
    ) in caller
    assert "cancel-in-progress: true" in caller
    assert "ready_for_review" in caller
    assert "converted_to_draft" in caller
    assert "github.event.pull_request.draft" in workflow
    assert "github.event.action != 'closed'" in workflow
