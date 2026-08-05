"""Replace one stale logical task-order assertion on PR 541."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Assert the exact task-revision axis and its aligned logical labels."""
    path = Path("tests/test_scoring_enterprise_issue_calibration.py")
    text = path.read_text(encoding="utf-8")
    old = '''    assert all(
        design.task_ids == ("task_alpha", "task_beta")
        for design in bundle.designs
    )
'''
    new = '''    expected_task_revisions = tuple(
        sorted(
            (
                _digest("task-revision:alpha"),
                _digest("task-revision:beta"),
            )
        )
    )
    expected_task_labels = {
        _digest("task-revision:alpha"): "task_alpha",
        _digest("task-revision:beta"): "task_beta",
    }
    assert all(
        design.task_revision_fingerprints == expected_task_revisions
        for design in bundle.designs
    )
    assert all(
        dict(
            zip(
                design.task_revision_fingerprints,
                design.task_ids,
                strict=True,
            )
        )
        == expected_task_labels
        for design in bundle.designs
    )
'''
    if text.count(old) != 1:
        raise SystemExit("expected one stale logical task-order assertion")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
