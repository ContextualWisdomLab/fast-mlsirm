"""Focused contract tests for facets-report audit axes."""

from __future__ import annotations

import fast_mlsirm.scoring.essay.calibration_report_html as report_html


def test_repeatable_aligned_audit_labels_are_preserved() -> None:
    """Logical labels may repeat when exact fingerprint axes stay unique."""
    repeated = ("shared_label", "shared_label")

    assert report_html._identifier_axis(
        repeated,
        "task_ids",
        require_unique=False,
    ) == repeated
