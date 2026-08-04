"""Public-package export tests for governed essay score reporting."""

from __future__ import annotations

import fast_mlsirm.scoring.essay as essay
import fast_mlsirm.scoring.essay.reporting as reporting


def test_score_reporting_symbols_are_exported_from_essay_package() -> None:
    """The issue-level essay namespace exposes the complete report API."""
    expected = {
        "EssayScoreReport",
        "MAX_ESSAY_REPORT_REVIEW_TRIGGERS",
        "build_essay_score_report",
    }

    assert expected <= set(essay.__all__)
    assert essay.EssayScoreReport is reporting.EssayScoreReport
    assert (
        essay.MAX_ESSAY_REPORT_REVIEW_TRIGGERS
        == reporting.MAX_ESSAY_REPORT_REVIEW_TRIGGERS
    )
    assert essay.build_essay_score_report is reporting.build_essay_score_report
    assert essay.EssayScoreReport.__doc__
    assert essay.build_essay_score_report.__doc__
