"""Regression tests for the essay HTML report title trust boundary."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from fast_mlsirm.scoring.essay import render_essay_score_report_html

_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_report_html.py"))
)
clean_report = _REPORT_FIXTURES["clean_report"]


class _HostileTitle(str):
    """String subclass whose text callbacks must never cross the public boundary."""

    def strip(self, *args: object, **kwargs: object) -> str:
        """Fail if validation invokes caller-controlled string behavior."""
        raise AssertionError("hostile title strip callback executed")

    def replace(self, *args: object, **kwargs: object) -> str:
        """Fail if HTML escaping invokes caller-controlled string behavior."""
        raise AssertionError("hostile title replace callback executed")


def test_renderer_rejects_string_subclass_without_callbacks(tmp_path: Path) -> None:
    """Custom titles reject string subclasses before any caller callback or write."""
    output = tmp_path / "hostile-title.html"

    with pytest.raises(ValueError, match="title must be a non-empty string"):
        render_essay_score_report_html(
            clean_report(),
            output,
            title=_HostileTitle("audit title"),
        )

    assert not output.exists()
