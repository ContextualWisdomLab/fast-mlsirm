"""Accessibility regressions for item-bank report table semantics."""

from __future__ import annotations

from pathlib import Path
import runpy

from fast_mlsirm.rubric.item_bank_report import render_item_bank_report_html


_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_report.py"))
)


def test_timeline_uses_semantic_row_headers_and_tabular_numbers() -> None:
    """Timeline identity cells remain row headers without visual weight drift."""
    records = _REPORT_FIXTURES["_lifecycle"]()

    rendered = render_item_bank_report_html(records)

    assert '<tr><th scope="row">piloting</th><td>pilot_admission</td>' in rendered
    assert '<tr><th scope="row">active</th><td>release_activation</td>' in rendered
    assert "font-variant-numeric:tabular-nums" in rendered
    assert "tbody th{font-weight:normal;}" in rendered
    assert "prefers-reduced-motion: reduce" in rendered
