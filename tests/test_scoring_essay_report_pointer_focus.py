"""Fail-first coverage for pointer focus styling in essay HTML reports."""

from __future__ import annotations

from fast_mlsirm.scoring.essay import report_html


def test_main_pointer_focus_suppresses_default_outline_without_hiding_keyboard_focus() -> None:
    """Pointer focus stays visually quiet while focus-visible remains explicit."""
    css = report_html._css()
    assert "main:focus { outline: none; }" in css
    assert "main:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }" in css
