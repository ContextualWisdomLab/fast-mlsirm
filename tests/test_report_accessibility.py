"""Accessibility regression tests for generated diagnostics reports."""

import json

from fast_mlsirm.report import render_diagnostics_report


def _render_accessibility_report(tmp_path):
    """Render one deterministic report containing exact-value export blocks."""
    source = tmp_path / "fit_diagnostics.json"
    output = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2},
                "itemfit": {
                    "item_id": [0],
                    "outfit_mnsq": [1.0],
                    "observed_count": [1],
                },
            }
        ),
        encoding="utf-8",
    )
    render_diagnostics_report(source, output, title="Example Fit")
    return output.read_text(encoding="utf-8")


def test_rendered_report_has_semantic_hero_metadata(tmp_path):
    """Keep decorative branding hidden and source metadata machine-readable."""
    html = _render_accessibility_report(tmp_path)

    assert '<p aria-hidden="true">fast-mlsirm diagnostics</p>' in html
    assert '<dl class="hero-meta">' in html
    assert "<dt>Source</dt>" in html
    assert "<dd>fit_diagnostics.json</dd>" in html


def test_export_blocks_are_named_keyboard_scroll_regions(tmp_path):
    """Wide JSON and CSV exports remain reachable and visible from a keyboard."""
    html = _render_accessibility_report(tmp_path)

    assert (
        '<pre role="region" aria-label="JSON export for Item Fit" tabindex="0">'
        in html
    )
    assert (
        '<pre role="region" aria-label="CSV export for Item Fit" tabindex="0">'
        in html
    )
    assert ".export-block pre:focus-visible {" in html
    assert "outline: 3px solid var(--teal);" in html
    assert "outline-offset: -2px;" in html
