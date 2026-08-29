"""Accessibility regressions for the shared essay-report color palette."""

import fast_mlsirm.scoring.essay.report_html as report_html


def test_shared_report_css_uses_distinct_theme_text_and_line_tokens() -> None:
    """Muted text and borders use explicit light/dark tokens, not system colors."""
    css = report_html._css()

    assert "GrayText" not in css
    assert "--muted: #60656f;" in css
    assert "--muted: #9e9e9e;" in css
    assert "--line: #d9ded6;" in css
    assert "--line: #333333;" in css
    assert (
        "section { margin-top: 20px; padding: 20px; "
        "border: 1px solid var(--line);"
    ) in css
    assert (
        "thead th, tbody th, td { padding: 10px; "
        "border: 1px solid var(--line);"
    ) in css
    assert (
        "pre { max-height: 32rem; overflow: auto; padding: 16px; "
        "border: 1px solid var(--line);"
    ) in css
    assert ".empty-state { font-style: italic; color: var(--muted); }" in css


def test_print_palette_resets_light_tokens_for_the_forced_white_canvas() -> None:
    """Dark-mode preference must not leak screen-dark palette tokens into print."""
    css = report_html._css()
    print_css = css.split("@media print", maxsplit=1)[1]

    assert (
        ":root { --review-required: #9c2f1f; --review-clear: #357a38; "
        "--muted: #60656f; --line: #d9ded6; }"
    ) in print_css
    assert "body { background: white; color: black; }" in print_css
