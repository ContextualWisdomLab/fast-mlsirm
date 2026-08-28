"""Accessibility regressions for the shared essay-report color palette."""

import fast_mlsirm.scoring.essay.report_html as report_html


def test_shared_report_css_uses_theme_muted_token_instead_of_system_gray() -> None:
    """All essay report renderers inherit one explicit light/dark muted token."""
    css = report_html._css()

    assert "GrayText" not in css
    assert "--muted: #60656f;" in css
    assert "--muted: #9e9e9e;" in css
    assert (
        "section { margin-top: 20px; padding: 20px; "
        "border: 1px solid var(--muted);"
    ) in css
    assert (
        "thead th, tbody th, td { padding: 10px; "
        "border: 1px solid var(--muted);"
    ) in css
    assert (
        "pre { max-height: 32rem; overflow: auto; padding: 16px; "
        "border: 1px solid var(--muted);"
    ) in css
    assert ".empty-state { font-style: italic; color: var(--muted); }" in css


def test_print_palette_resets_muted_token_for_the_forced_white_canvas() -> None:
    """Dark-mode preference must not leak a low-contrast muted token into print."""
    css = report_html._css()
    print_css = css.split("@media print", maxsplit=1)[1]

    assert ":root { --muted: #60656f; }" in print_css
    assert "body { background: white; color: black; }" in print_css
