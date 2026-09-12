"""Accessibility regressions for the shared essay-report color palette."""

import re

import fast_mlsirm.scoring.essay.report_html as report_html


def _hex_token(css: str, token: str) -> str:
    match = re.search(rf"{re.escape(token)}:\s*(#[0-9a-fA-F]{{6}});", css)
    assert match is not None, f"missing {token} token"
    return match.group(1)


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_shared_report_css_uses_distinct_theme_text_and_line_tokens() -> None:
    """Muted text and borders use explicit theme tokens, not system colors."""
    css = report_html._css()

    assert "GrayText" not in css
    assert "--muted: #60656f;" in css
    assert "--muted: #9e9e9e;" in css
    assert "--line:" in css
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


def test_report_boundary_tokens_meet_non_text_contrast_on_each_canvas() -> None:
    """Component boundaries keep at least 3:1 contrast in light, dark, and print."""
    css = report_html._css()
    light_css, dark_and_print = css.split(
        "@media (prefers-color-scheme: dark)", maxsplit=1
    )
    dark_css, print_css = dark_and_print.split("@media print", maxsplit=1)

    assert _contrast_ratio(_hex_token(light_css, "--line"), "#ffffff") >= 3.0
    assert _contrast_ratio(_hex_token(dark_css, "--line"), "#000000") >= 3.0
    assert _contrast_ratio(_hex_token(print_css, "--line"), "#ffffff") >= 3.0


def test_print_palette_resets_light_tokens_for_the_forced_white_canvas() -> None:
    """Dark-mode preference must not leak screen-dark palette tokens into print."""
    css = report_html._css()
    print_css = css.split("@media print", maxsplit=1)[1]

    assert "--review-required: #9c2f1f;" in print_css
    assert "--review-clear: #357a38;" in print_css
    assert "--muted: #60656f;" in print_css
    assert "--line:" in print_css
    assert "body { background: white; color: black; }" in print_css
