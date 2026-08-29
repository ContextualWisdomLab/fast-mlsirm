from scripts.build_figma_evidence_sync import _report_css


def _media_slice(css: str, marker: str) -> str:
    start = css.index(marker)
    return css[start:]


def test_dark_palette_is_screen_only() -> None:
    css = _report_css()

    assert "@media screen and (prefers-color-scheme: dark)" in css
    assert "@media (prefers-color-scheme: dark)" not in css


def test_print_palette_resets_to_light_tokens() -> None:
    css = _report_css()
    print_css = _media_slice(css, "@media print")

    for declaration in (
        "--text: #172026",
        "--bg: #f5f7f8",
        "--hero-bg: #12343b",
        "--hero-text: #fff",
        "--card-bg: #fff",
        "--border: #d8e1e3",
        "--meta-text: #5e6f76",
        "--focus-ring: #0f766e",
        "--table-border: #e8edef",
        "--hover-bg: #fbfcfa",
    ):
        assert declaration in print_css
