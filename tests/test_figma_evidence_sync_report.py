import base64
import hashlib

from scripts.build_figma_evidence_sync import (
    _content_security_policy,
    _render_report,
    _report_css,
)


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


def test_report_csp_hash_matches_exact_rendered_stylesheet() -> None:
    """The CSP must authorize only the exact package-owned rendered CSS bytes."""
    html = _render_report({})
    style_text = html.split("<style>", 1)[1].split("</style>", 1)[0]
    expected_hash = base64.b64encode(
        hashlib.sha256(style_text.encode("utf-8")).digest()
    ).decode("ascii")
    policy = _content_security_policy()

    assert style_text == _report_css()
    assert "'unsafe-inline'" not in policy
    assert f"style-src 'sha256-{expected_hash}'" in policy
