"""Focused accessibility evidence for cross-engine report boundaries."""

from __future__ import annotations

import re

from fast_mlsirm.cross_engine_report import _css


_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


def _channel(value: int) -> float:
    normalized = value / 255.0
    if normalized <= 0.04045:
        return normalized / 12.92
    return ((normalized + 0.055) / 1.055) ** 2.4


def _luminance(color: str) -> float:
    """Return WCAG relative luminance for one exact six-digit sRGB color."""
    assert _HEX_COLOR.fullmatch(color)
    red = _channel(int(color[1:3], 16))
    green = _channel(int(color[3:5], 16))
    blue = _channel(int(color[5:7], 16))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    lighter = max(_luminance(foreground), _luminance(background))
    darker = min(_luminance(foreground), _luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


def _token(scope: str, name: str) -> str:
    match = re.search(rf"--{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})", scope)
    assert match is not None
    return match.group(1)


def test_cross_engine_report_uses_theme_owned_three_to_one_boundary_contrast() -> None:
    """Table boundaries stay visible on the exact light, dark, and print canvases."""
    css = _css()
    assert "GrayText" not in css
    assert "border: 1px solid var(--line)" in css
    assert "background: var(--canvas)" in css

    root_end = css.index("}", css.index(":root")) + 1
    light_scope = css[:root_end]
    light_line = _token(light_scope, "line")
    light_canvas = _token(light_scope, "canvas")

    dark_marker = "@media screen and (prefers-color-scheme: dark)"
    dark_start = css.index(dark_marker)
    dark_end = css.index("}", css.index(":root", dark_start)) + 1
    dark_scope = css[dark_start:dark_end]
    dark_line = _token(dark_scope, "line")
    dark_canvas = _token(dark_scope, "canvas")

    print_start = css.index("@media print")
    print_scope = css[print_start:]
    print_line = _token(print_scope, "line")
    print_canvas = _token(print_scope, "canvas")

    assert print_line == light_line
    assert print_canvas == light_canvas
    assert _contrast(light_line, light_canvas) >= 3.0
    assert _contrast(dark_line, dark_canvas) >= 3.0
    assert _contrast(print_line, print_canvas) >= 3.0
