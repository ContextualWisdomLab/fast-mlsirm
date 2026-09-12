"""Palette and high-contrast contracts for standalone essay score reports."""

from __future__ import annotations

import re

import fast_mlsirm.scoring.essay.report_html as report_html


def _relative_luminance(hex_color: str) -> float:
    """Return WCAG relative luminance for one six-digit sRGB color."""
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    """Return the WCAG contrast ratio between two six-digit sRGB colors."""
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _theme_token(css: str, token: str, occurrence: int) -> str:
    """Return one six-digit theme token from the generated stylesheet."""
    matches = re.findall(rf"--{re.escape(token)}:\s*(#[0-9a-fA-F]{{6}});", css)
    assert len(matches) >= occurrence
    return matches[occurrence - 1]


def test_default_theme_palette_has_measured_text_and_boundary_contrast() -> None:
    """Default light/dark tokens preserve readable text and visible boundaries."""
    css = report_html._css()
    light_muted = _theme_token(css, "muted", 1)
    dark_muted = _theme_token(css, "muted", 2)
    light_line = _theme_token(css, "line", 1)
    dark_line = _theme_token(css, "line", 2)

    # Default-theme evidence uses the canonical light/dark Canvas baselines.
    # Browser/OS forced-color behavior is a separate contract below.
    assert _contrast_ratio(light_muted, "#ffffff") >= 4.5
    assert _contrast_ratio(dark_muted, "#000000") >= 4.5
    assert _contrast_ratio(light_line, "#ffffff") >= 3.0
    assert _contrast_ratio(dark_line, "#000000") >= 3.0


def test_forced_colors_delegate_palette_to_system_semantic_colors() -> None:
    """High-contrast mode must not pin report boundaries to author RGB values."""
    css = report_html._css()
    forced = re.search(
        r"@media \(forced-colors: active\)\s*\{(?P<body>.*?)\n\}",
        css,
        flags=re.DOTALL,
    )

    assert forced is not None
    body = forced.group("body")
    assert "--muted: CanvasText;" in body
    assert "--line: CanvasText;" in body


def test_graytext_is_not_used_as_active_report_content_color() -> None:
    """The active empty-state copy must not inherit disabled-text semantics."""
    assert "GrayText" not in report_html._css()
