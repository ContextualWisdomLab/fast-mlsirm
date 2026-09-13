"""Accessibility regression for diagnostics-report skip-link contrast."""

from __future__ import annotations

from fast_mlsirm.report import _css


def _token(block: str, name: str) -> str:
    """Return one hexadecimal custom-property value from a CSS root block."""
    prefix = f"--{name}: "
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).removesuffix(";")
    raise AssertionError(f"missing CSS token --{name}")


def _relative_luminance(value: str) -> float:
    """Return WCAG relative luminance for one six-digit hexadecimal color."""
    channels = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]

    def linear(channel: float) -> float:
        return (
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )

    red, green, blue = (linear(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(first: str, second: str) -> float:
    """Return the WCAG contrast ratio for two hexadecimal colors."""
    first_luminance = _relative_luminance(first)
    second_luminance = _relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def test_skip_link_uses_theme_background_text_with_aa_contrast() -> None:
    """The focused skip link must retain normal-text AA contrast in both themes."""
    css = _css()
    light_root = css.split(":root {", 1)[1].split("}", 1)[0]
    dark_media = css.split("@media (prefers-color-scheme: dark)", 1)[1]
    dark_root = dark_media.split(":root {", 1)[1].split("}", 1)[0]
    skip_link = css.split(".skip-link {", 1)[1].split("}", 1)[0]

    assert "background: var(--teal);" in skip_link
    assert "color: var(--bg);" in skip_link

    for root in (light_root, dark_root):
        assert _contrast_ratio(_token(root, "bg"), _token(root, "teal")) >= 4.5
