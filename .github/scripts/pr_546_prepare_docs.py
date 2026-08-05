"""Normalize prose boundaries for the reviewed PR 546 patch script."""

from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    """Replace one formatting-only prose boundary or fail closed."""
    target = Path(path)
    content = target.read_text(encoding="utf-8")
    if content.count(old) != 1:
        raise SystemExit(f"expected one prose boundary in {path}")
    target.write_text(content.replace(old, new), encoding="utf-8")


replace_exact(
    "docs/automated_essay_facets_calibration_reports.md",
    "malformed deserialization. These are integrity and replay checks.",
    "malformed deserialization.\nThese are integrity and replay checks.",
)
replace_exact(
    "docs/doctoring/enterprise-issue-facets-calibration-reports.md",
    "metadata behavior. The\nsuite also proves",
    "metadata behavior.\nThe suite also proves",
)
