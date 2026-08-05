"""Normalize one prose boundary for the reviewed PR 546 patch script."""

from pathlib import Path

path = Path("docs/automated_essay_facets_calibration_reports.md")
content = path.read_text(encoding="utf-8")
old = "malformed deserialization. These are integrity and replay checks."
new = "malformed deserialization.\nThese are integrity and replay checks."
if content.count(old) != 1:
    raise SystemExit("expected one automated-essay replay prose boundary")
path.write_text(content.replace(old, new), encoding="utf-8")
