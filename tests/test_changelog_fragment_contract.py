"""Contract tests for authoritative unreleased changelog fragments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "render_changelog_fragments.py"
RUBRIC_FRAGMENT = ROOT / "docs" / "changelog.d" / "394-rubric-blueprint-compiler.md"


def _module():
    """Load the repository script without making ``scripts`` a package."""
    spec = importlib.util.spec_from_file_location("render_changelog_fragments", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fragment_renderer_is_deterministic_and_preserves_release_sections(tmp_path):
    """Stable rendering groups fragments under the canonical Unreleased sections."""
    first = tmp_path / "100-first.md"
    second = tmp_path / "200-second.md"
    first.write_text("# First feature\n\n## Added\n\n- Alpha.\n", encoding="utf-8")
    second.write_text("# Second fix\n\n## Fixed\n\n- Beta.\n", encoding="utf-8")
    module = _module()
    rendered = module.render_unreleased((first, second))
    assert rendered == module.render_unreleased((first, second))
    assert rendered.startswith("## Unreleased\n")
    assert "### Added\n\n#### First feature\n\n- Alpha." in rendered
    assert "### Fixed\n\n#### Second fix\n\n- Beta." in rendered


def test_rubric_fragment_is_authoritative_and_renderable():
    """The public rubric slice is release-noted now rather than deferred silently."""
    module = _module()
    source = RUBRIC_FRAGMENT.read_text(encoding="utf-8")
    assert "will be folded" not in source
    rendered = module.render_unreleased((RUBRIC_FRAGMENT,))
    assert "#### Rubric blueprint compiler" in rendered
    assert "SHA-256" in rendered
    assert "JSON Schema 2020-12" in rendered


def test_every_repository_fragment_matches_the_authoritative_format():
    """A release cannot silently omit a malformed or unclassified fragment."""
    module = _module()
    paths = module.fragment_paths()
    assert paths
    rendered = module.render_unreleased(paths)
    for path in paths:
        title, _ = module.parse_fragment(path)
        assert f"#### {title}" in rendered
