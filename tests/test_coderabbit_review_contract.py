"""Contract tests for bounded CodeRabbit review volume."""

from __future__ import annotations

from pathlib import Path


_CONFIG = Path(__file__).parents[1] / ".coderabbit.yaml"


def _config_text() -> str:
    """Return the repository CodeRabbit configuration as UTF-8 text."""
    return _CONFIG.read_text(encoding="utf-8")


def test_draft_pull_requests_are_not_reviewed_automatically():
    """Work-in-progress branches must not consume scarce automated reviews."""
    config = _config_text()
    assert "drafts: false" in config


def test_incremental_reviews_require_an_explicit_stable_head_request():
    """Rapid successive commits must not trigger a new paid review each time."""
    config = _config_text()
    assert "auto_incremental_review: false" in config
    assert "enabled: true" in config
