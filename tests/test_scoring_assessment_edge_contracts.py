"""Boundary coverage for assessment rubric fingerprint materialization."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
import fast_mlsirm.scoring.assessment as assessment


_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)


def test_materialize_rubrics_hides_fingerprint_callback_failures(monkeypatch):
    """A rubric fingerprint failure becomes a stable package-owned error."""
    rubric = _FIXTURES["rubric"]()

    def _raise(_value):
        raise RuntimeError("fingerprint callback failed")

    monkeypatch.setattr(type(rubric), "fingerprint", property(_raise))
    with pytest.raises(AssessmentSpecError) as captured:
        assessment._materialize_rubrics((rubric,))
    assert captured.value.code == "invalid_rubric_fingerprint"
