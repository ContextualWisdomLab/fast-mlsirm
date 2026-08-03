"""Contract tests for independent automated-scoring schema evolution."""

from __future__ import annotations

import fast_mlsirm.scoring as scoring
from fast_mlsirm.scoring import ASSESSMENT_SCHEMA_VERSION

from test_scoring_contracts import _assessment


def test_assessment_schema_version_is_scoring_owned_and_exported() -> None:
    """Assessment wire evolution must not depend on the rubric schema constant."""
    assessment = _assessment()

    assert ASSESSMENT_SCHEMA_VERSION == "1.0"
    assert scoring.ASSESSMENT_SCHEMA_VERSION == ASSESSMENT_SCHEMA_VERSION
    assert assessment.schema_version == ASSESSMENT_SCHEMA_VERSION
    assert assessment.to_dict()["schema_version"] == ASSESSMENT_SCHEMA_VERSION
    assert "ASSESSMENT_SCHEMA_VERSION" in scoring.__all__
