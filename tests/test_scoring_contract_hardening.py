"""Fail-closed hardening contracts for the shared scoring specification layer."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring as scoring
from fast_mlsirm.scoring import (
    ASSESSMENT_SCHEMA_VERSION,
    AssessmentSpec,
    AssessmentSpecError,
    EnginePolicy,
    artifact_digest,
    canonical_json,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]


def test_scoring_schema_version_is_owned_and_exported_independently() -> None:
    """Assessment wire evolution must not reuse the rubric schema constant."""
    spec = assessment()
    assessment_source = (
        Path(__file__).parents[1]
        / "python"
        / "fast_mlsirm"
        / "scoring"
        / "assessment.py"
    ).read_text(encoding="utf-8")

    assert ASSESSMENT_SCHEMA_VERSION == "1.0"
    assert scoring.ASSESSMENT_SCHEMA_VERSION == ASSESSMENT_SCHEMA_VERSION
    assert spec.schema_version == ASSESSMENT_SCHEMA_VERSION
    assert spec.to_dict()["schema_version"] == ASSESSMENT_SCHEMA_VERSION
    assert "ASSESSMENT_SCHEMA_VERSION" in scoring.__all__
    assert "SCHEMA_VERSION" not in assessment_source


def test_public_validation_errors_are_structured_and_non_reflective() -> None:
    """Caller-controlled values and metadata keys must never escape in errors."""
    secret_key = "customer_secret\nvalue"
    with pytest.raises(AssessmentSpecError) as metadata_error:
        assessment(metadata={secret_key: "private response content"})
    assert metadata_error.value.code == "invalid_json_key"
    assert secret_key not in metadata_error.value.path
    assert secret_key not in str(metadata_error.value)
    assert "private response content" not in str(metadata_error.value)

    with pytest.raises(AssessmentSpecError) as identifier_error:
        EnginePolicy(
            policy_id="invalid",
            allow_human_raters=True,
            allow_automated_raters=False,
        )
    assert identifier_error.value.code == "invalid_identifier"
    assert "invalid" not in str(identifier_error.value)

    with pytest.raises(AssessmentSpecError) as canonical_error:
        canonical_json({"unsupported_value": object()})
    assert canonical_error.value.code == "unsupported_json_value"


def test_negative_zero_has_one_canonical_identity() -> None:
    """Signed floating zero must not create duplicate assessment identities."""
    negative = {"threshold_value": -0.0}
    positive = {"threshold_value": 0.0}
    assert canonical_json(negative) == canonical_json(positive)
    assert artifact_digest(negative) == artifact_digest(positive)
    assert assessment(metadata=negative).assessment_fingerprint == assessment(
        metadata=positive
    ).assessment_fingerprint


def test_signed_integer_and_utf8_boundaries_fail_closed() -> None:
    """Portable metadata boundaries return stable domain errors, never raw exceptions."""
    minimum = assessment(metadata={"integer_value": -(1 << 63)})
    maximum = assessment(metadata={"integer_value": (1 << 63) - 1})
    assert minimum.metadata["integer_value"] == -(1 << 63)
    assert maximum.metadata["integer_value"] == (1 << 63) - 1

    for value in (-(1 << 63) - 1, 1 << 63, 10**400):
        with pytest.raises(AssessmentSpecError) as integer_error:
            canonical_json({"integer_value": value})
        assert integer_error.value.code == "json_integer_out_of_range"

    with pytest.raises(AssessmentSpecError) as unicode_error:
        canonical_json({"text_value": "\ud800"})
    assert unicode_error.value.code == "invalid_utf8_text"


def test_direct_assessment_construction_uses_the_domain_error_contract() -> None:
    """The factory seal must fail through the same structured public boundary."""
    spec = assessment()
    with pytest.raises(AssessmentSpecError) as error:
        AssessmentSpec(
            assessment_id=spec.assessment_id,
            assessment_version=spec.assessment_version,
            constructs=spec.constructs,
            rubric_fingerprints=spec.rubric_fingerprints,
            response_type=spec.response_type,
            engine_policy=spec.engine_policy,
            calibration_policy=spec.calibration_policy,
            validation_policy=spec.validation_policy,
            adjudication_policy=spec.adjudication_policy,
            monitoring_policy=spec.monitoring_policy,
            reporting_policy=spec.reporting_policy,
            metadata=spec.metadata,
        )
    assert error.value.code == "factory_required"
