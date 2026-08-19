"""Tests for prompt/model/rubric stratification on essay validation evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest

import fast_mlsirm.scoring.essay.validation_reporting as base_validation_reporting
from fast_mlsirm.scoring import (
    AdjudicationPolicy,
    AssessmentResponseType,
    AssessmentSpecError,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    MonitoringPolicy,
    ReportingPolicy,
    ValidationPolicy,
    build_assessment_spec,
)
from fast_mlsirm.scoring.essay import (
    EssayValidationStratum,
    build_essay_prompt,
    build_essay_validation_evidence_report,
    build_essay_validation_stratum,
    render_essay_validation_evidence_report_html,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
_automated_engine = _FIXTURES["automated_engine"]
_human_engine = _FIXTURES["human_engine"]
_rubric = _FIXTURES["rubric"]

_AUTOMATED = np.array([0, 1, 2, 2, 1, 0], dtype=np.int64)
_REFERENCE = np.array([0, 1, 2, 1, 1, 0], dtype=np.int64)
_HUMAN_A = np.array([0, 1, 2, 2, 1, 0], dtype=np.int64)
_HUMAN_B = np.array([0, 1, 2, 1, 1, 0], dtype=np.int64)
_SUBGROUP = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
_METRICS = (
    "adjacent_agreement",
    "exact_agreement",
    "human_machine_degradation",
    "pearson_correlation",
    "quadratic_weighted_kappa",
    "standardized_mean_difference",
    "worst_subgroup_standardized_mean_difference",
)


def _assessment(rubric=None):
    """Return one assessment that authorizes the exact validation fixture."""
    rubric_value = _rubric() if rubric is None else rubric
    construct = ConstructSpec(
        construct_id="evidence_quality",
        construct_definition="Evidence-conditioned response quality.",
        rubric_fingerprints=(rubric_value.fingerprint,),
    )
    construct_ids = (construct.construct_id,)
    return build_assessment_spec(
        assessment_id="validation_assessment",
        assessment_version="1.0.0",
        constructs=(construct,),
        rubrics=(rubric_value,),
        response_type=AssessmentResponseType.CRITERION_LEVEL,
        engine_policy=EnginePolicy(
            policy_id="engine_policy",
            engine_ids=("fixture_engine",),
            allow_human_raters=True,
            allow_automated_raters=True,
            minimum_raters_per_response=2,
        ),
        calibration_policy=CalibrationPolicy(
            policy_id="calibration_policy",
            model_id="facets_ordinal",
            construct_ids=construct_ids,
        ),
        validation_policy=ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=_METRICS,
            construct_ids=construct_ids,
        ),
        adjudication_policy=AdjudicationPolicy(
            policy_id="adjudication_policy",
            trigger_ids=("scorer_disagreement",),
            construct_ids=construct_ids,
        ),
        monitoring_policy=MonitoringPolicy(
            policy_id="monitoring_policy",
            metric_ids=("severity_drift",),
            construct_ids=construct_ids,
        ),
        reporting_policy=ReportingPolicy(
            policy_id="reporting_policy",
            format_ids=("json_report",),
            construct_ids=construct_ids,
            include_exact_values=True,
        ),
    )


def _prompt(**overrides):
    """Return one content-addressed prompt for validation stratification."""
    values = {
        "prompt_id": "argument_prompt",
        "task_family_id": "essay_review",
        "prompt_content_fingerprint": "1" * 64,
        "language_id": "english_language",
        "genre_id": "argument_genre",
        "maximum_response_characters": 5_000,
        "maximum_response_units": 1_000,
    }
    values.update(overrides)
    return build_essay_prompt(**values)


def _stratum(*, prompt=None, rubric=None, engine=None):
    """Return one exact prompt/model/rubric validation stratum."""
    return build_essay_validation_stratum(
        prompt=_prompt() if prompt is None else prompt,
        rubric=_rubric() if rubric is None else rubric,
        automated_engine=_automated_engine() if engine is None else engine,
    )


def _report(*, stratum=_stratum(), rubric=None, engine=None):
    """Return one criterion report with optional explicit stratification."""
    rubric_value = _rubric() if rubric is None else rubric
    engine_value = _automated_engine() if engine is None else engine
    assessment = _assessment(rubric_value)
    construct = assessment.constructs[0]
    return build_essay_validation_evidence_report(
        report_id="criterion_validation_report",
        assessment=assessment,
        construct_id=construct.construct_id,
        rubric_fingerprint=construct.rubric_fingerprints[0],
        criterion_id="claim_support",
        automated_engine=engine_value,
        reference_engine=_human_engine(),
        validation_dataset_fingerprint="d" * 64,
        automated_labels=_AUTOMATED,
        reference_labels=_REFERENCE,
        category_count=3,
        human_human_labels=(_HUMAN_A, _HUMAN_B),
        subgroup_labels=_SUBGROUP,
        validation_stratum=stratum,
    )


def test_validation_stratum_preserves_exact_prompt_model_and_rubric_scope() -> None:
    """One immutable stratum carries every buyer-visible validation dimension."""
    prompt = _prompt()
    rubric = _rubric()
    engine = _automated_engine()
    stratum = _stratum(prompt=prompt, rubric=rubric, engine=engine)
    payload = stratum.to_dict()

    assert stratum.prompt_fingerprint == prompt.prompt_fingerprint
    assert stratum.prompt_id == prompt.prompt_id
    assert stratum.genre_id == prompt.genre_id
    assert stratum.language_id == prompt.language_id
    assert stratum.model_family_id == engine.engine_family_id
    assert stratum.rubric_fingerprint == rubric.fingerprint
    assert stratum.rubric_version == rubric.rubric_version
    assert payload["stratum_fingerprint"] == stratum.stratum_fingerprint
    assert stratum.stratum_handle.startswith("essay_validation_stratum_")
    assert stratum == _stratum(prompt=prompt, rubric=rubric, engine=engine)


def test_validation_report_records_stratum_and_routes_unstratified_evidence() -> None:
    """Reports expose explicit strata and flag pooled evidence when scope is absent."""
    stratum = _stratum()
    report = _report(stratum=stratum)
    payload = report.to_dict()

    assert report.validation_stratum == stratum
    assert payload["validation_stratum"] == stratum.to_dict()
    assert "validation_stratification_missing" not in report.review_trigger_ids

    pooled = _report(stratum=None)
    assert pooled.validation_stratum is None
    assert pooled.to_dict()["validation_stratum"] is None
    assert "validation_stratification_missing" in pooled.review_trigger_ids


def test_validation_report_rejects_stratum_identity_drift() -> None:
    """Rubric and automated-model strata cannot be replayed across report scopes."""
    alternate_rubric = replace(_rubric(), rubric_version="1.0.1")
    with pytest.raises(AssessmentSpecError) as rubric_error:
        _report(stratum=_stratum(rubric=alternate_rubric))
    assert rubric_error.value.code == "essay_validation_stratum_rubric_mismatch"

    alternate_engine = _automated_engine(engine_family_id="alternate_family")
    with pytest.raises(AssessmentSpecError) as engine_error:
        _report(stratum=_stratum(engine=alternate_engine))
    assert engine_error.value.code == "essay_validation_stratum_engine_mismatch"

    with pytest.raises(AssessmentSpecError) as type_error:
        _report(stratum=object())
    assert type_error.value.code == "invalid_essay_validation_stratum"


def test_validation_stratum_factory_rejects_untrusted_scope_inputs() -> None:
    """Only exact essay prompt, rubric, and automated engine contracts can scope evidence."""
    cases = (
        ({"prompt": object()}, "invalid_essay_validation_prompt"),
        ({"rubric": object()}, "invalid_essay_validation_rubric"),
        ({"automated_engine": object()}, "invalid_essay_validation_automated_engine"),
        ({"automated_engine": _human_engine()}, "invalid_essay_validation_automated_engine"),
        (
            {"prompt": _prompt(task_family_id="unknown_family")},
            "essay_validation_prompt_task_family_mismatch",
        ),
    )
    for overrides, code in cases:
        with pytest.raises(AssessmentSpecError) as caught:
            build_essay_validation_stratum(
                prompt=overrides.get("prompt", _prompt()),
                rubric=overrides.get("rubric", _rubric()),
                automated_engine=overrides.get(
                    "automated_engine", _automated_engine()
                ),
            )
        assert caught.value.code == code


def test_validation_stratum_direct_construction_is_rejected() -> None:
    """Callers cannot forge a stratification identity without the public factory."""
    valid = _stratum()
    with pytest.raises(AssessmentSpecError) as caught:
        EssayValidationStratum(
            prompt_fingerprint=valid.prompt_fingerprint,
            prompt_id=valid.prompt_id,
            genre_id=valid.genre_id,
            language_id=valid.language_id,
            model_family_id=valid.model_family_id,
            rubric_fingerprint=valid.rubric_fingerprint,
            rubric_version=valid.rubric_version,
        )
    assert caught.value.code == "unverified_essay_validation_stratum"


def test_validation_html_replay_preserves_explicit_and_pooled_strata(tmp_path: Path) -> None:
    """Standalone HTML replay must retain the exact stratification payload."""
    explicit = _report(stratum=_stratum())
    explicit_path = render_essay_validation_evidence_report_html(
        explicit,
        tmp_path / "explicit.html",
    )
    explicit_html = explicit_path.read_text(encoding="utf-8")
    assert explicit.validation_stratum is not None
    assert explicit.validation_stratum.stratum_fingerprint in explicit_html
    assert "validation_stratum" in explicit_html

    pooled = _report(stratum=None)
    pooled_path = render_essay_validation_evidence_report_html(
        pooled,
        tmp_path / "pooled.html",
    )
    pooled_html = pooled_path.read_text(encoding="utf-8")
    assert "validation_stratification_missing" in pooled_html
    assert "&quot;validation_stratum&quot;: null" in pooled_html


def test_validation_html_replay_preserves_legacy_base_reports(tmp_path: Path) -> None:
    """Existing direct-module base reports remain renderable after stratification."""
    rubric = _rubric()
    assessment = _assessment(rubric)
    construct = assessment.constructs[0]
    legacy = base_validation_reporting.build_essay_validation_evidence_report(
        report_id="legacy_validation_report",
        assessment=assessment,
        construct_id=construct.construct_id,
        rubric_fingerprint=construct.rubric_fingerprints[0],
        criterion_id="claim_support",
        automated_engine=_automated_engine(),
        reference_engine=_human_engine(),
        validation_dataset_fingerprint="e" * 64,
        automated_labels=_AUTOMATED,
        reference_labels=_REFERENCE,
        category_count=3,
        human_human_labels=(_HUMAN_A, _HUMAN_B),
        subgroup_labels=_SUBGROUP,
    )
    output = render_essay_validation_evidence_report_html(
        legacy,
        tmp_path / "legacy.html",
    )
    html = output.read_text(encoding="utf-8")
    assert legacy.report_fingerprint in html
    assert "validation_stratum" not in html
