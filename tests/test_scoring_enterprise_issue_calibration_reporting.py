"""Governed enterprise issue calibration-report orchestration tests."""

from __future__ import annotations

import inspect

import pytest

import fast_mlsirm.scoring._contract_safety as contract_safety
import fast_mlsirm.scoring.enterprise_issue as enterprise
import fast_mlsirm.scoring.enterprise_issue.calibration_reporting as reporting_module
from enterprise_issue_calibration_fixtures import _execution
from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    fit_enterprise_issue_facets_calibration_reports,
)


def _connected_executions():
    """Return one connected two-criterion enterprise calibration design."""
    return (
        _execution(
            issue_label="alpha",
            task_label="alpha",
            engine_label="alpha",
            scores=(0, 1),
        ),
        _execution(
            issue_label="alpha",
            task_label="beta",
            engine_label="beta",
            scores=(1, 2),
        ),
        _execution(
            issue_label="beta",
            task_label="alpha",
            engine_label="beta",
            scores=(2, 0),
        ),
        _execution(
            issue_label="beta",
            task_label="beta",
            engine_label="alpha",
            scores=(1, 2),
        ),
    )


def test_public_fitter_delegates_each_shared_design(monkeypatch) -> None:
    """Every criterion is fitted through the canonical shared report boundary."""
    calls: list[dict[str, object]] = []

    def fake_fit(**kwargs):
        calls.append(kwargs)
        return kwargs["report_id"]

    monkeypatch.setattr(
        reporting_module,
        "fit_scoring_facets_calibration_report",
        fake_fit,
    )
    reports = fit_enterprise_issue_facets_calibration_reports(
        _connected_executions(),
        report_id_prefix="enterprise_calibration",
        q_theta=17,
        max_iter=91,
        tol=1e-5,
        additional_review_trigger_ids=("human_validation_required",),
    )

    assert reports == (
        "enterprise_calibration_claim_support",
        "enterprise_calibration_source_alignment",
    )
    assert [call["design"].criterion_id for call in calls] == [
        "claim_support",
        "source_alignment",
    ]
    assert all(call["q_theta"] == 17 for call in calls)
    assert all(call["max_iter"] == 91 for call in calls)
    assert all(call["tol"] == 1e-5 for call in calls)
    assert all(
        call["additional_review_trigger_ids"]
        == ("human_validation_required",)
        for call in calls
    )
    bundle_fingerprints = {
        call["metadata"]["enterprise_issue_bundle_fingerprint"] for call in calls
    }
    assert len(bundle_fingerprints) == 1
    assert all(
        call["metadata"]["enterprise_issue_design_fingerprint"]
        == call["design"].design_fingerprint
        for call in calls
    )
    assert all(
        call["metadata"]["enterprise_issue_criterion_separation"] is True
        for call in calls
    )


def test_invalid_prefix_fails_before_bundle_assembly(monkeypatch) -> None:
    """Invalid public identities fail before provenance replay or Rust fitting."""
    visited = False

    def unexpected_builder(_executions):
        nonlocal visited
        visited = True
        raise AssertionError("bundle assembly must not run")

    monkeypatch.setattr(
        reporting_module,
        "build_enterprise_issue_facets_calibration_bundle",
        unexpected_builder,
    )
    with pytest.raises(AssessmentSpecError) as captured:
        fit_enterprise_issue_facets_calibration_reports(
            (),
            report_id_prefix="invalid",
        )

    assert captured.value.code == "invalid_report_id_prefix"
    assert captured.value.path == "$.report_id_prefix"
    assert visited is False


def test_prefix_validation_redacts_callback_failures(monkeypatch) -> None:
    """Unexpected identifier callbacks cannot leak data or reach calibration."""
    visited = False

    def failing_identifier(*_args, **_kwargs):
        raise RuntimeError("private_prefix_payload")

    def unexpected_builder(_executions):
        nonlocal visited
        visited = True
        raise AssertionError("bundle assembly must not run")

    monkeypatch.setattr(contract_safety, "_identifier", failing_identifier)
    monkeypatch.setattr(
        reporting_module,
        "build_enterprise_issue_facets_calibration_bundle",
        unexpected_builder,
    )

    with pytest.raises(AssessmentSpecError) as captured:
        fit_enterprise_issue_facets_calibration_reports(
            (),
            report_id_prefix="enterprise_report",
        )

    assert captured.value.code == "invalid_report_id_prefix"
    assert captured.value.path == "$.report_id_prefix"
    assert "private_prefix_payload" not in str(captured.value)
    assert visited is False


def test_all_generated_report_ids_fail_before_any_criterion_fit(monkeypatch) -> None:
    """One invalid later report identity must prevent every Rust-backed fit."""
    fitted_report_ids: list[str] = []

    def validating_fit(**kwargs):
        report_id = contract_safety.descriptive_identifier(
            kwargs["report_id"],
            "report_id",
        )
        fitted_report_ids.append(report_id)
        return report_id

    monkeypatch.setattr(
        reporting_module,
        "fit_scoring_facets_calibration_report",
        validating_fit,
    )
    prefix = "enterprise_" + ("x" * 102)

    with pytest.raises(AssessmentSpecError) as captured:
        fit_enterprise_issue_facets_calibration_reports(
            _connected_executions(),
            report_id_prefix=prefix,
        )

    assert captured.value.code == "invalid_report_id"
    assert captured.value.path == "$.report_id"
    assert fitted_report_ids == []


def test_review_trigger_generator_is_materialized_once(monkeypatch) -> None:
    """One callback-safe trigger collection is replayed for every criterion."""
    visits: list[str] = []
    captured: list[tuple[str, ...]] = []

    def trigger_values():
        visits.append("visited")
        yield "external_review_required"

    def fake_fit(**kwargs):
        captured.append(kwargs["additional_review_trigger_ids"])
        return kwargs["report_id"]

    monkeypatch.setattr(
        reporting_module,
        "fit_scoring_facets_calibration_report",
        fake_fit,
    )
    fit_enterprise_issue_facets_calibration_reports(
        _connected_executions(),
        report_id_prefix="enterprise_report",
        additional_review_trigger_ids=trigger_values(),
    )

    assert visits == ["visited"]
    assert captured == [
        ("external_review_required",),
        ("external_review_required",),
    ]


def test_public_surface_is_explicit_and_documented() -> None:
    """The additive enterprise report fitter remains discoverable and bounded."""
    assert "fit_enterprise_issue_facets_calibration_reports" in enterprise.__all__
    assert inspect.getdoc(fit_enterprise_issue_facets_calibration_reports)
    assert reporting_module.__all__ == [
        "fit_enterprise_issue_facets_calibration_reports"
    ]
