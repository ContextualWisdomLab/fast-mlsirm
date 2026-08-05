"""Bind enterprise issue calibration bundles to shared governed reports.

The adapter assembles exact enterprise issue scoring executions into the existing
criterion-specific calibration bundle and delegates every numerical fit to the
shared Rust-backed facets report boundary. It introduces no enterprise-specific
fit, report, estimator, validity, fairness, utility, or decision schema.
"""

from __future__ import annotations

from collections.abc import Iterable

from .._contract_safety import descriptive_identifier, sorted_identifiers
from ..calibration_reporting import (
    MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS,
    ScoringFacetsCalibrationReport,
    fit_scoring_facets_calibration_report,
)
from ..execution import EngineDescriptor, ScoringRequest, ScoringResult
from .calibration import build_enterprise_issue_facets_calibration_bundle
from .contracts import AtomicIssueRecord


def fit_enterprise_issue_facets_calibration_reports(
    executions: Iterable[
        tuple[AtomicIssueRecord, ScoringRequest, ScoringResult, EngineDescriptor]
    ],
    *,
    report_id_prefix: str,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    additional_review_trigger_ids: Iterable[str] = (),
) -> tuple[ScoringFacetsCalibrationReport, ...]:
    """Fit exact enterprise executions and return shared criterion reports.

    The report prefix is validated before any bundle assembly or Rust fitting.
    Exact enterprise provenance is replayed by
    :func:`build_enterprise_issue_facets_calibration_bundle`; each deterministic
    criterion design is then passed unchanged to
    :func:`~fast_mlsirm.scoring.calibration_reporting.fit_scoring_facets_calibration_report`.
    Report metadata binds the exact shared bundle and design fingerprints while
    retaining criterion separation.

    The returned objects are the existing shared calibration report aliases.
    Passing this boundary establishes provenance and estimator-output integrity
    only. It does not establish model adequacy, global optimality, reliability,
    fairness, construct validity, scorer interchangeability, causal effects, or
    authorization for consequential automation.
    """
    normalized_prefix = descriptive_identifier(
        report_id_prefix,
        "report_id_prefix",
    )
    review_trigger_ids = sorted_identifiers(
        additional_review_trigger_ids,
        "additional_review_trigger_ids",
        minimum=0,
        maximum=MAX_SCORING_FACETS_REPORT_REVIEW_TRIGGERS,
    )
    bundle = build_enterprise_issue_facets_calibration_bundle(executions)
    bundle_fingerprint = bundle.bundle_fingerprint
    return tuple(
        fit_scoring_facets_calibration_report(
            report_id=f"{normalized_prefix}_{design.criterion_id}",
            design=design,
            q_theta=q_theta,
            max_iter=max_iter,
            tol=tol,
            additional_review_trigger_ids=review_trigger_ids,
            metadata={
                "enterprise_issue_bundle_fingerprint": bundle_fingerprint,
                "enterprise_issue_design_fingerprint": design.design_fingerprint,
                "enterprise_issue_criterion_separation": True,
            },
        )
        for design in bundle.designs
    )


__all__ = ["fit_enterprise_issue_facets_calibration_reports"]
