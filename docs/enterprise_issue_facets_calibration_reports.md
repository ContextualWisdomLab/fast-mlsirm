# Governed enterprise issue facets calibration reports

`fit_enterprise_issue_facets_calibration_reports()` advances issue #404 by
assembling exact enterprise issue scoring executions into the existing shared
criterion-specific calibration bundle and immediately fitting each design through
the canonical Rust-backed scoring-facets report boundary.

## Workflow position

```text
enterprise source packet
  -> atomic issue and evidence contracts
  -> governed criterion observations
  -> enterprise provenance replay
  -> shared ScoringFacetsCalibrationBundle
  -> Rust-backed criterion-specific fits
  -> shared ScoringFacetsCalibrationReport values
```

The helper returns the established shared report objects. It does not create an
enterprise-specific fit, report, estimator, serialization, ranking, utility, or
decision schema. Report identifiers are derived deterministically from one
validated descriptive prefix and each criterion identifier.

Each report metadata payload binds:

- the exact shared calibration-bundle fingerprint;
- the exact criterion-design fingerprint; and
- an explicit marker that analytic criteria remain separate.

Additional review triggers are materialized once through the bounded shared
identifier validator and applied consistently to every criterion report.

## Numerical and scientific boundary

Python validates identifiers, replays enterprise provenance, marshals bounded
collections, and delegates. All likelihood, quadrature, gradient, parameter
update, optimization, and latent-trait arithmetic remains in the existing Rust
`fit_facets` path.

A returned report establishes only that the supplied governed executions, shared
design, and Rust fit output passed the declared integrity checks. It does not
establish model adequacy, global optimality, reliability, fairness, construct
validity, rater interchangeability, predictive validity, causal effects, or
permission for consequential automation. Criteria are not averaged, and no
enterprise action priority is inferred.

## Example

```python
from fast_mlsirm.scoring.enterprise_issue import (
    fit_enterprise_issue_facets_calibration_reports,
)

reports = fit_enterprise_issue_facets_calibration_reports(
    governed_executions,
    report_id_prefix="enterprise_calibration",
    additional_review_trigger_ids=("human_validation_required",),
)

for report in reports:
    persist(report.to_dict())
```

Persist the exact package version, Rust/PyO3 backend identity, estimator settings,
assessment and rubric revisions, validation dataset provenance, and human-review
decisions beside every report artifact.
