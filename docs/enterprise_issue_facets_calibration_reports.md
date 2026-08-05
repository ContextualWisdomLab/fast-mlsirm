# Governed enterprise issue facets calibration reports

`fit_enterprise_issue_facets_calibration_reports()` completes the governed
enterprise path from exact issue scoring executions to criterion-specific shared
many-facet reports.

## Workflow

```text
AtomicIssueRecord + ScoringRequest + ScoringResult + EngineDescriptor
    -> enterprise provenance replay
    -> ScoringFacetsCalibrationBundle
    -> validate every derived report identifier
    -> one Rust-backed fit per criterion
    -> ScoringFacetsCalibrationReport tuple
```

The function first calls
`build_enterprise_issue_facets_calibration_bundle()`. That boundary remains
authoritative for bounded execution consumption, tuple shape, issue and evidence
provenance, criterion separation, task-revision and rater identity, category
support, duplicate cells, and connectedness policy.

Each resulting design is passed once to the domain-neutral
`fit_scoring_facets_calibration_report()`. That shared boundary captures the exact
design fingerprint immediately before fitting, delegates the numerical work to
the existing Rust many-facet estimator, validates the returned fit, and emits the
existing canonical report type.

## Report identity and metadata

Callers provide a descriptive lower `snake_case` report prefix. One report ID is
created per criterion as `<prefix>_<criterion_id>`. The complete set of derived
report IDs is validated before any estimator call, so an invalid or overlong
prefix-and-criterion combination fails without returning or fitting a partial
batch. The output order follows the canonical criterion order in the shared
bundle and is independent of execution arrival order.

Every report contains package-managed metadata:

- the exact enterprise calibration bundle fingerprint;
- the exact criterion design fingerprint; and
- the criterion identifier.

Caller metadata is deep-frozen by the existing scoring contract and cannot
replace these fields. The workflow retains no raw source text, issue statement,
prompt, provider response, customer token, or credential.

Review-trigger IDs are normalized once and forwarded unchanged to every criterion
report. Shared mandatory triggers for nonconvergence or disconnectedness cannot be
suppressed.

## Compatibility

`ScoringFacetsCalibrationReport` is an exact domain-neutral alias of the
established report implementation. Legacy report handles, fingerprints,
structured errors, serialization, and essay imports remain unchanged. This
workflow does not define an enterprise-specific report schema.

## Interpretation limits

A generated report proves only that the supplied governed records, design, fit,
and provenance are internally consistent under the package contracts. It does
not establish:

- issue truth, completeness, materiality, or probability;
- construct validity or score reliability;
- fairness or rater interchangeability;
- model adequacy or global optimality;
- predictive validity;
- intervention value or urgency; or
- causal effects or high-stakes deployment readiness.

Production use must retain package and Rust/PyO3 versions, estimator settings,
assessment and rubric revisions, source and execution evidence, human-review
decisions, recovery studies, held-out validation, and policy approvals.

## Example

```python
from fast_mlsirm.scoring.enterprise_issue import (
    fit_enterprise_issue_facets_calibration_reports,
)

reports = fit_enterprise_issue_facets_calibration_reports(
    governed_executions,
    report_id_prefix="enterprise_calibration",
    q_theta=41,
    max_iter=500,
    tol=1e-6,
    additional_review_trigger_ids=("deployment_review",),
    metadata={"workflow_stage": "human_anchored_pilot"},
)
```

The output is a tuple of existing shared report objects, one for each analytic
criterion. Consumers may use the existing JSON, accessible HTML, CSV, and exact
value surfaces without an enterprise-specific renderer.
