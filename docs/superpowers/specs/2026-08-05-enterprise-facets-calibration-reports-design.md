# Enterprise facets calibration reports design

## Goal

Provide one governed enterprise entry point that validates exact issue scoring
executions, assembles the existing shared criterion-separated calibration bundle,
fits each design through the existing Rust-backed many-facet estimator, and
returns the existing shared calibration reports.

The design advances issues #544 and #404 without introducing an enterprise-specific
fit, report, estimator, rating, design, result, or decision schema.

## Public API

Add:

```python
fit_enterprise_issue_facets_calibration_reports(
    executions,
    *,
    report_id_prefix,
    q_theta=41,
    max_iter=500,
    tol=1e-6,
    require_connected=True,
    additional_review_trigger_ids=(),
    metadata=None,
) -> tuple[ScoringFacetsCalibrationReport, ...]
```

The function returns one canonical report per criterion in the deterministic
order owned by `ScoringFacetsCalibrationBundle.designs`.

## Reused boundaries

1. `build_enterprise_issue_facets_calibration_bundle()` owns execution bounds,
   tuple shape, issue/request/result/engine provenance replay, evidence replay,
   criterion separation, category support, duplicate-cell rejection, and
   connectedness policy.
2. `fit_scoring_facets_calibration_report()` owns exact design capture,
   Rust-backed fitting, fit replay, parameter dimensions, finite values,
   likelihood trace, connectedness, mandatory review triggers, and the existing
   report schema.
3. `ScoringFacetsCalibrationReport` remains the exact compatibility alias of the
   established report class. Existing report handles, fingerprints, error codes,
   and legacy imports remain unchanged.

No statistical arithmetic is implemented in the enterprise module.

## Provenance metadata

Every report receives package-managed metadata:

- `enterprise_calibration_bundle_fingerprint`
- `enterprise_calibration_design_fingerprint`
- `enterprise_calibration_criterion_id`

Caller metadata is deep-frozen by the existing scoring metadata boundary and may
not overwrite these keys. Raw source text, issue statements, prompts,
credentials, customer tokens, and provider responses remain prohibited.

Review-trigger identifiers are normalized once before fitting and forwarded
unchanged to each criterion. This prevents one-shot generators or caller order
from making criterion reports disagree.

## Determinism

Equivalent execution sets produce the same bundle, design order, managed
metadata, report identifiers, and report fingerprints. Report IDs are derived as
`<report_id_prefix>_<criterion_id>` after validating the prefix as a descriptive
two-or-more-token lower `snake_case` identifier.

## Error handling

- Invalid report prefixes use the established descriptive-identifier error.
- Reserved metadata fails before model fitting.
- Invalid executions and disconnected designs retain the existing enterprise and
  shared calibration errors.
- Estimator and report replay failures retain the existing shared report errors.
- Provider or source strings are never reflected in new errors.

## Scientific boundary

A completed report is evidence of contract and provenance consistency, not proof
of construct validity, reliability, fairness, rater interchangeability, model
adequacy, global optimality, predictive validity, intervention value, or causal
effect. Nonconvergence and disconnectedness remain human-review triggers.

## Testing

The focused suite covers:

- a realistic connected two-issue, two-task-revision, two-rater-family,
  two-criterion execution design;
- actual Rust-backed fitting with bounded quadrature and iterations;
- exact shared report identity and one report per criterion;
- bundle/design/criterion metadata and report-ID binding;
- tuning, review-trigger, and caller-metadata forwarding;
- execution-order invariance;
- invalid prefix and reserved metadata failures before fitting;
- absence of source and issue text from serialized reports;
- 100% statement and branch coverage and complete public docstrings.

## Standards and research

The design follows the existing many-facet estimator traceability and the
`Standards for Educational and Psychological Testing` distinction between a
technical score artifact and evidence supporting an intended interpretation or
use. ISO/IEC 42001:2023 and NIST AI 600-1 support retaining traceability, review,
and evaluation controls for AI-assisted scoring workflows; no conformity claim
is made.
