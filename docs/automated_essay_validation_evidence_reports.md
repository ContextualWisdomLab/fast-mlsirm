# Governed automated-essay validation evidence reports

`fast_mlsirm.scoring.essay.validation_reporting` binds one exact assessment, construct, rubric, criterion, validation dataset, automated engine, and human-reference engine to descriptive agreement and difference evidence produced by the existing Rust core. The report is an audit artifact. It is not an automated validity verdict, fairness certification, model-selection result, or deployment authorization.

## Why the report has no overall pass field

Agreement coefficients, correlations, standardized mean differences, and human–machine degradation summaries answer different questions and depend on the score scale, population, sampling design, rubric, task distribution, rater design, and intended use. A fixed threshold cannot establish construct validity or responsible use across contexts. Pearson correlation is retained only as a descriptive association statistic; it cannot substitute for agreement, calibration, subgroup analysis, evidence about construct representation, or consequences of use.

The legacy `fast_mlsirm.validation.validate_judge` API retains historical conjunctive gate fields for backward compatibility. The governed report deliberately copies only Rust-computed metric values and discards every legacy threshold and Boolean pass field. Interpretation-boundary identifiers may discuss why universal thresholds are invalid, but metric records and the report root contain no threshold or pass decision fields.

## Exact governed binding

`build_essay_validation_evidence_report` requires:

- one factory-verified shared `AssessmentSpec`;
- a construct and exact rubric fingerprint bound by that assessment;
- one descriptive criterion identifier;
- one assessment-authorized automated `EngineDescriptor`;
- one human-reference `EngineDescriptor`;
- one exact validation-dataset fingerprint;
- paired ordinal labels and the declared category count;
- optional paired human–human labels and subgroup labels.

The assessment validation policy must explicitly declare every metric emitted by the selected evidence design. The report persists neither essay text nor label vectors. It retains the exact dataset fingerprint, observation count, engine descriptors, assessment policy graph, metric values, interpretation boundaries, review triggers, and deterministic report identity.

## Rust-first computation

All metric arithmetic delegates to `mlsirm_core::agreement::validate_scoring` through the existing PyO3 wrapper. Python validates provenance, invokes the kernel, maps stable metric identities, freezes metadata, and serializes the result. It does not reimplement weighted kappa, correlation, standardized mean differences, agreement rates, degradation, subgroup statistics, ranking, likelihood, optimization, or utility arithmetic.

The current evidence set includes:

- quadratic-weighted kappa;
- exact and adjacent agreement;
- Pearson correlation as descriptive association only;
- standardized mean difference;
- human–machine degradation relative to a supplied human–human baseline;
- worst subgroup standardized mean difference when subgroup labels are supplied.

## Mandatory human-review routing

Every report carries `human_validation_required` and `correlation_descriptive_only`. Omitting a human–human comparator adds `human_human_baseline_missing`; omitting subgroup labels adds `subgroup_evidence_missing`. Callers may add policy-specific review triggers but cannot suppress these structural signals.

Even a report with complete comparator and subgroup inputs does not establish:

- score reliability or interchangeability across raters;
- construct representation or construct-irrelevant variance;
- fairness, DIF absence, or equal consequences across groups;
- predictive validity or incremental value over simpler baselines;
- causal utility of an intervention;
- safety for consequential automation.

Those claims require an identified validation design, representative data, uncertainty analysis, external and temporal replication, human review, and evidence appropriate to the intended interpretation and use.

## Example

```python
from fast_mlsirm.scoring.essay import (
    build_essay_validation_evidence_report,
    render_essay_validation_evidence_report_html,
)

report = build_essay_validation_evidence_report(
    report_id="claim_support_validation_report",
    assessment=assessment_spec,
    construct_id="evidence_quality",
    rubric_fingerprint=criterion_rubric.fingerprint,
    criterion_id="claim_support",
    automated_engine=automated_engine_descriptor,
    reference_engine=human_reference_descriptor,
    validation_dataset_fingerprint=validation_dataset_fingerprint,
    automated_labels=automated_scores,
    reference_labels=human_scores,
    category_count=3,
    human_human_labels=(human_rater_a, human_rater_b),
    subgroup_labels=declared_subgroups,
    metadata={"study_stage": "human_anchored_holdout"},
)

payload = report.to_dict()
assert "pass" not in payload
assert payload["human_review_required"] is True

render_essay_validation_evidence_report_html(
    report,
    "artifacts/claim_support_validation_report.html",
)
```

## Standalone HTML audit artifact

`render_essay_validation_evidence_report_html` writes a deterministic standalone artifact containing exact report, assessment, construct, rubric, dataset, engine, metric, review-trigger, and interpretation-boundary provenance. It intentionally excludes source text, score-label vectors, thresholds, and Boolean pass decisions.

The artifact is designed for audit review rather than interactive decision automation:

- no JavaScript or external network resources are loaded;
- a restrictive meta-delivered Content Security Policy permits only inline styling and data images;
- caller-controlled titles and all report values are HTML-escaped;
- semantic headings, definition lists, captions, column headers, a skip link, keyboard-focusable overflow regions, and exact visible values support accessible review;
- canonical deterministic JSON is embedded for reconstruction without requiring hover interactions.

A meta-delivered policy is defense in depth, not a substitute for output encoding or safe hosting controls. When the artifact is served over HTTP, operators should also set an equivalent or stricter `Content-Security-Policy` response header.

## Equation-to-source traceability

The Rust kernel computes quadratic-weighted kappa as

\[
\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}},
\qquad
w_{ij}=\frac{(i-j)^2}{(K-1)^2},
\]

where `O` is the observed joint proportion table and `E` is the table expected from the marginal proportions. Exact and adjacent agreement are observed proportions. Pearson correlation is a descriptive linear association. Standardized mean difference uses the human-score standard deviation as the reference scale in the current Rust implementation.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Fleiss, J. L., & Cohen, J. (1973). The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability. *Educational and Psychological Measurement, 33*(3), 613–619. https://doi.org/10.1177/001316447303300309

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x

World Wide Web Consortium. (2024, December 12). *Web Content Accessibility Guidelines (WCAG) 2.2* (W3C Recommendation). https://www.w3.org/TR/WCAG22/

World Wide Web Consortium. (2026, July 29). *Content Security Policy Level 3* (W3C Working Draft). https://www.w3.org/TR/CSP3/
