# Governed essay many-facet calibration reports

`fast_mlsirm.scoring.essay` can bind one exact criterion-specific governed scoring design to the unchanged output of the Rust-backed many-facet Rasch estimator. The report is a source-text-free audit artifact. It does not rescore essays, average criteria, rank raters, infer validity, or authorize deployment.

## Governed boundary

Use `fit_essay_facets_calibration_report` when fitting and reporting in one operation. The helper captures `ScoringFacetsDesign.design_fingerprint` immediately before delegating to `fit_scoring_facets_design`, then copies the returned `FacetsFit` arrays into immutable tuples. `build_essay_facets_calibration_report` remains available when a fit was produced separately, but callers must provide the exact source design fingerprint because a bare `FacetsFit` does not embed provenance. Persisting or exchanging an unbound `FacetsFit` as audit evidence is therefore unsupported.

The report retains:

- assessment, rubric, construct, occasion, and criterion identities;
- respondent identifiers aligned to EAP trait estimates;
- exact task-revision, logical task, and task-family axes aligned to item difficulty estimates;
- exact engine, engine-family, and engine-fingerprint axes aligned to rater severity estimates;
- original ordered category values aligned to common thresholds;
- the complete finite log-likelihood trace, iteration count, convergence flag, connectedness flags, and parameter count;
- deterministic report fingerprints, public handles, policy metadata, and review-trigger identifiers.

No prompt text, response text, evidence text, or source content is persisted. The shared metadata safety boundary rejects sensitive content keys.

## Fail-closed replay checks

Report construction rejects:

- a source design fingerprint that differs from the supplied design;
- non-vector, empty, non-finite, or axis-misaligned estimator output;
- a threshold count that differs from the ordered category scale;
- a material decrease in the reported EM log-likelihood trace;
- an iteration count that differs from the trace length;
- a parameter count that differs from the Rust model contract;
- connectedness that differs between the source design and returned fit.

The standalone HTML renderer repeats the numeric, axis, identity, parameter-count, iteration, monotonic-trace, and connectedness checks before serialization. This guards against post-construction mutation or malformed deserialization. These are integrity and replay checks. A monotone trace or converged optimizer does not establish a global optimum, model fit, score reliability, construct validity, fairness, rater interchangeability, or appropriate operational use. Quadrature size, iteration limits, and numerical tolerances are estimator controls rather than evidence of solution uniqueness or global optimality.

## Human-review routing

The following structural triggers are mandatory and cannot be removed by callers:

- `calibration_not_converged` when the Rust estimator does not report convergence;
- `calibration_disconnected` when the exact rating design and fit are disconnected.

Callers may add organization-specific policy triggers. The absence of a trigger is not evidence that an automated essay score is valid or safe for consequential decisions.

## Standalone HTML audit artifact

`render_essay_facets_calibration_report_html` writes a deterministic standalone artifact containing exact report, design, assessment, rubric, construct, occasion, criterion, respondent, task-revision, rater-engine, category, estimate, convergence, connectedness, iteration, and review-trigger evidence. It intentionally excludes source text and does not produce fit, reliability, fairness, scoreability, global-optimum, or deployment decisions.

The artifact is designed for audit review rather than interactive decision automation:

- no JavaScript or external network resource is loaded;
- a restrictive meta-delivered Content Security Policy permits only inline styling and data images;
- caller-controlled titles and report values are HTML-escaped;
- semantic headings, definition lists, captions, column headers, a skip link, keyboard-focusable overflow regions, and exact visible values support accessible review;
- canonical deterministic JSON is embedded for reconstruction without hover interactions.

A meta-delivered policy is defense in depth, not a substitute for output encoding or safe hosting controls. When the artifact is served over HTTP, operators should also set an equivalent or stricter `Content-Security-Policy` response header.

## Interpretation boundary

The current Rust many-facet baseline estimates respondent proficiency, task-revision difficulty, common category thresholds, and rater severity within one criterion. Criteria remain separate. The report does not emit a holistic score, severity ordering, difficulty ordering, discrimination estimate, range-compression diagnosis, drift estimate, fairness conclusion, DIF result, or model preference.

The model uses marginal maximum likelihood with a fixed standard-normal trait distribution. It is not Linacre's joint maximum likelihood implementation. Numerical agreement with FACETS software is therefore not asserted: any comparison requires matched identification constraints, estimator, quadrature, convergence rules, missing-data handling, and category treatment. Connectedness is required for design-based comparisons of item difficulty and rater severity; a shared population distribution alone does not make disconnected components interchangeable.

## Example

```python
from fast_mlsirm.scoring.essay import (
    fit_essay_facets_calibration_report,
    render_essay_facets_calibration_report_html,
)

report = fit_essay_facets_calibration_report(
    report_id="claim_support_calibration_report",
    design=criterion_design,
    q_theta=41,
    max_iter=500,
    tol=1e-6,
    metadata={"workflow_stage": "human_anchored_pilot"},
)

payload = report.to_dict()
assert payload["criterion_id"] == criterion_design.criterion_id

render_essay_facets_calibration_report_html(
    report,
    "artifacts/claim_support_calibration_report.html",
)
```

## Equation-to-source traceability

The delegated Rust estimator uses the rating-scale many-facet adjacent-category formulation

\[
\log\frac{P(Y_{pij}=k)}{P(Y_{pij}=k-1)}
= \theta_p-d_i-c_j-f_k,
\]

where `theta` is respondent proficiency, `d` task difficulty, `c` rater severity, and `f` the common category threshold. The package estimates this model by marginal maximum likelihood with EM and Gauss-Hermite quadrature. The report copies those outputs and does not reimplement the equation.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Andrich, D. (1978). A rating formulation for ordered response categories. *Psychometrika, 43*(4), 561–573. https://doi.org/10.1007/BF02293814

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459. https://doi.org/10.1007/BF02293801

Eckes, T. (2015). *Introduction to many-facet Rasch measurement* (2nd ed.). Peter Lang. https://doi.org/10.3726/978-3-653-04844-5

Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press.

World Wide Web Consortium. (2024, December 12). *Web Content Accessibility Guidelines (WCAG) 2.2* (W3C Recommendation). https://www.w3.org/TR/WCAG22/

World Wide Web Consortium. (2026, May 5). *Content Security Policy Level 3* (W3C Working Draft). https://www.w3.org/TR/CSP3/
