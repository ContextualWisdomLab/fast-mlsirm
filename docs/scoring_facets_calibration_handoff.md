# Governed criterion-level many-facet calibration handoff

`fast_mlsirm.scoring.calibration` bridges governed scoring observations to the
existing Rust-backed many-facet Rasch estimator without creating a second
observation schema or implementing psychometric arithmetic in Python.

The handoff is shared infrastructure for domain adapters such as automated essay
scoring and enterprise issue intelligence. Domain modules emit the ordinary
`ScoringRequest`, `ScoringResult`, and `EngineDescriptor` contracts; the
calibration layer projects those exact artifacts into criterion-specific designs.

## Measurement mapping

Each criterion is calibrated separately with the existing rating-scale
many-facet model:

\[
\log\frac{P(Y_{p i r}=k)}{P(Y_{p i r}=k-1)}
=\theta_p-d_i-c_r-f_k.
\]

The governed axes are:

- person: one exact `response_id` (with `respondent_id` retained as provenance);
- item: one exact `task_id`, such as an essay prompt revision;
- rater: one full `EngineDescriptor.engine_fingerprint`;
- category: the ordered rubric score scale.

In scoring wire schema 1.0, `task_id` is the estimator item identity. Callers must
therefore issue a new descriptive task identifier when task content changes;
they must not reuse one logical identifier across materially different prompt or
task revisions. Issue #499 tracks an explicit provider-neutral
`task_revision_fingerprint` for the next schema revision. Until that contract is
merged, the automated-essay vertical is not release-ready for cross-revision
calibration, even though each request still retains the exact prompt fingerprint
for audit. Linking changed revisions requires governed anchors and invariance/DIF
evidence rather than silent pooling.

The full engine fingerprint is used as the rater identity. A changed model,
prompt template, provider, version, or engine metadata therefore becomes a new
rater rather than being silently pooled with an earlier implementation.

Arbitrary ordered rubric values, for example `(1, 3, 5)`, remain visible in the
audit artifact. Immediately before Rust estimation they are mapped
order-preservingly to `(0, 1, 2)`, which is the category convention accepted by
`fit_facets`. The mapping changes no order or spacing claim: the rating-scale
model operates on ordered categories and does not interpret the numeric labels
as interval distances.

Construction and fitting use separate identification gates. A sparse pilot
design may be assembled and audited after at least two categories are observed
among `scored` records. Construction does not authorize fitting:
`to_fit_facets_kwargs()`, `fit_scoring_facets_design()`, and bundle fitting
require every declared category to be observed among `scored` records before
delegating to the Rust estimator, because an unobserved category leaves a
rating-scale threshold unidentified. `abstained`, `failed`, `excluded`, and
unassigned cells do not count as category observations.

## Missingness and terminal states

`scored`, `abstained`, `failed`, and `excluded` observations remain distinct in
the sparse content-addressed record. Only `scored` observations become numeric
categories. Terminal observations and cells that were never assigned are `NaN`
in the estimator tensor, but can be distinguished through `response_states()`.
They are never coerced to the lowest score.

The current Rust estimator treats these numeric missing cells under its
documented missing-at-random assumption. The handoff does not claim that the
assumption is substantively adequate; missingness, abstention, and failure
mechanisms must be examined in validation and monitoring.

## Connectedness and resource safety

Before dense allocation, the assembler:

- requires observed support for every response, task, and rater;
- requires at least two responses, two raters, and two observed categories per
  criterion;
- rejects duplicate response-task-rater cells;
- bounds the complete response-by-task-by-rater cross-product;
- verifies one assessment, rubric, construct, occasion, and score scale; and
- checks the observed task-rater bipartite graph for connectedness.

Disconnected task-rater designs fail closed by default. They may be retained for
explicit diagnostics, but fitting them requires a second explicit opt-in. In a
disconnected design, comparisons across components depend on the common latent
population assumption rather than direct rating-design links and must not be
reported as interchangeable rater or task effects.

## Interpretation boundary

This first baseline estimates response proficiency, task difficulty, common
category thresholds, and rater severity *within each criterion*. It deliberately
does not average analytic criteria or assert a general writing-quality factor.
It also does not yet estimate criterion-specific rater discrimination, range
restriction, rater drift, subgroup DIF, correlated dimensions, bifactor
structure, testlet effects, or latent-space residual interactions.

A valid handoff proves structural provenance and estimator compatibility only.
It does not establish convergence, adequate fit, score reliability, human/AI
interchangeability, fairness, scoreability, construct validity, causal utility,
or readiness for consequential automation. Those claims require recovery,
connected-design evidence, residual diagnostics, held-out prediction, human
anchors, subgroup analysis, drift monitoring, and an approved decision policy.

## Example

```python
from fast_mlsirm.scoring import (
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
    fit_scoring_facets_bundle,
)

records = []
for request, result, engine in governed_executions:
    records.extend(
        build_scoring_facets_rating_records(
            request=request,
            result=result,
            engine=engine,
        )
    )

bundle = build_scoring_facets_calibration_bundle(records)
fits = fit_scoring_facets_bundle(bundle)
```

All likelihood, EM, quadrature, and parameter-update arithmetic in the example is
performed by the compiled Rust core through `fast_mlsirm.fit_facets`.

## References

Andrich, D. (1978). A rating formulation for ordered response categories.
*Psychometrika, 43*(4), 561–573. https://doi.org/10.1007/BF02293814

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item
parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459.
https://doi.org/10.1007/BF02293801

Eckes, T. (2015). *Introduction to many-facet Rasch measurement* (2nd ed.).
Peter Lang. https://doi.org/10.3726/978-3-653-04844-5

Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press.
