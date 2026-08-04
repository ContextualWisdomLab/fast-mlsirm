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

- person: one respondent or system run, identified by `respondent_id`;
- item: one exact `task_id`, such as an essay prompt revision;
- rater: one full `EngineDescriptor.engine_fingerprint`;
- category: the ordered rubric score scale.

Each respondent-task cell retains its exact `response_id` and
`response_content_fingerprint`. Multiple raters may score one cell only when they
consume the same governed response revision. Reusing a response identifier for
changed content, or binding two response artifacts to one respondent-task cell,
fails before tensor allocation. Response-revision changes therefore propagate
through rating, design, and bundle identities.

In scoring wire schema 1.0, `task_id` is the estimator item identity. Callers must
issue a new descriptive task identifier when task content changes. Issue #499
tracks an explicit provider-neutral `task_revision_fingerprint` for the next
schema revision. Changed revisions require governed anchors and invariance/DIF
evidence rather than silent pooling.

The full engine fingerprint is the rater identity. A changed model, prompt
template, provider, version, or engine metadata becomes a new rater rather than
being silently combined with an earlier implementation.

Arbitrary ordered rubric values, for example `(1, 3, 5)`, remain visible in the
audit artifact. Immediately before Rust estimation they are mapped
order-preservingly to `(0, 1, 2)`, the category convention accepted by
`fit_facets`. The mapping changes no order or interval-spacing claim.

Construction and fitting use separate category-identification gates. A sparse
pilot design may be assembled and audited after at least two categories are
observed among `scored` records. Construction does not authorize fitting:
`to_fit_facets_kwargs()`, `fit_scoring_facets_design()`, and bundle fitting
require every declared category to be observed among `scored` records before
delegating to Rust. `abstained`, `failed`, `excluded`, and unassigned cells do
not identify thresholds.

## Missingness and terminal states

`scored`, `abstained`, `failed`, and `excluded` observations remain distinct in
the sparse content-addressed record. Only `scored` observations become numeric
categories. Terminal observations and cells that were never assigned are `NaN`
in the estimator tensor, but remain distinguishable through `response_states()`.
They are never coerced to the lowest score.

The current Rust estimator treats numeric missing cells under its documented
missing-at-random assumption. The handoff does not claim that assumption is
substantively adequate; missingness, abstention, and failure mechanisms require
validation and monitoring.

## Identification and resource safety

Before dense allocation, the assembler:

- requires at least two respondents, two tasks, two raters, and two observed
  categories per criterion;
- requires scored support for every respondent, task, and rater level;
- rejects duplicate respondent-task-rater cells;
- binds each logical response to one respondent, task, and exact content
  revision across all criteria and raters;
- binds each respondent-task cell to one exact response ID and content digest;
- bounds the complete respondent-by-task-by-rater allocation;
- verifies one assessment, rubric, construct, occasion, and score scale;
- checks the scored respondent-task bipartite graph for connectedness; and
- checks the scored task-rater bipartite graph for connectedness.

Both graphs are required. Common raters do not identify task difficulty when
respondents are observed on disjoint tasks, and respondent overlap does not
identify rater severity when raters are confounded with tasks. A disconnected
design can be retained only as an audit artifact with `require_connected=False`;
it cannot enter `fit_facets`, including through the legacy
`allow_disconnected` argument.

## Replay and provenance boundary

Projection revalidates package-owned child types and the complete result scope
before emitting any rating record. It rejects:

- an untyped observation child;
- duplicate observation identifiers;
- duplicate criterion observations;
- a result-declared criterion scope different from its request;
- missing, extra, or undeclared criterion coverage; and
- mismatched request, assessment, rubric, construct, granularity, or engine
  provenance on any observation.

Validation errors use stable caller-independent codes and paths and do not echo
rejected values.

## Interpretation boundary

This first baseline estimates respondent or system-run proficiency, task
difficulty, common category thresholds, and rater severity *within each
criterion*. It deliberately does not average analytic criteria or assert a
general writing-quality factor. It also does not estimate a response-level
random effect, criterion-specific rater discrimination, range restriction,
rater drift, subgroup DIF, correlated dimensions, bifactor structure, testlet
effects, or latent-space residual interactions.

A valid handoff proves structural provenance and estimator compatibility only.
It does not establish convergence, adequate fit, reliability, human/AI
interchangeability, fairness, scoreability, construct validity, causal utility,
or readiness for consequential automation. Those claims require recovery,
residual diagnostics, held-out prediction, human anchors, subgroup analysis,
drift monitoring, and an approved decision policy.

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

All likelihood, EM, quadrature, and parameter-update arithmetic is performed by
the compiled Rust core through `fast_mlsirm.fit_facets`.

## References

Andrich, D. (1978). A rating formulation for ordered response categories.
*Psychometrika, 43*(4), 561–573. https://doi.org/10.1007/BF02293814

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item
parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459.
https://doi.org/10.1007/BF02293801

Eckes, T. (2015). *Introduction to many-facet Rasch measurement* (2nd ed.).
Peter Lang. https://doi.org/10.3726/978-3-653-04844-5

Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press.
