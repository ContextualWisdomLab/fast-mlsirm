# Governed enterprise issue many-facet handoff

`build_enterprise_issue_facets_rating_records()` advances issue #404 by replaying
one exact enterprise issue scoring execution before it enters the existing shared
many-facet calibration contracts. The adapter returns only
`ScoringFacetsRatingRecord` values and introduces no enterprise-specific rating,
design, fit, report, or decision schema.

## Workflow position

The handoff sits after governed semantic extraction and criterion scoring:

1. `extract_enterprise_atomic_issues()` returns replay-verified
   `AtomicIssueRecord` values without retaining raw source text.
2. `build_enterprise_issue_scoring_request()` compiles an exact issue revision
   into the shared criterion-level `ScoringRequest` contract.
3. `build_enterprise_issue_score_observation()` preserves request-bound
   supporting, counter, and contextual evidence in shared `ScoreObservation`
   values.
4. `build_scoring_result()` closes one exact request/engine execution.
5. `build_enterprise_issue_facets_rating_records()` replays the enterprise
   provenance and delegates projection to
   `build_scoring_facets_rating_records()`.
6. `build_scoring_facets_calibration_bundle()` assembles separate
   criterion-specific respondent-by-task-revision-by-rater designs.
7. `fit_scoring_facets_bundle()` delegates all likelihood, quadrature, gradient,
   parameter-update, and optimization arithmetic to the existing Rust-backed
   estimator.

The atomic issue identifier remains the respondent axis and the exact issue
content fingerprint remains the response revision. Task revisions and exact
engine fingerprints remain the item and rater axes used by the shared calibration
boundary. Analytic criteria are not averaged into an undeclared holistic score.

## Fail-closed replay gates

Before shared rating records are built, the adapter requires exact package-owned
`AtomicIssueRecord`, `ScoringRequest`, `ScoringResult`, and `EngineDescriptor`
values. It then verifies:

- the request carries valid enterprise issue provenance;
- the request atomic-issue fingerprint equals the complete supplied issue record;
- the request issue-content fingerprint and response revision equal the supplied
  issue revision;
- the request respondent identifier equals the supplied issue identifier;
- every observation evidence reference was declared by the exact enterprise
  request;
- every non-abstained observation retains supporting evidence;
- declared counterevidence remains represented in every non-abstained
  observation; and
- package-managed observation fingerprints and evidence-role counts replay from
  the exact observation payload.

A generic shared observation with omitted evidence or copied-looking metadata
cannot cross this boundary unless all enterprise replay invariants are satisfied.
Abstention remains a terminal missing rating and is never converted into a low
score.

After these checks, the existing shared calibration builder remains authoritative
for request/result/engine matching, criterion coverage, response and task-revision
identity, duplicate-cell rejection, score-scale support, bounded dense allocation,
and respondent-task and task-rater connectedness.

## Numerical and scientific boundary

Python validates, canonicalizes, marshals, and delegates. It does not implement or
duplicate likelihood, gradient, Hessian, quadrature, optimization, scoring,
ranking, aggregation, utility, fairness, or causal arithmetic.

Passing provenance replay does not establish that an issue is true, complete,
material, probable, construct-valid, fair, or suitable for intervention. A
successful calibration handoff does not establish model adequacy, scoreability,
reliability, global optimality, rater interchangeability, predictive validity, or
high-stakes readiness. Connectedness is an identification gate, not evidence of
validity. A candidate intervention remains a hypothesis until an identified
design and human validation support a causal claim.

The delegated estimator equation and primary-source traceability are documented
in `automated_essay_facets_calibration_reports.md`; this adapter does not alter the
model or reimplement that equation.

## Example

```python
from fast_mlsirm.scoring import (
    build_scoring_facets_calibration_bundle,
    fit_scoring_facets_bundle,
)
from fast_mlsirm.scoring.enterprise_issue import (
    build_enterprise_issue_facets_rating_records,
)

records = []
for issue, request, result, engine in governed_executions:
    records.extend(
        build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )
    )

bundle = build_scoring_facets_calibration_bundle(records)
fits_by_criterion = fit_scoring_facets_bundle(bundle)
```

A production workflow must retain exact package versions, Rust/PyO3 backend
identity, estimator settings, assessment and rubric revisions, human-review
decisions, and held-out validation evidence alongside any persisted fit output.

## References

American Educational Research Association, American Psychological Association, &
National Council on Measurement in Education. (2014). *Standards for educational
and psychological testing*. American Educational Research Association.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item
parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459.
https://doi.org/10.1007/BF02293801

Eckes, T. (2015). *Introduction to many-facet Rasch measurement* (2nd ed.). Peter
Lang. https://doi.org/10.3726/978-3-653-04844-5

Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press.
