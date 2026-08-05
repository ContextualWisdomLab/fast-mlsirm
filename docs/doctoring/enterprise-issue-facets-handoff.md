# Enterprise issue many-facet handoff doctoring record

## Purpose

This record documents the issue #404 boundary that replays one exact governed
enterprise issue scoring execution before converting its criterion observations
into the existing shared `ScoringFacetsRatingRecord` contract. The adapter adds no
enterprise-specific rating, design, estimator, fit, report, or decision schema.
It emits neither a calibrated parameter estimate nor a model-selection decision.

Python verifies identities, evidence provenance, terminal states, and package-
managed observation metadata, then delegates record construction to
`build_scoring_facets_rating_records()`. All likelihood, quadrature, gradient,
parameter-update, and optimization arithmetic remains in the existing Rust-backed
many-facet implementation.

## Equation-to-source traceability

The handoff does not add or modify a statistical equation. The delegated
many-facet estimator, identification constraints, exact rating-scale semantics,
and numerical implementation traceability remain authoritative in
`automated_essay_facets_calibration_reports.md` and its doctoring record.

Bock and Aitkin (1981) provide the primary marginal maximum-likelihood and EM
foundation used by the delegated estimator family. Linacre (1989) is the original
many-facet Rasch monograph cited for the respondent-task-rater formulation, while
Eckes (2015) provides a later comprehensive treatment. The *Standards for
Educational and Psychological Testing* (American Educational Research
Association et al., 2014) govern the conservative interpretation boundary: a
connected calibration design and a successfully fitted model do not themselves
establish validity, reliability, fairness, scoreability, predictive utility, or
readiness for consequential use.

## Replay invariants

Before delegation, the adapter requires exact package-owned
`AtomicIssueRecord`, `ScoringRequest`, `ScoringResult`, and `EngineDescriptor`
values and verifies:

- complete atomic-issue and issue-content fingerprints;
- respondent identity and exact response revision;
- request-bound evidence membership;
- supporting evidence for every non-abstained observation;
- explicit counterevidence representation whenever the issue declares
  counterevidence;
- exact package-managed observation evidence fingerprints and supporting,
  counter, and context counts; and
- preservation of abstention as missingness rather than a low rating.

The shared rating builder remains authoritative for request/result/engine replay,
criterion coverage, score-category support, exact respondent, task-revision and
rater axes, duplicate-cell rejection, bounded dense allocation, and design
connectedness.

## Scientific and product limits

Passing this boundary proves only that the supplied records are mutually
consistent under the declared package contracts. It does not establish that an
issue is true, complete, material, probable, construct-valid, fair, or suitable
for intervention. It does not establish model adequacy, global optimality, rater
interchangeability, invariance, predictive validity, or causal effect.

Analytic criteria remain separate and are not averaged into an undeclared
holistic score. Connectedness is an identification gate rather than evidence of
validity. Any consequential deployment still requires human-anchored calibration,
held-out validation across relevant customers, periods, tasks, and judge
families, subgroup and fairness analysis, documented review and appeal paths, and
an identified causal design before intervention-effect claims.

## Verification contract

Merge requires the same unchanged pull-request commit to pass every gate below:

- exact issue/request/result/engine replay tests;
- observation evidence, counterevidence, abstention, and managed-metadata tests;
- shared-builder delegation and deterministic order-invariance tests;
- connected-design and separate-criterion integration tests;
- adversarial type, identity, provenance, privacy, and rollback-safe failure
  tests;
- complete public docstrings and 100% statement and branch coverage for added
  production code; and
- Python, Rust/PyO3, package, release-acceptance, GPU-no-skip, fuzz, Security Scan,
  SAST, changelog parity, and final review gates.

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
