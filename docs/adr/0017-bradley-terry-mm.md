# ADR-0017: Adopt Bradley–Terry MM and additive-ties BRATT for pairwise ranking

Status: **Accepted**
Date: 2026-08-16
Supersedes: none
Superseded by: none

## Context

Protected main already ships paired-comparison ranking kernels, including
Bradley–Terry MM (`bradley_terry_mm`) and an additive-ties variant
(`bratt_mm`). Implementation comments and changelog notes recorded that
Bradley and Terry (1952) and Hunter (2004) were unread at port time and that
`choix` / VGAM sources were the computational references. Those notes are
source-governance history. They are not a reason to leave the shipped
estimators without a primary-literature decision record, and they are not a
license to claim Rao–Kupper or Davidson ties models.

The implemented objects are psychometric ranking estimators. They are not
security controls.

## Decision drivers

- Pairwise win matrices are a reusable measurement primitive for rater,
  model, and item comparisons.
- The shipped MM update is Hunter's algorithm for the Bradley–Terry model;
  the decision record must say so.
- Ties are observed in some designs; the implemented ties likelihood is the
  additive-`alpha0` BRATT model, not Rao–Kupper or Davidson.
- Score-use interpretation, when worths are treated as educational or
  psychological scores, is governed by AERA, APA, and NCME (2014).

## Ownership and dependency direction

`fast-mlsirm` owns the reusable ranking kernels. Hosted leaderboards,
participant identity, and operational selection policy remain downstream.
This ADR does not change the ADR-0001 repository boundary.

## Decision

Adopt the Bradley–Terry paired-comparison model (Bradley & Terry, 1952)
fitted by Hunter (2004) MM as the repository's tie-free pairwise-ranking
estimator (`fast_mlsirm.bradley_terry_mm`).

Adopt the implemented additive-ties BRATT variant (`fast_mlsirm.bratt_mm`)
for data that contain ties:

```text
P(i beats j) = alpha_i / (alpha_i + alpha_j + alpha0)
P(i ties j)  = alpha0 / (alpha_i + alpha_j + alpha0)
```

Do not claim Rao–Kupper or Davidson unless a later model-design change
implements those likelihoods, gradients, tests, and documentation together.

LSR / I-LSR, Rank Centrality, and Plackett–Luce ranking remain separate
estimators. They are not aliases of Bradley–Terry MM.

Method documentation: [`../bradley_terry_mm.md`](../bradley_terry_mm.md).

## Invariants / acceptance evidence

1. `wins` is a square nonnegative matrix with a zero diagonal; non-finite or
   negative counts are rejected.
2. `bradley_terry_mm` rejects an all-zero matrix and, at `alpha = 0`, any
   object with zero wins (no finite log-worth).
3. `bratt_mm` rejects tie-free data and any contestant with zero wins.
4. Numeric work is Rust-owned; Python validates and marshals.
5. Documentation names Bradley and Terry (1952) and Hunter (2004) as the
   method basis and does not name Rao–Kupper or Davidson as implemented
   product behavior.

## Non-goals and claims not made

- Not Rao–Kupper (1967) or Davidson (1970) ties models.
- Not a Ford (1957) connectivity pre-check; disconnected graphs fail at
  estimation.
- Not an IRT, MLSIRM, or many-facet severity model.
- Not a causal ranking or high-stakes selection rule.
- `choix` and VGAM `bratt()` are computational comparison sources, not
  scientific oracles.

## Consequences and trade-offs

### Benefits

- Pairwise ranking has a named model and a named MM algorithm.
- The ties path is explicitly the additive-`alpha0` variant.
- Callers can choose `bradley_terry_mm` versus `bratt_mm` from the data
  contract (ties present or absent) rather than from package folklore.

### Costs / risks

- MM can fail to converge on sparse or one-sided graphs.
- Fractional counts are accepted as weights; they are a derived extension of
  integer pair lists and must not be described as a different model.
- Users may treat worths as IRT abilities unless the interpretation boundary
  stays adjacent to the API.

## Alternatives considered

### Claim Rao–Kupper or Davidson because the function mentions ties

Rejected. The implemented likelihood is additive `alpha0` in the denominator.
Those alternative ties models are not coded.

### Treat changelog "NOT READ" notes as the bibliographic record

Rejected. The estimators implement the named methods. Primary papers are the
method basis; unread-at-port-time notes are historical.

### Collapse all ranking kernels into one "Bradley–Terry" product claim

Rejected. LSR, Rank Centrality, and Plackett–Luce are different estimators.

## Failure, degraded, and recovery behavior

Invalid shapes, non-finite or negative counts, nonzero diagonals, empty
comparison graphs, zero-win objects (where the MLE is outside the positive
parameter space), and non-convergence return errors. `bratt_mm` redirects
tie-free data to `bradley_terry_mm` rather than returning `alpha0 = 0`.

## Security and privacy implications

Win/tie matrices can encode identifiable rater or candidate comparisons.
Purpose limitation follows ADR-0012. This ADR adds no new credential or
provider surface and is not a CWE/OWASP/NIST control.

## Compatibility, migration, and rollback

Public Python/Rust entry points are unchanged. This ADR records the
scientific identity of already-shipped kernels. Retiring or replacing the
likelihood requires a superseding model-design ADR.

## Verification and release evidence

- Rust unit oracles and Python contract tests for `bradley_terry_mm` and
  `bratt_mm` on protected main.
- Cross-algorithm agreement with I-LSR at `alpha = 0` is supplementary
  numerical evidence, not a substitute for the Bradley–Terry / Hunter
  citations.
- This ADR does not authorize a formula change.

## Research and standards basis

Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block
designs: I. The method of paired comparisons. *Biometrika, 39*(3/4), 324–345.
https://doi.org/10.2307/2334029

Hunter, D. R. (2004). MM algorithms for generalized Bradley–Terry models.
*The Annals of Statistics, 32*(1), 384–406.
https://doi.org/10.1214/aos/1079120141

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.

## Follow-ups

Rao–Kupper, Davidson, or other generalized paired-comparison likelihoods
remain Proposed unless implemented as a complete model path. LSR input-bound
doctoring remains a separate operational control
([`../doctoring/lsr_ranking_input_bounds.md`](../doctoring/lsr_ranking_input_bounds.md)).

## Reversal / supersession conditions

Supersede this ADR if the repository retires the MM kernels, changes the
Bradley–Terry or additive-ties likelihood, or implements a different ties
model as the default product behavior.
