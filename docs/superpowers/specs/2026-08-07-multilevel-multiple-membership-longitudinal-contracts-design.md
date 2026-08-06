# Multilevel, Multiple-Membership, and Longitudinal Contract Design

## Problem

The package can separate respondent, item/task, and rater facets, but those
facets are not sufficient when observations are nested or cross-classified in
schools, teams, sites, tenants, organizations, or other contexts. Treating all
contextual variation as an individual effect creates an atomistic-fallacy risk.
The current contracts also do not distinguish stable latent traits, systematic
growth, time-varying states, rater drift, item/task revision drift, and residual
same-item dependence across occasions.

Issue #565 defines the full Rust-first estimator roadmap. This first slice is
limited to immutable, provider-neutral design contracts and intentional RED
recovery contracts. It introduces no psychometric likelihood or optimization in
Python.

## Decision

Add a standalone `fast_mlsirm.multilevel` namespace whose initial public
contracts describe:

- sparse one-hot, cross-classified, and weighted multiple-membership designs;
- exact membership-set and revision provenance;
- ordered and irregular temporal occasions;
- random-intercept/slope and stationary AR(1) state specifications;
- separately switchable lagged-response dependence;
- bounded simulation specifications and true-parameter recovery assertions.

The contracts are reusable by scoring, enterprise-issue, RAG, and essay
orchestration without importing those domain modules.

## Canonical public contracts

### `ContextMembership`

One weighted edge from a governed observation to one context.

```python
ContextMembership(
    observation_id: str,
    context_id: str,
    membership_weight: float,
    membership_revision_fingerprint: str,
)
```

`observation_id` and `context_id` are descriptive two-or-more-token lower
`snake_case` identifiers. A membership weight is a finite real number in
`(0, 1]`; Booleans are not real weights. A full lowercase SHA-256 digest binds
the edge to the exact assignment revision.

### `ContextMembershipDesign`

A factory-sealed collection of ordered membership edges.

For every observation \(o\), non-negative weights must satisfy

\[
\sum_{h \in \mathcal H(o)} w_{oh} = 1
\]

within one documented absolute tolerance. One-hot nesting is the special case
with one edge of weight one. Duplicate observation–context cells, conflicting
revision provenance, empty observation groups, non-finite values, and resource
limits fail before numerical allocation.

Canonical order is `(observation_id, context_id, revision_fingerprint)`, so
input permutation cannot change content identity. The serialized form contains
no raw source or response text.

### `TemporalOccasion`

One repeated-measurement occasion.

```python
TemporalOccasion(
    respondent_id: str,
    occasion_id: str,
    sequence_index: int,
    time_offset_milliseconds: int,
    occasion_revision_fingerprint: str,
)
```

Sequence indices and time offsets are exact signed integers, not Boolean values.
Within one respondent, `sequence_index`, `occasion_id`, and time ordering must
be unique. Time offsets must increase strictly with sequence order; irregular
spacing is allowed.

### `LongitudinalStateSpec`

```python
LongitudinalStateSpec(
    state_kind: LongitudinalStateKind,
    autoregressive_coefficient: float | None,
    include_lagged_response_dependence: bool,
)
```

Supported initial state kinds:

- `random_intercept_slope`
- `stationary_autoregressive`

The stationary AR(1) contract requires finite
`-1 < autoregressive_coefficient < 1`. The growth contract forbids an AR
coefficient. Lagged-response dependence is a separate Boolean and never implied
by the latent-state model.

### `LongitudinalDesign`

A factory-sealed, content-addressed collection of respondent occasions and one
state specification. It must preserve respondent-level sequence grouping,
strict temporal ordering, irregular intervals, exact revision fingerprints,
and bounded allocation metadata.

## Identification boundary

These contracts do not claim that every accepted design is statistically
identified for every future estimator. They enforce necessary structural
conditions only. Estimator-specific checks must additionally reject:

- contextual random effects perfectly confounded with fixed effects;
- disconnected respondent–context–item–rater–time graphs;
- a drift parameter with no repeated anchored item or rater evidence;
- lagged-response parameters without repeated same-item observations;
- random slopes without within-respondent time variation;
- unsupported combinations of context effects and state models.

Diagnostic opt-in must not become the production default.

## Numerical ownership

Python validates, canonicalizes, hashes, marshals sparse designs, and returns
immutable result records. Rust owns every future likelihood, quadrature,
integration, gradient, Hessian-vector product, optimizer, uncertainty estimate,
and recovery calculation. GPU support is added only after profiling identifies
a batch shape that justifies transfer cost; CPU execution must never be labeled
as GPU.

## Security and resource boundaries

- bounded iterable materialization;
- bounded observations, contexts, memberships, respondents, and occasions;
- stable non-reflective errors with machine codes and JSON paths;
- UTF-8 and lowercase SHA-256 validation;
- no numeric object identifiers;
- no raw response, source, prompt, or provider output in serialized contracts;
- exact rejection of NaN, infinities, negative zero where identity would drift,
  callback conversion failures, and cyclic/unbounded collections.

## Recovery evidence

Subsequent estimator PRs must report scale-aligned bias, MAE, RMSE, confidence
or credible interval coverage, convergence, and failure classifications for:

- person/system latent states;
- context effects;
- membership effects;
- random slopes;
- AR coefficients;
- rater and item/task drift;
- lagged-response dependence.

Correlation is supplementary association evidence, not parameter-recovery
proof.

## MSA integration

The public design artifacts must be serializable across service boundaries and
usable unchanged by:

- standalone `fast_mlsirm` analysis;
- `fast_mlsirm.scoring` calibration;
- enterprise issue measurement;
- reference-free RAG evaluation;
- automated essay evaluation;
- `contextual-orchestrator` evaluation workflows.

No LLM provider dependency belongs in this namespace. Any later LLM integration
test uses `NVIDIA_NIM_API_KEY` and never `COPILOT_GITHUB_TOKEN`.

## Interpretation boundary

Multilevel and temporal measurement separates sources of variation; it does not
make contextual effects causal. Accepted contracts are not evidence of
measurement invariance, fairness, transportability, regulated-use readiness, or
valid intervention claims.

## APA 7 references

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership multiple
classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124.

Embretson, S. E. (1991). A multidimensional latent trait model for measuring
learning and change. *Psychometrika, 56*, 495–515.

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model using Gibbs sampling. *Psychometrika, 66*, 271–288.

Jeon, M., & Rabe-Hesketh, S. (2025). An autoregressive growth model for
longitudinal item analysis. *Psychometrika*. Advance online publication.

Uto, M. (2023). A Bayesian many-facet Rasch model with Markov modeling for
rater severity drift. *Behavior Research Methods, 55*, 3910–3928.
