# Multilevel, Multiple-Membership, and Longitudinal Contract Design

## Problem

Respondent, item/task, and rater facets are insufficient when observations are nested or cross-classified in schools, teams, sites, tenants, organizations, or other contexts. Collapsing contextual variation into an individual effect creates an atomistic-fallacy risk. The package also needs to distinguish stable traits, systematic growth, time-varying states, rater drift, item/task revision drift, and residual same-item dependence across occasions.

Issue #565 defines the Rust-first estimator roadmap. This first slice is limited to immutable, provider-neutral design contracts. It introduces no psychometric likelihood, integration, optimization, uncertainty, or recovery arithmetic in Python.

## Decision

Add a standalone `fast_mlsirm.multilevel` namespace whose initial public contracts describe:

- explicit context classifications and levels;
- one-hot nesting, weighted multiple membership, cross-classification, and multiple-membership multiple-classification;
- exact assignment and revision provenance;
- ordered irregular temporal occasions;
- random-intercept/slope and discrete occasion-step stationary AR(1) state specifications;
- separately switchable lagged-response dependence; and
- bounded, replay-verified, source-text-free artifacts.

The same contracts are reusable by standalone, scoring, enterprise-issue, RAG, essay, and contextual-orchestrator workflows without importing those domain modules.

## Canonical public contracts

### `ContextMembership`

One weighted edge from a governed observation to one level of one contextual classification.

```python
ContextMembership(
    observation_id: str,
    context_dimension_id: str,
    context_id: str,
    membership_weight: float,
    membership_revision_fingerprint: str,
)
```

`context_dimension_id` is required and identifies the random-effect family, such as `school_context`, `site_context`, `tenant_context`, or `team_context`. `context_id` identifies one level within that family. Schema 1.0 never infers a dimension from the context label and never creates a default dimension for the caller.

Identifiers are descriptive two-or-more-token lower `snake_case`. A membership weight is a finite real number in `(0, 1]`; Booleans are not weights. A full lowercase SHA-256 digest binds the exact observation, dimension, level, and weight revision.

### `ContextMembershipDesign`

A factory-sealed collection of ordered membership edges.

For every observation \(o\), context dimension \(d\), and levels \(h\in\mathcal H_d(o)\):

\[
\sum_{h\in\mathcal H_d(o)} w_{odh}=1.
\]

Weights normalize independently per observation × dimension. One-hot nesting is one weight-one edge in one dimension. Weighted multiple membership uses multiple edges in one dimension. Cross-classification uses multiple dimensions, each with its own complete normalized assignment.

Schema 1.0 requires every observation to carry every dimension declared anywhere in the design. Absence is not interpreted as a zero random effect. Duplicate cells are scoped by `(observation_id, context_dimension_id, context_id)`. Reusing a revision digest for another observation, dimension, level, or weight fails closed.

Canonical order is `(observation_id, context_dimension_id, context_id, revision_fingerprint)`. The design exposes deterministic observation IDs, dimension IDs, dimension-scoped context keys, flattened compatibility views, and exact per-observation/per-dimension counts and weights. It serializes no raw source or response text.

### `TemporalOccasion`

```python
TemporalOccasion(
    respondent_id: str,
    occasion_id: str,
    sequence_index: int,
    time_offset_milliseconds: int,
    occasion_revision_fingerprint: str,
)
```

Sequence indices and time offsets are exact signed integers, not Boolean values. Within one respondent, occasion IDs, sequence indices, and time offsets are unique. Time offsets increase strictly with sequence order; irregular spacing is preserved.

### `LongitudinalStateSpec`

```python
LongitudinalStateSpec(
    state_kind: LongitudinalStateKind,
    autoregressive_coefficient: float | None,
    include_lagged_response_dependence: bool,
)
```

Supported state kinds:

- `random_intercept_slope`
- `stationary_autoregressive`

The stationary AR(1) contract requires finite `-1 < autoregressive_coefficient < 1`. The coefficient is a **discrete occasion-step** parameter. Millisecond offsets are provenance only and do not automatically transform \(\phi\) for elapsed time. A later continuous-time or interval-adjusted model requires a separate parameterization, units, identification, and recovery contract in Rust.

The growth contract forbids an AR coefficient. Lagged-response dependence is an independent Boolean and is never implied by the latent-state model.

### `LongitudinalDesign`

A factory-sealed, content-addressed collection of replay-verified respondent occasions and one replay-verified state specification. It preserves respondent-level sequence grouping, strict temporal ordering, irregular intervals, exact revision fingerprints, and bounded allocation metadata.

## Integrity and callback boundary

Aggregate factories replay every package-owned child through the canonical factory and compare it with the sealed fingerprint before sorting, grouping, hashing, or serialization. Post-construction mutation through `object.__setattr__` therefore fails with stable non-reflective integrity codes.

Collection materialization and enum/value normalization convert ordinary caller-controlled exceptions into package-owned errors without echoing private exception messages. Process-control exceptions are not swallowed.

## Identification boundary

These contracts enforce necessary structural conditions but do not prove that every accepted design is statistically identified. Future Rust fitters must additionally reject:

- contextual random effects confounded with fixed effects;
- context dimensions with insufficient independently linked levels or observations;
- disconnected respondent–context–item–rater–time graphs;
- unanchored drift;
- lagged-response parameters without repeated same-item evidence;
- random slopes without within-respondent time variation; and
- unsupported combinations of context effects and state models.

Diagnostic opt-in cannot become the production default.

## Numerical ownership

Python validates, canonicalizes, hashes, marshals sparse designs, and serializes immutable artifacts. Rust owns every future likelihood, quadrature, random-effect integration, gradient, Hessian-vector product, optimizer, uncertainty estimate, CPU multithreading, justified GPU batching, and true-parameter recovery calculation. CPU execution must never be labeled as GPU.

## Security and resource boundaries

- bounded iterable materialization;
- bounded observations, dimensions, contexts, memberships, respondents, and occasions;
- stable non-reflective errors with machine codes and JSON paths;
- UTF-8 and lowercase SHA-256 validation;
- no numeric object identifiers;
- no raw response, source, prompt, or provider output;
- exact rejection of NaN, infinity, Boolean numeric coercion, revision rebinding, duplicate cells, and hostile callbacks; and
- deterministic canonical serialization and 128-bit public handles backed by full SHA-256 identity.

## Recovery evidence

Subsequent estimator PRs must report scale-aligned bias, MAE, RMSE, confidence or credible interval coverage, convergence, and failure classifications for person/system latent states, dimension-specific context effects, membership effects, random slopes, discrete-step and future continuous-time dynamics, rater/item drift, and lagged-response dependence. Correlation is supplementary only.

Recovery studies must include nested, crossed, weighted multiple-membership, multiple-membership multiple-classification, balanced/unbalanced occasions, missing observations, irregular gaps, zero and near-boundary variance components, and confounded designs expected to fail.

## MSA integration

The artifacts are serializable across service boundaries and usable unchanged by standalone analysis, `fast_mlsirm.scoring`, enterprise issue measurement, reference-free RAG evaluation, automated essay evaluation, and contextual-orchestrator workflows. No LLM provider dependency belongs in this namespace. Any future LLM integration test uses `NVIDIA_NIM_API_KEY`, never `COPILOT_GITHUB_TOKEN`.

## Interpretation boundary

Multilevel and temporal measurement separates sources of variation; it does not make contextual effects causal. Accepted contracts are not evidence of measurement invariance, fairness, transportability, continuous-time dynamics, regulated-use readiness, or valid intervention claims. Membership weighting is a substantive assumption and requires sensitivity analysis.

## APA 7 references

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership multiple classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124. https://doi.org/10.1177/1471082X0100100202

Embretson, S. E. (1991). A multidimensional latent trait model for measuring learning and change. *Psychometrika, 56*, 495–515. https://doi.org/10.1007/BF02294487

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model using Gibbs sampling. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839

Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for longitudinal item analysis. *Psychometrika, 81*(3), 830–850. https://doi.org/10.1007/s11336-015-9489-2

Tranmer, M., Steel, D., & Browne, W. J. (2014). Multiple-membership multiple-classification models for social network and group dependencies. *Journal of the Royal Statistical Society: Series A (Statistics in Society), 177*(2), 439–455. https://doi.org/10.1111/rssa.12021

Uto, M. (2023). A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods, 55*, 3910–3928. https://doi.org/10.3758/s13428-022-01997-z
