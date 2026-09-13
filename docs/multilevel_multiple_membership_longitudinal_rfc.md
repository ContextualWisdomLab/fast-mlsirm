# Multilevel, Multiple-Membership, and Longitudinal Measurement RFC

## Status

This RFC defines the provider-neutral contract boundary delivered by issue #565
and the first Rust-owned repeated-measurement state layer. The state layer is a
small, identified handoff for independent per-respondent OLS trends and
discrete-step AR(1) state predictions; it is not a claim that the full
multilevel IRT likelihood, Bayesian random-effect estimation, uncertainty, or
GPU state recurrence is complete.

## Product outcome

A buyer must be able to represent one person or AI system repeatedly observed
across tasks and raters while belonging to one or more changing organizational
contexts. Contextual variation must not be collapsed into an atomistic
individual score, and time-varying measurement must not silently pool item,
rater, membership, or occasion revisions.

The first public namespace is:

```python
from fast_mlsirm.multilevel import (
    LongitudinalStateKind,
    build_context_membership,
    build_context_membership_design,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
)
```

## Cross-classified multiple-membership contract

Every contextual edge names two identities explicitly:

- `context_dimension_id`: the random-effect family or classification, such as
  `school_context`, `site_context`, `tenant_context`, or `team_context`;
- `context_id`: the level within that dimension.

The dimension is required. Schema 1.0 never infers a variance-component family
from the text of `context_id` and never invents a default classification for the
caller. The same context label in two dimensions therefore remains two distinct
random-effect levels.

For observation \(o\), context dimension \(d\), and levels
\(h \in \mathcal H_d(o)\), weights are finite, strictly positive, and
normalize independently within each observation-by-dimension group:

\[
\sum_{h \in \mathcal H_d(o)} w_{odh}=1
\qquad \text{for every observed } (o,d).
\]

One-hot nesting is one weight-one edge within a dimension. Weighted multiple
membership uses two or more edges within a dimension. Cross-classification uses
more than one dimension, each with its own normalized assignment. Schema 1.0
requires every observation to carry every context dimension declared anywhere
in the design; absence is not silently interpreted as a zero random effect.

Membership edges retain:

- observation identity;
- context-dimension identity;
- context-level identity;
- exact membership weight;
- assignment-revision SHA-256;
- deterministic membership and design fingerprints.

A revision fingerprint is bound to the exact observation, dimension, context,
and weight. Reusing it for another assignment fails closed. Duplicate cells are
scoped by `(observation_id, context_dimension_id, context_id)`.

The design exposes deterministic dimension IDs, dimension-scoped context keys,
and per-observation/per-dimension counts and exact weights. These are audit and
future Rust-marshalling contracts, not variance-component estimates.

## Temporal contract

Each occasion retains respondent, occasion, sequence, exact integer time offset,
and revision identity. Within a respondent:

- occasion IDs are unique;
- sequence indices are unique;
- time offsets are unique;
- time offsets increase strictly with sequence order;
- irregular spacing is preserved as provenance.

The initial state-specification wire labels are:

- `random_intercept_slope`;
- `stationary_autoregressive` with \(-1<\phi<1\).

These labels are compatibility identifiers, not claims about the fitted
estimand. A `random_intercept_slope` result reports
`estimand_scope="independent_respondent_ols_trend"` and
`population_random_effects_estimated=False`; there is no population variance
component or shrinkage in this state predictor. A `stationary_autoregressive`
result reports `estimand_scope="discrete_ar_state_prediction"`,
`ar_coefficient_estimated=False`, and `ar_coefficient_source="caller_supplied"`.
The current `autoregressive_coefficient` is therefore a **discrete occasion-step
AR(1) parameter supplied by the caller**, not a coefficient estimated by this
state layer. The millisecond offsets do not transform \(\phi\), and the contract
does not claim continuous-time or interval-adjusted transitions. A later Rust
estimator may use irregular gaps only after a separate, explicit continuous-time
or elapsed-gap parameterization and recovery contract is introduced.

Residual lagged-response dependence is a separate Boolean contract. Enabling an
AR latent state does not implicitly enable same-item lag dependence, and vice
versa.

## MSA boundary

The same serialized artifacts are intended for:

- standalone psychometric analysis;
- human/AI scoring calibration;
- enterprise issue measurement;
- reference-free RAG evaluation;
- automated essay evaluation;
- contextual-orchestrator evaluation workflows.

The namespace has no provider SDK, network call, model credential, or raw text.
An LLM integration belongs in an adapter and, when tested in GitHub Actions, must
use `NVIDIA_NIM_API_KEY` rather than `COPILOT_GITHUB_TOKEN`.

## Numerical boundary

Python performs validation, canonicalization, hashing, sparse design marshalling,
and serialization. Rust owns:

- the multilevel and cross-classified weighted predictor;
- independent per-respondent OLS state estimates;
- discrete-step stationary AR(1) state prediction using caller-supplied \(\phi\);
- deterministic CPU respondent sharding and diagnostics for those state paths.

The following remain explicit future boundaries: full multilevel IRT random-effect
integration and estimation, uncertainty/intervals, joint item/context/state
likelihood and gradients, GPU batching for the recurrent state path, continuous
time transitions, and true-parameter recovery for the full joint model.

A Python fallback estimator is explicitly out of scope.

## Identification boundary

Structurally valid input is not automatically identified for every model. Future
fitters must reject random effects confounded with fixed effects, disconnected
measurement graphs, unanchored drift, random slopes without within-person time
variation, lagged-response terms without repeated same-item observations, and
context dimensions without enough independently linked levels and observations.

Dimension-specific weights describe the linear combination of context effects;
they do not by themselves prove that each variance component is estimable.

## Validation and recovery

Required estimator evidence includes scale-aligned bias, MAE, RMSE, interval
coverage, convergence, and classified failures for latent states, context
effects, membership effects, growth, discrete-step AR coefficients, future
continuous-time parameters, item/rater drift, and lagged dependence. Correlation
is supplementary only.

Recovery designs must include nested, crossed, weighted multiple-membership,
multiple-membership multiple-classification, balanced and unbalanced occasions,
missing observations, irregular gaps, zero and near-boundary variance
components, and confounded designs that are expected to fail.

## Data and naming

The contracts use non-numeric descriptive identifiers and two-or-more-token
lower `snake_case`. Future persistence uses names such as `context_dimension`,
`context_membership`, `membership_revision`, `temporal_occasion`,
`latent_state_estimate`, and `rater_drift_estimate`.

## Interpretation boundary

Multilevel and longitudinal measurement separates sources of variation. It does
not establish causal contextual effects, invariance, fairness, transportability,
or authorization for consequential decisions. Weighting choices carry
substantive assumptions and require sensitivity analysis rather than being
treated as neutral preprocessing.
