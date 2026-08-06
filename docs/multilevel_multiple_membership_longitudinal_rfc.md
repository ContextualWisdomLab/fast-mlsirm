# Multilevel, Multiple-Membership, and Longitudinal Measurement RFC

## Status

This RFC defines the provider-neutral contract boundary delivered by issue #565.
It does not yet add a numerical estimator. All future psychometric arithmetic
remains a Rust-core responsibility.

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

## Multiple-membership contract

Each observation has one or more weighted context edges. Within an observation,
weights are finite, strictly positive, and sum to one within the fixed contract
tolerance:

\[
\sum_{h \in \mathcal H(o)} w_{oh}=1.
\]

One-hot nesting is a single weight-one edge. Cross-classification and multiple
membership use two or more edges. The package preserves exact weights and rejects
materially invalid totals rather than silently renormalizing them.

Membership edges retain:

- observation identity;
- context identity;
- exact membership weight;
- assignment-revision SHA-256;
- deterministic membership and design fingerprints.

## Temporal contract

Each occasion retains respondent, occasion, sequence, exact integer time offset,
and revision identity. Within a respondent:

- occasion IDs are unique;
- sequence indices are unique;
- time offsets are unique;
- time offsets increase strictly with sequence order;
- irregular spacing is preserved.

The initial state specifications are:

- `random_intercept_slope`;
- `stationary_autoregressive` with \(-1<\phi<1\).

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
and serialization. Future Rust PRs own:

- multilevel and cross-classified predictors;
- random-effect integration;
- longitudinal state transitions;
- likelihood and gradients;
- optimization and uncertainty;
- CPU multithreading and justified GPU batching;
- true-parameter recovery.

A Python fallback estimator is explicitly out of scope.

## Identification boundary

Structurally valid input is not automatically identified for every model. Future
fitters must reject random effects confounded with fixed effects, disconnected
measurement graphs, unanchored drift, random slopes without within-person time
variation, and lagged-response terms without repeated same-item observations.

## Validation and recovery

Required estimator evidence includes scale-aligned bias, MAE, RMSE, interval
coverage, convergence, and classified failures for latent states, context
effects, membership effects, growth, AR coefficients, item/rater drift, and
lagged dependence. Correlation is supplementary only.

## Data and naming

The contracts use non-numeric descriptive identifiers and two-or-more-token
lower `snake_case`. Future persistence uses names such as `context_membership`,
`membership_revision`, `temporal_occasion`, `latent_state_estimate`, and
`rater_drift_estimate`.

## Interpretation boundary

Multilevel and longitudinal measurement separates sources of variation. It does
not establish causal contextual effects, invariance, fairness, transportability,
or authorization for consequential decisions.
