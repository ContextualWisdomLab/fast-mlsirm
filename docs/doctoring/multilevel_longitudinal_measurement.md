# Multilevel and longitudinal measurement doctoring

## Decision

`fast_mlsirm.multilevel` introduces immutable design contracts for nested,
cross-classified, weighted multiple-membership, and repeated longitudinal
measurement. The contracts prevent contextual and temporal provenance from
being collapsed into respondent IDs or unstructured metadata.

This first slice performs no statistical estimation. Python owns validation,
content identity, and serialization only. Likelihood, integration, gradients,
optimization, uncertainty, CPU multithreading, and future GPU batching remain in
Rust.

## Scientific rationale

A respondent-level model can commit an atomistic fallacy when cluster, team,
school, site, tenant, or other contextual variation is attributed entirely to
the individual. Multiple-membership designs are required when an observation is
influenced by more than one context, such as a learner taught by multiple
teachers, an employee attached to several teams, or an AI workflow using several
orchestrating agents.

Repeated measurement adds a separate requirement. Stable latent differences,
systematic growth, time-varying states, rater drift, item/task revision drift,
and residual same-item dependence must not be represented by one undifferentiated
occasion effect.

## Operational contract

The implementation:

- retains exact context weights and rejects materially invalid sums;
- treats one-hot nesting as a special case of multiple membership;
- canonicalizes input order without changing weights;
- retains exact membership and occasion revision fingerprints;
- requires strict respondent-level sequence and time ordering;
- distinguishes a random-intercept/slope state from stationary AR(1);
- keeps lagged-response dependence independently switchable;
- bounds all collections before aggregate allocation;
- rejects Boolean-as-number coercion, non-finite values, duplicate cells, and
  revision rebinding;
- stores no raw source, response, prompt, or provider output.

## Failure and rollback

A contract-validation failure produces a stable machine code, JSON path, and
non-reflective explanation. Callers must repair the design rather than bypassing
validation or renormalizing invalid weights silently.

Rollback consists of removing the new namespace before any estimator depends on
it. Once persisted artifacts or estimator APIs depend on schema `1.0`, changes
must use explicit schema migration rather than mutating the accepted contract.

## Verification boundary

The current evidence is limited to contract validation, deterministic identity,
resource bounds, and source-text-free serialization. It is not evidence of:

- estimator correctness;
- true-parameter recovery;
- interval coverage;
- measurement invariance or fairness;
- causal contextual effects;
- GPU performance or parity;
- high-stakes deployment readiness.

Those claims require separate Rust implementations and same-head recovery,
security, packaging, and operational evidence.

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
