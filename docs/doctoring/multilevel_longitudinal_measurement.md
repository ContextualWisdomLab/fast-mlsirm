# Multilevel and longitudinal measurement doctoring

## Decision

`fast_mlsirm.multilevel` introduces immutable design contracts for nested,
cross-classified, weighted multiple-membership, and repeated longitudinal
measurement. The contracts prevent contextual and temporal provenance from
being collapsed into respondent IDs or unstructured metadata. A Rust-owned
state layer now consumes the sealed longitudinal design for independent
per-respondent OLS trends and discrete-step AR(1) state prediction. A
separate joint MAP hierarchical continuous-time AR(1) Rasch slice estimates
shared population hyperparameters and person-occasion states. Estimated
multiple-membership `u_h` and Fox–Glas / adaptive-quadrature multilevel IRT
remain outside both slices.

Python owns validation, content identity, bounded collection handling, replay
protection, sparse design marshalling, and serialization. Rust owns the
weighted contextual predictor, the first state-layer arithmetic, and the
joint MAP hierarchical CT-AR Rasch kernel. The OLS path uses
respondent-sharded CPU threads; the AR path uses sequence-index gaps and a
caller-supplied coefficient; the hierarchical CT-AR path uses elapsed-day
Ornstein–Uhlenbeck transitions and packed L-BFGS. GPU batching for this
Rasch objective is not implemented: the existing wgpu path owns MLSIRM
distance/likelihood kernels, a different estimand.

## Scientific rationale

A respondent-level model can commit an atomistic fallacy when cluster, team,
school, site, tenant, or other contextual variation is attributed entirely to
the individual. Multiple-membership designs are required when an observation is
influenced by more than one level of one contextual classification, such as a
learner taught by multiple teachers or an employee attached to several teams.
Cross-classified designs are different: one observation can simultaneously
belong to distinct classifications such as school and residential area. The
contract therefore records `context_dimension_id` separately from `context_id`;
context labels are never parsed to infer their variance-component family.

For every observation \(o\) and contextual dimension \(d\), the exact positive
weights normalize independently:

\[
\sum_{h \in \mathcal H_d(o)} w_{odh}=1.
\]

A one-hot assignment is a special case within one dimension. Multiple
classification is represented by more than one dimension, each with its own
complete assignment. Schema 1.0 requires each observation to carry every
dimension declared in the design, because silently treating an absent
classification as a zero random effect would change the model.

Repeated measurement adds a separate requirement. Stable latent differences,
systematic growth, time-varying states, rater drift, item/task revision drift,
and residual same-item dependence must not be represented by one undifferentiated
occasion effect.

## Operational contract

The implementation:

- requires a descriptive `context_dimension_id` and `context_id` for every edge;
- scopes duplicate cells by observation, dimension, and context level;
- retains exact context weights and rejects materially invalid per-dimension
  sums instead of silently renormalizing them;
- treats one-hot nesting as a special case of dimension-scoped membership;
- supports weighted multiple membership and multiple classifications in one
  observation;
- requires every observation to contain every declared context dimension in
  schema 1.0;
- exposes dimension-scoped context keys plus deterministic per-observation,
  per-dimension counts and exact weights;
- binds each assignment-revision fingerprint to the exact observation,
  dimension, context, and weight;
- canonicalizes input order without changing assignments;
- retains exact membership and occasion revision fingerprints;
- requires strict respondent-level sequence and time ordering;
- distinguishes a random-intercept/slope wire label from stationary AR(1);
- fits the state layer with Rust-only arithmetic and deterministic respondent
  sharding, reporting independent OLS or caller-supplied AR estimand metadata;
- fits a separate joint MAP hierarchical CT-AR Rasch slice with estimated
  `(mu, tau, lambda)`, Wald observed-information intervals, and explicit
  exclusion of multiple-membership random effects and GPU parity;
- reports state RMSE, observed/transition counts, and worker-count-invariant
  results;
- keeps lagged-response dependence independently switchable;
- bounds all collections before aggregate allocation;
- rejects Boolean-as-number coercion, non-finite values, duplicate cells, and
  revision rebinding;
- replay-verifies exact package-owned child artifacts before aggregation; and
- stores no raw source, response, prompt, or provider output.

## Temporal interpretation boundary

The ADR-0018 `autoregressive_coefficient` is a discrete occasion-step
stationary AR(1) coefficient with \(-1<\phi<1\). Irregular millisecond
offsets are retained as exact ordering and audit provenance. They do **not**
imply that one \(\phi\) is automatically adjusted for elapsed time.

A separate joint MAP slice (ADR-0019) parameterizes elapsed time through
\(\phi_{pt}=\exp(-\lambda\Delta_{pt})\) with \(\Delta_{pt}\) in days. That
slice may be described as continuous-time AR(1) / Ornstein–Uhlenbeck. The
discrete AR path must not be described as continuous-time or interval-adjusted;
its coefficient is tied to sequence gaps. The OLS path uses exact day-scaled
offsets and does not estimate a continuous-time transition.

## Identification and interpretation limits

A structurally valid contract is not proof that a requested statistical model is
identified. Future Rust fitters must fail closed for confounded fixed and random
effects, dimensions with insufficient linked levels, disconnected measurement
graphs, random slopes without within-respondent time variation, unanchored drift,
and lagged-response terms without repeated same-item observations.

Membership weights define a substantive linear combination of contextual
effects. They are not neutral preprocessing. Alternative plausible weighting
schemes require sensitivity analysis. The contracts do not establish causal
contextual effects, measurement invariance, fairness, transportability, or
permission for consequential decisions.

## Failure and rollback

A contract-validation failure produces a stable machine code, JSON path, and
non-reflective explanation. Callers must repair the design rather than bypassing
validation or renormalizing invalid weights silently.

Rollback consists of removing the new namespace before any estimator depends on
it. Once persisted artifacts or estimator APIs depend on schema `1.0`, changes
must use explicit schema migration rather than mutating the accepted contract.

## Verification boundary

The current evidence includes contract validation, deterministic identity,
child replay, resource bounds, dimension-scoped assignment, strict temporal
ordering, Rust state-layer slope recovery, missing-occasion preservation,
discrete AR transition RMSE, worker-count determinism, source-text-free
serialization, and joint MAP hierarchical CT-AR Rasch recovery of known
states with measurement-information Wald interval coverage. Shared mean and
MAP-shrunk `tau` are recovered under documented RMSE bounds. Short series
leave `lambda` weakly identified; the transition claim is elapsed-day
`phi_pt=exp(-lambda Delta_pt)` with a finite positive decay, not tight
unbiased recovery of `lambda`. It is not evidence of:

- Fox and Glas Gibbs or Jeon and Rabe-Hesketh adaptive-quadrature ML;
- unbiased maximum-likelihood variance-component recovery;
- estimated multiple-membership or crossed `u_h`;
- measurement invariance or fairness;
- causal contextual effects;
- GPU recurrent-state performance or parity; or
- high-stakes deployment readiness.

Those claims require separate Rust implementations and same-head recovery,
security, packaging, and operational evidence. Recovery must report scale-aligned
bias, MAE, RMSE, interval coverage, convergence, and classified failures;
correlation alone is supplementary.

## APA 7 references

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership multiple
classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124.
https://doi.org/10.1177/1471082X0100100202

Embretson, S. E. (1991). A multidimensional latent trait model for measuring
learning and change. *Psychometrika, 56*, 495–515.
https://doi.org/10.1007/BF02294487

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model using Gibbs sampling. *Psychometrika, 66*, 271–288.
https://doi.org/10.1007/BF02294839

Huang, Z., & Cai, L. (2024). Estimation of the three-parameter logistic
cross-classified item response theory model. *Journal of Educational and
Behavioral Statistics, 49*(3), 525–558.
https://doi.org/10.3102/10769986231193351

Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for
longitudinal item analysis. *Psychometrika, 81*(3), 830–850.
https://doi.org/10.1007/s11336-015-9489-2

Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal
data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876

Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2011). A hierarchical
latent stochastic differential equation model for affective dynamics.
*Psychological Methods, 16*(2), 468–490. https://doi.org/10.1037/a0024375

Tranmer, M., Steel, D., & Browne, W. J. (2014). Multiple-membership
multiple-classification models for social network and group dependencies.
*Journal of the Royal Statistical Society: Series A (Statistics in Society),
177*(2), 439–455. https://doi.org/10.1111/rssa.12021

Uto, M. (2023). A Bayesian many-facet Rasch model with Markov modeling for
rater severity drift. *Behavior Research Methods, 55*, 3910–3928.
https://doi.org/10.3758/s13428-022-01997-z
