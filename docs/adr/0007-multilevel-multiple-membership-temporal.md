# ADR-0007: Multilevel, multiple-membership and temporal structure are first-class

Status: **Proposed**  
Date: 2026-08-09

## Context

Psychometric and AI-evaluation observations commonly sit inside schools, teams, organizations, prompts, testlets, documents, clients, time periods or other overlapping contexts. Repeated observations also evolve over time. Flattening those structures into independent rows can produce atomistic fallacy, understate uncertainty, confound stable traits with context effects and drift, and misinterpret temporal dependence.

A current open PR contains reusable contracts plus a Rust-owned respondent
state layer for a supported longitudinal handoff, but it is not yet
protected-integrated. Full multilevel IRT random-effect estimation, uncertainty,
joint likelihood recovery, and GPU recurrent-state parity are not accepted
production behavior. Therefore this ADR remains Proposed.

## Decision

The architecture treats the following as distinct, explicit structures:

- nested context;
- cross-classified context;
- weighted multiple membership;
- multiple-membership multiple-classification;
- testlet/shared-stimulus local dependence;
- repeated longitudinal occasions;
- discrete occasion-step autoregression;
- future continuous-time state transitions;
- rater/model/prompt drift.

The accepted state-layer boundary is deliberately narrower: random
intercept/slope states are fitted by Rust OLS on exact day-scaled offsets, and
stationary AR(1) states produce discrete-sequence predictions. The latter uses
sequence gaps, not elapsed milliseconds, so irregular calendar spacing cannot be
silently treated as a continuous-time decay.

### Contract rules

1. Context membership names an explicit `context_dimension_id` and dimension-scoped `context_id`.
2. Membership weights are provenance-bound and validated under the public contract; they are not silently inferred from labels.
3. Every observation carries the context dimensions required by the declared design.
4. Repeated occasions preserve respondent/system identity and exact temporal ordering/provenance.
5. A discrete occasion-step AR coefficient is not interpreted as elapsed-time decay. Continuous-time interpretation requires a separately parameterized model.
6. Local/testlet effects and substantive dimensions remain conceptually distinct.

### Numerical release rule

A new Rust estimator for these structures is not production-ready until realistic simulation establishes:

- identification under supported designs;
- parameter bias and RMSE;
- SE/interval coverage;
- convergence and failure classification;
- behavior under sparse/unbalanced membership;
- CPU/GPU parity where a GPU path exists;
- comparison against simpler models using relation-safe procedures.

## Consequences

The library can represent scientifically realistic designs before every estimator is implemented. Contract availability does not imply estimation capability or causal interpretation.

This architecture avoids forcing product-specific tenant/org structures into the core; contexts are provider/domain-neutral.

## Alternatives considered

- **Flatten all observations.** Rejected due to atomistic/ecological inference risk and underestimated dependency.
- **Use latent space to absorb all dependence.** Rejected because known hierarchy/time/testlet structures should be modeled explicitly before residual interactions.
- **Treat timestamps as labels only forever.** Rejected as a long-term architecture; timestamps are preserved so explicit temporal estimators can be added without data-model migration.

## Acceptance before status becomes Accepted

- governed contracts merged to protected main;
- architecture/serialization tests pass;
- at least one Rust estimator or clear handoff contract exists for a supported multilevel/temporal inference use case;
- the state-layer recovery fixture reports slope recovery, missing-occasion
  behavior, AR transition RMSE, and equality across worker counts;
- recovery evidence meets the numerical release rule.

## References

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*, 271–288.

Uto, M. (2022). A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods, 55*, 3910–3928.

Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for
longitudinal item analysis. *Psychometrika, 81*(3), 830–850.
https://doi.org/10.1007/s11336-015-9489-2

Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal
data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876
