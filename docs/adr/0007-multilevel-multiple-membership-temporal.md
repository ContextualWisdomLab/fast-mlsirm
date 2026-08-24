# ADR-0007: Multilevel, multiple-membership and temporal structure are first-class

Status: **Proposed**  
Date: 2026-08-09

## Context

Psychometric and AI-evaluation observations commonly sit inside schools, teams, organizations, prompts, testlets, documents, clients, time periods or other overlapping contexts. Repeated observations also evolve over time. Flattening those structures into independent rows can produce atomistic fallacy, understate uncertainty, confound stable traits with context effects and drift, and misinterpret temporal dependence.

Reusable contracts plus a Rust-owned respondent state layer define a supported
longitudinal handoff. ADR-0020 adds a separate joint MAP hierarchical
continuous-time AR(1) Rasch slice on that handoff. This ADR remains Proposed
until those boundaries are protected-integrated. Fox and Glas Gibbs sampling,
Jeon and Rabe-Hesketh adaptive-quadrature ML, estimated multiple-membership
`u_h`, and GPU recurrent-state parity are not accepted production behavior.

Protected main already contains the reusable nested, cross-classified, and
multiple-membership contracts plus crossed `u_h` MAP recovery evidence. The
longitudinal state and joint CT-AR slices remain separate estimands; Fox and
Glas Gibbs sampling, Jeon and Rabe-Hesketh adaptive-quadrature ML, and GPU
recurrent-state parity are not accepted production behavior.

This decision is about explicit context and time in the measurement design. It
is not a claim that latent-space MLSIRM/MLS2PLM interaction (Jeon, Jin,
Schweinberger, & Baugh, 2021; Kang & Jeon, 2025) absorbs hierarchy or drift.
Residual latent-space interaction remains a later layer after known
multilevel/temporal structure, consistent with ADR-0001 and ADR-0006.

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

The ADR-0019 state-layer boundary remains deliberately narrower: independent
per-respondent OLS trends are fitted by Rust on exact day-scaled offsets, and
stationary AR(1) states produce discrete-sequence predictions from a
caller-supplied coefficient. The latter uses sequence gaps, not elapsed
milliseconds, so irregular calendar spacing cannot be silently treated as a
continuous-time decay. Those predictors do not estimate population
random-effects distributions or AR-coefficient uncertainty.

ADR-0020 is a separate joint MAP slice. It estimates shared
`(mu, tau, lambda)`, shrinks person-occasion states toward `mu`, and uses
elapsed days in an Ornstein–Uhlenbeck / continuous-time AR(1) transition.
It does not estimate crossed or multiple-membership `u_h`.

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

## Research and standards basis

Score interpretation that depends on context, occasion, or drift remains governed by AERA, APA, and NCME (2014). Full multilevel/temporal estimators stay Proposed until the numerical release rule above is met. NIST, OWASP, and CWE catalogs are not the methodological basis.

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5

Uto, M. (2023). A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods, 55*, 3910–3928. https://doi.org/10.3758/s13428-022-01997-z
