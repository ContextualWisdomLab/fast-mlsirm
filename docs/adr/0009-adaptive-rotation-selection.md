# ADR-0009: Adaptive rotation uses criterion registry, multi-start and empirical selection

Status: **Proposed**  
Date: 2026-08-09

## Context

Exploratory factor rotation does not have one universally optimal criterion. Criterion behavior depends on loading complexity, factor correlation, cross-loadings, sample size, target information and local optima. Comparing raw objective values across different criteria is also invalid because those objectives have different definitions and scales.

An open Draft PR contains a substantial Rust rotation implementation, but full protected integration, platform/package verification, GPU batching and remaining criterion families are incomplete. This ADR remains Proposed.

## Decision

Rotation is structured in three layers.

### 1. Criterion registry

Each analytic criterion implements a common Rust value/gradient contract. The optimizer does not contain criterion-specific algebra except through that interface.

Targeted first-class families include orthomax/Crawford-Ferguson, oblimin, geomin, target/PST, bifactor/bi-geomin and selected additional analytic criteria. Procedural or derivative-free criteria may use explicit separate adapters rather than being forced into an invalid gradient contract.

### 2. Optimizer and solution search

The Rust optimizer supports appropriate orthogonal/oblique geometry and reports:

- criterion value;
- projected gradient/stationarity;
- transform/pattern/structure/factor-correlation matrices;
- termination reason;
- deterministic multi-start evidence;
- best-start index and best-observed basin support;
- sign/permutation canonicalization where semantically valid.

Finite multi-start returns the **best observed solution**, not proof of a global optimum.

### 3. Criterion-neutral selector

Criteria are compared using common evidence such as:

- loading/simple-structure complexity;
- cross-loading sparsity;
- degeneracy/near-singular factor-correlation penalties;
- bootstrap stability and Tucker congruence after global assignment/sign alignment;
- target/theory agreement when externally provided;
- split-sample or simulation recovery;
- convergence/basin stability.

The selector may expose policies such as `stability_first`, `recovery_first`, `theory_guided`, `bifactor_discovery` or `fully_exploratory`, but policy weights/ranks are documented choices, not universal scientific constants.

## Numerical constraints

- Matrix operations used by a criterion must respect its mathematical domain; SPD log determinants use SPD-safe decomposition such as Cholesky rather than sign-ambiguous row pivot heuristics.
- PST/target weights follow the exact documented criterion semantics. Binary-mask PST does not silently accept arbitrary continuous weights.
- Hyperparameters are included in solution provenance.
- CPU multi-start should use coarse parallelism; GPU batching is introduced only with objective/gradient/stationarity parity evidence.

## Consequences

The public product can answer both “what is the best solution we observed for this criterion?” and “which criterion is best supported for this use case?” without conflating the two.

## Alternatives considered

- **Default to varimax.** Rejected as universal policy.
- **Choose the criterion with the numerically smallest objective.** Rejected because objectives are not cross-criterion comparable.
- **Use one random start.** Rejected because local minima can dominate results.
- **Claim finite multi-start global optimization.** Rejected as scientifically unsupported.

## Acceptance before status becomes Accepted

- Rust registry/optimizer and Python/PyO3 public API protected-integrated;
- criterion gradient and covariance-preservation tests;
- deterministic multi-threaded multi-start tests;
- bootstrap/recovery selector evidence;
- packaging across supported platforms;
- GPU path only after parity evidence if released;
- primary-source traceability for implemented criterion formulas.

## References

Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and software for arbitrary rotation criteria in factor analysis. *Educational and Psychological Measurement, 65*(5), 676–696.
