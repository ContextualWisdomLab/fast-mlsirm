# Marginal-reduction allocation evidence

## Changed

- Three allocation-reducing NumPy marginal-estimator expressions now route
  through private, directly testable helpers without changing their equations.
- `benchmarks/benchmark_marginal_reductions.py` compares those production
  helpers with the former broadcast equations on exactly representable inputs
  and emits environment, configuration, parity, elapsed observations, and
  Python-traced peak allocations as JSON. The report explicitly makes no
  universal speed or memory claim.
- Every report binds the evidence to SHA-256 digests of the benchmark and
  production module plus a CI-injected revision, or the explicit value
  `unavailable` when no revision was supplied.
- Hand-derived unit fixtures and Rust/NumPy estimator-path tests protect the
  reduction axes, weights, and fitted-result parity.
- The three changed reductions now recover their generating sufficient
  statistics under the estimator's 121-node standard-normal Gauss-Hermite
  rule: the rule's analytic moment identities are checked independently;
  multilevel and population reductions match `math.fsum` oracles; and the
  covariate score is zero with positive Fisher information at its generating
  coefficient on the production `(group, item, theta-node, latent-node)` axes.
  Standard floating-point forward-error bounds replace empirical tolerances.
